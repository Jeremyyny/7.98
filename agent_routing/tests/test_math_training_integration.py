"""Opt-in real HF/PEFT/TRL smoke test: no downloaded models and no CUDA required.

MARGENT_CPU_INTEGRATION=1 python -m pytest -q tests/test_math_training_integration.py
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


@unittest.skipUnless(os.environ.get("MARGENT_CPU_INTEGRATION") == "1", "opt-in tiny CPU training")
class TrainingIntegrationTest(unittest.TestCase):
    def test_bounded_parquet_stream_exits_cleanly(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            pq.write_table(pa.table({"answer": list(range(20000))}), path)
            result = subprocess.run([sys.executable, "-c", """
import sys
from datasets import IterableDataset
from src.verifiable.data import parquet_rows
ds = IterableDataset.from_generator(parquet_rows, gen_kwargs={"files": [sys.argv[1]]})
it = iter(ds.shuffle(seed=42, buffer_size=10000))
assert isinstance(next(it)["answer"], int)
it.close()
""", str(path)], capture_output=True, text=True, timeout=30,
                env={**os.environ, "HF_HOME": str(Path(tmp) / "hf-cache")})
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_qwen35_multimodal_checkpoint_loads_identical_text_weights(self):
        import torch
        from transformers import Qwen3_5Config, Qwen3_5ForConditionalGeneration, Qwen3_5ForCausalLM
        config = Qwen3_5Config(text_config={"vocab_size": 16, "hidden_size": 32,
            "intermediate_size": 64, "num_hidden_layers": 1, "num_attention_heads": 2,
            "num_key_value_heads": 1, "head_dim": 16, "layer_types": ["full_attention"]},
            vision_config={"depth": 1, "hidden_size": 32, "intermediate_size": 64,
                           "num_heads": 2, "out_hidden_size": 32, "num_position_embeddings": 16})
        full = Qwen3_5ForConditionalGeneration(config)
        expected = full.model.language_model.embed_tokens.weight.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            full.save_pretrained(tmp)
            text = Qwen3_5ForCausalLM.from_pretrained(tmp)
            self.assertTrue(torch.equal(expected, text.model.embed_tokens.weight))

    def test_sft_continuation_then_reload_adapter(self):
        import torch
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
        from src.benchmarks.base import StandardRow
        from src.utils.io import write_jsonl
        from src.verifiable.backend import configure_tokenizer, load_model
        from src.verifiable.protocol import messages
        from src.verifiable.training import train_sft, train_rl
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "base"
            vocab = {v: i for i, v in enumerate(["<pad>", "<unk>", "<|im_start|>", "<|im_end|>",
                     "one", "two", "three", "four", "five", "six", "7", "8", "9", "10", "11", "12"])}
            raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
            raw.pre_tokenizer = Whitespace()
            tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>",
                                          pad_token="<pad>", eos_token="<|im_end|>",
                                          additional_special_tokens=["<|im_start|>"])
            configure_tokenizer(tok).save_pretrained(base)
            model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=32,
                    intermediate_size=64, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, head_dim=16, max_position_embeddings=4096,
                    pad_token_id=0, eos_token_id=3))
            model.save_pretrained(base)
            problem = StandardRow(0, "smoke", "free_response_math", "one plus one", {}, "2",
                                   metadata={"answer_type": "math"}, split="train")
            normalized = root / "train.jsonl"
            write_jsonl(str(normalized), [problem.to_dict(), {**problem.to_dict(),
                         "example_id": 1, "question": "two plus two", "ground_truth": "4"}])
            sft = root / "sft.jsonl"
            write_jsonl(str(sft), [{"prompt": messages(problem, direct=True),
                        "response": [{"role": "assistant", "content": r"FINAL_ANSWER: \boxed{2}"}],
                        "decision_type": "independent_solution", "split": "train", "protocol_version": 2}])
            cfg = {"base_model": str(base), "seed": 42, "advisor_url": "http://127.0.0.1:1",
                   "max_depth": 1, "max_new_tokens": 4, "advisor_max_tokens": 4,
                   "max_context": 4096, "max_seq_len": 4096, "sft_max_steps": 1,
                   "sft_accumulation": 1, "rl_max_steps": 1, "rl_accumulation": 2,
                   "num_generations": 2, "rl_max_completion_length": 4, "save_steps": 1}
            train_sft(cfg, str(base), str(sft), str(root / "sft"))
            self.assertTrue((root / "sft/adapter_model.safetensors").exists())
            train_sft(cfg, str(root / "sft"), str(sft), str(root / "continued"))
            self.assertTrue((root / "continued/adapter_model.safetensors").exists())
            _, restored = load_model(str(base), str(root / "continued"))
            self.assertFalse(any(p.requires_grad for p in restored.parameters()))
            # Idempotent completed-stage restart.
            train_sft(cfg, str(root / "sft"), str(sft), str(root / "continued"))
            metrics = json.loads((root / "continued/training_metrics.json").read_text())
            self.assertIn("train_loss", metrics)
            usage = [json.loads(line) for line in (root / "continued/usage.jsonl").read_text().splitlines()]
            self.assertTrue(any(r["role"] == "sft_train" and r["supervised_tokens"] > 0 for r in usage))
            self.assertEqual(json.loads((root / "continued/status.json").read_text())["status"], "completed")
            self.assertTrue((root / "continued/training_log.jsonl").exists())
