"""LoRA continuation, complete-solution SFT, and native-tool GRPO."""
from __future__ import annotations

from pathlib import Path
import hashlib
import inspect
import json
import time

from ..utils.io import append_jsonl, read_jsonl, write_json
from .answers import correct, extract_final
from .backend import HTTPAdvisors, load_model, render
from .data import identity, load_rows
from .experiment import root_state
from .protocol import KINDS, tool_schemas


def tokenize_turn(row, tok, max_seq_len):
    schemas = None if row.get("decision_type") == "independent_solution" else tool_schemas()
    prompt = render(tok, row["prompt"], schemas)
    full = render(tok, row["prompt"] + row["response"], schemas, generation=False)
    pids = tok(prompt, add_special_tokens=False)["input_ids"]
    ids = tok(full, add_special_tokens=False)["input_ids"]
    if len(ids) > max_seq_len:
        return None  # drop and report; do not train on truncated solutions
    prefix = 0
    for a, b in zip(pids, ids):
        if a != b:
            break
        prefix += 1
    if prefix >= len(ids):
        return None
    return {"input_ids": ids, "attention_mask": [1] * len(ids),
            "labels": [-100] * prefix + ids[prefix:]}


def _training_output(path, config, checkpoint, data_path, stage):
    from transformers.trainer_utils import get_last_checkpoint
    from .runner import checkpoint_identity
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    signature = {"config": config, "checkpoint": checkpoint_identity(checkpoint), "stage": stage,
                 "data_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest()}
    manifest = output / "training_run.json"
    if manifest.exists():
        if json.loads(manifest.read_text()) != signature:
            raise ValueError("Training settings or data changed; choose a new output directory")
    elif any(output.iterdir()):
        raise FileExistsError("Training directory has no matching run manifest")
    else:
        write_json(str(manifest), signature)
    complete = (output / "training_metrics.json").exists() and (output / "adapter_config.json").exists()
    return complete, get_last_checkpoint(str(output))


def train_sft(config, checkpoint, data_path, output):
    import torch
    from datasets import Dataset
    from transformers import DataCollatorForSeq2Seq, Trainer, TrainingArguments, set_seed
    finished, resume_checkpoint = _training_output(output, config, checkpoint, data_path, "sft")
    if finished:
        return
    set_seed(config["seed"])
    tok, model = load_model(config["base_model"], checkpoint, trainable=True,
                            lora_rank=config.get("lora_rank", 16))
    rows = read_jsonl(data_path)
    features = [tokenize_turn(row, tok, config["max_seq_len"]) for row in rows]
    kept = [f for f in features if f is not None]
    if not kept:
        raise ValueError("No complete SFT targets fit max_seq_len")
    report = {"input_turns": len(rows), "kept_turns": len(kept), "dropped_turns": len(rows) - len(kept),
              "input_tokens_per_epoch": sum(len(f["input_ids"]) for f in kept),
              "supervised_tokens_per_epoch": sum(sum(y != -100 for y in f["labels"]) for f in kept),
              "checkpoint": checkpoint, "data": data_path}
    write_json(str(Path(output) / "sft_data_report.json"), report)
    print(report, flush=True)
    args = TrainingArguments(output_dir=output, per_device_train_batch_size=1,
        gradient_accumulation_steps=config.get("sft_accumulation", 8),
        learning_rate=config.get("sft_learning_rate", 2e-5),
        num_train_epochs=config.get("sft_epochs", 1),
        max_steps=config.get("sft_max_steps", -1), bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, save_strategy="steps", save_steps=config.get("save_steps", 10),
        save_total_limit=2, report_to=[], remove_unused_columns=False,
        seed=config["seed"])
    trainer = Trainer(model=model, args=args, train_dataset=Dataset.from_list(kept),
                      data_collator=DataCollatorForSeq2Seq(tok, padding=True, label_pad_token_id=-100))
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output)
    tok.save_pretrained(output)
    write_json(str(Path(output) / "training_metrics.json"),
               {**result.metrics, "wall_seconds": time.monotonic() - start, "config": config})


def make_environment(rows, advisors, max_calls):
    index = {r.example_id: r for r in rows}

    class MathEnvironment:
        def reset(self, example_id: int, **kwargs):
            self.row = index[int(example_id)]
            self.used = []
            self.valid = True
            self.error = None
            self.usage = []
            return None

        def _call(self, kind, draft=""):
            if kind in self.used or len(self.used) >= max_calls:
                self.valid = False
                return '{"error": "advisor repeat or call budget exceeded"}'
            self.used.append(kind)
            try:
                response = advisors.call(kind, self.row, draft)
            except Exception as exc:
                self.error = str(exc)
                self.valid = False
                raise
            self.usage.append(response)
            return response["text"]

        def extractor_tool(self) -> str:
            """Extract the mathematical givens and constraints.

            Returns:
                Advice on the current problem.
            """
            return self._call("extractor")

        def reasoner_tool(self) -> str:
            """Suggest a mathematical solution approach.

            Returns:
                Advice on the current problem.
            """
            return self._call("reasoner")

        def verifier_tool(self, current_draft: str) -> str:
            """Audit the current derivation for mathematical errors.

            Args:
                current_draft: Your full current mathematical derivation, including candidate answer.

            Returns:
                An audit of the supplied derivation.
            """
            if not current_draft.strip():
                self.valid = False
                return '{"error": "missing derivation"}'
            return self._call("verifier", current_draft)

    return MathEnvironment


def reward_function(trace_path):
    def reward(completions, ground_truth, environments, question_hash=None, **kwargs):
        values, records = [], []
        for i, (completion, gold, env) in enumerate(zip(completions, ground_truth, environments)):
            if env.error:
                raise RuntimeError(f"Advisor infrastructure failed; refusing to turn this into a math reward: {env.error}")
            turns = [m for m in completion if m.get("role") == "assistant"] if isinstance(completion, list) else []
            text = str(turns[-1].get("content") or "") if turns else str(completion or "")
            valid = env.valid and (not turns or not turns[-1].get("tool_calls"))
            # Multiple calls in one assistant turn violate the collection protocol.
            valid = valid and all(len(m.get("tool_calls") or []) <= 1 for m in turns)
            valid = valid and all(not (m.get("tool_calls") and "FINAL_ANSWER:" in str(m.get("content")))
                                  for m in turns)
            valid = valid and "<tool_call" not in text
            value = float(valid and correct(text, str(gold)))
            values.append(value)
            records.append({"question_hash": question_hash[i] if question_hash else None,
                "correct": bool(value), "valid_answer": extract_final(text) is not None,
                "calls": len(env.used), "sequence": env.used, "advisor_usage": env.usage})
        append_jsonl(trace_path, records)
        return values
    reward.__name__ = "verified_math_terminal_correctness"
    return reward


def train_rl(config, checkpoint, data_path, output):
    import torch
    from datasets import Dataset
    from transformers import set_seed
    from trl import GRPOConfig, GRPOTrainer
    from .backend import HFBackend

    if "environment_factory" not in inspect.signature(GRPOTrainer.__init__).parameters:
        raise RuntimeError("TRL with environment_factory is required; use requirements-math.txt")
    finished, resume_checkpoint = _training_output(output, config, checkpoint, data_path, "rl")
    if finished:
        return
    set_seed(config["seed"])
    rows = load_rows(data_path, required_split="train")
    tok, model = load_model(config["base_model"], checkpoint, trainable=True,
                            lora_rank=config.get("lora_rank", 16))
    backend = HFBackend.__new__(HFBackend)
    backend.tokenizer, backend.model, backend.max_context = tok, model, config["max_context"]
    model.eval()
    prompts, root_costs = [], []
    root_file = Path(output) / "rl_prompts.jsonl"
    if root_file.exists():
        prompts = read_jsonl(str(root_file))
        if [x["question_hash"] for x in prompts] != [identity(r.question) for r in rows[:len(prompts)]]:
            raise ValueError("RL draft cache does not match training rows")
    for i, row in enumerate(rows[len(prompts):], start=len(prompts)):
        root, history = root_state(row, backend, config, config["seed"] + i * 1000)
        rec = {"prompt": history, "ground_truth": row.ground_truth,
               "example_id": row.example_id, "question_hash": identity(row.question)}
        append_jsonl(str(root_file), [rec])
        prompts.append(rec)
        root_costs.append({k: root[k] for k in ("prompt_tokens", "completion_tokens", "seconds", "truncated")})
    append_jsonl(str(Path(output) / "root_generation_usage.jsonl"), root_costs)
    model.train()
    advisors = HTTPAdvisors(config["advisor_url"], config["advisor_max_tokens"], config.get("advisor_models"))
    count = config.get("num_generations", 4)
    accumulation = config.get("rl_accumulation", count)
    if accumulation % count:
        raise ValueError("On one training GPU, rl_accumulation must be divisible by num_generations")
    args = GRPOConfig(output_dir=output, per_device_train_batch_size=1,
        gradient_accumulation_steps=accumulation, num_generations=count,
        max_completion_length=config.get("rl_max_completion_length", 4096),
        max_tool_calling_iterations=config["max_depth"],
        temperature=config.get("rl_temperature", 0.8), beta=config.get("rl_beta", 0.01),
        learning_rate=config.get("rl_learning_rate", 1e-6),
        max_steps=config.get("rl_max_steps", 50), bf16=torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        chat_template_kwargs={"enable_thinking": False}, use_vllm=False,
        remove_unused_columns=False, report_to=[], logging_steps=1, save_strategy="steps",
        save_steps=config.get("save_steps", 10), save_total_limit=2,
        mask_truncated_completions=True, seed=config["seed"])
    trainer = GRPOTrainer(model=model, args=args, train_dataset=Dataset.from_list(prompts),
        processing_class=tok, reward_funcs=reward_function(str(Path(output) / "rollouts.jsonl")),
        environment_factory=make_environment(rows, advisors, config["max_depth"]))
    start = time.monotonic()
    result = trainer.train(resume_from_checkpoint=resume_checkpoint)
    trainer.save_model(output)
    tok.save_pretrained(output)
    write_json(str(Path(output) / "training_metrics.json"), {**result.metrics,
        "wall_seconds": time.monotonic() - start, "config": config,
        "root_policy": "drafts regenerated from the SFT checkpoint once per RL phase; held fixed within phase"})
