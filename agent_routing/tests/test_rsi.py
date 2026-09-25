import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.verifiable.rsi import build_plan, verify_pilot
from src.verifiable.rsi_grpo import group_advantages, token_objective, turn_logprobs, validate_rl_config
from src.verifiable.experiment import policy_rollout
from src.benchmarks.base import StandardRow


def test_advantages_no_synthetic_reward_signal():
    assert group_advantages([0, 0, 0, 0]) == [0, 0, 0, 0]
    assert group_advantages([1, 1]) == [0, 0]
    a = group_advantages([0, 1])
    assert a[0] == -a[1] and a[1] > .99
    with pytest.raises(ValueError):
        group_advantages([1])


def test_clipped_objective_gradient_and_kl():
    import torch
    old = torch.tensor([-1.])
    current = torch.tensor([-1.], requires_grad=True)
    loss = token_objective(current, old, old, 1.)
    loss.backward()
    assert current.grad.item() == pytest.approx(-1.)
    current = torch.tensor([-0.5], requires_grad=True)
    loss = token_objective(current, old, old, 1., beta=0.)
    loss.backward()
    assert loss.item() == pytest.approx(-1.2)
    assert current.grad.item() == 0
    assert token_objective(old, old, old, 0.).item() == 0
    assert token_objective(torch.tensor([-2.]), old, old, 0.).item() > 0


def test_exact_response_positions_and_temperature():
    import torch
    # Each preceding token predicts the following response ID. Prefix/tool
    # tokens affect context, but produce no separate targets in this loss.
    logits = torch.tensor([[[0., 1., 2.], [2., 1., 0.], [1., 2., 0.], [3., 0., 0.]]], requires_grad=True)
    class Model:
        device = torch.device("cpu")
        def __call__(self, **kw):
            assert kw["input_ids"].tolist() == [[0, 1, 2, 0]]
            return SimpleNamespace(logits=logits[:, -kw["logits_to_keep"]:])
    result = turn_logprobs(Model(), {"prompt_ids": [0, 1], "completion_ids": [2, 0]}, .8)
    expected = torch.log_softmax(logits[0, 1:3] / .8, -1)[[0, 1], [2, 0]]
    assert torch.allclose(result, expected)
    result.sum().backward()
    assert logits.grad[0, 0].abs().sum() == 0
    assert logits.grad[0, 3].abs().sum() == 0


class ScriptedBackend:
    def __init__(self, texts):
        self.texts = iter(texts)
    def generate(self, messages, **kwargs):
        assert not any("secret_gold" in str(m) for m in messages)
        return dict(text=next(self.texts), prompt_tokens=3, completion_tokens=2, seconds=0., truncated=False)


def test_rl_protocol_commit_is_immutable_and_advice_is_context():
    row = StandardRow(0, "tiny", "math", "one plus one", {}, "2", split="train", metadata={"answer_type": "math"})
    root = dict(text=r"FINAL_ANSWER: \boxed{2}", valid=True)
    advisors = SimpleNamespace(call=lambda *a: dict(text="Try addition", prompt_tokens=2, completion_tokens=2, seconds=0.))
    cfg = {"max_depth": 2, "max_new_tokens": 64, "decision_max_tokens": 64}
    committed = policy_rollout(row, ScriptedBackend(["COMMIT"]), advisors, cfg, 42, root, [])
    assert committed["correct"] and committed["text"] == root["text"] and committed["calls"] == 0
    bad = policy_rollout(row, ScriptedBackend([r"FINAL_ANSWER: \boxed{2}"]), advisors, cfg, 42, root, [])
    assert not bad["valid"] and not bad["correct"]
    text = '<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>'
    revised = policy_rollout(row, ScriptedBackend([text, r"FINAL_ANSWER: \boxed{3}", "COMMIT"]), advisors, cfg, 42, root, [])
    assert not revised["correct"] and revised["calls"] == 1
    assert any(m["role"] == "tool" for m in revised["history"])
    repeated = policy_rollout(row, ScriptedBackend([text, r"FINAL_ANSWER: \boxed{2}", text]), advisors, cfg, 42, root, [])
    assert not repeated["valid"]


def test_plan_continues_each_arms_previous_grpo_and_static_reuses_first_labels(tmp_path):
    config = Path(__file__).parents[1] / "configs/math_rsi_pilot.json"
    plan = build_plan(config, tmp_path / "data", tmp_path / "run")
    def one(arm, stage):
        return next(p for p in plan if f"/{arm}/round_2/{stage}" in p["output"])
    def arg(step, flag):
        return step["command"][step["command"].index(flag) + 1]
    for arm in ("dynamic", "static", "success"):
        assert arg(one(arm, "collection"), "--checkpoint").endswith(f"/{arm}/round_1/grpo")
        assert arg(one(arm, "sft"), "--checkpoint").endswith(f"/{arm}/round_1/grpo")
        assert arg(one(arm, "grpo"), "--checkpoint").endswith(f"/{arm}/round_2/sft")
    assert arg(one("static", "sft"), "--data").endswith("/initial_collection/sft.jsonl")
    assert arg(one("dynamic", "sft"), "--data").endswith("/dynamic/round_2/collection/sft.jsonl")
    assert arg(one("success", "selection"), "--data").endswith("/success/round_2/collection/records.jsonl")
    assert not any("aime" in part.lower() for p in plan for part in p["command"])
    with pytest.raises(ValueError):
        build_plan(config, tmp_path, tmp_path, rounds=1)


def test_rl_temperature_is_separate_from_eval():
    from src.verifiable.runner import load_config
    cfg = load_config(Path(__file__).parents[1] / "configs/math_rsi_pilot.json")
    validate_rl_config(cfg)
    assert cfg["temperature"] == 0 and cfg["rl_temperature"] > 0
    with pytest.raises(ValueError):
        validate_rl_config({**cfg, "rl_temperature": 0})


def test_pilot_data_tampering_rejected(tmp_path):
    (tmp_path / "train.jsonl").write_text("changed")
    (tmp_path / "pilot_data.json").write_text(json.dumps({"sha256": {"train.jsonl": "not-a-digest"}}))
    with pytest.raises(ValueError, match="changed"):
        verify_pilot(tmp_path)


def tiny_checkpoint(root):
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
    from src.verifiable.backend import configure_tokenizer, load_model
    torch.set_num_threads(1)
    base = root / "base"
    vocab = {v: i for i, v in enumerate(["<pad>", "<unk>", "<|im_start|>", "<|im_end|>", "one", "two", "three", "COMMIT"])}
    raw = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
    raw.pre_tokenizer = Whitespace()
    tok = PreTrainedTokenizerFast(tokenizer_object=raw, unk_token="<unk>", pad_token="<pad>",
          eos_token="<|im_end|>", additional_special_tokens=["<|im_start|>"])
    configure_tokenizer(tok).save_pretrained(base)
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(vocab), hidden_size=32, intermediate_size=64,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        max_position_embeddings=4096, pad_token_id=0, eos_token_id=3))
    model.save_pretrained(base)
    tok, model = load_model(str(base), trainable=True, lora_rank=2)
    checkpoint = root / "sft"
    model.save_pretrained(checkpoint)
    tok.save_pretrained(checkpoint)
    return base, checkpoint


def test_real_lora_grpo_gradient_checkpoint_resume(tmp_path, monkeypatch):
    import torch
    from safetensors.torch import load_file
    from src.verifiable.rsi_grpo import train_grpo
    from src.utils.io import write_jsonl
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    monkeypatch.setenv("ACCELERATE_USE_CPU", "true")
    base, checkpoint = tiny_checkpoint(tmp_path)
    row = StandardRow(0, "tiny", "math", "one plus one", {}, "2", split="train", metadata={"answer_type": "math"})
    data = tmp_path / "train.jsonl"
    write_jsonl(str(data), [row.to_dict()])
    cfg = dict(base_model=str(base), seed=42, advisor_url="unused", advisor_max_tokens=8,
               max_context=4096, max_seq_len=4096, rl_temperature=.8, rl_max_steps=2,
               num_generations=2, rl_learning_rate=.01, rl_beta=.01, lora_rank=2)
    root_fn = lambda *args: ({"text": "candidate"}, [])
    calls = 0
    def rollout(row, backend, advisors, config, seed, direct, history):
        nonlocal calls
        calls += 1
        good = calls % 2 == 0
        backend.turns.append(dict(prompt_ids=[2, 4, 5], completion_ids=[7 if good else 6, 3], kind="decision"))
        return dict(correct=good, valid=True, calls=0, text="COMMIT")
    with patch("src.verifiable.rsi_grpo.verify_advisor", return_value={}), \
         patch("src.verifiable.rsi_grpo.root_state", side_effect=root_fn), \
         patch("src.verifiable.rsi_grpo.policy_rollout", side_effect=rollout):
        train_grpo(cfg, str(checkpoint), str(data), str(tmp_path / "full"))
    before = load_file(str(checkpoint / "adapter_model.safetensors"))
    after = load_file(str(tmp_path / "full/adapter_model.safetensors"))
    assert any(not torch.equal(before[k], after[k]) for k in before)
    assert not any("reference" in k for k in after)
    # Interrupt on the next group after a committed optimizer step.
    calls = 0
    def interrupted(*args):
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return rollout(*args)
    with patch("src.verifiable.rsi_grpo.verify_advisor", return_value={}), \
         patch("src.verifiable.rsi_grpo.root_state", side_effect=root_fn), \
         patch("src.verifiable.rsi_grpo.policy_rollout", side_effect=interrupted):
        with pytest.raises(RuntimeError, match="simulated"):
            train_grpo(cfg, str(checkpoint), str(data), str(tmp_path / "resume"))
    with patch("src.verifiable.rsi_grpo.verify_advisor", return_value={}), \
         patch("src.verifiable.rsi_grpo.root_state", side_effect=root_fn), \
         patch("src.verifiable.rsi_grpo.policy_rollout", side_effect=rollout):
        train_grpo(cfg, str(checkpoint), str(data), str(tmp_path / "resume"))
    resumed = load_file(str(tmp_path / "resume/adapter_model.safetensors"))
    for key in after:
        assert torch.equal(after[key], resumed[key]), key
    assert json.loads((tmp_path / "resume/training_metrics.json").read_text())["optimizer_steps"] == 2
    # The next SFT phase must load the GRPO adapter (not reset to base).
    from src.verifiable.training import train_sft
    targets = tmp_path / "targets.jsonl"
    write_jsonl(str(targets), [dict(prompt=[{"role": "user", "content": "one"}],
        response=[{"role": "assistant", "content": "two"}], decision_type="independent_solution",
        split="train", protocol_version=2)])
    train_sft({**cfg, "sft_max_steps": 1, "sft_accumulation": 1, "save_steps": 1},
              str(tmp_path / "resume"), str(targets), str(tmp_path / "next_sft"))
    assert (tmp_path / "next_sft/adapter_model.safetensors").exists()


def test_real_rollout_sampler_scores_exact_distribution(tmp_path):
    import torch
    from src.verifiable.backend import load_model
    from src.verifiable.rsi_grpo import RolloutBackend
    base, checkpoint = tiny_checkpoint(tmp_path)
    tok, model = load_model(str(base), str(checkpoint), trainable=True)
    model.eval()
    backend = RolloutBackend(tok, model, dict(max_context=4096, max_seq_len=4096, rl_temperature=.8))
    backend.turns = []
    seen = {}
    original = model.generate
    def recording_generate(**kwargs):
        generated = original(**kwargs, return_dict_in_generate=True, output_scores=True)
        seen["scores"] = generated.scores
        return generated.sequences
    with patch.object(model, "generate", side_effect=recording_generate):
        result = backend.generate([{"role": "user", "content": "one"}], max_tokens=6, seed=10)
    turn = backend.turns[0]
    assert len(turn["completion_ids"]) == result["completion_tokens"]
    lp = turn_logprobs(model, turn, .8)
    actual = torch.stack([torch.log_softmax(score[0].float(), -1)[idx]
                          for score, idx in zip(seen["scores"], turn["completion_ids"])])
    assert torch.allclose(lp, actual, atol=1e-5), (lp, actual)


def test_all_invalid_group_continues_with_zero_reward_and_visible_warning(tmp_path, monkeypatch):
    from src.verifiable.rsi_grpo import train_grpo
    from src.utils.io import write_jsonl
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    base, checkpoint = tiny_checkpoint(tmp_path)
    row = StandardRow(0, "tiny", "math", "one plus one", {}, "2", split="train", metadata={"answer_type": "math"})
    data = tmp_path / "train.jsonl"
    write_jsonl(str(data), [row.to_dict()])
    cfg = dict(base_model=str(base), seed=42, advisor_url="unused", advisor_max_tokens=8,
        max_context=4096, max_seq_len=4096, rl_temperature=.8, rl_max_steps=1,
        num_generations=2, lora_rank=2)
    def invalid(row, backend, *args):
        backend.turns.append(dict(prompt_ids=[2, 4], completion_ids=[7, 3], kind="decision"))
        return dict(correct=False, valid=False, calls=0, text="candidate", error="Unclosed tool call")
    with patch("src.verifiable.rsi_grpo.verify_advisor", return_value={}), \
         patch("src.verifiable.rsi_grpo.root_state", return_value=({"text": "candidate"}, [])), \
         patch("src.verifiable.rsi_grpo.policy_rollout", side_effect=invalid):
        train_grpo(cfg, str(checkpoint), str(data), str(tmp_path / "rejected"))
    root = tmp_path / "rejected"
    assert len(json.loads((root / "invalid_group.json").read_text())["trajectories"]) == 2
    assert (root / "training_metrics.json").exists()
    step_dir = json.loads((root / "resume.json").read_text())["directory"]
    step = json.loads((root / step_dir / "step.json").read_text())
    assert step["rewards"] == [0., 0.]
    assert step["advantages"] == [0., 0.]
    assert step["valid_rate"] == 0
    assert not step["mixed_reward_group"]
    assert step["policy_loss"] == 0
    assert len((root / "rollout_diagnostics.jsonl").read_text().splitlines()) == 2


def test_report_pairs_questions_and_preserves_regressions(tmp_path):
    from src.verifiable.rsi import report
    from src.utils.io import write_jsonl
    plan = []
    for name, direct, policy in (("initial_dev", False, True),
        ("dynamic/round_2/grpo_dev", True, True), ("static/round_2/grpo_dev", False, False)):
        out = tmp_path / name
        out.mkdir(parents=True)
        write_jsonl(str(out / "records.jsonl"), [dict(question_hash="q", direct_correct=direct,
                    policy=dict(correct=policy, calls=0, valid=True), costs=[])])
        (out / "summary.json").write_text(json.dumps(dict(n=1, independent_accuracy=int(direct), policy_accuracy=int(policy), mean_calls=0)))
        (out / ".rsi_complete.json").write_text(json.dumps(dict(wall_seconds=2)))
        plan.append(dict(stage="assess", output=str(out)))
    (tmp_path / "rsi_run.json").write_text(json.dumps(dict(plan=plan, rounds=2)))
    result = report(tmp_path)
    assert result["complete"]
    assert result["timeline"][1]["independent_new"] == 1
    assert result["timeline"][0]["delegation_rescues"] == 1
    assert result["paired_final_differences"][0]["difference"] == 1
    assert (tmp_path / "pilot_timeline.csv").exists()
    write_jsonl(str(tmp_path / "static/round_2/grpo_dev/records.jsonl"), [dict(question_hash="different")])
    with pytest.raises(ValueError, match="sets differ"):
        report(tmp_path)
