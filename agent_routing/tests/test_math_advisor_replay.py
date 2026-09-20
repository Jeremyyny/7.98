import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import replay_math_advisor as replay


def source(tmp_path):
    stage = tmp_path / "old" / "initial_dev"
    stage.mkdir(parents=True)
    record = {"role": "advisor", "advisor": "verifier", "truncated": True,
              "question": "Saved question", "context": "", "ground_truth": "SECRET_GOLD",
              "sequence": ["verifier"], "attempt": "old_attempt", "question_hash": "hash",
              "messages": [{"role": "system", "content": "Saved system prompt"},
                           {"role": "user", "content": "Saved question and derivation"}]}
    (stage / "generations.jsonl").write_text(json.dumps(record) + "\n")
    (stage / "run.json").write_text(json.dumps({"config": {
        "base_model": "fake", "base_model_revision": "fixed-revision", "max_context": 32768}}))
    return stage, record


@pytest.mark.parametrize("identical", [False, True])
@pytest.mark.parametrize("selected", [None, "fixed_sampled", "native_sampled"])
def test_replay_uses_exact_failed_messages_and_keeps_source_unchanged(tmp_path, monkeypatch, identical, selected):
    stage, record = source(tmp_path)
    before = {p.name: p.read_bytes() for p in stage.iterdir()}
    out = tmp_path / "review"
    monkeypatch.setenv("MARGENT_WANDB_MODE", "disabled")
    fixed, native = [SimpleNamespace(chat_template=t, pad_token_id=8, eos_token_id=7,
                    apply_chat_template=lambda *a, value=t, **kw: value) for t in ("fixed", "native")]
    if identical:
        native.apply_chat_template = fixed.apply_chat_template
    model = object()
    monkeypatch.setattr(replay, "HFBackend", lambda *a, **kw: SimpleNamespace(tokenizer=fixed, model=model))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: native)))
    calls = []
    def draw(m, tok, msgs, cap, context, seed, sampled):
        assert msgs == record["messages"]
        assert "SECRET_GOLD" not in json.dumps(msgs)
        assert m is model
        calls.append((tok.chat_template, sampled, cap, seed))
        return {"text": "diagnostic output", "raw_text": "diagnostic output<|im_end|>",
                "prompt_tokens": 12, "completion_tokens": 8, "seconds": 0.1,
                "truncated": not sampled, "prompt_sha256": tok.chat_template,
                "rendered_prompt": "actual template and saved messages"}
    monkeypatch.setattr(replay, "draw", draw)
    argv = ["replay", "--run-dir", str(stage), "--out", str(out)]
    if selected:
        argv += ["--variants", selected, "--max-tokens", "2048"]
    monkeypatch.setattr(sys, "argv", argv)
    replay.main()
    templates = ("fixed",) if identical else ("fixed", "native")
    expected = ([(selected.split("_")[0], True, 2048, 42)] if selected else
                [(t, s, 512, 42) for t in templates for s in (False, True)])
    assert calls == expected
    assert before == {p.name: p.read_bytes() for p in stage.iterdir()}
    summary = json.loads((out / "replay_summary.json").read_text())
    assert len(summary) == len(expected)
    assert sum(r["truncated"] for r in summary) == (0 if selected else len(templates))
    rows = [json.loads(line) for line in (out / "generations.jsonl").read_text().splitlines()]
    assert [r["phase"] for r in rows] == ([selected] if selected else
                                          [v[0] for v in replay.variants() if v[1] in templates])
    assert all(r["operation"] == "diagnostic_replay" for r in rows)
    assert json.loads((out / ((selected or "fixed_sampled") + ".json")).read_text())["raw_text"].endswith("<|im_end|>")
    assert not (out / "records.jsonl").exists()


def test_source_requires_actual_truncated_advisor_and_pinned_revision(tmp_path):
    stage, record = source(tmp_path)
    record["truncated"] = False
    (stage / "generations.jsonl").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="No truncated"):
        replay.load_failure(stage)
    record["truncated"] = True
    (stage / "generations.jsonl").write_text(json.dumps(record))
    (stage / "run.json").write_text('{"config": {}}')
    with pytest.raises(ValueError, match="pin a model revision"):
        replay.load_failure(stage)


def test_rejects_output_inside_experiment(tmp_path, monkeypatch):
    stage, _ = source(tmp_path)
    (stage.parent / "loop.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["replay", "--run-dir", str(stage),
                                     "--out", str(stage.parent / "replay")])
    with pytest.raises(SystemExit):
        replay.main()
    assert not (stage.parent / "replay").exists()


def test_presence_penalty_ignores_prompt_and_counts_generated_tokens_once():
    # Small array adapter tests the processor math without requiring CUDA/PyTorch.
    class Array:
        def __init__(self, a):
            self.a = np.asarray(a)
            self.shape = self.a.shape
        def __getitem__(self, key):
            return Array(self.a[key])
        def unique(self):
            return np.unique(self.a)
        def clone(self):
            return self.a.copy()
    tokens = Array([[0, 1, 3, 3], [2, 3, 1, 2]])
    scores = Array(np.zeros((2, 5)))
    actual = replay.presence_penalty(2, 1.5)(tokens, scores)
    np.testing.assert_equal(actual, [[0, 0, 0, -1.5, 0], [0, -1.5, -1.5, 0, 0]])
    np.testing.assert_equal(scores.a, np.zeros((2, 5)))
    np.testing.assert_equal(replay.presence_penalty(4, 1.5)(tokens, scores), scores.a)
