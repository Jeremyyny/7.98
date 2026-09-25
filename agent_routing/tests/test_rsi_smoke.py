import importlib.util
import json
from pathlib import Path
import os
import subprocess
import sys
import time

import pytest

spec = importlib.util.spec_from_file_location("rsi_smoke", Path(__file__).parents[1] / "scripts/runpod_rsi_smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


def test_timed_out_stage_is_terminated(tmp_path):
    pidfile = tmp_path / "pid"
    code = f"import os,time; open({str(pidfile)!r}, 'w').write(str(os.getpid())); time.sleep(30)"
    with pytest.raises(subprocess.TimeoutExpired):
        smoke.run_stage([sys.executable, "-c", code], tmp_path / "log", os.environ.copy(), time.monotonic() + 1)
    with pytest.raises(ProcessLookupError):
        os.kill(int(pidfile.read_text()), 0)


def test_busy_manager_gpu_refuses_duplicate(monkeypatch):
    replies = iter(["0, GPU-advisor, 63000\n1, GPU-manager, 70000\n", "GPU-manager, 1234\n"])
    monkeypatch.setattr(subprocess, "check_output", lambda *a, **k: next(replies))
    with pytest.raises(RuntimeError, match="occupied"):
        smoke.gpu_check("1", "0")


def fixtures(root):
    import torch
    from safetensors.torch import save_file
    for name in ("wandb_check", "collection", "sft", "grpo", "after_grpo", "next_sft"):
        stage = root / name
        stage.mkdir()
        smoke.write(stage / "wandb_link.json", {"mode": "online", "url": "https://wandb.ai/example"})
        for file in ("summary.json", "adapter_config.json"):
            smoke.write(stage / file, {})
        for file in ("records.jsonl", "sft.jsonl"):
            (stage / file).write_text("{}\n")
        smoke.write(stage / "training_metrics.json", {"optimizer_steps": 1, "train_loss": .5})
    for name, value in (("sft", 1.), ("grpo", 1.), ("next_sft", 2.)):
        save_file({"layer.lora_B.weight": torch.tensor([value])}, str(root / name / "adapter_model.safetensors"))
    (root / "grpo/step-1").mkdir()
    smoke.write(root / "grpo/resume.json", {"directory": "step-1"})
    smoke.write(root / "grpo/step-1/step.json", {"step": 1, "loss": 0., "gradient_norm": 0.,
                "mixed_reward_group": False, "protocol_valid": [True, True]})


def test_zero_advantage_is_not_reported_as_learning(tmp_path):
    fixtures(tmp_path)
    result = smoke.evidence(tmp_path)
    assert result["status"] == "plumbing_passed"
    assert result["grpo_learning_signal_observed"] is False
    assert result["grpo_changed_tensors"] == 0
    assert result["next_sft_changed_tensors"] == 1


def test_invalid_rollout_cannot_pass_smoke(tmp_path):
    fixtures(tmp_path)
    path = tmp_path / "grpo/step-1/step.json"
    step = json.loads(path.read_text())
    step["protocol_valid"] = [False, False]
    smoke.write(path, step)
    with pytest.raises(ValueError, match="invalid Manager"):
        smoke.evidence(tmp_path)
