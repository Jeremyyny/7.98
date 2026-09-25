"""One-hour, two-train/one-dev Qwen GPU plumbing check; never paper results.

Uses its own advisor on port 8002, leaves existing services alone, and stops
only processes it starts. Run with the existing margent virtualenv Python.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import traceback
from urllib.request import urlopen

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def stop_owned(process):
    if process is None:
        return
    # Children may remain even if their process-group leader already exited.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def run_stage(command, logfile, env, deadline):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Smoke wall-time limit reached")
    with logfile.open("w") as stream:
        process = subprocess.Popen(command, cwd=REPO, env=env, stdout=stream,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = process.wait(timeout=remaining)
            if code:
                raise RuntimeError(f"Stage exited {code}; see {logfile}")
        except BaseException:
            stop_owned(process)
            raise


def gpu_check(manager_gpu, advisor_gpu):
    if manager_gpu == advisor_gpu:
        raise ValueError("Use separate physical GPU indices for Manager and advisor")
    raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,memory.free",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    devices = {parts[0]: (parts[1], int(parts[2])) for line in raw.splitlines()
               if (parts := [part.strip() for part in line.split(",")]) and len(parts) == 3}
    if manager_gpu not in devices or advisor_gpu not in devices:
        raise ValueError("Requested GPUs not present")
    processes = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
        "--format=csv,noheader,nounits"], text=True, timeout=15)
    if any(line.split(",")[0].strip() == devices[manager_gpu][0] for line in processes.splitlines()):
        raise RuntimeError("Manager GPU is occupied; do not start a duplicate experiment")
    if devices[manager_gpu][1] < 60000 or devices[advisor_gpu][1] < 22000:
        raise RuntimeError("Need >=60000 MiB free on Manager GPU and >=22000 on advisor GPU")
    return {"devices": devices, "existing_compute_processes": processes}


def evidence(root):
    import torch
    from safetensors.torch import load_file
    from src.verifiable.runner import validate_stage_artifacts
    for name, kind in (("collection", "collect"), ("sft", "sft"),
                       ("grpo", "rl"), ("after_grpo", "assess"), ("next_sft", "sft")):
        validate_stage_artifacts(root / name, kind)
    adapters = {name: load_file(str(root / name / "adapter_model.safetensors"), device="cpu")
                for name in ("sft", "grpo", "next_sft")}
    for weights in adapters.values():
        if not weights or not all(torch.isfinite(t).all().item() for t in weights.values()):
            raise ValueError("Missing or nonfinite adapter weights")
    if not any(torch.count_nonzero(v).item() for k, v in adapters["sft"].items() if "lora_B" in k):
        raise ValueError("First SFT left all LoRA B weights zero")
    changed = lambda a, b: sum(not torch.equal(adapters[a][k], adapters[b][k]) for k in adapters[a])
    grpo_changed, next_changed = changed("sft", "grpo"), changed("grpo", "next_sft")
    if not next_changed:
        raise ValueError("Next SFT did not update the reloaded GRPO adapter")
    resume = json.loads((root / "grpo/resume.json").read_text())
    step = json.loads((root / "grpo" / resume["directory"] / "step.json").read_text())
    if step["step"] != 1 or not all(math.isfinite(step[k]) for k in ("loss", "gradient_norm")):
        raise ValueError("Missing optimizer step or nonfinite GRPO metrics")
    if not step["protocol_valid"] or not all(step["protocol_valid"]):
        raise ValueError("GRPO produced invalid Manager protocol output; inspect rollouts.json")
    for name in ("sft", "next_sft"):
        metrics = json.loads((root / name / "training_metrics.json").read_text())
        if metrics["optimizer_steps"] != 1 or not math.isfinite(metrics["train_loss"]):
            raise ValueError(f"Invalid SFT training result: {name}")
    links = [json.loads(p.read_text()) for p in root.glob("*/wandb_link.json")]
    if len(links) != 6 or any(link.get("mode") != "online" or not link.get("url") for link in links):
        raise ValueError("Expected six online W&B runs for logging check and five model stages")
    learning = bool(step["mixed_reward_group"] and step["gradient_norm"] > 0 and grpo_changed)
    return {"status": "plumbing_passed", "paper_result": False,
        "grpo_learning_signal_observed": learning,
        "grpo_step": step, "grpo_changed_tensors": grpo_changed,
        "next_sft_changed_tensors": next_changed, "wandb_runs": links,
        "note": "A same-reward group can pass plumbing with zero GRPO update; that does not validate learning. "
                "Checkpoint reload is tested; interrupted optimizer recovery is not tested here. "
                "W&B links do not verify delivery of every upload or any email alert."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="/workspace/margent-rsi-smoke-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S"))
    parser.add_argument("--data-dir", default="/workspace/margent-data-restart-20260925")
    parser.add_argument("--config", default=str(REPO / "configs/math_rsi_pilot.json"))
    parser.add_argument("--manager-gpu", default="1")
    parser.add_argument("--advisor-gpu", default="0")
    parser.add_argument("--port", type=int, default=8002)
    parser.add_argument("--minutes", type=float, default=60)
    args = parser.parse_args()
    if not 0 < args.minutes <= 60:
        parser.error("--minutes must be >0 and <=60")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    advisor = None
    deadline = time.monotonic() + args.minutes * 60
    current = "preflight"
    env = os.environ.copy()
    env.update(HF_HOME="/workspace/hf-cache", HF_HUB_CACHE="/workspace/hf-cache/hub",
               HF_DATASETS_CACHE="/workspace/hf-cache/datasets", HF_HUB_DISABLE_XET="1",
               TMPDIR="/workspace/margent-tmp", PYTHONUNBUFFERED="1",
               WANDB_ENTITY="yuningyangaillm", WANDB_PROJECT="MATH_rsi",
               MARGENT_WANDB_MODE="online", MARGENT_WANDB_TEXT="1")
    Path(env["TMPDIR"]).mkdir(parents=True, exist_ok=True)
    manager_env = {**env, "CUDA_VISIBLE_DEVICES": args.manager_gpu}
    print(f"SMOKE OUTPUT: {root}", flush=True)
    try:
        write(root / "gpu_preflight.json", gpu_check(args.manager_gpu, args.advisor_gpu))
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", args.port))
        from src.verifiable.rsi import prepare_pilot
        from src.verifiable.runner import load_config
        prepare_pilot(args.data_dir, root / "data", train_n=2, dev_n=1)
        cfg = load_config(args.config)
        cfg.update(advisor_url=f"http://127.0.0.1:{args.port}", sft_max_steps=1,
                   sft_accumulation=1, save_steps=1, rl_max_steps=1)
        config = root / "config.json"
        write(config, cfg)
        write(root / "smoke_manifest.json", {"purpose": "GPU plumbing only, not accuracy benchmark",
            "train_n": 2, "dev_n": 1, "max_minutes": args.minutes, "config": cfg,
            "source_config": str(Path(args.config).resolve())})

        def stage(name, command):
            nonlocal current
            current = name
            write(root / "smoke_status.json", {"status": "running", "stage": name})
            print(f"[smoke] {name}; log: {root / 'logs' / (name + '.log')}", flush=True)
            run_stage([sys.executable, *command], root / "logs" / (name + ".log"), manager_env, deadline)

        stage("wandb_check", ["-m", "src.verifiable", "wandb-check", "--out", str(root / "wandb_check")])
        current = "advisor_start"
        write(root / "smoke_status.json", {"status": "running", "stage": current})
        print(f"[smoke] advisor_start; log: {root / 'logs/advisor.log'}", flush=True)
        with (root / "logs/advisor.log").open("w") as stream:
            advisor = subprocess.Popen([sys.executable, "-m", "src.verifiable.serve", "--model", cfg["base_model"],
                "--revision", cfg["base_model_revision"], "--max-context", str(cfg["max_context"]),
                "--port", str(args.port)], cwd=REPO, env={**env, "CUDA_VISIBLE_DEVICES": args.advisor_gpu},
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        ready_deadline = min(deadline, time.monotonic() + 300)
        while True:
            if advisor.poll() is not None:
                raise RuntimeError("Smoke advisor exited; inspect logs/advisor.log")
            if time.monotonic() >= ready_deadline:
                raise TimeoutError("Smoke advisor did not become ready within startup/time budget")
            try:
                with urlopen(cfg["advisor_url"] + "/health", timeout=2) as response:
                    if json.load(response).get("status") == "ready":
                        break
            except OSError:
                pass
            time.sleep(1)
        stage("doctor", ["-m", "src.verifiable", "doctor", "--config", str(config), "--out", str(root / "environment.json")])

        def model_stage(name, kind, checkpoint, data):
            stage(name, ["-m", "src.verifiable.rsi", "stage", kind, "--config", str(config),
                        "--checkpoint", str(checkpoint), "--data", str(data), "--out", str(root / name)])

        model_stage("collection", "collect", cfg["base_model"], root / "data/train.jsonl")
        model_stage("sft", "sft", cfg["base_model"], root / "collection/sft.jsonl")
        model_stage("grpo", "grpo", root / "sft", root / "data/train.jsonl")
        model_stage("after_grpo", "assess", root / "grpo", root / "data/dev.jsonl")
        model_stage("next_sft", "sft", root / "grpo", root / "collection/sft.jsonl")
        current = "validate"
        report = evidence(root)
        write(root / "smoke_report.json", report)
        write(root / "smoke_status.json", {"status": report["status"], "stage": "complete"})
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    except BaseException as exc:
        write(root / "smoke_status.json", {"status": "failed", "stage": current, "error": str(exc)})
        (root / "error.log").write_text(traceback.format_exc())
        raise
    finally:
        stop_owned(advisor)


if __name__ == "__main__":
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    main()
