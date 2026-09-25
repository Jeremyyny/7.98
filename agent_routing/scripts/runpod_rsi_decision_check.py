"""Read-only GPU replay of saved decision states: raw vs finite-action syntax.

No training, no advisor calls, no rewards or answer labels supplied to the model.
The finite grammar is a changed decoding policy, NOT a claim of learned skill.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def decision_error(result):
    from src.verifiable.protocol import COMMIT, parse_calls
    if result.get("truncated"):
        return "truncated_decision"
    try:
        content, calls = parse_calls(result["text"])
    except ValueError as exc:
        return str(exc)
    if not calls:
        return None if content == COMMIT else "expected_COMMIT_or_tool_call"
    if len(calls) != 1 or content:
        return "decision_must_be_one_bare_tool_call"
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="/workspace/margent-rsi-smoke-01")
    parser.add_argument("--out", required=True)
    parser.add_argument("--require-mode", choices=("none", "finite_actions_v1"), default="finite_actions_v1")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle Manager GPU using CUDA_VISIBLE_DEVICES=1")
    gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not gpu.isdigit():
        raise ValueError("Use one physical numeric GPU index")
    uuid = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True, timeout=10).strip()
    active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader"], text=True, timeout=10)
    if any(line.split(",")[0].strip() == uuid and line.split(",")[1].strip() != str(os.getpid()) for line in active.splitlines()):
        raise RuntimeError("Manager GPU already has another process; refusing duplicate load")
    source, root = Path(args.source).resolve(), Path(args.out).resolve()
    cfg = json.loads((source / "config.json").read_text())
    records = [json.loads(line) for line in (source / "collection/records.jsonl").read_text().splitlines() if line]
    if len(records) != 2 or any(r.get("split") != "train" for r in records):
        raise ValueError("Expected the two saved smoke training states")
    checkpoint = source / "sft"
    if not (checkpoint / "adapter_config.json").is_file():
        raise ValueError("Saved SFT checkpoint missing")
    root.mkdir(parents=True, exist_ok=False)
    from src.verifiable.backend import load_model
    from src.verifiable.protocol import tool_schemas
    from src.verifiable.provenance import harness_identity
    from src.verifiable.runner import checkpoint_identity
    from src.verifiable.rsi_grpo import RolloutBackend
    from src.verifiable.telemetry import Monitor, atomic_json, generation, metrics, progress
    atomic_json(root / "run.json", {"config": cfg, "harness": harness_identity(),
        "purpose": "protocol replay only; new grammar is not a learned improvement",
        "source": str(source), "checkpoint": checkpoint_identity(str(checkpoint)),
        "collection_sha256": hashlib.sha256((source / "collection/records.jsonl").read_bytes()).hexdigest()})
    results = []
    try:
        with Monitor(root, "decision_check"):
            tok, model = load_model(cfg["base_model"], str(checkpoint), revision=cfg["base_model_revision"])
            backend = RolloutBackend(tok, model, cfg)
            for mode in ("none", "finite_actions_v1"):
                backend.decision_constraint = mode
                for row in records:
                    # Exact same stored state in both conditions; no answer key.
                    for sample in range(5):
                        decoding = "greedy" if sample == 0 else "sampled"
                        backend.turns = None if sample == 0 else []
                        seed = cfg["seed"] + sample
                        progress(condition=mode, decoding=decoding, completed=len(results), total=20)
                        result = backend.generate(row["base_messages"], tools=tool_schemas(),
                            max_tokens=cfg["decision_max_tokens"], seed=seed)
                        error = decision_error(result)
                        item = {"condition": mode, "decoding": decoding, "seed": seed,
                            "question_hash": row["question_hash"], "error": error, **result,
                            "exact_turns": backend.turns, "eos_token_id": tok.eos_token_id}
                        results.append(item)
                        atomic_json(root / "decision_outputs.json", results)
                        generation("manager", {**result, "valid": error is None}, messages=row["base_messages"],
                                   operation="decision_check", condition=mode, decoding=decoding)
                        print(json.dumps({k: v for k, v in item.items() if k != "exact_turns"}, ensure_ascii=False), flush=True)
            counts = {mode: {"valid": sum(r["error"] is None for r in results if r["condition"] == mode),
                             "total": sum(r["condition"] == mode for r in results)}
                      for mode in ("none", "finite_actions_v1")}
            report = {"status": "passed" if counts[args.require_mode]["valid"] == 10 else "failed",
                "required_mode": args.require_mode,
                "counts": counts, "gpu": torch.cuda.get_device_name(0),
                "scope": "syntax only; no advisor response, weight update, or benchmark improvement tested"}
            atomic_json(root / "decision_report.json", report)
            metrics({mode + "/valid_rate": c["valid"] / c["total"] for mode, c in counts.items()}, "protocol")
            if report["status"] != "passed":
                raise RuntimeError("Required action mode failed; inspect decision_outputs.json")
            print(json.dumps(report, indent=2), flush=True)
    except BaseException as exc:
        if not (root / "decision_report.json").exists():
            atomic_json(root / "decision_report.json", {"status": "failed", "error": str(exc), "completed": len(results)})
        raise


if __name__ == "__main__":
    os.environ.setdefault("HF_HOME", "/workspace/hf-cache")
    os.environ.setdefault("HF_HUB_CACHE", "/workspace/hf-cache/hub")
    os.environ.setdefault("WANDB_ENTITY", "yuningyangaillm")
    os.environ.setdefault("WANDB_PROJECT", "MATH_rsi")
    os.environ["MARGENT_WANDB_MODE"] = "online"
    os.environ["MARGENT_WANDB_TEXT"] = "1"
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    main()
