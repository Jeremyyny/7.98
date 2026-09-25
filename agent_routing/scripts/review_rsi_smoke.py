"""Publish existing smoke evidence to one W&B review run without training."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def review(source, target):
    from src.verifiable.rollout_reporting import reward_metrics, rollout_record
    from src.verifiable.telemetry import Monitor, atomic_json, metrics
    from src.verifiable.wandb_tracking import text_tracking_enabled, tracking_mode
    from src.verifiable.debug_records import question_info
    if tracking_mode() == "disabled" or not text_tracking_enabled():
        raise ValueError("Enable W&B and text logging; use review_rsi_smoke.sh")
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source in target.parents:
        raise ValueError("Review output must be outside original experiment")
    if not (source / "config.json").is_file():
        raise ValueError("Source smoke config not found")
    # Snapshot an explicit allowlist. No credentials, environment or arbitrary logs.
    paths = [source / n for n in ("config.json", "smoke_status.json", "smoke_report.json")]
    for stage in ("collection", "sft", "grpo", "after_grpo", "next_sft"):
        paths += [source / stage / name for name in
                  ("status.json", "training_metrics.json", "summary.json", "records.jsonl", "generations.jsonl")]
    paths += list(source.glob("grpo/step-*/step.json")) + list(source.glob("grpo/step-*/rollouts.json"))
    paths += [source / "grpo/invalid_group.json"]
    snapshots = {str(p.relative_to(source)): p.read_bytes() for p in paths if p.is_file()}
    if not any(name.endswith("rollouts.json") or name.endswith("invalid_group.json") for name in snapshots):
        raise ValueError("No saved GRPO rollout evidence; nothing to review yet")
    def read(name, fallback=None):
        return json.loads(snapshots[name]) if name in snapshots else fallback
    def jsonl(name):
        return [json.loads(line) for line in snapshots.get(name, b"").splitlines() if line.strip()]
    target.mkdir(parents=True, exist_ok=False)
    atomic_json(target / "run.json", {"config": {"purpose": "saved_smoke_review_no_training",
        "source": str(source), "original_config": read("config.json"),
        "source_sha256": {name: hashlib.sha256(content).hexdigest() for name, content in snapshots.items()}}})
    original = read("smoke_status.json", {})
    names = ("collection", "sft", "grpo", "after_grpo", "next_sft")
    stages = {name: read(name + "/status.json", {}).get("status", "unknown") for name in names}
    artifacts = all((source / stage / name).is_file() for stage, name in
        (("collection", "sft.jsonl"), ("after_grpo", "summary.json"),
         ("sft", "training_metrics.json"), ("grpo", "training_metrics.json"), ("next_sft", "training_metrics.json")))
    completed = artifacts and all(status == "completed" for status in stages.values())
    summary = {"source": str(source), "original_status": original, "stages": stages,
        "recorded_execution_completed": completed, "paper_result": False,
        "note": "Separate review of saved records; original files/status/rewards unchanged. "
                "Execution completion, rollout validity and learning evidence are distinct."}
    questions = {r["question_hash"]: question_info(r) for r in jsonl("collection/records.jsonl")}
    groups, all_rows = [], []
    with Monitor(target, "smoke_review") as monitor:
        metrics({"execution_completed": completed}, "review")
        for stage in names:
            state = read(stage + "/status.json", {})
            metrics({"elapsed_seconds": state.get("elapsed_seconds")}, stage)
            metrics(read(stage + "/training_metrics.json", {}), stage)
            metrics(read(stage + "/summary.json", {}), stage)
            for record in jsonl(stage + "/generations.jsonl"):
                monitor.log_text("generation", {**record, "source": stage})
        for name in sorted(snapshots):
            if not name.endswith("/rollouts.json"):
                continue
            saved = read(name)
            step = read(name.replace("rollouts.json", "step.json"), {})
            advantages = step.get("advantages", [0.] * len(saved["trajectories"]))
            stats = reward_metrics(saved["trajectories"], advantages)
            index = step.get("step", len(groups) + 1)
            metrics({**step, **stats}, "grpo", trainer_step=index)
            groups.append({"step": index, **stats})
            for i, (trajectory, advantage) in enumerate(zip(saved["trajectories"], advantages)):
                record = rollout_record(index, i, saved["root"], trajectory, advantage,
                    questions.get(step.get("question_hash"), {"question_hash": step.get("question_hash")}))
                all_rows.append(record)
                monitor.log_text("rollout", record)
        # Older guarded runs may have aborted before a committed optimizer step.
        if not groups and "grpo/invalid_group.json" in snapshots:
            saved = read("grpo/invalid_group.json")
            advantages = [0.] * len(saved["trajectories"])
            stats = reward_metrics(saved["trajectories"], advantages)
            groups.append({"step": saved["step"], **stats, "committed_step": False})
            metrics(stats, "grpo", trainer_step=saved["step"])
            for i, trajectory in enumerate(saved["trajectories"]):
                record = rollout_record(saved["step"], i, saved["root"], trajectory, 0.)
                all_rows.append(record)
                monitor.log_text("rollout", record)
        summary.update(grpo_groups=groups,
            failure_counts=dict(Counter(r["failure_type"] for r in all_rows if not r["valid"])),
            reward_signal_present=any(g["mixed_reward_group"] for g in groups))
        atomic_json(target / "review_summary.json", summary)
        atomic_json(target / "rollout_review.json", all_rows)
        monitor.tracker.run.summary.update(summary)
        monitor.flush_tables(force=True)
        # Keep exact original saved evidence accessible alongside bounded UI tables.
        artifact = monitor.tracker.sdk.Artifact("smoke-review-" + monitor.attempt, type="smoke-evidence")
        copied = target / "evidence"
        for name, content in snapshots.items():
            path = copied / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            artifact.add_file(str(path), name=name)
        artifact.add_file(str(target / "review_summary.json"), name="review_summary.json")
        artifact.add_file(str(target / "rollout_review.json"), name="rollout_review.json")
        monitor.tracker.run.log_artifact(artifact)
    if monitor.wandb_failed:
        raise RuntimeError("W&B upload failed; original data remains intact")
    link = json.loads((target / "wandb_link.json").read_text())
    print(json.dumps({"review_url": link.get("url"), "summary": summary}, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source")
    parser.add_argument("--out")
    args = parser.parse_args()
    target = args.out or "/workspace/margent-smoke-review-" + datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    review(args.source, target)
