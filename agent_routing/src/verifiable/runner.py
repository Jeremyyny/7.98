"""Resumable per-example collection and subprocess-isolated loop stages."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from ..utils.io import append_jsonl, read_jsonl, write_json, write_jsonl
from .backend import HFBackend, HTTPAdvisors
from .data import identity, load_rows, verify_manifest
from .experiment import collect_one, compare, policy_rollout, root_state, sft_rows, summary


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    for key in ("base_model", "advisor_url", "max_new_tokens", "advisor_max_tokens", "max_context",
                "max_seq_len", "max_depth", "seed"):
        if key not in cfg:
            raise ValueError(f"Missing config field: {key}")
    if cfg["max_depth"] not in (1, 2, 3):
        raise ValueError("max_depth must be 1, 2 or 3")
    for key in ("max_new_tokens", "advisor_max_tokens", "max_context", "max_seq_len"):
        if cfg[key] <= 0:
            raise ValueError(f"{key} must be positive")
    return cfg


def checkpoint_identity(checkpoint):
    path = Path(checkpoint)
    if not path.is_dir():
        return {"model_id": checkpoint}
    # Adapter weights are small enough to hash; full model shards use stat + config.
    return {p.name: (_digest(p) if p.suffix == ".json" or p.name.startswith("adapter_model") else
                     {"bytes": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns})
            for p in sorted(path.iterdir()) if p.is_file() and p.suffix in {".json", ".safetensors", ".bin"}}


def run_data(cfg, data, checkpoint, output, mode, resume=False, limit=0,
             selection="counterfactual", backend=None, advisors=None):
    allowed = {"collect": "train", "diagnose": "dev", "evaluate": "test"}
    rows = load_rows(data, required_split=allowed[mode])
    rows = sorted(rows, key=lambda r: identity(r.question))
    if limit > 0:
        rows = rows[:limit]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    signature = {"config": cfg, "data_sha256": _digest(data), "checkpoint": checkpoint_identity(checkpoint),
                 "mode": mode, "limit": limit, "selection": selection, "protocol_version": 1}
    meta = root / "run.json"
    if meta.exists():
        if not resume or json.loads(meta.read_text()) != signature:
            raise ValueError(f"Existing run differs or --resume missing: {output}")
    else:
        if any(root.iterdir()):
            raise ValueError("Output directory is not empty and has no matching run manifest")
        write_json(str(meta), signature)
    path = root / "records.jsonl"
    records = read_jsonl(str(path)) if path.exists() else []
    seen = {r["question_hash"] for r in records}
    if len(seen) != len(records) or seen - {identity(r.question) for r in rows}:
        raise ValueError("Invalid resume record identities")
    pending = [r for r in rows if identity(r.question) not in seen]
    if pending:
        backend = backend or HFBackend(cfg["base_model"], checkpoint, cfg["max_context"])
        advisors = advisors or HTTPAdvisors(cfg["advisor_url"], cfg["advisor_max_tokens"], cfg.get("advisor_models"))
    started = time.monotonic()
    for row in pending:
        seed = (cfg["seed"] + int(identity(row.question)[:8], 16)) % (2 ** 31)
        if mode == "evaluate":
            direct, history = root_state(row, backend, cfg, seed)
            from .answers import correct
            policy = policy_rollout(row, backend, advisors, cfg, seed, direct, history)
            record = {"question_hash": identity(row.question), "example_id": row.example_id,
                      "benchmark_name": row.benchmark_name, "split": row.split,
                      "direct_correct": bool(direct["valid"] and correct(direct["text"], row.ground_truth)),
                      "direct_text": direct["text"], "policy": policy,
                      "costs": [{"role": "manager", **direct}] + policy["costs"]}
        else:
            record = collect_one(row, backend, advisors, cfg, seed, evaluate_policy=mode == "diagnose")
        append_jsonl(str(path), [record])
        records.append(record)
        print(f"[{mode}] {len(records)}/{len(rows)} direct={record['direct_correct']}", flush=True)
    result = summary(records)
    result.update(checkpoint=checkpoint, verification_scope="terminal_answer", mode=mode)
    if mode == "collect":
        turns = sft_rows(records, cfg["seed"], cfg.get("commit_rescue_ratio", 1),
                         cfg.get("distill_solutions", True), selection)
        if not turns:
            raise ValueError("No successful trajectories: inspect pilot before launching training")
        write_jsonl(str(root / "sft.jsonl"), turns)
        result["sft_turns"] = len(turns)
    append_jsonl(str(root / "attempts.jsonl"), [{"completed_examples_this_attempt": len(pending),
                                               "wall_seconds": time.monotonic() - started}])
    write_json(str(root / "summary.json"), result)
    return result


def build_plan(config_path, data_dir, output, arm, rounds, initial=None):
    root = Path(output).resolve()
    data = Path(data_dir).resolve()
    cfg = load_config(config_path)
    checkpoint = initial or cfg["base_model"]
    prefix = [sys.executable, "-m", "src.verifiable"]
    common = ["--config", str(Path(config_path).resolve())]
    plan = []

    def add(stage, input_path, ckpt, out, extra=()):
        command = prefix + [stage] + common + ["--data", str(input_path), "--checkpoint", ckpt,
                  "--out", str(out)] + list(extra)
        plan.append({"stage": stage, "output": str(out), "command": command})

    add("diagnose", data / "dev.jsonl", checkpoint, root / "initial_dev", ["--resume"])
    for n in range(1, rounds + 1):
        rd = root / f"round_{n}"
        collection = root / "round_1" / "collection" if arm == "static_rl" else rd / "collection"
        if n == 1 or arm != "static_rl":
            add("collect", data / "train.jsonl", checkpoint, collection,
                ["--resume", "--selection", "success" if arm == "success_rl" else "counterfactual"])
        add("sft", collection / "sft.jsonl", checkpoint, rd / "sft")
        checkpoint = str(rd / "sft")
        add("diagnose", data / "dev.jsonl", checkpoint, rd / "sft_dev", ["--resume"])
        if arm != "dynamic_sft":
            add("rl", data / "train.jsonl", checkpoint, rd / "rl")
            checkpoint = str(rd / "rl")
            add("diagnose", data / "dev.jsonl", checkpoint, rd / "rl_dev", ["--resume"])
    return plan


def run_loop(config_path, data_dir, output, arm, rounds, initial=None, resume=False, dry_run=False):
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    manifest = verify_manifest(data_dir)
    plan = build_plan(config_path, data_dir, output, arm, rounds, initial)
    if dry_run:
        for step in plan:
            print(shlex.join(step["command"]))
        return plan
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    signature = {"config": load_config(config_path), "data_manifest": manifest, "arm": arm,
                 "rounds": rounds, "initial": initial, "plan": plan}
    runfile = root / "loop.json"
    if runfile.exists():
        if not resume or json.loads(runfile.read_text()) != signature:
            raise ValueError("Loop settings changed or --resume is missing; choose another output")
    else:
        if any(root.iterdir()):
            raise ValueError("Loop output is not empty")
        write_json(str(runfile), signature)
    for step in plan:
        out = Path(step["output"])
        done = out / ".stage_complete.json"
        if done.exists():
            if json.loads(done.read_text())["command"] != step["command"]:
                raise ValueError("Completed stage command changed")
            continue
        print(shlex.join(step["command"]), flush=True)
        start = time.monotonic()
        # A separate process releases all manager weights before the next stage.
        subprocess.run(step["command"], check=True)
        write_json(str(done), {"command": step["command"], "wall_seconds": time.monotonic() - start})
    checkpoints = [step for step in plan if step["stage"] == "diagnose"]
    reports = []
    before = read_jsonl(str(Path(checkpoints[0]["output"]) / "records.jsonl"))
    for step in checkpoints[1:]:
        after = read_jsonl(str(Path(step["output"]) / "records.jsonl"))
        reports.append({"checkpoint": step["output"], "vs_initial": compare(before, after),
                        "summary": json.loads((Path(step["output"]) / "summary.json").read_text())})
    write_json(str(root / "loop_report.json"), reports)
    return reports
