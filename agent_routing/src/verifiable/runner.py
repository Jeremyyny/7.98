"""Resumable per-example collection and subprocess-isolated loop stages."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import signal
import shlex
import subprocess
import sys
import time

from ..utils.io import append_jsonl, read_jsonl, write_json, write_jsonl
from .backend import HFBackend, HTTPAdvisors
from .data import identity, load_rows, verify_manifest
from .experiment import collect_one, compare, policy_rollout, root_state, sft_rows, summary
from .telemetry import Monitor, atomic_json, progress


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
    with Monitor(output, mode):
        if pending:
            backend = backend or HFBackend(cfg["base_model"], checkpoint, cfg["max_context"])
            advisors = advisors or HTTPAdvisors(cfg["advisor_url"], cfg["advisor_max_tokens"], cfg.get("advisor_models"))
        started = time.monotonic()
        for row in pending:
            progress(completed_examples=len(records), total_examples=len(rows), question_hash=identity(row.question))
            seed = (cfg["seed"] + int(identity(row.question)[:8], 16)) % (2 ** 31)
            if mode == "evaluate":
                direct, history = root_state(row, backend, cfg, seed)
                from .answers import correct
                policy = policy_rollout(row, backend, advisors, cfg, seed, direct, history)
                record = {"question_hash": identity(row.question), "example_id": row.example_id,
                          "benchmark_name": row.benchmark_name, "split": row.split,
                          "direct_correct": bool(direct["valid"] and correct(direct["text"], row.ground_truth)),
                          "direct_valid": direct["valid"], "direct_truncated": direct.get("truncated", False),
                          "direct_text": direct["text"], "policy": policy,
                          "costs": [{"role": "manager", **direct}] + policy["costs"]}
            else:
                record = collect_one(row, backend, advisors, cfg, seed, evaluate_policy=mode == "diagnose")
            append_jsonl(str(path), [record])
            records.append(record)
            progress(completed_examples=len(records), total_examples=len(rows))
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
    with Monitor(root, "loop"):
        for i, step in enumerate(plan):
            progress(stage_index=i + 1, total_stages=len(plan), current_stage=step["stage"], output=step["output"])
            if not (Path(step["output"]) / ".stage_complete.json").exists():
                verify_advisor(signature["config"], root)
            execute_stage(step, root / "logs")
        checkpoints = [step for step in plan if step["stage"] == "diagnose"]
        reports = []
        before = read_jsonl(str(Path(checkpoints[0]["output"]) / "records.jsonl"))
        for step in checkpoints[1:]:
            after = read_jsonl(str(Path(step["output"]) / "records.jsonl"))
            reports.append({"checkpoint": step["output"], "vs_initial": compare(before, after),
                            "summary": json.loads((Path(step["output"]) / "summary.json").read_text())})
        write_json(str(root / "loop_report.json"), reports)
        return reports


def execute_stage(step, log_dir):
    """Keep console output and propagate failure; never mark a failed stage done."""
    out = Path(step["output"])
    done = out / ".stage_complete.json"
    if done.exists():
        if json.loads(done.read_text())["command"] != step["command"]:
            raise ValueError("Completed stage command changed")
        validate_stage_artifacts(out, step["stage"])
        return
    print(shlex.join(step["command"]), flush=True)
    logs = Path(log_dir)
    logs.mkdir(parents=True, exist_ok=True)
    name = "_".join(out.parts[-3:]) + ".log"
    start = time.monotonic()
    with (logs / name).open("a", encoding="utf-8") as log:
        process = subprocess.Popen(step["command"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1, start_new_session=True,
                                   env={**os.environ, "PYTHONUNBUFFERED": "1"})
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end="", flush=True)
            code = process.wait()
            if code:
                raise subprocess.CalledProcessError(code, step["command"])
        except BaseException:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=5)
            raise
    validate_stage_artifacts(out, step["stage"])
    atomic_json(done, {"command": step["command"], "wall_seconds": time.monotonic() - start})


def validate_stage_artifacts(output, stage):
    names = ["training_metrics.json", "adapter_config.json"] if stage in {"sft", "rl"} else ["summary.json", "records.jsonl"]
    if stage == "collect":
        names.append("sft.jsonl")
    if stage in {"sft", "rl"} and not any((output / n).exists() for n in ("adapter_model.safetensors", "adapter_model.bin")):
        raise ValueError(f"Adapter weights missing from completed stage: {output}")
    for name in names:
        if not (output / name).exists():
            raise ValueError(f"Completed stage artifact missing: {output / name}")


def evaluate_suite(run_dir, data_dir, dry_run=False):
    root = Path(run_dir).resolve()
    run = json.loads((root / "loop.json").read_text())
    if verify_manifest(data_dir) != run["data_manifest"]:
        raise ValueError("External evaluation data differ from the frozen experiment manifest")
    training = [p for p in run["plan"] if p["stage"] in {"sft", "rl"}]
    if not training or not all((Path(p["output"]) / ".stage_complete.json").exists() for p in run["plan"]):
        raise ValueError("Complete the planned loop before the locked initial/final external evaluation")
    config_path = root / "evaluation_config.json"
    plan = []
    for label, checkpoint in [("initial", run["initial"] or run["config"]["base_model"]),
                              ("final", training[-1]["output"])]:
        for name in ("aime2026", "beyondaime"):
            output = root / "test" / label / name
            plan.append({"stage": "evaluate", "output": str(output), "command": [sys.executable, "-m",
                "src.verifiable", "evaluate", "--config", str(config_path), "--data",
                str(Path(data_dir).resolve() / f"{name}.jsonl"), "--checkpoint", checkpoint,
                "--out", str(output), "--resume"]})
    if dry_run:
        for step in plan:
            print(shlex.join(step["command"]))
        return plan
    atomic_json(config_path, run["config"])
    with Monitor(root / "test", "evaluate_suite"):
        for i, step in enumerate(plan):
            progress(stage_index=i + 1, total_stages=len(plan), output=step["output"])
            if not (Path(step["output"]) / ".stage_complete.json").exists():
                verify_advisor(run["config"], root)
            execute_stage(step, root / "logs")
    return {"evaluations": [p["output"] for p in plan]}


def verify_advisor(config, root):
    import requests
    response = requests.get(config["advisor_url"].rstrip("/") + "/health", timeout=15)
    response.raise_for_status()
    try:
        value = response.json()
    except ValueError:
        value = {}
    identity = value.get("margent_advisor")
    if identity is None:
        # External servers may provide an explicit immutable identity in config.
        identity = config.get("external_advisor_identity")
        if identity is None:
            raise ValueError("Advisor identity unavailable: use the bundled server or declare external_advisor_identity")
    path = Path(root) / "advisor_identity.json"
    if path.exists() and json.loads(path.read_text()) != identity:
        raise ValueError("Frozen advisor identity changed across stages; use the original advisor")
    atomic_json(path, identity)
    return identity
