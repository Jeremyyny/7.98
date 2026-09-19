"""Validate that the predeclared supplemental experiment has observed results."""
from pathlib import Path
import json
from collections import Counter

from .data import SOURCES
from .reporting import generate_report
from .telemetry import atomic_json


def paper_check(run_dirs, output, min_seeds=2):
    if min_seeds < 2:
        raise ValueError("Paper check requires at least two independent training seeds")
    out = Path(output)
    report = generate_report(run_dirs, out)
    failures, seeds, protocols = [], {"dynamic_sft": set(), "success_sft": set()}, set()
    for run_dir in run_dirs:
        root = Path(run_dir)
        run = json.loads((root / "loop.json").read_text())
        cfg, data = run["config"], run["data_manifest"]
        if run["arm"] not in {*seeds, "static_sft"}:
            failures.append(f"{root.name}: unsupported paper arm")
        if run["arm"] in seeds:
            if cfg["seed"] in seeds[run["arm"]]:
                failures.append(f"{root.name}: duplicate seed")
            seeds[run["arm"]].add(cfg["seed"])
        if run.get("harness", {}).get("protocol_version") != 2:
            failures.append(f"{root.name}: missing protocol-v2 harness identity")
        if data.get("smoke_only") or not cfg.get("base_model_revision"):
            failures.append(f"{root.name}: smoke data or unpinned base model")
        for name, (dataset, split) in SOURCES.items():
            source = data.get("sources", {}).get(name, {})
            if source.get("dataset") != dataset or not source.get("revision") or source.get("source_split") != split:
                failures.append(f"{root.name}: {name} is not a pinned official source")
        if cfg.get("max_depth", 0) < 2 or not cfg.get("distill_solutions") or run.get("rounds", 0) < 2:
            failures.append(f"{root.name}: paper design requires depth >= 2, solution distillation and two rounds")
        rounds = run.get("rounds", 0)
        expected_stages = {"diagnose": rounds + 1, "sft": rounds,
                           "collect": 1 if run["arm"] == "static_sft" else rounds}
        if Counter(step["stage"] for step in run["plan"]) != expected_stages:
            failures.append(f"{root.name}: plan does not match the predeclared SFT-only rounds")
        for step in run["plan"]:
            path = Path(step["output"])
            if not (path / ".stage_complete.json").exists():
                failures.append(f"{root.name}: incomplete {step['stage']} stage")
            if step["stage"] == "sft":
                from .runner import validate_stage_artifacts
                try:
                    validate_stage_artifacts(path, "sft")
                except ValueError as exc:
                    failures.append(f"{root.name}: {exc}")
                metrics_path = path / "training_metrics.json"
                trained = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
                if cfg.get("sft_max_steps", 0) <= 0 or trained.get("optimizer_steps") != cfg.get("sft_max_steps"):
                    failures.append(f"{root.name}: SFT optimizer budget was not completed")
                if not (path / "sft_data_report.json").exists() or not (path / "usage.jsonl").exists():
                    failures.append(f"{root.name}: missing SFT data/token accounting")
        for label in ("initial", "final"):
            for test in ("aime2026", "beyondaime"):
                if not (root / "test" / label / test / ".stage_complete.json").exists():
                    failures.append(f"{root.name}: missing locked {label}/{test} evaluation")
        advisor_path = root / "advisor_identity.json"
        comparable = {k: v for k, v in cfg.items() if k not in {"seed", "advisor_url"}}
        comparable.update(harness=run.get("harness"), data=data.get("sha256"),
                          rounds=run.get("rounds"), initial=run.get("initial"),
                          advisor=json.loads(advisor_path.read_text()) if advisor_path.exists() else None)
        if comparable["advisor"] is None:
            failures.append(f"{root.name}: missing frozen subagent identity")
        protocols.add(json.dumps(comparable, sort_keys=True))
    matched = seeds["dynamic_sft"] & seeds["success_sft"]
    if len(matched) < min_seeds or seeds["dynamic_sft"] != seeds["success_sft"]:
        failures.append("Need the same predeclared seeds for dynamic_sft and success_sft, with enough repeats")
    if len(protocols) != 1:
        failures.append("Runs differ in model, data, harness, subagent identity or training/generation budgets")
    result = {"complete": not failures, "failures": failures, "matched_seeds": sorted(matched),
              "report": report, "scope": "Completeness and protocol consistency only; no guarantee of positive gains, statistical significance, or acceptance."}
    atomic_json(out / "paper_readiness.json", result)
    return result
