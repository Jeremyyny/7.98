"""Question-aligned descriptive diagnostics; no gold information enters policies."""
from __future__ import annotations


def state_of(row):
    if row["direct_correct"]:
        return "independent"
    return "rescuable" if any(b["correct"] for b in row.get("branches", [])) else "unresolved"


def call_stats(rows):
    n = len(rows)
    return {"n": n, "call_rate": sum(r["policy"]["calls"] > 0 for r in rows) / n if n else None,
            "mean_calls": sum(r["policy"]["calls"] for r in rows) / n if n else None,
            "policy_accuracy": sum(r["policy"]["correct"] for r in rows) / n if n else None,
            "valid_failure_rate": sum(r["policy"]["valid"] and not r["policy"]["correct"] for r in rows) / n if n else None,
            "no_call_success_rate": sum(r["policy"]["calls"] == 0 and r["policy"]["correct"] for r in rows) / n if n else None,
            "invalid_rate": sum(not r["policy"]["valid"] for r in rows) / n if n else None}


def conditional_metrics(records, initial=None):
    out = {}
    if not records or not all("policy" in r for r in records):
        return out
    groups = {"currently_independent": [r for r in records if r["direct_correct"]]}
    if all("branches" in r for r in records):
        groups.update({"currently_" + state: [r for r in records if state_of(r) == state]
                       for state in ("rescuable", "unresolved")})
    if initial is not None:
        before = {r["question_hash"]: r for r in initial}
        after = {r["question_hash"]: r for r in records}
        if len(before) != len(initial) or len(after) != len(records) or set(before) != set(after):
            raise ValueError("Conditional metrics require unique, identical held-out question sets")
        rescued = {k for k, r in before.items() if state_of(r) == "rescuable"}
        learned = {k for k in rescued if after[k]["direct_correct"]}
        groups["initially_rescued_now_independent"] = [after[k] for k in sorted(learned)]
        groups["initially_rescued_still_rescuable"] = [after[k] for k in sorted(rescued) if state_of(after[k]) == "rescuable"]
        groups["initially_rescued_now_unresolved"] = [after[k] for k in sorted(rescued) if state_of(after[k]) == "unresolved"]
        # Compare policy calls on exactly the same learned subset at both times.
        groups["learned_subset_before"] = [before[k] for k in sorted(learned)]
        out.update(initial_rescued_n=len(rescued), rescued_now_independent_n=len(learned),
                   internalization_rate=len(learned) / len(rescued) if rescued else None)
        for old in ("independent", "rescuable", "unresolved"):
            for new in ("independent", "rescuable", "unresolved"):
                out[f"transition_{old}_to_{new}_n"] = sum(state_of(before[k]) == old and state_of(after[k]) == new for k in before)
    for name, rows in groups.items():
        for key, value in call_stats(rows).items():
            out[f"{name}_{key}"] = value
    return out


def pilot_cost(collect_dir, diagnose_dir, train_size=128, dev_size=64, arms=2, seeds=2, rounds=2):
    """Estimate generation time from an actual pilot, without claiming GPU FLOPs."""
    import json
    from pathlib import Path
    if min(train_size, dev_size, arms, seeds, rounds) < 1:
        raise ValueError("Pilot estimate sizes and run counts must be positive")
    def per_question(directory):
        rows = [json.loads(line) for line in (Path(directory) / "attempts.jsonl").read_text().splitlines()]
        n = sum(r["completed_examples_this_attempt"] for r in rows)
        if not n:
            raise ValueError("Pilot has no newly processed examples")
        return sum(r["wall_seconds"] for r in rows) / n
    collect_s, diagnose_s = per_question(collect_dir), per_question(diagnose_dir)
    count = arms * seeds
    hours = count * (rounds * train_size * collect_s + (rounds + 1) * dev_size * diagnose_s) / 3600
    return {"estimated_collection_and_dev_hours": hours, "collection_seconds_per_question": collect_s,
            "diagnosis_seconds_per_question": diagnose_s, "runs": count,
            "scope": "Excludes SFT, model loads, external tests, retries and throughput changes after learning; not a completion-time guarantee."}
