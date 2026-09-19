"""Same-draft interventions, complete successful solutions and policy evaluation."""
from __future__ import annotations

from copy import deepcopy
import random

from ..manager.marginal_value import choose_preferred_sequence
from .answers import correct, extract_final
from .data import identity
from .protocol import (FINAL_RULE, KINDS, call_message, messages, parse_calls, tool_schemas)
from .telemetry import progress

DECIDE = "Review your current candidate. Either commit with a complete solution or request one unused sub-agent. " + FINAL_RULE
PROBE = "For this forced-commit probe, use the available evidence and finish your solution without calling tools. " + FINAL_RULE


def candidate(text):
    return text.replace("FINAL_ANSWER:", "CANDIDATE_ANSWER:")


def _draw(backend, history, cfg, seed, tools=None, budget=None):
    result = backend.generate(history, tools=tools, max_tokens=budget or cfg["max_new_tokens"],
                              temperature=cfg.get("temperature", 0), seed=seed)
    result = dict(result)
    result["valid"] = not result.get("truncated", False) and extract_final(result["text"]) is not None
    return result


def _grade(result, row):
    return bool(result["valid"] and correct(result["text"], row.ground_truth))


def root_state(row, backend, cfg, seed):
    # Independent solve is also the shared candidate for all interventions.
    progress(phase="independent", question_hash=identity(row.question))
    root = _draw(backend, messages(row, direct=True), cfg, seed)
    history = messages(row, max_calls=cfg.get("max_depth", 2)) + [{"role": "assistant", "content": candidate(root["text"])},
                                {"role": "user", "content": DECIDE}]
    return root, history


def policy_rollout(row, backend, advisors, cfg, seed, root=None, history=None):
    if root is None:
        root, history = root_state(row, backend, cfg, seed)
    history = deepcopy(history)
    used, costs = [], []
    invalid = None
    for turn in range(cfg.get("max_depth", 2) + 1):
        progress(phase="policy", policy_turn=turn, sequence=used)
        generated = _draw(backend, history, cfg, seed + 100 + turn, tools=tool_schemas())
        costs.append({"role": "manager", **generated})
        if generated.get("truncated"):
            invalid = "truncated_manager_turn"
            break
        try:
            content, calls = parse_calls(generated["text"])
        except ValueError as exc:
            invalid = str(exc)
            break
        if not calls:
            return {"correct": _grade(generated, row), "valid": generated["valid"],
                    "text": content, "calls": len(used), "sequence": used, "costs": costs,
                    "history": history + [{"role": "assistant", "content": content}]}
        if len(calls) != 1 or turn >= cfg.get("max_depth", 2) or "FINAL_ANSWER:" in content:
            invalid = "invalid_call_count_or_final_with_call"
            break
        kind = calls[0]["name"].removesuffix("_tool")
        if kind in used:
            invalid = "repeated_advisor"
            break
        args = calls[0]["arguments"]
        if set(args) - ({"current_draft"} if kind == "verifier" else set()):
            invalid = "unexpected_tool_arguments"
            break
        if kind == "verifier" and not isinstance(args.get("current_draft"), str):
            invalid = "missing_verifier_derivation"
            break
        used.append(kind)
        answer = advisors.call(kind, row, args.get("current_draft", ""))
        costs.append({"role": "advisor", **answer})
        msg = call_message(kind, args.get("current_draft", ""), f"policy_{turn}")
        msg["content"] = content
        history += [msg, {"role": "tool", "tool_call_id": f"policy_{turn}",
                          "name": kind + "_tool", "content": answer["text"]}]
    return {"correct": False, "valid": False, "text": generated["text"],
            "calls": len(used), "sequence": used, "costs": costs, "error": invalid,
            "history": history}


def collect_one(row, backend, advisors, cfg, seed, evaluate_policy=False):
    root, base = root_state(row, backend, cfg, seed)
    root_correct = _grade(root, row)
    frontier = [{"sequence": [], "history": base, "draft": root["text"], "steps": []}]
    branches, costs = [], [{"role": "manager", **root}]
    # Exhaustive bounded search (no early stop): same branch budget per checkpoint.
    for depth in range(1, cfg.get("max_depth", 2) + 1):
        next_frontier = []
        for state in frontier:
            for kind in KINDS:
                if kind in state["sequence"]:
                    continue
                seq = state["sequence"] + [kind]
                progress(phase="counterfactual", sequence=seq, completed_branches=len(branches))
                call_id = "cf_" + "_".join(seq)
                msg = call_message(kind, state["draft"], call_id)
                msg["content"] = ""
                advice = advisors.call(kind, row, state["draft"] if kind == "verifier" else "")
                costs.append({"role": "advisor", **advice})
                call_history = state["history"] + [msg, {"role": "tool", "tool_call_id": call_id,
                               "name": kind + "_tool", "content": advice["text"]}]
                revision = _draw(backend, call_history + [{"role": "user", "content": PROBE}],
                                 cfg, seed + len(branches) + 1)
                costs.append({"role": "manager", **revision})
                steps = state["steps"] + [{"prompt": state["history"], "response": [msg]}]
                branch = {"sequence": seq, "correct": _grade(revision, row),
                          "valid": revision["valid"], "text": revision["text"],
                          "truncated": revision.get("truncated", False),
                          "steps": steps, "final_prompt": call_history}
                branches.append(branch)
                next_frontier.append({"sequence": seq, "steps": steps, "draft": revision["text"],
                    "history": call_history + [{"role": "assistant", "content": candidate(revision["text"])},
                                                {"role": "user", "content": DECIDE}]})
        frontier = next_frontier
    preferred = choose_preferred_sequence(root_correct, branches, tie_break_seed=seed)
    record = {"question_hash": identity(row.question), "example_id": row.example_id,
              "benchmark_name": row.benchmark_name, "split": row.split,
              "direct_correct": root_correct, "direct_valid": root["valid"],
              "direct_truncated": root.get("truncated", False),
              "direct_text": root["text"], "base_messages": base,
              "independent_prompt": messages(row, direct=True), "branches": branches,
              "preferred_sequence": list(preferred) if preferred is not None else None,
              "costs": costs, "ground_truth": row.ground_truth}
    if evaluate_policy:
        policy = policy_rollout(row, backend, advisors, cfg, seed, root, base)
        record["policy"] = policy
        record["costs"] += policy["costs"]
        # Equal maximum generated-token allowance: advisor + revision vs self continuation.
        progress(phase="self_continue")
        self_revision = _draw(backend, base + [{"role": "user", "content": PROBE}], cfg,
                              seed + 500, budget=cfg["max_new_tokens"] + cfg["advisor_max_tokens"])
        record["self_continue_correct"] = _grade(self_revision, row)
        record["self_continue_text"] = self_revision["text"]
        record["costs"].append({"role": "manager", **self_revision})
    return record


def sft_rows(records, seed=42, commit_rescue_ratio=1.0, distill_solutions=True, selection="counterfactual"):
    rng = random.Random(seed)
    rescues = [r for r in records if r["preferred_sequence"]]
    commits = [r for r in records if r["preferred_sequence"] == []]
    rng.shuffle(commits)
    if commit_rescue_ratio >= 0 and rescues:
        commits = commits[:int(commit_rescue_ratio * len(rescues))]
    selected = rescues + commits
    rng.shuffle(selected)
    rows = []
    for r in selected:
        if r.get("split") != "train":
            raise ValueError("Never export SFT targets from dev/test diagnostics")
        seq = r["preferred_sequence"]
        if selection == "success":
            options = ([None] if r["direct_correct"] else []) + [b for b in r["branches"] if b["correct"]]
            chosen = rng.choice(options)
            seq = chosen["sequence"] if chosen else []
        else:
            chosen = next((b for b in r["branches"] if b["sequence"] == seq), None)
        meta = {"question_hash": r["question_hash"], "example_id": r["example_id"],
                "preferred_sequence": seq}
        if seq:
            for step in chosen["steps"]:
                rows.append({**meta, **step, "decision_type": "call"})
            final_prompt, solution = chosen["final_prompt"], chosen["text"]
        else:
            final_prompt, solution = r["base_messages"], r["direct_text"]
        rows.append({**meta, "prompt": final_prompt,
                     "response": [{"role": "assistant", "content": solution}], "decision_type": "commit"})
        if distill_solutions:
            # Full model-generated successful derivation; never substitute a gold solution.
            rows.append({**meta, "prompt": r["independent_prompt"],
                         "response": [{"role": "assistant", "content": solution}],
                         "decision_type": "independent_solution"})
    return rows


def summary(records):
    if not records:
        raise ValueError("Cannot summarize empty evaluation")
    n = len(records)
    solved = lambda r: r["direct_correct"] or any(b["correct"] for b in r.get("branches", []))
    out = {"n": n, "independent_accuracy": sum(r["direct_correct"] for r in records) / n}
    if all("branches" in r for r in records):
        out["delegation_search_coverage"] = sum(solved(r) for r in records) / n
    if all("policy" in r for r in records):
        out.update(policy_accuracy=sum(r["policy"]["correct"] for r in records) / n,
                   mean_calls=sum(r["policy"]["calls"] for r in records) / n,
                   policy_valid_rate=sum(r["policy"]["valid"] for r in records) / n)
        if all("branches" in r for r in records):
            out["measured_union_coverage"] = sum(solved(r) or r["policy"]["correct"] for r in records) / n
    if all("self_continue_correct" in r for r in records):
        out["self_continue_accuracy"] = sum(r["self_continue_correct"] for r in records) / n
    costs = [c for r in records for c in r.get("costs", [])]
    out["usage"] = {"logical_prompt_tokens": sum(c["prompt_tokens"] for c in costs),
                    "logical_completion_tokens": sum(c["completion_tokens"] for c in costs),
                    "actual_prompt_tokens": sum(c.get("actual_prompt_tokens", c["prompt_tokens"]) for c in costs),
                    "actual_completion_tokens": sum(c.get("actual_completion_tokens", c["completion_tokens"]) for c in costs)}
    return out


def compare(before, after):
    a, b = ({r["question_hash"]: r for r in rows} for rows in (before, after))
    if set(a) != set(b):
        raise ValueError("Checkpoint comparison requires exactly the same question set")
    out = {"n": len(a)}
    for name, fn in {
        "independent": lambda r: r["direct_correct"],
        "search": lambda r: r["direct_correct"] or any(v["correct"] for v in r["branches"]),
    }.items():
        out[name] = {"newly_solved": [k for k in a if not fn(a[k]) and fn(b[k])],
                     "regressed": [k for k in a if fn(a[k]) and not fn(b[k])],
                     "retained": sum(fn(a[k]) and fn(b[k]) for k in a)}
    out["previously_rescued_now_independent"] = [k for k in a if not a[k]["direct_correct"]
        and any(v["correct"] for v in a[k]["branches"]) and b[k]["direct_correct"]]
    return out
