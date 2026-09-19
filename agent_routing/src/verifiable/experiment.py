"""Protocol v2: explicit routing actions, immutable COMMIT and paired revisions."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import random

from ..manager.marginal_value import choose_preferred_sequence
from .answers import correct, extract_final
from .data import identity
from .protocol import (COMMIT, DECIDE, REVISE, KINDS, PROTOCOL_VERSION,
                       call_message, messages, parse_calls, tool_schemas)
from .telemetry import progress


def candidate(text):
    return text.replace("FINAL_ANSWER:", "CANDIDATE_ANSWER:")


def _draw(backend, history, cfg, seed, tools=None, budget=None):
    result = dict(backend.generate(history, tools=tools,
        max_tokens=budget or cfg["max_new_tokens"],
        temperature=cfg.get("temperature", 0), seed=seed))
    result["valid"] = (not result.get("truncated", False)
                       and extract_final(result["text"]) is not None
                       and not any(token in result["text"] for token in ("<tool_call", "<|im_start|>", "<|im_end|>")))
    return result


def _grade(result, row):
    return bool(result["valid"] and correct(result["text"], row.ground_truth))


def branch_seed(seed, sequence):
    # Shared by forced branches and policy rollout, independent of traversal order.
    key = f"{seed}:" + "/".join(sequence)
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % (2 ** 31)


def decision_history(history, draft):
    return history + [{"role": "assistant", "content": candidate(draft)},
                      {"role": "user", "content": DECIDE}]


def root_state(row, backend, cfg, seed):
    progress(phase="independent", question_hash=identity(row.question))
    root = _draw(backend, messages(row, direct=True), cfg, seed)
    return root, decision_history(messages(row, max_calls=cfg.get("max_depth", 2)), root["text"])


def delegate(row, backend, advisors, cfg, seed, sequence, history, draft):
    kind = sequence[-1]
    msg = call_message(kind, draft, "call_" + "_".join(sequence))
    advice = advisors.call(kind, row, draft if kind == "verifier" else "")
    call_history = history + [msg, {"role": "tool", "tool_call_id": msg["tool_calls"][0]["id"],
                                    "name": kind + "_tool", "content": advice["text"]}]
    revision_prompt = call_history + [{"role": "user", "content": REVISE}]
    revision = _draw(backend, revision_prompt, cfg, branch_seed(seed, sequence))
    next_history = decision_history(revision_prompt, revision["text"])
    step = {"prompt": deepcopy(history), "response": [msg],
            "revision_prompt": revision_prompt, "revision_text": revision["text"]}
    return revision, next_history, step, [{"role": "advisor", **advice}, {"role": "manager", **revision}]


def policy_rollout(row, backend, advisors, cfg, seed, root=None, history=None):
    if root is None:
        root, history = root_state(row, backend, cfg, seed)
    history, current = deepcopy(history), dict(root)
    used, costs, decisions = [], [], []
    error = None
    for turn in range(cfg.get("max_depth", 2) + 1):
        progress(phase="policy", policy_turn=turn, sequence=used)
        # At the hard limit COMMIT is forced in collection and deployment alike.
        if turn == cfg.get("max_depth", 2):
            decisions.append({"action": COMMIT, "forced": True})
            break
        generated = _draw(backend, history, cfg, branch_seed(seed, ["decision", *used]),
                          tools=tool_schemas(), budget=cfg.get("decision_max_tokens", 128))
        costs.append({"role": "manager", **generated})
        if generated.get("truncated"):
            error = "truncated_decision"
            break
        try:
            content, calls = parse_calls(generated["text"])
        except ValueError as exc:
            error = str(exc)
            break
        if not calls:
            if content != COMMIT:
                error = "expected_COMMIT_or_tool_call"
            else:
                decisions.append({"action": COMMIT, "forced": False})
            break
        if len(calls) != 1 or content:
            error = "decision_must_be_one_bare_tool_call"
            break
        kind = calls[0]["name"].removesuffix("_tool")
        if kind in used:
            error = "repeated_subagent"
            break
        used.append(kind)
        decisions.append({"action": kind, "forced": False})
        current, history, _, extra = delegate(row, backend, advisors, cfg, seed, used, history, current["text"])
        costs.extend(extra)
    valid = error is None and current["valid"]
    return {"correct": bool(valid and _grade(current, row)), "valid": valid,
            "text": current["text"], "calls": len(used), "sequence": used,
            "costs": costs, "history": history, "decisions": decisions,
            "error": error, "protocol_version": PROTOCOL_VERSION}


def collect_one(row, backend, advisors, cfg, seed, evaluate_policy=False):
    root, base = root_state(row, backend, cfg, seed)
    root_correct = _grade(root, row)
    frontier = [{"sequence": [], "history": base, "draft": root["text"], "steps": []}]
    branches, costs = [], [{"role": "manager", **root}]
    for depth in range(1, cfg.get("max_depth", 2) + 1):
        next_frontier = []
        for state in frontier:
            for kind in KINDS:
                if kind in state["sequence"]:
                    continue
                seq = state["sequence"] + [kind]
                progress(phase="counterfactual", sequence=seq, completed_branches=len(branches))
                revision, history, step, extra = delegate(row, backend, advisors, cfg, seed,
                                                        seq, state["history"], state["draft"])
                costs.extend(extra)
                steps = state["steps"] + [step]
                branches.append({"sequence": seq, "correct": _grade(revision, row),
                    "valid": revision["valid"], "text": revision["text"],
                    "truncated": revision.get("truncated", False), "steps": steps,
                    "revision_prompt": step["revision_prompt"], "final_prompt": history})
                next_frontier.append({"sequence": seq, "history": history,
                                      "draft": revision["text"], "steps": steps})
        frontier = next_frontier
    preferred = choose_preferred_sequence(root_correct, branches, tie_break_seed=seed)
    record = {"question_hash": identity(row.question), "example_id": row.example_id,
              "benchmark_name": row.benchmark_name, "split": row.split,
              "protocol_version": PROTOCOL_VERSION, "direct_correct": root_correct,
              "direct_valid": root["valid"], "direct_truncated": root.get("truncated", False),
              "direct_text": root["text"], "base_messages": base,
              "independent_prompt": messages(row, direct=True), "branches": branches,
              "preferred_sequence": list(preferred) if preferred is not None else None,
              "costs": costs, "ground_truth": row.ground_truth}
    if evaluate_policy:
        record["policy"] = policy_rollout(row, backend, advisors, cfg, seed, root, base)
        record["costs"].extend(record["policy"]["costs"])
        # One self-revision control; ceiling matches ONE advisor plus one revision.
        # This is not a full multi-call equal-compute baseline.
        revision = _draw(backend, base + [{"role": "user", "content": REVISE}], cfg,
                        branch_seed(seed, ["self_continue"]),
                        budget=cfg["max_new_tokens"] + cfg["advisor_max_tokens"])
        record.update(self_continue_correct=_grade(revision, row), self_continue_text=revision["text"])
        record["costs"].append({"role": "manager", **revision})
    return record


def sft_rows(records, seed=42, commit_rescue_ratio=1.0, distill_solutions=True, selection="counterfactual"):
    if selection not in {"counterfactual", "success"}:
        raise ValueError("Unknown trajectory selector")
    if any(r.get("split") != "train" for r in records):
        raise ValueError("Never export SFT targets from dev/test diagnostics")
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
        seq = r["preferred_sequence"]
        if selection == "success":
            options = ([None] if r["direct_correct"] else []) + [b for b in r["branches"] if b["correct"]]
            chosen = rng.choice(options)
            seq = chosen["sequence"] if chosen else []
        else:
            chosen = next((b for b in r["branches"] if b["sequence"] == seq), None)
        meta = {"question_hash": r["question_hash"], "example_id": r["example_id"],
                "preferred_sequence": seq, "split": "train", "protocol_version": PROTOCOL_VERSION}
        if seq:
            for step in chosen["steps"]:
                rows.append({**meta, "prompt": step["prompt"], "response": step["response"], "decision_type": "call"})
            # Only the successful terminal revision is a solution target. Earlier
            # incorrect drafts remain context, never supervised reasoning targets.
            rows.append({**meta, "prompt": chosen["revision_prompt"],
                         "response": [{"role": "assistant", "content": chosen["text"]}], "decision_type": "revision"})
            final_prompt, solution = chosen["final_prompt"], chosen["text"]
        else:
            final_prompt, solution = r["base_messages"], r["direct_text"]
        rows.append({**meta, "prompt": final_prompt,
                     "response": [{"role": "assistant", "content": COMMIT}], "decision_type": "commit"})
        if distill_solutions:
            rows.append({**meta, "prompt": r["independent_prompt"],
                         "response": [{"role": "assistant", "content": solution}], "decision_type": "independent_solution"})
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
    from .analysis import conditional_metrics
    out.update(conditional_metrics(records))
    return out


def compare(before, after):
    a, b = ({r["question_hash"]: r for r in rows} for rows in (before, after))
    if len(a) != len(before) or len(b) != len(after) or not a or set(a) != set(b):
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
