"""Protocol v2 shared by collection, SFT and evaluation; legacy RL is disabled."""
from __future__ import annotations

import json
import re

KINDS = ("extractor", "reasoner", "verifier")
FINAL_RULE = r"End with exactly one final line: FINAL_ANSWER: \boxed{your answer}."
PROTOCOL_VERSION = 2
COMMIT = "COMMIT"
DECIDE = "Choose exactly one action: COMMIT to submit the stored candidate unchanged, or one unused sub-agent tool call with empty arguments. Do not write a new solution in this decision turn."
REVISE = "Using the problem and available advice, write a complete, self-contained revised solution. Include the needed reasoning; do not refer to earlier messages or tools. Do not call tools. " + FINAL_RULE
SYSTEM = """You solve free-response mathematics problems using stored solution candidates.
At a decision turn, output exactly COMMIT or one native sub-agent tool call with
empty arguments. COMMIT submits the stored candidate unchanged. The environment
passes that candidate to the verifier automatically. Each sub-agent may be used
at most once: extractor finds facts and constraints, reasoner suggests an approach,
and verifier audits the stored reasoning. These are fallible language models.
After a tool response, you will be asked to write a complete revised solution.
Only that revision phase may change the candidate. At the call budget, submit it.
"""
DIRECT_SYSTEM = "Solve the mathematics problem independently. Show your reasoning. " + FINAL_RULE


def messages(row, direct=False, max_calls=3):
    # Explicit allowlist: labels, source solutions and metadata never enter prompts.
    question = row.question + ("\nContext:\n" + row.context if row.context else "")
    system = DIRECT_SYSTEM if direct else SYSTEM + f"\nYour total sub-agent-call budget is {max_calls}."
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def tool_schemas():
    descriptions = {"extractor": "Extract the stated facts and constraints.",
                    "reasoner": "Suggest a solution approach using relevant principles.",
                    "verifier": "Audit the stored candidate, supplied automatically by the environment."}
    return [{"type": "function", "function": {"name": kind + "_tool",
             "description": descriptions[kind], "parameters": {"type": "object",
             "properties": {}, "required": [], "additionalProperties": False}}} for kind in KINDS]


def call_message(kind, draft, call_id):
    # Keep the helper signature compatible; candidates are bound by the environment.
    return {"role": "assistant", "content": "", "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": kind + "_tool", "arguments": {}}}]}


def parse_calls(text):
    """Parse the fixed JSON-tool template; reject malformed or unknown calls."""
    pattern = r"<tool_call>(.*?)</tool_call>"
    blocks = re.findall(pattern, text, flags=re.S)
    calls = []
    for block in blocks:
        try:
            obj = json.loads(block)
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON tool call; protocol v2 uses the fixed JSON template") from exc
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or obj["name"] not in {k + "_tool" for k in KINDS}:
            raise ValueError("Unknown tool call")
        if set(obj) - {"name", "arguments"}:
            raise ValueError("Unexpected tool-call fields")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
        if args:
            raise ValueError("Sub-agent arguments must be empty; the candidate is environment-bound")
        calls.append({"name": obj["name"], "arguments": args})
    content = re.sub(pattern, "", text, flags=re.S).strip()
    if "<tool_call" in content or "</tool_call>" in content:
        raise ValueError("Unclosed tool call")
    return content, calls


def advisor_messages(kind, row, draft):
    instructions = {
        "extractor": "Extract givens, constraints, variables and useful equivalent formulations.",
        "reasoner": "Develop a useful solution approach with intermediate deductions based on relevant principles.",
        "verifier": "Audit the supplied derivation. Identify specific invalid steps and suggest repairs.",
    }
    return [{"role": "system", "content": "You are a reasoning sub-agent. Use the subject matter of the supplied question. " + instructions[kind]
             + " Give concise concrete help. You have no answer key. Do not use tools."},
            {"role": "user", "content": row.question +
             ("\nContext:\n" + row.context if row.context else "") +
             ("\nCurrent derivation:\n" + draft if kind == "verifier" else "")}]
