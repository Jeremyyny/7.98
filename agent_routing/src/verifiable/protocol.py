"""The same native-tool protocol is used in collection, SFT, RL and evaluation."""
from __future__ import annotations

import json
import re

KINDS = ("extractor", "reasoner", "verifier")
FINAL_RULE = r"End with exactly one final line: FINAL_ANSWER: \boxed{your answer}."
SYSTEM = """You solve free-response mathematics problems. Show your mathematical reasoning.
You can commit to your solution or request help from three frozen sub-agents:
extractor_tool identifies constraints; reasoner_tool suggests a solution approach;
verifier_tool audits your full current derivation. The verifier sub-agent is a fallible
language model, not the answer-grading program.
Use native tool calls, at most one call per turn and each sub-agent at most once.
Before a call, write your current derivation. For verifier_tool pass that complete
derivation as current_draft. Stop when another sub-agent is unlikely to help.
Never put a FINAL_ANSWER line in a turn that calls a tool.
""" + FINAL_RULE
DIRECT_SYSTEM = "Solve the mathematics problem independently. Show your reasoning. " + FINAL_RULE


def messages(row, direct=False, max_calls=3):
    # Explicit allowlist: labels, source solutions and metadata never enter prompts.
    question = row.question + ("\nContext:\n" + row.context if row.context else "")
    system = DIRECT_SYSTEM if direct else SYSTEM + f"\nYour total sub-agent-call budget is {max_calls}."
    return [{"role": "system", "content": system},
            {"role": "user", "content": question}]


def tool_schemas():
    result = []
    for kind in KINDS:
        properties = ({"current_draft": {"type": "string", "description":
                       "Your full current reasoning, including candidate answer."}}
                      if kind == "verifier" else {})
        result.append({"type": "function", "function": {
            "name": kind + "_tool", "description": {
                "extractor": "Extract the stated facts and constraints.",
                "reasoner": "Suggest a solution approach using relevant principles.",
                "verifier": "Audit the current reasoning for errors."}[kind],
            "parameters": {"type": "object", "properties": properties,
                           "required": ["current_draft"] if kind == "verifier" else []}}})
    return result


def call_message(kind, draft, call_id):
    return {"role": "assistant", "content": draft, "tool_calls": [{
        "id": call_id, "type": "function", "function": {
            "name": kind + "_tool", "arguments":
            {"current_draft": draft} if kind == "verifier" else {}}}]}


def parse_calls(text):
    """Parse Qwen3 JSON or Qwen3.5 XML calls; reject malformed/unknown calls."""
    pattern = r"<tool_call>(.*?)</tool_call>"
    blocks = re.findall(pattern, text, flags=re.S)
    calls = []
    for block in blocks:
        try:
            obj = json.loads(block)
        except json.JSONDecodeError:
            name = re.search(r"<function=([^>]+)>", block)
            if not name:
                raise ValueError("Malformed tool call")
            obj = {"name": name.group(1), "arguments": dict(
                re.findall(r"<parameter=([^>]+)>(.*?)</parameter>", block, re.S))}
        if not isinstance(obj, dict) or obj.get("name") not in {k + "_tool" for k in KINDS}:
            raise ValueError("Unknown tool call")
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = json.loads(args)
        if not isinstance(args, dict):
            raise ValueError("Tool arguments must be an object")
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
