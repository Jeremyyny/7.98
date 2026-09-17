from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.benchmarks.base import StandardRow
from src.utils.io import write_jsonl
from src.verifiable.answers import correct, equivalent, extract_final
from src.verifiable.data import identity, normalize, partition, prepare, verify_manifest
from src.verifiable.experiment import collect_one, compare, policy_rollout, sft_rows, summary
from src.verifiable.protocol import advisor_messages, messages, parse_calls
from src.verifiable.runner import build_plan, run_data
from src.verifiable.training import make_environment, reward_function, tokenize_turn

CFG = {"base_model": "fake", "advisor_url": "http://fake", "seed": 42, "max_depth": 2,
       "max_new_tokens": 100, "advisor_max_tokens": 50, "max_context": 10000, "max_seq_len": 10000}


def row(index=0, answer="42", split="train"):
    return StandardRow(index, "math", "free_response_math", f"Compute a value, problem {index}.", {}, answer,
                       metadata={"answer_type": "math"}, split=split)


class Backend:
    def __init__(self, initial="0"):
        self.initial = initial
        self.inputs = []

    def generate(self, history, tools=None, **kwargs):
        self.inputs.append(deepcopy(history))
        helped = any(m.get("role") == "tool" and m.get("name") == "reasoner_tool" for m in history)
        if tools and not helped:
            text = '<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>'
        else:
            text = "A full derivation with intermediate steps.\nFINAL_ANSWER: \\boxed{" + ("42" if helped else self.initial) + "}"
        return {"text": text, "prompt_tokens": 10, "completion_tokens": 20,
                "truncated": False, "seconds": 0.001}


class Advisors:
    def __init__(self):
        self.calls = []

    def call(self, kind, problem, draft=""):
        self.calls.append((kind, problem.question, draft))
        return {"text": "Useful advice", "prompt_tokens": 5, "completion_tokens": 10,
                "seconds": 0.001, "truncated": False}


class AnswersTest(unittest.TestCase):
    def test_no_incidental_or_multiple_answers(self):
        for text in ["The intermediate value is 42", r"\boxed{42}",
                     "FINAL_ANSWER: \\boxed{42}\nI disagree", "FINAL_ANSWER: 42",
                     "FINAL_ANSWER: \\boxed{0}\nFINAL_ANSWER: \\boxed{42}"]:
            self.assertFalse(correct(text, "42"))

    def test_nested_latex_and_equivalence(self):
        self.assertEqual(extract_final(r"FINAL_ANSWER: \boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertTrue(correct(r"FINAL_ANSWER: \boxed{\frac{1}{2}}", "0.5"))
        self.assertTrue(equivalent("042", "42"))
        self.assertFalse(equivalent("41.99999", "42"))
        self.assertFalse(correct(r"FINAL_ANSWER: \boxed{42} or 43", "42"))


class DataTest(unittest.TestCase):
    def test_zero_not_dropped_and_gold_solution_not_copied(self):
        normalized = normalize({"problem": "Find zero", "answer": 0, "solution": "SECRET"}, "aime2026", 0)
        self.assertEqual(normalized.ground_truth, "0")
        self.assertEqual(normalized.split, "test")
        self.assertNotIn("SECRET", json.dumps(normalized.to_dict()))

    def test_numina_filters(self):
        valid = {"problem": "Find a value", "answer": "1", "problem_is_valid": "Yes", "solution_is_valid": "Yes"}
        self.assertIsNotNone(normalize(valid, "numina", 0))
        for change in [{"answer": "proof"}, {"solution_is_valid": "Incomplete"},
                       {"question_type": "proof"}, {"problem": "Use ![](image.png)"}]:
            self.assertIsNone(normalize({**valid, **change}, "numina", 0))

    def test_dedup_and_split(self):
        data = [row(i) for i in range(7)] + [row(0)]
        train, dev, _ = partition(data, {identity(row(1).question)}, 3, 2, 42)
        self.assertFalse({identity(r.question) for r in train} & {identity(r.question) for r in dev})
        self.assertNotIn(identity(row(1).question), {identity(r.question) for r in train + dev})
        self.assertEqual([r.split for r in train + dev], ["train"] * 3 + ["dev"] * 2)
        self.assertEqual(identity("A  B"), identity("a\nb"))

    def test_prepare_locks_data_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sources = {}
            for name in ("numina", "aime2026", "beyondaime"):
                path = root / (name + ".jsonl")
                raw = [{"problem": name + str(i), "answer": "42", "solution": "SECRET",
                        "problem_is_valid": "Yes", "solution_is_valid": "Yes"} for i in range(10)]
                write_jsonl(str(path), raw)
                sources[name] = str(path)
            out = root / "normalized"
            result = prepare(out, 4, 2, local_sources=sources)
            self.assertEqual(result["counts"]["train"], 4)
            self.assertNotIn("SECRET", (out / "train.jsonl").read_text())
            verify_manifest(out)
            (out / "train.jsonl").write_text("corrupted")
            with self.assertRaises(ValueError):
                verify_manifest(out)


class CounterfactualTest(unittest.TestCase):
    def test_full_branching_same_root_and_full_solution_export(self):
        backend, advisors = Backend(), Advisors()
        record = collect_one(row(), backend, advisors, CFG, 12, evaluate_policy=True)
        self.assertEqual(len(record["branches"]), 9)
        self.assertEqual(record["preferred_sequence"], ["reasoner"])
        self.assertTrue(record["policy"]["correct"])
        for kind, _, draft in advisors.calls:
            if kind == "verifier":
                self.assertIn("full derivation", draft)
        for branch in record["branches"]:
            self.assertEqual(branch["steps"][0]["prompt"], record["base_messages"])
        turns = sft_rows([record])
        self.assertEqual([r["decision_type"] for r in turns], ["call", "commit", "independent_solution"])
        self.assertIn("intermediate steps", turns[-1]["response"][0]["content"])
        self.assertNotIn("Useful advice", json.dumps(turns[-1]["prompt"]))
        self.assertEqual(summary([record])["measured_union_coverage"], 1)

    def test_direct_correct_still_probes_and_no_heldout_training(self):
        record = collect_one(row(split="dev"), Backend("42"), Advisors(), CFG, 12)
        self.assertEqual(record["preferred_sequence"], [])
        self.assertEqual(len(record["branches"]), 9)
        with self.assertRaises(ValueError):
            sft_rows([record])

    def test_labels_do_not_change_runtime_inputs(self):
        a, b = Backend(), Backend()
        collect_one(row(answer="42"), a, Advisors(), CFG, 12)
        collect_one(row(answer="999"), b, Advisors(), CFG, 12)
        self.assertEqual(a.inputs, b.inputs)
        self.assertEqual(messages(row(answer="42")), messages(row(answer="999")))
        self.assertEqual(advisor_messages("reasoner", row(answer="42"), ""),
                         advisor_messages("reasoner", row(answer="999"), ""))

    def test_resume_and_setting_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "data.jsonl"
            write_jsonl(str(source), [row().to_dict()])
            out = str(Path(tmp) / "run")
            run_data(CFG, str(source), "fake", out, "collect", backend=Backend(), advisors=Advisors())
            backend = Backend()
            run_data(CFG, str(source), "fake", out, "collect", resume=True, backend=backend, advisors=Advisors())
            self.assertFalse(backend.inputs)
            with self.assertRaises(ValueError):
                run_data({**CFG, "max_depth": 1}, str(source), "fake", out, "collect", resume=True)

    def test_policy_has_no_oracle_feedback(self):
        result = policy_rollout(row(answer="999"), Backend(), Advisors(), CFG, 12)
        self.assertFalse(result["correct"])
        self.assertEqual(result["sequence"], ["reasoner"])
        self.assertNotIn("999", json.dumps(result["history"]))

    def test_comparison_and_no_fake_coverage(self):
        before = collect_one(row(), Backend(), Advisors(), CFG, 12)
        after = collect_one(row(), Backend("42"), Advisors(), CFG, 12)
        diff = compare([before], [after])
        self.assertEqual(len(diff["previously_rescued_now_independent"]), 1)
        with self.assertRaises(ValueError):
            compare([before], [])
        final_only = {"direct_correct": False, "policy": {"correct": True, "calls": 1, "valid": True}}
        self.assertNotIn("delegation_search_coverage", summary([final_only]))


class ProtocolTest(unittest.TestCase):
    def test_both_qwen_call_forms(self):
        for text in ['<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>',
                     '<tool_call><function=verifier_tool><parameter=current_draft>x=2</parameter></function></tool_call>']:
            _, calls = parse_calls(text)
            self.assertEqual(len(calls), 1)
        for text in ['<tool_call>{bad}</tool_call>', '<tool_call>{"name":"wrong","arguments":{}}</tool_call>', '<tool_call>']:
            with self.assertRaises(ValueError):
                parse_calls(text)

    def test_rl_tool_budget_and_binary_reward(self):
        env = make_environment([row()], Advisors(), 1)()
        env.reset(0)
        env.reasoner_tool()
        with tempfile.TemporaryDirectory() as tmp:
            reward = reward_function(str(Path(tmp) / "trace.jsonl"))
            completions = [[{"role": "assistant", "content": r"FINAL_ANSWER: \boxed{42}"}]]
            self.assertEqual(reward(completions, ["42"], [env]), [1.0])
            env.reasoner_tool()
            self.assertEqual(reward(completions, ["42"], [env]), [0.0])

    def test_two_round_plan_continues_weights_and_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "cfg.json"
            config.write_text(json.dumps(CFG))
            plan = build_plan(config, tmp, str(Path(tmp) / "run"), "dynamic_rl", 2)
            collects = [p for p in plan if p["stage"] == "collect"]
            self.assertEqual(len(collects), 2)
            self.assertIn(str((Path(tmp) / "run/round_1/rl").resolve()), collects[1]["command"])
            fixed = build_plan(config, tmp, str(Path(tmp) / "fixed"), "static_rl", 2)
            self.assertEqual(sum(p["stage"] == "collect" for p in fixed), 1)
            self.assertFalse(any("aime2026.jsonl" in str(p) for p in plan))


if __name__ == "__main__":
    unittest.main()
