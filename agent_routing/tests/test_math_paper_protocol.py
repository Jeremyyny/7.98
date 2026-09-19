"""Regression tests for scientific attribution, lineage, and failure recovery."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from test_verifiable import Advisors, Backend, CFG, row
from test_math_reporting import fixture
from src.utils.io import write_jsonl, write_json
from src.verifiable.answers import correct
from src.verifiable.analysis import conditional_metrics
from src.verifiable.experiment import collect_one, compare, policy_rollout, sft_rows
from src.verifiable.protocol import parse_calls
from src.verifiable.runner import build_plan, run_data, verify_advisor
from src.verifiable.reporting import generate_report
from src.verifiable.training import train_rl, tokenize_turn
from src.verifiable.readiness import paper_check


class ActionBackend(Backend):
    def __init__(self, action="COMMIT", initial="0"):
        super().__init__(initial)
        self.action = action

    def generate(self, history, tools=None, **kwargs):
        result = super().generate(history, tools=tools, **kwargs)
        if tools:
            result["text"] = self.action
        return result


class ProtocolV2Test(unittest.TestCase):
    def test_sft_keeps_inference_prefix_when_bpe_merges_boundary_whitespace(self):
        class BoundaryTokenizer:
            def __call__(self, text, **kwargs):
                return {"input_ids": {"prompt\n\n": [1, 2], "\ncall": [3, 4],
                                      "prompt\n\n\ncall": [1, 5, 4]}[text]}
        sample = {"prompt": [], "response": [], "decision_type": "call"}
        with patch("src.verifiable.training.render", side_effect=["prompt\n\n", "prompt\n\n\ncall"]):
            result = tokenize_turn(sample, BoundaryTokenizer(), 20)
        self.assertEqual(result["input_ids"], [1, 2, 3, 4])
        self.assertEqual(result["labels"], [-100, -100, 3, 4])
        with patch("src.verifiable.training.render", side_effect=["prompt", "changed prompt"]):
            with self.assertRaisesRegex(ValueError, "prefix preserving"):
                tokenize_turn(sample, BoundaryTokenizer(), 20)

    def test_commit_cannot_repair_an_incorrect_candidate(self):
        result = policy_rollout(row(), ActionBackend(), Advisors(), CFG, 7)
        self.assertTrue(result["valid"])
        self.assertFalse(result["correct"])
        self.assertIn(r"\boxed{0}", result["text"])
        self.assertEqual(result["calls"], 0)
        # Returning a newly correct solution during routing is invalid, not rescue.
        result = policy_rollout(row(), ActionBackend(r"FINAL_ANSWER: \boxed{42}"), Advisors(), CFG, 7)
        self.assertFalse(result["valid"])
        self.assertFalse(result["correct"])

    def test_policy_and_forced_branch_use_identical_revision_context_and_seed(self):
        class Traced(Backend):
            def __init__(self):
                super().__init__()
                self.revisions = []
            def generate(self, history, tools=None, **kwargs):
                if history[-1]["role"] == "user" and "self-contained revised" in history[-1]["content"]:
                    self.revisions.append((history, kwargs["seed"]))
                return super().generate(history, tools=tools, **kwargs)
        backend = Traced()
        record = collect_one(row(), backend, Advisors(), CFG, 7, evaluate_policy=True)
        branch = next(b for b in record["branches"] if b["sequence"] == ["reasoner"])
        self.assertEqual(record["policy"]["text"], branch["text"])
        # Nine forced branches, then the policy revision, then self-continuation.
        self.assertEqual(backend.revisions[1], backend.revisions[9])

    def test_verifier_receives_stored_candidate_without_model_arguments(self):
        advisors = Advisors()
        action = '<tool_call>{"name":"verifier_tool","arguments":{}}</tool_call>'
        policy_rollout(row(), ActionBackend(action), advisors, {**CFG, "max_depth": 1}, 7)
        self.assertEqual(advisors.calls[0][0], "verifier")
        self.assertIn(r"\boxed{0}", advisors.calls[0][2])
        with self.assertRaises(ValueError):
            parse_calls('<tool_call>{"name":"verifier_tool","arguments":{"current_draft":"fake"}}</tool_call>')

    def test_multiple_calls_repeats_and_xml_are_rejected(self):
        call = '<tool_call>{"name":"reasoner_tool","arguments":{}}</tool_call>'
        for action in (call + call, "COMMIT" + call):
            self.assertFalse(policy_rollout(row(), ActionBackend(action), Advisors(), CFG, 7)["valid"])
        with self.assertRaises(ValueError):
            parse_calls('<tool_call><function=reasoner_tool>junk</function></tool_call>')
        with self.assertRaises(ValueError):
            parse_calls('<tool_call>{"name":[],"arguments":{}}</tool_call>')
        call = '<tool_call>{"name":"extractor_tool","arguments":{}}</tool_call>'
        self.assertFalse(policy_rollout(row(), ActionBackend(call), Advisors(), CFG, 7)["valid"])

    def test_sft_targets_only_successful_revision_and_immutable_commit(self):
        rec = collect_one(row(), Backend(), Advisors(), CFG, 7)
        turns = sft_rows([rec])
        self.assertEqual(next(t["response"][0]["content"] for t in turns if t["decision_type"] == "commit"), "COMMIT")
        for t in turns:
            if t["decision_type"] in {"revision", "independent_solution"}:
                self.assertTrue(correct(t["response"][0]["content"], "42"))
        bad = {**rec, "split": "dev", "preferred_sequence": None}
        with self.assertRaises(ValueError):
            sft_rows([bad])
        with self.assertRaises(ValueError):
            compare([rec, rec], [rec])

    def test_three_sft_arms_control_training_method(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / "cfg.json"
            config.write_text(json.dumps(CFG))
            plans = {arm: build_plan(config, tmp, str(Path(tmp) / arm), arm, 2)
                     for arm in ("dynamic_sft", "success_sft", "static_sft")}
            for arm, plan in plans.items():
                self.assertEqual(sum(p["stage"] == "sft" for p in plan), 2)
                self.assertFalse(any(p["stage"] == "rl" for p in plan))
            self.assertEqual(sum(p["stage"] == "collect" for p in plans["static_sft"]), 1)
            self.assertTrue(all(p["command"][-1] == "success" for p in plans["success_sft"] if p["stage"] == "collect"))
            with self.assertRaises(ValueError):
                build_plan(config, tmp, tmp, "success_rl", 2)
        with self.assertRaises(NotImplementedError):
            train_rl({}, "unused", "unused", "unused")

    def test_resume_recovers_partial_jsonl_but_rejects_changed_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data, out = root / "data.jsonl", root / "run"
            write_jsonl(str(data), [row().to_dict()])
            run_data(CFG, str(data), "fake", out, "collect", backend=Backend(), advisors=Advisors())
            with (out / "records.jsonl").open("a") as f:
                f.write('{"partial":')
            backend = Backend()
            run_data(CFG, str(data), "fake", out, "collect", resume=True, backend=backend, advisors=Advisors())
            self.assertFalse(backend.inputs)
            json.loads((out / "records.jsonl").read_text())
            with patch("src.verifiable.runner.harness_identity", return_value={"different": True}):
                with self.assertRaises(ValueError):
                    run_data(CFG, str(data), "fake", out, "collect", resume=True)

    def test_near_miss_numeric_latex_is_not_rounded_to_gold(self):
        self.assertFalse(correct(r"FINAL_ANSWER: \boxed{\frac{42000000001}{1000000000}}", "42"))
        self.assertTrue(correct(r"FINAL_ANSWER: \boxed{\frac{84}{2}}", "42"))


class AnalysisTest(unittest.TestCase):
    def test_incomplete_legacy_runs_cannot_pass_paper_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp, include_final=False)
            result = paper_check([root], Path(tmp) / "paper")
            self.assertFalse(result["complete"])
            self.assertTrue(any("SFT-only" in f for f in result["failures"]))
            self.assertTrue(any("missing locked" in f for f in result["failures"]))
            self.assertTrue(any("same predeclared seeds" in f for f in result["failures"]))

    def test_fixed_cohort_denominators_and_policy_behavior(self):
        def r(key, direct, rescue, calls, success):
            return {"question_hash": key, "direct_correct": direct, "branches": [{"correct": rescue}],
                    "policy": {"calls": calls, "correct": success, "valid": True}}
        before = [r("a", False, True, 1, True), r("b", False, True, 1, True), r("c", True, True, 0, True)]
        after = [r("c", False, False, 1, False), r("a", True, True, 0, True), r("b", False, True, 1, True)]
        value = conditional_metrics(after, before)
        self.assertEqual(value["internalization_rate"], .5)
        self.assertEqual(value["initially_rescued_now_independent_mean_calls"], 0)
        self.assertEqual(value["learned_subset_before_mean_calls"], 1)
        self.assertEqual(value["currently_rescuable_policy_accuracy"], 1)
        self.assertEqual(sum(v for k,v in value.items() if k.startswith("transition_")), 3)
        empty = conditional_metrics([r("x", True, True, 0, True)], [r("x", True, True, 0, True)])
        self.assertIsNone(empty["internalization_rate"])
        self.assertIsNone(empty["initially_rescued_now_independent_mean_calls"])

    def test_report_emits_paired_arm_comparisons(self):
        with tempfile.TemporaryDirectory() as tmp:
            runs = []
            for arm in ("dynamic_sft", "success_sft"):
                root = fixture(tmp, name=arm + "_s42")
                path = root / "loop.json"
                value = json.loads(path.read_text())
                value["arm"] = arm
                write_json(str(path), value)
                runs.append(root)
            out = Path(tmp) / "report"
            generate_report(runs, out, demo=True)
            self.assertIn("success_sft", (out / "arm_comparisons.csv").read_text())
            self.assertTrue((out / "arm_summary.csv").exists())
            self.assertTrue((out / "delegation_behavior.csv").exists())


if __name__ == "__main__":
    unittest.main()
