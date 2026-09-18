"""Synthetic fixtures validate accounting and plots, never experimental results."""
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest
from unittest.mock import patch

from src.utils.io import write_json, write_jsonl
from src.verifiable.experiment import summary
from src.verifiable.reporting import generate_report, paired_stats, stage_cost, wilson
from src.verifiable.telemetry import Monitor, status_snapshot
from src.verifiable.runner import execute_stage, verify_advisor


def fixture(root, name="dynamic_rl_s42", include_final=True):
    root = Path(root) / name
    root.mkdir(parents=True)
    plan = []
    cfg = {"base_model": "SYNTHETIC_FIXTURE", "seed": 42, "num_generations": 4}
    def records(n, benchmark, phase, dev):
        rows = []
        for i in range(n):
            direct = i % 10 < (3 if phase == 0 else 5) and not (phase > 0 and i % 10 == 0)
            rescued = i % 10 < 7
            policy = direct or (i % 10 < (5 if phase == 0 else 7))
            cost = {"prompt_tokens": 12, "completion_tokens": 8}
            r = {"question_hash": f"{benchmark}_{i}", "example_id": i, "benchmark_name": benchmark,
                 "split": "dev" if dev else "test", "direct_correct": direct, "direct_valid": True,
                 "direct_truncated": False, "costs": [cost],
                 "policy": {"correct": policy, "valid": True, "calls": int(not direct), "costs": [cost]}}
            if dev:
                r.update(branches=[{"correct": rescued, "sequence": ["reasoner"]}], self_continue_correct=direct)
            rows.append(r)
        return rows
    def result(path, rows):
        path.mkdir(parents=True)
        write_jsonl(str(path / "records.jsonl"), rows)
        write_json(str(path / "summary.json"), summary(rows))
        write_json(str(path / ".stage_complete.json"), {"command": []})
    for label, stage in [("initial_dev", "diagnose"), ("round_1/collection", "collect"),
                          ("round_1/sft", "sft"), ("round_1/sft_dev", "diagnose"),
                          ("round_1/rl", "rl"), ("round_1/rl_dev", "diagnose")]:
        path = root / label
        plan.append({"output": str(path.resolve()), "stage": stage, "command": []})
        if stage == "diagnose":
            result(path, records(40, "dev", 0 if label == "initial_dev" else 1, True))
        else:
            path.mkdir(parents=True)
            write_json(str(path / ".stage_complete.json"), {"command": []})
            write_jsonl(str(path / "events.jsonl"), [{"attempt": "a", "event": "started"},
                        {"attempt": "a", "event": "completed", "wall_seconds": 100}])
            write_jsonl(str(path / "usage.jsonl"), [{"role": "manager", "prompt_tokens": 500, "completion_tokens": 200}])
            if stage == "rl":
                write_jsonl(str(path / "training_log.jsonl"), [{"step": i, "reward": i / 10,
                    "frac_reward_zero_std": 1 - i / 10, "attempt": "a"} for i in range(1, 5)])
    write_json(str(root / "loop.json"), {"config": cfg, "arm": "dynamic_rl", "initial": None,
               "data_manifest": {"counts": {"dev": 40, "aime2026": 30, "beyondaime": 100}}, "plan": plan})
    write_json(str(root / "advisor_identity.json"), {"model": "SYNTHETIC_ADVISOR"})
    for benchmark, n in [("aime2026", 30), ("beyondaime", 100)]:
        for label, phase in [("initial", 0)] + ([("final", 1)] if include_final else []):
            result(root / "test" / label / benchmark, records(n, benchmark, phase, False))
    return root


class PaperReportTest(unittest.TestCase):
    def test_paired_statistics_and_zero_denominator(self):
        value = paired_stats([0, 0, 1, 1], [1, 1, 1, 0])
        self.assertEqual(value["newly_solved"], 2)
        self.assertEqual(value["regressed"], 1)
        self.assertEqual(value["delta_pp"], 25)
        self.assertEqual(value, paired_stats([0, 0, 1, 1], [1, 1, 1, 0]))
        self.assertEqual(wilson(0, 0), (None, None))
        self.assertLess(wilson(0, 30)[1], .12)

    def test_report_creates_all_figures_and_no_test_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            out = Path(tmp) / "paper"
            result = generate_report([root], out, demo=True)
            self.assertEqual(result["main_rows"], 4)
            self.assertEqual(len(result["figures"]), 5)
            self.assertNotIn("search_pct", (out / "main_results.csv").read_text())
            self.assertIn("SYNTHETIC", (out / "README.md").read_text())
            self.assertTrue((out / "paper_main.tex").exists())
            self.assertTrue((out / "fig3_external_tests.pdf").stat().st_size > 1000)

    def test_missing_results_are_not_imputed_and_tampering_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp, include_final=False)
            result = generate_report([root], Path(tmp) / "paper")
            self.assertEqual(result["main_rows"], 2)
            self.assertTrue(any("missing external evaluation" in w for w in result["warnings"]))
            path = root / "test/initial/aime2026/summary.json"
            value = json.loads(path.read_text())
            value["policy_accuracy"] = .999
            write_json(str(path), value)
            with self.assertRaisesRegex(ValueError, "disagree"):
                generate_report([root], Path(tmp) / "bad")

    def test_frozen_data_mismatch_and_report_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = fixture(tmp)
            out = Path(tmp) / "paper"
            generate_report([root], out)
            (root / "round_1/rl/training_log.jsonl").unlink()
            result = generate_report([root], out)
            self.assertNotIn("fig5_rl_training", result["figures"])
            self.assertFalse((out / "fig5_rl_training.pdf").exists())
            value = json.loads((root / "loop.json").read_text())
            value["data_manifest"]["sha256"] = {"dev.jsonl": "expected"}
            write_json(str(root / "loop.json"), value)
            write_json(str(root / "initial_dev/run.json"), {"data_sha256": "wrong", "config": value["config"]})
            with self.assertRaisesRegex(ValueError, "differs from frozen"):
                generate_report([root], Path(tmp) / "bad")

    def test_monitor_failure_and_incomplete_cost(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.verifiable.telemetry.subprocess.run", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(RuntimeError, "intentional"):
                    with Monitor(tmp, "test") as monitor:
                        monitor.update(completed_examples=2, total_examples=4)
                        monitor.usage("manager", {"prompt_tokens": 10, "completion_tokens": 5})
                        raise RuntimeError("intentional")
            state = status_snapshot(tmp)["states"][0]
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["progress"]["completed_examples"], 2)
            self.assertFalse(stage_cost(Path(tmp))["accounting_complete"])
            self.assertIn("intentional", (Path(tmp) / "errors.log").read_text())

    def test_failed_child_and_missing_outputs_are_not_marked_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stage"
            step = {"stage": "evaluate", "output": str(path), "command": [sys.executable, "-c", "print('expected failure'); raise SystemExit(3)"]}
            with self.assertRaises(subprocess.CalledProcessError):
                execute_stage(step, Path(tmp) / "logs")
            self.assertFalse((path / ".stage_complete.json").exists())
            self.assertIn("expected failure", next((Path(tmp) / "logs").glob("*.log")).read_text())
            step["command"] = [sys.executable, "-c", "print('no artifacts')"]
            with self.assertRaisesRegex(ValueError, "artifact missing"):
                execute_stage(step, Path(tmp) / "logs")
            self.assertFalse((path / ".stage_complete.json").exists())

    def test_advisor_identity_must_remain_frozen(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch("requests.get") as get:
                get.return_value.json.return_value = {"margent_advisor": {"model": "A", "revision": "1"}}
                verify_advisor({"advisor_url": "http://unused"}, tmp)
                get.return_value.json.return_value = {"margent_advisor": {"model": "A", "revision": "2"}}
                with self.assertRaisesRegex(ValueError, "changed"):
                    verify_advisor({"advisor_url": "http://unused"}, tmp)


if __name__ == "__main__":
    unittest.main()
