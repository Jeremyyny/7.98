import json
import sys
import tempfile
import types
import unittest
from argparse import Namespace
from pathlib import Path

# The loader tests do not execute model code. Keep them runnable in a minimal
# data-only environment where PyTorch is not installed.
if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.manual_seed = lambda _seed: None
    torch_stub.cuda = types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda _seed: None)
    sys.modules["torch"] = torch_stub

from src.benchmarks.aqua_rat import _from_record as aqua_from_record
from src.benchmarks.aqua_rat import load_aqua_rat
from src.benchmarks.arc_challenge import _from_record as arc_from_record
from src.pipeline.cli import _requested_benchmark


class AquaRatLoaderTest(unittest.TestCase):
    def test_strips_embedded_labels_and_does_not_leak_rationale(self):
        row = aqua_from_record(
            {
                "question": "What is 20% of 50?",
                "options": ["A)5", "B)10", "C)15", "D)20", "E)25"],
                "correct": "B",
                "rationale": "SECRET GOLD SOLUTION. Answer is B.",
                "_source_split": "validation",
            },
            idx=4,
            source_split="validation",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row.choices["A"], "5")
        self.assertEqual(row.choices["B"], "10")
        self.assertEqual(row.ground_truth, "B")
        self.assertEqual(row.split, "dev")
        self.assertTrue(row.metadata["has_gold_rationale"])
        self.assertNotIn("SECRET GOLD SOLUTION", json.dumps(row.to_dict()))

    def test_handles_duplicate_option_prefix_and_verbose_answer(self):
        row = aqua_from_record(
            {
                "question": "Choose one.",
                "options": ["A)A)$60", "B)$50", "C)$40", "D)$30", "E)$20"],
                "correct": "Answer: A",
            },
            idx=0,
            source_split="train",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row.choices["A"], "$60")
        self.assertEqual(row.ground_truth, "A")

    def test_official_local_dev_file_is_normalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev.json"
            # The official repository uses one JSON object per line even
            # though the filenames end in .json rather than .jsonl.
            records = [{
                    "question": "2 + 2 = ?",
                    "options": ["A)1", "B)2", "C)3", "D)4", "E)5"],
                    "correct": "D",
                    "rationale": "2 + 2 is 4.",
                }, {
                    "question": "3 + 3 = ?",
                    "options": ["A)2", "B)3", "C)4", "D)5", "E)6"],
                    "correct": "E",
                    "rationale": "3 + 3 is 6.",
                }]
            path.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
            rows = load_aqua_rat(source="local", local_path=tmp, splits="validation")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0].split, "dev")
        self.assertEqual(rows[0].ground_truth, "D")


class ArcChallengeLoaderTest(unittest.TestCase):
    def test_maps_source_labels_to_canonical_letters(self):
        row = arc_from_record(
            {
                "id": "arc-1",
                "question": "Which material conducts electricity?",
                "choices": {
                    "label": ["1", "2", "3", "4"],
                    "text": ["wood", "copper", "glass", "rubber"],
                },
                "answerKey": "2",
                "_source_split": "validation",
            },
            idx=0,
            source_split="validation",
        )
        self.assertIsNotNone(row)
        self.assertEqual(row.choices, {
            "A": "wood", "B": "copper", "C": "glass", "D": "rubber",
        })
        self.assertEqual(row.ground_truth, "B")
        self.assertEqual(row.split, "dev")
        self.assertEqual(row.metadata["source_labels"], ["1", "2", "3", "4"])

    def test_keeps_variable_option_count(self):
        row = arc_from_record(
            {
                "id": "arc-3",
                "question": "Choose the gas.",
                "choices": {
                    "label": ["A", "B", "C"],
                    "text": ["iron", "oxygen", "salt"],
                },
                "answerKey": "B",
            },
            idx=0,
            source_split="test",
        )
        self.assertIsNotNone(row)
        self.assertEqual(len(row.choices), 3)
        self.assertEqual(row.ground_truth, "B")

    def test_official_splits_remain_disjoint(self):
        rows = []
        for idx, split in enumerate(["train", "train", "dev", "test"]):
            row = arc_from_record(
                {
                    "id": f"arc-{idx}",
                    "question": f"Question {idx}",
                    "choices": {"label": ["A", "B"], "text": ["x", "y"]},
                    "answerKey": "A",
                },
                idx=idx,
                source_split=split,
            )
            rows.append(row)
        self.assertEqual([row.split for row in rows], ["train", "train", "dev", "test"])


def _selection_args(**overrides):
    values = {
        "stage": "build_marginal_sft",
        "benchmark": "auto",
        "medqa_normalized_cache": "",
        "medqa_local_path": "",
        "medqa_refresh_cache": False,
        "legalbench_normalized_cache": "",
        "legalbench_configs": "",
        "legalbench_refresh_cache": False,
        "gpqa_normalized_cache": "",
        "gpqa_refresh_cache": False,
        "mmlu_pro_normalized_cache": "",
        "mmlu_pro_categories": "",
        "mmlu_pro_refresh_cache": False,
        "aqua_rat_normalized_cache": "",
        "aqua_rat_local_path": "",
        "aqua_rat_refresh_cache": False,
        "arc_challenge_normalized_cache": "",
        "arc_challenge_refresh_cache": False,
    }
    values.update(overrides)
    return Namespace(**values)


class BenchmarkSelectionTest(unittest.TestCase):
    def test_no_dataset_flag_preserves_medqa_default(self):
        self.assertEqual(_requested_benchmark(_selection_args()), "medqa")

    def test_new_cache_selects_requested_benchmark(self):
        args = _selection_args(
            benchmark="aqua_rat",
            aqua_rat_normalized_cache="outputs/data/aqua.jsonl",
        )
        self.assertEqual(_requested_benchmark(args), "aqua_rat")

    def test_multiple_dataset_flags_raise_instead_of_using_priority(self):
        args = _selection_args(
            aqua_rat_normalized_cache="aqua.jsonl",
            arc_challenge_normalized_cache="arc.jsonl",
        )
        with self.assertRaisesRegex(ValueError, "Multiple benchmarks"):
            _requested_benchmark(args)

    def test_load_stage_cannot_disagree_with_explicit_benchmark(self):
        args = _selection_args(stage="load_arc_challenge", benchmark="aqua_rat")
        with self.assertRaisesRegex(ValueError, "conflicts"):
            _requested_benchmark(args)


if __name__ == "__main__":
    unittest.main()
