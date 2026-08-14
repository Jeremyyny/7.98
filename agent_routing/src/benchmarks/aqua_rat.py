"""AQuA-RAT benchmark loader.

The public ``deepmind/aqua_rat`` dataset contains five-choice algebra word
problems with gold rationales.  Rationales are deliberately *not* copied into
``StandardRow``: they are answer-bearing annotations and exposing them to the
manager or advisors would leak supervision into the runtime input.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .base import StandardRow
from ..utils.io import read_json, read_jsonl


HF_DEFAULT_DATASET = "deepmind/aqua_rat"
HF_DEFAULT_CONFIG = "raw"
DEFAULT_SPLITS = ("train", "validation", "test")


def _normalize_split(split: str) -> str:
    value = str(split or "").strip().lower()
    if value in {"validation", "valid", "val", "dev"}:
        return "dev"
    return value


def _strip_option_prefix(text: Any, label: str) -> str:
    """Remove AQuA's embedded ``A)``/``(A)`` prefix from one option."""
    value = str(text or "").strip()
    escaped = re.escape(label)
    prefix = re.compile(
        rf"^\s*(?:\(\s*{escaped}\s*\)|{escaped}\s*[\)\.:])\s*",
        flags=re.IGNORECASE,
    )
    # A few source rows contain a duplicated prefix such as ``A)A)$60``.
    for _ in range(2):
        cleaned = prefix.sub("", value, count=1)
        if cleaned == value:
            break
        value = cleaned.strip()
    return value


def _iter_json_records(raw: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(raw, list):
        for rec in raw:
            if isinstance(rec, dict):
                yield rec
    elif isinstance(raw, dict):
        # Support either an id -> record mapping or a single record.
        if "question" in raw:
            yield raw
        else:
            for rec in raw.values():
                if isinstance(rec, dict):
                    yield rec


def _read_records_file(path: Path) -> List[Dict[str, Any]]:
    """Read JSON arrays and the official JSONL files that use a .json suffix."""
    if path.suffix.lower() == ".jsonl":
        return list(_iter_json_records(read_jsonl(str(path))))
    try:
        return list(_iter_json_records(read_json(str(path))))
    except json.JSONDecodeError:
        return list(_iter_json_records(read_jsonl(str(path))))


def _from_record(rec: Dict[str, Any], idx: int, source_split: str) -> Optional[StandardRow]:
    question = str(rec.get("question") or "").strip()
    raw_options = rec.get("options")
    if not question or not isinstance(raw_options, (list, tuple)):
        return None

    choices: Dict[str, str] = {}
    for option_idx, raw_option in enumerate(raw_options):
        if option_idx >= 26:
            break
        label = chr(ord("A") + option_idx)
        text = _strip_option_prefix(raw_option, label)
        if not text:
            return None
        choices[label] = text
    if len(choices) < 2:
        return None

    raw_correct = str(rec.get("correct") or "").strip().upper()
    if raw_correct in choices:
        ground_truth = raw_correct
    else:
        answer_tokens = re.findall(r"(?<![A-Z])[A-Z](?![A-Z])", raw_correct)
        ground_truth = next((token for token in reversed(answer_tokens) if token in choices), "")
    if ground_truth not in choices:
        return None

    split = _normalize_split(rec.get("_source_split") or rec.get("split") or source_split)
    has_rationale = bool(str(rec.get("rationale") or "").strip())
    return StandardRow(
        example_id=idx,
        benchmark_name="aqua_rat",
        task_subtype="algebra_word_problem",
        question=question,
        choices=choices,
        ground_truth=ground_truth,
        context="",
        metadata={
            "n_options": len(choices),
            "source_split": split,
            # Record only presence. The answer-bearing rationale must never
            # enter the normalized runtime cache.
            "has_gold_rationale": has_rationale,
        },
        split=split,
    )


def _load_local(path: str, splits: Sequence[str]) -> List[Dict[str, Any]]:
    source = Path(path)
    if source.is_file():
        return [{"_source_split": "", **rec} for rec in _read_records_file(source)]
    if not source.is_dir():
        raise FileNotFoundError(f"AQuA-RAT local path not found: {path}")

    records: List[Dict[str, Any]] = []
    seen_paths = set()
    for requested_split in splits:
        split = _normalize_split(requested_split)
        stems = ["dev", "validation", "valid"] if split == "dev" else [split]
        for stem in stems:
            candidates = [source / f"{stem}.json", source / f"{stem}.jsonl"]
            for file_path in candidates:
                if not file_path.exists() or file_path.resolve() in seen_paths:
                    continue
                seen_paths.add(file_path.resolve())
                for rec in _read_records_file(file_path):
                    records.append({"_source_split": split, **rec})
    if not records:
        raise FileNotFoundError(
            f"No AQuA-RAT split files found under {path}; expected train/dev/test .json or .jsonl files"
        )
    return records


def _load_hf(
    dataset_name: str,
    config_name: str,
    cache_dir: Optional[str],
    splits: Sequence[str],
) -> List[Dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, config_name or None, cache_dir=cache_dir)
    records: List[Dict[str, Any]] = []
    for requested_split in splits:
        candidates = [requested_split]
        if _normalize_split(requested_split) == "dev":
            candidates = ["validation", "dev", "valid"]
        source_split = next((name for name in candidates if name in dataset), None)
        if source_split is None:
            continue
        normalized_split = _normalize_split(source_split)
        for rec in dataset[source_split]:
            records.append({"_source_split": normalized_split, **dict(rec)})
    return records


def load_aqua_rat(
    source: str = "hf",
    dataset_name: str = HF_DEFAULT_DATASET,
    config_name: str = HF_DEFAULT_CONFIG,
    local_path: Optional[str] = None,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    splits: "Sequence[str] | str" = DEFAULT_SPLITS,
) -> List[StandardRow]:
    """Load AQuA-RAT into the pipeline's common ``StandardRow`` schema."""
    if isinstance(splits, str):
        split_list = [item.strip() for item in splits.split(",") if item.strip()]
    else:
        split_list = [str(item).strip() for item in splits if str(item).strip()]
    if not split_list:
        split_list = list(DEFAULT_SPLITS)

    if source == "hf":
        raw_records = _load_hf(dataset_name, config_name, hf_cache_dir, split_list)
    elif source == "local":
        if not local_path:
            raise ValueError("source='local' requires local_path")
        raw_records = _load_local(local_path, split_list)
    else:
        raise ValueError(f"Unknown AQuA-RAT source: {source}")

    rows: List[StandardRow] = []
    for rec in raw_records:
        row = _from_record(rec, len(rows), str(rec.get("_source_split") or ""))
        if row is not None:
            rows.append(row)
        if max_examples > 0 and len(rows) >= max_examples:
            break

    for example_id, row in enumerate(rows):
        row.example_id = example_id
    return rows
