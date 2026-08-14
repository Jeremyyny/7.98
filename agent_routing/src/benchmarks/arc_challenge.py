"""ARC-Challenge benchmark loader."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from .base import StandardRow


HF_DEFAULT_DATASET = "allenai/ai2_arc"
HF_DEFAULT_CONFIG = "ARC-Challenge"
DEFAULT_SPLITS = ("train", "validation", "test")


def _normalize_split(split: str) -> str:
    value = str(split or "").strip().lower()
    return "dev" if value in {"validation", "valid", "val", "dev"} else value


def _extract_options(raw_choices: Any) -> Tuple[List[str], List[str]]:
    """Return source labels and texts while retaining official option order."""
    if isinstance(raw_choices, dict):
        texts = [str(item).strip() for item in (raw_choices.get("text") or [])]
        labels = [str(item).strip() for item in (raw_choices.get("label") or [])]
    elif isinstance(raw_choices, (list, tuple)):
        texts = []
        labels = []
        for idx, item in enumerate(raw_choices):
            if isinstance(item, dict):
                texts.append(str(item.get("text") or "").strip())
                labels.append(str(item.get("label") or "").strip())
            else:
                texts.append(str(item).strip())
                labels.append(chr(ord("A") + idx))
    else:
        return [], []

    if len(labels) != len(texts) or not all(labels):
        labels = [chr(ord("A") + idx) for idx in range(len(texts))]
    return labels, texts


def _from_record(rec: Dict[str, Any], idx: int, source_split: str) -> Optional[StandardRow]:
    question = str(rec.get("question") or "").strip()
    source_labels, texts = _extract_options(rec.get("choices"))
    if not question or len(texts) < 2 or any(not text for text in texts):
        return None
    if len(texts) > 26:
        return None

    normalized_source_labels = [label.upper() for label in source_labels]
    if len(set(normalized_source_labels)) != len(normalized_source_labels):
        return None

    choices = {
        chr(ord("A") + option_idx): text
        for option_idx, text in enumerate(texts)
    }
    source_to_canonical = {
        source_label: chr(ord("A") + option_idx)
        for option_idx, source_label in enumerate(normalized_source_labels)
    }
    raw_answer = str(rec.get("answerKey") or rec.get("answer") or "").strip().upper()
    ground_truth = source_to_canonical.get(raw_answer, "")
    if not ground_truth:
        # Some mirrors canonicalize answerKey to A/B/... while leaving source
        # labels numeric. Accept that representation only when it is in range.
        if raw_answer in choices:
            ground_truth = raw_answer
        else:
            return None

    split = _normalize_split(rec.get("_source_split") or rec.get("split") or source_split)
    return StandardRow(
        example_id=idx,
        benchmark_name="arc_challenge",
        task_subtype=HF_DEFAULT_CONFIG,
        question=question,
        choices=choices,
        ground_truth=ground_truth,
        context="",
        metadata={
            "source_id": str(rec.get("id") or ""),
            "source_labels": source_labels,
            "n_options": len(choices),
        },
        split=split,
    )


def load_arc_challenge(
    dataset_name: str = HF_DEFAULT_DATASET,
    config_name: str = HF_DEFAULT_CONFIG,
    hf_cache_dir: Optional[str] = None,
    max_examples: int = 0,
    splits: "Sequence[str] | str" = DEFAULT_SPLITS,
) -> List[StandardRow]:
    """Load ARC-Challenge from Hugging Face into ``StandardRow`` objects."""
    from datasets import load_dataset

    if isinstance(splits, str):
        split_list = [item.strip() for item in splits.split(",") if item.strip()]
    else:
        split_list = [str(item).strip() for item in splits if str(item).strip()]
    if not split_list:
        split_list = list(DEFAULT_SPLITS)

    dataset = load_dataset(dataset_name, config_name, cache_dir=hf_cache_dir)
    rows: List[StandardRow] = []
    for requested_split in split_list:
        candidates = [requested_split]
        if _normalize_split(requested_split) == "dev":
            candidates = ["validation", "dev", "valid"]
        source_split = next((name for name in candidates if name in dataset), None)
        if source_split is None:
            continue
        for rec in dataset[source_split]:
            rec_dict = {"_source_split": _normalize_split(source_split), **dict(rec)}
            row = _from_record(rec_dict, len(rows), source_split)
            if row is not None:
                rows.append(row)
            if max_examples > 0 and len(rows) >= max_examples:
                break
        if max_examples > 0 and len(rows) >= max_examples:
            break

    for example_id, row in enumerate(rows):
        row.example_id = example_id
    return rows
