"""Tiny arithmetic fixtures for GPU plumbing checks; never paper results."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.benchmarks.base import StandardRow
from src.utils.io import write_json, write_jsonl


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    args = p.parse_args()
    root = Path(args.out)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError("Choose an empty smoke-data directory")
    root.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for split, numbers in [("train", range(1, 9)), ("dev", range(9, 13))]:
        rows = [StandardRow(i, "arithmetic_smoke_only", "free_response_math",
                f"Compute {i} + {i + 1}.", {}, str(2 * i + 1),
                metadata={"answer_type": "math"}, split=split).to_dict() for i in numbers]
        path = root / (split + ".jsonl")
        write_jsonl(str(path), rows)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(str(root / "manifest.json"), {"smoke_only": True, "sha256": hashes})


if __name__ == "__main__":
    main()
