#!/usr/bin/env python3
"""
Count good/bad labels in JSONL shards and report dataset imbalance.

Example:
    python3 src/utils/counter_data.py
    python3 src/utils/counter_data.py --data-dir data/train --json-out reports/train_balance.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path
from typing import Iterable


VALID_LABELS = {"good", "bad"}


def iter_jsonl_files(data_dir: Path, pattern: str) -> list[Path]:
    files = sorted(data_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {data_dir.resolve()}"
        )
    return files


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    label = str(value).strip().lower()
    return label if label in VALID_LABELS else label


def count_labels_in_file(path: Path, label_key: str) -> dict:
    counts: Counter[str] = Counter()
    total_rows = 0
    malformed_rows = 0
    missing_label_rows = 0
    invalid_label_rows = 0

    with open_text(path) as handle:
        for line in handle:
            if not line.strip():
                continue

            total_rows += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue

            label = normalize_label(record.get(label_key))
            if label is None:
                missing_label_rows += 1
                continue

            if label not in VALID_LABELS:
                invalid_label_rows += 1
                counts[f"invalid:{label}"] += 1
                continue

            counts[label] += 1

    return {
        "file": str(path),
        "total_rows": total_rows,
        "good": counts["good"],
        "bad": counts["bad"],
        "malformed_rows": malformed_rows,
        "missing_label_rows": missing_label_rows,
        "invalid_label_rows": invalid_label_rows,
        "invalid_labels": {
            key.replace("invalid:", "", 1): value
            for key, value in counts.items()
            if key.startswith("invalid:")
        },
    }


def summarize(file_stats: Iterable[dict]) -> dict:
    files = list(file_stats)
    total_rows = sum(item["total_rows"] for item in files)
    good = sum(item["good"] for item in files)
    bad = sum(item["bad"] for item in files)
    usable = good + bad
    malformed = sum(item["malformed_rows"] for item in files)
    missing = sum(item["missing_label_rows"] for item in files)
    invalid = sum(item["invalid_label_rows"] for item in files)

    if usable:
        good_pct = good / usable * 100
        bad_pct = bad / usable * 100
    else:
        good_pct = 0.0
        bad_pct = 0.0

    majority_label = "good" if good >= bad else "bad"
    majority_count = max(good, bad)
    minority_label = "bad" if majority_label == "good" else "good"
    minority_count = min(good, bad)
    majority_baseline = majority_count / usable if usable else 0.0
    imbalance_ratio = (
        majority_count / minority_count if minority_count else float("inf")
    )
    minority_share = minority_count / usable if usable else 0.0

    return {
        "files": len(files),
        "total_rows": total_rows,
        "usable_labeled_rows": usable,
        "good": good,
        "bad": bad,
        "good_pct": good_pct,
        "bad_pct": bad_pct,
        "majority_label": majority_label,
        "majority_count": majority_count,
        "minority_label": minority_label,
        "minority_count": minority_count,
        "minority_share": minority_share,
        "majority_baseline_accuracy": majority_baseline,
        "imbalance_ratio_majority_to_minority": imbalance_ratio,
        "malformed_rows": malformed,
        "missing_label_rows": missing,
        "invalid_label_rows": invalid,
        "per_file": files,
    }


def imbalance_level(ratio: float) -> str:
    if ratio == float("inf"):
        return "single-class"
    if ratio < 1.5:
        return "low"
    if ratio < 3:
        return "moderate"
    if ratio < 10:
        return "high"
    return "severe"


def print_report(summary: dict, show_per_file: bool) -> None:
    ratio = summary["imbalance_ratio_majority_to_minority"]
    ratio_text = "infinite" if ratio == float("inf") else f"{ratio:.2f}:1"
    level = imbalance_level(ratio)

    print("\nDATASET LABEL BALANCE")
    print("=" * 72)
    print(f"Files scanned:         {summary['files']}")
    print(f"Total JSONL rows:      {summary['total_rows']:,}")
    print(f"Usable labeled rows:   {summary['usable_labeled_rows']:,}")
    print(f"Good:                  {summary['good']:,} ({summary['good_pct']:.2f}%)")
    print(f"Bad:                   {summary['bad']:,} ({summary['bad_pct']:.2f}%)")
    print(f"Majority class:        {summary['majority_label']}")
    print(f"Minority class:        {summary['minority_label']}")
    print(f"Imbalance ratio:       {ratio_text} ({level})")
    print(
        "Majority baseline:     "
        f"{summary['majority_baseline_accuracy']:.4f} accuracy "
        "if always predicting the majority class"
    )

    data_issues = (
        summary["malformed_rows"]
        + summary["missing_label_rows"]
        + summary["invalid_label_rows"]
    )
    if data_issues:
        print("\nDATA ISSUES")
        print("-" * 72)
        print(f"Malformed JSON rows:   {summary['malformed_rows']:,}")
        print(f"Missing labels:        {summary['missing_label_rows']:,}")
        print(f"Invalid labels:        {summary['invalid_label_rows']:,}")

    print("\nMETRIC SUGGESTION")
    print("-" * 72)
    if level == "low":
        print(
            "Use ROC-AUC for ranking plus F1, precision, recall, and a confusion "
            "matrix at your chosen threshold."
        )
    else:
        print(
            "Prefer balanced accuracy, macro F1, MCC, per-class precision/recall, "
            "and PR-AUC for the minority class. Plain accuracy can look good while "
            "ignoring the minority class."
        )
    print(
        "Also compare against the majority-class baseline above; your model "
        "should beat it meaningfully."
    )

    if show_per_file:
        print("\nPER-SHARD COUNTS")
        print("-" * 72)
        print(f"{'file':34} {'rows':>10} {'good':>10} {'bad':>10}")
        for item in summary["per_file"]:
            print(
                f"{Path(item['file']).name:34}"
                f"{item['total_rows']:>10,}"
                f"{item['good']:>10,}"
                f"{item['bad']:>10,}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count good/bad labels in JSONL shards and report imbalance."
    )
    parser.add_argument(
        "--data-dir",
        default="data/train",
        type=Path,
        help="Directory containing JSONL shards (default: data/train).",
    )
    parser.add_argument(
        "--pattern",
        default="*.jsonl",
        help="Glob pattern for shard files (default: *.jsonl).",
    )
    parser.add_argument(
        "--label-key",
        default="label",
        help="JSON field containing the label (default: label).",
    )
    parser.add_argument(
        "--per-file",
        action="store_true",
        help="Print counts for each shard.",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path where the full summary is saved as JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    files = iter_jsonl_files(args.data_dir, args.pattern)
    file_stats = [count_labels_in_file(path, args.label_key) for path in files]
    summary = summarize(file_stats)

    print_report(summary, show_per_file=args.per_file)

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()
