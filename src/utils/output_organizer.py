"""Organize jsonl.gz outputs by metadata.tag into CSV files.

Reads one .jsonl.gz file in the output root and one .jsonl.gz file
inside output/rejected, then writes two CSVs with texts grouped by tag.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class InputFile:
	label: str
	path: Path


def _find_single_jsonl_gz(folder: Path) -> Path:
	files = sorted(folder.glob("*.jsonl.gz"))
	if not files:
		raise FileNotFoundError(f"Nessun file .jsonl.gz trovato in {folder}")
	if len(files) > 1:
		names = ", ".join(f.name for f in files)
		raise RuntimeError(
			f"Trovati più file .jsonl.gz in {folder}: {names}. "
			"Specificare un file unico o ripulire la cartella."
		)
	return files[0]


def _iter_jsonl_gz(path: Path) -> Iterable[Dict]:
	with gzip.open(path, "rt", encoding="utf-8") as handle:
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				yield json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"JSON non valido in {path} alla riga {line_number}"
				) from exc


def _group_texts_by_tag(records: Iterable[Dict]) -> Dict[str, List[str]]:
	grouped: Dict[str, List[str]] = defaultdict(list)
	for item in records:
		metadata = item.get("metadata") or {}
		tag = metadata.get("tag")
		text = item.get("text")
		if tag is None or text is None:
			# Skip incomplete records, but keep processing.
			continue
		grouped[str(tag)].append(str(text))
	return grouped


def _write_grouped_csv(grouped: Dict[str, List[str]], output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(["tag", "count", "texts"])
		for tag, texts in sorted(grouped.items()):
			writer.writerow([tag, len(texts), "\n".join(texts)])


def _resolve_inputs(output_dir: Path) -> Tuple[InputFile, InputFile]:
	rejected_dir = output_dir / "rejected"
	if not rejected_dir.exists():
		raise FileNotFoundError(f"Cartella rejected non trovata: {rejected_dir}")

	root_file = _find_single_jsonl_gz(output_dir)
	rejected_file = _find_single_jsonl_gz(rejected_dir)

	return (
		InputFile(label="output", path=root_file),
		InputFile(label="rejected", path=rejected_file),
	)


def _default_csv_path(input_file: InputFile) -> Path:
	base = input_file.path.name
	if base.endswith(".jsonl.gz"):
		base = base[: -len(".jsonl.gz")]
	return input_file.path.parent / f"{base}.csv"


def run(output_dir: Path, output_csv_dir: Path | None = None) -> Tuple[Path, Path]:
	input_output, input_rejected = _resolve_inputs(output_dir)

	csv_paths: List[Path] = []
	for item in (input_output, input_rejected):
		grouped = _group_texts_by_tag(_iter_jsonl_gz(item.path))
		csv_path = (
			output_csv_dir / f"{item.label}.csv"
			if output_csv_dir
			else _default_csv_path(item)
		)
		_write_grouped_csv(grouped, csv_path)
		csv_paths.append(csv_path)

	return csv_paths[0], csv_paths[1]


def _build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Legge i file .jsonl.gz in output e output/rejected e genera "
			"due CSV con i testi raggruppati per metadata.tag."
		)
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path("output"),
		help="Cartella output montata dal container (default: output).",
	)
	parser.add_argument(
		"--csv-dir",
		type=Path,
		default=None,
		help=(
			"Cartella dove salvare i CSV. Se non indicata, usa la stessa "
			"cartella dei file di input."
		),
	)
	return parser


def main() -> None:
	parser = _build_parser()
	args = parser.parse_args()

	csv_output, csv_rejected = run(args.output_dir, args.csv_dir)
	print(f"CSV output: {csv_output}")
	print(f"CSV rejected: {csv_rejected}")


if __name__ == "__main__":
	main()
