"""Organize jsonl.gz outputs by metadata.tag into CSV files.

Reads one .jsonl.gz file in the output root and one .jsonl.gz file
inside output/rejected, then writes two CSVs with texts grouped by tag.
"""

from __future__ import annotations

import os
import argparse
import csv
import gzip
import json
from collections import defaultdict		# defaultdict permette di definire un dizionario al quale si associa ad ogni nuova chiave un valore di default
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class InputFile:
	label: str
	path: Path


def find_single_jsonl_gz(folder: Path) -> Path:
	# sorted() restituisce una nuova lista ordinata da un iterabile
	# mentre glob() viene usato per trovare file e directory all'interno del fs i quali corrispondono al pattern definito (in stile Unix)
	# Quindi l'istruzione seguente cerca e ordina in una lista tutti i file jsonl.gz
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


def iter_jsonl_gz(path: Path) -> Iterable[Dict]:
	with gzip.open(path, "rt", encoding="utf-8") as handle:
		# enumerate() restituisce una coppia (int, obj[int])
		for line_number, line in enumerate(handle, start=1):
			line = line.strip()
			if not line:
				continue
			try:
				# restituisce un iterabile sul json trsformato in dizionario
				yield json.loads(line)
			except json.JSONDecodeError as exc:
				raise ValueError(
					f"JSON non valido in {path} alla riga {line_number}"
				) from exc


def group_texts_by_tag(records: Iterable[Dict]) -> Dict[str, List[str]]:
	grouped: Dict[str, List[str]] = defaultdict(list)
	for item in records:
		metadata = item.get("metadata") or {}
		tag = metadata.get("tag")
		text = item.get("text")
		if tag is None or text is None:
			# Salto i record al quale manca il tag o il text
			continue
		# il dizionario grouped raccoglie una lista di testi (valori) sotto una delle chiavi tag ("bad","middle","good")
		grouped[str(tag)].append(str(text))
	return grouped


def write_grouped_csv(grouped: Dict[str, List[str]], output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.writer(handle)
		writer.writerow(["tag", "lenght", "texts"])
		for tag, texts in sorted(grouped.items()):
			writer.writerow([tag, len(texts), "\n".join(texts)])


def resolve_inputFile_objects(output_dir: Path | str) -> Tuple[InputFile, InputFile]:
	output_dir = Path(output_dir)
	rejected_dir = output_dir / "rejected"
	if not rejected_dir.exists():
		raise FileNotFoundError(f"Cartella rejected non trovata: {rejected_dir}")

	root_file = find_single_jsonl_gz(output_dir)
	rejected_file = find_single_jsonl_gz(rejected_dir)

	return (
		InputFile(label="output", path=root_file),
		InputFile(label="rejected", path=rejected_file),
	)


def default_csv_path(input_file: InputFile) -> Path:
	base = input_file.path.name
	if base.endswith(".jsonl.gz"):
		# taglio l'estensione e quindi estraggo il nome del file jsonl.gz per usarlo come nome del file csv
		base = base[: -len(".jsonl.gz")]
	return input_file.path.parent / f"{base}.csv"


def run(output_dir: Path | str, output_csv_dir: Path | str | None = None) -> Tuple[Path, Path]:
	output_dir = Path(output_dir)
	if output_csv_dir:
		output_csv_dir = Path(output_csv_dir)
	inputFileObj_output, inputFileObj_rejected = resolve_inputFile_objects(output_dir)

	csv_paths: List[Path] = []
	for item in (inputFileObj_output, inputFileObj_rejected):
		grouped = group_texts_by_tag(iter_jsonl_gz(item.path))
		csv_path = (
			output_csv_dir / f"{item.label}.csv"
			if output_csv_dir
			else default_csv_path(item)
		)
		write_grouped_csv(grouped, csv_path)
		csv_paths.append(csv_path)

	return csv_paths[0], csv_paths[1]


# def extract_args() -> argparse.ArgumentParser:
# 	parser = argparse.ArgumentParser(
# 		description=(
# 			"Legge i file .jsonl.gz in output e output/rejected e genera "
# 			"due CSV con i testi raggruppati per metadata.tag"
# 		)
# 	)
# 	parser.add_argument(
# 		"--output-dir",
# 		type=Path,
# 		default=Path("output"),
# 		help="Cartella output montata dal container (default: output).",
# 	)
# 	parser.add_argument(
# 		"--csv-dir",
# 		type=Path,
# 		default=None,
# 		help=(
# 			"Cartella dove salvare i CSV. Se non indicata, usa la stessa "
# 			"cartella dei file di input."
# 		),
# 	)
# 	return parser

def pie_graph(grouped: Dict[str, List[str]], title: str | None = None) -> None:
	plt.style.use('_mpl-gallery-nogrid')

	if not grouped:
		return

	labels: List[str] = []
	values: List[int] = []
	for tag, texts in grouped.items():
		labels.append(str(tag))
		values.append(len(texts))

	colors = plt.get_cmap('Blues')(np.linspace(0.2, 0.7, len(values)))

	fig, axes = plt.subplots()
	axes.pie(
		values,
		labels=labels,
		colors=colors, # type: ignore
		autopct="%1.1f%%",
		wedgeprops={"linewidth": 1, "edgecolor": "white"},
	)
	if title:
		axes.set_title(title)
	axes.axis("equal")
	plt.show()
	

def output_classification(output_dir: Path | str, output_csv_dir: Path | str | None = None) -> None:
	output_dir = Path(output_dir)
	if output_csv_dir:
		output_csv_dir = Path(output_csv_dir)
	run(output_dir, output_csv_dir)
	inputFileObj_output, inputFileObj_rejected = resolve_inputFile_objects(output_dir)

	for item in (inputFileObj_output, inputFileObj_rejected):
		grouped = group_texts_by_tag(iter_jsonl_gz(item.path))
		pie_graph(grouped, title=item.label)
	return

