from __future__ import annotations
import csv
import glob
import os
from typing import Optional
import sys


def aggregate_rank_csvs(
    feature_dir: str,
    final_name: str,
    label: Optional[str] = None,
    remove_parts: bool = True,
) -> bool:
    """
    Unisce i CSV temporanei prodotti dai worker DataTrove.

    Esempi:
        rank_0_doc_stats_per_file.csv
        rank_1_doc_stats_per_file.csv
        -> doc_stats_per_file.csv

        rank_0_spam_doc_features.csv
        rank_1_spam_doc_features.csv
        -> spam_doc_features.csv

    La funzione:
    - cerca i file rank_*_<final_name>
    - unisce i CSV evitando header duplicati
    - controlla che gli header siano coerenti
    - funziona sia su Linux sia su Windows
    - rimuove i file temporanei se remove_parts=True
    """

    csv.field_size_limit(sys.maxsize)
    label = label or final_name

    print("\n" + "=" * 60)
    print(f"AGGREGAZIONE FINALE CSV: {final_name}")
    print("=" * 60)

    os.makedirs(feature_dir, exist_ok=True)

    temp_pattern = os.path.join(feature_dir, f"rank_*_{final_name}")
    temp_files = sorted(glob.glob(temp_pattern))

    if not temp_files:
        print(f"[INFO] Nessun CSV temporaneo trovato per {label}.")
        print(f"[INFO] Pattern cercato: {temp_pattern}")
        return False

    final_output_csv = os.path.join(feature_dir, final_name)

    header = None
    total_rows = 0

    try:
        with open(final_output_csv, "w", newline="", encoding="utf-8") as fout:
            writer = None

            for path in temp_files:
                if not os.path.exists(path) or os.path.getsize(path) == 0:
                    print(f"[WARN] File vuoto o inesistente, salto: {path}")
                    continue

                with open(path, "r", newline="", encoding="utf-8") as fin:
                    reader = csv.reader(fin)

                    try:
                        current_header = next(reader)
                    except StopIteration:
                        print(f"[WARN] File senza header, salto: {path}")
                        continue

                    if header is None:
                        header = current_header
                        writer = csv.writer(fout)
                        writer.writerow(header)
                    elif current_header != header:
                        raise ValueError(
                            "Header CSV non coerente tra file temporanei.\n"
                            f"File problematico: {path}\n"
                            f"Header atteso: {header}\n"
                            f"Header trovato: {current_header}"
                        )

                    for row in reader:
                        if not row:
                            continue
                        writer.writerow(row)
                        total_rows += 1

        print(f"[OK] Unione {label} completata.")
        print(f"[OK] File finale: {final_output_csv}")
        print(f"[OK] File temporanei uniti: {len(temp_files)}")
        print(f"[OK] Righe dati scritte: {total_rows}")

        if remove_parts:
            for path in temp_files:
                try:
                    os.remove(path)
                except OSError as e:
                    print(f"[WARN] Non riesco a rimuovere {path}: {e}")

            print(f"[OK] File temporanei {label} rimossi.")

        return True

    except Exception as e:
        print(f"[ERRORE] Aggregazione CSV {label} fallita: {e}")
        return False