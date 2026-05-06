#contiene la logica dell'evaluation
# scripts/evaluate_spam_model.py

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from blocks.spam_classifier.spam_evaluation import evaluate_spam_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Valuta il modello spam su un CSV di feature etichettato."
    )

    parser.add_argument(
        "--model",
        default="models/spam_lgbm.joblib",
        help="Path del modello spam joblib.",
    )

    parser.add_argument(
        "--test-csv",
        required=True,
        help="CSV di test con feature spam e label ham/spam.",
    )

    parser.add_argument(
        "--output-dir",
        default="evaluation/spam",
        help="Cartella dove salvare i report.",
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help="Colonna label da usare. Se omessa prova spam_target_label e target_label.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Soglia spam. Se omessa usa quella salvata nel modello.",
    )

    args = parser.parse_args()

    evaluate_spam_model(
        model_path=args.model,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        label_column=args.label_column,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()


