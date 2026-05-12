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

    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Esegue anche un confronto modelli in cross validation.",
    )

    parser.add_argument(
        "--comparison-csv",
        default=None,
        help=(
            "CSV etichettato da usare per la cross validation. "
            "Se omesso usa --test-csv."
        ),
    )

    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero di fold per la cross validation. Default: 5.",
    )

    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=42,
        help="Seed per la cross validation. Default: 42.",
    )

    parser.add_argument(
        "--cv-models",
        nargs="+",
        default=None,
        help=(
            "Lista opzionale di modelli da confrontare. "
            "Valori ammessi: lightgbm logistic_regression random_forest extra_trees dummy"
        ),
    )

    parser.add_argument(
        "--no-feature-importance",
        action="store_true",
        help="Disattiva il calcolo della permutation importance.",
    )

    parser.add_argument(
        "--importance-review-epsilon",
        type=float,
        default=0.001,
        help=(
            "Soglia sotto cui una feature viene considerata quasi nulla "
            "nella permutation importance. Default: 0.001."
        ),
    )

    parser.add_argument(
        "--threshold-sweep",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Lista di soglie da valutare in parallelo. "
            "Esempio: --threshold-sweep 0.40 0.50 0.60 0.70 0.80"
        ),
    )


    args = parser.parse_args()

    evaluate_spam_model(
        model_path=args.model,
        test_csv=args.test_csv,
        output_dir=args.output_dir,
        label_column=args.label_column,
        threshold=args.threshold,
        save_feature_importance=not args.no_feature_importance,
        compare_models=args.compare_models,
        comparison_csv=args.comparison_csv,
        cv_folds=args.cv_folds,
        cv_random_state=args.cv_random_state,
        cv_model_names=args.cv_models,
        importance_review_epsilon=args.importance_review_epsilon,
        threshold_sweep=args.threshold_sweep,
    )



if __name__ == "__main__":
    main()



