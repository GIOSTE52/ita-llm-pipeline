from __future__ import annotations
import argparse
import os
import sys
import warnings

warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.blocks.spam_classifier.spam_classifier import SpamClassifier


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train LightGBM spam classifier from spam feature CSV."
    )

    parser.add_argument(
        "--csv-path",
        default="output/feature/spam_doc_features.csv",
        help="CSV con feature spam.",
    )

    parser.add_argument(
        "--model-path",
        default="models/spam_lgbm.joblib",
        help="Path dove salvare il modello.",
    )

    parser.add_argument(
        "--label-column",
        default=None,
        help="Colonna label. Se omessa prova spam_target_label e target_label.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.60,
        help="Soglia di classificazione spam.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Quota dati usata come test split.",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed per split e training.",
    )

    parser.add_argument(
        "--errors-output-dir",
        default="evaluation/spam",
        help="Cartella dove salvare predizioni ed errori del test split.",
    )



    
    args = parser.parse_args()

    print(f"🚀 Avvio training su: {args.csv_path}")
    
    try:
        result = SpamClassifier.train_from_csv(
            csv_path=args.csv_path,
            label_column=args.label_column,
            threshold=args.threshold,
            test_size=args.test_size,
            random_state=args.random_state,
            errors_output_dir=args.errors_output_dir,
        )

        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        SpamClassifier.save_model(result, args.model_path)
        print(f"✅ Modello salvato con successo in: {args.model_path}")
        
    except FileNotFoundError:
        print(f"❌ ERRORE: Non trovo il file CSV in {args.csv_path}")
    except Exception as e:
        print(f"❌ ERRORE durante il training: {e}")

if __name__ == "__main__":
    main()