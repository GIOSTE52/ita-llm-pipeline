from __future__ import annotations
import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from blocks.spam_classifier.spam_classifier import SpamClassifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Training del classificatore spam")
    
    parser.add_argument(
        "--csv-path",
        type=str,
        default=os.path.join("output", "feature", "spam_doc_features.csv"),
        help="Percorso del CSV con le 55 feature",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("models", "spam_lgbm.joblib"),
        help="Dove salvare il modello finale",
    )
    parser.add_argument(
        "--label-column",
        type=str,
        default=None,
        help="Colonna label (automatico se None)",
    )
    parser.add_argument(
        "--errors-dir",
        type=str,
        default=os.path.join("output", "debug", "spam"),
        help="Cartella dove salvare predizioni ed errori del test set",
    )



    
    args = parser.parse_args()

    print(f"🚀 Avvio training su: {args.csv_path}")
    
    try:
        result = SpamClassifier.train_from_csv(
            csv_path=args.csv_path,
            label_column=args.label_column,
            errors_output_dir=args.errors_dir,
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