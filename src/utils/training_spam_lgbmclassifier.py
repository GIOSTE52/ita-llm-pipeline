from __future__ import annotations
import argparse
import os
import sys
import warnings

# 1. Silenziamo i warning di Scikit-Learn e LightGBM per pulizia log
warnings.filterwarnings("ignore", category=UserWarning)

# 2. Aggiungiamo la cartella 'src' al path di sistema per evitare errori di import
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import assoluto dal tuo pacchetto blocks
from blocks.spam_classifier.spam_classifier import SpamClassifier

def main() -> None:
    parser = argparse.ArgumentParser(description="Training del classificatore spam")
    
    # Percorsi di default basati sulla tua struttura attuale
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
    
    args = parser.parse_args()

    print(f"Avvio training su: {args.csv_path}")
    
    try:
        # Addestramento
        result = SpamClassifier.train_from_csv(
            csv_path=args.csv_path,
            label_column=args.label_column,
        )
        
        # Creazione cartella models se manca
        os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
        
        # Salvataggio
        SpamClassifier.save_model(result, args.model_path)
        print(f"Modello salvato con successo in: {args.model_path}")
        
    except FileNotFoundError:
        print(f"ERRORE: Non trovo il file CSV in {args.csv_path}")
    except Exception as e:
        print(f"ERRORE durante il training: {e}")

if __name__ == "__main__":
    main()