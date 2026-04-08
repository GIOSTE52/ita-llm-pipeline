#!/usr/bin/env python3
"""
Script per valutare il classificatore di qualità su un dataset di test.

comando:
    python scripts/evaluate_model.py \\
        --model models/lgbm_quality_model.joblib \\
        --test-csv data/test/dataset_test.csv \\
        --output-dir output/evaluation \\
        --threshold 0.65

Questo script:
1. Carica il modello addestrato
2. Valuta il modello su un dataset di test
3. Stampa le statistiche a schermo
4. Salva un report dettagliato in JSON e HTML
"""

import argparse
import sys
import os

# Aggiungo src/ al path per importare i moduli del progetto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocks.classifiers import QualityClassifier


def main():
    parser = argparse.ArgumentParser(
        description="Valuta il classificatore di qualità su un dataset di test"
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Percorso al modello (.joblib)"
    )
    parser.add_argument(
        "--test-csv",
        required=True,
        help="Percorso al CSV di test con feature e label"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory per salvare i report (JSON, CSV, HTML). Se None, stampa a schermo."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.65,
        help="Soglia di decisione (default 0.65)"
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Nome della colonna con le label nel CSV (default 'label')"
    )

    args = parser.parse_args()

    # Valida gli argomenti
    if not os.path.exists(args.model):
        print(f"Errore: Modello non trovato: {args.model}")
        sys.exit(1)

    if not os.path.exists(args.test_csv):
        print(f"Errore: Dataset non trovato: {args.test_csv}")
        sys.exit(1)

    try:
        # Carica il modello
        print("\nCaricamento modello...")
        classifier = QualityClassifier(
            model_path=args.model,
            threshold=args.threshold
        )
        print("Modello caricato con successo!")

        # Valuta il modello
        print("\nValutazione in corso...")
        result = classifier.evaluate(
            csv_path=args.test_csv,
            label_column=args.label_column,
            output_dir=args.output_dir
        )

        print("\nValutazione completata con successo!")

        # Stampo i risultati più importanti
        print("\n" + "=" * 80)
        print("METRICHE INCIDENTI")
        print("=" * 80)
        print(f"Accuracy:          {result['accuracy']:.4f}")
        print(f"Balanced Accuracy: {result['balanced_accuracy']:.4f}")
        print(f"F1-Score:          {result['f1_score']:.4f}")
        print(f"ROC-AUC:           {result['roc_auc']:.4f}")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\nErrore durante la valutazione:")
        print(f"{type(e).__name__}: {str(e)}")
        # Importo il modulo traceback che mi permette di stampare a schermo l'origine dell'errore nel dettaglio
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
