#!/usr/bin/env python3
"""
Script per valutare il classificatore di qualità su un dataset di test scritto in un CSV.
Sono necessarie le label, quindi bisogna generare il CSV su un dataset con label presenti.

comando:
    python3 scripts/evaluate_model.py --model models/lgbm_quality_model.joblib --output-dir evaluation --compare-models

Questo script:
1. Carica il modello addestrato
2. Recupera il test split corretto dal modello oppure usa quello passato da CLI
3. Opzionalmente confronta piu modelli con cross validation
4. Stampa le statistiche a schermo
5. Salva un report dettagliato in JSON e HTML
"""

import argparse
import sys
import os
import joblib

# Aggiungo src/ al path per importare i moduli del progetto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from blocks.classifiers import QualityClassifier


def load_model_metadata(model_path: str) -> dict:
    """Carica i metadati utili del modello senza alterare il flusso di valutazione."""
    artifact = joblib.load(model_path)
    return {
        "threshold": artifact.get("threshold"),
        "training_metadata": artifact.get("training_metadata", {}),
        "model_name": artifact.get("model_name"),
    }


def resolve_test_csv(explicit_test_csv: str | None, training_metadata: dict) -> str | None:
    """Recupera il test set corretto, preferendo quello registrato nel modello."""
    if explicit_test_csv:
        return explicit_test_csv
    return training_metadata.get("test_csv")


def print_model_comparison(comparison_result: dict) -> None:
    """Stampa un confronto compatto tra i modelli valutati in cross validation."""
    print("\n" + "=" * 108)
    print(
        f"CONFRONTO MODELLI ({comparison_result['cv_folds']}-fold stratified CV)"
    )
    print("=" * 108)
    print(
        "Baseline: "
        f"{comparison_result['baseline_model_name']} | "
        f"Soglia: {comparison_result['threshold']}"
    )
    print(
        f"{'Modello':24}"
        f"{'ROC-AUC CV':>16}"
        f"{'F1 CV':>16}"
        f"{'Bal. Acc CV':>16}"
        f"{'Delta ROC-AUC':>18}"
        f"{'Delta F1':>12}"
    )
    print("-" * 108)
    for row in comparison_result["models"]:
        roc_auc_cell = f"{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}"
        f1_cell = f"{row['f1_score_mean']:.4f} ± {row['f1_score_std']:.4f}"
        bal_acc_cell = (
            f"{row['balanced_accuracy_mean']:.4f} ± "
            f"{row['balanced_accuracy_std']:.4f}"
        )
        delta_roc_auc_cell = f"{row['delta_vs_baseline']['roc_auc_mean']:+.4f}"
        delta_f1_cell = f"{row['delta_vs_baseline']['f1_score_mean']:+.4f}"
        print(
            f"{row['model_name']:24}"
            f"{roc_auc_cell:>16}"
            f"{f1_cell:>16}"
            f"{bal_acc_cell:>16}"
            f"{delta_roc_auc_cell:>18}"
            f"{delta_f1_cell:>12}"
        )
    print("-" * 108)
    winner = comparison_result["winner"]
    print(
        "Miglior modello per ROC-AUC medio: "
        f"{winner['model_name']} "
        f"(ROC-AUC {winner['roc_auc_mean']:.4f}, F1 {winner['f1_score_mean']:.4f})"
    )
    print("=" * 108 + "\n")


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
        default=None,
        help=(
            "Percorso al CSV di test con feature e label. "
            "Se omesso prova a usare il test split salvato nel modello."
        )
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory per salvare i report (JSON, CSV, HTML). Se None, stampa a schermo."
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Soglia di decisione. Se omessa usa quella salvata nel modello, "
            "altrimenti 0.65."
        )
    )
    parser.add_argument(
        "--label-column",
        default="label",
        help="Nome della colonna con le label nel CSV (default 'label')"
    )
    parser.add_argument(
        "--compare-models",
        action="store_true",
        help="Esegue anche un benchmark in cross validation con modelli alternativi."
    )
    parser.add_argument(
        "--comparison-csv",
        default=None,
        help=(
            "CSV etichettato da usare per la cross validation. "
            "Se omesso usa il dataset sorgente registrato nel modello oppure il test CSV risolto."
        )
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Numero di fold per la cross validation (default 5)."
    )
    parser.add_argument(
        "--cv-random-state",
        type=int,
        default=42,
        help="Seed per split e modelli della cross validation (default 42)."
    )
    parser.add_argument(
        "--cv-models",
        nargs="+",
        default=None,
        help=(
            "Lista opzionale di modelli da confrontare. "
            "Valori supportati: lightgbm random_forest extra_trees logistic_regression"
        )
    )

    args = parser.parse_args()

    # Valida gli argomenti
    if not os.path.exists(args.model):
        print(f"Errore: Modello non trovato: {args.model}")
        sys.exit(1)

    model_metadata = load_model_metadata(args.model)
    training_metadata = model_metadata.get("training_metadata", {})
    resolved_test_csv = resolve_test_csv(args.test_csv, training_metadata)

    if resolved_test_csv is None:
        print(
            "Errore: impossibile determinare il test set. "
            "Passa --test-csv oppure usa un modello salvato con i metadati dello split."
        )
        sys.exit(1)

    if not os.path.exists(resolved_test_csv):
        print(f"Errore: CSV di test non trovato: {resolved_test_csv}")
        sys.exit(1)

    if (
        args.test_csv
        and training_metadata.get("test_csv")
        and os.path.abspath(args.test_csv) != training_metadata["test_csv"]
    ):
        print(
            "Avviso: stai valutando su un CSV diverso dal test split registrato nel modello."
        )
        print(f"   Test registrato: {training_metadata['test_csv']}")
        print(f"   Test richiesto:  {os.path.abspath(args.test_csv)}")

    try:
        threshold = args.threshold
        if threshold is None:
            threshold = model_metadata.get("threshold") or 0.65

        # Carica il modello
        print("\nCaricamento modello...")
        classifier = QualityClassifier(
            model_path=args.model,
            threshold=threshold
        )
        print("Modello caricato con successo!")
        print(f"Threshold in uso: {threshold}")
        print(f"Test CSV in uso: {resolved_test_csv}")
        if training_metadata:
            print("Split di training trovati nel modello:")
            print(f"   Source: {training_metadata.get('source_csv')}")
            print(f"   Train:  {training_metadata.get('train_csv')}")
            print(f"   Val:    {training_metadata.get('validation_csv')}")
            print(f"   Test:   {training_metadata.get('test_csv')}")

        comparison_result = None
        if args.compare_models:
            comparison_csv = (
                args.comparison_csv
                or training_metadata.get("source_csv")
                or resolved_test_csv
            )
            if not os.path.exists(comparison_csv):
                print(f"Errore: Dataset per cross validation non trovato: {comparison_csv}")
                sys.exit(1)
            print("\nBenchmark cross validation in corso...")
            print(f"Dataset CV in uso: {comparison_csv}")
            comparison_result = QualityClassifier.cross_validate_models(
                csv_path=comparison_csv,
                label_column=args.label_column,
                threshold=threshold,
                cv_folds=args.cv_folds,
                random_state=args.cv_random_state,
                model_names=args.cv_models,
            )
            print_model_comparison(comparison_result)


#   Da qui in poi viene usato il metodo del classificatore. Da modificare se si vuole integrare la valutazione in questo script o in report.ipynb
        # Valutazione del modello con il metodo associato
        print("\nValutazione in corso...")
        result = classifier.evaluate(
            csv_path=resolved_test_csv,
            label_column=args.label_column,
            output_dir=args.output_dir,
            comparison_result=comparison_result,
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
