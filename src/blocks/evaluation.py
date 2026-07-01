import os
import json
import logging
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from sklearn.inspection import permutation_importance

from .classifiers import QualityClassifier

logger = logging.getLogger(__name__)

"""
Logica per la valutazione del modello LightLGBM usato come classificatore nella classe QualityClassifier
"""

def evaluate_model(
    classifier: QualityClassifier,
    csv_path: str,
    label_column: str = "label",
    output_dir: Optional[str] = None,
    comparison_result: Optional[Dict[str, Any]] = None,
) -> dict:
    """Valuta il modello - versione standalone (non metodo della classe)"""
    # carico il csv e effettuo la validazione delle feature e delle label
    X, y, _ = QualityClassifier._load_labeled_dataset(
        csv_path, classifier.feature_names, label_column
    )
    
    # Scaling (coerente con lo scaler usato durante il training)
    X_scaled = pd.DataFrame(
        classifier.scaler.transform(X),
        columns=classifier.feature_names,
        index=X.index,
    )
    
    # Predizioni
    # calcolo la probabilità della classe positiva "good", seconda colonna di predict_proba
    y_pred_proba = classifier.model.predict_proba(X_scaled)[:, 1]
    # calcolo la predizione binaria in base alla posizione della probbilità rispetto alla soglia scelta
    y_pred = (y_pred_proba >= classifier.threshold).astype(int)
    
    # calcolo metriche principali
    metrics = QualityClassifier._compute_binary_metrics(y, y_pred, y_pred_proba)
    
    # calcolo la permutation importance
    perm = permutation_importance(
        classifier.model,
        X_scaled,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    # creo un dataFrame per visualizzare i valori di importance calcolati, e li ordino dal 
    # livello più alto di importance_mean in poi
    importance_df = pd.DataFrame({
        "feature": classifier.feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)
    
    # calcolo le curve ROC e PR
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
    
    # genero la confusion matrix
    cm = confusion_matrix(y, y_pred)
    # produco il report
    report_dict = classification_report(
        y, y_pred, target_names=["bad", "good"], output_dict=True, zero_division=0
    )
    
    print_evaluation_report(
        accuracy=metrics["accuracy"],
        balanced_acc=metrics["balanced_accuracy"],
        roc_auc=metrics["roc_auc"],
        f1=metrics["f1_score"],
        cm=cm,
        importance_df=importance_df,
        csv_path=csv_path,
        model_path=classifier.model_path,
        threshold=classifier.threshold,
    )
    
    # organizzo l'output
    result = {
        "accuracy": round(metrics["accuracy"], 4),
        "balanced_accuracy": round(metrics["balanced_accuracy"], 4),
        "roc_auc": round(metrics["roc_auc"], 4),
        "f1_score": round(metrics["f1_score"], 4),
        # "importance_df":importance_df,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "threshold": classifier.threshold,
        "feature_names": classifier.feature_names,
        "top_features": importance_df.head(10).to_dict(orient="records"),
        "csv_path": csv_path,
        "timestamp": datetime.now().isoformat(),
        "model_name": classifier.model_name,
        "training_metadata": classifier.training_metadata,
    }
    
    if comparison_result:
        result["model_comparison"] = comparison_result
    
    # Salva se richiesto (se impostata l'output_dir)
    if output_dir:
        save_evaluation_report(result, output_dir, importance_df, fpr, tpr, precision_vals, recall_vals, classifier.model_path)
    
    return result


def print_evaluation_report(
    accuracy: float,
    balanced_acc: float,
    roc_auc: float,
    f1: float,
    cm,
    importance_df: pd.DataFrame,
    csv_path: str,
    model_path: str,
    threshold: float,
):
    print("\n" + "=" * 80)
    print("REPORT DI VALUTAZIONE DEL MODELLO")
    print("=" * 80)
    print(f"Dataset: {csv_path}")
    print(f"Modello: {model_path}")
    print(f"Soglia: {threshold}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n" + "-" * 80)
    print("METRICHE GLOBALI")
    print("-" * 80)
    print(f"Accuracy:          {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Balanced Accuracy: {balanced_acc:.4f} ({balanced_acc*100:.2f}%)")
    print(f"F1-Score:          {f1:.4f}")
    print(f"ROC-AUC:           {roc_auc:.4f}")
    
    print("\n" + "-" * 80)
    print("CONFUSION MATRIX")
    print("-" * 80)
    print("                Predicted:Bad  Predicted:Good")
    print(f"Actual:Bad         {cm[0, 0]:5d}           {cm[0, 1]:5d}")
    print(f"Actual:Good        {cm[1, 0]:5d}           {cm[1, 1]:5d}")
    
    # Questa parte la aggiungo separatamente in modo da poter decidere quando mostrarla
    # print("\n" + "-" * 80)
    # print("TOP 10 FEATURES")
    # print("-" * 80)
    # for _, row in importance_df.head(10).iterrows():
    #     importance_pct = (row["importance_mean"] / importance_df["importance_mean"].sum()) * 100
    #     bar_length = int(importance_pct / 2)
    #     bar = "█" * bar_length
    #     print(f"{row['feature']:30s} {bar:40s} {importance_pct:6.2f}%")
    
    # print("\n" + "=" * 80 + "\n")


def save_evaluation_report(
    evaluation_result: dict,
    output_dir: str,
    importance_df: pd.DataFrame,
    fpr,
    tpr,
    precision_vals,
    recall_vals,
    model_path,
):
    """Salva JSON, CSV, HTML"""
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON
    json_path = os.path.join(output_dir, "evaluation_report.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
    logger.info(f"JSON salvato: {json_path}")
    
    # CSV
    importance_csv_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_csv_path, index=False)
    logger.info(f"CSV salvato: {importance_csv_path}")
    
    # HTML
    html_path = os.path.join(output_dir, "evaluation_report.html")
    html_content = _generate_html_report(
        evaluation_result, importance_df, fpr, tpr, precision_vals, recall_vals, model_path
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    logger.info(f"HTML salvato: {html_path}")


def _generate_html_report(
    evaluation_result: dict,
    importance_df: pd.DataFrame,
    fpr,
    tpr,
    precision_vals,
    recall_vals,
    model_path,
) -> str:
    """
    Genera un report HTML usando Jinja2 template.
    
    Carica il template dal file evaluation_report.html e lo renderizza
    con i dati di valutazione passati come parametri.
    """
    # preparo i dati per il template
    cm = evaluation_result["confusion_matrix"]
    # converto le curve in punti e li metto dentro ad un json
    roc_points = json.dumps([{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)])
    pr_points = json.dumps([{"x": float(r), "y": float(p)} for r, p in zip(recall_vals, precision_vals)])
    
    # calcolo la somma delle importance_mean di ogni feature
    importance_total = float(importance_df["importance_mean"].sum()) or 1.0
    # selezione solo le prime dieci feature per maggiore importance
    top_features = importance_df.head(10).copy()
    # calcolo la perecentuale dell'importance
    top_features["importance_pct"] = (top_features["importance_mean"] / importance_total) * 100
    
    comparison_result = evaluation_result.get("model_comparison")
    
    # Configuro Jinja2
    template_dir = os.path.dirname(__file__)
    templates_path = os.path.join(template_dir, "templates")
    env = Environment(loader=FileSystemLoader(templates_path))
    template = env.get_template("evaluation_report.html")
    
    # Renderizza il template
    html_content = template.render(
        evaluation_result=evaluation_result,
        confusion_matrix=cm,
        top_features=top_features,
        roc_points=roc_points,
        pr_points=pr_points,
        model_path=model_path,
        comparison_result=comparison_result,
    )
    
    return html_content