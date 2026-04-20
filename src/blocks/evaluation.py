import os
import json
import logging
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np
from datetime import datetime
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
    X, y, _ = QualityClassifier._load_labeled_dataset(
        csv_path, classifier.feature_names, label_column
    )
    
    # Scaling
    X_scaled = pd.DataFrame(
        classifier.scaler.transform(X),
        columns=classifier.feature_names,
        index=X.index,
    )
    
    # Predizioni
    y_pred_proba = classifier.model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_pred_proba >= classifier.threshold).astype(int)
    
    # Metriche
    metrics = QualityClassifier._compute_binary_metrics(y, y_pred, y_pred_proba)
    
    # Feature importance
    perm = permutation_importance(
        classifier.model,
        X_scaled,
        y,
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )
    importance_df = pd.DataFrame({
        "feature": classifier.feature_names,
        "importance_mean": perm.importances_mean,
        "importance_std": perm.importances_std,
    }).sort_values(by="importance_mean", ascending=False)
    
    # Curve
    fpr, tpr, _ = roc_curve(y, y_pred_proba)
    precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)
    
    # Report
    cm = confusion_matrix(y, y_pred)
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
    
    # Salva se richiesto
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

    cm = evaluation_result["confusion_matrix"]
    roc_points = json.dumps([{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)])
    pr_points = json.dumps([{"x": float(r), "y": float(p)} for r, p in zip(recall_vals, precision_vals)])
    importance_total = float(importance_df["importance_mean"].sum()) or 1.0
    comparison_result = evaluation_result.get("model_comparison")

    html = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Valutazione Modello</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        :root {{
            --bg: #f3f1ec;
            --surface: #fffdf9;
            --border: #d8d1c7;
            --text: #1f1f1c;
            --muted: #666157;
            --accent: #2f5d50;
            --accent-soft: #e5efe9;
            --danger: #8e3b2f;
            --danger-soft: #f4e5e2;
        }}
        body {{
            font-family: Georgia, "Times New Roman", serif;
            background: var(--bg);
            color: var(--text);
            margin: 0;
            padding: 32px 20px;
        }}
        .container {{
            max-width: 980px;
            margin: 0 auto;
            background: var(--surface);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 32px;
            box-shadow: 0 10px 24px rgba(0, 0, 0, 0.04);
        }}
        h1 {{
            margin: 0 0 12px 0;
            font-size: 2rem;
            font-weight: 600;
            letter-spacing: -0.02em;
        }}
        h2 {{
            margin: 32px 0 16px 0;
            font-size: 1.15rem;
            font-weight: 600;
            border-top: 1px solid var(--border);
            padding-top: 24px;
        }}
        p {{
            line-height: 1.6;
        }}
        .intro {{
            color: var(--muted);
            margin: 0 0 24px 0;
        }}
        .meta {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 12px;
            margin: 20px 0 8px 0;
        }}
        .meta-item {{
            padding: 14px 16px;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #faf8f4;
        }}
        .meta-label {{
            display: block;
            font-size: 0.78rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: var(--muted);
            margin-bottom: 4px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin: 20px 0;
        }}
        .metric-card {{
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            background: #fcfbf8;
        }}
        .metric-card h3 {{
            margin: 0 0 8px 0;
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--muted);
        }}
        .metric-card .value {{
            font-size: 1.75rem;
            font-weight: 600;
            color: var(--text);
        }}
        .chart-container {{
            position: relative;
            height: 320px;
            margin: 18px 0;
            padding: 20px;
            background: #faf8f4;
            border: 1px solid var(--border);
            border-radius: 10px;
        }}
        .confusion-matrix {{
            margin: 20px 0;
        }}
        .confusion-matrix table {{
            width: 100%;
            border-collapse: collapse;
        }}
        .confusion-matrix th,
        .confusion-matrix td {{
            border: 1px solid var(--border);
            padding: 14px;
            text-align: center;
        }}
        .confusion-matrix th {{
            background: #f6f2eb;
            font-weight: 600;
        }}
        .confusion-matrix .row-label {{
            text-align: left;
            background: #f6f2eb;
            font-weight: 600;
        }}
        .tn, .tp {{
            background: var(--accent-soft);
            color: var(--accent);
            font-weight: 700;
        }}
        .fp, .fn {{
            background: var(--danger-soft);
            color: var(--danger);
            font-weight: 700;
        }}
        .features-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .features-table th, .features-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        .features-table th {{
            background: #f6f2eb;
            color: var(--text);
            font-weight: 600;
        }}
        .feature-share {{
            font-size: 0.85rem;
            color: var(--muted);
        }}
        .progress-bar {{
            width: 100%;
            height: 10px;
            background: #e7e1d7;
            border-radius: 999px;
            overflow: hidden;
            margin-top: 6px;
        }}
        .progress-fill {{
            height: 100%;
            background: var(--accent);
        }}
        .timestamp {{
            color: var(--muted);
            font-size: 0.9rem;
            margin-top: 28px;
            padding-top: 20px;
            border-top: 1px solid var(--border);
        }}
        @media (max-width: 640px) {{
            body {{
                padding: 16px;
            }}
            .container {{
                padding: 20px;
            }}
            .chart-container {{
                height: 260px;
            }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Report di valutazione del modello</h1>
        <p class="intro">Sintesi delle metriche principali, della matrice di confusione e delle feature piu rilevanti emerse durante la valutazione.</p>

        <div class="meta">
            <div class="meta-item">
                <span class="meta-label">Dataset</span>
                <strong>{evaluation_result['csv_path']}</strong>
            </div>
            <div class="meta-item">
                <span class="meta-label">Modello</span>
                <strong>{model_path}</strong>
            </div>
            <div class="meta-item">
                <span class="meta-label">Soglia</span>
                <strong>{evaluation_result['threshold']}</strong>
            </div>
        </div>

        <h2>Metriche principali</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{evaluation_result['accuracy']:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>Balanced Accuracy</h3>
                <div class="value">{evaluation_result['balanced_accuracy']:.2%}</div>
            </div>
            <div class="metric-card">
                <h3>F1-Score</h3>
                <div class="value">{evaluation_result['f1_score']:.4f}</div>
            </div>
            <div class="metric-card">
                <h3>ROC-AUC</h3>
                <div class="value">{evaluation_result['roc_auc']:.4f}</div>
            </div>
        </div>

        <h2>Confusion Matrix</h2>
        <div class="confusion-matrix">
            <table>
                <tr>
                    <th></th>
                    <th>Predetto bad</th>
                    <th>Predetto good</th>
                </tr>
                <tr>
                    <td class="row-label">Reale bad</td>
                    <td class="tn">{cm[0][0]}</td>
                    <td class="fp">{cm[0][1]}</td>
                </tr>
                <tr>
                    <td class="row-label">Reale good</td>
                    <td class="fn">{cm[1][0]}</td>
                    <td class="tp">{cm[1][1]}</td>
                </tr>
            </table>
        </div>

        <h2>Top 10 feature per importanza</h2>
        <table class="features-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Importanza media</th>
                    <th>Dev. std.</th>
                    <th>Peso relativo</th>
                </tr>
            </thead>
            <tbody>
"""
    for _, row in importance_df.head(10).iterrows():
        importance_pct = (row["importance_mean"] / importance_total) * 100
        html += f"""
            <tr>
                <td><strong>{row['feature']}</strong></td>
                <td>{row['importance_mean']:.6f}</td>
                <td>±{row['importance_std']:.6f}</td>
                <td>
                    <div class="feature-share">{importance_pct:.1f}%</div>
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {importance_pct}%;"></div>
                    </div>
                </td>
            </tr>
"""
        html += """
            </tbody>
        </table>

        <h2>Curve di valutazione</h2>
        <div class="chart-container">
            <canvas id="rocChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="prChart"></canvas>
        </div>
"""
        if comparison_result:
            html += """
        <h2>Confronto modelli con cross validation</h2>
        <p class="intro">Benchmark stratificato con la stessa soglia decisionale applicata a ciascun modello. Le differenze sono espresse rispetto al modello baseline.</p>
        <table class="features-table">
            <thead>
                <tr>
                    <th>Modello</th>
                    <th>ROC-AUC CV</th>
                    <th>F1 CV</th>
                    <th>Balanced Acc CV</th>
                    <th>Delta ROC-AUC</th>
                    <th>Delta F1</th>
                </tr>
            </thead>
            <tbody>
"""
            for row in comparison_result["models"]:
                html += f"""
                <tr>
                    <td><strong>{row['model_name']}</strong></td>
                    <td>{row['roc_auc_mean']:.4f} ± {row['roc_auc_std']:.4f}</td>
                    <td>{row['f1_score_mean']:.4f} ± {row['f1_score_std']:.4f}</td>
                    <td>{row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f}</td>
                    <td>{row['delta_vs_baseline']['roc_auc_mean']:+.4f}</td>
                    <td>{row['delta_vs_baseline']['f1_score_mean']:+.4f}</td>
                </tr>
"""
            html += """
            </tbody>
        </table>
"""

        html += """
    </div>

    <script>
        const rocCtx = document.getElementById('rocChart').getContext('2d');
        new Chart(rocCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'ROC Curve',
                        data: """ + roc_points + """,
                        borderColor: '#2f5d50',
                        backgroundColor: 'rgba(47, 93, 80, 0.08)',
                        fill: false,
                        tension: 0,
                        showLine: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'ROC Curve',
                        color: '#1f1f1c',
                        font: { size: 16, family: 'Georgia, Times New Roman, serif' }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'False Positive Rate', color: '#666157' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(0, 0, 0, 0.06)' }
                    },
                    y: {
                        title: { display: true, text: 'True Positive Rate', color: '#666157' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(0, 0, 0, 0.06)' }
                    }
                }
            }
        });

        const prCtx = document.getElementById('prChart').getContext('2d');
        new Chart(prCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Precision-Recall Curve',
                        data: """ + pr_points + """,
                        borderColor: '#8e3b2f',
                        backgroundColor: 'rgba(142, 59, 47, 0.08)',
                        fill: false,
                        tension: 0,
                        showLine: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    title: {
                        display: true,
                        text: 'Precision-Recall Curve',
                        color: '#1f1f1c',
                        font: { size: 16, family: 'Georgia, Times New Roman, serif' }
                    }
                },
                scales: {
                    x: {
                        title: { display: true, text: 'Recall', color: '#666157' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(0, 0, 0, 0.06)' }
                    },
                    y: {
                        title: { display: true, text: 'Precision', color: '#666157' },
                        min: 0,
                        max: 1,
                        grid: { color: 'rgba(0, 0, 0, 0.06)' }
                    }
                }
            }
        });
    </script>
</body>
</html>
"""
    return html