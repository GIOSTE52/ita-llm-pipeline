# src/blocks/spam_classifier/spam_evaluation.py

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import joblib
from sklearn.metrics import f1_score
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)

from sklearn.inspection import permutation_importance

from .spam_classifier import SpamClassifier, LABEL_MAP, INV_LABEL_MAP


def evaluate_spam_model(
    model_path: str,
    test_csv: str,
    output_dir: str,
    label_column: str | None = None,
    threshold: float | None = None,
    save_feature_importance: bool = True,
) -> dict:
    """
    Valuta il modello spam su un CSV già contenente le feature spam.

    Label attese:
        ham  -> 0
        spam -> 1

    Output principali:
        - spam_evaluation_report.json
        - spam_predictions.csv
        - spam_false_positives.csv
        - spam_false_negatives.csv
        - spam_feature_importance.csv
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    classifier = SpamClassifier(
        model_path=model_path,
        threshold=threshold,
    )

    df = pd.read_csv(test_csv)

    if label_column is None:
        label_column = classifier._resolve_label_column(df, None)

    if label_column not in df.columns:
        raise ValueError(f"Colonna label non trovata nel CSV: {label_column}")

    feature_names = list(classifier.feature_names)
    missing_features = [c for c in feature_names if c not in df.columns]

    if missing_features:
        raise ValueError(
            "Nel CSV mancano feature usate dal modello:\n"
            + "\n".join(missing_features)
        )

    y_raw = (
        df[label_column]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    y = y_raw.map(LABEL_MAP)
    valid_mask = y.notna()

    if not valid_mask.all():
        invalid_count = int((~valid_mask).sum())
        print(f"[WARN] Rimosse {invalid_count} righe con label non valida.")

    df_eval = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    X = (
        df_eval[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    X_scaled = pd.DataFrame(
        classifier.scaler.transform(X),
        columns=feature_names,
        index=X.index,
    )



    spam_proba = classifier.model.predict_proba(X_scaled)[:, 1]
    y_pred = (spam_proba >= classifier.threshold).astype(int)

    report_dict = classification_report(
        y,
        y_pred,
        target_names=["ham", "spam"],
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y, y_pred, labels=[0, 1])

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision_spam": precision_score(y, y_pred, pos_label=1, zero_division=0),
        "recall_spam": recall_score(y, y_pred, pos_label=1, zero_division=0),
        "f1_spam": f1_score(y, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y, spam_proba),
        "average_precision": average_precision_score(y, spam_proba),
    }

    pred_df = df_eval.copy()
    pred_df["true_label"] = [INV_LABEL_MAP[int(v)] for v in y]
    pred_df["pred_label"] = [INV_LABEL_MAP[int(v)] for v in y_pred]
    pred_df["spam_probability"] = spam_proba
    pred_df["threshold"] = classifier.threshold
    pred_df["is_error"] = pred_df["true_label"] != pred_df["pred_label"]

    pred_df["error_type"] = ""
    pred_df.loc[
        (pred_df["true_label"] == "ham") & (pred_df["pred_label"] == "spam"),
        "error_type",
    ] = "false_positive"

    pred_df.loc[
        (pred_df["true_label"] == "spam") & (pred_df["pred_label"] == "ham"),
        "error_type",
    ] = "false_negative"

    predictions_csv = output_path / "spam_predictions.csv"
    false_positives_csv = output_path / "spam_false_positives.csv"
    false_negatives_csv = output_path / "spam_false_negatives.csv"

    pred_df.to_csv(predictions_csv, index=False)
    pred_df[pred_df["error_type"] == "false_positive"].to_csv(false_positives_csv, index=False)
    pred_df[pred_df["error_type"] == "false_negative"].to_csv(false_negatives_csv, index=False)

    feature_importance_rows = []

    def spam_f1_scorer(estimator, X_perm, y_true):
        """
        Scorer custom per permutation_importance.

        Serve perché permutation_importance può passare array NumPy senza nomi colonna.
        Qui riconvertiamo sempre X in DataFrame con le feature corrette prima di chiamare LightGBM.
        """

        if isinstance(X_perm, pd.DataFrame):
            X_perm_df = X_perm[feature_names]
        else:
            X_perm_df = pd.DataFrame(
                X_perm,
                columns=feature_names,
            )

        spam_proba_perm = estimator.predict_proba(X_perm_df)[:, 1]
        y_perm_pred = (spam_proba_perm >= classifier.threshold).astype(int)

        return f1_score(
            y_true,
            y_perm_pred,
            pos_label=1,
            zero_division=0,
        )


    if save_feature_importance:
        perm = permutation_importance(
        classifier.model,
        X_scaled,
        y,
        n_repeats=5,
        random_state=42,
        scoring=spam_f1_scorer,
        n_jobs=1,
    )




        feature_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance_mean": perm.importances_mean,
                    "importance_std": perm.importances_std,
                }
            )
            .sort_values("importance_mean", ascending=False)
            .reset_index(drop=True)
        )

        feature_importance.to_csv(output_path / "spam_feature_importance.csv", index=False)
        feature_importance_rows = feature_importance.head(30).to_dict(orient="records")

    result = {
        "model_path": model_path,
        "test_csv": test_csv,
        "label_column": label_column,
        "threshold": classifier.threshold,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n_samples": int(len(y)),
        "metrics": metrics,
        "classification_report": report_dict,
        "confusion_matrix": {
            "labels": ["ham", "spam"],
            "matrix": cm.tolist(),
            "meaning": {
                "tn_ham_pred_ham": int(cm[0][0]),
                "fp_ham_pred_spam": int(cm[0][1]),
                "fn_spam_pred_ham": int(cm[1][0]),
                "tp_spam_pred_spam": int(cm[1][1]),
            },
        },
        "files": {
            "predictions_csv": str(predictions_csv),
            "false_positives_csv": str(false_positives_csv),
            "false_negatives_csv": str(false_negatives_csv),
            "feature_importance_csv": str(output_path / "spam_feature_importance.csv"),
        },
        "top_features": feature_importance_rows,
    }

    with open(output_path / "spam_evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("SPAM EVALUATION REPORT")
    print("=" * 60)
    print(f"Campioni valutati: {len(y)}")
    print(f"Soglia spam: {classifier.threshold}")
    print()
    print(classification_report(y, y_pred, target_names=["ham", "spam"], zero_division=0))
    print("Confusion Matrix [ham, spam]:")
    print(cm)
    print()
    print(f"Accuracy:          {metrics['accuracy']:.4f}")
    print(f"Balanced accuracy: {metrics['balanced_accuracy']:.4f}")
    print(f"Precision spam:    {metrics['precision_spam']:.4f}")
    print(f"Recall spam:       {metrics['recall_spam']:.4f}")
    print(f"F1 spam:           {metrics['f1_spam']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print(f"PR-AUC:            {metrics['average_precision']:.4f}")
    print()
    print(f"[OK] Report salvato in: {output_path}")

    return result


