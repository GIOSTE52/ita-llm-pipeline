from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler



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


def _threshold_metric_scorer(metric_func, threshold: float):
    """
    Crea uno scorer custom per usare la stessa soglia spam anche in cross validation.
    Serve perché di default sklearn usa predict(), quindi soglia 0.5.
    Noi invece vogliamo valutare i modelli con la soglia scelta ossia 0.75.
    """
    
    def scorer(estimator, X, y_true):
        spam_proba = estimator.predict_proba(X)[:, 1]
        y_pred = (spam_proba >= threshold).astype(int)
        return metric_func(y_true, y_pred)

    return scorer

def _build_spam_candidate_models(random_state: int) -> dict:
    """
    Modelli candidati per il confronto CV.

    Nota importante:
    - LightGBM, RandomForest, ExtraTrees e Dummy ricevono direttamente il DataFrame così da mantenere i nomi delle feature.
    - Solo LogisticRegression usa StandardScaler, perché ne ha realmente bisogno.
    """

    return {
        "lightgbm": lgb.LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=random_state,
            class_weight="balanced",
            verbosity=-1,
        ),

        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),

        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        "extra_trees": ExtraTreesClassifier(
            n_estimators=300,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        "dummy": DummyClassifier(
            strategy="prior",
            random_state=random_state,
        ),
    }




def compare_spam_models_cv(
    csv_path: str,
    output_dir: str,
    feature_names: list[str],
    label_column: str | None = None,
    threshold: float = 0.6,
    cv_folds: int = 5,
    random_state: int = 42,
    model_names: list[str] | None = None,
) -> dict:
    """
    Confronta più classificatori sulle stesse feature spam.
    Output:
    - spam_model_comparison_cv.csv
    - spam_model_comparison_report.json
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    label_column = SpamClassifier._resolve_label_column(df, label_column)

    missing_features = [c for c in feature_names if c not in df.columns]
    if missing_features:
        raise ValueError(
            "Nel CSV mancano feature richieste per la cross validation:\n"
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

    df_cv = df.loc[valid_mask].copy()
    y = y.loc[valid_mask].astype(int)

    if y.nunique() < 2:
        raise ValueError(
            "Cross validation impossibile: nel dataset è presente una sola classe."
        )

    X = (
        df_cv[feature_names]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )

    all_models = _build_spam_candidate_models(random_state=random_state)

    if model_names:
        unknown = [m for m in model_names if m not in all_models]
        if unknown:
            raise ValueError(
                "Modelli non supportati: "
                + ", ".join(unknown)
                + ". Valori ammessi: "
                + ", ".join(all_models.keys())
            )

        selected_models = {
            name: all_models[name]
            for name in model_names
        }
    else:
        selected_models = all_models

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=random_state,
    )

    scorers = {
        "roc_auc": "roc_auc",
        "average_precision": "average_precision",
        "f1_spam": _threshold_metric_scorer(
            lambda yt, yp: f1_score(yt, yp, pos_label=1, zero_division=0),
            threshold=threshold,
        ),
        "precision_spam": _threshold_metric_scorer(
            lambda yt, yp: precision_score(yt, yp, pos_label=1, zero_division=0),
            threshold=threshold,
        ),
        "recall_spam": _threshold_metric_scorer(
            lambda yt, yp: recall_score(yt, yp, pos_label=1, zero_division=0),
            threshold=threshold,
        ),
        "balanced_accuracy": _threshold_metric_scorer(
            lambda yt, yp: balanced_accuracy_score(yt, yp),
            threshold=threshold,
        ),
    }

    rows = []

    for model_name, estimator in selected_models.items():
        scores = cross_validate(
            estimator=estimator,
            X=X,
            y=y,
            scoring=scorers,
            cv=cv,
            n_jobs=1,
            error_score=np.nan,
            return_train_score=False,
        )

        row = {
            "model_name": model_name,
            "n_samples": int(len(y)),
            "n_features": int(len(feature_names)),
            "cv_folds": int(cv_folds),
            "threshold": float(threshold),
        }

        for metric_name in scorers.keys():
            values = scores[f"test_{metric_name}"]
            row[f"{metric_name}_mean"] = float(np.nanmean(values))
            row[f"{metric_name}_std"] = float(np.nanstd(values))

        rows.append(row)

    comparison_df = (
        pd.DataFrame(rows)
        .sort_values(
            by=["average_precision_mean", "f1_spam_mean", "roc_auc_mean"],
            ascending=False,
        )
        .reset_index(drop=True)
    )

    comparison_csv = output_path / "spam_model_comparison_cv.csv"
    comparison_df.to_csv(comparison_csv, index=False)

    winner = comparison_df.iloc[0].to_dict()

    result = {
        "csv_path": csv_path,
        "label_column": label_column,
        "threshold": float(threshold),
        "cv_folds": int(cv_folds),
        "random_state": int(random_state),
        "n_samples": int(len(y)),
        "n_features": int(len(feature_names)),
        "models": comparison_df.to_dict(orient="records"),
        "winner_metric": "average_precision_mean",
        "winner": winner,
        "files": {
            "model_comparison_csv": str(comparison_csv),
        },
    }

    report_json = output_path / "spam_model_comparison_report.json"
    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    result["files"]["model_comparison_report_json"] = str(report_json)

    print("\n" + "=" * 100)
    print(f"CONFRONTO MODELLI SPAM - {cv_folds}-fold Stratified CV")
    print("=" * 100)
    print(
        comparison_df[
            [
                "model_name",
                "average_precision_mean",
                "average_precision_std",
                "f1_spam_mean",
                "f1_spam_std",
                "precision_spam_mean",
                "recall_spam_mean",
                "balanced_accuracy_mean",
                "roc_auc_mean",
            ]
        ].to_string(index=False)
    )
    print("=" * 100)
    print(
        "Miglior modello per PR-AUC medio: "
        f"{winner['model_name']} "
        f"(PR-AUC {winner['average_precision_mean']:.4f}, "
        f"F1 spam {winner['f1_spam_mean']:.4f})"
    )
    print(f"[OK] Confronto modelli salvato in: {comparison_csv}")
    print()

    return result


def build_threshold_sweep(
    y_true,
    spam_proba,
    thresholds: list[float],
) -> pd.DataFrame:
    """
    Valuta il comportamento del classificatore spam al variare della soglia senza riaddestrare
    """

    rows = []

    for th in thresholds:
        y_pred = (spam_proba >= th).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true,
            y_pred,
            labels=[0, 1],
        ).ravel()

        rows.append(
            {
                "threshold": float(th),

                "accuracy": accuracy_score(y_true, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

                # =========================
                # Metriche spam
                # =========================
                "precision_spam": precision_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                "recall_spam": recall_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),
                "f1_spam": f1_score(
                    y_true,
                    y_pred,
                    pos_label=1,
                    zero_division=0,
                ),

                # =========================
                # Metriche ham classiche
                # =========================
                "precision_ham": precision_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "recall_ham": recall_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),
                "f1_ham": f1_score(
                    y_true,
                    y_pred,
                    pos_label=0,
                    zero_division=0,
                ),

                # =========================
                # Confusion matrix
                # =========================
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),

                # =========================
                # Conteggi operativi
                # =========================
                "true_ham": int(tn + fp),
                "true_spam": int(tp + fn),

                "predicted_ham": int(tn + fn),
                "predicted_spam": int(fp + tp),

                # =========================
                # Stats operative HAM
                # =========================
                "ham_correct": int(tn),
                "ham_lost_as_spam": int(fp),
                "ham_preservation_rate": float(tn / (tn + fp)) if (tn + fp) else 0.0,
                "ham_loss_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,

                # =========================
                # Contaminazione dello spam nei testi lasciati passare
                # =========================
                "spam_in_predicted_ham": int(fn),
                "spam_contamination_in_predicted_ham": float(fn / (tn + fn)) if (tn + fn) else 0.0,

                # =========================
                # Stats operative SPAM
                # =========================
                "spam_detected": int(tp),
                "spam_missed_as_ham": int(fn),
                "spam_detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
                "spam_miss_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,

                "fp_rate_on_ham": float(fp / (tn + fp)) if (tn + fp) else 0.0,
                "fn_rate_on_spam": float(fn / (tp + fn)) if (tp + fn) else 0.0,
            }
        )

    return pd.DataFrame(rows)

def evaluate_spam_model(
    model_path: str,
    test_csv: str,
    output_dir: str,
    label_column: str | None = None,
    threshold: float | None = None,
    save_feature_importance: bool = True,
    compare_models: bool = False,
    comparison_csv: str | None = None,
    cv_folds: int = 5,
    cv_random_state: int = 42,
    cv_model_names: list[str] | None = None,
    importance_review_epsilon: float = 0.001,
    threshold_sweep: list[float] | None = None,
) -> dict:


    
    """
    Valuta il modello spam su un CSV di feature già estratte.
    La funzione carica il modello, applica la soglia di classificazione, calcola le metriche principali, salva predizioni, falsi positivi,
    falsi negativi, feature importance e, opzionalmente, il confronto tra modelli e lo sweep delle soglie.

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

    tn, fp, fn, tp = cm.ravel()

    metrics = {
       "accuracy": accuracy_score(y, y_pred),
       "balanced_accuracy": balanced_accuracy_score(y, y_pred),

       # =========================
       # Metriche spam
       # =========================
       "precision_spam": precision_score(y, y_pred, pos_label=1, zero_division=0),
       "recall_spam": recall_score(y, y_pred, pos_label=1, zero_division=0),
       "f1_spam": f1_score(y, y_pred, pos_label=1, zero_division=0),

       # =========================
       # Metriche ham
       # =========================
       "precision_ham": precision_score(y, y_pred, pos_label=0, zero_division=0),
       "recall_ham": recall_score(y, y_pred, pos_label=0, zero_division=0),
       "f1_ham": f1_score(y, y_pred, pos_label=0, zero_division=0),

       # =========================
       # Curve globali
       # =========================
       "roc_auc": roc_auc_score(y, spam_proba),
       "pr_auc": average_precision_score(y, spam_proba),

       # =========================
       # Conteggi matrice
       # =========================
       "true_negative": int(tn),
       "false_positive": int(fp),
       "false_negative": int(fn),
       "true_positive": int(tp),

       # =========================
       # Stats operative ham
       # =========================
       "true_ham": int(tn + fp),
       "ham_correct": int(tn),
       "ham_lost_as_spam": int(fp),
       "ham_preservation_rate": float(tn / (tn + fp)) if (tn + fp) else 0.0,
       "ham_loss_rate": float(fp / (tn + fp)) if (tn + fp) else 0.0,

       # =========================
       # Contaminazione nei testi lasciati come ham
       # =========================
       "predicted_ham": int(tn + fn),
       "spam_in_predicted_ham": int(fn),
       "spam_contamination_in_predicted_ham": float(fn / (tn + fn)) if (tn + fn) else 0.0,

       # =========================
       # Stats operative spam
       # =========================
       "true_spam": int(tp + fn),
       "spam_detected": int(tp),
       "spam_missed_as_ham": int(fn),
       "spam_detection_rate": float(tp / (tp + fn)) if (tp + fn) else 0.0,
       "spam_miss_rate": float(fn / (tp + fn)) if (tp + fn) else 0.0,
    }


    threshold_sweep_rows = []
    threshold_sweep_csv = None

    if threshold_sweep:
        sweep_df = build_threshold_sweep(
            y_true=y,
            spam_proba=spam_proba,
            thresholds=threshold_sweep,
        )

        threshold_sweep_csv = output_path / "spam_threshold_sweep.csv"
        sweep_df.to_csv(threshold_sweep_csv, index=False)

        threshold_sweep_rows = sweep_df.to_dict(orient="records")

        print("\n" + "=" * 100)
        print("THRESHOLD SWEEP SPAM")
        print("=" * 100)
        print(
            sweep_df[
                [
                    "threshold",

                    "precision_spam",
                    "recall_spam",
                    "f1_spam",

                    "precision_ham",
                    "recall_ham",
                    "f1_ham",

                    "false_positive",
                    "false_negative",

                    "ham_lost_as_spam",
                    "ham_loss_rate",
                    "spam_contamination_in_predicted_ham",

                    "balanced_accuracy",
                ]
            ].to_string(index=False)
        )

        print("=" * 100)
        print(f"[OK] Threshold sweep salvato in: {threshold_sweep_csv}")


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



    feature_importance_rows = []
    feature_importance_files = {}

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

        feature_importance_csv = output_path / "spam_feature_importance.csv"
        feature_importance.to_csv(feature_importance_csv, index=False)

        strong_features = feature_importance[
            (feature_importance["importance_mean"] > importance_review_epsilon)
            & (
                feature_importance["importance_std"]
                <= feature_importance["importance_mean"].abs()
            )
        ].copy()

        features_to_review = feature_importance[
            (
                feature_importance["importance_mean"].abs()
                <= importance_review_epsilon
            )
            | (
                feature_importance["importance_std"]
                > feature_importance["importance_mean"].abs()
            )
        ].copy()

        negative_features = feature_importance[
            feature_importance["importance_mean"] < 0
        ].copy()

        strong_features_csv = output_path / "spam_features_strong.csv"
        features_to_review_csv = output_path / "spam_features_to_review.csv"
        negative_features_csv = output_path / "spam_features_negative_importance.csv"

        strong_features.to_csv(strong_features_csv, index=False)
        features_to_review.to_csv(features_to_review_csv, index=False)
        negative_features.to_csv(negative_features_csv, index=False)

        feature_importance_rows = feature_importance.head(30).to_dict(orient="records")

        feature_importance_files = {
            "feature_importance_csv": str(feature_importance_csv),
            "strong_features_csv": str(strong_features_csv),
            "features_to_review_csv": str(features_to_review_csv),
            "negative_features_csv": str(negative_features_csv),
        }

    comparison_result = None

    if compare_models:
        comparison_dataset = comparison_csv or test_csv

        comparison_result = compare_spam_models_cv(
            csv_path=comparison_dataset,
            output_dir=str(output_path),
            feature_names=feature_names,
            label_column=label_column,
            threshold=classifier.threshold,
            cv_folds=cv_folds,
            random_state=cv_random_state,
            model_names=cv_model_names,
        )




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
            **feature_importance_files,
            **(
                {"threshold_sweep_csv": str(threshold_sweep_csv)}
                if threshold_sweep_csv is not None
                else {}
            ),
        },

        "top_features": feature_importance_rows,
        "threshold_sweep": threshold_sweep_rows,
        "model_comparison": comparison_result,
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
    print("HAM STATS")
    print(f"Precision ham:                      {metrics['precision_ham']:.4f}")
    print(f"Recall ham / preservation rate:      {metrics['recall_ham']:.4f}")
    print(f"F1 ham:                             {metrics['f1_ham']:.4f}")
    print(f"Ham corretti:                       {metrics['ham_correct']}")
    print(f"Ham persi come spam:                {metrics['ham_lost_as_spam']}")
    print(f"Ham loss rate:                      {metrics['ham_loss_rate']:.4f}")
    print(f"Spam rimasto nei predicted ham:      {metrics['spam_in_predicted_ham']}")
    print(f"Spam contamination predicted ham:    {metrics['spam_contamination_in_predicted_ham']:.4f}")

    print()
    print(f"[OK] Report salvato in: {output_path}")

    return result


