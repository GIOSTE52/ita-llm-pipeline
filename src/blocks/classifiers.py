"""
Classificatore ML binario per la qualità dei documenti di testo.

Prende in input un vettore di feature (estratte da DocStatsCsv)
e classifica ogni documento come "good" oppure "bad".
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)
from sklearn.inspection import permutation_importance

import lightgbm as lgb
from datetime import datetime
import json

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

logger = logging.getLogger(__name__)

# Feature che il classificatore si aspetta di trovare in doc.metadata
# (devono coincidere con quelle prodotte da ItalianFeatureExtractor)
DEFAULT_FEATURE_NAMES: List[str] = [
"language_score",
"length",
"white_space_ratio",
"non_alpha_digit_ratio",
"digit_ratio",
"uppercase_ratio",
"elipsis_ratio",
"punctuation_ratio",
"word_count",
"sentence_count",
"vocabulary_size",
"lowercase_ratio",
"vowel_ratio",
"consonant_ratio",
"avg_word_length",
"avg_sentence_length", # molto importante per capire la scorrevolezza di un testo
"quote_ratio",
"parenthesis_ratio",
"comma_ratio",
"period_ratio",
"question_mark_ratio",
"exclamation_ratio",
"colon_ratio",
"semicolon_ratio",
"stopword_ratio", # può aiutare ma può anche creare bias (importante per capire la scorrevolezza di un testo)
"line_count",
"paragraph_count",
"avg_line_length",
"avg_paragraph_length",
"empty_line_ratio",
"bullet_point_count",
"bullet_point_ratio",
"url_count",
"url_density",
"email_count",
"email_density",
"html_tag_count",
"html_tag_ratio",
"special_char_ratio",
"most_common_word_freq",    # molto importante per capire la scorrevolezza di un testo
"repeated_word_count",
"repeated_word_ratio",  # molto importante per capire la scorrevolezza di un testo
"repeated_char_count",
"repeated_char_ratio",
"repeated_sequence_count",
"text_entropy", # molto importante per calcolare la qualità del testo
"unique_word_count",
"unique_word_ratio",    # molto importante per capire la scorrevolezza di un testo
"all_caps_word_ratio",
"all_lowercase_word_ratio",
"mixed_case_word_ratio",
"consecutive_spaces_count",
"consecutive_punctuation_count"
]
# length,white_space_ratio,non_alpha_digit_ratio,digit_ratio,uppercase_ratio,elipsis_ratio,punctuation_ratio,word_count,sentence_count,vocabulary_size,lowercase_ratio,vowel_ratio,consonant_ratio,avg_word_length,avg_sentence_length,quote_ratio,parenthesis_ratio,comma_ratio,period_ratio,question_mark_ratio,exclamation_ratio,colon_ratio,semicolon_ratio,stopword_ratio,line_count,paragraph_count,avg_line_length,avg_paragraph_length,empty_line_ratio,bullet_point_count,bullet_point_ratio,url_count,url_density,email_count,email_density,html_tag_count,html_tag_ratio,special_char_ratio,most_common_word_freq,repeated_word_count,repeated_word_ratio,repeated_char_count,repeated_char_ratio,repeated_sequence_count,text_entropy,unique_word_count,unique_word_ratio,all_caps_word_ratio,all_lowercase_word_ratio,mixed_case_word_ratio,consecutive_spaces_count,consecutive_punctuation_count

LABEL_MAP = {"bad": 0, "good": 1}


class QualityClassifier(PipelineStep):
    """
    Classificatore binario (good / bad) basato su LightGBM.

    Può essere usato come step della pipeline datatrove definita in main.py: legge le feature dai
    metadata del documento, esegue la predizione e scrive il risultato
    in ``doc.metadata["quality_label"]`` e ``doc.metadata["quality_score"]``.
    Parametri
    ---------
    model_path : str
        Percorso al modello serializzato (.joblib) da caricare per l'inferenza.
    feature_names : list[str] | None
        Nomi delle feature da leggere da doc.metadata.
        Se ``None`` usa ``DEFAULT_FEATURE_NAMES``.
    threshold : float
        Soglia di probabilità per la classe "good" (default 0.5).
        Documenti con probabilità >= threshold → "good", altrimenti → "bad".
    """

    name = "Quality Classifier"

    def __init__(
            
        self,
        model_path: str,
        feature_names: Optional[List[str]] = None,
        threshold: float = 0.5,
    ):
        super().__init__()
        self.model_path = model_path
        self.threshold = threshold

        # Carica modello e scaler dal file .joblib
        artifact = joblib.load(self.model_path)
        self.model: lgb.LGBMClassifier = artifact["model"]
        self.scaler: StandardScaler = artifact["scaler"]
        self._feature_names_train: List[str] = artifact["feature_names"]
        self.model_name: str = artifact.get(
            "model_name",
            self.model.__class__.__name__,
        )
        self.feature_names = feature_names or self._feature_names_train or DEFAULT_FEATURE_NAMES

        logger.info("Modello caricato da %s", self.model_path)

    # -----------------------------------------------------------------
    # Pipeline step: inferenza documento per documento
    # -----------------------------------------------------------------
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """Classifica ogni documento e produce il risultato nei metadata."""
        for doc in data:
            features = self._extract_features(doc)
            if features is None:
                # Se mancano feature, segna come "bad" e continua
                doc.metadata["quality_label"] = "bad"
                doc.metadata["quality_score"] = 0.0
                logger.warning(
                    "Feature mancanti per doc %s – classificato come bad", doc.id
                )
                yield doc
                continue

            # X = np.array(features).reshape(1, -1)
            X = pd.DataFrame([features], columns = self.feature_names) # riga da rimuovere se si vuole usare np.array
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns = self.feature_names) # riga d arimuovere se si vuole usare np.array

            proba = self.model.predict_proba(X_scaled)[0]  # [P(bad), P(good)]
            score_good = float(proba[1])
            label = "good" if score_good >= self.threshold else "bad"

            doc.metadata["quality_label"] = label
            doc.metadata["quality_score"] = round(score_good, 4)

            yield doc

    def _extract_features(self, doc) -> Optional[List[float]]:
        """Estrae il vettore di feature dal metadata del documento."""
        try:
            return [float(doc.metadata[f]) for f in self.feature_names]
        except (KeyError, TypeError) as e:
            logger.debug("Impossibile estrarre le feature: %s", e)
            return None

    # =================================================================
    # METODI STATICI PER IL TRAINING
    # =================================================================
    @staticmethod
    def _load_labeled_dataset(
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        label_column: str = "label",
    ) -> tuple[pd.DataFrame, pd.Series, List[str]]:
        """Carica un CSV etichettato e valida feature e label."""
        feat_names = feature_names or DEFAULT_FEATURE_NAMES
        df = pd.read_csv(csv_path)

        missing_cols = set(feat_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel CSV: {missing_cols}")
        if label_column not in df.columns:
            raise ValueError(
                f"Colonna label '{label_column}' non trovata nel CSV"
            )

        X = df[feat_names].copy()
        y = df[label_column].map(LABEL_MAP)

        if y.isna().any():
            invalid = df[label_column][y.isna()].unique().tolist()
            raise ValueError(
                f"Valori label non validi: {invalid}. Ammessi: 'good', 'bad'."
            )

        return X, y.astype(int), feat_names

    @staticmethod
    def _compute_binary_metrics(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
    ) -> Dict[str, float]:
        """Calcola un set coerente di metriche binarie."""
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_pred_proba)),
        }

    @staticmethod
    def _summarize_fold_metrics(
        fold_metrics: List[Dict[str, float]]
    ) -> Dict[str, float]:
        """Restituisce media e deviazione standard per ciascuna metrica di CV."""
        summary: Dict[str, float] = {}
        if not fold_metrics:
            return summary

        for metric_name in fold_metrics[0].keys():
            values = np.array(
                [fold[metric_name] for fold in fold_metrics],
                dtype=float,
            )
            summary[f"{metric_name}_mean"] = float(values.mean())
            summary[f"{metric_name}_std"] = float(values.std(ddof=0))
        return summary

    @staticmethod
    def _build_candidate_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
        """Restituisce i modelli candidati per il benchmark."""
        return {
            "lightgbm": {
                "display_name": "LightGBM",
                "estimator": lgb.LGBMClassifier(
                    objective="binary",
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=-1,
                    random_state=random_state,
                    verbose=-1,
                ),
            },
            "random_forest": {
                "display_name": "Random Forest",
                "estimator": RandomForestClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            },
            "extra_trees": {
                "display_name": "Extra Trees",
                "estimator": ExtraTreesClassifier(
                    n_estimators=400,
                    min_samples_leaf=2,
                    random_state=random_state,
                    n_jobs=-1,
                ),
            },
            "logistic_regression": {
                "display_name": "Logistic Regression",
                "estimator": Pipeline(
                    steps=[
                        ("scaler", StandardScaler()),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=random_state,
                            ),
                        ),
                    ]
                ),
            },
        }

    @staticmethod
    def cross_validate_models(
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        label_column: str = "label",
        threshold: float = 0.65,
        cv_folds: int = 5,
        random_state: int = 42,
        model_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Confronta piu modelli con cross validation stratificata sullo stesso dataset.
        """
        if cv_folds < 2:
            raise ValueError("cv_folds deve essere almeno 2")

        X, y, feat_names = QualityClassifier._load_labeled_dataset(
            csv_path=csv_path,
            feature_names=feature_names,
            label_column=label_column,
        )
        candidate_models = QualityClassifier._build_candidate_models(
            random_state=random_state
        )

        if model_names:
            invalid_model_names = sorted(set(model_names) - set(candidate_models))
            if invalid_model_names:
                raise ValueError(
                    "Modelli non supportati: "
                    f"{invalid_model_names}. Valori ammessi: {sorted(candidate_models)}"
                )
            selected_model_names = model_names
        else:
            selected_model_names = list(candidate_models.keys())

        cv = StratifiedKFold(
            n_splits=cv_folds,
            shuffle=True,
            random_state=random_state,
        )
        comparison_rows: List[Dict[str, Any]] = []

        for model_key in selected_model_names:
            spec = candidate_models[model_key]
            fold_metrics: List[Dict[str, float]] = []
            oof_proba = np.zeros(len(y), dtype=float)
            oof_pred = np.zeros(len(y), dtype=int)

            for train_idx, val_idx in cv.split(X, y):
                estimator = clone(spec["estimator"])
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                estimator.fit(X_train, y_train)
                val_pred_proba = estimator.predict_proba(X_val)[:, 1]
                val_pred = (val_pred_proba >= threshold).astype(int)

                fold_metrics.append(
                    QualityClassifier._compute_binary_metrics(
                        y_true=y_val,
                        y_pred=val_pred,
                        y_pred_proba=val_pred_proba,
                    )
                )
                oof_proba[val_idx] = val_pred_proba
                oof_pred[val_idx] = val_pred

            summary = QualityClassifier._summarize_fold_metrics(fold_metrics)
            overall_metrics = QualityClassifier._compute_binary_metrics(
                y_true=y,
                y_pred=oof_pred,
                y_pred_proba=oof_proba,
            )
            confusion = confusion_matrix(y, oof_pred)
            report = classification_report(
                y,
                oof_pred,
                target_names=["bad", "good"],
                output_dict=True,
                zero_division=0,
            )

            comparison_rows.append(
                {
                    "model_key": model_key,
                    "model_name": spec["display_name"],
                    "threshold": threshold,
                    "cv_folds": cv_folds,
                    "fold_metrics": fold_metrics,
                    "overall_metrics": {
                        metric: round(value, 4)
                        for metric, value in overall_metrics.items()
                    },
                    "classification_report": report,
                    "confusion_matrix": confusion.tolist(),
                    **{
                        metric_name: round(metric_value, 4)
                        for metric_name, metric_value in summary.items()
                    },
                }
            )

        comparison_rows.sort(
            key=lambda row: (row["roc_auc_mean"], row["f1_score_mean"]),
            reverse=True,
        )

        baseline_key = (
            "lightgbm"
            if "lightgbm" in selected_model_names
            else comparison_rows[0]["model_key"]
        )
        baseline_row = next(
            row for row in comparison_rows if row["model_key"] == baseline_key
        )
        metric_fields = [
            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_score_mean",
            "roc_auc_mean",
        ]
        for row in comparison_rows:
            row["delta_vs_baseline"] = {
                field: round(row[field] - baseline_row[field], 4)
                for field in metric_fields
            }

        return {
            "dataset_path": csv_path,
            "feature_names": feat_names,
            "threshold": threshold,
            "cv_folds": cv_folds,
            "baseline_model_key": baseline_key,
            "baseline_model_name": baseline_row["model_name"],
            "ranking_metric": "roc_auc_mean",
            "models": comparison_rows,
            "winner": {
                "model_key": comparison_rows[0]["model_key"],
                "model_name": comparison_rows[0]["model_name"],
                "roc_auc_mean": comparison_rows[0]["roc_auc_mean"],
                "f1_score_mean": comparison_rows[0]["f1_score_mean"],
            },
        }

    @staticmethod
    def train_from_csv(
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        label_column: str = "label",
        test_size: float = 0.2,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
    ) -> dict:
        """
        Addestra un modello LightGBM binario a partire da un CSV.

        Il CSV deve contenere:
        - le colonne corrispondenti alle feature (stessi nomi di DEFAULT_FEATURE_NAMES
          oppure quelli specificati in ``feature_names``)
        - una colonna ``label`` con valori "good" / "bad"

        Restituisce un dizionario con modello, scaler, metriche e
        importanza delle feature.

        Esempio
        -------
        result = QualityClassifier.train_from_csv("data/quality_dataset.csv")
        QualityClassifier.save_model(result, "models/quality_model.joblib")
        """
        # ------- 1. Caricamento dati -------
        X, y, feat_names = QualityClassifier._load_labeled_dataset(
            csv_path=csv_path,
            feature_names=feature_names,
            label_column=label_column,
        )
        
        # ------- 1b. Correlation Matrix -------
        # Calcola la matrice di correlazione tra tutte le feature + label
        corr_df = X.copy()
        corr_df["label"] = y
        correlation_matrix = corr_df.corr()

        print("=" * 60)
        print("CORRELATION MATRIX")
        print("=" * 60)
        # Mostra le correlazioni di ogni feature con la label, ordinate per valore assoluto
        label_corr = correlation_matrix["label"].drop("label").sort_values(
            key=abs, ascending=False
        )
        print("\nCorrelazione con la label (ordinate per |valore|):")
        print(label_corr.to_string())

        # Identifica coppie di feature altamente correlate tra loro (|r| > 0.9)
        high_corr_threshold = 0.9
        feature_corr = correlation_matrix.drop(columns=["label"], index=["label"])
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i + 1, len(feature_corr.columns)):
                r = feature_corr.iloc[i, j]
                if abs(r) > high_corr_threshold:
                    high_corr_pairs.append(
                        (feature_corr.columns[i], feature_corr.columns[j], round(r, 4))
                    )

        if high_corr_pairs:
            print(f"\nCoppie di feature con |correlazione| > {high_corr_threshold}:")
            for f1, f2, r in high_corr_pairs:
                print(f"  {f1}  ↔  {f2}  :  {r}")
        else:
            print(f"\nNessuna coppia di feature con |correlazione| > {high_corr_threshold}")

        # ------- 2. Train / Validation split -------
        X_train, X_val, y_train, y_val = train_test_split(
            X,
            y,
            test_size=test_size,    # default 0.2 -> 20% validazione
            stratify=y,     # mantiene la porzione good/bad
            random_state=random_state,
        )

        # ------- 3. Scaling -------
        # Fitto lo scaler solo sul training set per evitare data leakage.
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feat_names,
            index=X_train.index,
        )
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=feat_names,
            index=X_val.index,
        )

        # ------- 4. Training -------
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=-1,
            random_state=random_state,
            verbose=-1,
        )
        # Chiamata che effettua l'apprendimento
        model.fit(X_train_scaled, y_train)

        # ------- 5. Valutazione -------
        y_pred = model.predict(X_val_scaled)

        # Genera un dizionario 
        report = classification_report(
            y_val, y_pred, target_names=["bad", "good"], output_dict=True
        )
        # Genera una string stampabile a schermo
        report_str = classification_report(
            y_val, y_pred, target_names=["bad", "good"]
        )
        cm = confusion_matrix(y_val, y_pred)

        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(report_str)
        print("\nConfusion Matrix:")
        print(cm)

        # ------- 6. Feature Importance (permutation) -------
        print("\nCalcolo permutation feature importance...")
        perm = permutation_importance(
            model,
            X_val_scaled,
            y_val,
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )

        importance_df = pd.DataFrame(
            {
                "feature": feat_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values(by="importance_mean", ascending=False)

        print("\nFeature Importance:")
        print(importance_df.to_string(index=False))

        return {
            "model": model,
            "scaler": scaler,
            "model_name": "LightGBM",
            "feature_names": feat_names,
            "classification_report": report,
            "confusion_matrix": cm,
            "correlation_matrix": correlation_matrix,
            "feature_importance": importance_df,
        }

    @staticmethod
    def save_model(training_result: dict, output_path: str) -> str:
        """
        Salva il modello addestrato su disco (formato joblib).

        Parametri
        ---------
        training_result : dict
            Dizionario restituito da ``train_from_csv()``.
        output_path : str
            Percorso del file .joblib di output.

        Ritorna il percorso assoluto del file salvato.
        """
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        artifact = {
            "model": training_result["model"],
            "scaler": training_result["scaler"],
            "feature_names": training_result["feature_names"],
            "model_name": training_result.get(
                "model_name",
                training_result["model"].__class__.__name__,
            ),
        }
        joblib.dump(artifact, output_path)

        logger.info("Modello salvato in %s", output_path)
        print(f"\nModello salvato in: {output_path}")
        return os.path.abspath(output_path)
    
    def evaluate(
            self,
            csv_path: str,
            label_column: str = "label",
            output_dir: Optional[str] = None,
            comparison_result: Optional[Dict[str, Any]] = None,
    ) -> dict:
        """
        Valuta il modello su un dataset di test contenuto in un CSV.
        Il CSV che contiene il dataset di test viene generato eseguendo la pipeline con lettura in "train/*.jsonl"
        oppure utilizzando il file CSV salvato in notebooks

        Calcola tutte le metriche importanti: accuracy, precision, recall, F1,
        ROC-AUC, confusion matrix, e feature importance basata su permutazioni.
        Stampa i risultati a schermo e, opzionalmente, li salva su file JSON e HTML.

        Parametri
        ---------
        csv_path : str
            Percorso al file CSV con le feature e la label.
            Il CSV deve contenere colonne per tutte le feature (come in DEFAULT_FEATURE_NAMES)
            e una colonna per la label con valori "good" / "bad".
        label_column : str
            Nome della colonna contenente le label (default "label").
        output_dir : str | None
            Directory dove salvare il report in formato JSON e HTML.
            Se None, il report viene solo stampato a schermo.
        comparison_result : dict | None
            Risultato opzionale del benchmark multi-modello da includere nel report.

        -------
        Ritorna un dizionario contenente:
            - "accuracy": Accuratezza complessiva
            - "balanced_accuracy": Accuratezza bilanciata
            - "roc_auc": Punteggio ROC-AUC
            - "classification_report": Report dettagliato (dict)
            - "confusion_matrix": Matrice di confusione
            - "threshold": Soglia usata
            - "feature_names": Nomi delle feature impiegate
            - "timestamp": Data/ora della valutazione
        """
        X, y, _ = self._load_labeled_dataset(
            csv_path=csv_path,
            feature_names=self.feature_names,
            label_column=label_column,
        )

        # Scalo le feature usando lo scaler del modello
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Predizioni
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]  # Probabilità della classe "good"
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Calcolo metriche
        metrics = self._compute_binary_metrics(
            y_true=y,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )
        accuracy = metrics["accuracy"]
        balanced_acc = metrics["balanced_accuracy"]
        roc_auc = metrics["roc_auc"]
        f1 = metrics["f1_score"]
        cm = confusion_matrix(y, y_pred)
        report_dict = classification_report(
            y, y_pred, target_names=["bad", "good"], output_dict=True, zero_division=0
        )
        report_str = classification_report(
            y, y_pred, target_names=["bad", "good"], zero_division=0
        )

        # Feature importance (permutation)
        logger.info("Calcolo permutation feature importance...")
        perm = permutation_importance(
            self.model,
            X_scaled,
            y,
            n_repeats=10,
            random_state=42,
            n_jobs=-1,
        )
        importance_df = pd.DataFrame(
            {
                "feature": self.feature_names,
                "importance_mean": perm.importances_mean,
                "importance_std": perm.importances_std,
            }
        ).sort_values(by="importance_mean", ascending=False)

        # Calcolo curve ROC e Precision-Recall
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        precision_vals, recall_vals, _ = precision_recall_curve(y, y_pred_proba)

        # Stampo report formattato
        self._print_evaluation_report(
            accuracy=accuracy,
            balanced_acc=balanced_acc,
            roc_auc=roc_auc,
            f1=f1,
            report_str=report_str,
            cm=cm,
            importance_df=importance_df,
            csv_path=csv_path,
        )

        # Preparo il risultato
        result = {
            "accuracy": round(accuracy, 4),
            "balanced_accuracy": round(balanced_acc, 4),
            "roc_auc": round(roc_auc, 4),
            "f1_score": round(f1, 4),
            "classification_report": report_dict,
            "confusion_matrix": cm.tolist(),
            "threshold": self.threshold,
            "feature_names": self.feature_names,
            "top_features": importance_df.head(10).to_dict(orient="records"),
            "csv_path": csv_path,
            "timestamp": datetime.now().isoformat(),
            "model_name": self.model_name,
        }
        if comparison_result is not None:
            result["model_comparison"] = comparison_result

        # Salvo i report se specificato output_dir
        if output_dir:
            self.save_evaluation_report(result, output_dir, importance_df, fpr, tpr, precision_vals, recall_vals)

        return result

    def _print_evaluation_report(
            self,
            accuracy: float,
            balanced_acc: float,
            roc_auc: float,
            f1: float,
            report_str: str,
            cm,
            importance_df: pd.DataFrame,
            csv_path: str,
    ):
        """Stampa il report di valutazione in formato leggibile."""
        print("\n" + "=" * 80)
        print("REPORT DI VALUTAZIONE DEL MODELLO")
        print("=" * 80)
        print(f"Dataset: {csv_path}")
        print(f"Modello: {self.model_path}")
        print(f"Soglia decisione: {self.threshold}")
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

        print("\n" + "-" * 80)
        print("CLASSIFICATION REPORT (DETTAGLIATO)")
        print("-" * 80)
        print(report_str)

        print("\n" + "-" * 80)
        print("TOP 10 FEATURES (Feature Importance)")
        print("-" * 80)
        for idx, row in importance_df.head(10).iterrows():
            importance_pct = (row["importance_mean"] / importance_df["importance_mean"].sum()) * 100
            bar_length = int(importance_pct / 2)
            bar = "█" * bar_length
            print(f"{row['feature']:30s} {bar:40s} {importance_pct:6.2f}% "
                  f"(±{row['importance_std']:.4f})")

        print("\n" + "=" * 80 + "\n")

    def save_evaluation_report(
            self,
            evaluation_result: dict,
            output_dir: str,
            importance_df: pd.DataFrame,
            fpr,
            tpr,
            precision_vals,
            recall_vals,
    ):
        """
        Salva il report di valutazione in formato JSON e HTML.

        Parametri
        ---------
        evaluation_result : dict
            Dizionario restituito da evaluate()
        output_dir : str
            Directory di output
        importance_df : pd.DataFrame
            DataFrame con feature importance
        fpr, tpr : array
            Curve ROC
        precision_vals, recall_vals : array
            Curve Precision-Recall
        """
        os.makedirs(output_dir, exist_ok=True)

        # Salva JSON
        json_path = os.path.join(output_dir, "evaluation_report.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(evaluation_result, f, indent=2, ensure_ascii=False)
        logger.info(f"Report JSON salvato: {json_path}")

        # Salva CSV con feature importance
        importance_csv_path = os.path.join(output_dir, "feature_importance.csv")
        importance_df.to_csv(importance_csv_path, index=False)
        logger.info(f"Feature importance salvata: {importance_csv_path}")

        # Genera HTML
        html_path = os.path.join(output_dir, "evaluation_report.html")
        html_content = self._generate_html_report(
            evaluation_result, importance_df, fpr, tpr, precision_vals, recall_vals
        )
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        logger.info(f"Report HTML salvato: {html_path}")

        print(f"\n✅ Report salvato in:")
        print(f"   - JSON:  {json_path}")
        print(f"   - CSV:   {importance_csv_path}")
        print(f"   - HTML:  {html_path}")

    def _generate_html_report(
            self,
            evaluation_result: dict,
            importance_df: pd.DataFrame,
            fpr,
            tpr,
            precision_vals,
            recall_vals,
    ) -> str:
        """Genera un report HTML"""
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
                <strong>{self.model_path}</strong>
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
