from __future__ import annotations

import logging
import os
from typing import List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers import JsonlWriter


from .spam_stats import FEATURE_COLUMNS

logger = logging.getLogger(__name__)

LABEL_MAP = {"ham": 0, "spam": 1}
INV_LABEL_MAP = {0: "ham", 1: "spam"}

LABEL_COLUMNS = ["spam_target_label", "target_label"]

DEFAULT_FEATURE_NAMES: List[str] = [
    c for c in FEATURE_COLUMNS if c not in {"doc_id", "target_label", "spam_target_label"}
]


class SpamClassifier(PipelineStep):
    """
    Motore di inferenza spam.
    NON filtra i documenti: aggiunge solo metadata di predizione.
    """
    name = "Spam Classifier"

    def __init__(
        self,
        model_path: str,
        feature_names: Optional[List[str]] = None,
        threshold: Optional[float] = None,
    ):
        super().__init__()
        self.model_path = model_path
        
        artifact = joblib.load(self.model_path)
        
        self.model: lgb.LGBMClassifier = artifact["model"]
        self.scaler: StandardScaler = artifact["scaler"]
        self._feature_names_train: List[str] = artifact.get("feature_names", DEFAULT_FEATURE_NAMES)

        #se la trashold non è impostata usa come predefinito 0.5
        saved_threshold = artifact.get("threshold", 0.5)
        self.threshold = float(saved_threshold if threshold is None else threshold)

        self.feature_names = feature_names or self._feature_names_train

        logger.info(
            "Spam model caricato da %s | feature=%d | threshold=%.3f",
            self.model_path,
            len(self.feature_names),
            self.threshold,
        )

    def _extract_features(self, doc) -> Optional[List[float]]:
        values = []
        metadata = getattr(doc, "metadata", {}) or {}
        try:
            for f in self.feature_names:
                values.append(float(metadata.get(f, 0.0)))
            return values
        except Exception as e:
            logger.warning("Feature mancanti/non numeriche per doc %s: %s", getattr(doc, "id", ""), e)
            return None
    
    def _predict_from_features(self, feats: List[float]) -> Tuple[str, float]:
        X = pd.DataFrame([feats], columns=self.feature_names)
        Xs = self.scaler.transform(X)
        Xs = pd.DataFrame(Xs, columns=self.feature_names)
        proba = self.model.predict_proba(Xs)[0]
        spam_score = float(proba[1])
        pred_label = "spam" if spam_score >= self.threshold else "ham"
        return pred_label, spam_score 

    def predict_doc(self, doc) -> Tuple[str, float]:
        feats = self._extract_features(doc)
        if feats is None:
            return "ham", 0.0
        return self._predict_from_features(feats)
    
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            pred_label, spam_score = self.predict_doc(doc)

            if getattr(doc, "metadata", None) is None:
                doc.metadata = {}

            doc.metadata["spam_pred_label"] = pred_label
            doc.metadata["spam_pred_score"] = round(float(spam_score), 6)

            yield doc

    @staticmethod
    def _resolve_label_column(df: pd.DataFrame, label_column: Optional[str] = None) -> str:
        if label_column:
            if label_column not in df.columns:
                raise ValueError(f"Colonna label '{label_column}' non trovata nel CSV")
            return label_column
        
        for candidate in LABEL_COLUMNS:
            if candidate in df.columns:
                return candidate
        
        raise ValueError("Nessuna colonna label valida trovata nel CSV (spam_target_label/target_label)")

    @staticmethod
    def _resolve_feature_names(df: pd.DataFrame, feature_names: Optional[List[str]] = None) -> List[str]:
        requested = feature_names or DEFAULT_FEATURE_NAMES
        available = [c for c in requested if c in df.columns]
        if not available:
            raise ValueError("Nessuna feature valida trovata nel CSV")
        return available
    
    #serve per rimuovere features non utili al training o che renderebbero troppo facile il training
    @staticmethod
    def _drop_bad_features(X: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
        constant_cols = [c for c in X.columns if X[c].nunique(dropna=False) <= 1]
        near_constant_cols = []
        for c in X.columns:
            vc = X[c].value_counts(normalize=True, dropna=False)
            if not vc.empty and vc.iloc[0] >= 0.995:
                near_constant_cols.append(c)
        remove_cols = sorted(set(constant_cols))
        keep = [c for c in X.columns if c not in remove_cols]
        return X[keep].copy(), remove_cols, sorted(set(near_constant_cols) - set(remove_cols))

    #allena il modello sui dati estratti dal csv.
    @staticmethod
    def train_from_csv(
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        label_column: Optional[str] = None,
        test_size: float = 0.2,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        random_state: int = 42,
        threshold: float = 0.5,
    ) -> dict:
        df = pd.read_csv(csv_path)

        if "doc_id" not in df.columns:
            raise ValueError("Colonna 'doc_id' mancante nel CSV")

        label_column = SpamClassifier._resolve_label_column(df, label_column)
        feat_names = SpamClassifier._resolve_feature_names(df, feature_names)

        df[label_column] = df[label_column].astype(str).str.strip().str.lower()
        df = df[df[label_column].isin(["ham", "spam"])].copy()

        if df.empty:
            raise ValueError("Nessuna riga valida trovata nel CSV per il training")

        X = df[feat_names].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        y = df[label_column].map(LABEL_MAP)

        # remove constant junk before split
        X, dropped_constants, near_constants = SpamClassifier._drop_bad_features(X)
        feat_names = X.columns.tolist()

        if len(feat_names) < 2:
            raise ValueError("Dopo la pulizia restano troppo poche feature")

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=15,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.2,
            reg_lambda=0.2,
            random_state=random_state,
            class_weight="balanced",
            verbosity=-1,
        )
        model.fit(X_train_s, y_train)
        
        y_prob = model.predict_proba(X_test_s)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        print("=" * 60)
        print("SPAM CLASSIFICATION REPORT")
        print("=" * 60)
        print(f"Feature usate nel training: {len(feat_names)}")

        if dropped_constants:
            print(f"Feature costanti rimosse: {', '.join(dropped_constants)}")
        if near_constants:
            print(f"Feature quasi costanti rilevate: {', '.join(near_constants)}")

        print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        try:
            auc = roc_auc_score(y_test, y_prob)
            print(f"\nROC-AUC: {auc:.4f}")
        except Exception:
            auc = None

        perm = permutation_importance(
            model,
            X_test_s,
            y_test,
            n_repeats=10,
            random_state=random_state,
            n_jobs=1,
        )

        importances = pd.DataFrame({
            "feature": feat_names,
            "importance_mean": perm.importances_mean,
            "importance_std": perm.importances_std,
        }).sort_values("importance_mean", ascending=False)

        print("\nTop feature importance:")
        print(importances.head(30).to_string(index=False))

        return {
            "model": model,
            "scaler": scaler,
            "feature_names": feat_names,
            "label_column": label_column,
            "threshold": float(threshold),
            "classification_report": classification_report(
                y_test, 
                y_pred, 
                target_names=["ham", "spam"], 
                output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred),
            "roc_auc": auc,
            "feature_importance": importances,
            
        }

    @staticmethod
    def save_model(result: dict, output_path: str):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        artifact = {
            "model": result["model"],
            "scaler": result["scaler"],
            "feature_names": result["feature_names"],
            "label_column": result.get("label_column", "spam_target_label"),
            "threshold": float(result.get("threshold", 0.5)),
        }
        joblib.dump(artifact, output_path)
        print(f"[OK] Modello salvato in: {output_path}")

class SpamFilter(BaseFilter):
    """ 
    Filtro spam vero e proprio.
    - I documenti ham continuano nella pipeline
    - I documenti spam vengono scartati e salvati nella cartella rejected
    """
        
    name = "Spam Filter"

    def __init__(
        self,
        model_path: str,
        rejected_dir: str,
        threshold: Optional[float] = None,
    ):
        self.classifier = SpamClassifier(
            model_path=model_path,
            threshold=threshold,
        )

        exclusion_writer = JsonlWriter(
            output_folder=os.path.join(rejected_dir, "2_spam"),
            output_filename="spam_rejected_${rank}.jsonl",
            compression=None,
        )

        super().__init__(exclusion_writer=exclusion_writer)

    def filter(self, doc):
        if getattr(doc, "metadata", None) is None:
            doc.metadata = {}

        pred_label, spam_score = self.classifier.predict_doc(doc)

        doc.metadata["spam_pred_label"] = pred_label
        doc.metadata["spam_pred_score"] = round(float(spam_score), 6)

        if pred_label == "spam":
            doc.metadata["spam_reject_reason"] = "spam_detected"
            return False, "spam_detected"

        return True