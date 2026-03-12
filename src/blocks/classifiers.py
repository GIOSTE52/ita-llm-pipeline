"""
Classificatore ML binario per la qualità dei documenti di testo.

Prende in input un vettore di feature (estratte da ItalianFeatureExtractor)
e classifica ogni documento come "good" oppure "bad".

Utilizzo:
    1) Training:
        result = QualityClassifier.train_from_csv("quality_dataset.csv")
        QualityClassifier.save_model(result, "models/quality_model.joblib")

    2) Nella pipeline datatrove (inferenza):
        QualityClassifier(model_path="models/quality_model.joblib")
"""

from __future__ import annotations

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

import lightgbm as lgb

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

logger = logging.getLogger(__name__)

# Feature che il classificatore si aspetta di trovare in doc.metadata
# (devono coincidere con quelle prodotte da ItalianFeatureExtractor)
DEFAULT_FEATURE_NAMES: List[str] = [
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
"avg_sentence_length",
"quote_ratio",
"parenthesis_ratio",
"comma_ratio",
"period_ratio",
"question_mark_ratio",
"exclamation_ratio",
"colon_ratio",
"semicolon_ratio",
"stopword_ratio",
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
"most_common_word_freq",
"repeated_word_count",
"repeated_word_ratio",
"repeated_char_count",
"repeated_char_ratio",
"repeated_sequence_count",
"text_entropy",
"unique_word_count",
"unique_word_ratio",
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
        self.feature_names = feature_names or DEFAULT_FEATURE_NAMES
        self.threshold = threshold

        # Carica modello e scaler dal file .joblib
        artifact = joblib.load(self.model_path)
        self.model: lgb.LGBMClassifier = artifact["model"]
        self.scaler: StandardScaler = artifact["scaler"]
        self._feature_names_train: List[str] = artifact["feature_names"]

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

            X = np.array(features).reshape(1, -1)
            X_scaled = self.scaler.transform(X)

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
        feat_names = feature_names or DEFAULT_FEATURE_NAMES

        # ------- 1. Caricamento dati -------
        df = pd.read_csv(csv_path)

        # Mantengo la colonna degli id 
        doc_ids = df["doc_id"]
        df = df.drop(columns=["doc_id"])

        missing_cols = set(feat_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel CSV: {missing_cols}")
        if label_column not in df.columns:
            raise ValueError(
                f"Colonna label '{label_column}' non trovata nel CSV"
            )

        X = df[feat_names]
        y = df[label_column].map(LABEL_MAP)

        # isna() rileva la mancanza di valori (ovvero valori NaN, a seguito del mapping sopra efettuato) e mappa seocodno True e False
        if y.isna().any():
            invalid = df[label_column][y.isna()].unique().tolist()
            raise ValueError(
                f"Valori label non validi: {invalid}. Ammessi: 'good', 'bad'."
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

        # ------- 2. Scaling -------
        # Normalizzazione delle features
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            # fit_transform() calcola media e deviazione standard sul dataset e trasforma i dati
            scaler.fit_transform(X), columns=feat_names
        )

        # ------- 3. Train / Validation split -------
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled,
            y,
            test_size=test_size,    # default 0.2 -> 20% validazione
            stratify=y,     # mantiene la porzione good/bad
            random_state=random_state,
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
        model.fit(X_train, y_train)

        # ------- 5. Valutazione -------
        y_pred = model.predict(X_val)

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
            X_val,
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
            "feature_names": feat_names,
            "classification_report": report,
            "confusion_matrix": cm,
            "correlation_matrix":correlation_matrix,
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
        }
        joblib.dump(artifact, output_path)
        logger.info("Modello salvato in %s", output_path)
        print(f"\nModello salvato in: {output_path}")
        return os.path.abspath(output_path)