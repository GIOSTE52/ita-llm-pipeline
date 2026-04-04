"""
Classificatore ML binario per la qualità dei documenti di testo.

Prende in input un vettore di feature (estratte da DocStatsCsv)
e classifica ogni documento come "good" oppure "bad".
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
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    f1_score,
    accuracy_score,
    balanced_accuracy_score,
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
    ) -> dict:
        """
        Valuta il modello su un dataset di test contenuto in un CSV.

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
        # Carica il dataset
        df = pd.read_csv(csv_path)
        
        # Verifica colonne
        missing_cols = set(self.feature_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel CSV: {missing_cols}")
        if label_column not in df.columns:
            raise ValueError(f"Colonna label '{label_column}' non trovata nel CSV")

        # Estrai feature e label
        X = df[self.feature_names]
        y = df[label_column].map(LABEL_MAP)
        
        if y.isna().any():
            invalid = df[label_column][y.isna()].unique().tolist()
            raise ValueError(
                f"Valori label non validi: {invalid}. Ammessi: 'good', 'bad'."
            )

        # Scala le feature usando lo scaler del modello
        X_scaled = self.scaler.transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_names)

        # Predizioni
        y_pred = self.model.predict(X_scaled)
        y_pred_proba = self.model.predict_proba(X_scaled)[:, 1]  # Probabilità della classe "good"

        # Calcolo metriche
        accuracy = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        roc_auc = roc_auc_score(y, y_pred_proba)
        f1 = f1_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report_dict = classification_report(
            y, y_pred, target_names=["bad", "good"], output_dict=True
        )
        report_str = classification_report(
            y, y_pred, target_names=["bad", "good"]
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

        # Stampa report formattato
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

        # Prepara il risultato
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
        }

        # Salva i report se specificato output_dir
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
        """Genera un report HTML interattivo."""
        cm = evaluation_result["confusion_matrix"]
        
        html = f"""
<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Report Valutazione Modello</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            padding: 30px;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            padding-bottom: 5px;
            border-bottom: 1px solid #ecf0f1;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            opacity: 0.9;
        }}
        .metric-card .value {{
            font-size: 32px;
            font-weight: bold;
        }}
        .chart-container {{
            position: relative;
            height: 400px;
            margin: 30px 0;
            padding: 20px;
            background: #f9f9f9;
            border-radius: 8px;
        }}
        .confusion-matrix {{
            text-align: center;
            margin: 20px 0;
        }}
        .confusion-matrix table {{
            margin: 0 auto;
            border-collapse: collapse;
        }}
        .confusion-matrix td {{
            border: 2px solid #3498db;
            padding: 15px;
            font-size: 16px;
            font-weight: bold;
            width: 120px;
        }}
        .confusion-matrix td:first-child {{
            background: #ecf0f1;
        }}
        .confusion-matrix tr:first-child td {{
            background: #ecf0f1;
        }}
        .tn {{ background: #2ecc71; color: white; }}
        .fp {{ background: #e74c3c; color: white; }}
        .fn {{ background: #e67e22; color: white; }}
        .tp {{ background: #27ae60; color: white; }}
        .features-table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        .features-table th, .features-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        .features-table th {{
            background: #3498db;
            color: white;
        }}
        .features-table tr:hover {{
            background: #f5f5f5;
        }}
        .progress-bar {{
            width: 100%;
            height: 20px;
            background: #ecf0f1;
            border-radius: 10px;
            overflow: hidden;
            margin: 5px 0;
        }}
        .progress-fill {{
            height: 100%;
            background: linear-gradient(90deg, #3498db, #2ecc71);
            display: flex;
            align-items: center;
            justify-content: flex-end;
            padding-right: 10px;
            color: white;
            font-weight: bold;
            font-size: 12px;
        }}
        .timestamp {{
            color: #7f8c8d;
            font-size: 12px;
            margin-top: 20px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Report di Valutazione del Modello</h1>
        <p><strong>Dataset:</strong> {evaluation_result['csv_path']}</p>
        <p><strong>Modello:</strong> {self.model_path}</p>
        <p><strong>Soglia:</strong> {evaluation_result['threshold']}</p>
        
        <h2>🎯 Metriche Principali</h2>
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Accuracy</h3>
                <div class="value">{evaluation_result['accuracy']:.2%}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);">
                <h3>Balanced Accuracy</h3>
                <div class="value">{evaluation_result['balanced_accuracy']:.2%}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);">
                <h3>F1-Score</h3>
                <div class="value">{evaluation_result['f1_score']:.4f}</div>
            </div>
            <div class="metric-card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);">
                <h3>ROC-AUC</h3>
                <div class="value">{evaluation_result['roc_auc']:.4f}</div>
            </div>
        </div>

        <h2>🔲 Matrice di Confusione</h2>
        <div class="confusion-matrix">
            <table>
                <tr>
                    <td></td>
                    <td><strong>Pred: Bad</strong></td>
                    <td><strong>Pred: Good</strong></td>
                </tr>
                <tr>
                    <td><strong>Real: Bad</strong></td>
                    <td class="tn">{cm[0][0]}</td>
                    <td class="fp">{cm[0][1]}</td>
                </tr>
                <tr>
                    <td><strong>Real: Good</strong></td>
                    <td class="fn">{cm[1][0]}</td>
                    <td class="tp">{cm[1][1]}</td>
                </tr>
            </table>
        </div>

        <h2>⭐ Top 10 Features (Importanza)</h2>
        <table class="features-table">
            <thead>
                <tr>
                    <th>Feature</th>
                    <th>Importanza Media</th>
                    <th>Dev. Std.</th>
                    <th>Visualizzazione</th>
                </tr>
            </thead>
            <tbody>
"""
        for idx, row in importance_df.head(10).iterrows():
            importance_pct = (row["importance_mean"] / importance_df["importance_mean"].sum()) * 100
            html += f"""
                <tr>
                    <td><strong>{row['feature']}</strong></td>
                    <td>{row['importance_mean']:.6f}</td>
                    <td>±{row['importance_std']:.6f}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: {importance_pct}%;">
                                {importance_pct:.1f}%
                            </div>
                        </div>
                    </td>
                </tr>
"""
        html += """
            </tbody>
        </table>

        <h2>📈 Curve di Valutazione</h2>
        <div class="chart-container">
            <canvas id="rocChart"></canvas>
        </div>
        <div class="chart-container">
            <canvas id="prChart"></canvas>
        </div>

        <div class="timestamp">
            <strong>Generato:</strong> """ + evaluation_result['timestamp'] + """
        </div>
    </div>

    <script>
        // ROC Curve
        const rocCtx = document.getElementById('rocChart').getContext('2d');
        new Chart(rocCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'ROC Curve',
                        data: """ + json.dumps([{"x": float(f), "y": float(t)} for f, t in zip(fpr, tpr)]) + """,
                        borderColor: '#3498db',
                        backgroundColor: 'rgba(52, 152, 219, 0.1)',
                        fill: false,
                        tension: 0.1,
                        showLine: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'ROC Curve' }
                },
                scales: {
                    x: { title: { display: true, text: 'False Positive Rate' }, min: 0, max: 1 },
                    y: { title: { display: true, text: 'True Positive Rate' }, min: 0, max: 1 }
                }
            }
        });

        // Precision-Recall Curve
        const prCtx = document.getElementById('prChart').getContext('2d');
        new Chart(prCtx, {
            type: 'scatter',
            data: {
                datasets: [
                    {
                        label: 'Precision-Recall Curve',
                        data: """ + json.dumps([{"x": float(r), "y": float(p)} for r, p in zip(recall_vals, precision_vals)]) + """,
                        borderColor: '#e74c3c',
                        backgroundColor: 'rgba(231, 76, 60, 0.1)',
                        fill: false,
                        tension: 0.1,
                        showLine: true
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: true },
                    title: { display: true, text: 'Precision-Recall Curve' }
                },
                scales: {
                    x: { title: { display: true, text: 'Recall' }, min: 0, max: 1 },
                    y: { title: { display: true, text: 'Precision' }, min: 0, max: 1 }
                }
            }
        });
    </script>
</body>
</html>
"""
        return html