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
# (devono coincidere con quelle prodotte da DocStatsCsv)
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
        threshold: float = 0.65,
    ):
        super().__init__()
        # Percorso al modello salvato
        self.model_path = model_path
        # Soglia di confidenza, ovvero
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
        # Recupero i metadata dei documenti utilizzati e salvati durante il training per mantenere la coerenza
        self.training_metadata: Dict[str, Any] = artifact.get(
            "training_metadata",
            {},
        )
        self.feature_names = feature_names or self._feature_names_train or DEFAULT_FEATURE_NAMES

        logger.info("Modello caricato da %s", self.model_path)

    # -----------------------------------------------------------------
    # Pipeline step: inferenza documento per documento
    # -----------------------------------------------------------------
    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        """Classifica ogni documento e produce il risultato nei metadata."""
        # I documenti vengono processati uno alla volta (alta efficienza in memoria)
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

            # Creo un DataFrame con i nomi delle features scelte in fase di progettazione come colonne e 
            # riempo le righe con i valori associati al doc processato 
            X = pd.DataFrame([features], columns = self.feature_names) # riga da rimuovere se si vuole usare np.array
            # Applico lo stesso scaler usato in fase di training per mantenere la coerenza
            X_scaled = self.scaler.transform(X)
            X_scaled = pd.DataFrame(X_scaled, columns = self.feature_names) # riga d arimuovere se si vuole usare np.array

            # Calcolo predizione con il modello e seleziono l'etichetta in base alla soglia scelta 
            proba = self.model.predict_proba(X_scaled)[0]  # [P(bad), P(good)]
            score_good = float(proba[1])
            label = "good" if score_good >= self.threshold else "bad"

            doc.metadata["quality_label"] = label
            doc.metadata["quality_score"] = round(score_good, 4)

            yield doc


    def _extract_features(self, doc) -> Optional[List[float]]:
        """Estrae il vettore di feature dai metadata del documento, dove sono state salvate le feature calcolate in DocStatsCsv."""
        try:
            return [float(doc.metadata[f]) for f in self.feature_names]
        except (KeyError, TypeError) as e:
            logger.debug("Impossibile estrarre le feature: %s", e)
            return None

    # METODI STATICI UTILI DURANTE IL TRAINING
    @staticmethod
    def _load_labeled_dataset(
        csv_path: str,
        feature_names: Optional[List[str]] = None,
        label_column: str = "label",
    ) -> tuple[pd.DataFrame, pd.Series, List[str]]:
        """Carica un CSV etichettato e valida feature e label,,poi ocnverte le ulime in formato numerico"""
        feat_names = feature_names or DEFAULT_FEATURE_NAMES
        df = pd.read_csv(csv_path)

        # Controllo se le feature scelte corrispondono alle colonne del csv ovvero alle feature calcolate
        missing_cols = set(feat_names) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Colonne mancanti nel CSV: {missing_cols}")
        if label_column not in df.columns:
            raise ValueError(
                f"Colonna label '{label_column}' non trovata nel CSV"
            )

        X = df[feat_names].copy()
        y = df[label_column].map(LABEL_MAP)

        # Controllo che non ci siano campi NaN in y (ovvero che l'operazione di mapping
        # precedente sia andata a buon fine)
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
        """Calcola un set coerente di metriche"""
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

    # Genera la lista di modelli con cui confrontare le prestazioni del modello scelto (cross-validation)
    @staticmethod
    def _build_candidate_models(random_state: int = 42) -> Dict[str, Dict[str, Any]]:
        """Restituisce i modelli scelti per il benchmarking"""
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

    # Metodo che permette di effettuare la cross-validation con i modelli istanziati in _build_candidate_models()
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
            # L'utilizzo di sorted, nel controllo della lista di modelli selezionati dall'utente con quelli disponibili,
            # permette di visualizzare un messaggio di output (errore) sempre nello stesso ordine 
            invalid_model_names = sorted(set(model_names) - set(candidate_models))
            if invalid_model_names:
                raise ValueError(
                    "Modelli non supportati: "
                    f"{invalid_model_names}. Valori ammessi: {sorted(candidate_models)}"
                )
            selected_model_names = model_names
        else:
            selected_model_names = list(candidate_models.keys())

        # creo lo splitter dei dati per la cross-validation stratificato, ovvero dove mantengo la stessa proporzione di "good" e "bad"
        cv = StratifiedKFold(
            n_splits=cv_folds,
            # mescolo i dati prima della divisione nei vari fold
            shuffle=True,
            # imposto il random_seed per mantenere riproducibilità nelle esecuzioni successive
            random_state=random_state,
        )
        # Lista per contenere i risultati di ogni modello
        comparison_rows: List[Dict[str, Any]] = []

        for model_key in selected_model_names:
            spec = candidate_models[model_key]
            # Lista per accumulare le metriche calcolate su ogni fold
            fold_metrics: List[Dict[str, float]] = []
            # creo due array: oof_proba contiene le probabilità predette out-of-fold, mentre oof_pred contiene le classi predette out-of-fold
            oof_proba = np.zeros(len(y), dtype=float)
            oof_pred = np.zeros(len(y), dtype=int)

            for train_idx, val_idx in cv.split(X, y):
                # creo una copia del modello per evitare che uno stato venga riusato tra fold diversi
                estimator = clone(spec["estimator"])
                # divido feature e label in training e validation per il fold corrente.
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # addestro il modello sul training fold
                estimator.fit(X_train, y_train)
                # calcolo la probabilità della classe positiva, in questo caso "good", infatti prendo la seconda colonna di predict_proba
                val_pred_proba = estimator.predict_proba(X_val)[:, 1]
                # converto le probabilità in predizioni binarie in base a dove si posizionano rispetto alla soglia threshold
                val_pred = (val_pred_proba >= threshold).astype(int)

                # calcolo le metriche del fold e le salvo in fold_metrics
                fold_metrics.append(
                    QualityClassifier._compute_binary_metrics(
                        y_true=y_val,
                        y_pred=val_pred,
                        y_pred_proba=val_pred_proba,
                    )
                )
                # salvo le predizioni nella posizione della riga del dataset
                oof_proba[val_idx] = val_pred_proba
                oof_pred[val_idx] = val_pred

            # calcolo media e deviazione standard delle metriche sui fold
            summary = QualityClassifier._summarize_fold_metrics(fold_metrics)
            # calcolo le metriche complessive del modello usando tutte le predizioni out-of-fold
            overall_metrics = QualityClassifier._compute_binary_metrics(
                y_true=y,
                y_pred=oof_pred,
                y_pred_proba=oof_proba,
            )
            # genero la confusion matrix
            confusion = confusion_matrix(y, oof_pred)
            # genero un report sklearn con le metriche più rilevanti
            report = classification_report(
                y,
                oof_pred,
                target_names=["bad", "good"],
                output_dict=True,
                zero_division=0,
            )

            # aggiungo alla lista il risultato del modello corrente
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

        # ordino i modelli dal migliore al peggiore (in base ai valori roc_auc_mean e f1_score_mean)
        comparison_rows.sort(
            key=lambda row: (row["roc_auc_mean"], row["f1_score_mean"]),
            reverse=True,
        )

        # come baseline uso lightgbm se si trova nella lista dei modelli confrontati, altrimenti uso il primo in classifica
        baseline_key = (
            "lightgbm"
            if "lightgbm" in selected_model_names
            else comparison_rows[0]["model_key"]
        )
        # recupero la riga dei risultati del modello baseline
        baseline_row = next(
            row for row in comparison_rows if row["model_key"] == baseline_key
        )
        # definisco le metriche su cui calcolare le differenze con la baseline
        metric_fields = [
            "accuracy_mean",
            "balanced_accuracy_mean",
            "precision_mean",
            "recall_mean",
            "f1_score_mean",
            "roc_auc_mean",
        ]
        # calcolo i delta rispetto alla baseline e costruisco un dizionario
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
        validation_csv_path: Optional[str] = None,
        test_size: float = 0.2,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        threshold: float = 0.65,
        random_state: int = 42,
    ) -> dict:
        """
        Addestra un modello LightGBM binario a partire da un CSV.

        Il CSV deve contenere:
        - le colonne corrispondenti alle feature (stessi nomi di DEFAULT_FEATURE_NAMES
          oppure quelli specificati in ``feature_names``)
        - una colonna ``label`` con valori "good" / "bad"
        - opzionalmente un CSV di validazione separato, per evitare split interni

        Restituisce un dizionario con modello, scaler, metriche e
        importanza delle feature.
        """
        # 1. Caricamento dati
        X, y, feat_names = QualityClassifier._load_labeled_dataset(
            csv_path=csv_path,
            feature_names=feature_names,
            label_column=label_column,
        )
        
        # 1b. Correlation Matrix
        # Calcolo la matrice di correlazione tra tutte le feature e la label
        corr_df = X.copy()
        corr_df["label"] = y
        correlation_matrix = corr_df.corr()

        print("=" * 60)
        print("CORRELATION MATRIX")
        print("=" * 60)
        # Mostro le correlazioni di ogni feature con la label(eliminando la correlazione con se stessa),
        # ordinate per valore assoluto
        label_corr = correlation_matrix["label"].drop("label").sort_values(
            key=abs, ascending=False
        )
        print("\nCorrelazione con la label (ordinate per valore assoluto):")
        print(label_corr.to_string())

        # Identifica coppie di feature altamente correlate tra loro (|r| > 0.9)
        high_corr_threshold = 0.9
        feature_corr = correlation_matrix.drop(columns=["label"], index=["label"])
        high_corr_pairs = []
        for i in range(len(feature_corr.columns)):
            for j in range(i + 1, len(feature_corr.columns)):
                # recupero il coefficiente di correlazione tra le due feature correnti (i,j)
                r = feature_corr.iloc[i, j]
                if abs(r) > high_corr_threshold:
                    high_corr_pairs.append(
                        (feature_corr.columns[i], feature_corr.columns[j], round(r, 4))
                    )

        if high_corr_pairs:
            print(f"\nCoppie di feature con correlazione assoluta > {high_corr_threshold}:")
            for f1, f2, r in high_corr_pairs:
                print(f"  {f1}  ↔  {f2}  :  {r}")
        else:
            print(f"\nNessuna coppia di feature con correlazione assoluta > {high_corr_threshold}")

        # 2. Train / Validation split
        if validation_csv_path:
            X_train = X
            y_train = y
            X_val, y_val, _ = QualityClassifier._load_labeled_dataset(
                csv_path=validation_csv_path,
                feature_names=feat_names,
                label_column=label_column,
            )
            split_metadata = {
                "split_strategy": "precomputed_validation_csv",
                "train_csv": os.path.abspath(csv_path),
                "validation_csv": os.path.abspath(validation_csv_path),
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
                "random_state": random_state,
            }
        else:
            # creo uno split interno del dataset
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y,
                test_size=test_size,    # default 0.2 , ovvero 20% validazione
                stratify=y,     # mantiene la porzione good/bad in train e validation
                random_state=random_state,
            )
            split_metadata = {
                "split_strategy": "internal_train_test_split",
                "train_csv": os.path.abspath(csv_path),
                "validation_csv": None,
                "train_rows": int(len(X_train)),
                "validation_rows": int(len(X_val)),
                "validation_fraction": float(test_size),
                "random_state": random_state,
            }

        # 3. Scaling 
        # Fitto lo scaler solo sul training set per evitare data leakage.
        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=feat_names,
            index=X_train.index,
        )
        # applico alla validazione lo stesso scaler addestrato sul training set
        X_val_scaled = pd.DataFrame(
            scaler.transform(X_val),
            columns=feat_names,
            index=X_val.index,
        )

        # 4. Training 
        # creo il modello binario
        model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=-1,
            random_state=random_state,
            verbose=-1,
        )
        # Chiamata che effettua l'apprendimento sui dati di training
        model.fit(X_train_scaled, y_train)

        # 5. Valutazione
        # calcolo la probabilità della classe positiva "good", seconda colonna di predict_proba 
        y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
        # converto la probabilità in classe binaria in base alla posizione rispetto alla soglia scelta
        y_pred = (y_pred_proba >= threshold).astype(int)
        # calcolo metriche principali
        metrics = QualityClassifier._compute_binary_metrics(
            y_true=y_val,
            y_pred=y_pred,
            y_pred_proba=y_pred_proba,
        )

        # Genero un report in formato dizionario 
        report = classification_report(
            y_val, y_pred, target_names=["bad", "good"], output_dict=True,
            zero_division=0,
        )
        # Genero una string stampabile a schermo
        report_str = classification_report(
            y_val, y_pred, target_names=["bad", "good"], zero_division=0
        )
        # genero la confusion matrix
        cm = confusion_matrix(y_val, y_pred)

        print("=" * 60)
        print("CLASSIFICATION REPORT")
        print("=" * 60)
        print(f"Soglia decisione validazione: {threshold:.2f}")
        print(report_str)
        print("\nConfusion Matrix:")
        print(cm)

        # 6. Feature Importance (permutation)
        # misura quanto peggiora il modello quando una feature viene mescoalta (permutazioni)
        print("\nCalcolo permutation feature importance...")
        perm = permutation_importance(
            model,
            X_val_scaled,
            y_val,
            # numero di ripetizioni del mescolamento
            n_repeats=10,
            random_state=random_state,
            n_jobs=-1,
        )

        # creo un dataFrame per visualizzare i valori di importance calcolati, e li ordino dal 
        # livello più alto di importance_mean in poi
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
            "threshold": threshold,
            "validation_metrics": {
                metric_name: round(metric_value, 4)
                for metric_name, metric_value in metrics.items()
            },
            "split_metadata": split_metadata,
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
            Percorso dove salvare il file .joblib.

        Restituisce il percorso assoluto del file salvato.
        """
        # Validazione di training_result
        if not training_result or not isinstance(training_result, dict):
            error_msg = "Errore: training_result è vuoto o non è un dizionario"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Verifica delle chiavi essenziali
        required_keys = ["model", "scaler", "feature_names"]
        missing_keys = [key for key in required_keys if key not in training_result]
        if missing_keys:
            error_msg = f"Errore: training_result non contiene le chiavi richieste: {missing_keys}"
            logger.error(error_msg)
            raise KeyError(error_msg)
        
        # Verifica che il modello sia stato effettivamente fittato
        model = training_result["model"]
        if not hasattr(model, 'booster_'):
            error_msg = f"Errore: il modello non è stato addestrato (booster_ non trovato)"
            logger.error(error_msg)
            raise RuntimeError(error_msg)
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        artifact = {
            "model": training_result["model"],
            "scaler": training_result["scaler"],
            "feature_names": training_result["feature_names"],
            "model_name": training_result.get(
                "model_name",
                training_result["model"].__class__.__name__,
            ),
            "threshold": training_result.get("threshold"),
            "validation_metrics": training_result.get("validation_metrics"),
            "training_metadata": training_result.get(
                "training_metadata",
                training_result.get("split_metadata", {}),
            ),
        }
        # comando per salvare il modello addestrato
        joblib.dump(artifact, output_path)

        logger.info("Modello salvato in %s", output_path)
        print(f"\nModello salvato in: {output_path}")
        return os.path.abspath(output_path)