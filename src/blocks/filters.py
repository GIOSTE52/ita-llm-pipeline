import os
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers import JsonlWriter
from typing import List, Tuple
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.writers.disk_base import DiskWriter

from blocks.classifiers import QualityClassifier, DEFAULT_FEATURE_NAMES

import pandas as pd

def get_language_filter(rejected_dir: str, threshold: float = 0.65, languages = "it"):
    """
    Inizializza il filtro per la lingua italiana.
    
    Parametri:
    - rejected_dir: Cartella dove salvare i testi non in italiano.
    - threshold: Soglia di confidenza del modello fasttext (0.65 consigliata).
    """
    return LanguageFilter(
        languages=languages,
        language_threshold=threshold,
        exclusion_writer=JsonlWriter(
            output_folder=os.path.join(rejected_dir, "1_language"),
            output_filename="non_italiano_${rank}.jsonl",
            compression=None
        )
    )

class CustomItalianFilter(BaseFilter):
    """
    Filtro euristico per la pulizia del rumore testuale (Menu, Footer, Link).
    
    Scarta i documenti basandosi su:
    - Lunghezza minima (< 120 char).
    - Pattern di navigazione (troppi caratteri '|').
    - Parole chiave sospette (Home, Login, etc.).
    """
    def __init__(self, output_folder: str, filename: str):
        super().__init__()
        self.name = "🇮🇹 Custom Italian Filter"
        self.exclusion_writer = JsonlWriter(
            output_folder=output_folder,
            output_filename=filename,
            compression=None
        )
        self.bad_keywords = ["home", "contatti", "cookie policy", "privacy", "login"]

    def filter(self, doc) -> bool:
        text_lower = doc.text.lower()
        # Soglia da controllare sull'excel fatto durante lo sviluppo della pipeline RedPajama
        if len(text_lower) < 120: return False
        # Da monitorare poichè alcuni testi ottimi non passano
        if text_lower.count("|") > 6: return False
        bad_count = sum(1 for word in self.bad_keywords if word in text_lower)
        return bad_count < 3
    
# Da sviluppare in seguito, ogni testo etichettato dal classificatore in input a questo blocco
# deve essere poi maneggiato per fare sì che si trovi in /output oppure /rejected
class ItalianClassification(BaseFilter):
    """
    Filtro di qualita basato su ``QualityClassifier``.

    I documenti ``good`` vengono inoltrati allo step successivo della pipeline,
    quindi verranno scritti dal writer finale in output.
    I documenti ``bad`` vengono invece scartati e scritti dal writer di esclusione
    nella directory ``rejected``.
    """

    def __init__(
            self,
            model_path: str,
            rejected_dir: str | None = None,
            output_folder: str | None = None,
            output_filename:str = "quality_rejectd_${rank}.jsonl",
            exclusion_writer: DiskWriter | None = None,
            feature_names: List[str] = DEFAULT_FEATURE_NAMES,
            threshold: float = 0.65,
            batch_size: int = 1,
    ):
        if exclusion_writer is None and (rejected_dir or output_folder):
            exclusion_writer = JsonlWriter(
                output_folder= os.path.join(rejected_dir, "3_quality"),
                output_filename= output_filename,
                compression= None
            )

        super().__init__(exclusion_writer=exclusion_writer, batch_size=batch_size)
        self.name = "🇮🇹 Italian Classification"
        self.classifier = QualityClassifier(
            model_path= model_path,
            feature_names= feature_names,
            threshold= threshold
        )

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        features = self.classifier._extract_features(doc)
        if features is None:
            doc.metadata["quality_label"] = "bad"
            doc.metadata["quality_score"] = 0.0
            return False, "quality_missing_features"
        
        score_good, label = self._predict(features)
        doc.metadata["quality_label"] = label
        doc.metadata["quality_score"] = round(score_good, 4)

        if label == "good":
            return True
        return False, "quality_bad"

    def filter_batch(self, batch: List[Document]) -> List[bool | Tuple[bool, str]]:
        return [self.filter(doc) for doc in batch]
    
    def _predict(self, features: List[float]) -> Tuple[float, str]:
        x = pd.DataFrame([features], columns=self.classifier.feature_names) 
        x_scaled = self.classifier.scaler.transform(x)
        x_scaled = pd.DataFrame(x_scaled, columns=self.classifier.feature_names) 
        proba = self.classifier.model.predict_proba(x_scaled)[0]
        score_good = float(proba[1])
        label = "good" if score_good >= self.classifier.threshold else "bad"
        return score_good, label