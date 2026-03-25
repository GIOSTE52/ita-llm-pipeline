import os
from datatrove.pipeline.filters import LanguageFilter
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers import JsonlWriter
from typing import List, Callable, Tuple
from datatrove.data import Document, DocumentsPipeline
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.pipeline.writers.disk_base import DiskWriter

def get_language_filter(rejected_dir: str, threshold: float = 0.65):
    """
    Inizializza il filtro per la lingua italiana.
    
    Parametri:
    - rejected_dir: Cartella dove salvare i testi non in italiano.
    - threshold: Soglia di confidenza del modello fasttext (0.65 consigliata).
    """
    return LanguageFilter(
        languages="it",
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
        if len(text_lower) < 120: return False
        if text_lower.count("|") > 6: return False
        bad_count = sum(1 for word in self.bad_keywords if word in text_lower)
        return bad_count < 3
    

class ItalianFilter(BaseFilter):

    def __init__(self, filter_function: Callable[[Document], bool], exclusion_writer: DiskWriter = None, batch_size: int = 1):
        super().__init__(exclusion_writer, batch_size)
        self.filter_function = filter_function

    def filter(self, doc: Document) -> bool | Tuple[bool, str]:
        return self.filter_function(doc)

    def filter_batch(self, batch: List[Document]) -> List[bool | Tuple[bool, str]]:
        return super().filter_batch(batch)
        