import re
import csv
import math
from collections import Counter
from typing import List, Optional
from datatrove.data import Document
from datatrove.io import DataFolderLike
from datatrove.pipeline.stats.doc_stats import DocStats
from loguru import logger
from datatrove.utils.lid import FT176LID

# --- REGEX PRE-COMPILATE ---
# L'uso di re.compile fuori dal loop di processamento ottimizza le performance,
# evitando la ricompilazione dell'espressione regolare per ogni documento

RE_EMAIL = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")
RE_HTML = re.compile(r"<[^>]+>")
RE_BULLET = re.compile(r"^\s*[-*•]\s", re.MULTILINE)
RE_REPEATED_CHARS = re.compile(r"(.)\1{2,}") # Rileva caratteri ripetuti (es. "belllezzza")
RE_REPEATED_SEQ = re.compile(r"(\b\w+\b\s+)\1+", re.IGNORECASE) # Rileva parole duplicate consecutive
RE_SPACES = re.compile(r" {2,}")
RE_PUNC_SEQ = re.compile(r"[.!?,;:]{2,}")
RE_ELIPSIS = re.compile(r"\.\.\.|…")
RE_URL = re.compile(r"https?://[^\s)>\]\"\'\}]*")

# Lista di stopword italiane per il calcolo della stopword_ratio.
# Fondamentale per distinguere testi naturali da liste di parole o contenuti spazzatura.

ITALIAN_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una","l'", "un'",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "del", "della","dello", "dei", "delle", "degli", 
    "al", "alla", "allo","ai", "agli", "alle", "all'",
    "dal", "dalla", "dallo", "dalla", "dai","dagli", "dall'",
    "nel", "nei", "nelle", "nello", "nella", "negli",
    "col", "coi", "sul", "sulla", "sui", "sulle", "sugli","sullo", "sull'",
    "e", "o", "ma", "però", "tuttavia", "dunque", "quindi", "anche", "dove",
    "pure", "inoltre", "infatti", "invece", "mentre", "quando", "perché",
    "poiché", "benché", "affinché", "purché", "siccome", "ciò",
    "io", "tu", "egli", "lui", "lei", "esso", "noi", "voi", "loro", "essi",
    "mi", "ti", "ci", "vi", "si", "li", "ne",
    "me", "te", "ce", "ve", "se",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "che", "cui", "chi", "c'è", "come",
    "quali", "quale", "quanto", "quanta", "quanti", "quante",  
    "alcuni","alcune", "qualche", "qualcosa", "qualcuno","qualcuna",
    "nulla", "niente", "nessuno", "nessuna", 
    "tutto", "tutte", "tutti", "tutta",
    "altro", "altri", "altre", "altra",
    "stesso", "stessa", "stessi", "stesse",
    "non", "sì", "no", "più", "meno", "molto", "poco", "troppo", "così", "allora",
    "già", "ancora", "sempre", "spesso", "mai", "qui", "qua", "li", "là", 
    "essere", "sono", "sei", "è", "siamo", "siete", "era", "erano", "sia", "siano",
    "avere", "ha", "hanno", "hai", "ho", "avete", "abbiamo" , "po'", "com'", "c'", "d'"
}

class DocStatsCsv(DocStats):

    """
    Componente della pipeline DataTrove per l'estrazione di feature statistiche avanzate.
    Queste feature sono progettate specificamente per il dataset italiano e serviranno
    come input per l'addestramento di un modello classificatore LightGBM (Setaccio).
    """

    name = "Italian Advanced Features CSV"

    def __init__(
        self,
        output_folder: DataFolderLike,
        csv_filename: str = "doc_stats_per_file.csv",
        languages: str = "it",
        **kwargs  #--->accetta i parametri extra come groups_to_compute
    ) -> None:
        # Passiamo i kwargs (incluso groups_to_compute) alla classe base DocStats
        super().__init__(output_folder, **kwargs)
        self.csv_filename = csv_filename
        self.languages = languages
        self.all_docs_stats = []
        self._lid_model = None

    #variabilizzo
    @property
    def lid_model(self):
        """Inizializza il modello FastText LID solo se necessario (risparmio memoria)."""
        if self._lid_model is None:
            self._lid_model = FT176LID([self.languages])
        return self._lid_model

    def _calculate_entropy(self, text: str) -> float:
        """Calcola l'entropia di Shannon a livello di caratteri per misurare la compressione del testo."""
        if not text: return 0.0
        counts = Counter(text)
        text_len = len(text)
        return -sum((count / text_len) * math.log2(count / text_len) for count in counts.values())

    def extract_stats(self, doc: Document) -> dict:
        """
        Motore di estrazione: trasforma il testo grezzo in un vettore di 52 feature numeriche.
        Le feature sono divise in: Base, Linguistiche, Strutturali e di Anomalia.
        """
        text = doc.text
        if not text or len(text) == 0:
            return self._get_empty_stats()

        words = text.split()
        word_count = len(words)
        char_count = len(text)
        lines = text.splitlines()
        line_count = len(lines)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        
        # 1. BASE: Metriche di composizione dei caratteri
        base = {
            "length": char_count,
            "white_space_ratio": sum(1 for c in text if c.isspace()) / char_count,
            "non_alpha_digit_ratio": sum(1 for c in text if not c.isalnum()) / char_count,
            "digit_ratio": sum(1 for c in text if c.isdigit()) / char_count,
            "uppercase_ratio": sum(1 for c in text if c.isupper()) / char_count,
            "elipsis_ratio": len(RE_ELIPSIS.findall(text)) / char_count,
            "punctuation_ratio": sum(1 for c in text if c in '.,;:!?()[]""\'\'') / char_count,
        }

        # 2. LINGUISTICHE: Analisi sintattica superficiale e stopword
        # L'uso delle vocali accentate è specifico per la lingua italiana.
        periods, questions, exclamations = text.count('.'), text.count('?'), text.count('!')
        sentence_count = max(1, periods + questions + exclamations)
        linguistic = {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "vocabulary_size": len(set(w.lower() for w in words)),
            "lowercase_ratio": sum(1 for c in text if c.islower()) / char_count,
            "vowel_ratio": sum(1 for c in text.lower() if c in "aeiouàèéìòù") / char_count,
            "consonant_ratio": sum(1 for c in text if c.isalpha() and c.lower() not in "aeiouàèéìòù") / char_count,
            "avg_word_length": sum(len(w) for w in words) / word_count if word_count > 0 else 0,
            "avg_sentence_length": word_count / sentence_count,
            "quote_ratio": (text.count('"') + text.count("'") + text.count("«") + text.count("»")) / char_count,
            "parenthesis_ratio": (text.count('(') + text.count(')')) / char_count,
            "comma_ratio": text.count(',') / char_count,
            "period_ratio": periods / char_count,
            "question_mark_ratio": questions / char_count,
            "exclamation_ratio": exclamations / char_count,
            "colon_ratio": text.count(':') / char_count,
            "semicolon_ratio": text.count(';') / char_count,
            "stopword_ratio": sum(1 for w in words if w.lower() in ITALIAN_STOPWORDS) / word_count if word_count > 0 else 0,
        }

        # 3. STRUTTURALI: Layout del documento e presenza di rumore (HTML, Email, URL)
        structural = {
            "line_count": line_count,
            "paragraph_count": len(paragraphs),
            "avg_line_length": char_count / line_count if line_count > 0 else 0,
            "avg_paragraph_length": char_count / len(paragraphs) if paragraphs else 0,
            "empty_line_ratio": sum(1 for l in lines if not l.strip()) / line_count if line_count > 0 else 0,
            "bullet_point_count": len(RE_BULLET.findall(text)),
            "bullet_point_ratio": len(RE_BULLET.findall(text)) / char_count,
            "url_count": len(RE_URL.findall(text)),
            "url_density": len(RE_URL.findall(text)) / word_count if word_count > 0 else 0,
            "email_count": len(RE_EMAIL.findall(text)),
            "email_density": len(RE_EMAIL.findall(text)) / word_count if word_count > 0 else 0,
            "html_tag_count": len(RE_HTML.findall(text)),
            "html_tag_ratio": len(RE_HTML.findall(text)) / char_count,
            "special_char_ratio": sum(1 for c in text if c in "@#$%^&*+=") / char_count,
        }

        # 4. ANOMALIA: Identificazione di potenziali testi generati, boilerplate o spam
        word_counts = Counter(w.lower() for w in words)
        unique_words = len(word_counts)
        anomaly = {
            "most_common_word_freq": word_counts.most_common(1)[0][1] if word_counts else 0,
            "repeated_word_count": sum(1 for c in word_counts.values() if c > 1),
            "repeated_word_ratio": sum(1 for c in word_counts.values() if c > 1) / word_count if word_count > 0 else 0,
            "repeated_char_count": len(RE_REPEATED_CHARS.findall(text)),
            "repeated_char_ratio": len(RE_REPEATED_CHARS.findall(text)) / char_count,
            "repeated_sequence_count": len(RE_REPEATED_SEQ.findall(text)),
            "text_entropy": self._calculate_entropy(text),
            "unique_word_count": unique_words,
            "unique_word_ratio": unique_words / word_count if word_count > 0 else 0,
            "all_caps_word_ratio": sum(1 for w in words if w.isupper() and len(w) > 1) / word_count if word_count > 0 else 0,
            "all_lowercase_word_ratio": sum(1 for w in words if w.islower()) / word_count if word_count > 0 else 0,
            "mixed_case_word_ratio": sum(1 for w in words if any(c.isupper() for c in w) and any(c.islower() for c in w)) / word_count if word_count > 0 else 0,
            "consecutive_spaces_count": len(RE_SPACES.findall(text)),
            "consecutive_punctuation_count": len(RE_PUNC_SEQ.findall(text)),
        }

        return {**base, **linguistic, **structural, **anomaly}
    
    def run(self, data, rank=0, world_size=1):
        """
        Esecuzione della pipeline in parallelo. 
        Implementa una scrittura streaming su CSV per mantenere l'occupazione di memoria costante (O(1)).
        """

        # Creazione di un file unico per worker per evitare race conditions in ambienti distributed
        temp_csv_name = f"rank_{rank}_{self.csv_filename}"
        
        # 2. Apriamo il file in modalità scrittura immediata
        # Usiamo self.output_folder.open per essere compatibili con DataTrove
        with self.output_folder.open(temp_csv_name, "wt") as f:
            writer = None
            
            for doc in data:
                with self.track_time():
                    # Estraiamo le 41 features
                    doc_features = self.extract_stats(doc)
                    
                    # Gestione della Language Identification (LID)
                    lang_score = doc.metadata.get("language_score")
                    if lang_score is None:
                        _, lang_score = self.lid_model.predict(doc)
                    
                    row = {
                        "doc_id": doc.id,
                        "label": doc.metadata.get("label", "unknown").lower(),
                        "language_score": lang_score,
                        **doc_features
                    }
                    
                    # Inizializziamo l'header solo al primo documento
                    if writer is None:
                        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
                        writer.writeheader()
                    
                    # Scrittura immediata su disco (flush implicito) per prevenire saturazione RAM
                    writer.writerow(row)
                    
                    # Propagazione delle feature nei metadati per eventuali step successivi della pipeline
                    doc.metadata.update(doc_features)
                    doc.metadata["language_score"] = lang_score
                
                yield doc
        
        logger.info(f"Worker {rank} ha finito di scrivere il suo file parziale.")

    def _save_to_csv(self):
        pass
           

    def _get_empty_stats(self) -> dict:
        # Metodo di fallback per doc vuoti (ritorna 0 per tutte le chiavi)
        keys = ["length", "white_space_ratio", "non_alpha_digit_ratio", "digit_ratio", "uppercase_ratio", "elipsis_ratio", "punctuation_ratio", "word_count", "sentence_count", "vocabulary_size", "lowercase_ratio", "vowel_ratio", "consonant_ratio", "avg_word_length", "avg_sentence_length", "quote_ratio", "parenthesis_ratio", "comma_ratio", "period_ratio", "question_mark_ratio", "exclamation_ratio", "colon_ratio", "semicolon_ratio", "stopword_ratio", "line_count", "paragraph_count", "avg_line_length", "avg_paragraph_length", "empty_line_ratio", "bullet_point_count", "bullet_point_ratio", "url_count", "url_density", "email_count", "email_density", "html_tag_count", "html_tag_ratio", "special_char_ratio", "most_common_word_freq", "repeated_word_count", "repeated_word_ratio", "repeated_char_count", "repeated_char_ratio", "repeated_sequence_count", "text_entropy", "unique_word_count", "unique_word_ratio", "all_caps_word_ratio", "all_lowercase_word_ratio", "mixed_case_word_ratio", "consecutive_spaces_count", "consecutive_punctuation_count"]
        return {k: 0 for k in keys}