import re
import csv
import string
import math
from collections import Counter
from typing import get_args
from datatrove.data import Document
from datatrove.io import DataFolderLike, get_datafolder
from datatrove.pipeline.stats.doc_stats import DocStats
from datatrove.pipeline.stats.config import DEFAULT_TOP_K_CONFIG, GROUP, TopKConfig
from datatrove.utils.text import PUNCTUATION
from loguru import logger


ELIPSIS = ["...", "…"]

# Stopwords italiano (base)
ITALIAN_STOPWORDS = {
    "il", "lo", "la", "i", "gli", "le", "un", "uno", "una", "e", "è", "di", "da", 
    "che", "per", "con", "a", "in", "su", "se", "non", "come", "ma", "o", "nel",
    "della", "dei", "delle", "degli", "al", "agli", "alla", "alle", "nel", "nella",
    "nei", "nelle", "dal", "dallo", "dalla", "dagli", "dalle", "tra", "fra", "cui"
}


class DocStatsCsv(DocStats):
    """
    Estrattore di features avanzate per classificazione documenti.
    Include 41 features + label per training di 3 classificatori LightGBM personalizzati.
    
    Salva un CSV con:
    - doc_id: identificativo documento
    - label: 'good' o 'bad' (basato su euristiche)
    - 41 features linguistiche, strutturali, anomalie
    
    Features per 3 tesi diverse:
    - PERSONA 1: Focus linguistico italiano
    - PERSONA 2: Focus qualità strutturale
    - PERSONA 3: Focus anomalie/spam detection
    """

    name = "🇮🇹 Italian Advanced Features CSV"

    def __init__(
        self,
        output_folder: DataFolderLike,
        csv_filename: str = "doc_stats_per_file.csv",
        groups_to_compute: list[GROUP] = list(get_args(GROUP)),
        histogram_round_digits: int = 3,
        top_k_config: TopKConfig = DEFAULT_TOP_K_CONFIG,
    ) -> None:
        super().__init__(output_folder, groups_to_compute, histogram_round_digits, top_k_config)
        self.csv_filename = csv_filename
        self.all_docs_stats = []
        self.elipsis_regex = re.compile("|".join([f"(?:{re.escape(elipsis)})" for elipsis in ELIPSIS]))
        self.punc_regex = re.compile("|".join([f"(?:{re.escape(punc)})" for punc in PUNCTUATION]))

    def extract_stats(self, doc: Document) -> dict:
        text = doc.text
        
        if len(text) == 0:
            logger.warning(f"Document {doc.id} is empty")
            return self._get_empty_stats()

        try:
            # === FEATURES BASE (da DocStats) ===
            base_stats = self._extract_base_stats(text)
            
            # === FEATURES LINGUISTICHE ===
            linguistic_stats = self._extract_linguistic_stats(text)
            
            # === FEATURES STRUTTURALI ===
            structural_stats = self._extract_structural_stats(text)
            
            # === FEATURES ANOMALIE/SPAM ===
            anomaly_stats = self._extract_anomaly_stats(text)
            
            # === LABEL (good/bad) basato su euristiche ===
            label = doc.metadata["label"]
            
            # Combina tutte le features
            all_stats = {
                **base_stats,
                **linguistic_stats,
                **structural_stats,
                **anomaly_stats,
                "label": label,
            }
            
            return all_stats
        
        except Exception as e:
            logger.error(f"Error extracting stats from {doc.id}: {e}")
            return self._get_empty_stats()

    def _extract_base_stats(self, text: str) -> dict:
        """Features base da DocStats"""
        return {
            "length": len(text),
            "white_space_ratio": sum([1 for c in text if c.isspace()]) / len(text),
            "non_alpha_digit_ratio": sum([1 for c in text if not c.isalpha() and not c.isdigit()]) / len(text),
            "digit_ratio": sum([1 for c in text if c.isdigit()]) / len(text),
            "uppercase_ratio": sum([1 for c in text if c.isupper()]) / len(text),
            "elipsis_ratio": sum(len(e) for e in self.elipsis_regex.findall(text)) / len(text),
            "punctuation_ratio": sum(len(p) for p in self.punc_regex.findall(text)) / len(text),
        }

    def _extract_linguistic_stats(self, text: str) -> dict:
        """Features linguistiche italiane - PERSONA 1"""
        words = text.split()
        word_count = len(words)
        
        # Conteggi di caratteri specifici
        lowercase_count = sum(1 for c in text if c.islower())
        vowels_count = sum(1 for c in text.lower() if c in "aeiouàèéìòù")
        consonants_count = sum(1 for c in text if c.isalpha() and c not in "aeiouàèéìòùAEIOUÀÈÉÌÒÙ")
        
        # Punteggiatura specifica
        quotes_count = text.count('"') + text.count("'") + text.count("`") + text.count("«") + text.count("»")
        parenthesis_count = text.count("(") + text.count(")")
        commas_count = text.count(",")
        periods_count = text.count(".")
        question_marks = text.count("?")
        exclamations = text.count("!")
        colons = text.count(":")
        semicolons = text.count(";")
        
        # Conteggio frasi (approssimativo)
        sentence_count = max(1, periods_count + question_marks + exclamations)
        
        # Lunghezze medie
        avg_word_length = sum(len(w) for w in words) / word_count if word_count > 0 else 0
        avg_sentence_length = word_count / sentence_count if sentence_count > 0 else 0
        
        # Stopwords italiani
        stopword_count = sum(1 for w in words if w.lower() in ITALIAN_STOPWORDS)
        
        return {
            # Conteggi
            "word_count": word_count,
            "sentence_count": sentence_count,
            "vocabulary_size": len(set(w.lower() for w in words)),
            
            # Rapporti di caratteri
            "lowercase_ratio": lowercase_count / len(text) if text else 0,
            "vowel_ratio": vowels_count / len(text) if text else 0,
            "consonant_ratio": consonants_count / len(text) if text else 0,
            
            # Medie
            "avg_word_length": avg_word_length,
            "avg_sentence_length": avg_sentence_length,
            
            # Punteggiatura specifica
            "quote_ratio": quotes_count / len(text) if text else 0,
            "parenthesis_ratio": parenthesis_count / len(text) if text else 0,
            "comma_ratio": commas_count / len(text) if text else 0,
            "period_ratio": periods_count / len(text) if text else 0,
            "question_mark_ratio": question_marks / len(text) if text else 0,
            "exclamation_ratio": exclamations / len(text) if text else 0,
            "colon_ratio": colons / len(text) if text else 0,
            "semicolon_ratio": semicolons / len(text) if text else 0,
            
            # Stopwords
            "stopword_ratio": stopword_count / word_count if word_count > 0 else 0,
        }

    def _extract_structural_stats(self, text: str) -> dict:
        """Features strutturali e di formattazione - PERSONA 2"""
        lines = text.split("\n")
        line_count = len(lines)
        
        # Paragrafi (linee vuote come separatori)
        paragraphs = [p for p in text.split("\n\n") if p.strip()]
        paragraph_count = len(paragraphs)
        
        # Linee vuote
        empty_lines = sum(1 for line in lines if line.strip() == "")
        
        # Bullet points (cerca -, *, •, ecc.)
        bullet_points = len(re.findall(r"^\s*[-*•]\s", text, re.MULTILINE))
        
        # URL e email
        urls = re.findall(r"https?://[^\s]+", text)
        emails = re.findall(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", text)
        
        # Tag HTML
        html_tags = len(re.findall(r"<[^>]+>", text))
        
        # Caratteri speciali
        special_chars = sum(1 for c in text if c in "!@#$%^&*()_+-=[]{}|;:',.<>?/`~")
        
        # Medie
        avg_line_length = sum(len(line) for line in lines) / line_count if line_count > 0 else 0
        avg_paragraph_length = sum(len(p) for p in paragraphs) / paragraph_count if paragraph_count > 0 else 0
        
        return {
            "line_count": line_count,
            "paragraph_count": paragraph_count,
            "avg_line_length": avg_line_length,
            "avg_paragraph_length": avg_paragraph_length,
            "empty_line_ratio": empty_lines / line_count if line_count > 0 else 0,
            "bullet_point_count": bullet_points,
            "bullet_point_ratio": bullet_points / len(text) if text else 0,
            "url_count": len(urls),
            "url_density": len(urls) / len(text.split()) if text.split() else 0,
            "email_count": len(emails),
            "email_density": len(emails) / len(text.split()) if text.split() else 0,
            "html_tag_count": html_tags,
            "html_tag_ratio": html_tags / len(text) if text else 0,
            "special_char_ratio": special_chars / len(text) if text else 0,
        }

    def _extract_anomaly_stats(self, text: str) -> dict:
        """Features per rilevare anomalie e spam - PERSONA 3"""
        words = text.split()
        word_count = len(words)
        
        # Parole ripetute
        word_counts = Counter(w.lower() for w in words)
        most_common_word_freq = word_counts.most_common(1)[0][1] if word_counts else 0
        repeated_words = sum(1 for count in word_counts.values() if count > 1)
        
        # Caratteri ripetuti consecutivi
        repeated_chars = len(re.findall(r"(.)\1{2,}", text))
        
        # Sequenze ripetute (es. "aaa bbb ccc")
        repeated_sequences = len(re.findall(r"(\w+\s+){2,}\1", text, re.IGNORECASE))
        
        # Entropia (disordine del testo)
        entropy = self._calculate_entropy(text)
        
        # Unicità di parole
        unique_words = len(word_counts)
        
        # Case mixing (maiuscole/minuscole/miste)
        all_uppercase = sum(1 for w in words if w.isupper() and len(w) > 1)
        all_lowercase = sum(1 for w in words if w.islower())
        mixed_case = sum(1 for w in words if any(c.isupper() for c in w) and any(c.islower() for c in w))
        
        # Spazi bianchi anomali
        consecutive_spaces = len(re.findall(r" {2,}", text))
        
        # Punteggiatura consecutiva
        consecutive_punctuation = len(re.findall(r"[.!?,;:]{2,}", text))
        
        return {
            "most_common_word_freq": most_common_word_freq,
            "repeated_word_count": repeated_words,
            "repeated_word_ratio": repeated_words / word_count if word_count > 0 else 0,
            "repeated_char_count": repeated_chars,
            "repeated_char_ratio": repeated_chars / len(text) if text else 0,
            "repeated_sequence_count": repeated_sequences,
            "text_entropy": entropy,
            "unique_word_count": unique_words,
            "unique_word_ratio": unique_words / word_count if word_count > 0 else 0,
            "all_caps_word_ratio": all_uppercase / word_count if word_count > 0 else 0,
            "all_lowercase_word_ratio": all_lowercase / word_count if word_count > 0 else 0,
            "mixed_case_word_ratio": mixed_case / word_count if word_count > 0 else 0,
            "consecutive_spaces_count": consecutive_spaces,
            "consecutive_punctuation_count": consecutive_punctuation,
        }

    # def _assign_label(self,doc: Document, text: str, base_stats: dict, linguistic_stats: dict, 
    #                   structural_stats: dict, anomaly_stats: dict) -> str:
    #     """
    #     Assegna un label 'good' o 'bad' basato su euristiche.
    #     Questa è una baseline - potete personalizzarla per ogni persona!
    #     """
    #     # Se il documento ha già una label nei metadata, usiamo quella!
    #     if "label" in doc.metadata:
    #         return doc.metadata["label"]
    #     score = 0  # Score da 0 a 100, >50 = good
        
    #     # === PENALITÀ (bad indicators) ===
        
    #     # Documento troppo corto
    #     if base_stats["length"] < 100:
    #         score -= 20
        
    #     # Documento molto corto
    #     if base_stats["length"] < 50:
    #         score -= 30
        
    #     # Alto rapporto di caratteri speciali/non-alfanumerici
    #     if base_stats["non_alpha_digit_ratio"] > 0.4:
    #         score -= 15
        
    #     # Entropya molto alta (testo disordinato/rumore)
    #     if anomaly_stats["text_entropy"] > 5.5:
    #         score -= 20
        
    #     # Troppe sequenze ripetute (spam indicator)
    #     if anomaly_stats["repeated_sequence_count"] > 3:
    #         score -= 25
        
    #     # Troppi spazi bianchi consecutivi
    #     if anomaly_stats["consecutive_spaces_count"] > 5:
    #         score -= 15
        
    #     # Punteggiatura eccessiva consecutiva
    #     if anomaly_stats["consecutive_punctuation_count"] > 10:
    #         score -= 15
        
    #     # Troppi caratteri ripetuti (spam)
    #     if anomaly_stats["repeated_char_ratio"] > 0.05:
    #         score -= 20
        
    #     # Testo tutto maiuscolo
    #     if base_stats["uppercase_ratio"] > 0.7:
    #         score -= 20
        
    #     # === BONUS (good indicators) ===
        
    #     # Documento di buona lunghezza
    #     if 500 < base_stats["length"] < 15000:
    #         score += 20
        
    #     # Numero di frasi ragionevole
    #     if 3 < linguistic_stats["sentence_count"] < 100:
    #         score += 10
        
    #     # Buona lunghezza media delle parole (segno di qualità)
    #     if 4 < linguistic_stats["avg_word_length"] < 8:
    #         score += 10
        
    #     # Buon rapporto di stopwords (italiano naturale)
    #     if 0.15 < linguistic_stats["stopword_ratio"] < 0.5:
    #         score += 15
        
    #     # Paragrafazione presente
    #     if structural_stats["paragraph_count"] > 1:
    #         score += 10
        
    #     # Uso di punteggiatura varia (non monotono)
    #     punc_list = [
    #         linguistic_stats["period_ratio"] > 0,
    #         linguistic_stats["comma_ratio"] > 0,
    #         linguistic_stats["question_mark_ratio"] > 0,
    #         linguistic_stats["exclamation_ratio"] > 0
    #     ]
    #     punc_variety = sum(punc_list)
    #     if punc_variety >= 2:
    #         score += 10
        
    #     # Buona diversità di vocabolario
    #     if linguistic_stats["vocabulary_size"] > linguistic_stats["word_count"] * 0.6:
    #         score += 10
        
    #     # Entropia media (testo coerente)
    #     if 4 < anomaly_stats["text_entropy"] <= 5.5:
    #         score += 10
        
    #     # Assegna il label
    #     return "good" if score > 50 else "bad"

    def _calculate_entropy(self, text: str) -> float:
        """Calcola l'entropia di Shannon del testo"""
        if not text:
            return 0.0
        
        char_counts = Counter(text)
        entropy = 0.0
        text_len = len(text)
        
        for count in char_counts.values():
            probability = count / text_len
            entropy -= probability * math.log2(probability) if probability > 0 else 0
        
        return entropy

    def _get_empty_stats(self) -> dict:
        """Ritorna un dizionario di 0 per documenti vuoti"""
        return {
            # Base
            "length": 0, "white_space_ratio": 0, "non_alpha_digit_ratio": 0, "digit_ratio": 0,
            "uppercase_ratio": 0, "elipsis_ratio": 0, "punctuation_ratio": 0,
            # Linguistiche
            "word_count": 0, "sentence_count": 0, "vocabulary_size": 0, "lowercase_ratio": 0,
            "vowel_ratio": 0, "consonant_ratio": 0, "avg_word_length": 0, "avg_sentence_length": 0,
            "quote_ratio": 0, "parenthesis_ratio": 0, "comma_ratio": 0, "period_ratio": 0,
            "question_mark_ratio": 0, "exclamation_ratio": 0, "colon_ratio": 0, "semicolon_ratio": 0,
            "stopword_ratio": 0,
            # Strutturali
            "line_count": 0, "paragraph_count": 0, "avg_line_length": 0, "avg_paragraph_length": 0,
            "empty_line_ratio": 0, "bullet_point_count": 0, "bullet_point_ratio": 0,
            "url_count": 0, "url_density": 0, "email_count": 0, "email_density": 0,
            "html_tag_count": 0, "html_tag_ratio": 0, "special_char_ratio": 0,
            # Anomalie
            "most_common_word_freq": 0, "repeated_word_count": 0, "repeated_word_ratio": 0,
            "repeated_char_count": 0, "repeated_char_ratio": 0, "repeated_sequence_count": 0,
            "text_entropy": 0, "unique_word_count": 0, "unique_word_ratio": 0,
            "all_caps_word_ratio": 0, "all_lowercase_word_ratio": 0, "mixed_case_word_ratio": 0,
            "consecutive_spaces_count": 0, "consecutive_punctuation_count": 0,
            # Metadata da filtri
            "language_score": 0.0,
            # Label
            "label": "bad",
        }

    def run(self, data, rank=0, world_size=1):
        """Override del metodo run per raccogliere e salvare stats in CSV"""
        self.all_docs_stats = []
        
        for doc in data:
            with self.track_time():
                try:
                    doc_stats = self.extract_stats(doc)
                except Exception as e:
                    logger.error(f"Error extracting stats from {doc.id}", exc_info=e)
                    raise e

                # Punteggio lingua calcolato da LanguageFilter (se presente in metadata)
                language_score = doc.metadata.get("language_score", 0.0)
                
                row = {
                    "doc_id": doc.id,
                    "language_score": language_score,
                    **doc_stats
                }
                self.all_docs_stats.append(row)
                doc.metadata.update(doc_stats)
            
            yield doc
        
        # Salva in CSV (solo dal rank 0 per evitare conflitti)
        if rank == 0:
            self._save_to_csv()

    def _save_to_csv(self):
        """Salva tutti i dati nel file CSV"""
        if not self.all_docs_stats:
            logger.warning("⚠️ Nessun documento statistiche da salvare")
            return
        
        fieldnames = list(self.all_docs_stats[0].keys())
        csv_path = f"{self.csv_filename}"
        
        try:
            with self.output_folder.open(csv_path, "wt", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(self.all_docs_stats)
            
            # Statistiche di distribuzione
            good_count = sum(1 for row in self.all_docs_stats if row.get("label") == "good")
            bad_count = sum(1 for row in self.all_docs_stats if row.get("label") == "bad")
            
            logger.info(f"✅ Statistiche salvate in CSV: {csv_path}")
            logger.info(f"📊 Totale documenti: {len(self.all_docs_stats)}")
            logger.info(f"✨ GOOD: {good_count} ({100*good_count/len(self.all_docs_stats):.1f}%)")
            logger.info(f"❌ BAD:  {bad_count} ({100*bad_count/len(self.all_docs_stats):.1f}%)")
            logger.info(f"📋 Colonne: {len(fieldnames)}")
            logger.info(f"Features: {len(fieldnames)-2}")
        
        except Exception as e:
            logger.error(f"❌ Errore nel salvataggio CSV: {e}")
            raise e