from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

from .spam_keywords import (
    keyword_bundle,
    quick_pattern_counts,
    token_count,
    unique_token_count,
)

#controllo su non-stringhe o stringhe vuote
def _safe_text(value) -> str:
    return value if isinstance(value, str) else ""

# converte tutti i numeri in float
def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default

# normalizzazione della label spam
def _normalize_spam_label(raw_label: Optional[str]) -> str:
    value = _safe_text(raw_label).strip().lower()
    if value in {"ham", "not_spam", "non_spam", "legit"}:
        return "ham"
    if value in {"spam", "junk"}:
        return "spam"
    return ""


def _extract_spam_label(metadata: dict) -> str:
    for key in ("spam_label_gold", "spam_label", "spam_gold_label"):
        label = _normalize_spam_label(metadata.get(key))
        if label:
            return label
    return ""

# calcolo feature sui caratteri di 'text'
def _basic_char_stats(text: str) -> Dict[str, float]:
    char_count = len(text)
    if char_count == 0:
        return {
            "char_count": 0.0,
            "digit_count": 0.0,
            "digit_ratio": 0.0,
            "uppercase_ratio": 0.0,
            "punctuation_ratio": 0.0,
            "whitespace_ratio": 0.0,
            "exclamation_count": 0.0,
            "question_count": 0.0,
            "newline_count": 0.0,
            "currency_symbol_count": 0.0,
        }

    digit_count = 0
    upper_count = 0
    punct_count = 0
    space_count = 0
    exclamation_count = 0
    question_count = 0
    newline_count = 0
    currency_symbol_count = 0

    for c in text:
        if c.isdigit():
            digit_count += 1
        elif c.isupper():
            upper_count += 1
        elif c.isspace():
            space_count += 1
            if c == "\n":
                newline_count += 1
        elif not c.isalnum():
            punct_count += 1
            if c == "!":
                exclamation_count += 1
            elif c == "?":
                question_count += 1
            elif c in {"€", "$"}:
                currency_symbol_count += 1

    return {
        "char_count": float(char_count),
        "digit_count": float(digit_count),
        "digit_ratio": digit_count / char_count,
        "uppercase_ratio": upper_count / char_count,
        "punctuation_ratio": punct_count / char_count,
        "whitespace_ratio": space_count / char_count,
        "exclamation_count": float(exclamation_count),
        "question_count": float(question_count),
        "newline_count": float(newline_count),
        "currency_symbol_count": float(currency_symbol_count),
    }

# estrae le stats da 'doc'
def extract_spam_features(doc) -> Dict[str, float | str]:
    text = _safe_text(getattr(doc, "text", ""))
    metadata = getattr(doc, "metadata", {}) or {}

    basic = _basic_char_stats(text)
    text_tokens = text.split()
    word_count = token_count(text)
    unique_word_count = unique_token_count(text)
    avg_word_length = (
        sum(len(tok) for tok in text_tokens) / len(text_tokens) if text_tokens else 0.0
    )

    kw = keyword_bundle(text)
    pat = quick_pattern_counts(text)

    lang = _safe_text(metadata.get("language")).lower()
    lang_score = _safe_float(metadata.get("language_score"), 0.0)
    
    url_count = float(pat["url_count"])
    email_count = float(pat["email_count"])

    features = {
        "doc_id": _safe_text(getattr(doc, "id", "")) or _safe_text(metadata.get("id")),
        "target_label": _extract_spam_label(metadata),
        "spam_target_label": _extract_spam_label(metadata),
        "char_count": basic["char_count"],
        "word_count": float(word_count),
        "unique_word_count": float(unique_word_count),
        "unique_word_ratio": (unique_word_count / word_count) if word_count else 0.0,
        "avg_word_length": avg_word_length,
        "digit_count": basic["digit_count"],
        "digit_ratio": basic["digit_ratio"],
        "uppercase_ratio": basic["uppercase_ratio"],
        "punctuation_ratio": basic["punctuation_ratio"],
        "whitespace_ratio": basic["whitespace_ratio"],
        "exclamation_count": basic["exclamation_count"],
        "question_count": basic["question_count"],
        "newline_count": basic["newline_count"],
        "currency_symbol_count": basic["currency_symbol_count"],
        "url_count_text": url_count,
        "unique_url_count_text": float(pat["unique_url_count"]),
        "email_count_text": email_count,
        "url_density": (url_count / word_count) if word_count else 0.0,
        "email_density": (email_count / word_count) if word_count else 0.0,
        "amount_pattern_count": float(pat["amount_pattern_count"]),
        "promo_code_pattern_count": float(pat["promo_code_pattern_count"]),
        "suspicious_tld_count": float(pat["suspicious_tld_count"]),
        "shortener_url_count": float(pat["shortener_url_count"]),
        "cta_plus_url_score": float(pat["cta_plus_url_score"]),
        "brand_plus_link_score": float(pat["brand_plus_link_score"]),
        "spam_keyword_hits": float(kw.spam_keywords),
        "urgency_keyword_hits": float(kw.urgency_keywords),
        "money_keyword_hits": float(kw.money_keywords),
        "cta_keyword_hits": float(kw.cta_keywords),
        "account_keyword_hits": float(kw.account_keywords),
        "security_keyword_hits": float(kw.security_keywords),
        "delivery_keyword_hits": float(kw.delivery_keywords),
        "brand_keyword_hits": float(kw.brand_keywords),
        "unsubscribe_keyword_hits": float(kw.unsubscribe_keywords),
        "promo_keyword_hits": float(kw.promo_code_keywords),
        "lang_score": lang_score,
        "lang_is_ita": 1.0 if lang in {"ita", "it", "italian"} else 0.0,
    }
    return features


FEATURE_COLUMNS: List[str] = [
    "doc_id",
    "target_label",
    "spam_target_label",
    "char_count",
    "word_count",
    "unique_word_count",
    "unique_word_ratio",
    "avg_word_length",
    "digit_count",
    "digit_ratio",
    "uppercase_ratio",
    "punctuation_ratio",
    "whitespace_ratio",
    "exclamation_count",
    "question_count",
    "newline_count",
    "currency_symbol_count",
    "url_count_text",
    "unique_url_count_text",
    "email_count_text",
    "url_density",
    "email_density",
    "amount_pattern_count",
    "promo_code_pattern_count",
    "suspicious_tld_count",
    "shortener_url_count",
    "cta_plus_url_score",
    "brand_plus_link_score",
    "spam_keyword_hits",
    "urgency_keyword_hits",
    "money_keyword_hits",
    "cta_keyword_hits",
    "account_keyword_hits",
    "security_keyword_hits",
    "delivery_keyword_hits",
    "brand_keyword_hits",
    "unsubscribe_keyword_hits",
    "promo_keyword_hits",
    "lang_score",
    "lang_is_ita",
]

# è il blocco chiamato dalla pipeline
# estrae i documenti, calcola le features e le insersce nel csv
class SpamFeatureExtractor(PipelineStep):
    name = "Spam Feature Extractor"

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        for doc in data:
            feats = extract_spam_features(doc)
            if doc.metadata is None:
                doc.metadata = {}
            for k, v in feats.items():
                doc.metadata[k] = v
            yield doc

# prima i writer scrivevano in parallelo sullo stesso csv, 
#ora sono separati e poi vengono uniti alla fine

class SpamFeatureCsvWriter(PipelineStep):
    name = "Spam Feature CSV Writer"

    def __init__(self, output_folder: str, csv_filename: str = "spam_doc_features.csv"):
        super().__init__()
        self.output_folder = output_folder
        self.csv_filename = csv_filename

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1):
        os.makedirs(self.output_folder, exist_ok=True)
        rank_filename = f"rank_{rank}_{self.csv_filename}"
        csv_path = os.path.join(self.output_folder, rank_filename)

        file_exists = os.path.exists(csv_path)
        write_header = not file_exists or os.path.getsize(csv_path) == 0

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEATURE_COLUMNS)

            if write_header:
                writer.writeheader()

            for doc in data:
                metadata = getattr(doc, "metadata", {}) or {}
                row = {}

                for col in FEATURE_COLUMNS:
                    if col == "doc_id":
                        row[col] = (
                            metadata.get("doc_id")
                            or getattr(doc, "id", "")
                            or metadata.get("id", "")
                        )
                    else:
                        row[col] = metadata.get(col, "")

                writer.writerow(row)
                yield doc