from __future__ import annotations

import csv
import os
from typing import Dict, List, Optional
import re

from datatrove.pipeline.base import PipelineStep
from datatrove.data import DocumentsPipeline

from .spam_keywords import (
    keyword_bundle,
    quick_pattern_counts,
    ITALIAN_STOPWORDS_MINI,
    ITALIAN_COMMON_WORDS,
    ACCENTED_CHARS,
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
#gestione language score
def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _tokenize_lang_words(text: str) -> list[str]:
    return re.findall(r"\b[\wàèéìòùÀÈÉÌÒÙ']+\b", text.lower(), flags=re.UNICODE)


def _sentence_chunks(text: str) -> list[str]:
    chunks = re.split(r"[.!?\n\r;:]+", text)
    return [c.strip() for c in chunks if c.strip()]


def compute_custom_lang_score(text: str, metadata: dict | None = None) -> float:
    metadata = metadata or {}
    text = _safe_text(text).strip()

    if not text:
        return 0.0

    tokens = _tokenize_lang_words(text)
    if not tokens:
        return 0.0

    total_chars = len(text)
    alpha_chars = sum(ch.isalpha() for ch in text)
    digit_chars = sum(ch.isdigit() for ch in text)
    punct_chars = sum((not ch.isalnum()) and (not ch.isspace()) for ch in text)
    accented_count = sum(ch.lower() in ACCENTED_CHARS for ch in text)

    token_count = len(tokens)
    long_tokens = sum(len(t) >= 4 for t in tokens)
    short_tokens = sum(len(t) <= 2 for t in tokens)
    avg_word_len = sum(len(t) for t in tokens) / token_count

    stopword_hits = sum(t in ITALIAN_STOPWORDS_MINI for t in tokens)
    common_word_hits = sum(t in ITALIAN_COMMON_WORDS for t in tokens)

    stopword_ratio = _safe_div(stopword_hits, token_count)
    common_word_ratio = _safe_div(common_word_hits, token_count)
    alpha_ratio = _safe_div(alpha_chars, total_chars)
    digit_ratio = _safe_div(digit_chars, total_chars)
    punct_ratio = _safe_div(punct_chars, total_chars)
    accented_ratio = _safe_div(accented_count, total_chars)
    long_token_ratio = _safe_div(long_tokens, token_count)
    short_token_ratio = _safe_div(short_tokens, token_count)

    chunks = _sentence_chunks(text)
    avg_chunk_len = sum(len(c) for c in chunks) / len(chunks) if chunks else len(text)

    raw_lang_score = _safe_float(metadata.get("language_score"), 0.0)
    raw_lang_score = _clip01(raw_lang_score)

    score = 0.0
    score += 0.22 * _clip01(alpha_ratio / 0.75)
    score += 0.22 * _clip01(stopword_ratio / 0.18)
    score += 0.12 * _clip01(common_word_ratio / 0.08)

    if 4.0 <= avg_word_len <= 8.5:
        avg_len_score = 1.0
    elif 3.0 <= avg_word_len < 4.0:
        avg_len_score = (avg_word_len - 3.0) / 1.0
    elif 8.5 < avg_word_len <= 11.0:
        avg_len_score = 1.0 - ((avg_word_len - 8.5) / 2.5)
    else:
        avg_len_score = 0.0
    score += 0.10 * _clip01(avg_len_score)

    score += 0.08 * _clip01(long_token_ratio / 0.45)

    if 20 <= avg_chunk_len <= 180:
        chunk_score = 1.0
    elif 10 <= avg_chunk_len < 20:
        chunk_score = (avg_chunk_len - 10) / 10
    elif 180 < avg_chunk_len <= 260:
        chunk_score = 1.0 - ((avg_chunk_len - 180) / 80)
    else:
        chunk_score = 0.0
    score += 0.08 * _clip01(chunk_score)

    score += 0.04 * _clip01(accented_ratio / 0.01)
    score += 0.06 * raw_lang_score

    penalty = 0.0
    penalty += 0.10 * _clip01(digit_ratio / 0.20)
    penalty += 0.08 * _clip01(punct_ratio / 0.22)
    penalty += 0.06 * _clip01(short_token_ratio / 0.45)

    final_score = score - penalty
    final_score = 0.15 + (0.75 * _clip01(final_score))

    return round(_clip01(final_score), 4)

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
    lang_score = compute_custom_lang_score(text,metadata)
    
    url_count = float(pat["url_count"])
    email_count = float(pat["email_count"])

     # --- NUOVE COMBO CHEAP ---
    # booleane leggere che spesso rendono più delle feature singole
    has_link_and_cta = 1.0 if (pat["url_count"] > 0 and pat["action_phrase_count"] > 0) else 0.0
    has_urgency_and_cta = 1.0 if (kw.urgency_keywords > 0 and pat["action_phrase_count"] > 0) else 0.0
    has_brand_and_link = 1.0 if (kw.brand_keywords > 0 and pat["url_count"] > 0) else 0.0
    has_money_and_cta = 1.0 if (kw.money_keywords > 0 and pat["action_phrase_count"] > 0) else 0.0
    has_account_and_security = 1.0 if (kw.account_keywords > 0 and kw.security_keywords > 0) else 0.0
    has_delivery_and_link = 1.0 if (kw.delivery_keywords > 0 and pat["url_count"] > 0) else 0.0

    # score aggregato cheap sulla pressione simbolica
    symbol_pressure_score = float(
        basic["exclamation_count"] +
        basic["currency_symbol_count"] +
        pat["promo_symbol_count"]
    )

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
        "unique_domain_count_text": float(pat["unique_domain_count"]),
        "email_count_text": email_count,
        "url_density": (url_count / word_count) if word_count else 0.0,
        "email_density": (email_count / word_count) if word_count else 0.0,
        "amount_pattern_count": float(pat["amount_pattern_count"]),
        "promo_code_pattern_count": float(pat["promo_code_pattern_count"]),
        "action_phrase_count": float(pat["action_phrase_count"]),
        "promo_symbol_count": float(pat["promo_symbol_count"]),
        "uppercase_token_count": float(pat["uppercase_token_count"]),
        "short_line_count": float(pat["short_line_count"]),
        "short_token_count": float(pat["short_token_count"]),
        "suspicious_tld_count": float(pat["suspicious_tld_count"]),
        "shortener_url_count": float(pat["shortener_url_count"]),
        "cta_plus_url_score": float(pat["cta_plus_url_score"]),
        "brand_plus_link_score": float(pat["brand_plus_link_score"]),
        "urgency_cta_url_combo": float(pat["urgency_cta_url_combo"]),
        "money_cta_combo": float(pat["money_cta_combo"]),
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

        # --- NUOVE COMBO BOOLEANE ---
        "has_link_and_cta": has_link_and_cta,
        "has_urgency_and_cta": has_urgency_and_cta,
        "has_brand_and_link": has_brand_and_link,
        "has_money_and_cta": has_money_and_cta,
        "has_account_and_security": has_account_and_security,
        "has_delivery_and_link": has_delivery_and_link,

        # --- score aggregato cheap ---
        "symbol_pressure_score": symbol_pressure_score,
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
    "unique_domain_count_text",
    "email_count_text",
    "url_density",
    "email_density",
    "amount_pattern_count",
    "promo_code_pattern_count",
    "short_line_count",
    "short_token_count",
    "suspicious_tld_count",
    "shortener_url_count",
    "cta_plus_url_score",
    "brand_plus_link_score",
    "ham_business_hits",
    "action_phrase_count",
    "promo_symbol_count",
    "uppercase_token_count",
    "digit_run_count",
    "has_link_and_cta",
    "has_urgency_and_cta",
    "has_brand_and_link",
    "has_money_and_cta",
    "has_account_and_security",
    "has_delivery_and_link",
    "symbol_pressure_score",
    "urgency_cta_url_combo",
    "money_cta_combo",
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