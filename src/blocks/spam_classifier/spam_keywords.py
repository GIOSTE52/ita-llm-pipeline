from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Pattern, Sequence, Set


URGENCY_TERMS: Set[str] = {
    "urgente", "subito", "immediato", "scade", "scadenza", "oggi",
    "entro oggi", "ultime ore", "adesso", "ora", "ultimo avviso",
    "bloccato", "sospeso",
}

MONEY_TERMS: Set[str] = {
    "gratis", "gratuito", "offerta", "sconto", "bonus", "premio",
    "vincita", "guadagno", "cashback", "rimborso", "pagamento",
    "bonifico", "saldo", "fattura", "euro", "€",
}

CTA_TERMS: Set[str] = {
    "clicca qui", "clicca", "apri", "scopri di più", "conferma ora",
    "verifica ora", "registrati", "accedi", "rispondi", "compila",
    "scarica", "richiedi", "attiva",
}

ACCOUNT_TERMS: Set[str] = {
    "account", "profilo", "accesso", "password", "credenziali",
    "verifica account", "reimposta password", "login",
    "conferma identità",
}

SECURITY_TERMS: Set[str] = {
    "sicurezza", "alert", "allerta", "anomalo", "sospetto",
    "tentativo di accesso", "attività insolita", "blocco",
    "sospensione", "violazione",
}

DELIVERY_TERMS: Set[str] = {
    "spedizione", "consegna", "corriere", "pacco", "giacenza",
    "tracking", "ordine", "ritiro", "dogana", "indirizzo errato",
}

BRAND_TERMS: Set[str] = {
    "poste italiane", "poste", "inps", "agenzia entrate", "paypal",
    "amazon", "dhl", "gls", "ups", "fedex", "banca", "intesa",
    "unicredit", "nexi", "visa", "mastercard", "apple", "google",
    "microsoft", "enel", "tim",
}

UNSUBSCRIBE_TERMS: Set[str] = {
    "unsubscribe", "disiscriviti", "cancella iscrizione",
    "annulla iscrizione", "rimuovi", "opt-out",
}

PROMO_CODE_TERMS: Set[str] = {
    "codice", "coupon", "promo", "promocode", "voucher", "gift card",
}

SPAM_TERMS: Set[str] = set().union(
    URGENCY_TERMS,
    MONEY_TERMS,
    CTA_TERMS,
    ACCOUNT_TERMS,
    SECURITY_TERMS,
    DELIVERY_TERMS,
    BRAND_TERMS,
    UNSUBSCRIBE_TERMS,
    PROMO_CODE_TERMS,
)

SUSPICIOUS_TLDS: Set[str] = {
    ".xyz", ".top", ".click", ".live", ".shop", ".loan", ".buzz",
    ".win", ".cf", ".tk", ".ml", ".ga",
}

URL_SHORTENERS: Set[str] = {
    "bit.ly", "tinyurl.com", "t.co", "ow.ly", "cutt.ly",
    "rebrand.ly", "is.gd", "buff.ly",
}


WORD_RE: Pattern[str] = re.compile(r"\b[\wÀ-ÖØ-öø-ÿ'-]+\b", re.UNICODE)
URL_RE: Pattern[str] = re.compile(r"(?:https?://|www\.)[^\s<>()\[\]{}\"']+", re.IGNORECASE)
EMAIL_RE: Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
AMOUNT_RE: Pattern[str] = re.compile(
    r"(?:€\s?\d+[\d.,]*|\d+[\d.,]*\s?€|\b\d+[\d.,]*\s?(?:euro|eur)\b)",
    re.IGNORECASE,
)
PROMO_CODE_RE: Pattern[str] = re.compile(
    r"\b(?:codice|coupon|promo(?:code)?|voucher)\s*[:=-]?\s*[A-Z0-9_-]{4,}\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class KeywordBundle:
    spam_keywords: int
    urgency_keywords: int
    money_keywords: int
    cta_keywords: int
    account_keywords: int
    security_keywords: int
    delivery_keywords: int
    brand_keywords: int
    unsubscribe_keywords: int
    promo_code_keywords: int


def normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(stripped.split())

# conta parole o frasi target
def count_term_matches(normalized_text: str, terms: Iterable[str]) -> int:
    total = 0
    padded = f" {normalized_text} "
    for term in terms:
        candidate = f" {normalize_text(term)} "
        total += padded.count(candidate)
    return total

# restituisce un oggetto con i conteggi per ogni keyword
def keyword_bundle(text: str) -> KeywordBundle:
    normalized = normalize_text(text)
    return KeywordBundle(
        spam_keywords=count_term_matches(normalized, SPAM_TERMS),
        urgency_keywords=count_term_matches(normalized, URGENCY_TERMS),
        money_keywords=count_term_matches(normalized, MONEY_TERMS),
        cta_keywords=count_term_matches(normalized, CTA_TERMS),
        account_keywords=count_term_matches(normalized, ACCOUNT_TERMS),
        security_keywords=count_term_matches(normalized, SECURITY_TERMS),
        delivery_keywords=count_term_matches(normalized, DELIVERY_TERMS),
        brand_keywords=count_term_matches(normalized, BRAND_TERMS),
        unsubscribe_keywords=count_term_matches(normalized, UNSUBSCRIBE_TERMS),
        promo_code_keywords=count_term_matches(normalized, PROMO_CODE_TERMS),
    )

# conta il match delle regex
def regex_count(pattern: Pattern[str], text: str) -> int:
    return sum(1 for _ in pattern.finditer(text))

# estrae url
def extract_urls(text: str) -> Sequence[str]:
    return URL_RE.findall(text)

# estrae email
def extract_emails(text: str) -> Sequence[str]:
    return EMAIL_RE.findall(text)

# conto dei token (parole)
def token_count(text: str) -> int:
    return sum(1 for _ in WORD_RE.finditer(text))

# conto dei token unici
def unique_token_count(text: str) -> int:
    return len({m.group(0).lower() for m in WORD_RE.finditer(text)})

# conto dei tld sospetti
def count_suspicious_tlds(text: str) -> int:
    lowered = text.lower()
    return sum(lowered.count(tld) for tld in SUSPICIOUS_TLDS)

# conta url shortner
def count_shortener_urls(urls: Iterable[str]) -> int:
    total = 0
    for url in urls:
        lowered = url.lower()
        if any(shortener in lowered for shortener in URL_SHORTENERS):
            total += 1
    return total

# frasi in cui ci sono sia URL sia CTA (clicca, accedi, veierfica)
def count_cta_url_cooccurrence(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_text(chunk)
        if URL_RE.search(chunk) and count_term_matches(normalized, CTA_TERMS) > 0:
            total += 1
    return total

# frasi in cui ci sono sia URL sia brand
def count_brand_url_cooccurrence(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_text(chunk)
        if URL_RE.search(chunk) and count_term_matches(normalized, BRAND_TERMS) > 0:
            total += 1
    return total

# frasi in cui ci sono sia URL sia CTA (clicca, accedi, veierfica)
def quick_pattern_counts(text: str) -> Dict[str, int]:
    urls = extract_urls(text)
    return {
        "url_count": len(urls),
        "unique_url_count": len(set(u.lower() for u in urls)),
        "email_count": regex_count(EMAIL_RE, text),
        "amount_pattern_count": regex_count(AMOUNT_RE, text),
        "promo_code_pattern_count": regex_count(PROMO_CODE_RE, text),
        "suspicious_tld_count": count_suspicious_tlds(text),
        "shortener_url_count": count_shortener_urls(urls),
        "cta_plus_url_score": count_cta_url_cooccurrence(text),
        "brand_plus_link_score": count_brand_url_cooccurrence(text),
    }