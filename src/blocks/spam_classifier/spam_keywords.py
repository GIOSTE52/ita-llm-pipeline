from __future__ import annotations


import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, Pattern, Sequence, Set


DELIVERY_BRANDS: Set[str] = {
    "sda", "bartolini", "brt", "dpd", "gls",
    "poste delivery", "fedex", "ups", "dhl",
    "poste italiane", "poste delivery", "amazon",
    "mondial relay", "parcel", "tracking number",
    "punto di ritiro", "indirizzo incompleto",
    "consegna fallita", "tentata consegna",
    "ritiro entro", "spese doganali", "oneri doganali",
}


# lessico banca/pagamenti/phishing economico
BANK_PAYMENT_TERMS: Set[str] = {
    "iban", "addebito", "pagamento non autorizzato",
    "transazione", "movimento", "verifica pagamento",
    "bonifico istantaneo", "conto corrente", "carta",
    "carta bloccata", "scadenza carta", "wallet",
    "saldo disponibile", "rimborso in sospeso",
}


# verifica identità / kyc / documenti
IDENTITY_VERIFICATION_TERMS: Set[str] = {
    "verifica identita", "documento di identita", "conferma documento",
    "selfie", "riconoscimento", "identificazione", "kyc",
    "aggiornamento dati", "conferma dati", "dati mancanti",
    "profilo incompleto",
}


# spam promo aggressivo / pressione commerciale
PROMO_PRESSURE_TERMS: Set[str] = {
    "solo per oggi", "posti limitati", "pezzi limitati",
    "offerta esclusiva", "prezzo speciale", "fino al",
    "risparmia", "approfitta ora", "ultimo giorno",
    "non perdere", "offerta riservata", "accesso immediato",
}


# ham business / amministrativo / formale
HAM_BUSINESS_TERMS: Set[str] = {
    "in allegato", "resto a disposizione", "cordiali saluti",
    "in riferimento a", "come concordato", "distinti saluti",
    "pratica", "documentazione", "fattura allegata",
    "gentile", "spettabile", "ufficio", "fornitore",
    "cliente", "riunione", "preventivo",
}


# verbi/frasi di azione cheap, senza NLP
ACTION_PHRASE_TERMS: Set[str] = {
    "clicca", "clicca qui", "verifica", "verifica ora",
    "conferma", "conferma ora", "accedi", "scarica",
    "attiva", "richiedi", "aggiorna", "scopri",
}




ITALIAN_STOPWORDS_MINI = {
    "il", "lo", "la", "i", "gli", "le",
    "un", "uno", "una",
    "di", "a", "da", "in", "con", "su", "per", "tra", "fra",
    "e", "o", "ma", "che", "chi", "cui",
    "mi", "ti", "si", "ci", "vi",
    "sono", "sei", "è", "siamo", "siete",
    "ho", "hai", "ha", "abbiamo", "avete",
    "del", "della", "dello", "dei", "degli", "delle",
    "al", "allo", "alla", "ai", "agli", "alle",
    "dal", "dallo", "dalla", "dai", "dagli", "dalle",
    "nel", "nello", "nella", "nei", "negli", "nelle",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "non", "più", "come", "dove", "quando", "anche",
    "grazie", "gentile", "salve", "buongiorno",
}


ITALIAN_COMMON_WORDS = {
    "pagamento", "ordine", "consegna", "fattura", "cliente", "servizio",
    "offerta", "spedizione", "verifica", "accesso", "conto", "account",
    "documento", "messaggio", "richiesta", "conferma", "sicurezza",
    "codice", "clicca", "aggiorna", "attiva", "confermare", "ricevuto",
    "numero", "telefono", "indirizzo", "ufficio", "azienda", "supporto",
}


PROMO_SYMBOLS = {"%", "€", "$", "!"}
ACCENTED_CHARS = set("àèéìòù")


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
    "clicca", "clicca qui", "clicca sul link", "premi qui",
    "accedi", "accedi ora", "verifica", "verifica ora",  "verifica subito",
    "conferma", "conferma ora", "conferma subito", "aggiorna", 
    "aggiorna ora", "scarica", "scarica ora", "attiva",
    "attiva ora", "richiedi", "richiedi ora", "completa",
    "completa ora", "procedi", "procedi ora", "continua",
    "continua qui", "visualizza", "visualizza qui", "consulta",
    "scopri", "scopri ora", 
}


ACCOUNT_TERMS: Set[str] = {
    "account", "profilo", "accesso", "password", "credenziali",
    "verifica account", "reimposta password", "login",
    "conferma identità", "utenza", "conto", "conto corrente",
    "carta", "wallet", "profilo incompleto", "dati account", 
}


SECURITY_TERMS: Set[str] = {
    "sicurezza", "alert", "allerta", "anomalo", "sospetto",
    "tentativo di accesso", "attività insolita", "blocco",
    "sospensione", "violazione", "accesso non autorizzato",
    "attivita sospetta", "verifica di sicurezza",
    "account compromesso", "anomalia", "blocco preventivo",
    "misura di sicurezza", "conferma identita", "protezione account",
}


DELIVERY_TERMS: Set[str] = {
    "consegna", "spedizione", "pacco", "corriere",
    "tracking", "tracciamento", "tentata consegna",
    "consegna fallita", "indirizzo errato", "indirizzo incompleto",
    "ritiro", "ritiro entro", "punto di ritiro",
    "giacenza", "fermo deposito", "spese di consegna",
    "spese doganali", "oneri doganali", "spedizione bloccata",
    "ordine in arrivo", "ordine sospeso", 

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
    DELIVERY_BRANDS,
    BANK_PAYMENT_TERMS,
    IDENTITY_VERIFICATION_TERMS,
    PROMO_PRESSURE_TERMS,
    UNSUBSCRIBE_TERMS,
    PROMO_CODE_TERMS,
    ACTION_PHRASE_TERMS,
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
# - https:/, www., domini nudi tipo .win o name.com/path
URL_RE: Pattern[str] = re.compile(r"""
    (?:
        \bhttps?:/{1,2}[^\s<>()\[\]{}"']+ |
        \bwww\.[^\s<>()\[\]{}"']+ |
        \b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+
        (?:[a-z]{2,})(?:/[^\s<>()\[\]{}"']*)?
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)
EMAIL_RE: Pattern[str] = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
AMOUNT_RE: Pattern[str] = re.compile(
    r"(?:€\s?\d+[\d.,]*|\d+[\d.,]*\s?€|\b\d+[\d.,]*\s?(?:euro|eur)\b)",
    re.IGNORECASE,
)


PROMO_CODE_RE: Pattern[str] = re.compile(
    r"(?i)\b(?:codice(?:\s+(?:promo|sconto|offerta))?|coupon|promo(?:code)?|voucher)\b\s*[:=-]?\s*(?-i:[A-Z0-9][A-Z0-9_-]{3,})\b"
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


# normalizza i testi
def normalize_text(text: str) -> str:
    if not text:
        return ""
    lowered = text.lower()
    normalized = unicodedata.normalize("NFKD", lowered)
    stripped = "".join(ch for ch in normalized if not unicodedata.combining(ch))
    return " ".join(stripped.split())


# normalizza url
def normalize_url(url: str) -> str:
    if not url:
        return ""
    cleaned = url.strip().strip(".,;:!?()[]{}<>\"'")
    cleaned = re.sub(r"^https:/([^/])", r"https://\1", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^http:/([^/])", r"http://\1", cleaned, flags=re.IGNORECASE)
    return cleaned




# normalizza casi di termini attaccati a segni
def normalize_for_matching(text: str) -> str:
    normalized = normalize_text(text)
    normalized = normalized.replace("'", " ")
    normalized = re.sub(r"[^0-9a-zA-ZÀ-ÖØ-öø-ÿ]+", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip()




# --- FIX IMPORTANTE ---
# usa la stessa normalizzazione sia per il testo sia per i termini,
# così non perdi match con punteggiatura tipo "INPS:" o "clicca!"
def count_term_matches(normalized_text: str, terms: Iterable[str]) -> int:
    total = 0
    padded = f" {normalize_for_matching(normalized_text)} "
    for term in terms:
        candidate = f" {normalize_for_matching(term)} "
        total += padded.count(candidate)
    return total


# restituisce un oggetto con i conteggi per ogni keyword
def keyword_bundle(text: str) -> KeywordBundle:
    normalized = normalize_for_matching(text)
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


# estrae il dominio host da una URL in modo leggero
def extract_domain(url: str) -> str:
    cleaned = normalize_url(url).lower()
    cleaned = re.sub(r"^https?://", "", cleaned)
    cleaned = re.sub(r"^www\.", "", cleaned)
    return cleaned.split("/", 1)[0].split(":", 1)[0]


# token tutti maiuscoli e abbastanza lunghi da essere informativi
def count_uppercase_tokens(text: str) -> int:
    total = 0
    for m in WORD_RE.finditer(text):
        tok = m.group(0)
        letters = [c for c in tok if c.isalpha()]
        if len(letters) >= 2 and tok.upper() == tok and tok.lower() != tok:
            total += 1
    return total


# righe molto corte: tipiche di CTA, blocchi promozionali o testo frammentato
def count_short_lines(text: str, max_len: int = 40) -> int:
    return sum(1 for line in text.splitlines() if line.strip() and len(line.strip()) <= max_len)


# token molto corti spesso usati in testi spezzati o rumorosi
# ignora parole di una sola lettera per non penalizzare troppo l'italiano
def count_short_tokens(text: str, max_len: int = 2) -> int:
    total = 0
    for m in WORD_RE.finditer(text):
        tok = m.group(0)
        if tok.isalnum() and 2 <= len(tok) <= max_len:
            total += 1
    return total


ACTION_PATTERN_RE: Pattern[str] = re.compile(
    r"\b(?:clicca(?:\s+qui)?|verifica|conferma|accedi|sblocca|aggiorna|attiva|firma\s+online|richiedi)\b",
    re.IGNORECASE,
)


PROMO_SYMBOL_RE: Pattern[str] = re.compile(r"[%€$!]")


# frasi in cui ci sono sia URL sia CTA (clicca, accedi, veierfica)
def count_cta_url_cooccurrence(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_for_matching(chunk)
        if URL_RE.search(chunk) and count_term_matches(normalized, CTA_TERMS) > 0:
            total += 1
    return total


# frasi in cui ci sono sia URL sia brand
def count_brand_url_cooccurrence(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_for_matching(chunk)
        if URL_RE.search(chunk) and count_term_matches(normalized, BRAND_TERMS) > 0:
            total += 1
    return total


# combo leggere ma molto mirate al caso spam
def count_urgency_cta_url_combo(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_for_matching(chunk)
        if (
            URL_RE.search(chunk)
            and count_term_matches(normalized, URGENCY_TERMS) > 0
            and count_term_matches(normalized, CTA_TERMS) > 0
        ):
            total += 1
    return total


def count_money_cta_combo(text: str) -> int:
    total = 0
    for chunk in re.split(r"[\n.!?]+", text):
        if not chunk.strip():
            continue
        normalized = normalize_for_matching(chunk)
        has_money = count_term_matches(normalized, MONEY_TERMS) > 0 or regex_count(AMOUNT_RE, chunk) > 0
        has_cta = count_term_matches(normalized, CTA_TERMS) > 0
        if has_money and has_cta:
            total += 1
    return total


# conta i match del lessico business/ham
def count_ham_business_terms(text: str) -> int:
    normalized = normalize_for_matching(text)
    return count_term_matches(normalized, HAM_BUSINESS_TERMS)




# conta i match di frasi/verbi di azione
def count_action_phrases(text: str) -> int:
    normalized = normalize_for_matching(text)
    return count_term_matches(normalized, ACTION_PHRASE_TERMS)




# conta simboli promo/pressione come %, €, $, !
def count_promo_symbols(text: str) -> int:
    return sum(text.count(sym) for sym in PROMO_SYMBOLS)




# conta token completamente in maiuscolo (almeno 2 caratteri)
def count_uppercase_tokens(text: str) -> int:
    total = 0
    for m in WORD_RE.finditer(text):
        token = m.group(0)
        if len(token) >= 2 and token.isupper():
            total += 1
    return total




# conta sequenze numeriche lunghe tipo tracking, codici, riferimenti
def count_digit_runs(text: str) -> int:
    return len(re.findall(r"\b\d{4,}\b", text))


# pattern veloci e poco costosi per la parte spam
def quick_pattern_counts(text: str) -> Dict[str, int]:
    urls = extract_urls(text)
    domains = [extract_domain(u) for u in urls if extract_domain(u)]
    return {
        "url_count": len(urls),
        "unique_url_count": len(set(u.lower() for u in urls)),
        "unique_domain_count": len(set(domains)),
        "email_count": regex_count(EMAIL_RE, text),
        "amount_pattern_count": regex_count(AMOUNT_RE, text),
        "promo_code_pattern_count": regex_count(PROMO_CODE_RE, text),
        "short_line_count": count_short_lines(text),
        "short_token_count": count_short_tokens(text),
        "suspicious_tld_count": count_suspicious_tlds(text),
        "shortener_url_count": count_shortener_urls(urls),
        "cta_plus_url_score": count_cta_url_cooccurrence(text),
        "brand_plus_link_score": count_brand_url_cooccurrence(text),
        "urgency_cta_url_combo": count_urgency_cta_url_combo(text),
        "money_cta_combo": count_money_cta_combo(text),
         "ham_business_hits": count_ham_business_terms(text),
        "action_phrase_count": count_action_phrases(text),
        "promo_symbol_count": count_promo_symbols(text),
        "uppercase_token_count": count_uppercase_tokens(text),
        "digit_run_count": count_digit_runs(text),
    }





