
import os
import json
import math
import argparse
from datetime import datetime
import time

import regex as re  # pip install regex

# DataTrove
from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import JsonlReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

# AGGIUNGI UNO STOPWORD KILLER, ossia stopwrds di altre lingue che aiutano a capire cosa non è italiano (soprattuto lingue simili come latino spagnolo portoghese francese)
# -----------------------------
# Config / Resources
# -----------------------------
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
    "io", "tu", "lui", "egli", "lui", "lei", "esso", "noi", "voi", "loro", "essi",
    "mi", "ti", "ci", "vi", "si", "li", "ne",
    "me", "te", "ce", "ve", "se", "glie",
    "questo", "questa", "questi", "queste",
    "quello", "quella", "quelli", "quelle",
    "che", "cui", "chi", "c'è", "non", "più", "anche", "come",
    "quali", "quale", "quanto", "quanta", "quanti", "quante",  
    "alcuni","alcune", "qualche","qualcosa","qualcuno","qualcuna",
    "nulla", "niente", "nessuno", "nessuna", 
    "tutto", "tutte", "tutti", "tutta",
    "altro", "altri", "altre", "altra",
    "stesso", "stessa", "stessi", "stesse",
    "non", "sì", "no", "più", "meno", "molto", "poco", "troppo", "così", "allora",
    "già", "ancora", "sempre", "spesso", "mai", "qui", "qua", "li", "là", 
    "essere", "sono", "sei", "è", "siamo", "siete", "era", "erano", "sia", "siano",
    "avere", "ha", "hanno", "hai", "ho", "avete", "abbiamo" , "po'", "com'", "c'", "d'",
    
}

COMMON_IT_BIGRAMS = {
    ("di", "cui"), ("che", "si"), ("non", "è"), ("non", "ha"), ("non", "sono"),
    ("si", "può"), ("si", "deve"), ("si", "fa"), ("in", "cui"), ("per", "il"),
    ("per", "la"), ("per", "un"), ("per", "una"), ("con", "il"), ("con", "la"),
    ("del", "tutto"), ("alla", "fine"), ("a", "cui"), ("da", "cui"), ("tra", "cui"),
    ("anche", "se"), ("in", "modo"), ("un", "po'"), ("po'", "di"), ("il", "fatto"),
    ("il", "modo"), ("la", "parte"), ("nel", "caso"), ("nel", "modo"), ("nel", "tempo"),
    ("alla", "luce"), ("in", "quanto"), ("per", "quanto"), ("dal", "punto"), ("punto", "di"), ("punto", "vista"),
}

# PAROLE PER EVITARE FALSI POSITIVI CON ALTRE LINGUE

ANTI_ES_WORDS = {
    "que", "los", "las", "del", "para", "por", "como", "pero",
    "también", "porque", "una", "uno", "unos", "unas"
}

ANTI_PT_WORDS = {
    "não", "voce", "você", "para", "por", "como", "porque",
    "também", "uma", "umas", "um", "uns"
}

ANTI_FR_WORDS = {
    "les", "des", "est", "une", "un", "pour", "pas", "que",
    "avec", "mais", "dans"
}


RE_HTML_TAG = re.compile(r"<[^>]+>") # toglie qualunque cosa tra < > per html
RE_URL = re.compile(r"https?://\S+|www\.\S+")
RE_EMAIL = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
RE_BASE64 = re.compile(r"\b[A-Za-z0-9+/]{120,}={0,2}\b")  # molto grezzo ma utile per encoded e dump
RE_WHITESPACE = re.compile(r"\s+") # spazi multipli
RE_NON_LETTER = re.compile(r"[^\p{L}]+", re.UNICODE) # non lettere
RE_WORD = re.compile(r"\p{L}+(?:['’]\p{L}+)?", re.UNICODE)

# “Code-ish” / dump
RE_CODE = re.compile(r"(?:\bfunction\b|\bclass\b|=>|{.*}|;\s*$|#include|\bimport\b|\bdef\b)", re.IGNORECASE | re.MULTILINE) #intercetta segnali ma rivedila è un po' aggressiva

# Ripetizioni tipo "aaaaaa" o "!!!!!"
RE_LONG_REPEAT = re.compile(r"(.)(?:\1){7,}")  # stesso char ripetuto >= 8


# Linee boilerplate tipiche crawl , se compaiono almeno due di questi termini li scarta come boilerplate, occhio a non filtrare anche cose che parlano di privacy/cookie
BOILERPLATE_HINTS = (
    "accetta", "impostazioni", "abbonati", "iscriviti", "newsletter", 
    "pubblicità", "advertising", "sponsored", "all rights reserved"
)
'''
# -----------------------------
# Optional Language ID backends
# -----------------------------
def try_load_fasttext_lid(model_path: str):
    """
    Prova a usare fasttext per language ID.
    Richiede: pip install fasttext
    Modello tipico: lid.176.bin (o lid.176.ftz)
    """
    if not model_path:
        return None
    try:
        import fasttext  # type: ignore
        if not os.path.exists(model_path):
            return None
        return fasttext.load_model(model_path)
    except Exception:
        return None


def lang_id_fasttext(text: str, ft_model, threshold: float = 0.70):
    """
    Ritorna (is_it, prob, label)
    """
    if ft_model is None:
        return None
    # fasttext vuole una singola riga
    t = text.replace("\n", " ")[:20000]
    labels, probs = ft_model.predict(t, k=1)
    label = labels[0].replace("__label__", "")
    prob = float(probs[0])
    return (label == "it" and prob >= threshold, prob, label)


def lang_id_langdetect(text: str):
    """
    Fallback su langdetect.
    Richiede: pip install langdetect
    """
    try:
        from langdetect import detect_langs  # type: ignore
        t = text.replace("\n", " ")
        # detect_langs ritorna [LangProbability]
        res = detect_langs(t[:5000])
        if not res:
            return None
        top = res[0]
        # top.lang, top.prob
        return (top.lang == "it" and float(top.prob) >= 0.70, float(top.prob), top.lang)
    except Exception:
        return None
'''

# -----------------------------
# Text cleaning & scoring
# -----------------------------
def basic_normalize(text: str) -> str:
    # rimuovi html grezzo, url/email, normalizza spazi
    t = RE_HTML_TAG.sub(" ", text)
    t = RE_URL.sub(" ", t)
    t = RE_EMAIL.sub(" ", t)
    t = t.replace("\u00a0", " ")
    t = RE_WHITESPACE.sub(" ", t).strip()
    t=normalize_apostrophes(t)
    return t


def char_stats(text: str):
    total = max(len(text), 1)
    letters = sum(1 for c in text if c.isalpha()) # lettere
    digits = sum(1 for c in text if c.isdigit())            # cifre
    spaces = sum(1 for c in text if c.isspace())            # spazi
    punct = total - letters - digits - spaces               # simboli, punteggiatura, etc ...
    return {
        "total": total,
        "letters": letters,
        "digits": digits,
        "spaces": spaces,
        "punct": punct,
        "letter_ratio": letters / total,
        "digit_ratio": digits / total,
        "punct_ratio": punct / total,
        "space_ratio": spaces / total,
    }

def normalize_apostrophes(s: str) -> str: # normalizzazione apostrofi
    return s.replace("’", "'")

def token_stats(text: str):
    # tokenizzazione Unicode per italiano + normalizzazione apostrofi
    t = normalize_apostrophes(text.lower())

    # Estrae token tipo: "l'acqua", "dell'aria", "perche", "sì"
    tokens = RE_WORD.findall(t)

    n = len(tokens)
    if n == 0:
        return {"n_tokens": 0, "stop_ratio": 0.0, "unique_ratio": 0.0}

    # Conta stopwords: gestisci anche la parte prima dell'apostrofo (l', dell', all'...)
    stop = 0
    for tok in tokens:
        if tok in ITALIAN_STOPWORDS:
            stop += 1
        else:
            # se contiene apostrofo, controlla anche il prefisso con apostrofo
            if "'" in tok:
                prefix = tok.split("'", 1)[0] + "'"
                if prefix in ITALIAN_STOPWORDS:
                    stop += 1

    unique_ratio = len(set(tokens)) / n
    return {
        "n_tokens": n,
        "stop_ratio": stop / n,
        "unique_ratio": unique_ratio,
    }

# per evitare falsi positivi di altre lingue
def anti_language_stats(text: str):
    t = normalize_apostrophes(text.lower())
    tokens = RE_WORD.findall(t)
    n = len(tokens)
    if n == 0:
        return {
            "anti_es_ratio": 0.0,
            "anti_pt_ratio": 0.0,
            "anti_fr_ratio": 0.0,
        }

    es = sum(1 for tok in tokens if tok in ANTI_ES_WORDS)
    pt = sum(1 for tok in tokens if tok in ANTI_PT_WORDS)
    fr = sum(1 for tok in tokens if tok in ANTI_FR_WORDS)

    return {
        "anti_es_ratio": es / n,
        "anti_pt_ratio": pt / n,
        "anti_fr_ratio": fr / n,
    }



def bigram_stats(text: str):
    t = normalize_apostrophes(text.lower())
    tokens = RE_WORD.findall(t)
    if len(tokens) < 2:
        return {"n_bigrams": 0, "bigram_ratio": 0.0}

    bigrams = list(zip(tokens, tokens[1:]))
    hits = sum(1 for bg in bigrams if bg in COMMON_IT_BIGRAMS)
    return {
        "n_bigrams": len(bigrams),
        "bigram_ratio": hits / len(bigrams),
    }


def italian_heuristic_score(text: str) -> float:
    """
    Score euristico [0..1] “quanto sembra italiano”.
    Include: token stats, bigrammi, ratio caratteri + penalità anti-lingua.
    """
    ts = token_stats(text)
    bs = bigram_stats(text)
    als = anti_language_stats(text)
    cs = char_stats(text)

    score = 0.0

    # 1) stopword ratio (forte)
    score += min(ts["stop_ratio"] * 3.0, 1.0) * 0.38

    # 2) bigrammi comuni italiani (molto discriminante)
    # con whitelist piccola, 1–4% è già “buono”
    score += min(bs["bigram_ratio"] / 0.04, 1.0) * 0.24

    # 3) letter ratio
    score += min(max(cs["letter_ratio"] - 0.45, 0.0) / 0.35, 1.0) * 0.20

    # 4) rumore simbolico
    noise_penalty = min(max(cs["punct_ratio"] - 0.25, 0.0) / 0.35, 1.0)
    score += (1.0 - noise_penalty) * 0.10

    # 5) varietà lessicale
    score += min(max(ts["unique_ratio"] - 0.25, 0.0) / 0.55, 1.0) * 0.08

    # --- Penalità anti-lingua (ES/PT/FR)
    # Evita falsi positivi: spagnolo/portoghese con stopwords simili.
    # Penalizza solo se stop_ratio IT non è alto (se è alto, probabilmente è italiano vero).
    if ts["stop_ratio"] < 0.10:
        anti = max(als["anti_es_ratio"], als["anti_pt_ratio"], als["anti_fr_ratio"])
        # 3% già sospetto su testi lunghi
        anti_pen = min(max(anti - 0.02, 0.0) / 0.08, 1.0)  # da 0.02 a 0.10
        score -= anti_pen * 0.25

    return max(0.0, min(score, 1.0))



 # aggiungi percentuale di righe ripetute, lunghezza media delle parole, ratio caratteri non ASCII, ratio di token unici  su dinestre (es: ripetere gli stessi token 15/20 volte è spam)
def detect_noise(text: str):
    """
    Ritorna (is_noise, reason)
    """

    # testo vuoto
    if not text or len(text.strip()) == 0:   
        return True, "empty"

    t = text.strip()

    # troppo corto 
    if len(t) < 3:
        return True, "too_short"

    # base64 / blob
    if RE_BASE64.search(t):
        return True, "base64_blob"

    # ripetizioni eccessive
    if RE_LONG_REPEAT.search(t):
        return True, "long_repetition"

    # codice / log / dump
    if RE_CODE.search(t):
        return True, "code_like"

    # boilerplate tipico
    low = t.lower()
    bp_hits = sum(1 for h in BOILERPLATE_HINTS if h in low)
    if bp_hits >= 2:
        return True, "boilerplate"

    # ratio caratteri
    cs = char_stats(t)
    if cs["letter_ratio"] < 0.55 and cs["punct_ratio"] > 0.25:
        return True, "symbol_heavy"

    if cs["digit_ratio"] > 0.35 and cs["letter_ratio"] < 0.45:
        return True, "digit_heavy"
    
    ts = token_stats(t)

    if ts["stop_ratio"] <0.1 and ts["stop_ratio"] > 0.4:
        return True, "stop_ratio sballato"
    
    return False, ""


def is_italian(text: str, ft_model=None):
    """
    Ritorna (is_it, confidence, backend)
    backend: fasttext:* | langdetect:* | heuristic | heuristic_gate:*
    """
    '''
    # 1) fasttext
    ft = lang_id_fasttext(text, ft_model) if ft_model is not None else None
    if ft is not None:
        ok, prob, label = ft
        return ok, prob, f"fasttext:{label}"

    # 2) langdetect
    ld = lang_id_langdetect(text)
    if ld is not None:
        ok, prob, label = ld
        return ok, prob, f"langdetect:{label}"
    '''
    # 1) euristica + gates
    score = italian_heuristic_score(text)
    ts = token_stats(text)
    bs = bigram_stats(text)
    als = anti_language_stats(text)

    # base decision
    if score < 0.60:
        return False, score, "heuristic"

    # gate anti ES/PT: se stop_ratio IT è basso e anti_language è alto → scarta
    if ts["stop_ratio"] < 0.10:
        if als["anti_es_ratio"] > 0.03:
            return False, score, "heuristic_gate:anti_es"
        if als["anti_pt_ratio"] > 0.03:
            return False, score, "heuristic_gate:anti_pt"
        if als["anti_fr_ratio"] > 0.03:
            return False, score, "heuristic_gate:anti_fr"

    # gate bigrammi: score ok ma zero bigrammi italiani -> spesso non IT “ben formattato”
    # (non farlo troppo duro: alcuni testi corti o tecnici hanno pochi bigrammi)
    if ts["n_tokens"] >= 80 and bs["bigram_ratio"] < 0.008 and score < 0.72:
        return False, score, "heuristic_gate:low_bigrams"

    return True, score, "heuristic"





# -----------------------------
# Filter factory (writes rejects)
# -----------------------------
def make_filter(reject_path: str, ft_model=None):
    os.makedirs(os.path.dirname(reject_path), exist_ok=True)
    rej_f = open(reject_path, "a", encoding="utf-8")

    def _filter(doc):
        # doc.text deve esistere
        raw = getattr(doc, "text", "") or ""
        cleaned = basic_normalize(raw)
        id=getattr(doc,"id",None)

        # noise detection
        is_noise, noise_reason = detect_noise(cleaned)
        if is_noise:
            rej_f.write(json.dumps({
                "id": id,
                "ts": datetime.utcnow().isoformat() + "Z",
                "reason": f"noise:{noise_reason}",
                "text_preview": cleaned[:400],
            }, ensure_ascii=False) + "\n")
            rej_f.flush()
            return False

        # italian detection
        ok_it, conf, backend = is_italian(cleaned, ft_model=ft_model)
        ts = token_stats(cleaned)
        bs = bigram_stats(cleaned)
        als = anti_language_stats(cleaned)
        


        if not ok_it:
            rej_f.write(json.dumps({
                "id": id,
                "ts": datetime.utcnow().isoformat() + "Z",
                "reason": f"non_it:{backend}",
                "confidence": conf,
                "text_preview": cleaned[:400],
                "stop_ratio": ts["stop_ratio"],
                "unique_ratio": ts["unique_ratio"],
                "n_tokens": ts["n_tokens"],
                "bigram_ratio": bs["bigram_ratio"],
                "anti_es_ratio": als["anti_es_ratio"],
                "anti_pt_ratio": als["anti_pt_ratio"],
                "anti_fr_ratio": als["anti_fr_ratio"],

            }, ensure_ascii=False) + "\n")
            rej_f.flush()
            return False

        # passa
        return True

    return _filter


# -----------------------------
# Main
# -----------------------------
def main():
    t0=time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="input/mixed", help="Cartella con file .jsonl")
    ap.add_argument("--output", default="output", help="Cartella output (kept)")
    ap.add_argument("--rejects", default="debug/rejected.jsonl", help="File jsonl degli scarti")
    ap.add_argument("--text-key", default="text", help="Chiave del testo nel json")
    ap.add_argument("--fasttext-model", default="", help="Percorso a lid.176.bin o lid.176.ftz (opzionale)")
    args = ap.parse_args()

    # ft_model = try_load_fasttext_lid(args.fasttext_model) if args.fasttext_model else None

    pipeline = [
        JsonlReader(
            data_folder=args.input,
            text_key=args.text_key
        ),
        LambdaFilter(make_filter(args.rejects, ft_model=ft_model)),
        JsonlWriter(
            output_folder=args.output
        )
    ]

    # tasks=1 per scrivere rejects senza race conditions
    executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1)
    executor.run()

    t1=time.perf_counter()
    print(f"[STATS] TEMPO TOTALE IMPIEGATO DAL PROGRAMMA 'run_senza_regex.py': {t1 - t0:.2f}secondi ")


if __name__ == "__main__":
    main()


