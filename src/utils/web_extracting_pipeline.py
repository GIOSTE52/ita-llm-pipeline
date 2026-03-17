from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers import WarcReader
from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import LanguageFilter, URLFilter
from datatrove.pipeline.writers.jsonl import JsonlWriter
import os

# opzionali (come in fineweb.py)
from datatrove.pipeline.filters import (
    GopherRepetitionFilter,
    GopherQualityFilter,
    C4QualityFilter,
    FineWebQualityFilter,
)

WARC_PATHS_FILE="data/warc_paths"
WARC_CANDIDATES=50

OUT_BASE = "my_web_dump_output" 
os.makedirs(OUT_BASE, exist_ok=True)

LOGS = f"{OUT_BASE}/logs"
OUT_JSONL = f"{OUT_BASE}/jsonl"

TARGET_N = 1000

def take_n(data, rank: int = 0, world_size: int = 1):
    """Ferma la pipeline dopo N documenti (per prototipi locali)."""
    c = 0
    for doc in data:
        yield doc
        c += 1
        if c >= TARGET_N:
            break


if __name__ == "__main__":
    pipeline = [
        WarcReader(
            data_folder="https://data.commoncrawl.org",
            paths_file=WARC_PATHS_FILE,
            limit=-1,
            shuffle_files=True,
        ),

        # (opzionale) salva gli scarti
        URLFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/url")),

        # HTML -> testo
        Trafilatura(favour_precision=True),

        # filtri qualità stile fineweb.py
        LanguageFilter(
            exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/non_english"),
            languages=["it"],
            language_threshold=0.8,
        ),
        GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/gopher_rep")),
        GopherQualityFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/gopher_qual")),
        C4QualityFilter(
            filter_no_terminal_punct=False,
            exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/c4")
        ),
        FineWebQualityFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/fineweb_qual")),

        # datatrove permette di collegarla agli altri step della pipeline in modo automatico inserendone in input l'iteratore del blocco precedente
        # e così via
        take_n,     # limita a ~1000 i testi estratti

        JsonlWriter(
            OUT_JSONL,
            output_filename="sample_${rank}.jsonl",  # evita collisioni tra task
            compression=None,  # così è JSONL “plain”
        ),
    ]

    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        logging_dir=LOGS,
    )

    executor.run()

#################################################
# Versione 2 (URL from HTTP request)
#################################################

# import hashlib
# import time
# from typing import Iterable, List, Dict, Any

# import requests

# from datatrove.data import Document
# from datatrove.executor.local import LocalPipelineExecutor
# from datatrove.pipeline.extractors import Trafilatura
# from datatrove.pipeline.filters import LanguageFilter
# from datatrove.pipeline.writers.jsonl import JsonlWriter

# URLS: List[str] = [
#     # metti qui una lista di URL (o caricala da file)
# ]

# TARGET_N = 1000

# def http_fetch_reader(urls: List[str], timeout_s: int = 20, sleep_s: float = 0.0):
#     """
#     Genera Document con HTML in doc.text e url in doc.metadata.
#     id deterministico da url (sha1).
#     """
#     session = requests.Session()
#     headers = {"User-Agent": "my-datatrove-pipeline/0.1"}

#     n = 0
#     for url in urls:
#         if n >= TARGET_N:
#             break
#         try:
#             r = session.get(url, timeout=timeout_s, headers=headers)
#             r.raise_for_status()
#             html = r.text

#             doc_id = hashlib.sha1(url.encode("utf-8")).hexdigest()
#             yield Document(
#                 id=doc_id,
#                 text=html,  # sarà convertito a testo da Trafilatura
#                 metadata={
#                     "url": url,
#                     "status_code": r.status_code,
#                     "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
#                     "content_type": r.headers.get("content-type"),
#                 },
#             )
#             n += 1
#         except Exception as e:
#             # in un prototipo: skip; in produzione: loggare / salvare gli errori
#             continue
#         finally:
#             if sleep_s:
#                 time.sleep(sleep_s)

# OUT_BASE = "data/http_sample"
# pipeline = [
#     http_fetch_reader(URLS),
#     Trafilatura(favour_precision=True),  # HTML -> testo
#     LanguageFilter(),                     # opzionale
#     JsonlWriter(OUT_BASE, output_filename="sample_${rank}.jsonl", compression=None),
# ]

# executor = LocalPipelineExecutor(pipeline=pipeline, tasks=1, logging_dir=f"{OUT_BASE}/logs")
# executor.run()
