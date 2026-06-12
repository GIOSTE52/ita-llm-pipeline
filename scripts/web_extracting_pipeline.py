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

        # HTML -> testo (estrattore del testo da codice HTML)
        Trafilatura(favour_precision=True),

        # filtri qualità stile fineweb.py
        LanguageFilter(
            exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/non_english"),
            languages=["it"],
            language_threshold=0.8,
        ),
        # Per ora commento questi filtri per constatare se sono necessari
        # GopherRepetitionFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/gopher_rep")),
        # GopherQualityFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/gopher_qual")),
        # C4QualityFilter(
        #     filter_no_terminal_punct=False,
        #     exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/c4")
        # ),
        FineWebQualityFilter(exclusion_writer=JsonlWriter(f"{OUT_BASE}/removed/fineweb_qual")),

        # datatrove permette di collegarla agli altri step della pipeline in modo automatico inserendone in input l'iteratore del blocco precedente
        # e così via
        take_n,     # limita a circa 1000 i testi estratti

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