import os
from datatrove.pipeline.readers import JsonlReader

def get_jsonl_reader(data_dir: str, pattern: str):
    """
    Inizializza il lettore per file JSONL.
    
    Legge i dati dalla cartella specificata cercando il pattern dei file.
    Predefinito: rp_normalized.jsonl (il tuo dataset da 11k).
    """
    return JsonlReader(
        data_folder=data_dir,
        glob_pattern=pattern
    )