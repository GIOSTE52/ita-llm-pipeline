import os
from datatrove.pipeline.writers import JsonlWriter

def get_jsonl_writer(output_dir: str, filename: str = "italiano_pulito_${rank}.jsonl"):
    """
    Inizializza il modulo di scrittura finale.
    
    Salva i documenti che hanno superato tutti i filtri in formato JSONL.
    La compressione è disattivata per facilitare l'ispezione manuale dei dati.
    """
    return JsonlWriter(
        output_folder=output_dir,
        output_filename=filename,
        compression=None
    )