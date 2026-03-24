import os
from blocks.readers import get_jsonl_reader
from blocks.writers import get_jsonl_writer
from blocks.filters import get_language_filter, CustomItalianFilter
from blocks.stats import DocStatsCsv

def build_italian_cleaning_pipeline(data_dir, output_dir, rejected_dir):
    """
    Costruisce la pipeline modulare assemblando i blocchetti pre-configurati.
    """
    return [
        # 1. Lettura
        get_jsonl_reader(data_dir),
        
        # 2. Filtro Lingua (Ora richiamato dal tuo modulo filters)
        get_language_filter(rejected_dir, threshold=0.65),

        # 3. Filtro Custom per Rumore Web
        CustomItalianFilter(
            output_folder=os.path.join(rejected_dir, "2_custom_filter"),
            filename="custom_rejected_${rank}.jsonl"
        ),

        # 4. Estrazione Statistiche (CSV)
        DocStatsCsv(
            output_folder=os.path.join(output_dir, "feature"),
            csv_filename="doc_stats_per_file.csv",
            groups_to_compute=["summary"]
        ),

        # 5. Scrittura Finale
        get_jsonl_writer(output_dir)
    ]