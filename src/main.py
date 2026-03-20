import os
import argparse
# from datatrove.data import Document
from datatrove.pipeline.readers import JsonlReader
# from datatrove.pipeline.extractors import Trafilatura
from datatrove.pipeline.filters import (
    C4QualityFilter,
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    LanguageFilter,
    URLFilter,
)
from datatrove.pipeline.stats import DocStats, WordStats

#per le stats di ogni file
from blocks.stats import DocStatsCsv  

from datatrove.pipeline.writers import JsonlWriter
from datatrove.executor import LocalPipelineExecutor
from utils import output_organizer

#Come alternativa è possibile usare load_dotenv
def load_config(config_path:str)->None:
    """Carico il file config nel quale vengono dichiarate le variabili di environment interessate"""
    if not os.path.exists(config_path):
        print(f"Config non trovato: {config_path}")
        return
    
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            #Non mi interessano le righe vuote o le righe commentate
            if not line or line.startswith("#"):
                continue
            #Rimuovo eventuali "export"
            if line.startswith("export "):
                line = line[7:]
            #Faccio il parsing per recuperare chiave e valore
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('""').strip("'")
                os.environ[key] = value
    # print(f"Config caricato: {config_path}")

def extract_args() -> argparse.ArgumentParser:
    # Rilevo se siamo in Docker (/app/src esiste) per usare path corretti
    in_docker = os.path.exists("/app/src")
    default_root = "/app" if in_docker else os.path.expandvars("$HOME/ita-llm-pipeline")
    default_output = "/app/output" if in_docker else os.path.expandvars("$HOME/output_data")
    
    parser = argparse.ArgumentParser(description="ITA LLM Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Insert relative path to file config (e.g. configs/default.conf)"
    )
    parser.add_argument(
        "--root-dir",
        type=str,
        default=default_root,
        help="Insert absolute path to project root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=default_output,
        help="Insert absolute path to output data"
    )
    parser.add_argument(
        "--rejected-dir",
        type=str,
        default=None,
        help="Insert absolute path to directory that contains rejected file by filters (e.g. home/user/output_data/rejected)"
    )
    parser.add_argument(
        "--csv-dir",
        type=str,
        default=None,
        help="Insert absolute path to csv directory"
    )
    parser.add_argument(
        "--feature-dir",
        type=str,
        default=None,
        help="Insert absolute path to directory that contains feature stats from your data (e.g home/user/output_data/feature)"
    )

    return parser

def pipeline_design() -> None:
    # pipeline = [
    #     JsonlReader(
    #         data_folder = DATA_DIR,
    #         # glob_pattern="input.jsonl"
    #         glob_pattern=os.path.join("train", "gold.jsonl")
    #     ),

    #     # 1. Filtro Lingua
    #     LanguageFilter(
    #         languages="it",
    #         language_threshold=0.65,
    #         exclusion_writer=JsonlWriter(os.path.join(REJECTED_DIR, "1_language"),output_filename="non_italian_${rank}.jsonl",compression=None)
    #     ),

    #     # 2. Gopher Repetition Filter (PERFETTO per test_it_bad_repetition)
    #     # Riconosce sequenze ripetute di parole e caratteri.
    #     # GopherRepetitionFilter(
    #     #     exclusion_writer=JsonlWriter(os.path.join(REJECTED_DIR, "2_repetition"),output_filename="Repetition${rank}.jsonl",compression=None),
    #     # ),


    #     # 3. Gopher Quality 
    #     # GopherQualityFilter(
    #     #     min_doc_words=5, # Questo serve per i tuoi test brevi
    #     #     # Rimuoviamo max_symbol_to_word_ratio perché causa il TypeError
    #     #     exclusion_writer=JsonlWriter(
    #     #         os.path.join(REJECTED_DIR, "3_gopher_quality"),
    #     #         output_filename="Quality${rank}.jsonl",
    #     #         compression=None
    #     #     )
    #     # ),

    #     #4
    #     FineWebQualityFilter(
    #         exclusion_writer=JsonlWriter(
    #             output_folder=os.path.join(REJECTED_DIR, "4_fineweb"),    # In questo modo creo una cartella per visualizzare tutti i documents rifiutati durante lo step relativo al filtro FineWebQualityFilter
    #             # output_folder=REJECTED_DIR,
    #             output_filename="fineweb_rejected_${rank}.jsonl",
    #             compression=None
    #             ) if os.path.exists(REJECTED_DIR) else None
    #     ),

    #     #Stats modificate per avere le statistiche di ogni singolo file
    #     DocStatsCsv(
    #         output_folder=os.path.join(OUTPUT_DIR, "feature"),
    #         csv_filename="doc_stats_per_file.csv",
    #         groups_to_compute=["summary"]
    #     ),
    #     # DocStatsCsv(
    #     #     output_folder=os.path.join(OUTPUT_DIR, "feature"),
    #     #     csv_filename="doc_stats_per_file.csv"
    #     # ),

    #     # ItalianFeatureExtractor(),
    #     JsonlWriter(
    #         output_folder = OUTPUT_DIR,
    #         output_filename = "italiano_pulito_${rank}.jsonl", 
    #         compression = None #per lavorare levo un attimo la compressione cosi sono piu veloce ~silvio
    #     )
    # ]


  

    pipeline = [
        # 1. LETTURA
        JsonlReader(
            data_folder=DATA_DIR,
            glob_pattern=os.path.join("train", "rp_normalized.jsonl")
        ),

        # 2. FILTRO LINGUA
        # Produce file chiamati: non_italiano_00000.jsonl
        LanguageFilter(
            languages="it",
            language_threshold=0.65,
            exclusion_writer=JsonlWriter(
                output_folder=os.path.join(REJECTED_DIR, "1_language"),
                output_filename="non_italiano_${rank}.jsonl",
                compression=None
            )
        ),

        # 3. FINEWEB QUALITY FILTER
        # Produce file chiamati: fineweb_rejected_00000.jsonl
        # FineWebQualityFilter(
        #     exclusion_writer=JsonlWriter(
        #         output_folder=os.path.join(REJECTED_DIR, "2_fineweb"),
        #         output_filename="fineweb_rejected_${rank}.jsonl",
        #         compression=None
        #     )
        # ),

        # 4. ESTRATTORE DI FEATURE (Il CSV per il tuo amico)
        DocStatsCsv(
            output_folder=os.path.join(OUTPUT_DIR, "feature"),
            csv_filename="doc_stats_per_file.csv",
            groups_to_compute=["summary"]
        ),

        # 5. SCRITTURA FINALE (I sopravvissuti)
        # Produce file chiamati: italiano_pulito_00000.jsonl
        JsonlWriter(
            output_folder=OUTPUT_DIR,
            output_filename="italiano_pulito_${rank}.jsonl",
            compression=None
        )
    ]

    # Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        workers=1
    )
    executor.run()
    # Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline,
        tasks=1,
        workers=1
    )
    executor.run()


    #Eseguo la pipeline sopra definita
    executor = LocalPipelineExecutor(
        pipeline = pipeline,
        # default logging_dir (viene creata automaticamente la cartella logs con le sue subdirectory)
        # Per test semplici eseguo la pipeline attraverso un singolo worker che esegue un singolo task
        tasks = 1,
        workers = 1
        # Se si vuole impostare un parallelismo si può mettere workers=-1 e il limite verrò gestito dal sistema, ovviamente questa
        # operazione ha senso se si hanno più file e quindi anche multipli task su cui operare 
    )
    executor.run()
    return





def main() -> None:
    pipeline_design()



if __name__ == "__main__":
    # Gli argomenti e le costanti globali le definisco qui per avere il libero accesso da tutte le funzioni
    args = extract_args().parse_args()

    # Carico il file config se specificato tramite argomento
    if args.config:
        config_path = os.path.join(args.root_dir, args.config)
        load_config(config_path)
    # Leggo le variabili di environment (se impostate nel config), altrimenti uso i valori degli argomenti
    ROOT_DIR = os.environ.get("ROOT_DIR", args.root_dir)
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", args.output_dir)
    # Per REJECTED_DIR e CSV_DIR uso sottocartelle di OUTPUT_DIR come default
    REJECTED_DIR = os.environ.get("REJECTED_DIR", args.rejected_dir or os.path.join(OUTPUT_DIR, "rejected"))
    CSV_DIR = os.environ.get("CSV_DIR", args.csv_dir or os.path.join(OUTPUT_DIR, "csv"))
    FEATURE_DIR = os.environ.get("FEATURE_DIR", args.feature_dir or os.path.join(OUTPUT_DIR, "feature"))

    # Crea le cartelle output e rejected se non esistono
    if OUTPUT_DIR:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
    if REJECTED_DIR:
        os.makedirs(REJECTED_DIR, exist_ok=True)

    main()

    # Creo la cartella per contenere i file csv con la classificazione dei dati in output    
    if CSV_DIR:
        os.makedirs(CSV_DIR, exist_ok=True)

    # Eseguo lo script ausiliario per creare i file csv che classificano i testi risultanti dalla pipeline
    # Controllo che ci siano dei file .jsonl.gz in OUTPUT_DIR oppure in REJECTED_DIR
    has_output_files = any(f.endswith('.jsonl.gz') for f in os.listdir(OUTPUT_DIR)) if os.path.exists(OUTPUT_DIR) else False
    has_rejected_files = any(f.endswith('.jsonl.gz') for f in os.listdir(REJECTED_DIR)) if os.path.exists(REJECTED_DIR) else False
    
    if has_output_files or has_rejected_files:
        output_organizer.output_classification(OUTPUT_DIR, CSV_DIR, REJECTED_DIR)  # type: ignore
