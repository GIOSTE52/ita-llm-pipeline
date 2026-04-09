import os
import argparse
import glob

def extract_args():
    """Configura i parametri da riga di comando."""
    in_docker = os.path.exists("/app/src")
    default_root = "/app" if in_docker else os.path.abspath(".")
    default_output = "/app/output" if in_docker else os.path.join(default_root, "output")

    # Calcolo dinamico dei workers di default
    cpus = os.cpu_count() or 1
    default_workers = max(1, cpus - 2)

    parser = argparse.ArgumentParser(description="ITA LLM Pipeline Configuration")
    parser.add_argument("--config", type=str, default=None, help="Path al file .conf")
    parser.add_argument("--root-dir", type=str, default=default_root)
    parser.add_argument("--output-dir", type=str, default=default_output)
    parser.add_argument("--rejected-dir", type=str, default=None, help="Path to rejected files")
    parser.add_argument("--tasks", type=int, default=None, help="Numero di task (se omesso, vengono contati i file .jsonl)")
    parser.add_argument("--workers", type=int, default=default_workers, help=f"Numero di workers (default rilevato per questo PC: {default_workers})")
    parser.add_argument("--csv-dir", type=str, default=None, help="Path to csv directory")
    parser.add_argument("--feature-dir", type=str, default=None, help="Path to feature stats")
    parser.add_argument("--model-path", type=str, default=None)
    return parser.parse_args()

def get_config():
    """
    Ritorna un dizionario con tutti i percorsi pronti all'uso.
    Combina Argomenti e Variabili di Ambiente.
    """
    args = extract_args()
    
    # 1. Base Directories
    ROOT_DIR = os.environ.get("ROOT_DIR", args.root_dir)
    OUTPUT_DIR = os.environ.get("OUTPUT_DIR", args.output_dir)
    DATA_DIR = os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data"))
    #modificare qui per cambiare la cartella del dataset
    INPUT_SUB_PATTERN = "dataset/*.jsonl"
    
    # --- LOGICA DINAMICA TASK ---
    full_search_path = os.path.join(DATA_DIR, INPUT_SUB_PATTERN)
    found_files = glob.glob(full_search_path)
    num_tasks = len(found_files)
    # ----------------------------

    
    
    # 2. Sottocartelle (Default calcolati)
# 2. Sottocartelle (Logica: Ambiente > Argomento > Default)
    config = {
        "DATA_DIR": DATA_DIR,
        "INPUT_SUB_PATTERN": INPUT_SUB_PATTERN,
        "OUTPUT_DIR": OUTPUT_DIR,
        "REJECTED_DIR": os.environ.get("REJECTED_DIR", args.rejected_dir or os.path.join(OUTPUT_DIR, "rejected")),
        "FEATURE_DIR": os.environ.get("FEATURE_DIR", args.feature_dir or os.path.join(OUTPUT_DIR, "feature")),
        # "CSV_DIR": os.environ.get("CSV_DIR", args.csv_dir or os.path.join(OUTPUT_DIR, "csv")),
        "MODEL_PATH": os.environ.get("MODEL_PATH", os.path.join(ROOT_DIR, "models")),
        "MAX_WORKERS": int(os.environ.get("MAX_WORKERS", args.workers)),
        "NUM_TASKS": num_tasks,
    }

    # 3. Creazione automatica cartelle (gestendo il file del modello)
# Creazione automatica cartelle
    for key, path in config.items():
        if key in ["MAX_WORKERS", "NUM_TASKS"]:
            continue
        os.makedirs(os.path.dirname(path) if key == "MODEL_PATH" else path, exist_ok=True)
            
    print(f"🚀 Pipeline: {config['MAX_WORKERS']} workers | {config['NUM_TASKS']} tasks.")
    # Verifica di sicurezza: il modello esiste?
    if not os.path.exists(config["MODEL_PATH"]):
        print(f"⚠️ [WARNING] Modello non trovato in: {config['MODEL_PATH']}")
        
    return config