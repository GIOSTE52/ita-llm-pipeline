import os
import argparse

def extract_args():
    """Configura i parametri da riga di comando."""
    in_docker = os.path.exists("/app/src")
    default_root = "/app" if in_docker else os.path.abspath(".")
    default_output = "/app/output" if in_docker else os.path.join(default_root, "output")
    
    parser = argparse.ArgumentParser(description="ITA LLM Pipeline Configuration")
    parser.add_argument("--config", type=str, default=None, help="Path al file .conf")
    parser.add_argument("--root-dir", type=str, default=default_root)
    parser.add_argument("--output-dir", type=str, default=default_output)
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
    
    # 2. Sottocartelle (Default calcolati)
    config = {
        "DATA_DIR": os.environ.get("DATA_DIR", os.path.join(ROOT_DIR, "data")),
        "OUTPUT_DIR": OUTPUT_DIR,
        "REJECTED_DIR": os.environ.get("REJECTED_DIR", os.path.join(OUTPUT_DIR, "rejected")),
        "FEATURE_DIR": os.environ.get("FEATURE_DIR", os.path.join(OUTPUT_DIR, "feature")),
    }

    # 3. Creazione automatica cartelle
    for path in config.values():
        os.makedirs(path, exist_ok=True)
        
    return config