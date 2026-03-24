from datatrove.executor import LocalPipelineExecutor
from config_loader import get_config
from pipeline_factory import build_italian_cleaning_pipeline
from utils.output_organizer import output_classification

def main():
    """
    Main Entry Point: Carica config, costruisce la pipeline ed esegue.
    """
    # 1. Carica i percorsi
    cfg = get_config()
    
    # 2. Crea i blocchi (passando i percorsi corretti)
    pipeline_blocks = build_italian_cleaning_pipeline(
        data_dir=cfg["DATA_DIR"],
        output_dir=cfg["OUTPUT_DIR"],
        rejected_dir=cfg["REJECTED_DIR"]
    )
    
    # 3. Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline_blocks,
        tasks=1,
        workers=1
    )
    executor.run()

    # 4. Lo script "usa e getta" per il debug
    print("\n--- 🔍 Analisi Etichette (Scarti) ---")
    output_classification(cfg["REJECTED_DIR"], cfg["OUTPUT_DIR"])

    print("\n--- 🔍 Analisi Accuratezza Filtri ---")
    output_classification(cfg["REJECTED_DIR"], cfg["OUTPUT_DIR"])

if __name__ == "__main__":
    main()