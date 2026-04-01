from datatrove.executor import LocalPipelineExecutor
from config_loader import get_config
from pipeline_factory import build_italian_cleaning_pipeline
from utils.output_organizer import output_classification
from datatrove.utils.stats import PipelineStats

def main():
    """
    Main Entry Point: Carica config, costruisce la pipeline ed esegue.
    """
    # 1. Carica i percorsi dal file config selezionato
    cfg = get_config()
    
    # 2. Crea i blocchi (passando i percorsi corretti)
    pipeline_blocks = build_italian_cleaning_pipeline(
        data_dir=cfg["DATA_DIR"],
        output_dir=cfg["OUTPUT_DIR"],
        rejected_dir=cfg["REJECTED_DIR"],
        model_path=cfg["MODEL_PATH"]
    )
    
    # 3. Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline_blocks,
        tasks=1,
        workers=1
    )
    # 4. Avvio della pipeline
    print(f"\nPipeline avviata su: {cfg['DATA_DIR']}")
    executor.run()

    # 5. Recupero delle statistiche di interesse
    stats = PipelineStats(
        
    )

    # tempo totale in secondi (restituisce 0)
    # total_time = stats.total_time
    # print(total_time)

    # tempo totale in rappresentazione formattata
    # total_time_repr = stats.get_repr("ita-llm-pipeline")
    # total_std_dev = stats.total_std_dev
    
    # Ottiene una rappresentazione formattata (restituisce 0)
    # print(stats.get_repr("ita-llm-pipeline"))

    # 6. Analisi finale degli scarti (cartelle 1_..., 2_..., ecc.)
    print("\n--- 🔍 Analisi Risultati ---")
    output_classification(cfg["REJECTED_DIR"], cfg["OUTPUT_DIR"])
    print(f"\n✅ Operazione completata. Statistiche in: {cfg['FEATURE_DIR']}")

if __name__ == "__main__":
    main() 