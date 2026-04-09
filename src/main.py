from datatrove.executor import LocalPipelineExecutor
from config_loader import get_config
from pipeline_factory import build_italian_cleaning_pipeline
from utils.output_organizer import output_classification
from datatrove.utils.stats import PipelineStats
import os

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
        pattern=cfg["INPUT_SUB_PATTERN"],
        model_path=cfg["MODEL_PATH"]
    )
    
    # 3. Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline_blocks,
        tasks=cfg["NUM_TASKS"],
        workers=cfg["MAX_WORKERS"]
    )
    # 4. Avvio della pipeline
    executor.run()

    # 5. Recupero delle statistiche di interesse
    stats = PipelineStats(
        executor.pipeline
    )

    # Descrizione di alcune statistiche fornite da DataTrove a fronte del report di progetto
    # print("=" * 60)
    # print("REPORT ESECUZIONE PIPELINE")
    # print("=" * 60)

    # # Per ogni step
    # for i, step_stats in enumerate(stats.stats):
    #     print(f"\n--- Step {i+1}: {step_stats.name} ---")
    #     print(f"  ⏱️  Tempo globale: {step_stats.time_stats.global_mean:.2f}s")
    #     print(f"  📊 Min: {step_stats.time_stats.global_min:.2f}s, Max: {step_stats.time_stats.global_max:.2f}s")
    #     print(f"  📈 ±{step_stats.time_stats.global_std_dev:.2f}s (std dev)")
        
    #     # Metriche custom
    #     if step_stats.stats:
    #         print(f"  📋 Metriche:")
    #         for metric_name, metric_stats in step_stats.stats.items():
    #             print(f"     - {metric_name}:")
    #             print(f"       Total: {metric_stats.total}")
    #             print(f"       Mean: {metric_stats.mean:.2f}/{metric_stats.unit}")
    #             print(f"       Range: {metric_stats.min} - {metric_stats.max}")


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
    print(f"\n✅ Operazione completata. Inspection in: {os.path.join(cfg['OUTPUT_DIR'], "inspection")}")

if __name__ == "__main__":
    main() 