from datatrove.executor import LocalPipelineExecutor
from config_loader import get_config
from pipeline_factory import build_italian_cleaning_pipeline
from utils.output_organizer import output_classification
from datatrove.utils.stats import PipelineStats
from utils.csv_aggregator import aggregate_rank_csvs
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
        model_path=cfg["MODEL_PATH"],
    )
  
    # 3. Esecuzione
    executor = LocalPipelineExecutor(
        pipeline=pipeline_blocks,
        tasks=cfg["NUM_TASKS"],
        workers=cfg["MAX_WORKERS"]
    )
    # 4. Avvio della pipeline
    executor.run()

    # 5. Aggregazione csv spam e quality
    feature_dir = cfg["FEATURE_DIR"]


    csv_outputs = [
        ("doc_stats_per_file.csv", "quality"),
        ("spam_doc_features.csv", "spam"),
    ]


    for final_name, label in csv_outputs:
        aggregate_rank_csvs(
            feature_dir=feature_dir,
            final_name=final_name,
            label=label,
            remove_parts=True,
        )

    # --- FINE AGGREGAZIONE ---

    # 6. Recupero delle statistiche di interesse (PipelineStats)
    # stats = PipelineStats(executor.pipeline)
    # print("\n" + "=" * 60)
    # print("REPORT ESECUZIONE PIPELINE")
    # print("=" * 60)

    # for i, step_stats in enumerate(stats.stats):
    #     print(f"\n--- Step {i+1}: {step_stats.name} ---")
    #     print(f"   Tempo globale: {step_stats.time_stats.global_mean:.2f}s")
    #     if step_stats.stats:
    #         print(f"  Metriche:")
    #         for metric_name, metric_stats in step_stats.stats.items():
    #             print(f"     - {metric_name}: {metric_stats.total}")

    # 7. Analisi finale degli scarti
    print("\n--- Analisi Risultati ---")
    output_classification(cfg["REJECTED_DIR"], cfg["OUTPUT_DIR"])
    print(f"\nOperazione completata. Inspection in: {os.path.join(cfg['OUTPUT_DIR'], 'inspection')}")

if __name__ == "__main__":
    main()