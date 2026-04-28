import os
from blocks.readers import get_jsonl_reader
from blocks.writers import get_jsonl_writer
from blocks.filters import get_language_filter, CustomItalianFilter, ItalianClassification
from blocks.stats import DocStatsCsv

from blocks.spam_classifier.spam_classifier import SpamFilter
from blocks.spam_classifier.spam_stats import SpamFeatureExtractor, SpamFeatureCsvWriter

def build_italian_cleaning_pipeline(data_dir, output_dir, rejected_dir, pattern, model_path):
    """
    Costruisce la pipeline modulare assemblando i blocchetti pre-configurati.
    """
    return [
        # 1. Lettura
        get_jsonl_reader(data_dir,  pattern = pattern),
        
        # 2. Filtro Lingua (Ora richiamato dal tuo modulo filters)
        get_language_filter(rejected_dir, threshold=0.75, languages = "it"),

        # 3. Filtro Custom per Rumore Web
        # CustomItalianFilter(
        #     output_folder=os.path.join(rejected_dir, "2_custom_filter"),
        #     filename="custom_rejected_${rank}.jsonl"
        # ),

        # 4. Estrazione feature spam nei metadata
        SpamFeatureExtractor(),

        # 5. Scrittura CSV feature spam serve per addestrare il modello poi si può togliere
        SpamFeatureCsvWriter(
        # Usiamo os.path.join per essere sicuri che funzioni su ogni sistema
            output_folder=os.path.join(output_dir, "feature"), 
            csv_filename="spam_doc_features.csv"
        ),

        # 6. Filtro spam
        SpamFilter(
            model_path=os.path.join(model_path, "spam_lgbm.joblib"),
            rejected_dir=rejected_dir,
            threshold=0.5  se la levo la trashold viene decisa dal joblib
        ),

        
        
        # 6. Estrazione Statistiche (CSV)
        DocStatsCsv(
            output_folder=os.path.join(output_dir, "feature"),
            csv_filename="doc_stats_per_file.csv",
            groups_to_compute=["summary"],
            languages="it",
        ),


        # 7. Classificazione italiana con QualityClassifier
        ItalianClassification(
            model_path = os.path.join(model_path, "lgbm_quality_model.joblib"),
            rejected_dir = rejected_dir,
            output_folder = output_dir,
            threshold = 0.65
        ),

        # 7. Scrittura Finale
        get_jsonl_writer(output_dir)
    ]