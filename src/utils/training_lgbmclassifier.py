from ..blocks.classifiers import QualityClassifier
import os

"""
Avviare come modulo:
python3 -m src.utils.training_lgbmclassifier
"""
csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "output", "feature", "doc_stats_per_file.csv")


result = QualityClassifier.train_from_csv(csv_path)