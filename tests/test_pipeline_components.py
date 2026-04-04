"""
Test dei componenti principali della pipeline usando dati reali in `data/train/`.
"""

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from blocks.spam_classifier.spam_stats import extract_spam_features, SpamFeatureCsvWriter
from blocks.stats import DocStatsCsv
from blocks.classifiers import QualityClassifier, DEFAULT_FEATURE_NAMES
from pipeline_factory import build_italian_cleaning_pipeline


class TestSpamFeatureExtraction:
    """Test per l'estrazione feature spam da documenti."""
    
    def test_extracts_expected_features_from_real_hand_labeled_doc(self, hand_labeled_docs):
        """Verifica che le feature spam vengano estratte correttamente."""
        doc = hand_labeled_docs[0]

        features = extract_spam_features(
            SimpleNamespace(
                id=doc["id"],
                text=doc["text"],
                metadata=doc.get("metadata", {}),
            )
        )

        assert features["doc_id"] == doc["id"]
        assert features["char_count"] > 0
        assert features["word_count"] > 0
        assert features["lang_is_ita"] == 1.0
        assert features["lang_score"] > 0.0

    def test_extracts_features_from_english_document(self):
        """Verifica che le feature vengano estratte anche per documenti non italiani."""
        features = extract_spam_features(
            SimpleNamespace(
                id="en_doc",
                text="This is an English document about spam detection.",
                metadata={},
            )
        )

        assert features["doc_id"] == "en_doc"
        assert features["char_count"] > 0
        assert features["word_count"] > 0
        assert features["lang_score"] >= 0.0


class TestSpamFeatureCsvWriter:
    """Test per il writer CSV delle feature spam."""
    
    def test_initializes_with_correct_output_folder(self, temp_dir):
        """Verifica che il writer sia inizializzato con il percorso corretto."""
        output_folder = temp_dir / "output" / "feature"
        
        writer = SpamFeatureCsvWriter(
            output_folder=str(output_folder),
            csv_filename="spam_features.csv"
        )
        
        assert writer.output_folder == str(output_folder)
        assert writer.csv_filename == "spam_features.csv"


class TestDocStatsCsv:
    """Test per l'estrazione statistiche documenti."""
    
    def test_initializes_with_correct_parameters(self, temp_dir):
        """Verifica che DocStatsCsv sia inizializzato correttamente."""
        output_folder = temp_dir / "output" / "feature"
        
        stats = DocStatsCsv(
            output_folder=str(output_folder),
            csv_filename="doc_stats.csv",
            groups_to_compute=["summary"]
        )
        
        # output_folder è un DataFolder object - accediamo al path
        assert hasattr(stats.output_folder, 'path')
        assert stats.csv_filename == "doc_stats.csv"
        assert stats.name == "🇮🇹 Italian Advanced Features CSV"


class TestQualityClassifier:
    """Test per il classificatore di qualità LightGBM."""
    
    def test_default_feature_names_exist(self):
        """Verifica che le feature default siano definite."""
        assert len(DEFAULT_FEATURE_NAMES) > 0
        assert "length" in DEFAULT_FEATURE_NAMES
        assert "word_count" in DEFAULT_FEATURE_NAMES
        assert "text_entropy" in DEFAULT_FEATURE_NAMES

    def test_quality_classifier_requires_valid_model_path(self, temp_dir):
        """Verifica che QualityClassifier sollevi errore con modello non trovato."""
        import pytest
        from pathlib import Path
        
        model_path = temp_dir / "nonexistent_model.joblib"
        
        with pytest.raises(FileNotFoundError):
            QualityClassifier(
                model_path=str(model_path),
                threshold=0.5
            )


class TestPipelineFactory:
    """Test per la costruzione della pipeline."""
    
    def test_build_pipeline_requires_valid_model_files(self, temp_dir):
        """Verifica che la pipeline-factory richieda modelli validi."""
        output_dir = temp_dir / "output"
        rejected_dir = temp_dir / "rejected"
        (temp_dir / "models").mkdir(exist_ok=True)

        # Senza modello, la pipeline factory solleva FileNotFoundError
        import pytest
        with pytest.raises(FileNotFoundError):
            build_italian_cleaning_pipeline(
                data_dir=str(temp_dir),
                output_dir=str(output_dir),
                rejected_dir=str(rejected_dir),
                model_path=str(temp_dir / "models")
            )
    
    def test_pipeline_factory_signature(self):
        """Verifica la firma della pipeline factory."""
        import inspect
        sig = inspect.signature(build_italian_cleaning_pipeline)
        
        params = list(sig.parameters.keys())
        assert "data_dir" in params
        assert "output_dir" in params
        assert "rejected_dir" in params
        assert "model_path" in params
