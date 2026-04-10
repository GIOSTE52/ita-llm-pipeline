"""
Test dei componenti principali della pipeline.
"""

from types import SimpleNamespace

import pytest

import pipeline_factory
from blocks.classifiers import DEFAULT_FEATURE_NAMES, QualityClassifier
from blocks.filters import get_language_filter
from blocks.readers import get_jsonl_reader
from blocks.spam_classifier.spam_stats import (
    SpamFeatureCsvWriter,
    extract_spam_features,
)
from blocks.stats import DocStatsCsv
from blocks.writers import get_jsonl_writer


class TestSpamFeatureExtraction:
    def test_extracts_expected_features_from_real_hand_labeled_doc(self, hand_labeled_docs):
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
        assert features["lang_score"] >= 0.0

    def test_extracts_enriched_spam_features_and_normalized_label(self):
        text = (
            "ATTENZIONE! Usa il codice SCONTO50 ora. "
            "Scrivi a promo@example.com oppure visita https://bit.ly/offerta."
        )
        features = extract_spam_features(
            SimpleNamespace(
                id="spam_doc",
                text=text,
                metadata={
                    "spam_label_gold": "JUNK",
                    "language": "it",
                    "language_score": "0.91",
                },
            )
        )

        assert features["doc_id"] == "spam_doc"
        assert features["target_label"] == "spam"
        assert features["spam_target_label"] == "spam"
        assert features["url_count_text"] == 2.0
        assert features["unique_url_count_text"] == 2.0
        assert features["email_count_text"] == 1.0
        assert features["shortener_url_count"] == 1.0
        assert features["promo_code_pattern_count"] >= 1.0
        assert features["cta_keyword_hits"] >= 1.0
        assert features["lang_is_ita"] == 1.0


class TestSpamFeatureCsvWriter:
    def test_initializes_with_correct_output_folder(self, temp_dir):
        output_folder = temp_dir / "output" / "feature"

        writer = SpamFeatureCsvWriter(
            output_folder=str(output_folder),
            csv_filename="spam_features.csv",
        )

        assert writer.output_folder == str(output_folder)
        assert writer.csv_filename == "spam_features.csv"


class TestDocStatsCsv:
    def test_initializes_with_correct_parameters(self, temp_dir):
        output_folder = temp_dir / "output" / "feature"

        stats = DocStatsCsv(
            output_folder=str(output_folder),
            csv_filename="doc_stats.csv",
            groups_to_compute=["summary"],
        )

        assert hasattr(stats.output_folder, "path")
        assert stats.csv_filename == "doc_stats.csv"
        assert stats.name == "🇮🇹 Italian Advanced Features CSV"

    def test_extract_stats_contains_extended_feature_set(self, temp_dir):
        stats = DocStatsCsv(
            output_folder=str(temp_dir / "feature"),
            csv_filename="doc_stats.csv",
            groups_to_compute=["summary"],
        )
        doc = SimpleNamespace(
            id="doc_stats_1",
            text=(
                "Contattaci via email a test@example.com.\n\n"
                "- Primo punto\n- Secondo punto\n"
                "Visita https://example.com ora!!! <b>Promo</b>"
            ),
            metadata={},
        )

        features = stats.extract_stats(doc)

        expected_keys = {
            "url_count",
            "email_count",
            "html_tag_count",
            "bullet_point_count",
            "text_entropy",
            "unique_word_ratio",
            "all_lowercase_word_ratio",
            "consecutive_punctuation_count",
        }

        assert expected_keys.issubset(features.keys())
        assert features["url_count"] == 1
        assert features["email_count"] == 1
        assert features["html_tag_count"] == 2
        assert features["bullet_point_count"] == 2
        assert features["text_entropy"] > 0
        assert features["consecutive_punctuation_count"] >= 1


class TestQualityClassifier:
    def test_default_feature_names_exist(self):
        assert len(DEFAULT_FEATURE_NAMES) > 0
        assert "language_score" in DEFAULT_FEATURE_NAMES
        assert "word_count" in DEFAULT_FEATURE_NAMES
        assert "text_entropy" in DEFAULT_FEATURE_NAMES
        assert "unique_word_ratio" in DEFAULT_FEATURE_NAMES
        assert "consecutive_punctuation_count" in DEFAULT_FEATURE_NAMES

    def test_quality_classifier_requires_valid_model_path(self, temp_dir):
        model_path = temp_dir / "nonexistent_model.joblib"

        with pytest.raises(FileNotFoundError):
            QualityClassifier(
                model_path=str(model_path),
                threshold=0.5,
            )


class TestReaderWriterAndFilters:
    def test_jsonl_reader_uses_input_pattern(self):
        reader = get_jsonl_reader("/tmp/data", pattern="train/*.jsonl")
        data_folder = getattr(reader.data_folder, "path", reader.data_folder)

        assert data_folder == "/tmp/data"
        assert reader.glob_pattern == "train/*.jsonl"

    def test_jsonl_writer_uses_default_filename(self):
        writer = get_jsonl_writer("/tmp/output")
        output_folder = getattr(writer.output_folder, "path", writer.output_folder)

        assert output_folder == "/tmp/output"
        assert writer.output_filename.template == "italiano_pulito_${rank}.jsonl"
        assert writer.compression is None

    def test_language_filter_writes_non_italian_docs_in_expected_folder(self):
        language_filter = get_language_filter(
            rejected_dir="/tmp/output/rejected",
            threshold=0.75,
            languages="it",
        )
        languages = language_filter.languages
        exclusion_output_folder = getattr(
            language_filter.exclusion_writer.output_folder,
            "path",
            language_filter.exclusion_writer.output_folder,
        )

        assert languages == ["it"] or languages == "it"
        assert language_filter.language_threshold == 0.75
        assert exclusion_output_folder.endswith("rejected/1_language")
        assert language_filter.exclusion_writer.output_filename.template == "non_italiano_${rank}.jsonl"


class TestPipelineFactory:
    def test_pipeline_factory_signature(self):
        import inspect

        sig = inspect.signature(pipeline_factory.build_italian_cleaning_pipeline)

        params = list(sig.parameters.keys())
        assert "data_dir" in params
        assert "output_dir" in params
        assert "rejected_dir" in params
        assert "pattern" in params
        assert "model_path" in params

    def test_build_pipeline_wires_components_in_expected_order(self, monkeypatch):
        captured = {}

        def fake_reader(data_dir, pattern):
            captured["reader"] = (data_dir, pattern)
            return "reader"

        def fake_language_filter(rejected_dir, threshold=0.75, languages="it"):
            captured["language_filter"] = {
                "rejected_dir": rejected_dir,
                "threshold": threshold,
                "languages": languages,
            }
            return "language_filter"

        def fake_stats_csv(output_folder, csv_filename, groups_to_compute, languages):
            captured["stats"] = {
                "output_folder": output_folder,
                "csv_filename": csv_filename,
                "groups_to_compute": groups_to_compute,
                "languages": languages,
            }
            return "stats"

        def fake_quality_classifier(model_path, rejected_dir, output_folder, threshold):
            captured["classification"] = {
                "model_path": model_path,
                "rejected_dir": rejected_dir,
                "output_folder": output_folder,
                "threshold": threshold,
            }
            return "classification"

        def fake_writer(output_dir):
            captured["writer"] = output_dir
            return "writer"

        monkeypatch.setattr(pipeline_factory, "get_jsonl_reader", fake_reader)
        monkeypatch.setattr(pipeline_factory, "get_language_filter", fake_language_filter)
        monkeypatch.setattr(pipeline_factory, "DocStatsCsv", fake_stats_csv)
        monkeypatch.setattr(pipeline_factory, "ItalianClassification", fake_quality_classifier)
        monkeypatch.setattr(pipeline_factory, "get_jsonl_writer", fake_writer)

        pipeline = pipeline_factory.build_italian_cleaning_pipeline(
            data_dir="/tmp/data",
            output_dir="/tmp/output",
            rejected_dir="/tmp/output/rejected",
            pattern="train/*.jsonl",
            model_path="/tmp/models",
        )

        assert pipeline == ["reader", "language_filter", "stats", "classification", "writer"]
        assert captured["reader"] == ("/tmp/data", "train/*.jsonl")
        assert captured["language_filter"] == {
            "rejected_dir": "/tmp/output/rejected",
            "threshold": 0.75,
            "languages": "it",
        }
        assert captured["stats"]["output_folder"].endswith("/tmp/output/feature")
        assert captured["stats"]["csv_filename"] == "doc_stats_per_file.csv"
        assert captured["stats"]["groups_to_compute"] == ["summary"]
        assert captured["classification"]["model_path"].endswith(
            "/tmp/models/lgbm_quality_model.joblib"
        )
        assert captured["classification"]["rejected_dir"] == "/tmp/output/rejected"
        assert captured["classification"]["output_folder"] == "/tmp/output"
        assert captured["classification"]["threshold"] == 0.65
        assert captured["writer"] == "/tmp/output"
