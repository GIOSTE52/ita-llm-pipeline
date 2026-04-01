"""
Test dei componenti principali della pipeline usando dati reali in `data/train/`.
"""

from types import SimpleNamespace

from blocks.filters import CustomItalianFilter
from blocks.spam_classifier.spam_stats import extract_spam_features
from pipeline_factory import build_italian_cleaning_pipeline


class TestCustomItalianFilter:
    def test_accepts_real_training_documents(self, training_good_docs, temp_dir):
        filter_block = CustomItalianFilter(
            output_folder=str(temp_dir / "rejects"),
            filename="custom_rejected_${rank}.jsonl",
        )

        for doc in training_good_docs[:3]:
            accepted = filter_block.filter(SimpleNamespace(text=doc["text"]))
            assert accepted is True, f"{doc['id']} non dovrebbe essere scartato dal filtro custom"

    def test_rejects_short_navigation_like_text(self, temp_dir):
        filter_block = CustomItalianFilter(
            output_folder=str(temp_dir / "rejects"),
            filename="custom_rejected_${rank}.jsonl",
        )

        accepted = filter_block.filter(
            SimpleNamespace(text="Home | Login | Contatti | Cookie policy | Privacy")
        )

        assert accepted is False


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
        assert features["lang_is_ita"] == 1.0
        assert features["lang_score"] > 0.0
        assert features["has_url_meta"] == 1.0
        assert features["minhash_cluster_size"] >= 0.0


class TestPipelineFactory:
    def test_build_pipeline_wires_expected_blocks_and_paths(self, temp_dir):
        output_dir = temp_dir / "output"
        rejected_dir = temp_dir / "rejected"

        pipeline = build_italian_cleaning_pipeline(
            data_dir=str(temp_dir),
            output_dir=str(output_dir),
            rejected_dir=str(rejected_dir),
            model_path=str(temp_dir / "models" / "spam_lgbm.joblib"),
        )

        assert len(pipeline) == 7
        assert pipeline[0].data_folder == str(temp_dir)
        assert pipeline[0].glob_pattern == "train/*.jsonl"
        assert pipeline[1].languages == "it"
        assert pipeline[1].language_threshold == 0.65
        assert pipeline[1].exclusion_writer.output_folder.endswith("1_language")
        assert pipeline[2].exclusion_writer.output_folder.endswith("2_custom_filter")
        assert pipeline[4].output_folder == str(output_dir / "feature")
        assert pipeline[5].output_folder == str(output_dir / "feature")
        assert pipeline[6].output_folder == str(output_dir)
