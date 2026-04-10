"""
Test per le utility dello script di valutazione modello.
"""

import sys

import pytest

from scripts import evaluate_model


class TestEvaluateModelHelpers:
    def test_load_model_metadata_extracts_expected_fields(self, monkeypatch):
        artifact = {
            "threshold": 0.72,
            "training_metadata": {"test_csv": "/tmp/test.csv"},
            "model_name": "LightGBM Custom",
        }
        monkeypatch.setattr(evaluate_model.joblib, "load", lambda path: artifact)

        metadata = evaluate_model.load_model_metadata("/tmp/model.joblib")

        assert metadata == {
            "threshold": 0.72,
            "training_metadata": {"test_csv": "/tmp/test.csv"},
            "model_name": "LightGBM Custom",
        }

    def test_resolve_test_csv_prefers_explicit_value(self):
        resolved = evaluate_model.resolve_test_csv(
            explicit_test_csv="/tmp/explicit.csv",
            model_metadata={"training_metadata": {"test_csv": "/tmp/model.csv"}},
        )

        assert resolved == "/tmp/explicit.csv"

    def test_resolve_test_csv_uses_model_metadata_when_cli_value_missing(self):
        resolved = evaluate_model.resolve_test_csv(
            explicit_test_csv=None,
            model_metadata={"training_metadata": {"test_csv": "/tmp/model.csv"}},
        )

        assert resolved == "/tmp/model.csv"

    def test_print_model_comparison_outputs_summary(self, capsys):
        comparison_result = {
            "cv_folds": 3,
            "baseline_model_name": "LightGBM",
            "threshold": 0.65,
            "models": [
                {
                    "model_name": "LightGBM",
                    "roc_auc_mean": 0.91,
                    "roc_auc_std": 0.01,
                    "f1_score_mean": 0.88,
                    "f1_score_std": 0.02,
                    "balanced_accuracy_mean": 0.86,
                    "balanced_accuracy_std": 0.02,
                    "delta_vs_baseline": {
                        "roc_auc_mean": 0.0,
                        "f1_score_mean": 0.0,
                    },
                },
                {
                    "model_name": "Random Forest",
                    "roc_auc_mean": 0.89,
                    "roc_auc_std": 0.03,
                    "f1_score_mean": 0.85,
                    "f1_score_std": 0.04,
                    "balanced_accuracy_mean": 0.83,
                    "balanced_accuracy_std": 0.03,
                    "delta_vs_baseline": {
                        "roc_auc_mean": -0.02,
                        "f1_score_mean": -0.03,
                    },
                },
            ],
            "winner": {
                "model_name": "LightGBM",
                "roc_auc_mean": 0.91,
                "f1_score_mean": 0.88,
            },
        }

        evaluate_model.print_model_comparison(comparison_result)
        output = capsys.readouterr().out

        assert "CONFRONTO MODELLI" in output
        assert "LightGBM" in output
        assert "Random Forest" in output
        assert "Miglior modello per ROC-AUC medio" in output


class TestEvaluateModelMain:
    def test_main_uses_model_metadata_and_runs_cross_validation(self, monkeypatch):
        captured = {}
        metadata = {
            "threshold": 0.71,
            "training_metadata": {
                "source_csv": "/tmp/source.csv",
                "train_csv": "/tmp/train.csv",
                "validation_csv": "/tmp/val.csv",
                "test_csv": "/tmp/test.csv",
            },
            "model_name": "LightGBM",
        }
        comparison_result = {
            "cv_folds": 3,
            "baseline_model_name": "LightGBM",
            "threshold": 0.71,
            "models": [],
            "winner": {
                "model_name": "LightGBM",
                "roc_auc_mean": 0.9,
                "f1_score_mean": 0.88,
            },
        }

        class FakeQualityClassifier:
            def __init__(self, model_path, threshold):
                captured["classifier_init"] = {
                    "model_path": model_path,
                    "threshold": threshold,
                }

            def evaluate(
                self,
                csv_path,
                label_column="label",
                output_dir=None,
                comparison_result=None,
            ):
                captured["evaluate_call"] = {
                    "csv_path": csv_path,
                    "label_column": label_column,
                    "output_dir": output_dir,
                    "comparison_result": comparison_result,
                }
                return {
                    "accuracy": 0.9,
                    "balanced_accuracy": 0.89,
                    "f1_score": 0.88,
                    "roc_auc": 0.91,
                }

            @staticmethod
            def cross_validate_models(
                csv_path,
                label_column="label",
                threshold=0.65,
                cv_folds=5,
                random_state=42,
                model_names=None,
            ):
                captured["cross_validate_call"] = {
                    "csv_path": csv_path,
                    "label_column": label_column,
                    "threshold": threshold,
                    "cv_folds": cv_folds,
                    "random_state": random_state,
                    "model_names": model_names,
                }
                return comparison_result

        def fake_exists(path):
            return path in {
                "/tmp/model.joblib",
                "/tmp/test.csv",
                "/tmp/source.csv",
            }

        monkeypatch.setattr(evaluate_model, "load_model_metadata", lambda path: metadata)
        monkeypatch.setattr(evaluate_model, "QualityClassifier", FakeQualityClassifier)
        monkeypatch.setattr(evaluate_model, "print_model_comparison", lambda result: captured.setdefault("printed_comparison", result))
        monkeypatch.setattr(evaluate_model.os.path, "exists", fake_exists)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_model.py",
                "--model",
                "/tmp/model.joblib",
                "--output-dir",
                "/tmp/evaluation",
                "--compare-models",
                "--cv-folds",
                "3",
                "--cv-random-state",
                "7",
                "--cv-models",
                "lightgbm",
                "random_forest",
            ],
        )

        evaluate_model.main()

        assert captured["classifier_init"] == {
            "model_path": "/tmp/model.joblib",
            "threshold": 0.71,
        }
        assert captured["cross_validate_call"] == {
            "csv_path": "/tmp/source.csv",
            "label_column": "label",
            "threshold": 0.71,
            "cv_folds": 3,
            "random_state": 7,
            "model_names": ["lightgbm", "random_forest"],
        }
        assert captured["evaluate_call"] == {
            "csv_path": "/tmp/test.csv",
            "label_column": "label",
            "output_dir": "/tmp/evaluation",
            "comparison_result": comparison_result,
        }
        assert captured["printed_comparison"] == comparison_result

    def test_main_exits_when_test_csv_cannot_be_resolved(self, monkeypatch, capsys):
        monkeypatch.setattr(
            evaluate_model,
            "load_model_metadata",
            lambda path: {"threshold": 0.65, "training_metadata": {}, "model_name": "LightGBM"},
        )
        monkeypatch.setattr(evaluate_model.os.path, "exists", lambda path: path == "/tmp/model.joblib")
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "evaluate_model.py",
                "--model",
                "/tmp/model.joblib",
            ],
        )

        with pytest.raises(SystemExit) as exc_info:
            evaluate_model.main()

        assert exc_info.value.code == 1
        assert "impossibile determinare il test set" in capsys.readouterr().out
