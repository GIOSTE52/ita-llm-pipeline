"""
Test per configurazione ed entrypoint principale.
"""

import os
import sys
from types import SimpleNamespace
from pathlib import Path

import config_loader
import main


class TestExtractArgs:
    def test_default_values_outside_docker(self, monkeypatch):
        original_exists = os.path.exists

        def fake_exists(path):
            if path == "/app/src":
                return False
            return original_exists(path)

        monkeypatch.setattr(config_loader.os.path, "exists", fake_exists)
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = config_loader.extract_args()

        assert args.config is None
        assert args.root_dir == os.path.abspath(".")
        assert args.output_dir == os.path.join(args.root_dir, "output")

    def test_default_values_inside_docker(self, monkeypatch):
        original_exists = os.path.exists

        def fake_exists(path):
            if path == "/app/src":
                return True
            return original_exists(path)

        monkeypatch.setattr(config_loader.os.path, "exists", fake_exists)
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = config_loader.extract_args()

        assert args.root_dir == "/app"
        assert args.output_dir == "/app/output"

    def test_custom_arguments_are_parsed(self, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "prog",
                "--config",
                "configs/custom.conf",
                "--root-dir",
                "/tmp/root",
                "--output-dir",
                "/tmp/output",
                "--rejected-dir",
                "/tmp/rejected",
                "--csv-dir",
                "/tmp/csv",
                "--feature-dir",
                "/tmp/feature",
                "--model-path",
                "/tmp/model.joblib",
            ],
        )

        args = config_loader.extract_args()

        assert args.config == "configs/custom.conf"
        assert args.root_dir == "/tmp/root"
        assert args.output_dir == "/tmp/output"
        assert args.rejected_dir == "/tmp/rejected"
        assert args.csv_dir == "/tmp/csv"
        assert args.feature_dir == "/tmp/feature"
        assert args.model_path == "/tmp/model.joblib"


class TestGetConfig:
    """Test per il caricamento della configurazione."""
    
    def test_config_loads_successfully_with_default_values(self, monkeypatch, clean_env):
        """Verifica che la configurazione si carichi correttamente."""
        monkeypatch.setattr(
            config_loader,
            "extract_args",
            lambda: SimpleNamespace(
                root_dir=".",
                output_dir="./output",
                rejected_dir=None,
                csv_dir=None,
                feature_dir=None,
                model_path=None,
                config=None,
            ),
        )

        config = config_loader.get_config()

        assert isinstance(config, dict)
        assert "DATA_DIR" in config
        assert "OUTPUT_DIR" in config
        assert "MODEL_PATH" in config


class TestMainEntrypoint:
    """Test per il main entry point della pipeline."""
    
    def test_main_wires_pipeline_executor_and_output_analysis(self, monkeypatch):
        """Verifica che main() instanzi l'executor e chiami l'output analysis."""
        captured = {}
        fake_config = {
            "DATA_DIR": "/tmp/data",
            "OUTPUT_DIR": "/tmp/output",
            "REJECTED_DIR": "/tmp/output/rejected",
            "FEATURE_DIR": "/tmp/output/feature",
            "MODEL_PATH": "/tmp/models/spam_lgbm.joblib",
        }
        fake_pipeline = ["reader", "filter", "writer"]

        class FakeExecutor:
            def __init__(self, pipeline, tasks, workers):
                self.pipeline = pipeline  # Aggiungo l'attributo pipeline
                captured["pipeline"] = pipeline
                captured["tasks"] = tasks
                captured["workers"] = workers

            def run(self):
                captured["run_called"] = True

        monkeypatch.setattr(main, "get_config", lambda: fake_config)
        monkeypatch.setattr(
            main,
            "build_italian_cleaning_pipeline",
            lambda data_dir, output_dir, rejected_dir, model_path: fake_pipeline,
        )
        monkeypatch.setattr(main, "LocalPipelineExecutor", FakeExecutor)
        monkeypatch.setattr(
            main,
            "output_classification",
            lambda rejected_dir, output_dir: captured.setdefault(
                "classification_args",
                (rejected_dir, output_dir),
            ),
        )

        main.main()

        assert captured["pipeline"] == fake_pipeline
        assert captured["tasks"] == 1
        assert captured["workers"] == 1
        assert captured["run_called"] is True
        assert captured["classification_args"] == (
            fake_config["REJECTED_DIR"],
            fake_config["OUTPUT_DIR"],
        )
