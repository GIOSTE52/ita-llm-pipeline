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
    def test_builds_paths_and_creates_directories(self, monkeypatch, temp_dir, clean_env):
        root_dir = temp_dir / "workspace"
        output_dir = temp_dir / "output"

        monkeypatch.setattr(
            config_loader,
            "extract_args",
            lambda: SimpleNamespace(
                root_dir=str(root_dir),
                output_dir=str(output_dir),
                rejected_dir=None,
                csv_dir=None,
                feature_dir=None,
                model_path=str(temp_dir / "models" / "custom.joblib"),
                config=None,
            ),
        )
        default_model_path = root_dir / "models" / "spam_lgbm.joblib"
        monkeypatch.setattr(config_loader.os.path, "exists", lambda path: path != str(default_model_path))

        config = config_loader.get_config()

        assert config["DATA_DIR"] == str(root_dir / "data")
        assert config["OUTPUT_DIR"] == str(output_dir)
        assert config["REJECTED_DIR"] == str(output_dir / "rejected")
        assert config["CSV_DIR"] == str(output_dir / "csv")
        assert config["FEATURE_DIR"] == str(output_dir / "feature")
        assert config["MODEL_PATH"] == str(default_model_path)
        assert Path(config["OUTPUT_DIR"]).exists()
        assert Path(config["REJECTED_DIR"]).exists()
        assert Path(config["CSV_DIR"]).exists()
        assert Path(config["FEATURE_DIR"]).exists()
        assert Path(config["MODEL_PATH"]).parent.exists()

    def test_environment_variables_override_cli_defaults(self, monkeypatch, temp_dir, clean_env):
        cli_root = temp_dir / "cli-root"
        cli_output = temp_dir / "cli-output"
        env_output = temp_dir / "env-output"

        monkeypatch.setenv("ROOT_DIR", str(temp_dir / "env-root"))
        monkeypatch.setenv("OUTPUT_DIR", str(env_output))
        monkeypatch.setenv("REJECTED_DIR", str(temp_dir / "env-rejected"))
        monkeypatch.setenv("FEATURE_DIR", str(temp_dir / "env-feature"))
        monkeypatch.setenv("CSV_DIR", str(temp_dir / "env-csv"))
        monkeypatch.setenv("MODEL_PATH", str(temp_dir / "env-models" / "spam.joblib"))

        monkeypatch.setattr(
            config_loader,
            "extract_args",
            lambda: SimpleNamespace(
                root_dir=str(cli_root),
                output_dir=str(cli_output),
                rejected_dir=str(temp_dir / "cli-rejected"),
                csv_dir=str(temp_dir / "cli-csv"),
                feature_dir=str(temp_dir / "cli-feature"),
                model_path=str(temp_dir / "cli-models" / "spam.joblib"),
                config=None,
            ),
        )
        monkeypatch.setattr(config_loader.os.path, "exists", lambda path: False)

        config = config_loader.get_config()

        assert config["OUTPUT_DIR"] == str(env_output)
        assert config["REJECTED_DIR"] == str(temp_dir / "env-rejected")
        assert config["FEATURE_DIR"] == str(temp_dir / "env-feature")
        assert config["CSV_DIR"] == str(temp_dir / "env-csv")
        assert config["MODEL_PATH"] == str(temp_dir / "env-models" / "spam.joblib")


class TestMainEntrypoint:
    def test_main_wires_pipeline_executor_and_output_analysis(self, monkeypatch):
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
