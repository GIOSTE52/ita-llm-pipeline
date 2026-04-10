"""
Test per configurazione ed entrypoint principale.
"""

import os
import sys
from types import SimpleNamespace

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
        monkeypatch.setattr(config_loader.os, "cpu_count", lambda: 8)
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = config_loader.extract_args()

        assert args.config is None
        assert args.tasks is None
        assert args.root_dir == os.path.abspath(".")
        assert args.output_dir == os.path.join(args.root_dir, "output")
        assert args.workers == 6

    def test_default_values_inside_docker(self, monkeypatch):
        original_exists = os.path.exists

        def fake_exists(path):
            if path == "/app/src":
                return True
            return original_exists(path)

        monkeypatch.setattr(config_loader.os.path, "exists", fake_exists)
        monkeypatch.setattr(config_loader.os, "cpu_count", lambda: 4)
        monkeypatch.setattr(sys, "argv", ["prog"])

        args = config_loader.extract_args()

        assert args.root_dir == "/app"
        assert args.output_dir == "/app/output"
        assert args.workers == 2

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
                "--tasks",
                "11",
                "--workers",
                "3",
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
        assert args.tasks == 11
        assert args.workers == 3
        assert args.csv_dir == "/tmp/csv"
        assert args.feature_dir == "/tmp/feature"
        assert args.model_path == "/tmp/model.joblib"


class TestGetConfig:
    def test_config_loads_successfully_with_dynamic_defaults(
        self, monkeypatch, clean_env, temp_dir
    ):
        monkeypatch.setattr(
            config_loader,
            "extract_args",
            lambda: SimpleNamespace(
                root_dir=str(temp_dir),
                output_dir=str(temp_dir / "output"),
                rejected_dir=None,
                csv_dir=None,
                feature_dir=None,
                model_path=None,
                config=None,
                workers=5,
                tasks=None,
            ),
        )
        monkeypatch.setattr(
            config_loader.glob,
            "glob",
            lambda pattern: ["train/shard_0000.jsonl", "train/shard_0001.jsonl"],
        )

        config = config_loader.get_config()

        assert isinstance(config, dict)
        assert config["DATA_DIR"] == str(temp_dir / "data")
        assert config["INPUT_SUB_PATTERN"] == "train/*.jsonl"
        assert config["OUTPUT_DIR"] == str(temp_dir / "output")
        assert config["REJECTED_DIR"] == str(temp_dir / "output" / "rejected")
        assert config["FEATURE_DIR"] == str(temp_dir / "output" / "feature")
        assert config["MODEL_PATH"] == str(temp_dir / "models")
        assert config["MAX_WORKERS"] == 5
        assert config["NUM_TASKS"] == 2

    def test_environment_variables_override_argument_defaults(
        self, monkeypatch, clean_env, temp_dir
    ):
        monkeypatch.setenv("ROOT_DIR", str(temp_dir / "env_root"))
        monkeypatch.setenv("OUTPUT_DIR", str(temp_dir / "env_output"))
        monkeypatch.setenv("DATA_DIR", str(temp_dir / "env_data"))
        monkeypatch.setenv("REJECTED_DIR", str(temp_dir / "env_rejected"))
        monkeypatch.setenv("FEATURE_DIR", str(temp_dir / "env_feature"))
        monkeypatch.setenv("MODEL_PATH", str(temp_dir / "env_models"))
        monkeypatch.setenv("MAX_WORKERS", "9")
        monkeypatch.setattr(
            config_loader,
            "extract_args",
            lambda: SimpleNamespace(
                root_dir="/tmp/ignored_root",
                output_dir="/tmp/ignored_output",
                rejected_dir="/tmp/ignored_rejected",
                csv_dir=None,
                feature_dir="/tmp/ignored_feature",
                model_path="/tmp/ignored_models",
                config=None,
                workers=2,
                tasks=None,
            ),
        )
        monkeypatch.setattr(
            config_loader.glob,
            "glob",
            lambda pattern: ["a.jsonl", "b.jsonl", "c.jsonl"],
        )

        config = config_loader.get_config()

        assert config["DATA_DIR"] == str(temp_dir / "env_data")
        assert config["OUTPUT_DIR"] == str(temp_dir / "env_output")
        assert config["REJECTED_DIR"] == str(temp_dir / "env_rejected")
        assert config["FEATURE_DIR"] == str(temp_dir / "env_feature")
        assert config["MODEL_PATH"] == str(temp_dir / "env_models")
        assert config["MAX_WORKERS"] == 9
        assert config["NUM_TASKS"] == 3


class TestMainEntrypoint:
    def test_main_wires_pipeline_executor_aggregation_and_output_analysis(
        self, monkeypatch
    ):
        captured = {}
        fake_config = {
            "DATA_DIR": "/tmp/data",
            "OUTPUT_DIR": "/tmp/output",
            "REJECTED_DIR": "/tmp/output/rejected",
            "FEATURE_DIR": "/tmp/output/feature",
            "INPUT_SUB_PATTERN": "train/*.jsonl",
            "MODEL_PATH": "/tmp/models",
            "NUM_TASKS": 7,
            "MAX_WORKERS": 4,
        }
        fake_pipeline = ["reader", "stats", "classifier", "writer"]

        class FakeExecutor:
            def __init__(self, pipeline, tasks, workers):
                captured["pipeline"] = pipeline
                captured["tasks"] = tasks
                captured["workers"] = workers

            def run(self):
                captured["run_called"] = True

        def fake_build_pipeline(data_dir, output_dir, rejected_dir, pattern, model_path):
            captured["build_args"] = {
                "data_dir": data_dir,
                "output_dir": output_dir,
                "rejected_dir": rejected_dir,
                "pattern": pattern,
                "model_path": model_path,
            }
            return fake_pipeline

        def fake_subprocess_run(command, shell=False, check=False):
            captured.setdefault("subprocess_calls", []).append(
                {"command": command, "shell": shell, "check": check}
            )
            return SimpleNamespace(returncode=0)

        monkeypatch.setattr(main, "get_config", lambda: fake_config)
        monkeypatch.setattr(main, "build_italian_cleaning_pipeline", fake_build_pipeline)
        monkeypatch.setattr(main, "LocalPipelineExecutor", FakeExecutor)
        monkeypatch.setattr(main.time, "sleep", lambda seconds: None)
        monkeypatch.setattr(main.subprocess, "run", fake_subprocess_run)
        monkeypatch.setattr(
            main,
            "output_classification",
            lambda rejected_dir, output_dir: captured.setdefault(
                "classification_args",
                (rejected_dir, output_dir),
            ),
        )

        main.main()

        assert captured["build_args"] == {
            "data_dir": fake_config["DATA_DIR"],
            "output_dir": fake_config["OUTPUT_DIR"],
            "rejected_dir": fake_config["REJECTED_DIR"],
            "pattern": fake_config["INPUT_SUB_PATTERN"],
            "model_path": fake_config["MODEL_PATH"],
        }
        assert captured["pipeline"] == fake_pipeline
        assert captured["tasks"] == fake_config["NUM_TASKS"]
        assert captured["workers"] == fake_config["MAX_WORKERS"]
        assert captured["run_called"] is True
        assert len(captured["subprocess_calls"]) == 2
        assert "rank_*_doc_stats_per_file.csv" in captured["subprocess_calls"][0]["command"]
        assert captured["subprocess_calls"][0]["shell"] is True
        assert captured["subprocess_calls"][0]["check"] is True
        assert captured["subprocess_calls"][1]["command"].startswith("rm ")
        assert captured["classification_args"] == (
            fake_config["REJECTED_DIR"],
            fake_config["OUTPUT_DIR"],
        )
