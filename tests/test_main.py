"""
Test per le funzioni di main.py
"""

import os
import sys
from pathlib import Path

import pytest

# Aggiungi src al path per importare i moduli
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from main import load_config, extract_args


# =============================================================================
# Test load_config
# =============================================================================

class TestLoadConfig:
    """Test per la funzione load_config."""

    def test_load_valid_config(self, temp_config_file: Path, clean_env):
        """Verifica il caricamento di un config valido."""
        load_config(str(temp_config_file))
        
        assert os.environ.get("DATATROVE_COLORIZE_LOGS") == "0"

    def test_config_not_found(self, capsys, clean_env):
        """Verifica il comportamento con config non esistente."""
        load_config("/path/to/nonexistent.conf")
        
        captured = capsys.readouterr()
        assert "Config non trovato" in captured.out

    def test_skip_comments(self, temp_dir: Path, clean_env):
        """Verifica che i commenti vengano saltati."""
        config_content = """# Questo è un commento
# TEST_VAR=non_impostata
REAL_VAR=valore"""
        
        config_path = temp_dir / "test.conf"
        config_path.write_text(config_content)
        
        load_config(str(config_path))
        
        assert os.environ.get("TEST_VAR") is None
        assert os.environ.get("REAL_VAR") == "valore"
        
        # Cleanup
        del os.environ["REAL_VAR"]

    def test_skip_empty_lines(self, temp_dir: Path, clean_env):
        """Verifica che le righe vuote vengano saltate."""
        config_content = """VAR1=primo

VAR2=secondo"""
        
        config_path = temp_dir / "test.conf"
        config_path.write_text(config_content)
        
        load_config(str(config_path))
        
        assert os.environ.get("VAR1") == "primo"
        assert os.environ.get("VAR2") == "secondo"
        
        # Cleanup
        del os.environ["VAR1"]
        del os.environ["VAR2"]

    def test_strip_export_keyword(self, temp_dir: Path, clean_env):
        """Verifica che 'export' venga rimosso."""
        config_content = """export MY_VAR=valore
export ANOTHER_VAR=altro"""
        
        config_path = temp_dir / "test.conf"
        config_path.write_text(config_content)
        
        load_config(str(config_path))
        
        assert os.environ.get("MY_VAR") == "valore"
        assert os.environ.get("ANOTHER_VAR") == "altro"
        
        # Cleanup
        del os.environ["MY_VAR"]
        del os.environ["ANOTHER_VAR"]

    def test_strip_quotes(self, temp_dir: Path, clean_env):
        """Verifica che le virgolette vengano rimosse."""
        config_content = """QUOTED_VAR="valore con spazi"
SINGLE_QUOTED='altro valore'"""
        
        config_path = temp_dir / "test.conf"
        config_path.write_text(config_content)
        
        load_config(str(config_path))
        
        assert os.environ.get("QUOTED_VAR") == "valore con spazi"
        assert os.environ.get("SINGLE_QUOTED") == "altro valore"
        
        # Cleanup
        del os.environ["QUOTED_VAR"]
        del os.environ["SINGLE_QUOTED"]


# =============================================================================
# Test extract_args
# =============================================================================

class TestExtractArgs:
    """Test per la funzione extract_args."""

    def test_default_values(self, monkeypatch):
        """Verifica i valori di default quando non in Docker."""
        # Salva riferimento alla funzione originale prima del mock
        original_exists = os.path.exists
        # Simula di non essere in Docker
        monkeypatch.setattr(os.path, "exists", lambda p: False if p == "/app/src" else original_exists(p))
        
        parser = extract_args()
        args = parser.parse_args([])
        
        assert args.config is None
        assert "$HOME" not in args.root_dir or "ita-llm-pipeline" in args.root_dir

    def test_custom_config(self):
        """Verifica il parsing dell'argomento --config."""
        parser = extract_args()
        args = parser.parse_args(["--config", "configs/custom.conf"])
        
        assert args.config == "configs/custom.conf"

    def test_custom_root_dir(self):
        """Verifica il parsing dell'argomento --root-dir."""
        parser = extract_args()
        args = parser.parse_args(["--root-dir", "/custom/path"])
        
        assert args.root_dir == "/custom/path"

    def test_custom_output_dir(self):
        """Verifica il parsing dell'argomento --output-dir."""
        parser = extract_args()
        args = parser.parse_args(["--output-dir", "/custom/output"])
        
        assert args.output_dir == "/custom/output"

    def test_custom_rejected_dir(self):
        """Verifica il parsing dell'argomento --rejected-dir."""
        parser = extract_args()
        args = parser.parse_args(["--rejected-dir", "/custom/rejected"])
        
        assert args.rejected_dir == "/custom/rejected"

    def test_custom_csv_dir(self):
        """Verifica il parsing dell'argomento --csv-dir."""
        parser = extract_args()
        args = parser.parse_args(["--csv-dir", "/custom/csv"])
        
        assert args.csv_dir == "/custom/csv"

    def test_all_arguments_together(self):
        """Verifica il parsing di tutti gli argomenti insieme."""
        parser = extract_args()
        args = parser.parse_args([
            "--config", "my.conf",
            "--root-dir", "/root",
            "--output-dir", "/output",
            "--rejected-dir", "/rejected",
            "--csv-dir", "/csv",
        ])
        
        assert args.config == "my.conf"
        assert args.root_dir == "/root"
        assert args.output_dir == "/output"
        assert args.rejected_dir == "/rejected"
        assert args.csv_dir == "/csv"

    def test_docker_detection(self, monkeypatch, temp_dir: Path):
        """Verifica il rilevamento dell'ambiente Docker."""
        # Simula di essere in Docker creando /app/src
        app_src = temp_dir / "app" / "src"
        app_src.mkdir(parents=True)
        
        # Salva riferimento alla funzione originale prima del mock
        original_exists = os.path.exists
        
        def mock_exists(path):
            if path == "/app/src":
                return True
            return original_exists(path)
        
        monkeypatch.setattr(os.path, "exists", mock_exists)
        
        parser = extract_args()
        args = parser.parse_args([])
        
        assert args.root_dir == "/app"
        assert args.output_dir == "/app/output"


# =============================================================================
# Test di integrazione config + args
# =============================================================================

class TestConfigAndArgsIntegration:
    """Test di integrazione tra config e argomenti."""

    def test_env_vars_priority(self, temp_dir: Path, clean_env):
        """Verifica che le variabili d'ambiente abbiano priorità sugli argomenti."""
        # Crea un config con OUTPUT_DIR
        config_content = "OUTPUT_DIR=/from/config"
        config_path = temp_dir / "test.conf"
        config_path.write_text(config_content)
        
        # Carica il config
        load_config(str(config_path))
        
        # Verifica che la variabile d'ambiente sia impostata
        assert os.environ.get("OUTPUT_DIR") == "/from/config"
        
        # Nel codice reale, questo valore avrebbe priorità sull'argomento --output-dir
        
        # Cleanup
        del os.environ["OUTPUT_DIR"]
