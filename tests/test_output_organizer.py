"""
Test per le funzioni di output_organizer.py
"""

import csv
import gzip
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest

# Aggiungi src al path per importare i moduli
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.output_organizer import (
    find_single_jsonl_gz,
    iter_jsonl_gz,
    group_texts_by_tag,
    write_grouped_csv,
    resolve_inputFile_objects,
    InputFile,
)


# =============================================================================
# Test find_single_jsonl_gz
# =============================================================================

class TestFindSingleJsonlGz:
    """Test per la funzione find_single_jsonl_gz."""

    def test_find_single_file(self, temp_dir: Path):
        """Verifica che trovi un singolo file .jsonl.gz."""
        # Crea un file di test
        test_file = temp_dir / "test.jsonl.gz"
        with gzip.open(test_file, "wt") as f:
            f.write('{"text": "test"}\n')
        
        result = find_single_jsonl_gz(temp_dir)
        assert result == test_file

    def test_no_file_raises_error(self, temp_dir: Path):
        """Verifica che sollevi errore se non ci sono file."""
        with pytest.raises(FileNotFoundError) as exc_info:
            find_single_jsonl_gz(temp_dir)
        
        assert "Nessun file .jsonl.gz trovato" in str(exc_info.value)

    def test_multiple_files_raises_error(self, temp_dir: Path):
        """Verifica che sollevi errore se ci sono più file."""
        # Crea due file di test
        for name in ["test1.jsonl.gz", "test2.jsonl.gz"]:
            with gzip.open(temp_dir / name, "wt") as f:
                f.write('{"text": "test"}\n')
        
        with pytest.raises(RuntimeError) as exc_info:
            find_single_jsonl_gz(temp_dir)
        
        assert "Trovati più file .jsonl.gz" in str(exc_info.value)


# =============================================================================
# Test iter_jsonl_gz
# =============================================================================

class TestIterJsonlGz:
    """Test per la funzione iter_jsonl_gz."""

    def test_iterate_valid_jsonl(self, temp_dir: Path):
        """Verifica l'iterazione su un file JSONL valido."""
        test_file = temp_dir / "test.jsonl.gz"
        data = [
            {"text": "primo", "id": 1},
            {"text": "secondo", "id": 2},
            {"text": "terzo", "id": 3},
        ]
        
        with gzip.open(test_file, "wt", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        
        result = list(iter_jsonl_gz(test_file))
        assert len(result) == 3
        assert result[0]["text"] == "primo"
        assert result[2]["id"] == 3

    def test_skip_empty_lines(self, temp_dir: Path):
        """Verifica che le righe vuote vengano saltate."""
        test_file = temp_dir / "test.jsonl.gz"
        
        with gzip.open(test_file, "wt", encoding="utf-8") as f:
            f.write('{"text": "primo"}\n')
            f.write('\n')  # Riga vuota
            f.write('   \n')  # Riga con solo spazi
            f.write('{"text": "secondo"}\n')
        
        result = list(iter_jsonl_gz(test_file))
        assert len(result) == 2

    def test_invalid_json_raises_error(self, temp_dir: Path):
        """Verifica che JSON non valido sollevi errore."""
        test_file = temp_dir / "test.jsonl.gz"
        
        with gzip.open(test_file, "wt", encoding="utf-8") as f:
            f.write('{"text": "valido"}\n')
            f.write('non json valido\n')
        
        with pytest.raises(ValueError) as exc_info:
            list(iter_jsonl_gz(test_file))
        
        assert "JSON non valido" in str(exc_info.value)
        assert "riga 2" in str(exc_info.value)


# =============================================================================
# Test group_texts_by_tag
# =============================================================================

class TestGroupTextsByTag:
    """Test per la funzione group_texts_by_tag."""

    def test_group_by_tag(self):
        """Verifica il raggruppamento per tag."""
        records = [
            {"text": "buono 1", "metadata": {"tag": "good"}},
            {"text": "buono 2", "metadata": {"tag": "good"}},
            {"text": "cattivo 1", "metadata": {"tag": "bad"}},
            {"text": "medio 1", "metadata": {"tag": "middle"}},
        ]
        
        result = group_texts_by_tag(iter(records))
        
        assert len(result["good"]) == 2
        assert len(result["bad"]) == 1
        assert len(result["middle"]) == 1
        assert "buono 1" in result["good"]

    def test_skip_missing_tag(self):
        """Verifica che i record senza tag vengano saltati."""
        records = [
            {"text": "con tag", "metadata": {"tag": "good"}},
            {"text": "senza tag", "metadata": {}},
            {"text": "senza metadata"},
        ]
        
        result = group_texts_by_tag(iter(records))
        
        assert len(result) == 1
        assert len(result["good"]) == 1

    def test_skip_missing_text(self):
        """Verifica che i record senza text vengano saltati."""
        records = [
            {"text": "con testo", "metadata": {"tag": "good"}},
            {"metadata": {"tag": "good"}},  # Senza text
        ]
        
        result = group_texts_by_tag(iter(records))
        
        assert len(result["good"]) == 1

    def test_empty_records(self):
        """Verifica il comportamento con nessun record."""
        result = group_texts_by_tag(iter([]))
        assert len(result) == 0


# =============================================================================
# Test write_grouped_csv
# =============================================================================

class TestWriteGroupedCsv:
    """Test per la funzione write_grouped_csv."""

    def test_write_csv(self, temp_dir: Path):
        """Verifica la scrittura del CSV."""
        grouped = {
            "good": ["testo buono 1", "testo buono 2"],
            "bad": ["testo cattivo"],
        }
        
        output_csv = temp_dir / "output.csv"
        write_grouped_csv(grouped, output_csv)
        
        assert output_csv.exists()
        
        # Leggi e verifica il contenuto
        with output_csv.open("r", encoding="utf-8") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Header + 2 righe dati
        assert len(rows) == 3
        assert rows[0] == ["tag", "lenght", "texts"]  # Nota: "lenght" come nel codice originale
        
        # Verifica contenuto (ordinato alfabeticamente)
        assert rows[1][0] == "bad"
        assert rows[1][1] == "1"
        assert rows[2][0] == "good"
        assert rows[2][1] == "2"

    def test_create_parent_dirs(self, temp_dir: Path):
        """Verifica che vengano create le directory parent."""
        grouped = {"good": ["test"]}
        output_csv = temp_dir / "subdir" / "nested" / "output.csv"
        
        write_grouped_csv(grouped, output_csv)
        
        assert output_csv.exists()

    def test_empty_grouped(self, temp_dir: Path):
        """Verifica la scrittura con dizionario vuoto."""
        grouped: Dict[str, List[str]] = {}
        output_csv = temp_dir / "empty.csv"
        
        write_grouped_csv(grouped, output_csv)
        
        assert output_csv.exists()
        
        with output_csv.open("r") as f:
            reader = csv.reader(f)
            rows = list(reader)
        
        # Solo header
        assert len(rows) == 1


# =============================================================================
# Test resolve_inputFile_objects
# =============================================================================

class TestResolveInputFileObjects:
    """Test per la funzione resolve_inputFile_objects."""

    def test_resolve_both_files(self, temp_output_structure: Path):
        """Verifica che vengano risolti entrambi i file."""
        output_file, rejected_file = resolve_inputFile_objects(temp_output_structure)
        
        assert output_file.label == "output"
        assert output_file.path.exists()
        assert rejected_file.label == "rejected"
        assert rejected_file.path.exists()

    def test_missing_rejected_dir(self, temp_dir: Path):
        """Verifica errore se manca la directory rejected."""
        # Crea solo output senza rejected
        output_dir = temp_dir / "output"
        output_dir.mkdir()
        
        with gzip.open(output_dir / "test.jsonl.gz", "wt") as f:
            f.write('{"text": "test"}\n')
        
        with pytest.raises(FileNotFoundError) as exc_info:
            resolve_inputFile_objects(output_dir)
        
        assert "rejected non trovata" in str(exc_info.value)


# =============================================================================
# Test InputFile dataclass
# =============================================================================

class TestInputFile:
    """Test per il dataclass InputFile."""

    def test_create_input_file(self, temp_dir: Path):
        """Verifica la creazione di un InputFile."""
        test_path = temp_dir / "test.jsonl.gz"
        input_file = InputFile(label="test", path=test_path)
        
        assert input_file.label == "test"
        assert input_file.path == test_path

    def test_input_file_frozen(self, temp_dir: Path):
        """Verifica che InputFile sia immutabile."""
        test_path = temp_dir / "test.jsonl.gz"
        input_file = InputFile(label="test", path=test_path)
        
        with pytest.raises(AttributeError):
            input_file.label = "modified"  # type: ignore


# =============================================================================
# Test di integrazione
# =============================================================================

class TestIntegration:
    """Test di integrazione per output_organizer."""

    def test_full_pipeline(self, temp_output_structure: Path, temp_dir: Path):
        """Test del flusso completo di classificazione."""
        output_file, rejected_file = resolve_inputFile_objects(temp_output_structure)
        
        # Raggruppa i testi
        output_grouped = group_texts_by_tag(iter_jsonl_gz(output_file.path))
        rejected_grouped = group_texts_by_tag(iter_jsonl_gz(rejected_file.path))
        
        # Verifica raggruppamento
        assert "good" in output_grouped
        assert "middle" in output_grouped
        assert "bad" in rejected_grouped
        
        # Scrivi CSV
        csv_dir = temp_dir / "csv"
        csv_dir.mkdir()
        
        write_grouped_csv(output_grouped, csv_dir / "output.csv")
        write_grouped_csv(rejected_grouped, csv_dir / "rejected.csv")
        
        assert (csv_dir / "output.csv").exists()
        assert (csv_dir / "rejected.csv").exists()
