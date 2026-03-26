"""
Fixtures condivise per i test della pipeline ITA-LLM.
"""

import gzip
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest


# =============================================================================
# Fixtures per documenti di test
# =============================================================================

@pytest.fixture
def italian_texts() -> List[Dict[str, str]]:
    """Testi italiani validi che dovrebbero passare i filtri."""
    return [
        {
            "id": "it_dante",
            "text": "Nel mezzo del cammin di nostra vita mi ritrovai per una selva oscura, "
                    "ché la diritta via era smarrita. Ahi quanto a dir qual era è cosa dura "
                    "esta selva selvaggia e aspra e forte che nel pensier rinova la paura!"
        },
        {
            "id": "it_colloquiale",
            "text": "Oggi sono andato al mercato e ho comprato delle mele bellissime. "
                    "Il fruttivendolo mi ha fatto un ottimo prezzo perché erano le ultime. "
                    "Domani tornerò a comprare anche le arance che erano molto buone."
        },
        {
            "id": "it_romanesco",
            "text": "Aò, che bella giornata oggi! Er sole picchia forte già da le otto. "
                    "Me so' svejato co' 'na scimmia addosso che la metà basta, ma appena "
                    "ho visto 'sta luce, m'è tornata la voja de campà."
        },
        {
            "id": "it_formale",
            "text": "La presente comunicazione ha lo scopo di informare tutti i dipendenti "
                    "che a partire dal prossimo mese entreranno in vigore le nuove normative "
                    "relative alla sicurezza sul lavoro. Si prega di prendere visione del documento."
        },
    ]


@pytest.fixture
def non_italian_texts() -> List[Dict[str, str]]:
    """Testi non italiani che dovrebbero essere filtrati."""
    return [
        {
            "id": "en_sample",
            "text": "The quick brown fox jumps over the lazy dog. This is a sample text "
                    "in English that should be filtered out by the Italian language detector."
        },
        {
            "id": "es_sample",
            "text": "Hola, cómo estás? Hoy hace un día muy bonito para pasear por el parque. "
                    "Los pájaros cantan y las flores están floreciendo en primavera."
        },
        {
            "id": "fr_sample",
            "text": "Bonjour, comment allez-vous? Aujourd'hui il fait très beau pour se promener "
                    "dans le parc. Les oiseaux chantent et les fleurs sont en pleine floraison."
        },
        {
            "id": "pt_sample",
            "text": "Olá, como você está? Hoje é um dia muito bonito para passear no parque. "
                    "Os pássaros cantam e as flores estão florescendo na primavera."
        },
    ]


@pytest.fixture
def noise_texts() -> List[Dict[str, str]]:
    """Testi rumorosi che dovrebbero essere filtrati."""
    return [
        {
            "id": "noise_empty",
            "text": ""
        },
        {
            "id": "noise_short",
            "text": "ab"
        },
        {
            "id": "noise_html",
            "text": "<div class='footer'>© 2026 • P.IVA • Contatti • Cookie • Privacy</div>"
        },
        {
            "id": "noise_code",
            "text": "function test() { return x => x * 2; } class MyClass { constructor() {} }"
        },
        {
            "id": "noise_repetition",
            "text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbb"
        },
        {
            "id": "noise_log",
            "text": "INFO 2026-01-22 17:34:28 - User logged in from IP 192.168.1.1. Status: OK. "
                    "INFO 2026-01-22 17:34:28 - User logged in from IP 192.168.1.1. Status: OK."
        },
        {
            "id": "noise_boilerplate",
            "text": "Accetta i cookie e iscriviti alla newsletter per ricevere pubblicità personalizzata"
        },
    ]


# =============================================================================
# Fixtures per directory temporanee
# =============================================================================

@pytest.fixture
def temp_dir():
    """Crea una directory temporanea per i test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_config_file(temp_dir: Path) -> Path:
    """Crea un file di configurazione temporaneo."""
    config_content = """# Test config
DATATROVE_COLORIZE_LOGS=0
DATA_DIR={data_dir}
OUTPUT_DIR={output_dir}
""".format(data_dir=temp_dir / "data", output_dir=temp_dir / "output")
    
    config_path = temp_dir / "test.conf"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def temp_data_dir(temp_dir: Path, italian_texts: List[Dict]) -> Path:
    """Crea una directory con file JSONL di test."""
    data_dir = temp_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Crea file JSONL di test
    jsonl_path = data_dir / "test_input.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for doc in italian_texts:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")
    
    return data_dir


@pytest.fixture
def temp_output_structure(temp_dir: Path) -> Path:
    """Crea la struttura di output con file .jsonl.gz per test output_organizer."""
    output_dir = temp_dir / "output"
    rejected_dir = output_dir / "rejected"
    output_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)
    
    # Dati di esempio con metadata.tag
    output_data = [
        {"text": "Testo buono di esempio", "metadata": {"tag": "good"}},
        {"text": "Altro testo buono", "metadata": {"tag": "good"}},
        {"text": "Testo medio", "metadata": {"tag": "middle"}},
    ]
    
    rejected_data = [
        {"text": "Testo scartato", "metadata": {"tag": "bad"}},
        {"text": "Altro scarto", "metadata": {"tag": "bad"}},
    ]
    
    # Scrivi file gzipped
    output_file = output_dir / "risultati_0.jsonl.gz"
    with gzip.open(output_file, "wt", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    rejected_file = rejected_dir / "risultati_0.jsonl.gz"
    with gzip.open(rejected_file, "wt", encoding="utf-8") as f:
        for item in rejected_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    return output_dir


# =============================================================================
# Fixtures per environment variables
# =============================================================================

@pytest.fixture
def clean_env():
    """Pulisce le variabili d'ambiente relative alla pipeline prima del test."""
    env_vars = ["ROOT_DIR", "DATA_DIR", "OUTPUT_DIR", "REJECTED_DIR", "CSV_DIR"]
    original = {var: os.environ.get(var) for var in env_vars}
    
    # Rimuovi le variabili
    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield
    
    # Ripristina i valori originali
    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]
