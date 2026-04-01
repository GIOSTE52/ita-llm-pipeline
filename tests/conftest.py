"""
Fixtures condivise per i test della pipeline ITA-LLM.
"""

import gzip
import importlib.util
import json
import os
import re as stdlib_re
import string
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _install_regex_stub() -> None:
    if "regex" in sys.modules or importlib.util.find_spec("regex") is not None:
        return

    regex_module = types.ModuleType("regex")

    def translate(pattern: str) -> str:
        pattern = pattern.replace(
            r"\p{L}+(?:['’]\p{L}+)?",
            r"[^\W\d_]+(?:['’][^\W\d_]+)?",
        )
        pattern = pattern.replace(r"[^\p{L}]+", r"(?:[\W\d_])+")
        pattern = pattern.replace(r"\p{L}", r"[^\W\d_]")
        return pattern

    def compile(pattern: str, flags: int = 0):
        return stdlib_re.compile(translate(pattern), flags)

    regex_module.compile = compile
    regex_module.UNICODE = stdlib_re.UNICODE
    regex_module.IGNORECASE = stdlib_re.IGNORECASE
    regex_module.MULTILINE = stdlib_re.MULTILINE
    regex_module.__version__ = "test-stub"
    sys.modules["regex"] = regex_module


def _install_loguru_stub() -> None:
    if "loguru" in sys.modules or importlib.util.find_spec("loguru") is not None:
        return

    loguru_module = types.ModuleType("loguru")

    class _Logger:
        def warning(self, *args, **kwargs):
            return None

        def error(self, *args, **kwargs):
            return None

        def info(self, *args, **kwargs):
            return None

    loguru_module.logger = _Logger()
    sys.modules["loguru"] = loguru_module


def _install_datatrove_stub() -> None:
    if "datatrove" in sys.modules or importlib.util.find_spec("datatrove") is not None:
        return

    datatrove_module = types.ModuleType("datatrove")

    executor_module = types.ModuleType("datatrove.executor")
    pipeline_module = types.ModuleType("datatrove.pipeline")
    readers_module = types.ModuleType("datatrove.pipeline.readers")
    filters_module = types.ModuleType("datatrove.pipeline.filters")
    filters_base_module = types.ModuleType("datatrove.pipeline.filters.base_filter")
    writers_module = types.ModuleType("datatrove.pipeline.writers")
    writers_disk_base_module = types.ModuleType("datatrove.pipeline.writers.disk_base")
    base_module = types.ModuleType("datatrove.pipeline.base")
    stats_module = types.ModuleType("datatrove.pipeline.stats")
    stats_doc_module = types.ModuleType("datatrove.pipeline.stats.doc_stats")
    stats_config_module = types.ModuleType("datatrove.pipeline.stats.config")
    data_module = types.ModuleType("datatrove.data")
    io_module = types.ModuleType("datatrove.io")
    utils_module = types.ModuleType("datatrove.utils")
    utils_text_module = types.ModuleType("datatrove.utils.text")

    class PipelineStep:
        name = "Pipeline Step"

        def run(self, data, rank: int = 0, world_size: int = 1):
            yield from data

    class DiskWriter:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

    class JsonlWriter(DiskWriter):
        def __init__(self, output_folder: str, output_filename: str | None = None, compression=None):
            super().__init__(output_folder, output_filename, compression)
            self.output_folder = output_folder
            self.output_filename = output_filename
            self.compression = compression

    class JsonlReader:
        def __init__(self, data_folder: str, glob_pattern: str | None = None, text_key: str = "text"):
            self.data_folder = data_folder
            self.glob_pattern = glob_pattern
            self.text_key = text_key

    class LanguageFilter(PipelineStep):
        def __init__(self, languages, language_threshold, exclusion_writer=None):
            self.languages = languages
            self.language_threshold = language_threshold
            self.exclusion_writer = exclusion_writer

    class LambdaFilter(PipelineStep):
        def __init__(self, function):
            self.function = function

    class BaseFilter(PipelineStep):
        def __init__(self, exclusion_writer=None, batch_size: int = 1):
            self.exclusion_writer = exclusion_writer
            self.batch_size = batch_size

        def filter(self, doc):
            return True

        def filter_batch(self, batch):
            return [self.filter(doc) for doc in batch]

    class LocalPipelineExecutor:
        def __init__(self, pipeline, tasks: int = 1, workers: int = 1):
            self.pipeline = pipeline
            self.tasks = tasks
            self.workers = workers
            self.run_called = False

        def run(self):
            self.run_called = True

    @dataclass
    class Document:
        text: str
        id: str | None = None
        metadata: dict | None = None

    class DocStats(PipelineStep):
        def __init__(
            self,
            output_folder,
            groups_to_compute,
            histogram_round_digits: int = 3,
            top_k_config=None,
        ):
            self.output_folder = output_folder
            self.groups_to_compute = groups_to_compute
            self.histogram_round_digits = histogram_round_digits
            self.top_k_config = top_k_config

        def _get_empty_stats(self):
            return {
                "length": 0,
                "white_space_ratio": 0.0,
                "non_alpha_digit_ratio": 0.0,
                "digit_ratio": 0.0,
                "uppercase_ratio": 0.0,
                "elipsis_ratio": 0.0,
                "punctuation_ratio": 0.0,
                "label": "",
            }

    class TopKConfig(dict):
        pass

    def get_datafolder(value):
        return value

    executor_module.LocalPipelineExecutor = LocalPipelineExecutor
    readers_module.JsonlReader = JsonlReader
    filters_module.LanguageFilter = LanguageFilter
    filters_module.LambdaFilter = LambdaFilter
    filters_base_module.BaseFilter = BaseFilter
    writers_module.JsonlWriter = JsonlWriter
    writers_disk_base_module.DiskWriter = DiskWriter
    base_module.PipelineStep = PipelineStep
    stats_doc_module.DocStats = DocStats
    stats_config_module.DEFAULT_TOP_K_CONFIG = TopKConfig()
    stats_config_module.GROUP = str
    stats_config_module.TopKConfig = TopKConfig
    data_module.Document = Document
    data_module.DocumentsPipeline = Iterable[Document]
    io_module.DataFolderLike = str
    io_module.get_datafolder = get_datafolder
    utils_text_module.PUNCTUATION = list(string.punctuation)

    datatrove_module.executor = executor_module
    datatrove_module.pipeline = pipeline_module
    datatrove_module.data = data_module
    datatrove_module.io = io_module
    datatrove_module.utils = utils_module

    pipeline_module.readers = readers_module
    pipeline_module.filters = filters_module
    pipeline_module.writers = writers_module
    pipeline_module.base = base_module
    pipeline_module.stats = stats_module
    stats_module.doc_stats = stats_doc_module
    stats_module.config = stats_config_module
    filters_module.base_filter = filters_base_module
    writers_module.disk_base = writers_disk_base_module
    utils_module.text = utils_text_module

    sys.modules["datatrove"] = datatrove_module
    sys.modules["datatrove.executor"] = executor_module
    sys.modules["datatrove.pipeline"] = pipeline_module
    sys.modules["datatrove.pipeline.readers"] = readers_module
    sys.modules["datatrove.pipeline.filters"] = filters_module
    sys.modules["datatrove.pipeline.filters.base_filter"] = filters_base_module
    sys.modules["datatrove.pipeline.writers"] = writers_module
    sys.modules["datatrove.pipeline.writers.disk_base"] = writers_disk_base_module
    sys.modules["datatrove.pipeline.base"] = base_module
    sys.modules["datatrove.pipeline.stats"] = stats_module
    sys.modules["datatrove.pipeline.stats.doc_stats"] = stats_doc_module
    sys.modules["datatrove.pipeline.stats.config"] = stats_config_module
    sys.modules["datatrove.data"] = data_module
    sys.modules["datatrove.io"] = io_module
    sys.modules["datatrove.utils"] = utils_module
    sys.modules["datatrove.utils.text"] = utils_text_module


_install_regex_stub()
_install_loguru_stub()
_install_datatrove_stub()


def _load_jsonl_records(path: Path, limit: int | None = None) -> List[Dict]:
    records: List[Dict] = []
    with path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                records.append(json.loads(raw_line))
            except json.JSONDecodeError:
                continue
            if limit is not None and len(records) >= limit:
                break
    return records


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
            "esta selva selvaggia e aspra e forte che nel pensier rinova la paura!",
        },
        {
            "id": "it_colloquiale",
            "text": "Oggi sono andato al mercato e ho comprato delle mele bellissime. "
            "Il fruttivendolo mi ha fatto un ottimo prezzo perché erano le ultime. "
            "Domani tornerò a comprare anche le arance che erano molto buone.",
        },
        {
            "id": "it_romanesco",
            "text": "Aò, che bella giornata oggi! Er sole picchia forte già da le otto. "
            "Me so' svejato co' 'na scimmia addosso che la metà basta, ma appena "
            "ho visto 'sta luce, m'è tornata la voja de campà.",
        },
        {
            "id": "it_formale",
            "text": "La presente comunicazione ha lo scopo di informare tutti i dipendenti "
            "che a partire dal prossimo mese entreranno in vigore le nuove normative "
            "relative alla sicurezza sul lavoro. Si prega di prendere visione del documento.",
        },
    ]


@pytest.fixture
def non_italian_texts() -> List[Dict[str, str]]:
    """Testi non italiani che dovrebbero essere filtrati."""
    return [
        {
            "id": "en_sample",
            "text": "The quick brown fox jumps over the lazy dog. This is a sample text "
            "in English that should be filtered out by the Italian language detector.",
        },
        {
            "id": "es_sample",
            "text": "Hola, cómo estás? Hoy hace un día muy bonito para pasear por el parque. "
            "Los pájaros cantan y las flores están floreciendo en primavera.",
        },
        {
            "id": "fr_sample",
            "text": "Bonjour, comment allez-vous? Aujourd'hui il fait très beau pour se promener "
            "dans le parc. Les oiseaux chantent et les fleurs sont en pleine floraison.",
        },
        {
            "id": "pt_sample",
            "text": "Olá, como você está? Hoje é um dia muito bonito para passear no parque. "
            "Os pássaros cantam e as flores estão florescendo na primavera.",
        },
    ]


@pytest.fixture
def noise_texts() -> List[Dict[str, str]]:
    """Testi rumorosi che dovrebbero essere filtrati."""
    return [
        {"id": "noise_empty", "text": ""},
        {"id": "noise_short", "text": "ab"},
        {
            "id": "noise_html",
            "text": "<div class='footer'>© 2026 • P.IVA • Contatti • Cookie • Privacy</div>",
        },
        {
            "id": "noise_code",
            "text": "function test() { return x => x * 2; } class MyClass { constructor() {} }",
        },
        {
            "id": "noise_repetition",
            "text": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa bbbbbbbbbbbbbbbbbbbbbbbbb",
        },
        {
            "id": "noise_log",
            "text": "INFO 2026-01-22 17:34:28 - User logged in from IP 192.168.1.1. Status: OK. "
            "INFO 2026-01-22 17:34:28 - User logged in from IP 192.168.1.1. Status: OK.",
        },
        {
            "id": "noise_boilerplate",
            "text": "Accetta i cookie e iscriviti alla newsletter per ricevere pubblicità personalizzata",
        },
    ]


@pytest.fixture
def training_data_path() -> Path:
    return PROJECT_ROOT / "data" / "train" / "training_data.jsonl"


@pytest.fixture
def hand_label_path() -> Path:
    return PROJECT_ROOT / "data" / "train" / "hand_label.jsonl"


@pytest.fixture
def training_good_docs(training_data_path: Path) -> List[Dict]:
    """Campione reale di documenti italiani dal dataset di training."""
    return _load_jsonl_records(training_data_path, limit=8)


@pytest.fixture
def hand_labeled_docs(hand_label_path: Path) -> List[Dict]:
    """Campione reale dal file annotato manualmente, saltando eventuali righe malformate."""
    return _load_jsonl_records(hand_label_path, limit=8)


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

    output_data = [
        {"text": "Testo buono di esempio", "metadata": {"tag": "good"}},
        {"text": "Altro testo buono", "metadata": {"tag": "good"}},
        {"text": "Testo medio", "metadata": {"tag": "middle"}},
    ]

    rejected_data = [
        {"text": "Testo scartato", "metadata": {"tag": "bad"}},
        {"text": "Altro scarto", "metadata": {"tag": "bad"}},
    ]

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
    env_vars = [
        "ROOT_DIR",
        "DATA_DIR",
        "OUTPUT_DIR",
        "REJECTED_DIR",
        "CSV_DIR",
        "FEATURE_DIR",
        "MODEL_PATH",
    ]
    original = {var: os.environ.get(var) for var in env_vars}

    for var in env_vars:
        if var in os.environ:
            del os.environ[var]

    yield

    for var, value in original.items():
        if value is not None:
            os.environ[var] = value
        elif var in os.environ:
            del os.environ[var]
