"""
Fixtures condivise per i test della pipeline ITA-LLM.
"""

import gzip
import importlib.util
import json
import os
import pickle
import re as stdlib_re
import string
import sys
import tempfile
import types
from contextlib import nullcontext
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
    utils_lid_module = types.ModuleType("datatrove.utils.lid")
    utils_stats_module = types.ModuleType("datatrove.utils.stats")
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

        def track_time(self):
            return nullcontext()

    class TopKConfig(dict):
        pass

    class DataFolder:
        def __init__(self, path: str):
            self.path = str(path)

        def open(self, filename: str, mode: str = "rt", encoding: str = "utf-8"):
            file_path = Path(self.path) / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            return file_path.open(mode, encoding=encoding)

    def get_datafolder(value):
        return value if hasattr(value, "open") else DataFolder(value)

    class FT176LID:
        def __init__(self, languages: str = "it"):
            self.languages = languages

        def predict(self, doc):
            metadata = getattr(doc, "metadata", {}) or {}
            return self.languages, float(metadata.get("language_score", 0.99))

    class PipelineStats:
        def __init__(self, pipeline):
            self.pipeline = pipeline
            self.stats = []

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
    utils_lid_module.FT176LID = FT176LID
    utils_stats_module.PipelineStats = PipelineStats

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
    utils_module.lid = utils_lid_module
    utils_module.stats = utils_stats_module

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
    sys.modules["datatrove.utils.lid"] = utils_lid_module
    sys.modules["datatrove.utils.stats"] = utils_stats_module
    sys.modules["datatrove.utils.text"] = utils_text_module


def _install_joblib_stub() -> None:
    if "joblib" in sys.modules or importlib.util.find_spec("joblib") is not None:
        return

    joblib_module = types.ModuleType("joblib")

    def dump(obj, path):
        with open(path, "wb") as handle:
            pickle.dump(obj, handle)
        return path

    def load(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)

    joblib_module.dump = dump
    joblib_module.load = load
    sys.modules["joblib"] = joblib_module


def _install_sklearn_stub() -> None:
    if "sklearn" in sys.modules or importlib.util.find_spec("sklearn") is not None:
        return

    import copy
    import numpy as np

    sklearn_module = types.ModuleType("sklearn")
    base_module = types.ModuleType("sklearn.base")
    ensemble_module = types.ModuleType("sklearn.ensemble")
    linear_model_module = types.ModuleType("sklearn.linear_model")
    model_selection_module = types.ModuleType("sklearn.model_selection")
    pipeline_module = types.ModuleType("sklearn.pipeline")
    preprocessing_module = types.ModuleType("sklearn.preprocessing")
    metrics_module = types.ModuleType("sklearn.metrics")
    inspection_module = types.ModuleType("sklearn.inspection")

    class _SimpleEstimator:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self._positive_rate = 0.5

        def fit(self, X, y):
            values = np.asarray(y, dtype=float)
            self._positive_rate = float(values.mean()) if len(values) else 0.5
            return self

        def predict_proba(self, X):
            rows = len(X)
            positive = min(max(self._positive_rate, 0.0), 1.0)
            negative = 1.0 - positive
            return np.tile(np.array([[negative, positive]], dtype=float), (rows, 1))

    class RandomForestClassifier(_SimpleEstimator):
        pass

    class ExtraTreesClassifier(_SimpleEstimator):
        pass

    class LogisticRegression(_SimpleEstimator):
        pass

    class StandardScaler:
        def fit(self, X):
            return self

        def transform(self, X):
            return np.asarray(X, dtype=float)

        def fit_transform(self, X):
            self.fit(X)
            return self.transform(X)

    class Pipeline:
        def __init__(self, steps):
            self.steps = steps

        def fit(self, X, y):
            self.steps[-1][1].fit(X, y)
            return self

        def predict_proba(self, X):
            return self.steps[-1][1].predict_proba(X)

    class StratifiedKFold:
        def __init__(self, n_splits=5, shuffle=False, random_state=None):
            self.n_splits = n_splits

        def split(self, X, y):
            indices = np.arange(len(y))
            folds = np.array_split(indices, self.n_splits)
            for fold_idx in range(self.n_splits):
                val_idx = folds[fold_idx]
                train_idx = np.concatenate(
                    [fold for idx, fold in enumerate(folds) if idx != fold_idx]
                )
                yield train_idx, val_idx

    def train_test_split(X, y, test_size=0.2, stratify=None, random_state=None):
        split_idx = max(1, int(len(X) * (1 - test_size)))
        return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]

    def clone(estimator):
        return copy.deepcopy(estimator)

    def _to_numpy(values):
        return np.asarray(values, dtype=float)

    def confusion_matrix(y_true, y_pred):
        y_true = _to_numpy(y_true).astype(int)
        y_pred = _to_numpy(y_pred).astype(int)
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        return np.array([[tn, fp], [fn, tp]])

    def accuracy_score(y_true, y_pred):
        y_true = _to_numpy(y_true)
        y_pred = _to_numpy(y_pred)
        return float((y_true == y_pred).mean()) if len(y_true) else 0.0

    def balanced_accuracy_score(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        tn, fp = cm[0]
        fn, tp = cm[1]
        tnr = tn / (tn + fp) if (tn + fp) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        return float((tnr + tpr) / 2)

    def precision_score(y_true, y_pred, zero_division=0):
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fp = cm[0, 1]
        return float(tp / (tp + fp)) if (tp + fp) else float(zero_division)

    def recall_score(y_true, y_pred, zero_division=0):
        cm = confusion_matrix(y_true, y_pred)
        tp = cm[1, 1]
        fn = cm[1, 0]
        return float(tp / (tp + fn)) if (tp + fn) else float(zero_division)

    def f1_score(y_true, y_pred, zero_division=0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        if precision + recall == 0:
            return float(zero_division)
        return float(2 * precision * recall / (precision + recall))

    def roc_auc_score(y_true, y_score):
        y_true = _to_numpy(y_true)
        y_score = _to_numpy(y_score)
        if len(np.unique(y_true)) < 2:
            return 0.5
        positive = y_score[y_true == 1]
        negative = y_score[y_true == 0]
        wins = sum(float(p > n) + 0.5 * float(p == n) for p in positive for n in negative)
        total = len(positive) * len(negative)
        return float(wins / total) if total else 0.5

    def roc_curve(y_true, y_score):
        y_true = _to_numpy(y_true)
        y_score = _to_numpy(y_score)
        thresholds = np.array([np.inf, 0.5, -np.inf])
        points = []
        for threshold in thresholds:
            y_pred = (y_score >= threshold).astype(int)
            cm = confusion_matrix(y_true, y_pred)
            tn, fp = cm[0]
            fn, tp = cm[1]
            fpr = fp / (fp + tn) if (fp + tn) else 0.0
            tpr = tp / (tp + fn) if (tp + fn) else 0.0
            points.append((fpr, tpr))
        fpr, tpr = zip(*points)
        return np.array(fpr), np.array(tpr), thresholds

    def precision_recall_curve(y_true, y_score):
        y_true = _to_numpy(y_true)
        y_score = _to_numpy(y_score)
        thresholds = np.array([0.5])
        y_pred = (y_score >= 0.5).astype(int)
        precision = precision_score(y_true, y_pred, zero_division=1)
        recall = recall_score(y_true, y_pred, zero_division=0)
        return np.array([precision, 1.0]), np.array([recall, 0.0]), thresholds

    def classification_report(y_true, y_pred, target_names=None, output_dict=False, zero_division=0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        f1 = f1_score(y_true, y_pred, zero_division=zero_division)
        accuracy = accuracy_score(y_true, y_pred)
        if output_dict:
            return {
                "accuracy": accuracy,
                "weighted avg": {
                    "precision": precision,
                    "recall": recall,
                    "f1-score": f1,
                },
            }
        return (
            "classification report\n"
            f"accuracy: {accuracy:.4f}\n"
            f"precision: {precision:.4f}\n"
            f"recall: {recall:.4f}\n"
            f"f1-score: {f1:.4f}\n"
        )

    class _PermutationResult:
        def __init__(self, feature_count):
            self.importances_mean = np.zeros(feature_count)
            self.importances_std = np.zeros(feature_count)

    def permutation_importance(model, X, y, n_repeats=10, random_state=None, n_jobs=None):
        feature_count = getattr(X, "shape", [0, 0])[1] if len(getattr(X, "shape", [])) > 1 else 0
        return _PermutationResult(feature_count)

    base_module.clone = clone
    ensemble_module.ExtraTreesClassifier = ExtraTreesClassifier
    ensemble_module.RandomForestClassifier = RandomForestClassifier
    linear_model_module.LogisticRegression = LogisticRegression
    model_selection_module.StratifiedKFold = StratifiedKFold
    model_selection_module.train_test_split = train_test_split
    pipeline_module.Pipeline = Pipeline
    preprocessing_module.StandardScaler = StandardScaler
    metrics_module.classification_report = classification_report
    metrics_module.confusion_matrix = confusion_matrix
    metrics_module.roc_auc_score = roc_auc_score
    metrics_module.roc_curve = roc_curve
    metrics_module.precision_recall_curve = precision_recall_curve
    metrics_module.f1_score = f1_score
    metrics_module.accuracy_score = accuracy_score
    metrics_module.balanced_accuracy_score = balanced_accuracy_score
    metrics_module.precision_score = precision_score
    metrics_module.recall_score = recall_score
    inspection_module.permutation_importance = permutation_importance

    sklearn_module.base = base_module
    sklearn_module.ensemble = ensemble_module
    sklearn_module.linear_model = linear_model_module
    sklearn_module.model_selection = model_selection_module
    sklearn_module.pipeline = pipeline_module
    sklearn_module.preprocessing = preprocessing_module
    sklearn_module.metrics = metrics_module
    sklearn_module.inspection = inspection_module

    sys.modules["sklearn"] = sklearn_module
    sys.modules["sklearn.base"] = base_module
    sys.modules["sklearn.ensemble"] = ensemble_module
    sys.modules["sklearn.linear_model"] = linear_model_module
    sys.modules["sklearn.model_selection"] = model_selection_module
    sys.modules["sklearn.pipeline"] = pipeline_module
    sys.modules["sklearn.preprocessing"] = preprocessing_module
    sys.modules["sklearn.metrics"] = metrics_module
    sys.modules["sklearn.inspection"] = inspection_module


def _install_lightgbm_stub() -> None:
    if "lightgbm" in sys.modules or importlib.util.find_spec("lightgbm") is not None:
        return

    import numpy as np

    lightgbm_module = types.ModuleType("lightgbm")

    class LGBMClassifier:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs
            self._positive_rate = 0.5

        def fit(self, X, y):
            values = np.asarray(y, dtype=float)
            self._positive_rate = float(values.mean()) if len(values) else 0.5
            return self

        def predict_proba(self, X):
            rows = len(X)
            positive = min(max(self._positive_rate, 0.0), 1.0)
            negative = 1.0 - positive
            return np.tile(np.array([[negative, positive]], dtype=float), (rows, 1))

    lightgbm_module.LGBMClassifier = LGBMClassifier
    sys.modules["lightgbm"] = lightgbm_module


_install_regex_stub()
_install_loguru_stub()
_install_datatrove_stub()
_install_joblib_stub()
_install_sklearn_stub()
_install_lightgbm_stub()


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
    preferred_paths = [
        PROJECT_ROOT / "data" / "train" / "training_data.jsonl",
        PROJECT_ROOT / "data" / "train" / "shard_0000.jsonl",
    ]
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


@pytest.fixture
def hand_label_path() -> Path:
    preferred_paths = [
        PROJECT_ROOT / "data" / "train" / "hand_label.jsonl",
        PROJECT_ROOT / "data" / "spam" / "spam_dataset_300.jsonl",
        PROJECT_ROOT / "data" / "train" / "shard_0000.jsonl",
    ]
    for path in preferred_paths:
        if path.exists():
            return path
    return preferred_paths[0]


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
