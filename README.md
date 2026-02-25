# ITA-LLM-Pipeline

Pipeline di elaborazione dati per Large Language Models italiani, basata su [DataTrove](https://github.com/huggingface/datatrove).

Questo progetto permette di filtrare e processare dataset in formato JSONL applicando filtri di qualità (es. FineWebQualityFilter) e generando output organizzati con classificazione automatica.

---

## 📋 Indice

- [Requisiti](#-requisiti)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Configurazione](#-configurazione)
- [Avvio tramite Docker](#-avvio-tramite-docker)
- [Avvio tramite CLI (Locale)](#-avvio-tramite-cli-locale)
- [Argomenti CLI](#-argomenti-cli)
- [Testing](#-testing)
- [Output](#-output)

---

## 📦 Requisiti

### Per esecuzione Docker:
- Docker >= 20.10
- Docker Compose >= 2.0

### Per esecuzione locale:
- Python 3.12+
- pip

---

## 📁 Struttura del Progetto

```
ita-llm-pipeline/
├── configs/
│   └── default.conf          # File di configurazione delle variabili d'ambiente
├── data/
│   ├── mixed.jsonl           # Dataset di esempio
│   ├── test1.jsonl
│   └── test2.jsonl
├── src/
│   ├── main.py               # Entry point principale della pipeline
│   ├── run_local_it.py       # Script alternativo con filtri italiani custom
│   ├── blocks/
│   └── utils/
│       └── output_organizer.py   # Utility per classificazione output
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## ⚙️ Configurazione

Il file `configs/default.conf` contiene le variabili d'ambiente per la pipeline:

```properties
# Colorazione logs (0=disabilitata, 1=abilitata)
DATATROVE_COLORIZE_LOGS=0
DATATROVE_COLORIZE_LOG_FILES=0

# Directory della pipeline
ROOT_DIR=/home/stefano/ita-llm-pipeline
DATA_DIR=/home/stefano/ita-llm-pipeline/data
OUTPUT_DIR=/home/stefano/output_data
REJECTED_DIR=/home/stefano/output_data/rejected
```

> **Nota:** Per l'esecuzione Docker, queste variabili vengono sovrascritte automaticamente con i path interni al container (`/app/...`).

---

## 🐳 Avvio tramite Docker

### 1. Build dell'immagine

```bash
docker-compose build
```

### 2. Avvio della pipeline

```bash
docker-compose up
```

### 3. Avvio in background (detached mode)

```bash
docker-compose up -d
```

### 4. Visualizzazione logs

```bash
docker-compose logs -f
```

### 5. Stop del container

```bash
docker-compose down
```

### Volumi montati

| Host | Container | Modalità | Descrizione |
|------|-----------|----------|-------------|
| `./data` | `/app/data` | read-only | Dati in input |
| `./output` | `/app/output` | read-write | Output della pipeline |
| `./logs` | `/app/logs` | read-write | File di log |

### Personalizzazione del comando Docker

Puoi sovrascrivere il comando di default modificando `docker-compose.yml`:

```yaml
command: ["python3", "src/main.py", "--config", "configs/default.conf"]
```

Oppure eseguendo direttamente:

```bash
docker-compose run pipeline python3 src/main.py --config configs/default.conf
```

---

## 💻 Avvio tramite CLI (Locale)

### 1. Creazione ambiente virtuale

```bash
python3.12 -m venv datatrove_venv
source datatrove_venv/bin/activate
```

### 2. Installazione dipendenze

```bash
pip install -r requirements.txt
```

### 3. Esecuzione della pipeline

#### Metodo base (con configurazione di default):

```bash
python3 src/main.py --config configs/default.conf
```

#### Specificando directory custom:

```bash
python3 src/main.py \
    --root-dir /path/to/project \
    --output-dir /path/to/output \
    --rejected-dir /path/to/rejected \
    --csv-dir /path/to/csv
```

#### Esempio completo:

```bash
python3 src/main.py \
    --config configs/default.conf \
    --root-dir ~/ita-llm-pipeline \
    --output-dir ~/output_data \
    --rejected-dir ~/output_data/rejected \
    --csv-dir ~/output_data/csv
```

### 4. Esecuzione script alternativo (filtri italiani custom)

```bash
python3 src/run_local_it.py
```

---

## 🔧 Argomenti CLI

| Argomento | Tipo | Default | Descrizione |
|-----------|------|---------|-------------|
| `--config` | `str` | `None` | Path relativo al file di configurazione (es. `configs/default.conf`) |
| `--root-dir` | `str` | `~/ita-llm-pipeline` (locale) o `/app` (Docker) | Path assoluto alla root del progetto |
| `--output-dir` | `str` | `~/output_data` (locale) o `/app/output` (Docker) | Path assoluto alla directory di output |
| `--rejected-dir` | `str` | `<output-dir>/rejected` | Path per i documenti scartati dai filtri |
| `--csv-dir` | `str` | `<output-dir>/csv` | Path per i file CSV di classificazione |

### Priorità delle variabili

1. **Variabili d'ambiente** (da file config) → Priorità massima
2. **Argomenti CLI** → Usati se le variabili d'ambiente non sono impostate
3. **Valori di default** → Usati come fallback

---

## 🧪 Testing

Il progetto include una suite di test basata su pytest.

### Struttura dei test

```
tests/
├── conftest.py                # Fixtures condivise (dati test, temp dirs)
├── test_main.py               # Test per load_config() e extract_args()
├── test_run_local_it.py       # Test per le funzioni di filtering italiano
└── test_output_organizer.py   # Test per utils/output_organizer.py
```

### Installazione dipendenze di sviluppo

```bash
pip install -r requirements-dev.txt
```

### Esecuzione dei test

#### Esegui tutti i test

```bash
pytest
```

#### Esegui con output verbose

```bash
pytest -v
```

#### Esegui test specifici

```bash
# Test di un singolo file
pytest tests/test_run_local_it.py

# Test di una singola classe
pytest tests/test_run_local_it.py::TestItalianHeuristicScore

# Test di una singola funzione
pytest tests/test_run_local_it.py::TestItalianHeuristicScore::test_high_score_for_italian
```

#### Esegui con coverage report

```bash
# Coverage in terminale
pytest --cov=src

# Coverage con report HTML
pytest --cov=src --cov-report=html
# Apri htmlcov/index.html nel browser
```

#### Esegui test in parallelo

```bash
pytest -n auto
```

#### Esegui solo test veloci (escludendo quelli lenti)

```bash
pytest -m "not slow"
```

### Test coperti

| Modulo | Funzioni testate |
|--------|------------------|
| `main.py` | `load_config()`, `extract_args()` |
| `run_local_it.py` | `basic_normalize()`, `char_stats()`, `token_stats()`, `bigram_stats()`, `italian_heuristic_score()`, `detect_noise()`, `is_italian()` |
| `output_organizer.py` | `find_single_jsonl_gz()`, `iter_jsonl_gz()`, `group_texts_by_tag()`, `write_grouped_csv()` |

---

## 📤 Output

La pipeline genera i seguenti output:

### Directory di output (`OUTPUT_DIR`)

```
output/
├── risultati_0.jsonl         # Documenti che hanno passato i filtri
├── rejected/
│   └── risultati_0.jsonl     # Documenti scartati dai filtri
└── csv/
    └── classification.csv    # Classificazione dei risultati
```

### Logs

```
logs/
└── [timestamp]/
    └── pipeline.log          # Log dettagliati dell'esecuzione
```

---

## 🔍 Esempio di dati in input

I file JSONL devono avere il seguente formato:

```json
{"text": "Testo del documento in italiano...", "id": "doc_001"}
{"text": "Altro documento da processare...", "id": "doc_002"}
```

---


## 📝 Note

- Il progetto rileva automaticamente se è in esecuzione dentro Docker e adatta i path di conseguenza
- I logs colorati possono essere abilitati impostando `DATATROVE_COLORIZE_LOGS=1`
- Per elaborazioni parallele, modificare i parametri `tasks` e `workers` in `LocalPipelineExecutor`

---

## 📄 Licenza

[Inserire licenza]

---

## 👥 Contributi

[Inserire linee guida per i contributi]
