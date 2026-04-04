# ITA-LLM-Pipeline

Pipeline avanzata di elaborazione e pulizia dati per Large Language Models italiani, basata su [DataTrove](https://github.com/huggingface/datatrove).

Fornisce un sistema completo di filtri linguistici, classificatori ML di qualità e estrazione di feature statistiche per il preprocessing di dataset JSONL in italiano.

---

## 📋 Indice

- [Funzionalità Principali](#-funzionalità-principali)
- [Requisiti](#-requisiti)
- [Struttura del Progetto](#-struttura-del-progetto)
- [Configurazione](#-configurazione)
- [Avvio tramite Docker](#-avvio-tramite-docker)
- [Avvio tramite CLI (Locale)](#-avvio-tramite-cli-locale)
- [Componenti della Pipeline](#-componenti-della-pipeline)
- [Testing](#-testing)
- [Output e Struttura Risultati](#-output-e-struttura-risultati)

---

## ✨ Funzionalità Principali

- **Filtro Linguistico**: Identificazione e scarto automatico di documenti non in italiano usando fastText
- **Classificazione della Qualità**: Modello LightGBM addestrato per classificare documenti come "buoni" o "cattivi"
- **Estrazione di Feature**: Calcolo automatico di metriche statistiche per ogni documento:
  - Lunghezza e rapporto di whitespace
  - Rapporto di caratteri speciali e digit
  - Rapporto maiuscole, ellissi e punteggiatura
  - E molte altre feature rilevanti
- **Rilevamento Spam**: Estrazione feature specifiche per identificazione e scarto di contenuti spam (opzionale)
- **Statistiche Documentali**: Generazione automatica di report CSV con metriche aggregate
- **Organizzazione Automatica**: Scarto e archiviazione strutturata di documenti rifiutati
- **Reporting Completo**: Statistiche di esecuzione dettagliate per ogni fase della pipeline

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
│   ├── default.conf                # Configurazione per esecuzione locale
│   ├── gabriele.conf
│   ├── silvio.conf
│   └── stefano.conf
├── data/
│   ├── rp_normalized.jsonl         # Dataset di input
│   ├── test/                       # Dataset di test
│   │   ├── dataset_test.jsonl
│   │   ├── dataset_train.jsonl
│   │   └── dataset_validation.jsonl
│   └── train/                      # Dataset di training
│       ├── hand_label.jsonl
│       ├── it_head_0001.jsonl
│       └── rpDataset.jsonl
├── models/
│   ├── lgbm_quality_model.joblib   # Modello LightGBM per classificazione qualità
│   └── spam_lgbm.joblib            # Modello per rilevamento spam
├── src/
│   ├── main.py                     # Entry point principale della pipeline
│   ├── run_local_it.py             # Script con filtri italiani custom
│   ├── config_loader.py            # Caricamento file di configurazione
│   ├── pipeline_factory.py         # Costruzione modulare della pipeline
│   ├── blocks/
│   │   ├── readers.py              # Lettori JSONL
│   │   ├── writers.py              # Writer JSONL
│   │   ├── filters.py              # Filtri (linguistici, custom)
│   │   ├── classifiers.py          # Classificatori ML
│   │   ├── stats.py                # Estrazione statistiche
│   │   └── spam_classifier/        # Modulo rilevamento spam
│   │       ├── spam_classifier.py
│   │       ├── spam_stats.py
│   │       └── spam_keywords.py
│   └── utils/
│       ├── output_organizer.py     # Organizzazione output e analisi risultati
│       ├── training_lgbmclassifier.py
│       ├── training_spam_lgbmclassifier.py
│       └── web_extracting_pipeline.py
├── tests/
│   ├── conftest.py
│   ├── test_main.py
│   ├── test_pipeline_components.py
│   ├── test_output_organizer.py
│   └── test_run_local_it.py
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## ⚙️ Configurazione

Il progetto utilizza file di configurazione `.conf` per gestire le variabili di ambiente. 

### File di Configurazione

Ogni file di configurazione contiene:

```properties
# Colorazione logs (0=disabilitata, 1=abilitata)
DATATROVE_COLORIZE_LOGS=0
DATATROVE_COLORIZE_LOG_FILES=0

# Directory della pipeline
ROOT_DIR=/path/to/project
DATA_DIR=/path/to/input/data
OUTPUT_DIR=/path/to/output
REJECTED_DIR=/path/to/output/rejected
MODEL_PATH=/path/to/models
FEATURE_DIR=/path/to/output/feature
```

### Variabili Disponibili

| Variabile | Descrizione | Default |
|-----------|-------------|---------|
| `DATA_DIR` | Directory con dataset JSONL in input | `./data` |
| `OUTPUT_DIR` | Directory per output filtrati | `./output` |
| `REJECTED_DIR` | Directory per documenti scartati | `./output/rejected` |
| `MODEL_PATH` | Directory con modelli ML (.joblib) | `./models` |
| `FEATURE_DIR` | Directory per CSV con feature e statistiche | `./output/feature` |
| `DATATROVE_COLORIZE_LOGS` | Abilita colori nei log (Docker: 0) | `0` |

### Esecuzione con Profili Diversi

Per eseguire con un profilo di configurazione specifico, usa:

**Docker:**
```bash
docker-compose --file docker-compose.yml run pipeline \
  python3 src/main.py --config configs/stefano.conf
```

**Local:**
```bash
python3 src/main.py --config configs/stefano.conf
```

---

## 🐳 Avvio tramite Docker

### Prerequisiti

- Docker >= 20.10
- Docker Compose >= 2.0

### Quick Start

```bash
# Build dell'immagine
docker-compose build

# Esecuzione della pipeline
docker-compose up
```

### Comandi Utili

```bash
# Avvio in background (detached mode)
docker-compose up -d

# Visualizzazione log in tempo reale
docker-compose logs -f

# Stop e pulizia container
docker-compose down

# Avvio con configurazione specifica
docker-compose run --rm pipeline \
  python3 src/main.py --config configs/stefano.conf

# Esecuzione con bash per debug
docker-compose run --rm pipeline bash
```

### Volumi Montati

| Host | Container | Modalità | Descrizione |
|------|-----------|----------|-------------|
| `./data` | `/app/data` | read-only | Dataset in input |
| `./output` | `/app/output` | read-write | Output filtrati |
| `./logs` | `/app/logs` | read-write | File di log |
| `./models` | `/app/models` | read-only | Modelli ML |
| `./configs` | `/app/configs` | read-only | File configurazione |

---

## 💻 Avvio tramite CLI (Locale)

### Prerequisiti

- Python 3.12+
- pip

### Setup dell'Ambiente

```bash
# 1. Creare ambiente virtuale
python3.12 -m venv datatrove_venv

# 2. Attivare ambiente
source datatrove_venv/bin/activate  # Linux/Mac
# o
.\datatrove_venv\Scripts\activate  # Windows

# 3. Installare dipendenze
pip install -r requirements.txt
```

### Esecuzione della Pipeline

#### Modo base (con configurazione default)

```bash
python3 src/main.py --config configs/default.conf
```

#### Modo personalizzato

```bash
python3 src/main.py \
    --config configs/stefano.conf \
    --root-dir ~/ita-llm-pipeline \
    --output-dir ~/output_data \
    --rejected-dir ~/output_data/rejected
```

### Struttura dell'Esecuzione

La pipeline segue questi step:

1. **Lettura**: Carica file JSONL da `DATA_DIR`
2. **Filtro Linguistico**: Identifica e scarta i non-italiani (soglia default: 0.75)
3. **Estrazione Feature Spam**: Calcola feature specifiche per spam detection
4. **Statistiche Documentali**: Estrae metriche per ogni documento (CSV)
5. **Classificazione Qualità**: Applica modello LightGBM per classificare qualità
6. **Scrittura Finale**: Salva documenti filtrati in JSONL
7. **Analisi Risultati**: Organizza documenti scartati nelle cartelle di rifiuto

## 🔧 Componenti della Pipeline

### 1. Reader (`blocks/readers.py`)
- **Funzione**: Legge file JSONL dalla directory specificata
- **Output**: DocumentsPipeline pronto per elaborazione
- **Formato**: Supporta JSONL standard DataTrove

### 2. Language Filter (`blocks/filters.py`)
- **Modello**: fastText
- **Funzione**: Identifica e scarta documenti non in italiano
- **Soglia default**: 0.75 (confidenza lingua italiana)
- **Output**: Documenti scartati in `rejected/1_language/`

### 3. Spam Feature Extractor (`blocks/spam_classifier/spam_stats.py`)
- **Funzione**: Estrae feature specifiche per rilevamento spam
- **Feature**: Keyword patterns, densità, struttura testuale
- **Output**: Metadata arricchito per documento

### 4. Spam Feature CSV Writer (`blocks/spam_classifier/spam_stats.py`)
- **Funzione**: Esporta feature spam in formato CSV
- **Output**: `output/feature/spam_doc_features.csv`
- **Utilità**: Analisi e debug spam detection

### 5. Document Statistics (`blocks/stats.py`)
- **Funzione**: Calcola metriche statistiche per ogni documento
- **Feature**: Lunghezza, whitespace ratio, digit ratio, uppercase ratio, ellissi, punteggiatura, ecc.
- **Output**: `output/feature/doc_stats_per_file.csv`

### 6. Quality Classifier (`blocks/classifiers.py`)
- **Modello**: LightGBM pre-addestrato
- **Percorso**: `models/lgbm_quality_model.joblib`
- **Soglia default**: 0.7
- **Input**: Feature calcolate al step 5
- **Output**: Classificazione "buono" o "cattivo" + scarto in `rejected/quality/`
- **Feature**: Lunghezza, rapporti caratteri, metriche di qualità

### 7. Writer Finale (`blocks/writers.py`)
- **Funzione**: Scrive documenti approvati in JSONL finale
- **Output**: `output/italiano_pulito_*.jsonl`
- **Formato**: JSONL standard con metadata

### 8. Output Organizer (`utils/output_organizer.py`)
- **Funzione**: Analizza ed organizza documenti scartati
- **Output**: Statistiche e report in `output/feature/`
- **Genera**: Analisi dettagliata dei motivi di scarto

---

## 🧪 Testing

Il progetto include una suite completa di test basata su pytest.

### Installazione dipendenze di sviluppo

```bash
pip install -r requirements-dev.txt
```

### Esecuzione dei test

#### Esegui tutti i test

```bash
pytest
pytest -v  # Con output verbose
```

#### Esegui i test via Docker

```bash
docker compose --profile test run --rm test
```

#### Esegui test specifici

```bash
# Test di un singolo file
pytest tests/test_main.py

# Con output verbose
pytest tests/test_pipeline_components.py -v
```

#### Esegui con coverage report

```bash
# Coverage in terminale
pytest --cov=src

# Coverage con report HTML
pytest --cov=src --cov-report=html
# Apri htmlcov/index.html nel browser
```

### Test disponibili

| File | Copertura |
|------|-----------|
| `test_main.py` | Entry point e caricamento configurazione |
| `test_pipeline_components.py` | Componenti della pipeline (reader, writer, filter) |
| `test_run_local_it.py` | Filtri italiani custom e scoring |
| `test_output_organizer.py` | Organizzazione output e analisi risultati |

---

## 📤 Output e Struttura Risultati

La pipeline produce diversi output organizzati in cartelle strutturate.

### Directory di Output Principale (`OUTPUT_DIR`)

```
output/
├── italiano_pulito_00000.jsonl     # Documenti approvati (passa tutti i filtri)
├── feature/                         # Statistiche e feature estratte
│   ├── spam_doc_features.csv       # Feature spam per ogni documento
│   ├── doc_stats_per_file.csv      # Statistiche documentali aggregate
│   └── classification_report.json  # Report di classificazione
└── rejected/                        # Documenti scartati divisi per motivo
    ├── 1_language/
    │   └── non_italiano_0.jsonl    # Documenti non in italiano
    ├── quality/
    │   └── low_quality_0.jsonl     # Documenti a bassa qualità
    └── stats/
        └── rejection_analysis.csv  # Analisi dettagliata scarti
```

### File CSV di Feature

#### `spam_doc_features.csv`
Feature estratte per rilevamento spam:
- Parole chiave spam
- Densità link
- Struttura testuale sospetta
- Pattern email/numero

#### `doc_stats_per_file.csv`
Statistiche per ogni documento:
- `length`: Lunghezza totale
- `white_space_ratio`: Rapporto spazi bianchi
- `non_alpha_digit_ratio`: Rapporto caratteri non-alfanumerici
- `digit_ratio`: Rapporto digit
- `uppercase_ratio`: Rapporto maiuscole
- `elipsis_ratio`: Rapporto ellissi
- `punctuation_ratio`: Rapporto punteggiatura

### Logs di Esecuzione

```
logs/
└── YYYY-MM-DD_HH-MM-SS_XXXXX/
    ├── pipeline.log                # Log complessivo
    └── step_[numero].log           # Log per singolo step
```

### Report di Esecuzione

Al termine della pipeline, viene stampato un report che contiene:
- Tempo totale di esecuzione
- Tempo per ogni step
- Metriche di processing (documenti processati, scarti, approvati)
- Statistiche dettagliate per ogni componente

Viene anche salvato un file JSON con le statistiche:
```
pipeline_stats.json                 # Statistiche in formato JSON
```

### Statistiche di Output

La pipeline conta e reporta:
- **Documenti processati**: Totale input
- **Documenti approvati**: Passa tutti i filtri
- **Documenti scartati per lingua**: Non italiano
- **Documenti scartati per qualità**: Sotto soglia LightGBM
- **Percentuale di approvazione**: % documenti mantenuti

---

## 📋 Formato Input

I file JSONL devono avere il seguente formato minimo:

```json
{"text": "Testo del documento in italiano...", "id": "doc_001"}
{"text": "Altro documento da processare...", "id": "doc_002"}
{"text": "Ancora un documento...", "url": "https://example.com"}
```

Campi supportati:
- `text` (obbligatorio): Contenuto testuale
- `id` (consigliato): Identificatore unico
- `url`: URL sorgente (opzionale)
- `metadata`: Metadata aggiuntivi (opzionale)

---

## 🎯 Utilizzo Tipico

### 1. Preparazione dati

```bash
# Verifica il tuo file JSONL
head -5 data/mio_dataset.jsonl
```

### 2. Configurazione

```bash
# Crea una copia del file config
cp configs/default.conf configs/mio_progetto.conf

# Modifica i path
nano configs/mio_progetto.conf
```

### 3. Esecuzione

```bash
# Local
python3 src/main.py --config configs/mio_progetto.conf

# Oppure Docker
docker-compose run --rm pipeline \
  python3 src/main.py --config configs/mio_progetto.conf
```

### 4. Analisi Risultati

```bash
# Controlla output
ls -lh output/
cat output/feature/doc_stats_per_file.csv | head -10

# Analizza scarti
ls output/rejected/
```

---

## 🔧 Configurazione Avanzata

### Modificare Soglie di Filtraggio

Modifica `pipeline_factory.py`:

```python
# Lingua (0.0-1.0, default 0.75)
get_language_filter(rejected_dir, threshold=0.80)

# Qualità (0.0-1.0, default 0.7)
ItalianClassification(
    ...
    threshold = 0.65  # Più permissivo
)

# Spam (se abilitato, default 0.7)
SpamClassifier(
    ...
    threshold=0.8
)
```

### Elaborazione Parallela

Modifica `main.py`:

```python
executor = LocalPipelineExecutor(
    pipeline=pipeline_blocks,
    tasks=4,      # Numero di task paralleli
    workers=2     # Worker per task
)
```

### Disabilitare Componenti

Commenta le sezioni in `pipeline_factory.py`:

```python
# # 4. SPAM: Estrattore Feature
# SpamFeatureExtractor(),

# # 5. SPAM: Classifier
# SpamClassifier(...),

# # Spam Writer
# SpamFeatureCsvWriter(...)
```

---

## 🐛 Troubleshooting

### Problema: "ModuleNotFoundError: No module named 'datatrove'"

**Soluzione:**
```bash
# Reinstalla le dipendenze
pip install --upgrade -r requirements.txt
```

### Problema: "No such file or directory: models/lgbm_quality_model.joblib"

**Soluzione:**
```bash
# Verifica modelli presenti
ls -lh models/

# Se assenti, si può:
# 1. Addestrare il modello (vedi training_lgbmclassifier.py)
# 2. Disabilitare il quality classifier
```

### Problema: Dataset vuoto dopo filtri

Possibili cause:
- Soglia lingua troppo alta → ridurre da 0.75 a 0.65
- Soglia qualità troppo alta → ridurre da 0.7 a 0.5
- Formato input non valido → verifica JSONL

**Debug:**
```bash
# Esamina file rifiutati
wc -l output/rejected/1_language/*.jsonl
tail output/rejected/1_language/*.jsonl
```

### Problema: Memoria insufficiente (OutOfMemoryError)

**Soluzione:**
```python
# In LocalPipelineExecutor, riduci workers
executor = LocalPipelineExecutor(
    pipeline=pipeline_blocks,
    tasks=1,    # Uno alla volta
    workers=1   # Un worker
)
```

---

## 📊 Performance e Ottimizzazione

### Metriche Tipiche

| Dataset | Documenti | Tempo | RAM |
|---------|-----------|-------|-----|
| 1000 doc | 1,000 | ~30s | 512MB |
| 10K doc | 10,000 | ~3min | 1GB |
| 100K doc | 100,000 | ~30min | 2GB |

*(Da considerare come ordine di grandezza, varia con la qualità del testo)*

### Tips Ottimizzazione

1. **Velocità**: Aumenta `tasks` e `workers`
2. **Memoria**: Riduci `workers` o elabora in batch
3. **Accuratezza**: Abbassa soglie di filtering
4. **Pulizia**: Aumenta soglie di filtering

---

## 🔗 Risorse

- [DataTrove Documentation](https://github.com/huggingface/datatrove)
- [LightGBM](https://lightgbm.readthedocs.io/)
- [fastText Language Identification](https://fasttext.cc/docs/en/language-identification.html)

---

## 📝 Note Importanti

- La pipeline è idempotente: lanciare più volte produce gli stessi risultati
- I file di output non vengono sovrascritti automaticamente; elimina `output/` prima di rirunnare
- Per batch processing: combina con uno script wrapper che processa file incrementalmente
- I modelli ML richiedono specifici features; modifiche a `stats.py` devono allinearsi con il training data

---

## 📞 Support

Per segnalare problemi o richiedere aiuto:

1. Controlla il [log di esecuzione](logs/)
2. Esamina i [file di scarto](output/rejected/)
3. Verifica la [configurazione usata](configs/)

---
