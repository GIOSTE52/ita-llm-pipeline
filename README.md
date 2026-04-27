# ITA-LLM-Pipeline

Pipeline di pulizia e classificazione per corpora italiani basata su DataTrove. Il progetto legge shard JSONL, filtra i documenti non italiani, estrae feature statistiche, classifica la qualità con LightGBM e salva sia gli output validi sia gli scarti organizzati per tipologia.

## Funzionalità

- Filtro lingua italiano con soglia configurabile.
- Estrazione estesa di feature documentali in `DocStatsCsv`.
- Classificazione qualità tramite modello `lgbm_quality_model.joblib`.
- Aggregazione finale dei CSV parziali `rank_*_doc_stats_per_file.csv`.
- Analisi post-run degli scarti con `utils/output_organizer.py`.
- Script dedicato per valutazione modello e confronto multi-modello in cross validation.

## Struttura del repository

```text
ita-llm-pipeline/
├── configs/
│   ├── default.conf
│   ├── gabriele.conf
│   ├── silvio.conf
│   └── stefano.conf
├── data/
│   ├── dataset/
│   ├── splits/
│   ├── spam/
│   ├── test/
│   ├── train/
│   └── warc_paths
├── evaluation/
│   ├── evaluation_report.html
│   ├── evaluation_report.json
│   └── feature_importance.csv
├── models/
│   ├── lgbm_quality_model.joblib
│   └── spam_lgbm.joblib
├── scripts/
│   ├── evaluate_model.py
│   ├── training_lgbmclassifier.py
│   └── web_extracting_pipeline.py
├── src/
│   ├── main.py
│   ├── config_loader.py
│   ├── pipeline_factory.py
│   ├── blocks/
│   │   ├── classifiers.py
│   │   ├── filters.py
│   │   ├── readers.py
│   │   ├── stats.py
│   │   ├── writers.py
│   │   └── spam_classifier/
│   └── utils/
├── tests/
│   ├── conftest.py
│   ├── test_evaluate_model.py
│   ├── test_main.py
│   └── test_pipeline_components.py
├── docker-compose.yml
├── Dockerfile
├── EVALUATE_MODEL.md
├── README.md
├── requirements-dev.txt
└── requirements.txt
```

## Requisiti

### Esecuzione locale

- Python 3.12+
- `pip`
- Dipendenze di `requirements.txt`

### Esecuzione Docker

- Docker
- Docker Compose plugin (`docker compose`)

## Configurazione

La pipeline usa argomenti CLI e variabili d'ambiente. I file `.conf` in `configs/` servono a valorizzare le variabili base, soprattutto in Docker.

Variabili e valori principali:

| Variabile | Significato | Default effettivo |
| --- | --- | --- |
| `ROOT_DIR` | root del progetto | `/app` in Docker, cartella corrente in locale |
| `DATA_DIR` | directory di input | `ROOT_DIR/data` |
| `OUTPUT_DIR` | directory di output | `ROOT_DIR/output` |
| `REJECTED_DIR` | scarti della pipeline | `OUTPUT_DIR/rejected` |
| `FEATURE_DIR` | CSV e artefatti derivati | `OUTPUT_DIR/feature` |
| `MODEL_PATH` | cartella con i modelli | `ROOT_DIR/models` |
| `MAX_WORKERS` | numero worker DataTrove | default `cpu_count() - 2`, minimo `1` |

## Gestione Dataset (Interno vs Esterno)
- `config_loader.get_config()` usa come pattern input fisso `INPUT_SUB_PATTERN = train/*.jsonl`, bisogna modificarlo manualmente per cambiare la scelta del 
### 1. Utilizzo Dataset della Repository (Default)
Per utilizzare i file contenuti in `data/train/*.jsonl`:
- Nel file `src/config_loader.py`, imposta: `USE_EXTERNAL_DATA = False`.
- (Opzionale) Nel `docker-compose.yml`, puoi lasciare commentata la riga del volume esterno.

### 2. Utilizzo Dataset Esterno (Locale)
Per puntare a una cartella esterna (es. un disco rigido o la Scrivania) senza spostare i file:
1. **Docker Compose**: Decommenta o aggiungi la riga del volume nel servizio `pipeline`:
   ```yaml
   volumes:
     - /tuo/percorso/sul/pc:/app/external_data:ro

Note operative:

dataset in input.
- `NUM_TASKS` viene calcolato contando i file che matchano `DATA_DIR/INPUT_SUB_PATTERN`.
- `--tasks` viene parsato dalla CLI e sovrascrive `NUM_TASKS` se specificato.

## Esecuzione

### Locale

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 src/main.py --config configs/default.conf
```

Esempio con override dei path:

```bash
python3 src/main.py \
  --config configs/stefano.conf \
  --root-dir /path/progetto \
  --output-dir /path/output \
  --rejected-dir /path/output/rejected \
  --workers 4
```

### Docker

```bash
docker compose build
docker compose up
```
oppure
```bash
docker compose up --build
```


## Pipeline attuale

`build_italian_cleaning_pipeline()` costruisce questi step:

1. `get_jsonl_reader(data_dir, pattern)`
2. `get_language_filter(rejected_dir, threshold=0.75, languages="it")`
3. `DocStatsCsv(..., csv_filename="doc_stats_per_file.csv", groups_to_compute=["summary"])`
4. `ItalianClassification(..., threshold=0.65)`
5. `get_jsonl_writer(output_dir)`

Dopo l'esecuzione della pipeline, `src/main.py` esegue anche:

1. aggregazione dei CSV per-rank in `FEATURE_DIR/doc_stats_per_file.csv`
2. rimozione dei file temporanei `rank_*_doc_stats_per_file.csv`
3. analisi finale degli output con `output_classification(REJECTED_DIR, OUTPUT_DIR)`

## Output prodotti

Struttura tipica:

```text
output/
├── italiano_pulito_${rank}.jsonl
├── feature/
│   └── doc_stats_per_file.csv
├── inspection/
│   ├── rejected_was_bad.jsonl
│   └── rejected_was_good.jsonl
└── rejected/
    ├── 1_language/
    │   └── non_italiano_${rank}.jsonl
    └── 3_quality/
        └── quality_rejectd_${rank}.jsonl
```

`doc_stats_per_file.csv` contiene, tra le altre, queste feature:

- `language_score`
- `length`
- `word_count`
- `text_entropy`
- `url_count`
- `email_count`
- `unique_word_ratio`
- `consecutive_punctuation_count`

## Testing

Le suite `pytest` coperte dal repository sono:

- `tests/test_main.py`: parsing CLI, config dinamica, wiring di `main()`
- `tests/test_pipeline_components.py`: reader, writer, filtro lingua, stats, spam feature extraction, pipeline factory
- `tests/test_evaluate_model.py`: utility e orchestrazione di `scripts/evaluate_model.py`

Installazione dipendenze di sviluppo:

```bash
pip install -r requirements-dev.txt
```

Esecuzione locale:

```bash
python3 -m pytest -q
python3 -m pytest tests/test_main.py -v
python3 -m pytest tests/test_evaluate_model.py -v
```

Esecuzione via Docker:

```bash
docker compose --profile test run --rm test
```

## Valutazione del modello

Per la valutazione del classificatore qualità usa:

- [VALUTAZIONE_MODELLO.md](./VALUTAZIONE_MODELLO.md)
- `scripts/evaluate_model.py`

Esempio rapido:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models
```

Lo script prova a recuperare automaticamente il test set dai metadati del modello; se non li trova, va passato `--test-csv`.
