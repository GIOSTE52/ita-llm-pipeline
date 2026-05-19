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
│   ├── spam/
│   │   ├── test/
│   │   ├── train/
│   │   ├── spam_data.jsonl
│   │   ├── spam_dataset_300.jsonl
│   │   └── tutti_gli_spam.jsonl
│   ├── splits/
│   ├── test/
│   ├── train/
│   └── warc_paths
├── evaluation/
│   ├── spam/
│   ├── evaluation_report.html
│   ├── evaluation_report.json
│   └── feature_importance.csv
├── models/
│   ├── lgbm_quality_model.joblib
│   └── spam_lgbm.joblib
├── my_web_dump_output/
├── notebooks/
├── scripts/
│   ├── spam/
│   │   └── evaluate_spam_model.py
│   ├── evaluate_model.py
│   ├── training_lgbmclassifier.py
│   ├── training_spam_lgbmclassifier.py
│   └── web_extracting_pipeline.py
├── src/
│   ├── main.py
│   ├── config_loader.py
│   ├── pipeline_factory.py
│   ├── blocks/
│   │   ├── __init__.py
│   │   ├── classifiers.py
│   │   ├── evaluation.py
│   │   ├── filters.py
│   │   ├── readers.py
│   │   ├── stats.py
│   │   ├── writers.py
│   │   └── spam_classifier/
│   │       ├── __init__.py
│   │       ├── spam_classifier.py
│   │       ├── spam_evaluation.py
│   │       ├── spam_keywords.py
│   │       └── spam_stats.py
│   └── utils/
│       ├── __init__.py
│       ├── counter_data.py
│       ├── csv_aggregator.py
│       ├── fix_rpDataset.py
│       └── output_organizer.py
├── .gitignore
├── Dockerfile
├── README.md
├── VALUTAZIONE_MODELLO.md
├── docker-compose.yml
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
3. `SpamFeatureExtractor()`
4. `SpamFeatureCsvWriter(... , csv_filename="spam_doc_features.csv")`
5. `SpamFilter(model_path, ... , threshold=0.75)`
6. `DocStatsCsv(..., csv_filename="doc_stats_per_file.csv", groups_to_compute=["summary"])`
7. `ItalianClassification(..., threshold=0.65)`
8. `get_jsonl_writer(output_dir)`

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
|   └── spam_doc_features.csv
├── inspection/
│   ├── rejected_was_bad.jsonl
│   └── rejected_was_good.jsonl
└── rejected/
    ├── 1_language/
    │   └── non_italiano_${rank}.jsonl
    ├── 2_spam/
    │   └── spam_rejected_${rank}.jsonl
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

## Filtro spam

Il filtro spam utilizzato nella pipeline è implementato in:

```text
src/blocks/spam_classifier/spam_classifier.py
```

Il documento viene scartato solo se:

1. il modello predice `spam`
2. la probabilità supera la soglia impostata
3. sono presenti evidenze forti di spam

Esempi di evidenze forti:

- TLD sospetti
- URL abbreviati associati a CTA
- CTA + URL
- urgenza + CTA
- denaro + CTA
- keyword account + security
- keyword delivery + URL
- brand + URL + CTA
- molte keyword spam con CTA, urgenza o denaro
- probabilità molto alta senza segnali business/ham

Se il modello sospetta spam ma mancano evidenze forti, il documento non viene scartato. 
In quel caso passa alla fase di quality classification e nei metadata viene aggiunto:

```text
spam_uncertain_reason = "high_score_but_weak_spam_evidence"
```

## Training del modello spam

Lo script di training spam è:

```text
scripts/training_spam_lgbmclassifier.py
```

Esempio:

```bash
python3 scripts/training_spam_lgbmclassifier.py \
  --csv-path output/feature/spam_doc_features.csv \
  --model-path models/spam_lgbm.joblib \
  --label-column spam_target_label \
  --threshold 0.75 \
  --test-size 0.30 \
  --random-state 42 \
  --errors-output-dir evaluation/spam/spam_train_errors_th075
```

Parametri principali:

| Parametro | Significato | 
|---|---|
| `--csv-path` | CSV con le feature spam |
| `--model-path` | Path di salvataggio modello |
| `--label-column` | Colonna label |
| `--threshold` | Soglia classificazione spam |
| `--test-size` | Quota test split |
| `--random-state` | Seed |
| `--errors-output-dir` | Directory output training/evaluation interna |

Output prodotti dal training:

```text
evaluation/spam/spam_train_errors_th075/
├── spam_train_features.csv
├── spam_test_features.csv
├── spam_features_with_split.csv
├── test_predictions.csv
├── test_misclassified.csv
├── false_positives.csv
└── false_negatives.csv
```

Il training stampa anche:

- numero di feature usate
- feature costanti rimosse
- feature quasi costanti rilevate
- classification report
- confusion matrix
- ROC-AUC
- permutation importance

Il modello salvato in `models/spam_lgbm.joblib` contiene:

- modello LightGBM
- scaler
- lista feature usate
- soglia
- label column
- metadata di training

---

## Evaluation del modello spam

Lo script di evaluation spam è:

```text
scripts/spam/evaluate_spam_model.py
```

Esempio:

```bash
python3 scripts/spam/evaluate_spam_model.py \
  --model models/spam_lgbm.joblib \
  --test-csv evaluation/spam/spam_train_errors_th075/spam_test_features.csv \
  --output-dir evaluation/spam/train_evaluation \
  --label-column spam_target_label \
  --threshold 0.75 \
  --threshold-sweep 0.40 0.50 0.60 0.70 0.75 0.80 0.90 \
  --compare-models \
  --comparison-csv output/feature/spam_doc_features.csv \
  --cv-folds 5
```

Parametri principali:

| Parametro | Significato |
|---|---|
| `--model` | Modello spam |
| `--test-csv` | CSV etichettato su cui valutare |
| `--output-dir` | Directory dei report |
| `--label-column` | Colonna label da usare |
| `--threshold` | Soglia spam da applicare |
| `--threshold-sweep` | Lista di soglie da confrontare |
| `--compare-models` | Attiva confronto multi-modello in cross validation |
| `--comparison-csv` | Dataset da usare per cross validation |
| `--cv-folds` | Numero fold CV |
| `--cv-models` | Modelli specifici da confrontare |
| `--no-feature-importance` | Disattiva permutation importance |

Modelli confrontabili con `--cv-models`:

- `lightgbm`
- `logistic_regression`
- `random_forest`
- `extra_trees`
- `dummy`

Output prodotti dall'evaluation spam:

```text
evaluation/spam/train_evaluation/
├── spam_evaluation_report.json
├── spam_predictions.csv
├── spam_false_positives.csv
├── spam_false_negatives.csv
├── spam_feature_importance.csv
├── spam_features_strong.csv
├── spam_features_to_review.csv
├── spam_features_negative_importance.csv
├── spam_threshold_sweep.csv
├── spam_model_comparison_cv.csv
└── spam_model_comparison_report.json
```

Alcuni file sono prodotti solo se la relativa opzione è attiva:

- `spam_threshold_sweep.csv` richiede `--threshold-sweep`
- `spam_model_comparison_cv.csv` e `spam_model_comparison_report.json` richiedono `--compare-models`
- i file di feature importance non vengono prodotti se si usa `--no-feature-importance`

