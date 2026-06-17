# ITA-LLM-Pipeline

Pipeline di pulizia e classificazione per corpus italiani basata su DataTrove. Il progetto legge shard JSONL, filtra i documenti non italiani, rimuove contenuti spam, estrae feature statistiche, classifica la qualità con LightGBM e salva sia gli output validi sia gli scarti organizzati per tipologia.

## Funzionalità

- Filtro lingua italiano con soglia configurabile.
- Estrazione estesa di feature documentali in `DocStatsCsv` e `SpamFeatureExtractor`.
- Classificazione spam tramite modello `spam_lgbm.joblib`.
- Classificazione qualità tramite modello `lgbm_quality_model.joblib`.
- Aggregazione finale dei CSV parziali `rank_*_doc_stats_per_file.csv`.
- Analisi post-run degli scarti con `utils/output_organizer.py`.
- Script dedicato per valutazione modello e confronto multi-modello in cross validation.

## Struttura del repository

```text
ita-llm-pipeline/
│
├── Dockerfile                          # Build immagine Docker (Python 3.12-slim)
├── docker-compose.yml                  # Orchestrazione servizi
├── requirements.txt                    # Dipendenze Python
├── .gitignore                          # Esclusioni Git
├── README.md                           # Questa guida
├── VALUTAZIONE_MODELLO.md              # Guida valutazione e metriche modelli
│
├── configs/
│   └── default.conf                    # Variabili ambiente e path
│
├── data/
│   ├── dataset/                        # Dataset principale (JSONL)
│   ├── train/                          # Shard training per pipeline
│   ├── test/                           # Shard test
│   ├── spam/                           # Dataset annotati spam
│   │   ├── spam_data.jsonl
│   │   ├── spam_dataset_300.jsonl
│   │   ├── tutti_gli_spam.jsonl
│   │   ├── test/
│   │   └── train/
│   ├── splits/                         # CSV split dataset per training
│   │   ├── doc_stats_test.csv
│   │   ├── doc_stats_train.csv
│   │   └── doc_stats_val.csv
│   └── warc_paths                      # Path WARC per estrazione shard del web
│
├── models/
│   ├── lgbm_quality_model.joblib       # Classificatore qualità LightGBM
│   └── spam_lgbm.joblib                # Classificatore spam LightGBM
│
├── evaluation/
│   ├── evaluation_report.json          # Report metriche qualità
│   ├── evaluation_report.html          # Report interattivo qualità
│   ├── feature_importance.csv          # Feature importance qualità
│   └── spam/                           # Valutazione modello spam
│
├── output/                             # Generato da esecuzione
│   ├── italiano_pulito_*.jsonl         # Documenti processati
│   ├── feature/                        # CSV aggregati
│   ├── inspection/                     # Analisi scarti
│   └── rejected/                       # Documenti scartati per fase
│
├── logs/                               # Log timestamped di ogni esecuzione
│
├── notebooks/
│   └── report.ipynb                    # Analisi interattiva modello QualityClassifier
│
├── my_web_dump_output/                 # Output web_extracting_pipeline.py
│
├── src/
│   ├── main.py                         # Entry point pipeline
│   ├── config_loader.py                # Caricamento configurazione
│   ├── pipeline_factory.py             # Factory pattern
│   ├── blocks/
│   │   ├── classifiers.py              # Classificatori LightGBM
│   │   ├── evaluation.py               # Valutazione modelli
│   │   ├── filters.py                  # Filtri lingua/contenuto
│   │   ├── readers.py                  # Reader JSONL
│   │   ├── stats.py                    # Statistiche documenti
│   │   ├── writers.py                  # Writer JSONL
│   │   └── spam_classifier/
│   │       ├── spam_classifier.py
│   │       ├── spam_evaluation.py
│   │       ├── spam_keywords.py
│   │       └── spam_stats.py
│   └── utils/
│       ├── csv_aggregator.py           # Aggregazione CSV
|       ├── fix_rpDataset.py            # Standardizza il dataset di RedPajama per lo scopo di questa pipeline
│       └── output_organizer.py         # Organizzazione output
│
└── scripts/
    ├── evaluate_model.py               # Valutazione qualità
    ├── training_lgbmclassifier.py      # Training qualità
    ├── training_spam_lgbmclassifier.py # Training spam
    ├── web_extracting_pipeline.py      # Estrazione web
    └── spam/
        └── evaluate_spam_model.py      # Valutazione spam

```

## Requisiti

### Esecuzione locale

- **Python 3.12+** (consigliato: 3.12.0+)
- **pip** (gestione pacchetti)
- **Dipendenze di `requirements.txt`** (datatrove, lightgbm, spacy, fasttext...)
- **Spazio disco**: 10+ GB (modelli, dati, output)

### Esecuzione Docker

- **Docker** 20.10+
- **Docker Compose** (plugin `docker compose`)
- **Spazio disco**: 15+ GB (incluso volume HuggingFace cache)

## Configurazione

La pipeline legge le variabili di configurazione da **3 fonti** (in ordine di precedenza):
1. **Argomenti CLI** (`--root-dir`, `--output-dir`, etc.)
2. **File `.conf`** in `configs/` (caricato con `--config`)
3. **Default interni** nel codice

### Variabili Configurazione Principali

| Variabile | Significato | Default |
| --- | --- | --- |
| `ROOT_DIR` | Radice progetto | `/app` (Docker), `./` (locale) |
| `DATA_DIR` | Directory input | `ROOT_DIR/data` |
| `OUTPUT_DIR` | Directory output | `ROOT_DIR/output` |
| `REJECTED_DIR` | Scarti pipeline | `OUTPUT_DIR/rejected` |
| `FEATURE_DIR` | CSV aggregati | `OUTPUT_DIR/feature` |
| `MODEL_PATH` | Cartella modelli | `ROOT_DIR/models` |
| `MAX_WORKERS` | Worker DataTrove | `cpu_count() - 2` (min 1) |
| `INPUT_SUB_PATTERN` | Pattern file input | `train/*.jsonl` |
| `THRESHOLD_SPAM` | Soglia spam (0-1) | `0.75` |
| `THRESHOLD_QUALITY` | Soglia qualità (0-1) | `0.65` |
| `LANG_THRESHOLD` | Soglia lingua (0-1) | `0.75` |

### File Configurazione Disponibili

In `configs/`:
- **`default.conf`** - Configurazione di base (paths Docker)

**Nota**: Il file `default.conf` usa path Docker. Per esecuzione locale, sovrascrivere con argomenti CLI.

## Gestione Dataset (Interno vs Esterno)

La pipeline supporta due modalità di input dati, controllate dal flag **`USE_EXTERNAL_DATA`** in `src/config_loader.py`. Questo meccanismo permette di scegliere facilmente se usare i dati contenuti nella repository o puntare a una cartella esterna.

### 1. Utilizzo Dataset della Repository (Default)

**Modalità**: Esecuzione locale con dati contenuti in `data/dataset/*.jsonl`

```python
# In src/config_loader.py
USE_EXTERNAL_DATA = False
```

Quando `USE_EXTERNAL_DATA = False`:
- `DATA_DIR` = `ROOT_DIR/data` (punta sempre a `./data`)
- `INPUT_SUB_PATTERN` = `dataset/*.jsonl` (per default, configurabile)
- La pipeline legge gli JSONL dalla repository

**Comandi**:
```bash
# Locale
python3 src/main.py --root-dir . --output-dir ./output

# Docker (già configurato)
docker compose up --build
```

### 2. Utilizzo Dataset Esterno (Dischi Esterni, Percorsi Alternativi)

**Modalità**: Puntare a una cartella esterna senza spostare i file.

```python
# In src/config_loader.py
USE_EXTERNAL_DATA = True
```

Quando `USE_EXTERNAL_DATA = True`:
- `DATA_DIR` = `/app/external_data` (o il percorso specificato)
- `INPUT_SUB_PATTERN` = `*.jsonl` 
- La pipeline legge JSONL dal volume montato

**Setup per Docker Compose**:

1. Apri `docker-compose.yml` e modifica il servizio `pipeline`:
   ```yaml
   services:
     pipeline:
       # ...
       volumes:
         - /tuo/percorso/locale:/app/external_data:ro
         - ./output:/app/output:rw
   ```
   Sostituisci `/tuo/percorso/locale` con il percorso reale 

2. Imposta il flag in `src/config_loader.py`:
   ```python
   USE_EXTERNAL_DATA = True
   ```

3. Esegui:
   ```bash
   docker compose up --build
   ```

**Setup per esecuzione locale**:

1. Imposta il flag in `src/config_loader.py`:
   ```python
   USE_EXTERNAL_DATA = True
   ```

2. Esegui con il percorso esterno:
   ```bash
   python3 src/main.py \
     --root-dir . \
     --data-dir /percorso/personalizzato \
     --output-dir ./output
   ```

### 3. Personalizzazione Pattern Input

Se vuoi leggere solo un sottoinsieme di file, modifica `INPUT_SUB_PATTERN` in `src/config_loader.py`:

```python
# Legge tutti i JSONL in data/
INPUT_SUB_PATTERN = "*.jsonl"

# Legge da shard specifici
INPUT_SUB_PATTERN = "train/shard_0{0..5}.jsonl" 
```

### Note Operative

- **`NUM_TASKS`** viene calcolato contando automaticamente i file che matchano `DATA_DIR/INPUT_SUB_PATTERN`
- **`--tasks`** dalla CLI sovrascrive il conteggio automatico se specificato: `python3 src/main.py --tasks 100`

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

## Quick Start (Esecuzione Rapida)

### 1️ Primo Avvio Locale (5 minuti)

```bash
# Attiva environment
source venv/bin/activate

# Esegui pipeline con config di default
python3 src/main.py --config configs/default.conf

# (Oppure con override path locale)
python3 src/main.py \
  --root-dir . \
  --data-dir ./data/train \
  --output-dir ./output \
  --workers 4
```

L'esecuzione genererà file in `output/` e log in `logs/`.

### 2️ Analisi Risultati con Notebook (2 minuti)

```bash
# Apri Jupyter
jupyter notebook notebooks/report.ipynb
```

Il notebook carica automaticamente il modello `models/lgbm_quality_model.joblib` e:
- Calcola metriche di valutazione
- Mostra feature importance
- Visualizza confusion matrix
- Genera curve ROC e Precision-Recall
- Analizza SHAP values per interpretabilità

### 3️ Valutazione Modelli (1 minuto)

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models
```

Genera report HTML interattivo in `evaluation/evaluation_report.html`.

`build_italian_cleaning_pipeline()` costruisce questi step:

1. `get_jsonl_reader(data_dir, pattern)` - Legge file JSONL
2. `get_language_filter(rejected_dir, threshold=0.75, languages="it")` - Filtra italiano
3. `SpamFeatureExtractor()` - Estrae feature spam
4. `SpamFeatureCsvWriter(... , csv_filename="spam_doc_features.csv")` - Salva feature spam
5. `SpamFilter(model_path, ... , threshold=0.75)` - Classifica spam
6. `DocStatsCsv(..., csv_filename="doc_stats_per_file.csv", groups_to_compute=["summary"])` - Estrae feature qualità
7. `ItalianClassification(..., threshold=0.65)` - Classifica qualità
8. `get_jsonl_writer(output_dir)` - Salva documenti validi

Dopo l'esecuzione della pipeline, `src/main.py` esegue anche:

1. aggregazione dei CSV per-rank in `FEATURE_DIR/doc_stats_per_file.csv`
2. aggregazione dei CSV spam in `FEATURE_DIR/spam_doc_features.csv`
3. rimozione dei file temporanei `rank_*_*_.csv`
4. analisi finale degli output con `output_classification(REJECTED_DIR, OUTPUT_DIR)`

## Output prodotti

### Struttura Directory Output

Struttura tipica dopo un esecuzione:

```text
output/
├── italiano_pulito_${rank}.jsonl          # Documenti passati tutti i filtri (VALIDI)
├── feature/                               # CSV aggregati
│   ├── doc_stats_per_file.csv             # Feature qualità aggregato
│   └── spam_doc_features.csv              # Feature spam aggregato
├── inspection/                            # Analisi scarti post-run
│   ├── rejected_was_bad.jsonl             # Scarti corretti (false negative evitati)
│   └── rejected_was_good.jsonl            # Falsi scarti (false positive)
└── rejected/                              # Documenti scartati per fase
    ├── 1_language/
    │   └── non_italiano_${rank}.jsonl     # Language score < 0.75
    ├── 2_spam/
    │   └── spam_rejected_${rank}.jsonl    # Spam classifier score > 0.75
    └── 3_quality/
        └── quality_rejected_${rank}.jsonl # Quality classifier score < 0.65
```

### File CSV di Output: Feature Quality

**File**: `output/feature/doc_stats_per_file.csv`

Questo CSV contiene le feature estratte da ogni documento **processato** (prima della classificazione) e serve sia per:
1. **Input al QualityClassifier** per predire la qualità
2. **Analisi post-processing** delle statistiche documentali

**Colonne principali**:

| Feature | Tipo | Interpretazione |
|---------|------|------------------|
| `doc_id` | str | ID univoco documento |
| `language_score` | float (0-1) | Confidence lingua italiana (fastText) |
| `length` | int | Lunghezza testo in caratteri |
| `word_count` | int | Numero di token (words) |
| `text_entropy` | float | Entropia Shannon del testo (diversità) |
| `url_count` | int | Numero di URL nel testo |
| `email_count` | int | Numero di indirizzi email |
| `unique_word_ratio` | float (0-1) | Rapporto parole uniche / totali |
| `consecutive_punctuation_count` | int | Sequenze di punteggiatura consecutive |
| `punctuation_ratio` | float (0-1) | Rapporto punteggiatura / lunghezza |

**Utilizzo tipico**:
```bash
# Analizzare le statistiche
python3 -c "
import pandas as pd
df = pd.read_csv('output/feature/doc_stats_per_file.csv')

# Statistiche descrittive
print(df[['language_score', 'text_entropy', 'word_count', 'unique_word_ratio']].describe())

# Documenti anomali
print('\nDocumenti con scarsa diversità lessicale:')
print(df[df['unique_word_ratio'] < 0.3])
"
```

### File CSV di Output: Feature Spam

**File**: `output/feature/spam_doc_features.csv`

Questo CSV contiene le feature estratte specificamente per il **SpamClassifier**. Viene usato sia per:
1. **Input al SpamFilter** per predire probabilità spam
2. **Training/Valutazione** del modello spam in `scripts/training_spam_lgbmclassifier.py`

**Colonne principali**:

| Feature | Tipo | Interpretazione |
|---------|------|------------------|
| `doc_id` | str | ID univoco documento |
| `keyword_count` | int | Numero keyword spam rilevate |
| `keyword_density` | float (0-1) | Percentuale keyword spam nel testo |
| `suspicious_chars_ratio` | float (0-1) | Rapporto caratteri sospetti / lunghezza |
| `repetition_score` | float | Score ripetizioni parole/sequenze |
| `url_spam_ratio` | float (0-1) | Rapporto URL sospetti / URL totali |
| `email_spam_ratio` | float (0-1) | Rapporto email spam-like / email totali |

**Utilizzo tipico**:
```bash
# Analizzare prevalenza spam
python3 -c "
import pandas as pd
df = pd.read_csv('output/feature/spam_doc_features.csv')

# Quanto spam è stato rilevato
print('Distribuzione keyword spam:')
print(df['keyword_count'].value_counts())

# Correlazione con classificazione
print('\nDocumenti ad alta densità di keyword spam:')
print(df[df['keyword_density'] > 0.5])
"
```

### Quando i CSV vengono generati

Nella pipeline (`src/pipeline_factory.py`), i CSV vengono scritti **per-rank** durante l'esecuzione:
- **`rank_*_doc_stats_per_file.csv`** - Feature quality aggregati dal blocco `DocStatsCsv()`
- **`rank_*_spam_doc_features.csv`** - Feature spam dal blocco `SpamFeatureCsvWriter()`

Al termine dell'esecuzione, `src/main.py` chiama `aggregate_rank_csvs()` che:
1. Legge tutti i file per-rank
2. Li concatena in un unico CSV
3. Salva come `doc_stats_per_file.csv` e `spam_doc_features.csv`
4. Rimuove i file temporanei (se `remove_parts=True`)

## Analisi Risultati

### Interpretare gli Output

Dopo l'esecuzione della pipeline, i documenti vengono classificati e organizzati per fase di scarto:

| Cartella | Documenti | Motivo Scarto |
|----------|-----------|---------------|
| `output/italiano_pulito_*.jsonl` | **VALIDI** | Superano tutti i filtri |
| `output/rejected/1_language/` | Non italiani | `language_score < 0.75` |
| `output/rejected/2_spam/` | Spam | Modello spam predice positivo + evidenze forti |
| `output/rejected/3_quality/` | Scarsa qualità | Modello qualità predice negativo (score < 0.65) |
| `output/inspection/rejected_was_bad.jsonl` | Scarti corretti | Analisi post-run: effettivamente scarti |
| `output/inspection/rejected_was_good.jsonl` | Falsi scarti | Analisi post-run: erroneamente eliminati |

### Esaminare le Statistiche

```bash
# Vedere quanti documenti per categoria
# Il comando calcola e mostra il numero totale di righe contenute in tutti i file .jsonl presenti nella cartella output/rejected e nelle sue sottocartelle.
find output/rejected -name "*.jsonl" -exec wc -l {} + | tail -1

# Leggere il CSV di statistiche
python3 -c "
import pandas as pd
df = pd.read_csv('output/feature/doc_stats_per_file.csv')
print(df[['language_score', 'text_entropy', 'word_count']].describe())
"
```

### Usare il Notebook report.ipynb

Il notebook **`notebooks/report.ipynb`** fornisce un'analisi interattiva:

1. **Caricamento Modello**: Carica automaticamente il modello salvato
2. **Metriche di Valutazione**: Accuracy, Precision, Recall, F1-Score
3. **Confusion Matrix**: Visualizza TP, TN, FP, FN
4. **Feature Importance**: Permutation importance delle top 10 feature
5. **Correlation Matrix**: Relazioni tra le feature
6. **Cross-Validation**: Stabilità metriche su fold diversi
7. **ROC-AUC & Precision-Recall**: Curve di performance
8. **Model Comparison**: LGBM vs LogisticRegression baseline
9. **SHAP Values**: Interpretabilità decisioni modello
10. **Data Leakage Check**: Verifica assenza data leakage

**Per eseguire il notebook:**
```bash
jupyter notebook notebooks/report.ipynb

# Oppure se non hai jupyter installato
pip install jupyter
jupyter notebook notebooks/report.ipynb
```

## Valutazione del modello

Per la valutazione del classificatore qualità consulta:

- **[VALUTAZIONE_MODELLO.md](./VALUTAZIONE_MODELLO.md)** - Guida completa metriche e interpretazione
- **`scripts/evaluate_model.py`** - Script valutazione CLI

### Valutazione Rapida

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models
```

Genera:
- Report JSON in `evaluation/evaluation_report.json`
- Report HTML interattivo in `evaluation/evaluation_report.html`
- CSV feature importance in `evaluation/feature_importance.csv`

Lo script tenta di recuperare automaticamente il test set dai metadati del modello. Se fallisce, passare `--test-csv`:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --test-csv data/splits/doc_stats_test.csv \
  --output-dir evaluation
```

### Valutazione Modello Spam

Per la valutazione del classificatore spam:

```bash
python3 scripts/spam/evaluate_spam_model.py \
  --model models/spam_lgbm.joblib \
  --test-csv evaluation/spam/spam_test_features.csv \
  --output-dir evaluation/spam \
  --compare-models
```

Consulta il modulo **`src/blocks/spam_classifier/spam_evaluation.py`** per dettagli.

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

## Training del modello di qualità

Lo script di training qualità è:

```text
scripts/training_lgbmclassifier.py
```

Esempio con CSV di feature:

```bash
python3 scripts/training_lgbmclassifier.py \
  --csv-path output/feature/doc_stats_per_file.csv \
  --model-path models/lgbm_quality_model_new.joblib \
  --label-column label \
  --threshold 0.65 \
  --test-size 0.20 \
  --random-state 42 \
  --n-estimators 300 \
  --learning-rate 0.05
```

**Output prodotti:**
```text
models/lgbm_quality_model_new.joblib    # Modello addestrato
evaluation/                              # Report metriche (opzionale con --output-dir)
```

**Parametri disponibili:**
- `--csv-path`: CSV con feature e label
- `--model-path`: Percorso salvataggio modello
- `--label-column`: Colonna con etichette ("good"/"bad")
- `--threshold`: Soglia decisione (default 0.65)
- `--test-size`: Quota test split (default 0.20)
- `--n-estimators`: Numero alberi (default 300)
- `--learning-rate`: Learning rate (default 0.05)
- `--random-state`: Seed (default 42)

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

---

## Troubleshooting & FAQ

### Errore: `ModuleNotFoundError: No module named 'src'`

**Soluzione**: Assicurati di eseguire da root directory del progetto:
```bash
cd /home/stefano/ita-llm-pipeline
python3 src/main.py --config configs/default.conf
```

### Errore: `File not found: data/train/*.jsonl`

**Soluzione**: Verifica che i file JSONL esistano in `data/train/`:
```bash
ls -la data/train/ | head -10

# Oppure usa dataset esterno decommentando docker-compose.yml
```

### Errore: `GPU out of memory`

**Soluzione**: Riduci il numero di worker:
```bash
python3 src/main.py --workers 2 --config configs/default.conf
```

### Errore nel notebook Jupyter: `ImportError: No module named 'shap'`

**Soluzione**: Installa dipendenze aggiuntive:
```bash
pip install shap jupyter matplotlib seaborn scikit-learn
```

### Output non generato

**Verifiche**:
1. Controlla che la pipeline sia completata (ultimo log non contiene errori):
   ```bash
   tail logs/*/pipeline.log
   ```

2. Verifica i file temporanei:
   ```bash
   ls output/feature/ | grep rank
   ```

3. Se i file `rank_*` ancora esistono, l'aggregazione è fallita:
   ```bash
   python3 src/utils/csv_aggregator.py --feature-dir output/feature
   ```

### Modello predice sempre la stessa classe

**Cause possibili**:
- Dataset sbilanciato
- Threshold troppo alto/basso
- Features mancanti o costanti

**Verifica**:
```bash
python3 -c "
import pandas as pd
df = pd.read_csv('output/feature/doc_stats_per_file.csv')
print(df.describe())  # Controlla range feature
print(df.isnull().sum())  # Controlla valori mancanti
"
```

---

## Documentazione Correlata

- **[VALUTAZIONE_MODELLO.md](./VALUTAZIONE_MODELLO.md)** - Metriche di valutazione dettagliate
- **[notebooks/report.ipynb](./notebooks/report.ipynb)** - Analisi interattiva modelli
- **Codice sorgente**:
  - `src/blocks/classifiers.py` - Logica classificatori LightGBM
  - `src/blocks/spam_classifier/` - Modulo spam detection
  - `src/pipeline_factory.py` - Costruzione pipeline

---

## Note Importanti

1. **Path Configurazione**: Il file `default.conf` usa path Docker. Per esecuzione locale, usa argomenti CLI per override.

2. **Modelli Pre-addestrati**: I file `.joblib` in `models/` contengono scaler e metadata. Non eliminare.

3. **Dataset Split**: I CSV in `data/splits/` contengono le split per training/validation/test. Rigenerarli comporta risultati diversi.

4. **Soglie Tuning**: Le soglie di spam (0.75) e qualità (0.65) possono essere tuned con `--threshold-sweep` nei script di evaluation.

5. **Parallelizzazione**: `MAX_WORKERS` di default è `cpu_count() - 2`. Aumentare per HW potente, ridurre per sistemi con memoria limitata.