# Valutazione dei Modelli

Questo documento spiega come valutare i modelli usati dalla pipeline:

- classificatore di qualità: `models/lgbm_quality_model.joblib`
- classificatore spam: `models/spam_lgbm.joblib`

## Comando rapido

Il comando consigliato è:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models
```

Lo script prova a recuperare automaticamente il test set dai metadati salvati nel modello. Per il modello attuale il test set registrato è:

```text
data/splits/doc_stats_test.csv
```

Se il recupero automatico non funziona, passare il CSV esplicitamente:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --test-csv data/splits/doc_stats_test.csv \
  --output-dir evaluation
```

## Parametri principali

| Parametro | Significato |
| --- | --- |
| `--model` | Percorso del modello `.joblib` da valutare. |
| `--test-csv` | CSV di test con feature e label. Se omesso, lo script usa il test split registrato nel modello. |
| `--output-dir` | Directory in cui salvare JSON, CSV e HTML. |
| `--threshold` | Soglia decisionale. Se omessa, usa quella salvata nel modello; in mancanza usa `0.65`. |
| `--label-column` | Nome della colonna label. Default: `label`. |
| `--compare-models` | Esegue anche un confronto in cross-validation tra modelli. |
| `--comparison-csv` | Dataset etichettato per la cross-validation. Se omesso, usa il dataset sorgente registrato nel modello oppure il test CSV. |
| `--cv-folds` | Numero di fold per la cross-validation. Default: `5`. |
| `--cv-models` | Sottoinsieme di modelli da confrontare: `lightgbm`, `random_forest`, `extra_trees`, `logistic_regression`. |

## Formato del CSV richiesto

Il CSV di valutazione deve contenere:

- `doc_id`, utile per tracciare i documenti
- `label`, con valori `good` o `bad`
- tutte le feature attese dal modello, definite in `DEFAULT_FEATURE_NAMES` in `src/blocks/classifiers.py`

Esempio semplificato:

```csv
doc_id,label,language_score,length,word_count,text_entropy,unique_word_ratio,...
doc_001,good,0.98,1234,210,4.52,0.71,...
doc_002,bad,0.91,456,72,2.10,0.32,...
```

Gli split prodotti dallo script di training qualità sono:

```text
data/splits/doc_stats_train.csv
data/splits/doc_stats_val.csv
data/splits/doc_stats_test.csv
```

Lo script `scripts/training_lgbmclassifier.py` usa uno split:

| Split | Quota |
| --- | --- |
| Train | 70% |
| Validation | 15% |
| Test | 15% |

Il modello viene addestrato sul train set, validato sul validation set e valutato sul test set.

## Output generati

Con `--output-dir evaluation`, la valutazione qualità genera:

```text
evaluation/
├── evaluation_report.json
├── evaluation_report.html
└── feature_importance.csv
```

I file hanno questi ruoli:

| File | Contenuto |
| --- | --- |
| `evaluation_report.json` | Metriche, confusion matrix, metadati modello, top feature e confronto modelli se richiesto. |
| `feature_importance.csv` | Permutation importance delle feature. |
| `evaluation_report.html` | Report interattivo con metriche, confusion matrix, feature importance, ROC curve e Precision-Recall curve. Generato da template Jinja2. |

## Visualizzazione del report HTML

Il report HTML viene generato a partire dal template `src/blocks/templates/evaluation_report.html`, che viene riempito con i dati di valutazione usando **Jinja2**.

### Aprire il report nel browser

Una volta generato, è possibile aprire il report direttamente nel browser:

**Linux/macOS:**
```bash
# Linux
xdg-open evaluation/evaluation_report.html

# macOS
open evaluation/evaluation_report.html
```

**Windows:**
```bash
start evaluation/evaluation_report.html
```

### Servire il report localmente (opzionale)

Per servire il report tramite un server HTTP:

```bash
cd evaluation
python -m http.server 8000
# Poi visita http://localhost:8000/evaluation_report.html
```

## Metriche

Le metriche principali sono:

| Metrica | Interpretazione |
| --- | --- |
| Accuracy | Percentuale totale di predizioni corrette. |
| Balanced Accuracy | Accuracy bilanciata tra classi, utile se `good` e `bad` sono sbilanciate. |
| Precision | Quanto sono affidabili le predizioni positive. |
| Recall | Quanti esempi positivi reali vengono recuperati. |
| F1-score | Media armonica tra precision e recall. |
| ROC-AUC | Capacità del modello di separare le classi al variare della soglia. |

Nel classificatore qualità la classe positiva è `good`, perché in `LABEL_MAP`:

```python
LABEL_MAP = {"bad": 0, "good": 1}
```

## Confusion matrix

La matrice di confusione prodotta da `src/blocks/evaluation.py` segue questa convenzione:

```text
                Predicted bad   Predicted good
Actual bad          TN              FP
Actual good         FN              TP
```

Significato operativo:
+ `TN` sta per True Negative, ovvero documenti `bad` classificati correttamente come `bad`
+ `FP` sta per False Negative, ovvero documenti `bad` lasciati passare come `good`
+ `FN` sta per False Negative, ovvero documenti `good` scartati per errore come `bad`
+ `TP` sta per True Positive, ovvero documenti `good` classificati correttamente come `good`

Per la pipeline, gli errori hanno impatti diversi:

- molti `FP` significano che documenti scadenti restano nel corpus finale
- molti `FN` significano che documenti validi vengono scartati

La soglia `0.65` controlla questo compromesso.

## Risultati attuali

Il report attuale in `evaluation/evaluation_report.json` indica:

| Metrica | Valore |
| --- | ---: |
| Accuracy | `0.9189` |
| Balanced Accuracy | `0.8813` |
| F1-score | `0.8529` |
| ROC-AUC | `0.9705` |
| Threshold | `0.65` |

Confusion matrix:

```text
                Predicted bad   Predicted good
Actual bad          2251             59
Actual good          208            774
```

Le prime feature per permutation importance nel report attuale sono:

1. `avg_line_length`
2. `stopword_ratio`
3. `language_score`
4. `avg_word_length`
5. `exclamation_ratio`

Con `--compare-models`, il report attuale registra come miglior modello per ROC-AUC medio `Extra Trees`, con ROC-AUC medio `0.9869` e F1 medio `0.8624`. Questo non sostituisce automaticamente il modello in pipeline: indica solo che vale la pena confrontare i modelli prima di un eventuale riaddestramento.

## Notebook report.ipynb

Il notebook `notebooks/report.ipynb` fornisce un'analisi interattiva del modello qualità.

Usa lo stesso modello e lo stesso test split registrato nel file `.joblib`, quindi rimane coerente con:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation
```

Il notebook include:

- caricamento del modello e dei metadati di training
- metriche globali e metriche per classe
- confusion matrix
- permutation importance delle feature
- correlation matrix
- ROC curve e Precision-Recall curve
- confronto LGBM contro LogisticRegression
- controlli di data leakage
- SHAP summary plot e SHAP bar plot

Per avviarlo:

```bash
jupyter notebook notebooks/report.ipynb
```

## Confronto modelli

La cross-validation del classificatore qualità si attiva con:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models
```

I modelli supportati sono:

- `lightgbm`
- `random_forest`
- `extra_trees`
- `logistic_regression`

Per limitarla a un sottoinsieme:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --output-dir evaluation \
  --compare-models \
  --cv-models lightgbm logistic_regression
```

Il confronto serve a verificare se LightGBM è ancora una buona scelta rispetto a baseline o modelli alternativi. Non cambia il modello salvato in `models/`.

## Esempio Completo

### Addestramento
```python
result = QualityClassifier.train_from_csv(
    csv_path="data/quality_dataset.csv",
    n_estimators=300,
    learning_rate=0.05
)
QualityClassifier.save_model(result, "models/my_model.joblib")
```

### Valutazione su Test Set
```bash
python scripts/evaluate_model.py \
    --model models/my_model.joblib \
    --test-csv data/test/dataset_test.csv \
    --output-dir evaluation \
    --threshold 0.65
```

### Visualizzare il Report HTML
1. Apri il file `evaluation/evaluation_report.html` in un browser
2. Visualizza interattivamente:
   - Metriche principali
   - Matrice di confusione
   - Top 10 features con barre colorate
   - Curve ROC e Precision-Recall

---

## Valutazione modello spam

Il modello spam è valutato con `scripts/spam/evaluate_spam_model.py`.

Il parametro `--test-csv` deve puntare a un CSV di feature spam etichettato, per esempio il file `spam_test_features.csv` prodotto dal training spam. Esempio:

```bash
python3 scripts/spam/evaluate_spam_model.py \
  --model models/spam_lgbm.joblib \
  --test-csv path/al/tuo/spam_test_features.csv \
  --output-dir evaluation/spam/train_evaluation \
  --label-column spam_target_label \
  --threshold 0.75 \
  --threshold-sweep 0.40 0.50 0.60 0.70 0.75 0.80 0.90 \
  --compare-models \
  --comparison-csv output/feature/spam_doc_features.csv
```

La label spam usa questa convenzione:

```text
ham  -> 0
spam -> 1
```

Output principali:

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

Nel repository sono già presenti report generati in:

```text
evaluation/spam/train_evaluation/
evaluation/spam/test_evaluation/
```

I file `spam_threshold_sweep.csv`, `spam_model_comparison_cv.csv` e `spam_model_comparison_report.json` vengono generati solo se si usano rispettivamente `--threshold-sweep` e `--compare-models`.

Nel filtro operativo della pipeline, un documento viene scartato come spam solo se:

1. il modello predice `spam`
2. la probabilità supera la soglia
3. sono presenti evidenze forti di spam

Se il modello sospetta spam ma mancano evidenze forti, il documento non viene scartato subito e passa alla fase di classificazione qualità.

## Errori comuni

### CSV di test non trovato

Usare il path corretto:

```bash
python3 scripts/evaluate_model.py \
  --model models/lgbm_quality_model.joblib \
  --test-csv data/splits/doc_stats_test.csv \
  --output-dir evaluation
```

### Colonne mancanti nel CSV

Controllare che il CSV contenga tutte le feature richieste da:

```text
src/blocks/classifiers.py
```

In particolare, verificare `DEFAULT_FEATURE_NAMES`.