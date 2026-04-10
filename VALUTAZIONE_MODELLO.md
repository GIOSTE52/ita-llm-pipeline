# 📊 Guida: Valutazione del Modello di Classificazione

## Panoramica

Hai due modi per valutare il classificatore di qualità:

### 1. **Script Python** (Consigliato per uso rapido)

```bash
python scripts/evaluate_model.py \
    --model models/lgbm_quality_model.joblib \
    --test-csv data/test/dataset_test.csv \
    --output-dir output/evaluation \
    --threshold 0.65 \
    --compare-models
```

**Parametri:**
- `--model`: Percorso al modello addestrato (.joblib)
- `--test-csv`: Percorso al CSV con feature e label
- `--output-dir`: (Opzionale) Directory per salvare report JSON, CSV e HTML
- `--threshold`: (Opzionale) Soglia di decisione (default 0.5)
- `--label-column`: (Opzionale) Nome colonna label nel CSV (default "label")
- `--compare-models`: (Opzionale) Esegue anche la cross validation comparando LightGBM con altri modelli
- `--comparison-csv`: (Opzionale) CSV etichettato da usare per la cross validation; se omesso usa `--test-csv`
- `--cv-folds`: (Opzionale) Numero di fold per il benchmark (default `5`)
- `--cv-models`: (Opzionale) Sottoinsieme di modelli da confrontare (`lightgbm`, `random_forest`, `extra_trees`, `logistic_regression`)

**Output:**
- Stampa report a schermo con metriche, confusion matrix, feature importance
- Tabella comparativa tra modelli con differenze rispetto al baseline LightGBM
- Genera `evaluation_report.json` - Dati strutturati per ulteriore analisi
- Genera `feature_importance.csv` - Importanza delle feature in formato CSV
- Genera `evaluation_report.html` - Report interattivo con grafici ROC e Precision-Recall

---

### 2. **Programmatico** (Per integrazione nel codice)

```python
from blocks.classifiers import QualityClassifier

# Carica il modello
classifier = QualityClassifier(
    model_path="models/lgbm_quality_model.joblib",
    threshold=0.65
)

# Valuta su un dataset di test
result = classifier.evaluate(
    csv_path="data/test/dataset_test.csv",
    label_column="label",
    output_dir="output/evaluation"
)

# Accedi alle metriche
print(f"Accuracy: {result['accuracy']}")
print(f"ROC-AUC: {result['roc_auc']}")
print(f"Top features: {result['top_features']}")
```

---

## Formato del CSV di Test

Il CSV deve contenere:
1. **Colonne di feature** - Contiene le features elencate in `DEFAULT_FEATURE_NAMES` all'interno di `classifiers.py`
2. **Colonna label** - Con valori "good" o "bad"
3. **Colonna doc_id** - Per tracciare i documenti

**Esempio:**
```csv
doc_id,length,white_space_ratio,non_alpha_digit_ratio,...,label
doc_001,1234,0.15,0.05,...,good
doc_002,456,0.22,0.10,...,bad
...
```

---

## Metriche Calcolate

### Globali
- **Accuracy**: Percentuale di predizioni corrette
- **Balanced Accuracy**: Accuracy pesata per classi sbilanciate
- **F1-Score**: Media armonica di precision e recall
- **ROC-AUC**: Area sotto la curva ROC (0.5 = random, 1.0 = perfetto)

### Confusion Matrix
```
              Predicted:Bad  Predicted:Good
Actual:Bad         TN              FP
Actual:Good        FN              TP
```

### Feature Importance
- **Importanza Media**: Media della variazione di performance rimuovendo ogni feature
- **Std Dev**: Deviazione standard dell'importanza

### Curve di Valutazione
- **ROC Curve**: Trade-off tra True Positive Rate e False Positive Rate
- **Precision-Recall Curve**: Trade-off tra Precision e Recall

---

## Interpretazione dei Risultati

| Metrica | Intervallo | Interpretazione |
|---------|-----------|-----------------|
| Accuracy | 0–1 | % di predizioni corrette. Alto = buono. |
| Balanced Acc | 0–1 | Come Accuracy ma pesata. Utile con classi sbilanciate. |
| ROC-AUC | 0–1 | 0.5 = casuale, 1.0 = perfetto. >0.7 = buono. |
| F1-Score | 0–1 | Bilancia precision e recall. 1.0 = perfetto. |
| Confusion Matrix | - | TN e TP grandi = buono. FP e FN piccoli = buono. |

---

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

## Risoluzione dei Problemi

### "Colonne mancanti nel CSV"
- Verifica che il CSV contenga tutte le feature richieste
- Controlla i nomi delle colonne
- Usa `DEFAULT_FEATURE_NAMES` come riferimento

### "Valori label non validi"
- Il CSV deve contenere solo "good" o "bad" nella colonna label
- Non accetta numeri (usa la mappa `LABEL_MAP = {"bad": 0, "good": 1}`)

### "Feature mancanti per doc"
- Se durante valutazione un documento non ha feature, viene saltato
- Pulisci il CSV da righe incomplete

---

## Salvare e Condividere i Risultati

I report vengono salvati in tre formati:

1. **JSON** - Dati strutturati per analisi programmatica
   ```python
   import json
   with open("evaluation_report.json") as f:
       data = json.load(f)
   ```

2. **CSV** - Feature importance in formato tabulare per Excel/Sheets

3. **HTML** - Report interattivo professionale per presentazioni

---

## Personalizzazione Avanzata

E' possibile calcolare metriche personalizzate:

```python
class MyQualityClassifier(QualityClassifier):
    def evaluate(self, csv_path, **kwargs):
        # Valutazione standard
        result = super().evaluate(csv_path, **kwargs)
        
        # Aggiungi metriche custom
        result['my_metric'] = calculate_my_metric(...)
        
        return result
```

---

