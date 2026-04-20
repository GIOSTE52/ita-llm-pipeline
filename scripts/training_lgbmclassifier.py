import sys
import os

# Aggiungi il parent directory al path per importare src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.blocks.classifiers import QualityClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

"""
Avviare come:
python3 scripts/training_lgbmclassifier.py

Questo script:
1. Effettua lo slit del dataset in train (70%), val (15%), test (15%)
2. Allena il modello su train
3. Valuta su val
4. Salva tutte e 3 le parti per usi futuri (es. per valutazione performance del modello)
"""

# === Percorsi ===
csv_path = os.path.join(project_root, "output", "feature", "doc_stats_per_file.csv")
if not os.path.exists(csv_path):
    print("Errore: file CSV non trovato.")
    print(f"Percorso atteso: {csv_path}")
    print("Esegui prima la pipeline su un dataset etichettato così che genera un dataset formato csv. Utilizza docker compose up --build")
    sys.exit(1)
output_dir = os.path.join(project_root, "data", "splits")
# scrivo i valori di default per il random_state e la threshold
random_state = 42
validation_threshold = 0.65

# Se non esiste, creo la cartella che ospiterà i 3 datasets
os.makedirs(output_dir, exist_ok=True)

# === STEP 1: Leggi il dataset completo ===
print("Caricamento dataset...")
print(f"   Percorso: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Totale documenti: {len(df)}")

# === STEP 2: Split 70% train, 15% val, 15% test ===
print("\nSplitting dataset (70% train, 15% val, 15% test)...")
train_df, temp_df = train_test_split(
    df,
    test_size=0.3,
    stratify=df["label"],
    random_state=random_state,
)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    stratify=temp_df["label"],
    random_state=random_state,
)

print(f"   Train: {len(train_df)} documenti ({len(train_df)/len(df)*100:.1f}%)")
print(f"   Val:   {len(val_df)} documenti ({len(val_df)/len(df)*100:.1f}%)")
print(f"   Test:  {len(test_df)} documenti ({len(test_df)/len(df)*100:.1f}%)")

# === STEP 3: Salva i tre set ===
print(f"\nSalvataggio dei dataset splittati in: {output_dir}")
train_csv = os.path.join(output_dir, "doc_stats_train.csv")
val_csv = os.path.join(output_dir, "doc_stats_val.csv")
test_csv = os.path.join(output_dir, "doc_stats_test.csv")

train_df.to_csv(train_csv, index=False)
val_df.to_csv(val_csv, index=False)
test_df.to_csv(test_csv, index=False)

print(f"Training: {train_csv}")
print(f"Validation: {val_csv}")
print(f"Test: {test_csv}")

# === STEP 4: Allena il modello sul training set ===
print("\nAddestramento del modello...")
result = QualityClassifier.train_from_csv(
    csv_path=train_csv,
    validation_csv_path=val_csv,
    threshold=validation_threshold,
    random_state=random_state,
)
# scrivo metadata possibilmente utili in futuro
result["training_metadata"] = {
    "source_csv": os.path.abspath(csv_path),
    "train_csv": os.path.abspath(train_csv),
    "validation_csv": os.path.abspath(val_csv),
    "test_csv": os.path.abspath(test_csv),
    "split_strategy": "train_val_test",
    "split_random_state": random_state,
    "train_fraction": 0.70,
    "validation_fraction": 0.15,
    "test_fraction": 0.15,
    "train_rows": int(len(train_df)),
    "validation_rows": int(len(val_df)),
    "test_rows": int(len(test_df)),
    "validation_threshold": validation_threshold,
}

# === STEP 5: Salva il modello ===
print("\nSalvataggio del modello...")
model_path = os.path.join(project_root, "models", "lgbm_quality_model.joblib")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
QualityClassifier.save_model(result, model_path)

print("\nSplit registrati nel modello e salvati in data/splits:")
print(f"   Train: {train_csv}")
print(f"   Val:   {val_csv}")
print(f"   Test:  {test_csv}")
print("\nOra puoi valutare il modello sul test set")
