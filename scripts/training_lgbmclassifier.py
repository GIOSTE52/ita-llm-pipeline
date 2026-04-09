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
1. Splitte il dataset in train (70%), val (15%), test (15%)
2. Allena il modello su train
3. Valuta su val
4. Salva tutte e 3 le parti per usi futuri
"""

# === Percorsi ===
csv_path = os.path.join(project_root, "output", "feature", "doc_stats_per_file.csv")
output_dir = os.path.join(project_root, "data", "splits")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# === STEP 1: Leggi il dataset completo ===
print("Caricamento dataset...")
print(f"   Percorso: {csv_path}")
df = pd.read_csv(csv_path)
print(f"   Totale documenti: {len(df)}")

# === STEP 2: Split 70% train, 15% val, 15% test ===
print("\n🔀 Split dataset (70% train, 15% val, 15% test)...")
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

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
result = QualityClassifier.train_from_csv(train_csv)

# === STEP 5: Salva il modello ===
print("\nSalvataggio del modello...")
model_path = os.path.join(project_root, "models", "lgbm_quality_model.joblib")
os.makedirs(os.path.dirname(model_path), exist_ok=True)
QualityClassifier.save_model(result, model_path)

print("\nDone!")
print("\nPuoi ora valutare il modello sul test set con:")
print(f"   python3 scripts/evaluate_model.py --model {model_path} --test-csv {test_csv} --output-dir evaluation --threshold 0.7")