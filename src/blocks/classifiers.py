import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

import lightgbm as lgb


# -----------------------
# 1. Load dataset
# -----------------------

df = pd.read_csv("quality_dataset.csv")

# Separate features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Encode labels automatically (good/middle/bad → 0/1/2)
y = y.astype("category")
label_mapping = dict(enumerate(y.cat.categories))
y = y.cat.codes


# -----------------------
# 2. Train / validation split
# -----------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)


# -----------------------
# 3. Train model
# -----------------------

model = lgb.LGBMClassifier(
    objective="multiclass",
    num_class=3,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=-1,
    random_state=42
)

model.fit(X_train, y_train)


# -----------------------
# 4. Evaluation
# -----------------------

y_pred = model.predict(X_val)

print("Classification Report:")
print(classification_report(y_val, y_pred, target_names=label_mapping.values()))

print("Confusion Matrix:")
print(confusion_matrix(y_val, y_pred))


# -----------------------
# 5. Permutation Feature Importance
# -----------------------

print("\nComputing permutation importance...")
result = permutation_importance(
    model,
    X_val,
    y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

importance_df = pd.DataFrame({
    "feature": X.columns,
    "importance_mean": result.importances_mean,
    "importance_std": result.importances_std
}).sort_values(by="importance_mean", ascending=False)

print("\nFeature Importance:")
print(importance_df)