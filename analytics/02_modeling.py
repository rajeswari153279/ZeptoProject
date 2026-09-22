# ============================================================
# MODULE 2 - STAGE 2: PREPROCESSING + SPLIT + 3 CLASSIFIERS
# ============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix
)

# ------------------------------------------------------------
# STEP 1: LOAD (reuse the offline fallback saved in Stage 1)
# ------------------------------------------------------------
df = pd.read_csv("analytics/titanic.csv")
print(f"Loaded titanic.csv: {df.shape}")

# Basic cleaning (same threshold rule as Stage 1)
missing_pct = (df.isnull().sum() / len(df) * 100)
for col in missing_pct[missing_pct > 0].index:
    if missing_pct[col] > 50:
        df = df.drop(columns=[col])
    elif not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

# ------------------------------------------------------------
# STEP 2: SELECT FEATURES + TARGET
# ------------------------------------------------------------
target = "survived"
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
features = [f for f in features if f in df.columns]

X = df[features]
y = df[target]

numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
categorical_features = [f for f in features if f not in numeric_features]
print(f"\nNumeric features: {numeric_features}")
print(f"Categorical features: {categorical_features}")

# ------------------------------------------------------------
# STEP 3: STRATIFIED TRAIN/TEST SPLIT (before any preprocessing)
# ------------------------------------------------------------
# We use stratify=y so the train and test sets keep the SAME survival
# ratio as the full dataset. Without this, a random split could put too
# many/few survivors in either set, skewing evaluation.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
print(f"Train survival rate: {y_train.mean():.3f}, Test survival rate: {y_test.mean():.3f}")

# ------------------------------------------------------------
# STEP 4: PREPROCESSING PIPELINE (fit on TRAIN only)
# ------------------------------------------------------------
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features),
])

# fit_transform on TRAIN (learns the imputation values / scaling / categories)
X_train_processed = preprocessor.fit_transform(X_train)
# transform ONLY (never fit) on TEST - this avoids data leakage
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed train shape: {X_train_processed.shape}")
print(f"Processed test shape: {X_test_processed.shape}")

# ------------------------------------------------------------
# STEP 5: TRAIN 3 CLASSIFIERS
# ------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
}

results = []
for name, model in models.items():
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)
    y_proba = model.predict_proba(X_test_processed)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f} | F1: {f1:.3f} | ROC-AUC: {auc:.3f}")
    print(f"Confusion Matrix:\n{cm}")

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1": f1, "ROC-AUC": auc
    })

print("\n--- Model Comparison Table ---")
print(pd.DataFrame(results).to_string(index=False))
print("\n--- STAGE 2 COMPLETE ---")