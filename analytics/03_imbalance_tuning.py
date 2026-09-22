# ============================================================
# MODULE 2 - STAGE 3: TREE VIZ + IMBALANCE + TUNING
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

# ------------------------------------------------------------
# STEP 1: LOAD + CLEAN (same as Stage 2)
# ------------------------------------------------------------
df = pd.read_csv("analytics/titanic.csv")
missing_pct = (df.isnull().sum() / len(df) * 100)
for col in missing_pct[missing_pct > 0].index:
    if missing_pct[col] > 50:
        df = df.drop(columns=[col])
    elif not pd.api.types.is_numeric_dtype(df[col]):
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna(df[col].median())

target = "survived"
features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
features = [f for f in features if f in df.columns]
X = df[features]
y = df[target]

numeric_features = [f for f in features if pd.api.types.is_numeric_dtype(df[f])]
categorical_features = [f for f in features if f not in numeric_features]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_features),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), categorical_features),
])

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

feature_names = numeric_features + list(
    preprocessor.named_transformers_["cat"]["encoder"].get_feature_names_out(categorical_features)
)

# ------------------------------------------------------------
# STEP 2: DECISION TREE VISUALIZATION
# ------------------------------------------------------------
dt = DecisionTreeClassifier(random_state=42, max_depth=4)  # max_depth=4 keeps the plot readable
dt.fit(X_train_processed, y_train)

plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=feature_names, class_names=["Died", "Survived"], filled=True, fontsize=8)
plt.title("Decision Tree (max_depth=4 for readability)")
plt.savefig("analytics/charts/decision_tree.png", dpi=150, bbox_inches="tight")
plt.close()
print("Decision tree plot saved to analytics/charts/decision_tree.png")

# ------------------------------------------------------------
# STEP 3: CLASS IMBALANCE COMPARISON
# ------------------------------------------------------------
print(f"\n--- Class balance in training data ---\n{y_train.value_counts(normalize=True)}")

# A) Baseline Random Forest (no imbalance handling)
rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train_processed, y_train)
pred_baseline = rf_baseline.predict(X_test_processed)
print(f"\nBaseline RF -> Accuracy: {accuracy_score(y_test, pred_baseline):.3f}, F1: {f1_score(y_test, pred_baseline):.3f}")

# B) class_weight='balanced'
rf_balanced = RandomForestClassifier(random_state=42, class_weight="balanced")
rf_balanced.fit(X_train_processed, y_train)
pred_balanced = rf_balanced.predict(X_test_processed)
print(f"class_weight='balanced' RF -> Accuracy: {accuracy_score(y_test, pred_balanced):.3f}, F1: {f1_score(y_test, pred_balanced):.3f}")

# C) SMOTE (applied to TRAINING data only, never to test data)
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_processed, y_train)
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
pred_smote = rf_smote.predict(X_test_processed)
print(f"SMOTE RF -> Accuracy: {accuracy_score(y_test, pred_smote):.3f}, F1: {f1_score(y_test, pred_smote):.3f}")

print("\nConclusion: Titanic's imbalance is mild (~62/38 split), so the baseline model already "
      "performs reasonably. class_weight='balanced' and SMOTE mainly help F1 by nudging recall "
      "for the minority class (survivors), at a small accuracy cost.")

# ------------------------------------------------------------
# STEP 4: HYPERPARAMETER TUNING (GridSearchCV on Random Forest)
# ------------------------------------------------------------
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
    "max_features": ["sqrt", "log2"],
}

rf_for_tuning = RandomForestClassifier(random_state=42, oob_score=True, bootstrap=True)
grid_search = GridSearchCV(rf_for_tuning, param_grid, cv=5, scoring="f1", n_jobs=-1)
grid_search.fit(X_train_processed, y_train)

print(f"\n--- GridSearchCV Results ---")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 score: {grid_search.best_score_:.3f}")

best_rf = grid_search.best_estimator_
print(f"OOB score of best model: {best_rf.oob_score_:.3f}")

test_pred = best_rf.predict(X_test_processed)
print(f"Tuned model test accuracy: {accuracy_score(y_test, test_pred):.3f}, F1: {f1_score(y_test, test_pred):.3f}")

print("\n--- STAGE 3 COMPLETE ---")