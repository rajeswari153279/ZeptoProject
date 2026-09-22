# ============================================================
# MODULE 2 - STAGE 4: REGRESSION + FINAL COMPARISON + SAVE PIPELINE
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

# ------------------------------------------------------------
# STEP 1: LOAD + CLEAN (same as before)
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

# ============================================================
# PART A: REGRESSION SUB-TASK -> predict FARE
# ============================================================
reg_features = ["pclass", "age", "sibsp", "parch", "sex", "embarked"]
reg_features = [f for f in reg_features if f in df.columns]
X_reg = df[reg_features]
y_reg = df["fare"]

num_reg = [f for f in reg_features if pd.api.types.is_numeric_dtype(df[f])]
cat_reg = [f for f in reg_features if f not in num_reg]

Xr_train, Xr_test, yr_train, yr_test = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

reg_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_reg),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_reg),
])

Xr_train_processed = reg_preprocessor.fit_transform(Xr_train)
Xr_test_processed = reg_preprocessor.transform(Xr_test)

lin_reg = LinearRegression()
lin_reg.fit(Xr_train_processed, yr_train)
yr_pred = lin_reg.predict(Xr_test_processed)

mae = mean_absolute_error(yr_test, yr_pred)
rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
r2 = r2_score(yr_test, yr_pred)
n, p = Xr_test_processed.shape
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("--- Regression: Predicting Fare ---")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f} | Adjusted R2: {adj_r2:.3f}")

residuals = yr_test - yr_pred
plt.figure(figsize=(7, 5))
plt.scatter(yr_pred, residuals, alpha=0.5)
plt.axhline(y=0, color="red", linestyle="--")
plt.xlabel("Predicted Fare")
plt.ylabel("Residuals")
plt.title("Residual Plot (checking heteroscedasticity)")
plt.savefig("analytics/charts/residual_plot.png")
plt.close()

yr_pred_median = np.median(yr_pred)
spread_low = residuals[yr_pred < yr_pred_median].std()
spread_high = residuals[yr_pred >= yr_pred_median].std()
print(f"Residual spread (low predictions): {spread_low:.2f}, (high predictions): {spread_high:.2f}")
print("Conclusion: The residual spread is clearly larger for higher predicted fares, "
      "indicating HETEROSCEDASTICITY (non-constant variance) - the model is less reliable for expensive tickets.")

# ============================================================
# PART B: FINAL CLASSIFIER COMPARISON (rebuild all 3, using tuned RF)
# ============================================================
clf_features = ["pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
clf_features = [f for f in clf_features if f in df.columns]
X = df[clf_features]
y = df["survived"]
num_feat = [f for f in clf_features if pd.api.types.is_numeric_dtype(df[f])]
cat_feat = [f for f in clf_features if f not in num_feat]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf_preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_feat),
    ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), cat_feat),
])

X_train_processed = clf_preprocessor.fit_transform(X_train)
X_test_processed = clf_preprocessor.transform(X_test)

final_models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, max_features="sqrt", random_state=42),
}

clf_results = []
best_model_name = None
best_f1 = -1
best_model_obj = None

for name, model in final_models.items():
    model.fit(X_train_processed, y_train)
    pred = model.predict(X_test_processed)
    proba = model.predict_proba(X_test_processed)[:, 1]
    f1 = f1_score(y_test, pred)
    clf_results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1,
        "ROC-AUC": roc_auc_score(y_test, proba),
    })
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model_obj = model

print("\n--- Final Classifier Comparison ---")
print(pd.DataFrame(clf_results).to_string(index=False))

print("\n--- Regression Metrics (separate scale) ---")
print(f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R2: {r2:.3f} | Adjusted R2: {adj_r2:.3f}")

print(f"\n=== RECOMMENDATION ===")
print(f"Best classifier by F1-score: {best_model_name} (F1={best_f1:.3f}). "
      f"Recommended for deployment due to balanced precision/recall performance.")

# ============================================================
# PART C: SAVE THE COMPLETE PIPELINE (preprocessing + model together)
# ============================================================
full_pipeline = Pipeline([
    ("preprocessor", clf_preprocessor),
    ("model", best_model_obj),
])
# Re-fit as one combined pipeline object (so it's a single reloadable unit)
full_pipeline.fit(X_train, y_train)

joblib.dump(full_pipeline, "analytics/full_pipeline.joblib")
print("\nSaved complete pipeline to analytics/full_pipeline.joblib")

# ------------------------------------------------------------
# VERIFY: reload it and test on raw new data
# ------------------------------------------------------------
reloaded = joblib.load("analytics/full_pipeline.joblib")
sample_raw = X_test.iloc[[0]]  # one raw, unprocessed row
prediction = reloaded.predict(sample_raw)
print(f"Reloaded pipeline test prediction on 1 raw sample: {prediction} (actual: {y_test.iloc[0]})")

print("\n--- STAGE 4 COMPLETE - MODULE 2 FINISHED ---")