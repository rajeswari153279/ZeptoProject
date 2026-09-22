# ============================================================
# MODULE 2 - STAGE 1: DATA UNDERSTANDING + EDA
# ============================================================

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# ------------------------------------------------------------
# STEP 1: LOAD THE DATASET (only once, save offline copy)
# ------------------------------------------------------------
if os.path.exists("analytics/titanic.csv"):
    df = pd.read_csv("analytics/titanic.csv")
    print("Loaded from local titanic.csv (offline fallback)")
else:
    df = sns.load_dataset("titanic")
    df.to_csv("analytics/titanic.csv", index=False)
    print("Loaded from Seaborn and saved as analytics/titanic.csv")

print(f"\nShape: {df.shape}")
print(f"\nData types:\n{df.dtypes}")

# ------------------------------------------------------------
# STEP 2: MISSING VALUES
# ------------------------------------------------------------
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_report = pd.DataFrame({"missing_count": missing, "missing_pct": missing_pct})
missing_report = missing_report[missing_report["missing_count"] > 0].sort_values("missing_pct", ascending=False)
print(f"\n--- Missing Value Report ---\n{missing_report}")

# Threshold rule: if a column is missing > 50%, drop it (too little info to use).
# If missing <= 50%, impute it (fill with a reasonable value) instead of dropping rows.
for col in missing_report.index:
    pct = missing_report.loc[col, "missing_pct"]
    if pct > 50:
        print(f"'{col}': {pct}% missing -> DROPPED (exceeds 50% threshold)")
        df = df.drop(columns=[col])
    else:
        if not pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].mode()[0]
            df[col] = df[col].fillna(fill_value)
            print(f"'{col}': {pct}% missing -> imputed with mode ('{fill_value}') since <=50% threshold")
        else:
            fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)
            print(f"'{col}': {pct}% missing -> imputed with median ({fill_value}) since <=50% threshold")

# ------------------------------------------------------------
# STEP 3: OUTLIERS (IQR method) for age and fare
# ------------------------------------------------------------
def iqr_outlier_count(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return ((series < lower) | (series > upper)).sum()

age_outliers = iqr_outlier_count(df["age"])
fare_outliers = iqr_outlier_count(df["fare"])
print(f"\n--- Outlier Counts (IQR method) ---")
print(f"Age outliers: {age_outliers}")
print(f"Fare outliers: {fare_outliers}")

mean_fare = df["fare"].mean()
median_fare = df["fare"].median()
mode_fare = df["fare"].mode()[0]
print(f"\nFare -> mean: {mean_fare:.2f}, median: {median_fare:.2f}, mode: {mode_fare:.2f}")
print("Since mean > median > mode, Fare is right-skewed (a few very high fares pull the average up).")

# ------------------------------------------------------------
# STEP 4: BIVARIATE ANALYSIS - survival rates
# ------------------------------------------------------------
print(f"\n--- Survival rate by sex ---\n{df.groupby('sex')['survived'].mean()}")
print(f"\n--- Survival rate by pclass ---\n{df.groupby('pclass')['survived'].mean()}")
print(f"\n--- Survival rate by sex + pclass ---\n{df.groupby(['sex', 'pclass'])['survived'].mean()}")

# ------------------------------------------------------------
# STEP 5: CORRELATION MATRIX (6 specified columns)
# ------------------------------------------------------------
corr_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
corr_matrix = df[corr_cols].corr()
print(f"\n--- Correlation Matrix ---\n{corr_matrix}")

# Find the two strongest off-diagonal correlations
corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
corr_pairs = corr_pairs[corr_pairs < 1.0]  # remove self-correlation (always 1.0)
top_pairs = corr_pairs.drop_duplicates().head(2)
print(f"\nTwo strongest correlations:\n{top_pairs}")

os.makedirs("analytics/charts", exist_ok=True)

plt.figure(figsize=(7, 5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("analytics/charts/correlation_heatmap.png")
plt.close()

# ------------------------------------------------------------
# STEP 6: REQUIRED CHARTS (4+)
# ------------------------------------------------------------
plt.figure(figsize=(7, 5))
sns.histplot(df["age"], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("analytics/charts/age_histogram.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.boxplot(x=df["fare"])
plt.title("Fare Boxplot")
plt.savefig("analytics/charts/fare_boxplot.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.barplot(x="pclass", y="survived", data=df)
plt.title("Survival Rate by Passenger Class")
plt.savefig("analytics/charts/survival_by_pclass.png")
plt.close()

plt.figure(figsize=(7, 5))
sns.barplot(x="sex", y="survived", hue="pclass", data=df)
plt.title("Survival Rate by Sex and Class")
plt.savefig("analytics/charts/survival_by_sex_pclass.png")
plt.close()

print("\n4 charts saved to analytics/charts/")
print("\n--- EDA COMPLETE ---")