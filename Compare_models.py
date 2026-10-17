import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")  # avoids needing a display; saves chart straight to file
import matplotlib.pyplot as plt

# --- Step 1: Load dataset ONCE and recreate the exact same test split used in every run ---
df = pd.read_csv("data/UCI_Credit_Card.csv")
if "ID" in df.columns:
    df = df.drop(columns=["ID"])

target_col = "default.payment.next.month" if "default.payment.next.month" in df.columns else "default"
y = df[target_col]
X = df.drop(columns=[target_col])

# --- Step 2: List every saved model version to compare ---
# Each entry: (Label to show, filename)
model_files = [
    ("Baseline XGBoost",              "api/model/credit_risk_model_backup.pkl"),
    ("Tuned (scale_pos_weight)",      "api/model/credit_risk_model_backup_before_smote.pkl"),
    ("Plain SMOTE",                   "api/model/credit_risk_model_backup_before_kmeans_smote.pkl"),
    ("K-Means SMOTE",                 "api/model/credit_risk_model_backup_before_combined.pkl"),
    ("K-Means SMOTE + Tuning",        "api/model/credit_risk_model.pkl"),
]

results = []

for label, path in model_files:
    if not os.path.exists(path):
        print(f"⚠️  Skipping '{label}' — file not found: {path}")
        continue

    bundle = joblib.load(path)
    feature_adder = bundle['feature_adder']
    pipeline = bundle['pipeline']

    # Apply this model's own feature engineering, then split with the SAME
    # random_state/test_size used during every training run, so the test set
    # is identical across all models — this is what makes the comparison fair.
    X_engineered = feature_adder.transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=0.2, random_state=42, stratify=y
    )

    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    results.append({
        "Model": label,
        "Accuracy": acc,
        "Default Precision": prec,
        "Default Recall": rec,
        "Default F1": f1,
        "Confusion Matrix": cm
    })

# --- Step 3: Print a clean comparison table ---
results_df = pd.DataFrame(results).drop(columns=["Confusion Matrix"])
results_df_display = results_df.copy()
for col in ["Accuracy", "Default Precision", "Default Recall", "Default F1"]:
    results_df_display[col] = (results_df_display[col] * 100).round(1)

print("\n" + "=" * 80)
print("MODEL COMPARISON — all values in %")
print("=" * 80)
print(results_df_display.to_string(index=False))
print("=" * 80)

# --- Step 4: Recommend the best model ---
# We recommend based on Default F1 (best balance of catching defaulters
# without too many false alarms) since that's the metric that matters most
# for a real credit risk system — not raw accuracy alone.
best_by_f1 = results_df.loc[results_df["Default F1"].idxmax()]
best_by_accuracy = results_df.loc[results_df["Accuracy"].idxmax()]

print(f"\n🏆 Best by Accuracy:      {best_by_accuracy['Model']} ({best_by_accuracy['Accuracy']*100:.2f}%)")
print(f"🏆 Best by Default F1:    {best_by_f1['Model']} (F1 = {best_by_f1['Default F1']:.3f})")

print("\n--- Recommendation ---")
print(f"For a credit risk system, catching actual defaulters matters more than raw accuracy.")
print(f"Recommended model: '{best_by_f1['Model']}'")
print(f"  Accuracy: {best_by_f1['Accuracy']*100:.2f}%")
print(f"  Default Precision: {best_by_f1['Default Precision']*100:.1f}%")
print(f"  Default Recall: {best_by_f1['Default Recall']*100:.1f}%")
print(f"  Default F1-score: {best_by_f1['Default F1']:.3f}")

# --- Step 5: Save a bar chart comparing all models visually ---
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(results_df))
width = 0.2

ax.bar(x - 1.5*width, results_df["Accuracy"] * 100, width, label="Accuracy")
ax.bar(x - 0.5*width, results_df["Default Precision"] * 100, width, label="Default Precision")
ax.bar(x + 0.5*width, results_df["Default Recall"] * 100, width, label="Default Recall")
ax.bar(x + 1.5*width, results_df["Default F1"] * 100, width, label="Default F1")

ax.set_ylabel("Score (%)")
ax.set_title("Model Comparison — Credit Default Risk Prediction")
ax.set_xticks(x)
ax.set_xticklabels(results_df["Model"], rotation=20, ha="right")
ax.legend()
ax.set_ylim(0, 100)
plt.tight_layout()

os.makedirs("reports", exist_ok=True)
chart_path = "reports/model_comparison_chart.png"
plt.savefig(chart_path, dpi=150)
print(f"\n📊 Comparison chart saved to {chart_path}")
print("You can open this image and show it directly to your guide.")