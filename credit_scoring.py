

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")            # save charts as image files (no pop-up windows)
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

import os
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

np.random.seed(42)        # makes results repeatable every time we run it
sns.set_style("whitegrid")

#  CREATE A SAMPLE DATASET
print("=" * 60)
print("STEP 1: Creating sample credit dataset")
print("=" * 60)

n = 1000   # number of "people" (rows) in our dataset

annual_income = np.random.normal(50000, 18000, n).clip(15000, 150000)
existing_debt = np.random.normal(12000, 8000, n).clip(0, 60000)
loan_amount = np.random.normal(15000, 7000, n).clip(1000, 50000)
payment_history_score = np.random.normal(70, 15, n).clip(0, 100)
employment_years = np.random.normal(6, 4, n).clip(0, 30)
age = np.random.normal(38, 11, n).clip(18, 70)

# A simple "debt-to-income ratio" - a well-known real banking feature
debt_to_income = (existing_debt + loan_amount) / annual_income

# Good payment history + low debt-to-income + stable job => good credit
score = (
    (payment_history_score / 100) * 0.5
    + (1 - np.clip(debt_to_income, 0, 1)) * 0.3
    + (np.clip(employment_years, 0, 10) / 10) * 0.2
)

score += np.random.normal(0, 0.08, n)
creditworthy = (score > 0.55).astype(int)

df = pd.DataFrame({
    "annual_income": annual_income.round(0),
    "existing_debt": existing_debt.round(0),
    "loan_amount": loan_amount.round(0),
    "payment_history_score": payment_history_score.round(1),
    "employment_years": employment_years.round(1),
    "age": age.round(0),
    "debt_to_income": debt_to_income.round(2),
    "creditworthy": creditworthy
})

df.to_csv(f"{OUT_DIR}/credit_dataset.csv", index=False)
print(f"Dataset created: {df.shape[0]} people, {df.shape[1]-1} features")
print("\nCreditworthy count (1 = good credit, 0 = bad credit):")
print(df["creditworthy"].value_counts())

# QUICK LOOK AT THE DATA (simple chart)
plt.figure(figsize=(5, 4))
sns.countplot(x="creditworthy", data=df, palette=["#7B2732", "#5A9272"])
plt.title("Creditworthy vs Not Creditworthy")
plt.xlabel("0 = Bad Credit, 1 = Good Credit")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/01_class_distribution.png", dpi=150)
plt.close()
print("\nSaved chart: 01_class_distribution.png")

# SPLIT DATA INTO TRAIN / TEST
print("\n" + "=" * 60)
print("STEP 2: Splitting data (80% train, 20% test)")
print("=" * 60)

X = df.drop(columns=["creditworthy"])
y = df["creditworthy"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scaling: puts all numbers on a similar range (helps Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Train set: {X_train.shape[0]} people | Test set: {X_test.shape[0]} people")

# TRAIN THE TWO MODELS

print("\n" + "=" * 60)
print("STEP 3: Training models")
print("=" * 60)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = []
roc_data = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc
    })

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_data[name] = (fpr, tpr, auc)

    # Confusion matrix chart
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Bad Credit", "Good Credit"],
                yticklabels=["Bad Credit", "Good Credit"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"{OUT_DIR}/cm_{safe_name}.png", dpi=150)
    plt.close()

    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred, target_names=["Bad Credit", "Good Credit"]))

results_df = pd.DataFrame(results).sort_values("ROC-AUC", ascending=False)
results_df.to_csv(f"{OUT_DIR}/model_comparison.csv", index=False)

# COMPARE THE TWO MODELS
print("\n" + "=" * 60)
print("STEP 4: Model comparison")
print("=" * 60)
print(results_df.to_string(index=False))

plot_df = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]]
plot_df.plot(kind="bar", figsize=(8, 5), colormap="viridis")
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.ylim(0, 1.05)
plt.xticks(rotation=0)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/02 comparison_bar.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Random Guess")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/03_roc_curves.png", dpi=150)
plt.close()

# Decision tree
print("\n" + "=" * 60)
print("STEP 5: Which features matter most (Decision Tree)")
print("=" * 60)

tree_model = models["Decision Tree"]
importance = pd.Series(tree_model.feature_importances_, index=X.columns)
importance = importance.sort_values(ascending=False)
print(importance)

plt.figure(figsize=(7, 5))
sns.barplot(x=importance.values, y=importance.index, palette="mako")
plt.title("Feature Importance - Decision Tree")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/04_feature_importance.png", dpi=150)
plt.close()

# Draw the actual decision tree so you can SEE how it makes decisions
plt.figure(figsize=(16, 8))
plot_tree(tree_model, feature_names=X.columns, class_names=["Bad", "Good"],
          filled=True, rounded=True, fontsize=8)
plt.title("Decision Tree - How It Decides Creditworthiness")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/05_decision_tree_diagram.png", dpi=150)
plt.close()

print("\nSaved charts: 02_model_comparison_bar.png, 03_roc_curves.png,")
print("              04_feature_importance.png, 05_decision_tree_diagram.png")

# simple Summary
best_model = results_df.iloc[0]["Model"]
with open(f"{OUT_DIR}/results_summary.txt", "w") as f:
    f.write("CodeAlpha ML Internship - Task 1: Credit Scoring Model\n")
    f.write("=" * 60 + "\n\n")
    f.write(f"Dataset: {df.shape[0]} people, {X.shape[1]} features (sample data)\n\n")
    f.write("Model Comparison:\n\n")
    f.write(results_df.to_string(index=False))
    f.write(f"\n\nBest Model: {best_model}\n")
    f.write("\nTop 3 Most Important Features:\n")
    f.write(importance.head(3).to_string())

print(f"\nBest performing model: {best_model}")

