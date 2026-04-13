import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)

data = load_breast_cancer()

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print("\n--- First 5 rows ---")
print(df.head())

print("\n--- Dataset shape ---")
print(df.shape)

print("\n--- Feature names ---")
print(data.feature_names)

print("\n--- Target distribution ---")
print(df['target'].value_counts())

print("\n--- Discussion ---")
print("Dataset is slightly imbalanced (more benign than malignant).")

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

def evaluate_model(y_test, y_pred, model_name):
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"\n--- {model_name} ---")
    print("Accuracy:", acc)
    print("Precision:", prec)
    print("Recall:", rec)
    print("F1-score:", f1)

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    return acc, prec, rec, f1

metrics_lr = evaluate_model(y_test, y_pred_lr, "Logistic Regression")
metrics_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

def plot_conf_matrix(y_test, y_pred, title):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plot_conf_matrix(y_test, y_pred_lr, "Logistic Regression Confusion Matrix")
plot_conf_matrix(y_test, y_pred_rf, "Random Forest Confusion Matrix")

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

auc_lr = auc(fpr_lr, tpr_lr)
auc_rf = auc(fpr_rf, tpr_rf)

plt.figure()
plt.plot(fpr_lr, tpr_lr, label=f"Logistic Regression (AUC={auc_lr:.3f})")
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC={auc_rf:.3f})")
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

labels = ['Accuracy', 'Precision', 'Recall', 'F1']

lr_vals = list(metrics_lr)
rf_vals = list(metrics_rf)

x = np.arange(len(labels))
width = 0.35

plt.figure()
plt.bar(x - width/2, lr_vals, width, label='Logistic Regression')
plt.bar(x + width/2, rf_vals, width, label='Random Forest')

plt.xticks(x, labels)
plt.title("Model Comparison")
plt.legend()
plt.show()

print("\n--- Interpretation ---")

print("1. Better overall model: Usually Random Forest (handles non-linearity better).")
print("2. Better Recall: Check above values (important for cancer detection).")
print("3. Better Precision: Depends on FP tradeoff.")
print("4. Better AUC:", "LR" if auc_lr > auc_rf else "RF")
print("5. Accuracy alone is not enough — confusion matrix shows FN/FP importance.")

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

precisions = []
recalls = []
f1s = []

for t in thresholds:
    y_pred_thresh = (y_prob_lr >= t).astype(int)

    precisions.append(precision_score(y_test, y_pred_thresh))
    recalls.append(recall_score(y_test, y_pred_thresh))
    f1s.append(f1_score(y_test, y_pred_thresh))

plt.figure()
plt.plot(thresholds, precisions, marker='o', label='Precision')
plt.plot(thresholds, recalls, marker='o', label='Recall')
plt.plot(thresholds, f1s, marker='o', label='F1-score')

plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold Tuning (Logistic Regression)")
plt.legend()
plt.show()

print("\n--- Threshold Insight ---")
print("Lower threshold → higher recall, lower precision")
print("Higher threshold → higher precision, lower recall")