import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc

digits = load_digits()
X = digits.data
y = digits.target

print(X.shape)
print(len(np.unique(y)))
print(np.unique(y))

plt.matshow(digits.images[0])
plt.show()

print("Min pixel:", np.min(X))
print("Max pixel:", np.max(X))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

log_reg = LogisticRegression(max_iter=1000, multi_class='ovr')
svm = SVC(probability=True)

log_reg.fit(X_train_scaled, y_train)
svm.fit(X_train_scaled, y_train)

y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_svm = svm.predict(X_test_scaled)

print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

cm_lr = confusion_matrix(y_test, y_pred_lr)
cm_svm = confusion_matrix(y_test, y_pred_svm)

plt.figure(figsize=(8,6))
sns.heatmap(cm_lr, annot=True, fmt='d')
plt.title("Logistic Regression Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(cm_svm, annot=True, fmt='d')
plt.title("SVM Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("Logistic Regression Report")
print(classification_report(y_test, y_pred_lr))

print("SVM Report")
print(classification_report(y_test, y_pred_svm))

for name, y_pred in [("LogReg", y_pred_lr), ("SVM", y_pred_svm)]:
    print(name)
    print("Macro Precision:", precision_score(y_test, y_pred, average='macro'))
    print("Macro Recall:", recall_score(y_test, y_pred, average='macro'))
    print("Macro F1:", f1_score(y_test, y_pred, average='macro'))
    print("Weighted Precision:", precision_score(y_test, y_pred, average='weighted'))
    print("Weighted Recall:", recall_score(y_test, y_pred, average='weighted'))
    print("Weighted F1:", f1_score(y_test, y_pred, average='weighted'))

lb = LabelBinarizer()
y_test_bin = lb.fit_transform(y_test)

y_score_lr = log_reg.predict_proba(X_test_scaled)
y_score_svm = svm.predict_proba(X_test_scaled)

for model_name, y_score in [("LogReg", y_score_lr), ("SVM", y_score_svm)]:
    plt.figure(figsize=(10,8))
    for i in range(10):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {i} (AUC={roc_auc:.2f})")
    plt.plot([0,1],[0,1],'k--')
    plt.title(f"{model_name} ROC Curves")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

indices = np.random.choice(len(X_test), 10, replace=False)

plt.figure(figsize=(10,4))
for i, idx in enumerate(indices):
    plt.subplot(2,5,i+1)
    plt.imshow(X_test[idx].reshape(8,8), cmap='gray')
    pred = y_pred_lr[idx]
    true = y_test[idx]
    color = 'green' if pred == true else 'red'
    plt.title(f"T:{true} P:{pred}", color=color)
    plt.axis('off')
plt.show()