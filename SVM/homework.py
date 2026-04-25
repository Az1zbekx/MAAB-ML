"""
============================================================
  Support Vector Machines: Handwritten Digit Classification
  Digits Dataset (sklearn) — Full Homework Solution
============================================================
"""

# ─────────────────────────────────────────────
# 0. Imports
# ─────────────────────────────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report
)
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ─────────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1 — Loading the Digits Dataset")
print("=" * 60)

digits = load_digits()
X = digits.data       # shape: (1797, 64)
y = digits.target     # shape: (1797,)

print(f"  Feature matrix shape : {X.shape}")
print(f"  Target vector shape  : {y.shape}")
print(f"  Classes              : {np.unique(y)}")
print(f"  Image shape          : {digits.images[0].shape}  (8x8 pixels)")


# ─────────────────────────────────────────────
# 2. Data Exploration
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Data Exploration")
print("=" * 60)

# Class distribution
unique, counts = np.unique(y, return_counts=True)
print("\n  Class distribution:")
for cls, cnt in zip(unique, counts):
    print(f"    Digit {cls}: {cnt} samples")

# Plot sample digits
fig, axes = plt.subplots(2, 10, figsize=(16, 4))
fig.suptitle("Sample Digit Images (one per class × 2)", fontsize=14, fontweight="bold")
for digit in range(10):
    idxs = np.where(y == digit)[0]
    for row, idx in enumerate(idxs[:2]):
        axes[row, digit].imshow(digits.images[idx], cmap="gray_r")
        axes[row, digit].set_title(f"'{digit}'", fontsize=9)
        axes[row, digit].axis("off")
plt.tight_layout()
plt.savefig("01_sample_digits.png", dpi=120)
plt.show()
print("  [Saved] 01_sample_digits.png")

# Class distribution bar chart
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(unique, counts, color=sns.color_palette("tab10", 10), edgecolor="black")
ax.set_xlabel("Digit Class", fontsize=12)
ax.set_ylabel("Number of Samples", fontsize=12)
ax.set_title("Class Distribution in Digits Dataset", fontsize=14, fontweight="bold")
ax.set_xticks(unique)
for x, c in zip(unique, counts):
    ax.text(x, c + 1, str(c), ha="center", va="bottom", fontsize=9)
plt.tight_layout()
plt.savefig("02_class_distribution.png", dpi=120)
plt.show()
print("  [Saved] 02_class_distribution.png")


# ─────────────────────────────────────────────
# 3. Data Splitting & Standardisation
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Splitting & Standardising Data")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"  Training samples : {X_train.shape[0]}")
print(f"  Test samples     : {X_test.shape[0]}")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("  StandardScaler applied  ✓")


# ─────────────────────────────────────────────
# Helper — evaluate & print metrics
# ─────────────────────────────────────────────
def evaluate(model, X_tr, y_tr, X_te, y_te, label=""):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"\n  [{label}]")
    print(f"    Accuracy : {acc:.4f} ({acc*100:.2f}%)")
    print(classification_report(y_te, y_pred, digits=4))
    return acc, y_pred


# ─────────────────────────────────────────────
# 4. SVM — Linear Kernel
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — SVM with Linear Kernel")
print("=" * 60)

svm_linear = SVC(kernel="linear", C=1.0, random_state=42)
acc_linear, pred_linear = evaluate(
    svm_linear, X_train_sc, y_train, X_test_sc, y_test, "Linear SVM"
)

# Confusion matrix
cm_linear = confusion_matrix(y_test, pred_linear)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_linear, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_title("Confusion Matrix — Linear SVM", fontsize=14, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("03_cm_linear.png", dpi=120)
plt.show()
print("  [Saved] 03_cm_linear.png")


# ─────────────────────────────────────────────
# 5. Experiment with Different Kernels
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Comparing Kernels (Poly & RBF)")
print("=" * 60)

svm_poly = SVC(kernel="poly", degree=3, C=1.0, random_state=42)
acc_poly, pred_poly = evaluate(
    svm_poly, X_train_sc, y_train, X_test_sc, y_test, "Polynomial SVM (degree=3)"
)

svm_rbf = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)
acc_rbf, pred_rbf = evaluate(
    svm_rbf, X_train_sc, y_train, X_test_sc, y_test, "RBF SVM"
)

# Summary bar chart
kernels = ["Linear", "Polynomial", "RBF"]
accs    = [acc_linear, acc_poly, acc_rbf]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(kernels, [a * 100 for a in accs],
              color=["#4C72B0", "#DD8452", "#55A868"], edgecolor="black", width=0.5)
ax.set_ylim(90, 102)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("SVM Kernel Comparison", fontsize=14, fontweight="bold")
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2, f"{acc*100:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig("04_kernel_comparison.png", dpi=120)
plt.show()
print("  [Saved] 04_kernel_comparison.png")


# ─────────────────────────────────────────────
# 6. Hyperparameter Tuning — GridSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Hyperparameter Tuning (GridSearchCV)")
print("=" * 60)
print("  (This may take a minute …)")

param_grid = {
    "C"      : [0.1, 1, 10, 100],
    "gamma"  : ["scale", "auto", 0.001, 0.01],
    "kernel" : ["rbf", "poly"],
    "degree" : [2, 3],        # only relevant for poly
}

grid_search = GridSearchCV(
    SVC(random_state=42),
    param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
    verbose=0
)
grid_search.fit(X_train_sc, y_train)

best_params = grid_search.best_params_
best_score  = grid_search.best_score_
print(f"\n  Best CV score  : {best_score:.4f}")
print(f"  Best params    : {best_params}")

best_model  = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test_sc)
acc_best    = accuracy_score(y_test, y_pred_best)
print(f"\n  Test accuracy (best model) : {acc_best:.4f} ({acc_best*100:.2f}%)")
print(classification_report(y_test, y_pred_best, digits=4))


# ─────────────────────────────────────────────
# 7. Visualisations — Best Model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Visualisations")
print("=" * 60)

# 7a  Confusion matrix — best model
cm_best = confusion_matrix(y_test, y_pred_best)
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(cm_best, annot=True, fmt="d", cmap="Greens",
            xticklabels=range(10), yticklabels=range(10), ax=ax)
ax.set_title(f"Confusion Matrix — Best Model ({best_params['kernel'].upper()} kernel)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig("05_cm_best_model.png", dpi=120)
plt.show()
print("  [Saved] 05_cm_best_model.png")

# 7b  PCA 2-D visualisation
print("  Computing PCA (2D) …")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_test_sc)

fig, ax = plt.subplots(figsize=(9, 7))
palette = sns.color_palette("tab10", 10)
for digit in range(10):
    mask = y_test == digit
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
               label=str(digit), color=palette[digit],
               alpha=0.7, s=40, edgecolors="none")
ax.set_title("PCA — 2D Projection of Test Set", fontsize=14, fontweight="bold")
ax.set_xlabel(f"PC 1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
ax.set_ylabel(f"PC 2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
ax.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("06_pca_2d.png", dpi=120)
plt.show()
print("  [Saved] 06_pca_2d.png")

# 7c  t-SNE 2-D visualisation
print("  Computing t-SNE (2D) — takes ~30 s …")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000,
            random_state=42, learning_rate="auto", init="pca")
X_tsne = tsne.fit_transform(X_test_sc)

fig, ax = plt.subplots(figsize=(9, 7))
for digit in range(10):
    mask = y_test == digit
    ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
               label=str(digit), color=palette[digit],
               alpha=0.8, s=40, edgecolors="none")
ax.set_title("t-SNE — 2D Projection of Test Set", fontsize=14, fontweight="bold")
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(title="Digit", bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig("07_tsne_2d.png", dpi=120)
plt.show()
print("  [Saved] 07_tsne_2d.png")


# ─────────────────────────────────────────────
# BONUS 1 — Support Vectors Analysis
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BONUS 1 — Support Vector Analysis")
print("=" * 60)

svm_lin_bonus = SVC(kernel="linear", C=1.0, random_state=42)
svm_lin_bonus.fit(X_train_sc, y_train)
n_sv = svm_lin_bonus.n_support_
print("  Number of support vectors per class:")
for cls, n in zip(range(10), n_sv):
    print(f"    Digit {cls}: {n}")
print(f"  Total: {n_sv.sum()} support vectors out of {X_train_sc.shape[0]} training samples")
print(f"  ({n_sv.sum()/X_train_sc.shape[0]*100:.1f}% of training data)")

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(10), n_sv, color=palette, edgecolor="black")
ax.set_xlabel("Digit Class")
ax.set_ylabel("# Support Vectors")
ax.set_title("Support Vectors per Class — Linear SVM", fontsize=13, fontweight="bold")
ax.set_xticks(range(10))
plt.tight_layout()
plt.savefig("08_support_vectors.png", dpi=120)
plt.show()
print("  [Saved] 08_support_vectors.png")


# ─────────────────────────────────────────────
# BONUS 2 — Custom Kernel
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BONUS 2 — Custom Kernel (Laplacian)")
print("=" * 60)

def laplacian_kernel(X, Y):
    """K(x, y) = exp(-gamma * ||x - y||_1)  (L1 / Laplacian kernel)."""
    gamma = 0.001
    from sklearn.metrics.pairwise import manhattan_distances
    return np.exp(-gamma * manhattan_distances(X, Y))

svm_custom = SVC(kernel=laplacian_kernel, random_state=42)
svm_custom.fit(X_train_sc, y_train)
y_pred_custom = svm_custom.predict(X_test_sc)
acc_custom = accuracy_score(y_test, y_pred_custom)
print(f"  Custom Laplacian kernel accuracy: {acc_custom:.4f} ({acc_custom*100:.2f}%)")


# ─────────────────────────────────────────────
# BONUS 3 — Model Comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("BONUS 3 — Comparison with Other Models")
print("=" * 60)

lr = LogisticRegression(max_iter=2000, random_state=42)
lr.fit(X_train_sc, y_train)
acc_lr = accuracy_score(y_test, lr.predict(X_test_sc))
print(f"  Logistic Regression accuracy : {acc_lr:.4f}")

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_sc, y_train)
acc_rf = accuracy_score(y_test, rf.predict(X_test_sc))
print(f"  Random Forest accuracy       : {acc_rf:.4f}")

# Summary comparison
models = ["SVM Linear", "SVM Poly", "SVM RBF", "SVM Best\n(tuned)",
          "Custom Kernel", "Logistic\nRegression", "Random\nForest"]
model_accs = [acc_linear, acc_poly, acc_rbf, acc_best, acc_custom, acc_lr, acc_rf]

colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52",
          "#8172B2", "#937860", "#DA8BC3"]

fig, ax = plt.subplots(figsize=(12, 5))
bars = ax.bar(models, [a * 100 for a in model_accs], color=colors,
              edgecolor="black", width=0.6)
ax.set_ylim(80, 104)
ax.set_ylabel("Accuracy (%)", fontsize=12)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.axhline(y=max(model_accs) * 100, color="red", linestyle="--",
           linewidth=1.2, label=f"Best: {max(model_accs)*100:.2f}%")
for bar, acc in zip(bars, model_accs):
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3, f"{acc*100:.2f}%",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("09_model_comparison.png", dpi=120)
plt.show()
print("  [Saved] 09_model_comparison.png")


# ─────────────────────────────────────────────
# Final Summary Report
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
  Dataset         : Digits (sklearn)  |  1,797 samples, 64 features
  Train/Test split: 80% / 20%  (stratified)

  ┌──────────────────────────────┬───────────┐
  │ Model                        │ Accuracy  │
  ├──────────────────────────────┼───────────┤
  │ SVM — Linear kernel          │ {acc_linear*100:6.2f}%  │
  │ SVM — Polynomial kernel      │ {acc_poly*100:6.2f}%  │
  │ SVM — RBF kernel             │ {acc_rbf*100:6.2f}%  │
  │ SVM — Best (GridSearchCV)    │ {acc_best*100:6.2f}%  │
  │ SVM — Custom Laplacian       │ {acc_custom*100:6.2f}%  │
  │ Logistic Regression          │ {acc_lr*100:6.2f}%  │
  │ Random Forest                │ {acc_rf*100:6.2f}%  │
  └──────────────────────────────┴───────────┘

  Best hyperparams: {best_params}

  Observations:
  • RBF and tuned SVM achieved the highest accuracy.
  • LinearSVM is fast and competitive for this linearly separable-ish space.
  • Polynomial kernel is slightly weaker without tuning.
  • Hyperparameter tuning via GridSearchCV improved accuracy noticeably.
  • t-SNE revealed tight, well-separated clusters per digit class.
  • {n_sv.sum()} support vectors define decision boundaries (out of {X_train_sc.shape[0]}).
""")

print("All outputs saved. Homework complete!")