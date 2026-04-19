import numpy as np
import pandas as pd

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine = load_wine()

X = pd.DataFrame(wine.data, columns=wine.feature_names)
y = pd.Series(wine.target)

print("First 5 rows:")
print(X.head())

print("\nShape:", X.shape)

print("\nSummary statistics:")
print(X.describe())

print("\nClass distribution:")
print(y.value_counts())

print("\nDiscussion:")
print("- Dataset is relatively balanced (not perfectly).")
print("- Some features (e.g., proline, color_intensity) show high variance → possible outliers.\n")


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

baseline_model = LogisticRegression(max_iter=5000)
baseline_model.fit(X_train, y_train)

y_pred = baseline_model.predict(X_test)
baseline_acc = accuracy_score(y_test, y_pred)

print("Baseline Test Accuracy:", baseline_acc)


kf = KFold(n_splits=5, shuffle=True, random_state=42)

model = LogisticRegression(max_iter=5000)

cv_scores = cross_val_score(model, X, y, cv=kf)

print("\nK-Fold CV Scores:", cv_scores)
print("Mean CV Score:", np.mean(cv_scores))
print("Std Dev:", np.std(cv_scores))

print("\nInterpretation:")
print("- Mean CV is more reliable because it uses multiple splits.")
print("- Std deviation shows stability: lower = more stable model.\n")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

log_model = LogisticRegression(max_iter=5000)
rf_model = RandomForestClassifier(random_state=42)

log_scores = cross_val_score(log_model, X, y, cv=skf)
rf_scores = cross_val_score(rf_model, X, y, cv=skf)

print("Logistic Regression CV Mean:", np.mean(log_scores))
print("Logistic Regression Std:", np.std(log_scores))

print("\nRandom Forest CV Mean:", np.mean(rf_scores))
print("Random Forest Std:", np.std(rf_scores))

print("\nAnalysis:")
print("- Compare mean → accuracy")
print("- Compare std → stability")
print("- RF usually higher accuracy but may be more complex\n")


scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

log_results = cross_validate(log_model, X, y, cv=skf, scoring=scoring)
rf_results = cross_validate(rf_model, X, y, cv=skf, scoring=scoring)

print("=== Logistic Regression Metrics ===")
for metric in scoring:
    print(f"{metric}: Mean={np.mean(log_results['test_' + metric]):.4f}, Std={np.std(log_results['test_' + metric]):.4f}")

print("\n=== Random Forest Metrics ===")
for metric in scoring:
    print(f"{metric}: Mean={np.mean(rf_results['test_' + metric]):.4f}, Std={np.std(rf_results['test_' + metric]):.4f}")

print("\nDiscussion:")
print("- Check which model dominates across all metrics")
print("- Look for metric where difference is largest\n")


params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 4, 6]
}

grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=params,
    scoring='accuracy',
    cv=skf,
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Best CV Accuracy:", grid.best_score_)
print("Best Parameters:", grid.best_params_)
print("Best Estimator:", grid.best_estimator_)


best_model = grid.best_estimator_
best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)
best_test_acc = accuracy_score(y_test, y_pred_best)

print("\nFinal Test Accuracy (Tuned Model):", best_test_acc)
print("Baseline Accuracy:", baseline_acc)

print("\nConclusion:")
print("- If tuned accuracy > baseline → tuning helped")
print("- Otherwise → baseline already sufficient")
