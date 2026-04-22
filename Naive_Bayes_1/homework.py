# ============================================================
#  Bernoulli Naive Bayes — SMS Spam Classification
# ============================================================

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ----------------------------------------------------------
# Step 1: Load the dataset
# ----------------------------------------------------------
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

# Try loading from URL; fall back to requests (handles redirects / headers)
try:
    df = pd.read_csv(url, sep="\t", header=None, names=["label", "message"])
except Exception:
    import requests, io
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers, timeout=15)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), sep="\t", header=None, names=["label", "message"])

print("=" * 55)
print("  Dataset Overview")
print("=" * 55)
print(f"  Total samples : {len(df)}")
print(f"  Columns       : {list(df.columns)}")
print(f"\n  Label distribution:\n{df['label'].value_counts().to_string()}")
print()

# ----------------------------------------------------------
# Step 2: Encode labels  (spam → 1, ham → 0)
# ----------------------------------------------------------
df["label_bin"] = df["label"].map({"spam": 1, "ham": 0})

print("  Label encoding applied  →  spam = 1 | ham = 0")
print()

# ----------------------------------------------------------
# Step 3: Binary text features with CountVectorizer
# ----------------------------------------------------------
vectorizer = CountVectorizer(binary=True)          # presence / absence only
X = vectorizer.fit_transform(df["message"])
y = df["label_bin"]

print(f"  Vocabulary size (features) : {X.shape[1]:,}")
print(f"  Feature matrix shape       : {X.shape}")
print()

# ----------------------------------------------------------
# Step 4: Train / Test split  (70 % / 30 %)
# ----------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42, stratify=y
)

print("  Train / Test Split (70 / 30)")
print(f"    Training samples : {X_train.shape[0]}")
print(f"    Test samples     : {X_test.shape[0]}")
print()

# ----------------------------------------------------------
# Step 5: Train BernoulliNB
# ----------------------------------------------------------
model = BernoulliNB()
model.fit(X_train, y_train)

print("  BernoulliNB model trained successfully.")
print()

# ----------------------------------------------------------
# Step 6: Evaluate on the test set
# ----------------------------------------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
cm       = confusion_matrix(y_test, y_pred)
report   = classification_report(y_test, y_pred, target_names=["ham", "spam"])

print("=" * 55)
print("  Evaluation Results")
print("=" * 55)
print(f"  Accuracy : {accuracy * 100:.2f} %")
print()

print("  Confusion Matrix:")
print("              Predicted Ham   Predicted Spam")
print(f"  Actual Ham       {cm[0][0]:<10}     {cm[0][1]}")
print(f"  Actual Spam      {cm[1][0]:<10}     {cm[1][1]}")
print()

print("  Classification Report:")
print(report)

# ----------------------------------------------------------
# Bonus: Top spam-indicative words
# ----------------------------------------------------------
feature_names = np.array(vectorizer.get_feature_names_out())
spam_log_prob  = model.feature_log_prob_[1]          # class index 1 = spam
top_n          = 15
top_spam_idx   = np.argsort(spam_log_prob)[-top_n:][::-1]

print("=" * 55)
print(f"  Top {top_n} Spam-Indicative Words")
print("=" * 55)
for rank, idx in enumerate(top_spam_idx, 1):
    print(f"  {rank:>2}. {feature_names[idx]:<20}  log-prob: {spam_log_prob[idx]:.4f}")
print()
