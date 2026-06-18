"""
train_model.py — ResumeAI Pro  |  One-time Model Trainer
=========================================================
Run this ONCE before launching the app to train and save the ML model.

    python train_model.py

Outputs:
  • resume_model.pkl          — trained model + vectorisers
  • category_distribution.png — bar chart of dataset categories

Pipeline:
  1. Load & clean Resume.csv
  2. TF-IDF feature extraction  (word n-grams + char n-grams)
  3. Train 4 individual classifiers, evaluate on hold-out set
  4. Build calibrated soft-voting ensemble
  5. Pick best model, cross-validate, save
"""

import pandas as pd
import numpy as np
import re
import pickle
import warnings
warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.ensemble import VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, classification_report
from scipy.sparse import hstack

import nltk
for pkg in ["stopwords", "punkt", "wordnet"]:
    nltk.download(pkg, quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# ══════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════

STOP_WORDS = set(stopwords.words("english"))
KEEP_WORDS = {
    "not", "no", "very", "highly", "strong", "senior", "junior",
    "lead", "head", "chief", "principal", "associate", "assistant",
}
STOP_WORDS -= KEEP_WORDS
lemmatizer = WordNetLemmatizer()


def clean_text(text: str, lemmatize: bool = True) -> str:
    """Multi-stage NLP cleaning pipeline optimised for resumes."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[\+\(\)]?[0-9][0-9 .\-]{7,}[0-9]", " ", text)
    text = re.sub(r"c\+\+",    "cplusplus", text, flags=re.IGNORECASE)
    text = re.sub(r"c#",       "csharp",    text, flags=re.IGNORECASE)
    text = re.sub(r"\.net",    "dotnet",    text, flags=re.IGNORECASE)
    text = re.sub(r"node\.js", "nodejs",    text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    tokens = text.split()
    if lemmatize:
        tokens = [lemmatizer.lemmatize(w) for w in tokens
                  if w not in STOP_WORDS and len(w) > 1]
    else:
        tokens = [w for w in tokens if w not in STOP_WORDS and len(w) > 1]
    return " ".join(tokens)


# ══════════════════════════════════════════════════════════════════════
# LOAD DATASET
# ══════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  ResumeAI Pro — Model Trainer")
print("=" * 60)
print("\n[1/6] Loading Resume.csv ...")

df = pd.read_csv(
    "Resume.csv",
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip",
    escapechar="\\",
)
print(f"      Raw shape  : {df.shape}")
print(f"      Columns    : {list(df.columns)}")

df = df[["Resume_str", "Category"]].rename(columns={"Resume_str": "Resume"})
df = df.dropna(subset=["Resume", "Category"])
df["Resume"] = df["Resume"].astype(str)
print(f"      Clean rows : {len(df):,}  |  Categories: {df['Category'].nunique()}")


# ══════════════════════════════════════════════════════════════════════
# CATEGORY DISTRIBUTION CHART
# ══════════════════════════════════════════════════════════════════════

print("\n[2/6] Generating category distribution chart ...")

fig, ax = plt.subplots(figsize=(14, 5))
counts = df["Category"].value_counts()
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(counts)))
bars   = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)
ax.set_title("Resume Category Distribution", fontsize=15, fontweight="bold", pad=14)
ax.set_xlabel("Category", fontsize=11)
ax.set_ylabel("Count", fontsize=11)
ax.tick_params(axis="x", rotation=45)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
            str(int(bar.get_height())), ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("category_distribution.png", dpi=100, bbox_inches="tight")
plt.close()
print("      Saved: category_distribution.png")


# ══════════════════════════════════════════════════════════════════════
# TEXT CLEANING
# ══════════════════════════════════════════════════════════════════════

print("\n[3/6] Cleaning resume text (~30 s) ...")
df["cleaned"]    = df["Resume"].apply(clean_text)
df["word_count"] = df["cleaned"].str.split().str.len()
print(f"      Avg words after cleaning: {df['word_count'].mean():.0f}")


# ══════════════════════════════════════════════════════════════════════
# TF-IDF FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════

print("\n[4/6] Building TF-IDF features ...")
X = df["cleaned"]
y = df["Category"]

tfidf_word = TfidfVectorizer(
    analyzer="word", stop_words="english", ngram_range=(1, 3),
    max_features=30_000, sublinear_tf=True, min_df=2, max_df=0.95,
)
tfidf_char = TfidfVectorizer(
    analyzer="char_wb", ngram_range=(3, 5),
    max_features=20_000, sublinear_tf=True, min_df=3, max_df=0.95,
)

X_word     = tfidf_word.fit_transform(X)
X_char     = tfidf_char.fit_transform(X)
X_combined = hstack([X_word, X_char])
print(f"      Feature matrix: {X_combined.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.2, random_state=42, stratify=y,
)
print(f"      Train: {X_train.shape[0]:,}  |  Test: {X_test.shape[0]:,}")


# ══════════════════════════════════════════════════════════════════════
# TRAIN CLASSIFIERS
# ══════════════════════════════════════════════════════════════════════

print("\n[5/6] Training classifiers ...")

classifiers = {
    "LinearSVC":    LinearSVC(C=2.0, max_iter=3000, loss="hinge", dual=True),
    "LogReg":       LogisticRegression(C=5.0, max_iter=1000, solver="saga", n_jobs=-1),
    "SGD":          SGDClassifier(loss="modified_huber", alpha=1e-4, max_iter=200, random_state=42, n_jobs=-1),
    "ComplementNB": ComplementNB(alpha=0.3),
}

results = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results[name] = acc
    print(f"      {name:<20}  {acc * 100:.2f}%")

best_single = max(results, key=results.get)
print(f"\n      Best single: {best_single} ({results[best_single] * 100:.2f}%)")

# Calibrated soft-voting ensemble
print("      Building calibrated ensemble ...")
svc_cal = CalibratedClassifierCV(LinearSVC(C=2.0, max_iter=3000, loss="hinge"), cv=3)
lr_best = LogisticRegression(C=5.0, max_iter=1000, solver="saga", n_jobs=-1)
sgd_cal = CalibratedClassifierCV(SGDClassifier(loss="hinge", alpha=1e-4, max_iter=200, random_state=42), cv=3)

ensemble = VotingClassifier(
    estimators=[("svc", svc_cal), ("lr", lr_best), ("sgd", sgd_cal)],
    voting="soft", n_jobs=-1,
)
ensemble.fit(X_train, y_train)
ens_acc = accuracy_score(y_test, ensemble.predict(X_test))
print(f"      Ensemble accuracy: {ens_acc * 100:.2f}%")

if ens_acc >= results[best_single]:
    best_model = ensemble
    model_name = "Calibrated Ensemble (SVC + LR + SGD)"
    final_acc  = ens_acc
else:
    best_model = classifiers[best_single]
    model_name = best_single
    final_acc  = results[best_single]

print(f"\n  FINAL MODEL : {model_name}")
print(f"  ACCURACY    : {final_acc * 100:.2f}%")

# 5-fold cross-validation
print("\n      5-fold cross-validation ...")
lr_cv = LogisticRegression(C=5.0, max_iter=1000, solver="saga", n_jobs=-1)
cv_scores = cross_val_score(
    lr_cv, X_combined, y,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    scoring="accuracy", n_jobs=-1,
)
print(f"      CV: {cv_scores.mean() * 100:.2f}% +/- {cv_scores.std() * 100:.2f}%")

print("\n      Classification Report (test set):")
print(classification_report(y_test, best_model.predict(X_test), zero_division=0))


# ══════════════════════════════════════════════════════════════════════
# SAVE MODEL
# ══════════════════════════════════════════════════════════════════════

print("[6/6] Saving model ...")

artifacts = {
    "model":      best_model,
    "tfidf_word": tfidf_word,
    "tfidf_char": tfidf_char,
    "accuracy":   final_acc,
    "model_name": model_name,
}
with open("resume_model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print(f"\n  Saved: resume_model.pkl")
print(f"  Model    : {model_name}")
print(f"  Accuracy : {final_acc * 100:.2f}%")
print(f"\n  Next step: streamlit run app.py")
print("=" * 60 + "\n")