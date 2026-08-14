"""Training and evaluation pipeline for the Spam Email Classifier.

Task: binary classification of SMS messages (ham=0 / spam=1).

Approaches compared:
  1. TF-IDF + Multinomial Naive Bayes   (classic NLP baseline)
  2. TF-IDF + Logistic Regression       (strong linear baseline)
  3. TF-IDF + Linear SVM                (SGDClassifier)

Evaluation: stratified 80/20 holdout (random_state=42) + 5-fold CV.
Because spam detection is imbalanced (13.4% spam), we report precision /
recall / F1 / AUC alongside accuracy, and emphasize recall on the spam class.
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from src.data import clean_text, load_data

RANDOM_STATE = 42

sns.set_theme(style="whitegrid")


def build_vectorizer():
    return TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
        stop_words="english",
    )


def run_experiments(df) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], df["label"], test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"]
    )

    models = {
        "Naive Bayes": MultinomialNB(alpha=0.5),
        "Logistic Regression": LogisticRegression(max_iter=2000, C=1.0, random_state=RANDOM_STATE),
        "Linear SVM (SGD)": SGDClassifier(loss="modified_huber", random_state=RANDOM_STATE, max_iter=2000),
    }

    results = {}
    for name, model in models.items():
        pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", model)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.decision_function(X_test) if hasattr(pipe.named_steps["clf"], "decision_function") \
            else pipe.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision_score(y_test, y_pred)), 4),
            "recall": round(float(recall_score(y_test, y_pred)), 4),
            "f1": round(float(f1_score(y_test, y_pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, y_prob)), 4),
        }
        print(f"  [{name}] acc={results[name]['accuracy']}  prec={results[name]['precision']}  "
              f"rec={results[name]['recall']}  f1={results[name]['f1']}  auc={results[name]['roc_auc']}")

        # CV
        cv = cross_val_score(pipe, df["clean"], df["label"], cv=5, scoring="f1")
        results[name]["cv_f1_mean"] = round(float(cv.mean()), 4)
        results[name]["cv_f1_std"] = round(float(cv.std()), 4)
        print(f"         5-fold CV F1: {cv.mean():.4f} +/- {cv.std():.4f}")

    # ---- artifacts for the BEST model (highest holdout F1) ----
    best_name = max(results, key=lambda k: results[k]["f1"])
    print(f"\n>>> Best model: {best_name}")
    best_model = models[best_name]
    pipe = Pipeline([("tfidf", build_vectorizer()), ("clf", best_model)])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    joblib.dump(pipe, "models/spam_classifier_best.joblib")

    # confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"])
    ax.set_title(f"{best_name} — Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    fig.savefig("images/confusion_matrix.png", dpi=150)
    plt.close(fig)

    # top spam/ham-indicative words
    vectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]
    vocab = np.array(vectorizer.get_feature_names_out())
    order = np.argsort(coefs)
    all_words, all_coefs = vocab[order], coefs[order]

    top_ham = all_words[:15][::-1]
    top_spam = all_words[-15:][::-1]
    (Path("reports") / "top_words.txt").write_text(
        "Top spam words: " + ", ".join(map(str, top_spam)) +
        "\nTop ham words: " + ", ".join(map(str, top_ham))
    )

    # feature weights plot
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].barh(top_ham, all_coefs[:15][::-1], color="#0f6b8c")
    axes[0].set_title("Top ham-indicative words")
    axes[1].barh(top_spam, all_coefs[-15:][::-1], color="#e07a5f")
    axes[1].set_title("Top spam-indicative words")
    fig.tight_layout()
    fig.savefig("images/top_words.png", dpi=150)
    plt.close(fig)

    report = classification_report(y_test, y_pred, target_names=["Ham", "Spam"])
    (Path("reports") / "classification_report.txt").write_text(report)
    print(report)

    return results


def main() -> None:
    df = load_data("data")
    df["clean"] = df["message"].map(clean_text)
    print(f"Dataset: {len(df)} messages | spam rate: {df['label'].mean():.1%}")

    # quick length analysis
    df["len_chars"] = df["message"].str.len()
    print(f"Avg length (chars): ham={df.loc[df.label==0,'len_chars'].mean():.0f}, "
          f"spam={df.loc[df.label==1,'len_chars'].mean():.0f}")

    results = run_experiments(df)

    summary = {
        "dataset": {"messages": int(len(df)), "spam_rate": float(df["label"].mean())},
        "models": results,
    }
    (Path("reports") / "metrics.json").write_text(json.dumps(summary, indent=2))
    print("\nSaved reports/metrics.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
