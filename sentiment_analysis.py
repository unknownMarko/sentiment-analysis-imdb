"""
Sentiment Analysis: VADER vs TF-IDF + Logistic Regression
==========================================================
Compares two approaches on the IMDb Large Movie Review Dataset:
  - Method A: VADER (lexicon-based, no training)
  - Method B: TF-IDF + Logistic Regression (supervised ML)
"""

import os
import re
import string
import time
import warnings

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# 1) Data Loading
# ---------------------------------------------------------------------------

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "aclImdb")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


def load_reviews(split: str) -> tuple[list[str], list[int]]:
    """Load reviews from aclImdb/{split}/pos and aclImdb/{split}/neg.

    Returns:
        texts: list of raw review strings
        labels: list of ints (1=pos, 0=neg)
    """
    texts, labels = [], []
    for label_dir, label_val in [("pos", 1), ("neg", 0)]:
        folder = os.path.join(DATA_DIR, split, label_dir)
        for fname in sorted(os.listdir(folder)):
            fpath = os.path.join(folder, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                texts.append(f.read())
            labels.append(label_val)
    return texts, labels


# ---------------------------------------------------------------------------
# 2) Preprocessing
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def preprocess(text: str) -> str:
    """Basic text preprocessing:
    - lowercase
    - remove HTML tags
    - remove punctuation
    - normalize whitespace
    """
    text = text.lower()
    text = _HTML_TAG_RE.sub(" ", text)
    text = text.translate(_PUNCT_TABLE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# 3) Method A: VADER
# ---------------------------------------------------------------------------


def vader_predict(texts: list[str]) -> tuple[list[int], list[float]]:
    """Classify reviews using VADER compound score.

    compound >= 0  → positive (1)
    compound < 0   → negative (0)

    Returns:
        predictions: list of 0/1
        compound_scores: list of floats
    """
    sia = SentimentIntensityAnalyzer()
    predictions = []
    compound_scores = []
    for text in texts:
        scores = sia.polarity_scores(text)
        compound = scores["compound"]
        compound_scores.append(compound)
        predictions.append(1 if compound >= 0 else 0)
    return predictions, compound_scores


# ---------------------------------------------------------------------------
# 4) Method B: TF-IDF + Logistic Regression
# ---------------------------------------------------------------------------


def tfidf_lr_train_predict(
    train_texts: list[str],
    train_labels: list[int],
    test_texts: list[str],
) -> list[int]:
    """Train TF-IDF + Logistic Regression on train data, predict on test.

    Returns:
        predictions: list of 0/1
    """
    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=2,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test).tolist()
    return predictions


# ---------------------------------------------------------------------------
# 5) Evaluation
# ---------------------------------------------------------------------------


def evaluate(y_true: list[int], y_pred: list[int], method_name: str) -> dict:
    """Compute and print classification metrics.

    Returns dict with accuracy, precision, recall, f1, confusion_matrix.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\n{'=' * 60}")
    print(f"  {method_name} — Evaluation Results")
    print(f"{'=' * 60}")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                 Predicted Neg  Predicted Pos")
    print(f"  Actual Neg     {cm[0][0]:>12}  {cm[0][1]:>12}")
    print(f"  Actual Pos     {cm[1][0]:>12}  {cm[1][1]:>12}")
    print()
    print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# 6) Visualization
# ---------------------------------------------------------------------------


def plot_metrics_comparison(vader_metrics: dict, tfidf_metrics: dict, output_dir: str):
    """Bar chart comparing Accuracy and F1 for VADER vs TF-IDF."""
    metrics_names = ["Accuracy", "Precision", "Recall", "F1"]
    vader_vals = [
        vader_metrics["accuracy"],
        vader_metrics["precision"],
        vader_metrics["recall"],
        vader_metrics["f1"],
    ]
    tfidf_vals = [
        tfidf_metrics["accuracy"],
        tfidf_metrics["precision"],
        tfidf_metrics["recall"],
        tfidf_metrics["f1"],
    ]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(
        x - width / 2, vader_vals, width, label="VADER", color="#e74c3c", alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2,
        tfidf_vals,
        width,
        label="TF-IDF + LR",
        color="#3498db",
        alpha=0.85,
    )

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(
        "VADER vs TF-IDF + Logistic Regression — Metrics Comparison",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars1:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    for bar in bars2:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    path = os.path.join(output_dir, "metrics_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_confusion_matrix(cm: np.ndarray, method_name: str, output_dir: str):
    """Plot and save a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Negative", "Positive"]
    )
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — {method_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    safe_name = method_name.lower().replace(" ", "_").replace("+", "")
    path = os.path.join(output_dir, f"confusion_matrix_{safe_name}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_vader_histogram(
    compound_scores: list[float], test_labels: list[int], output_dir: str
):
    """Histogram of VADER compound scores split by true label."""
    scores_pos = [s for s, l in zip(compound_scores, test_labels) if l == 1]
    scores_neg = [s for s, l in zip(compound_scores, test_labels) if l == 0]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        scores_neg,
        bins=50,
        alpha=0.6,
        label="True Negative",
        color="#e74c3c",
        edgecolor="white",
    )
    ax.hist(
        scores_pos,
        bins=50,
        alpha=0.6,
        label="True Positive",
        color="#3498db",
        edgecolor="white",
    )
    ax.axvline(x=0, color="black", linestyle="--", linewidth=1.5, label="Threshold (0)")
    ax.set_xlabel("VADER Compound Score", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(
        "Distribution of VADER Compound Scores by True Label",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "vader_compound_histogram.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# 7) Error Analysis
# ---------------------------------------------------------------------------


def find_misclassified_examples(
    texts: list[str],
    labels: list[int],
    predictions: list[int],
    method_name: str,
    n: int = 2,
):
    """Find and print n misclassified positive and n misclassified negative reviews."""
    false_negatives = []  # true=pos, pred=neg
    false_positives = []  # true=neg, pred=pos

    for i, (text, true_label, pred_label) in enumerate(zip(texts, labels, predictions)):
        if true_label == 1 and pred_label == 0 and len(false_negatives) < n:
            false_negatives.append((i, text))
        elif true_label == 0 and pred_label == 1 and len(false_positives) < n:
            false_positives.append((i, text))
        if len(false_negatives) >= n and len(false_positives) >= n:
            break

    print(f"\n{'=' * 60}")
    print(f"  {method_name} — Misclassified Examples")
    print(f"{'=' * 60}")

    print(f"\n  FALSE NEGATIVES (truly positive, predicted negative):")
    for idx, (i, text) in enumerate(false_negatives, 1):
        snippet = text[:300].replace("\n", " ")
        print(f"\n  Example {idx} (review #{i}):")
        print(f'  "{snippet}..."')

    print(f"\n  FALSE POSITIVES (truly negative, predicted positive):")
    for idx, (i, text) in enumerate(false_positives, 1):
        snippet = text[:300].replace("\n", " ")
        print(f"\n  Example {idx} (review #{i}):")
        print(f'  "{snippet}..."')


# ---------------------------------------------------------------------------
# 8) Analysis Report
# ---------------------------------------------------------------------------


def print_analysis(vader_metrics: dict, tfidf_metrics: dict):
    """Print a short analysis comparing the two methods."""
    print(f"\n{'#' * 60}")
    print(f"  ANALYSIS")
    print(f"{'#' * 60}")

    print("""
  1) WHICH METHOD PERFORMED BETTER AND WHY?
  ------------------------------------------
  TF-IDF + Logistic Regression significantly outperforms VADER.
  
  - TF-IDF + LR is a supervised method trained on the same domain (IMDb reviews),
    so it learns domain-specific vocabulary and patterns (e.g., "boring", "waste",
    "masterpiece") directly from the data distribution.
  - VADER is a general-purpose lexicon designed for social media text. It relies
    on a fixed dictionary of sentiment words with intensity scores. Movie reviews
    often use nuanced language, sarcasm, and domain-specific expressions that
    VADER's lexicon doesn't capture well.
  - TF-IDF captures bigrams (e.g., "not good", "really bad") which helps with
    negation handling, while VADER has limited negation processing.

  2) TYPICAL ERRORS
  ------------------
  VADER typical errors:
  - Misclassifies reviews with mixed sentiment (e.g., "The acting was great but
    the plot was terrible") — compound score averages out to near zero.
  - Fails on sarcasm/irony: "What a brilliant way to waste 2 hours" — detects
    "brilliant" as positive.
  - Struggles with domain-specific vocabulary: film jargon like "formulaic",
    "predictable", "derivative" aren't in VADER's lexicon.
  - Overly sensitive to punctuation and capitalization artifacts.

  TF-IDF + LR typical errors:
  - Struggles with very short or ambiguous reviews lacking strong sentiment words.
  - Can misclassify reviews that discuss negative plot elements positively
    (e.g., a horror film praised for being "terrifying" and "disturbing").
  - May fail on reviews with heavy negation chains or unusual sentence structures.
  - Slightly biased by training data distribution — rare expressions may be
    mishandled.

  3) HOW COULD VADER BE OPTIMIZED FOR THIS TASK?
  ------------------------------------------------
  a) Custom lexicon: Add domain-specific words (e.g., "masterpiece"=+3,
     "formulaic"=-2, "predictable"=-1.5) to the VADER lexicon via
     SentimentIntensityAnalyzer().lexicon.update({...}).
  b) Threshold tuning: Instead of compound >= 0, find the optimal threshold
     on a validation split (e.g., compound >= 0.05 might reduce false positives).
  c) Preprocessing adjustments: Keep some punctuation/capitalization that VADER
     uses as intensity modifiers (! and ALL CAPS boost scores in VADER).
  d) Ensemble approach: Combine VADER compound scores as a feature alongside
     TF-IDF features in the Logistic Regression model.
  e) Text segmentation: Analyze sentences individually and aggregate, rather
     than scoring the entire review at once — this avoids the averaging effect
     on long mixed-sentiment reviews.
""")

    # Print metric comparison table
    print(f"  {'Metric':<15} {'VADER':>10} {'TF-IDF+LR':>12} {'Difference':>12}")
    print(f"  {'-' * 49}")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        v = vader_metrics[metric]
        t = tfidf_metrics[metric]
        diff = t - v
        print(f"  {metric.capitalize():<15} {v:>10.4f} {t:>12.4f} {diff:>+12.4f}")
    print()


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Download VADER lexicon if needed ---
    nltk.download("vader_lexicon", quiet=True)

    # --- Load data ---
    print("Loading training data...")
    t0 = time.time()
    train_texts_raw, train_labels = load_reviews("train")
    print(
        f"  Loaded {len(train_texts_raw)} training reviews in {time.time() - t0:.1f}s"
    )

    print("Loading test data...")
    t0 = time.time()
    test_texts_raw, test_labels = load_reviews("test")
    print(f"  Loaded {len(test_texts_raw)} test reviews in {time.time() - t0:.1f}s")

    # --- Preprocess ---
    print("\nPreprocessing...")
    t0 = time.time()
    train_texts = [preprocess(t) for t in train_texts_raw]
    test_texts = [preprocess(t) for t in test_texts_raw]
    print(f"  Done in {time.time() - t0:.1f}s")
    print(f"  Sample (raw):  {test_texts_raw[0][:100]}...")
    print(f"  Sample (clean): {test_texts[0][:100]}...")

    # --- VADER ---
    print("\n" + "=" * 60)
    print("  Running VADER Sentiment Analysis...")
    print("=" * 60)
    t0 = time.time()
    vader_preds, compound_scores = vader_predict(test_texts)
    print(f"  VADER completed in {time.time() - t0:.1f}s")

    # --- TF-IDF + LR ---
    print("\n" + "=" * 60)
    print("  Training TF-IDF + Logistic Regression...")
    print("=" * 60)
    t0 = time.time()
    tfidf_preds = tfidf_lr_train_predict(train_texts, train_labels, test_texts)
    print(f"  TF-IDF + LR completed in {time.time() - t0:.1f}s")

    # --- Evaluation ---
    vader_metrics = evaluate(test_labels, vader_preds, "VADER")
    tfidf_metrics = evaluate(test_labels, tfidf_preds, "TF-IDF + Logistic Regression")

    # --- Visualizations ---
    print("\nGenerating visualizations...")
    plot_metrics_comparison(vader_metrics, tfidf_metrics, OUTPUT_DIR)
    plot_confusion_matrix(vader_metrics["confusion_matrix"], "VADER", OUTPUT_DIR)
    plot_confusion_matrix(tfidf_metrics["confusion_matrix"], "TF-IDF + LR", OUTPUT_DIR)
    plot_vader_histogram(compound_scores, test_labels, OUTPUT_DIR)

    # --- Error examples ---
    find_misclassified_examples(test_texts_raw, test_labels, vader_preds, "VADER")
    find_misclassified_examples(test_texts_raw, test_labels, tfidf_preds, "TF-IDF + LR")

    # --- Analysis ---
    print_analysis(vader_metrics, tfidf_metrics)

    print("=" * 60)
    print("  ALL DONE. Outputs saved to:", OUTPUT_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
