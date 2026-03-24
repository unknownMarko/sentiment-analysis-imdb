# Sentiment Analysis

Comparison of two sentiment analysis approaches on [IMDb movie reviews](https://ai.stanford.edu/~amaas/data/sentiment/).

## How to Run

```bash
wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar xzf aclImdb_v1.tar.gz
pip install -r requirements.txt
python3 sentiment_analysis.py
```

Requires Python 3.10+ — runs in ~2.5 minutes on 50,000 reviews.

## Dataset

IMDb Large Movie Review Dataset — 25,000 train + 25,000 test reviews, balanced across positive and negative.

```
aclImdb/
├── train/
│   ├── pos/  (12,500)
│   └── neg/  (12,500)
└── test/
    ├── pos/  (12,500)
    └── neg/  (12,500)
```

## Methods

| Method | Type | Training |
|--------|------|----------|
| VADER | Lexicon-based (NLTK) | None — pre-built dictionary |
| TF-IDF + Logistic Regression | Supervised ML (sklearn) | Trained on train set |

**VADER** scores each review with a compound value and classifies by threshold (`>= 0` → positive).

**TF-IDF + LR** converts text to TF-IDF vectors (unigrams + bigrams, 50k features) and trains a Logistic Regression classifier.

## Preprocessing

Applied to all reviews before classification:

- Lowercase
- HTML tag removal
- Punctuation removal
- Whitespace normalization

## Results

| Metric | VADER | TF-IDF + LR |
|--------|-------|-------------|
| Accuracy | 0.698 | **0.900** |
| Precision | 0.651 | **0.896** |
| Recall | 0.857 | **0.906** |
| F1 | 0.740 | **0.901** |

## Output

The script generates four plots in `output/`:

### Metrics Comparison

![Metrics Comparison](output/metrics_comparison.png)

### Confusion Matrix — VADER

![VADER Confusion Matrix](output/confusion_matrix_vader.png)

### Confusion Matrix — TF-IDF + LR

![TF-IDF Confusion Matrix](output/confusion_matrix_tf-idf__lr.png)

### VADER Compound Score Distribution

![VADER Histogram](output/vader_compound_histogram.png)

It also prints misclassified examples and a written analysis comparing both methods.
