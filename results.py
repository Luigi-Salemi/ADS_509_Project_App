"""
Finalized results for the ADS-509 final project
===============================================
Project: Traditional NLP vs. Transformers for Sentiment Classification
Course : ADS-509 Applied Text Mining, University of San Diego
Team   : Gagandeep Singh, Shivam Patel, Luigi Salemi

Source of truth: the team's finalized results presentation
("Traditional-NLP-vs-Transformers-for-Sentiment-Classification.pdf"),
cross-checked against the executed team notebook
(`ADS_509_Final_Team_Project_.ipynb`).

Note on reproducibility: the notebook scrapes the Google Play Store live, so
each run pulls slightly different reviews and produces slightly different
numbers. The values below are the team's reported final-run results.
"""

# ----------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------
PROJECT = {
    "title": "Traditional NLP vs. Transformers for Sentiment Classification",
    "course": "ADS-509 — Applied Text Mining",
    "school": "University of San Diego",
    "team": ["Gagandeep Singh", "Shivam Patel", "Luigi Salemi"],
    "objective": (
        "Build an end-to-end text-analytics pipeline that classifies the "
        "sentiment of app-store reviews, comparing a traditional "
        "TF-IDF + Logistic Regression baseline against a fine-tuned DistilBERT "
        "transformer."
    ),
}

# four "project overview" cards (mirrors the presentation)
OVERVIEW_CARDS = [
    ("Data Collection", "Scraped 10,000 reviews from the Google Play Store across 5 retail apps."),
    ("EDA & Preprocessing", "Explored text structure, word frequency, and sentiment distributions."),
    ("Two Modeling Approaches", "TF-IDF + Logistic Regression baseline vs. fine-tuned DistilBERT."),
    ("Generative AI Assist", "LLM tools aided code generation, model setup, and visualization."),
]

# ----------------------------------------------------------------------
# Data collection
# ----------------------------------------------------------------------
SOURCE_APPS = ["Amazon Shopping", "Walmart", "Target", "Best Buy", "eBay"]
SOURCE_PLATFORM = "Google Play Store"
REVIEWS_PER_APP = 2000
RAW_REVIEW_COUNT = 10000
SCRAPER_LIB = "google_play_scraper"
DATA_FIELDS = ["review text", "star rating (1-5)", "review date", "app name"]

# raw star-rating distribution across all 10,000 scraped reviews
RATING_DISTRIBUTION = {5: 5015, 1: 3043, 2: 697, 4: 624, 3: 621}

# ----------------------------------------------------------------------
# Label engineering + preprocessing
# ----------------------------------------------------------------------
# 4-5 stars -> positive, 1-2 stars -> negative, 3 stars dropped.
LABEL_BALANCE = {"Positive (1)": 3250, "Negative (0)": 3250}
BALANCED_TOTAL = 6500
TRAIN_SIZE = 5200
TEST_SIZE = 1300
UNIQUE_REVIEWS = 6095     # executed notebook (cell 27)
DUPLICATE_REVIEWS = 405   # executed notebook (cell 27)

# four-stage preprocessing pipeline (mirrors the presentation)
PIPELINE_STEPS = [
    ("Scrape Raw Reviews", "Pull reviews via google_play_scraper"),
    ("Clean Text", "Remove NLTK stopwords, handle null values"),
    ("Map Ratings", "Star ratings -> binary sentiment labels"),
    ("Balance & Split", "6,500 balanced samples, stratified 80/20 split"),
]

# average review length (words) by sentiment label — executed notebook (cell 13)
AVG_REVIEW_LENGTH = {"Negative (0)": 32.7, "Positive (1)": 10.9}

# ----------------------------------------------------------------------
# Word frequency analysis  (top 20, stopwords removed)
# ----------------------------------------------------------------------
TOP_WORDS = [
    ("app", 1774), ("get", 593), ("use", 510), ("easy", 481), ("love", 450),
    ("The", 426), ("good", 412), ("order", 403), ("search", 403), ("great", 402),
    ("time", 389), ("Amazon", 385), ("like", 355), ("can't", 330), ("service", 324),
    ("slow", 323), ("even", 323), ("I'm", 319), ("items", 318), ("app.", 301),
]

# ----------------------------------------------------------------------
# TF-IDF analysis  (TfidfVectorizer, english stopwords)
# ----------------------------------------------------------------------
TFIDF_TOP_FEATURES = [
    "ai", "amazon", "app", "best", "buy", "delivery", "don", "easy", "ebay",
    "good", "great", "items", "just", "love", "order", "search", "service",
    "slow", "time", "use",
]
# TF-IDF surfaced these as especially discriminative between classes
TFIDF_DISCRIMINATIVE = ["slow", "delivery", "search", "items"]

# ======================================================================
# MODEL A — TF-IDF + Logistic Regression (baseline)
# TfidfVectorizer max_features=5000, LogisticRegression max_iter=1000
# Stratified 80/20 split -> 5,200 train / 1,300 test
# ======================================================================
# executed notebook (cell 25): Accuracy 0.9092, Precision 0.9138,
# Recall 0.9092, F1 0.909
LR_METRICS = {
    "Accuracy": 0.9092,
    "Precision (weighted)": 0.9138,
    "Recall (weighted)": 0.9092,
    "F1 (weighted)": 0.909,
}
# per-class classification report (from the executed notebook run; rounds to
# the 0.91 figures reported on the baseline slide)
LR_REPORT = {
    "Negative (0)": {"precision": 0.87, "recall": 0.96, "f1": 0.91, "support": 650},
    "Positive (1)": {"precision": 0.96, "recall": 0.86, "f1": 0.90, "support": 650},
    "macro avg":    {"precision": 0.91, "recall": 0.91, "f1": 0.91, "support": 1300},
    "weighted avg": {"precision": 0.91, "recall": 0.91, "f1": 0.91, "support": 1300},
}

# ======================================================================
# MODEL B — Fine-tuned DistilBERT (transformer)
# distilbert-base-uncased, WordPiece max_len=128, 1 epoch,
# AdamW lr=2e-5, batch_size=8. Evaluated on 1,300 held-out reviews.
# ======================================================================
DISTILBERT_CONFIG = {
    "Base model": "distilbert-base-uncased",
    "Tokenizer": "WordPiece (max length 128)",
    "Epochs": "1",
    "Optimizer": "AdamW (lr = 2e-5)",
    "Batch size": "8",
    "Eval set": "1,300 held-out reviews",
}
# fine-tuning workflow (mirrors the presentation)
DISTILBERT_WORKFLOW = [
    ("Tokenize", "Reviews tokenized with WordPiece, max length 128 tokens"),
    ("Load Model", "Pre-trained DistilBERT with classification head (2 labels)"),
    ("Fine-Tune", "1 epoch, AdamW optimizer, learning rate 2e-5, batch size 8"),
    ("Evaluate", "Accuracy and F1 computed on the held-out 1,300 test reviews"),
]
# NOTE: from the team's final presentation. The saved notebook run did NOT log
# this score (its training cell was interrupted), so unlike every other figure
# in this file it is not reproducible from the saved notebook.
DISTILBERT_METRICS = {"Accuracy": 0.946, "F1 (weighted)": 0.946}
DISTILBERT_SOURCE_NOTE = (
    "From the team's final presentation. The saved notebook run did not log this "
    "score (the training cell was interrupted), so it is the one figure here not "
    "reproduced from the executed notebook."
)
DISTILBERT_REPORT = {
    "Negative": {"precision": 0.93, "recall": 0.96, "f1": 0.95, "support": 625},
    "Positive": {"precision": 0.96, "recall": 0.93, "f1": 0.95, "support": 675},
}
# confusion matrix: rows = actual, cols = predicted ; order = [Negative, Positive]
DISTILBERT_CONFUSION = [
    [602, 23],    # actual Negative -> 602 correct, 23 called Positive
    [46, 629],    # actual Positive -> 46 called Negative, 629 correct
]
DISTILBERT_CORRECT = 1231
DISTILBERT_MISCLASSIFIED = 69   # 23 FP + 46 FN

# ----------------------------------------------------------------------
# Model comparison
# ----------------------------------------------------------------------
MODEL_COMPARISON = [
    {"Model": "TF-IDF + Logistic Regression",
     "Accuracy": LR_METRICS["Accuracy"],
     "F1 (weighted)": LR_METRICS["F1 (weighted)"]},
    {"Model": "Fine-Tuned DistilBERT",
     "Accuracy": DISTILBERT_METRICS["Accuracy"],
     "F1 (weighted)": DISTILBERT_METRICS["F1 (weighted)"]},
]
# gap computed live in the app from the figures above; the presentation
# reported ~3.5 points (using its own slightly different baseline run).
PRESENTATION_GAIN = 3.5

# ----------------------------------------------------------------------
# Data provenance / cross-check  (executed notebook vs final presentation)
# ----------------------------------------------------------------------
PROVENANCE = (
    "Reproducible figures — the dataset stats, EDA, word frequency, TF-IDF, and "
    "the Logistic Regression baseline — are taken from the executed notebook. "
    "DistilBERT's score was not captured in the saved notebook run (its training "
    "cell was interrupted), so it comes from the team's final presentation. "
    "Because the notebook scrapes Google Play live, the presentation's later run "
    "differs slightly from the saved run; the table below shows both."
)
# rows: (metric, executed notebook, final presentation)
CROSSCHECK = [
    ("Raw reviews scraped", "10,000", "10,000"),
    ("Balanced sample", "6,500 (3,250 / 3,250)", "6,500"),
    ("Avg words — negative", "32.7", "32.5"),
    ("Avg words — positive", "10.9", "10.8"),
    ("Duplicate reviews", "405", "417"),
    ("LR accuracy", "90.9%", "91.2%"),
    ("LR weighted F1", "90.9%", "91.2%"),
    ("DistilBERT accuracy", "not captured", "94.6%"),
    ("DistilBERT weighted F1", "not captured", "94.6%"),
]

# ----------------------------------------------------------------------
# Live-demo example reviews (product reviews — match the bundled corpus)
# ----------------------------------------------------------------------
EXAMPLE_REVIEWS = {
    "Negative · blender": "This blender broke after just two weeks. The plastic feels cheap and the motor started smelling like it was burning.",
    "Positive · headphones": "These wireless headphones are amazing — crystal-clear sound, super comfortable, and the battery easily lasts all day.",
    "Mixed · coffee maker": "The coffee maker brews fine, but the carafe drips every time I pour and the lid feels flimsy.",
}
DEFAULT_EXAMPLE = "Negative · blender"

# ----------------------------------------------------------------------
# Conclusions  (mirrors the presentation)
# ----------------------------------------------------------------------
KEY_TAKEAWAYS = [
    "DistilBERT outperformed the TF-IDF + Logistic Regression baseline on both accuracy and F1.",
    "Contextual embeddings capture nuance that bag-of-words models miss.",
    "Generative AI tools accelerated code generation and pipeline setup.",
    "A balanced, real-world dataset ensured reliable, unbiased evaluation.",
]
LIMITATIONS = [
    "Single epoch — additional training may further improve performance.",
    "Binary only — neutral (3-star) reviews were excluded; multi-class remains open.",
    "Domain scope — limited to 5 retail apps; generalization untested.",
    "Next: multi-class sentiment, aspect-level analysis, larger models (BERT, RoBERTa).",
]
