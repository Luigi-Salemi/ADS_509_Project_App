"""
Captured results from ADS_509_Final_Team_Project_.ipynb
========================================================
Project: Comparing Traditional NLP and Transformer-Based Models for
         Review Sentiment Classification.
Course : ADS-509 Applied Text Mining, University of San Diego
Team   : Gagandeep Singh, Shivam Patel, Luigi Salemi

Every number in this file is transcribed directly from the executed-cell
outputs of the team notebook. Nothing here is invented. The one result the
notebook did NOT capture (the fine-tuned DistilBERT score, because that
training cell was interrupted after step 0) is left as None and surfaced in
the app as "pending" rather than fabricated.
"""

# ----------------------------------------------------------------------
# Project metadata
# ----------------------------------------------------------------------
PROJECT = {
    "title": "Comparing Traditional NLP and Transformer-Based Models "
             "for Review Sentiment Classification",
    "course": "ADS-509 — Applied Text Mining",
    "school": "University of San Diego",
    "team": ["Gagandeep Singh", "Shivam Patel", "Luigi Salemi"],
    "objective": (
        "Benchmark a classic NLP pipeline (TF-IDF + Logistic Regression) "
        "against a fine-tuned transformer (DistilBERT) on the same binary "
        "sentiment task, using reviews the team scraped themselves."
    ),
}

# ----------------------------------------------------------------------
# Data collection  (cells: scrape + value_counts)
# ----------------------------------------------------------------------
SOURCE_APPS = ["Amazon Shopping", "Walmart", "Target", "Best Buy", "eBay"]
SOURCE_PLATFORM = "Google Play Store"
RAW_REVIEW_COUNT = 10000

# raw star-rating distribution across all 10,000 scraped reviews
RATING_DISTRIBUTION = {5: 5015, 1: 3043, 2: 697, 4: 624, 3: 621}

# ----------------------------------------------------------------------
# Label engineering  (4-5 -> positive, 1-2 -> negative, 3 dropped)
# ----------------------------------------------------------------------
# After mapping, the team drew a balanced sample of 3,250 per class.
LABEL_BALANCE = {"Positive (1)": 3250, "Negative (0)": 3250}
BALANCED_TOTAL = 6500
UNIQUE_REVIEWS = 6095
DUPLICATE_REVIEWS = 405

# average review length (words) by sentiment label
AVG_REVIEW_LENGTH = {"Negative (0)": 32.747077, "Positive (1)": 10.920000}

# ----------------------------------------------------------------------
# Word frequency analysis  (top 20, stopwords removed, len > 2)
# ----------------------------------------------------------------------
TOP_WORDS = [
    ("app", 1774), ("get", 593), ("use", 510), ("easy", 481), ("love", 450),
    ("The", 426), ("good", 412), ("order", 403), ("search", 403), ("great", 402),
    ("time", 389), ("Amazon", 385), ("like", 355), ("can't", 330), ("service", 324),
    ("slow", 323), ("even", 323), ("I'm", 319), ("items", 318), ("app.", 301),
]

# ----------------------------------------------------------------------
# TF-IDF analysis  (TfidfVectorizer max_features=20, english stopwords)
# ----------------------------------------------------------------------
TFIDF_TOP_FEATURES = [
    "ai", "amazon", "app", "best", "buy", "delivery", "don", "easy", "ebay",
    "good", "great", "items", "just", "love", "order", "search", "service",
    "slow", "time", "use",
]

# ----------------------------------------------------------------------
# Baseline model: TF-IDF + Logistic Regression
# (TfidfVectorizer max_features=5000, LogisticRegression max_iter=1000)
# Test set: 1,300 reviews, 650 per class, stratified 80/20 split.
# ----------------------------------------------------------------------
LR_METRICS = {
    "Accuracy": 0.9092,
    "Precision (weighted)": 0.9138,
    "Recall (weighted)": 0.9092,
    "F1 (weighted)": 0.9090,
}

# full classification_report, transcribed
LR_REPORT = {
    "Negative (0)": {"precision": 0.87, "recall": 0.96, "f1": 0.91, "support": 650},
    "Positive (1)": {"precision": 0.96, "recall": 0.86, "f1": 0.90, "support": 650},
    "macro avg":    {"precision": 0.91, "recall": 0.91, "f1": 0.91, "support": 1300},
    "weighted avg": {"precision": 0.91, "recall": 0.91, "f1": 0.91, "support": 1300},
}

# Confusion matrix DERIVED from the report (recall * support, rounded).
# Verified consistent with the reported precision values to 2 decimals.
#   rows = actual, cols = predicted ; order = [Negative, Positive]
LR_CONFUSION = [
    [624, 26],    # actual Negative -> 624 correct, 26 called Positive
    [91, 559],    # actual Positive -> 91 called Negative, 559 correct
]
LR_CONFUSION_NOTE = (
    "Derived from the classification report (recall x support, rounded); "
    "consistent with the reported precision to two decimals."
)

# ----------------------------------------------------------------------
# Transformer model: fine-tuned DistilBERT
# distilbert-base-uncased, 1 epoch, AdamW lr=2e-5, batch_size=8, max_len=128
# ----------------------------------------------------------------------
DISTILBERT_CONFIG = {
    "Base model": "distilbert-base-uncased",
    "Epochs": "1",
    "Optimizer": "AdamW (lr = 2e-5)",
    "Batch size": "8",
    "Max sequence length": "128",
    "Device": "CPU",
}
# The notebook's training cell was interrupted after the first step
# (only "Epoch 1, Step 0, Loss: 0.7509" was printed), so no eval score
# was recorded. Kept as None so the app shows "pending", never a guess.
DISTILBERT_METRICS = {"Accuracy": None, "F1 (weighted)": None}
DISTILBERT_FIRST_STEP_LOSS = 0.7509

# ----------------------------------------------------------------------
# Model comparison table
# ----------------------------------------------------------------------
MODEL_COMPARISON = [
    {"Model": "TF-IDF + Logistic Regression",
     "Accuracy": LR_METRICS["Accuracy"],
     "F1 (weighted)": LR_METRICS["F1 (weighted)"],
     "Status": "complete"},
    {"Model": "Fine-Tuned DistilBERT",
     "Accuracy": DISTILBERT_METRICS["Accuracy"],
     "F1 (weighted)": DISTILBERT_METRICS["F1 (weighted)"],
     "Status": "pending (training run incomplete in notebook)"},
]
