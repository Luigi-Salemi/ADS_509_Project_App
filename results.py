"""
Results for the ADS-509 Option 1 dashboard — matches the team's final
presentation ("Traditional NLP vs. Transformers for Sentiment Classification").

Data: 10,000 reviews scraped from the Google Play Store across 5 retail apps,
balanced to 6,500 and split 80/20. Figures below are the presentation's reported
numbers (TF-IDF + Logistic Regression vs. fine-tuned DistilBERT).
"""

PROJECT = {
    "title": "Pre-trained LLM Text Analysis — App Review Sentiment",
    "course": "Applied Large Language Models for Data Science (ADS-509-01)",
    "school": "University of San Diego",
    "team": ["Gagandeep Singh", "Shivam Patel", "Luigi Salemi"],
    "objective": (
        "A small-scale text-analytics pipeline on Google Play app-store reviews: "
        "EDA of text structure, frequency, and sentiment distribution, then a "
        "fine-tuned open-source LLM (DistilBERT) for sentiment classification — "
        "benchmarked against a TF-IDF + Logistic Regression baseline, with "
        "generative-AI-assisted development."
    ),
}

# ----------------------------------------------------------------------
# Dataset — Google Play Store, 5 retail apps
# ----------------------------------------------------------------------
DATASET_NAME = "Google Play Store reviews (5 retail apps)"
DATASET_URL = "https://play.google.com/store"
RAW_REVIEW_COUNT = 10000
CATEGORIES = ["Amazon Shopping", "Walmart", "Target", "Best Buy", "eBay"]   # the 5 apps
CATEGORY_DIST = {"Amazon Shopping": 2000, "Walmart": 2000, "Target": 2000,
                 "Best Buy": 2000, "eBay": 2000}
RATING_DISTRIBUTION = {1: 3043, 2: 697, 3: 621, 4: 624, 5: 5015}

# ----------------------------------------------------------------------
# Label engineering + preprocessing
# ----------------------------------------------------------------------
LABEL_BALANCE = {"Positive (1)": 3250, "Negative (0)": 3250}
BALANCED_TOTAL = 6500
TRAIN_SIZE = 5200
TEST_SIZE = 1300
UNIQUE_REVIEWS = 6083
DUPLICATE_REVIEWS = 417

AVG_REVIEW_LENGTH = {"Negative (0)": 32.5, "Positive (1)": 10.8}

# ----------------------------------------------------------------------
# Word frequency + TF-IDF
# ----------------------------------------------------------------------
TOP_WORDS = [
    ["app", 1774], ["get", 593], ["use", 510], ["easy", 481], ["love", 450],
    ["good", 412], ["order", 403], ["search", 403], ["great", 402], ["time", 389],
    ["amazon", 385], ["like", 355], ["service", 324], ["slow", 323], ["even", 323],
    ["items", 318], ["delivery", 300], ["update", 295], ["work", 288], ["fast", 281],
]
TFIDF_TOP_FEATURES = ["ai", "amazon", "app", "best", "buy", "delivery", "don", "easy",
                      "ebay", "good", "great", "items", "just", "love", "order",
                      "search", "service", "slow", "time", "use"]
TFIDF_DISCRIMINATIVE = ["slow", "delivery", "search", "items"]

# ======================================================================
# Model A — TF-IDF + Logistic Regression (presentation numbers)
# ======================================================================
LR_METRICS = {
    "Accuracy": 0.912,
    "Precision (weighted)": 0.916,
    "Recall (weighted)": 0.912,
    "F1 (weighted)": 0.912,
}
LR_REPORT = {
    "Negative (0)": {"precision": 0.87, "recall": 0.96, "f1": 0.91, "support": 650},
    "Positive (1)": {"precision": 0.96, "recall": 0.86, "f1": 0.90, "support": 650},
}

# ======================================================================
# Model B — Fine-tuned DistilBERT (presentation numbers)
# ======================================================================
DISTILBERT_CONFIG = {
    "Base model": "distilbert-base-uncased",
    "Tokenizer": "WordPiece (max length 128)",
    "Epochs": "1", "Optimizer": "AdamW (lr = 2e-5)", "Batch size": "8",
    "Eval set": "1,300 held-out reviews",
}
DISTILBERT_WORKFLOW = [
    ("Tokenize", "WordPiece, max length 128 tokens"),
    ("Load Model", "Pre-trained DistilBERT + classification head (2 labels)"),
    ("Fine-Tune", "1 epoch, AdamW, lr 2e-5, batch size 8"),
    ("Evaluate", "Accuracy and F1 on the 1,300 held-out reviews"),
]
DISTILBERT_METRICS = {"Accuracy": 0.946, "F1 (weighted)": 0.946}
DISTILBERT_REPORT = {
    "Negative": {"precision": 0.93, "recall": 0.96, "f1": 0.95, "support": 625},
    "Positive": {"precision": 0.96, "recall": 0.93, "f1": 0.95, "support": 675},
}
DISTILBERT_CONFUSION = [[602, 23], [46, 629]]
DISTILBERT_CORRECT = 1231
DISTILBERT_MISCLASSIFIED = 69

MODEL_COMPARISON = [
    {"Model": "TF-IDF + Logistic Regression", "Accuracy": LR_METRICS["Accuracy"], "F1 (weighted)": LR_METRICS["F1 (weighted)"]},
    {"Model": "Fine-Tuned DistilBERT", "Accuracy": DISTILBERT_METRICS["Accuracy"], "F1 (weighted)": DISTILBERT_METRICS["F1 (weighted)"]},
]

PROVENANCE = (
    "Figures match the team's final presentation: 10,000 reviews scraped from the "
    "Google Play Store across 5 retail apps (Amazon, Walmart, Target, Best Buy, "
    "eBay), balanced to 6,500 and split 80/20. The live demo uses a representative "
    "fine-tuned DistilBERT, since the original scraped sample was not exported."
)
KEY_TAKEAWAYS = [
    "DistilBERT outperformed TF-IDF + Logistic Regression by ~3.5 points on both accuracy and F1.",
    "Contextual embeddings capture nuance that bag-of-words models miss.",
    "Generative AI tools accelerated code generation and pipeline setup.",
    "A balanced, real-world dataset ensured reliable, unbiased evaluation.",
]
LIMITATIONS = [
    "Single epoch — additional training may further improve performance.",
    "Binary only — neutral (3-star) reviews excluded; multi-class remains open.",
    "Domain scope — limited to 5 retail apps; generalization untested.",
    "Next: multi-class sentiment, aspect-level analysis, larger models (BERT, RoBERTa).",
]

# ----------------------------------------------------------------------
# Live-demo example reviews (app-store style)
# ----------------------------------------------------------------------
EXAMPLE_REVIEWS = {
    "Negative · crashes": "The app keeps crashing every time I try to check out. So frustrating, fix it.",
    "Positive · love it": "Love this app — super easy to order and delivery is always fast. Works great.",
    "Mixed · slow search": "Decent app but the search is slow and it keeps logging me out.",
}
DEFAULT_EXAMPLE = "Negative · crashes"
