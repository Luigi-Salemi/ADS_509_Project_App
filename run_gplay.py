"""
Full Google Play pipeline (reproduces the team notebook) — self-run end-to-end.
Scrapes 5 retail apps, runs EDA + TF-IDF/LogReg + fine-tunes DistilBERT, and saves:
  - data/reviews.csv      (collected reviews, used by the live demo)
  - model/                (fine-tuned DistilBERT, used by the live demo)
  - results_real.json     (every figure the dashboard shows)
Run:  python run_gplay.py
"""
import json, os, re, time
from collections import Counter
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
APPS = {"Amazon Shopping": "com.amazon.mShop.android.shopping", "Walmart": "com.walmart.android",
        "Target": "com.target.ui", "Best Buy": "com.bestbuy.android", "eBay": "com.ebay.mobile"}

# ---- 1. scrape ----
from google_play_scraper import reviews, Sort
rows = []
for name, app_id in APPS.items():
    try:
        res, _ = reviews(app_id, lang="en", country="us", sort=Sort.NEWEST, count=2000)
        for r in res:
            rows.append({"product_name": name, "text": r["content"], "rating": r["score"],
                         "review_date": str(r["at"])[:10]})
        print("scraped", name, len(res), flush=True)
    except Exception as e:
        print("fail", name, e, flush=True)
    time.sleep(2)
raw = pd.DataFrame(rows).dropna(subset=["text"])
raw_count = len(raw)
rating_dist = {int(k): int(v) for k, v in raw["rating"].value_counts().sort_index().items()}
app_dist = {str(k): int(v) for k, v in raw["product_name"].value_counts().items()}

# ---- 2. label + balance ----
raw["label"] = raw["rating"].apply(lambda x: 1 if x >= 4 else (0 if x <= 2 else None))
df = raw.dropna(subset=["label"]).copy(); df["label"] = df["label"].astype(int)
n = min(3250, int((df.label == 1).sum()), int((df.label == 0).sum()))
bal = pd.concat([df[df.label == 1].sample(n, random_state=SEED),
                 df[df.label == 0].sample(n, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)
bal["text"] = bal["text"].fillna("")
balanced_total = len(bal)
label_balance = {"Positive (1)": int((bal.label == 1).sum()), "Negative (0)": int((bal.label == 0).sum())}
unique_reviews = int(bal["text"].nunique()); duplicate_reviews = balanced_total - unique_reviews

# save the collected reviews for the live demo (binary subset)
os.makedirs(os.path.join(HERE, "data"), exist_ok=True)
df[["product_name", "text", "rating", "review_date", "label"]].to_csv(os.path.join(HERE, "data", "reviews.csv"), index=False)
print("balanced", balanced_total, label_balance, flush=True)

# ---- 3. EDA ----
bal["rlen"] = bal["text"].apply(lambda x: len(x.split()))
avg = bal.groupby("label")["rlen"].mean()
avg_len = {"Negative (0)": round(float(avg[0]), 1), "Positive (1)": round(float(avg[1]), 1)}
import nltk; nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords
stop = set(stopwords.words("english"))
toks = re.findall(r"[A-Za-z']{3,}", " ".join(bal["text"]))
freq = Counter(w.lower() for w in toks if w.lower() not in stop)
top_words = [[w, int(c)] for w, c in freq.most_common(20)]
top_words_full = [[w, int(c)] for w, c in freq.most_common(120)]
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_features = sorted(TfidfVectorizer(max_features=20, stop_words="english").fit(bal["text"]).get_feature_names_out().tolist())

# ---- 4. TF-IDF + Logistic Regression ----
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix)
Xtr, Xte, ytr, yte = train_test_split(bal["text"], bal["label"], test_size=0.2, random_state=SEED, stratify=bal["label"])
vec = TfidfVectorizer(stop_words="english", max_features=5000)
lrm = LogisticRegression(max_iter=1000, random_state=SEED).fit(vec.fit_transform(Xtr), ytr)
lp = lrm.predict(vec.transform(Xte)); rep = classification_report(yte, lp, output_dict=True)
lr = {"accuracy": round(accuracy_score(yte, lp), 4), "precision": round(precision_score(yte, lp, average="weighted"), 4),
      "recall": round(recall_score(yte, lp, average="weighted"), 4), "f1": round(f1_score(yte, lp, average="weighted"), 4),
      "report": {"Negative (0)": {k: round(rep["0"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(rep["0"]["support"])},
                 "Positive (1)": {k: round(rep["1"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(rep["1"]["support"])}},
      "train_size": int(Xtr.shape[0]), "test_size": int(Xte.shape[0])}
print("LR", lr["accuracy"], lr["f1"], flush=True)

# ---- 5. fine-tune DistilBERT + save model ----
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
hf = Dataset.from_pandas(bal[["text", "label"]], preserve_index=False).train_test_split(test_size=0.2, seed=SEED)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = hf.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True, remove_columns=["text"]).rename_column("label", "labels")
enc = enc.remove_columns([c for c in enc["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]])
coll = DataCollatorWithPadding(tokenizer=tok)
tl = DataLoader(enc["train"], shuffle=True, batch_size=8, collate_fn=coll); el = DataLoader(enc["test"], batch_size=8, collate_fn=coll)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5); model.train()
for step, b in enumerate(tl):
    b = {k: v.to(dev) for k, v in b.items()}; o = model(**b); o.loss.backward(); opt.step(); opt.zero_grad()
    if step % 150 == 0: print("step", step, round(o.loss.item(), 4), flush=True)
model.eval(); preds, labels = [], []
with torch.no_grad():
    for b in el:
        y = b["labels"]; b = {k: v.to(dev) for k, v in b.items()}
        preds += torch.argmax(model(**b).logits, -1).cpu().tolist(); labels += y.tolist()
drep = classification_report(labels, preds, output_dict=True); cm = confusion_matrix(labels, preds)
distilbert = {"accuracy": round(accuracy_score(labels, preds), 4), "f1": round(f1_score(labels, preds, average="weighted"), 4),
              "report": {"Negative": {k: round(drep["0"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(drep["0"]["support"])},
                         "Positive": {k: round(drep["1"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(drep["1"]["support"])}},
              "confusion": cm.tolist(), "correct": int(np.trace(cm)), "misclassified": int(cm.sum() - np.trace(cm)), "test_size": int(len(labels))}
print("DistilBERT", distilbert["accuracy"], distilbert["f1"], "cm", cm.tolist(), flush=True)
model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}; model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
model.save_pretrained(os.path.join(HERE, "model")); tok.save_pretrained(os.path.join(HERE, "model"))

# ---- 6. save results_real.json ----
json.dump({"dataset": "Google Play Store reviews (5 retail apps)", "raw_count": raw_count,
           "rating_distribution": rating_dist, "app_dist": app_dist, "balanced_total": balanced_total,
           "label_balance": label_balance, "unique_reviews": unique_reviews, "duplicate_reviews": duplicate_reviews,
           "avg_review_length": avg_len, "top_words": top_words, "top_words_full": top_words_full,
           "tfidf_features": tfidf_features, "lr": lr, "distilbert": distilbert},
          open(os.path.join(HERE, "results_real.json"), "w"), indent=2)
print("WROTE results_real.json + data/reviews.csv + model/", flush=True)
