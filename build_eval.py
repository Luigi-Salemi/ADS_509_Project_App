"""Re-evaluate ALL models on their exact held-out splits (leakage-free) and emit
results_eval.json: LR, Naive Bayes, DistilBERT metrics + reports + confusion + ROC.
Reproduces run_gplay.py's splits exactly (SEED=42) so DistilBERT is scored on the
same test set the bundled model was trained against."""
import json, os, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_curve, auc)

HERE = os.path.dirname(os.path.abspath(__file__)); SEED = 42
df = pd.read_csv(os.path.join(HERE, "data", "reviews.csv"))
df["text"] = df["text"].astype(str); df["label"] = df["label"].astype(int)
# reconstruct the exact balanced set from run_gplay.py
n = min(3250, int((df.label == 1).sum()), int((df.label == 0).sum()))
bal = pd.concat([df[df.label == 1].sample(n, random_state=SEED),
                 df[df.label == 0].sample(n, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)

def ds(fpr, tpr, k=70):
    idx = np.linspace(0, len(fpr) - 1, min(k, len(fpr))).astype(int)
    return [round(float(x), 4) for x in fpr[idx]], [round(float(x), 4) for x in tpr[idx]]

def rep2(y, p, neg="Negative (0)", pos="Positive (1)"):
    r = classification_report(y, p, output_dict=True)
    return {neg: {k: round(r["0"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(r["0"]["support"])},
            pos: {k: round(r["1"][k], 2) for k in ["precision", "recall", "f1-score"]} | {"support": int(r["1"]["support"])}}

def metrics(y, p):
    return {"Accuracy": round(accuracy_score(y, p), 4),
            "Precision (weighted)": round(precision_score(y, p, average="weighted"), 4),
            "Recall (weighted)": round(recall_score(y, p, average="weighted"), 4),
            "F1 (weighted)": round(f1_score(y, p, average="weighted"), 4)}

out = {"roc": {}}
# ---- LR + NB on the sklearn split (their held-out set) ----
Xtr, Xte, ytr, yte = train_test_split(bal["text"], bal["label"], test_size=0.2, random_state=SEED, stratify=bal["label"])
vec = TfidfVectorizer(stop_words="english", max_features=5000)
Xtr_t, Xte_t = vec.fit_transform(Xtr), vec.transform(Xte)
for key, label, mdl in [("lr", "Logistic Regression", LogisticRegression(max_iter=1000, random_state=SEED)),
                        ("nb", "Naive Bayes", MultinomialNB())]:
    mdl.fit(Xtr_t, ytr)
    p = mdl.predict(Xte_t); prob = mdl.predict_proba(Xte_t)[:, 1]
    out[key] = {"metrics": metrics(yte, p), "report": rep2(yte, p)}
    fpr, tpr, _ = roc_curve(yte, prob); f, t = ds(fpr, tpr)
    out["roc"][label] = {"fpr": f, "tpr": t, "auc": round(float(auc(fpr, tpr)), 4)}
    print(key, out[key]["metrics"]["Accuracy"], out[key]["metrics"]["F1 (weighted)"], "AUC", out["roc"][label]["auc"], flush=True)

# ---- DistilBERT on the HuggingFace split (the model's true held-out set) ----
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
hf = Dataset.from_pandas(bal[["text", "label"]], preserve_index=False).train_test_split(test_size=0.2, seed=SEED)
te = hf["test"]; texts, ylab = list(te["text"]), list(te["label"])
tok = AutoTokenizer.from_pretrained(os.path.join(HERE, "model"))
model = AutoModelForSequenceClassification.from_pretrained(os.path.join(HERE, "model")).to(dev).eval()
preds, probs = [], []
with torch.no_grad():
    for i in range(0, len(texts), 32):
        enc = tok(texts[i:i + 32], truncation=True, padding=True, max_length=128, return_tensors="pt").to(dev)
        logits = model(**enc).logits
        preds += torch.argmax(logits, -1).cpu().tolist()
        probs += torch.softmax(logits, 1)[:, 1].cpu().tolist()
cm = confusion_matrix(ylab, preds)
dm = metrics(ylab, preds)
out["distilbert"] = {"metrics": dm,
                     "report": rep2(ylab, preds, "Negative", "Positive"),
                     "confusion": cm.tolist(), "correct": int(np.trace(cm)),
                     "misclassified": int(cm.sum() - np.trace(cm)), "test_size": int(len(ylab))}
fpr, tpr, _ = roc_curve(ylab, probs); f, t = ds(fpr, tpr)
out["roc"]["Fine-Tuned DistilBERT"] = {"fpr": f, "tpr": t, "auc": round(float(auc(fpr, tpr)), 4)}
print("distilbert", dm["Accuracy"], dm["F1 (weighted)"], "AUC", out["roc"]["Fine-Tuned DistilBERT"]["auc"], "cm", cm.tolist(), flush=True)

out["comparison"] = [
    {"Model": "TF-IDF + Logistic Regression", "Accuracy": out["lr"]["metrics"]["Accuracy"], "F1 (weighted)": out["lr"]["metrics"]["F1 (weighted)"]},
    {"Model": "TF-IDF + Naive Bayes", "Accuracy": out["nb"]["metrics"]["Accuracy"], "F1 (weighted)": out["nb"]["metrics"]["F1 (weighted)"]},
    {"Model": "Fine-Tuned DistilBERT", "Accuracy": out["distilbert"]["metrics"]["Accuracy"], "F1 (weighted)": out["distilbert"]["metrics"]["F1 (weighted)"]},
]
json.dump(out, open(os.path.join(HERE, "results_eval.json"), "w"), indent=2)
print("WROTE results_eval.json", flush=True)
