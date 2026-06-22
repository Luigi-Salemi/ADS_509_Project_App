"""Fine-tune DistilBERT on the Amazon Reviews (McAuley) data and save it to ./model
so the dashboard's live demo can run it in-process. Mirrors run_from_csv.py's training."""
import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 42
df = pd.read_csv(os.path.join(HERE, "data", "reviews.csv"))
df = df[df["label"].isin([0, 1])].dropna(subset=["text"]).copy()
df["label"] = df["label"].astype(int)
n = min(int((df.label == 1).sum()), int((df.label == 0).sum()))
df = pd.concat([df[df.label == 1].sample(n, random_state=SEED),
                df[df.label == 0].sample(n, random_state=SEED)]).sample(frac=1, random_state=SEED).reset_index(drop=True)[["text", "label"]]
print("training rows:", len(df), flush=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
hf = Dataset.from_pandas(df, preserve_index=False).train_test_split(test_size=0.2, seed=SEED)
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
enc = hf.map(lambda b: tok(b["text"], truncation=True, max_length=128), batched=True,
             remove_columns=["text"]).rename_column("label", "labels")
enc = enc.remove_columns([c for c in enc["train"].column_names if c not in ["input_ids", "attention_mask", "labels"]])
coll = DataCollatorWithPadding(tokenizer=tok)
tl = DataLoader(enc["train"], shuffle=True, batch_size=8, collate_fn=coll)
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
model.train()
for step, b in enumerate(tl):
    b = {k: v.to(device) for k, v in b.items()}
    out = model(**b); out.loss.backward(); opt.step(); opt.zero_grad()
    if step % 150 == 0:
        print(f"step {step}/{len(tl)} loss {out.loss.item():.4f}", flush=True)

model.config.id2label = {0: "NEGATIVE", 1: "POSITIVE"}
model.config.label2id = {"NEGATIVE": 0, "POSITIVE": 1}
out_dir = os.path.join(HERE, "model")
model.save_pretrained(out_dir); tok.save_pretrained(out_dir)
print("saved fine-tuned model to", out_dir, flush=True)
