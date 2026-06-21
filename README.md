# ADS-509 — Sentiment Model Comparison Dashboard

An interactive [Streamlit](https://streamlit.io) app that presents the results of the
ADS-509 Applied Text Mining final project:
**Comparing Traditional NLP and Transformer-Based Models for Review Sentiment Classification.**

> **Team:** Gagandeep Singh · Shivam Patel · Luigi Salemi
> **Course:** ADS-509 Applied Text Mining — University of San Diego

## What it shows

| Section | Contents |
|---|---|
| 🏠 Overview | Project objective, pipeline, headline metrics |
| 🗂️ Dataset | Star-rating distribution, balanced binary sample, dedup stats |
| 🔎 Exploratory Analysis | Review length by sentiment, top-20 words, word cloud |
| 🧮 TF-IDF Features | Top distinctive terms selected by TF-IDF |
| 🤖 Model Results | LR metrics + report; DistilBERT metrics, workflow, classification report & confusion matrix |
| ⚖️ Model Comparison | Classic baseline (91.2%) vs. fine-tuned DistilBERT (94.6%) |
| 📌 Conclusion | Key takeaways, limitations, and next steps |
| 🧪 Try it Live | Classify your own review with a **classic** model **and** a **pretrained transformer** |

Results are the team's finalized figures from the project presentation
(*Traditional NLP vs. Transformers for Sentiment Classification*), cross-checked against the
executed team notebook (`ADS_509_Final_Team_Project_.ipynb`). Headline result: the
fine-tuned **DistilBERT (94.6% accuracy / F1)** beats the **TF-IDF + Logistic Regression
baseline (91.2%)** by ~3.5 points. Because the notebook scrapes Google Play **live**, each
run varies slightly, so the dashboard pins the presentation's reported numbers.

> **Note on the live demo:** the *classic* model is trained at runtime on a bundled review
> corpus, and the *transformer* is an off-the-shelf **pretrained** sentiment model
> (`distilbert-base-uncased-finetuned-sst-2-english`) used for inference. It is a hands-on
> demonstration and is intentionally separate from the project's own fine-tuned DistilBERT.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The transformer in **Try it Live** downloads ~250 MB on first use; leave its checkbox off
for an instant classic-model-only result.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (already done if you cloned it from there).
2. Go to **https://share.streamlit.io** and sign in.
3. **New app → From existing repo**, pick this repo, branch `main`, main file `app.py`.
4. Click **Deploy**. First build takes a few minutes (it installs PyTorch).

## Files

```
app.py            # the dashboard
results.py        # captured results from the notebook (single source of truth)
requirements.txt  # dependencies (CPU-only torch)
data/             # bundled review CSV used by the live classic model
assets/           # word cloud image
```
