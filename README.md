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
| 🤖 Model Results | Logistic Regression metrics, classification report, confusion matrix; DistilBERT config |
| ⚖️ Model Comparison | Classic baseline vs. transformer |
| 🧪 Try it Live | Classify your own review with a **classic** model **and** a **pretrained transformer** |

All numbers are transcribed directly from the executed team notebook
(`ADS_509_Final_Team_Project_.ipynb`). The fine-tuned DistilBERT score is shown as
**pending** because that training cell was interrupted in the saved notebook — no value is
estimated.

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
