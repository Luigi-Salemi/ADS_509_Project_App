# ADS-509 — Sentiment Model Comparison

A simple [Streamlit](https://streamlit.io) dashboard for the ADS-509 final project:
**Traditional NLP vs. Transformers for Review Sentiment Classification.**

> **Live app:** https://ads509projectresultsteamgroup4.streamlit.app
> **Team:** Gagandeep Singh · Shivam Patel · Luigi Salemi
> **Course:** ADS-509 Applied Text Mining — University of San Diego

One scrolling page: dataset overview, EDA, model comparison, and a live "try it" classifier.

## Data & results

All figures come from **one end-to-end run** of the notebook's pipeline on the project's real
dataset — **Amazon Reviews 2023 (McAuley-Lab)**, ~10K reviews across 10 product categories,
balanced and split stratified 80/20. No scraping; nothing estimated.

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + Logistic Regression | **90.0%** | 90.0% |
| Fine-tuned DistilBERT | **94.0%** | 94.0% |

The pipeline is reproduced in `run_from_csv.py`; `build_results.py` writes `results.py` from its
output (`results_real.json`).

> **Live demo note:** the *classic* model is trained at runtime on the bundled reviews; the
> *transformer* is an off-the-shelf pretrained model
> (`distilbert-base-uncased-finetuned-sst-2-english`) used for inference (optional checkbox).

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Files

```
app.py             # the dashboard (single page)
results.py         # numbers from the real run (generated)
run_from_csv.py    # reproduces the notebook pipeline on the real dataset
build_results.py   # results_real.json -> results.py
data/reviews.csv   # real review data used by the live classic model
assets/            # word cloud image
```
