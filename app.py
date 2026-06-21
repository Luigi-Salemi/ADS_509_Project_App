"""
ADS-509 Final Project — Results Dashboard
=========================================
An interactive Streamlit app that presents the results of the team notebook
`ADS_509_Final_Team_Project_.ipynb` (Comparing Traditional NLP and
Transformer-Based Models for Review Sentiment Classification) and lets a
visitor try both model families live on their own text.

Run locally:   streamlit run app.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import results as R

# ----------------------------------------------------------------------
# Page config + light styling
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="ADS-509 — Sentiment Model Comparison",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAVY = "#003B70"   # USD blue
GOLD = "#C69214"   # USD gold
POS = "#2E8B57"
NEG = "#C0392B"
NEU = "#9aa0a6"

st.markdown(
    f"""
    <style>
      .main .block-container {{ padding-top: 2rem; max-width: 1200px; }}
      h1, h2, h3 {{ color: {NAVY}; }}
      .metric-card {{
        background: #ffffff; border: 1px solid #e6e9ef; border-left: 5px solid {NAVY};
        border-radius: 10px; padding: 16px 18px; box-shadow: 0 1px 3px rgba(0,0,0,.05);
      }}
      .metric-card .v {{ font-size: 30px; font-weight: 700; color: {NAVY}; }}
      .metric-card .l {{ font-size: 13px; color: #5f6571; text-transform: uppercase; letter-spacing:.04em; }}
      .tag {{
        display:inline-block; background:{NAVY}10; color:{NAVY}; border:1px solid {NAVY}33;
        border-radius:999px; padding:4px 12px; margin:4px 6px 4px 0; font-size:13px; font-weight:600;
      }}
      .pending {{ color:{GOLD}; font-weight:700; }}
      .note {{ color:#5f6571; font-size:13px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def card(value, label):
    st.markdown(
        f"<div class='metric-card'><div class='v'>{value}</div>"
        f"<div class='l'>{label}</div></div>",
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown(f"### 📊 ADS-509 Final Project")
    st.caption(R.PROJECT["course"] + " · " + R.PROJECT["school"])
    section = st.radio(
        "Navigate",
        [
            "🏠 Overview",
            "🗂️ Dataset",
            "🔎 Exploratory Analysis",
            "🧮 TF-IDF Features",
            "🤖 Model Results",
            "⚖️ Model Comparison",
            "🧪 Try it Live",
        ],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown("**Team**")
    for m in R.PROJECT["team"]:
        st.caption("• " + m)
    st.divider()
    st.caption(
        "Numbers shown are transcribed from the executed team notebook. "
        "The fine-tuned DistilBERT score was not captured (training cell "
        "interrupted), so it is shown as *pending* rather than estimated."
    )


# ======================================================================
# OVERVIEW
# ======================================================================
if section == "🏠 Overview":
    st.title(R.PROJECT["title"])
    st.markdown(f"**{R.PROJECT['course']} · {R.PROJECT['school']}**")
    st.markdown("##### " + " · ".join(R.PROJECT["team"]))
    st.write("")
    st.info("**Objective** — " + R.PROJECT["objective"])

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(f"{R.RAW_REVIEW_COUNT:,}", "Reviews scraped")
    with c2:
        card(f"{R.BALANCED_TOTAL:,}", "Balanced sample")
    with c3:
        card(f"{R.LR_METRICS['Accuracy']*100:.1f}%", "Baseline accuracy")
    with c4:
        card(f"{len(R.SOURCE_APPS)}", "App sources")

    st.write("")
    st.subheader("How the project works")
    st.markdown(
        f"""
1. **Collect** — {R.RAW_REVIEW_COUNT:,} reviews self-scraped from the
   {R.SOURCE_PLATFORM} ({', '.join(R.SOURCE_APPS)}).
2. **Label** — 4–5★ → *positive*, 1–2★ → *negative*, 3★ dropped; then a
   balanced sample of **{R.BALANCED_TOTAL:,}** reviews (3,250 per class).
3. **Explore** — review length, word frequency, and TF-IDF analysis.
4. **Model A (classic)** — TF-IDF + Logistic Regression baseline.
5. **Model B (transformer)** — fine-tuned DistilBERT.
6. **Compare** — accuracy / F1 of both approaches.
        """
    )
    st.markdown(
        "<span class='note'>Use the sidebar to walk through each stage. "
        "The final tab lets you classify your own review with both a classic "
        "model and a live transformer.</span>",
        unsafe_allow_html=True,
    )


# ======================================================================
# DATASET
# ======================================================================
elif section == "🗂️ Dataset":
    st.title("🗂️ Dataset")
    st.caption(
        f"{R.RAW_REVIEW_COUNT:,} reviews scraped from the {R.SOURCE_PLATFORM}: "
        + ", ".join(R.SOURCE_APPS)
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Star-rating distribution")
        rd = pd.DataFrame(
            {"Stars": list(R.RATING_DISTRIBUTION.keys()),
             "Reviews": list(R.RATING_DISTRIBUTION.values())}
        ).sort_values("Stars")

        def star_color(s):
            return NEG if s <= 2 else (NEU if s == 3 else POS)

        rd["color"] = rd["Stars"].apply(star_color)
        fig = px.bar(rd, x="Stars", y="Reviews", text="Reviews")
        fig.update_traces(marker_color=rd["color"], textposition="outside")
        fig.update_layout(
            xaxis_title="Star rating", yaxis_title="Number of reviews",
            showlegend=False, height=380, margin=dict(t=10, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(
            "<span class='note'>Ratings are skewed toward 5★ and 1★ — typical "
            "of app-store reviews, where people write when delighted or upset. "
            "3★ reviews were dropped for the binary task.</span>",
            unsafe_allow_html=True,
        )

    with right:
        st.subheader("Balanced binary sample")
        lb = pd.DataFrame(
            {"Label": list(R.LABEL_BALANCE.keys()),
             "Reviews": list(R.LABEL_BALANCE.values())}
        )
        fig2 = px.pie(lb, names="Label", values="Reviews", hole=0.55,
                      color="Label",
                      color_discrete_map={"Positive (1)": POS, "Negative (0)": NEG})
        fig2.update_layout(height=300, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            card(f"{R.UNIQUE_REVIEWS:,}", "Unique reviews")
        with c2:
            card(f"{R.DUPLICATE_REVIEWS:,}", "Duplicates")


# ======================================================================
# EDA
# ======================================================================
elif section == "🔎 Exploratory Analysis":
    st.title("🔎 Exploratory Analysis")

    st.subheader("Average review length by sentiment")
    al = pd.DataFrame(
        {"Sentiment": list(R.AVG_REVIEW_LENGTH.keys()),
         "Avg words": [round(v, 1) for v in R.AVG_REVIEW_LENGTH.values()]}
    )
    figl = px.bar(al, x="Sentiment", y="Avg words", text="Avg words",
                  color="Sentiment",
                  color_discrete_map={"Positive (1)": POS, "Negative (0)": NEG})
    figl.update_traces(textposition="outside")
    figl.update_layout(showlegend=False, height=330, margin=dict(t=10, b=10),
                       yaxis_title="Average words per review")
    st.plotly_chart(figl, use_container_width=True)
    st.markdown(
        "<span class='note'>Negative reviews are ~3× longer on average "
        "(32.7 vs 10.9 words): unhappy users explain what went wrong, while "
        "happy users leave short praise.</span>",
        unsafe_allow_html=True,
    )

    st.divider()
    st.subheader("Top 20 most frequent words")
    tw = pd.DataFrame(R.TOP_WORDS, columns=["Word", "Frequency"]).sort_values("Frequency")
    figw = px.bar(tw, x="Frequency", y="Word", orientation="h", text="Frequency")
    figw.update_traces(marker_color=NAVY, textposition="outside")
    figw.update_layout(height=560, margin=dict(t=10, b=10),
                       yaxis_title="", xaxis_title="Frequency (stopwords removed)")
    st.plotly_chart(figw, use_container_width=True)

    wc_path = os.path.join(os.path.dirname(__file__), "assets", "wordcloud.png")
    if os.path.exists(wc_path):
        st.subheader("Word cloud")
        st.image(wc_path, use_container_width=True)


# ======================================================================
# TF-IDF
# ======================================================================
elif section == "🧮 TF-IDF Features":
    st.title("🧮 TF-IDF Feature Analysis")
    st.write(
        "TF-IDF (Term Frequency–Inverse Document Frequency) highlights words "
        "that are distinctive to reviews while down-weighting words that appear "
        "everywhere. These are the top features the vectorizer selected."
    )
    st.subheader("Top 20 TF-IDF terms")
    st.markdown(
        "".join(f"<span class='tag'>{t}</span>" for t in R.TFIDF_TOP_FEATURES),
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        "<span class='note'>Brand/app terms (<b>amazon</b>, <b>ebay</b>, "
        "<b>best buy</b>), action terms (<b>order</b>, <b>search</b>, "
        "<b>delivery</b>), and sentiment cues (<b>love</b>, <b>good</b>, "
        "<b>great</b>, <b>slow</b>) dominate — exactly the signal a linear "
        "classifier can exploit.</span>",
        unsafe_allow_html=True,
    )


# ======================================================================
# MODEL RESULTS
# ======================================================================
elif section == "🤖 Model Results":
    st.title("🤖 Model Results")

    tab1, tab2 = st.tabs(["TF-IDF + Logistic Regression", "Fine-Tuned DistilBERT"])

    # ---- Logistic Regression ----
    with tab1:
        st.subheader("Baseline — TF-IDF + Logistic Regression")
        cols = st.columns(len(R.LR_METRICS))
        for col, (k, v) in zip(cols, R.LR_METRICS.items()):
            with col:
                card(f"{v*100:.1f}%", k)

        st.write("")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Classification report**")
            rep = pd.DataFrame(R.LR_REPORT).T
            rep = rep.rename(columns={"f1": "f1-score"})
            rep["support"] = rep["support"].astype(int)
            st.dataframe(
                rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                                  "f1-score": "{:.2f}", "support": "{:d}"}),
                use_container_width=True,
            )
        with c2:
            st.markdown("**Confusion matrix**")
            labels = ["Negative", "Positive"]
            figc = go.Figure(data=go.Heatmap(
                z=R.LR_CONFUSION, x=labels, y=labels,
                text=R.LR_CONFUSION, texttemplate="%{text}",
                colorscale="Blues", showscale=False,
            ))
            figc.update_layout(
                height=320, margin=dict(t=10, b=10),
                xaxis_title="Predicted", yaxis_title="Actual",
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(figc, use_container_width=True)
            st.caption(R.LR_CONFUSION_NOTE)

    # ---- DistilBERT ----
    with tab2:
        st.subheader("Transformer — Fine-Tuned DistilBERT")
        st.markdown("**Configuration**")
        cfg = pd.DataFrame(
            {"Setting": list(R.DISTILBERT_CONFIG.keys()),
             "Value": list(R.DISTILBERT_CONFIG.values())}
        )
        st.table(cfg)
        st.warning(
            "⚠️ **Result pending.** In the saved notebook the DistilBERT "
            "training cell was interrupted after the first step "
            f"(only the initial loss of {R.DISTILBERT_FIRST_STEP_LOSS} was "
            "recorded), so no evaluation accuracy / F1 was captured. To keep "
            "this dashboard honest, no value is shown here instead of an "
            "estimate. Re-running the notebook's fine-tuning cell to "
            "completion will fill this in."
        )
        st.markdown(
            "<span class='note'>Want a transformer in action right now? The "
            "<b>🧪 Try it Live</b> tab runs a pretrained sentiment transformer "
            "on any text you enter.</span>",
            unsafe_allow_html=True,
        )


# ======================================================================
# COMPARISON
# ======================================================================
elif section == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison")
    st.caption("Classic NLP baseline vs. the transformer approach.")

    comp = pd.DataFrame(R.MODEL_COMPARISON)
    show = comp.copy()
    show["Accuracy"] = show["Accuracy"].apply(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "— pending —")
    show["F1 (weighted)"] = show["F1 (weighted)"].apply(
        lambda v: f"{v*100:.1f}%" if pd.notna(v) else "— pending —")
    st.dataframe(show, use_container_width=True, hide_index=True)

    # bar chart — only the measured model has bars; pending shown as annotation
    plot_df = comp.dropna(subset=["Accuracy", "F1 (weighted)"]).melt(
        id_vars="Model", value_vars=["Accuracy", "F1 (weighted)"],
        var_name="Metric", value_name="Score")
    figb = px.bar(plot_df, x="Model", y="Score", color="Metric",
                  barmode="group", text=plot_df["Score"].apply(lambda v: f"{v*100:.1f}%"),
                  color_discrete_sequence=[NAVY, GOLD])
    figb.update_traces(textposition="outside")
    figb.update_layout(height=400, yaxis=dict(range=[0.8, 1.0], tickformat=".0%"),
                       margin=dict(t=30, b=10), yaxis_title="Score")
    figb.add_annotation(
        x="Fine-Tuned DistilBERT" if "Fine-Tuned DistilBERT" in comp["Model"].values else 1,
        y=0.9, text="DistilBERT<br>pending", showarrow=False,
        font=dict(color=GOLD, size=13),
    )
    st.plotly_chart(figb, use_container_width=True)

    st.success(
        "**Takeaway so far** — the lightweight TF-IDF + Logistic Regression "
        f"baseline already reaches **{R.LR_METRICS['Accuracy']*100:.1f}% "
        f"accuracy / {R.LR_METRICS['F1 (weighted)']*100:.1f}% F1**. It is fast, "
        "interpretable, and a strong bar for the transformer to beat once its "
        "fine-tuning run is completed."
    )


# ======================================================================
# LIVE DEMO
# ======================================================================
elif section == "🧪 Try it Live":
    st.title("🧪 Try it Live")
    st.write(
        "Type a product/app review and see how each model family classifies it. "
        "This mirrors the project's two approaches: a **classic** TF-IDF + "
        "Logistic Regression model and a **transformer**."
    )
    st.markdown(
        "<span class='note'><b>Note:</b> the classic model is trained live on a "
        "bundled review corpus, and the transformer is an off-the-shelf "
        "<i>pretrained</i> sentiment model "
        "(<code>distilbert-base-uncased-finetuned-sst-2-english</code>) used for "
        "inference. This is a hands-on demo — it is separate from the project's "
        "own fine-tuned DistilBERT (whose score is still pending).</span>",
        unsafe_allow_html=True,
    )

    # ---------- classic model: train once on bundled CSV ----------
    @st.cache_resource(show_spinner="Training the classic TF-IDF + Logistic Regression model…")
    def get_classic_model():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        path = os.path.join(os.path.dirname(__file__), "data",
                            "product_reviews_dataset.csv")
        df = pd.read_csv(path)
        df = df[df["label"].isin([0, 1])].dropna(subset=["text"])
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(df["text"].astype(str), df["label"].astype(int))
        return pipe, len(df)

    # ---------- transformer: lazy load pretrained pipeline ----------
    @st.cache_resource(show_spinner="Loading the pretrained transformer (first run downloads ~250 MB)…")
    def get_transformer():
        from transformers import pipeline
        return pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
        )

    example = "The app keeps crashing every time I try to checkout. So frustrating."
    text = st.text_area("Your review", value=example, height=110)

    run_tf = st.checkbox(
        "Also run the transformer (downloads & loads the model on first use)",
        value=False,
        help="Leave off for an instant classic-model result. Turn on to compare "
             "with a real transformer — slower on first click while the model loads.",
    )

    if st.button("Classify", type="primary") and text.strip():
        col1, col2 = st.columns(2)

        # classic
        with col1:
            st.markdown("#### Classic · TF-IDF + LogReg")
            try:
                model, n = get_classic_model()
                proba = model.predict_proba([text])[0]
                pred = int(proba.argmax())
                label = "Positive 😊" if pred == 1 else "Negative 😠"
                conf = proba[pred]
                (st.success if pred == 1 else st.error)(f"**{label}**")
                st.metric("Confidence", f"{conf*100:.1f}%")
                st.caption(f"Trained live on {n:,} bundled labeled reviews.")
            except Exception as e:
                st.warning(f"Classic model unavailable: {e}")

        # transformer
        with col2:
            st.markdown("#### Transformer · DistilBERT (SST-2)")
            if not run_tf:
                st.info("Enable the checkbox above to run the transformer.")
            else:
                try:
                    clf = get_transformer()
                    out = clf(text[:512])[0]
                    pos = out["label"].upper() == "POSITIVE"
                    label = "Positive 😊" if pos else "Negative 😠"
                    (st.success if pos else st.error)(f"**{label}**")
                    st.metric("Confidence", f"{out['score']*100:.1f}%")
                    st.caption("Pretrained distilbert-base-uncased-finetuned-sst-2-english.")
                except Exception as e:
                    st.warning(
                        "Transformer couldn't load in this environment "
                        f"(often a memory limit on free hosting): {e}"
                    )

    st.divider()
    st.caption(
        "Tip: try a long, detailed complaint vs. a short \"love it!\" — the EDA "
        "showed negative reviews tend to be much longer."
    )
