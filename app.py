"""
ADS-509 Final Project — Results Dashboard
=========================================
Interactive Streamlit app presenting the results of
"Traditional NLP vs. Transformers for Sentiment Classification"
(team notebook `ADS_509_Final_Team_Project_.ipynb`), plus a live demo that
lets a visitor classify their own review with both model families.

Run locally:   streamlit run app.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import results as R

# ----------------------------------------------------------------------
# Page config + styling
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
        background:#fff; border:1px solid #e6e9ef; border-left:5px solid {NAVY};
        border-radius:10px; padding:16px 18px; box-shadow:0 1px 3px rgba(0,0,0,.05);
      }}
      .metric-card.gold {{ border-left-color:{GOLD}; }}
      .metric-card .v {{ font-size:30px; font-weight:700; color:{NAVY}; }}
      .metric-card.gold .v {{ color:{GOLD}; }}
      .metric-card .l {{ font-size:13px; color:#5f6571; text-transform:uppercase; letter-spacing:.04em; }}
      .tag {{
        display:inline-block; background:{NAVY}10; color:{NAVY}; border:1px solid {NAVY}33;
        border-radius:999px; padding:4px 12px; margin:4px 6px 4px 0; font-size:13px; font-weight:600;
      }}
      .tag.hot {{ background:{GOLD}1a; color:#8a6400; border-color:{GOLD}66; }}
      .note {{ color:#5f6571; font-size:13px; }}
      .pipe {{ background:#f4f6fa; border:1px solid #e6e9ef; border-radius:12px; padding:14px;
               text-align:center; height:140px; }}
      .pipe-n {{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
                 border-radius:50%; background:{NAVY}; color:#fff; font-weight:700; margin-bottom:6px; }}
      .pipe-t {{ font-weight:700; color:{NAVY}; font-size:15px; }}
      .pipe-d {{ font-size:12px; color:#5f6571; margin-top:4px; }}
      .pipe-arrow {{ font-size:26px; color:{GOLD}; text-align:center; padding-top:48px; }}
      .step {{ border-left:4px solid {NAVY}; background:#f4f6fa; border-radius:0 8px 8px 0;
               padding:10px 14px; margin-bottom:10px; }}
      .step b {{ color:{NAVY}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def card(value, label, gold=False):
    cls = "metric-card gold" if gold else "metric-card"
    st.markdown(
        f"<div class='{cls}'><div class='v'>{value}</div>"
        f"<div class='l'>{label}</div></div>",
        unsafe_allow_html=True,
    )


def report_table(report: dict):
    rep = pd.DataFrame(report).T.rename(columns={"f1": "f1-score"})
    rep["support"] = rep["support"].astype(int)
    return rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                             "f1-score": "{:.2f}", "support": "{:d}"})


def confusion_fig(matrix, colorscale="Blues"):
    labels = ["Negative", "Positive"]
    fig = go.Figure(data=go.Heatmap(
        z=matrix, x=labels, y=labels, text=matrix, texttemplate="%{text}",
        textfont=dict(size=18), colorscale=colorscale, showscale=False,
    ))
    fig.update_layout(height=330, margin=dict(t=10, b=10),
                      xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"))
    return fig


# ----------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 ADS-509 Final Project")
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
            "📌 Conclusion",
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
        "Results shown are the team's finalized figures (from the project "
        "presentation), cross-checked against the executed notebook. The "
        "notebook scrapes live data, so each run varies slightly."
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
        card(f"{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%", "Best accuracy (DistilBERT)", gold=True)
    with c4:
        card(f"{len(R.SOURCE_APPS)}", "App sources")

    st.write("")
    st.subheader("Project at a glance")
    cards = R.OVERVIEW_CARDS
    row1 = st.columns(2)
    row2 = st.columns(2)
    for col, (title, desc) in zip(row1 + row2, cards):
        with col:
            st.markdown(
                f"<div class='step'><b>{title}</b><br>"
                f"<span class='note'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        "<span class='note'>Use the sidebar to walk through each stage — from "
        "data collection to the final model comparison. The last tab lets you "
        "classify your own review with both a classic model and a live "
        "transformer.</span>",
        unsafe_allow_html=True,
    )


# ======================================================================
# DATASET
# ======================================================================
elif section == "🗂️ Dataset":
    st.title("🗂️ Dataset")
    st.caption(
        f"{R.RAW_REVIEW_COUNT:,} reviews ({R.REVIEWS_PER_APP:,} per app) scraped "
        f"from the {R.SOURCE_PLATFORM} via `{R.SCRAPER_LIB}`: "
        + ", ".join(R.SOURCE_APPS)
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Star-rating distribution")
        rd = pd.DataFrame(
            {"Stars": list(R.RATING_DISTRIBUTION.keys()),
             "Reviews": list(R.RATING_DISTRIBUTION.values())}
        ).sort_values("Stars")
        rd["color"] = rd["Stars"].apply(lambda s: NEG if s <= 2 else (NEU if s == 3 else POS))
        fig = px.bar(rd, x="Stars", y="Reviews", text="Reviews")
        fig.update_traces(marker_color=rd["color"], textposition="outside")
        fig.update_layout(xaxis_title="Star rating", yaxis_title="Number of reviews",
                          showlegend=False, height=360, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Balanced binary sample")
        lb = pd.DataFrame({"Label": list(R.LABEL_BALANCE.keys()),
                           "Reviews": list(R.LABEL_BALANCE.values())})
        fig2 = px.pie(lb, names="Label", values="Reviews", hole=0.55, color="Label",
                      color_discrete_map={"Positive (1)": POS, "Negative (0)": NEG})
        fig2.update_layout(height=280, margin=dict(t=10, b=10),
                           legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(fig2, use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            card(f"{R.BALANCED_TOTAL:,}", "Total samples")
        with c2:
            card(f"{R.DUPLICATE_REVIEWS:,}", "Duplicates handled")

    st.divider()
    st.subheader("Text preprocessing pipeline")
    steps = R.PIPELINE_STEPS
    cols = st.columns(len(steps) * 2 - 1)
    for i, (name, desc) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(
                f"<div class='pipe'><div class='pipe-n'>{i+1}</div>"
                f"<div class='pipe-t'>{name}</div>"
                f"<div class='pipe-d'>{desc}</div></div>",
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[i * 2 + 1]:
                st.markdown("<div class='pipe-arrow'>→</div>", unsafe_allow_html=True)
    st.caption(
        f"Star ratings mapped to binary labels (4-5 → positive, 1-2 → negative, "
        f"3 dropped); balanced to {R.BALANCED_TOTAL:,} samples and split 80/20 "
        f"→ {R.TRAIN_SIZE:,} train / {R.TEST_SIZE:,} test."
    )


# ======================================================================
# EDA
# ======================================================================
elif section == "🔎 Exploratory Analysis":
    st.title("🔎 Exploratory Analysis")

    st.subheader("Average review length by sentiment")
    al = pd.DataFrame({"Sentiment": list(R.AVG_REVIEW_LENGTH.keys()),
                       "Avg words": [round(v, 1) for v in R.AVG_REVIEW_LENGTH.values()]})
    figl = px.bar(al, x="Sentiment", y="Avg words", text="Avg words", color="Sentiment",
                  color_discrete_map={"Positive (1)": POS, "Negative (0)": NEG})
    figl.update_traces(textposition="outside")
    figl.update_layout(showlegend=False, height=330, margin=dict(t=10, b=10),
                       yaxis_title="Average words per review")
    st.plotly_chart(figl, use_container_width=True)
    neg = R.AVG_REVIEW_LENGTH["Negative (0)"]
    pos = R.AVG_REVIEW_LENGTH["Positive (1)"]
    st.markdown(
        f"<span class='note'>Negative reviews average <b>{neg} words</b> vs. "
        f"<b>{pos} words</b> for positive ones — unhappy users elaborate on "
        f"complaints, while happy users leave short praise.</span>",
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
        st.subheader("Word cloud of reviews")
        st.image(wc_path, use_container_width=True)


# ======================================================================
# TF-IDF
# ======================================================================
elif section == "🧮 TF-IDF Features":
    st.title("🧮 TF-IDF Feature Analysis")
    st.write(
        "TF-IDF (Term Frequency–Inverse Document Frequency) highlights words "
        "distinctive to reviews while down-weighting words that appear "
        "everywhere. Terms shaded gold were especially discriminative between "
        "positive and negative reviews."
    )
    st.subheader("Top TF-IDF terms")
    chips = []
    for t in R.TFIDF_TOP_FEATURES:
        hot = "hot" if t in R.TFIDF_DISCRIMINATIVE else ""
        chips.append(f"<span class='tag {hot}'>{t}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)
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
            st.dataframe(report_table(R.LR_REPORT), use_container_width=True)
        with c2:
            st.markdown("**Approach**")
            st.markdown(
                "- TF-IDF, up to **5,000 features**\n"
                "- Logistic Regression, **1,000 max iterations**\n"
                f"- Stratified 80/20 split ({R.TRAIN_SIZE:,} train / {R.TEST_SIZE:,} test)\n"
                "- Weighted averaging for metric aggregation"
            )

    # ---- DistilBERT ----
    with tab2:
        st.subheader("Transformer — Fine-Tuned DistilBERT")
        cols = st.columns(2 + len(R.DISTILBERT_METRICS))
        with cols[0]:
            card(f"{R.DISTILBERT_CORRECT:,}", "Correct / 1,300", gold=True)
        with cols[1]:
            card(f"{R.DISTILBERT_MISCLASSIFIED}", "Misclassified")
        for col, (k, v) in zip(cols[2:], R.DISTILBERT_METRICS.items()):
            with col:
                card(f"{v*100:.1f}%", k, gold=True)

        st.write("")
        st.markdown("**Fine-tuning workflow**")
        for i, (name, desc) in enumerate(R.DISTILBERT_WORKFLOW, 1):
            st.markdown(
                f"<div class='step'><b>{i}. {name}</b> — "
                f"<span class='note'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.write("")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Classification report**")
            st.dataframe(report_table(R.DISTILBERT_REPORT), use_container_width=True)
        with c2:
            st.markdown("**Confusion matrix**")
            st.plotly_chart(confusion_fig(R.DISTILBERT_CONFUSION, "Viridis"),
                            use_container_width=True)
        st.success(
            f"DistilBERT correctly classified **{R.DISTILBERT_CORRECT:,} of "
            f"{R.TEST_SIZE:,}** held-out reviews "
            f"(**{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%** accuracy, "
            f"**{R.DISTILBERT_METRICS['F1 (weighted)']*100:.1f}%** weighted F1) — "
            "only 69 misclassifications, with strong symmetry across classes."
        )


# ======================================================================
# COMPARISON
# ======================================================================
elif section == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison")
    st.caption("Classic NLP baseline vs. the fine-tuned transformer.")

    comp = pd.DataFrame(R.MODEL_COMPARISON)
    show = comp.copy()
    for c in ["Accuracy", "F1 (weighted)"]:
        show[c] = show[c].apply(lambda v: f"{v*100:.1f}%")
    st.dataframe(show, use_container_width=True, hide_index=True)

    plot_df = comp.melt(id_vars="Model", value_vars=["Accuracy", "F1 (weighted)"],
                        var_name="Metric", value_name="Score")
    figb = px.bar(plot_df, x="Model", y="Score", color="Metric", barmode="group",
                  text=plot_df["Score"].apply(lambda v: f"{v*100:.1f}%"),
                  color_discrete_sequence=[NAVY, GOLD])
    figb.update_traces(textposition="outside")
    figb.update_layout(height=420, yaxis=dict(range=[0.8, 1.0], tickformat=".0%"),
                       margin=dict(t=30, b=10), yaxis_title="Score")
    st.plotly_chart(figb, use_container_width=True)

    st.success(
        f"**DistilBERT outperforms the baseline** by ~{R.IMPROVEMENT_POINTS} "
        f"percentage points on both accuracy and F1 "
        f"({R.LR_METRICS['Accuracy']*100:.1f}% → "
        f"{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%). Contextual embeddings "
        "capture nuance that the bag-of-words baseline misses."
    )

    st.subheader("DistilBERT — per-class performance")
    cc = st.columns(2)
    for col, (cls, m) in zip(cc, R.DISTILBERT_REPORT.items()):
        with col:
            st.markdown(
                f"<div class='step'><b>{cls} class</b><br>"
                f"<span class='note'>Precision {m['precision']:.2f} · "
                f"Recall {m['recall']:.2f} · F1 {m['f1']:.2f}</span></div>",
                unsafe_allow_html=True,
            )


# ======================================================================
# CONCLUSION
# ======================================================================
elif section == "📌 Conclusion":
    st.title("📌 Results, Limitations & Next Steps")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Key takeaways")
        for t in R.KEY_TAKEAWAYS:
            st.markdown(f"- {t}")
    with c2:
        st.subheader("Limitations & future work")
        for t in R.LIMITATIONS:
            st.markdown(f"- {t}")
    st.divider()
    a, b, c = st.columns(3)
    with a:
        card(f"{R.LR_METRICS['Accuracy']*100:.1f}%", "Baseline accuracy")
    with b:
        card(f"{R.DISTILBERT_METRICS['Accuracy']*100:.1f}%", "DistilBERT accuracy", gold=True)
    with c:
        card(f"+{R.IMPROVEMENT_POINTS} pts", "Transformer gain", gold=True)


# ======================================================================
# LIVE DEMO
# ======================================================================
elif section == "🧪 Try it Live":
    st.title("🧪 Try it Live")
    st.write(
        "Type a product/app review and see how each model family classifies it — "
        "mirroring the project's two approaches: a **classic** TF-IDF + Logistic "
        "Regression model and a **transformer**."
    )
    st.markdown(
        "<span class='note'><b>Note:</b> the classic model is trained live on a "
        "bundled review corpus, and the transformer is an off-the-shelf "
        "<i>pretrained</i> sentiment model "
        "(<code>distilbert-base-uncased-finetuned-sst-2-english</code>) used for "
        "inference. It is a hands-on demo of the two approaches; the project's "
        "own fine-tuned DistilBERT reached "
        f"<b>{R.DISTILBERT_METRICS['Accuracy']*100:.1f}% accuracy</b> on the "
        "task (see Model Results).</span>",
        unsafe_allow_html=True,
    )

    @st.cache_resource(show_spinner="Training the classic TF-IDF + Logistic Regression model…")
    def get_classic_model():
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        path = os.path.join(os.path.dirname(__file__), "data", "product_reviews_dataset.csv")
        df = pd.read_csv(path)
        df = df[df["label"].isin([0, 1])].dropna(subset=["text"])
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ])
        pipe.fit(df["text"].astype(str), df["label"].astype(int))
        return pipe, len(df)

    @st.cache_resource(show_spinner="Loading the pretrained transformer (first run downloads ~250 MB)…")
    def get_transformer():
        from transformers import pipeline
        return pipeline("sentiment-analysis",
                        model="distilbert-base-uncased-finetuned-sst-2-english")

    example = "The app keeps crashing every time I try to checkout. So frustrating."
    text = st.text_area("Your review", value=example, height=110)
    run_tf = st.checkbox(
        "Also run the transformer (downloads & loads the model on first use)",
        value=False,
        help="Leave off for an instant classic-model result. Turn on to compare "
             "with a real transformer — slower on first click while it loads.",
    )

    if st.button("Classify", type="primary") and text.strip():
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Classic · TF-IDF + LogReg")
            try:
                model, n = get_classic_model()
                proba = model.predict_proba([text])[0]
                pred = int(proba.argmax())
                (st.success if pred == 1 else st.error)(
                    f"**{'Positive 😊' if pred == 1 else 'Negative 😠'}**")
                st.metric("Confidence", f"{proba[pred]*100:.1f}%")
                st.caption(f"Trained live on {n:,} bundled labeled reviews.")
            except Exception as e:
                st.warning(f"Classic model unavailable: {e}")
        with col2:
            st.markdown("#### Transformer · DistilBERT (SST-2)")
            if not run_tf:
                st.info("Enable the checkbox above to run the transformer.")
            else:
                try:
                    clf = get_transformer()
                    out = clf(text[:512])[0]
                    pos = out["label"].upper() == "POSITIVE"
                    (st.success if pos else st.error)(
                        f"**{'Positive 😊' if pos else 'Negative 😠'}**")
                    st.metric("Confidence", f"{out['score']*100:.1f}%")
                    st.caption("Pretrained distilbert-base-uncased-finetuned-sst-2-english.")
                except Exception as e:
                    st.warning(
                        "Transformer couldn't load in this environment "
                        f"(often a memory limit on free hosting): {e}")

    st.divider()
    st.caption(
        "Tip: try a long, detailed complaint vs. a short \"love it!\" — the EDA "
        "showed negative reviews tend to be much longer."
    )
