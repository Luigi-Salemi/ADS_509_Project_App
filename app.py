"""
ADS-509 Final Project — Results Dashboard
=========================================
Interactive Streamlit app presenting the results of
"Traditional NLP vs. Transformers for Sentiment Classification"
(team notebook `ADS_509_Final_Team_Project_.ipynb`), plus a live demo.

Dark theme using BlueprintJS v6 colors (gray base, blue primary) + Lucide icons.
Uses Streamlit 1.4x+ features: st.navigation/st.Page, @st.fragment, st.pills,
st.popover, st.badge, and width="stretch".

Run locally:   streamlit run app.py
"""

import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

import results as R

# ----------------------------------------------------------------------
# BlueprintJS v6 palette (dark)
# ----------------------------------------------------------------------
BG       = "#1C2127"
PANEL    = "#252A31"
ELEV     = "#2F343C"
BORDER   = "#383E47"
BORDER2  = "#404854"
TEXT     = "#F6F7F9"
SUBTLE   = "#ABB3BF"
MUTE     = "#738091"
PRIMARY  = "#4C90F0"
PRIMARY2 = "#8ABBFF"
GOLD     = "#F0B726"
POS      = "#32A467"
NEG      = "#E76A6E"
NEU      = "#8F99A8"

st.set_page_config(
    page_title="ADS-509 — Sentiment Model Comparison",
    page_icon="📊", layout="wide", initial_sidebar_state="expanded",
)
pio.templates.default = "plotly_dark"

# DistilBERT accuracy gain over the baseline, computed from the figures
GAIN = round((R.DISTILBERT_METRICS["Accuracy"] - R.LR_METRICS["Accuracy"]) * 100, 1)

# ----------------------------------------------------------------------
# Lucide icons (inline SVG, MIT licensed)
# ----------------------------------------------------------------------
_ICONS = {
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5V19A9 3 0 0 0 21 19V5"/><path d="M3 12A9 3 0 0 0 21 12"/>',
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "calculator": '<rect width="16" height="20" x="4" y="2" rx="2"/><line x1="8" x2="16" y1="6" y2="6"/><line x1="16" x2="16" y1="14" y2="18"/><path d="M16 10h.01"/><path d="M12 10h.01"/><path d="M8 10h.01"/><path d="M12 14h.01"/><path d="M8 14h.01"/><path d="M12 18h.01"/><path d="M8 18h.01"/>',
    "bot": '<path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/>',
    "scale": '<path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h2c2 0 5-1 7-2 2 1 5 2 7 2h2"/>',
    "flag": '<path d="M4 15s1-1 4-1 5 2 8 2 4-1 4-1V3s-1 1-4 1-5-2-8-2-4 1-4 1z"/><line x1="4" x2="4" y1="22" y2="15"/>',
    "flask": '<path d="M14 2v6a2 2 0 0 0 .245.96l5.51 10.08A2 2 0 0 1 18 22H6a2 2 0 0 1-1.755-2.96l5.51-10.08A2 2 0 0 0 10 8V2"/><path d="M6.453 15h11.094"/><path d="M8.5 2h7"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "sparkles": '<path d="M9.937 15.5A2 2 0 0 0 8.5 14.063l-6.135-1.582a.5.5 0 0 1 0-.962L8.5 9.936A2 2 0 0 0 9.937 8.5l1.582-6.135a.5.5 0 0 1 .963 0L14.063 8.5A2 2 0 0 0 15.5 9.937l6.135 1.581a.5.5 0 0 1 0 .964L15.5 14.063a2 2 0 0 0-1.437 1.437l-1.582 6.135a.5.5 0 0 1-.963 0z"/>',
    "users": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "list": '<path d="M8 6h13"/><path d="M8 12h13"/><path d="M8 18h13"/><path d="M3 6h.01"/><path d="M3 12h.01"/><path d="M3 18h.01"/>',
    "alert": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>',
    "info": '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4"/><path d="M12 8h.01"/>',
    "thumbsUp": '<path d="M7 10v12"/><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z"/>',
    "thumbsDown": '<path d="M17 14V2"/><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z"/>',
    "layers": '<path d="M12.83 2.18a2 2 0 0 0-1.66 0L2.6 6.08a1 1 0 0 0 0 1.83l8.58 3.91a2 2 0 0 0 1.66 0l8.58-3.9a1 1 0 0 0 0-1.83Z"/><path d="m22 17.65-9.17 4.16a2 2 0 0 1-1.66 0L2 17.65"/><path d="m22 12.65-9.17 4.16a2 2 0 0 1-1.66 0L2 12.65"/>',
    "factcheck": '<path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/>',
}


def icon(name, size=22, color=PRIMARY, sw=2.0, mr=10):
    return (
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{size}' height='{size}' "
        f"viewBox='0 0 24 24' fill='none' stroke='{color}' stroke-width='{sw}' "
        f"stroke-linecap='round' stroke-linejoin='round' "
        f"style='vertical-align:middle;margin-right:{mr}px'>{_ICONS[name]}</svg>"
    )


# ----------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------
st.markdown(
    f"""
    <style>
      .stApp {{ background:{BG}; }}
      .main .block-container {{ padding-top: 2rem; max-width: 1180px; }}
      h1, h2, h3, h4 {{ color:{TEXT}; }}
      .metric-card {{ background:{PANEL}; border:1px solid {BORDER}; border-left:5px solid {PRIMARY};
        border-radius:10px; padding:16px 18px; }}
      .metric-card .v {{ font-size:30px; font-weight:700; line-height:1.1; }}
      .metric-card .l {{ font-size:12px; color:{SUBTLE}; text-transform:uppercase; letter-spacing:.05em; margin-top:4px; }}
      .tag {{ display:inline-block; background:{PRIMARY}1f; color:{PRIMARY2}; border:1px solid {PRIMARY}55;
        border-radius:999px; padding:4px 12px; margin:4px 6px 4px 0; font-size:13px; font-weight:600; }}
      .tag.hot {{ background:{GOLD}1f; color:{GOLD}; border-color:{GOLD}66; }}
      .note {{ color:{SUBTLE}; font-size:13px; }}
      .pipe {{ background:{PANEL}; border:1px solid {BORDER}; border-radius:12px; padding:14px; text-align:center; height:150px; }}
      .pipe-n {{ display:inline-flex; align-items:center; justify-content:center; width:30px; height:30px;
        border-radius:50%; background:{PRIMARY}; color:{BG}; font-weight:700; margin-bottom:8px; }}
      .pipe-t {{ font-weight:700; color:{TEXT}; font-size:15px; }}
      .pipe-d {{ font-size:12px; color:{SUBTLE}; margin-top:6px; }}
      .step {{ border:1px solid {BORDER}; border-left:4px solid {PRIMARY}; background:{PANEL};
        border-radius:0 10px 10px 0; padding:12px 16px; margin-bottom:10px; }}
      .step b {{ color:{TEXT}; }}
      section[data-testid="stSidebar"] {{ border-right:1px solid {BORDER}; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def pct(v):
    return f"{v*100:.1f}%" if v is not None else "—"


def card(value, label, accent=PRIMARY, icon_name=None, highlight=False):
    ic = (f"<div style='float:right;opacity:.9'>{icon(icon_name, 22, accent, 2, 0)}</div>"
          if icon_name else "")
    vcolor = accent if highlight else TEXT
    st.markdown(
        f"<div class='metric-card' style='border-left-color:{accent}'>{ic}"
        f"<div class='v' style='color:{vcolor}'>{value}</div>"
        f"<div class='l'>{label}</div></div>",
        unsafe_allow_html=True,
    )


def panel(icon_name, html, accent=PRIMARY):
    st.markdown(
        f"<div style='background:{accent}14; border:1px solid {accent}44; border-left:4px solid {accent}; "
        f"border-radius:8px; padding:14px 16px; display:flex; gap:12px; align-items:flex-start; margin:6px 0'>"
        f"<div style='flex:0 0 auto; margin-top:1px'>{icon(icon_name, 20, accent, 2, 0)}</div>"
        f"<div style='color:{TEXT}; font-size:14px; line-height:1.5'>{html}</div></div>",
        unsafe_allow_html=True,
    )


def header(icon_name, title):
    st.markdown(
        f"<h1 style='display:flex; align-items:center; gap:4px'>"
        f"{icon(icon_name, 30, PRIMARY, 2.2)}<span>{title}</span></h1>",
        unsafe_allow_html=True,
    )


def style_fig(fig, h=None):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=SUBTLE, family="sans-serif"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=SUBTLE)),
        margin=dict(t=20, b=10, l=10, r=10),
    )
    fig.update_xaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER2, color=SUBTLE)
    fig.update_yaxes(gridcolor=BORDER, zerolinecolor=BORDER, linecolor=BORDER2, color=SUBTLE)
    if h:
        fig.update_layout(height=h)
    return fig


def report_table(report: dict):
    rep = pd.DataFrame(report).T.rename(columns={"f1": "f1-score"})
    rep["support"] = rep["support"].astype(int)
    return rep.style.format({"precision": "{:.2f}", "recall": "{:.2f}",
                             "f1-score": "{:.2f}", "support": "{:d}"})


def confusion_fig(matrix):
    labels = ["Negative", "Positive"]
    fig = go.Figure(go.Heatmap(
        z=matrix, x=labels, y=labels, text=matrix, texttemplate="%{text}",
        textfont=dict(size=22, color=TEXT),
        colorscale=[[0, PANEL], [0.5, "#2D72D2"], [1, PRIMARY]],
        showscale=False, xgap=4, ygap=4,
    ))
    fig.update_layout(height=330, xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"))
    return style_fig(fig)


def result_badge(is_pos, conf, caption):
    color = POS if is_pos else NEG
    ic = icon("thumbsUp" if is_pos else "thumbsDown", 24, color, 2.2, 8)
    label = "Positive" if is_pos else "Negative"
    st.markdown(
        f"<div style='background:{color}1f; border:1px solid {color}55; border-radius:10px; padding:14px 16px'>"
        f"<div style='display:flex; align-items:center; font-size:20px; font-weight:700; color:{color}'>{ic}{label}</div>"
        f"<div style='color:{SUBTLE}; font-size:13px; margin-top:8px'>Confidence: "
        f"<b style='color:{TEXT}'>{conf*100:.1f}%</b></div>"
        f"<div style='color:{MUTE}; font-size:12px; margin-top:4px'>{caption}</div></div>",
        unsafe_allow_html=True,
    )


# ======================================================================
# PAGES
# ======================================================================
def page_overview():
    st.markdown(f"<h1 style='line-height:1.15'>{R.PROJECT['title']}</h1>", unsafe_allow_html=True)
    st.markdown(f"**{R.PROJECT['course']} · {R.PROJECT['school']}**")
    st.markdown("##### " + " · ".join(R.PROJECT["team"]))
    st.write("")
    panel("target", "<b>Objective</b> — " + R.PROJECT["objective"])

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        card(f"{R.RAW_REVIEW_COUNT:,}", "Reviews scraped", PRIMARY, "database")
    with c2:
        card(f"{R.BALANCED_TOTAL:,}", "Balanced sample", PRIMARY, "layers")
    with c3:
        card(pct(R.DISTILBERT_METRICS["Accuracy"]), "Best accuracy · DistilBERT",
             GOLD, "bot", highlight=True)
    with c4:
        card(f"{len(R.SOURCE_APPS)}", "App sources", PRIMARY, "users")

    st.write("")
    st.subheader("Project at a glance")
    card_icons = ["database", "search", "scale", "sparkles"]
    cols = st.columns(2) + st.columns(2)
    for col, (title, desc), ic in zip(cols, R.OVERVIEW_CARDS, card_icons):
        with col:
            st.markdown(
                f"<div class='step'><b>{icon(ic, 18, PRIMARY, 2, 8)}{title}</b><br>"
                f"<span class='note'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

    st.write("")
    with st.popover("Data provenance & cross-check", icon=":material/fact_check:"):
        st.markdown("**Executed notebook vs. final presentation**")
        cc = pd.DataFrame(R.CROSSCHECK, columns=["Figure", "Executed notebook", "Presentation"])
        st.dataframe(cc, width="stretch", hide_index=True)
        st.caption(R.PROVENANCE)


def page_dataset():
    header("database", "Dataset")
    st.caption(
        f"{R.RAW_REVIEW_COUNT:,} reviews ({R.REVIEWS_PER_APP:,} per app) scraped from the "
        f"{R.SOURCE_PLATFORM} via {R.SCRAPER_LIB}: " + ", ".join(R.SOURCE_APPS)
    )

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Star-rating distribution")
        rd = pd.DataFrame({"Stars": list(R.RATING_DISTRIBUTION.keys()),
                           "Reviews": list(R.RATING_DISTRIBUTION.values())}).sort_values("Stars")
        rd["color"] = rd["Stars"].apply(lambda s: NEG if s <= 2 else (NEU if s == 3 else POS))
        fig = px.bar(rd, x="Stars", y="Reviews", text="Reviews")
        fig.update_traces(marker_color=rd["color"], textposition="outside", textfont_color=TEXT)
        fig.update_layout(xaxis_title="Star rating", yaxis_title="Reviews", showlegend=False)
        st.plotly_chart(style_fig(fig, 360), width="stretch")
    with right:
        st.subheader("Balanced sample")
        labels = list(R.LABEL_BALANCE.keys())
        fig2 = go.Figure(go.Pie(
            labels=labels, values=list(R.LABEL_BALANCE.values()), hole=0.55,
            marker=dict(colors=[POS if "Positive" in l else NEG for l in labels]),
            textfont=dict(color=TEXT)))
        fig2.update_layout(legend=dict(orientation="h", y=-0.1))
        st.plotly_chart(style_fig(fig2, 280), width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            card(f"{R.UNIQUE_REVIEWS:,}", "Unique reviews", PRIMARY, "layers")
        with c2:
            card(f"{R.DUPLICATE_REVIEWS:,}", "Duplicates", PRIMARY, "list")

    st.divider()
    st.subheader("Text preprocessing pipeline")
    steps = R.PIPELINE_STEPS
    cols = st.columns(len(steps) * 2 - 1)
    for i, (name, desc) in enumerate(steps):
        with cols[i * 2]:
            st.markdown(
                f"<div class='pipe'><div class='pipe-n'>{i+1}</div>"
                f"<div class='pipe-t'>{name}</div><div class='pipe-d'>{desc}</div></div>",
                unsafe_allow_html=True,
            )
        if i < len(steps) - 1:
            with cols[i * 2 + 1]:
                st.markdown(
                    f"<div style='text-align:center; padding-top:54px; color:{GOLD}; font-size:26px'>→</div>",
                    unsafe_allow_html=True,
                )
    st.caption(
        f"Star ratings mapped to binary labels (4-5 → positive, 1-2 → negative, 3 dropped); "
        f"balanced to {R.BALANCED_TOTAL:,} samples, split 80/20 "
        f"→ {R.TRAIN_SIZE:,} train / {R.TEST_SIZE:,} test."
    )


def page_eda():
    header("search", "Exploratory Analysis")
    st.subheader("Average review length by sentiment")
    sent = list(R.AVG_REVIEW_LENGTH.keys())
    vals = [round(v, 1) for v in R.AVG_REVIEW_LENGTH.values()]
    figl = go.Figure(go.Bar(
        x=sent, y=vals, text=vals, textposition="outside",
        marker_color=[NEG if "Negative" in s else POS for s in sent],
        textfont=dict(color=TEXT)))
    figl.update_layout(showlegend=False, yaxis_title="Average words per review")
    st.plotly_chart(style_fig(figl, 330), width="stretch")
    neg = R.AVG_REVIEW_LENGTH["Negative (0)"]
    pos = R.AVG_REVIEW_LENGTH["Positive (1)"]
    panel("alert",
          f"Negative reviews average <b>{neg} words</b> vs. <b>{pos} words</b> for positive "
          "ones — unhappy users elaborate on complaints, while happy users leave short praise.",
          GOLD)

    st.divider()
    st.subheader("Top 20 most frequent words")
    tw = pd.DataFrame(R.TOP_WORDS, columns=["Word", "Frequency"]).sort_values("Frequency")
    figw = px.bar(tw, x="Frequency", y="Word", orientation="h", text="Frequency")
    figw.update_traces(marker_color=PRIMARY, textposition="outside", textfont_color=SUBTLE)
    figw.update_layout(yaxis_title="", xaxis_title="Frequency (stopwords removed)")
    st.plotly_chart(style_fig(figw, 560), width="stretch")

    wc_path = os.path.join(os.path.dirname(__file__), "assets", "wordcloud.png")
    if os.path.exists(wc_path):
        st.subheader("Word cloud of reviews")
        st.image(wc_path, width="stretch")


def page_tfidf():
    header("calculator", "TF-IDF Feature Analysis")
    st.write(
        "TF-IDF (Term Frequency–Inverse Document Frequency) highlights words distinctive to "
        "reviews while down-weighting words that appear everywhere. Terms shaded gold were "
        "especially discriminative between positive and negative reviews."
    )
    st.subheader("Top TF-IDF terms")
    chips = []
    for t in R.TFIDF_TOP_FEATURES:
        hot = "hot" if t in R.TFIDF_DISCRIMINATIVE else ""
        chips.append(f"<span class='tag {hot}'>{t}</span>")
    st.markdown("".join(chips), unsafe_allow_html=True)
    st.write("")
    panel("calculator",
          "Brand/app terms (<b>amazon</b>, <b>ebay</b>, <b>best buy</b>), action terms "
          "(<b>order</b>, <b>search</b>, <b>delivery</b>), and sentiment cues (<b>love</b>, "
          "<b>good</b>, <b>great</b>, <b>slow</b>) dominate — exactly the signal a linear "
          "classifier can exploit.")


def page_models():
    header("bot", "Model Results")
    tab1, tab2 = st.tabs(["TF-IDF + Logistic Regression", "Fine-Tuned DistilBERT"])

    with tab1:
        st.subheader("Baseline — TF-IDF + Logistic Regression")
        cols = st.columns(len(R.LR_METRICS))
        for col, (k, v) in zip(cols, R.LR_METRICS.items()):
            with col:
                card(pct(v), k)
        st.write("")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Classification report**")
            st.dataframe(report_table(R.LR_REPORT), width="stretch")
        with c2:
            st.markdown("**Approach**")
            st.markdown(
                "- TF-IDF, up to **5,000 features**\n"
                "- Logistic Regression, **1,000 max iterations**\n"
                f"- Stratified 80/20 split ({R.TRAIN_SIZE:,} train / {R.TEST_SIZE:,} test)\n"
                "- Weighted averaging for metric aggregation"
            )

    with tab2:
        c = st.container()
        c.subheader("Transformer — Fine-Tuned DistilBERT")
        st.badge("Best model", color="green", icon=":material/trophy:")
        cols = st.columns(2 + len(R.DISTILBERT_METRICS))
        with cols[0]:
            card(f"{R.DISTILBERT_CORRECT:,}", "Correct / 1,300", GOLD, "check", highlight=True)
        with cols[1]:
            card(f"{R.DISTILBERT_MISCLASSIFIED}", "Misclassified", PRIMARY)
        for col, (k, v) in zip(cols[2:], R.DISTILBERT_METRICS.items()):
            with col:
                card(pct(v), k, GOLD, highlight=True)
        panel("info", R.DISTILBERT_SOURCE_NOTE, PRIMARY)

        st.markdown("**Fine-tuning workflow**")
        for i, (name, desc) in enumerate(R.DISTILBERT_WORKFLOW, 1):
            st.markdown(
                f"<div class='step'><b>{i}. {name}</b> — <span class='note'>{desc}</span></div>",
                unsafe_allow_html=True,
            )

        st.write("")
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("**Classification report**")
            st.dataframe(report_table(R.DISTILBERT_REPORT), width="stretch")
        with c2:
            st.markdown("**Confusion matrix**")
            st.plotly_chart(confusion_fig(R.DISTILBERT_CONFUSION), width="stretch")
        panel("check",
              f"DistilBERT correctly classified <b>{R.DISTILBERT_CORRECT:,} of {R.TEST_SIZE:,}</b> "
              f"held-out reviews (<b>{pct(R.DISTILBERT_METRICS['Accuracy'])}</b> accuracy, "
              f"<b>{pct(R.DISTILBERT_METRICS['F1 (weighted)'])}</b> weighted F1) — only "
              f"{R.DISTILBERT_MISCLASSIFIED} misclassifications, with strong symmetry across classes.",
              POS)


def page_comparison():
    header("scale", "Model Comparison")
    st.caption("Classic NLP baseline vs. the fine-tuned transformer.")
    comp = pd.DataFrame(R.MODEL_COMPARISON)
    show = comp.copy()
    for c in ["Accuracy", "F1 (weighted)"]:
        show[c] = show[c].apply(pct)
    st.dataframe(show, width="stretch", hide_index=True)

    models = comp["Model"].tolist()
    figb = go.Figure()
    figb.add_bar(name="Accuracy", x=models, y=comp["Accuracy"], marker_color=PRIMARY,
                 text=[pct(v) for v in comp["Accuracy"]], textposition="outside")
    figb.add_bar(name="F1 (weighted)", x=models, y=comp["F1 (weighted)"], marker_color=GOLD,
                 text=[pct(v) for v in comp["F1 (weighted)"]], textposition="outside")
    figb.update_traces(textfont_color=TEXT)
    figb.update_layout(barmode="group", yaxis=dict(range=[0.8, 1.0], tickformat=".0%"),
                       yaxis_title="Score")
    st.plotly_chart(style_fig(figb, 420), width="stretch")

    panel("check",
          f"<b>DistilBERT outperforms the baseline</b> by ~{GAIN} percentage points on both "
          f"accuracy and F1 ({pct(R.LR_METRICS['Accuracy'])} → "
          f"{pct(R.DISTILBERT_METRICS['Accuracy'])}). Contextual embeddings capture nuance "
          f"that the bag-of-words baseline misses. <span style='color:{MUTE}'>(The team's "
          f"presentation reports ~{R.PRESENTATION_GAIN} pts using its own baseline run.)</span>",
          POS)

    st.subheader("DistilBERT — per-class performance")
    cc = st.columns(2)
    for col, (cls, m) in zip(cc, R.DISTILBERT_REPORT.items()):
        with col:
            st.markdown(
                f"<div class='step'><b>{cls} class</b><br><span class='note'>"
                f"Precision {m['precision']:.2f} · Recall {m['recall']:.2f} · "
                f"F1 {m['f1']:.2f}</span></div>",
                unsafe_allow_html=True,
            )


def page_conclusion():
    header("flag", "Results, Limitations & Next Steps")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"{icon('check', 20, POS)}<b style='color:{TEXT}; font-size:18px'>Key takeaways</b>",
                    unsafe_allow_html=True)
        for t in R.KEY_TAKEAWAYS:
            st.markdown(f"- {t}")
    with c2:
        st.markdown(f"{icon('list', 20, GOLD)}<b style='color:{TEXT}; font-size:18px'>Limitations & future work</b>",
                    unsafe_allow_html=True)
        for t in R.LIMITATIONS:
            st.markdown(f"- {t}")
    st.divider()
    a, b, c = st.columns(3)
    with a:
        card(pct(R.LR_METRICS["Accuracy"]), "Baseline accuracy", PRIMARY, "calculator")
    with b:
        card(pct(R.DISTILBERT_METRICS["Accuracy"]), "DistilBERT accuracy", GOLD, "bot", highlight=True)
    with c:
        card(f"+{GAIN} pts", "Transformer gain", GOLD, "scale", highlight=True)


# ---- live demo (uses @st.fragment so classifying reruns only this block) ----
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


@st.fragment
def live_demo_runner():
    if "review_text" not in st.session_state:
        st.session_state["review_text"] = R.EXAMPLE_REVIEWS[R.DEFAULT_EXAMPLE]
    pick = st.pills("Try an example product review", list(R.EXAMPLE_REVIEWS.keys()),
                    selection_mode="single", default=None, key="ex_pick")
    if pick and pick != st.session_state.get("_last_pick"):
        st.session_state["review_text"] = R.EXAMPLE_REVIEWS[pick]
        st.session_state["_last_pick"] = pick

    st.text_area("Your review", key="review_text", height=110)
    run_tf = st.checkbox(
        "Also run the transformer (downloads & loads the model on first use)", value=False,
        help="Leave off for an instant classic-model result. Turn on to compare with a real "
             "transformer — slower on first click while it loads.",
    )

    if st.button("Classify", type="primary"):
        text = st.session_state["review_text"]
        if not text.strip():
            st.warning("Enter a review first.")
            return
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"{icon('calculator', 20, PRIMARY)}<b style='color:{TEXT}'>Classic · TF-IDF + LogReg</b>",
                        unsafe_allow_html=True)
            try:
                model, n = get_classic_model()
                proba = model.predict_proba([text])[0]
                pred = int(proba.argmax())
                result_badge(pred == 1, proba[pred], f"Trained live on {n:,} bundled product reviews.")
            except Exception as e:
                panel("alert", f"Classic model unavailable: {e}", NEG)
        with col2:
            st.markdown(f"{icon('bot', 20, PRIMARY)}<b style='color:{TEXT}'>Transformer · DistilBERT (SST-2)</b>",
                        unsafe_allow_html=True)
            if not run_tf:
                panel("info", "Enable the checkbox above to run the transformer.")
            else:
                try:
                    clf = get_transformer()
                    out = clf(text[:512])[0]
                    is_pos = out["label"].upper() == "POSITIVE"
                    result_badge(is_pos, out["score"],
                                 "Pretrained distilbert-base-uncased-finetuned-sst-2-english.")
                except Exception as e:
                    panel("alert",
                          f"Transformer couldn't load in this environment (often a memory "
                          f"limit on free hosting): {e}", NEG)


def page_live():
    header("flask", "Try it Live")
    st.write(
        "Type a product review and see how each model family classifies it — mirroring the "
        "project's two approaches: a **classic** TF-IDF + Logistic Regression model and a "
        "**transformer**."
    )
    with st.popover("Which models is this using?", icon=":material/help:"):
        st.markdown(
            "**This demo does _not_ load the team's fine-tuned DistilBERT weights.** "
            "Those were trained inside the project notebook on Colab and never exported, "
            "so they aren't stored in the app.\n\n"
            "- **Classic** — a TF-IDF + Logistic Regression model trained *live* on a bundled "
            "product-review corpus when you open this page.\n"
            "- **Transformer** — an off-the-shelf *pretrained* sentiment model "
            "(`distilbert-base-uncased-finetuned-sst-2-english`), used only for inference.\n\n"
            f"The project's own fine-tuned DistilBERT scored **{pct(R.DISTILBERT_METRICS['Accuracy'])}** "
            "on the task — see the **Model Results** page."
        )
    live_demo_runner()
    st.divider()
    st.caption(
        "Tip: try a long, detailed complaint vs. a short \"love it!\" — the EDA showed "
        "negative reviews tend to be much longer."
    )


# ======================================================================
# NAVIGATION (st.navigation / st.Page — modern multipage API)
# ======================================================================
PAGES = [
    ("overview", "Overview", page_overview, True),
    ("dataset", "Dataset", page_dataset, False),
    ("eda", "Exploratory Analysis", page_eda, False),
    ("tfidf", "TF-IDF Features", page_tfidf, False),
    ("models", "Model Results", page_models, False),
    ("comparison", "Model Comparison", page_comparison, False),
    ("conclusion", "Conclusion", page_conclusion, False),
    ("live", "Try it Live", page_live, False),
]

# smoke-test hook: APP_SMOKE=<url_path> runs a single page standalone (no nav)
_smoke = os.environ.get("APP_SMOKE")
if _smoke:
    {u: f for u, _, f, _ in PAGES}[_smoke]()
else:
    nav = st.navigation(
        [st.Page(f, title=t, url_path=u, default=d) for u, t, f, d in PAGES],
        position="sidebar",
    )
    with st.sidebar:
        st.markdown(
            f"<div style='display:flex; align-items:center; gap:6px; font-size:16px; font-weight:700; "
            f"color:{TEXT}; margin-bottom:2px'>{icon('chart', 20, PRIMARY)}ADS-509 Final Project</div>",
            unsafe_allow_html=True,
        )
        st.caption(R.PROJECT["course"] + " · " + R.PROJECT["school"])
        st.divider()
        st.markdown(f"{icon('users', 16, SUBTLE, 2, 6)}<b style='color:{TEXT}'>Team</b>",
                    unsafe_allow_html=True)
        for m in R.PROJECT["team"]:
            st.caption("• " + m)
    nav.run()
