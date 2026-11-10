import html
import json
import os
import re

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from scipy.sparse import csr_matrix, hstack

from core import prep, style_features, style_vector, red_flags, LABELS

st.set_page_config(page_title="CredibilityLens", page_icon="🔎", layout="wide")
st.markdown(
    """<style>
    .block-container{max-width:1100px;padding-top:2rem}
    .hl{line-height:2;font-size:1.02rem}
    .hl span{padding:1px 3px;border-radius:4px}
    .bar{height:16px;border-radius:8px;background:linear-gradient(90deg,#e45756,#f2cf5b,#54a24b);position:relative;margin:6px 0 2px}
    .pin{position:absolute;top:-5px;width:4px;height:26px;background:#111;border-radius:2px}
    .card{border:1px solid #8884;border-radius:12px;padding:14px 18px;margin-bottom:10px}
    </style>""",
    unsafe_allow_html=True,
)

def ensure_model():
    if os.path.exists("bundle.joblib"):
        return
    with st.spinner("First-time setup: downloading the news dataset and training the CredibilityLens model. This may take a few minutes..."):
        try:
            from train import main as train_model
            train_model(None)
        except Exception as exc:
            st.error("Automatic model setup failed. Check the deployment logs and dataset access.")
            st.exception(exc)
            st.stop()

@st.cache_resource
def load_model():
    return joblib.load("bundle.joblib")

ensure_model()
bundle = load_model()
model = bundle["model"]
tfidf = bundle["tfidf"]
scaler = bundle["scaler"]
baseline = bundle["baseline"]
style_weight = bundle["style_w"]
real_class_index = list(model.classes_).index("REAL")

def analyze(text):
    text_matrix = tfidf.transform([prep(text)])
    style_matrix = csr_matrix(scaler.transform([style_vector(text)]) * style_weight)
    probability_real = model.predict_proba(hstack([text_matrix, style_matrix]))[0][real_class_index]

    coefficients = model.coef_[0]
    contributions = {}
    sparse_row = text_matrix.tocoo()
    inverse_vocab = {index: word for word, index in tfidf.vocabulary_.items()}
    for index, value in zip(sparse_row.col, sparse_row.data):
        contributions[inverse_vocab[index]] = float(coefficients[index] * value)
    return probability_real, contributions, style_features(text)

def highlight_text(text, contributions):
    scale = max([abs(value) for value in contributions.values()] + [1e-9])
    output = []
    for token in re.findall(r"\s+|[^\s]+", text):
        word = re.sub(r"[^a-z]", "", token.lower())
        contribution = contributions.get(word)
        if contribution is not None and abs(contribution) / scale > 0.12:
            opacity = min(0.15 + 0.7 * abs(contribution) / scale, 0.85)
            color = f"rgba(84,162,75,{opacity})" if contribution > 0 else f"rgba(228,87,86,{opacity})"
            output.append(f'<span style="background:{color}">{html.escape(token)}</span>')
        else:
            output.append(html.escape(token))
    return "".join(output)

SAMPLES = {
    "Sensational": "BREAKING!!! SHOCKING secret REVEALED: Doctors HATE this miracle cure the government doesn't want you to know about! Share before they delete it!",
    "Measured report": "The education ministry said on Tuesday that it will expand digital classrooms to 2,000 rural schools over the next two years. According to a statement, the programme will be funded from the existing budget, officials told reporters.",
}

st.title("🔎 CredibilityLens")
st.caption("Not just *fake or real* — **why**, which writing red flags appear, and whether the model can be trusted.")
analyze_tab, results_tab, about_tab = st.tabs(["Analyze", "Research results", "About"])

with analyze_tab:
    columns = st.columns(len(SAMPLES) + 1)
    for index, (name, sample) in enumerate(SAMPLES.items()):
        if columns[index].button(f"Try: {name}"):
            st.session_state.news_text = sample

    news_text = st.text_area("Paste a news article or headline", key="news_text", height=170)
    if st.button("Analyze", type="primary"):
        if len(news_text.split()) < 8:
            st.warning("Please paste at least a couple of sentences.")
        else:
            probability_real, contributions, features = analyze(news_text)
            score = round(probability_real * 100)
            verdict = (
                "✅ Likely credible" if score >= 65 else
                "🚩 Likely fake / misleading" if score <= 35 else
                "⚠️ Uncertain — verify with other sources"
            )
            left, right = st.columns([1, 2])
            with left:
                st.metric("Credibility score", f"{score} / 100")
                st.markdown(
                    f'<div class="bar"><div class="pin" style="left:{score}%"></div></div>',
                    unsafe_allow_html=True,
                )
                st.subheader(verdict)
                if len(news_text.split()) < 40:
                    st.info("Short text: predictions are less reliable.")
            with right:
                st.markdown("**Writing red flags**")
                flags = red_flags(features, baseline)
                if not flags:
                    st.success("No unusual writing patterns compared with real news.")
                for name, message in flags:
                    st.markdown(
                        f'<div class="card">🚩 <b>{html.escape(name)}</b> — {html.escape(message)}</div>',
                        unsafe_allow_html=True,
                    )

            st.markdown(
                "**Why the model decided this** — "
                "<span style='background:rgba(228,87,86,.5)'>pushes toward FAKE</span> · "
                "<span style='background:rgba(84,162,75,.5)'>pushes toward REAL</span>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="hl card">{highlight_text(news_text, contributions)}</div>',
                unsafe_allow_html=True,
            )

            ranked = sorted(contributions.items(), key=lambda item: item[1])
            fake_column, real_column = st.columns(2)
            fake_column.write("Top FAKE cues")
            fake_column.write(", ".join(word for word, value in ranked[:8] if value < 0) or "–")
            real_column.write("Top REAL cues")
            real_column.write(", ".join(word for word, value in ranked[::-1][:8] if value > 0) or "–")

            st.markdown("**Your text vs typical real news**")
            chart_data = pd.DataFrame(
                {
                    "Your text": [features[key] for key in LABELS],
                    "Typical real news": [baseline[key]["median"] for key in LABELS],
                },
                index=list(LABELS.values()),
            )
            st.bar_chart(chart_data, horizontal=True)
            st.caption(
                "This estimates writing patterns learned from labelled data. "
                "It cannot verify facts — always check original sources."
            )

with results_tab:
    if os.path.exists("results.json"):
        with open("results.json", encoding="utf-8") as file:
            results = json.load(file)
        st.dataframe(pd.DataFrame(results["table"]), hide_index=True, use_container_width=True)
        st.metric(
            "Shortcut gap (hybrid raw accuracy minus hybrid masked accuracy)",
            f'{results["shortcut_gap"]} percentage points',
        )
        if os.path.exists("ablation.png"):
            st.image("ablation.png", caption="Five-experiment accuracy comparison")
    else:
        st.info("Run train.py to generate the experimental results.")

with about_tab:
    st.markdown(
        """
**Pipeline:** text cleaning and artifact masking → TF-IDF + ten writing-style features
→ class-balanced Logistic Regression → credibility score, word-level cues, red flags.

**Experimental contribution:** five controlled feature configurations compare style-only,
raw text, raw hybrid, masked text, and masked hybrid models to investigate reliance on
source and formatting shortcuts.

**Important limitation:** CredibilityLens estimates similarity to patterns learned from
labelled data. It does not independently verify facts or prove that a news article is true.
"""
    )
