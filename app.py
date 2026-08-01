import streamlit as st
import pickle
import re
import os
from textblob import TextBlob

st.set_page_config(page_title="Fake News Intelligence System", page_icon="📰")

st.title("📰 AI Fake News Intelligence System")
st.write("Enter a news article below to analyze whether it is REAL or FAKE.")

# Load model and vectorizer
try:
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\\S+", "", text)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    return text


clickbait_words = [
    "breaking", "shocking", "secret",
    "revealed", "viral", "truth"
]


def clickbait_score(text):
    score = 0
    text = text.lower()
    for word in clickbait_words:
        if word in text:
            score += 1
    return score


news = st.text_area("Paste News Article")

if st.button("Analyze"):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:

        cleaned = clean_text(news)

        vector = tfidf.transform([cleaned])

        prediction = model.predict(vector)[0]

        sentiment = TextBlob(news).sentiment.polarity

        clickbait = clickbait_score(news)

        st.subheader("Prediction")
        st.success(prediction)

        st.subheader("Analysis")

        st.write("Sentiment Score:", round(sentiment, 2))
        st.write("Clickbait Score:", clickbait)

        if clickbait >= 3:
            st.error("Risk Level: HIGH")
        elif clickbait == 2:
            st.warning("Risk Level: MEDIUM")
        else:
            st.success("Risk Level: LOW") 