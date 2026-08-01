import streamlit as st
import pickle
import re
from textblob import TextBlob


# ==============================
# PAGE CONFIG
# ==============================

st.set_page_config(
    page_title="AI Fake News Intelligence System",
    page_icon="📰",
    layout="wide"
)


# ==============================
# CUSTOM CSS
# ==============================

st.markdown(
"""
<style>

body{
    background:#f8fafc;
}


.hero{

    background:linear-gradient(
        135deg,
        #2563eb,
        #7c3aed
    );

    padding:35px;
    border-radius:25px;
    color:white;
    text-align:center;
}


.hero h1{

    font-size:45px;
    margin-bottom:10px;

}


.hero p{

    font-size:18px;

}



.card{

    background:white;
    padding:25px;
    border-radius:20px;
    text-align:center;
    box-shadow:
    0px 5px 20px rgba(0,0,0,0.1);

}



.stButton button{

    width:100%;
    height:50px;
    border-radius:12px;
    background:#2563eb;
    color:white;
    font-size:18px;
    font-weight:bold;

}



.stTextArea textarea{

    border-radius:15px;

}


</style>

""",
unsafe_allow_html=True
)



# ==============================
# HEADER
# ==============================


st.markdown(
"""
<div class="hero">

<h1>
📰 AI Fake News Intelligence System
</h1>

<p>
Machine Learning + NLP based News Credibility Analyzer
</p>

</div>

""",
unsafe_allow_html=True
)


st.write("")



# ==============================
# LOAD MODEL
# ==============================


@st.cache_resource
def load_model():

    model = pickle.load(
        open(
            "fake_news_model.pkl",
            "rb"
        )
    )


    tfidf = pickle.load(
        open(
            "tfidf_vectorizer.pkl",
            "rb"
        )
    )


    return model,tfidf



try:

    model,tfidf = load_model()


except Exception as e:

    st.error(
        "Model loading failed: "
        + str(e)
    )

    st.stop()



# ==============================
# FUNCTIONS
# ==============================


def clean_text(text):

    text=text.lower()

    text=re.sub(
        r"http\S+",
        "",
        text
    )


    text=re.sub(
        "[^a-zA-Z ]",
        " ",
        text
    )


    text=re.sub(
        "\s+",
        " ",
        text
    )


    return text



clickbait_words=[

    "breaking",
    "shocking",
    "secret",
    "revealed",
    "viral",
    "truth",
    "amazing"

]



def clickbait_score(text):

    score=0

    text=text.lower()


    for word in clickbait_words:

        if word in text:

            score+=1


    return score



def sentiment_analysis(text):

    score=TextBlob(text).sentiment.polarity


    if score>0.3:

        emotion="Positive"

    elif score<-0.3:

        emotion="Negative"

    else:

        emotion="Neutral"


    return score,emotion




# ==============================
# SAMPLE NEWS
# ==============================


st.subheader(
    "🧪 Try Sample Articles"
)


sample_col1,sample_col2=st.columns(2)



news=""



with sample_col1:

    if st.button(
        "✅ Sample Real News"
    ):

        news="""
        The government announced a new education
        policy to improve digital learning
        facilities across schools.
        """



with sample_col2:


    if st.button(
        "⚠️ Sample Suspicious News"
    ):

        news="""
        Breaking shocking secret revealed!
        Scientists discovered a miracle cure
        doctors don't want you to know.
        """



# ==============================
# USER INPUT
# ==============================


news=st.text_area(

    "📝 Enter News Article",

    value=news,

    height=250,

    placeholder=
    "Paste the complete news article here..."

)



analyze=st.button(
    "🚀 Analyze News Credibility"
)




# ==============================
# PREDICTION
# ==============================


if analyze:


    if news.strip()=="":


        st.warning(
            "Please enter news content."
        )

        st.stop()



    cleaned=clean_text(news)



    vector=tfidf.transform(
        [cleaned]
    )



    prediction=model.predict(
        vector
    )[0]



    sentiment_score,emotion=sentiment_analysis(
        news
    )



    click_score=clickbait_score(
        news
    )



    # credibility calculation

    credibility=100


    credibility -= click_score*10


    if emotion=="Negative":

        credibility-=10



    credibility=max(
        0,
        credibility
    )



    if credibility>=70:

        risk="🟢 Low Risk"


    elif credibility>=40:

        risk="🟡 Medium Risk"


    else:

        risk="🔴 High Risk"



    # ==============================
    # RESULT CARDS
    # ==============================


    st.divider()


    st.subheader(
        "📊 AI Credibility Report"
    )



    c1,c2,c3=st.columns(3)



    with c1:

        st.markdown(

        f"""

        <div class="card">

        <h3>Prediction</h3>

        <h2>{prediction}</h2>

        </div>

        """,

        unsafe_allow_html=True

        )



    with c2:


        st.markdown(

        f"""

        <div class="card">

        <h3>Credibility Score</h3>

        <h2>{credibility}%</h2>

        </div>

        """,

        unsafe_allow_html=True

        )



    with c3:


        st.markdown(

        f"""

        <div class="card">

        <h3>Risk Level</h3>

        <h2>{risk}</h2>

        </div>

        """,

        unsafe_allow_html=True

        )




    st.write("")


    st.subheader(
        "🔍 NLP Analysis"
    )


    col1,col2,col3=st.columns(3)


    col1.metric(
        "Emotion",
        emotion
    )


    col2.metric(
        "Sentiment Score",
        round(sentiment_score,2)
    )


    col3.metric(
        "Clickbait Score",
        f"{click_score}/7"
    )



# ==============================
# ABOUT SECTION
# ==============================


st.divider()


with st.expander(
    "ℹ️ How this AI system works"
):

    st.write(
    """
    1. News text is cleaned using NLP preprocessing.

    2. TF-IDF converts text into numerical features.

    3. Machine Learning model predicts Fake/Real.

    4. Sentiment analysis detects emotional tone.

    5. Clickbait detection identifies suspicious patterns.

    6. AI generates credibility assessment.
    """
    )
