
import streamlit as st
import joblib
import re
from pathlib import Path

BASE_DIR = Path(__file__).parent

bundle = joblib.load(BASE_DIR / "truthlens_ai_models.joblib")
stop_words = joblib.load(BASE_DIR / "stop_words.joblib")

model = bundle["article_model"]
vectorizer = bundle["article_tfidf"]

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'<.*?>', ' ', text)
    text = re.sub(r'\breuters\b', ' ', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words and len(w) > 1]
    return ' '.join(words)

st.set_page_config(
    page_title="TruthLens AI",
    page_icon="🔎",
    layout="centered"
)

st.title("🔎 TruthLens AI")
st.subheader("AI-Powered Fake News Detection System")

st.write(
    "Analyze a news article using Machine Learning "
    "and Natural Language Processing."
)

st.info("Paste the complete news article below.")

news = st.text_area(
    "Enter News Article",
    height=220,
    placeholder="Paste your news article here..."
)

if st.button("🔍 Detect Fake News", use_container_width=True):
    if len(news.strip()) < 30:
        st.warning("Please enter a longer news article.")
    else:
        cleaned = clean_text(news)
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]

        st.divider()
        st.subheader("Analysis Result")

        if prediction == 1:
            st.error("🚨 FAKE NEWS")
            st.write("The model identifies this article as potentially fake.")
        else:
            st.success("✅ REAL NEWS")
            st.write("The model identifies this article as potentially real.")

        st.caption(
            "This is a machine-learning prediction, not independent "
            "verification of the article's claims. Check reliable "
            "news sources before sharing."
        )

st.divider()
st.subheader("About the Project")

st.write("""
**TruthLens AI** is a machine-learning-based fake news detection system.

**Technologies used:**
- Python
- Machine Learning
- Natural Language Processing
- TF-IDF Vectorization
- Passive Aggressive Classifier

The system analyzes article text and predicts whether it is REAL or FAKE.
""")

st.caption("Academic Mini Project | TruthLens AI")
