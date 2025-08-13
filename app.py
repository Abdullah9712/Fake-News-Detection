import streamlit as st
import joblib
import requests
from transformers import pipeline
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load Fake News Detection model

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Load NLI Model for Fact Checking

nli_model = pipeline("text-classification", model="facebook/bart-large-mnli")

# Trusted news sources
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "apnews.com", "cnn.com", "aljazeera.com",
    "nytimes.com", "theguardian.com", "geo.tv", "dawn.com"
]

# Helper Functions


def google_search(query, num_results=5):
    """Search Google for relevant news articles from trusted sources."""
    try:
        url = f"https://www.googleapis.com/customsearch/v1?q={query}&cx={CSE_ID}&key={API_KEY}"
        response = requests.get(url)
        results = response.json()
        articles = []
        if "items" in results:
            for item in results["items"]:
                link = item["link"]
                if any(source in link for source in TRUSTED_SOURCES):
                    articles.append({"title": item["title"], "snippet": item["snippet"]})
        return articles[:num_results]
    except Exception as e:
        st.error(f"Google Search failed: {e}")
        return []


def verify_with_nli(claim, articles):
    """Verify claim against trusted source snippets using NLI."""
    if not articles:
        return False, 0.0

    agree_count = 0
    total_confidence = 0

    for article in articles:
        snippet = article["snippet"]
        result = nli_model(f"{claim} </s></s> {snippet}", truncation=True)[0]
        if result["label"].upper() == "ENTAILMENT":
            agree_count += 1
            total_confidence += result["score"]

    # Require at least 2 trusted articles to agree
    avg_confidence = (total_confidence / agree_count) if agree_count > 0 else 0
    return agree_count >= 2, avg_confidence


def classify_news(text):
    """Classify as True or Fake using ML model + NLI verification."""
    # Step 1: Initial ML prediction
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = model.predict_proba(features).max()

    # Step 2: Fact-check with NLI
    articles = google_search(text)
    verified, nli_conf = verify_with_nli(text, articles)

    if verified:
        final_label = "True"
        final_confidence = max(confidence, nli_conf)
    else:
        final_label = "Fake"
        final_confidence = max(confidence, nli_conf)

    return final_label, final_confidence, articles

# Streamlit UI


st.set_page_config(page_title="AI Fake News Detector", layout="centered")
st.title("📰 AI-Powered Fake News Detection")
st.markdown("This app detects **fake news** and verifies information against **trusted news sources**.")

user_input = st.text_area("Enter a news headline or statement:", height=100)

if st.button("Check News"):
    if user_input.strip():
        label, conf, sources = classify_news(user_input)

        st.subheader("Prediction")
        st.write(f"**Result:** {label}")
        st.progress(conf)
        st.write(f"Confidence: **{conf:.2f}**")

        st.subheader("Trusted Source Check")
        if sources:
            for s in sources:
                st.write(f"- **{s['title']}** — {s['snippet']}")
        else:
            st.warning("No trusted sources found for verification.")
    else:
        st.warning("Please enter some text.")
