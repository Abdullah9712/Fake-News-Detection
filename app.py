import streamlit as st
import joblib
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os

# Load environment variables

load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load Fake News model & vectorizer

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


# Load Natural Language Inference (NLI) model without meta tensor issue

@st.cache_resource
def load_nli_model():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-mnli")
    model_nli = AutoModelForSequenceClassification.from_pretrained("facebook/bart-large-mnli")
    return pipeline(
        "text-classification",
        model=model_nli,
        tokenizer=tokenizer,
        device=-1
    )

nli_model = load_nli_model()


# Trusted news sources

TRUSTED_SOURCES = [
    "bbc.com", "cnn.com", "reuters.com", "apnews.com", "nytimes.com",
    "theguardian.com", "forbes.com", "bloomberg.com", "aljazeera.com"
]


# Google Search API

def google_search(query):
    url = (
        f"https://www.googleapis.com/customsearch/v1?"
        f"q={query}&key={API_KEY}&cx={CSE_ID}"
    )
    response = requests.get(url)
    if response.status_code == 200:
        return response.json().get("items", [])
    return []


# Classify news as Real or Fake

def classify_news(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)[0]
    return "Real" if prediction == 1 else "Fake"


# Check against trusted sources using NLI

def verify_with_nli(news_text):
    search_results = google_search(news_text)
    for result in search_results:
        link = result.get("link", "")
        snippet = result.get("snippet", "")

        if any(source in link for source in TRUSTED_SOURCES):
            nli_result = nli_model(
                {"text": snippet, "text_pair": news_text}
            )
            if nli_result[0]['label'] == 'ENTAILMENT':
                return True, link
    return False, None


# Streamlit App

st.title("📰 Fake News Detection with NLI Verification")
st.write("Enter a news headline or paragraph to verify if it is real or fake.")

user_input = st.text_area("Enter News Text", "")

if st.button("Check News"):
    if user_input.strip():
        # Step 1: ML Model Prediction
        classification = classify_news(user_input)

        # Step 2: Verify with Trusted Sources using NLI
        verified, source_link = verify_with_nli(user_input)

        # Step 3: Display Results
        st.subheader("Prediction:")
        st.write(f"**ML Model Output:** {classification}")

        if verified:
            st.success(f"Verified ✅ - Matches trusted source: [Link]({source_link})")
        else:
            st.warning("No matching trusted source found ❌")

    else:
        st.error("Please enter some news text before checking.")
