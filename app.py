import streamlit as st
import joblib
import requests
from transformers import pipeline
from dotenv import load_dotenv
import os
import torch

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load ML model and vectorizer
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Determine device: -1 = CPU, 0 = first GPU
device = 0 if torch.cuda.is_available() else -1

# Load NLI model safely
nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",  # can switch to 'bart-base-mnli' for less RAM usage
    device=device,
    torch_dtype=torch.float32
)

# Trusted news sources
TRUSTED_SOURCES = [
    "bbc.com", "cnn.com", "reuters.com", "nytimes.com", "theguardian.com",
    "aljazeera.com", "apnews.com", "npr.org", "bloomberg.com", "wsj.com"
]

# Function to search trusted sources
def search_trusted_sources(query):
    search_url = (
        f"https://www.googleapis.com/customsearch/v1?q={query}"
        f"&cx={CSE_ID}&key={API_KEY}"
    )
    response = requests.get(search_url)
    results = response.json()
    links = []
    if "items" in results:
        for item in results["items"]:
            for source in TRUSTED_SOURCES:
                if source in item["link"]:
                    links.append(item["link"])
    return links

# Function to verify using NLI
def verify_with_nli(user_text):
    links = search_trusted_sources(user_text)
    if not links:
        return False, None

    first_link = links[0]
    try:
        response = requests.get(first_link, timeout=5)
        page_text = response.text[:1000]  # Short snippet for testing

        nli_result = nli_model(f"{user_text} </s> {page_text}")

        if isinstance(nli_result, list) and len(nli_result) > 0:
            label = nli_result[0].get("label", "").upper()
            if label == "ENTAILMENT":
                return True, first_link

        return False, first_link

    except Exception as e:
        st.error(f"Verification error: {e}")
        return False, first_link

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Detection App")
st.write(
    "This app predicts whether a news article is **Fake** or **Real** using a trained ML model "
    "and verifies it against trusted news sources."
)

user_input = st.text_area("Enter the news text:", height=200)

if st.button("Predict"):
    if not user_input.strip():
        st.warning("⚠ Please enter some news text before predicting.")
    else:
        # Step 1: ML Model Prediction
        input_vectorized = vectorizer.transform([user_input])
        prediction = model.predict(input_vectorized)[0]

        # Step 2: Verify with NLI
        verified, source_link = verify_with_nli(user_input)

        # Step 3: Display results
        if prediction == 0:
            st.error("🚨 The model predicts this news is **FAKE**.")
        else:
            st.success("✅ The model predicts this news is **REAL**.")

        if verified:
            st.info(f"📌 Verified with trusted source: [Read here]({source_link})")
        else:
            if source_link:
                st.warning(f"⚠ Could not verify. Closest match found: [Check here]({source_link})")
            else:
                st.warning("⚠ No related trusted sources found.")
