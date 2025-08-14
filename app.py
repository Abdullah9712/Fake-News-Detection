import streamlit as st
import joblib
import requests
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from dotenv import load_dotenv
import os
import torch
from requests.exceptions import RequestException

# Load environment variables
load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

# Load ML model and vectorizer (not shown in UI, but kept in case needed later)
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Force CPU device for NLI model
device = torch.device("cpu")

# Load smaller, faster NLI model eagerly (avoid meta tensors)
model_name = "typeform/distilbert-base-uncased-mnli"
tokenizer = AutoTokenizer.from_pretrained(model_name)
nli_model_hf = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map=None  # Force full load to CPU
).to(device)

# Create NLI pipeline
nli_model = pipeline(
    "text-classification",
    model=nli_model_hf,
    tokenizer=tokenizer,
    device=-1  # CPU
)

# Trusted news sources
TRUSTED_SOURCES = [
    "bbc.com", "cnn.com", "reuters.com", "nytimes.com", "theguardian.com",
    "aljazeera.com", "apnews.com", "npr.org", "bloomberg.com", "wsj.com",
    "washingtonpost.com", "cnbc.com", "foxnews.com", "usatoday.com", "time.com",
    "financialtimes.com", "abcnews.go.com", "pbs.org", "euronews.com", "cbc.ca",
    "espn.com", "cricbuzz.com", "skysports.com", "sify.com/sports",
    "techcrunch.com", "wired.com", "nature.com", "sciencemag.org", "scientificamerican.com",
    "dawn.com", "geo.tv", "express.pk", "arynews.tv", "tribune.com.pk", "thenews.com.pk",
    "bbc.co.uk", "thehindu.com", "hindustantimes.com", "scroll.in"
]

# Search trusted sources using Google Custom Search
def search_trusted_sources(query):
    search_url = (
        f"https://www.googleapis.com/customsearch/v1?q={query}"
        f"&cx={CSE_ID}&key={API_KEY}"
    )
    try:
        response = requests.get(search_url, timeout=5)
        results = response.json()
        links = []
        if "items" in results:
            for item in results["items"]:
                for source in TRUSTED_SOURCES:
                    if source in item["link"]:
                        links.append(item["link"])
        return links
    except RequestException as e:
        print(f"Search error: {e}")
        return []

# Verify using multiple trusted sources with NLI
def verify_with_nli(user_text):
    links = search_trusted_sources(user_text)
    if not links:
        return None, None  # No trusted sources found

    for link in links[:5]:  # Check first 5 sources
        try:
            response = requests.get(link, timeout=5)
            page_text = response.text[:800]  # Short snippet for NLI

            # Run NLI model
            nli_result = nli_model(f"{user_text} </s> {page_text}")

            if isinstance(nli_result, list) and len(nli_result) > 0:
                label = nli_result[0].get("label", "").upper()
                if label == "ENTAILMENT":
                    return True, link

        except RequestException as e:
            print(f"Skipping {link} due to error: {e}")
            continue

    return False, links[0] if links else None

# Streamlit UI
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")
st.title("📰 Fake News Verification App")
st.write(
    "This app checks news text against **trusted sources**. "
    "If verified by a trusted source, it's marked **REAL**; "
    "if no match is found, it may be **Fake or Unverified**."
)

user_input = st.text_area("Enter the news text:", height=200)

if st.button("Check News"):
    if not user_input.strip():
        st.warning("⚠ Please enter some news text before checking.")
    else:
        verified, source_link = verify_with_nli(user_input)

        if verified:
            st.success("✅ Verified: This news is REAL.")
            st.info(f"📌 Source: [Read here]({source_link})")
        elif verified is False:
            st.error("🚨 No trusted sources confirm this. Possibly Fake or Misinformation.")
            if source_link:
                st.warning(f"Closest related link: [Check here]({source_link})")
        else:
            st.warning("⚠ No trusted sources found. This might be very recent breaking news.")
