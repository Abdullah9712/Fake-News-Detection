import streamlit as st
import joblib
import requests

# ===============================
# Load model and vectorizer
# ===============================
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# ===============================
# Google API Setup
# ===============================
API_KEY = "AIzaSyByheoFeDqF5gruBxTDpI_9rfhv1L74gfk"  # Replace with your real API key
CSE_ID = "c1f2092a6d6f64f40"  # Replace with your real CSE ID

# Trusted news sources
TRUSTED_SOURCES = [
    # International
    "bbc.com", "reuters.com", "cnn.com", "aljazeera.com", "apnews.com", "theguardian.com", "nytimes.com", 
    "washingtonpost.com", "bloomberg.com", "forbes.com", "time.com", "economist.com",
    
    # Sports
    "espn.com", "espncricinfo.com", "cricbuzz.com", "skysports.com", "goal.com", "uefa.com", "fifa.com",
    
    # Pakistani News
    "dawn.com", "geo.tv", "tribune.com.pk", "arynews.tv", "thenews.com.pk", "92news.tv", "dunyanews.tv",
    
    # Tech
    "techcrunch.com", "wired.com", "theverge.com", "cnet.com", "gsmarena.com",
    
    # Fact-checking
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org"
]

# ===============================
# Google Search Function
# ===============================
def google_search(query):
    """Search Google using Custom Search JSON API."""
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
    res = requests.get(url)
    return res.json().get("items", [])

# ===============================
# Prediction Function
# ===============================
def blended_prediction(news_text):
    """Blend model prediction with Google trusted sources check."""
    transformed = vectorizer.transform([news_text])
    model_pred = model.predict(transformed)[0]  # 1 = REAL, 0 = FAKE

    search_results = google_search(news_text)
    credible_found = False
    trusted_links = []

    for item in search_results:
        for source in TRUSTED_SOURCES:
            if source in item["link"]:
                credible_found = True
                trusted_links.append((item["title"], item["link"]))
                break

    if credible_found:
        return "REAL (Confirmed by trusted sources)", trusted_links, search_results
    else:
        if model_pred == 1:
            return "POSSIBLY REAL (Model says real but not confirmed)", [], search_results
        else:
            return "FAKE (No trusted sources found)", [], search_results

# ===============================
# Streamlit UI
# ===============================
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection with Google Verification")

news_input = st.text_area("Enter news headline or article:")

if st.button("Check News"):
    if news_input.strip():
        st.write("🔍 Checking...")
        final_pred, trusted_links, all_results = blended_prediction(news_input)

        st.subheader("✅ Final Prediction:")
        st.success(final_pred)

        st.subheader("🌐 Google Search Results:")
        for item in all_results:
            st.write(f"[{item['title']}]({item['link']})")

        if trusted_links:
            st.subheader("🔒 Confirmed Trusted Sources:")
            for title, link in trusted_links:
                st.write(f"[{title}]({link})")
    else:
        st.warning("Please enter some news text to check.")
