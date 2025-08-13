# import streamlit as st
# import joblib
# import requests
# from transformers import pipeline
# from dotenv import load_dotenv
# import os

# # Load environment variables
# load_dotenv()
# API_KEY = os.getenv("API_KEY")
# CSE_ID = os.getenv("CSE_ID")

# if not API_KEY or not CSE_ID:
#     st.error("❌ Missing API_KEY or CSE_ID in environment variables. Please check your .env or Streamlit secrets.")
#     st.stop()

# # Load model and vectorizer
# model = joblib.load("fake_news_model.pkl")
# vectorizer = joblib.load("vectorizer.pkl")

# # Trusted news sources
# TRUSTED_SOURCES = [
#     "bbc.com", "reuters.com", "cnn.com", "aljazeera.com", "apnews.com", "theguardian.com", "nytimes.com",
#     "washingtonpost.com", "bloomberg.com", "forbes.com", "time.com", "economist.com",
#     "espn.com", "espncricinfo.com", "cricbuzz.com", "skysports.com",
#     "dawn.com", "geo.tv", "tribune.com.pk", "arynews.tv", "thenews.com.pk", "92news.tv", "dunyanews.tv",
#     "snopes.com", "factcheck.org", "politifact.com", "fullfact.org"
# ]

# # Load NLI model (CPU mode for compatibility)
# nli_model = pipeline(
#     "text-classification",
#     model="facebook/bart-large-mnli",
#     device=-1,
#     trust_remote_code=True
# )

# # Google Search
# def google_search(query):
#     url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
#     res = requests.get(url)
#     return res.json().get("items", [])

# # Check claim using NLI
# def check_claim_with_nli(claim, snippet):
#     if not snippet.strip():
#         return "UNSURE"
    
#     result = nli_model(f"{claim} </s> {snippet}", return_all_scores=True)[0]

#     # Map possible label formats
#     label_map = {
#         "LABEL_0": "ENTAILMENT",
#         "LABEL_1": "NEUTRAL",
#         "LABEL_2": "CONTRADICTION",
#         "ENTAILMENT": "ENTAILMENT",
#         "NEUTRAL": "NEUTRAL",
#         "CONTRADICTION": "CONTRADICTION"
#     }
#     scores = {label_map.get(item['label'], item['label']): item['score'] for item in result}

#     if scores.get("ENTAILMENT", 0) > 0.5:
#         return "AGREES"
#     elif scores.get("CONTRADICTION", 0) > 0.5:
#         return "DISAGREES"
#     else:
#         return "UNSURE"

# # Prediction function
# def blended_prediction(news_text):
#     transformed = vectorizer.transform([news_text])
#     model_pred = model.predict(transformed)[0]

#     search_results = google_search(news_text)
#     credible_found = False
#     trusted_links = []
#     disagreement_found = False

#     for item in search_results:
#         link = item.get("link", "")
#         snippet = item.get("snippet", "")
#         for source in TRUSTED_SOURCES:
#             if source in link:
#                 credibility_check = check_claim_with_nli(news_text, snippet)
#                 if credibility_check == "AGREES":
#                     credible_found = True
#                     trusted_links.append((item["title"], link))
#                 elif credibility_check == "DISAGREES":
#                     disagreement_found = True
#                 break

#     if disagreement_found:
#         return "FAKE (Contradicted by trusted sources)", trusted_links, search_results
#     elif credible_found:
#         return "REAL (Confirmed by trusted sources)", trusted_links, search_results
#     else:
#         if model_pred == 1:
#             return "POSSIBLY REAL (Model says real but not confirmed)", [], search_results
#         else:
#             return "FAKE (No trusted sources found)", [], search_results

# # Streamlit UI
# st.set_page_config(page_title="Fake News Detector", layout="wide")
# st.title("📰 Fake News Detection with Google + NLI Verification")

# news_input = st.text_area("Enter news headline or article:")

# if st.button("Check News"):
#     if news_input.strip():
#         st.write("🔍 Checking...")
#         final_pred, trusted_links, all_results = blended_prediction(news_input)

#         st.subheader("✅ Final Prediction:")
#         if "FAKE" in final_pred:
#             st.error(final_pred)
#         elif "REAL" in final_pred:
#             st.success(final_pred)
#         else:
#             st.warning(final_pred)

#         st.subheader("🌐 Google Search Results:")
#         for item in all_results:
#             st.write(f"[{item['title']}]({item['link']}) - {item.get('snippet','')}")

#         if trusted_links:
#             st.subheader("🔒 Confirmed Trusted Sources:")
#             for title, link in trusted_links:
#                 st.write(f"[{title}]({link})")
#     else:
#         st.warning("Please enter some news text to check.")


import streamlit as st
import joblib
import requests
from transformers import pipeline
from dotenv import load_dotenv
import os

# -----------------------
# Load Environment Vars
# -----------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
CSE_ID = os.getenv("CSE_ID")

if not API_KEY or not CSE_ID:
    st.error("❌ Missing API_KEY or CSE_ID in environment variables.")
    st.stop()

# -----------------------
# Load Model & Vectorizer
# -----------------------
model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Trusted sources
TRUSTED_SOURCES = [
    "bbc.com", "reuters.com", "cnn.com", "aljazeera.com", "apnews.com", "theguardian.com", "nytimes.com",
    "washingtonpost.com", "bloomberg.com", "forbes.com", "time.com", "economist.com",
    "espn.com", "espncricinfo.com", "cricbuzz.com", "skysports.com",
    "dawn.com", "geo.tv", "tribune.com.pk", "arynews.tv", "thenews.com.pk", "92news.tv", "dunyanews.tv",
    "snopes.com", "factcheck.org", "politifact.com", "fullfact.org"
]

# NLI Model
nli_model = pipeline(
    "text-classification",
    model="facebook/bart-large-mnli",
    device=-1
)

# -----------------------
# Google Search
# -----------------------
def google_search(query):
    url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&cx={CSE_ID}"
    res = requests.get(url)
    return res.json().get("items", [])

# NLI Check
def check_claim_with_nli(claim, snippet):
    if not snippet.strip():
        return "UNSURE"
    result = nli_model(f"{claim} </s> {snippet}", return_all_scores=True)[0]
    label_map = {"LABEL_0": "ENTAILMENT", "LABEL_1": "NEUTRAL", "LABEL_2": "CONTRADICTION"}
    scores = {label_map.get(item['label'], item['label']): item['score'] for item in result}
    if scores.get("ENTAILMENT", 0) > 0.5:
        return "AGREES"
    elif scores.get("CONTRADICTION", 0) > 0.5:
        return "DISAGREES"
    else:
        return "UNSURE"

# Prediction
def blended_prediction(news_text):
    transformed = vectorizer.transform([news_text])
    model_proba = model.predict_proba(transformed)[0]  # [prob_fake, prob_real]
    model_pred = 1 if model_proba[1] > 0.5 else 0

    search_results = google_search(news_text)
    agrees, disagrees, others = [], [], []

    for item in search_results:
        link = item.get("link", "")
        snippet = item.get("snippet", "")
        title = item.get("title", "No Title")

        if any(source in link for source in TRUSTED_SOURCES):
            credibility_check = check_claim_with_nli(news_text, snippet)
            if credibility_check == "AGREES":
                agrees.append((title, link, snippet))
            elif credibility_check == "DISAGREES":
                disagrees.append((title, link, snippet))
        else:
            others.append((title, link, snippet))

    # NEW DECISION RULES
    if disagrees:  # Any trusted contradiction → FAKE
        return "❌ FAKE", f"Contradicted by {len(disagrees)} trusted sources.", agrees, disagrees, others, model_proba

    if len(agrees) >= 2:  # Multiple trusted agrees → REAL
        return "✅ REAL", f"Confirmed by {len(agrees)} trusted sources.", agrees, disagrees, others, model_proba

    # No strong external evidence → use ML confidence
    if model_proba[1] >= 0.6:
        return "⚠️ POSSIBLY REAL", f"Model confidence: {model_proba[1]*100:.1f}%", agrees, disagrees, others, model_proba
    elif model_proba[1] <= 0.4:
        return "⚠️ POSSIBLY FAKE", f"Model confidence: {model_proba[1]*100:.1f}%", agrees, disagrees, others, model_proba
    else:
        return "🤔 UNSURE", "Insufficient evidence from trusted sources.", agrees, disagrees, others, model_proba

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Fake News Detector", layout="wide")
st.title("📰 Fake News Detection")
st.caption("Hybrid model using ML + Google Search + NLI Verification")

news_input = st.text_area("Enter news headline or article:")

if st.button("Check News"):
    if news_input.strip():
        with st.spinner("🔍 Analyzing news..."):
            final_status, desc, agrees, disagrees, others, model_proba = blended_prediction(news_input)

        # Prediction
        st.subheader("Prediction Result")
        st.markdown(f"### {final_status}")
        st.write(desc)

        # Show model confidence
        st.progress(model_proba[1])
        st.write(f"Model REAL confidence: **{model_proba[1]*100:.1f}%**")
        st.write(f"Model FAKE confidence: **{model_proba[0]*100:.1f}%**")

        # Trusted sources (Agree)
        if agrees:
            st.success(f"✅ Trusted sources confirming the news ({len(agrees)} found):")
            for title, link, snippet in agrees:
                st.markdown(f"**[{title}]({link})**  \n_{snippet}_")

        # Trusted sources (Disagree)
        if disagrees:
            st.error(f"❌ Trusted sources contradicting the news ({len(disagrees)} found):")
            for title, link, snippet in disagrees:
                st.markdown(f"**[{title}]({link})**  \n_{snippet}_")

        # Other results
        with st.expander("🌐 Other Google Search Results"):
            for title, link, snippet in others:
                st.markdown(f"**[{title}]({link})**  \n_{snippet}_")
    else:
        st.warning("Please enter some news text to check.")
