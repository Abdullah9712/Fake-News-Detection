import os
import difflib
import requests
import streamlit as st
from dotenv import load_dotenv
from requests.exceptions import RequestException, Timeout

# ---- Optional NLI (loaded safely) ----
# We try to load a small MNLI model eagerly on CPU to avoid meta tensors.
# If anything fails, we run without NLI and use a robust similarity fallback.
nli_model = None
tokenizer = None
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

    MODEL_NAME = "typeform/distilbert-base-uncased-mnli"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # IMPORTANT: low_cpu_mem_usage=False + no device_map => eager, real tensors (no meta)
    hf_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        torch_dtype=None,
        low_cpu_mem_usage=False,
        device_map=None,
    )
    nli_model = pipeline(
        "text-classification",
        model=hf_model,          # already on CPU
        tokenizer=tokenizer,
        device=-1,               # force CPU
        top_k=None               # return only top label by default
    )
except Exception as e:
    # We keep running without NLI; fallback similarity will be used.
    nli_model = None

# ----------------- ENV + CONFIG -----------------
load_dotenv()
API_KEY = os.getenv("API_KEY", "").strip()
CSE_ID = os.getenv("CSE_ID", "").strip()

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)
REQ_HEADERS = {"User-Agent": USER_AGENT}

TRUSTED_SOURCES = [
    # International mainstream news
    "bbc.com", "cnn.com", "reuters.com", "nytimes.com", "theguardian.com",
    "aljazeera.com", "apnews.com", "npr.org", "bloomberg.com", "wsj.com",
    "washingtonpost.com", "cnbc.com", "foxnews.com", "usatoday.com", "time.com",
    "financialtimes.com", "abcnews.go.com", "pbs.org", "euronews.com", "cbc.ca",
    # Sports
    "espn.com", "cricbuzz.com", "skysports.com", "sify.com/sports",
    # Tech & Science
    "techcrunch.com", "wired.com", "nature.com", "sciencemag.org", "scientificamerican.com",
    # Pakistan / Regional
    "dawn.com", "geo.tv", "express.pk", "arynews.tv", "tribune.com.pk", "thenews.com.pk",
    # Others
    "bbc.co.uk", "thehindu.com", "hindustantimes.com", "scroll.in"
]

# ----------------- SEARCH -----------------
def search_trusted_sources(query: str, max_results: int = 10):
    """Use Google Custom Search to find trusted-source links for a claim."""
    if not API_KEY or not CSE_ID:
        return []

    url = (
        "https://www.googleapis.com/customsearch/v1"
        f"?q={requests.utils.quote(query)}"
        f"&cx={CSE_ID}&key={API_KEY}&num={max_results}"
    )
    try:
        r = requests.get(url, timeout=8, headers=REQ_HEADERS)
        r.raise_for_status()
        data = r.json()
    except (RequestException, Timeout):
        return []

    items = data.get("items", []) or []
    results = []
    for it in items:
        link = it.get("link", "")
        title = it.get("title", "")
        snippet = it.get("snippet", "")
        if any(src in link for src in TRUSTED_SOURCES):
            results.append({"link": link, "title": title, "snippet": snippet})
    return results

# ----------------- VERIFICATION -----------------
def nli_entails(hypothesis: str, premise: str) -> bool:
    """
    Return True iff NLI model says the premise entails the hypothesis.
    We truncate premise for speed/safety.
    """
    if nli_model is None:
        return False
    premise = premise[:900]  # keep short for speed & stability
    try:
        # Pass as pair: (hypothesis, premise)
        out = nli_model((hypothesis, premise))
        if isinstance(out, list) and len(out) > 0 and "label" in out[0]:
            label = out[0]["label"].upper()
            # Some models may output labels like "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
            if label == "ENTAILMENT":
                return True
    except Exception:
        return False
    return False

def similarity_match(claim: str, title: str, snippet: str, threshold: float = 0.60) -> bool:
    """
    Fallback: check fuzzy similarity between the claim and the result's title+snippet.
    This is not NLI, but works well as a backup when NLI isn't available.
    """
    text = f"{title} {snippet}".lower()
    claim_l = claim.lower()
    ratio = difflib.SequenceMatcher(None, claim_l, text).ratio()
    return ratio >= threshold

def verify_claim_with_sources(claim: str, max_links_to_check: int = 5):
    """
    Try up to N links:
    1) If NLI is loaded, use NLI entailment with the page text.
    2) Else fallback to similarity check on title+snippet from search.
    Return (verified_bool_or_None, verifying_link_or_None, checked_links_list)
    """
    hits = search_trusted_sources(claim, max_results=max_links_to_check)
    if not hits:
        return None, None, []

    # If NLI available, fetch pages & try entailment
    if nli_model is not None:
        for h in hits:
            link = h["link"]
            try:
                resp = requests.get(link, timeout=8, headers=REQ_HEADERS)
                if resp.status_code != 200:
                    continue
                page_text = resp.text or ""
                if nli_entails(hypothesis=claim, premise=page_text):
                    return True, link, hits
            except (RequestException, Timeout):
                continue
        # If none entailed, still return first link for reference
        return False, hits[0]["link"], hits

    # Fallback: no NLI → use title/snippet similarity
    for h in hits:
        if similarity_match(claim, h.get("title", ""), h.get("snippet", "")):
            return True, h["link"], hits

    return False, hits[0]["link"], hits

# ----------------- UI -----------------
st.set_page_config(page_title="Fake News Verification", page_icon="📰", layout="centered")
st.title("📰 Fake News Verification App")
st.write(
    "This app checks your claim against **trusted news sources**. "
    "If any trusted outlet confirms it, it’s marked **REAL**; otherwise **Unverified/Possibly Fake**."
)

with st.expander("API Setup Help", expanded=False):
    st.markdown(
        "- Set environment variables in a `.env` file:\n"
        "  - `API_KEY=<your_google_custom_search_api_key>`\n"
        "  - `CSE_ID=<your_custom_search_engine_id>`"
    )

claim = st.text_area("Enter the news claim (e.g., 'Pakistan won the first ODI against West Indies'):", height=120)

if st.button("Verify"):
    if not claim.strip():
        st.warning("⚠ Please enter a claim first.")
    else:
        with st.spinner("Checking trusted sources..."):
            verified, link, checked = verify_claim_with_sources(claim, max_links_to_check=8)

        if verified is True:
            st.success("✅ Verified: This news appears REAL from trusted sources.")
            if link:
                st.info(f"📌 Source: [Read here]({link})")
        elif verified is False:
            st.error("🚨 No trusted sources confirm this. Possibly Fake or Unverified.")
            if link:
                st.warning(f"Closest related link: [Check here]({link})")
        else:
            st.warning("⚠ No trusted sources found (maybe very recent breaking news or API limit).")

        # Show what we checked
        if checked:
            with st.expander("Show checked sources"):
                for i, h in enumerate(checked, 1):
                    st.markdown(f"{i}. [{h['title']}]({h['link']})  \n{h['snippet']}")
