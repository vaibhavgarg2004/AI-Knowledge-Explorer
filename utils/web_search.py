import os
import requests
from config.config import SERPAPI_KEY
import logging

# utils/web_search.py

logger = logging.getLogger(__name__)

def serpapi_search(query, num_results=3):
    """Return a list of text snippets from SerpAPI results. Requires SERPAPI_KEY environment variable."""
    if not SERPAPI_KEY:
        logger.warning("SERPAPI_KEY not configured.")
        return []
    params = {
        "q": query,
        "api_key": SERPAPI_KEY,
        "engine": "google",
        "num": num_results,
    }
    try:
        resp = requests.get("https://serpapi.com/search.json", params=params, timeout=10)
        data = resp.json()
        snippets = []
        for r in data.get("organic_results", [])[:num_results]:
            snippet = r.get("snippet") or r.get("title") or r.get("snippet_text") or ""
            snippets.append(snippet)
        return snippets
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return []
