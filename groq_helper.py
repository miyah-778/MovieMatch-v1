import logging
import requests

logger = logging.getLogger(__name__)

def explain(movie, liked_titles, api_key=None):
    title = str(movie.get("clean_title") or movie.get("title") or "this movie")
    genres = str(movie.get("genres") or "").replace("|", ", ")
    overview = str(movie.get("overview") or "")
    source = str(movie.get("recommendation_source") or "")
    if source == "Popularity baseline" or not liked_titles:
        fallback = f"{title} was selected from your discovery category using historical audience ratings. Its genres include {genres}. This is not a personalized recommendation."
    elif source == "Genre-based":
        fallback = f"{title} was selected using your genre preferences. Its genres include {genres}."
    else:
        fallback = f"{title} was selected based on ratings from similar viewers. Its genres include {genres}."
    if not api_key: return fallback, False
    try:
        context = f"User liked: {', '.join(liked_titles[:8])}." if liked_titles and source != "Popularity baseline" else "No personal viewing preferences are available."
        prompt = (f"Write a friendly, factual movie discovery explanation in 35-55 words for {title}. "
                  f"Genres: {genres}. Synopsis: {overview[:650]}. {context} "
                  f"Recommendation method: {source}. Explain the method accurately: never say genres drove ranking when collaborative filtering did, "
                  "and never imply personalization for popularity recommendations. Do not invent plot facts or user preferences.")
        response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "llama-3.3-70b-versatile", "messages": [{"role": "user", "content": prompt}], "max_tokens": 160, "temperature": 0.4}, timeout=15)
        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"].strip()
        if not answer: return fallback, False
        return answer, True
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError) as exc:
        logger.warning("Groq explanation unavailable: %s", exc)
        return fallback, False
