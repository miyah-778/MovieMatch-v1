
import logging
import requests

logger = logging.getLogger(__name__)

# Groq API configuration
URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL = "openai/gpt-oss-20b"

def explain(movie, liked_titles, api_key=None):
    title = str(movie.get("clean_title") or movie.get("title") or "this movie")
    genres = str(movie.get("genres") or "").replace("|", ", ")
    overview = str(movie.get("overview") or "")
    source = str(movie.get("recommendation_source") or "")

    # Fallback explanation
    if source == "Popularity baseline":
        fallback = f"{title} was selected based on historical audience ratings within your chosen discovery category. Its genres include {genres}. This recommendation is not personalized."
    elif source == "Genre-based":
        fallback = f"{title} was selected based on your genre preferences. Its genres include {genres}."
    elif source in ["Collaborative filtering", "Hybrid (CF + genre)"]:
        fallback = f"{title} was recommended based on ratings from viewers with similar preferences. Its genres include {genres}."
    else:
        fallback = f"{title} is part of your selected discovery category. Its genres include {genres}."

    if not api_key:
        logger.warning("Groq API key is missing.")
        return fallback, False

    # Generate AI explanation
    try:
        context = f"Movies the user liked: {', '.join(liked_titles[:8])}." if liked_titles and source != "Popularity baseline" else "No personal viewing preferences are available."

        prompt = (
            f"Write a friendly movie recommendation explanation in 35-55 words for {title}. "
            f"Genres: {genres}. Synopsis: {overview[:650]}. {context} "
            f"Recommendation method: {source}. "
            "Explain why the movie may be interesting without inventing plot details. "
            "If collaborative filtering was used, explain that similar viewers influenced the recommendation. "
            "If genre-based filtering was used, explain the connection to genre preferences. "
            "If popularity was used, explain that the recommendation is based on historical audience ratings, not personalization. "
            "Do not claim to know user preferences unless they were provided."
        )

        response = requests.post(
            URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_completion_tokens": 350,
                "temperature": 0.4,
                "reasoning_effort": "low"
            },
            timeout=25
        )

        if response.status_code != 200:
            logger.warning("Groq API returned HTTP %s: %s", response.status_code, response.text[:300])

        response.raise_for_status()
        answer = response.json()["choices"][0]["message"]["content"]

        if not answer or not answer.strip():
            logger.warning("Groq returned an empty explanation.")
            return fallback, False

        return answer.strip(), True

    except requests.Timeout:
        logger.warning("Groq request timed out.")
        return fallback, False

    except requests.RequestException as exc:
        logger.warning("Groq request failed: %s", type(exc).__name__)
        return fallback, False

    except (KeyError, IndexError, TypeError, ValueError):
        logger.warning("Groq returned an unexpected response format.")
        return fallback, False
