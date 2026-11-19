import requests

def explain(movie, liked_titles, api_key=None):
    genres = movie.get("genres", "")
    fallback = f"Suggested from your selected discovery category and available movie information ({genres.replace('|', ', ')})."
    if not liked_titles: fallback = f"This is a generally well-rated selection within your chosen discovery category ({genres.replace('|', ', ')})."
    else: fallback = f"Its genres ({genres.replace('|', ', ')}) may match themes in movies you rated highly."
    if not api_key: return fallback, False
    try:
        prompt = f"In at most 45 words explain why {movie.get('clean_title')} might suit someone who liked {liked_titles}. Genres: {genres}. Do not invent plot details or claim certainty."
        response = requests.post("https://api.x.ai/v1/chat/completions", headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, json={"model": "grok-4-fast-non-reasoning", "messages": [{"role": "user", "content": prompt}], "max_tokens": 110}, timeout=12)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"], True
    except Exception: return fallback, False
