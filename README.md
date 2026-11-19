# MovieMatch

Streamlit movie discovery app based on MovieLens latest-small. Two modes: personalized (minimum 3 watched-movie ratings) and beginner (Bayesian popularity). Discovery bands: Deep Cuts 10-24 ratings, Hidden Gems 25-49, Under the Radar 50-149, Popular Picks 150+. Bayesian eligibility cutoff: 3.7 (same as original discovery evaluation). Runtime is a strict maximum when chosen.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

The bundled `data/movies.csv` and `data/ratings.csv` support recommendations immediately, **but runtime, posters, and synopses require your group's `movie_enriched.csv`**. Put that file at `data/movie_enriched.csv` before deployment; it is not included here because it wasn't provided in this conversation. Without it, choose "No time limit" because strict runtime filters exclude movies with unknown runtime.

Optional Grok explanations: add `GROK_API_KEY` to Streamlit Community Cloud's Secrets configuration. For local testing, copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and insert your key; never commit the latter. Recommendations still work without Grok.

This is a prototype: CF is user-based Pearson KNN with 3 co-rated movies and shrinkage; genre matching is cosine similarity of a rating-weighted genre profile. The earlier validation selected alpha=1.0, so the deployed ranking uses CF where available and genre fallback otherwise. Popularity baseline uses a Bayesian score with smoothing k=50. MovieLens rating counts proxy popularity, not global viewership.

**Before production:** verify the 3.7 cutoff and runtime availability across categories, test results with real users, and add safeguards around API cost and caching.
