import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DISCOVERY_BANDS = {"Deep Cuts": (10, 24), "Hidden Gems": (25, 49), "Under the Radar": (50, 149), "Popular Picks": (150, float("inf"))}
MIN_QUALITY = 3.7
ALPHA = 1.0  # Selected during small-data validation.

class MovieMatch:
    def __init__(self, data_dir="data"):
        data_dir = Path(data_dir)
        if not data_dir.is_absolute(): data_dir = Path(__file__).resolve().parent / data_dir
        self.ratings = pd.read_csv(data_dir / "ratings.csv")
        enriched = data_dir / "movie_enriched.csv"
        if enriched.exists():
            self.movies = pd.read_csv(enriched)
            self.has_enrichment = True
        else:
            self.movies = pd.read_csv(data_dir / "movies.csv")
            self.has_enrichment = False
        self.movies = self.movies.drop_duplicates("movieId").copy()
        if "clean_title" not in self.movies: self.movies["clean_title"] = self.movies["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True).str.strip()
        for col in ["runtime", "overview", "poster_url"]:
            if col not in self.movies: self.movies[col] = np.nan
        self.movies["runtime"] = pd.to_numeric(self.movies["runtime"], errors="coerce")
        self.movies.loc[self.movies["runtime"] <= 0, "runtime"] = np.nan
        self.movies["genres"] = self.movies["genres"].fillna("(no genres listed)")
        genre_cols = [c for c in self.movies if c.startswith("genre_") and c != "genre_list"]
        if not genre_cols:
            dummy = self.movies["genres"].replace("(no genres listed)", "").str.get_dummies(sep="|").add_prefix("genre_")
            self.movies = pd.concat([self.movies, dummy], axis=1)
            genre_cols = dummy.columns.tolist()
        self.metadata = self.movies.set_index("movieId")
        self.genres = self.metadata[genre_cols].fillna(0).astype(float)
        self.matrix = self.ratings.pivot(index="userId", columns="movieId", values="rating")
        self.means = self.matrix.mean(axis=1)
        stats = self.ratings.groupby("movieId").agg(rating_count=("rating", "count"), avg_rating=("rating", "mean"))
        C = self.ratings["rating"].mean()
        stats["popularity_score"] = (stats["rating_count"] * stats["avg_rating"] + 50 * C) / (stats["rating_count"] + 50)
        self.popularity = stats

    def neighbors(self, new_ratings, min_common=3, shrinkage=10, k=20):
        rated = pd.Series(new_ratings, dtype=float)
        found = []
        for uid, row in self.matrix.iterrows():
            common = rated.index.intersection(row.dropna().index)
            if len(common) < min_common: continue
            a, b = rated.loc[common], row.loc[common]
            if a.std() == 0 or b.std() == 0: continue
            sim = a.corr(b) * len(common) / (len(common) + shrinkage)
            if pd.notna(sim) and sim > 0: found.append((uid, sim))
        return pd.DataFrame(found, columns=["userId", "similarity"]).sort_values("similarity", ascending=False).head(k) if found else pd.DataFrame(columns=["userId", "similarity"])

    def cf_scores(self, new_ratings, neighbors):
        if neighbors.empty: return pd.DataFrame(columns=["movieId", "cf_score", "support"])
        selected = self.matrix.loc[neighbors["userId"]].stack().rename("rating").reset_index()
        selected = selected.merge(neighbors, on="userId")
        selected = selected[~selected["movieId"].isin(new_ratings)]
        selected["weighted"] = (selected["rating"] - selected["userId"].map(self.means)) * selected["similarity"]
        scores = selected.groupby("movieId").agg(num=("weighted", "sum"), den=("similarity", "sum"), support=("userId", "nunique")).reset_index()
        scores["cf_score"] = np.clip(np.mean(list(new_ratings.values())) + scores["num"] / scores["den"], 0.5, 5)
        return scores[["movieId", "cf_score", "support"]]

    def genre_profile(self, new_ratings):
        rated = pd.Series(new_ratings, dtype=float)
        common = rated.index.intersection(self.genres.index)
        if not len(common): return pd.Series(0.0, index=self.genres.columns)
        profile = self.genres.loc[common].T.dot(rated.loc[common] - 3)
        counts = self.genres.loc[common].sum(axis=0).replace(0, np.nan)
        return (profile / counts).fillna(0)

    def genre_scores(self, new_ratings):
        profile = self.genre_profile(new_ratings)
        if np.linalg.norm(profile.values) == 0: values = np.full(len(self.genres), 0.5)
        else: values = (cosine_similarity(self.genres.values, profile.values.reshape(1, -1)).flatten() + 1) / 2
        return pd.DataFrame({"movieId": self.genres.index, "genre_score": values})

    def recommend(self, ratings=None, level="Under the Radar", n=3, max_runtime=None, min_quality=MIN_QUALITY, alpha=ALPHA):
        if level not in DISCOVERY_BANDS: raise ValueError("Invalid discovery level")
        ratings = {int(i): float(r) for i, r in (ratings or {}).items()}
        if ratings and len(ratings) < 3: raise ValueError("Rate at least three movies for personalized mode")
        if any(not np.isfinite(r) or r < 0.5 or r > 5 for r in ratings.values()): raise ValueError("Ratings must be between 0.5 and 5")
        if not isinstance(n, (int, np.integer)) or not 1 <= n <= 5: raise ValueError("Choose between 1 and 5 recommendations")
        if not 0 <= alpha <= 1: raise ValueError("Alpha must be between 0 and 1")
        if max_runtime is not None and (not np.isfinite(max_runtime) or max_runtime <= 0): raise ValueError("Runtime limit must be positive")
        if any(i not in self.metadata.index for i in ratings): raise ValueError("One or more rated movies are not in the catalog")
        lo, hi = DISCOVERY_BANDS[level]
        pool = self.popularity[self.popularity["rating_count"].between(lo, hi) & (self.popularity["popularity_score"] >= min_quality)].copy()
        pool = pool.reset_index().merge(self.movies[["movieId", "clean_title", "genres", "runtime", "overview", "poster_url"]], on="movieId", how="left")
        pool = pool[~pool["movieId"].isin(ratings)]
        if max_runtime is not None: pool = pool[pool["runtime"].notna() & (pool["runtime"] <= max_runtime)]
        mode = "General popularity (not personalized)"
        if ratings:
            neighbors = self.neighbors(ratings)
            genre = self.genre_scores(ratings)
            pool = pool.merge(genre, on="movieId", how="left")
            pool["genre_score"] = pool["genre_score"].fillna(0.5)
            has_genre_signal = np.linalg.norm(self.genre_profile(ratings).values) > 1e-12
            if neighbors.empty:
                if has_genre_signal:
                    pool["rank_score"] = pool["genre_score"]
                    pool["recommendation_source"] = "Genre-based"
                    mode = "Personalized by genre (no similar viewers found)"
                else:
                    pool["rank_score"] = pool["popularity_score"]
                    pool["recommendation_source"] = "Popularity baseline"
                    mode = "General popularity (not personalized; insufficient preference signal)"
            else:
                cf = self.cf_scores(ratings, neighbors)
                pool = pool.merge(cf, on="movieId", how="left")
                cf_norm = ((pool["cf_score"] - 0.5) / 4.5).clip(0, 1)
                pool["rank_score"] = alpha * cf_norm + (1 - alpha) * pool["genre_score"]
                pool["recommendation_source"] = "Collaborative filtering" if alpha == 1 else "Hybrid (CF + genre)"
                missing_cf = pool["cf_score"].isna()
                if has_genre_signal:
                    pool.loc[missing_cf, "rank_score"] = pool.loc[missing_cf, "genre_score"]
                    pool.loc[missing_cf, "recommendation_source"] = "Genre-based"
                else:
                    pool.loc[missing_cf, "rank_score"] = (pool.loc[missing_cf, "popularity_score"] - 0.5) / 4.5
                    pool.loc[missing_cf, "recommendation_source"] = "Popularity baseline"
                mode = "Personalized recommendations with per-movie fallback labels"
        else:
            pool["rank_score"] = pool["popularity_score"]
            pool["recommendation_source"] = "Popularity baseline"
        pool = pool.sort_values(["rank_score", "rating_count"], ascending=False).head(n).reset_index(drop=True)
        return pool, mode
