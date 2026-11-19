
from pathlib import Path
import pandas as pd
import streamlit as st
from recommender import MovieMatch, DISCOVERY_BANDS
from grok_helper import explain

# Page setup
st.set_page_config(page_title="MovieMatch", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

# Styling
st.markdown("""
<style>
.stApp {background: #FAF8F4; color: #292725;}
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
h1, h2, h3 {color: #292725; letter-spacing: -0.03em;}
p, label {color: #49443F;}
div.stButton > button {border-radius: 12px; font-weight: 600; border: 1px solid #DED6CD;}
div.stButton > button[kind="primary"], div.stButton > button[data-testid="stBaseButton-primary"] {background: #8F3547; color: #FFFFFF !important; border-color: #8F3547;}
div.stButton > button[kind="primary"] *, div.stButton > button[data-testid="stBaseButton-primary"] * {color: #FFFFFF !important;}
div.stButton > button[kind="primary"]:hover {background: #75283A; color: #FFFFFF !important;}
div.stButton > button[kind="secondary"] {background: #FFFFFF; color: #292725 !important; border: 1px solid #DED6CD;}
div.stButton > button[kind="secondary"] * {color: #292725 !important;}
div.stButton > button:hover {border-color: #8F3547;}
div[data-testid="stTabs"] button[aria-selected="true"] {color: #8F3547;}
div.st-key-mode_personal button, div.st-key-mode_beginner button {
    border-radius: 14px !important;
    min-height: 100px;
    white-space: pre-wrap;
    border: 1px solid #DED6CD !important;
    background: #FFFFFF !important;
    color: #292725 !important;
}
div.st-key-mode_personal button *, div.st-key-mode_beginner button * {color: #292725 !important;}
div.st-key-mode_personal button[kind="primary"], div.st-key-mode_beginner button[kind="primary"] {
    background: #8F3547 !important;
    border: 2px solid #8F3547 !important;
    color: #FFFFFF !important;
}
div.st-key-mode_personal button[kind="primary"] *, div.st-key-mode_beginner button[kind="primary"] * {color: #FFFFFF !important;}
div.st-key-mode_personal button:hover, div.st-key-mode_beginner button:hover {border-color: #8F3547 !important;}
.moviematch-progress {max-width: 380px; margin: 0 0 24px 0;}
.moviematch-progress-header {display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;}
.moviematch-progress-title {font-size: 14px; font-weight: 500; color: #746D66;}
.moviematch-progress-step {font-size: 12px; font-weight: 600; color: #8F3547;}
.moviematch-progress-bars {display: flex; gap: 5px;}
.moviematch-progress-bar {height: 5px; flex: 1; border-radius: 8px; background: #E8E1DA;}
.moviematch-progress-bar.active {background: #8F3547;}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    return MovieMatch(Path(__file__).parent / "data")

try:
    model = load_model()
except Exception as e:
    st.error(f"MovieMatch data could not be loaded: {e}")
    st.stop()

# Session state
defaults = {"step": 1, "mode": None, "ratings": {}, "level": "Under the Radar", "runtime_choice": "120 minutes", "count": 3, "results": None, "result_mode": "", "result_signature": None, "explanations": {}}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Helper functions
def clear_results():
    st.session_state.results = None
    st.session_state.result_mode = ""
    st.session_state.result_signature = None
    st.session_state.explanations = {}

def select_mode(mode):
    if st.session_state.mode != mode:
        st.session_state.mode = mode
        clear_results()

def next_step():
    st.session_state.step = min(3, st.session_state.step + 1)

def previous_step():
    st.session_state.step = max(1, st.session_state.step - 1)

def update_rating(mid):
    choice = st.session_state[f"rate_{mid}"]
    if choice == "Not watched":
        st.session_state.ratings.pop(mid, None)
    else:
        st.session_state.ratings[mid] = float(choice.split()[0])
    clear_results()

def remove_rating(mid):
    st.session_state.ratings.pop(mid, None)
    st.session_state.pop(f"rate_{mid}", None)
    clear_results()

def current_signature():
    ratings = st.session_state.ratings if st.session_state.mode == "personal" else {}
    return (st.session_state.mode, st.session_state.level, st.session_state.runtime_choice, st.session_state.count, tuple(sorted(ratings.items())))

def format_runtime(value):
    try:
        if pd.notna(value) and float(value) > 0:
            return f"{int(float(value))} min"
    except (TypeError, ValueError):
        pass
    return "Runtime unavailable"

def recommendation_label(source):
    labels = {"Popularity baseline": "General recommendation · Not personalized", "Genre-based": "Personalized using genre preferences", "Collaborative filtering": "Personalized using similar viewers", "Hybrid (CF + genre)": "Personalized using similar viewers and genres"}
    return labels.get(source, source or "Recommendation")

# Prepare movie catalog
catalog = model.movies.drop_duplicates("movieId").copy()
catalog["display_title"] = catalog["title"].fillna(catalog["clean_title"]).astype(str)
movie_titles = dict(zip(catalog["movieId"].astype(int), catalog["display_title"]))

# Header
st.title("🎬 MovieMatch")
st.caption("Discover beyond the obvious · Personalized movie discovery")

tab1, tab2 = st.tabs(["🔎 Discover Movies", "🎞️ Your Movie Matches"])

with tab1:
    # Compact progress tracker
    step = st.session_state.step
    bars = "".join([f'<div class="moviematch-progress-bar {"active" if i <= step else ""}"></div>' for i in range(1, 4)])
    st.markdown(f"""
    <div class="moviematch-progress">
        <div class="moviematch-progress-header">
            <span class="moviematch-progress-title">Your discovery journey</span>
            <span class="moviematch-progress-step">Step {step} of 3</span>
        </div>
        <div class="moviematch-progress-bars">{bars}</div>
    </div>
    """, unsafe_allow_html=True)

    # STEP 1: Choose mode
    if step == 1:
        st.header("1 · How would you like to explore?")
        st.write("Choose your starting point.")

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.button("❤️ Personalize My Recommendations\nRate movies to match your taste", key="mode_personal", type="primary" if st.session_state.mode == "personal" else "secondary", on_click=select_mode, args=("personal",), use_container_width=True)
        with col2:
            st.button("🍿 I'm New to Movies\nNo ratings required", key="mode_beginner", type="primary" if st.session_state.mode == "beginner" else "secondary", on_click=select_mode, args=("beginner",), use_container_width=True)

        if st.session_state.mode == "personal":
            st.info("We'll use your movie ratings to find recommendations that match your preferences.")
        elif st.session_state.mode == "beginner":
            st.info("Explore well-rated movies without providing personal ratings.")

        st.write("")
        forward, spacer = st.columns([1, 4])
        with forward:
            st.button("Continue →", type="primary", disabled=st.session_state.mode is None, on_click=next_step, use_container_width=True)

    # STEP 2: Discovery preferences
    elif step == 2:
        st.header("2 · Set your discovery preferences")
        st.write("Tell us what you're in the mood for.")

        st.selectbox("How adventurous are you feeling?", list(DISCOVERY_BANDS), key="level", on_change=clear_results)

        descriptions = {
            "Popular Picks": "Play it safe — Familiar, widely rated movies that many viewers have enjoyed.",
            "Under the Radar": "A little adventurous — Movies with a moderate following that you might have overlooked.",
            "Hidden Gems": "Explore something different — Lesser-known movies with some audience feedback.",
            "Deep Cuts": "Surprise me — Niche titles with limited rating history."
        }
        st.info(descriptions[st.session_state.level])

        col1, col2 = st.columns(2, gap="medium")
        with col1:
            st.selectbox("Available viewing time", ["No time limit", "90 minutes", "120 minutes", "180 minutes"], key="runtime_choice", help="Your time limit is strict. Movies without known runtimes won't appear when a limit is selected.", on_change=clear_results)
        with col2:
            st.slider("Number of recommendations", min_value=1, max_value=5, key="count", on_change=clear_results)

        st.write("")
        back, forward, spacer = st.columns([1, 1, 3])
        with back:
            st.button("← Back", on_click=previous_step, use_container_width=True)
        with forward:
            st.button("Continue →", type="primary", on_click=next_step, use_container_width=True)

    # STEP 3: Ratings or beginner summary
    elif step == 3:
        if st.session_state.mode == "personal":
            st.header("3 · Rate movies you've watched")
            st.write("Rate at least **3 movies**. Try including two you enjoyed and one you didn't.")

            # Popular movie gallery
            top_ids = model.popularity.sort_values("rating_count", ascending=False).head(20).index
            gallery = catalog[catalog["movieId"].isin(top_ids)].set_index("movieId").reindex(top_ids).dropna(subset=["clean_title"]).reset_index()

            for start in range(0, len(gallery), 5):
                cols = st.columns(5, gap="small")
                for col, (_, movie) in zip(cols, gallery.iloc[start:start+5].iterrows()):
                    mid = int(movie["movieId"])
                    with col:
                        poster = movie.get("poster_url")
                        if isinstance(poster, str) and poster.startswith("https://"):
                            st.image(poster, use_container_width=True)
                        else:
                            st.info("🎞️ Poster unavailable")

                        st.caption(str(movie["clean_title"])[:38])
                        choices = ["Not watched"] + [f"{x/2:g} ★" for x in range(10, 0, -1)]
                        key = f"rate_{mid}"

                        if key not in st.session_state:
                            existing = st.session_state.ratings.get(mid)
                            st.session_state[key] = f"{existing:g} ★" if existing else "Not watched"

                        st.selectbox("Rating", choices, key=key, label_visibility="collapsed", on_change=update_rating, args=(mid,))

            # Search additional movies
            st.divider()
            st.subheader("Can't find a movie you've watched?")
            st.caption("Search by title, select a movie, and add your rating.")

            search_query = st.text_input("Search movies", placeholder="e.g. Titanic, Inception, The Matrix", key="movie_search_query")

            if len(search_query.strip()) >= 2:
                matches = catalog[catalog["display_title"].str.contains(search_query.strip(), case=False, regex=False, na=False)].sort_values("display_title").head(15)

                if matches.empty:
                    st.info("No movies found. Try another title.")
                else:
                    match_ids = matches["movieId"].astype(int).tolist()
                    selected_id = st.selectbox("Select a movie", match_ids, format_func=lambda mid: movie_titles[mid], key=f"search_movie_{search_query.strip().lower()}")
                    rating_options = [x/2 for x in range(10, 0, -1)]
                    selected_rating = st.selectbox("Your rating", [None] + rating_options, format_func=lambda x: "Select a rating..." if x is None else f"{x:g} ★", key="search_movie_rating")

                    if st.button("＋ Add to My Ratings", type="primary", key="add_search_movie", disabled=selected_rating is None):
                        st.session_state.ratings[int(selected_id)] = float(selected_rating)
                        clear_results()
                        st.success(f"Added {movie_titles[selected_id]} — {selected_rating:g} ★")
            else:
                st.caption("Enter at least two characters to search.")

            # Rating progress
            rated_count = len(st.session_state.ratings)
            st.divider()
            st.write(f"**{rated_count} movies rated · 3 minimum**")

            if rated_count >= 3:
                st.success("You're ready for personalized recommendations.")
                positive = sum(r >= 3.5 for r in st.session_state.ratings.values())
                negative = sum(r <= 2.5 for r in st.session_state.ratings.values())
                if positive < 2 or negative < 1:
                    st.caption("Tip: Rating two liked movies and one disliked movie may help us understand your preferences.")
            else:
                st.warning(f"Please rate {3 - rated_count} more movie(s) to continue.")

            if rated_count:
                with st.expander("Review your ratings"):
                    for mid, rating in list(st.session_state.ratings.items()):
                        a, b = st.columns([4, 1])
                        a.write(f"{movie_titles.get(mid, mid)} — {rating:g} ★")
                        b.button("Remove", key=f"remove_{mid}", on_click=remove_rating, args=(mid,))

        else:
            st.header("3 · Ready to discover?")
            st.write("You chose **I'm New to Movies**.")
            st.info("We'll recommend well-rated movies using historical MovieLens ratings. These recommendations are not personalized.")

        # Summary
        st.divider()
        st.subheader("Your discovery settings")
        st.info(f"**Mode:** {'Personalized' if st.session_state.mode == 'personal' else 'New to Movies'}\n\n**Discovery level:** {st.session_state.level}\n\n**Viewing time:** {st.session_state.runtime_choice}\n\n**Recommendations:** {st.session_state.count}")

        st.write("")
        back, generate, spacer = st.columns([1, 2, 2])
        with back:
            st.button("← Back", on_click=previous_step, use_container_width=True)

        with generate:
            valid = st.session_state.mode == "beginner" or len(st.session_state.ratings) >= 3
            if st.button("✨ Find My Movie Matches", type="primary", disabled=not valid, use_container_width=True):
                ratings = st.session_state.ratings.copy() if st.session_state.mode == "personal" else None
                runtime = None if st.session_state.runtime_choice == "No time limit" else int(st.session_state.runtime_choice.split()[0])

                try:
                    with st.spinner("Finding your movie matches..."):
                        results, label = model.recommend(ratings, level=st.session_state.level, n=st.session_state.count, max_runtime=runtime)
                    st.session_state.results = results.to_dict("records")
                    st.session_state.result_mode = label
                    st.session_state.result_signature = current_signature()
                    st.session_state.explanations = {}
                except Exception as e:
                    clear_results()
                    st.error(f"Unable to generate recommendations: {e}")

        # Quick preview
        if st.session_state.results is not None and st.session_state.result_signature == current_signature():
            st.divider()
            st.subheader("Your movie matches")
            st.caption(st.session_state.result_mode)

            if not st.session_state.results:
                st.warning("No movies match your current settings. Try another discovery level or a longer viewing time.")
            else:
                if len(st.session_state.results) < st.session_state.count:
                    st.info(f"We found {len(st.session_state.results)} eligible movies out of {st.session_state.count} requested.")

                for i, movie in enumerate(st.session_state.results, 1):
                    st.write(f"**{i}. {movie['clean_title']}**")
                    st.caption(f"{str(movie.get('genres', '')).replace('|', ' · ')} · {format_runtime(movie.get('runtime'))} · {recommendation_label(movie.get('recommendation_source', ''))}")

                st.success("Your matches are ready! Open the **🎞️ Your Movie Matches** tab for more details.")

# TAB 2: Movie details
with tab2:
    if st.session_state.results is None or st.session_state.result_signature != current_signature():
        st.info("Complete your discovery journey in the first tab to unlock your movie matches.")

    elif not st.session_state.results:
        st.warning("No matches found. Return to Discover Movies and adjust your preferences.")

    else:
        st.header("🎞️ Your Movie Matches")
        st.caption(st.session_state.result_mode)

        try:
            api_key = st.secrets.get("GROK_API_KEY", None)
        except Exception:
            api_key = None

        liked = []
        if st.session_state.mode == "personal":
            liked = [str(model.metadata.loc[mid, "clean_title"]) for mid, r in st.session_state.ratings.items() if r >= 3.5 and mid in model.metadata.index]

        for i, movie in enumerate(st.session_state.results, 1):
            st.divider()
            left, right = st.columns([1, 3], gap="large")

            with left:
                poster = movie.get("poster_url")
                if isinstance(poster, str) and poster.startswith("https://"):
                    st.image(poster, use_container_width=True)
                else:
                    st.info("🎞️ Poster unavailable")

            with right:
                st.subheader(f"{i}. {movie['clean_title']}")
                st.caption(f"{str(movie.get('genres', '')).replace('|', ' · ')} · {format_runtime(movie.get('runtime'))}")

                source = movie.get("recommendation_source", "")
                st.caption(recommendation_label(source))

                overview = movie.get("overview")
                st.write(overview if isinstance(overview, str) and overview.strip() else "Synopsis unavailable.")

                mid = int(movie["movieId"])
                if st.button("✨ Why this movie?", key=f"explain_{mid}"):
                    with st.spinner("Preparing your explanation..."):
                        explanation, ai = explain(movie, liked if source != "Popularity baseline" else [], api_key)
                    st.session_state.explanations[mid] = (explanation, ai)

                if mid in st.session_state.explanations:
                    explanation, ai = st.session_state.explanations[mid]
                    st.write(explanation)
                    if not ai:
                        st.caption("Template explanation · AI explanation unavailable or not configured.")

        st.divider()
        st.info("Want different recommendations? Return to Discover Movies and adjust your preferences.")
