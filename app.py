from pathlib import Path
import streamlit as st
from recommender import MovieMatch, DISCOVERY_BANDS
from grok_helper import explain

st.set_page_config(page_title="MovieMatch", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp {background: #FAF8F4; color: #292725;}
.block-container {max-width: 1180px; padding-top: 2rem; padding-bottom: 4rem;}
h1, h2, h3 {color: #292725; letter-spacing: -0.03em;}
p, label {color: #49443F;}
div.stButton > button {border-radius: 12px; min-height: 58px; font-weight: 600; border: 1px solid #DED6CD;}
div.stButton > button[kind="primary"] {background: #8F3547; color: white; border-color: #8F3547;}
div.stButton > button[kind="secondary"] {background: #FFFFFF; color: #292725;}
div.stButton > button:hover {border-color: #8F3547; color: #8F3547;}
div.stButton > button[kind="primary"]:hover {background: #75283A; color: white;}
div[data-testid="stTabs"] button[aria-selected="true"] {color: #8F3547;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model(): return MovieMatch(Path(__file__).parent / "data")
try: model = load_model()
except Exception as e:
    st.error(f"MovieMatch data could not be loaded: {e}")
    st.stop()
for key, value in {"mode":"personal", "ratings":{}, "results":None, "result_mode":"", "signature":None}.items():
    if key not in st.session_state: st.session_state[key] = value

st.title("🎬 MovieMatch")
st.caption("Discover beyond the obvious · Personalized movie discovery")
tab1, tab2 = st.tabs(["🔎 Discover Movies", "🎞️ Your Movie Matches"])
with tab1:
    st.subheader("1 · How would you like to explore?")
    left, right = st.columns(2)
    with left:
        if st.button("❤️  Personalize My Recommendations\n\nRate movies to match your taste", use_container_width=True, type="primary" if st.session_state.mode=="personal" else "secondary"):
            st.session_state.mode = "personal"; st.session_state.results = None; st.rerun()
    with right:
        if st.button("🍿  I'm New to Movies\n\nNo ratings required", use_container_width=True, type="primary" if st.session_state.mode=="beginner" else "secondary"):
            st.session_state.mode = "beginner"; st.session_state.results = None; st.rerun()
    st.subheader("2 · Set your discovery preferences")
    level = st.selectbox("How adventurous are you feeling?", list(DISCOVERY_BANDS), index=2, help="Popularity is based on MovieLens rating counts, not worldwide viewership.")
    c1, c2 = st.columns(2)
    with c1: runtime_choice = st.selectbox("Available viewing time", ["No time limit", "90 minutes", "120 minutes", "180 minutes"], index=2)
    with c2: count = st.slider("Number of recommendations", 1, 5, 3)
    runtime = None if runtime_choice == "No time limit" else int(runtime_choice.split()[0])
    if st.session_state.mode == "personal":
        st.subheader("3 · Rate movies you've watched")
        st.caption("Rate at least 3 movies. Try including two you enjoyed and one you didn't. You can skip movies you haven't watched.")
        top = model.popularity.sort_values("rating_count", ascending=False).head(20).index
        gallery = model.movies[model.movies["movieId"].isin(top)].set_index("movieId").loc[top].reset_index()
        for start in range(0, len(gallery), 5):
            cols = st.columns(5)
            for col, (_, movie) in zip(cols, gallery.iloc[start:start+5].iterrows()):
                mid = int(movie["movieId"])
                with col:
                    poster = movie.get("poster_url")
                    if isinstance(poster, str) and poster.startswith("https://"): st.image(poster, use_container_width=True)
                    else: st.markdown("🎞️ **Poster unavailable**")
                    st.caption(str(movie["clean_title"])[:35])
                    choices = ["Not watched"] + [f"{x/2:g} ★" for x in range(1,11)]
                    current = st.session_state.ratings.get(mid)
                    idx = int(current*2) if current else 0
                    choice = st.selectbox("Rating", choices, index=idx, key=f"rate_{mid}", label_visibility="collapsed")
                    if choice == "Not watched": st.session_state.ratings.pop(mid, None)
                    else: st.session_state.ratings[mid] = float(choice.split()[0])
        st.caption("Can't find a movie? Search the full catalog below.")
        options = model.movies.sort_values("clean_title")
        ids = options["movieId"].astype(int).tolist()
        titles = dict(zip(options["movieId"].astype(int), options["clean_title"]))
        selected = st.selectbox("Search another movie", [None] + ids, format_func=lambda x: "Select a movie..." if x is None else f"{titles[x]} ({x})")
        if selected is not None:
            rating = st.select_slider("Your rating for this movie", options=[x/2 for x in range(1,11)], value=st.session_state.ratings.get(selected, 3.0))
            if st.button("Add / update this rating"): st.session_state.ratings[selected] = rating; st.rerun()
        st.write(f"**{len(st.session_state.ratings)} movies rated** · 3 minimum")
        if st.session_state.ratings:
            with st.expander("Review your ratings"):
                for mid, rating in list(st.session_state.ratings.items()):
                    a,b = st.columns([4,1]); a.write(f"{titles.get(mid, mid)} — {rating:g} ★")
                    if b.button("Remove", key=f"remove_{mid}"): st.session_state.ratings.pop(mid, None); st.rerun()
    st.subheader("Your selected mode")
    st.info(f"{'Personalized hybrid' if st.session_state.mode=='personal' else 'General Bayesian popularity (not personalized)'} · {level} · {runtime_choice} · {count} movie(s)")
    valid = st.session_state.mode=="beginner" or len(st.session_state.ratings)>=3
    if st.button("✨ Find My Movie Matches", type="primary", disabled=not valid, use_container_width=True):
        ratings = st.session_state.ratings.copy() if st.session_state.mode=="personal" else None
        result, label = model.recommend(ratings, level=level, n=count, max_runtime=runtime)
        st.session_state.results = result.to_dict("records"); st.session_state.result_mode = label
        st.session_state.signature = (st.session_state.mode, level, runtime, count, tuple(sorted((ratings or {}).items())))
    if st.session_state.results is not None:
        st.success(f"Recommendation method: {st.session_state.result_mode}")
        if not st.session_state.results: st.warning("No suitable movies match these filters. Try a longer time limit or another discovery level.")
        else:
            st.write("### Your quick preview")
            for i, m in enumerate(st.session_state.results, 1): st.write(f"**{i}. {m['clean_title']}** · {m['genres'].replace('|', ' · ')} · {int(m['runtime']) if m.get('runtime') is not None and str(m.get('runtime'))!='nan' else 'Runtime unknown'} min")
            st.info("Open the **Your Movie Matches** tab for posters, descriptions, and explanations.")
with tab2:
    if st.session_state.results is None: st.info("Complete your preferences in Discover Movies and click Find My Movie Matches first.")
    elif not st.session_state.results: st.warning("No matches found. Please adjust your discovery settings in the first tab.")
    else:
        st.subheader("Your Movie Matches")
        st.caption(st.session_state.result_mode)
        try: api_key = st.secrets.get("GROK_API_KEY", None)
        except Exception: api_key = None
        liked = [model.metadata.loc[mid, "clean_title"] for mid, r in st.session_state.ratings.items() if r>=3.5 and mid in model.metadata.index] if st.session_state.mode=="personal" else []
        for m in st.session_state.results:
            st.divider(); a,b = st.columns([1,3])
            with a:
                if isinstance(m.get("poster_url"), str) and m["poster_url"].startswith("https://"): st.image(m["poster_url"], use_container_width=True)
                else: st.info("Poster unavailable")
            with b:
                st.subheader(m["clean_title"])
                st.caption(f"{m['genres'].replace('|', ' · ')} · {int(m['runtime']) if m.get('runtime') is not None and str(m.get('runtime'))!='nan' else 'Runtime unknown'} minutes")
                overview = m.get("overview")
                st.write(overview if isinstance(overview,str) and overview.strip() else "Synopsis unavailable.")
                if st.button("Why this movie?", key=f"explain_{m['movieId']}"):
                    explanation, ai = explain(m, liked, api_key)
                    st.write(explanation)
                    if not ai: st.caption("Template explanation; AI explanation unavailable or not configured.")
