from pathlib import Path
import pandas as pd
import streamlit as st
from recommender import MovieMatch, DISCOVERY_BANDS
from groq_helper import explain

st.set_page_config(page_title="MovieMatch", page_icon="🎬", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.stApp {background:#FAF8F4;color:#292725;}
.block-container {max-width:1120px;padding-top:2rem;padding-bottom:4rem;}
h1,h2,h3 {color:#292725;letter-spacing:-.025em;}
p,label {color:#49443F;}
div.stButton button {border-radius:12px !important;min-height:45px;font-weight:600;}
div.stButton button[kind="secondary"],div.stButton button[data-testid="stBaseButton-secondary"] {background:#FFFFFF !important;border:1px solid #DED6CD !important;color:#292725 !important;}
div.stButton button[kind="secondary"] *,div.stButton button[data-testid="stBaseButton-secondary"] * {color:#292725 !important;}
div.stButton button[kind="primary"],div.stButton button[data-testid="stBaseButton-primary"] {background:#8F3547 !important;border:1px solid #8F3547 !important;color:#FFFFFF !important;}
div.stButton button[kind="primary"] *,div.stButton button[data-testid="stBaseButton-primary"] * {color:#FFFFFF !important;}
div.stButton button[kind="primary"]:hover {background:#75283A !important;}
/* Continue buttons remain white even on hover. */
div.st-key-continue_step1 button,div.st-key-continue_step2 button,
div.st-key-continue_step1 button:hover,div.st-key-continue_step2 button:hover {background:#FFFFFF !important;border:1px solid #DED6CD !important;color:#292725 !important;}
div.st-key-continue_step1 button:hover,div.st-key-continue_step2 button:hover {border:2px solid #8F3547 !important;}
div.st-key-continue_step1 button *,div.st-key-continue_step2 button * {color:#292725 !important;opacity:1 !important;}
div.st-key-continue_step1 button:hover *,div.st-key-continue_step2 button:hover * {color:#8F3547 !important;}
div.st-key-continue_step1 button:disabled {background:#F0EAE2 !important;border-color:#DED6CD !important;opacity:.7 !important;}
/* Selected mode cards have burgundy background and visible white text. */
div.st-key-mode_personal button,div.st-key-mode_beginner button {min-height:100px;border-radius:15px !important;white-space:pre-wrap;}
div.st-key-mode_personal button[kind="primary"],div.st-key-mode_beginner button[kind="primary"] {background:#8F3547 !important;border:2px solid #8F3547 !important;}
div.st-key-mode_personal button[kind="primary"] *,div.st-key-mode_beginner button[kind="primary"] * {color:#FFFFFF !important;}
div.st-key-mode_personal button[kind="secondary"],div.st-key-mode_beginner button[kind="secondary"] {background:#FFFFFF !important;border:1px solid #DED6CD !important;}
div.st-key-mode_personal button[kind="secondary"] *,div.st-key-mode_beginner button[kind="secondary"] * {color:#292725 !important;}
div[data-testid="stTabs"] button[aria-selected="true"] {color:#8F3547;}
.mm-progress {max-width:340px;margin:8px 0 26px;}
.mm-progress-header {display:flex;justify-content:space-between;margin-bottom:9px;font-size:13px;color:#746D66;}
.mm-progress-header strong {color:#8F3547;}
.mm-bars {display:flex;gap:5px;}
.mm-bar {height:5px;flex:1;border-radius:8px;background:#E8E1DA;}
.mm-bar.active {background:#8F3547;}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return MovieMatch(Path(__file__).resolve().parent / "data")

try:
    model = load_model()
except Exception as e:
    st.error(f"Could not load MovieMatch data: {e}")
    st.stop()

initial = {"step":1,"mode":None,"ratings":{},"level":"Under the Radar","runtime_choice":"120 minutes","count":3,"results":None,"result_mode":"","result_signature":None,"explanations":{},"show_results":False}
for key,value in initial.items():
    if key not in st.session_state: st.session_state[key] = value

def clear_results():
    st.session_state.results = None
    st.session_state.result_mode = ""
    st.session_state.result_signature = None
    st.session_state.explanations = {}
    st.session_state.show_results = False

def select_mode(value):
    if st.session_state.mode != value:
        st.session_state.mode = value
        clear_results()

def next_step(): st.session_state.step = min(3, st.session_state.step + 1)
def previous_step(): st.session_state.step = max(1, st.session_state.step - 1)

def update_rating(mid):
    value = st.session_state[f"rate_{mid}"]
    if value == "Not watched": st.session_state.ratings.pop(mid, None)
    else: st.session_state.ratings[mid] = float(value.split()[0])
    clear_results()

def remove_rating(mid):
    st.session_state.ratings.pop(mid, None)
    if f"rate_{mid}" in st.session_state: st.session_state[f"rate_{mid}"] = "Not watched"
    clear_results()

def adjust_preferences():
    st.session_state.show_results = False
    st.session_state.step = 2
    st.session_state.level_selector = st.session_state.level

def current_signature():
    ratings = st.session_state.ratings if st.session_state.mode == "personal" else {}
    return (st.session_state.mode,st.session_state.level,st.session_state.runtime_choice,st.session_state.count,tuple(sorted(ratings.items())))

def format_runtime(value):
    try:
        if pd.notna(value) and float(value) > 0: return f"{int(float(value))} min"
    except (TypeError, ValueError): pass
    return "Runtime unavailable"

def recommendation_label(source):
    labels = {"Popularity baseline":"General recommendation · Not personalized","Genre-based":"Personalized using genre preferences","Collaborative filtering":"Personalized using similar viewers","Hybrid (CF + genre)":"Personalized using similar viewers and genres"}
    return labels.get(source, source or "Recommendation")

catalog = model.movies.drop_duplicates("movieId").copy()
catalog["display_title"] = catalog["title"].fillna(catalog["clean_title"]).astype(str)
movie_titles = dict(zip(catalog["movieId"].astype(int),catalog["display_title"]))

st.title("🎬 MovieMatch")
st.caption("Discover beyond the obvious · Personalized movie discovery")
tab1,tab2 = st.tabs(["🔎 Discover Movies","🎞️ Your Movie Matches"])

with tab1:
    if st.session_state.show_results and st.session_state.results is not None and st.session_state.result_signature == current_signature():
        st.header("🎬 Your Movie Matches")
        st.caption(f"{st.session_state.level} · {st.session_state.runtime_choice} · {st.session_state.result_mode}")
        if not st.session_state.results:
            st.warning("No movies match your settings. Try another discovery level or a longer viewing time.")
        else:
            st.success("Your recommendations are ready!")
            if len(st.session_state.results) < st.session_state.count:
                st.info(f"Found {len(st.session_state.results)} of {st.session_state.count} requested movies.")
            for i,movie in enumerate(st.session_state.results,1):
                st.markdown(f"### {i}. {movie['clean_title']}")
                st.caption(f"{str(movie.get('genres','')).replace('|',' · ')} · {format_runtime(movie.get('runtime'))} · {recommendation_label(movie.get('recommendation_source',''))}")
            st.markdown("**Open the 🎞️ Your Movie Matches tab above for posters, descriptions, and explanations.**")
        st.button("← Adjust My Preferences",key="adjust_preferences",on_click=adjust_preferences)
    else:
        step = st.session_state.step
        bars = "".join(f'<span class="mm-bar {"active" if i <= step else ""}"></span>' for i in range(1,4))
        st.markdown(f'<div class="mm-progress"><div class="mm-progress-header"><span>Your discovery journey</span><strong>Step {step} of 3</strong></div><div class="mm-bars">{bars}</div></div>',unsafe_allow_html=True)

        if step == 1:
            st.header("How would you like to explore?")
            st.write("Choose your starting point.")
            a,b = st.columns(2,gap="medium")
            with a:
                st.button("❤️  Personalize My Recommendations\nRate movies to match your taste",key="mode_personal",type="primary" if st.session_state.mode == "personal" else "secondary",on_click=select_mode,args=("personal",),use_container_width=True)
            with b:
                st.button("🍿  I'm New to Movies\nNo ratings required",key="mode_beginner",type="primary" if st.session_state.mode == "beginner" else "secondary",on_click=select_mode,args=("beginner",),use_container_width=True)
            if st.session_state.mode == "personal": st.caption("We'll use your ratings to find movies that match your taste.")
            elif st.session_state.mode == "beginner": st.caption("We'll recommend generally well-rated movies without personal ratings.")
            st.write("")
            nav,_ = st.columns([1,3])
            with nav: st.button("Continue →",key="continue_step1",disabled=st.session_state.mode is None,on_click=next_step,use_container_width=True)

        elif step == 2:
            st.header("Set your discovery preferences")
            st.write("Choose the kind of movie experience you're looking for.")
            descriptions = {
                "Popular Picks":"Play it safe — Familiar movies that many viewers have rated and enjoyed.",
                "Under the Radar":"A little adventurous — Discover movies with a following that you might have overlooked.",
                "Hidden Gems":"Explore something different — Find lesser-known movies with some positive audience feedback.",
                "Deep Cuts":"Surprise me — Explore niche movies with limited audience feedback."
            }
            if "level_selector" not in st.session_state: st.session_state.level_selector = st.session_state.level
            st.selectbox("How adventurous are you feeling?",list(DISCOVERY_BANDS),key="level_selector",on_change=update_level)
            st.info(descriptions[st.session_state.level])
            a,b = st.columns(2,gap="medium")
            with a: st.selectbox("Available viewing time",["No time limit","90 minutes","120 minutes","180 minutes"],key="runtime_choice",help="Your time limit is strict. Movies without known runtimes won't appear when a limit is selected.",on_change=clear_results)
            with b: st.slider("Number of recommendations",1,5,key="count",on_change=clear_results)
            st.write("")
            back,forward,_ = st.columns([1,1,3])
            with back: st.button("← Back",on_click=previous_step,use_container_width=True)
            with forward: st.button("Continue →",key="continue_step2",on_click=next_step,use_container_width=True)

        else:
            if st.session_state.mode == "personal":
                st.header("Rate movies you've watched")
                st.write("Rate at least **3 movies**. For better matches, try including movies you liked and disliked.")
                top_ids = model.popularity.sort_values("rating_count",ascending=False).head(20).index
                gallery = catalog[catalog["movieId"].isin(top_ids)].set_index("movieId").reindex(top_ids).dropna(subset=["clean_title"]).reset_index()
                choices = ["Not watched"] + [f"{x/2:g} ★" for x in range(10,0,-1)]
                for start in range(0,len(gallery),5):
                    cols = st.columns(5,gap="small")
                    for col,(_,movie) in zip(cols,gallery.iloc[start:start+5].iterrows()):
                        mid = int(movie["movieId"])
                        with col:
                            poster = movie.get("poster_url")
                            if isinstance(poster,str) and poster.startswith("https://"): st.image(poster,use_container_width=True)
                            else: st.caption("🎞️ Poster unavailable")
                            st.caption(str(movie["clean_title"])[:38])
                            key = f"rate_{mid}"
                            if key not in st.session_state:
                                existing = st.session_state.ratings.get(mid)
                                st.session_state[key] = f"{existing:g} ★" if existing else "Not watched"
                            st.selectbox("Rating",choices,key=key,label_visibility="collapsed",on_change=update_rating,args=(mid,))

                st.divider()
                st.subheader("Can't find a movie you've watched?")
                st.caption("Search by title, select a movie, and add your rating.")
                query = st.text_input("Search movies",placeholder="e.g. Titanic, Inception, The Matrix",key="movie_search_query").strip()
                if len(query) >= 2:
                    matches = catalog[catalog["display_title"].str.contains(query,case=False,regex=False,na=False)].sort_values("display_title").head(15)
                    if matches.empty: st.info("No movies found. Try another title.")
                    else:
                        ids = matches["movieId"].astype(int).tolist()
                        selected = st.selectbox("Select a movie",ids,format_func=lambda mid:movie_titles[mid],key=f"search_{query.lower()}")
                        rating = st.selectbox("Your rating",[None]+[x/2 for x in range(10,0,-1)],format_func=lambda x:"Select a rating..." if x is None else f"{x:g} ★",key="search_rating")
                        if st.button("＋ Add to My Ratings",key="add_search",type="primary",disabled=rating is None):
                            st.session_state.ratings[int(selected)] = float(rating)
                            clear_results()
                            st.success(f"Added {movie_titles[selected]} — {rating:g} ★")
                else: st.caption("Enter at least two characters to search.")

                count_rated = len(st.session_state.ratings)
                st.write(f"**{count_rated} movies rated** · 3 minimum")
                if count_rated >= 3:
                    st.success("You're ready for personalized recommendations.")
                    positive = sum(r >= 3.5 for r in st.session_state.ratings.values())
                    negative = sum(r <= 2.5 for r in st.session_state.ratings.values())
                    if positive < 2 or negative < 1: st.caption("Tip: Including two liked movies and one disliked movie can help describe your taste. This is optional.")
                else: st.warning(f"Rate {3-count_rated} more movie(s) to continue.")
                if count_rated:
                    with st.expander("Review your ratings"):
                        for mid,value in list(st.session_state.ratings.items()):
                            left,right = st.columns([5,1])
                            left.write(f"{movie_titles.get(mid,mid)} — {value:g} ★")
                            right.button("Remove",key=f"remove_{mid}",on_click=remove_rating,args=(mid,))
            else:
                st.header("Your discovery settings")

            if st.session_state.mode == "personal":
                st.divider()
                st.subheader("Your discovery settings")
            st.write(f"**{'Personalized' if st.session_state.mode == 'personal' else 'New to Movies'}** · {st.session_state.level} · {st.session_state.runtime_choice} · {st.session_state.count} movie(s)")
            if st.session_state.mode == "beginner": st.caption("Based on historical ratings, not your personal preferences.")
            st.write("")
            back,generate,_ = st.columns([1,2,2])
            with back: st.button("← Back",on_click=previous_step,use_container_width=True)
            with generate:
                ready = st.session_state.mode == "beginner" or len(st.session_state.ratings) >= 3
                if st.button("✨ Find My Movie Matches",type="primary",disabled=not ready,use_container_width=True):
                    ratings = st.session_state.ratings.copy() if st.session_state.mode == "personal" else None
                    runtime = None if st.session_state.runtime_choice == "No time limit" else int(st.session_state.runtime_choice.split()[0])
                    try:
                        with st.spinner("Finding your movie matches..."):
                            results,method = model.recommend(ratings,level=st.session_state.level,n=st.session_state.count,max_runtime=runtime)
                        st.session_state.results = results.to_dict("records")
                        st.session_state.result_mode = method
                        st.session_state.result_signature = current_signature()
                        st.session_state.explanations = {}
                        st.session_state.show_results = True
                        st.rerun()
                    except Exception as e:
                        clear_results()
                        st.error(f"Could not generate recommendations: {e}")


with tab2:
    if st.session_state.results is None or st.session_state.result_signature != current_signature():
        st.info("Finish the steps in Discover Movies to see your recommendations here.")
    elif not st.session_state.results:
        st.warning("No matches yet. Return to Discover Movies to adjust your preferences.")
    else:
        st.header("🎞️ Your Movie Matches")
        st.caption(st.session_state.result_mode)
        try: api_key = st.secrets.get("GROQ_API_KEY")
        except Exception: api_key = None
        liked = []
        if st.session_state.mode == "personal":
            liked = [str(model.metadata.loc[mid,"clean_title"]) for mid,r in st.session_state.ratings.items() if r >= 3.5 and mid in model.metadata.index]
        for i,movie in enumerate(st.session_state.results,1):
            st.divider()
            left,right = st.columns([1,3],gap="large")
            with left:
                poster = movie.get("poster_url")
                if isinstance(poster,str) and poster.startswith("https://"): st.image(poster,use_container_width=True)
                else: st.caption("🎞️ Poster unavailable")
            with right:
                st.subheader(f"{i}. {movie['clean_title']}")
                st.caption(f"{str(movie.get('genres','')).replace('|',' · ')} · {format_runtime(movie.get('runtime'))}")
                source = movie.get("recommendation_source","")
                st.caption(recommendation_label(source))
                overview = movie.get("overview")
                st.write(overview if isinstance(overview,str) and overview.strip() else "Synopsis unavailable.")
                mid = int(movie["movieId"])
                if st.button("✨ Why this movie?",key=f"explain_{mid}"):
                    with st.spinner("Preparing your explanation..."):
                        explanation,ai = explain(movie,liked if source != "Popularity baseline" else [],api_key)
                    st.session_state.explanations[mid] = (explanation,ai)
                if mid in st.session_state.explanations:
                    explanation,ai = st.session_state.explanations[mid]
                    st.write(explanation)
                    if not ai: st.caption("Template explanation · AI explanation unavailable.")
        st.info("Want different recommendations? Return to Discover Movies and adjust your settings.")
