import streamlit as st

from src.rag import ask
from src.search import retrieve, songs_with_word, word_stats

st.set_page_config(page_title="Billboard Lyrics RAG", page_icon="🎵")
st.title("🎵 Billboard Lyrics RAG")
st.caption("Semantic search and question answering over Billboard Hot 100 songs (1946-2022)")

mode = st.sidebar.radio(
    "Mode",
    ["Ask about lyrics (RAG)", "Find similar songs", "Exact word search", "Word statistics"],
)

use_years = st.sidebar.checkbox("Filter by year", value=False)
year_from = year_to = None
if use_years:
    year_from, year_to = st.sidebar.slider("Years", 1946, 2022, (1980, 1999))


def show_songs(rows):
    for title, artist, year in rows:
        st.write(f"- **{title}** - {artist} ({year})")


if mode == "Ask about lyrics (RAG)":
    st.write("Best for specific questions, like *which songs describe being separated by distance, and how?*")
    q = st.text_input("Your question")
    if st.button("Ask") and q:
        with st.spinner("Searching lyrics and writing an answer..."):
            answer, hits = ask(q, year_from=year_from, year_to=year_to)
        st.markdown(answer)
        if hits:
            with st.expander("Songs retrieved"):
                show_songs([(h[1], h[2], h[3]) for h in hits])

elif mode == "Find similar songs":
    st.write("Search by meaning, not exact words.")
    q = st.text_input("Describe a theme or feeling")
    k = st.slider("Number of songs", 3, 15, 5)
    if st.button("Search") and q:
        hits = retrieve(q, k=k, year_from=year_from, year_to=year_to)
        for _, title, artist, year, _, sim in hits:
            st.write(f"- **{title}** - {artist} ({year})  `{sim:.2f}`")

elif mode == "Exact word search":
    st.write("Lists every song containing a word (whole-word match).")
    w = st.text_input("Word")
    if st.button("Find songs") and w:
        rows = songs_with_word(w, year_from=year_from, year_to=year_to)
        st.write(f"{len(rows)} songs found (showing up to 50)")
        show_songs(rows)

else:
    st.write("Exact counts across all songs, computed with code, not the LLM.")
    w = st.text_input("Word")
    if st.button("Count") and w:
        s = word_stats(w)
        c1, c2, c3 = st.columns(3)
        c1.metric("Songs containing it", s["songs_containing"])
        c2.metric("% of songs", f'{s["pct_of_songs"]}%')
        c3.metric("Total occurrences", s["total_occurrences"])