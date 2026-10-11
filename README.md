# Billboard Lyrics RAG

Semantic search and question answering over about 6,000 Billboard Hot 100 songs (1946-2022),
using sentence embeddings, PostgreSQL/pgvector, and a local LLM (Ollama).


## What it does
- **Ask about lyrics (RAG):** retrieves relevant lyric excerpts and answers from them, refusing when they don't contain the answer
- **Find similar songs:** semantic search by theme or feeling, with year filtering
- **Exact word search / word statistics:** SQL and pandas for questions LLMs can't answer reliably (counts, "which songs contain X")

The app shows only song titles, artists, and years, and the LLM is told to paraphrase instead of quoting lyrics.

## Stack
Python, pandas, sentence-transformers (all-MiniLM-L6-v2), PostgreSQL + pgvector (Docker),
Ollama (llama3.1:8b), Streamlit

## How it works
```
CSV -> clean -> chunk (~150 words, 30 overlap) -> embed -> pgvector
Question -> embed -> retrieve closest chunks -> prompt -> LLM answer
```

## Prerequisites
Python 3.10+, Git, [Docker Desktop](https://www.docker.com/products/docker-desktop/), and [Ollama](https://ollama.com).
Docker Desktop and Ollama must be running. About 16 GB of RAM is recommended for the default LLM.

## Run it
All commands run from the `server/` folder unless noted.

1. **Get the dataset** (not included, for copyright reasons): [Billboard's Top 100 Songs 1946 to 2022 (lyrics) on Kaggle](https://www.kaggle.com/datasets/rhaamrozenberg/billboards-top-100-song-1946-to-2022-lyrics).
   Save the CSV as `server/data/billboard_top_100_1946_2022_lyrics.csv`.
2. **Create a virtual environment and install packages**
```
   cd server
   python -m venv .venv
   .venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
```
3. **Create your settings file:** copy `.env.example` to `.env`.
4. **Start the database:** `docker compose up -d`
5. **Download the LLM:** `ollama pull llama3.1:8b`
6. **Build the database** (several minutes, one time): `python -m src.ingest`
7. **Start the app:** `streamlit run app.py`

## Repo layout
- `server/src/` - application code (ingest, search, RAG)
- `server/app.py` - Streamlit interface
- `server/notebooks/` - exploration notebooks (outputs cleared); `src/` is the source of truth

## Design choices
- **pgvector:** vectors sit next to metadata, so year filtering is plain SQL
- **Chunking:** the embedding model reads about 256 tokens, so songs are split into overlapping chunks
- **Hybrid tools:** counting and exact-word queries go to SQL/pandas, not the LLM

## What I tried and learned
- **Retrieval vs. generation:** I checked the retrieved sources first to see which stage was failing
- **Model swap:** moving from llama3.2 to llama3.1:8b fixed a misdescribed song, but the sources were identical, so retrieval was unchanged
- **Whole-dataset questions:** RAG correctly refuses questions like "most used lyrics" because it only sees a few chunks

## Limitations
- A small local LLM sometimes includes loosely related songs (for example, a song about music in an answer about distance)
- Retrieval is imperfect on vague, subjective themes, and similarity scores of about 0.5 to 0.6 are typical
- Answers depend on only the top retrieved songs, so broad trend questions are weak

## Related
[Billboard Lyrics Analysis](https://github.com/LanzLinag/Lyrics-Analysis): word-frequency analysis of the same dataset
