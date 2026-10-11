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

1. **Get the dataset** (not included, for copyright reasons): TODO: add the dataset source link.
   Save the CSV as `server/data/billboard_top_100_1946_2022_lyrics-edited.csv`.
2. **Create a virtual environment and install packages**
```
   cd server
   python -m venv .venv
   .venv\Scripts\activate        # Windows (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
```
3. **Create your settings file:** copy `.env.example` to `.env`.
4. **Start