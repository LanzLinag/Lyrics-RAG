import re
import pandas as pd

from .config import DATA_DIR, get_conn, get_model

def retrieve(query, k=5, year_from=None, year_to=None):
    """Semantic search: best chunk per song, most similar first."""
    q_vec = get_model().encode(query, normalize_embeddings=True)
    where, params = [], []
    if year_from is not None:
        where.append("year >= %s"); params.append(year_from)
    if year_to is not None:
        where.append("year <= %s"); params.append(year_to)
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT DISTINCT ON (song_id) song_id, title, artist, year, chunk_text,
               1 - (embedding <=> %s) AS similarity
        FROM chunks {where_sql}
        ORDER BY song_id, embedding <=> %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, [q_vec] + params + [q_vec]).fetchall()
    rows.sort(key=lambda r: r[5], reverse=True)
    return rows[:k]

def songs_with_word(word, year_from=None, year_to=None, limit=50):
    """Exact-word search: every song containing the word."""
    sql = """
        SELECT DISTINCT title, artist, year FROM chunks
        WHERE chunk_text ~* %s
          AND (%s::int IS NULL OR year >= %s)
          AND (%s::int IS NULL OR year <= %s)
        ORDER BY year LIMIT %s
    """
    with get_conn() as conn:
        return conn.execute(
            sql, (rf"\m{re.escape(word)}\M", year_from, year_from, year_to, year_to, limit)
        ).fetchall()

def word_stats(word):
    """Exact counts over all songs (no chunk overlap, no LLM)."""
    songs = pd.read_csv(DATA_DIR / "songs_clean.csv")
    counts = songs["lyrics"].str.lower().str.count(rf"\b{re.escape(word.lower())}\b")
    return {
        "word": word,
        "total_occurrences": int(counts.sum()),
        "songs_containing": int((counts > 0).sum()),
        "total_songs": len(songs),
        "pct_of_songs": round((counts > 0).mean() * 100, 1),
    }