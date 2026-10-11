import ast
import pandas as pd

from .config import DATA_DIR, get_conn, get_model

RAW_CSV = DATA_DIR / "billboard_top_100_1946_2022_lyrics-edited.csv"

def load_songs():
    df = pd.read_csv(RAW_CSV)
    df["artist"] = df["Artist Names"].apply(lambda s: ", ".join(ast.literal_eval(s)))
    df["lyrics"] = df["Lyrics"].apply(lambda s: " ".join(ast.literal_eval(s)))
    df["title"] = df["Song"].str.replace(
        r"\s+-\s+(\d{4}\s+)?(remaster(ed)?|single version|mono|stereo).*$",
        "", regex=True, case=False,
    )
    df = df.rename(columns={"Hot100 Ranking Year": "year", "Hot100 Rank": "rank"})
    df = df[["title", "artist", "year", "rank", "lyrics"]]
    df = df[df["lyrics"].str.split().str.len() >= 30]
    df = df.drop_duplicates(subset=["title", "artist", "year"]).reset_index(drop=True)
    df.to_csv(DATA_DIR / "songs_clean.csv", index=False)  # used by word_stats
    return df

def chunk_text(text, size=150, overlap=30):
    words = text.split()
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + size]
        if len(piece) < 30 and chunks:
            chunks[-1] += " " + " ".join(piece)
            break
        chunks.append(" ".join(piece))
    return chunks

def build_chunks(df):
    rows = []
    for song_id, row in df.iterrows():
        for i, chunk in enumerate(chunk_text(row["lyrics"])):
            rows.append((song_id, i, row["title"], row["artist"],
                         int(row["year"]), int(row["rank"]), chunk))
    return rows

def main():
    df = load_songs()
    rows = build_chunks(df)
    print(f"{len(df)} songs -> {len(rows)} chunks")

    model = get_model()
    vectors = model.encode([r[6] for r in rows], batch_size=64,
                           show_progress_bar=True, normalize_embeddings=True)

    conn = get_conn()
    conn.execute("DROP TABLE IF EXISTS chunks")
    conn.execute("""
        CREATE TABLE chunks (
            id SERIAL PRIMARY KEY, song_id INT, chunk_index INT,
            title TEXT, artist TEXT, year INT, rank INT,
            chunk_text TEXT, embedding vector(384)
        )
    """)
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO chunks
               (song_id, chunk_index, title, artist, year, rank, chunk_text, embedding)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
            [r + (v,) for r, v in zip(rows, vectors)],
        )
    conn.commit()
    print("Inserted", conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0], "chunks")

if __name__ == "__main__":
    main()