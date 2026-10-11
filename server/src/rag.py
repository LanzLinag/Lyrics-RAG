import ollama

from .config import LLM_MODEL
from .search import retrieve

SYSTEM_PROMPT = (
    "You answer questions about song lyrics using ONLY the numbered excerpts provided.\n"
    "Rules:\n"
    "1. Every claim must name the song it comes from, like \"Title\" by Artist (Year).\n"
    "2. Only describe a song if its excerpt supports what you say.\n"
    "3. Ignore excerpts that are not relevant to the question.\n"
    "4. Be specific. Avoid generic summaries.\n"
    "5. Paraphrase instead of quoting lyrics at length.\n"
    "6. If the excerpts do not answer the question, say so plainly."
)

def ask(question, k=8, year_from=None, year_to=None, min_similarity=0.35):
    hits = [h for h in retrieve(question, k, year_from, year_to) if h[5] >= min_similarity]
    if not hits:
        return "I couldn't find relevant lyrics for that question.", []

    context = "\n\n".join(
        f'[{i+1}] "{title}" by {artist} ({year})\n{text}'
        for i, (_, title, artist, year, text, _) in enumerate(hits)
    )
    user = f"Lyric excerpts:\n\n{context}\n\nQuestion: {question}"

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user}],
        options={"temperature": 0.1, "num_ctx": 8192},
    )
    return response["message"]["content"], hits

if __name__ == "__main__":
    answer, hits = ask("Which songs describe being separated from someone by distance, and how?")
    print(answer)
    print("\nSources:")
    for _, title, artist, year, _, sim in hits:
        print(f"- {title} - {artist} ({year}) [{sim:.2f}]")