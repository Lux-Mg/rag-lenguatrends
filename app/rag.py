"""
Núcleo del RAG: retrieve + generate.

Dos use cases sobre el mismo pipeline:
  - trends(): resumen analítico de qué opinan los usuarios sobre un tema
  - recommend(): pelis ranqueadas que matchean una descripción
"""
import os
import json
from collections import defaultdict
from ollama import Client

from app.db import get_conn
from app.embeddings import embed_query

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

_llm = Client(host=OLLAMA_HOST)


# ---------- retrieval ----------

def retrieve(
    query_text: str,
    top_k: int = 30,
    language: str | None = None,
    movie_id: int | None = None,
    sentiment: str | None = None,
) -> list[dict]:
    """Búsqueda semántica con filtros estructurados opcionales."""
    qvec = embed_query(query_text)

    where = ["embedding IS NOT NULL"]
    params: list = []
    if language:
        where.append("c.language = %s")
        params.append(language)
    if movie_id is not None:
        where.append("c.movie_id = %s")
        params.append(movie_id)
    if sentiment:
        where.append("c.sentiment = %s")
        params.append(sentiment)
    where_sql = " AND ".join(where)

    sql = f"""
        SELECT
            c.id,
            c.text,
            c.language,
            c.sentiment,
            c.topic_label,
            m.id      AS movie_id,
            m.title   AS movie_title,
            1 - (c.embedding <=> %s::vector) AS similarity
        FROM comments c
        JOIN movies m ON m.id = c.movie_id
        WHERE {where_sql}
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s;
    """
    params = [qvec] + params + [qvec, top_k]

    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "id": r[0],
            "text": r[1],
            "language": r[2],
            "sentiment": r[3],
            "topic": r[4],
            "movie_id": r[5],
            "movie_title": r[6],
            "similarity": float(r[7]),
        }
        for r in rows
    ]


# ---------- fase 1: tendencias ----------

_TRENDS_PROMPT = """You are analyzing user comments scraped from YouTube reviews of recent movies.

User question: "{query}"

Below are the {n} most relevant comments retrieved from a multilingual corpus. Each line starts with [Movie · Language · Sentiment · similarity].

<comments>
{comments_block}
</comments>

Hard rules:
1. Answer ONLY about movies that appear in the comments block above. Never mention a movie not listed there.
2. Anchor at least one statement to a specific movie title from the block, e.g. "Comments about <Movie X> say...".
3. Quote one short comment verbatim (max 2 sentences) to support your synthesis. Translate it to English in parentheses if it's not English.
4. Write 3-5 sentences. Synthesize — do NOT list comments one by one.
5. Reply in English."""


def trends(query: str, top_k: int = 30, language: str | None = None) -> dict:
    docs = retrieve(query, top_k=top_k, language=language)
    if not docs:
        return {"answer": "No matching comments found.", "sources": []}

    block = "\n\n".join(
        f"[{d['movie_title']} · {d['language']} · {d['sentiment']} · sim={d['similarity']:.2f}]\n{d['text']}"
        for d in docs
    )
    prompt = _TRENDS_PROMPT.format(query=query, n=len(docs), comments_block=block)

    resp = _llm.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    return {
        "answer": resp["message"]["content"],
        "sources": docs,
    }


# ---------- fase 2: recomendación ----------

_RECOMMEND_PROMPT = """You are a movie recommender. The user describes what they want; you pick from a fixed catalog of recent movies based on real user comments.

User wants: "{query}"

Below are aggregated stats per movie based on the {n} most relevant comments retrieved from a multilingual corpus. For each movie you have: how many of the top matches it received, the dominant sentiment, sample comments, and topics.

<movies>
{movies_block}
</movies>

Pick UP TO 3 movies that best fit the user's request. For each, give a one-sentence reason grounded in what the comments actually say (paraphrase, do not fabricate). If nothing in the corpus matches well, say so honestly and recommend at most 1.

Return your answer as JSON with this exact shape:
{{
  "recommendations": [
    {{"title": "...", "reason": "..."}}
  ],
  "caveats": "..."
}}"""


def recommend(query: str, top_k: int = 50, language: str | None = None) -> dict:
    docs = retrieve(query, top_k=top_k, language=language)
    if not docs:
        return {"recommendations": [], "caveats": "No matches in corpus.", "sources": []}

    # Agrego por peli
    by_movie: dict[int, list[dict]] = defaultdict(list)
    for d in docs:
        by_movie[d["movie_id"]].append(d)

    # Ordeno por cantidad de hits (proxy de relevancia agregada)
    ranked = sorted(by_movie.items(), key=lambda kv: len(kv[1]), reverse=True)[:6]

    blocks = []
    for _, comments in ranked:
        title = comments[0]["movie_title"]
        sent_counts = defaultdict(int)
        for c in comments:
            sent_counts[c["sentiment"] or "unknown"] += 1
        sent_str = ", ".join(f"{k}={v}" for k, v in sent_counts.items())
        samples = "\n".join(f"  - {c['text'][:180]}" for c in comments[:3])
        topics = ", ".join({c["topic"] for c in comments if c["topic"]}) or "—"
        blocks.append(
            f"{title}\n  hits: {len(comments)} | sentiment: {sent_str}\n  topics: {topics}\n  samples:\n{samples}"
        )
    movies_block = "\n\n".join(blocks)

    prompt = _RECOMMEND_PROMPT.format(query=query, n=len(docs), movies_block=movies_block)

    resp = _llm.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format={
            "type": "object",
            "properties": {
                "recommendations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "title":  {"type": "string"},
                            "reason": {"type": "string"},
                        },
                        "required": ["title", "reason"],
                    },
                },
                "caveats": {"type": "string"},
            },
            "required": ["recommendations", "caveats"],
        },
    )
    parsed = json.loads(resp["message"]["content"])
    parsed["sources"] = docs
    return parsed
