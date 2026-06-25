# RAG over LenguaTrends

A multilingual retrieval-augmented question-answering service over **17,130 YouTube comments** about recent movies (English, Spanish, Russian), with two endpoints:

- `POST /trends` — natural-language analysis of what users are saying about a topic
- `POST /recommend` — movie recommendations grounded in the comments themselves

The corpus comes from my undergraduate thesis project, [LenguaTrends](https://github.com/Lux-Mg). This demo extends it: instead of dashboards, you ask questions in any of the three languages and get answers backed by retrieved evidence.

## What it does

```bash
# Trends — analytical summary
curl -X POST http://localhost:8000/trends \
  -H "Content-Type: application/json" \
  -d '{"query": "Что зрители думают про спецэффекты в Аватаре?", "language": "ru"}'

# Recommendation — ranked movies with justification
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Intense action movie with practical stunts"}'
```

Both endpoints share the same retrieval pipeline (`multilingual-e5-base` embeddings + `pgvector` cosine search); only the post-processing and the LLM prompt differ.

## Results

Numbers below come from `eval/eval_rag.py --judge` over the 8 test queries in `eval/queries_test.json` (queries in English, Spanish, and Russian).

| Metric | Value |
|---|---|
| Corpus size | 17,130 comments · 15 movies · 3 languages |
| Retrieval recall@5 | **71.4%** (5/7 queries with ground truth) |
| Retrieval recall@10 | **100.0%** (7/7) |
| Retrieval recall@30 | 100.0% |
| LLM-as-judge — grounded (1–5) | 2.88 |
| LLM-as-judge — relevant (1–5) | 4.25 |
| LLM-as-judge — honest (1–5) | 3.88 |
| End-to-end latency, warm | 7–11 s per request |
| End-to-end latency, first request | ~2 min (e5 model load + Ollama warmup) |

A few things worth pointing out:

- **The retriever always finds the right movie in the top 10.** Recall@10 is 100%; recall@5 drops to 71% because pure semantic similarity occasionally surfaces a thematic sibling above the actual target (e.g. *Send Help* / *Hoppers* outrank *The Super Mario Galaxy Movie* for "nostalgic childhood games"). A hybrid BM25 + vector search would close this gap.
- **The generator is the weak link.** Qwen 2.5 7B occasionally drifts to a secondary movie in the context block. The judge score for *grounded* (2.88) reflects this honestly — *relevant* stays high (4.25) because the answer still addresses the user's intent. A larger LLM (Qwen 2.5 14B+) or a reranker would help; both are Demo #2 concerns.
- **LLM-as-judge is biased — same family judges itself.** I use the same Qwen 2.5 7B for generation *and* judging. The absolute numbers are softer than they would be under a stronger external judge (e.g. GPT-4o-mini); the deltas between prompt versions remain trustworthy.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Host (WSL2 on Windows, RTX 2060 Super 8GB VRAM)                 │
│                                                                  │
│   ┌─────────────────┐    ┌──────────────────┐    ┌────────────┐  │
│   │  Ollama service │◄───│   FastAPI app    │───▶│ Postgres   │  │
│   │  Qwen 2.5 7B    │    │   /trends        │    │ +pgvector  │  │
│   │  port 11434     │    │   /recommend     │    │ port 5434  │  │
│   └─────────────────┘    └──────────────────┘    └────────────┘  │
│                                  │                      ▲        │
│                                  ▼                      │        │
│                          ┌─────────────────┐            │        │
│                          │  e5-base        │            │        │
│                          │  (768d, local)  │            │        │
│                          └─────────────────┘            │        │
│                                                         │        │
│         ┌───────────────────────────────────────────────┘        │
│         │  one-time ingest from thesis Postgres (port 5432)      │
│         │  → copies movies + comments + sentiment + topic        │
│         │  generate_embeddings.py fills the embedding column     │
└─────────┼────────────────────────────────────────────────────────┘
```

A few things worth pointing out:

- **Embeddings model is multilingual.** `intfloat/multilingual-e5-base` handles English, Spanish, and Russian out of the same vector space, so a Russian query retrieves Spanish comments if they're semantically close. This is the whole point — most off-the-shelf RAG demos are English-only.
- **The thesis DB is never touched.** The ingest script reads from the source Postgres (port 5432) and writes to an isolated container on port 5434. The source remains read-only.
- **Two use cases, one retrieval.** Both endpoints call `retrieve()`; what differs is the prompt and the post-processing. `/recommend` adds an aggregation-by-movie step before sending context to the LLM. Keeping the pipeline shared makes the comparison between modes honest.

## Stack

- **Python 3.12** — application code
- **FastAPI + Pydantic + Uvicorn** — API layer
- **Postgres 16 + pgvector 0.3.6** — vector store with IVFFlat cosine index
- **sentence-transformers** + `intfloat/multilingual-e5-base` — 768-dim multilingual embeddings (CPU-friendly)
- **Ollama + Qwen 2.5 7B** — local LLM, JSON-schema-constrained output for `/recommend`
- **Docker Compose** — packages the Postgres container; the API runs in a venv for fast iteration, with a Dockerfile available for deployment

## Running it

You need Ollama on the host with the model pulled (one-time):

```bash
ollama pull qwen2.5:7b
```

Then:

```bash
git clone <this-repo>
cd rag-lenguatrends
cp .env.example .env          # edit SOURCE_DATABASE_URL to point at your data

docker compose up -d           # postgres+pgvector on :5434
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

python -m scripts.ingest_corpus           # ~10s — copies metadata
python -m scripts.generate_embeddings     # ~15-30 min on CPU (one-time)

python -m uvicorn app.api:app --reload --port 8000
# Swagger UI: http://localhost:8000/docs
```

Without the LenguaTrends source DB you can still run the stack — `ingest_corpus.py` is a no-op if the target is empty and there's no source. A small synthetic seed script is on the TODO list.

## Repository layout

```
.
├── app/
│   ├── api.py             # FastAPI: /trends, /recommend, /health
│   ├── rag.py             # retrieve() + trends() + recommend()
│   ├── embeddings.py      # lazy-loaded e5-base wrapper
│   └── db.py              # pg connection context manager
├── scripts/
│   ├── init_schema.sql    # pgvector + tables (runs on first container start)
│   ├── ingest_corpus.py   # source DB → target DB, one-shot
│   └── generate_embeddings.py  # populates the embedding column, resumable
├── eval/
│   ├── queries_test.json  # 8 test queries, 3 languages
│   ├── eval_rag.py        # recall@k + LLM-as-judge
│   └── results.json       # written by eval_rag.py
├── docker-compose.yml     # postgres+pgvector
├── Dockerfile             # for the API container (optional)
├── requirements.txt
├── .env.example
└── README.md
```

## Evaluation methodology

There is no manually labeled gold standard for movie-comment QA, so accuracy claims would be hand-wavy. The eval has two parts:

1. **Recall@k on retrieval.** For each test query annotated with an expected movie, check whether any chunk from that movie is in the top-k retrieved. This isolates the retriever from the generator — if recall@10 is low, no amount of prompting will save the answer.
2. **LLM-as-judge on the generated answer.** Three axes, 1–5: *grounded* (does it cite the retrieved comments?), *relevant* (does it address the query?), *honest* (does it admit when retrieval is weak?). LLM-as-judge is biased, especially when the judge and generator are the same model — the absolute numbers are less interesting than the deltas between prompt versions.

```bash
python -m eval.eval_rag           # recall only
python -m eval.eval_rag --judge   # recall + judge (slower, ~3 min)
```

## Limitations

- **Corpus is small and curated.** 15 movies, all recent blockbusters scraped from YouTube. Don't expect long-tail or niche recommendations.
- **No multi-turn memory.** Each call is independent — there's no chat state. Adding it is a Demo #2 concern (LangGraph).
- **Embeddings are static.** No re-embedding job; if new comments land in the source, you re-run `generate_embeddings.py`.
- **The recommendation aggregation is naive.** Comments are aggregated per movie by hit-count; a better version would weight by similarity and adjust for sentiment skew.
- **The `'other'`/`'unsupported'` language bucket is excluded from retrieval filtering** (~9% of the source corpus). Those comments are in the index, just not surfaced via `language=ru/es/en`.
- **LLM-as-judge is the same family as the generator.** A different judge (e.g. GPT-4o-mini) would give a more honest read.

## What I learned

In rough order of usefulness:

1. **`pgvector` over Postgres is the cheap, boring, correct default for any RAG that doesn't need >1M vectors.** Adding it to an existing Postgres is a single `CREATE EXTENSION`. No new infra, no separate vector DB, joins with relational metadata just work.
2. **Shared retrieval, divergent prompts beat duplicated pipelines.** Building `/trends` and `/recommend` as two callers of one `retrieve()` made the code half the size and forced honest design choices.
3. **The e5 family wants prefixes.** `query:` for queries, `passage:` for documents. Skipping the prefix degrades retrieval quietly — there's no error, just worse results. Worth a comment in the code.
4. **Multilingual embedding spaces are real.** A Russian query landing English-language comments isn't a bug, it's the design.
5. **Schema-constrained JSON output keeps the API contract trustworthy.** `format={...}` in Ollama means the recommender always returns the same shape — no defensive parsing in the FastAPI layer.

---

Built by Luis Mendoza · [GitHub](https://github.com/Lux-Mg) · 2026
