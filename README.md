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
| Retrieval recall@5 | **85.7%** (6/7 queries with ground truth) |
| Retrieval recall@10 | 85.7% (6/7) |
| Retrieval recall@30 | **100.0%** (7/7) |
| LLM-as-judge — grounded (1–5) | **4.75** |
| LLM-as-judge — relevant (1–5) | **5.00** |
| LLM-as-judge — honest (1–5) | 4.00 |
| End-to-end latency, warm | 7–12 s per request |
| End-to-end latency, first request | ~90 s (BGE-M3 model load + Ollama warmup) |

### Stack upgrade — before vs after

The first version used `multilingual-e5-base` (768d) + `Qwen 2.5 7B`. After running the eval, both pieces were swapped for measurable upgrades:

| | First version | Current |
|---|---|---|
| Embeddings | `intfloat/multilingual-e5-base` (768d) | **`BAAI/bge-m3`** (1024d, #1 on MTEB multilingual leaderboard) |
| LLM | `qwen2.5:7b` | **`qwen3:8b`** (better structured output and grounding) |
| Recall@5 | 71.4% | **85.7%** (+14.3pp) |
| Grounded (judge) | 2.88 | **4.75** (+1.87) |
| Relevant (judge) | 4.25 | **5.00** (+0.75) |

A few things worth pointing out:

- **The retriever finds the right movie in the top 5 for 6/7 queries.** The one miss (`"nostalgic childhood games"` → *Lilo & Stitch* / *Hoppers* outranking *Super Mario Galaxy*) shows the limit of pure semantic similarity — a hybrid BM25 + vector search would close it.
- **The grounded score went from 2.88 to 4.75.** That single number is the bug the first version had: Qwen 2.5 7B drifted to the wrong movie in multi-movie context blocks. Qwen 3 8B stays anchored on the retrieved comments.
- **LLM-as-judge is biased — same family judges itself.** I use the same Qwen 3 8B for generation *and* judging. The absolute numbers are softer than they would be under a stronger external judge (e.g. GPT-4o-mini); the deltas between configurations remain trustworthy.

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
- **Postgres 16 + pgvector 0.8** — vector store with IVFFlat cosine index (`probes=100`)
- **sentence-transformers** + `BAAI/bge-m3` — 1024-dim multilingual embeddings (#1 on MTEB multilingual)
- **Ollama + Qwen 3 8B** — local LLM, JSON-schema-constrained output for `/recommend`
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
python -m scripts.generate_embeddings     # ~2-15 min depending on hardware (one-time)

# API
python -m uvicorn app.api:app --reload --port 8000
# Swagger UI:  http://localhost:8000/docs

# Optional UI (in a separate terminal)
streamlit run ui.py
# Open:  http://localhost:8501
```

Without the LenguaTrends source DB you can still run the stack — `ingest_corpus.py` is a no-op if the target is empty and there's no source. A small synthetic seed script is on the TODO list.

### The UI

`ui.py` is a small Streamlit front-end that talks to the API over HTTP. Two tabs:

- **Trends** — open-ended questions about what users are saying. Picks a language, a top-K, and shows the synthesized answer plus the comments that fed into it.
- **Recommend** — describe what you'd like to watch; get a ranked list with justifications anchored in real comments.

The UI is intentionally a thin layer over the API — same calls you'd make from `curl`, just nicer to demo. Keeping it decoupled means I can swap the front-end without touching the retrieval/generation code.

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
├── ui.py                  # Streamlit front-end (talks to the API over HTTP)
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
- **The recommendation aggregation is naive.** Comments are aggregated per movie by hit-count. A better version would (a) weight by similarity instead of counting, and (b) adjust for sentiment skew so a movie with mostly negative comments isn't recommended when the user asked for "a good one." Both changes are small; I left them out because validating the improvement requires re-running the eval, and the overall demo isn't bottlenecked on this.
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
