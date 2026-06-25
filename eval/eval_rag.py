"""
Evaluación del RAG. Dos métricas:

1. Recall@k del retrieval: para cada query con `expected_movies`, ¿aparece
   alguna entre los top-k retrieved?
2. (Opcional) LLM-as-judge sobre las respuestas generadas — flag --judge.
"""
import os
import sys
import json
import argparse
from pathlib import Path
from dotenv import load_dotenv
from ollama import Client

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

load_dotenv(dotenv_path=ROOT / ".env")

from app.rag import retrieve, trends, recommend

OLLAMA_HOST  = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")

JUDGE_PROMPT = """You are evaluating a movie-comment RAG system.

User query: "{query}"

System answer: "{answer}"

Rate the answer on 3 dimensions (1-5):
- grounded: does it cite or paraphrase concrete comments?
- relevant: does it address the user's query?
- honest: does it acknowledge limits when retrieval is weak?

Return ONLY valid JSON: {{"grounded": N, "relevant": N, "honest": N}}"""


def recall_at_k(queries: list[dict], k_values=(5, 10, 30)) -> dict:
    print(f"\n=== Recall@k sobre {len(queries)} queries ===")
    hits = {k: 0 for k in k_values}
    eligible = 0

    for q in queries:
        expected = set(q.get("expected_movies", []))
        if not expected:
            continue
        eligible += 1

        docs = retrieve(q["query"], top_k=max(k_values), language=q.get("language"))
        retrieved_titles = [d["movie_title"] for d in docs]

        for k in k_values:
            top_titles = set(retrieved_titles[:k])
            if expected & top_titles:
                hits[k] += 1

        top5 = retrieved_titles[:5]
        print(f"  [{q['id']:35s}] expected={list(expected)} | top5={top5}")

    print(f"\nResults ({eligible} queries with ground truth):")
    for k in k_values:
        rate = hits[k] / eligible if eligible else 0
        print(f"  Recall@{k:<2} = {hits[k]}/{eligible} = {rate:.1%}")

    return {f"recall@{k}": hits[k] / eligible if eligible else 0 for k in k_values}


def llm_judge(queries: list[dict]) -> dict:
    print(f"\n=== LLM-as-judge sobre {len(queries)} queries ===")
    llm = Client(host=OLLAMA_HOST)
    scores = []

    for q in queries:
        if q.get("use_case") == "trends":
            result = trends(q["query"], top_k=20, language=q.get("language"))
            answer = result["answer"]
        else:
            result = recommend(q["query"], top_k=30, language=q.get("language"))
            answer = json.dumps(result.get("recommendations", []))

        resp = llm.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(query=q["query"], answer=answer)}],
            format={
                "type": "object",
                "properties": {
                    "grounded": {"type": "integer", "minimum": 1, "maximum": 5},
                    "relevant": {"type": "integer", "minimum": 1, "maximum": 5},
                    "honest":   {"type": "integer", "minimum": 1, "maximum": 5},
                },
                "required": ["grounded", "relevant", "honest"],
            },
        )
        s = json.loads(resp["message"]["content"])
        s["query_id"] = q["id"]
        scores.append(s)
        print(f"  [{q['id']:35s}] grounded={s['grounded']} relevant={s['relevant']} honest={s['honest']}")

    avg = {k: sum(s[k] for s in scores) / len(scores) for k in ("grounded", "relevant", "honest")}
    print("\nPromedios:")
    for k, v in avg.items():
        print(f"  {k:10s} = {v:.2f}/5")
    return avg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true", help="Ejecutar LLM-as-judge (lento)")
    args = parser.parse_args()

    queries_path = ROOT / "eval" / "queries_test.json"
    queries = json.loads(queries_path.read_text())

    results = {"recall": recall_at_k(queries)}
    if args.judge:
        results["judge"] = llm_judge(queries)

    out = ROOT / "eval" / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
