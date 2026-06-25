"""FastAPI: dos endpoints, uno por fase."""
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import trends, recommend

app = FastAPI(
    title="RAG LenguaTrends",
    description="RAG conversacional + recomendador sobre comentarios multilingües de YouTube.",
    version="0.1.0",
)


class TrendsRequest(BaseModel):
    query: str = Field(..., examples=["¿Qué opinan los rusos sobre Avatar?"])
    language: str | None = Field(None, examples=["ru", "es", "en"])
    top_k: int = Field(30, ge=1, le=100)


class RecommendRequest(BaseModel):
    query: str = Field(..., examples=["Algo de acción intensa con buena banda sonora"])
    language: str | None = None
    top_k: int = Field(50, ge=1, le=100)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/trends")
def trends_endpoint(req: TrendsRequest):
    return trends(req.query, top_k=req.top_k, language=req.language)


@app.post("/recommend")
def recommend_endpoint(req: RecommendRequest):
    return recommend(req.query, top_k=req.top_k, language=req.language)
