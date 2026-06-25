"""Wrapper del modelo de embeddings. Carga lazy (al primer uso)."""
import os
from sentence_transformers import SentenceTransformer

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_query(text: str) -> list[float]:
    """e5 espera prefijo 'query: ' para queries (no 'passage: ')."""
    model = get_model()
    vec = model.encode(
        [f"query: {text}"],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vec.tolist()
