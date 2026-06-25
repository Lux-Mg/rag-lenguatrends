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
    """BGE-M3 no usa prefijos (a diferencia de e5)."""
    model = get_model()
    vec = model.encode(
        [text],
        normalize_embeddings=True,
        show_progress_bar=False,
    )[0]
    return vec.tolist()
