"""
Genera embeddings (multilingual-e5-base, 768d) para todos los comentarios
sin embedding. Reanudable: si se corta, vuelve a correr y sigue.

Convención e5: prefijos "passage: " para documentos, "query: " para queries.
"""
import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
MODEL_NAME   = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-base")
BATCH_SIZE   = 64

if not DATABASE_URL:
    print("Falta DATABASE_URL en .env", file=sys.stderr)
    sys.exit(1)


def main():
    print(f"Cargando modelo {MODEL_NAME}...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Dim del modelo: {dim}")

    conn = psycopg2.connect(DATABASE_URL)

    # Total pendiente
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM comments WHERE embedding IS NULL;")
        pending = cur.fetchone()[0]
    print(f"Pendientes de embedar: {pending}")
    if pending == 0:
        return

    pbar = tqdm(total=pending, desc="embeddings")

    while True:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, text
                FROM comments
                WHERE embedding IS NULL
                ORDER BY id
                LIMIT %s;
            """, (BATCH_SIZE,))
            rows = cur.fetchall()

        if not rows:
            break

        ids   = [r[0] for r in rows]
        texts = [f"passage: {r[1]}" for r in rows]
        vecs  = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)

        # UPDATE batch
        payload = [(vec.tolist(), id_) for vec, id_ in zip(vecs, ids)]
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE comments SET embedding = %s::vector WHERE id = %s;",
                payload,
            )
        conn.commit()
        pbar.update(len(rows))

    pbar.close()

    # Crea el índice vectorial cuando todo está poblado
    print("Creando índice IVFFlat...")
    with conn.cursor() as cur:
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_comments_embedding
            ON comments USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        conn.commit()
    print("Listo.")
    conn.close()


if __name__ == "__main__":
    main()
