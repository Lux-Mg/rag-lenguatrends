"""
Copia las pelis y comentarios procesados desde la DB de la tesis a la DB del demo.
La columna `embedding` queda NULL — la llena generate_embeddings.py después.

One-shot: si la nueva DB ya tiene filas, salta sin tocar nada.
"""
import os
import sys
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import execute_values

load_dotenv()

SOURCE_URL = os.getenv("SOURCE_DATABASE_URL")
TARGET_URL = os.getenv("DATABASE_URL")

if not SOURCE_URL or not TARGET_URL:
    print("Faltan SOURCE_DATABASE_URL o DATABASE_URL en .env", file=sys.stderr)
    sys.exit(1)


def ingest():
    src = psycopg2.connect(SOURCE_URL)
    tgt = psycopg2.connect(TARGET_URL)

    with tgt.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM comments;")
        existing = cur.fetchone()[0]
        if existing > 0:
            print(f"DB target ya tiene {existing} comentarios — salgo sin hacer nada.")
            return

    # Pelis
    with src.cursor() as scur:
        scur.execute("""
            SELECT id, title, title_es, title_ru, tmdb_id, type
            FROM media_entities
            ORDER BY id;
        """)
        movies = scur.fetchall()

    with tgt.cursor() as tcur:
        execute_values(
            tcur,
            "INSERT INTO movies (id, title, title_es, title_ru, tmdb_id, type) VALUES %s;",
            movies,
        )
    tgt.commit()
    print(f"Insertadas {len(movies)} pelis.")

    # Comentarios (con sentiment + topic via join)
    with src.cursor(name="comments_cursor") as scur:
        scur.itersize = 2000
        scur.execute("""
            SELECT
                c.id,
                c.media_entity_id,
                c.text,
                c.language,
                sr.label,
                sr.score,
                tr.topic_label,
                tr.probability
            FROM comments c
            INNER JOIN sentiment_results sr ON sr.comment_id = c.id
            LEFT  JOIN topic_results     tr ON tr.comment_id = c.id
            WHERE c.processed = TRUE
              AND c.media_entity_id IS NOT NULL
            ORDER BY c.id;
        """)

        batch = []
        total = 0
        for row in scur:
            batch.append(row)
            if len(batch) >= 1000:
                _flush(tgt, batch)
                total += len(batch)
                batch = []
                print(f"  ... {total} comentarios", flush=True)

        if batch:
            _flush(tgt, batch)
            total += len(batch)

    print(f"Insertados {total} comentarios.")
    src.close()
    tgt.close()


def _flush(conn, batch):
    with conn.cursor() as cur:
        execute_values(
            cur,
            """
            INSERT INTO comments
              (id, movie_id, text, language, sentiment, sentiment_score, topic_label, topic_prob)
            VALUES %s;
            """,
            batch,
        )
    conn.commit()


if __name__ == "__main__":
    ingest()
