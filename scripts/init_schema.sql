-- Schema inicial de la DB del demo
-- Se ejecuta automáticamente al crear el contenedor (volume init)

CREATE EXTENSION IF NOT EXISTS vector;

-- Películas (copiadas desde la tesis)
CREATE TABLE IF NOT EXISTS movies (
    id           INTEGER PRIMARY KEY,
    title        VARCHAR(255) NOT NULL,
    title_es     VARCHAR(255),
    title_ru     VARCHAR(255),
    tmdb_id      INTEGER UNIQUE,
    type         VARCHAR(50)
);

-- Comentarios con sentiment + topic ya resuelto (denormalizado: una sola tabla
-- en lugar de 4 joins de la tesis, porque acá no muta y se consulta mucho)
CREATE TABLE IF NOT EXISTS comments (
    id              INTEGER PRIMARY KEY,
    movie_id        INTEGER REFERENCES movies(id),
    text            TEXT NOT NULL,
    language        VARCHAR(20) NOT NULL,
    sentiment       VARCHAR(20),     -- positive / negative / neutral / NULL
    sentiment_score REAL,
    topic_label     VARCHAR(255),
    topic_prob      REAL,
    embedding       vector(1024)     -- BAAI/bge-m3
);

-- Índices
CREATE INDEX IF NOT EXISTS idx_comments_movie    ON comments(movie_id);
CREATE INDEX IF NOT EXISTS idx_comments_lang     ON comments(language);
CREATE INDEX IF NOT EXISTS idx_comments_sent     ON comments(sentiment);

-- Índice vectorial (IVFFlat con cosine distance). Lists=100 está bien para ~17k filas.
-- Se crea DESPUÉS de cargar los embeddings; lo deja pendiente el script de ingest.
