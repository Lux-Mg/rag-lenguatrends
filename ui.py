"""
Streamlit UI para el RAG. Habla con el API por HTTP (no importa rag.py directo
para no duplicar la carga del modelo de embeddings).

Correr:  streamlit run ui.py
"""
import os
import requests
import streamlit as st
import pandas as pd

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000")

MOVIES = [
    "Mickey 17", "Thunderbolts*", "Avatar: Fire and Ash", "Lilo & Stitch",
    "Project Hail Mary", "Sinners", "Scream 7",
    "The Super Mario Galaxy Movie", "Mission: Impossible - The Final Reckoning",
    "Hoppers", "GOAT", "Send Help", "Crime 101", "Pretty Lethal",
    "Mike & Nick & Nick & Alice",
]

LANG_OPTIONS = {
    "Todos los idiomas": None,
    "Inglés": "en",
    "Español": "es",
    "Ruso": "ru",
}

st.set_page_config(
    page_title="RAG LenguaTrends",
    page_icon="🎬",
    layout="wide",
)

# --- sidebar ---
with st.sidebar:
    st.markdown("## RAG LenguaTrends")
    st.markdown(
        "Pregunta sobre **17,130 comentarios** de YouTube sobre películas "
        "recientes, en inglés, español y ruso."
    )

    health_ok = False
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        health_ok = r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        pass

    if health_ok:
        st.success(f"API activa en {API_URL}")
    else:
        st.error(f"API no responde en {API_URL}")
        st.caption("Arrancalo con `uvicorn app.api:app --port 8000`")

    st.markdown("### Películas en el corpus")
    st.caption(", ".join(MOVIES))

    st.markdown("---")
    st.caption("Stack: pgvector · e5-base · Qwen 2.5 7B")
    st.caption("Demo #1 del portfolio AI/ML.")


# --- tabs ---
tab_trends, tab_recommend = st.tabs(["📊 Trends", "🎯 Recomendar"])

# ---------------- TRENDS ----------------
with tab_trends:
    st.markdown("### ¿Qué opina la gente sobre...?")
    st.caption(
        "Hacé una pregunta abierta. El sistema busca los comentarios más "
        "parecidos semánticamente y el LLM los sintetiza."
    )

    col_q, col_l, col_k = st.columns([4, 1, 1])
    with col_q:
        q_trends = st.text_input(
            "Tu pregunta",
            value="What do viewers think of the visual effects in Avatar?",
            key="q_trends",
            label_visibility="collapsed",
            placeholder="Ejemplo: ¿Qué dicen los rusos sobre Mickey 17?",
        )
    with col_l:
        lang_trends = st.selectbox("Idioma", LANG_OPTIONS.keys(), key="lang_trends")
    with col_k:
        k_trends = st.slider("Top K", 5, 50, 20, key="k_trends")

    if st.button("Consultar", key="btn_trends", type="primary", use_container_width=True):
        if not q_trends.strip():
            st.warning("Escribí una pregunta primero.")
        elif not health_ok:
            st.error("El API no está disponible.")
        else:
            with st.spinner("Buscando comentarios + generando análisis..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/trends",
                        json={
                            "query": q_trends,
                            "language": LANG_OPTIONS[lang_trends],
                            "top_k": k_trends,
                        },
                        timeout=180,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as e:
                    st.error(f"Error: {e}")
                    st.stop()

            st.markdown("#### Respuesta")
            st.write(data["answer"])

            with st.expander(f"Ver los {len(data['sources'])} comentarios usados", expanded=False):
                df = pd.DataFrame([
                    {
                        "Película": s["movie_title"],
                        "Idioma": s["language"],
                        "Sentiment": s["sentiment"],
                        "Sim": round(s["similarity"], 3),
                        "Comentario": s["text"][:200] + ("…" if len(s["text"]) > 200 else ""),
                    }
                    for s in data["sources"]
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)

# ---------------- RECOMMEND ----------------
with tab_recommend:
    st.markdown("### ¿Qué peli ver?")
    st.caption(
        "Describí qué te gustaría ver. El sistema busca comentarios que "
        "encajen y rankea las películas correspondientes."
    )

    col_q, col_l, col_k = st.columns([4, 1, 1])
    with col_q:
        q_rec = st.text_input(
            "Tu deseo",
            value="Algo de acción intensa con buenos efectos",
            key="q_rec",
            label_visibility="collapsed",
            placeholder="Ejemplo: Una película de terror con vampiros",
        )
    with col_l:
        lang_rec = st.selectbox("Idioma", LANG_OPTIONS.keys(), key="lang_rec")
    with col_k:
        k_rec = st.slider("Top K", 10, 100, 50, key="k_rec")

    if st.button("Recomendar", key="btn_rec", type="primary", use_container_width=True):
        if not q_rec.strip():
            st.warning("Escribí algo primero.")
        elif not health_ok:
            st.error("El API no está disponible.")
        else:
            with st.spinner("Buscando candidatas + ranqueando..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/recommend",
                        json={
                            "query": q_rec,
                            "language": LANG_OPTIONS[lang_rec],
                            "top_k": k_rec,
                        },
                        timeout=180,
                    )
                    resp.raise_for_status()
                    data = resp.json()
                except requests.RequestException as e:
                    st.error(f"Error: {e}")
                    st.stop()

            recs = data.get("recommendations", [])
            if recs:
                st.markdown("#### Recomendaciones")
                for i, rec in enumerate(recs, 1):
                    st.markdown(f"**{i}. {rec['title']}**  \n{rec['reason']}")
            else:
                st.info("No se encontraron recomendaciones fuertes en el corpus.")

            if data.get("caveats"):
                st.caption(f"⚠️ {data['caveats']}")

            with st.expander(f"Ver los {len(data.get('sources', []))} comentarios usados"):
                df = pd.DataFrame([
                    {
                        "Película": s["movie_title"],
                        "Idioma": s["language"],
                        "Sentiment": s["sentiment"],
                        "Sim": round(s["similarity"], 3),
                        "Comentario": s["text"][:200] + ("…" if len(s["text"]) > 200 else ""),
                    }
                    for s in data.get("sources", [])
                ])
                st.dataframe(df, use_container_width=True, hide_index=True)
