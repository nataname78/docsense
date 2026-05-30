"""
Application Streamlit : interface web de l'agent RAG multi-document.

V2.4 : support multi-PDF avec filtrage par source.
"""
import os
from pathlib import Path
import streamlit as st

from src.pipeline import (
    index_pdf, ask, ask_stream, ask_verified, summarize, build_hybrid_retriever,
    list_sources, analyze, ask_with_agent,
)


st.set_page_config(
    page_title="RAG Doc Analyzer",
    page_icon="📄",
    layout="wide",
)


UPLOAD_DIR = Path("data/raw")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


# Initialisation du session_state
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "tables_index" not in st.session_state:
    st.session_state.tables_index = []
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []  # liste des noms de fichiers indexes
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "summary" not in st.session_state:
    st.session_state.summary = None


# ============== SIDEBAR ==============
with st.sidebar:
    st.title("📄 RAG Doc Analyzer")
    st.caption("Agent IA d'analyse documentaire — Mistral + FAISS")

    st.divider()
    st.subheader("1. Ajouter un PDF")

    uploaded_file = st.file_uploader(
        "Glissez-deposez un PDF",
        type=["pdf"],
        help="Vous pouvez en ajouter plusieurs successivement.",
        key="file_uploader",
    )

    if uploaded_file is not None:
        # Sauvegarde locale
        pdf_path = UPLOAD_DIR / uploaded_file.name

        # On ne reindexe que si ce fichier n'a pas deja ete indexe
        if uploaded_file.name not in st.session_state.indexed_files:
            with open(pdf_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Indexation de {uploaded_file.name} en cours (~30s)..."):
                vs, all_chunks, new_tables, meta = index_pdf(
                    str(pdf_path),
                    existing_vectorstore=st.session_state.vectorstore,
                    existing_chunks=st.session_state.chunks,
                )
                st.session_state.vectorstore = vs
                st.session_state.chunks = all_chunks
                st.session_state.tables_index.extend(new_tables)
                st.session_state.indexed_files.append(uploaded_file.name)
                # Reconstruction du retriever avec TOUS les chunks
                st.session_state.retriever = build_hybrid_retriever(vs, all_chunks)
                st.session_state.summary = None  # on invalide le resume

            n_new = len([c for c in all_chunks if c.metadata.get("source") == uploaded_file.name])
            st.success(f"{uploaded_file.name} indexe : {n_new} nouveaux chunks.")
        else:
            st.info(f"{uploaded_file.name} est deja indexe.")

    st.divider()

    if st.session_state.indexed_files:
        st.subheader("2. Documents indexes")
        for f in st.session_state.indexed_files:
            n_chunks = len([c for c in st.session_state.chunks if c.metadata.get("source") == f])
            st.write(f"• {f} ({n_chunks} chunks)")

        st.divider()
        st.subheader("3. Filtre de recherche")
        sources = ["Tous les documents"] + list_sources(st.session_state.chunks)
        selected_source = st.selectbox(
            "Restreindre la recherche a :",
            options=sources,
            help="'Tous les documents' = recherche dans tout le corpus."
        )

        st.divider()
        st.subheader("4. Parametres avances")
        use_agent = st.toggle(
            "🤖 Mode Agent (ReAct multi-step)",
            value=False,
            help=(
                "Agent autonome qui decompose la question, fait plusieurs "
                "recherches et calculs si necessaire, et affiche son raisonnement. "
                "Plus lent mais plus puissant sur les questions complexes."
            ),
        )
        
        use_double_check = st.toggle(
            "🛡️ Double check anti-hallucination",
            value=False,
            help=(
                "Une seconde passe LLM verifie que chaque affirmation factuelle de "
                "la reponse est bien supportee par le contexte. Si une hallucination "
                "est detectee, la reponse est remplacee par un refus explicite. "
                "Trade-off : +1 appel LLM (latence x2) et pas de streaming."
            ),
        )
        
        use_reranking = st.toggle(
            "Activer le re-ranking (cross-encoder)",
            value=True,
            help="Re-classe les candidats hybrid via un cross-encoder pour plus de precision.",
        )
        
        retrieve_n = st.slider(
            "Candidats avant re-ranking",
            min_value=10, max_value=40, value=20, step=5,
            disabled=not use_reranking,
        )

        st.divider()
        st.caption("Tech stack")
        st.caption("• LLM : Mistral Small")
        st.caption("• Embeddings : mistral-embed (1024d)")
        st.caption("• Vector store : FAISS")
        st.caption("• Retrieval : Hybrid (BM25 + FAISS + RRF)")
        st.caption("• Re-ranking : bge-reranker-base")
        st.caption("• Tableaux : pdfplumber + Markdown")
        st.caption("• Multi-document : oui")
        st.caption("• Agent ReAct : optionnel (toggle)")


# Defauts hors sidebar
use_double_check = locals().get("use_double_check", False)
use_reranking = locals().get("use_reranking", True)
retrieve_n = locals().get("retrieve_n", 20)
selected_source = locals().get("selected_source", "Tous les documents")
use_agent = locals().get("use_agent", False)
source_filter = None if selected_source == "Tous les documents" else selected_source


# ============== ZONE PRINCIPALE ==============
st.title("Analyse documentaire intelligente")

if st.session_state.vectorstore is None:
    st.info("👈 Commencez par uploader un PDF dans la barre laterale.")
    st.markdown("""
    ### Fonctionnalites

    - **Multi-document** : indexez plusieurs PDFs et croisez les sources
    - **Hybrid search** : BM25 + sematique pour un retrieval robuste
    - **Re-ranking** : cross-encoder pour raffiner les resultats
    - **Tableaux structures** : extraction des tableaux financiers en Markdown
    - **Anti-hallucination** : l'agent refuse d'inventer si l'info n'est pas presente
    - **Citation des sources** : chaque affirmation est sourcee (page, document)

    ### Cas d'usage
    Cabinets comptables, audit, analyse financiere, documentation technique,
    veille reglementaire.
    """)
else:
    # Banniere indiquant le filtre actif
    if source_filter:
        st.info(f"🔍 Recherche restreinte au document : **{source_filter}**")
    else:
        n_docs = len(st.session_state.indexed_files)
        if n_docs > 1:
            st.info(f"🔍 Recherche dans **tous les documents** ({n_docs} indexes)")

    tab_chat, tab_analyses, tab_summary = st.tabs(["💬 Chat", "🔍 Analyses metier", "📊 Resume"])

    # ============== ONGLET CHAT ==============
    with tab_chat:
        for entry in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(entry["question"])
                if entry.get("source_filter"):
                    st.caption(f"🔍 Filtre actif : {entry['source_filter']}")
            with st.chat_message("assistant"):
                st.write(entry["answer"])
                with st.expander(f"📎 Sources utilisees ({len(entry['sources'])} chunks)"):
                    for i, src in enumerate(entry["sources"], start=1):
                        st.markdown(
                            f"**[{i}] {src.metadata.get('source', '?')} - Page {src.metadata['page']}**"
                        )
                        st.text(src.page_content[:500] + ("..." if len(src.page_content) > 500 else ""))
                        st.divider()

        question = st.chat_input("Posez votre question...")
        if question:
            with st.chat_message("user"):
                st.write(question)
                if source_filter:
                    st.caption(f"🔍 Filtre actif : {source_filter}")

            with st.chat_message("assistant"):
                if use_agent:
                    # ====== MODE AGENT REACT ======
                    with st.spinner("🤖 Agent en cours de raisonnement..."):
                        agent_result = ask_with_agent(
                            st.session_state.retriever,
                            question,
                            all_chunks=st.session_state.chunks,
                            tables=st.session_state.tables_index,
                            source_filter=source_filter,
                        )
                    st.write(agent_result["answer"])

                    n_steps = agent_result["iterations"]
                    converged = "✅" if agent_result["completed"] else "⚠️"
                    with st.expander(
                        f"🧠 Trace du raisonnement ({n_steps} etapes) {converged}",
                        expanded=False,
                    ):
                        for step in agent_result["trace"]:
                            st.markdown(f"**Etape {step['step']}**")
                            if step.get("thought"):
                                st.markdown(f"💭 **THOUGHT** : {step['thought']}")
                            if step.get("action"):
                                st.code(f"ACTION : {step['action']}", language="python")
                            if step.get("observation") and not step.get("is_final"):
                                obs_preview = step["observation"][:500]
                                if len(step["observation"]) > 500:
                                    obs_preview += "\n... [tronque dans l'affichage]"
                                st.markdown("👁️ **OBSERVATION** :")
                                st.text(obs_preview)
                            st.divider()

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": agent_result["answer"],
                        "sources": [],
                        "source_filter": source_filter,
                        "agent_trace": agent_result["trace"],
                        "agent_iterations": n_steps,
                        "agent_completed": agent_result["completed"],
                    })

                else:
                    # ====== MODE RAG CLASSIQUE ======
                    if use_double_check:
                        # Mode VERIFIED : pas de streaming, 2 appels LLM
                        with st.spinner("🛡️ Generation + verification anti-hallucination (~10s)..."):
                            verified = ask_verified(
                                st.session_state.retriever,
                                question,
                                k=4,
                                use_hybrid=True,
                                use_reranking=use_reranking,
                                retrieve_n=retrieve_n,
                                source_filter=source_filter,
                            )
                        st.write(verified["answer"])

                        # Badge de verdict
                        verdict = verified["verdict"]
                        if verdict == "VALIDE":
                            st.success(f"✅ Verdict du verificateur : VALIDE — {verified['explanation']}")
                        elif verdict == "INVALIDE":
                            st.error(f"❌ Verdict du verificateur : INVALIDE — {verified['explanation']}")
                            with st.expander("Voir la reponse initiale (avant remplacement)"):
                                st.text(verified["original_answer"])
                        elif verdict == "PARTIEL":
                            st.warning(f"⚠️ Verdict du verificateur : PARTIEL — {verified['explanation']}")
                        else:
                            st.info(f"ℹ️ Verdict du verificateur : INCONNU — {verified['explanation']}")

                        sources = verified["sources"]
                        full_answer = verified["answer"]

                        with st.expander(f"📎 Sources utilisees ({len(sources)} chunks)"):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**[{i}] {src.metadata.get('source', '?')} - Page {src.metadata['page']}**"
                                )
                                preview = src.page_content[:500]
                                if len(src.page_content) > 500:
                                    preview += "..."
                                st.text(preview)
                                st.divider()

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": full_answer,
                            "sources": sources,
                            "source_filter": source_filter,
                            "verdict": verdict,
                        })

                    else:
                        # Mode STREAMING (V3.2) : 1 appel LLM, reponse token par token
                        with st.spinner("Recherche des passages pertinents..."):
                            token_generator, sources = ask_stream(
                                st.session_state.retriever,
                                question,
                                k=4,
                                use_hybrid=True,
                                use_reranking=use_reranking,
                                retrieve_n=retrieve_n,
                                source_filter=source_filter,
                            )
                        full_answer = st.write_stream(token_generator)

                        with st.expander(f"📎 Sources utilisees ({len(sources)} chunks)"):
                            for i, src in enumerate(sources, start=1):
                                st.markdown(
                                    f"**[{i}] {src.metadata.get('source', '?')} - Page {src.metadata['page']}**"
                                )
                                preview = src.page_content[:500]
                                if len(src.page_content) > 500:
                                    preview += "..."
                                st.text(preview)
                                st.divider()

                        st.session_state.chat_history.append({
                            "question": question,
                            "answer": full_answer,
                            "sources": sources,
                            "source_filter": source_filter,
                        })



    # ============== ONGLET ANALYSES METIER ==============
    with tab_analyses:
        st.markdown("### Analyses pre-construites")
        st.caption(
            "Cliquez sur une analyse pour la lancer automatiquement. "
            "Le filtre source actif s'applique."
        )

        # Initialise le cache des analyses dans session_state
        if "analyses_results" not in st.session_state:
            st.session_state.analyses_results = {}

        # Selection des chunks selon le filtre actif
        if source_filter:
            chunks_to_analyze = [
                c for c in st.session_state.chunks
                if c.metadata.get("source") == source_filter
            ]
            st.caption(f"📂 Analyse restreinte a : **{source_filter}**")
        else:
            chunks_to_analyze = st.session_state.chunks
            st.caption(f"📂 Analyse sur **tous les documents** indexes")

        # Grille 2 colonnes de 5 boutons
        analyses_definitions = [
            ("chiffres_cles", "📊 Chiffres cles", "Extrait CA, resultat net, dette, effectifs..."),
            ("risques", "⚠️ Risques & alertes", "Identifie les risques mentionnes."),
            ("strategie", "🎯 Strategie & objectifs", "Liste axes strategiques, objectifs, projets."),
            ("evolution", "📈 Evolution vs N-1", "Tableau comparatif annee N vs N-1."),
            ("note_synthese", "✍️ Note de synthese", "Note de 1 page executive pour manager."),
        ]

        cols = st.columns(2)
        for i, (key, label, description) in enumerate(analyses_definitions):
            col = cols[i % 2]
            with col:
                if st.button(label, help=description, use_container_width=True, key=f"btn_{key}"):
                    with st.spinner(f"Generation de l'analyse '{label}' (~15-30s)..."):
                        result = analyze(chunks_to_analyze, key)
                    # Cle de cache integrant le filtre actif
                    cache_key = f"{key}__{source_filter or 'all'}"
                    st.session_state.analyses_results[cache_key] = result
                    st.rerun()

        st.divider()

        # Affichage des resultats des analyses lancees
        for key, label, _ in analyses_definitions:
            cache_key = f"{key}__{source_filter or 'all'}"
            if cache_key in st.session_state.analyses_results:
                with st.expander(f"{label} - resultat", expanded=True):
                    st.markdown(st.session_state.analyses_results[cache_key])
                    if st.button(f"Effacer {label}", key=f"clear_{key}"):
                        del st.session_state.analyses_results[cache_key]
                        st.rerun()

    # ============== ONGLET RESUME ==============
    with tab_summary:
        st.write("Genere un resume structure du corpus actif (en respectant le filtre source).")

        if st.session_state.summary is None:
            if st.button("Generer le resume", type="primary"):
                # On respecte le filtre actif pour le resume aussi
                if source_filter:
                    chunks_to_summarize = [
                        c for c in st.session_state.chunks
                        if c.metadata.get("source") == source_filter
                    ]
                else:
                    chunks_to_summarize = st.session_state.chunks

                with st.spinner("Generation du resume par Mistral (~15s)..."):
                    st.session_state.summary = summarize(chunks_to_summarize)
                st.rerun()
        else:
            st.markdown(st.session_state.summary)
            if st.button("Regenerer"):
                st.session_state.summary = None
                st.rerun()