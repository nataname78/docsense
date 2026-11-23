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
    page_title="DocSense · Nathan Makila",
    page_icon="📄",
    layout="wide",
)

# Look & feel : meme esprit "carnet de labo" que le portfolio (nataname78.github.io).
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,800&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&family=Reenie+Beanie&display=swap');
.stApp{background-color:#f2f6f8;background-image:linear-gradient(#d6e3ea 1px,transparent 1px),linear-gradient(90deg,#d6e3ea 1px,transparent 1px);background-size:32px 32px;font-family:'IBM Plex Sans',sans-serif;color:#0f2038}
[data-testid="stHeader"]{background:transparent}
[data-testid="stSidebar"]{background:#fbfdfe;border-right:2px solid #0f2038}
.kicker{font:500 12px 'IBM Plex Mono',monospace;letter-spacing:.08em;color:#51627a;text-transform:uppercase;margin-top:8px}
.ttl{font:800 clamp(34px,5vw,52px)/1.02 'Bricolage Grotesque',sans-serif;letter-spacing:-.02em;margin:6px 0 10px}
.ttl mark{background:linear-gradient(transparent 55%,#ffe55c 55%);color:inherit;padding:0 2px}
.lede{font-size:18px;line-height:1.5;max-width:60ch;margin:0 0 4px}
.hand{font:400 30px/1 'Reenie Beanie',cursive;color:#cf3a2c;transform:rotate(-1.5deg);display:inline-block;margin:2px 0 14px}
.card{background:#fbfdfe;border:2px solid #0f2038;border-radius:8px;box-shadow:4px 4px 0 #0f2038;padding:14px 18px;margin:8px 0 22px;font-size:15px;line-height:1.55;max-width:760px}
.card b{font:500 11px 'IBM Plex Mono',monospace;letter-spacing:.07em;text-transform:uppercase;color:#51627a;display:block;margin-bottom:4px}
.card ul{margin:4px 0 8px 18px;padding:0}
.fine{font:400 12.5px 'IBM Plex Mono',monospace;color:#51627a}
.foot{font:400 13px 'IBM Plex Mono',monospace;color:#51627a;margin:36px 0 12px;border-top:2px dashed #b9cad6;padding-top:12px}
.foot a{color:#0f2038}
div.stButton>button{border:2px solid #0f2038;border-radius:6px;box-shadow:3px 3px 0 #0f2038;font-weight:600;transition:transform .12s,box-shadow .12s}
div.stButton>button:hover{transform:translate(2px,2px);box-shadow:1px 1px 0 #0f2038;border-color:#0f2038}
div.stButton>button[kind="primary"]{background:#0f2038;color:#fff}
h1,h2,h3,h4{font-family:'Bricolage Grotesque',sans-serif}
</style>
""",
    unsafe_allow_html=True,
)

# --- Protection de la demo publique : la cle Mistral est payee par l'auteur,
# donc on plafonne les usages par session et par jour (configurable via les secrets).
import threading
from datetime import date

SESSION_CAP = int(os.getenv("DEMO_SESSION_CAP", "6"))
DAILY_CAP = int(os.getenv("DEMO_DAILY_CAP", "80"))


@st.cache_resource
def _daily_counter() -> dict:
    return {"day": date.today(), "n": 0, "lock": threading.Lock()}


def take_slot():
    """Reserve un appel au modele. Renvoie None si OK, sinon un message expliquant la limite."""
    if st.session_state.get("used", 0) >= SESSION_CAP:
        return (f"Limite de la demo atteinte : {SESSION_CAP} actions par session. "
                "Recharge la page pour une nouvelle session, ou lance DocSense en local (voir le depot GitHub).")
    c = _daily_counter()
    with c["lock"]:
        if c["day"] != date.today():
            c["day"], c["n"] = date.today(), 0
        if c["n"] >= DAILY_CAP:
            return "La demo a atteint sa limite du jour. Reviens demain, ou lance DocSense en local (voir le depot GitHub)."
        c["n"] += 1
    st.session_state["used"] = st.session_state.get("used", 0) + 1
    return None


SAMPLE_NAME = "Lumeria_rapport_annuel_2025_exemple.pdf"
SAMPLE_PAGES = [
    [("Lumeria SA - Rapport annuel 2025", 18),
     ("Document fictif, ecrit pour la demonstration de DocSense. Aucune donnee reelle.", 9),
     ("", 6),
     ("1. Message de la direction", 13),
     ("Lumeria, fabricant francais de capteurs pour la gestion de l'eau, a poursuivi sa croissance en 2025. "
      "Le chiffre d'affaires atteint 84,2 millions d'euros, en hausse de 11,8 % par rapport a 2024 (75,3 millions). "
      "Le resultat net s'etablit a 6,9 millions d'euros contre 5,1 millions en 2024. "
      "L'effectif moyen est de 412 collaborateurs au 31 decembre 2025 (385 un an plus tot).", 10),
     ("", 6),
     ("2. Strategie 2026-2028", 13),
     ("Trois axes : (a) ouvrir une usine d'assemblage a Lyon pour reduire de 30 % les delais de livraison ; "
      "(b) lancer la gamme AquaSense Pro destinee aux collectivites ; (c) atteindre 25 % du chiffre d'affaires a l'international d'ici 2028, "
      "contre 14 % en 2025. Le plan d'investissement s'eleve a 18 millions d'euros sur trois ans.", 10)],
    [("3. Principaux risques", 13),
     ("Approvisionnement en composants electroniques : deux fournisseurs asiatiques representent 62 % des achats de puces. "
      "Risque de change : 9 % des ventes sont facturees en dollars americains, sans couverture integrale. "
      "Risque reglementaire : la revision de la directive europeenne sur l'eau potable pourrait imposer de nouvelles certifications en 2027. "
      "Risque de concentration client : les trois premiers clients pesent 28 % du chiffre d'affaires.", 10),
     ("", 6),
     ("4. Chiffres cles (en millions d'euros)", 13)],
]
SAMPLE_TABLE = [
    ["Indicateur", "2025", "2024", "Variation"],
    ["Chiffre d'affaires", "84,2", "75,3", "+11,8 %"],
    ["Resultat d'exploitation", "9,8", "7,6", "+28,9 %"],
    ["Resultat net", "6,9", "5,1", "+35,3 %"],
    ["Dette nette", "12,4", "15,0", "-17,3 %"],
    ["Effectif moyen", "412", "385", "+7,0 %"],
]


def make_sample_pdf(path):
    """Fabrique le PDF d'exemple (texte + vrai tableau) avec PyMuPDF."""
    import fitz
    doc = fitz.open()
    for pi, blocks in enumerate(SAMPLE_PAGES):
        page = doc.new_page()
        y = 60
        for text, size in blocks:
            if not text:
                y += size
                continue
            h = size * 1.6 * (len(text) // (88 if size <= 10 else 55) + 1) + 8
            page.insert_textbox(fitz.Rect(50, y, 545, y + h), text, fontsize=size, fontname="helv")
            y += h + 4
        if pi == len(SAMPLE_PAGES) - 1:
            x0, cw, rh = 50, [190, 100, 100, 100], 24
            for r, row in enumerate(SAMPLE_TABLE):
                x = x0
                for c, cell in enumerate(row):
                    rect = fitz.Rect(x, y + r * rh, x + cw[c], y + (r + 1) * rh)
                    page.draw_rect(rect, color=(0, 0, 0), width=0.8)
                    page.insert_textbox(rect + (4, 6, -4, 0), cell, fontsize=10, fontname="helv")
                    x += cw[c]
    doc.save(str(path))
    doc.close()


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
    st.title("📄 DocSense")
    st.caption("Agent IA d'analyse documentaire — Mistral + FAISS")

    st.divider()
    st.subheader("1. Ajouter un PDF")

    uploaded_file = st.file_uploader(
        "Glissez-deposez un PDF",
        type=["pdf"],
        help="Vous pouvez en ajouter plusieurs successivement.",
        key="file_uploader",
    )

    use_sample = st.button("Essayer avec un rapport d'exemple", type="primary", use_container_width=True,
                           help="Rapport annuel fictif de 2 pages, genere pour la demo.")
    st.caption("Ou charge ton propre PDF ci-dessus. Les fichiers ne sont pas conserves apres la session.")

    if use_sample and SAMPLE_NAME not in st.session_state.indexed_files:
        _lim = take_slot()
        if _lim:
            st.warning(_lim)
        else:
            _p = UPLOAD_DIR / SAMPLE_NAME
            make_sample_pdf(_p)
            with st.spinner("Indexation du rapport d'exemple..."):
                vs, all_chunks, new_tables, meta = index_pdf(
                    str(_p),
                    existing_vectorstore=st.session_state.vectorstore,
                    existing_chunks=st.session_state.chunks,
                )
                st.session_state.vectorstore = vs
                st.session_state.chunks = all_chunks
                st.session_state.tables_index.extend(new_tables)
                st.session_state.indexed_files.append(SAMPLE_NAME)
                st.session_state.retriever = build_hybrid_retriever(vs, all_chunks)
                st.session_state.summary = None
            st.rerun()

    if uploaded_file is not None:
        # Sauvegarde locale
        pdf_path = UPLOAD_DIR / uploaded_file.name

        # On ne reindexe que si ce fichier n'a pas deja ete indexe
        _upload_lim = take_slot() if uploaded_file.name not in st.session_state.indexed_files else None
        if _upload_lim:
            st.warning(_upload_lim)
        elif uploaded_file.name not in st.session_state.indexed_files:
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
            value=False,
            help="Re-classe les candidats via un cross-encoder (plus precis, plus lent ; desactive par defaut sur la demo en ligne, serveur limite).",
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
use_reranking = locals().get("use_reranking", False)
retrieve_n = locals().get("retrieve_n", 20)
selected_source = locals().get("selected_source", "Tous les documents")
use_agent = locals().get("use_agent", False)
source_filter = None if selected_source == "Tous les documents" else selected_source


# ============== ZONE PRINCIPALE ==============
st.markdown(
    """
<div class="kicker">Carnet de labo · démo n°2</div>
<div class="ttl">Doc<mark>Sense</mark></div>
<p class="lede">Pose des questions à un PDF et obtiens des réponses sourcées (page, passage). S'il ne sait pas, il doit le dire plutôt qu'inventer.</p>
<div class="hand">charge un PDF, ou prends mon rapport d'exemple.</div>
""",
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.markdown(
        """
<div class="card"><b>Pour tester en 2 minutes</b>
<ul>
<li>Clique sur « Essayer avec un rapport d'exemple » dans la barre de gauche.</li>
<li>Pose une question dont la réponse est dans le document : <i>« Quel est le chiffre d'affaires 2025 ? »</i></li>
<li>Puis essaie de le piéger : <i>« Quel est le salaire du directeur ? »</i> (l'info n'y est pas).</li>
</ul>
<span class="fine">Recherche hybride (BM25 + vecteurs) · tableaux extraits · sources citées · mode agent et double vérification en option.<br>
Démo limitée à quelques actions par visite. Ne charge pas de document confidentiel.</span></div>
""",
        unsafe_allow_html=True,
    )
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

        question = st.chat_input("Pose ta question sur le document...")
        if question and (_lim := take_slot()) is not None:
            st.warning(_lim)
            question = None
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
                    if (_lim := take_slot()) is not None:
                        st.warning(_lim)
                        st.stop()
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
                if (_lim := take_slot()) is not None:
                    st.warning(_lim)
                    st.stop()
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


st.markdown(
    '<div class="foot">Construit par Nathan Makila · '
    '<a href="https://nataname78.github.io" target="_blank">portfolio</a> · '
    '<a href="https://github.com/nataname78/docsense" target="_blank">code source</a></div>',
    unsafe_allow_html=True,
)
