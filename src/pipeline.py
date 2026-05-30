"""
Module pipeline : orchestrateur de haut niveau.

V2.4 : support multi-document
- index_pdf : indexe UN PDF (creation ou ajout)
- ask : avec filtrage optionnel par source
- list_sources : liste les documents indexes
"""
from src.logger import get_logger
logger = get_logger(__name__)

from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_pages
from src.vectorstore import (
    build_vectorstore,
    add_chunks_to_vectorstore,
    save_vectorstore,
    load_vectorstore,
)
from src.retrieval import retrieve_chunks, HybridRetriever, format_chunks_for_prompt
from src.reranker import rerank
from src.generation import (
    generate_answer, generate_summary, run_analysis,
    generate_answer_stream, generate_answer_with_verification,
)
from src.agent import run_agent
from src.table_extractor import extract_tables_structured


def index_pdf(pdf_path, existing_vectorstore=None, existing_chunks=None,
              save_to=None, extract_tables=True):
    """
    Indexe UN PDF, soit en creant un nouvel index, soit en l'ajoutant a un existant.

    Args:
        pdf_path: chemin du PDF
        existing_vectorstore: si fourni, on AJOUTE les chunks a cet index (V2.4)
        existing_chunks: liste des chunks deja indexes (pour reconstruire BM25 plus tard)
        save_to: dossier d'export de l'index (optionnel)
        extract_tables: True pour enrichir avec les tableaux pdfplumber

    Returns:
        (vectorstore, all_chunks, metadata)
        - vectorstore : index FAISS (nouveau ou enrichi)
        - all_chunks : TOUS les chunks indexes (anciens + nouveaux)
        - metadata : meta du PDF qu'on vient d'indexer
    """
    # Etape 1 : extraction
    result = extract_text_from_pdf(pdf_path)
    pages = result["pages"]
    metadata = result["metadata"]

    # Etape 2 : enrichissement tableaux
    if extract_tables:
        from src.table_extractor import enrich_pages_with_tables
        logger.info("Pipeline : enrichissement avec tableaux structures (pdfplumber)...")
        pages = enrich_pages_with_tables(pages, pdf_path)

    # Etape 3 : chunking
    new_chunks = chunk_pages(pages=pages, file_name=metadata["file_name"])

    # Etape 4 : indexation (creation ou ajout)
    if existing_vectorstore is not None:
        logger.info(f"Pipeline : ajout de {len(new_chunks)} chunks a l'index existant...")
        vectorstore = add_chunks_to_vectorstore(existing_vectorstore, new_chunks)
        all_chunks = (existing_chunks or []) + new_chunks
    else:
        logger.info(f"Pipeline : creation d'un nouvel index avec {len(new_chunks)} chunks...")
        vectorstore = build_vectorstore(new_chunks)
        all_chunks = new_chunks

    if save_to:
        save_vectorstore(vectorstore, save_to)

    # V3.3 : extraction des tableaux structures pour l'agent
    new_tables = extract_tables_structured(pdf_path)

    return vectorstore, all_chunks, new_tables, metadata


def build_hybrid_retriever(vectorstore, chunks):
    """Construit un HybridRetriever pret a l'emploi."""
    return HybridRetriever(vectorstore=vectorstore, chunks=chunks)


def ask(retriever, question, k=4, use_hybrid=True, use_reranking=True,
        retrieve_n=20, source_filter=None):
    """
    Pose une question au RAG, avec hybrid + reranking + filtrage optionnel.

    Args:
        retriever: HybridRetriever (V2)
        question: str
        k: chunks finaux envoyes au LLM
        use_hybrid: True pour hybrid, False pour semantique pur
        use_reranking: True pour appliquer le cross-encoder
        retrieve_n: candidats avant reranking
        source_filter: str ou None - si fourni, restreint la recherche a CE document

    Returns:
        dict avec 'answer' (str) et 'sources' (list[Document])
    """
    # Etape 1 : retrieval (avec filtre source si demande)
    if use_hybrid:
        n = retrieve_n if use_reranking else k
        candidates = retriever.retrieve(question, k=n, source_filter=source_filter)
    else:
        n = retrieve_n if use_reranking else k
        candidates = retrieve_chunks(retriever, question, k=n)

    # Etape 2 : reranking optionnel
    if use_reranking and len(candidates) > k:
        chunks = rerank(question, candidates, top_k=k)
    else:
        chunks = candidates[:k]

    # Etape 3 : generation
    context = format_chunks_for_prompt(chunks)
    answer = generate_answer(question, context)

    return {
        "answer": answer,
        "sources": chunks,
    }


def ask_stream(retriever, question, k=4, use_hybrid=True, use_reranking=True,
               retrieve_n=20, source_filter=None):
    """
    Variante streaming de ask() : retourne (generator, sources).

    Le retrieval se fait normalement (rapide), puis la generation est streamee.
    Permet a l'UI d'afficher les sources immediatement et la reponse au fur et a mesure.

    Returns:
        tuple : (token_generator, list[Document])
        - token_generator : generator qui yield les morceaux de reponse
        - sources : les chunks utilises comme contexte (deja disponibles)
    """
    # Etape 1 : retrieval (rapide, non-streame)
    if use_hybrid:
        n = retrieve_n if use_reranking else k
        candidates = retriever.retrieve(question, k=n, source_filter=source_filter)
    else:
        n = retrieve_n if use_reranking else k
        candidates = retrieve_chunks(retriever, question, k=n)

    # Etape 2 : reranking
    if use_reranking and len(candidates) > k:
        chunks = rerank(question, candidates, top_k=k)
    else:
        chunks = candidates[:k]

    # Etape 3 : preparation du contexte
    context = format_chunks_for_prompt(chunks)

    # Etape 4 : generation EN STREAMING (on retourne un generateur)
    token_generator = generate_answer_stream(question, context)

    return token_generator, chunks


def ask_verified(retriever, question, k=4, use_hybrid=True, use_reranking=True,
                 retrieve_n=20, source_filter=None):
    """
    Pose une question avec double check anti-hallucination.

    Pipeline identique a ask() mais avec verification par 2eme LLM en fin.
    Si la reponse contient des affirmations non supportees par le contexte,
    elle est remplacee par un refus explicite.

    NOTE : pas de streaming (impossible avec verification, qui necessite la
    reponse complete avant verdict).

    Returns:
        dict avec :
            - 'answer' (str, la reponse finale)
            - 'sources' (list[Document])
            - 'verdict' (str, 'VALIDE' / 'INVALIDE' / 'PARTIEL')
            - 'is_valid' (bool)
            - 'explanation' (str)
            - 'original_answer' (str, la reponse avant remplacement)
    """
    # Etape 1 : retrieval
    if use_hybrid:
        n = retrieve_n if use_reranking else k
        candidates = retriever.retrieve(question, k=n, source_filter=source_filter)
    else:
        n = retrieve_n if use_reranking else k
        candidates = retrieve_chunks(retriever, question, k=n)

    # Etape 2 : reranking
    if use_reranking and len(candidates) > k:
        chunks = rerank(question, candidates, top_k=k)
    else:
        chunks = candidates[:k]

    # Etape 3 : preparation du contexte
    context = format_chunks_for_prompt(chunks)

    # Etape 4 : generation + verification (2 appels LLM)
    result = generate_answer_with_verification(question, context)

    return {
        "answer": result["answer"],
        "sources": chunks,
        "verdict": result["verdict"],
        "is_valid": result["is_valid"],
        "explanation": result["explanation"],
        "original_answer": result["original_answer"],
    }


def list_sources(chunks):
    """
    Retourne la liste des sources (noms de fichiers) presentes dans les chunks.

    Utile pour proposer un filtre par document dans l'UI.
    """
    sources = set()
    for chunk in chunks:
        src = chunk.metadata.get("source")
        if src:
            sources.add(src)
    return sorted(sources)


def summarize(chunks):
    """Genere un resume du document (ou du corpus entier)."""
    return generate_summary(chunks)


def analyze(chunks, analysis_type):
    """
    Lance une analyse metier predefinie sur les chunks.

    Args:
        chunks: list[Document]
        analysis_type: str parmi 'chiffres_cles', 'risques', 'strategie',
                       'evolution', 'note_synthese'

    Returns:
        str: analyse generee en Markdown
    """
    return run_analysis(analysis_type, chunks)

def ask_with_agent(retriever, question, all_chunks, tables=None, source_filter=None,
                   max_iterations=8):
    """
    Pose une question au mode AGENT (ReAct).

    L'agent peut faire plusieurs recherches, calculer, comparer,
    avant de produire une reponse finale.

    Args:
        retriever: HybridRetriever
        question: str
        all_chunks: list[Document], tous les chunks (pour list_documents)
        source_filter: optionnel
        max_iterations: garde-fou contre boucles infinies

    Returns:
        dict avec 'answer', 'trace', 'iterations', 'completed', 'sources'
    """
    result = run_agent(
        question=question,
        retriever=retriever,
        all_chunks=all_chunks,
        tables=tables,
        source_filter=source_filter,
        max_iterations=max_iterations,
        verbose=False,  # pas de print dans Streamlit
    )

    # On extrait les sources (chunks) referencees pendant la trace
    # depuis les actions de type search
    sources = []
    for step in result["trace"]:
        if step.get("action", "").strip().startswith("search"):
            # On peut parser pour recuperer les chunks ; ici on fait simple :
            # les observations contiennent deja les sources formatees
            pass

    result["sources"] = sources  # vide pour l'instant, on utilise la trace pour l'UI
    return result