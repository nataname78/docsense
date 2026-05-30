"""
Module reranker : re-classement des chunks via cross-encoder.

Pattern retrieve-then-rerank :
1. Hybrid retrieval recupere un top-N large (ex: 20) candidats
2. Cross-encoder re-classe specifiquement selon (question, chunk)
3. On garde le top-K final (ex: 4)

Pourquoi : un cross-encoder evalue la pertinence en voyant
question ET chunk ensemble (vs bi-encoder qui les encode separement).
Plus precis, mais plus lent -> on l'applique sur un petit nombre de candidats.

Modele utilise : BAAI/bge-reranker-base
- Multi-langue (francais OK)
- ~280 Mo, tourne sur CPU
- Telechargement automatique au premier lancement
"""
from sentence_transformers import CrossEncoder
from src.query_expansion import expand_query
from src.logger import get_logger
logger = get_logger(__name__)


# Singleton : on charge le modele UNE SEULE FOIS (lazy loading)
_reranker_model = None


def get_reranker():
    """
    Retourne le cross-encoder, en le chargeant une seule fois (lazy).
    Premier appel : telechargement (~280 Mo) + chargement.
    Appels suivants : instantane.
    """
    global _reranker_model
    if _reranker_model is None:
        logger.info("Reranker : chargement de BAAI/bge-reranker-base...")
        logger.info("  [Reranker] (premier lancement : telechargement ~280Mo)")
        _reranker_model = CrossEncoder(
            "BAAI/bge-reranker-base",
            max_length=512,  # tronque les chunks trop longs pour le cross-encoder
        )
        logger.info("  [Reranker] OK, modele charge.")
    return _reranker_model


def rerank(question, chunks, top_k=4):
    """
    Re-classe une liste de chunks selon leur pertinence pour la question.

    Args:
        question: str
        chunks: list[Document], les candidats issus du retrieval
        top_k: int, nombre de chunks a retourner apres reranking

    Returns:
        list[Document]: les top_k chunks, du plus pertinent au moins pertinent
    """
    question = expand_query(question)
    if not chunks:
        return []

    reranker = get_reranker()

    # Construit les paires (question, chunk_content) pour le cross-encoder
    pairs = [(question, chunk.page_content) for chunk in chunks]

    # Calcule le score de pertinence pour chaque paire
    # show_progress_bar=False pour ne pas polluer la sortie en mode app
    scores = reranker.predict(pairs, show_progress_bar=False)

    # Associe chaque chunk a son score, trie par score decroissant
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Retourne les top_k chunks
    return [chunk for chunk, score in scored[:top_k]]


def rerank_with_scores(question, chunks, top_k=4):
    """
    Version qui retourne aussi les scores (utile pour debug et evaluation).

    Returns:
        list[tuple[Document, float]]: (chunk, score) tries par score decroissant
    """
    if not chunks:
        return []

    reranker = get_reranker()
    pairs = [(question, chunk.page_content) for chunk in chunks]
    scores = reranker.predict(pairs, show_progress_bar=False)

    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)

    return scored[:top_k]


# Bloc de test : comparaison hybrid SEUL vs hybrid + reranking
if __name__ == "__main__":
    from src.vectorstore import load_vectorstore
    from src.ingestion import extract_text_from_pdf
    from src.chunking import chunk_pages
    from src.retrieval import HybridRetriever

    print("Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    print("Reconstruction des chunks pour BM25...")
    result = extract_text_from_pdf("data/raw/sample.pdf")
    all_chunks = chunk_pages(pages=result["pages"], file_name=result["metadata"]["file_name"])

    hybrid = HybridRetriever(vectorstore=vs, chunks=all_chunks)

    # Questions de test
    questions = [
        "Quels sont les chiffres concernant IFRS 16 ?",
        "Quelles sont les performances de Sephora ?",
        "Combien de postes le groupe prevoit-il de creer en France ?",
        "Quels sont les principaux risques mentionnes ?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print(f"QUESTION : {question}")
        print("=" * 80)

        # Etape 1 : hybrid retrieve un top-20 large
        candidates = hybrid.retrieve(question, k=20)
        print(f"\n[Hybrid] {len(candidates)} candidats recuperes")

        # Etape 2 : rerank pour avoir le top-4
        reranked = rerank_with_scores(question, candidates, top_k=4)
        print(f"[Rerank] Top-4 apres re-classement cross-encoder :")
        for i, (chunk, score) in enumerate(reranked, start=1):
            preview = chunk.page_content[:150].replace("\n", " ")
            print(f"  [{i}] score={score:.3f} | Page {chunk.metadata['page']} : {preview}...")

        # Pour comparer : top-4 du hybrid SANS reranking
        print(f"\n[Hybrid seul] Top-4 sans reranking (pour comparaison) :")
        for i, chunk in enumerate(candidates[:4], start=1):
            preview = chunk.page_content[:150].replace("\n", " ")
            print(f"  [{i}] Page {chunk.metadata['page']} : {preview}...")