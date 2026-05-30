"""
Module retrieval : recherche des chunks pertinents.

Deux strategies disponibles :
1. Semantique pure (FAISS) - la version V1
2. Hybride (BM25 + FAISS + Reciprocal Rank Fusion) - la version V2

Pourquoi hybrid :
- BM25 (lexical) excelle sur noms propres, references exactes, codes
- FAISS (semantique) excelle sur paraphrases, sens global
- La fusion RRF combine les deux classements sans heuristique fragile
"""
import re
from rank_bm25 import BM25Okapi
from src.query_expansion import expand_query
from src.logger import get_logger
logger = get_logger(__name__)


# ============== UTILITAIRES ==============

def _tokenize_fr(text):
    """
    Tokenizer simple pour le francais.

    Pour un projet pro, on utiliserait spaCy ou nltk avec stemming + stopwords.
    Ici on reste leger : minuscules + split par non-mot + filtre des tokens vides.

    Args:
        text: str

    Returns:
        list[str]: tokens (mots)
    """
    text = text.lower()
    # Split sur tout ce qui n'est pas une lettre/chiffre/accent
    tokens = re.findall(r"[a-z0-9àâäçéèêëîïôöùûüÿñæœ]+", text)
    # On garde les tokens d'au moins 2 caracteres (filtre 'a', 'l', etc.)
    tokens = [t for t in tokens if len(t) >= 2]
    return tokens


# ============== V1 : RETRIEVAL SEMANTIQUE PUR ==============

def retrieve_chunks(vectorstore, question, k=4):
    """
    [V1] Retrieval semantique pur via FAISS.
    Conserve pour compatibilite et comparaison avec le hybrid.
    """
    return vectorstore.similarity_search(question, k=k)


# ============== V2 : HYBRID SEARCH ==============

class HybridRetriever:
    """
    Retriever hybride : BM25 + FAISS, fusionnes via Reciprocal Rank Fusion.

    Usage typique :
        retriever = HybridRetriever(vectorstore, all_chunks)
        results = retriever.retrieve("ma question", k=4)
    """

    def __init__(self, vectorstore, chunks, k_each=20, rrf_k=60):
        """
        Args:
            vectorstore: FAISS, l'index semantique
            chunks: list[Document], TOUS les chunks indexes (necessaire pour BM25)
            k_each: int, nombre de candidats recuperes par CHAQUE moteur avant fusion
            rrf_k: int, constante de RRF (60 = valeur standard de la litterature)
        """
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.k_each = k_each
        self.rrf_k = rrf_k

        # Construit l'index BM25 sur tous les chunks
        # BM25Okapi attend une liste de listes de tokens
        logger.info(f"HybridRetriever : Indexation BM25 sur {len(chunks)} chunks...")
        tokenized_corpus = [_tokenize_fr(c.page_content) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        logger.info("HybridRetriever : OK, BM25 pret.")

    def _bm25_search(self, question):
        """Retourne les indices des top-k_each chunks selon BM25."""
        tokenized_query = _tokenize_fr(question)
        # get_scores retourne un score pour CHAQUE chunk de l'index
        scores = self.bm25.get_scores(tokenized_query)
        # On recupere les indices tries par score decroissant
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return top_indices[: self.k_each]

    def _faiss_search(self, question):
        """Retourne les indices des top-k_each chunks selon FAISS."""
        # similarity_search retourne des Documents, on les rattache aux indices originaux
        # via comparaison du contenu (rapide pour quelques milliers de chunks)
        faiss_docs = self.vectorstore.similarity_search(question, k=self.k_each)
        # Mapping contenu -> index dans self.chunks
        content_to_idx = {c.page_content: i for i, c in enumerate(self.chunks)}
        indices = []
        for doc in faiss_docs:
            idx = content_to_idx.get(doc.page_content)
            if idx is not None:
                indices.append(idx)
        return indices

    def _reciprocal_rank_fusion(self, bm25_ranks, faiss_ranks):
        """
        Fusionne deux classements via RRF.

        Formule : score(doc) = 1/(rrf_k + rang_BM25) + 1/(rrf_k + rang_FAISS)

        Args:
            bm25_ranks: list[int], indices ordonnes par BM25 (meilleur en premier)
            faiss_ranks: list[int], indices ordonnes par FAISS (meilleur en premier)

        Returns:
            list[int]: indices fusionnes, ordonnes du meilleur au moins bon
        """
        scores = {}

        for rank, idx in enumerate(bm25_ranks, start=1):
            scores[idx] = scores.get(idx, 0) + 1 / (self.rrf_k + rank)

        for rank, idx in enumerate(faiss_ranks, start=1):
            scores[idx] = scores.get(idx, 0) + 1 / (self.rrf_k + rank)

        # Tri par score decroissant
        return sorted(scores.keys(), key=lambda i: scores[i], reverse=True)

    def retrieve(self, question, k=4, source_filter=None, verbose=False):
        """
        Recherche hybride : BM25 + FAISS + RRF, avec filtrage optionnel par source.

        Args:
            question: str
            k: nombre de chunks a retourner apres fusion
            source_filter: str ou None. Si fourni, ne renvoie que les chunks
                          dont metadata['source'] == source_filter
            verbose: si True, affiche les details du retrieval

        Returns:
            list[Document]: les k meilleurs chunks
        """
        # V3.2.5 : query expansion automatique sur termes financiers
        question = expand_query(question)
        bm25_ranks = self._bm25_search(question)
        faiss_ranks = self._faiss_search(question)
        fused = self._reciprocal_rank_fusion(bm25_ranks, faiss_ranks)

        # Application du filtre source si demande
        if source_filter:
            filtered = [
                idx for idx in fused
                if self.chunks[idx].metadata.get("source") == source_filter
            ]
            top_k_indices = filtered[:k]
        else:
            top_k_indices = fused[:k]

        top_k_chunks = [self.chunks[i] for i in top_k_indices]

        if verbose:
            print(f"\n  [HybridRetriever] Question : '{question}'")
            if source_filter:
                print(f"  [HybridRetriever] Filtre source : {source_filter}")
            print(f"  [HybridRetriever] BM25 top-5 (indices)  : {bm25_ranks[:5]}")
            print(f"  [HybridRetriever] FAISS top-5 (indices) : {faiss_ranks[:5]}")
            print(f"  [HybridRetriever] RRF top-{k} (indices) : {top_k_indices}")

        return top_k_chunks


# ============== FORMATAGE POUR LE PROMPT ==============

def format_chunks_for_prompt(chunks):
    """
    Formate les chunks en un bloc texte injectable dans le prompt.
    Chaque chunk est prefixe par sa source et son numero de page.
    """
    formatted = []
    for i, chunk in enumerate(chunks, start=1):
        page = chunk.metadata.get("page", "?")
        source = chunk.metadata.get("source", "?")
        formatted.append(
            f"[Extrait {i} - source: {source}, page {page}]\n{chunk.page_content}"
        )
    return "\n\n---\n\n".join(formatted)


# ============== BLOC DE TEST : COMPARAISON V1 vs V2 ==============

if __name__ == "__main__":
    from src.vectorstore import load_vectorstore
    from src.ingestion import extract_text_from_pdf
    from src.chunking import chunk_pages

    print("Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    # Pour le hybrid, on a besoin des chunks bruts. On re-extrait depuis le PDF.
    print("Reconstruction des chunks pour BM25 (re-extraction du PDF)...")
    pdf_path = "data/raw/sample.pdf"
    result = extract_text_from_pdf(pdf_path)
    all_chunks = chunk_pages(pages=result["pages"], file_name=result["metadata"]["file_name"])
    print(f"  -> {len(all_chunks)} chunks reconstruits\n")

    # Initialise le hybrid retriever
    hybrid = HybridRetriever(vectorstore=vs, chunks=all_chunks)

    # Questions de test
    questions = [
        "Quel est le chiffre d'affaires du groupe en 2023 ?",
        "Quelles sont les performances de Sephora ?",
        "Quelles sont les performances de Tiffany ?",
        "Que dit le document sur la norme IFRS 16 ?",  # nom propre + reference
        "Combien de boutiques Marc Jacobs ?",  # nom propre specifique
    ]

    for question in questions:
        print("=" * 80)
        print(f"QUESTION : {question}")
        print("=" * 80)

        # V1 : semantique pur
        print("\n--- V1 (semantique pur) ---")
        v1_results = retrieve_chunks(vs, question, k=3)
        for i, doc in enumerate(v1_results, start=1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  [{i}] Page {doc.metadata['page']} : {preview}...")

        # V2 : hybrid
        print("\n--- V2 (hybrid BM25 + FAISS + RRF) ---")
        v2_results = hybrid.retrieve(question, k=3)
        for i, doc in enumerate(v2_results, start=1):
            preview = doc.page_content[:150].replace("\n", " ")
            print(f"  [{i}] Page {doc.metadata['page']} : {preview}...")

        print()