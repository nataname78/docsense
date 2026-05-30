"""
Script temporaire de debug du retrieval.
A executer apres la V3.1/V3.2 pour diagnostiquer pourquoi 'chiffre d'affaires'
ne remonte pas le bon chunk alors que 'ventes' fonctionne.
"""
from pathlib import Path
from src.vectorstore import load_vectorstore
from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_pages
from src.table_extractor import enrich_pages_with_tables
from src.retrieval import HybridRetriever


def main():
    print("=" * 70)
    print("DEBUG RETRIEVAL - 'chiffre d'affaires' vs 'ventes'")
    print("=" * 70)

    # Charge l'index
    print("\n[Setup] Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    # Reconstruit les chunks (texte + tableaux)
    print("[Setup] Reconstruction des chunks...")
    all_chunks = []
    for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
        result = extract_text_from_pdf(str(pdf_path))
        pages = enrich_pages_with_tables(result["pages"], str(pdf_path))
        chunks = chunk_pages(pages=pages, file_name=pdf_path.name)
        all_chunks.extend(chunks)
    print(f"[Setup] Total : {len(all_chunks)} chunks reconstruits.")

    # Construit le retriever
    retriever = HybridRetriever(vectorstore=vs, chunks=all_chunks)

    # ============ TEST 1 : la question qui PLANTE ============
    print("\n" + "=" * 70)
    print("TEST 1 : 'Quel est le chiffre d affaires en 2023 ?'")
    print("=" * 70)

    chunks1 = retriever.retrieve("Quel est le chiffre d affaires en 2023 ?", k=4)
    for i, c in enumerate(chunks1, start=1):
        print(f"\n[Chunk {i}] source: {c.metadata['source']} | page: {c.metadata['page']}")
        preview = c.page_content[:400]
        if len(c.page_content) > 400:
            preview += "\n... (tronque dans l'affichage)"
        print(preview)
        print("-" * 70)

    # ============ TEST 2 : la question qui MARCHE ============
    print("\n" + "=" * 70)
    print("TEST 2 : 'Quelles sont les ventes en 2023 ?'")
    print("=" * 70)

    chunks2 = retriever.retrieve("Quelles sont les ventes en 2023 ?", k=4)
    for i, c in enumerate(chunks2, start=1):
        print(f"\n[Chunk {i}] source: {c.metadata['source']} | page: {c.metadata['page']}")
        preview = c.page_content[:400]
        if len(c.page_content) > 400:
            preview += "\n... (tronque dans l'affichage)"
        print(preview)
        print("-" * 70)

    # ============ COMPARAISON ============
    print("\n" + "=" * 70)
    print("COMPARAISON des chunks remontes")
    print("=" * 70)
    pages_q1 = [(c.metadata["source"], c.metadata["page"]) for c in chunks1]
    pages_q2 = [(c.metadata["source"], c.metadata["page"]) for c in chunks2]
    print(f"Q1 (CA)     : {pages_q1}")
    print(f"Q2 (ventes) : {pages_q2}")
    overlap = set(pages_q1) & set(pages_q2)
    print(f"Pages communes : {overlap if overlap else 'AUCUNE'}")


if __name__ == "__main__":
    main()