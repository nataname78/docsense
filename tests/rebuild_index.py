"""
Script utilitaire : reconstruit l'index FAISS a partir de TOUS les PDFs
presents dans data/raw/.

A lancer avant evaluate_rag.py si l'index existant ne contient pas tous les docs.
"""
from pathlib import Path
from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_pages
from src.table_extractor import enrich_pages_with_tables
from src.vectorstore import build_vectorstore, save_vectorstore


def rebuild():
    raw_dir = Path("data/raw")
    pdf_files = sorted([p for p in raw_dir.glob("*.pdf")])

    if not pdf_files:
        print("ERREUR : aucun PDF dans data/raw/")
        return

    print(f"Indexation de {len(pdf_files)} PDF(s) ensemble :")
    for p in pdf_files:
        print(f"  - {p.name}")

    all_chunks = []
    for pdf_path in pdf_files:
        print(f"\n[Extraction] {pdf_path.name}...")
        result = extract_text_from_pdf(str(pdf_path))
        pages = enrich_pages_with_tables(result["pages"], str(pdf_path))
        chunks = chunk_pages(pages=pages, file_name=result["metadata"]["file_name"])
        print(f"  -> {len(chunks)} chunks")
        all_chunks.extend(chunks)

    print(f"\n[Total] {len(all_chunks)} chunks au total")
    print("[Indexation FAISS] (~30s pour les embeddings Mistral)...")
    vectorstore = build_vectorstore(all_chunks)

    save_path = "data/indexes/sample"
    print(f"[Sauvegarde] vers {save_path}...")
    save_vectorstore(vectorstore, save_path)

    print("\nOK ! Index reconstruit avec tous les PDF.")


if __name__ == "__main__":
    rebuild()