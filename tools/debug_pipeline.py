"""
Test du pipeline COMPLET (retrieval + formatage + LLM) sans Streamlit.
Simule exactement ce que fait l'app avec la question 'chiffre d'affaires'.
"""
from pathlib import Path
from src.vectorstore import load_vectorstore
from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_pages
from src.table_extractor import enrich_pages_with_tables
from src.retrieval import HybridRetriever
from src.pipeline import ask

print("=" * 70)
print("TEST PIPELINE COMPLET (comme l'app Streamlit)")
print("=" * 70)

# Setup identique a l'app
print("\n[Setup] Chargement de l'index FAISS...")
vs = load_vectorstore("data/indexes/sample")

print("[Setup] Reconstruction des chunks...")
all_chunks = []
for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
    result = extract_text_from_pdf(str(pdf_path))
    pages = enrich_pages_with_tables(result["pages"], str(pdf_path))
    chunks = chunk_pages(pages=pages, file_name=pdf_path.name)
    all_chunks.extend(chunks)
print(f"[Setup] Total : {len(all_chunks)} chunks.")

retriever = HybridRetriever(vectorstore=vs, chunks=all_chunks)

# Appel EXACTEMENT comme dans l'app
print("\n[Pipeline] Appel ask() avec la question piege...")
question = "Quel est le chiffre d'affaires en 2023 ?"

result = ask(
    retriever,
    question,
    k=4,
    use_hybrid=True,
    use_reranking=True,
    retrieve_n=20,
    source_filter=None,
)

print("\n" + "=" * 70)
print("REPONSE DU PIPELINE :")
print("=" * 70)
print(result["answer"])

print("\n" + "=" * 70)
print("SOURCES UTILISEES :")
print("=" * 70)
for i, src in enumerate(result["sources"], start=1):
    print(f"\n[{i}] {src.metadata['source']} - page {src.metadata['page']}")
    print(src.page_content[:300])
    if len(src.page_content) > 300:
        print("... (tronque)")