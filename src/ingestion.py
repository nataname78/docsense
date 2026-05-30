"""
Module d'ingestion : lecture des fichiers PDF et extraction du texte brut.

Responsabilité unique : transformer un PDF (binaire) en texte exploitable
par le reste du pipeline. Aucune logique IA ici.
"""
import fitz  # PyMuPDF
from pathlib import Path


def extract_text_from_pdf(pdf_path):
    """
    Extrait le texte d'un PDF, page par page.

    Args:
        pdf_path: chemin vers le fichier PDF.

    Returns:
        dict avec :
        - 'text' : str, le texte concaténé du document
        - 'pages' : list[dict], une entrée par page avec son texte et son numéro
        - 'metadata' : dict, métadonnées du PDF (titre, auteur, nb pages...)
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    # Ouverture du PDF avec PyMuPDF
    doc = fitz.open(pdf_path)

    pages = []
    full_text = []

    for page_num, page in enumerate(doc, start=1):
        # get_text() extrait le texte avec un layout raisonnable
        page_text = page.get_text()
        pages.append({
            "page_number": page_num,
            "text": page_text,
        })
        full_text.append(page_text)

    metadata = {
        "file_name": pdf_path.name,
        "n_pages": len(doc),
        "title": doc.metadata.get("title", ""),
        "author": doc.metadata.get("author", ""),
    }

    doc.close()

    return {
        "text": "\n\n".join(full_text),
        "pages": pages,
        "metadata": metadata,
    }


# Bloc de test : si on lance ce fichier directement, il fait un test
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion <chemin_du_pdf>")
        sys.exit(1)

    result = extract_text_from_pdf(sys.argv[1])
    print(f"Fichier : {result['metadata']['file_name']}")
    print(f"Pages   : {result['metadata']['n_pages']}")
    print(f"Auteur  : {result['metadata']['author']}")
    print(f"\n--- Apercu des 500 premiers caracteres ---\n")
    print(result["text"][:500])
    print(f"\n--- Longueur totale : {len(result['text'])} caracteres ---")