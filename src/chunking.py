"""
Module de chunking : decoupage du texte en chunks exploitables par le LLM.

On utilise RecursiveCharacterTextSplitter de LangChain, qui decoupe
en respectant la structure naturelle du texte (paragraphes -> phrases -> mots).

Chaque chunk conserve ses metadonnees (numero de page, fichier source)
pour permettre la citation des sources dans la reponse finale.
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


# Configuration par defaut du chunking
# 3200 caracteres ~ 800 tokens (regle empirique : 1 token ~ 4 caracteres en francais)
DEFAULT_CHUNK_SIZE = 3200
DEFAULT_CHUNK_OVERLAP = 400


def chunk_pages(pages, file_name, chunk_size=DEFAULT_CHUNK_SIZE, chunk_overlap=DEFAULT_CHUNK_OVERLAP):
    """
    Decoupe une liste de pages en chunks, en preservant les metadonnees.

    Args:
        pages: liste de dicts [{'page_number': int, 'text': str}, ...]
               (format produit par src.ingestion.extract_text_from_pdf)
        file_name: nom du fichier PDF d'origine (pour la metadata)
        chunk_size: taille cible d'un chunk en caracteres
        chunk_overlap: chevauchement entre chunks consecutifs en caracteres

    Returns:
        list[Document]: liste de Documents LangChain.
        Chaque Document a :
        - .page_content : str, le texte du chunk
        - .metadata : dict avec 'source' (file_name) et 'page' (numero de page)
    """
    # Initialise le splitter avec sa hierarchie de separateurs
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Hierarchie : on coupe d'abord sur paragraphes, puis lignes, puis phrases, puis mots
        separators=["\n\n", "\n", ". ", " ", ""],
        # length_function par defaut = len() (compte les caracteres)
        length_function=len,
    )

    all_chunks = []

    # On chunke page par page pour preserver l'info "ce chunk vient de la page X"
    for page in pages:
        page_text = page["text"]
        page_number = page["page_number"]

        # Skip les pages vides (ca arrive : pages de garde, separateurs...)
        if not page_text.strip():
            continue

        # Decoupe le texte de la page en chunks
        page_chunks_text = splitter.split_text(page_text)

        # Pour chaque chunk, on cree un Document avec ses metadonnees
        for chunk_text in page_chunks_text:
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": file_name,
                    "page": page_number,
                },
            )
            all_chunks.append(doc)

    return all_chunks


# Bloc de test : permet de lancer ce module directement pour voir les chunks
if __name__ == "__main__":
    import sys
    from src.ingestion import extract_text_from_pdf

    if len(sys.argv) < 2:
        print("Usage: python -m src.chunking <chemin_du_pdf>")
        sys.exit(1)

    # Etape 1 : extraire le texte du PDF (Phase 1)
    print(f"Extraction du PDF : {sys.argv[1]}")
    result = extract_text_from_pdf(sys.argv[1])
    print(f"  -> {result['metadata']['n_pages']} pages, {len(result['text'])} caracteres")

    # Etape 2 : chunker
    print(f"\nChunking en cours (taille={DEFAULT_CHUNK_SIZE}, overlap={DEFAULT_CHUNK_OVERLAP})...")
    chunks = chunk_pages(
        pages=result["pages"],
        file_name=result["metadata"]["file_name"],
    )
    print(f"  -> {len(chunks)} chunks generes")

    # Etape 3 : afficher les 3 premiers chunks pour visualiser
    print("\n" + "=" * 70)
    print("APERCU DES 3 PREMIERS CHUNKS")
    print("=" * 70)
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk #{i + 1} ---")
        print(f"Source : {chunk.metadata['source']} | Page : {chunk.metadata['page']}")
        print(f"Taille : {len(chunk.page_content)} caracteres")
        print(f"Contenu :")
        print(chunk.page_content[:400])
        if len(chunk.page_content) > 400:
            print(f"... [tronque, total {len(chunk.page_content)} caracteres]")

# Experience : comparons differentes tailles de chunks
    print("\n" + "=" * 70)
    print("EXPERIENCE : impact de la taille de chunk")
    print("=" * 70)
    for size in [500, 1500, 3200, 6000]:
        chunks_exp = chunk_pages(
            pages=result["pages"],
            file_name=result["metadata"]["file_name"],
            chunk_size=size,
            chunk_overlap=int(size * 0.125),  # 12.5% d'overlap
        )
        print(f"chunk_size={size:5d} -> {len(chunks_exp):4d} chunks")

    # Etape 4 : statistiques globales
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    sizes = [len(c.page_content) for c in chunks]
    print(f"Nombre de chunks    : {len(chunks)}")
    print(f"Taille moyenne      : {sum(sizes) // len(sizes)} caracteres")
    print(f"Taille min / max    : {min(sizes)} / {max(sizes)} caracteres")
    print(f"Pages couvertes     : de {chunks[0].metadata['page']} a {chunks[-1].metadata['page']}")