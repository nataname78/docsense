"""
Module vectorstore : gestion de l'index FAISS.

V2.4 : support multi-document
- build_vectorstore : creer un nouvel index a partir de chunks
- add_chunks_to_vectorstore : ajouter des chunks a un index existant
- save / load comme avant
"""
from pathlib import Path
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embeddings_model
from src.logger import get_logger
logger = get_logger(__name__)


def build_vectorstore(documents):
    """
    Construit un index FAISS a partir d'une liste de Documents LangChain.
    """
    embed_model = get_embeddings_model()
    vectorstore = FAISS.from_documents(
        documents=documents,
        embedding=embed_model,
    )
    return vectorstore


def add_chunks_to_vectorstore(vectorstore, new_documents):
    """
    Ajoute de nouveaux chunks a un index FAISS existant.

    FAISS supporte add_documents() qui fait le travail :
    - embed les nouveaux chunks via le meme modele
    - les ajoute a l'index existant
    - leurs metadonnees sont preservees

    Args:
        vectorstore: index FAISS existant
        new_documents: list[Document] a ajouter

    Returns:
        vectorstore: le meme index, enrichi
    """
    vectorstore.add_documents(new_documents)
    return vectorstore


def save_vectorstore(vectorstore, index_path):
    """Sauvegarde un index FAISS sur disque."""
    index_path = Path(index_path)
    index_path.mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(str(index_path))
    print(f"  Index sauvegarde dans : {index_path}")


def load_vectorstore(index_path):
    """Recharge un index FAISS existant depuis le disque."""
    index_path = Path(index_path)
    if not index_path.exists():
        raise FileNotFoundError(f"Index FAISS introuvable : {index_path}")

    embed_model = get_embeddings_model()
    vectorstore = FAISS.load_local(
        folder_path=str(index_path),
        embeddings=embed_model,
        allow_dangerous_deserialization=True,
    )
    return vectorstore