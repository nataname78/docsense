"""
Test de recherche pure dans un index FAISS existant.
On NE re-indexe PAS : on recharge l'index sauvegarde et on interroge.
Demontre la valeur de la persistance.
"""
from src.vectorstore import load_vectorstore

print("Chargement de l'index FAISS depuis le disque (instantane)...")
vectorstore = load_vectorstore("data/indexes/sample")
print("OK, index charge.\n")

# Tape tes propres questions ici, adaptees au contenu de TON PDF
questions = [
    "Quel est le chiffre d'affaires du groupe en 2023 ?",
    "Comment a evolue l'activite Mode et Maroquinerie ?",
    "Combien de postes le groupe prevoit-il de creer en France ?",
    "Quelles sont les performances de Sephora ?",
]

for question in questions:
    print("=" * 70)
    print(f"QUESTION : {question}")
    print("=" * 70)
    results = vectorstore.similarity_search_with_score(question, k=3)
    for i, (doc, score) in enumerate(results, start=1):
        # similarity_search_with_score retourne la DISTANCE (plus c'est bas, plus c'est proche)
        print(f"\n  [{i}] Page {doc.metadata['page']} | distance L2 : {score:.3f}")
        preview = doc.page_content[:300].replace("\n", " ")
        print(f"      {preview}...")
    print()