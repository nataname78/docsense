"""
Metrique 3 : CONTEXT PRECISION

Mesure : parmi les chunks remontes, combien sont vraiment pertinents
pour repondre a la question ?

Algorithme :
1. Pour chaque chunk dans la liste retrieved_chunks
2. Demander au juge : "ce chunk est-il utile pour repondre a la question ?"
3. Score = nombre de OUI / nombre total de chunks

Score = 1.0 -> tous les chunks sont pertinents (retrieval parfait)
Score = 0.0 -> aucun chunk n'est pertinent
"""
from tests.llm_judge import judge, parse_yes_no


_PROMPT_CHECK_RELEVANCE = """Tu es un evaluateur expert. Determine si le chunk de texte ci-dessous est UTILE pour repondre a la question.

REGLES :
- "OUI" si le chunk contient des informations qui aident a repondre, meme partiellement.
- "NON" si le chunk est hors-sujet, redondant ou ne contient rien d'utile pour cette question precise.

Reponds UNIQUEMENT par "OUI" ou "NON".

QUESTION :
{question}

CHUNK A EVALUER :
{chunk}

VERDICT (OUI/NON) :"""


def compute_context_precision(question, retrieved_chunks):
    """
    Calcule la precision du retrieval.

    Args:
        question: str
        retrieved_chunks: list[Document] ou list[str]

    Returns:
        dict avec 'score', 'n_total', 'n_relevant', 'details'
    """
    if not retrieved_chunks:
        return {"score": 0.0, "n_total": 0, "n_relevant": 0, "note": "Aucun chunk."}

    n_relevant = 0
    details = []

    for i, chunk in enumerate(retrieved_chunks):
        # Support des objets Document ou des strings
        if hasattr(chunk, "page_content"):
            chunk_text = chunk.page_content
        else:
            chunk_text = str(chunk)

        # Tronque si trop long pour ne pas exploser le contexte du juge
        if len(chunk_text) > 2000:
            chunk_text = chunk_text[:2000] + "..."

        prompt = _PROMPT_CHECK_RELEVANCE.format(
            question=question,
            chunk=chunk_text,
        )
        response = judge(prompt, max_tokens=10)
        is_relevant = parse_yes_no(response)
        if is_relevant is True:
            n_relevant += 1
            details.append((i + 1, True))
        else:
            details.append((i + 1, False))

    score = n_relevant / len(retrieved_chunks)
    return {
        "score": score,
        "n_total": len(retrieved_chunks),
        "n_relevant": n_relevant,
        "details": details,
    }


# Test
if __name__ == "__main__":
    print("Test context precision\n" + "=" * 50)

    chunks_test = [
        "Ventes 2023 : 86 153 millions d'euros, en progression de 9%.",
        "Tiffany & Co. a ouvert un nouveau magasin a New York en 2023.",
        "Le resultat operationnel courant 2023 atteint 22,8 milliards d'euros.",
        "Recette de cuisine : mettre l'eau a bouillir et ajouter le sel.",
    ]

    result = compute_context_precision(
        question="Quel est le chiffre d'affaires 2023 ?",
        retrieved_chunks=chunks_test,
    )
    print(f"Score : {result['score']}")
    print(f"Pertinents : {result['n_relevant']}/{result['n_total']}")
    for idx, rel in result["details"]:
        print(f"  Chunk {idx} : {'OUI' if rel else 'NON'}")