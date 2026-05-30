"""
Metrique 4 : CONTEXT RECALL

Mesure : tous les elements de la ground truth (reponse attendue) sont-ils
presents dans les chunks remontes ?

Algorithme :
1. Decomposer la ground truth en elements factuels
2. Pour chaque element, demander au juge s'il est present dans le contexte
3. Score = nombre de OUI / nombre d'elements

Score = 1.0 -> tous les elements attendus sont dans les chunks (rappel parfait)
Score = 0.0 -> aucun element attendu n'est dans les chunks
"""
from tests.llm_judge import judge, parse_yes_no


_PROMPT_EXTRACT_GT_ELEMENTS = """Tu es un evaluateur expert. Decompose la reponse de reference en elements factuels distincts.

REGLES :
- Un element = un fait verifiable (chiffre, date, nom, relation).
- Limite a 6 elements maximum.
- Format : un element par ligne, prefixe par "- ".

REPONSE DE REFERENCE :
{ground_truth}

ELEMENTS FACTUELS :
"""


_PROMPT_CHECK_PRESENCE = """Tu es un evaluateur expert. Determine si l'element factuel suivant est PRESENT dans le contexte fourni.

REGLES :
- "OUI" si l'information se trouve dans le contexte (meme avec une formulation differente).
- "NON" si l'information est absente ou contradictoire.

Reponds UNIQUEMENT par "OUI" ou "NON".

CONTEXTE :
{context}

ELEMENT A CHERCHER :
{element}

VERDICT (OUI/NON) :"""


def _extract_gt_elements(ground_truth):
    """Decompose la ground truth en elements factuels."""
    prompt = _PROMPT_EXTRACT_GT_ELEMENTS.format(ground_truth=ground_truth)
    response = judge(prompt, max_tokens=500)
    elements = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            element = line.lstrip("- ").strip()
            if element and len(element) > 5:
                elements.append(element)
    return elements


def compute_context_recall(question, ground_truth, retrieved_chunks):
    """
    Calcule le recall du retrieval contre la ground truth.

    Args:
        question: str (pas utilise, garde pour symetrie)
        ground_truth: str, la reponse attendue
        retrieved_chunks: list[Document] ou list[str]

    Returns:
        dict avec 'score', 'n_elements', 'n_found', 'details'
    """
    if not ground_truth or not ground_truth.strip():
        return {"score": None, "note": "Pas de ground truth fournie."}

    # Concatene tous les chunks en un seul contexte
    if not retrieved_chunks:
        return {"score": 0.0, "n_elements": 0, "n_found": 0, "note": "Aucun chunk."}

    chunks_text = []
    for chunk in retrieved_chunks:
        if hasattr(chunk, "page_content"):
            chunks_text.append(chunk.page_content)
        else:
            chunks_text.append(str(chunk))
    context = "\n\n---\n\n".join(chunks_text)

    # Tronque si trop long
    if len(context) > 8000:
        context = context[:8000] + "..."

    # Extrait les elements de la GT
    elements = _extract_gt_elements(ground_truth)
    if not elements:
        return {"score": None, "note": "Aucun element extractible de la ground truth."}

    n_found = 0
    details = []
    for element in elements:
        prompt = _PROMPT_CHECK_PRESENCE.format(context=context, element=element)
        response = judge(prompt, max_tokens=10)
        is_present = parse_yes_no(response)
        if is_present is True:
            n_found += 1
            details.append((element, True))
        else:
            details.append((element, False))

    score = n_found / len(elements)
    return {
        "score": score,
        "n_elements": len(elements),
        "n_found": n_found,
        "details": details,
    }


# Test
if __name__ == "__main__":
    print("Test context recall\n" + "=" * 50)

    # Cas : retrieval qui rate des elements
    gt = "Le CA 2023 est 86 153 M€ et le resultat net est 15 174 M€."
    chunks = [
        "Ventes 2023 : 86 153 millions d'euros (+9% vs 2022).",
        "Tiffany & Co. a connu une excellente annee 2023.",
    ]

    result = compute_context_recall(
        question="CA et resultat net 2023",
        ground_truth=gt,
        retrieved_chunks=chunks,
    )
    print(f"Score : {result['score']}")
    print(f"Elements trouves : {result['n_found']}/{result['n_elements']}")
    for elem, found in result["details"]:
        print(f"  [{'OUI' if found else 'NON'}] {elem}")