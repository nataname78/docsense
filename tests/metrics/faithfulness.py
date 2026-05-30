"""
Metrique 1 : FAITHFULNESS

Mesure : la reponse est-elle FIDELE au contexte fourni ?

Algorithme :
1. Decomposer la reponse en affirmations atomiques
2. Pour chaque affirmation, demander au juge si elle est supportee par le contexte
3. Score = nombre de OUI / nombre total d'affirmations

Score = 1.0 -> aucune hallucination, parfait
Score = 0.0 -> tout est invente
"""
from tests.llm_judge import judge, parse_yes_no


_PROMPT_EXTRACT_CLAIMS = """Tu es un evaluateur expert. Decompose la reponse ci-dessous en affirmations factuelles atomiques.

REGLES :
- Une affirmation = un fait verifiable (un chiffre, une date, un nom, une relation).
- Ignore les formules de politesse, les transitions, les meta-commentaires.
- Limite a 8 affirmations maximum (les plus importantes).
- Format de sortie : une affirmation par ligne, prefixee par "- ".

REPONSE A ANALYSER :
{answer}

AFFIRMATIONS ATOMIQUES :
"""


_PROMPT_CHECK_CLAIM = """Tu es un evaluateur expert. Determine si l'affirmation suivante est SUPPORTEE par le contexte fourni.

REGLES STRICTES :
- "OUI" si l'affirmation peut etre DIRECTEMENT deduite du contexte (meme paraphrase).
- "NON" si l'affirmation n'est pas dans le contexte OU contredit le contexte OU n'est pas verifiable.
- Sois STRICT : un chiffre legerement different (15 200 vs 15 174) = NON.
- Reponds UNIQUEMENT par "OUI" ou "NON", rien d'autre.

CONTEXTE :
{context}

AFFIRMATION :
{claim}

VERDICT (OUI/NON) :"""


def _extract_claims(answer):
    """Decompose une reponse en affirmations atomiques via le LLM."""
    prompt = _PROMPT_EXTRACT_CLAIMS.format(answer=answer)
    response = judge(prompt, max_tokens=500)
    # Parse les lignes commencant par "-"
    claims = []
    for line in response.split("\n"):
        line = line.strip()
        if line.startswith("-"):
            claim = line.lstrip("- ").strip()
            if claim and len(claim) > 5:  # filtre lignes trop courtes
                claims.append(claim)
    return claims


def compute_faithfulness(question, answer, context):
    """
    Calcule le score de faithfulness pour une (question, reponse, contexte).

    Args:
        question: str (pas utilise directement mais garde pour symetrie API)
        answer: str, la reponse generee par le RAG
        context: str, le contexte fourni au RAG (chunks formates)

    Returns:
        dict avec :
        - 'score' : float 0.0-1.0
        - 'n_claims' : int, nombre d'affirmations extraites
        - 'n_supported' : int, nombre d'affirmations supportees
        - 'claims_detail' : list[(claim, supported)] pour audit
    """
    claims = _extract_claims(answer)

    if not claims:
        # Pas d'affirmations extraites : on ne peut pas evaluer
        return {
            "score": None,
            "n_claims": 0,
            "n_supported": 0,
            "claims_detail": [],
            "note": "Aucune affirmation extractible.",
        }

    supported_count = 0
    details = []
    for claim in claims:
        prompt = _PROMPT_CHECK_CLAIM.format(context=context, claim=claim)
        response = judge(prompt, max_tokens=10)
        is_supported = parse_yes_no(response)
        if is_supported is True:
            supported_count += 1
            details.append((claim, True))
        else:
            details.append((claim, False))

    score = supported_count / len(claims)
    return {
        "score": score,
        "n_claims": len(claims),
        "n_supported": supported_count,
        "claims_detail": details,
    }


# Test
if __name__ == "__main__":
    print("Test faithfulness\n" + "=" * 50)

    # Cas 1 : reponse 100% fidele
    print("\n--- Cas 1 : reponse fidele ---")
    result = compute_faithfulness(
        question="Quel est le CA 2023 ?",
        answer="Le CA 2023 de LVMH est de 86 153 millions d'euros.",
        context="Ventes 2023 : 86 153 millions d'euros [page 3]",
    )
    print(f"Score : {result['score']}")
    print(f"Affirmations : {result['n_claims']}, supportees : {result['n_supported']}")
    for claim, supp in result["claims_detail"]:
        print(f"  - [{supp}] {claim}")

    # Cas 2 : reponse avec hallucination
    print("\n--- Cas 2 : reponse avec hallucination ---")
    result2 = compute_faithfulness(
        question="Quel est le CA 2023 ?",
        answer="Le CA 2023 de LVMH est de 86 153 millions d'euros. Le PDG est Bernard Arnault.",
        context="Ventes 2023 : 86 153 millions d'euros [page 3]",
    )
    print(f"Score : {result2['score']}")
    print(f"Affirmations : {result2['n_claims']}, supportees : {result2['n_supported']}")
    for claim, supp in result2["claims_detail"]:
        print(f"  - [{supp}] {claim}")
    