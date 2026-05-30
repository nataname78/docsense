"""
Metrique 5 : REFUSAL QUALITY

Mesure : quand une question est impossible (info absente du corpus),
le systeme a-t-il bien refuse au lieu d'halluciner ?

S'applique UNIQUEMENT aux questions du golden dataset marquees avec
expected_behavior == "refusal".

Algorithme :
1. Verifie si la reponse contient une phrase de refus claire
2. Verifie qu'aucune affirmation factuelle inventee n'est presente
3. Combine en un score 0.0-1.0

Score = 1.0 -> refus parfait : explicite + zero fabrication
Score = 0.5 -> refus implicite ou avec ambiguites
Score = 0.0 -> hallucination (a invente une reponse au lieu de refuser)
"""
import re
from tests.llm_judge import judge, parse_yes_no


# Phrases reconnues comme refus explicite (insensible casse/ponctuation)
_REFUSAL_PATTERNS = [
    r"l'?information n'est pas disponible",
    r"information non disponible",
    r"n'est pas mentionn[ée]\w* dans les documents",
    r"n'est pas pr[ée]sent\w* dans les documents",
    r"je ne peux pas r[ée]pondre",
    r"je n'ai pas l'information",
    r"je ne dispose pas",
    r"impossible de r[ée]pondre",
    r"pas mentionn[ée]\w* dans les extraits",
]


_PROMPT_DETECT_FABRICATION = """Tu es un evaluateur expert. La question ci-dessous a une reponse VOLONTAIREMENT impossible (l'info n'existe pas dans le contexte).

La reponse du systeme contient-elle des affirmations factuelles INVENTEES (chiffres, noms propres, dates specifiques) qui ne sont pas dans le contexte ?

REGLES :
- "OUI" : si la reponse INVENTE des faits (un nom de PDG, un chiffre, une date specifique).
- "NON" : si la reponse refuse honnetement OU si elle ne fait que reformuler le contexte sans inventer.

Une reponse qui refuse en disant "information non disponible" est consideree comme "NON" (pas d'invention).

QUESTION : {question}

CONTEXTE (incomplet, l'info n'y est pas) :
{context}

REPONSE DU SYSTEME :
{answer}

VERDICT (OUI = a invente / NON = pas d'invention) :"""


def _detect_explicit_refusal(answer):
    """
    Detecte si la reponse contient une formule de refus explicite.
    Retourne True/False.
    """
    if not answer:
        return False
    answer_lower = answer.lower()
    for pattern in _REFUSAL_PATTERNS:
        if re.search(pattern, answer_lower):
            return True
    return False


def _detect_fabrication(question, answer, context):
    """
    Verifie via LLM si la reponse contient des affirmations inventees.
    Retourne True (a invente) / False (pas d'invention) / None (parse echec).
    """
    prompt = _PROMPT_DETECT_FABRICATION.format(
        question=question,
        context=context[:3000] if context else "(contexte vide)",
        answer=answer,
    )
    response = judge(prompt, max_tokens=10)
    return parse_yes_no(response)


def compute_refusal_quality(question, answer, context):
    """
    Calcule le score de qualite du refus pour une question impossible.

    Args:
        question: str, la question
        answer: str, la reponse du systeme
        context: str, le contexte fourni au RAG (peut etre vide ou non-pertinent)

    Returns:
        dict :
            - 'score' : float 0.0-1.0
            - 'is_explicit_refusal' : bool, refus formel detecte
            - 'has_fabrication' : bool, hallucination detectee
            - 'note' : str, explication du score
    """
    if not answer or not answer.strip():
        return {
            "score": 0.0,
            "is_explicit_refusal": False,
            "has_fabrication": False,
            "note": "Reponse vide.",
        }

    # Etape 1 : detecter le refus explicite (regex, pas d'appel LLM)
    is_explicit = _detect_explicit_refusal(answer)

    # Etape 2 : detecter une fabrication via LLM
    fabrication_detected = _detect_fabrication(question, answer, context)
    has_fabrication = fabrication_detected is True

    # Etape 3 : calcul du score
    # Scoring scheme :
    #   Refus explicite + zero fabrication = 1.0 (parfait)
    #   Refus implicite + zero fabrication = 0.75 (acceptable)
    #   Refus explicite + fabrication = 0.5 (refuse mais a quand meme invente des trucs)
    #   Pas de refus + fabrication = 0.0 (echec total)
    #   Pas de refus + pas de fabrication = 0.25 (cas bizarre : ne refuse pas mais n'invente pas)

    if is_explicit and not has_fabrication:
        score = 1.0
        note = "Refus explicite + zero hallucination. Parfait."
    elif is_explicit and has_fabrication:
        score = 0.5
        note = "Refus explicite, mais reponse contient aussi des elements inventes."
    elif not is_explicit and not has_fabrication:
        score = 0.75
        note = "Pas de refus explicite, mais pas d'hallucination non plus."
    elif not is_explicit and has_fabrication:
        score = 0.0
        note = "Hallucination : le systeme a invente une reponse au lieu de refuser."
    else:
        score = 0.0
        note = "Cas non gere."

    return {
        "score": score,
        "is_explicit_refusal": is_explicit,
        "has_fabrication": has_fabrication,
        "note": note,
    }


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    print("Test de refusal_quality\n" + "=" * 60)

    # Cas 1 : refus parfait
    print("\n--- Cas 1 : refus parfait ---")
    r1 = compute_refusal_quality(
        question="Quel est le nom du PDG de LVMH ?",
        answer="L'information n'est pas disponible dans les documents fournis.",
        context="LVMH realise en 2023 des ventes de 86,2 milliards d'euros.",
    )
    print(f"  Score : {r1['score']}")
    print(f"  Explicit refusal : {r1['is_explicit_refusal']}")
    print(f"  Fabrication : {r1['has_fabrication']}")
    print(f"  Note : {r1['note']}")

    # Cas 2 : hallucination totale
    print("\n--- Cas 2 : hallucination ---")
    r2 = compute_refusal_quality(
        question="Quel est le nom du PDG de LVMH ?",
        answer="Le PDG de LVMH est Bernard Arnault.",
        context="LVMH realise en 2023 des ventes de 86,2 milliards d'euros.",
    )
    print(f"  Score : {r2['score']}")
    print(f"  Explicit refusal : {r2['is_explicit_refusal']}")
    print(f"  Fabrication : {r2['has_fabrication']}")
    print(f"  Note : {r2['note']}")

    # Cas 3 : refus implicite (ne dit pas explicitement non, mais ne hallucine pas)
    print("\n--- Cas 3 : refus implicite (verbose) ---")
    r3 = compute_refusal_quality(
        question="Quel est le PDG de LVMH ?",
        answer="Les documents fournis concernent les ventes et la performance financiere de LVMH, mais ne contiennent pas d'information sur la direction de l'entreprise.",
        context="LVMH realise en 2023 des ventes de 86,2 milliards d'euros.",
    )
    print(f"  Score : {r3['score']}")
    print(f"  Explicit refusal : {r3['is_explicit_refusal']}")
    print(f"  Fabrication : {r3['has_fabrication']}")
    print(f"  Note : {r3['note']}")