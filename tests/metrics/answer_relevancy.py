"""
Metrique 2 : ANSWER RELEVANCY

Mesure : la reponse repond-elle vraiment a la question posee ?

Approche simplifiee : on demande au juge de noter directement 0.0-1.0.
(L'approche RAGAS standard genere des questions a partir de la reponse,
mais c'est plus complexe pour un gain marginal sur notre use case.)

Score = 1.0 -> reponse parfaitement adressee a la question
Score = 0.0 -> reponse totalement hors-sujet
"""
from tests.llm_judge import judge, parse_score_0_1


_PROMPT_RELEVANCY = """Tu es un evaluateur expert. Note la pertinence de la reponse par rapport a la question.

CRITERES DE NOTATION (0.0 a 1.0) :
- 1.0 : la reponse repond directement et completement a la question.
- 0.7-0.9 : la reponse repond mais avec des digressions ou des manques mineurs.
- 0.4-0.6 : la reponse aborde le sujet mais ne repond pas vraiment.
- 0.1-0.3 : la reponse est en partie hors-sujet.
- 0.0 : la reponse est totalement hors-sujet ou vide.

IMPORTANT : on evalue UNIQUEMENT si la reponse adresse la question, PAS si elle est vraie ou fidele au contexte (c'est l'objet d'une autre metrique).

QUESTION :
{question}

REPONSE :
{answer}

NOTE (un seul nombre entre 0.0 et 1.0, sans explication) :"""


def compute_answer_relevancy(question, answer, context=None):
    """
    Calcule le score de pertinence de la reponse.

    Args:
        question: str
        answer: str
        context: ignore (garde pour symetrie d'API)

    Returns:
        dict avec 'score' (float 0.0-1.0)
    """
    if not answer or not answer.strip():
        return {"score": 0.0, "note": "Reponse vide."}

    prompt = _PROMPT_RELEVANCY.format(question=question, answer=answer)
    response = judge(prompt, max_tokens=20)
    score = parse_score_0_1(response)

    if score is None:
        return {"score": None, "note": f"Parse echoue. Reponse brute : {response}"}

    return {"score": score, "raw_response": response.strip()}


# Test
if __name__ == "__main__":
    print("Test answer relevancy\n" + "=" * 50)

    # Cas 1 : reponse pertinente
    print("\n--- Cas 1 : reponse pertinente ---")
    result = compute_answer_relevancy(
        question="Quel est le chiffre d'affaires 2023 ?",
        answer="Le chiffre d'affaires 2023 est de 86,2 milliards d'euros.",
    )
    print(f"Score : {result['score']}")

    # Cas 2 : reponse hors-sujet
    print("\n--- Cas 2 : reponse hors-sujet ---")
    result2 = compute_answer_relevancy(
        question="Quel est le chiffre d'affaires 2023 ?",
        answer="LVMH est un groupe de luxe fonde en 1987, present sur cinq continents.",
    )
    print(f"Score : {result2['score']}")

    # Cas 3 : partiellement pertinent
    print("\n--- Cas 3 : partiellement pertinent ---")
    result3 = compute_answer_relevancy(
        question="Quel est le chiffre d'affaires 2023 ?",
        answer="Le chiffre d'affaires a beaucoup progresse cette annee.",
    )
    print(f"Score : {result3['score']}")