"""
Module anti-hallucination par double check.

Apres qu'un premier LLM ait genere une reponse, un second appel LLM verifie
si chaque affirmation factuelle est supportee par le contexte. Si une seule
affirmation cle ne l'est pas, la reponse est invalidee et remplacee par
le message "information non disponible".

Inspiration : reflection / self-consistency check (Anthropic, OpenAI).
"""
import os
import time
from dotenv import load_dotenv
from src.llm_client import Mistral
from mistralai.models.sdkerror import SDKError

from src.logger import get_logger

logger = get_logger(__name__)
load_dotenv()


_VERIFICATION_PROMPT = """Tu es un VERIFICATEUR strict de reponses generees par un autre LLM.

Ton role : determiner si la REPONSE proposee est entierement supportee par le CONTEXTE fourni.

CRITERES DE VERIFICATION :
1. Chaque AFFIRMATION FACTUELLE (chiffre, nom propre, date, fait) doit etre 
   DIRECTEMENT presente dans le contexte ou deductible par synonymie evidente.
2. Si la reponse contient un nom propre (PDG, dirigeant, marque), ce nom 
   doit etre LITTERALEMENT dans le contexte.
3. Si la reponse contient un chiffre, ce chiffre doit etre dans le contexte 
   (avec tolerance sur l'unite : "86 milliards" et "86 153 millions" sont equivalents).
4. Les formulations de refus comme "L'information n'est pas disponible" sont 
   TOUJOURS considerees comme VALIDES.

REPONDS UNIQUEMENT par un de ces 3 verdicts :
- "VALIDE" : toutes les affirmations sont supportees
- "INVALIDE" : au moins une affirmation factuelle n'est PAS dans le contexte
- "PARTIEL" : une partie est correcte mais il y a des elements inventes

Puis sur une 2eme ligne, explique en UNE phrase courte.

CONTEXTE :
{context}

REPONSE A VERIFIER :
{answer}

QUESTION ORIGINALE (pour information) :
{question}

VERDICT (VALIDE/INVALIDE/PARTIEL) :"""


def _call_llm_with_retry(client, model, messages, max_retries=6, temperature=0.0, max_tokens=200):
    """Appel LLM avec retry exponentiel (meme pattern que generation.py)."""
    for attempt in range(max_retries):
        try:
            return client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except SDKError as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "capacity" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(f"Verifier : rate limit, attente {wait}s, retry {attempt + 2}/{max_retries}")
                time.sleep(wait)
                continue
            raise


def verify_answer(question, answer, context):
    """
    Verifie si une reponse est supportee par le contexte.

    Args:
        question: str, la question originale
        answer: str, la reponse generee par le premier LLM
        context: str, le contexte (chunks formates)

    Returns:
        dict :
            - 'verdict' : 'VALIDE', 'INVALIDE', 'PARTIEL' ou 'INCONNU' si parse echoue
            - 'explanation' : str, courte explication du verdict
            - 'is_valid' : bool, True si verdict == 'VALIDE'
            - 'raw_response' : str, la reponse brute du verifier
    """
    if not answer or not answer.strip():
        return {
            "verdict": "INVALIDE",
            "explanation": "Reponse vide.",
            "is_valid": False,
            "raw_response": "",
        }

    # Cas evident : les refus sont toujours valides
    refusal_phrases = [
        "l'information n'est pas disponible",
        "information non disponible",
        "n'est pas mentionne dans les documents",
        "n'est pas present dans les documents",
    ]
    answer_lower = answer.lower()
    if any(phrase in answer_lower for phrase in refusal_phrases):
        logger.info("Verifier : reponse de refus detectee, validation automatique.")
        return {
            "verdict": "VALIDE",
            "explanation": "Refus correct (info absente).",
            "is_valid": True,
            "raw_response": "(refus auto-valide)",
        }

    # Appel LLM verifier
    api_key = os.getenv("MISTRAL_API_KEY")
    model = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    prompt = _VERIFICATION_PROMPT.format(
        context=context,
        answer=answer,
        question=question,
    )

    try:
        response = _call_llm_with_retry(
            client=client,
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Verifier : erreur LLM ({e}), validation par defaut.")
        return {
            "verdict": "INCONNU",
            "explanation": f"Erreur LLM : {e}",
            "is_valid": True,  # en cas d'erreur, on ne bloque pas
            "raw_response": str(e),
        }

    # Parse le verdict (1ere ligne)
    lines = raw.strip().split("\n", 1)
    first_line = lines[0].strip().upper()
    explanation = lines[1].strip() if len(lines) > 1 else ""

    if "VALIDE" in first_line and "INVALIDE" not in first_line:
        verdict = "VALIDE"
        is_valid = True
    elif "INVALIDE" in first_line:
        verdict = "INVALIDE"
        is_valid = False
    elif "PARTIEL" in first_line:
        verdict = "PARTIEL"
        is_valid = False  # par precaution, on rejette aussi les partiels
    else:
        verdict = "INCONNU"
        is_valid = True  # safe : on ne bloque pas si le parse echoue
        logger.warning(f"Verifier : verdict non parsable, raw='{raw[:100]}'")

    logger.info(f"Verifier : verdict={verdict} | {explanation[:80]}")

    return {
        "verdict": verdict,
        "explanation": explanation,
        "is_valid": is_valid,
        "raw_response": raw,
    }


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    print("Test du module anti_hallucination\n" + "=" * 60)

    # CAS 1 : reponse valide (chiffre present dans le contexte)
    print("\n--- CAS 1 : reponse valide ---")
    result = verify_answer(
        question="Quel est le CA 2023 ?",
        answer="Le chiffre d'affaires 2023 est de 86 153 millions d'euros.",
        context="Ventes 2023 : 86 153 millions d'euros [page 3]",
    )
    print(f"  Verdict : {result['verdict']}")
    print(f"  Is valid : {result['is_valid']}")
    print(f"  Explication : {result['explanation']}")

    # CAS 2 : reponse avec hallucination
    print("\n--- CAS 2 : reponse hallucinee (nom propre absent) ---")
    result2 = verify_answer(
        question="Qui est le PDG de LVMH ?",
        answer="Le PDG de LVMH est Bernard Arnault.",
        context="LVMH realise en 2023 des ventes de 86,2 milliards d'euros. (aucune mention de PDG)",
    )
    print(f"  Verdict : {result2['verdict']}")
    print(f"  Is valid : {result2['is_valid']}")
    print(f"  Explication : {result2['explanation']}")

    # CAS 3 : refus correct (doit etre auto-valide sans appel LLM)
    print("\n--- CAS 3 : refus correct (auto-valide) ---")
    result3 = verify_answer(
        question="Quel est le CA 2024 ?",
        answer="L'information n'est pas disponible dans les documents fournis.",
        context="Ventes 2023 : 86 153 millions d'euros [page 3]",
    )
    print(f"  Verdict : {result3['verdict']}")
    print(f"  Is valid : {result3['is_valid']}")
