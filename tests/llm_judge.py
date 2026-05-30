"""
Moteur LLM-as-judge : appelle Mistral pour evaluer des reponses.

Centralise :
- Le client Mistral
- Le retry en cas de 429
- Une fonction generique 'judge' qui prend un prompt et retourne une reponse propre

Les 4 metriques (faithfulness, relevancy, precision, recall) utilisent toutes ce moteur.
"""
import os
import time
import re
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

load_dotenv()


_client = None


def _get_client():
    """Singleton client Mistral."""
    global _client
    if _client is None:
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY manquante dans .env")
        _client = Mistral(api_key=api_key)
    return _client


def judge(prompt, max_retries=6, temperature=0.0, max_tokens=600):
    """
    Appelle Mistral comme juge sur un prompt donne.

    temperature=0.0 : on veut des reponses STABLES (peu de variance)
                      pour que les scores soient reproductibles.

    Returns:
        str: la reponse texte du LLM
    """
    client = _get_client()
    model = os.getenv("LLM_MODEL", "mistral-small-latest")

    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except SDKError as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "capacity" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [Judge: rate limit, attente {wait}s, retry {attempt + 2}/{max_retries}]")
                time.sleep(wait)
                continue
            raise


def parse_yes_no(text):
    """
    Parse une reponse 'OUI' / 'NON' (tolerant a la casse, espaces, ponctuation).

    Retourne True (OUI), False (NON), ou None (parse echoue).
    """
    if not text:
        return None
    cleaned = text.strip().upper()
    # Cherche OUI ou NON en debut de reponse
    if re.match(r"^\s*(OUI|YES)\b", cleaned):
        return True
    if re.match(r"^\s*(NON|NO)\b", cleaned):
        return False
    # Fallback : si OUI/NON apparait n'importe ou
    if "OUI" in cleaned or "YES" in cleaned:
        return True
    if "NON" in cleaned or " NO " in f" {cleaned} ":
        return False
    return None


def parse_score_0_1(text):
    """
    Parse une note 0.0-1.0 dans la reponse du LLM.

    Cherche en priorite un nombre decimal entre 0 et 1.
    Fallback : un entier 0-10 a normaliser.
    """
    if not text:
        return None
    # Cherche un decimal 0.X ou 1.0
    match = re.search(r"\b([01](?:\.\d+)?)\b", text)
    if match:
        try:
            value = float(match.group(1))
            if 0.0 <= value <= 1.0:
                return value
        except ValueError:
            pass
    # Fallback : un entier 0-10
    match = re.search(r"\b(\d{1,2})\b", text)
    if match:
        try:
            value = int(match.group(1))
            if 0 <= value <= 10:
                return value / 10.0
        except ValueError:
            pass
    return None


# Test
if __name__ == "__main__":
    print("Test du LLM-as-judge\n" + "=" * 50)

    # Test simple : OUI / NON
    prompt = (
        "Reponds par 'OUI' ou 'NON' uniquement.\n"
        "La phrase 'Le ciel est bleu' est-elle vraie ?"
    )
    response = judge(prompt)
    print(f"Reponse brute : {response}")
    print(f"Parse OUI/NON : {parse_yes_no(response)}")

    # Test note
    prompt2 = (
        "Note de 0.0 a 1.0 la pertinence de la reponse 'Le CA est 86 Md€' "
        "a la question 'Quel est le CA ?'. Reponds UNIQUEMENT par le nombre."
    )
    response2 = judge(prompt2)
    print(f"\nReponse brute : {response2}")
    print(f"Parse score   : {parse_score_0_1(response2)}")