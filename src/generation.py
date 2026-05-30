"""
Module generation : appel au LLM Mistral pour produire la reponse finale.

Le LLM recoit :
- Le prompt system (defini dans prompts/system_prompt.txt)
- Les chunks pertinents formates en contexte
- La question utilisateur

Et retourne une reponse en francais, fondee sur le contexte, avec citations.
"""
import os
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from src.anti_hallucination import verify_answer

load_dotenv()


# Charge le prompt system une fois au demarrage du module
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "system_prompt.txt"
SYSTEM_PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8")


def _call_llm_with_retry(client, model, messages, max_retries=6, temperature=0.2, max_tokens=1024):
    """
    Appelle Mistral chat avec retry exponentiel sur 429 (rate limit free tier).
    """
    for attempt in range(max_retries):
        try:
            response = client.chat.complete(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response
        except SDKError as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "capacity" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [Rate limit, attente {wait}s, retry {attempt + 2}/{max_retries}...]")
                time.sleep(wait)
                continue
            raise


def _call_llm_stream_with_retry(client, model, messages, max_retries=6, temperature=0.2, max_tokens=1024):
    """
    Variante streaming : retourne un GENERATEUR qui yield les morceaux de reponse.

    Comme _call_llm_with_retry, mais utilise client.chat.stream() au lieu de complete().
    Mistral SDK retourne un iterateur de chunks ; chaque chunk a un delta.content
    contenant le morceau de texte a ajouter.
    """
    for attempt in range(max_retries):
        try:
            # IMPORTANT : on appelle stream() qui retourne un generateur Mistral
            response_stream = client.chat.stream(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # On itere sur les chunks recus
            for chunk in response_stream:
                # Le SDK Mistral retourne des objets avec data.choices[0].delta.content
                # Le contenu est dans le delta (parfois None si chunk de controle)
                if chunk.data.choices and chunk.data.choices[0].delta.content:
                    yield chunk.data.choices[0].delta.content
                    time.sleep(0.02)  # 20ms par token = ~50 tokens/sec, plus visible
            return  # streaming termine normalement
        except SDKError as e:
            error_str = str(e)
            is_rate_limit = "429" in error_str or "capacity" in error_str.lower()
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"  [Stream rate limit, attente {wait}s, retry {attempt + 2}/{max_retries}]")
                time.sleep(wait)
                continue
            raise


def generate_answer(question, context, temperature=0.2):
    """
    Genere la reponse finale en appelant Mistral chat.

    Args:
        question: str, la question utilisateur
        context: str, les chunks formates (sortie de format_chunks_for_prompt)
        temperature: float, 0=deterministe, 1=cretif.
                     Pour un RAG, on veut BAS (0.1-0.3) pour eviter les inventions.

    Returns:
        str: la reponse generee
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    # On injecte le contexte et la question dans le template de prompt
    full_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    # Mistral chat utilise un format messages (comme OpenAI)
    messages = [
        {"role": "user", "content": full_prompt}
    ]

    response = _call_llm_with_retry(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

    return response.choices[0].message.content

def generate_answer_stream(question, context, temperature=0.2):
    """
    Variante streaming de generate_answer : yield les morceaux de reponse au fur et a mesure.

    Utilise dans Streamlit avec st.write_stream() pour afficher la reponse
    token par token (effet ChatGPT).

    Args:
        question: str, la question utilisateur
        context: str, les chunks formates en contexte
        temperature: float, idem generate_answer

    Yields:
        str: morceau de reponse (typiquement un token ou quelques caracteres)
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    # Meme prompt que la version non-streaming
    full_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        context=context,
        question=question,
    )

    messages = [
        {"role": "user", "content": full_prompt}
    ]

    # Delegation au stream avec retry
    yield from _call_llm_stream_with_retry(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
    )

def generate_answer_with_verification(question, context, temperature=0.2):
    """
    Variante de generate_answer avec verification anti-hallucination par 2eme LLM.

    Pipeline :
    1. Genere la reponse via le LLM principal (mode normal, pas stream).
    2. Verifie chaque affirmation factuelle via verify_answer (2eme LLM en mode juge).
    3. Si INVALIDE -> remplace par "L'information n'est pas disponible..."
    4. Sinon -> retourne la reponse originale.

    Args:
        question: str, la question utilisateur
        context: str, les chunks formates
        temperature: float, idem generate_answer

    Returns:
        dict :
            - 'answer' : str, la reponse finale (originale ou remplacee)
            - 'verdict' : str, 'VALIDE', 'INVALIDE', 'PARTIEL', 'INCONNU'
            - 'is_valid' : bool
            - 'explanation' : str, raison du verdict
            - 'original_answer' : str, la reponse avant verification (utile pour debug)
    """
    # Etape 1 : generation classique
    original_answer = generate_answer(question, context, temperature=temperature)

    # Etape 2 : verification
    verification = verify_answer(question, original_answer, context)

    # Etape 3 : decision
    if verification["is_valid"]:
        final_answer = original_answer
    else:
        final_answer = (
            "L'information n'est pas disponible dans les documents fournis. "
            "(Note : une premiere reponse a ete generee mais elle contenait des "
            f"elements non supportes par le contexte : {verification['explanation']})"
        )

    return {
        "answer": final_answer,
        "verdict": verification["verdict"],
        "is_valid": verification["is_valid"],
        "explanation": verification["explanation"],
        "original_answer": original_answer,
    }

# Charge aussi le prompt de resume
SUMMARY_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "summary_prompt.txt"
SUMMARY_PROMPT_TEMPLATE = SUMMARY_PROMPT_PATH.read_text(encoding="utf-8")


def generate_summary(chunks, temperature=0.3):
    """
    Genere un resume structure du document a partir d'un echantillon de chunks.

    Pour un document complet, on prend un echantillon representatif (debut, milieu, fin)
    pour rester dans la limite de contexte du LLM.

    Args:
        chunks: list[Document], tous les chunks du document
        temperature: float, un peu plus haut que la generation (0.3 vs 0.2)
                     pour permettre une formulation plus naturelle

    Returns:
        str: le resume genere
    """
    # On prend jusqu'a 10 chunks representatifs (debut, milieu, fin du doc)
    # pour eviter d'envoyer 50 chunks au LLM (cher + risque de dilution)
    n = len(chunks)
    if n <= 10:
        sample = chunks
    else:
        # On prend les 4 premiers, 2 du milieu, 4 derniers
        sample = chunks[:4] + chunks[n // 2 - 1 : n // 2 + 1] + chunks[-4:]

    # Format chunks comme contexte
    from src.retrieval import format_chunks_for_prompt
    context = format_chunks_for_prompt(sample)

    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    full_prompt = SUMMARY_PROMPT_TEMPLATE.format(context=context)
    messages = [{"role": "user", "content": full_prompt}]

    response = _call_llm_with_retry(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=1500,
    )

    return response.choices[0].message.content

# ============== V2.5 : ANALYSES METIER PREDEFINIES ==============

# Charge les 5 prompts d'analyse au demarrage
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

ANALYSIS_PROMPTS = {
    "chiffres_cles": (PROMPTS_DIR / "analyse_chiffres_cles.txt").read_text(encoding="utf-8"),
    "risques": (PROMPTS_DIR / "analyse_risques.txt").read_text(encoding="utf-8"),
    "strategie": (PROMPTS_DIR / "analyse_strategie.txt").read_text(encoding="utf-8"),
    "evolution": (PROMPTS_DIR / "analyse_evolution.txt").read_text(encoding="utf-8"),
    "note_synthese": (PROMPTS_DIR / "analyse_note_synthese.txt").read_text(encoding="utf-8"),
}


def run_analysis(analysis_type, chunks, temperature=0.3):
    """
    Execute une analyse metier predefinie sur un ensemble de chunks.

    Args:
        analysis_type: str parmi 'chiffres_cles', 'risques', 'strategie',
                       'evolution', 'note_synthese'
        chunks: list[Document], les chunks a analyser (sample du document)
        temperature: float, 0.3 par defaut pour les analyses structurees

    Returns:
        str: l'analyse generee, format Markdown
    """
    if analysis_type not in ANALYSIS_PROMPTS:
        raise ValueError(
            f"Type d'analyse inconnu : '{analysis_type}'. "
            f"Choix possibles : {list(ANALYSIS_PROMPTS.keys())}"
        )

    # Pour les analyses, on prend un echantillon large mais pas tout
    # (cout + risque de dilution)
    n = len(chunks)
    if n <= 15:
        sample = chunks
    else:
        # Echantillonnage : debut, milieu, fin pour couvrir tout le doc
        sample = chunks[:6] + chunks[n // 2 - 2 : n // 2 + 2] + chunks[-6:]

    from src.retrieval import format_chunks_for_prompt
    context = format_chunks_for_prompt(sample)

    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    full_prompt = ANALYSIS_PROMPTS[analysis_type].format(context=context)
    messages = [{"role": "user", "content": full_prompt}]

    response = _call_llm_with_retry(
        client=client,
        model=model_name,
        messages=messages,
        temperature=temperature,
        max_tokens=2000,  # plus genereux pour les analyses structurees
    )

    return response.choices[0].message.content

# Bloc de test : pipeline RAG complet en CLI
if __name__ == "__main__":
    import sys
    from src.vectorstore import load_vectorstore
    from src.retrieval import retrieve_chunks, format_chunks_for_prompt

    # Question : depuis la ligne de commande ou question par defaut
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "Quel est le chiffre d'affaires du groupe en 2023 ?"

    print(f"Question : {question}\n")

    # Etape 1 : charger l'index
    print("[1/3] Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    # Etape 2 : retrieval
    print("[2/3] Recherche des chunks pertinents (top-4)...")
    chunks = retrieve_chunks(vs, question, k=4)
    pages = sorted(set(c.metadata["page"] for c in chunks))
    print(f"      -> Chunks recuperes depuis les pages : {pages}")

    # Etape 3 : generation
    print("[3/3] Generation de la reponse par Mistral...\n")
    context = format_chunks_for_prompt(chunks)
    answer = generate_answer(question, context)

    print("=" * 70)
    print("REPONSE :")
    print("=" * 70)
    print(answer)
    print("=" * 70)
    