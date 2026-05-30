"""
Module agent : moteur ReAct (Reasoning + Acting).

Implemente la boucle d'orchestration :
- LLM produit un THOUGHT + une ACTION
- On parse, on execute l'outil
- On lui renvoie l'OBSERVATION
- Boucle jusqu'a final_answer ou max_iterations

Format strict attendu du LLM :
    THOUGHT: ...
    ACTION: tool_name(args)
"""
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError

from src.tools import (
    tool_search,
    tool_calculate,
    tool_compare_values,
    tool_list_documents,
    tool_extract_table_value,
    parse_tool_call,
    TOOLS_DESCRIPTION,
)

from src.logger import get_logger
logger = get_logger(__name__)

load_dotenv()


# Charge le prompt agent
AGENT_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "agent_system.txt"
AGENT_PROMPT_TEMPLATE = AGENT_PROMPT_PATH.read_text(encoding="utf-8")


# ============== APPEL LLM AVEC RETRY ==============

def _call_llm_with_retry(client, model, messages, max_retries=6, temperature=0.1, max_tokens=600):
    """
    Appelle Mistral chat avec retry exponentiel sur 429.
    Temperature basse (0.1) pour stabiliser le format ReAct.
    max_tokens=600 : on n'a pas besoin de longues sorties par etape.
    """
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
                logger.warning(f"Agent : rate limit, attente {wait}s, retry {attempt + 2}/{max_retries}")
                time.sleep(wait)
                continue
            raise


# ============== PARSER DE LA SORTIE LLM ==============

def parse_llm_output(text):
    """
    Extrait THOUGHT et ACTION d'une sortie LLM.

    Format attendu :
        THOUGHT: ...
        ACTION: tool_name(args)

    On est tolerant : on accepte les variations de casse, les espaces, etc.

    Returns:
        dict avec 'thought' (str) et 'action' (str) ou None si parse echoue
    """
    if not text:
        return None

    # On cherche THOUGHT et ACTION via regex tolerante
    thought_match = re.search(
        r"THOUGHT\s*:\s*(.+?)(?=\n\s*ACTION\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )
    action_match = re.search(
        r"ACTION\s*:\s*(.+?)(?=\n\s*(?:THOUGHT|OBSERVATION)\s*:|\Z)",
        text,
        re.IGNORECASE | re.DOTALL,
    )

    if not action_match:
        return None

    thought = thought_match.group(1).strip() if thought_match else ""
    action = action_match.group(1).strip()

    # On nettoie : enleve markdown code fences eventuels
    action = re.sub(r"^```\w*\s*", "", action)
    action = re.sub(r"\s*```$", "", action)
    action = action.strip()

    return {"thought": thought, "action": action}


# ============== EXECUTION D'UN OUTIL ==============

def execute_tool(tool_call_str, retriever, all_chunks, tables=None, source_filter=None):
    """
    Execute un appel d'outil, retourne le resultat sous forme de string (OBSERVATION).

    Args:
        tool_call_str: str, ex 'search("ventes 2023")'
        retriever: HybridRetriever
        all_chunks: list[Document], tous les chunks (pour list_documents)
        source_filter: optionnel, source filter actif

    Returns:
        tuple (observation_str, is_final, final_answer_str)
        - observation_str : ce qu'on renvoie a l'agent comme OBSERVATION
        - is_final : True si l'outil etait final_answer
        - final_answer_str : la reponse finale si is_final, sinon None
    """
    name, args, kwargs = parse_tool_call(tool_call_str)

    if name is None:
        return (
            "ERREUR : action mal formee. Format attendu : tool_name(args). "
            "Reessaie avec un format correct.",
            False,
            None,
        )

    # Dispatching selon le nom de l'outil
    if name == "search":
        query = args[0] if args else kwargs.get("query", "")
        sf = kwargs.get("source_filter", source_filter)
        if not query:
            return ("ERREUR : search() necessite une requete. Ex: search(\"ventes 2023\")", False, None)
        result = tool_search(retriever, query, source_filter=sf)
        return (result, False, None)

    elif name == "calculate":
        expr = args[0] if args else kwargs.get("expression", "")
        if not expr:
            return ("ERREUR : calculate() necessite une expression. Ex: calculate(\"(15174-14084)/14084*100\")", False, None)
        result = tool_calculate(expr)
        return (f"Resultat du calcul : {result}", False, None)

    elif name == "compare_values":
        if len(args) >= 2:
            va, vb = args[0], args[1]
            la = args[2] if len(args) > 2 else kwargs.get("label_a", "A")
            lb = args[3] if len(args) > 3 else kwargs.get("label_b", "B")
        else:
            va = kwargs.get("value_a")
            vb = kwargs.get("value_b")
            la = kwargs.get("label_a", "A")
            lb = kwargs.get("label_b", "B")
        if va is None or vb is None:
            return ("ERREUR : compare_values() necessite value_a et value_b. Ex: compare_values(14084, 15174, \"2022\", \"2023\")", False, None)
        result = tool_compare_values(va, vb, la, lb)
        return (result, False, None)

    elif name == "extract_table_value":
        if tables is None or len(tables) == 0:
            return (
                "ERREUR : aucun tableau structure disponible dans le corpus. "
                "Fallback : utilise search() a la place.",
                False,
                None,
            )
        metric = args[0] if args else kwargs.get("metric", "")
        year = args[1] if len(args) > 1 else kwargs.get("year", "")
        doc = args[2] if len(args) > 2 else kwargs.get("document", None)
        if not metric or not year:
            return (
                "ERREUR : extract_table_value() necessite metric ET year. "
                "Ex: extract_table_value(\"resultat net\", 2023)",
                False,
                None,
            )
        result = tool_extract_table_value(metric, year, tables, document=doc)
        return (result, False, None)
    
    elif name == "list_documents":
        result = tool_list_documents(all_chunks)
        return (result, False, None)

    elif name == "final_answer":
        answer = args[0] if args else kwargs.get("answer", "")
        if not answer:
            return ("ERREUR : final_answer() necessite une reponse. Ex: final_answer(\"Votre reponse...\")", False, None)
        return (None, True, str(answer))

    else:
        return (
            f"ERREUR : outil '{name}' inconnu. Outils disponibles : search, calculate, "
            f"compare_values, extract_table_value, list_documents, final_answer.",
            False,
            None,
        )


# ============== BOUCLE D'AGENT ==============

def run_agent(question, retriever, all_chunks, tables=None, source_filter=None,
              max_iterations=8, verbose=True):
    """
    Execute la boucle ReAct pour repondre a une question.

    Args:
        question: str
        retriever: HybridRetriever
        all_chunks: list[Document], tous les chunks indexes
        source_filter: str ou None
        max_iterations: int, garde-fou contre boucles infinies
        verbose: si True, print les etapes (utile pour debug)

    Returns:
        dict avec :
        - 'answer' : str, la reponse finale (ou message d'erreur)
        - 'trace' : list[dict], chaque etape avec thought/action/observation
        - 'iterations' : int, nombre d'etapes effectives
        - 'completed' : bool, True si final_answer atteinte
    """
    api_key = os.getenv("MISTRAL_API_KEY")
    model_name = os.getenv("LLM_MODEL", "mistral-small-latest")
    client = Mistral(api_key=api_key)

    trace = []
    history_str = ""

    for step in range(1, max_iterations + 1):
        # Construit le prompt complet
        full_prompt = AGENT_PROMPT_TEMPLATE.format(
            tools_description=TOOLS_DESCRIPTION,
            question=question,
            history=history_str,
            step=step,
        )

        # Appel LLM
        if verbose:
            print(f"\n[Agent] Etape {step} : appel LLM...")

        try:
            response = _call_llm_with_retry(
                client=client,
                model=model_name,
                messages=[{"role": "user", "content": full_prompt}],
            )
            llm_output = response.choices[0].message.content
        except Exception as e:
            return {
                "answer": f"Erreur LLM : {str(e)}",
                "trace": trace,
                "iterations": step - 1,
                "completed": False,
            }

        # Parse la sortie
        parsed = parse_llm_output(llm_output)
        if parsed is None:
            # Format invalide : on injecte une observation d'erreur et on continue
            error_obs = (
                "ERREUR : ta sortie n'avait pas le format attendu. "
                "Tu DOIS produire 'THOUGHT: ...' suivi de 'ACTION: tool_name(args)'. "
                "Reessaie."
            )
            trace.append({
                "step": step,
                "thought": "(format invalide)",
                "action": "(format invalide)",
                "observation": error_obs,
                "raw_output": llm_output,
            })
            history_str += f"\nEtape {step} :\n{llm_output}\nOBSERVATION: {error_obs}\n"
            if verbose:
                print(f"  [Agent] Sortie LLM non parsable, on injecte erreur.")
            continue

        thought = parsed["thought"]
        action = parsed["action"]

        if verbose:
            print(f"  THOUGHT : {thought[:100]}{'...' if len(thought) > 100 else ''}")
            print(f"  ACTION  : {action[:100]}{'...' if len(action) > 100 else ''}")

        # Execute l'outil
        observation, is_final, final_answer = execute_tool(
            action, retriever, all_chunks, tables=tables, source_filter=source_filter
        )

        if is_final:
            # On enregistre l'etape finale dans la trace
            trace.append({
                "step": step,
                "thought": thought,
                "action": action,
                "observation": "(final_answer)",
                "is_final": True,
            })
            if verbose:
                print(f"  [Agent] Final answer atteinte en {step} etape(s).")
            return {
                "answer": final_answer,
                "trace": trace,
                "iterations": step,
                "completed": True,
            }

        # Tronque l'observation si trop longue (evite explosion de contexte)
        obs_for_history = observation
        if len(obs_for_history) > 3000:
            obs_for_history = obs_for_history[:3000] + "\n... [observation tronquee]"

        trace.append({
            "step": step,
            "thought": thought,
            "action": action,
            "observation": observation,
            "is_final": False,
        })

        history_str += (
            f"\nEtape {step} :\n"
            f"THOUGHT: {thought}\n"
            f"ACTION: {action}\n"
            f"OBSERVATION: {obs_for_history}\n"
        )

        if verbose:
            obs_preview = observation[:150].replace("\n", " ")
            print(f"  OBSERVATION : {obs_preview}{'...' if len(observation) > 150 else ''}")

    # Si on sort de la boucle sans final_answer
    return {
        "answer": (
            "L'agent n'a pas pu converger en " + str(max_iterations) + " etapes. "
            "Reformulez votre question ou augmentez max_iterations."
        ),
        "trace": trace,
        "iterations": max_iterations,
        "completed": False,
    }


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    from src.vectorstore import load_vectorstore
    from src.ingestion import extract_text_from_pdf
    from src.chunking import chunk_pages
    from src.table_extractor import enrich_pages_with_tables
    from src.retrieval import HybridRetriever

    print("Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    from src.table_extractor import extract_tables_structured

    print("Reconstruction des chunks (avec tableaux)...")
    chunks = []
    tables = []
    for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
        result = extract_text_from_pdf(str(pdf_path))
        pages = enrich_pages_with_tables(result["pages"], str(pdf_path))
        chunks.extend(chunk_pages(pages=pages, file_name=pdf_path.name))
        tables.extend(extract_tables_structured(str(pdf_path)))
    print(f"  -> {len(chunks)} chunks, {len(tables)} tableaux structures.")

    retriever = HybridRetriever(vectorstore=vs, chunks=chunks)

    print("\n" + "=" * 70)
    print("TEST AGENT : question simple")
    print("=" * 70)
    question1 = "Quel est le chiffre d'affaires 2023 ?"
    result1 = run_agent(question1, retriever, chunks, tables=tables)
    print(f"\n>>> REPONSE FINALE : {result1['answer']}")
    print(f">>> Etapes utilisees : {result1['iterations']}")

    print("\n" + "=" * 70)
    print("TEST AGENT : question multi-step (la grosse demo)")
    print("=" * 70)
    question2 = "Quelle est l'evolution du resultat net entre 2022 et 2023 en pourcentage ?"
    result2 = run_agent(question2, retriever, chunks, tables=tables)
    print(f"\n>>> REPONSE FINALE : {result2['answer']}")
    print(f">>> Etapes utilisees : {result2['iterations']}")