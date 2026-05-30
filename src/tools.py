"""
Module tools : outils que l'agent peut utiliser pour raisonner.

Chaque outil = une fonction Python + un descripteur que le LLM peut "lire"
pour decider quand l'appeler.

Outils disponibles :
- search : recherche dans le corpus indexe (notre RAG)
- calculate : evalue une expression mathematique
- compare_values : compare deux valeurs et calcule la variation
- list_documents : liste les documents indexes
- final_answer : termine la boucle d'agent

L'agent appelle ces outils via leur nom + parametres,
on les execute, et on lui retourne le resultat sous forme de string.
"""
import re
import ast
import operator
from src.retrieval import format_chunks_for_prompt
from src.reranker import rerank
from src.table_query import extract_table_value


# ============== OUTIL 1 : SEARCH (retrieval RAG) ==============

def tool_search(retriever, query, k=4, retrieve_n=20, source_filter=None):
    """
    Recherche dans le corpus indexe.

    Args:
        retriever: HybridRetriever
        query: str, la requete de recherche
        k: chunks finaux retournes
        retrieve_n: candidats avant reranking
        source_filter: optionnel, restreint a un document

    Returns:
        str: contexte formate avec sources, exploitable par le LLM
    """
    candidates = retriever.retrieve(query, k=retrieve_n, source_filter=source_filter)
    if len(candidates) > k:
        chunks = rerank(query, candidates, top_k=k)
    else:
        chunks = candidates[:k]

    if not chunks:
        return "Aucun resultat trouve."

    return format_chunks_for_prompt(chunks)


# ============== OUTIL 2 : CALCULATE (mathematiques) ==============

# Operations autorisees (on n'utilise PAS eval() pour des raisons de securite)
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    """Evaluation securisee d'un AST mathematique (sans eval())."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("Seuls les nombres sont autorises.")
    elif isinstance(node, ast.BinOp):
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Operateur non autorise : {type(node.op).__name__}")
        return op_func(left, right)
    elif isinstance(node, ast.UnaryOp):
        operand = _safe_eval(node.operand)
        op_func = _ALLOWED_OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Operateur unaire non autorise : {type(node.op).__name__}")
        return op_func(operand)
    else:
        raise ValueError(f"Type d'expression non autorise : {type(node).__name__}")


def tool_calculate(expression):
    """
    Evalue une expression mathematique de maniere securisee.

    Supporte : +, -, *, /, **, %, parentheses, nombres entiers et decimaux.
    Ne supporte PAS : fonctions, variables, eval() Python.

    Args:
        expression: str, ex "((15174-14084)/14084)*100"

    Returns:
        str: resultat sous forme de chaine, ou message d'erreur
    """
    try:
        # Nettoie l'expression (espaces, virgules francaises -> points)
        cleaned = expression.replace(" ", "").replace(",", ".")
        # Retire les separateurs de milliers eventuels (apostrophes typographiques)
        cleaned = cleaned.replace("'", "").replace("\u202f", "").replace("\xa0", "")

        # Parse en AST puis evalue
        tree = ast.parse(cleaned, mode="eval")
        result = _safe_eval(tree.body)

        # Formate joliment : entier si entier, sinon arrondi a 2 decimales
        if isinstance(result, float) and result.is_integer():
            return str(int(result))
        if isinstance(result, float):
            return f"{result:.4f}".rstrip("0").rstrip(".")
        return str(result)

    except Exception as e:
        return f"Erreur de calcul : {str(e)}. Verifie ton expression."


# ============== OUTIL 3 : COMPARE_VALUES ==============

def tool_compare_values(value_a, value_b, label_a="A", label_b="B"):
    """
    Compare deux valeurs numeriques et retourne la variation absolue et relative.

    Utile pour eviter au LLM de poser le calcul lui-meme.

    Args:
        value_a: float ou str (premiere valeur, generalement N-1)
        value_b: float ou str (deuxieme valeur, generalement N)
        label_a: str, libelle de la valeur A
        label_b: str, libelle de la valeur B

    Returns:
        str: rapport de comparaison structure
    """
    try:
        # Nettoyage des valeurs
        va = float(str(value_a).replace(" ", "").replace(",", ".").replace("'", ""))
        vb = float(str(value_b).replace(" ", "").replace(",", ".").replace("'", ""))

        delta = vb - va
        if va == 0:
            pct = "n/a (division par zero)"
        else:
            pct = f"{((vb - va) / va) * 100:.2f}%"

        sign = "+" if delta >= 0 else ""

        return (
            f"{label_a} : {va}\n"
            f"{label_b} : {vb}\n"
            f"Variation absolue : {sign}{delta}\n"
            f"Variation relative : {sign}{pct}"
        )
    except Exception as e:
        return f"Erreur de comparaison : {str(e)}"


# ============== OUTIL 5 : EXTRACT_TABLE_VALUE (V3.3) ==============

def tool_extract_table_value(metric, year, tables, document=None):
    """
    Extrait la valeur EXACTE d'une metrique pour une annee depuis les tableaux structures.

    A utiliser quand on veut un chiffre PRECIS d'un tableau, sans risque d'arrondi
    communique (ex: "86 153" et non "86,2 milliards").

    Args:
        metric: str, la metrique (ex: "resultat net", "CA", "dette")
        year: str ou int, l'annee (ex: 2023)
        tables: list[dict], les tableaux structures du corpus
        document: str ou None, optionnel : filtre par fichier

    Returns:
        str: une chaine lisible pour l'agent, avec la valeur et la source
    """
    result = extract_table_value(metric, year, tables, document=document)

    if result["found"]:
        return (
            f"{result['value']} "
            f"({result['metric_matched']}, "
            f"document: {result['document']}, "
            f"page {result['page']})"
        )
    return f"Aucun match. {result['error']}"

# ============== OUTIL 6 : LIST_DOCUMENTS ==============

def tool_list_documents(chunks):
    """
    Retourne la liste des documents presents dans le corpus.

    Utile quand l'agent veut savoir s'il a plusieurs sources disponibles.
    """
    sources = set()
    chunks_per_source = {}
    for chunk in chunks:
        src = chunk.metadata.get("source")
        if src:
            sources.add(src)
            chunks_per_source[src] = chunks_per_source.get(src, 0) + 1

    if not sources:
        return "Aucun document indexe."

    lines = ["Documents indexes :"]
    for src in sorted(sources):
        lines.append(f"- {src} ({chunks_per_source[src]} chunks)")
    return "\n".join(lines)


# ============== DESCRIPTEURS POUR LE PROMPT AGENT ==============

# Cette description est INSEREE dans le prompt system de l'agent
# pour que le LLM sache quels outils existent et comment les appeler.
TOOLS_DESCRIPTION = """
OUTILS DISPONIBLES :

1. search(query, source_filter=None)
   - Cherche dans le corpus indexe via retrieval semantique + lexical.
   - query : la requete de recherche en langage naturel.
   - source_filter : optionnel, nom de fichier pour restreindre la recherche.
   - Exemple : search("strategie Sephora") ou search("ventes", source_filter="lvmh_2022.pdf")
   - PREFERER search pour le texte narratif (strategie, performance, evolution qualitative).

2. calculate(expression)
   - Evalue une expression mathematique.
   - Exemple : calculate("((15174-14084)/14084)*100")
   - Supporte : +, -, *, /, **, parentheses. PAS de fonctions complexes.

3. compare_values(value_a, value_b, label_a, label_b)
   - Compare deux valeurs et retourne variation absolue + relative.
   - Exemple : compare_values(14084, 15174, "2022", "2023")

4. extract_table_value(metric, year, document=None)
   - Extrait une valeur EXACTE depuis les tableaux structures du corpus.
   - PREFERER cet outil pour les chiffres precis (CA, resultat net, dette, etc.)
     car il evite les arrondis communiques ("86,2 milliards" vs "86 153").
   - metric : nom de la metrique (ex: "resultat net", "CA", "dette financiere nette").
     Les synonymes sont geres automatiquement (CA = Ventes, benefice = Resultat net).
   - year : annee cherchee (ex: 2023 ou "2023").
   - document : optionnel, filtre par fichier (ex: "lvmh_2023.pdf").
   - Exemple : extract_table_value("resultat net", 2023) ou extract_table_value("CA", 2022, document="lvmh_2022.pdf")

5. list_documents()
   - Liste les documents indexes (utile pour les questions sur le corpus).

6. final_answer(answer)
   - TERMINE la boucle et retourne la reponse finale a l'utilisateur.
   - L'argument 'answer' doit etre la reponse complete, avec citations [page X].
"""


# ============== PARSER DES APPELS D'OUTILS ==============

def parse_tool_call(text):
    """
    Parse le texte d'une action LLM pour extraire nom_outil et arguments.

    Format attendu : 'tool_name(arg1, arg2, ...)' ou avec arguments nommes.

    Args:
        text: str, par exemple 'search("resultat net 2023")'

    Returns:
        tuple (tool_name, args_list, kwargs_dict) ou (None, None, None) si parse echoue
    """
    text = text.strip()

    # Regex pour extraire 'tool_name(...)'
    match = re.match(r"^(\w+)\s*\((.*)\)\s*$", text, re.DOTALL)
    if not match:
        return None, None, None

    tool_name = match.group(1)
    args_str = match.group(2).strip()

    if not args_str:
        return tool_name, [], {}

    # Parse les arguments via ast.parse en mode 'eval' sur un appel factice
    try:
        # On encapsule pour que ast comprenne
        fake_call = f"_({args_str})"
        tree = ast.parse(fake_call, mode="eval")
        call_node = tree.body
        if not isinstance(call_node, ast.Call):
            return tool_name, [], {}

        # Args positionnels : on extrait litteraux
        args = []
        for arg in call_node.args:
            args.append(_extract_literal(arg))

        # Args nommes
        kwargs = {}
        for kw in call_node.keywords:
            kwargs[kw.arg] = _extract_literal(kw.value)

        return tool_name, args, kwargs

    except Exception:
        # Si echec du parse, on retourne brut
        return tool_name, [args_str], {}


def _extract_literal(node):
    """Extrait une valeur litterale d'un noeud AST (str, int, float, None, True, False)."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return -_extract_literal(node.operand)
    # Fallback : on prend la representation textuelle
    return ast.unparse(node)


# ============== TEST ==============

if __name__ == "__main__":
    print("Test des outils\n" + "=" * 60)

    # Test calculate
    print("\n--- calculate ---")
    print(f"calculate('((15174-14084)/14084)*100') = {tool_calculate('((15174-14084)/14084)*100')}")
    print(f"calculate('86 153 - 79 184') = {tool_calculate('86 153 - 79 184')}")
    print(f"calculate('2**10') = {tool_calculate('2**10')}")
    print(f"calculate('1+abc') = {tool_calculate('1+abc')}")

    # Test compare_values
    print("\n--- compare_values ---")
    print(tool_compare_values(14084, 15174, "2022", "2023"))

    # Test parse_tool_call
    print("\n--- parse_tool_call ---")
    tests = [
        'search("resultat net 2023")',
        'search("ventes", source_filter="lvmh_2022.pdf")',
        'calculate("((15174-14084)/14084)*100")',
        'compare_values(14084, 15174, "2022", "2023")',
        'final_answer("Le resultat net est passe de 14084 a 15174.")',
        'list_documents()',
    ]
    for t in tests:
        name, args, kwargs = parse_tool_call(t)
        print(f"  '{t}'")
        print(f"    -> name={name}, args={args}, kwargs={kwargs}")