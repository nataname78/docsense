"""
Module de Query Expansion : augmente la question utilisateur avec des synonymes metier
AVANT de l'envoyer au retrieval et au re-ranking.

Probleme resolu :
Le re-ranker bge-reranker-base ne reconnait pas certains synonymes financiers
(ex : "chiffre d'affaires" vs "Ventes"). Resultat : il retrograde les bons chunks
quand la question utilise un terme different de celui du document.

Solution :
On detecte les termes financiers dans la question et on ajoute leurs synonymes
entre parentheses dans la requete. Le re-ranker voit alors le mot "ventes" dans
la question et classe correctement les chunks qui le contiennent.

Exemple :
  Input  : "Quel est le chiffre d'affaires en 2023 ?"
  Output : "Quel est le chiffre d'affaires (ventes, revenus, produits des activites ordinaires) en 2023 ?"
"""
import re


# ============== TABLE DE SYNONYMES METIER ==============
# Chaque entree : un terme detecte -> liste de synonymes a injecter
# Tous en minuscules ; le matching est case-insensitive.

FINANCIAL_SYNONYMS = {
    # Performance financiere
    "chiffre d'affaires": ["ventes", "revenus", "produits des activites ordinaires"],
    "ca": ["ventes", "chiffre d'affaires", "revenus"],
    "ventes": ["chiffre d'affaires", "revenus"],
    "revenus": ["chiffre d'affaires", "ventes"],

    "benefice": ["resultat net", "resultat net part du groupe", "profit"],
    "benefices": ["resultat net", "resultat net part du groupe", "profits"],
    "profit": ["benefice", "resultat net"],
    "resultat net": ["benefice", "profit"],

    "ebit": ["resultat d'exploitation", "resultat operationnel courant"],
    "ebitda": ["excedent brut d'exploitation"],
    "marge brute": ["marge commerciale"],

    # Structure financiere
    "dette": ["dette financiere", "dette financiere nette", "endettement"],
    "endettement": ["dette", "dette financiere"],
    "tresorerie": ["liquidites", "cash", "disponibilites"],

    # Investissements
    "investissement": ["capex", "investissements d'exploitation"],
    "investissements": ["capex", "investissements d'exploitation"],
    "capex": ["investissements", "investissements d'exploitation"],

    # Ressources humaines
    "salaries": ["effectifs", "collaborateurs", "personnel"],
    "effectifs": ["salaries", "collaborateurs", "personnel"],
    "collaborateurs": ["salaries", "effectifs", "personnel"],
    "personnel": ["salaries", "effectifs", "collaborateurs"],

    # Autres concepts financiers
    "actionnaires": ["proprietaires", "investisseurs"],
    "dividende": ["dividendes"],
    "dividendes": ["dividende"],
}


def expand_query(query, max_synonyms_per_term=3, verbose=False):
    """
    Augmente une question avec les synonymes des termes financiers detectes.

    Args:
        query: str, la question originale de l'utilisateur
        max_synonyms_per_term: int, max de synonymes ajoutes par terme detecte
        verbose: bool, affiche les termes detectes (debug)

    Returns:
        str: la question avec synonymes injectes entre parentheses
             Si aucun terme metier detecte, retourne la question originale inchangee.
    """
    if not query or not query.strip():
        return query

    # On travaille sur une version en minuscules pour la detection
    # mais on garde la query originale pour l'output
    query_lower = query.lower()

    # On detecte les termes presents, en evitant les doublons
    # (ex: si "chiffre d'affaires" est detecte, on ne re-detecte pas "ventes" comme synonyme a re-expander)
    detected_terms = []
    detected_synonyms = set()

    # Tri par longueur decroissante : on detecte d'abord les expressions longues
    # ("chiffre d'affaires" avant "ventes")
    sorted_terms = sorted(FINANCIAL_SYNONYMS.keys(), key=len, reverse=True)

    for term in sorted_terms:
        # On utilise des word boundaries pour eviter les faux positifs
        # (ex: "ca" matchant "cabinet")
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, query_lower):
            # Eviter de detecter un synonyme deja injecte d'un autre terme
            if term in detected_synonyms:
                continue
            synonyms = FINANCIAL_SYNONYMS[term][:max_synonyms_per_term]
            detected_terms.append((term, synonyms))
            detected_synonyms.update(synonyms)

    if not detected_terms:
        if verbose:
            print(f"  [QueryExpansion] Aucun terme metier detecte, query inchangee.")
        return query

    # Construction de la query augmentee
    expanded = query
    for term, synonyms in detected_terms:
        # Reconstruit le pattern en preservant la casse originale
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        # Recupere la version exacte (avec casse) qui matche dans la query
        match = pattern.search(expanded)
        if match:
            original_match = match.group(0)
            # Injecte les synonymes apres le terme detecte
            synonyms_str = ", ".join(synonyms)
            replacement = f"{original_match} ({synonyms_str})"
            # remplace une seule fois (count=1) pour eviter les remplacements multiples
            expanded = pattern.sub(replacement, expanded, count=1)

    if verbose:
        terms_str = ", ".join(t for t, _ in detected_terms)
        print(f"  [QueryExpansion] Termes detectes : {terms_str}")
        print(f"  [QueryExpansion] Original  : {query}")
        print(f"  [QueryExpansion] Expanded  : {expanded}")

    return expanded


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    print("Test de la Query Expansion\n" + "=" * 60)

    test_queries = [
        "Quel est le chiffre d'affaires en 2023 ?",
        "Quel est le benefice de LVMH en 2023 ?",
        "Combien d'effectifs LVMH a-t-il en 2023 ?",
        "Quel est l'EBITDA du groupe ?",
        "Quelle est la dette nette en 2023 ?",
        "Compare les ventes de 2022 et 2023.",
        "Quelle est la capitale de la France ?",  # aucun terme metier
        "Parle-moi de Tiffany & Co.",  # aucun terme metier
    ]

    for q in test_queries:
        expanded = expand_query(q, verbose=True)
        print(f"\nIN  : {q}")
        print(f"OUT : {expanded}")
        print("-" * 60)