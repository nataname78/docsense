"""
Module de requete sur tableaux structures.

Fournit la fonction extract_table_value() qui matche une metrique et une annee
dans une liste de tableaux extraits (issus de table_extractor.extract_tables_structured).

Cette fonction est l'outil cle de la V3.3 : elle permet a l'agent ReAct de piocher
DIRECTEMENT dans les tableaux exacts (ex: "Resultat net (part du Groupe)" 2023 = 15 174)
sans passer par le retrieval/re-ranking qui peut introduire des arrondis communiques.

Matching intelligent :
- Insensible a la casse et aux accents
- Synonymes financiers (CA = Ventes = Revenus, benefice = Resultat net, etc.)
- Tolerance aux variations de formulation (resultat net matche "Resultat net (part du Groupe)")
"""
import unicodedata
import re
from src.query_expansion import FINANCIAL_SYNONYMS


# ============== HELPERS DE NORMALISATION ==============

def _normalize_text(text):
    """
    Normalise un texte pour le matching :
    - Minuscules
    - Sans accents (resume -> resume, eleve -> eleve, etc.)
    - Espaces multiples reduits a 1
    - Ponctuation interne preservee pour distinction (mais simplifiee)

    Exemple : "Résultat net (part du Groupe)" -> "resultat net (part du groupe)"
    """
    if not text:
        return ""
    # Supprime les accents : "é" -> "e"
    nfkd = unicodedata.normalize("NFKD", str(text))
    no_accents = "".join(c for c in nfkd if not unicodedata.combining(c))
    # Minuscules + strip
    normalized = no_accents.lower().strip()
    # Compresse les espaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _expand_metric_synonyms(metric):
    """
    Pour une metrique donnee, retourne une LISTE de termes a essayer matcher,
    incluant les synonymes financiers.

    Exemple :
        "CA" -> ["ca", "ventes", "chiffre d'affaires", "revenus"]
        "resultat net" -> ["resultat net", "benefice", "profit"]
    """
    metric_norm = _normalize_text(metric)
    candidates = [metric_norm]  # toujours essayer le terme original d'abord

    # Cherche les synonymes (la table FINANCIAL_SYNONYMS contient les synonymes en minuscules sans accents normalises)
    for key, synonyms in FINANCIAL_SYNONYMS.items():
        key_norm = _normalize_text(key)
        synonyms_norm = [_normalize_text(s) for s in synonyms]

        # Si la metrique correspond a une cle, ajoute ses synonymes
        if key_norm == metric_norm:
            candidates.extend(synonyms_norm)

        # Si la metrique est UN synonyme, ajoute la cle + les autres synonymes
        elif metric_norm in synonyms_norm:
            candidates.append(key_norm)
            candidates.extend([s for s in synonyms_norm if s != metric_norm])

    # Dedoublonne en preservant l'ordre
    seen = set()
    unique = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            unique.append(c)

    return unique


# ============== MATCHING LIGNES/COLONNES ==============

def _row_matches_metric(row_first_cell, metric_candidates):
    """
    Verifie si la 1ere cellule d'une ligne (le label de metrique) matche
    l'un des candidats fournis.

    Retourne un score (0-2) :
        2 = match exact apres normalisation
        1 = la metrique est INCLUSE dans le label (ex: "resultat net" dans "resultat net (part du groupe)")
        0 = aucun match
    """
    label_norm = _normalize_text(row_first_cell)
    if not label_norm:
        return 0

    best_score = 0
    for candidate in metric_candidates:
        if not candidate:
            continue

        # Match exact
        if label_norm == candidate:
            return 2

        # Match par inclusion : le candidat est une sous-chaine du label
        # ex: "resultat net" dans "resultat net (part du groupe)"
        if candidate in label_norm:
            best_score = max(best_score, 1)
        # Inverse aussi : "resultat net (part du groupe)" inclus dans "resultat net" est moins courant
        # mais on le tolere
        elif label_norm in candidate and len(label_norm) >= 5:
            best_score = max(best_score, 1)

    return best_score


def _find_matching_row(metric, table):
    """
    Trouve la ligne du tableau dont la 1ere colonne matche le mieux la metrique.

    Args:
        metric: str, le terme cherche (ex: "resultat net")
        table: dict avec 'rows' (list[list[str]])

    Returns:
        (row, score) ou (None, 0) si pas de match exploitable
    """
    metric_candidates = _expand_metric_synonyms(metric)

    best_row = None
    best_score = 0

    for row in table["rows"]:
        if not row:
            continue
        score = _row_matches_metric(row[0], metric_candidates)
        if score > best_score:
            best_score = score
            best_row = row

    return best_row, best_score


def _find_matching_column(year, headers):
    """
    Trouve l'index de la colonne correspondant a l'annee demandee.

    Args:
        year: str ou int, l'annee cherchee (ex: 2023 ou "2023")
        headers: list[str], la 1ere ligne du tableau

    Returns:
        int ou None : index de la colonne, ou None si non trouvee
    """
    year_str = str(year).strip()

    for i, header in enumerate(headers):
        header_norm = _normalize_text(header)
        # Match si l'annee apparait dans le header (couvre "2023" et "Variation 2023/2022")
        # Mais on prefere un match exact si possible
        if header_norm == year_str:
            return i

    # 2eme passe : match par sous-chaine (uniquement comme fallback)
    for i, header in enumerate(headers):
        if year_str in str(header):
            # On exclut les headers qui contiennent l'annee dans un contexte de "variation"
            # car ce n'est pas la valeur mais le pourcentage de variation
            header_lower = str(header).lower()
            if "variation" in header_lower or "evolution" in header_lower:
                continue
            return i

    return None


# ============== FONCTION PRINCIPALE ==============

def extract_table_value(metric, year, tables, document=None, verbose=False):
    """
    Extrait la valeur exacte d'une metrique pour une annee donnee, en cherchant
    dans une liste de tableaux structures.

    Args:
        metric: str, la metrique cherchee (ex: "resultat net", "CA", "dette financiere nette")
        year: str ou int, l'annee cherchee (ex: 2023)
        tables: list[dict], les tableaux structures (issus de extract_tables_structured)
        document: str ou None, optionnel : filtre par nom de fichier (ex: "lvmh_2023.pdf")
        verbose: bool, affiche les details du matching

    Returns:
        dict :
            {
                "found": bool,
                "value": str | None,      # la valeur trouvee (ex: "15 174")
                "metric_matched": str,    # le label exact du tableau (ex: "Resultat net (part du Groupe)")
                "year_matched": str,      # le header exact (ex: "2023")
                "document": str,          # le doc source
                "page": int,              # la page source
                "table_index": int,       # index du tableau dans la page
                "match_score": int,       # 2 = exact, 1 = inclusion
                "error": str | None,      # message d'erreur si found=False
            }
    """
    # Filtre par document si demande
    if document:
        candidate_tables = [t for t in tables if t["document"] == document]
        if not candidate_tables:
            return {
                "found": False,
                "value": None,
                "error": f"Aucun tableau trouve pour le document '{document}'.",
            }
    else:
        candidate_tables = tables

    if not candidate_tables:
        return {
            "found": False,
            "value": None,
            "error": "Aucun tableau dans le corpus.",
        }

    # On cherche le meilleur match parmi tous les tableaux
    best_match = None
    best_score = 0

    for table in candidate_tables:
        # Etape 1 : trouve la ligne metrique
        row, row_score = _find_matching_row(metric, table)
        if row_score == 0:
            continue

        # Etape 2 : trouve la colonne annee
        col_idx = _find_matching_column(year, table["headers"])
        if col_idx is None:
            continue

        # Etape 3 : extrait la valeur (verifie que la ligne a assez de colonnes)
        if col_idx >= len(row):
            continue

        value = row[col_idx].strip()
        if not value:
            continue

        # Verifie si ce match est meilleur que le precedent
        if row_score > best_score:
            best_score = row_score
            best_match = {
                "found": True,
                "value": value,
                "metric_matched": row[0],
                "year_matched": table["headers"][col_idx],
                "document": table["document"],
                "page": table["page"],
                "table_index": table["table_index"],
                "match_score": row_score,
                "error": None,
            }

        if verbose:
            print(f"  [Match candidat] {table['document']} p{table['page']} "
                  f"| ligne '{row[0]}' (score {row_score}) | "
                  f"colonne '{table['headers'][col_idx]}' | valeur '{value}'")

    if best_match:
        return best_match

    return {
        "found": False,
        "value": None,
        "error": (
            f"Aucun match trouve pour metric='{metric}', year={year}"
            + (f", document='{document}'" if document else "")
            + ". Verifie l'orthographe ou essaye un synonyme."
        ),
    }


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    from src.table_extractor import extract_tables_structured
    from pathlib import Path

    print("Test de extract_table_value\n" + "=" * 70)

    # Charge les tableaux des 2 PDFs LVMH (si presents)
    all_tables = []
    for pdf_path in sorted(Path("data/raw").glob("*.pdf")):
        print(f"  [Setup] Extraction des tableaux de {pdf_path.name}...")
        tables = extract_tables_structured(str(pdf_path))
        print(f"  [Setup]   -> {len(tables)} tableaux")
        all_tables.extend(tables)

    print(f"\n[Setup] Total : {len(all_tables)} tableaux dans le corpus.\n")

    # Cas de test : (metric, year, attendu_approximatif)
    test_cases = [
        ("resultat net", 2023, "15 174"),
        ("Resultat net (part du Groupe)", 2023, "15 174"),
        ("ventes", 2023, "86 153"),
        ("chiffre d'affaires", 2023, "86 153"),
        ("CA", 2023, "86 153"),
        ("benefice", 2023, "15 174"),
        ("dette financiere nette", 2023, "10 746"),
        ("dette", 2023, "10 746"),
        ("ventes", 2022, "79 184"),
        ("capitaux propres", 2023, "62 701"),
        # Cas qui doivent echouer proprement :
        ("metrique_inexistante", 2023, None),
        ("ventes", 1999, None),
    ]

    n_pass = 0
    n_fail = 0
    for metric, year, expected in test_cases:
        result = extract_table_value(metric, year, all_tables)

        if expected is None:
            # On s'attend a un echec
            if not result["found"]:
                print(f"  [OK]   '{metric}' annee {year} -> echec attendu : {result.get('error', '')[:60]}")
                n_pass += 1
            else:
                print(f"  [FAIL] '{metric}' annee {year} -> match inattendu : {result['value']}")
                n_fail += 1
        else:
            if result["found"] and expected in result["value"]:
                print(f"  [OK]   '{metric}' annee {year} -> '{result['value']}' "
                      f"(match: '{result['metric_matched']}' p{result['page']})")
                n_pass += 1
            else:
                got = result.get("value", "(non trouve)")
                print(f"  [FAIL] '{metric}' annee {year} -> attendu ~{expected}, obtenu {got}")
                if not result["found"]:
                    print(f"         Erreur : {result.get('error')}")
                n_fail += 1

    print(f"\n{'=' * 70}")
    print(f"Resultats : {n_pass} OK / {n_fail} FAIL sur {len(test_cases)} tests")
    print("=" * 70)