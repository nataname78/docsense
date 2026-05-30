"""
Module d'extraction de tableaux : pdfplumber + conversion Markdown.

Pourquoi un module separe :
- L'extraction PyMuPDF aplatit les tableaux financiers
  -> "(7 478) (4 969) (2 664)" sans contexte de colonnes
- pdfplumber detecte les tableaux par analyse geometrique
  -> on recupere la STRUCTURE ligne/colonne
- On convertit en Markdown : le LLM lit nativement les tableaux Markdown
  -> il comprend que "7 478" est sous la colonne "2023"

Strategie : pour chaque page, on extrait
- le texte normal (PyMuPDF, dans ingestion.py)
- les tableaux convertis en Markdown (ce module)
On concatene les deux dans le texte de la page.
"""
import pdfplumber
from pathlib import Path

from src.logger import get_logger
logger = get_logger(__name__)


def _clean_cell(cell):
    """Nettoie une cellule de tableau (None -> '', strip espaces, \n -> espace)."""
    if cell is None:
        return ""
    return str(cell).strip().replace("\n", " ")


def _table_to_markdown(table):
    """
    Convertit un tableau (liste de listes) en Markdown.

    Exemple d'entree :
        [["", "2023", "2022"],
         ["Ventes", "86 153", "79 184"],
         ["Resultat net", "15 174", "14 084"]]

    Sortie :
        | | 2023 | 2022 |
        |---|---|---|
        | Ventes | 86 153 | 79 184 |
        | Resultat net | 15 174 | 14 084 |
    """
    if not table or len(table) == 0:
        return ""

    # Nettoie toutes les cellules
    cleaned = [[_clean_cell(cell) for cell in row] for row in table]

    # Filtre les lignes completement vides
    cleaned = [row for row in cleaned if any(cell for cell in row)]

    if len(cleaned) == 0:
        return ""

    # Premiere ligne = entete (convention markdown)
    header = cleaned[0]
    n_cols = len(header)

    lines = []
    # Ligne d'entete
    lines.append("| " + " | ".join(header) + " |")
    # Ligne de separation Markdown
    lines.append("|" + "|".join(["---"] * n_cols) + "|")
    # Lignes de donnees
    for row in cleaned[1:]:
        # Ajuste si la ligne a un nombre de cellules different
        padded = row + [""] * (n_cols - len(row))
        padded = padded[:n_cols]
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def extract_tables_per_page(pdf_path):
    """
    Extrait les tableaux de chaque page d'un PDF, convertis en Markdown.

    Args:
        pdf_path: chemin vers le PDF

    Returns:
        dict[int, list[str]]: {page_number (1-indexed): [markdown_table1, ...]}
        Pages sans tableau : absentes du dict (ou liste vide).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    tables_by_page = {}

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            # extract_tables retourne une liste de tableaux (chacun = liste de listes)
            raw_tables = page.extract_tables()

            if not raw_tables:
                continue

            markdown_tables = []
            for raw_table in raw_tables:
                md = _table_to_markdown(raw_table)
                if md:  # on ignore les tableaux vides apres nettoyage
                    markdown_tables.append(md)

            if markdown_tables:
                tables_by_page[page_idx] = markdown_tables

    return tables_by_page


def extract_tables_structured(pdf_path):
    """
    Extrait les tableaux d'un PDF sous forme STRUCTUREE (pas Markdown).

    Cette fonction est destinee a l'outil 'extract_table_value' de l'agent ReAct.
    Contrairement a extract_tables_per_page() qui produit du Markdown texte
    (pour le retrieval), ici on retourne une structure exploitable pour matcher
    une metrique (ex: "resultat net") avec une annee (ex: 2023).

    Args:
        pdf_path: chemin vers le PDF

    Returns:
        list[dict]: liste de tableaux structures, chacun ayant :
            - 'document'    : str, nom du fichier PDF
            - 'page'        : int, numero de page (1-indexed)
            - 'table_index' : int, position du tableau dans la page (1, 2, ...)
            - 'headers'     : list[str], la 1ere ligne du tableau
            - 'rows'        : list[list[str]], les lignes de donnees (sans le header)
            - 'title_hint'  : str, indice de titre (texte juste avant le tableau)

        Format adapte pour une recherche par 'metric' (1ere colonne) et 'year' (header).
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF introuvable : {pdf_path}")

    pdf_name = pdf_path.name
    structured_tables = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_idx, page in enumerate(pdf.pages, start=1):
            raw_tables = page.extract_tables()

            if not raw_tables:
                continue

            # Recupere le texte de la page pour identifier les titres
            page_text = page.extract_text() or ""

            for table_idx, raw_table in enumerate(raw_tables, start=1):
                if not raw_table or len(raw_table) < 2:
                    continue  # On ignore les tableaux degeneres (1 ligne ou moins)

                # Nettoyage des cellules
                cleaned = [[_clean_cell(cell) for cell in row] for row in raw_table]
                # Filtre les lignes completement vides
                cleaned = [row for row in cleaned if any(cell for cell in row)]

                if len(cleaned) < 2:
                    continue  # Apres nettoyage, plus assez de lignes

                headers = cleaned[0]
                rows = cleaned[1:]

                # Heuristique simple pour deviner le titre du tableau :
                # on prend les 100 premiers caracteres du texte de la page,
                # comme indice de contexte (pas parfait mais utile en debug)
                title_hint = page_text[:200].strip().replace("\n", " ")[:150]

                structured_tables.append({
                    "document": pdf_name,
                    "page": page_idx,
                    "table_index": table_idx,
                    "headers": headers,
                    "rows": rows,
                    "title_hint": title_hint,
                })

    return structured_tables


def enrich_pages_with_tables(pages, pdf_path):
    """
    Enrichit le texte de chaque page avec les tableaux extraits en Markdown.

    Args:
        pages: liste de dicts [{'page_number': int, 'text': str}, ...]
               (format produit par src.ingestion.extract_text_from_pdf)
        pdf_path: chemin vers le PDF (necessaire pour relire les tableaux)

    Returns:
        liste de dicts au meme format, avec text enrichi.
        Les tableaux sont APPENDED a la fin du texte de la page, encadres par
        des marqueurs explicites pour que le LLM les reconnaisse.
    """
    tables_by_page = extract_tables_per_page(pdf_path)

    enriched_pages = []
    n_pages_with_tables = 0

    for page in pages:
        page_num = page["page_number"]
        text = page["text"]

        if page_num in tables_by_page:
            n_pages_with_tables += 1
            tables_md = tables_by_page[page_num]
            tables_block = "\n\n".join([
                f"[TABLEAU {i + 1} (page {page_num})]\n{md}"
                for i, md in enumerate(tables_md)
            ])
            # On concatene : texte original + section TABLEAUX
            enriched_text = f"{text}\n\n--- TABLEAUX DE LA PAGE ---\n\n{tables_block}"
        else:
            enriched_text = text

        enriched_pages.append({
            "page_number": page_num,
            "text": enriched_text,
        })

    logger.info(f"Tables : {n_pages_with_tables} page(s) avec tableaux extraits.")
    return enriched_pages


# Bloc de test : visualiser ce que pdfplumber extrait
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.table_extractor <chemin_du_pdf> [structured]")
        print("  - sans argument : affiche les tableaux en Markdown")
        print("  - avec 'structured' : affiche la structure pour l'agent")
        sys.exit(1)

    pdf_path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) > 2 else "markdown"

    # === Mode 'structured' : nouvelle sortie pour V3.3 ===
    if mode == "structured":
        print(f"Extraction structuree des tableaux de : {pdf_path}\n")
        tables = extract_tables_structured(pdf_path)

        if not tables:
            print("Aucun tableau detecte.")
            sys.exit(0)

        print(f"=> {len(tables)} tableau(x) extrait(s)\n")

        for t in tables:
            print("=" * 70)
            print(f"DOC: {t['document']} | PAGE: {t['page']} | TABLE #{t['table_index']}")
            print(f"Title hint : {t['title_hint'][:80]}...")
            print("=" * 70)
            print(f"Headers ({len(t['headers'])} colonnes) :")
            for i, h in enumerate(t['headers']):
                print(f"  [col {i}] {h!r}")
            print(f"\nRows ({len(t['rows'])} lignes) :")
            for i, row in enumerate(t['rows'][:8], start=1):  # max 8 lignes pour debug
                print(f"  [row {i}] {row}")
            if len(t['rows']) > 8:
                print(f"  ... ({len(t['rows']) - 8} lignes de plus tronquees)")
            print()
        sys.exit(0)

    # === Mode 'markdown' : comportement existant ===
    print(f"Extraction des tableaux de : {pdf_path}\n")

    tables_by_page = extract_tables_per_page(pdf_path)

    if not tables_by_page:
        print("Aucun tableau detecte dans le PDF.")
        sys.exit(0)

    print(f"=> {len(tables_by_page)} page(s) contiennent des tableaux\n")

    for page_num in sorted(tables_by_page.keys()):
        tables = tables_by_page[page_num]
        print("=" * 70)
        print(f"PAGE {page_num} - {len(tables)} tableau(x)")
        print("=" * 70)
        for i, md in enumerate(tables, start=1):
            print(f"\n--- Tableau {i} (page {page_num}) ---")
            print(md)
            print()