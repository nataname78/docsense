"""
Configuration pytest globale : fixtures partagées entre les tests.

Une "fixture" pytest est une fonction qui prépare un état réutilisable
(par exemple : un Document factice, un chunk type, un index FAISS pré-construit).
Les tests reçoivent les fixtures comme arguments.
"""
import pytest
from langchain_core.documents import Document


@pytest.fixture
def sample_chunk_text():
    """Un chunk de texte type (sans tableau)."""
    return (
        "Le chiffre d'affaires de LVMH en 2023 atteint 86,2 milliards d'euros, "
        "en croissance organique de 13% par rapport à 2022."
    )


@pytest.fixture
def sample_chunk_with_table():
    """Un chunk avec un tableau Markdown intégré (comme produit par enrich_pages_with_tables)."""
    return """Chiffres clés du Groupe.

[TABLEAU 1 (page 3)]
| En millions d'euros | 2022 | 2023 | Variation |
|---|---|---|---|
| Ventes | 79 184 | 86 153 | +9% |
| Résultat net (part du Groupe) | 14 084 | 15 174 | +8% |
| Dette financière nette | 9 201 | 10 746 | +17% |
"""


@pytest.fixture
def sample_document():
    """Un Document langchain typique."""
    return Document(
        page_content="Le résultat net 2023 est de 15 174 millions d'euros.",
        metadata={"source": "lvmh_2023.pdf", "page": 3},
    )


@pytest.fixture
def sample_documents_list():
    """Une liste de Documents pour tester le retrieval, le formatage, etc."""
    return [
        Document(
            page_content="Ventes 2023 : 86 153 millions d'euros.",
            metadata={"source": "lvmh_2023.pdf", "page": 3},
        ),
        Document(
            page_content="Résultat net 2023 : 15 174 millions d'euros.",
            metadata={"source": "lvmh_2023.pdf", "page": 3},
        ),
        Document(
            page_content="Tiffany & Co. a connu une croissance significative en 2023.",
            metadata={"source": "lvmh_2023.pdf", "page": 4},
        ),
    ]


@pytest.fixture
def sample_structured_table():
    """Un tableau structuré (format extract_tables_structured)."""
    return {
        "document": "lvmh_2023.pdf",
        "page": 3,
        "table_index": 1,
        "headers": ["", "2022", "2023", "Variation"],
        "rows": [
            ["Ventes", "79 184", "86 153", "+9%"],
            ["Résultat net (part du Groupe)", "14 084", "15 174", "+8%"],
            ["Dette financière nette", "9 201", "10 746", "+17%"],
        ],
        "title_hint": "Chiffres cles En millions d'euros",
    }