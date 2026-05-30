"""
Suite de tests pytest pour les modules critiques du projet.

Couverture :
- query_expansion : detection synonymes financiers
- tools.parse_tool_call : parsing des actions agent
- tools.tool_calculate : calculs securises (anti-eval)
- tools.tool_compare_values : calcul variation absolue/relative
- table_query.extract_table_value : matching metric/year avec synonymes
- retrieval.format_chunks_for_prompt : formatage du contexte LLM

Lancer la suite :
    pytest
    pytest tests/test_modules.py -v
    pytest -k "test_calculate"          # filtre par nom

Marqueurs disponibles :
    pytest -m "not slow"                # exclut les tests lents
"""
import pytest


# ============== TESTS QUERY EXPANSION ==============

class TestQueryExpansion:
    """Tests du module src.query_expansion."""

    def test_expansion_chiffre_affaires(self):
        """Le terme 'chiffre d'affaires' doit etre augmente avec ses synonymes."""
        from src.query_expansion import expand_query
        result = expand_query("Quel est le chiffre d'affaires en 2023 ?")
        # On verifie qu'au moins un synonyme cle est present
        assert "ventes" in result.lower() or "revenus" in result.lower()
        # La question originale est preservee
        assert "chiffre d'affaires" in result.lower()

    def test_expansion_benefice(self):
        """'benefice' doit etre augmente avec 'resultat net'."""
        from src.query_expansion import expand_query
        result = expand_query("Quel est le benefice 2023 ?")
        assert "resultat net" in result.lower()

    def test_no_expansion_when_no_financial_term(self):
        """Une question sans terme financier doit etre inchangee."""
        from src.query_expansion import expand_query
        original = "Quelle est la capitale de la France ?"
        result = expand_query(original)
        assert result == original

    def test_expansion_preserves_question_structure(self):
        """L'expansion ne casse pas la structure de la question (mots-cles preserves)."""
        from src.query_expansion import expand_query
        result = expand_query("Compare les ventes de 2022 et 2023")
        # Les chiffres et le verbe doivent etre preserves
        assert "2022" in result
        assert "2023" in result
        assert "compare" in result.lower()


# ============== TESTS TOOLS (PARSING + CALCUL) ==============

class TestToolParsing:
    """Tests du parsing des appels d'outils de l'agent."""

    def test_parse_simple_call(self):
        """Parse 'search("query")' -> name='search', args=['query']."""
        from src.tools import parse_tool_call
        name, args, kwargs = parse_tool_call('search("resultat net 2023")')
        assert name == "search"
        assert args == ["resultat net 2023"]
        assert kwargs == {}

    def test_parse_call_with_kwargs(self):
        """Parse 'search(query="x", source_filter="y")' avec kwargs."""
        from src.tools import parse_tool_call
        name, args, kwargs = parse_tool_call(
            'search("ventes", source_filter="lvmh_2022.pdf")'
        )
        assert name == "search"
        assert args == ["ventes"]
        assert kwargs == {"source_filter": "lvmh_2022.pdf"}

    def test_parse_call_with_numbers(self):
        """Parse 'compare_values(14084, 15174, "2022", "2023")' avec mix int/str."""
        from src.tools import parse_tool_call
        name, args, kwargs = parse_tool_call(
            'compare_values(14084, 15174, "2022", "2023")'
        )
        assert name == "compare_values"
        assert args == [14084, 15174, "2022", "2023"]

    def test_parse_invalid_call(self):
        """Une chaine mal formee doit retourner (None, None, None)."""
        from src.tools import parse_tool_call
        name, args, kwargs = parse_tool_call("ceci n'est pas un appel")
        assert name is None
        assert args is None


class TestToolCalculate:
    """Tests de la fonction tool_calculate (eval securisee)."""

    def test_addition(self):
        from src.tools import tool_calculate
        assert tool_calculate("1+2") == "3"

    def test_complex_expression(self):
        """Calcul de variation type (15174-14084)/14084*100 ~= 7.74."""
        from src.tools import tool_calculate
        result = tool_calculate("((15174-14084)/14084)*100")
        # On accepte une tolerance sur l'arrondi
        assert "7.7" in result

    def test_supports_french_decimal(self):
        """Les virgules francaises sont converties en points."""
        from src.tools import tool_calculate
        assert tool_calculate("1,5+1,5") == "3"

    def test_rejects_eval_injection(self):
        """tool_calculate ne doit PAS executer du code Python arbitraire."""
        from src.tools import tool_calculate
        # Si tool_calculate utilisait eval() naivement, cette ligne casserait le test
        result = tool_calculate("__import__('os').system('ls')")
        assert "Erreur" in result  # doit refuser, pas executer

    def test_handles_division_by_zero(self):
        """Pas de crash sur division par zero, retourne une erreur."""
        from src.tools import tool_calculate
        result = tool_calculate("1/0")
        assert "Erreur" in result


class TestToolCompareValues:
    """Tests de tool_compare_values."""

    def test_positive_variation(self):
        """Variation positive : 14084 -> 15174 = +1090 (+7.74%)."""
        from src.tools import tool_compare_values
        result = tool_compare_values(14084, 15174, "2022", "2023")
        assert "+1090" in result.replace(".0", "")
        assert "+7.7" in result

    def test_negative_variation(self):
        """Variation negative."""
        from src.tools import tool_compare_values
        result = tool_compare_values(100, 80, "A", "B")
        assert "-20" in result

    def test_handles_string_values(self):
        """Les valeurs str sont converties (avec virgules / espaces)."""
        from src.tools import tool_compare_values
        # Format francais courant : "1 234,5"
        result = tool_compare_values("1 234", "2 468", "A", "B")
        assert "+1234" in result.replace(".0", "")


# ============== TESTS TABLE QUERY ==============

class TestTableQuery:
    """Tests de extract_table_value et helpers."""

    def test_match_exact_metric(self, sample_structured_table):
        """Match exact d'une metrique avec son label exact dans le tableau."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "Ventes",
            "2023",
            [sample_structured_table],
        )
        assert result["found"] is True
        assert result["value"] == "86 153"

    def test_match_synonym(self, sample_structured_table):
        """'CA' doit matcher 'Ventes' via synonymes financiers."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "CA",
            "2023",
            [sample_structured_table],
        )
        assert result["found"] is True
        assert result["value"] == "86 153"

    def test_match_partial_inclusion(self, sample_structured_table):
        """'resultat net' doit matcher 'Resultat net (part du Groupe)'."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "resultat net",
            "2023",
            [sample_structured_table],
        )
        assert result["found"] is True
        assert result["value"] == "15 174"

    def test_match_accents_insensitive(self, sample_structured_table):
        """'Resultat' sans accent doit matcher 'Résultat' avec accent."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "Resultat net",
            "2023",
            [sample_structured_table],
        )
        assert result["found"] is True

    def test_year_mismatch_returns_not_found(self, sample_structured_table):
        """Une annee absente du tableau doit retourner found=False."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "Ventes",
            "1999",
            [sample_structured_table],
        )
        assert result["found"] is False

    def test_metric_not_found_returns_clean_error(self, sample_structured_table):
        """Une metrique absente retourne un message d'erreur, pas un crash."""
        from src.table_query import extract_table_value
        result = extract_table_value(
            "metrique_inexistante",
            "2023",
            [sample_structured_table],
        )
        assert result["found"] is False
        assert "error" in result


# ============== TESTS RETRIEVAL.FORMAT ==============

class TestFormatChunks:
    """Tests du formatage des chunks pour le prompt LLM."""

    def test_format_includes_source_and_page(self, sample_documents_list):
        """Le format inclut source + page de chaque chunk."""
        from src.retrieval import format_chunks_for_prompt
        result = format_chunks_for_prompt(sample_documents_list)
        # Chaque chunk doit avoir son entete avec page et source
        assert "lvmh_2023.pdf" in result
        assert "page 3" in result
        assert "page 4" in result
        # Le contenu des chunks doit etre present
        assert "86 153" in result
        assert "15 174" in result

    def test_format_uses_separator(self, sample_documents_list):
        """Les chunks sont separes par '---' pour aider le LLM."""
        from src.retrieval import format_chunks_for_prompt
        result = format_chunks_for_prompt(sample_documents_list)
        assert "---" in result


# ============== TESTS ANTI-HALLUCINATION (refus auto-valide) ==============

class TestAntiHallucination:
    """Tests du module anti_hallucination (sans appel LLM, juste la logique de refus)."""

    def test_auto_validate_explicit_refusal(self):
        """Une reponse de refus explicite doit etre auto-validee sans appel LLM."""
        from src.anti_hallucination import verify_answer
        result = verify_answer(
            question="Quel est le PDG ?",
            answer="L'information n'est pas disponible dans les documents fournis.",
            context="(contexte qui ne mentionne pas le PDG)",
        )
        assert result["is_valid"] is True
        assert result["verdict"] == "VALIDE"
        # Verifie que l'auto-validation s'est faite (pas d'appel LLM)
        assert "refus" in result["explanation"].lower()


# ============== TESTS REFUSAL_QUALITY (regex uniquement, pas d'appel LLM) ==============

class TestRefusalQuality:
    """Tests partiels de refusal_quality (juste la detection regex)."""

    def test_detect_refusal_explicit(self):
        """La detection regex reconnait 'l'information n'est pas disponible'."""
        from tests.metrics.refusal_quality import _detect_explicit_refusal
        assert _detect_explicit_refusal("L'information n'est pas disponible dans les documents.") is True

    def test_detect_refusal_variant(self):
        """Variante avec 'non disponible'."""
        from tests.metrics.refusal_quality import _detect_explicit_refusal
        assert _detect_explicit_refusal("Information non disponible dans le corpus.") is True

    def test_no_refusal_on_answer(self):
        """Une vraie reponse n'est pas detectee comme un refus."""
        from tests.metrics.refusal_quality import _detect_explicit_refusal
        assert _detect_explicit_refusal("Le CA 2023 est de 86 153 millions d'euros.") is False