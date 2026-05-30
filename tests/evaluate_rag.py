"""
Script d'evaluation complet du RAG.

Pipeline :
1. Charge l'index FAISS et reconstruit le HybridRetriever
2. Iter sur chaque question du golden dataset
3. Pour chaque question :
   a. Retrieve les chunks (top-4 via hybrid + reranking)
   b. Genere la reponse via le LLM
   c. Calcule les 4 metriques (faithfulness, relevancy, precision, recall)
   d. Stocke le resultat
4. Agrege :
   - Scores moyens globaux
   - Scores par categorie
   - Scores par difficulte
5. Genere un rapport Markdown reutilisable pour le README

Duree estimee : 10-15 minutes (132+ appels Mistral avec retries).
"""
import os
import time
import json
from pathlib import Path
from datetime import datetime

# Modules du projet
from src.vectorstore import load_vectorstore
from src.ingestion import extract_text_from_pdf
from src.chunking import chunk_pages
from src.table_extractor import enrich_pages_with_tables
from src.retrieval import HybridRetriever
from src.pipeline import ask

# Modules d'evaluation
from tests.evaluation_dataset import EVAL_DATASET
from tests.metrics.faithfulness import compute_faithfulness
from tests.metrics.answer_relevancy import compute_answer_relevancy
from tests.metrics.context_precision import compute_context_precision
from tests.metrics.context_recall import compute_context_recall
from tests.metrics.refusal_quality import compute_refusal_quality


# ============== CHEMINS ==============
RESULTS_DIR = Path("data/eval_results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_one_question(entry, retriever, idx, total):
    """
    Evalue UNE question : retrieve + generate + 4 metriques.

    Args:
        entry: dict du golden dataset
        retriever: HybridRetriever
        idx: index courant (pour affichage)
        total: nombre total de questions

    Returns:
        dict avec tous les resultats
    """
    question = entry["question"]
    expected = entry["expected_answer"]

    print(f"\n[{idx}/{total}] {entry['category']} / {entry['difficulty']}")
    print(f"  Question : {question}")

    # ----- ETAPE 1 : retrieve + generate -----
    print(f"  [Run RAG]...")
    t0 = time.time()
    result = ask(
        retriever,
        question,
        k=4,
        use_hybrid=True,
        use_reranking=True,
        retrieve_n=20,
    )
    rag_duration = time.time() - t0
    answer = result["answer"]
    chunks = result["sources"]
    print(f"  -> Reponse generee en {rag_duration:.1f}s ({len(chunks)} chunks)")

    # Format le contexte pour les metriques (concatene les chunks)
    context_str = "\n\n---\n\n".join([c.page_content for c in chunks])

    # ----- ETAPE 2 : metriques (refusal_quality pour les questions hard impossibles, sinon les 4 classiques) -----
    print(f"  [Evaluation des metriques]...")

    is_refusal_question = entry.get("expected_behavior") == "refusal"

    try:
        if is_refusal_question:
            # Pour les questions volontairement impossibles : refusal_quality uniquement
            print(f"  [Mode 'refusal_quality' (question volontairement impossible)]")
            refusal = compute_refusal_quality(question, answer, context_str)
            time.sleep(1)

            # Les 4 metriques classiques ne sont pas applicables ici (renvoient None)
            faith = {"score": None, "note": "N/A : question de type refusal"}
            relev = {"score": None, "note": "N/A : question de type refusal"}
            prec = {"score": None, "note": "N/A : question de type refusal"}
            recall = {"score": None, "note": "N/A : question de type refusal"}
        else:
            # Questions normales : les 4 metriques classiques
            faith = compute_faithfulness(question, answer, context_str)
            time.sleep(1)
            relev = compute_answer_relevancy(question, answer)
            time.sleep(1)
            prec = compute_context_precision(question, chunks)
            time.sleep(1)
            recall = compute_context_recall(question, expected, chunks)
            time.sleep(1)

            # refusal_quality non applicable
            refusal = {"score": None, "note": "N/A : question normale"}
    except Exception as e:
        print(f"  ! Erreur eval : {e}")
        return {
            "question": question,
            "category": entry["category"],
            "difficulty": entry["difficulty"],
            "answer": answer,
            "expected_answer": expected,
            "error": str(e),
        }

    scores = {
        "faithfulness": faith.get("score"),
        "answer_relevancy": relev.get("score"),
        "context_precision": prec.get("score"),
        "context_recall": recall.get("score"),
        "refusal_quality": refusal.get("score"),
    }

    # Affichage compact
    # Affichage compact
    def _fmt(v):
        return f"{v:.2f}" if isinstance(v, (int, float)) else "N/A"
    if is_refusal_question:
        print(f"  Scores : refusal_quality={_fmt(scores['refusal_quality'])}")
    else:
        print(
            f"  Scores : faith={_fmt(scores['faithfulness'])} "
            f"relev={_fmt(scores['answer_relevancy'])} "
            f"prec={_fmt(scores['context_precision'])} "
            f"recall={_fmt(scores['context_recall'])}"
        )

    return {
        "question": question,
        "category": entry["category"],
        "difficulty": entry["difficulty"],
        "answer": answer,
        "expected_answer": expected,
        "n_chunks": len(chunks),
        "rag_duration_s": round(rag_duration, 2),
        "scores": scores,
        "faithfulness_detail": faith,
        "answer_relevancy_detail": relev,
        "context_precision_detail": prec,
        "context_recall_detail": recall,
        "refusal_quality_detail": refusal,
        "is_refusal_question": is_refusal_question,
    }


def aggregate_scores(results):
    """
    Agrege les scores : moyenne globale, par categorie, par difficulte.

    Returns:
        dict avec 'global', 'by_category', 'by_difficulty'
    """
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "refusal_quality"]

    def _avg(values):
        valid = [v for v in values if isinstance(v, (int, float))]
        if not valid:
            return None
        return sum(valid) / len(valid)

    # Global
    global_avg = {}
    for m in metrics:
        values = [r.get("scores", {}).get(m) for r in results if "scores" in r]
        global_avg[m] = _avg(values)

    # Par categorie
    by_category = {}
    categories = set(r["category"] for r in results if "scores" in r)
    for cat in sorted(categories):
        cat_results = [r for r in results if r.get("category") == cat and "scores" in r]
        by_category[cat] = {
            m: _avg([r["scores"].get(m) for r in cat_results])
            for m in metrics
        }
        by_category[cat]["n_questions"] = len(cat_results)

    # Par difficulte
    by_difficulty = {}
    difficulties = set(r["difficulty"] for r in results if "scores" in r)
    for diff in sorted(difficulties):
        diff_results = [r for r in results if r.get("difficulty") == diff and "scores" in r]
        by_difficulty[diff] = {
            m: _avg([r["scores"].get(m) for r in diff_results])
            for m in metrics
        }
        by_difficulty[diff]["n_questions"] = len(diff_results)

    return {
        "global": global_avg,
        "by_category": by_category,
        "by_difficulty": by_difficulty,
        "n_questions_total": len([r for r in results if "scores" in r]),
        "n_errors": len([r for r in results if "error" in r]),
    }


def generate_markdown_report(results, aggregates, output_path):
    """Genere un rapport Markdown lisible et reutilisable dans le README."""

    def _fmt(v):
        if v is None or not isinstance(v, (int, float)):
            return "N/A"
        return f"{v:.2f}"

    lines = []
    lines.append("# Rapport d'evaluation RAG")
    lines.append("")
    lines.append(f"_Genere le {datetime.now().strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append(f"**Corpus evalue** : 2 documents (lvmh_2022.pdf + lvmh_2023.pdf)")
    lines.append(f"**Jeu de reference** : {aggregates['n_questions_total']} questions ")
    lines.append(f"**Erreurs d'evaluation** : {aggregates['n_errors']}")
    lines.append("")
    lines.append("## Scores moyens globaux")
    lines.append("")
    g = aggregates["global"]
    lines.append("| Metrique | Score | Applicable a |")
    lines.append("|---|---|---|")
    lines.append(f"| **Faithfulness** (fidelite anti-hallucination) | **{_fmt(g['faithfulness'])}** | Questions normales |")
    lines.append(f"| **Answer Relevancy** (pertinence de la reponse) | **{_fmt(g['answer_relevancy'])}** | Questions normales |")
    lines.append(f"| **Context Precision** (precision du retrieval) | **{_fmt(g['context_precision'])}** | Questions normales |")
    lines.append(f"| **Context Recall** (rappel du retrieval) | **{_fmt(g['context_recall'])}** | Questions normales |")
    lines.append(f"| **Refusal Quality** (qualite des refus) | **{_fmt(g['refusal_quality'])}** | Questions impossibles (hard) |")
    lines.append("")
    lines.append("## Scores par categorie de question")
    lines.append("")
    lines.append("| Categorie | N | Faith | Relev | Prec | Recall | RefusalQ |")
    lines.append("|---|---|---|---|---|---|---|")
    for cat, s in aggregates["by_category"].items():
        lines.append(
            f"| {cat} | {s['n_questions']} | {_fmt(s['faithfulness'])} | "
            f"{_fmt(s['answer_relevancy'])} | {_fmt(s['context_precision'])} | "
            f"{_fmt(s['context_recall'])} | {_fmt(s.get('refusal_quality'))} |"
        )
    lines.append("")
    lines.append("## Scores par difficulte")
    lines.append("")
    lines.append("| Difficulte | N | Faith | Relev | Prec | Recall | RefusalQ |")
    lines.append("|---|---|---|---|---|---|---|")
    for diff in ["easy", "medium", "hard"]:
        if diff in aggregates["by_difficulty"]:
            s = aggregates["by_difficulty"][diff]
            lines.append(
                f"| {diff} | {s['n_questions']} | {_fmt(s['faithfulness'])} | "
                f"{_fmt(s['answer_relevancy'])} | {_fmt(s['context_precision'])} | "
                f"{_fmt(s['context_recall'])} | {_fmt(s.get('refusal_quality'))} |"
            )
    lines.append("")
    lines.append("## Detail par question")
    lines.append("")
    for i, r in enumerate(results, start=1):
        if "error" in r:
            lines.append(f"### {i}. [{r['category']} / {r['difficulty']}] - ERREUR")
            lines.append(f"**Q** : {r['question']}")
            lines.append(f"**Erreur** : {r['error']}")
            lines.append("")
            continue

        s = r["scores"]
        lines.append(f"### {i}. [{r['category']} / {r['difficulty']}]")
        lines.append(f"**Q** : {r['question']}")
        lines.append(f"**Reponse generee** : {r['answer']}")
        lines.append("")
        if r.get("is_refusal_question"):
            lines.append(f"Score : RefusalQ={_fmt(s['refusal_quality'])} (question de type refusal)")
        else:
            lines.append(
                f"Scores : Faith={_fmt(s['faithfulness'])} | "
                f"Relev={_fmt(s['answer_relevancy'])} | "
                f"Prec={_fmt(s['context_precision'])} | "
                f"Recall={_fmt(s['context_recall'])}"
            )
        lines.append("")

    lines.append("## Methodologie")
    lines.append("")
    lines.append("- **LLM juge** : Mistral Small (meme modele que la generation)")
    lines.append("- **Temperature juge** : 0.0 (pour reproductibilite)")
    lines.append("- **Mode RAG evalue** : hybrid search (BM25 + FAISS + RRF) + reranking (bge-base) + LLM generation")
    lines.append("- **Configuration** : k=4, retrieve_n=20")
    lines.append("- **Implementation des metriques** : maison, inspiree de RAGAS, LLM-as-judge")
    lines.append("")
    lines.append("## Limites connues")
    lines.append("")
    lines.append("- L'evaluation utilise le meme LLM (Mistral) comme juge et comme generateur. ")
    lines.append("  L'etat de l'art recommande un LLM-juge different pour eviter les biais d'auto-evaluation. ")
    lines.append("  En revanche, comme metrique relative pour comparer versions, ca reste valide.")
    lines.append("- Le golden dataset est petit (12 questions). Pour une evaluation rigoureuse en production, ")
    lines.append("  on viserait 50-200 questions diversifiees, idealement annotees par plusieurs experts metier.")
    lines.append("")

    output_path = Path(output_path)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nRapport Markdown ecrit dans : {output_path}")


# ============== MAIN ==============

def main():
    print("=" * 70)
    print("EVALUATION DU RAG - lancement")
    print("=" * 70)

    # Etape 0 : Charger l'index et reconstruire le retriever
    print("\n[Setup] Chargement de l'index FAISS...")
    vs = load_vectorstore("data/indexes/sample")

    print("[Setup] Reconstruction des chunks (avec tableaux)...")
    # Reconstruit a partir de tous les PDFs presents dans data/raw/
    all_chunks = []
    raw_dir = Path("data/raw")
    pdf_files = sorted([p for p in raw_dir.glob("*.pdf")])
    if not pdf_files:
        print("ERREUR : aucun PDF dans data/raw/")
        return

    for pdf_path in pdf_files:
        print(f"  - Re-extraction de {pdf_path.name}...")
        result = extract_text_from_pdf(str(pdf_path))
        pages = enrich_pages_with_tables(result["pages"], str(pdf_path))
        chunks = chunk_pages(pages=pages, file_name=result["metadata"]["file_name"])
        all_chunks.extend(chunks)
    print(f"[Setup] Total : {len(all_chunks)} chunks reconstruits a partir de {len(pdf_files)} PDF(s)")

    retriever = HybridRetriever(vectorstore=vs, chunks=all_chunks)

    # Etape 1 : Evaluer chaque question
    print(f"\n[Eval] {len(EVAL_DATASET)} questions a evaluer...")
    print(f"[Eval] Duree estimee : 10-15 minutes (~132 appels LLM avec retries)")
    print("=" * 70)

    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for i, entry in enumerate(EVAL_DATASET, start=1):
        try:
            result = evaluate_one_question(entry, retriever, i, len(EVAL_DATASET))
            results.append(result)
        except KeyboardInterrupt:
            print("\n\n[INTERRUPTION UTILISATEUR] Sauvegarde des resultats partiels...")
            break
        except Exception as e:
            print(f"  ! Erreur sur la question {i} : {e}")
            results.append({
                "question": entry["question"],
                "category": entry["category"],
                "difficulty": entry["difficulty"],
                "error": str(e),
            })

        # Sauvegarde partielle apres chaque question (au cas ou)
        partial_path = RESULTS_DIR / f"eval_partial_{timestamp}.json"
        with open(partial_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    # Etape 2 : Aggregation
    print("\n" + "=" * 70)
    print("AGREGATION DES SCORES")
    print("=" * 70)
    aggregates = aggregate_scores(results)

    print("\nScores moyens GLOBAUX :")
    g = aggregates["global"]
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        v = g.get(m)
        v_str = f"{v:.3f}" if isinstance(v, (int, float)) else "N/A"
        print(f"  {m:25s} : {v_str}")

    # Etape 3 : Rapports
    json_path = RESULTS_DIR / f"eval_full_{timestamp}.json"
    md_path = RESULTS_DIR / f"eval_report_{timestamp}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "results": results,
            "aggregates": aggregates,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResultats JSON ecrits dans : {json_path}")

    generate_markdown_report(results, aggregates, md_path)

    print("\n" + "=" * 70)
    print("EVALUATION TERMINEE")
    print("=" * 70)


if __name__ == "__main__":
    main()