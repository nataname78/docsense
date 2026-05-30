"""
Jeu de reference (golden dataset) pour evaluer le RAG.

Format de chaque entree :
- question : str, la question utilisateur
- expected_answer : str, la reponse attendue (ground truth)
- expected_pages : list[int], pages ou l'info devrait se trouver
- category : str, type de question (pour analyse par categorie)
- difficulty : str, easy / medium / hard
- expected_behavior : str, comportement attendu pour les cas anti-hallucination

IMPORTANT : ce dataset est cale sur le corpus LVMH (lvmh_2023.pdf + lvmh_2022.pdf).
Pour evaluer sur d'autres documents, il faudrait adapter les questions et reponses.
"""

EVAL_DATASET = [
    # ====== CATEGORIE 1 : FACTUEL CHIFFRE SIMPLE (easy) ======
    {
        "question": "Quel est le chiffre d'affaires de LVMH en 2023 ?",
        "expected_answer": "Le chiffre d'affaires (ventes) de LVMH en 2023 s'eleve a 86 153 millions d'euros (86,2 milliards d'euros).",
        "expected_pages": [1, 3],
        "category": "factuel_chiffre",
        "difficulty": "easy",
    },
    {
        "question": "Quel est le resultat net 2023 de LVMH ?",
        "expected_answer": "Le resultat net part du Groupe en 2023 est de 15 174 millions d'euros (15,2 milliards d'euros).",
        "expected_pages": [3, 7],
        "category": "factuel_chiffre",
        "difficulty": "easy",
    },
    {
        "question": "Quelle est la dette financiere nette de LVMH en 2023 ?",
        "expected_answer": "La dette financiere nette de LVMH en 2023 est de 10 746 millions d'euros.",
        "expected_pages": [3],
        "category": "factuel_chiffre",
        "difficulty": "easy",
    },

    # ====== CATEGORIE 2 : TABLEAU FINANCIER (easy-medium) ======
    {
        "question": "Quelles sont les ventes par activite de LVMH en 2023 ?",
        "expected_answer": (
            "Ventes par activite en 2023 : Vins et Spiritueux 6 602 M€, "
            "Mode et Maroquinerie 42 169 M€, Parfums et Cosmetiques 8 271 M€, "
            "Montres et Joaillerie 10 902 M€, Distribution selective 17 885 M€, "
            "Autres et eliminations 324 M€. Total : 86 153 M€."
        ),
        "expected_pages": [3],
        "category": "tableau",
        "difficulty": "medium",
    },
    {
        "question": "Combien d'investissements d'exploitation LVMH a-t-il realises en 2023 ?",
        "expected_answer": "LVMH a realise 7 478 millions d'euros d'investissements d'exploitation en 2023.",
        "expected_pages": [11],
        "category": "tableau",
        "difficulty": "easy",
    },

    # ====== CATEGORIE 3 : COMPARAISON MULTI-PERIODES (medium) ======
    {
        "question": "Quelle est l'evolution du chiffre d'affaires entre 2022 et 2023 ?",
        "expected_answer": (
            "Le chiffre d'affaires est passe de 79 184 M€ en 2022 a 86 153 M€ en 2023, "
            "soit une progression de +6 969 M€ (environ +8,8% en variation publiee)."
        ),
        "expected_pages": [3],
        "category": "evolution",
        "difficulty": "medium",
    },
    {
        "question": "De combien le resultat net a-t-il progresse entre 2022 et 2023 en pourcentage ?",
        "expected_answer": (
            "Le resultat net est passe de 14 084 M€ (2022) a 15 174 M€ (2023), "
            "soit une progression de +1 090 M€ (+7,7% environ)."
        ),
        "expected_pages": [3, 7],
        "category": "evolution",
        "difficulty": "medium",
    },

    # ====== CATEGORIE 4 : PARAPHRASE / SEMANTIQUE (medium) ======
    {
        "question": "Quel est le benefice de LVMH en 2023 ?",
        "expected_answer": (
            "Le benefice (resultat net) de LVMH en 2023 s'eleve a environ 15 174 millions d'euros "
            "(15,2 milliards d'euros). Note : le document utilise les termes 'resultat net' plutot que 'benefice'."
        ),
        "expected_pages": [3, 7],
        "category": "paraphrase",
        "difficulty": "medium",
    },

    # ====== CATEGORIE 5 : NOMS PROPRES (BM25 test) ======
    {
        "question": "Que dit le document sur la performance de Tiffany & Co. ?",
        "expected_answer": (
            "Tiffany & Co. a marque un nouveau chapitre de son histoire en 2023 avec la "
            "reouverture du Landmark a New York, qui a recu un accueil exceptionnel. "
            "La collection Lock poursuit son deploiement mondial avec succes. "
            "L'activite Montres et Joaillerie (dont Tiffany fait partie) a enregistre "
            "une croissance organique de 7% en 2023."
        ),
        "expected_pages": [4],
        "category": "nom_propre",
        "difficulty": "medium",
    },
    {
        "question": "Quels sont les chiffres mentionnes concernant IFRS 16 ?",
        "expected_answer": (
            "Concernant IFRS 16 : remboursements des dettes locatives de 2 818 M€ en 2023 "
            "(vs 2 751 en 2022 et 2 453 en 2021), amortissements des droits d'utilisation de "
            "3 031 M€ en 2023 (vs 3 007 en 2022), interets payes sur dettes locatives de "
            "356 M€ en 2023 (vs 240 en 2022)."
        ),
        "expected_pages": [11],
        "category": "nom_propre_reference",
        "difficulty": "medium",
    },

    # ====== CATEGORIE 6 : ANTI-HALLUCINATION (hard - reponse attendue : refus) ======
    {
        "question": "Quel est le nom du PDG de LVMH ?",
        "expected_answer": "L'information n'est pas disponible dans les documents fournis.",
        "expected_pages": [],
        "category": "anti_hallucination",
        "difficulty": "hard",
        "expected_behavior": "refusal",
    },
    {
        "question": "Quel est le chiffre d'affaires de LVMH en 2024 ?",
        "expected_answer": "L'information n'est pas disponible dans les documents fournis. Les documents couvrent les exercices 2022 et 2023.",
        "expected_pages": [],
        "category": "anti_hallucination",
        "difficulty": "hard",
        "expected_behavior": "refusal",
    },
]


# ============== ANALYSE DU DATASET ==============

if __name__ == "__main__":
    print(f"Dataset d'evaluation : {len(EVAL_DATASET)} questions")
    print("=" * 60)

    # Comptage par categorie
    from collections import Counter
    cat_count = Counter(q["category"] for q in EVAL_DATASET)
    diff_count = Counter(q["difficulty"] for q in EVAL_DATASET)

    print("\nRepartition par categorie :")
    for cat, count in cat_count.most_common():
        print(f"  - {cat:30s} : {count} question(s)")

    print("\nRepartition par difficulte :")
    for diff, count in diff_count.most_common():
        print(f"  - {diff:10s} : {count} question(s)")

    print("\n--- Apercu des questions ---")
    for i, entry in enumerate(EVAL_DATASET, start=1):
        print(f"\n{i:2d}. [{entry['category']} / {entry['difficulty']}]")
        print(f"    Q : {entry['question']}")
        expected_short = entry['expected_answer'][:120]
        if len(entry['expected_answer']) > 120:
            expected_short += "..."
        print(f"    A : {expected_short}")