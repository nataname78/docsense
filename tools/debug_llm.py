"""
Test isole : on donne au LLM le chunk page 1 EXACTEMENT (celui qu'on sait contenir "Ventes 86,2 milliards"),
et on lui demande la question "chiffre d'affaires".
Objectif : confirmer si le bug est dans le LLM ou ailleurs.
"""
from src.generation import generate_answer

# Le contexte fourni : on simule un chunk qui contient EXPLICITEMENT le bon chiffre
context = """
[Extrait 1 - source: lvmh_2023.pdf, page 1]
2023 : Nouvelle annee record pour LVMH

. Ventes 86,2 milliards d'euros
. Resultat operationnel courant 22,8 milliards d'euros
. Croissance au 4eme trimestre superieure a celle du 3eme
. Une empreinte economique et sociale forte en France et dans le monde

Paris, le 25 janvier 2024

LVMH Moet Hennessy Louis Vuitton, leader mondial des produits de luxe,
realise en 2023 des ventes de 86,2 milliards d'euros, en croissance organique de 13 %.
"""

question = "Quel est le chiffre d'affaires en 2023 ?"

print("=" * 70)
print("TEST ISOLE LLM")
print("=" * 70)
print(f"\nQuestion : {question}\n")
print(f"Contexte fourni au LLM :\n{context}\n")
print("=" * 70)
print("Reponse du LLM :")
print("=" * 70)

result = generate_answer(question, context)
print(result)