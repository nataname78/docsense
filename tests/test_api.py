"""Test rapide : la clé API Mistral fonctionne-t-elle ?"""
import os
from dotenv import load_dotenv
from mistralai import Mistral

# Charge les variables du fichier .env
load_dotenv()

# Initialise le client Mistral (il lit MISTRAL_API_KEY automatiquement)
client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))

# Test 1 : un appel LLM minimal
print("Test 1 : appel LLM en cours...")
response = client.chat.complete(
    model=os.getenv("LLM_MODEL"),
    messages=[{"role": "user", "content": "Dis 'OK' en un mot."}],
    max_tokens=10,
)
print(f"Reponse LLM : {response.choices[0].message.content}")

# Test 2 : un embedding
print("\nTest 2 : generation d'embedding en cours...")
embedding = client.embeddings.create(
    model=os.getenv("EMBEDDING_MODEL"),
    inputs=["Test de vectorisation"],
)
vector = embedding.data[0].embedding
print(f"Embedding genere : {len(vector)} dimensions")
print(f"Premieres valeurs : {vector[:5]}")

print("\nTout fonctionne ! Tu peux passer a la suite.")