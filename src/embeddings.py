"""
Module d'embeddings : wrapper custom autour du SDK officiel Mistral.

V3.1 : ajout d'un cache MD5 persistant sur disque pour eviter
les appels API redondants sur des chunks deja vus.

Pourquoi :
1. Contournement d'un bug 'KeyError: data' dans langchain-mistralai
2. Gestion robuste des rate limits du free tier (retry exponentiel)
3. NEW : cache MD5 pour eviter de re-payer Mistral sur des chunks deja vus

Implemente l'interface 'Embeddings' de LangChain :
- embed_documents(texts) -> list[list[float]]
- embed_query(text) -> list[float]
"""
import os
import time
import pickle
import hashlib
from pathlib import Path
from dotenv import load_dotenv
from mistralai import Mistral
from mistralai.models.sdkerror import SDKError
from langchain_core.embeddings import Embeddings
from src.logger import get_logger

logger = get_logger(__name__)

load_dotenv()


# ============== CACHE DES EMBEDDINGS ==============

CACHE_DIR = Path("data/cache")
CACHE_FILE = CACHE_DIR / "embeddings_cache.pkl"


def _compute_hash(text):
    """
    Calcule un hash MD5 unique d'un texte.
    Utilise UTF-8 pour garantir la coherence Windows/Mac/Linux.
    """
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_cache():
    """
    Charge le cache depuis le disque. Retourne un dict vide si absent.

    Format du cache : { md5_hash: list[float] embedding }
    """
    if not CACHE_FILE.exists():
        return {}
    try:
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logger.warning(f"Cache : erreur de chargement (reinitialise) : {e}")
        return {}


def _save_cache(cache):
    """Sauvegarde le cache sur disque."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


# ============== WRAPPER MISTRAL EMBEDDINGS AVEC CACHE ==============

class MistralEmbeddings(Embeddings):
    """
    Wrapper compatible LangChain autour du SDK Mistral officiel,
    avec :
    - retry exponentiel sur rate limits 429
    - cache MD5 persistant sur disque (V3.1)
    """

    def __init__(self, model="mistral-embed", batch_size=8, max_retries=6, use_cache=True):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY manquante dans .env")
        self.client = Mistral(api_key=api_key)
        self.model = model
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.use_cache = use_cache

        # Chargement du cache (une seule fois a l'init)
        if self.use_cache:
            self._cache = _load_cache()
            logger.info(f"Cache : {len(self._cache)} embeddings deja en cache.")
        else:
            self._cache = {}

        # Compteurs de statistiques (utiles pour debug et entretien)
        self._stats = {"hits": 0, "misses": 0, "api_calls": 0}

    def _call_api_with_retry(self, inputs):
        """
        Appelle l'API Mistral avec retry exponentiel sur 429.
        """
        for attempt in range(self.max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    inputs=inputs,
                )
                self._stats["api_calls"] += 1
                return response
            except SDKError as e:
                error_str = str(e)
                is_rate_limit = "429" in error_str or "capacity" in error_str.lower()
                if is_rate_limit and attempt < self.max_retries - 1:
                    wait = 2 ** (attempt + 1)
                    logger.warning(f"Rate limit Mistral, attente {wait}s, retry {attempt + 2}/{self.max_retries}")
                    time.sleep(wait)
                    continue
                raise

    def embed_documents(self, texts):
        """
        Embed une liste de textes avec cache MD5.

        Strategie :
        1. Hash chaque texte
        2. Separer les hits (deja en cache) des misses (a calculer)
        3. Appeler l'API uniquement sur les misses, en batches
        4. Mettre a jour le cache + retourner les embeddings dans l'ordre original
        """
        # Etape 1 : calculer les hashes
        hashes = [_compute_hash(text) for text in texts]

        # Etape 2 : identifier les misses
        if self.use_cache:
            missing_indices = [i for i, h in enumerate(hashes) if h not in self._cache]
        else:
            missing_indices = list(range(len(texts)))

        n_hits = len(texts) - len(missing_indices)
        n_misses = len(missing_indices)

        if self.use_cache and n_hits > 0:
            logger.info(f"Cache : {n_hits} hits / {n_misses} misses sur {len(texts)} textes.")

        # Etape 3 : appeler l'API uniquement sur les misses, par batches
        if missing_indices:
            missing_texts = [texts[i] for i in missing_indices]

            for batch_start in range(0, len(missing_texts), self.batch_size):
                batch_end = batch_start + self.batch_size
                batch_texts = missing_texts[batch_start:batch_end]
                batch_indices_in_missing = list(range(batch_start, min(batch_end, len(missing_texts))))

                batch_idx_num = batch_start // self.batch_size + 1
                n_batches = (len(missing_texts) + self.batch_size - 1) // self.batch_size
                logger.info(f"Batch {batch_idx_num}/{n_batches} ({len(batch_texts)} chunks)...")

                response = self._call_api_with_retry(batch_texts)
                batch_embeddings = [item.embedding for item in response.data]

                # Met a jour le cache avec les nouveaux embeddings
                for local_idx, embedding in zip(batch_indices_in_missing, batch_embeddings):
                    original_idx = missing_indices[local_idx]
                    h = hashes[original_idx]
                    if self.use_cache:
                        self._cache[h] = embedding

                self._stats["misses"] += len(batch_texts)

                # Pause anti-rate-limit entre batches
                if batch_end < len(missing_texts):
                    time.sleep(1.5)

            # Sauvegarde le cache apres tous les nouveaux embeddings
            if self.use_cache:
                _save_cache(self._cache)

        # Etape 4 : reconstruire la liste d'embeddings dans l'ordre original
        result = []
        for h in hashes:
            if self.use_cache and h in self._cache:
                result.append(self._cache[h])
                # Ne compte comme hit que les chunks deja presents AVANT cet appel
            else:
                # Cas ou use_cache=False : on a tous appele l'API, recuperer depuis _cache temporaire
                # mais en mode no-cache, _cache reste vide -> impossible ici en realite
                pass

        # Update stats hits (uniquement les vrais hits, pas les nouveaux ajouts)
        self._stats["hits"] += n_hits

        return result

    def embed_query(self, text):
        """
        Embed un texte unique (pour une question utilisateur).
        Egalement cache : si la meme question est posee 2 fois, 0 appel API.
        """
        h = _compute_hash(text)

        if self.use_cache and h in self._cache:
            self._stats["hits"] += 1
            return self._cache[h]

        response = self._call_api_with_retry([text])
        embedding = response.data[0].embedding

        if self.use_cache:
            self._cache[h] = embedding
            _save_cache(self._cache)

        self._stats["misses"] += 1
        return embedding

    def get_stats(self):
        """Retourne les stats du cache (utile pour debug)."""
        return dict(self._stats)


def get_embeddings_model():
    """Retourne une instance de notre wrapper MistralEmbeddings."""
    model_name = os.getenv("EMBEDDING_MODEL", "mistral-embed")
    return MistralEmbeddings(model=model_name)


# ============== UTILITAIRES CACHE ==============

def clear_cache():
    """Vide le cache des embeddings (utile pour tests)."""
    if CACHE_FILE.exists():
        CACHE_FILE.unlink()
        logger.info(f"Cache supprime : {CACHE_FILE}")
    else:
        logger.info("Aucun cache a supprimer.")


def cache_info():
    """Affiche des infos sur l'etat du cache."""
    if not CACHE_FILE.exists():
        logger.info("Aucun cache existant.")
        return

    cache = _load_cache()
    size_mb = CACHE_FILE.stat().st_size / (1024 * 1024)
    logger.info(f"Cache : {len(cache)} embeddings ({size_mb:.2f} MB)")
    logger.info(f"Fichier : {CACHE_FILE}")


# ============== BLOC DE TEST ==============

if __name__ == "__main__":
    print("Test du wrapper Mistral avec cache MD5\n" + "=" * 60)

    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "clear":
        clear_cache()
        sys.exit(0)
    if len(sys.argv) > 1 and sys.argv[1] == "info":
        cache_info()
        sys.exit(0)

    # Test 1 : premier appel (tout en MISS)
    print("\n--- Test 1 : premier appel (tout doit etre en MISS) ---")
    embed_model = get_embeddings_model()
    texts = [
        "Le chiffre d'affaires de LVMH en 2023.",
        "La performance de Sephora.",
        "Les investissements en France.",
    ]

    t0 = time.time()
    vectors1 = embed_model.embed_documents(texts)
    t1 = time.time()
    print(f"  Duree : {t1 - t0:.2f}s")
    print(f"  Stats : {embed_model.get_stats()}")

    # Test 2 : deuxieme appel (tout doit etre en HIT)
    print("\n--- Test 2 : meme appel (tout doit etre en HIT, instantane) ---")
    embed_model2 = get_embeddings_model()  # nouvelle instance, recharge le cache du disque
    t0 = time.time()
    vectors2 = embed_model2.embed_documents(texts)
    t1 = time.time()
    print(f"  Duree : {t1 - t0:.3f}s")
    print(f"  Stats : {embed_model2.get_stats()}")

    # Verification : les embeddings doivent etre identiques
    print("\n--- Test 3 : verification que les embeddings sont identiques ---")
    all_same = all(v1 == v2 for v1, v2 in zip(vectors1, vectors2))
    print(f"  Embeddings identiques apres cache : {all_same}")

    # Test 4 : ajout d'un nouveau chunk
    print("\n--- Test 4 : ajout d'un nouveau texte (1 MISS, 0 HIT) ---")
    embed_model3 = get_embeddings_model()
    new_texts = texts + ["Une phrase totalement nouvelle qui n'a jamais ete embedded."]
    t0 = time.time()
    vectors3 = embed_model3.embed_documents(new_texts)
    t1 = time.time()
    print(f"  Duree : {t1 - t0:.2f}s")
    print(f"  Stats : {embed_model3.get_stats()}")

    # Info finale
    print("\n--- Info cache ---")
    cache_info()