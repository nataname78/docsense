"""
Module de configuration du logging centralise.

Utilisation dans n'importe quel module du projet :

    from src.logger import get_logger
    logger = get_logger(__name__)
    
    logger.debug("Trace detaillee")     # n'apparait que si LOG_LEVEL=DEBUG
    logger.info("Etape importante")     # apparait par defaut
    logger.warning("Anomalie")          # apparait toujours
    logger.error("Erreur grave")        # apparait toujours

Configuration via la variable d'environnement LOG_LEVEL :
- DEBUG    : tout
- INFO     : info + warning + error (defaut)
- WARNING  : warning + error
- ERROR    : juste error

Format de sortie :
    [HH:MM:SS] [INFO ] [src.embeddings] message...
"""
import os
import logging
import sys
from dotenv import load_dotenv

load_dotenv()


# Niveau via .env, defaut = INFO
_LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Format avec timestamp court + niveau + module + message
_LOG_FORMAT = "[%(asctime)s] [%(levelname)-5s] [%(name)s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

# Configuration globale du root logger : niveau, format, sortie stdout
_root_logger = logging.getLogger()

# On verifie qu'on configure une seule fois (Streamlit reload peut re-importer)
if not _root_logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)
    handler.setFormatter(formatter)
    _root_logger.addHandler(handler)
    _root_logger.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

    # On rend les libs externes moins verbeuses
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name):
    """
    Retourne un logger nomme pour le module appelant.
    
    Usage typique :
        logger = get_logger(__name__)
    
    Le name est automatiquement defini par Python comme 'src.embeddings',
    'src.retrieval', etc. Ca permet de filtrer/identifier la source d'un log.
    """
    return logging.getLogger(name)


# Bloc de test
if __name__ == "__main__":
    print("Test du module logger\n" + "=" * 60)

    log = get_logger("test")
    log.debug("Ceci est un DEBUG (visible si LOG_LEVEL=DEBUG)")
    log.info("Ceci est un INFO (visible par defaut)")
    log.warning("Ceci est un WARNING")
    log.error("Ceci est un ERROR")

    print("\nNiveau actuel configurable via .env :")
    print(f"  LOG_LEVEL={_LOG_LEVEL}")
    print("  Pour voir les DEBUG : LOG_LEVEL=DEBUG dans .env, puis relancer.")