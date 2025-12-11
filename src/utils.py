"""
Segédfüggvények, például a központi logger beállításához.
"""
import logging
import sys

def get_logger(name="legal_text_decoder"):
    """
    Beállít és visszaad egy logger-t, amely a standard output-ra ír.
    A Docker ezt a stream-et fogja elkapni és a log fájlba irányítani.
    """
    logger = logging.getLogger(name)
    if not logger.handlers: # Handler-ek duplikálásának elkerülése
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
