import logging

LEVEL = logging.DEBUG

logger = logging.getLogger(__name__)

logger.setLevel(LEVEL)
ch = logging.StreamHandler()
ch.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(ch)
