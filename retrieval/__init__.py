
from .extractors import GlobalExtractor

__all__ = ['GlobalExtractor']

__version__ = '0.0'


try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning("Could not import loguru")


try:
    import core
    logger.info(f"Visloc core version {core.__version__}")
except ImportError:
    logger.warning("Could not import visloc_core")
