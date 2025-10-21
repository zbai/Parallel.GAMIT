import logging
from logging import StreamHandler, Formatter


def setup_etm_logging(level=logging.INFO, format_string=' -- %(name)s: %(message)s'):
    """Setup logging for the entire ETM package"""
    # Configure the parent 'etm' logger
    etm_logger = logging.getLogger('geode.etm')

    # Avoid duplicate handlers
    if not etm_logger.handlers:
        handler = StreamHandler()
        if level == logging.INFO:
            handler.setFormatter(Formatter(' -- %(message)s'))
        else:
            handler.setFormatter(Formatter(format_string))
        etm_logger.addHandler(handler)
        etm_logger.setLevel(level)
        # Prevent propagation to root logger to avoid duplicate messages
        etm_logger.propagate = False

    return etm_logger