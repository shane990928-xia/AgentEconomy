import logging
import sys

def get_logger(name, level=logging.INFO):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)      

        handler = logging.StreamHandler(sys.stdout)

        formatter = logging.Formatter(
            '%(asctime)s | PID:%(process)d | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.propagate = False
        
    return logger