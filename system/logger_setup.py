import logging
import sys
from logging.handlers import RotatingFileHandler
from system.config import LOG_FORMAT, LOG_LEVEL

_global_file_handler = None

def get_global_file_handler():
    global _global_file_handler
    if _global_file_handler is None:
        try:
            _global_file_handler = RotatingFileHandler(
                "tracking.log",
                maxBytes=10 * 1024 * 1024,   # 10 MB per file
                backupCount=5,
                encoding="utf-8",
            )
            file_format = logging.Formatter("%(asctime)s [%(levelname)s] [%(name)s] %(message)s")
            _global_file_handler.setFormatter(file_format)
        except Exception as e:
            print(f"Warning: Could not create tracking.log file handler: {e}")
            _global_file_handler = logging.NullHandler()
    return _global_file_handler

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[90m',       # Dark Grey
        'INFO': '\033[94m',        # Blue
        'WARNING': '\033[93m',     # Yellow
        'ERROR': '\033[91m',       # Red
        'CRITICAL': '\033[1;91m',  # Bold Red
    }
    RESET = '\033[0m'
    CYAN = '\033[96m'
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        
        # Clean dashboard style format
        name_str = f"{self.CYAN}[{record.name:<16}]{self.RESET}"
        level_str = f"{color}[{record.levelname:<8}]{self.RESET}"
        
        # Override the format template per-record
        self._style._fmt = f"{level_str} {name_str} %(message)s"
        return super().format(record)

def setup_logger(name: str) -> logging.Logger:
    """
    Sets up a beautiful, dashboard-style terminal logger.
    Uses RotatingFileHandler to prevent disk from filling up.
    Rotates at 10 MB, keeps 5 backups (~60 MB max on disk).
    """
    logger = logging.getLogger(name)
    logger.setLevel(LOG_LEVEL)

    # Add handlers to the logger, but only if it doesn't have handlers already
    if not hasattr(logger, '_is_setup'):
        # 1. Console Output
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(ColoredFormatter())
        logger.addHandler(console_handler)
        
        # 2. File Output with rotation (shared single instance)
        file_handler = get_global_file_handler()
        if not isinstance(file_handler, logging.NullHandler):
            logger.addHandler(file_handler)
            
        # Stop propagation to root logger to prevent ugly duplicate logs
        logger.propagate = False
        logger._is_setup = True

    return logger