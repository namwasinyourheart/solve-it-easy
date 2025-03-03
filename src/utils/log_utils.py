import logging
from colorama import Fore, Style, init
from typing import Union

# Initialize colorama
init(autoreset=True)


class ColorFormatter(logging.Formatter):
    """Custom formatter to color the entire log message."""
    
    LOG_COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        log_color = self.LOG_COLORS.get(record.levelno, "")
        message = super().format(record)  
        return f"{log_color}{message}{Style.RESET_ALL}" 

def setup_logger(name: Union[str, None] = None):
    """Set up a logger with colored output."""
    logger = logging.getLogger(name)
    handler = logging.StreamHandler()
    formatter = ColorFormatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger
