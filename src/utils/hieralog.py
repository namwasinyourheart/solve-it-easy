import sys
import logging
import re
import atexit
from tqdm import tqdm
from colorama import Fore, Style, init
from typing import Union

# Initialize colorama
init(autoreset=True)

# Global variables
current_level = 1
FIXED_FRAME_WIDTH = 160

def print_raw(*args, end="\n", **kwargs):
    """Print directly to stdout without interception."""
    sys.__stdout__.write(" ".join(map(str, args)) + end)
    sys.__stdout__.flush()

# ================================
# 🚀 Process Tracker for Level 1
# ================================
class ProcessTracker:
    def __init__(self):
        self.active_processes = {}
        self.last_level_1 = None  # Track last Level 1 process

    def start_process(self, key):
        """Start a new Level 1 process, auto-completing the previous one if needed."""
        if self.last_level_1 and self.last_level_1 != key:
            self.complete_process(self.last_level_1)  # Auto-complete previous Level 1 process
            print_raw("")  # Force a newline before new Level 1 log
        self.last_level_1 = key
        if key not in self.active_processes:
            self.active_processes[key] = "processing"
            self.display_status(key)

    def complete_process(self, key):
        """Mark a Level 1 process as completed, printing only a checkmark."""
        if key in self.active_processes and self.active_processes[key] == "processing":
            self.active_processes[key] = "completed"
            print_raw("✅")  # Print only the checkmark

    def get_icon(self, key, completed=False):
        return "✅" if completed else "⏳"

    def display_status(self, key, completed=False):
        """Display the process status (icon and key) without an extra newline."""
        icon = self.get_icon(key, completed)
        message = f"{Fore.YELLOW}{icon} {key}{Style.RESET_ALL}"
        print_raw(message, end="")

tracker = ProcessTracker()

# ============================
# 🎨 Custom Logging Formatter
# ============================
def format_multiline_message(message, indent, color):
    """
    Formats multi-line messages with hierarchical indentation and color.
    The first line is prefixed with "├──🛠️   " and subsequent lines with "│     ".
    """
    lines = message.split("\n")
    formatted_lines = [f"{indent}{color}├──🛠️  {lines[0]}{Style.RESET_ALL}"]
    formatted_lines += [f"{indent}{color}│     {line}{Style.RESET_ALL}" for line in lines[1:]]
    return "\n".join(formatted_lines)

class CustomFormatter(logging.Formatter):
    def format(self, record):
        message = record.getMessage()
        level, clean_message = extract_level_and_message(message)
        
        if level == 1:
            color = Fore.GREEN
        elif level == 2:
            color = Fore.BLUE
        elif level == 3:
            color = Fore.CYAN
        elif level == 4:
            color = Fore.MAGENTA
        elif level == 5:
            color = Fore.RED
        else:
            color = Fore.WHITE

        if level == 1:
            tracker.start_process(clean_message)
            return ""
        elif level in [2, 3, 4, 5]:
            indent = "│  " * (level - 2)
            return format_multiline_message(clean_message, indent, color)
        elif message.startswith("[COMPLETE]"):
            process_name = message[10:]
            tracker.complete_process(process_name)
            return ""
        return color + message + Style.RESET_ALL

def setup_logger(name: Union[str, None] = None):
    """Sets up and returns a logger (to be called manually in user code)."""
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = CustomFormatter("%(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

# ============================
# Utility Functions
# ============================
def extract_level_and_message(message):
    """
    Extracts the level from a message in the form "[x] Message".
    Returns (level, message) if matched; otherwise (None, message).
    """
    if isinstance(message, str):
        # match = re.match(r"\[(\d+)\]\s*(.*)", message, re.DOTALL)
        match = re.match(r"(\d+):\s*(.*)", message, re.DOTALL)
        if match:
            return int(match.group(1)), match.group(2)
    return None, message

# ----------------------------
# Hierarchical Print (hprint)
# ----------------------------
def hprint(*args, logger=None, **kwargs):
    """
    Hierarchical print: auto-detects the level from a "[x]" prefix,
    and prints using the enhanced logging system (framed output).
    """
    global current_level
    if not args:
        return
    message = " ".join(map(str, args)).strip()
    detected_level, clean_message = extract_level_and_message(message)
    if logger is None:
        logger = setup_logger()
    if detected_level is None:
        fprint(message, level=current_level)
        return
    if detected_level == 1:
        tracker.start_process(clean_message)
    if detected_level in [1, 2, 3, 4, 5]:
        logger.info(f"{detected_level}: {clean_message}")
        current_level = detected_level
        return
    fprint(clean_message, level=current_level)

# ----------------------------
# Plain Print (pprint)
# ----------------------------
def pprint(*args, **kwargs):
    """
    Plain print: prints messages with hierarchical indentation (without a frame).
    Auto-detects the level from a "[x]" prefix; if not provided, uses the current level's indent.
    Does not prepend the level number.
    """
    global current_level
    if not args:
        return
    message = " ".join(map(str, args)).strip()
    detected_level, clean_message = extract_level_and_message(message)
    if detected_level is None:
        detected_level = current_level
        text = message
    else:
        text = clean_message
        current_level = detected_level
    indent = "│  " * (detected_level - 1) + " " * 3
    for line in text.split("\n"):
        print_raw(f"{indent}{line}")

# ----------------------------
# Framed Print (fprint)
# ----------------------------
def fprint(*args, **kwargs):
    """
    Framed print: prints a message inside a frame with hierarchical indentation.
    Auto-detects the level from a "[x]" prefix; if not, uses the current level.
    """
    global current_level
    if not args:
        return
    message = " ".join(map(str, args)).strip()
    detected_level, clean_message = extract_level_and_message(message)
    if detected_level is None:
        detected_level = current_level
        text = message
    else:
        text = clean_message
        current_level = detected_level
    _frame_print(text, level=detected_level)

def _frame_print(output, level=1):
    """Internal function to print output inside a frame with proper indent."""
    output = str(output).strip()
    indent = "│  " * (level - 1)
    if output:
        lines = output.split("\n")
        wrapped_lines = []
        for line in lines:
            while len(line) > FIXED_FRAME_WIDTH:
                wrapped_lines.append(line[:FIXED_FRAME_WIDTH])
                line = line[FIXED_FRAME_WIDTH:]
            wrapped_lines.append(line)
        border = f"{indent}╭─{'─' * FIXED_FRAME_WIDTH}─╮"
        print_raw(border)
        for line in wrapped_lines:
            print_raw(f"{indent}│ {line.ljust(FIXED_FRAME_WIDTH)} │")
        print_raw(f"{indent}╰─{'─' * FIXED_FRAME_WIDTH}─╯")

# ----------------------------
# Progress Write (progress_write)
# ----------------------------
def progress_write(message, level=None):
    """
    Writes a progress message with proper indentation using tqdm.write.
    If level is not provided, uses the global current_level.
    """
    global current_level
    if level is None:
        level = current_level
    indent = "│  " * (level - 1)
    from tqdm import tqdm
    tqdm.write(f"{indent}{message}")

# ----------------------------
# Final Completion Log
# ----------------------------
def print_complete_log():
    """Prints the final completion message on program exit."""
    
    print_raw("\n\n✅ Well done!")

# atexit.register(print_complete_log)
