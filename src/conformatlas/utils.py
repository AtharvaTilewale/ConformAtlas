"""Utility functions, constants, and logging configuration for ConformAtlas."""

import datetime
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Gas constant R in kJ / (mol * K)
R_GAS_CONSTANT = 0.008314462618

# ANSI Color codes
if sys.stdout.isatty():
    RED = "\033[0;31m"
    GREEN = "\033[0;32m"
    YELLOW = "\033[1;33m"
    BLUE = "\033[0;34m"
    CYAN = "\033[0;36m"
    BOLD = "\033[1m"
    NC = "\033[0m"
else:
    RED = ""
    GREEN = ""
    YELLOW = ""
    BLUE = ""
    CYAN = ""
    BOLD = ""
    NC = ""


BANNER_RAW = """ ▄████▄   ▒█████   ███▄    █   █████▒▒█████   ██▀███   ███▄ ▄███▓ ▄▄▄     ▄▄▄█████▓ ██▓    ▄▄▄        ██████ 
▒██▀ ▀█  ▒██▒  ██▒ ██ ▀█   █ ▓██   ▒▒██▒  ██▒▓██ ▒ ██▒▓██▒▀█▀ ██▒▒████▄   ▓  ██▒ ▓▒▓██▒   ▒████▄    ▒██    ▒ 
▒▓█    ▄ ▒██░  ██▒▓██  ▀█ ██▒▒████ ░▒██░  ██▒▓██ ░▄█ ▒▓██    ▓██░▒██  ▀█▄ ▒ ▓██░ ▒░▒██░   ▒██  ▀█▄  ░ ▓██▄   
▒▓▓▄ ▄██▒▒██   ██░▓██▒  ▐▌██▒░▓█▒  ░▒██   ██░▒██▀▀█▄  ▒██    ▒██ ░██▄▄▄▄██░ ▓██▓ ░ ▒██░   ░██▄▄▄▄██   ▒   ██▒
▒ ▓███▀ ░░ ████▓▒░▒██░   ▓██░░▒█░   ░ ████▓▒░░██▓ ▒██▒▒██▒   ░██▒ ▓█   ▓██▒ ▒██▒ ░ ░██████▒▓█   ▓██▒▒██████▒▒
░ ░▒ ▒  ░░ ▒░▒░▒░ ░ ▒░   ▒ ▒  ▒ ░   ░ ▒░▒░▒░ ░ ▒▓ ░▒▓░░ ▒░   ░  ░ ▒▒   ▓▒█░ ▒ ░░   ░ ▒░▓  ░▒▒   ▓▒█░▒ ▒▓▒ ▒ ░
  ░  ▒     ░ ▒ ▒░ ░ ░░   ░ ▒░ ░       ░ ▒ ▒░   ░▒ ░ ▒░░  ░      ░  ▒   ▒▒ ░   ░    ░ ░ ▒  ░ ▒   ▒▒ ░░ ░▒  ░ ░
░        ░ ░ ░ ▒     ░   ░ ░  ░ ░   ░ ░ ░ ▒    ░░   ░ ░      ░     ░   ▒    ░        ░ ░    ░   ▒   ░  ░  ░  
░ ░          ░ ░           ░            ░ ░     ░            ░         ░  ░            ░  ░     ░  ░      ░  
░                                                                                                            """


def get_banner(color: bool = True) -> str:
    """Returns the ConformAtlas banner with vibrant gradient coloring."""
    lines = BANNER_RAW.splitlines()
    if not color:
        return "\n".join(lines)

    stops = [
        (0.0, (0, 245, 235)),  # Electric Cyan
        (0.5, (60, 130, 255)),  # Cobalt / Azure Blue
        (1.0, (185, 75, 250)),  # Neon Violet
    ]
    max_x = max(len(ln) for ln in lines) - 1
    max_y = len(lines) - 1

    rendered_lines = []
    for y, line in enumerate(lines):
        line_chars = []
        if y < 5:
            v_mult = 1.0
        else:
            v_mult = 1.0 - ((y - 4) / (max_y - 4)) * 0.35

        last_color = None
        for x, ch in enumerate(line):
            if ch == " ":
                line_chars.append(" ")
                continue
            pos = x / max_x
            for idx in range(len(stops) - 1):
                p0, c0 = stops[idx]
                p1, c1 = stops[idx + 1]
                if p0 <= pos <= p1 or idx == len(stops) - 2:
                    t = (pos - p0) / (p1 - p0) if p1 > p0 else 0
                    t = max(0.0, min(1.0, t))
                    r = int((c0[0] * (1 - t) + c1[0] * t) * v_mult)
                    g = int((c0[1] * (1 - t) + c1[1] * t) * v_mult)
                    b = int((c0[2] * (1 - t) + c1[2] * t) * v_mult)
                    cur_color = (r, g, b)
                    if cur_color != last_color:
                        line_chars.append(f"\033[38;2;{r};{g};{b}m")
                        last_color = cur_color
                    line_chars.append(ch)
                    break
        line_chars.append("\033[0m")
        rendered_lines.append("".join(line_chars))
    return "\n".join(rendered_lines)


def print_banner(ctx=None):
    """Prints the ConformAtlas ASCII art banner with vibrant colors."""
    if ctx is None:
        try:
            import click

            ctx = click.get_current_context(silent=True)
        except Exception:
            ctx = None

    if ctx is not None:
        root_ctx = ctx.find_root() if hasattr(ctx, "find_root") else ctx
        if getattr(root_ctx, "_banner_printed", False):
            return
        root_ctx._banner_printed = True

    use_color = (
        sys.stdout.isatty() and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"
    )
    print(get_banner(color=use_color))
    print()


def setup_logging(log_dir: Path, log_name_prefix: str = "conformatlas") -> Path:
    """Configures logging to file and console without leaking handlers.

    Parameters
    ----------
    log_dir : Path
        Directory where log file will be saved.
    log_name_prefix : str
        Prefix for the log file.

    Returns
    -------
    Path
        Absolute path to the created log file.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    now = datetime.datetime.now()
    log_filename = f"{log_name_prefix}_{now.strftime('%Y-%m-%d_%H-%M-%S')}.log"
    log_path = log_dir / log_filename

    logger = logging.getLogger("conformatlas")
    logger.setLevel(logging.DEBUG)

    # Clean existing handlers to avoid duplicates on repeated test runs
    logger.handlers.clear()

    # File handler
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(message)s")
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    logger.info(f"Log initialized at {log_path.resolve()}")
    return log_path


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to safely serialize NumPy data types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (datetime.date, datetime.datetime)):
            return obj.isoformat()
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(data: dict[str, Any], filepath: str | Path, indent: int = 2):
    """Save dictionary to JSON with NumPy and Path conversion support."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with filepath.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, cls=NumpyEncoder)
