"""Rigorous GROMACS XPM parser and converter."""

import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_xpm(xpm_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse a GROMACS XPM file and extract x-axis, y-axis, and 2D matrix values.

    Parameters
    ----------
    xpm_path : Union[str, Path]
        Path to the GROMACS XPM file.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        (x_axis, y_axis, matrix_2d) where matrix_2d has shape (len(y_axis), len(x_axis)).

    Raises
    ------
    FileNotFoundError
        If xpm_path does not exist.
    ValueError
        If the XPM structure, header, axes, or symbols cannot be parsed.
    """
    path = Path(xpm_path)
    if not path.exists():
        raise FileNotFoundError(f"XPM file not found: {path}")

    with path.open("r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # 1. Parse header dimensions: "<width> <height> <ncolors> <chars_per_pixel>"
    header_match = re.search(r'static\s+char\s*\*\s*\w+\[\]\s*=\s*\{\s*["\']\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*["\']', content)
    if not header_match:
        # Fallback: look for the first quoted string containing 4 integers
        header_match = re.search(r'["\']\s*(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s*["\']', content)

    if not header_match:
        raise ValueError(f"Could not parse valid XPM header with 4 dimensions in {path}")

    width = int(header_match.group(1))
    height = int(header_match.group(2))
    ncolors = int(header_match.group(3))
    cpp = int(header_match.group(4))

    # 2. Parse x-axis and y-axis comments
    # Axes can span multiple comment lines
    x_floats: list[float] = []
    y_floats: list[float] = []

    # Find all x-axis comment blocks
    for match in re.finditer(r'/\*\s*x-axis:\s*(.*?)\*/', content, re.DOTALL):
        tokens = match.group(1).split()
        for tok in tokens:
            try:
                x_floats.append(float(tok))
            except ValueError:
                continue

    # Find all y-axis comment blocks
    for match in re.finditer(r'/\*\s*y-axis:\s*(.*?)\*/', content, re.DOTALL):
        tokens = match.group(1).split()
        for tok in tokens:
            try:
                y_floats.append(float(tok))
            except ValueError:
                continue

    # If axes were not found in comments, generate 0..width-1 and 0..height-1
    if len(x_floats) != width:
        if len(x_floats) == 0:
            x_floats = [float(i) for i in range(width)]
        else:
            raise ValueError(f"Parsed {len(x_floats)} x-axis values but XPM header specifies width {width}.")

    if len(y_floats) != height:
        if len(y_floats) == 0:
            y_floats = [float(j) for j in range(height)]
        else:
            raise ValueError(f"Parsed {len(y_floats)} y-axis values but XPM header specifies height {height}.")

    # 3. Parse color map lines
    # Example formats in GROMACS:
    # "a c #FFFFFF "/* "0.0" */,
    # "ab c #00FF00 "/* "15.23" */,
    # "   c #000000 "/* "0" */,
    symbol_to_val: dict[str, float] = {}

    lines = content.splitlines()
    # Find lines that define color mappings
    color_line_pattern = re.compile(r'^"(.{' + str(cpp) + r'})\s+c\s+([^\s"]+)\s*"\s*(?:/\*\s*["\']?([^"\'\*]+)["\']?\s*\*/)?')

    for line in lines:
        stripped = line.strip()
        m = color_line_pattern.match(stripped)
        if m:
            symbol = m.group(1)
            val_str = m.group(3)
            if val_str is not None:
                try:
                    val = float(val_str.strip())
                except ValueError:
                    val = float(len(symbol_to_val))
            else:
                # Fallback: check if rightmost token is float
                tokens = stripped.replace("/*", " ").replace("*/", " ").replace('"', " ").split()
                val = None
                for tok in reversed(tokens):
                    try:
                        val = float(tok)
                        break
                    except ValueError:
                        continue
                if val is None:
                    val = float(len(symbol_to_val))
            symbol_to_val[symbol] = val

    if len(symbol_to_val) < ncolors:
        # Fallback: find any line of form "symbol c ... "
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('"') and ' c ' in stripped:
                sym = stripped[1:1+cpp]
                if sym not in symbol_to_val:
                    # extract any float inside /* "..." */
                    fm = re.search(r'/\*\s*["\']?([0-9\.\-\+eE]+)["\']?\s*\*/', stripped)
                    if fm:
                        symbol_to_val[sym] = float(fm.group(1))

    if not symbol_to_val:
        raise ValueError(f"Could not parse any color map mappings in {path}")

    # 4. Parse matrix rows
    # Matrix rows are lines starting with '"' and having length approximately width*cpp + 2 quotes
    raw_rows: list[str] = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"') and not stripped.startswith('/*') and ' c ' not in stripped and not stripped.startswith(f'"{width} '):
            # Extract row string inside quotes
            # Handle possible trailing comma
            match_row = re.match(r'^"([^"]*)"', stripped)
            if match_row:
                row_str = match_row.group(1)
                if len(row_str) == width * cpp:
                    raw_rows.append(row_str)

    if len(raw_rows) != height:
        raise ValueError(
            f"Expected {height} matrix rows of width {width*cpp}, found {len(raw_rows)} rows in {path}"
        )

    # GROMACS XPM orders rows top-down (row 0 is maximum y).
    # Reversing raw_rows matches increasing y_axis order (y_axis[0] to y_axis[height-1])
    raw_rows.reverse()

    matrix = np.full((height, width), np.nan, dtype=np.float64)

    for y_idx, row in enumerate(raw_rows):
        for x_idx in range(width):
            sym = row[x_idx * cpp : (x_idx + 1) * cpp]
            if sym not in symbol_to_val:
                raise ValueError(
                    f"Unknown symbol '{sym}' at row {y_idx}, col {x_idx}. "
                    f"Available symbols: {list(symbol_to_val.keys())[:10]}"
                )
            matrix[y_idx, x_idx] = symbol_to_val[sym]

    return np.array(x_floats), np.array(y_floats), matrix


def xpm_to_dataframe(xpm_path: str | Path) -> pd.DataFrame:
    """Convert XPM file into a pandas DataFrame with columns ['x', 'y', 'z']."""
    x_axis, y_axis, matrix = parse_xpm(xpm_path)
    records = []
    for y_idx, y_val in enumerate(y_axis):
        for x_idx, x_val in enumerate(x_axis):
            z_val = matrix[y_idx, x_idx]
            if not np.isnan(z_val):
                records.append((x_val, y_val, z_val))
    return pd.DataFrame(records, columns=["x", "y", "z"])


def xpm_to_dat(xpm_path: str | Path, dat_path: str | Path) -> Path:
    """Convert a GROMACS XPM file into a 3-column DAT file (x y z)."""
    df = xpm_to_dataframe(xpm_path)
    out_path = Path(dat_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, header=False, float_format="%.5f")
    return out_path
