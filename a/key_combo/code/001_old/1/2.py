from __future__ import annotations

import random

import numpy as np


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored pixels left while preventing same-color repeats in any previously settled column.

    Input structure:
    - A black square grid contains several non-empty rows.
    - Each non-empty row has one to three colored singleton pixels, with no repeated color inside a row.
    - The same color may appear in different rows.

    Rule:
    - Process rows from top to bottom and pixels within each row from left to right.
    - Move each pixel to the smallest column not already used in its output row and not containing the same color in any earlier output row.
    - All original positions not selected by the process become black.

    Key insight:
    This is not ordinary left compression; every earlier row creates column exclusions for later pixels of the same color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = np.zeros_like(g)
    for r in range(h):
        row_pixels = [(c, int(g[r, c])) for c in range(w) if int(g[r, c]) != 0]
        used_cols: set[int] = set()
        for _old_c, color in row_pixels:
            placed = False
            for c in range(w):
                if c in used_cols:
                    continue
                if np.any(out[:r, c] == color):
                    continue
                out[r, c] = color
                used_cols.add(c)
                placed = True
                break
            if not placed:
                raise ValueError("no legal parking column")
    return out


def generate_input() -> np.ndarray:
    for _attempt in range(200):
        n = random.randint(8, 12)
        g = np.zeros((n, n), dtype=int)
        row_count = random.randint(4, min(8, n))
        rows = sorted(random.sample(range(n), row_count))
        colors = list(range(1, 10))
        common = random.choice(colors)
        common_rows = set(random.sample(rows, min(3, len(rows))))
        for r in rows:
            k = random.randint(1, 3)
            row_colors: list[int] = []
            if r in common_rows:
                row_colors.append(common)
            available = [c for c in colors if c not in row_colors]
            row_colors.extend(random.sample(available, k - len(row_colors)))
            cols = sorted(random.sample(range(1, n), len(row_colors)))
            random.shuffle(row_colors)
            for c, color in zip(cols, row_colors):
                g[r, c] = color
        try:
            main(g)
        except ValueError:
            continue
        return g
    raise RuntimeError("failed to generate puzzle input")
