from pathlib import Path
import importlib.util
import random

import numpy as np


def _load_common():
    path = Path(__file__).with_name("_common.py")
    spec = importlib.util.spec_from_file_location("_arc051_common", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


U = _load_common()


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a diagonal color cycle only where row and column labels match.

    Input structure:
    The rightmost column contains row labels and the bottom row contains column
    labels. Interior colored samples appear only at matching label crossings.

    Rule:
    Matching row/column labels define the active mask. Infer the diagonal cycle
    from existing active samples using (row + col) modulo the number of sample
    colors, fill active black cells from that mapping, force inactive interior
    cells to black, and preserve the labels.

    Key insight:
    Label equality is only a mask; the actual filled colors come from a separate
    diagonal periodic pattern.
    """
    g = np.array(grid, copy=True)
    out = g.copy()
    h, w = g.shape
    active = np.zeros((h - 1, w - 1), dtype=bool)
    for r in range(h - 1):
        for c in range(w - 1):
            active[r, c] = g[r, w - 1] == g[h - 1, c]
    samples = [(r, c, int(g[r, c])) for r in range(h - 1) for c in range(w - 1)
               if active[r, c] and g[r, c] != 0]
    colors = sorted({v for _, _, v in samples}) or [3, 6]
    k = len(colors)
    mapping = {}
    for r, c, v in samples:
        mapping[(r + c) % k] = v
    for i, col in enumerate(colors):
        mapping.setdefault(i, col)
    for r in range(h - 1):
        for c in range(w - 1):
            out[r, c] = mapping[(r + c) % k] if active[r, c] else 0
    return out


def generate_input() -> np.ndarray:
    h = random.randint(8, 11)
    w = random.randint(8, 11)
    g = np.zeros((h, w), dtype=int)
    labels = [1, 2, 4]
    for r in range(h - 1):
        g[r, w - 1] = random.choice(labels)
    for c in range(w - 1):
        g[h - 1, c] = random.choice(labels)
    g[h - 1, w - 1] = 0
    colors = random.sample([3, 6, 7], random.randint(2, 3))
    need = set(range(len(colors)))
    active_cells = [(r, c) for r in range(h - 1) for c in range(w - 1) if g[r, w - 1] == g[h - 1, c]]
    random.shuffle(active_cells)
    for r, c in active_cells:
        res = (r + c) % len(colors)
        if res in need or random.random() < 0.12:
            g[r, c] = colors[res]
            need.discard(res)
    return g
