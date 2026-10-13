import random
from collections import Counter

import numpy as np


BLACK = 0
GREY = 5
MARKERS = [1, 2, 3, 4, 6, 7, 8, 9]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each straight path segment after every perpendicular grey gate using that gate's adjacent marker color.

    Input structure:
    - Black background with straight one-cell horizontal or vertical colored paths.
    - Short grey gates cross those paths perpendicularly.
    - Each gate has one colored marker immediately outside one end.

    Rule:
    - Horizontal paths are read left-to-right; vertical paths are read top-to-bottom.
    - The segment before the first gate keeps the path color.
    - After crossing a gate, the path color changes to that gate's marker color until the next gate.

    Key insight:
    - Gates define an ordered recoloring sequence along the path direction.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    markers = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if g[r, c] != GREY:
                continue
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = r + dr * 2, c + dc * 2
                mr, mc = r + dr * 3, c + dc * 3
                if 0 <= mr < g.shape[0] and 0 <= mc < g.shape[1] and g[mr, mc] not in (BLACK, GREY):
                    markers[(r, c)] = int(g[mr, mc])
    colors = [c for c in np.unique(g) if c not in (BLACK, GREY)]
    for color in colors:
        pts = np.argwhere(g == color)
        if not len(pts):
            continue
        row_counts = Counter(int(r) for r, _ in pts)
        col_counts = Counter(int(c) for _, c in pts)
        if max(row_counts.values()) >= max(col_counts.values()):
            row = row_counts.most_common(1)[0][0]
            cols = sorted({int(c) for r, c in pts if int(r) == row})
            gates = sorted(c for c in cols if g[row, c] == color and any((row, c + d) in markers for d in (-1, 0, 1)))
            paint = color
            gate_cols = sorted(c for (r, c), _ in markers.items() if r == row)
            last = min(cols)
            for gc in gate_cols:
                out[row, last:gc] = paint
                paint = markers[(row, gc)]
                last = gc + 1
            out[row, last : max(cols) + 1] = paint
        else:
            col = col_counts.most_common(1)[0][0]
            rows = sorted({int(r) for r, c in pts if int(c) == col})
            paint = color
            gate_rows = sorted(r for (r, c), _ in markers.items() if c == col)
            last = min(rows)
            for gr in gate_rows:
                out[last:gr, col] = paint
                paint = markers[(gr, col)]
                last = gr + 1
            out[last : max(rows) + 1, col] = paint
    out[g == GREY] = GREY
    return out


def generate_input() -> np.ndarray:
    g = np.zeros((20, 20), dtype=int)
    color = random.choice([1, 2, 3, 4])
    row = random.randint(5, 14)
    g[row, 2:18] = color
    gate_cols = sorted(random.sample(range(5, 16), random.randint(2, 3)))
    for gc, mk in zip(gate_cols, random.sample(MARKERS, len(gate_cols))):
        g[row - 1 : row + 2, gc] = GREY
        g[row - 3, gc] = mk
    return g
