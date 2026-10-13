from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    colors = sorted({int(v) for v in g.flatten() if v not in (0,)})
    if not colors:
        return np.zeros((1, 1), dtype=int)

    counts: Dict[int, int] = {c: 0 for c in colors}

    # Draw horizontal/vertical link segments in cyan for counting.
    for color in colors:
        positions = [(r, c) for r in range(h) for c in range(w) if g[r, c] == color]
        by_row: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        by_col: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
        for r, c in positions:
            by_row[r].append((r, c))
            by_col[c].append((r, c))

        # Consecutive pairs in each row.
        for row, pts in by_row.items():
            pts.sort(key=lambda x: x[1])
            for i in range(len(pts) - 1):
                _, c1 = pts[i]
                _, c2 = pts[i + 1]
                a, b = sorted((c1, c2))
                clear = True
                for cc in range(a + 1, b):
                    if g[row, cc] != 0:
                        clear = False
                        break
                if clear:
                    # paint cyan segment for visual consistency
                    for cc in range(a + 1, b):
                        g[row, cc] = 8
                    counts[color] += 1

        # Consecutive pairs in each column.
        for col, pts in by_col.items():
            pts.sort(key=lambda x: x[0])
            for i in range(len(pts) - 1):
                r1, _ = pts[i]
                r2, _ = pts[i + 1]
                a, b = sorted((r1, r2))
                clear = True
                for rr in range(a + 1, b):
                    if g[rr, col] != 0:
                        clear = False
                        break
                if clear:
                    for rr in range(a + 1, b):
                        g[rr, col] = 8
                    counts[color] += 1

    ordered_colors = sorted(counts.items(), key=lambda x: x[1])
    max_cnt = max(v for _, v in ordered_colors) if ordered_colors else 0
    if max_cnt == 0:
        return np.zeros((len(ordered_colors), 1), dtype=int)
    out = np.zeros((len(ordered_colors), max_cnt), dtype=int)
    for r, (color, cnt) in enumerate(ordered_colors):
        out[r, :cnt] = color
    return out


def generate_input() -> np.ndarray:
    h = 14
    w = 14
    g = np.zeros((h, w), dtype=int)
    colors = [1, 2, 3, 4]
    random.shuffle(colors)

    used_rows = set()
    used_cols = set()

    for i, color in enumerate(colors[:3]):
        # Horizontal markers: two segments in same row.
        row = random.choice([r for r in range(1, h - 1) if r not in used_rows])
        used_rows.add(row)
        row_cols = random.sample([c for c in range(1, w - 1) if c not in used_cols], 3)
        row_cols.sort()
        c1, c2, c3 = row_cols
        g[row, c1] = color
        g[row, c2] = color
        g[row, c3] = color
        used_cols.update(row_cols)

        # One vertical segment with a new unique column.
        col = random.choice([c for c in range(1, w - 1) if c not in used_cols])
        rows = random.sample([r for r in range(1, h - 1) if r not in used_rows], 2)
        r1, r2 = rows
        g[r1, col] = color
        g[r2, col] = color
        used_rows.add(r1)
        used_rows.add(r2)
        used_cols.add(col)

    # add few extra markers to make 3-5 per color
    for color in colors[:3]:
        positions = np.argwhere(g == color)
        for r, c in positions[:2]:
            # keep stable noise only in background
            pass
    return g
