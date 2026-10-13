from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
LIGHT_GREY = 7
DIRS = {
    "U": (-1, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (0, -1),
}
DIR_ORDER = ["U", "R", "D", "L"]
PATTERN_POS = {
    "U": (0, 0),
    "R": (0, 1),
    "D": (1, 0),
    "L": (1, 1),
}


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _top_directions(points: list[tuple[int, int]]) -> frozenset[str]:
    pts = set(points)
    counts = {d: 0 for d in DIR_ORDER}
    for r, c in pts:
        for d, (dr, dc) in DIRS.items():
            if (r + dr, c + dc) not in pts:
                counts[d] += 1
    ranked = sorted(DIR_ORDER, key=lambda d: (-counts[d], DIR_ORDER.index(d)))
    return frozenset(ranked[:2])


def _extract_legend(grid: np.ndarray) -> dict[frozenset[str], int]:
    h, w = grid.shape
    mapping: dict[frozenset[str], int] = {}
    if h < 4 or w < 2:
        return mapping
    for c in range(w - 1):
        color_block = grid[h - 2 : h, c : c + 2]
        color = int(color_block[0, 0])
        if color in (BLACK, GREY, LIGHT_GREY) or not bool(np.all(color_block == color)):
            continue
        pattern = grid[h - 4 : h - 2, c : c + 2]
        flat = [(int(pattern[rr, cc]), d) for d, (rr, cc) in PATTERN_POS.items()]
        if not all(v in (GREY, LIGHT_GREY) for v, _d in flat):
            continue
        dark_dirs = frozenset(d for _v, d in sorted(flat, key=lambda item: (item[0], DIR_ORDER.index(item[1])))[:2])
        mapping[dark_dirs] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey foreground objects by matching their two strongest exposed boundary directions to a bottom legend.

    Input structure:
    - Black is the background and Grey objects occupy the main area.
    - The last four rows form a legend: each entry has a 2x2 grey direction pattern over a 2x2 color swatch.
    - In the pattern, positions encode up, right, down, and left; the two darker cells name the object signature.

    Rule:
    - Count, for each Grey object, how many object pixels expose a side toward up, right, down, and left.
    - Select the two directions with largest counts and find the legend entry with those two dark pattern cells.
    - Recolor the whole Grey object with that entry's swatch color while leaving the bottom legend unchanged.

    Key insight: The small grey pattern is a directional fingerprint for the object's exposed perimeter.
    """
    g = np.asarray(grid, dtype=int)
    h, _w = g.shape
    legend = _extract_legend(g)
    out = g.copy()
    if h <= 4 or not legend:
        return out

    main_mask = g[: h - 4, :] == GREY
    for comp in _components(main_mask):
        signature = _top_directions(comp)
        color = legend.get(signature)
        if color is None:
            continue
        for r, c in comp:
            out[r, c] = int(color)
    return out


def _draw_legend(grid: np.ndarray, entries: list[tuple[frozenset[str], int]]) -> None:
    h, w = grid.shape
    step = 3
    total = len(entries) * 2 + (len(entries) - 1)
    c = max(0, (w - total) // 2)
    for dirs, color in entries:
        pattern = np.full((2, 2), LIGHT_GREY, dtype=int)
        for d in dirs:
            rr, cc = PATTERN_POS[d]
            pattern[rr, cc] = GREY
        grid[h - 4 : h - 2, c : c + 2] = pattern
        grid[h - 2 : h, c : c + 2] = int(color)
        c += step


def _can_place(occupied: np.ndarray, r0: int, c0: int, hh: int, ww: int) -> bool:
    h, w = occupied.shape
    r1 = min(h, r0 + hh + 1)
    c1 = min(w, c0 + ww + 1)
    r0p = max(0, r0 - 1)
    c0p = max(0, c0 - 1)
    return not bool(occupied[r0p:r1, c0p:c1].any())


def generate_input() -> np.ndarray:
    rows = random.randint(16, 24)
    cols = random.randint(17, 28)
    grid = np.full((rows, cols), BLACK, dtype=int)
    entries = [(frozenset(("U", "D")), random.choice([1, 2, 3, 4])), (frozenset(("R", "L")), random.choice([6, 8, 9]))]
    if entries[0][1] == entries[1][1]:
        entries[1] = (entries[1][0], 9)
    _draw_legend(grid, entries)

    occupied = np.zeros((rows - 4, cols), dtype=bool)
    obj_count = random.randint(3, 6)
    for _ in range(obj_count):
        vertical_signature = random.choice([True, False])
        if vertical_signature:
            ww = random.randint(1, 3)
            hh = random.randint(ww + 2, min(8, ww + 6))
        else:
            hh = random.randint(1, 3)
            ww = random.randint(hh + 2, min(8, hh + 6))
        placed = False
        for _attempt in range(100):
            if rows - 4 - hh <= 1 or cols - ww <= 1:
                break
            r0 = random.randint(0, rows - 5 - hh)
            c0 = random.randint(0, cols - ww)
            if not _can_place(occupied, r0, c0, hh, ww):
                continue
            grid[r0 : r0 + hh, c0 : c0 + ww] = GREY
            occupied[r0 : r0 + hh, c0 : c0 + ww] = True
            placed = True
            break
        if not placed:
            continue
    return grid


if __name__ == "__main__":
    sample = generate_input()
    print(sample)
    print(main(sample))
