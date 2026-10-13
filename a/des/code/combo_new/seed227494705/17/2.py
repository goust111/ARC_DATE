from collections import defaultdict, deque
import random

import numpy as np

BLACK, BLUE, RED, GREEN, YELLOW, GREY, MAGENTA, ORANGE, AZURE, MAROON = range(10)


def _bbox(cells):
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not mask[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(cells)
    return comps


def _pair_components(grid):
    by_color = defaultdict(list)
    for color in np.unique(grid):
        color = int(color)
        if color in (ORANGE, GREY, BLACK, AZURE):
            continue
        for comp in _components(grid == color):
            by_color[color].append(comp)
    return by_color


def _best_alignment(a, b):
    aset = set(a)
    bset = set(b)
    best = None
    for dr in range(-30, 31):
        for dc in range(-30, 31):
            moved = {(r + dr, c + dc) for r, c in bset}
            union = aset | moved
            inter = aset & moved
            if not inter:
                continue
            r0, c0, r1, c1 = _bbox(union)
            area = (r1 - r0 + 1) * (c1 - c0 + 1)
            if area != len(union):
                continue
            candidate = (area, union, inter)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return None if best is None else best[1:]


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay each same-color fragment pair into a solid rectangle and mark overlaps.

    Input structure: Orange background contains pairs of separated same-color
    connected fragments, plus a grey/black lower-right legend listing colors.
    Rule: For each color, translate one fragment without rotation so the two
    fragments jointly make a solid rectangle with some overlap.  Draw only that
    assembled rectangle in the output: singly covered cells keep the color and
    doubly covered cells become azure.  The legend is cleared to orange.
    Key insight: The visible fragments are two offset halves of one rectangle; the
    azure cells are found only after mentally completing the overlap.
    """
    out = np.full_like(grid, ORANGE)
    for color, comps in _pair_components(grid).items():
        if len(comps) != 2:
            continue
        aligned = _best_alignment(comps[0], comps[1])
        if aligned is None:
            continue
        union, inter = aligned
        for r, c in union:
            if 0 <= r < out.shape[0] and 0 <= c < out.shape[1]:
                out[r, c] = AZURE if (r, c) in inter else color
    return out


def _place(grid, mask, top, left, color):
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = color


def generate_input() -> np.ndarray:
    h, w = 30, 30
    grid = np.full((h, w), ORANGE, dtype=int)
    n = random.randint(2, 4)
    colors = random.sample([1, 2, 3, 4, 6, 8, 9], n)
    anchors = [(2, 2), (2, 11), (10, 2), (10, 11)]
    remotes = [(13, 18), (4, 18), (17, 4), (17, 12)]
    for i, color in enumerate(colors):
        rh, rw = random.randint(3, 5), random.randint(4, 6)
        cut = random.randint(1, rw - 2)
        a = np.zeros((rh, rw), dtype=bool)
        b = np.zeros((rh, rw), dtype=bool)
        a[:, : cut + 1] = True
        b[:, cut:] = True
        ar, ac = anchors[i]
        br, bc = remotes[i]
        _place(grid, a, ar, ac, color)
        _place(grid, b, br, bc, color)
    lr = h - n - 2
    lc = w - 5
    grid[lr : lr + n + 1, lc : lc + 4] = GREY
    for i, color in enumerate(colors):
        grid[lr + i, lc] = color
        grid[lr + i, lc + 1 : lc + 4] = BLACK
    return grid
