import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
LEGEND_COLORS = (RED, YELLOW, GREEN)

GEN_RETRIES = 400
PLACEMENT_RETRIES = 160
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue

        q = deque([(r, c)])
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while q:
            cr, cc = q.popleft()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(comp)

    return comps


def _find_legend(grid: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    for r in range(h):
        for c in range(w - 2):
            if tuple(map(int, grid[r, c : c + 3])) != LEGEND_COLORS:
                continue
            if r in (0, h - 1) or c == 0 or c + 2 == w - 1:
                return int(r), int(c)
    return None


def _extract_rectangles(grid: np.ndarray) -> list[dict[str, int]]:
    mask = (grid == GREY) | (grid == BLUE)
    rects: list[dict[str, int]] = []

    for comp in _components(mask):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all((patch == GREY) | (patch == BLUE)):
            continue
        if len(comp) != int(patch.size):
            continue
        marker_count = int(np.count_nonzero(patch == BLUE))
        if 1 <= marker_count <= 3:
            rects.append(
                {
                    "r0": int(r0),
                    "c0": int(c0),
                    "r1": int(r1),
                    "c1": int(c1),
                    "markers": marker_count,
                }
            )

    rects.sort(key=lambda rect: (rect["r0"], rect["c0"], rect["r1"], rect["c1"]))
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle by peeling and repainting as many border layers as its blue marker count.

    Input structure:
    - The background is black.
    - One edge of the grid contains a 3-cell horizontal legend whose colors, from left to right, are red, yellow, and green.
    - Elsewhere, one or more solid grey rectangles appear, and each rectangle contains 1 to 3 blue marker pixels somewhere inside it.

    Rule:
    - For each grey rectangle, count its blue markers to obtain N.
    - Repaint the rectangle's outermost layer red, the next layer yellow, and the third layer green, stopping after N layers.
    - Any remaining inner cells stay grey, and blue markers are removed by taking on the color of the layer or grey core they belong to.

    Key insight:
    - The blue markers are not kept as objects; only their count matters, because it controls how many nested border shells of the rectangle must be recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert _find_legend(g) is not None, "expected a border legend of red-yellow-green"

    out = g.copy()
    for rect in _extract_rectangles(g):
        r0, c0, r1, c1 = rect["r0"], rect["c0"], rect["r1"], rect["c1"]
        marker_count = rect["markers"]
        height = r1 - r0 + 1
        width = c1 - c0 + 1

        rr = np.arange(height)[:, None]
        cc = np.arange(width)[None, :]
        layer = np.minimum(np.minimum(rr, height - 1 - rr), np.minimum(cc, width - 1 - cc))

        patch = np.full((height, width), GREY, dtype=int)
        for depth, color in enumerate(LEGEND_COLORS[:marker_count]):
            patch[layer == depth] = int(color)
        out[r0 : r1 + 1, c0 : c1 + 1] = patch

    return out


def _marker_counts(rng: random.Random, n_rects: int) -> list[int]:
    colors = [1, 2, 3]
    if n_rects == 1:
        return [rng.choice(colors)]
    if n_rects == 2:
        picked = rng.sample(colors, 2)
        rng.shuffle(picked)
        return picked

    picked = colors[:]
    rng.shuffle(picked)
    while len(picked) < n_rects:
        picked.append(rng.choice(colors))
    rng.shuffle(picked)
    return picked


def _reserve(occ: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = occ.shape
    occ[max(0, r0 - pad) : min(h, r1 + pad + 1), max(0, c0 - pad) : min(w, c1 + pad + 1)] = True


def _place_legend(grid: np.ndarray, occ: np.ndarray, rng: random.Random) -> tuple[int, int]:
    h, w = grid.shape
    row = rng.choice([0, h - 1])
    start = rng.randint(0, w - 3)
    grid[row, start : start + 3] = LEGEND_COLORS
    _reserve(occ, row, start, row, start + 2, pad=1)
    return row, start


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = rng.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    grid = np.full((h, w), BLACK, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    _place_legend(grid, occ, rng)

    max_rects = 4 if min(h, w) >= 22 else 3
    n_rects = rng.randint(1, max_rects)
    marker_counts = _marker_counts(rng, n_rects)

    placed: list[int] = []
    for marker_count in sorted(marker_counts, reverse=True):
        min_side = max(4, 2 * marker_count + 1)
        max_height = min(h - 2, min_side + 4 + (1 if n_rects == 1 else 0))
        max_width = min(w - 2, min_side + 5 + (1 if n_rects <= 2 else 0))
        if max_height < min_side or max_width < min_side:
            return None

        success = False
        for _ in range(PLACEMENT_RETRIES):
            rect_h = rng.randint(min_side, max_height)
            rect_w = rng.randint(min_side, max_width)
            r0 = rng.randint(1, h - rect_h - 1)
            c0 = rng.randint(1, w - rect_w - 1)
            r1 = r0 + rect_h - 1
            c1 = c0 + rect_w - 1

            if occ[max(0, r0 - 1) : min(h, r1 + 2), max(0, c0 - 1) : min(w, c1 + 2)].any():
                continue

            interior = [(r, c) for r in range(r0 + 1, r1) for c in range(c0 + 1, c1)]
            if len(interior) < marker_count:
                continue

            success = True
            grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
            for mr, mc in rng.sample(interior, marker_count):
                grid[mr, mc] = BLUE
            _reserve(occ, r0, c0, r1, c1, pad=1)
            placed.append(marker_count)
            break

        if not success:
            return None

    if _find_legend(grid) is None:
        return None

    rects = _extract_rectangles(grid)
    if len(rects) != n_rects:
        return None
    if sorted(rect["markers"] for rect in rects) != sorted(placed):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
