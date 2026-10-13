from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GRAY = 5

BG_CHOICES = [0, 4, 6, 7, 8, 9]
GRID_H_RANGE = (8, 15)
GRID_W_RANGE = (8, 15)
MAX_SEEDS = 4
GEN_RETRIES = 400

_NEXT_CROSS = False


def _most_common_color(values: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(values).reshape(-1))
    ranked = counts.most_common(2)
    assert ranked, "expected a nonempty grid"
    if len(ranked) > 1:
        assert ranked[0][1] > ranked[1][1], "expected a unique background color"
    return int(ranked[0][0])


def _edge_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _row_direction(g: np.ndarray, row: int) -> int:
    left = int(g[row, 0]) == RED
    right = int(g[row, -1]) == RED
    assert left ^ right, "each seed row must have exactly one horizontal border marker"
    return 1 if left else -1


def _col_direction(g: np.ndarray, col: int) -> int:
    top = int(g[0, col]) == RED
    bottom = int(g[-1, col]) == RED
    assert top ^ bottom, "each seed column must have exactly one vertical border marker"
    return 1 if top else -1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use border arrows to grow rays from blue seeds; crossing rays from different seeds turn green.

    Input structure
    - bg = most common color.
    - A one-pixel gray border surrounds the grid.
    - Some border cells are red direction markers:
      top -> down, bottom -> up, left -> right, right -> left.
    - Inside the border there are blue single-pixel seeds.

    Rule
    - For each seed, its row marker picks the horizontal direction and its
      column marker picks the vertical direction.
    - Draw a blue ray from the seed horizontally until just before the border.
    - Draw a blue ray from the seed vertically until just before the border.
    - Any non-seed cell covered by rays from two or more different seeds turns green.
    - Seeds stay blue and the border stays unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 5 and w >= 5, "grid is too small for a bordered interior"

    bg = _most_common_color(g)
    edge = _edge_mask(g.shape)
    assert set(map(int, np.unique(g[edge]))) <= {GRAY, RED}, "expected only gray/red cells on the border"
    assert int((g[edge] == RED).sum()) > 0, "expected at least one border marker"

    interior = g[1:-1, 1:-1]
    assert set(map(int, np.unique(interior))) <= {bg, BLUE}, "expected only background and blue seeds inside"

    seed_coords = [tuple(map(int, rc)) for rc in np.argwhere(g == BLUE)]
    assert seed_coords, "expected at least one seed"
    assert all(0 < r < h - 1 and 0 < c < w - 1 for r, c in seed_coords), "seeds must be inside the border"

    row_dirs = {r: _row_direction(g, r) for r in {r for r, _c in seed_coords}}
    col_dirs = {c: _col_direction(g, c) for c in {c for _r, c in seed_coords}}

    seed_set = set(seed_coords)
    covered_by: dict[tuple[int, int], set[int]] = defaultdict(set)

    for seed_id, (r, c) in enumerate(seed_coords):
        dc = row_dirs[r]
        cc = c + dc
        while 0 < cc < w - 1:
            covered_by[(r, cc)].add(seed_id)
            cc += dc

        dr = col_dirs[c]
        rr = r + dr
        while 0 < rr < h - 1:
            covered_by[(rr, c)].add(seed_id)
            rr += dr

    out = g.copy()
    for (r, c), owners in covered_by.items():
        if (r, c) in seed_set:
            continue
        out[r, c] = GREEN if len(owners) >= 2 else BLUE

    return out


def _safe_positions(size: int) -> list[int]:
    positions = list(range(2, size - 2))
    assert len(positions) >= 2, "need room for seeds away from the border"
    return positions


def _pick_positions(size: int, count: int) -> list[int]:
    positions = _safe_positions(size)
    want_gap_two = len(positions) >= 2 * count - 1 and random.random() < 0.5
    for _ in range(80):
        picked = sorted(random.sample(positions, count))
        if not want_gap_two or all(b - a >= 2 for a, b in zip(picked, picked[1:])):
            return picked
    raise RuntimeError("failed to sample well-spaced positions")


def _choose_directions(want_cross: bool, increasing: bool) -> tuple[int, int]:
    if increasing:
        options = [(1, -1), (-1, 1)] if want_cross else [(1, 1), (-1, -1)]
    else:
        options = [(1, 1), (-1, -1)] if want_cross else [(1, -1), (-1, 1)]
    return random.choice(options)


def _build_case(
    h: int,
    w: int,
    bg: int,
    rows: list[int],
    cols: list[int],
    increasing: bool,
    want_cross: bool,
) -> np.ndarray:
    g = np.full((h, w), bg, dtype=int)
    g[0, :] = GRAY
    g[-1, :] = GRAY
    g[:, 0] = GRAY
    g[:, -1] = GRAY

    hdir, vdir = _choose_directions(want_cross, increasing)

    row_marker_col = 0 if hdir == 1 else w - 1
    col_marker_row = 0 if vdir == 1 else h - 1
    for r in rows:
        g[r, row_marker_col] = RED
    for c in cols:
        g[col_marker_row, c] = RED

    seed_cols = cols if increasing else list(reversed(cols))
    for r, c in zip(rows, seed_cols):
        g[r, c] = BLUE

    return g


def _is_valid_case(grid: np.ndarray, want_cross: bool) -> bool:
    try:
        out = main(grid)
    except Exception:
        return False

    bg = _most_common_color(grid)
    interior = out[1:-1, 1:-1]
    green_count = int((out == GREEN).sum())
    changed = int((out != grid).sum())
    colored_inside = int((interior != bg).sum())
    interior_area = int(interior.size)

    if changed <= int((grid == BLUE).sum()):
        return False
    if want_cross != (green_count > 0):
        return False
    if colored_inside < 6 or colored_inside > int(0.7 * interior_area):
        return False
    if not np.array_equal(out[0, :], grid[0, :]):
        return False
    if not np.array_equal(out[-1, :], grid[-1, :]):
        return False
    if not np.array_equal(out[:, 0], grid[:, 0]):
        return False
    if not np.array_equal(out[:, -1], grid[:, -1]):
        return False
    return True


def generate_input() -> np.ndarray:
    global _NEXT_CROSS

    want_cross = _NEXT_CROSS
    _NEXT_CROSS = not _NEXT_CROSS

    for _ in range(GEN_RETRIES):
        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        bg = random.choice(BG_CHOICES)

        max_seed_count = min(MAX_SEEDS, len(_safe_positions(h)), len(_safe_positions(w)))
        if max_seed_count < 2:
            continue
        seed_count = random.randint(2, max_seed_count)

        rows = _pick_positions(h, seed_count)
        cols = _pick_positions(w, seed_count)
        increasing = bool(random.getrandbits(1))

        grid = _build_case(
            h=h,
            w=w,
            bg=bg,
            rows=rows,
            cols=cols,
            increasing=increasing,
            want_cross=want_cross,
        )

        if _is_valid_case(grid, want_cross):
            return grid

    raise RuntimeError("failed to generate a valid bordered-ray puzzle")
