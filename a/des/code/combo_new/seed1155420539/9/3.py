from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATE_ATTEMPTS = 400


def _horizontal_segment(
    row: int,
    c0: int,
    c1: int,
    grey_cols: set[int],
) -> tuple[list[tuple[int, int]], int]:
    step = 1 if c1 >= c0 else -1
    points = [(row, c) for c in range(c0, c1 + step, step) if c not in grey_cols]
    lo, hi = sorted((c0, c1))
    crossings = sum(1 for c in range(lo + 1, hi) if c in grey_cols)
    return points, crossings


def _vertical_segment(
    col: int,
    r0: int,
    r1: int,
    grey_rows: set[int],
) -> tuple[list[tuple[int, int]], int]:
    step = 1 if r1 >= r0 else -1
    points = [(r, col) for r in range(r0, r1 + step, step) if r not in grey_rows]
    lo, hi = sorted((r0, r1))
    crossings = sum(1 for r in range(lo + 1, hi) if r in grey_rows)
    return points, crossings


def _candidate_path(
    p0: tuple[int, int],
    p1: tuple[int, int],
    grey_rows: set[int],
    grey_cols: set[int],
    horizontal_first: bool,
) -> tuple[list[tuple[int, int]], int]:
    r0, c0 = p0
    r1, c1 = p1
    if horizontal_first:
        first, cross0 = _horizontal_segment(r0, c0, c1, grey_cols)
        second, cross1 = _vertical_segment(c1, r0, r1, grey_rows)
    else:
        first, cross0 = _vertical_segment(c0, r0, r1, grey_rows)
        second, cross1 = _horizontal_segment(r1, c0, c1, grey_cols)
    path = first + second[1:]
    return path, cross0 + cross1


def _chosen_path(
    p0: tuple[int, int],
    p1: tuple[int, int],
    grey_rows: set[int],
    grey_cols: set[int],
) -> tuple[list[tuple[int, int]], int]:
    h_path, h_cross = _candidate_path(p0, p1, grey_rows, grey_cols, horizontal_first=True)
    v_path, v_cross = _candidate_path(p0, p1, grey_rows, grey_cols, horizontal_first=False)
    if h_cross <= v_cross:
        return h_path, h_cross
    return v_path, v_cross


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect each same-colored endpoint pair with an orthogonal rope that skips across full grey barrier lines.

    Input structure:
    - Black background fills the grid.
    - One-cell-wide Grey rows and/or columns span the full grid width or full grid height.
    - Every other color appears exactly twice as isolated single-pixel endpoints placed off the Grey lines.

    Rule:
    - Pair the two pixels of each non-Black, non-Grey color.
    - For each pair, compare the two shortest L-shaped routes: horizontal-then-vertical or vertical-then-horizontal; choose the route with fewer Grey-line crossings, breaking ties in favor of the horizontal-first route.
    - Recolor the route's background cells with the pair color, but never draw on Grey cells: when the route reaches a Grey row or column, skip that one-cell barrier and continue on the opposite side. Keep the Grey bands unchanged.

    Key insight:
    - The task is endpoint pairing plus ordered Manhattan routing: the full-span Grey lines are not painted through, they are treated as removable one-cell gaps inside the chosen L-shaped connection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    grey_rows = {int(r) for r in range(g.shape[0]) if bool(np.all(g[r, :] == GREY))}
    grey_cols = {int(c) for c in range(g.shape[1]) if bool(np.all(g[:, c] == GREY))}
    assert grey_rows or grey_cols

    out = g.copy()
    colors = [int(color) for color in np.unique(g) if int(color) not in {BG, GREY}]
    for color in sorted(colors):
        pts = [tuple(map(int, p)) for p in np.argwhere(g == color)]
        assert len(pts) == 2, "each non-background, non-grey color must appear exactly twice"
        path, _ = _chosen_path(pts[0], pts[1], grey_rows, grey_cols)
        for r, c in path:
            if out[r, c] != GREY:
                out[r, c] = color
    return out


def _sample_bands(size: int, max_count: int) -> list[int]:
    target = random.randint(0, max_count)
    candidates = list(range(1, size - 1))
    random.shuffle(candidates)
    chosen: list[int] = []
    for pos in candidates:
        if all(abs(pos - prev) >= 3 for prev in chosen):
            chosen.append(pos)
            if len(chosen) == target:
                break
    return sorted(chosen)


def _nonband_indices(size: int, bands: Iterable[int]) -> list[int]:
    band_set = set(int(x) for x in bands)
    return [i for i in range(size) if i not in band_set]


def _path_is_clear(
    path: list[tuple[int, int]],
    reserved: set[tuple[int, int]],
    endpoints: tuple[tuple[int, int], tuple[int, int]],
) -> bool:
    p0, p1 = endpoints
    clean_path = set(path)
    if len(clean_path) <= 2:
        return False
    if any(cell in reserved for cell in clean_path):
        return False
    if p0 in reserved or p1 in reserved:
        return False
    return True


def _sample_pair(
    kind: str,
    free_rows: list[int],
    free_cols: list[int],
    grey_rows: set[int],
    grey_cols: set[int],
    reserved: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]], int] | None:
    for _ in range(250):
        if kind.startswith("same_row"):
            if len(free_cols) < 2:
                return None
            r = random.choice(free_rows)
            c0, c1 = random.sample(free_cols, 2)
            p0 = (r, c0)
            p1 = (r, c1)
        elif kind.startswith("same_col"):
            if len(free_rows) < 2:
                return None
            c = random.choice(free_cols)
            r0, r1 = random.sample(free_rows, 2)
            p0 = (r0, c)
            p1 = (r1, c)
        else:
            if len(free_rows) < 2 or len(free_cols) < 2:
                return None
            r0, r1 = random.sample(free_rows, 2)
            c0, c1 = random.sample(free_cols, 2)
            p0 = (r0, c0)
            p1 = (r1, c1)

        path, crossings = _chosen_path(p0, p1, grey_rows, grey_cols)
        if kind.endswith("cross") and crossings == 0:
            continue
        if kind.endswith("local") and crossings != 0:
            continue
        if abs(p0[0] - p1[0]) + abs(p0[1] - p1[1]) < 2:
            continue
        if not _path_is_clear(path, reserved, (p0, p1)):
            continue
        return p0, p1, path, crossings
    return None


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(11, 20)
    width = random.randint(11, 20)

    grey_row_list = _sample_bands(height, max_count=2)
    grey_col_list = _sample_bands(width, max_count=2)
    if not grey_row_list and not grey_col_list:
        if random.random() < 0.5:
            grey_row_list = _sample_bands(height, max_count=1) or [random.randint(1, height - 2)]
        else:
            grey_col_list = _sample_bands(width, max_count=1) or [random.randint(1, width - 2)]

    grey_rows = set(grey_row_list)
    grey_cols = set(grey_col_list)
    free_rows = _nonband_indices(height, grey_rows)
    free_cols = _nonband_indices(width, grey_cols)
    if len(free_rows) < 2 or len(free_cols) < 2:
        return None

    pair_cap = min(5, len(PALETTE), max(2, (len(free_rows) * len(free_cols)) // 18))
    pair_count = random.randint(2, pair_cap)

    kinds: list[str] = []
    kinds.append("diag_cross" if (grey_rows or grey_cols) and random.random() < 0.7 else "diag_local")
    if grey_cols:
        kinds.append(random.choice(["same_row_cross", "same_row_local"]))
    else:
        kinds.append("same_row_local")
    if grey_rows:
        kinds.append(random.choice(["same_col_cross", "same_col_local"]))
    else:
        kinds.append("same_col_local")

    pool = ["diag_local", "diag_cross", "same_row_local", "same_col_local"]
    if grey_cols:
        pool.append("same_row_cross")
    if grey_rows:
        pool.append("same_col_cross")
    while len(kinds) < pair_count:
        kinds.append(random.choice(pool))
    random.shuffle(kinds)
    kinds = kinds[:pair_count]

    reserved: set[tuple[int, int]] = set()
    placements: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    used_colors = random.sample(PALETTE, pair_count)

    for color, kind in zip(used_colors, kinds):
        sampled = _sample_pair(kind, free_rows, free_cols, grey_rows, grey_cols, reserved)
        if sampled is None:
            fallback_pool = list(dict.fromkeys(pool + kinds))
            for fallback_kind in fallback_pool:
                sampled = _sample_pair(fallback_kind, free_rows, free_cols, grey_rows, grey_cols, reserved)
                if sampled is not None:
                    break
        if sampled is None:
            return None
        p0, p1, path, _ = sampled
        reserved.update(path)
        placements.append((color, p0, p1))

    if len(placements) < 2:
        return None
    if not any(p0[0] != p1[0] and p0[1] != p1[1] for _, p0, p1 in placements):
        return None
    if (grey_rows or grey_cols) and not any(_chosen_path(p0, p1, grey_rows, grey_cols)[1] > 0 for _, p0, p1 in placements):
        return None

    grid = np.full((height, width), BG, dtype=int)
    for r in grey_rows:
        grid[r, :] = GREY
    for c in grey_cols:
        grid[:, c] = GREY
    for color, p0, p1 in placements:
        grid[p0] = color
        grid[p1] = color

    try:
        _ = main(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_ATTEMPTS} attempts")
