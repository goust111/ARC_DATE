from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
LEGEND_COLORS = (1, 2, 3, 4)
GRID_MIN = 12
GRID_MAX = 18
MARGIN = 3
MIN_QUADRANT_AREA = 5
GEN_RETRIES = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _classify_cell(r: int, c: int, origin_r: int, origin_c: int) -> str:
    dr = int(r) - int(origin_r)
    dc = int(c) - int(origin_c)
    if abs(dr) == abs(dc):
        return "ray"
    if abs(dr) > abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def _paint_diagonal_rays(grid: np.ndarray, origin_r: int, origin_c: int, color: int = GREY) -> None:
    h, w = grid.shape
    grid[origin_r, origin_c] = int(color)
    for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
        r = int(origin_r) + dr
        c = int(origin_c) + dc
        while 0 <= r < h and 0 <= c < w:
            grid[r, c] = int(color)
            r += dr
            c += dc


def _flood_fill_black(grid: np.ndarray, start_r: int, start_c: int, color: int) -> None:
    h, w = grid.shape
    if not (0 <= start_r < h and 0 <= start_c < w):
        return
    if int(grid[start_r, start_c]) != BG:
        return
    q: deque[tuple[int, int]] = deque([(int(start_r), int(start_c))])
    grid[start_r, start_c] = int(color)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
                grid[nr, nc] = int(color)
                q.append((int(nr), int(nc)))


def _find_origin(grid: np.ndarray) -> tuple[int, int]:
    coords = np.argwhere(grid == GREY)
    assert coords.shape[0] == 1, "expected exactly one grey origin pixel"
    r, c = map(int, coords[0])
    return r, c


def _legend_map(grid: np.ndarray) -> dict[str, int]:
    legend = np.asarray(grid[:2, :2], dtype=int)
    values = tuple(int(v) for v in legend.flat)
    assert set(values) == set(LEGEND_COLORS), "top-left 2x2 legend must be a permutation of 1,2,3,4"
    return {
        "up": int(legend[0, 0]),
        "right": int(legend[0, 1]),
        "left": int(legend[1, 0]),
        "down": int(legend[1, 1]),
    }


def _quadrant_areas(shape: tuple[int, int], origin_r: int, origin_c: int) -> dict[str, int]:
    h, w = shape
    counts = {"up": 0, "right": 0, "left": 0, "down": 0}
    for r in range(h):
        for c in range(w):
            region = _classify_cell(r, c, origin_r, origin_c)
            if region != "ray":
                counts[region] += 1
    return counts


def _legend_profile(origin_r: int, origin_c: int) -> tuple[str, str, str, str]:
    return tuple(
        _classify_cell(r, c, origin_r, origin_c)
        for r, c in ((0, 0), (0, 1), (1, 0), (1, 1))
    )


def _independent_expected(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    origin_r, origin_c = _find_origin(g)
    mapping = _legend_map(g)
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            region = _classify_cell(r, c, origin_r, origin_c)
            out[r, c] = GREY if region == "ray" else int(mapping[region])
    return out


def _validate_instance(grid: np.ndarray, out: np.ndarray) -> None:
    g = np.asarray(grid, dtype=int)
    expected = _independent_expected(g)
    assert np.array_equal(out, expected), "solver output must match the quadrant-by-quadrant semantics"

    origin_r, origin_c = _find_origin(g)
    assert origin_r >= MARGIN and origin_c >= MARGIN
    assert origin_r <= g.shape[0] - MARGIN - 1 and origin_c <= g.shape[1] - MARGIN - 1

    areas = _quadrant_areas(g.shape, origin_r, origin_c)
    assert all(area > 0 for area in areas.values())
    assert len(set(areas.values())) >= 3, "generator should prefer non-degenerate quadrant area splits"

    non_bg = {(int(r), int(c)): int(g[r, c]) for r, c in np.argwhere(g != BG)}
    assert len(non_bg) == 5, "input should contain exactly the 2x2 legend plus one origin pixel"
    assert set(non_bg.values()) == {1, 2, 3, 4, 5}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend four grey diagonal rays from a single origin, then decode the top-left 2x2 legend into quadrant fill colors.

    Input structure:
    - Black background with a 2x2 legend block fixed at the top-left corner.
    - The legend contains the four distinct colors 1, 2, 3, and 4.
    - Exactly one isolated grey pixel appears away from the border and away from the legend.

    Rule:
    - Use the grey pixel as the origin and draw grey rays in the four diagonal directions until the grid edge.
    - The rays partition the remaining cells into four 4-connected regions: up, right, left, and down.
    - Decode legend positions as quadrant colors: top-left -> up, top-right -> right,
      bottom-left -> left, bottom-right -> down.
    - Fill each quadrant region with its decoded color. Ray cells, including the origin, stay grey.

    Key insight:
    - The 2x2 legend is not copied spatially; its local layout is a topological code for the four regions created by the diagonal cross.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30

    h, w = g.shape
    mapping = _legend_map(g)
    origin_r, origin_c = _find_origin(g)

    out = np.full((h, w), BG, dtype=int)
    _paint_diagonal_rays(out, origin_r, origin_c, GREY)

    seeds = {
        "up": (origin_r - 1, origin_c),
        "right": (origin_r, origin_c + 1),
        "left": (origin_r, origin_c - 1),
        "down": (origin_r + 1, origin_c),
    }
    for region, (seed_r, seed_c) in seeds.items():
        _flood_fill_black(out, seed_r, seed_c, int(mapping[region]))

    assert not np.any(out == BG), "rays should partition the full grid into four fillable regions"
    _validate_instance(g, out)
    return out


def _origin_candidates(h: int, w: int, branch: str) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r in range(MARGIN, h - MARGIN):
        for c in range(MARGIN, w - MARGIN):
            areas = _quadrant_areas((h, w), r, c)
            if min(areas.values()) < MIN_QUADRANT_AREA:
                continue
            if len(set(areas.values())) < 3:
                continue

            profile = _legend_profile(r, c)
            profile_set = set(profile)
            if branch == "split":
                if len(profile_set) < 2:
                    continue
                if "up" not in profile_set and "left" not in profile_set:
                    continue
            elif branch == "single":
                if profile_set not in ({"up"}, {"left"}):
                    continue
            else:
                raise ValueError(f"unknown branch: {branch}")
            candidates.append((r, c))
    return candidates


def _sample_shape_and_origin() -> tuple[int, int, int, int]:
    branch = random.choice(("split", "single"))
    for _ in range(120):
        h = random.randint(GRID_MIN, GRID_MAX)
        w = random.randint(GRID_MIN, GRID_MAX)
        candidates = _origin_candidates(h, w, branch)
        if candidates:
            origin_r, origin_c = random.choice(candidates)
            return h, w, origin_r, origin_c
    raise RuntimeError("failed to sample a valid origin for either legend-coverage branch")


def _gen_attempt() -> np.ndarray | None:
    h, w, origin_r, origin_c = _sample_shape_and_origin()
    grid = np.zeros((h, w), dtype=int)

    colors = list(LEGEND_COLORS)
    random.shuffle(colors)
    grid[0, 0] = colors[0]
    grid[0, 1] = colors[1]
    grid[1, 0] = colors[2]
    grid[1, 1] = colors[3]
    grid[origin_r, origin_c] = GREY

    try:
        out = main(grid)
    except AssertionError:
        return None

    _validate_instance(grid, out)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray legend instance")
