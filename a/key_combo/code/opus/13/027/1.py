from __future__ import annotations

import random
from typing import Iterable

import numpy as np


MAX_GENERATION_ATTEMPTS = 600
GRID_MIN = 10
GRID_MAX = 16
COUNT_PATTERNS = (
    (1, 2, 4),
    (1, 3, 4),
    (1, 3, 5),
    (2, 3, 4),
    (2, 3, 5),
    (2, 4, 5),
    (3, 4, 5),
)
COLOR_POOL = tuple(range(1, 10))


def _role_colors(grid: np.ndarray) -> tuple[int, int, int]:
    nz = grid[grid != 0]
    colors, counts = np.unique(nz, return_counts=True)
    assert len(colors) == 3, "expected exactly three non-black colors"
    assert len(set(map(int, counts))) == 3, "non-black color frequencies must be unique"
    order = np.argsort(counts)
    return tuple(int(colors[idx]) for idx in order)


def _source_coords(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(grid == int(color))]


def _is_placeable(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = pos
    if pos in occupied:
        return False
    for rr, cc in occupied:
        if abs(rr - r) + abs(cc - c) == 1:
            return False
    return True


def _sample_spaced_values(low: int, high: int, count: int, min_gap: int = 2) -> list[int] | None:
    values = list(range(low, high + 1))
    if count == 0:
        return []
    if len(values) < count:
        return None
    for _ in range(200):
        chosen = sorted(random.sample(values, count))
        if all(chosen[i + 1] - chosen[i] >= min_gap for i in range(len(chosen) - 1)):
            return chosen
    return None


def _line_blocker_ok(pos: tuple[int, int], sources: Iterable[tuple[int, int]], shape: tuple[int, int]) -> bool:
    r, c = pos
    h, w = shape
    for sr, sc in sources:
        if r == sr and c != sc:
            if (c < sc and c > 0) or (c > sc and c < w - 1):
                return True
        if c == sc and r != sr:
            if (r < sr and r > 0) or (r > sr and r < h - 1):
                return True
    return False


def _pick_positions(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    need: int,
    predicate,
) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = [(r, c) for r in range(h) for c in range(w) if predicate(r, c)]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    blocked = set(occupied)
    for pos in candidates:
        if _is_placeable(pos, blocked):
            chosen.append(pos)
            blocked.add(pos)
            if len(chosen) == need:
                return chosen
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the rarest color as ray sources, then alternately fill the ray-split rectangles by band parity.

    Input structure:
    - The grid is a black background canvas with exactly three non-black colors.
    - Every non-black object is a singleton pixel; the three colors have distinct frequencies.
    - The rarest color marks ray sources, while the other two colors are preserved singleton pixels.

    Rule:
    - Rank the three non-black colors by frequency: rarest = ray color, middle = even-fill color, most frequent = odd-fill color.
    - From each ray-color pixel, extend in the four cardinal directions to the grid edge, recoloring only black cells to the ray color and skipping over any non-black singleton pixels.
    - Treat the source rows and source columns as band separators; for every remaining black cell, count how many source rows lie above it and how many source columns lie left of it.
    - Fill that black cell with the even-fill color when the band-index sum is even, otherwise with the odd-fill color.
    - Keep every original non-ray singleton and every ray pixel unchanged.

    Key insight:
    - The task is solved by frequency-based role assignment first; once the rarest color is identified, its rows and columns define a rectangular checkerboard of background bands for the other two colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert 0 < g.shape[0] <= 30 and 0 < g.shape[1] <= 30, "grid must satisfy ARC size limits"

    ray_color, even_color, odd_color = _role_colors(g)
    ray_sources = _source_coords(g, ray_color)
    assert ray_sources, "expected at least one ray source"

    ray_rows = sorted({r for r, _ in ray_sources})
    ray_cols = sorted({c for _, c in ray_sources})
    assert len(ray_rows) == len(ray_sources), "ray sources must not share rows"
    assert len(ray_cols) == len(ray_sources), "ray sources must not share columns"

    out = g.copy()
    h, w = out.shape
    for r, c in ray_sources:
        for cc in range(w):
            if out[r, cc] == 0:
                out[r, cc] = ray_color
        for rr in range(h):
            if out[rr, c] == 0:
                out[rr, c] = ray_color

    row_breaks = np.array(ray_rows, dtype=int)
    col_breaks = np.array(ray_cols, dtype=int)
    for r0, c0 in np.argwhere(out == 0):
        r = int(r0)
        c = int(c0)
        row_band = int(np.searchsorted(row_breaks, r, side="right"))
        col_band = int(np.searchsorted(col_breaks, c, side="right"))
        out[r, c] = even_color if (row_band + col_band) % 2 == 0 else odd_color

    return out


def _has_skip_case(grid: np.ndarray) -> bool:
    ray_color, _, _ = _role_colors(grid)
    sources = _source_coords(grid, ray_color)
    h, w = grid.shape
    for r0, c0 in np.argwhere(grid != 0):
        r = int(r0)
        c = int(c0)
        if int(grid[r, c]) == ray_color:
            continue
        if _line_blocker_ok((r, c), sources, (h, w)):
            return True
    return False


def _has_enough_regions(grid: np.ndarray) -> bool:
    ray_color, _, _ = _role_colors(grid)
    ray_sources = _source_coords(grid, ray_color)
    rows = sorted(r for r, _ in ray_sources)
    cols = sorted(c for _, c in ray_sources)
    return len(rows) >= 1 and len(cols) >= 1 and len(rows) == len(set(rows)) and len(cols) == len(set(cols))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    ray_count, even_count, odd_count = random.choice(COUNT_PATTERNS)
    ray_color, even_color, odd_color = random.sample(COLOR_POOL, 3)
    force_skip = random.random() < 0.55

    ray_rows = _sample_spaced_values(1, h - 2, ray_count, min_gap=2)
    ray_cols = _sample_spaced_values(1, w - 2, ray_count, min_gap=2)
    if ray_rows is None or ray_cols is None:
        return None

    permuted_cols = ray_cols[:]
    random.shuffle(permuted_cols)
    sources = list(zip(ray_rows, permuted_cols))
    occupied = set(sources)
    grid = np.zeros((h, w), dtype=int)
    for r, c in sources:
        grid[r, c] = ray_color

    remaining = even_count + odd_count
    line_rows = set(ray_rows)
    line_cols = set(ray_cols)

    if force_skip:
        blocker_need = random.randint(1, min(3, remaining))
        blockers = _pick_positions(
            (h, w),
            occupied,
            blocker_need,
            lambda r, c: (
                ((r in line_rows) ^ (c in line_cols))
                and (r, c) not in occupied
                and _line_blocker_ok((r, c), sources, (h, w))
            ),
        )
        if blockers is None:
            return None
        for pos in blockers:
            occupied.add(pos)
        filler_need = remaining - len(blockers)
        fillers = _pick_positions(
            (h, w),
            occupied,
            filler_need,
            lambda r, c: (r, c) not in occupied,
        )
        if fillers is None:
            return None
        positions = blockers + fillers
    else:
        positions = _pick_positions(
            (h, w),
            occupied,
            remaining,
            lambda r, c: (
                (r not in line_rows)
                and (c not in line_cols)
                and (r, c) not in occupied
            ),
        )
        if positions is None:
            return None

    random.shuffle(positions)
    even_positions = positions[:even_count]
    odd_positions = positions[even_count:]
    for r, c in even_positions:
        grid[r, c] = even_color
    for r, c in odd_positions:
        grid[r, c] = odd_color

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not _has_enough_regions(grid):
        return None
    if force_skip and not _has_skip_case(grid):
        return None
    if not force_skip and _has_skip_case(grid):
        return None
    if np.array_equal(out, grid):
        return None
    if len(np.unique(out)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC puzzle 13/027/1")
