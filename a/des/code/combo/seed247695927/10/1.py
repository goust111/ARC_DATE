from __future__ import annotations

import random

import numpy as np


COLOR_OFFSETS: dict[int, tuple[int, int]] = {
    7: (-2, 0),  # orange -> up 2
    2: (0, -2),  # red -> left 2
    9: (2, 0),   # maroon -> down 2
    1: (0, 1),   # blue -> right 1
}

ALL_COLORS = tuple(range(10))
GRID_HEIGHT_RANGE = (9, 18)
GRID_WIDTH_RANGE = (9, 18)
FOREGROUND_COUNT_RANGE = (2, 12)
MAX_GENERATION_RETRIES = 400
ISOLATION_RADIUS = 1


def _dominant_background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    max_count = int(counts.max())
    winners = values[counts == max_count]
    assert winners.size == 1, "background color must be uniquely dominant"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move every foreground pixel by the fixed offset indicated by its color.

    Input structure:
    - The grid has a unique dominant background color.
    - The foreground is a sparse set of colored pixels.
    - Each foreground color is one of the trained motion colors: orange, red,
      maroon, or blue.

    Rule:
    - Clear the canvas to the background color.
    - Translate orange pixels up 2 cells, red pixels left 2 cells, maroon
      pixels down 2 cells, and blue pixels right 1 cell.
    - Preserve each moved pixel's original color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    bg = _dominant_background(g)
    out = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    for r, c in map(tuple, np.argwhere(g != bg)):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        assert color in COLOR_OFFSETS, "foreground colors must be known motion colors"
        dr, dc = COLOR_OFFSETS[color]
        nr = r + dr
        nc = c + dc
        assert 0 <= nr < h and 0 <= nc < w, "translated pixel must stay in bounds"
        assert (nr, nc) not in occupied, "translated pixels must not collide"
        out[nr, nc] = color
        occupied.add((nr, nc))

    return out


def _positive_partition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    return [b - a for a, b in zip([0] + cuts, cuts + [total])]


def _isolated(pos: tuple[int, int], placed: set[tuple[int, int]]) -> bool:
    r, c = pos
    for rr in range(r - ISOLATION_RADIUS, r + ISOLATION_RADIUS + 1):
        for cc in range(c - ISOLATION_RADIUS, c + ISOLATION_RADIUS + 1):
            if (rr, cc) in placed:
                return False
    return True


def _sample_design() -> tuple[int, int, int, list[int], int]:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    total = random.randint(*FOREGROUND_COUNT_RANGE)
    num_colors = random.randint(1, min(4, total))
    chosen_colors = random.sample(list(COLOR_OFFSETS), num_colors)
    counts = _positive_partition(total, num_colors)

    color_sequence: list[int] = []
    for color, count in zip(chosen_colors, counts):
        color_sequence.extend([color] * count)
    random.shuffle(color_sequence)

    bg = random.choice([c for c in ALL_COLORS if c not in COLOR_OFFSETS])
    return h, w, bg, color_sequence, num_colors


def _place_pixels(h: int, w: int, bg: int, color_sequence: list[int]) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    sources: set[tuple[int, int]] = set()
    destinations: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    for color in color_sequence:
        dr, dc = COLOR_OFFSETS[color]
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

        row_lo = max(0, -dr)
        row_hi = min(h, h - dr)
        col_lo = max(0, -dc)
        col_hi = min(w, w - dc)

        for r in range(row_lo, row_hi):
            for c in range(col_lo, col_hi):
                src = (r, c)
                dst = (r + dr, c + dc)
                if src in reserved or dst in reserved:
                    continue
                if not _isolated(src, sources):
                    continue
                if not _isolated(dst, destinations):
                    continue
                candidates.append((src, dst))

        if not candidates:
            return None

        src, dst = random.choice(candidates)
        grid[src] = color
        sources.add(src)
        destinations.add(dst)
        reserved.add(src)
        reserved.add(dst)

    return grid


def _is_valid_instance(grid: np.ndarray, min_colors: int) -> bool:
    bg = _dominant_background(grid)
    foreground = [int(grid[r, c]) for r, c in np.argwhere(grid != bg)]
    if len(set(foreground)) < min_colors:
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return out.shape == grid.shape


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        h, w, bg, color_sequence, min_colors = _sample_design()
        grid = _place_pixels(h, w, bg, color_sequence)
        if grid is None:
            continue
        if _is_valid_instance(grid, min_colors):
            return grid
    raise RuntimeError("failed to generate a valid input")
