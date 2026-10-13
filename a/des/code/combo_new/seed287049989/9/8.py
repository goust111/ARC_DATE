from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
PALETTE_SIZE_RANGE = (2, 5)
SOURCE_COUNT_RANGE = (3, 5)
GRID_SIZE_MAX = 18
MAX_RETRIES = 300


def _palette_priority(grid: np.ndarray) -> dict[int, int]:
    palette_cols = [int(c) for c in np.flatnonzero(grid[0] != BG)]
    assert len(palette_cols) >= 2, "palette row must contain at least two colors"
    assert all(b - a >= 2 for a, b in zip(palette_cols, palette_cols[1:])), "palette colors must be separated by black cells"

    palette = [int(grid[0, c]) for c in palette_cols]
    assert len(palette) == len(set(palette)), "palette colors must be distinct"
    return {color: rank for rank, color in enumerate(palette)}


def _sources(grid: np.ndarray) -> list[tuple[int, int, int]]:
    pts = []
    for r, c in np.argwhere(grid[1:] != BG):
        rr = int(r) + 1
        cc = int(c)
        pts.append((rr, cc, int(grid[rr, cc])))
    assert pts, "expected at least one signal source below the palette row"
    return pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill blank intersections of row/column broadcasts using the palette-row priority order.

    Input structure:
    - Black background.
    - The top row is a palette row: distinct single-pixel colors appear from left to right, with black gaps between them.
    - Below the palette row are isolated colored single pixels acting as signal sources; every source color appears in the palette row.

    Rule:
    - Each source broadcasts across its entire row and entire column, excluding the palette row.
    - Blank cells covered by only one source stay black.
    - Blank cells covered by two or more sources are filled with the highest-priority covered color, where priority is defined by the palette row from left (low) to right (high).
    - Source pixels and the palette row remain unchanged.

    Key insight:
    - The task is determined by cross-shaped row/column coverage, and color conflicts are resolved by the visible left-to-right palette order rather than by numeric color ids.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2, "expected a 2D grid with a palette row"

    priority = _palette_priority(g)
    sources = _sources(g)
    counts = Counter(color for _, _, color in sources)
    assert all(color in priority for color in counts), "every source color must appear in the palette row"
    assert all(count <= 2 for count in counts.values()), "each color may appear at most twice"

    row_colors: dict[int, list[int]] = {}
    col_colors: dict[int, list[int]] = {}
    for r, c, color in sources:
        row_colors.setdefault(r, []).append(color)
        col_colors.setdefault(c, []).append(color)

    out = g.copy()
    h, w = g.shape
    for r in range(1, h):
        row_cover = row_colors.get(r, [])
        for c in range(w):
            if g[r, c] != BG:
                continue
            cover = row_cover + col_colors.get(c, [])
            if len(cover) < 2:
                continue
            out[r, c] = max(cover, key=priority.__getitem__)
    return out


def _sample_spaced_positions(rng: random.Random, length: int, count: int, offset: int = 0) -> list[int]:
    assert length >= 2 * count - 1, "not enough room for separated positions"
    raw = sorted(rng.sample(range(length - count + 1), count))
    return [offset + raw[i] + i for i in range(count)]


def _sample_source_colors(rng: random.Random, palette: list[int], source_count: int) -> list[int]:
    min_used = max(2, (source_count + 1) // 2)
    max_used = min(len(palette), source_count - 1)
    used_count = rng.randint(min_used, max_used)
    used_colors = rng.sample(palette, used_count)

    counts = {color: 1 for color in used_colors}
    extra_slots = source_count - used_count
    for color in rng.sample(used_colors, extra_slots):
        counts[color] += 1

    out = []
    for color in used_colors:
        out.extend([color] * counts[color])
    rng.shuffle(out)
    return out


def _branch_stats(grid: np.ndarray) -> tuple[int, int]:
    sources = _sources(grid)
    row_owner = {r: color for r, _, color in sources}
    col_owner = {c: color for _, c, color in sources}
    locs = {(r, c) for r, c, _ in sources}

    same_color = 0
    diff_color = 0
    for r, row_color in row_owner.items():
        for c, col_color in col_owner.items():
            if (r, c) in locs:
                continue
            if row_color == col_color:
                same_color += 1
            else:
                diff_color += 1
    return same_color, diff_color


def _is_valid_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.any(grid[1] != BG):
        return False

    sources = _sources(grid)
    rows = [r for r, _, _ in sources]
    cols = [c for _, c, _ in sources]
    colors = [color for _, _, color in sources]
    if len(rows) != len(set(rows)) or len(cols) != len(set(cols)):
        return False
    if len(set(colors)) < 2 or max(Counter(colors).values()) < 2:
        return False

    same_color, diff_color = _branch_stats(grid)
    if same_color == 0 or diff_color == 0:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    palette_size = rng.randint(*PALETTE_SIZE_RANGE)
    max_sources = min(SOURCE_COUNT_RANGE[1], 2 * palette_size)
    source_count = rng.randint(SOURCE_COUNT_RANGE[0], max_sources)

    min_h = max(7, 2 * source_count + 1)
    min_w = max(7, 2 * source_count - 1, 2 * palette_size - 1)
    h = rng.randint(min_h, GRID_SIZE_MAX)
    w = rng.randint(min_w, GRID_SIZE_MAX)

    palette = rng.sample(COLOR_POOL, palette_size)
    palette_cols = _sample_spaced_positions(rng, w, palette_size)
    source_rows = _sample_spaced_positions(rng, h - 2, source_count, offset=2)
    source_cols = _sample_spaced_positions(rng, w, source_count)
    rng.shuffle(source_cols)
    source_colors = _sample_source_colors(rng, palette, source_count)

    grid = np.zeros((h, w), dtype=int)
    grid[0, palette_cols] = np.array(palette, dtype=int)
    for r, c, color in zip(source_rows, source_cols, source_colors):
        grid[r, c] = color

    if not _is_valid_generated_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_RETRIES} attempts")
