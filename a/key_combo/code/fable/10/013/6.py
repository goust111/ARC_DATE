from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLUE = 1

MISSING_TO_DELTA = {
    (0, 0): (-1, -1),
    (0, 1): (-1, 1),
    (1, 0): (1, -1),
    (1, 1): (1, 1),
}

NON_BLUE_COLORS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
GEN_RETRIES = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1, *sorted(separators), size]
    bands = [(a + 1, b) for a, b in zip(bounds, bounds[1:])]
    assert len(bands) == 3
    assert all(start < stop for start, stop in bands)
    return bands


def _find_separator_color(grid: np.ndarray, bg: int) -> tuple[int, list[int], list[int]]:
    h, w = grid.shape
    candidates: list[tuple[int, list[int], list[int]]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        rows = [r for r in range(h) if np.all(grid[r, :] == color)]
        cols = [c for c in range(w) if np.all(grid[:, c] == color)]
        if rows and cols:
            candidates.append((color, rows, cols))
    assert len(candidates) == 1, "expected one separator color"
    color, rows, cols = candidates[0]
    assert len(rows) == 2 and len(cols) == 2, "expected a 3x3 partition"
    return color, rows, cols


def _parse_region(region: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    coords = np.argwhere(region != bg)
    assert coords.shape[0] == 3, "each region must contain exactly three foreground cells"
    colors = set(map(int, region[region != bg]))
    assert len(colors) == 1, "each L sprite must be monochrome"
    color = colors.pop()

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert (int(r1 - r0 + 1), int(c1 - c0 + 1)) == (2, 2), "each sprite must fit a 2x2 box"

    sprite = region[r0 : r1 + 1, c0 : c1 + 1]
    missing = np.argwhere(sprite == bg)
    assert missing.shape == (1, 2), "expected one missing corner in the 2x2 box"
    return int(color), (int(missing[0, 0]), int(missing[0, 1]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each partition's L color into the diagonally adjacent partition indicated by its missing corner.

    Input structure:
    - One background color fills most cells.
    - One non-background color forms exactly two full separator rows and two full separator columns, splitting the board into a 3x3 table of rectangular partitions.
    - Each partition contains exactly one monochrome three-pixel L sprite occupying a 2x2 bounding box with one missing corner.

    Rule:
    - In each partition, read the missing corner of the 2x2 L sprite as a diagonal direction.
    - Send that sprite's color to the diagonally adjacent partition in that direction; projections that leave the 3x3 table are discarded.
    - Build a 3x3 output grid where a cell stays background if it receives nothing, takes the incoming color if it receives exactly one projection, and becomes blue if it receives two or more projections.

    Key insight:
    - The output records incoming diagonal projections between partitions, not the colors of the partitions themselves, so the separator table supplies the coordinate system and the L missing corner supplies the arrow.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    bg = _get_bg(g)
    _separator_color, sep_rows, sep_cols = _find_separator_color(g, bg)
    row_bands = _bands(g.shape[0], sep_rows)
    col_bands = _bands(g.shape[1], sep_cols)

    incoming: list[list[list[int]]] = [[[] for _ in range(3)] for _ in range(3)]
    for pr, (r0, r1) in enumerate(row_bands):
        for pc, (c0, c1) in enumerate(col_bands):
            region = g[r0:r1, c0:c1]
            color, missing = _parse_region(region, bg)
            dr, dc = MISSING_TO_DELTA[missing]
            tr, tc = pr + dr, pc + dc
            if 0 <= tr < 3 and 0 <= tc < 3:
                incoming[tr][tc].append(color)

    out = np.full((3, 3), bg, dtype=int)
    for r in range(3):
        for c in range(3):
            hit_count = len(incoming[r][c])
            if hit_count == 1:
                out[r, c] = incoming[r][c][0]
            elif hit_count >= 2:
                out[r, c] = BLUE
    return out


def _manual_output(
    bg: int,
    colors: list[list[int]],
    missings: list[list[tuple[int, int]]],
) -> tuple[np.ndarray, np.ndarray]:
    out = np.full((3, 3), bg, dtype=int)
    indegree = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            dr, dc = MISSING_TO_DELTA[missings[r][c]]
            tr, tc = r + dr, c + dc
            if 0 <= tr < 3 and 0 <= tc < 3:
                indegree[tr, tc] += 1
                if indegree[tr, tc] == 1:
                    out[tr, tc] = colors[r][c]
                else:
                    out[tr, tc] = BLUE
    return out, indegree


def _sample_partition_sizes() -> tuple[list[int], list[int]]:
    row_heights = [random.randint(3, 7) for _ in range(3)]
    col_widths = [random.randint(3, 7) for _ in range(3)]
    assert sum(row_heights) + 2 <= 30
    assert sum(col_widths) + 2 <= 30
    return row_heights, col_widths


def _sample_palette() -> tuple[int, int, list[int]]:
    bg = random.choice(NON_BLUE_COLORS)
    sep_choices = [c for c in NON_BLUE_COLORS if c != bg]
    sep = random.choice(sep_choices)
    sprite_choices = [c for c in NON_BLUE_COLORS if c not in {bg, sep}]
    return bg, sep, sprite_choices


def _sample_colors(sprite_choices: list[int]) -> list[list[int]]:
    pool_size = random.randint(4, min(7, len(sprite_choices)))
    pool = random.sample(sprite_choices, pool_size)
    flat = [random.choice(pool) for _ in range(9)]
    if len(set(flat)) < 4:
        return _sample_colors(sprite_choices)
    return [flat[0:3], flat[3:6], flat[6:9]]


def _sample_missings(mode: str) -> list[list[tuple[int, int]]]:
    choices = list(MISSING_TO_DELTA)
    for _ in range(200):
        flat = [random.choice(choices) for _ in range(9)]
        grid = [flat[0:3], flat[3:6], flat[6:9]]
        _, indegree = _manual_output(0, [[2] * 3 for _ in range(3)], grid)
        distinct_dirs = len(set(flat))
        offboard = sum(
            1
            for r in range(3)
            for c in range(3)
            if not (0 <= r + MISSING_TO_DELTA[grid[r][c]][0] < 3 and 0 <= c + MISSING_TO_DELTA[grid[r][c]][1] < 3)
        )
        singles = int(np.sum(indegree == 1))
        conflicts = int(np.sum(indegree >= 2))
        empties = int(np.sum(indegree == 0))
        max_in = int(indegree.max())
        if distinct_dirs < 3 or offboard == 0 or empties == 0:
            continue
        if mode == "no_conflict" and max_in == 1 and singles >= 3:
            return grid
        if mode == "conflict" and conflicts >= 1 and singles >= 1 and empties >= 1:
            return grid
        if mode == "mixed" and conflicts >= 1 and singles >= 2 and offboard >= 2:
            return grid
    raise RuntimeError(f"failed to sample missing-corner layout for mode={mode}")


def _render_input(
    bg: int,
    sep: int,
    row_heights: list[int],
    col_widths: list[int],
    colors: list[list[int]],
    missings: list[list[tuple[int, int]]],
) -> np.ndarray:
    h = sum(row_heights) + 2
    w = sum(col_widths) + 2
    grid = np.full((h, w), bg, dtype=int)

    sep_rows = [row_heights[0], row_heights[0] + 1 + row_heights[1]]
    sep_cols = [col_widths[0], col_widths[0] + 1 + col_widths[1]]
    grid[sep_rows, :] = sep
    grid[:, sep_cols] = sep

    row_starts = [0, row_heights[0] + 1, row_heights[0] + row_heights[1] + 2]
    col_starts = [0, col_widths[0] + 1, col_widths[0] + col_widths[1] + 2]

    for pr in range(3):
        for pc in range(3):
            rh = row_heights[pr]
            cw = col_widths[pc]
            top = row_starts[pr] + random.randint(0, rh - 2)
            left = col_starts[pc] + random.randint(0, cw - 2)
            color = colors[pr][pc]
            missing = missings[pr][pc]
            for rr in range(2):
                for cc in range(2):
                    if (rr, cc) != missing:
                        grid[top + rr, left + cc] = color
    return grid


def _validate_generated_case(
    grid: np.ndarray,
    expected: np.ndarray,
    indegree: np.ndarray,
    colors: list[list[int]],
    missings: list[list[tuple[int, int]]],
    mode: str,
) -> bool:
    if not np.array_equal(main(grid), expected):
        return False

    bg = _get_bg(grid)
    if bg == BLUE:
        return False

    own_colors = np.array(colors, dtype=int)
    single_mask = (indegree == 1) & (expected != BLUE)
    if not np.any(single_mask):
        return False
    if not np.any(expected[single_mask] != own_colors[single_mask]):
        return False
    if np.array_equal(expected, own_colors):
        return False

    if mode == "no_conflict" and np.any(expected == BLUE):
        return False
    if mode != "no_conflict" and not np.any(expected == BLUE):
        return False
    if not np.any(indegree == 0):
        return False

    parsed_dirs = []
    sep_color, sep_rows, sep_cols = _find_separator_color(grid, bg)
    if sep_color in {bg, BLUE}:
        return False
    row_bands = _bands(grid.shape[0], sep_rows)
    col_bands = _bands(grid.shape[1], sep_cols)
    for r in range(3):
        for c in range(3):
            region = grid[row_bands[r][0] : row_bands[r][1], col_bands[c][0] : col_bands[c][1]]
            color, missing = _parse_region(region, bg)
            if color != colors[r][c] or missing != missings[r][c]:
                return False
            parsed_dirs.append(missing)
    return len(set(parsed_dirs)) >= 3


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["no_conflict", "conflict", "mixed"])
    row_heights, col_widths = _sample_partition_sizes()
    bg, sep, sprite_choices = _sample_palette()
    colors = _sample_colors(sprite_choices)
    missings = _sample_missings(mode)
    expected, indegree = _manual_output(bg, colors, missings)

    visible_colors = set(map(int, expected[expected != bg]))
    if len(visible_colors) < 2:
        return None

    grid = _render_input(bg, sep, row_heights, col_widths, colors, missings)
    if not _validate_generated_case(grid, expected, indegree, colors, missings, mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
