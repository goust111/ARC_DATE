from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
HEIGHT_RANGE = (14, 22)
WIDTH_OPTIONS = tuple(range(15, 30, 2))
TOP_MARGIN_RANGE = (1, 3)
BOTTOM_MARGIN_RANGE = (1, 3)
MARKER_GAP = 3
EXTRA_MARKER_PROBABILITY = 0.6
GENERATION_ATTEMPTS = 500
MIN_CHANGED_CELLS = 16
MAX_CHANGED_FRACTION = 0.48


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _find_divider_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == DIVIDER)]
    assert len(cols) == 1, "expected exactly one full-height grey divider"
    return int(cols[0])


def _base_colors_by_frequency(grid: np.ndarray, divider_col: int) -> set[int]:
    del divider_col
    counts = Counter(int(v) for v in grid.ravel() if int(v) not in (BG, DIVIDER))
    assert len(counts) >= 4, "expected two base colors and two marker colors"

    ordered = sorted(counts.values(), reverse=True)
    assert ordered[1] > ordered[2], "base/marker frequency split is ambiguous"
    cutoff = ordered[1]
    base_colors = {color for color, count in counts.items() if count >= cutoff}
    assert len(base_colors) == 2, "expected exactly two large base colors"
    return base_colors


def _marker_pixels(
    grid: np.ndarray,
    divider_col: int,
    base_colors: set[int],
) -> list[tuple[int, int, int]]:
    left_mask = np.arange(grid.shape[1])[None, :] < int(divider_col)
    base_mask = np.isin(grid, list(base_colors))
    marker_mask = left_mask & ~base_mask & (grid != BG) & (grid != DIVIDER)
    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(marker_mask)]
    assert 2 <= len(markers) <= 3, "expected 2 or 3 left-side marker pixels"
    markers.sort(key=lambda item: (item[0], item[1]))
    return markers


def _base_to_marker_map(
    grid: np.ndarray,
    base_colors: set[int],
    markers: list[tuple[int, int, int]],
) -> dict[int, int]:
    base_to_marker: dict[int, int] = {}
    marker_to_base: dict[int, int] = {}

    for r, c, marker_color in markers:
        neighbor_counts: Counter[int] = Counter()
        for rr, cc in _neighbors8(r, c):
            if not _in_bounds(grid.shape, rr, cc):
                continue
            color = int(grid[rr, cc])
            if color in base_colors:
                neighbor_counts[color] += 1

        assert neighbor_counts, "marker must touch a base region"
        best = max(neighbor_counts.values())
        candidates = [color for color, count in neighbor_counts.items() if count == best]
        assert len(candidates) == 1, "marker must identify a unique base color"
        base_color = int(candidates[0])

        if base_color in base_to_marker:
            assert base_to_marker[base_color] == marker_color, "one base color cannot map to two markers"
        else:
            base_to_marker[base_color] = marker_color

        if marker_color in marker_to_base:
            assert marker_to_base[marker_color] == base_color, "one marker color cannot represent two bases"
        else:
            marker_to_base[marker_color] = base_color

    assert set(base_to_marker) == set(base_colors), "each base color needs a marker"
    assert len(marker_to_base) == 2, "expected exactly two marker colors"
    return base_to_marker


def _transform(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    divider_col = _find_divider_col(g)
    base_colors = _base_colors_by_frequency(g, divider_col)
    markers = _marker_pixels(g, divider_col, base_colors)
    base_to_marker = _base_to_marker_map(g, base_colors, markers)

    out = g.copy()
    rows, cols = np.indices(g.shape)
    for r, c, _marker in markers:
        mirror_c = 2 * divider_col - c
        assert 0 <= mirror_c < g.shape[1], "mirrored marker must remain in-bounds"
        ray_mask = (rows - cols == r - mirror_c) | (rows + cols == r + mirror_c)
        for base_color, marker_color in base_to_marker.items():
            out[ray_mask & (g == base_color)] = marker_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror each left-side marker across the divider, then paint diagonal rays using the crossed base colors' paired marker colors.

    Input structure:
    - A black background contains one full-height grey divider column that splits the canvas into left and right halves.
    - The non-black cells on both sides are filled by exactly two large base colors, with a different irregular partition on each side.
    - The left half contains 2 or 3 single-pixel markers; each marker sits inside one base-color region and reveals that base color's paired marker color.

    Rule:
    - Find the grey divider column and the two large base colors.
    - For every left-side marker at `(r, c)`, reflect it to `(r, 2k - c)` across divider column `k`, then activate both diagonals through that mirrored point.
    - Whenever an activated cell's original color is one of the two base colors, recolor it to that base color's paired marker color; leave black background, the grey divider, and original marker pixels unchanged.

    Key insight:
    - The firing point is not the marker itself but its mirror image, while the color written on each ray cell is determined by the cell's original base color rather than by the ray source alone.
    """
    return _transform(np.asarray(grid, dtype=int))


def _clamped_walk(length: int, lo: int, hi: int) -> list[int]:
    start = random.randint(lo, hi)
    values = [start]
    for _ in range(length - 1):
        step = random.choice((-2, -1, 0, 1, 2))
        values.append(max(lo, min(hi, values[-1] + step)))
    return values


def _side_pattern(height: int, width: int, base_a: int, base_b: int) -> np.ndarray | None:
    if height < 7 or width < 7:
        return None

    pattern = np.full((height, width), base_b, dtype=int)
    mode = random.choice(("vertical_wave", "horizontal_wave"))

    if mode == "vertical_wave":
        lo = 2
        hi = height - 3
        if lo > hi:
            return None
        boundary = _clamped_walk(width, lo, hi)
        for c, b in enumerate(boundary):
            pattern[: b + 1, c] = base_a
    else:
        lo = 2
        hi = width - 3
        if lo > hi:
            return None
        boundary = _clamped_walk(height, lo, hi)
        for r, b in enumerate(boundary):
            pattern[r, : b + 1] = base_a

    if random.random() < 0.5:
        pattern = np.where(pattern == base_a, base_b, base_a)
    return pattern


def _interior_cells(
    grid: np.ndarray,
    row0: int,
    row1: int,
    col0: int,
    col1: int,
    color: int,
) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(row0 + 1, row1):
        for c in range(col0 + 1, col1):
            if int(grid[r, c]) != int(color):
                continue
            if np.all(grid[r - 1 : r + 2, c - 1 : c + 2] == color):
                cells.append((r, c))
    return cells


def _pick_spaced(
    cells: list[tuple[int, int]],
    count: int,
    used: list[tuple[int, int]],
) -> list[tuple[int, int]] | None:
    available = list(cells)
    random.shuffle(available)
    chosen: list[tuple[int, int]] = []
    for cell in available:
        if all(max(abs(cell[0] - r), abs(cell[1] - c)) >= MARKER_GAP for r, c in used):
            chosen.append(cell)
            used.append(cell)
            if len(chosen) == count:
                return chosen
    return None


def _build_input() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.choice(WIDTH_OPTIONS)
    divider_col = width // 2
    side_width = divider_col

    top_margin = random.randint(*TOP_MARGIN_RANGE)
    bottom_margin = random.randint(*BOTTOM_MARGIN_RANGE)
    row0 = top_margin
    row1 = height - bottom_margin - 1
    active_height = row1 - row0 + 1
    if active_height < 8:
        return None

    colors = list(PALETTE)
    random.shuffle(colors)
    base_colors = [int(colors[0]), int(colors[1])]
    marker_colors = [int(colors[2]), int(colors[3])]
    base_to_marker = {
        base_colors[0]: marker_colors[0],
        base_colors[1]: marker_colors[1],
    }

    left = _side_pattern(active_height, side_width, base_colors[0], base_colors[1])
    right = _side_pattern(active_height, side_width, base_colors[0], base_colors[1])
    if left is None or right is None:
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[:, divider_col] = DIVIDER
    grid[row0 : row1 + 1, :divider_col] = left
    grid[row0 : row1 + 1, divider_col + 1 :] = right

    candidates = {
        base_colors[0]: _interior_cells(grid, row0, row1, 0, divider_col - 1, base_colors[0]),
        base_colors[1]: _interior_cells(grid, row0, row1, 0, divider_col - 1, base_colors[1]),
    }
    if not candidates[base_colors[0]] or not candidates[base_colors[1]]:
        return None

    marker_counts = {base_colors[0]: 1, base_colors[1]: 1}
    if random.random() < EXTRA_MARKER_PROBABILITY:
        extra_base = random.choice(base_colors)
        marker_counts[extra_base] += 1

    used: list[tuple[int, int]] = []
    order = base_colors[:]
    random.shuffle(order)
    for base_color in order:
        chosen = _pick_spaced(candidates[base_color], marker_counts[base_color], used)
        if chosen is None:
            return None
        for r, c in chosen:
            grid[r, c] = base_to_marker[base_color]

    try:
        out = _transform(grid)
    except AssertionError:
        return None

    changed = out != grid
    changed_count = int(changed.sum())
    if changed_count < MIN_CHANGED_CELLS:
        return None
    if changed_count > int(MAX_CHANGED_FRACTION * grid.size):
        return None
    if not np.any(changed[:, :divider_col]) or not np.any(changed[:, divider_col + 1 :]):
        return None

    for marker_color in marker_colors:
        if int(np.count_nonzero(out == marker_color)) <= int(np.count_nonzero(grid == marker_color)):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _build_input()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirrored-ray input")
