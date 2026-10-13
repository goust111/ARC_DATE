import random

import numpy as np


FORWARD = 2  # red: top->bottom / left->right
REVERSE = 1  # blue: bottom->top / right->left
INDICATORS = (FORWARD, REVERSE)

GEN_ATTEMPTS = 400
HEIGHT_RANGE = (6, 13)
WIDTH_RANGE = (6, 13)
BG_CHOICES = (0, 3, 4, 5, 6, 7, 8, 9)
COLOR_POOL = tuple(range(10))


def _most_common_color(arr: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(arr, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _is_forward(marker: int) -> bool:
    marker = int(marker)
    assert marker in INDICATORS, f"unexpected direction marker: {marker}"
    return marker == FORWARD


def _propagate_line(src: np.ndarray, dst: np.ndarray, bg: int, forward: bool) -> None:
    length = int(src.shape[0])
    order = range(length) if forward else range(length - 1, -1, -1)
    carry = None
    for idx in order:
        value = int(src[idx])
        if value != bg:
            carry = value
        elif carry is not None and int(dst[idx]) == bg:
            dst[idx] = carry


def _solve_core(interior: np.ndarray, row_markers: np.ndarray, col_markers: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    row_filled = np.array(interior, copy=True)
    for r in range(interior.shape[0]):
        _propagate_line(interior[r], row_filled[r], bg, _is_forward(int(row_markers[r])))

    final = np.array(row_filled, copy=True)
    for c in range(interior.shape[1]):
        _propagate_line(row_filled[:, c], final[:, c], bg, _is_forward(int(col_markers[c])))

    return row_filled, final


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use edge direction markers to relay each seen color across rows first and then across columns.

    Input structure:
    - The top row and left column are direction indicators and are not part of the final answer.
    - Red indicators mean the positive scan direction: left-to-right for rows and top-to-bottom for columns.
    - Blue indicators mean the reverse scan direction: right-to-left for rows and bottom-to-top for columns.
    - Inside the bordered area, the background is dominant and a few non-background pixels of other colors act as relay seeds.

    Rule:
    - Remove the indicator row and column conceptually and work on the interior area.
    - For each interior row, read its left-edge indicator, scan in that direction, and once a non-background pixel is found, copy its color through later background cells until another non-background pixel replaces the relay color.
    - After finishing all rows, process each column on the row-filled grid using the top-edge indicators with the same relay rule.
    - Column propagation may only paint cells that are still background; original colored pixels and row-filled pixels are never overwritten.

    Key insight:
    - The edge colors do not label objects; they encode scan direction, and every encountered color acts like a baton that keeps filling forward until a new colored pixel takes over.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    top_markers = g[0, 1:]
    row_markers = g[1:, 0]
    interior = g[1:, 1:]

    assert top_markers.shape[0] == interior.shape[1]
    assert row_markers.shape[0] == interior.shape[0]
    assert set(map(int, np.unique(top_markers))) <= set(INDICATORS)
    assert set(map(int, np.unique(row_markers))) <= set(INDICATORS)

    bg = _most_common_color(interior)
    assert bg not in INDICATORS
    assert np.any(interior == bg), "interior must contain background"
    assert not np.any(np.isin(interior, INDICATORS)), "interior colors must differ from indicator colors"

    _, final = _solve_core(interior, row_markers, top_markers, bg)
    return final


def _mixed_markers(length: int, rng: random.Random) -> np.ndarray:
    markers = np.array([FORWARD if rng.random() < 0.5 else REVERSE for _ in range(length)], dtype=int)
    if length > 1 and len(set(map(int, markers))) == 1:
        flip_idx = rng.randrange(length)
        markers[flip_idx] = FORWARD if int(markers[flip_idx]) == REVERSE else REVERSE
    return markers


def _sample_row_positions(width: int, forward: bool, force_multi: bool, rng: random.Random) -> list[int]:
    max_k = min(3, width - 1)
    min_k = 2 if force_multi and max_k >= 2 else 1
    choices = list(range(min_k, max_k + 1))
    weights = [2 if k == 2 else 1 for k in choices]

    for _ in range(80):
        k = rng.choices(choices, weights=weights, k=1)[0]
        pool = list(range(1, width)) if forward else list(range(0, width - 1))
        positions = sorted(rng.sample(pool, k))

        line = np.zeros(width, dtype=int)
        line[positions] = 9
        out = line.copy()
        _propagate_line(line, out, 0, forward)
        if np.count_nonzero((line == 0) & (out != 0)) >= 1:
            return positions

    return []


def _seed_colors(count: int, palette: list[int], rng: random.Random, force_distinct: bool) -> list[int]:
    if count >= 2 and force_distinct and len(palette) >= 2:
        colors = rng.sample(palette, 2) + [rng.choice(palette) for _ in range(count - 2)]
        rng.shuffle(colors)
        return colors
    return [rng.choice(palette) for _ in range(count)]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*HEIGHT_RANGE)
    width = rng.randint(*WIDTH_RANGE)

    bg = rng.choice(BG_CHOICES)
    palette = [color for color in COLOR_POOL if color not in (bg, FORWARD, REVERSE)]
    rng.shuffle(palette)
    palette = palette[: rng.randint(3, min(5, len(palette)))]

    row_markers = _mixed_markers(height, rng)
    col_markers = _mixed_markers(width, rng)

    active_count = rng.randint(max(2, height // 3), max(2, height - 2))
    active_rows = sorted(rng.sample(range(height), active_count))
    if active_rows[0] >= height // 2 or active_rows[-1] < height // 2:
        return None

    interior = np.full((height, width), bg, dtype=int)
    relay_row = rng.choice(active_rows)

    for row in active_rows:
        positions = _sample_row_positions(width, _is_forward(int(row_markers[row])), row == relay_row, rng)
        if not positions:
            return None
        colors = _seed_colors(len(positions), palette, rng, row == relay_row)
        for col, color in zip(positions, colors):
            interior[row, col] = color

    row_filled, final = _solve_core(interior, row_markers, col_markers, bg)

    seed_count = int(np.count_nonzero(interior != bg))
    row_fills = int(np.count_nonzero((interior == bg) & (row_filled != bg)))
    col_fills = int(np.count_nonzero((row_filled == bg) & (final != bg)))
    bg_count = int(np.count_nonzero(interior == bg))
    non_bg_counts = [int(np.count_nonzero(interior == color)) for color in np.unique(interior) if int(color) != bg]
    final_bg = int(np.count_nonzero(final == bg))
    row_seed_counts = np.count_nonzero(interior != bg, axis=1)

    if seed_count < active_count:
        return None
    if row_fills < max(3, active_count):
        return None
    if col_fills < max(2, height - active_count):
        return None
    if not np.any(row_seed_counts >= 2):
        return None
    if len(set(map(int, interior[interior != bg].tolist()))) < 2:
        return None
    if bg_count <= max(non_bg_counts):
        return None
    if final_bg < max(2, width // 3):
        return None

    grid = np.full((height + 1, width + 1), bg, dtype=int)
    grid[0, 1:] = col_markers
    grid[1:, 0] = row_markers
    grid[1:, 1:] = interior
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
