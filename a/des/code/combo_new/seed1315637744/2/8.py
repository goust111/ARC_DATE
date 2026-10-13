from __future__ import annotations

import random

import numpy as np


SEP = 7
BLACK = 0
COLOR_POOL = tuple(c for c in range(10) if c not in (BLACK, SEP))
COMBO_ORDER = (("h", 1), ("h", 2), ("v", 1), ("v", 2))
LAYOUT_SHAPE = (2, 2)
SEPARATOR_WIDTH = 1
CELL_SIZE_MIN = 6
CELL_SIZE_MAX = 14
MAX_GENERATION_ATTEMPTS = 300

SHAPE_LIBRARY = (
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 0, 0], [0, 1, 0, 0]], dtype=bool),
)


def _bands(size: int, separators: list[int]) -> list[tuple[int, int]]:
    bands: list[tuple[int, int]] = []
    start = 0
    for cut in separators:
        if start < cut:
            bands.append((start, cut - 1))
        start = cut + 1
    if start < size:
        bands.append((start, size - 1))
    return bands


def _stripe_pattern(
    shape: tuple[int, int], color_a: int, color_b: int, orient: str, period: int, phase: int
) -> np.ndarray:
    h, w = map(int, shape)
    if orient == "h":
        idx = np.arange(h, dtype=int)[:, None]
    else:
        idx = np.arange(w, dtype=int)[None, :]
    mask = ((idx // int(period)) + int(phase)) % 2 == 1
    out = np.full((h, w), int(color_a), dtype=int)
    out[np.broadcast_to(mask, (h, w))] = int(color_b)
    return out


def _split_regions(g: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = map(int, g.shape)
    row_colors = {int(g[r, 0]) for r in range(h) if np.all(g[r, :] == g[r, 0])}
    col_colors = {int(g[0, c]) for c in range(w) if np.all(g[:, c] == g[0, c])}
    sep_colors = row_colors & col_colors
    assert len(sep_colors) == 1, "expected one separator color shared by full rows and columns"

    sep = next(iter(sep_colors))
    sep_rows = [r for r in range(h) if np.all(g[r, :] == sep)]
    sep_cols = [c for c in range(w) if np.all(g[:, c] == sep)]
    assert sep_rows and sep_cols, "expected both horizontal and vertical separators"

    row_bands = _bands(h, sep_rows)
    col_bands = _bands(w, sep_cols)
    assert len(row_bands) >= 2 and len(col_bands) >= 2

    heights = {r1 - r0 + 1 for r0, r1 in row_bands}
    widths = {c1 - c0 + 1 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "regions must all have the same size"
    return row_bands, col_bands


def _analyze_region(cell: np.ndarray) -> tuple[int, str, int]:
    colors, counts = np.unique(cell, return_counts=True)
    order = np.argsort(-counts)
    assert colors.size >= 3, "each region must contain two stripe colors and one object color"

    bg_colors = tuple(int(colors[idx]) for idx in order[:2])
    object_colors = {int(color) for color in colors if int(color) not in bg_colors}
    assert len(object_colors) == 1, "expected exactly one non-background object color"
    object_color = next(iter(object_colors))
    object_mask = cell == int(object_color)
    assert bool(object_mask.any())

    matches: list[tuple[str, int]] = []
    for orient, period in COMBO_ORDER:
        for phase in (0, 1):
            expected = _stripe_pattern(cell.shape, bg_colors[0], bg_colors[1], orient, period, phase)
            if np.array_equal(cell[~object_mask], expected[~object_mask]):
                matches.append((orient, int(period)))
                break

    unique_matches = sorted(set(matches), key=COMBO_ORDER.index)
    assert len(unique_matches) == 1, "region texture must identify one orientation-period pair"
    orient, period = unique_matches[0]
    return int(object_color), orient, int(period)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Order the region colors by the stripe texture behind their identical objects.

    Input structure:
    - Orange full rows and columns form a separator lattice that splits the grid into equal rectangular regions.
    - Every region contains the same small object shape drawn in its own object color.
    - The visible background of each region is a black-and-light striped texture, with stripes either horizontal or vertical and with period 1 or 2.

    Rule:
    - Split the grid into regions using the full separator rows and columns.
    - For each region, ignore the object shape and recover the background stripe attributes: orientation and period.
    - Output a single-column list of the object colors, ordered first by horizontal before vertical stripes, then by period 1 before period 2 within the same orientation.

    Key insight:
    - The object colors are not ordered by position, size, or color value; the hidden key is the faint stripe texture underneath each object.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    row_bands, col_bands = _split_regions(g)
    items: list[tuple[int, int, int]] = []
    seen_keys: set[tuple[str, int]] = set()
    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            color, orient, period = _analyze_region(cell)
            key = (orient, int(period))
            assert key not in seen_keys, "texture combinations must be unique across regions"
            seen_keys.add(key)
            items.append((0 if orient == "h" else 1, int(period), int(color)))

    items.sort()
    return np.array([[color] for _axis, _period, color in items], dtype=int)


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated), np.flipud(rotated)):
            key = (candidate.shape, candidate.tobytes())
            variants[key] = candidate.astype(bool)
    return list(variants.values())


def _sample_shape(cell_h: int, cell_w: int) -> np.ndarray:
    candidates = [shape for base in SHAPE_LIBRARY for shape in _all_transforms(base)]
    fitting = [shape for shape in candidates if shape.shape[0] <= cell_h - 2 and shape.shape[1] <= cell_w - 2]
    assert fitting, "cell must leave a background margin around the object"
    return np.array(random.choice(fitting), copy=True)


def _place_object(cell: np.ndarray, shape: np.ndarray, color: int) -> None:
    cell_h, cell_w = map(int, cell.shape)
    shape_h, shape_w = map(int, shape.shape)
    top = random.randint(1, cell_h - shape_h - 1)
    left = random.randint(1, cell_w - shape_w - 1)
    window = cell[top : top + shape_h, left : left + shape_w]
    window[shape] = int(color)


def _render_region(
    cell_h: int, cell_w: int, accent: int, orient: str, period: int, shape: np.ndarray, obj_color: int
) -> np.ndarray:
    phase = random.randint(0, 1)
    cell = _stripe_pattern((cell_h, cell_w), BLACK, int(accent), orient, int(period), phase)
    _place_object(cell, shape, obj_color)
    return cell


def _render_input(
    cell_h: int, cell_w: int, assignments: list[tuple[tuple[int, int], tuple[str, int], int, int]], shape: np.ndarray
) -> np.ndarray:
    rows, cols = LAYOUT_SHAPE
    height = rows * cell_h + (rows - 1) * SEPARATOR_WIDTH
    width = cols * cell_w + (cols - 1) * SEPARATOR_WIDTH
    grid = np.full((height, width), SEP, dtype=int)

    stride_h = cell_h + SEPARATOR_WIDTH
    stride_w = cell_w + SEPARATOR_WIDTH
    for (ri, ci), (orient, period), obj_color, accent in assignments:
        r0 = int(ri) * stride_h
        c0 = int(ci) * stride_w
        region = _render_region(cell_h, cell_w, accent, orient, int(period), shape, int(obj_color))
        grid[r0 : r0 + cell_h, c0 : c0 + cell_w] = region
    return grid


def _expected_output(assignments: list[tuple[tuple[int, int], tuple[str, int], int, int]]) -> np.ndarray:
    color_by_combo = {(orient, int(period)): int(obj_color) for _slot, (orient, period), obj_color, _accent in assignments}
    return np.array([[color_by_combo[key]] for key in COMBO_ORDER], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    cell_h = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    cell_w = random.randint(CELL_SIZE_MIN, CELL_SIZE_MAX)
    assert max(2 * cell_h + 1, 2 * cell_w + 1) <= 30

    palette = random.sample(COLOR_POOL, len(COMBO_ORDER) * 2)
    object_colors = palette[: len(COMBO_ORDER)]
    accent_colors = palette[len(COMBO_ORDER) :]

    shape = _sample_shape(cell_h, cell_w)
    slots = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(slots)
    shuffled_combos = list(COMBO_ORDER)
    random.shuffle(shuffled_combos)

    assignments = [
        (slot, combo, int(obj_color), int(accent))
        for slot, combo, obj_color, accent in zip(slots, shuffled_combos, object_colors, accent_colors)
    ]
    grid = _render_input(cell_h, cell_w, assignments, shape)
    expected = _expected_output(assignments)

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid striped-region ARC input")
