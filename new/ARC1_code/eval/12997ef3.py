from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste
import random


def _marker_pixels(g: np.ndarray, marker_colors: list[int]) -> list[tuple[int, int, int]]:
    markers: list[tuple[int, int, int]] = []
    for color in marker_colors:
        coords = np.argwhere(g == color)
        assert coords.shape[0] == 1, "each marker color must appear as one pixel"
        r, c = map(int, coords[0])
        markers.append((r, c, int(color)))
    assert len(markers) >= 2, "expected at least two color markers"
    return markers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp recolored copies of the foreground template in the order given by the marker pixels.

    Input structure
    - A dominant background contains one multi-pixel foreground template color.
    - Every other foreground color is a single marker pixel.
    - The marker pixels lie on one common row or one common column.

    Rule
    - Crop the template color to its bounding box and use it as a mask.
    - Sort markers left-to-right when they share a row, or top-to-bottom when they share a column.
    - For each sorted marker, make one copy of the template mask recolored to that marker color.
    - Concatenate the copies horizontally for a marker row, or vertically for a marker column.

    Key insight
    - The single pixels are an ordered palette; the larger foreground shape is a reusable stencil.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    counts = Counter(map(int, g.ravel()))
    foreground = [color for color in counts if color != bg]
    template_colors = [color for color in foreground if counts[color] > 1]
    assert len(template_colors) == 1, "expected exactly one multi-pixel template color"
    template_color = int(template_colors[0])

    template_bbox = bbox(g == template_color)
    assert template_bbox is not None, "template mask must be nonempty"
    mask = crop_bbox(g == template_color, template_bbox)
    assert mask.any(), "template mask must be nonempty"
    th, tw = map(int, mask.shape)

    marker_colors = [color for color in foreground if color != template_color]
    markers = _marker_pixels(g, marker_colors)
    marker_rows = {r for r, _c, _color in markers}
    marker_cols = {c for _r, c, _color in markers}
    horizontal = len(marker_rows) == 1
    vertical = len(marker_cols) == 1
    assert horizontal != vertical, "markers must form exactly one row or one column"

    if horizontal:
        markers.sort(key=lambda item: item[1])
        out = np.full((th, tw * len(markers)), int(bg), dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = color
            paste(out, patch, 0, i * tw)
    else:
        markers.sort(key=lambda item: item[0])
        out = np.full((th * len(markers), tw), int(bg), dtype=int)
        for i, (_r, _c, color) in enumerate(markers):
            patch = np.full(mask.shape, int(bg), dtype=int)
            patch[mask] = color
            paste(out, patch, i * th, 0)

    return out

# --------------- generator (o5x) ---------------

# Canvas and sparsity constraints.
CANVAS_SIZE_RANGE = range(9, 15)
TEMPLATE_SIZE = 3
BACKGROUND_COLOR, TEMPLATE_COLOR = 0, 1
MIN_BACKGROUND_FRACTION = 0.9
SPARSE_AREA_FACTOR = 10

# Template and marker sampling.
MARKER_COUNT_BOUNDS = (2, 4)
TEMPLATE_FILL_BOUNDS = (4, 6)
MARKER_COLORS = list(range(2, 10))
MARKER_SPACING = 2
BOOL_CHOICES = [False, True]

# Retry policy.
MAX_GENERATION_ATTEMPTS = 300


def _gen_template_mask(fill_count: int) -> np.ndarray:
    mask = np.zeros((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)

    cols = list(range(TEMPLATE_SIZE))
    random.shuffle(cols)
    cells = [(r, cols[r]) for r in range(TEMPLATE_SIZE)]

    remaining = [(r, c) for r in range(TEMPLATE_SIZE) for c in range(TEMPLATE_SIZE) if (r, c) not in cells]
    random.shuffle(remaining)
    cells.extend(remaining[: fill_count - TEMPLATE_SIZE])

    for r, c in cells:
        mask[r, c] = True

    if random.choice(BOOL_CHOICES):
        mask = np.flipud(mask)
    if random.choice(BOOL_CHOICES):
        mask = np.fliplr(mask)
    if random.choice(BOOL_CHOICES):
        mask = mask.T
    return np.array(mask, dtype=bool)


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BACKGROUND_COLOR, dtype=int)
    patch[mask] = color
    return patch


def _gen_expected_output(mask: np.ndarray, marker_colors: list[int], horizontal: bool) -> np.ndarray:
    th, tw = map(int, mask.shape)
    if horizontal:
        out = np.full((th, tw * len(marker_colors)), BACKGROUND_COLOR, dtype=int)
        for i, color in enumerate(marker_colors):
            paste(out, _colored_patch(mask, color), 0, i * tw)
    else:
        out = np.full((th * len(marker_colors), tw), BACKGROUND_COLOR, dtype=int)
        for i, color in enumerate(marker_colors):
            paste(out, _colored_patch(mask, color), i * th, 0)
    return out


def _place_marker_line(grid: np.ndarray, marker_colors: list[int], horizontal: bool, tr: int, tc: int) -> bool:
    """Place the marker line outside the template span; return False if none fits."""
    h, w = map(int, grid.shape)
    line_limit, run_limit = (h, w) if horizontal else (w, h)
    template_start = tr if horizontal else tc
    lines = [i for i in range(line_limit) if not (template_start <= i <= template_start + TEMPLATE_SIZE - 1)]
    if not lines:
        return False
    line = random.choice(lines)
    starts = list(range(run_limit - MARKER_SPACING * (len(marker_colors) - 1)))
    if not starts:
        return False
    start = random.choice(starts)
    for i, color in enumerate(marker_colors):
        offset = start + MARKER_SPACING * i
        if horizontal:
            grid[line, offset] = color
        else:
            grid[offset, line] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    marker_count = random.randint(*MARKER_COUNT_BOUNDS)
    fill_count = random.randint(*TEMPLATE_FILL_BOUNDS)
    valid_dims = [
        (h, w)
        for h in CANVAS_SIZE_RANGE
        for w in CANVAS_SIZE_RANGE
        if h * w > SPARSE_AREA_FACTOR * (marker_count + fill_count)
    ]
    if not valid_dims:
        return None

    h, w = random.choice(valid_dims)
    horizontal = random.choice(BOOL_CHOICES)
    grid = np.full((h, w), BACKGROUND_COLOR, dtype=int)
    mask = _gen_template_mask(fill_count)
    template = _colored_patch(mask, TEMPLATE_COLOR)
    tr = random.randint(0, h - TEMPLATE_SIZE)
    tc = random.randint(0, w - TEMPLATE_SIZE)
    paste(grid, template, tr, tc, transparent=BACKGROUND_COLOR)

    marker_colors = random.sample(MARKER_COLORS, marker_count)
    if not _place_marker_line(grid, marker_colors, horizontal, tr, tc):
        return None

    if np.count_nonzero(grid == BACKGROUND_COLOR) / grid.size <= MIN_BACKGROUND_FRACTION:
        return None

    expected = _gen_expected_output(mask, marker_colors, horizontal)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 12997ef3 input")
