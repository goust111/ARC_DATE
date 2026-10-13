from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared_035_7",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
paste = _ARC_UTILS.paste


BLACK = int(Color.BLACK)
BLUE = int(Color.BLUE)
RED = int(Color.RED)

# ARC size cap + >=3 empty rows between bars means at most 8 one-row bars fit.
# We keep counts in [5, 8] so K = number of selected bars varies between 2 and 3.
BAR_COUNTS = (5, 6, 7, 8)
WIDTH_RANGE = (4, 8)
GRID_WIDTH_RANGE = (11, 18)
MIN_GRID_HEIGHT = 18
GEN_RETRIES = 300


def _extract_bars(grid: np.ndarray) -> list[tuple[int, int, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    colors = {int(x) for x in np.unique(g)}
    assert colors.issubset({BLACK, BLUE}), "expected only black background and blue bars"

    objects = find_components(
        g,
        bg=BLACK,
        colors=[BLUE],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    assert objects, "expected at least one bar"

    ordered = sorted(objects, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
    bars: list[tuple[int, int, int]] = []
    prev_row: int | None = None
    for obj in ordered:
        bar = crop_bbox(g, obj["bbox"])
        assert bbox_h(obj["bbox"]) == 1, "each object must be one row tall"
        assert obj["area"] == bbox_w(obj["bbox"]), "each object must be a solid rectangle"
        assert np.all(bar == BLUE), "each object must be blue"

        row, col = int(obj["bbox"][0]), int(obj["bbox"][1])
        width = int(bbox_w(obj["bbox"]))
        if prev_row is not None:
            assert row - prev_row >= 4, "bars must have at least three blank rows between them"
        bars.append((row, col, width))
        prev_row = row

    return bars


def _draw_square_border(out: np.ndarray, r: int, c: int, radius: int, color: int) -> None:
    h, w = out.shape
    r0, r1 = int(r - radius), int(r + radius)
    c0, c1 = int(c - radius), int(c + radius)

    clip_c0 = max(0, c0)
    clip_c1 = min(w - 1, c1)
    clip_r0 = max(0, r0)
    clip_r1 = min(h - 1, r1)

    if clip_c0 <= clip_c1:
        if 0 <= r0 < h:
            out[r0, clip_c0 : clip_c1 + 1] = color
        if 0 <= r1 < h:
            out[r1, clip_c0 : clip_c1 + 1] = color
    if clip_r0 <= clip_r1:
        if 0 <= c0 < w:
            out[clip_r0 : clip_r1 + 1, c0] = color
        if 0 <= c1 < w:
            out[clip_r0 : clip_r1 + 1, c1] = color


def _transform_from_bars(shape: tuple[int, int], bars: list[tuple[int, int, int]]) -> np.ndarray:
    out = np.full(shape, BLACK, dtype=int)
    rows = [row for row, _col, _width in bars]
    reversed_rows = rows[::-1]
    selected_count = sum(index % 3 == 0 for index in range(len(bars)))
    red_centers: list[tuple[int, int]] = []

    for index, (row, col, width) in enumerate(bars):
        color = RED if index % 3 == 0 else BLUE
        new_row = reversed_rows[index]
        patch = np.full((1, width), color, dtype=int)
        paste(out, patch, new_row, col)
        if color == RED:
            red_centers.append((new_row, col + width // 2))

    for center_r, center_c in red_centers:
        for radius in range(1, selected_count + 1):
            _draw_square_border(out, center_r, center_c, radius, RED)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every third horizontal bar, reverse the stack, and grow red concentric square borders from the moved red bars.

    Input structure:
    - The grid has a black background and several solid blue horizontal one-row bars.
    - The bars are separated vertically by at least three empty rows, and each bar is a maximal 4-connected rectangle.

    Rule:
    - Order the bars from top to bottom and recolor ranks 1, 4, 7, ... red; let K be the number of red bars.
    - Reverse the vertical order of all bars while preserving each bar's width and columns, so each bar moves to the row slot of the symmetrically ranked bar from the bottom.
    - For every moved red bar, use its center pixel at column left + width // 2 as a seed and draw red square borders of radii 1 through K, clipping any out-of-bounds border cells.

    Key insight:
    - The same ordinal-selection step both chooses which bars become red and defines the global growth depth K used after the full stack is vertically reversed.
    """
    bars = _extract_bars(grid)
    return _transform_from_bars(np.asarray(grid, dtype=int).shape, bars)


def _random_partition(total: int, slots: int, rng: random.Random) -> list[int]:
    parts = [0] * slots
    for _ in range(total):
        parts[rng.randrange(slots)] += 1
    return parts


def _sample_rows(count: int, height: int, rng: random.Random) -> list[int]:
    min_height = 1 + 4 * (count - 1)
    slack = height - min_height
    extras = _random_partition(slack, count + 1, rng)

    rows = [extras[0]]
    for gap_extra in extras[1:count]:
        rows.append(rows[-1] + 4 + gap_extra)
    assert rows[-1] < height
    return rows


def _sample_widths(count: int, max_width: int, rng: random.Random) -> list[int]:
    lo, hi = WIDTH_RANGE
    hi = min(hi, max_width)
    assert lo <= hi
    for _ in range(20):
        widths = [rng.randint(lo, hi) for _ in range(count)]
        if len(set(widths)) > 1:
            return widths
    return [rng.randint(lo, hi) for _ in range(count)]


def _build_input_from_specs(
    height: int,
    width: int,
    rows: list[int],
    lefts: list[int],
    widths: list[int],
) -> np.ndarray:
    grid = np.full((height, width), BLACK, dtype=int)
    for row, left, bar_width in zip(rows, lefts, widths):
        patch = np.full((1, bar_width), BLUE, dtype=int)
        paste(grid, patch, row, left)
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.choice(BAR_COUNTS)
    min_height = 1 + 4 * (count - 1)
    height = rng.randint(max(MIN_GRID_HEIGHT, min_height), 30)
    width = rng.randint(GRID_WIDTH_RANGE[0], GRID_WIDTH_RANGE[1])

    alignment = rng.choice(("left", "center"))
    if alignment == "left":
        max_left = min(3, width - WIDTH_RANGE[1])
        if max_left < 0:
            return None
        anchor = rng.randint(0, max_left)
        widths = _sample_widths(count, width - anchor, rng)
        lefts = [anchor] * count
    else:
        widths = _sample_widths(count, width, rng)
        lefts = [(width - bar_width) // 2 for bar_width in widths]

    rows = _sample_rows(count, height, rng)
    grid = _build_input_from_specs(height, width, rows, lefts, widths)

    bars = [(row, left, bar_width) for row, left, bar_width in zip(rows, lefts, widths)]
    expected = _transform_from_bars(grid.shape, bars)

    try:
        parsed = _extract_bars(grid)
        actual = main(grid)
    except Exception:
        return None

    if parsed != bars:
        return None
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
