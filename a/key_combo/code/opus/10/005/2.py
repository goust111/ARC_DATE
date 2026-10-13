from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(ARC_UTILS_DIR))

from arc_utils import Color, crop_bbox, find_components, get_bg, paste


BACKGROUND = Color.BLACK
GRAY = Color.GREY
RECT_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE)
GRID_HEIGHT_RANGE = (11, 22)
GRID_WIDTH_LIMIT = 30
RECT_COUNT_CHOICES = (4, 5)
RECT_SIZE_RANGE = range(2, 6)
GAP_RANGE = (1, 2)
MARGIN_RANGE = (0, 2)
MAX_GENERATION_ATTEMPTS = 500


def _rectangle_from_component(grid: np.ndarray, obj: dict, bg: int) -> dict:
    r0, c0, r1, c1 = map(int, obj["bbox"])
    patch = crop_bbox(grid, (r0, c0, r1, c1))
    h, w = map(int, patch.shape)

    assert r1 == grid.shape[0] - 1, "each rectangle must touch the bottom row"
    assert not np.any(patch == bg), "rectangle bbox must be fully filled"

    colors = set(map(int, np.unique(patch)))
    assert GRAY in colors, "each rectangle must contain a gray bar"
    body_colors = colors - {GRAY}
    assert len(body_colors) == 1, "each rectangle must have exactly one body color"
    body = int(next(iter(body_colors)))

    gray_mask = patch == GRAY
    gray_rows = np.flatnonzero(gray_mask.any(axis=1))
    assert len(gray_rows) == 1, "gray pixels must lie on exactly one row"
    bar_row = int(gray_rows[0])
    gray_cols = np.flatnonzero(gray_mask[bar_row])
    assert gray_cols.size > 0, "gray bar must be non-empty"
    c_start = int(gray_cols[0])
    c_end = int(gray_cols[-1])
    assert c_end - c_start + 1 == gray_cols.size, "gray bar must be contiguous"
    assert not np.any(gray_mask[np.arange(h) != bar_row]), "gray bar must be one-pixel thick"
    assert np.all(patch[bar_row, c_start : c_end + 1] == GRAY)

    expected = np.full((h, w), body, dtype=int)
    expected[bar_row, c_start : c_end + 1] = GRAY
    assert np.array_equal(patch, expected), "rectangle must be solid except for the gray bar"

    return {
        "bbox": (r0, c0, r1, c1),
        "body_color": body,
        "height": h,
        "width": w,
        "bar_row": bar_row,
        "bar_start": c_start,
        "bar_length": int(gray_cols.size),
    }


def _extract_rectangles(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    assert bg == BACKGROUND, "expected a black background"

    objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
    rectangles = [_rectangle_from_component(g, obj, bg) for obj in objects]
    assert 4 <= len(rectangles) <= 6, "expected 4 to 6 bottom-grounded rectangles"

    rectangles.sort(key=lambda item: int(item["bbox"][1]))

    left_edges = [int(rect["bbox"][1]) for rect in rectangles]
    assert len(set(left_edges)) == len(left_edges), "rectangle left edges must be unique"

    body_colors = [int(rect["body_color"]) for rect in rectangles]
    assert len(set(body_colors)) == len(body_colors), "rectangle colors must be distinct"

    bar_lengths = [int(rect["bar_length"]) for rect in rectangles]
    assert len(set(bar_lengths)) == len(bar_lengths), "gray bar lengths must be distinct"

    return rectangles


def _render_chain(shape: tuple[int, int], rectangles: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)

    row = 0
    col = 0
    prev = None
    for rect in rectangles:
        patch = np.full((int(rect["height"]), int(rect["width"])), int(rect["body_color"]), dtype=int)
        paste(out, patch, row, col, clip=True)
        if prev is not None and 0 <= row < shape[0] and 0 <= col < shape[1]:
            winner = prev if int(prev["bar_length"]) > int(rect["bar_length"]) else rect
            out[row, col] = int(winner["body_color"])
        prev = rect
        row += int(rect["height"]) - 1
        col += int(rect["width"]) - 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repack the bottom row of colored rectangles into a diagonal chain whose overlap winners are set by gray-bar length.

    Input structure:
    - The background is black.
    - There are 4 to 6 bottom-grounded filled rectangles with distinct body colors.
    - Each rectangle contains exactly one gray horizontal bar, and all bar lengths are distinct.

    Rule:
    - Sort the rectangles by left edge and place the first at the upper-left corner.
    - Place each next rectangle so its upper-left corner lands on the previous rectangle's lower-right corner.
    - At the single shared corner pixel, the rectangle with the longer gray bar keeps that pixel.
    - Remove gray from the final result by restoring every bar cell to its rectangle body color, and clip any part of the chain that leaves the canvas.

    Key insight:
    - The bottom arrangement determines the chain order, but the gray-bar length ranking independently decides the winner of each corner overlap.
    """
    rectangles = _extract_rectangles(np.asarray(grid, dtype=int))
    return _render_chain(np.asarray(grid, dtype=int).shape, rectangles)


def _chain_height(rectangles: list[dict]) -> int:
    return 1 + sum(int(rect["height"]) - 1 for rect in rectangles)


def _chain_width(rectangles: list[dict]) -> int:
    return 1 + sum(int(rect["width"]) - 1 for rect in rectangles)


def _start_rows(rectangles: list[dict]) -> list[int]:
    cur = 0
    starts = []
    for rect in rectangles:
        starts.append(cur)
        cur += int(rect["height"]) - 1
    return starts


def _build_rectangles() -> list[dict] | None:
    count = random.choice(RECT_COUNT_CHOICES)
    body_colors = random.sample(RECT_COLORS, count)

    available_lengths = list(range(1, 6))
    lengths = random.sample(available_lengths, count)

    while True:
        random.shuffle(lengths)
        signs = [np.sign(lengths[i + 1] - lengths[i]) for i in range(count - 1)]
        visible_signs = signs[: min(3, len(signs))]
        if any(sign < 0 for sign in visible_signs) and any(sign > 0 for sign in visible_signs):
            break

    rectangles = []
    for color, length in zip(body_colors, lengths):
        width = random.randint(max(2, int(length)), 5)
        height = random.randint(2, 5)
        bar_row = random.randint(0, height - 1)
        bar_start = random.randint(0, width - int(length))
        rectangles.append(
            {
                "body_color": int(color),
                "height": int(height),
                "width": int(width),
                "bar_row": int(bar_row),
                "bar_start": int(bar_start),
                "bar_length": int(length),
            }
        )
    return rectangles


def _pick_canvas(rectangles: list[dict], clip_case: bool) -> tuple[int, int, int, list[int]] | None:
    count = len(rectangles)
    gaps = [random.randint(*GAP_RANGE) for _ in range(count - 1)]
    left_margin = random.randint(*MARGIN_RANGE)
    right_margin = random.randint(*MARGIN_RANGE)

    input_width = left_margin + sum(int(rect["width"]) for rect in rectangles) + sum(gaps) + right_margin
    if input_width > GRID_WIDTH_LIMIT:
        return None

    starts = _start_rows(rectangles)
    chain_h = _chain_height(rectangles)
    max_h = max(int(rect["height"]) for rect in rectangles)
    min_visible_h = int(starts[min(3, count - 1)]) + 1

    if clip_case:
        low = max(max_h, min_visible_h)
        high = chain_h - 1
        if low > high:
            return None
        height = random.randint(low, high)
    else:
        height = random.randint(max(chain_h, GRID_HEIGHT_RANGE[0]), min(30, max(chain_h, GRID_HEIGHT_RANGE[0]) + 4))

    if not (max_h <= height <= 30):
        return None
    return height, input_width, left_margin, gaps


def _render_input(shape: tuple[int, int], rectangles: list[dict], left_margin: int, gaps: list[int]) -> np.ndarray:
    grid = np.full(shape, BACKGROUND, dtype=int)
    col = int(left_margin)
    h_total = shape[0]
    for index, rect in enumerate(rectangles):
        h = int(rect["height"])
        w = int(rect["width"])
        top = h_total - h
        grid[top : top + h, col : col + w] = int(rect["body_color"])
        bar_r = top + int(rect["bar_row"])
        bar_c0 = col + int(rect["bar_start"])
        bar_c1 = bar_c0 + int(rect["bar_length"])
        grid[bar_r, bar_c0:bar_c1] = GRAY
        rect["bbox"] = (top, col, top + h - 1, col + w - 1)
        col += w
        if index < len(gaps):
            col += int(gaps[index])
    return grid


def _visible_overlap_directions(rectangles: list[dict], height: int) -> list[int]:
    directions = []
    row = 0
    for prev, cur in zip(rectangles, rectangles[1:]):
        row += int(prev["height"]) - 1
        if row < height:
            if int(prev["bar_length"]) > int(cur["bar_length"]):
                directions.append(-1)
            else:
                directions.append(1)
    return directions


def _validate_instance(inp: np.ndarray, rectangles: list[dict], clip_case: bool) -> bool:
    try:
        parsed = _extract_rectangles(inp)
    except AssertionError:
        return False

    expected_tuples = [
        (
            int(rect["body_color"]),
            int(rect["height"]),
            int(rect["width"]),
            int(rect["bar_row"]),
            int(rect["bar_start"]),
            int(rect["bar_length"]),
        )
        for rect in sorted(rectangles, key=lambda item: int(item["bbox"][1]))
    ]
    parsed_tuples = [
        (
            int(rect["body_color"]),
            int(rect["height"]),
            int(rect["width"]),
            int(rect["bar_row"]),
            int(rect["bar_start"]),
            int(rect["bar_length"]),
        )
        for rect in parsed
    ]
    if parsed_tuples != expected_tuples:
        return False

    out = _render_chain(inp.shape, parsed)
    try:
        solved = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(solved, out):
        return False
    if np.any(out == GRAY):
        return False
    if np.array_equal(inp, out):
        return False

    visible = _visible_overlap_directions(parsed, inp.shape[0])
    if len(visible) < 2:
        return False
    if not (any(v < 0 for v in visible) and any(v > 0 for v in visible)):
        return False

    chain_h = _chain_height(parsed)
    if clip_case:
        if not chain_h > inp.shape[0]:
            return False
    else:
        if not chain_h <= inp.shape[0]:
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    clip_case = random.choice((False, True))
    rectangles = _build_rectangles()
    if rectangles is None:
        return None

    canvas = _pick_canvas(rectangles, clip_case)
    if canvas is None:
        return None
    height, width, left_margin, gaps = canvas

    inp = _render_input((height, width), rectangles, left_margin, gaps)
    if not _validate_instance(inp, rectangles, clip_case):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-chain ranking puzzle")
