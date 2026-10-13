from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg, paste, random_free_location


def _segment_info(component: dict) -> dict[str, int] | None:
    r0, c0, r1, c1 = map(int, component["bbox"])
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    area = int(component["area"])
    if area != height * width:
        return None
    if height == 1 and width >= 2:
        return {"orientation": "h", "index": r0, "length": width, "color": int(component["color"])}
    if width == 1 and height >= 2:
        return {"orientation": "v", "index": c0, "length": height, "color": int(component["color"])}
    return None


def _extract_real_segments(grid: np.ndarray) -> tuple[list[dict[str, int]], list[dict[str, int]], int]:
    bg = int(get_bg(grid))
    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []

    for component in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        info = _segment_info(component)
        if info is None or info["length"] < 3:
            continue
        if info["orientation"] == "h":
            horizontals.append(info)
        else:
            verticals.append(info)

    horizontals.sort(key=lambda item: item["index"])
    verticals.sort(key=lambda item: item["index"])
    assert horizontals and verticals, "expected both horizontal and vertical real segments"
    assert len({item["index"] for item in horizontals}) == len(horizontals), "real horizontal rows must be unique"
    assert len({item["index"] for item in verticals}) == len(verticals), "real vertical columns must be unique"

    lengths = [item["length"] for item in horizontals + verticals]
    assert len(lengths) == len(set(lengths)), "real segment lengths must be unique"
    return horizontals, verticals, bg


def _render_output(
    shape: tuple[int, int],
    horizontals: list[dict[str, int]],
    verticals: list[dict[str, int]],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for item in horizontals:
        out[item["index"], :] = item["color"]
    for item in verticals:
        out[:, item["index"]] = item["color"]
    for h_item in horizontals:
        for v_item in verticals:
            out[h_item["index"], v_item["index"]] = (
                h_item["color"] if h_item["length"] > v_item["length"] else v_item["color"]
            )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter away short distractor segments, extend the remaining bars to the borders, and color each crossing by the longer original bar.

    Input structure:
    - A black background contains several monochrome 4-connected foreground components.
    - The meaningful components are 1-pixel-thick horizontal or vertical line segments.
    - Extra singleton pixels and 2-pixel fragments act as distractors and may reuse any foreground color.

    Rule:
    - Identify straight horizontal and vertical monochrome components.
    - Discard every such component whose original length is less than 3.
    - Remember each surviving segment's original length before any extension.
    - Extend each surviving horizontal segment across its full row and each surviving vertical segment across its full column.
    - At every horizontal-row / vertical-column crossing, use the color of the segment with the larger original length.
    - Remove all filtered distractors so only the extended real segments remain.

    Key insight:
    - Crossing colors depend on the pre-extension segment lengths, so the solver must filter and store length metadata before drawing the full rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    horizontals, verticals, bg = _extract_real_segments(g)
    return _render_output(g.shape, horizontals, verticals, bg)


# --------------- generator (o5x) ---------------

BG = Color.BLACK
PALETTE = tuple(range(1, 10))
HEIGHT_RANGE = (12, 18)
WIDTH_RANGE = (12, 18)
SPACING_MIN = 3
TRUE_LENGTHS = (3, 4, 5, 6)
NOISE_COUNT_RANGE = (4, 8)
NOISE_SHAPES = ((1, 1), (1, 2), (2, 1))
GENERATION_ATTEMPTS = 400
INTERLEAVED_HORIZONTAL_LENGTHS = ((3, 5), (3, 6), (4, 5), (4, 6))


def _sample_spaced_pair(lo: int, hi: int, min_gap: int) -> tuple[int, int] | None:
    pairs = [(a, b) for a in range(lo, hi + 1) for b in range(a + min_gap, hi + 1)]
    return None if not pairs else random.choice(pairs)


def _sample_span(axis_len: int, length: int, forbidden: set[int]) -> int | None:
    starts = [
        start
        for start in range(axis_len - length + 1)
        if all(not (start <= point < start + length) for point in forbidden)
    ]
    if not starts:
        return None

    interior = [start for start in starts if 1 <= start and start + length <= axis_len - 1]
    return random.choice(interior or starts)


def _choose_lengths() -> tuple[list[int], list[int]]:
    horizontal = list(random.choice(INTERLEAVED_HORIZONTAL_LENGTHS))
    vertical = [length for length in TRUE_LENGTHS if length not in horizontal]
    random.shuffle(horizontal)
    random.shuffle(vertical)
    return horizontal, vertical


def _place_true_segments(grid: np.ndarray) -> tuple[list[dict[str, int]], list[dict[str, int]]] | None:
    height, width = map(int, grid.shape)
    row_pair = _sample_spaced_pair(1, height - 2, SPACING_MIN)
    col_pair = _sample_spaced_pair(1, width - 2, SPACING_MIN)
    if row_pair is None or col_pair is None:
        return None

    rows = list(row_pair)
    cols = list(col_pair)
    random.shuffle(rows)
    random.shuffle(cols)
    horizontal_lengths, vertical_lengths = _choose_lengths()
    colors = random.sample(PALETTE, 4)

    horizontals: list[dict[str, int]] = []
    verticals: list[dict[str, int]] = []
    forbidden_cols = set(cols)
    forbidden_rows = set(rows)

    for row, length, color in zip(rows, horizontal_lengths, colors[:2], strict=True):
        start = _sample_span(width, length, forbidden_cols)
        if start is None:
            return None
        paste(grid, np.full((1, length), color, dtype=int), row, start)
        horizontals.append({"orientation": "h", "index": row, "length": length, "color": color})

    for col, length, color in zip(cols, vertical_lengths, colors[2:], strict=True):
        start = _sample_span(height, length, forbidden_rows)
        if start is None:
            return None
        paste(grid, np.full((length, 1), color, dtype=int), start, col)
        verticals.append({"orientation": "v", "index": col, "length": length, "color": color})

    # Each puzzle should show both possible crossing winners.
    all_horizontal_wins = all(h["length"] > v["length"] for h in horizontals for v in verticals)
    all_vertical_wins = all(h["length"] < v["length"] for h in horizontals for v in verticals)
    if all_horizontal_wins or all_vertical_wins:
        return None

    assert all(grid[h["index"], v["index"]] == BG for h in horizontals for v in verticals)
    return horizontals, verticals


def _noise_shape_sequence(count: int) -> list[tuple[int, int]]:
    shapes: list[tuple[int, int]] = []
    if count >= 1:
        shapes.append((1, 1))
    if count >= 2:
        shapes.append(random.choice(((1, 2), (2, 1))))
    while len(shapes) < count:
        shapes.append(random.choice(NOISE_SHAPES))
    random.shuffle(shapes)
    return shapes


def _place_noise(
    grid: np.ndarray,
    horizontal_rows: set[int],
    vertical_cols: set[int],
) -> bool:
    blocked = grid.copy()
    reserve_color = 10
    blocked[list(horizontal_rows), :] = reserve_color
    blocked[:, list(vertical_cols)] = reserve_color

    noise_count = random.randint(*NOISE_COUNT_RANGE)
    for shape in _noise_shape_sequence(noise_count):
        color = random.choice(PALETTE)
        patch = np.full(shape, color, dtype=int)
        try:
            r0, c0 = random_free_location(blocked, patch, transparent=BG, padding=1)
        except ValueError:
            return False
        paste(grid, patch, r0, c0)
        paste(blocked, patch, r0, c0)
    return True


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    grid = np.full((height, width), BG, dtype=int)

    placed = _place_true_segments(grid)
    if placed is None:
        return None
    horizontals, verticals = placed

    if not _place_noise(
        grid,
        {item["index"] for item in horizontals},
        {item["index"] for item in verticals},
    ):
        return None

    expected = _render_output(grid.shape, horizontals, verticals, BG)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid filtering-and-extension ARC input")
