from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

from arc_utils import colors_present


BG = 0
PALETTE = tuple(range(1, 10))

GRID_H_RANGE = (9, 20)
GRID_W_RANGE = (9, 20)
RECT_COUNT_RANGE = (2, 5)
ROW_LEN_RANGE = (2, 7)
COL_LEN_RANGE = (2, 7)
MIN_GAP = 1
GEN_RETRIES = 500
PLACEMENT_RETRIES = 200


def _parse_segments(line: np.ndarray) -> dict[int, tuple[int, int]]:
    """Return color -> inclusive span for one 1D wall, rejecting split colors."""
    arr = np.asarray(line, dtype=int).ravel()
    spans: dict[int, tuple[int, int]] = {}
    i = 0
    while i < len(arr):
        color = int(arr[i])
        if color == BG:
            i += 1
            continue
        j = i
        while j + 1 < len(arr) and int(arr[j + 1]) == color:
            j += 1
        assert color not in spans, "each wall color must form one contiguous segment"
        spans[color] = (i, j)
        i = j + 1
    return spans


def _segment_len(span: tuple[int, int]) -> int:
    return int(span[1] - span[0] + 1)


def _rectangles_from_grid(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "grid must leave an interior plus two clue walls"
    assert int(g[-1, -1]) == BG, "bottom-right corner must stay background"
    assert np.all(g[:-1, :-1] == BG), "only the bottom wall and right wall may be non-background"

    bottom = _parse_segments(g[-1, :-1])
    right = _parse_segments(g[:-1, -1])
    assert bottom, "expected at least one bottom projection segment"
    assert set(bottom) == set(right), "bottom and right walls must use the same color set"

    edge_colors = set(colors_present(g, exclude={BG}))
    assert edge_colors == set(bottom) == set(right), "all non-background colors must live on both walls"

    rects: list[dict[str, int]] = []
    for color in sorted(bottom):
        c0, c1 = bottom[color]
        r0, r1 = right[color]
        area = _segment_len((r0, r1)) * _segment_len((c0, c1))
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "area": int(area),
            }
        )

    areas = [rect["area"] for rect in rects]
    assert len(set(areas)) == len(areas), "rectangle areas must be pairwise distinct"
    return rects


def _render_rectangles(base: np.ndarray, rects: list[dict[str, int]]) -> np.ndarray:
    out = np.asarray(base, dtype=int).copy()
    for rect in sorted(rects, key=lambda item: item["area"], reverse=True):
        out[rect["r0"] : rect["r1"] + 1, rect["c0"] : rect["c1"] + 1] = rect["color"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct filled rectangles from color-matched bottom and right projection ticks.

    Input structure:
    - The grid has a black background.
    - The bottom row contains several disjoint colored horizontal tick segments, and the
      rightmost column contains the same colors as vertical tick segments.
    - Everything except these two walls is empty black interior.

    Rule:
    - For each color, the bottom-wall segment gives the rectangle's column interval and the
      right-wall segment gives its row interval.
    - Paint the solid rectangle spanned by those two intervals into the interior, drawing
      rectangles from larger area to smaller area so smaller ones would cover larger ones.
    - Keep the original bottom-row and right-column tick segments unchanged.

    Key insight:
    - The colored edge segments are not rays to extend; they are paired orthogonal
      projections of invisible rectangles, and the draw order is determined by derived area.
    """
    rects = _rectangles_from_grid(grid)
    return _render_rectangles(np.asarray(grid, dtype=int), rects)


def _max_count(limit: int, min_len: int) -> int:
    return max(0, (limit + MIN_GAP) // (min_len + MIN_GAP))


def _sample_spans(limit: int, count: int, length_range: tuple[int, int]) -> list[tuple[int, int]] | None:
    lo, hi = map(int, length_range)
    hi = min(hi, limit)
    if lo > hi:
        return None

    for _ in range(PLACEMENT_RETRIES):
        lengths = [random.randint(lo, hi) for _ in range(count)]
        used = sum(lengths) + MIN_GAP * (count - 1)
        if used > limit:
            continue

        slack = limit - used
        extra_gaps = list(np.random.multinomial(slack, [1.0 / (count + 1)] * (count + 1)))

        spans: list[tuple[int, int]] = []
        pos = int(extra_gaps[0])
        for idx, length in enumerate(lengths):
            start = pos
            end = start + int(length) - 1
            spans.append((int(start), int(end)))
            pos = end + 1
            if idx < count - 1:
                pos += MIN_GAP + int(extra_gaps[idx + 1])

        assert spans[-1][1] < limit
        return spans
    return None


def _build_input(
    shape: tuple[int, int],
    row_spans: dict[int, tuple[int, int]],
    col_spans: dict[int, tuple[int, int]],
) -> np.ndarray:
    h, w = map(int, shape)
    grid = np.full((h, w), BG, dtype=int)

    for color, (r0, r1) in row_spans.items():
        grid[r0 : r1 + 1, -1] = int(color)
    for color, (c0, c1) in col_spans.items():
        grid[-1, c0 : c1 + 1] = int(color)

    grid[-1, -1] = BG
    return grid


def _expected_from_specs(
    shape: tuple[int, int],
    row_spans: dict[int, tuple[int, int]],
    col_spans: dict[int, tuple[int, int]],
) -> np.ndarray:
    expected = _build_input(shape, row_spans, col_spans)
    rects = []
    for color in sorted(row_spans):
        r0, r1 = row_spans[color]
        c0, c1 = col_spans[color]
        rects.append(
            {
                "color": int(color),
                "r0": int(r0),
                "r1": int(r1),
                "c0": int(c0),
                "c1": int(c1),
                "area": _segment_len((r0, r1)) * _segment_len((c0, c1)),
            }
        )
    return _render_rectangles(expected, rects)


def _validate_case(
    inp: np.ndarray,
    expected: np.ndarray,
    row_spans: dict[int, tuple[int, int]],
    col_spans: dict[int, tuple[int, int]],
) -> None:
    assert inp.shape == expected.shape
    assert inp.ndim == 2 and max(inp.shape) <= 30
    assert np.all(inp[:-1, :-1] == BG)
    assert int(inp[-1, -1]) == BG

    parsed = _rectangles_from_grid(inp)
    parsed_rows = {rect["color"]: (rect["r0"], rect["r1"]) for rect in parsed}
    parsed_cols = {rect["color"]: (rect["c0"], rect["c1"]) for rect in parsed}
    assert parsed_rows == row_spans
    assert parsed_cols == col_spans

    independent = _expected_from_specs(inp.shape, row_spans, col_spans)
    assert np.array_equal(independent, expected), "spec-rendered output must match expected output"

    solved = main(inp)
    assert np.array_equal(solved, expected), "main() must match the rule-defined rendering"


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    inner_h = h - 1
    inner_w = w - 1

    max_count = min(
        RECT_COUNT_RANGE[1],
        len(PALETTE),
        _max_count(inner_h, ROW_LEN_RANGE[0]),
        _max_count(inner_w, COL_LEN_RANGE[0]),
    )
    if max_count < RECT_COUNT_RANGE[0]:
        return None

    count = random.randint(RECT_COUNT_RANGE[0], max_count)
    row_spans_list = _sample_spans(inner_h, count, ROW_LEN_RANGE)
    col_spans_list = _sample_spans(inner_w, count, COL_LEN_RANGE)
    if row_spans_list is None or col_spans_list is None:
        return None

    colors = random.sample(PALETTE, count)
    row_colors = colors[:]
    col_colors = colors[:]
    random.shuffle(row_colors)
    random.shuffle(col_colors)
    if count > 1 and row_colors == col_colors:
        col_colors = col_colors[1:] + col_colors[:1]

    row_spans = {color: span for color, span in zip(row_colors, row_spans_list)}
    col_spans = {color: span for color, span in zip(col_colors, col_spans_list)}

    areas = {
        color: _segment_len(row_spans[color]) * _segment_len(col_spans[color])
        for color in colors
    }
    if len(set(areas.values())) != count:
        return None

    inp = _build_input((h, w), row_spans, col_spans)
    expected = _expected_from_specs((h, w), row_spans, col_spans)

    try:
        _validate_case(inp, expected, row_spans, col_spans)
    except AssertionError:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        candidate = _generate_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid projection-rectangle sample")
