from __future__ import annotations

import random

import numpy as np


LEGEND_BG = 5
INTERIOR_BG = 0
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

OBJECT_COUNT_RANGE = (2, 5)
SEGMENT_LENGTH_RANGE = (1, 7)
LEADING_MARGIN_RANGE = (0, 3)
TRAILING_MARGIN_RANGE = (0, 3)
MIDDLE_GAP_RANGE = (1, 3)
AXIS_INTERIOR_MAX = 29

MAX_GENERATION_ATTEMPTS = 400
MAX_LAYOUT_ATTEMPTS = 200
MAX_SPEC_ATTEMPTS = 400


def _extract_runs(line: np.ndarray, filler: int, start_index: int) -> dict[int, tuple[int, int]]:
    spans: dict[int, tuple[int, int]] = {}
    idx = int(start_index)
    while idx < len(line):
        color = int(line[idx])
        if color == filler:
            idx += 1
            continue
        run_start = idx
        while idx + 1 < len(line) and int(line[idx + 1]) == color:
            idx += 1
        if color in spans:
            raise AssertionError(f"color {color} appears in multiple runs")
        spans[color] = (run_start, idx)
        idx += 1
    return spans


def _interior_bg(grid: np.ndarray) -> int:
    interior = np.asarray(grid[1:, 1:], dtype=int)
    colors, counts = np.unique(interior, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _rectangles_from_legends(
    top_spans: dict[int, tuple[int, int]],
    left_spans: dict[int, tuple[int, int]],
) -> list[tuple[int, int, int, int, int, int]]:
    top_colors = set(top_spans)
    left_colors = set(left_spans)
    if top_colors != left_colors or not top_colors:
        raise AssertionError("top-row colors and left-column colors must match exactly")

    rectangles: list[tuple[int, int, int, int, int, int]] = []
    for color in top_colors:
        c0, c1 = top_spans[color]
        r0, r1 = left_spans[color]
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        rectangles.append((area, int(color), int(r0), int(r1), int(c0), int(c1)))
    return rectangles


def _render_output(
    shape: tuple[int, int],
    top_spans: dict[int, tuple[int, int]],
    left_spans: dict[int, tuple[int, int]],
    legend_bg: int,
    interior_bg: int,
) -> np.ndarray:
    out = np.full(shape, int(interior_bg), dtype=int)
    out[0, :] = int(legend_bg)
    out[:, 0] = int(legend_bg)

    for color, (c0, c1) in top_spans.items():
        out[0, c0 : c1 + 1] = int(color)
    for color, (r0, r1) in left_spans.items():
        out[r0 : r1 + 1, 0] = int(color)

    rectangles = _rectangles_from_legends(top_spans, left_spans)
    if len({area for area, *_rest in rectangles}) != len(rectangles):
        raise AssertionError("rectangle areas must be pairwise distinct")

    # Larger rectangles go down first so smaller-area rectangles stay visible on top.
    rectangles.sort(key=lambda item: (-item[0], item[1], item[2], item[4]))
    for _area, color, r0, r1, c0, c1 in rectangles:
        out[r0 : r1 + 1, c0 : c1 + 1] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct solid rectangles from top-row and left-column color projections.

    Input structure:
    - The grid interior is a uniform background, while the top row and left column contain colored legend runs separated by one filler color.
    - Each rectangle color appears in exactly one contiguous run on the top row and exactly one contiguous run on the left column.
    - The top-row run gives that color's column span and the left-column run gives its row span.

    Rule:
    - Parse every non-filler color run on the top row and on the left column.
    - Pair runs by color to recover each solid rectangle's row range and column range.
    - Draw all recovered rectangles into the interior; when rectangles overlap, the one with smaller area remains visible. Keep the top row and left column unchanged.

    Key insight:
    - The visible edge runs are one-dimensional projections of hidden filled rectangles, so the task is to cross row spans with column spans rather than to extend lines or copy the legends themselves.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or g.shape[0] < 2 or g.shape[1] < 2:
        raise AssertionError("expected a 2D grid with at least 2 rows and 2 columns")

    legend_bg = int(g[0, 0])
    bg = _interior_bg(g)
    top_spans = _extract_runs(g[0], legend_bg, 1)
    left_spans = _extract_runs(g[:, 0], legend_bg, 1)
    return _render_output(g.shape, top_spans, left_spans, legend_bg, bg)


def _sample_order(colors: list[int]) -> list[int]:
    order = colors[:]
    random.shuffle(order)
    return order


def _sample_lengths(count: int) -> list[int]:
    return [random.randint(SEGMENT_LENGTH_RANGE[0], SEGMENT_LENGTH_RANGE[1]) for _ in range(count)]


def _sample_spans(order: list[int], lengths: list[int]) -> dict[int, tuple[int, int]] | None:
    if len(order) != len(lengths):
        raise AssertionError("order and lengths must have the same length")

    for _ in range(MAX_LAYOUT_ATTEMPTS):
        lead = random.randint(LEADING_MARGIN_RANGE[0], LEADING_MARGIN_RANGE[1])
        trail = random.randint(TRAILING_MARGIN_RANGE[0], TRAILING_MARGIN_RANGE[1])
        gaps = [
            random.randint(MIDDLE_GAP_RANGE[0], MIDDLE_GAP_RANGE[1])
            for _ in range(max(0, len(order) - 1))
        ]
        total = lead + sum(lengths) + sum(gaps) + trail
        if total > AXIS_INTERIOR_MAX:
            continue

        spans: dict[int, tuple[int, int]] = {}
        cursor = 1 + lead
        for idx, color in enumerate(order):
            length = int(lengths[idx])
            spans[int(color)] = (cursor, cursor + length - 1)
            cursor += length
            if idx < len(gaps):
                cursor += gaps[idx]
        return spans
    return None


def _build_input(
    top_spans: dict[int, tuple[int, int]],
    left_spans: dict[int, tuple[int, int]],
) -> np.ndarray:
    width = 1 + max(c1 for c0, c1 in top_spans.values())
    height = 1 + max(r1 for r0, r1 in left_spans.values())
    grid = np.full((height, width), LEGEND_BG, dtype=int)
    grid[1:, 1:] = INTERIOR_BG

    for color, (c0, c1) in top_spans.items():
        grid[0, c0 : c1 + 1] = int(color)
    for color, (r0, r1) in left_spans.items():
        grid[r0 : r1 + 1, 0] = int(color)
    return grid


def _gen_attempt() -> np.ndarray | None:
    for _ in range(MAX_SPEC_ATTEMPTS):
        count = random.randint(OBJECT_COUNT_RANGE[0], OBJECT_COUNT_RANGE[1])
        colors = random.sample(PALETTE, count)

        top_order = _sample_order(colors)
        left_order = _sample_order(colors)
        if count > 1 and top_order == left_order:
            left_order = top_order[1:] + top_order[:1]

        top_lengths = _sample_lengths(count)
        left_lengths = _sample_lengths(count)
        if len(set(top_lengths)) == 1 and len(set(left_lengths)) == 1:
            continue

        top_spans = _sample_spans(top_order, top_lengths)
        left_spans = _sample_spans(left_order, left_lengths)
        if top_spans is None or left_spans is None:
            continue

        rectangles = _rectangles_from_legends(top_spans, left_spans)
        if len({area for area, *_rest in rectangles}) != count:
            continue

        grid = _build_input(top_spans, left_spans)
        expected = _render_output(grid.shape, top_spans, left_spans, LEGEND_BG, INTERIOR_BG)
        if not np.array_equal(main(grid), expected):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
