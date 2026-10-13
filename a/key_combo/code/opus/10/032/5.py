from __future__ import annotations

import random
from collections import deque

import numpy as np


BACKGROUND = 0
FOREGROUND_COLORS = tuple(range(1, 10))
GEN_ATTEMPTS = 400
HALF_HEIGHT_RANGE = (6, 12)
WIDTH_RANGE = (14, 24)
LANE_COUNT_CHOICES = (3, 4)
LANE_HEIGHT_RANGE = (1, 3)
MIN_LANE_GAP = 1


def _unique_foreground_color(half: np.ndarray) -> int:
    colors = sorted(set(map(int, np.unique(half))) - {BACKGROUND})
    assert len(colors) == 1, "each half must contain exactly one foreground color"
    return colors[0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(coords)
    return out


def _rectangles_only(mask: np.ndarray) -> bool:
    components = _components(mask)
    if not components:
        return False
    for coords in components:
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(coords) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
    return True


def _expected_output(top: np.ndarray, bottom: np.ndarray, top_color: int, bottom_color: int) -> np.ndarray:
    rows, cols = np.indices(top.shape)
    top_fg = top != BACKGROUND
    bottom_fg = bottom != BACKGROUND

    out = np.zeros_like(top)
    overlap = top_fg & bottom_fg
    top_only = top_fg & ~bottom_fg
    bottom_only = bottom_fg & ~top_fg

    out[top_only] = top_color
    out[bottom_only] = bottom_color
    out[overlap & ((rows + cols) % 2 == 0)] = top_color
    out[overlap & ((rows + cols) % 2 == 1)] = bottom_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two stacked half-masks cell by cell and merge them into a single checkerboard-or-solid output.

    Input structure:
    - The input grid is split into two equally sized halves stacked vertically.
    - Each half uses black as background and contains exactly one non-black foreground color.
    - Foreground pixels in each half form one or more solid axis-aligned rectangles.

    Rule:
    - Align the top half and bottom half by row and column and produce one output grid with the size of a single half.
    - If both aligned cells are foreground, fill that output cell with a checkerboard anchored at the output's top-left:
      even `(r + c)` takes the top half's foreground color and odd `(r + c)` takes the bottom half's foreground color.
    - If only one aligned cell is foreground, copy that half's foreground color; if both are background, keep black.

    Key insight:
    - The transform is a three-way per-cell classification of the two halves, and the overlap class needs both detected
      foreground colors plus the global output parity anchor to resolve its final color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape[0] % 2 == 0, "expected two equally tall stacked halves"

    half_h = g.shape[0] // 2
    top = g[:half_h, :]
    bottom = g[half_h:, :]

    top_color = _unique_foreground_color(top)
    bottom_color = _unique_foreground_color(bottom)

    assert set(map(int, np.unique(top))) <= {BACKGROUND, top_color}
    assert set(map(int, np.unique(bottom))) <= {BACKGROUND, bottom_color}

    return _expected_output(top, bottom, top_color, bottom_color)


def _sample_row_lanes(half_h: int, lane_count: int) -> list[tuple[int, int]] | None:
    min_h, max_h = LANE_HEIGHT_RANGE
    min_needed = lane_count * min_h + (lane_count - 1) * MIN_LANE_GAP
    if half_h < min_needed:
        return None

    for _ in range(100):
        heights = [random.randint(min_h, max_h) for _ in range(lane_count)]
        base = sum(heights) + (lane_count - 1) * MIN_LANE_GAP
        if base > half_h:
            continue

        slack = half_h - base
        extras = [0] * (lane_count + 1)
        for _ in range(slack):
            extras[random.randrange(lane_count + 1)] += 1

        lanes: list[tuple[int, int]] = []
        r = extras[0]
        for i, height in enumerate(heights):
            lanes.append((r, r + height - 1))
            r += height
            if i < lane_count - 1:
                r += MIN_LANE_GAP + extras[i + 1]
        return lanes
    return None


def _sample_span(width: int, min_width: int, max_width: int) -> tuple[int, int]:
    span_width = random.randint(min_width, min(max_width, width))
    start = random.randint(0, width - span_width)
    return start, start + span_width - 1


def _sample_mixed_spans(width: int) -> tuple[tuple[int, int], tuple[int, int]]:
    max_total = min(width - 2, 10)
    total = random.randint(4, max_total)
    overlap_width = random.randint(2, total - 2)
    top_only_width = random.randint(1, total - overlap_width - 1)
    bottom_only_width = total - overlap_width - top_only_width
    start = random.randint(0, width - total)

    if random.random() < 0.5:
        top = (start, start + top_only_width + overlap_width - 1)
        bottom = (start + top_only_width, start + total - 1)
    else:
        bottom = (start, start + bottom_only_width + overlap_width - 1)
        top = (start + bottom_only_width, start + total - 1)
    return top, bottom


def _paint_rect(grid: np.ndarray, row_span: tuple[int, int], col_span: tuple[int, int], color: int) -> None:
    r0, r1 = row_span
    c0, c1 = col_span
    grid[r0 : r1 + 1, c0 : c1 + 1] = color


def _semantic_check(inp: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or inp.ndim != 2:
        return False
    if inp.shape[0] % 2 != 0 or max(inp.shape) > 30:
        return False

    half_h = inp.shape[0] // 2
    top = inp[:half_h, :]
    bottom = inp[half_h:, :]

    try:
        top_color = _unique_foreground_color(top)
        bottom_color = _unique_foreground_color(bottom)
    except AssertionError:
        return False

    if top_color == BACKGROUND or bottom_color == BACKGROUND or top_color == bottom_color:
        return False
    if not _rectangles_only(top == top_color):
        return False
    if not _rectangles_only(bottom == bottom_color):
        return False

    top_fg = top == top_color
    bottom_fg = bottom == bottom_color
    overlap = top_fg & bottom_fg
    top_only = top_fg & ~bottom_fg
    bottom_only = bottom_fg & ~top_fg
    both_bg = ~top_fg & ~bottom_fg

    if not (overlap.any() and top_only.any() and bottom_only.any() and both_bg.any()):
        return False
    overlap_parities = set(map(int, ((np.argwhere(overlap).sum(axis=1)) % 2)))
    if overlap_parities != {0, 1}:
        return False

    expected = _expected_output(top, bottom, top_color, bottom_color)
    try:
        actual = main(inp)
    except AssertionError:
        return False
    if not np.array_equal(actual, expected):
        return False

    rows, cols = np.indices(expected.shape)
    if not np.all(actual[top_only] == top_color):
        return False
    if not np.all(actual[bottom_only] == bottom_color):
        return False
    if not np.all(actual[both_bg] == BACKGROUND):
        return False
    even_overlap = overlap & ((rows + cols) % 2 == 0)
    odd_overlap = overlap & ((rows + cols) % 2 == 1)
    if not np.all(actual[even_overlap] == top_color):
        return False
    if not np.all(actual[odd_overlap] == bottom_color):
        return False

    return True


def _build_candidate() -> np.ndarray | None:
    half_h = random.randint(*HALF_HEIGHT_RANGE)
    width = random.randint(*WIDTH_RANGE)
    lane_count = random.choice(LANE_COUNT_CHOICES)
    lanes = _sample_row_lanes(half_h, lane_count)
    if lanes is None:
        return None

    top_color, bottom_color = random.sample(FOREGROUND_COLORS, 2)
    top = np.zeros((half_h, width), dtype=int)
    bottom = np.zeros((half_h, width), dtype=int)

    lane_types = ["mixed", "top", "bottom"]
    if lane_count > 3:
        lane_types.extend(random.choices(["mixed", "top", "bottom"], weights=[2, 1, 1], k=lane_count - 3))
    random.shuffle(lane_types)

    for row_span, lane_type in zip(lanes, lane_types):
        if lane_type == "mixed":
            top_span, bottom_span = _sample_mixed_spans(width)
            _paint_rect(top, row_span, top_span, top_color)
            _paint_rect(bottom, row_span, bottom_span, bottom_color)
        elif lane_type == "top":
            _paint_rect(top, row_span, _sample_span(width, 2, 8), top_color)
        else:
            _paint_rect(bottom, row_span, _sample_span(width, 2, 8), bottom_color)

    inp = np.vstack([top, bottom])
    return inp if _semantic_check(inp) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        candidate = _build_candidate()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input for puzzle 5")
