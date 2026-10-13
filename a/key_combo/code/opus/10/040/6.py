from __future__ import annotations

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

_RNG = np.random.default_rng()
_INPUT_COLORS = {BLACK, RED, GREEN}

_BAR_HEIGHT_RANGE = (8, 13)  # high is exclusive
_WIDTH_FLOOR = 18
_WIDTH_EXTRA_RANGE = (7, 16)
_TOP_SLACK_RANGE = (0, 5)
_BOTTOM_MARGIN_RANGE = (1, 8)
_TWO_MARKER_PROB = 0.5
_MAX_GENERATION_ATTEMPTS = 400


def _contiguous_run(mask: np.ndarray) -> tuple[int, int]:
    rows = np.flatnonzero(mask)
    assert rows.size > 0, "expected a non-empty edge bar"
    start = int(rows[0])
    end = int(rows[-1])
    assert end - start + 1 == int(rows.size), "edge bar must be contiguous"
    return start, end


def _parse_input(grid: np.ndarray) -> tuple[int, int, list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert height <= 30 and width <= 30, "grid must fit ARC bounds"
    assert set(map(int, np.unique(g))) <= _INPUT_COLORS, "unexpected input colors"
    assert width >= 3, "need room for two edge bars and interior"
    assert np.all(g[:, 1:-1] == BLACK), "interior must be empty in the input"

    left = g[:, 0]
    right = g[:, width - 1]
    left_mask = left != BLACK
    right_mask = right != BLACK
    left_top, left_bottom = _contiguous_run(left_mask)
    right_top, right_bottom = _contiguous_run(right_mask)
    assert left_top == right_top and left_bottom == right_bottom, "bars must be top-aligned with equal height"
    assert np.all(np.isin(left[left_top : left_bottom + 1], [RED, GREEN])), "left bar may only contain red and green"
    assert np.all(right[right_top : right_bottom + 1] == RED), "right bar must stay red"
    assert np.all(left[:left_top] == BLACK) and np.all(left[left_bottom + 1 :] == BLACK)
    assert np.all(right[:right_top] == BLACK) and np.all(right[right_bottom + 1 :] == BLACK)

    marker_rows = np.flatnonzero(left == GREEN)
    assert 1 <= len(marker_rows) <= 2, "expected one or two green markers"
    offsets = [int(row - left_top) for row in marker_rows]
    assert len(set(offsets)) == len(offsets), "marker rows must be distinct"
    return left_top, left_bottom - left_top + 1, sorted(offsets)


def _paint_wedge(out: np.ndarray, top: int, bar_height: int, offset: int) -> None:
    width = bar_height - 2 * offset
    if width <= 0:
        return

    center_row = top + offset
    right_col = int(out.shape[1] - 1)
    radius = (width - 1) // 2
    for layer in range(radius + 1):
        layer_width = width - 2 * layer
        if layer_width <= 0:
            break
        color = GREEN if layer % 2 == 0 else YELLOW
        rows = [center_row] if layer == 0 else [center_row - layer, center_row + layer]
        col_start = right_col - layer_width
        col_stop = right_col
        assert 0 <= col_start < col_stop <= right_col, "wedge would leave the grid"
        for row in rows:
            assert 0 <= row < out.shape[0], "wedge would leave the grid"
            span = out[row, col_start:col_stop]
            fill_mask = span == BLACK
            span[fill_mask] = color
            out[row, col_start:col_stop] = span


def _expected_output(grid: np.ndarray, top: int, bar_height: int, offsets: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for offset in offsets:
        out[top + offset, 0] = YELLOW
    for offset in sorted(offsets):
        _paint_wedge(out, top, bar_height, offset)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor left-bar markers and map their row offsets to shrinking two-color wedges emitted from the opposite edge bar.

    Input structure:
    - The grid has a black background, a left edge bar and a right edge bar sharing the same top row and height.
    - Both bars are red except that one or two green marker pixels replace cells on distinct rows of the left bar.
    - All interior cells are background, so the marker offsets are encoded only by their rows within the left edge bar.

    Rule:
    - Detect the paired edge bars and compute each green marker's offset d from the common top row.
    - Recolor every green marker to yellow.
    - For each offset d, use the same row on the right bar as the wedge center and draw leftward rows of width H-2d, then H-2d-2, and so on on rows R, R±1, R±2, ...
    - Each wedge row starts immediately left of the right bar, alternates green/yellow by layer parity, never paints over red bar cells, and lower-priority larger-d wedges may only fill still-black cells.

    Key insight:
    - The marker's vertical offset inside the left bar simultaneously determines the center row and the initial width of a symmetric shrinking wedge anchored to the opposite edge.
    """
    top, bar_height, offsets = _parse_input(grid)
    return _expected_output(grid, top, bar_height, offsets)


def _nonoverlapping_positive_offsets(bar_height: int) -> list[int]:
    positives = [offset for offset in range(bar_height) if bar_height - 2 * offset > 0]
    assert positives, "bar height must admit at least one wedge"
    return positives


def _sample_case() -> tuple[np.ndarray, dict]:
    bar_height = int(_RNG.integers(*_BAR_HEIGHT_RANGE))
    positive_offsets = _nonoverlapping_positive_offsets(bar_height)
    productive = int(_RNG.choice(positive_offsets))

    offsets = [productive]
    if _RNG.random() < _TWO_MARKER_PROB:
        dead_offsets = list(range((bar_height + 1) // 2, bar_height))
        if dead_offsets:
            offsets.append(int(_RNG.choice(dead_offsets)))
    offsets = sorted(set(offsets))

    width0 = bar_height - 2 * productive
    radius = (width0 - 1) // 2
    top_need = max(0, radius - productive)
    top = top_need + int(_RNG.integers(*_TOP_SLACK_RANGE))
    bottom_margin = int(_RNG.integers(*_BOTTOM_MARGIN_RANGE))
    height = top + bar_height + bottom_margin

    min_width = max(_WIDTH_FLOOR, width0 + _WIDTH_EXTRA_RANGE[0])
    max_width = min(30, width0 + _WIDTH_EXTRA_RANGE[1])
    if min_width > max_width:
        min_width = max(width0 + 3, min(30, min_width))
        max_width = max(min_width, max_width)
    width = int(_RNG.integers(min_width, max_width + 1))

    grid = np.zeros((height, width), dtype=int)
    grid[top : top + bar_height, 0] = RED
    grid[top : top + bar_height, width - 1] = RED
    for offset in offsets:
        grid[top + offset, 0] = GREEN

    meta = {
        "top": top,
        "bar_height": bar_height,
        "offsets": offsets,
        "productive": productive,
    }
    return grid, meta


def _predicted_wedge_cells(top: int, bar_height: int, offset: int, shape: tuple[int, int]) -> dict[tuple[int, int], int]:
    _height, width = map(int, shape)
    right_col = width - 1
    initial_width = bar_height - 2 * offset
    cells: dict[tuple[int, int], int] = {}
    if initial_width <= 0:
        return cells
    center_row = top + offset
    radius = (initial_width - 1) // 2
    for layer in range(radius + 1):
        layer_width = initial_width - 2 * layer
        if layer_width <= 0:
            break
        color = GREEN if layer % 2 == 0 else YELLOW
        rows = [center_row] if layer == 0 else [center_row - layer, center_row + layer]
        for row in rows:
            for col in range(right_col - layer_width, right_col):
                cells[(int(row), int(col))] = color
    return cells


def _semantic_checks(grid: np.ndarray, meta: dict) -> np.ndarray:
    parsed_top, parsed_height, parsed_offsets = _parse_input(grid)
    assert parsed_top == meta["top"], "parsed top row disagrees with generator metadata"
    assert parsed_height == meta["bar_height"], "parsed bar height disagrees with generator metadata"
    assert parsed_offsets == meta["offsets"], "parsed marker offsets disagrees with generator metadata"

    productive_offsets = [offset for offset in meta["offsets"] if meta["bar_height"] - 2 * offset > 0]
    assert productive_offsets == [meta["productive"]], "generator must expose exactly one productive wedge"

    expected = _expected_output(grid, parsed_top, parsed_height, parsed_offsets)
    wedge_cells = _predicted_wedge_cells(parsed_top, parsed_height, meta["productive"], grid.shape)
    changed = {
        (int(r), int(c)): int(expected[r, c])
        for r, c in np.argwhere(expected != grid)
    }
    for offset in parsed_offsets:
        changed.pop((parsed_top + offset, 0), None)
    assert changed == wedge_cells, "output changes must match the intended wedge cells"
    assert int(expected[parsed_top + meta["productive"], grid.shape[1] - 2]) == GREEN, "wedge must start beside the right bar"
    return expected


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid, meta = _sample_case()
        expected = _semantic_checks(grid, meta)
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
