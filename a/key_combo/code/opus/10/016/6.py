from __future__ import annotations

import random

import numpy as np


BG = 0
H_BAR_COLOR = 1
V_BAR_COLOR = 3
CROSS_COLOR = 4
FILL_COLOR = 7

EXTRA_COLOR_POOL = (2, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400
_EXTRA_COUNT_CYCLE: list[int] = []


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _distinct_non_black_color_count(grid: np.ndarray) -> int:
    return len({int(value) for value in np.unique(grid) if int(value) != BG})


def _bar_segment_role(grid: np.ndarray, color: int) -> tuple[str, int, tuple[int, int]]:
    coords = np.argwhere(grid == int(color))
    assert 2 <= coords.shape[0] <= 4, "bar segment must have length 2-4"

    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})
    if len(rows) == 1:
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal bar must be contiguous"
        return "horizontal", rows[0], (cols[0], cols[-1])
    if len(cols) == 1:
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical bar must be contiguous"
        return "vertical", cols[0], (rows[0], rows[-1])
    raise AssertionError("bar color must form a straight 1-cell-thick segment")


def _black_components(grid: np.ndarray) -> list[dict[str, object]]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(height):
        for c in range(width):
            if seen[r, c] or int(grid[r, c]) != BG:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            min_r = max_r = r
            min_c = max_c = c
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                min_r = min(min_r, rr)
                max_r = max(max_r, rr)
                min_c = min(min_c, cc)
                max_c = max(max_c, cc)
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < height and 0 <= nc < width and not seen[nr, nc] and int(grid[nr, nc]) == BG:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"cells": cells, "bbox": (min_r, min_c, max_r, max_c)})
    return components


def _sorted_black_components(grid: np.ndarray) -> list[dict[str, object]]:
    return sorted(_black_components(grid), key=lambda comp: tuple(comp["bbox"][:2]))


def _extended_cross(grid: np.ndarray, h_row: int, v_col: int) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    out[h_row, :] = H_BAR_COLOR
    out[:, v_col] = V_BAR_COLOR
    out[h_row, v_col] = CROSS_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend the blue/green bar segments into a full cross, then periodically fill cross-partitioned regions by color-count period.

    Input structure:
    - A black background contains exactly one blue horizontal 1-cell-thick segment and exactly one green vertical 1-cell-thick segment.
    - Their row/column crossing cell is black in the input, so the two segments identify a latent cross without already intersecting.
    - There are also 1-3 isolated singleton pixels in distinct non-black colors, none lying on the horizontal row or vertical column.

    Rule:
    - Extend the blue segment across its entire row and the green segment across its entire column, then recolor their unique crossing cell yellow.
    - Count how many distinct non-black colors were present in the original input; call this number N.
    - Find all 4-connected black regions after the cross is drawn, sort them by the top-left corner of each region's bounding box in reading order, and flood-fill regions with indices 0, N, 2N, ... using orange.

    Key insight:
    - The short bar segments determine the geometric partition, while the otherwise isolated singleton colors only matter through the distinct-color count that sets the fill period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h_role, h_row, _ = _bar_segment_role(g, H_BAR_COLOR)
    v_role, v_col, _ = _bar_segment_role(g, V_BAR_COLOR)
    assert h_role == "horizontal", "blue cells must form the horizontal bar"
    assert v_role == "vertical", "green cells must form the vertical bar"
    assert int(g[h_row, v_col]) == BG, "bar crossing must start as background"

    out = _extended_cross(g, h_row, v_col)
    period = _distinct_non_black_color_count(g)
    assert period >= 2

    for index, component in enumerate(_sorted_black_components(out)):
        if index % period != 0:
            continue
        for r, c in component["cells"]:
            out[r, c] = FILL_COLOR
    return out


def _quadrant_boxes(height: int, width: int, h_row: int, v_col: int) -> list[tuple[int, int, int, int]]:
    return [
        (0, 0, h_row - 1, v_col - 1),
        (0, v_col + 1, h_row - 1, width - 1),
        (h_row + 1, 0, height - 1, v_col - 1),
        (h_row + 1, v_col + 1, height - 1, width - 1),
    ]


def _candidate_cells_in_box(box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    inner_r0 = r0 + 1 if r1 - r0 + 1 >= 4 else r0
    inner_c0 = c0 + 1 if c1 - c0 + 1 >= 4 else c0
    inner_r1 = r1 - 1 if r1 - r0 + 1 >= 4 else r1
    inner_c1 = c1 - 1 if c1 - c0 + 1 >= 4 else c1
    return [(r, c) for r in range(inner_r0, inner_r1 + 1) for c in range(inner_c0, inner_c1 + 1)]


def _sample_extra_count() -> int:
    if not _EXTRA_COUNT_CYCLE:
        _EXTRA_COUNT_CYCLE.extend([1, 2, 3])
        random.shuffle(_EXTRA_COUNT_CYCLE)
    return _EXTRA_COUNT_CYCLE.pop()


def _choose_extra_positions(
    boxes: list[tuple[int, int, int, int]], extra_count: int
) -> list[tuple[int, int]] | None:
    region_indices = random.sample(range(len(boxes)), k=extra_count)
    chosen: list[tuple[int, int]] = []
    for region_index in region_indices:
        candidates = _candidate_cells_in_box(boxes[region_index])
        random.shuffle(candidates)
        pick = None
        for cell in candidates:
            if all(max(abs(cell[0] - r), abs(cell[1] - c)) > 1 for r, c in chosen):
                pick = cell
                break
        if pick is None:
            return None
        chosen.append(pick)
    return chosen


def _segment_starts(length: int, span: int, forbidden_index: int) -> list[int]:
    starts: list[int] = []
    for start in range(1, span - length):
        if forbidden_index < start or forbidden_index > start + length - 1:
            starts.append(start)
    return starts


def _singleton_cells(grid: np.ndarray) -> list[tuple[int, int]]:
    mask = (grid != BG) & (grid != H_BAR_COLOR) & (grid != V_BAR_COLOR)
    return [tuple(map(int, cell)) for cell in np.argwhere(mask)]


def _semantic_expected_output(grid: np.ndarray) -> np.ndarray:
    h_role, h_row, _ = _bar_segment_role(grid, H_BAR_COLOR)
    v_role, v_col, _ = _bar_segment_role(grid, V_BAR_COLOR)
    assert h_role == "horizontal"
    assert v_role == "vertical"

    expected = _extended_cross(grid, h_row, v_col)
    components = _sorted_black_components(expected)
    boxes = [tuple(comp["bbox"]) for comp in components]
    target_boxes = _quadrant_boxes(grid.shape[0], grid.shape[1], h_row, v_col)
    assert boxes == target_boxes, "cross regions must be the four quadrants created by the extension lines"

    period = _distinct_non_black_color_count(grid)
    for index, component in enumerate(components):
        if index % period != 0:
            continue
        for r, c in component["cells"]:
            expected[r, c] = FILL_COLOR
    return expected


def _grid_is_semantically_valid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    try:
        h_role, h_row, h_span = _bar_segment_role(grid, H_BAR_COLOR)
        v_role, v_col, v_span = _bar_segment_role(grid, V_BAR_COLOR)
    except AssertionError:
        return False

    if h_role != "horizontal" or v_role != "vertical":
        return False
    if h_row <= 1 or h_row >= grid.shape[0] - 2 or v_col <= 1 or v_col >= grid.shape[1] - 2:
        return False
    if h_span[0] <= 0 or h_span[1] >= grid.shape[1] - 1:
        return False
    if v_span[0] <= 0 or v_span[1] >= grid.shape[0] - 1:
        return False
    if int(grid[h_row, v_col]) != BG:
        return False

    extras = _singleton_cells(grid)
    if not (1 <= len(extras) <= 3):
        return False
    seen_colors: set[int] = set()
    for r, c in extras:
        color = int(grid[r, c])
        if color in seen_colors:
            return False
        seen_colors.add(color)
        if r == h_row or c == v_col or int(np.count_nonzero(grid == color)) != 1:
            return False
    for i, (r0, c0) in enumerate(extras):
        for r1, c1 in extras[i + 1 :]:
            if max(abs(r0 - r1), abs(c0 - c1)) <= 1:
                return False

    if _distinct_non_black_color_count(grid) != 2 + len(extras):
        return False

    try:
        expected = _semantic_expected_output(grid)
    except AssertionError:
        return False

    if len(_sorted_black_components(_extended_cross(grid, h_row, v_col))) != 4:
        return False

    actual = main(grid)
    return bool(np.array_equal(actual, expected))


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(10, 16)
    width = random.randint(10, 16)

    h_row = random.randint(2, height - 3)
    v_col = random.randint(2, width - 3)
    h_len = random.randint(2, 4)
    v_len = random.randint(2, 4)

    h_starts = _segment_starts(h_len, width, v_col)
    v_starts = _segment_starts(v_len, height, h_row)
    if not h_starts or not v_starts:
        return None

    grid = np.zeros((height, width), dtype=int)

    h_start = random.choice(h_starts)
    v_start = random.choice(v_starts)
    grid[h_row, h_start : h_start + h_len] = H_BAR_COLOR
    grid[v_start : v_start + v_len, v_col] = V_BAR_COLOR

    extra_count = _sample_extra_count()
    boxes = _quadrant_boxes(height, width, h_row, v_col)
    positions = _choose_extra_positions(boxes, extra_count)
    if positions is None:
        return None

    extra_colors = random.sample(list(EXTRA_COLOR_POOL), k=extra_count)
    for (r, c), color in zip(positions, extra_colors):
        grid[r, c] = int(color)

    if not _grid_is_semantically_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
