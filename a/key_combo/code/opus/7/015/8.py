import random
from typing import Any

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
TEAL = 8
MAROON = 9

GEN_ATTEMPTS = 500


def _find_reference_rows(grid: np.ndarray) -> tuple[int, int, list[int]]:
    candidates: list[tuple[int, int, list[int]]] = []
    for red_row in range(grid.shape[0] - 1):
        red_cols = np.flatnonzero(grid[red_row] == RED)
        if len(red_cols) < 3:
            continue
        if np.any((grid[red_row] != BLACK) & (grid[red_row] != RED)):
            continue
        grey_row = red_row + 1
        if np.any((grid[grey_row] != BLACK) & (grid[grey_row] != GREY)):
            continue
        grey_cols = np.flatnonzero(grid[grey_row] == GREY)
        if np.array_equal(red_cols, grey_cols):
            candidates.append((red_row, grey_row, red_cols.astype(int).tolist()))
    assert len(candidates) == 1, f"expected one red/grey baseline pair, got {len(candidates)}"
    return candidates[0]


def _parse_bar_column(grid: np.ndarray, red_row: int, grey_row: int, col: int) -> dict[str, Any]:
    top_slice = grid[:red_row, col]
    teal_mask = np.isin(top_slice, (TEAL, YELLOW))
    teal_rows = np.flatnonzero(teal_mask)
    assert len(teal_rows) >= 2, "each bar needs a teal/yellow top segment"
    top = int(teal_rows[0])
    assert teal_rows[-1] == red_row - 1, "top segment must touch the red row"
    expected_rows = np.arange(top, red_row, dtype=int)
    assert np.array_equal(teal_rows, expected_rows), "top segment must be contiguous"
    height = red_row - top

    yellow_rows = np.flatnonzero(top_slice == YELLOW)
    assert len(yellow_rows) <= 1, "a bar may have at most one embedded marker"
    marker_offset = None if len(yellow_rows) == 0 else int(yellow_rows[0] - top)

    lower_slice = grid[grey_row + 1 :, col]
    blue_rows = np.flatnonzero(lower_slice == BLUE)
    assert len(blue_rows) >= 1, "each bar needs a blue lower segment"
    blue_top = int(grey_row + 1 + blue_rows[0])
    blue_bottom = int(grey_row + 1 + blue_rows[-1])
    expected_blue = np.arange(blue_top, blue_bottom + 1, dtype=int) - (grey_row + 1)
    assert np.array_equal(blue_rows, expected_blue), "blue segment must be contiguous"

    allowed = {BLACK, TEAL, YELLOW, BLUE}
    assert set(map(int, np.unique(grid[:, col]))).issubset(allowed), "bar columns only use puzzle colors"

    return {
        "col": int(col),
        "teal_top": int(top),
        "teal_height": int(height),
        "marker_offset": marker_offset,
        "blue_top": int(blue_top),
        "blue_height": int(blue_bottom - blue_top + 1),
        "blue_bottom": int(blue_bottom),
    }


def _parse_scene(grid: np.ndarray) -> dict[str, Any]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    red_row, grey_row, marker_cols = _find_reference_rows(g)

    bar_cols = [marker_cols[i] + 1 for i in range(len(marker_cols) - 1) if marker_cols[i + 1] - marker_cols[i] == 2]
    assert len(bar_cols) >= 3, "expected several alternating bar columns"

    bars = [_parse_bar_column(g, red_row, grey_row, col) for col in bar_cols]
    assert all(2 <= bar["teal_height"] <= 6 for bar in bars)
    assert all(1 <= bar["blue_height"] <= 4 for bar in bars)
    return {
        "grid": g,
        "red_row": int(red_row),
        "grey_row": int(grey_row),
        "marker_cols": marker_cols,
        "bar_cols": bar_cols,
        "bars": bars,
        "h_max": max(bar["teal_height"] for bar in bars),
    }


def _normalized_grid(scene: dict[str, Any]) -> tuple[np.ndarray, list[tuple[int, int]]]:
    g = scene["grid"]
    red_row = scene["red_row"]
    grey_row = scene["grey_row"]
    h_max = scene["h_max"]
    out = g.copy()
    markers: list[tuple[int, int]] = []

    for bar in scene["bars"]:
        col = bar["col"]
        out[:, col] = BLACK

        teal_top = red_row - h_max
        out[teal_top:red_row, col] = TEAL
        if bar["marker_offset"] is not None:
            marker_row = teal_top + bar["marker_offset"]
            out[marker_row, col] = YELLOW
            markers.append((int(marker_row), int(col)))

        blue_top = grey_row + 1
        blue_bottom = blue_top + bar["blue_height"] - 1
        out[blue_top : blue_bottom + 1, col] = BLUE

        if blue_bottom < bar["blue_bottom"]:
            out[blue_bottom + 1 : bar["blue_bottom"] + 1, col] = MAROON

    return out, sorted(markers)


def _adjacent_same_row_pairs(points: list[tuple[int, int]]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    grouped: dict[int, list[int]] = {}
    for r, c in points:
        grouped.setdefault(int(r), []).append(int(c))

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for row, cols in grouped.items():
        cols.sort()
        for i in range(len(cols) - 1):
            pairs.append(((row, cols[i]), (row, cols[i + 1])))
    return pairs


def _draw_connectors(grid: np.ndarray, markers: list[tuple[int, int]]) -> np.ndarray:
    out = grid.copy()
    for (row, col0), (_, col1) in _adjacent_same_row_pairs(markers):
        if np.all(out[row, col0 + 1 : col1] == BLACK):
            out[row, col0 + 1 : col1] = GREEN
    return out


def _input_markers(scene: dict[str, Any]) -> list[tuple[int, int]]:
    markers: list[tuple[int, int]] = []
    for bar in scene["bars"]:
        if bar["marker_offset"] is None:
            continue
        row = bar["teal_top"] + bar["marker_offset"]
        markers.append((int(row), int(bar["col"])))
    return sorted(markers)


def _clear_pairs_on_grid(points: list[tuple[int, int]], grid: np.ndarray) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for (row, col0), (_, col1) in _adjacent_same_row_pairs(points):
        if np.all(grid[row, col0 + 1 : col1] == BLACK):
            pairs.append(((row, col0), (row, col1)))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """Normalize uneven teal bars, move their embedded markers with the normalized tops, then connect newly aligned markers.

    Input structure:
    - The background is black.
    - One red row sits immediately above one grey row, each containing matching isolated baseline pixels in alternating columns.
    - In the columns between neighboring baseline pixels, each bar has a contiguous teal top segment ending at the red row and a contiguous blue lower segment somewhere below the grey row; some teal segments contain one embedded yellow marker pixel.

    Rule:
    - Find the maximum teal-segment height across all bar columns.
    - Rebuild every teal segment so it has that maximum height while keeping its bottom flush with the red row; an embedded yellow marker keeps the same offset from the teal segment's top, so its absolute row may change.
    - Move every blue segment upward so it starts immediately below the grey row, keep its height unchanged, and fill the vacated cells down to the original blue bottom with maroon.
    - After normalization, collect all yellow markers; for each row, sort them by column and only consider adjacent pairs. If the horizontal gap between an adjacent pair is entirely black, draw a green connector across that gap without overwriting existing non-black cells.

    Key insight:
    - The yellow marker is anchored to its teal segment rather than to the red baseline, so equal top-offsets become equal rows only after every teal segment is normalized to the same height.
    """
    scene = _parse_scene(np.asarray(grid, dtype=int))
    normalized, markers = _normalized_grid(scene)
    return _draw_connectors(normalized, markers)


def _global_bar_layout(group_sizes: list[int], left_margin: int, group_gap: int) -> tuple[list[int], list[int]]:
    marker_cols: list[int] = []
    bar_cols: list[int] = []
    start = left_margin
    for gi, bars_in_group in enumerate(group_sizes):
        cols = [start + 2 * k for k in range(bars_in_group + 1)]
        marker_cols.extend(cols)
        bar_cols.extend(start + 2 * k + 1 for k in range(bars_in_group))
        if gi < len(group_sizes) - 1:
            start = cols[-1] + group_gap + 1
    return marker_cols, bar_cols


def _choose_group_sizes(total_bars: int) -> list[int]:
    if total_bars >= 5 and random.random() < 0.45:
        split = random.randint(2, total_bars - 2)
        return [split, total_bars - split]
    return [total_bars]


def _make_attempt() -> np.ndarray | None:
    total_bars = random.randint(4, 7)
    group_sizes = _choose_group_sizes(total_bars)
    group_gap = random.randint(3, 6) if len(group_sizes) == 2 else 0
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)
    marker_cols, bar_cols = _global_bar_layout(group_sizes, left_margin, group_gap)
    width = marker_cols[-1] + 1 + right_margin
    if width > 30:
        return None

    teal_heights = [random.randint(2, 6) for _ in range(total_bars)]
    h_max = max(teal_heights)
    if all(h == h_max for h in teal_heights):
        teal_heights[random.randrange(total_bars)] -= 1
        h_max = max(teal_heights)

    blue_heights = [random.randint(1, 4) for _ in range(total_bars)]
    blue_shifts = [random.randint(0, 2) for _ in range(total_bars)]
    if all(shift == 0 for shift in blue_shifts):
        blue_shifts[random.randrange(total_bars)] = 1

    top_margin = random.randint(1, 3)
    red_row = h_max + top_margin
    bottom_need = max(1 + shift + height for shift, height in zip(blue_shifts, blue_heights))
    height = red_row + 2 + bottom_need + random.randint(1, 3)
    if height > 30:
        return None

    target_marks = max(2, min(total_bars - 1, int(round(total_bars * random.uniform(0.4, 0.6)))))
    bar_order = list(range(total_bars))
    key_left = random.randint(0, total_bars - 2)
    key_right = key_left + 1

    offset_limit = min(teal_heights[key_left], teal_heights[key_right]) - 1
    shared_offset = random.randint(0, offset_limit)
    marked_offsets: dict[int, int] = {key_left: shared_offset, key_right: shared_offset}

    extra_candidates = [idx for idx in bar_order if idx not in marked_offsets]
    random.shuffle(extra_candidates)
    while len(marked_offsets) < target_marks and extra_candidates:
        idx = extra_candidates.pop()
        if random.random() < 0.35 and len(marked_offsets) >= 2:
            marked_offsets[idx] = shared_offset if shared_offset < teal_heights[idx] else random.randint(0, teal_heights[idx] - 1)
        else:
            marked_offsets[idx] = random.randint(0, teal_heights[idx] - 1)

    if len(marked_offsets) / total_bars < 0.4 or len(marked_offsets) / total_bars > 0.61:
        return None

    if teal_heights[key_left] == teal_heights[key_right]:
        return None

    grid = np.zeros((height, width), dtype=int)

    for c in marker_cols:
        grid[red_row, c] = RED
        grid[red_row + 1, c] = GREY

    for idx, col in enumerate(bar_cols):
        teal_top = red_row - teal_heights[idx]
        grid[teal_top:red_row, col] = TEAL
        if idx in marked_offsets:
            grid[teal_top + marked_offsets[idx], col] = YELLOW

        blue_top = red_row + 2 + blue_shifts[idx]
        blue_bottom = blue_top + blue_heights[idx]
        grid[blue_top:blue_bottom, col] = BLUE

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        scene = _parse_scene(grid)
        out = main(grid)
    except AssertionError:
        return False

    if out.shape != grid.shape:
        return False
    if np.array_equal(out, grid):
        return False

    if max(grid.shape) > 30 or max(out.shape) > 30:
        return False

    if not any(bar["teal_height"] < scene["h_max"] for bar in scene["bars"]):
        return False
    if not any(bar["blue_top"] > scene["grey_row"] + 1 for bar in scene["bars"]):
        return False
    if not np.any(out == MAROON):
        return False
    if not np.any(out == GREEN):
        return False

    input_markers = _input_markers(scene)
    output_base, output_markers = _normalized_grid(scene)
    input_pairs = _clear_pairs_on_grid(input_markers, grid)
    output_pairs = _clear_pairs_on_grid(output_markers, output_base)

    if input_pairs:
        return False
    if not output_pairs:
        return False

    input_by_col = {c: r for r, c in input_markers}
    if not any(input_by_col[c0] != input_by_col[c1] for (_, c0), (_, c1) in output_pairs):
        return False

    painted = out == GREEN
    expected = _draw_connectors(output_base, output_markers)
    if not np.array_equal(out, expected):
        return False
    if np.count_nonzero(painted) < 1:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _make_attempt()
        if grid is not None and _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid normalized-bar alignment puzzle")
