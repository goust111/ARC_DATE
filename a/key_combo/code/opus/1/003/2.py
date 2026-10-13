from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

GENERATION_RETRIES = 400


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _depth_grid(height: int, width: int) -> np.ndarray:
    rows = np.arange(height, dtype=int)[:, None]
    cols = np.arange(width, dtype=int)[None, :]
    return np.minimum(
        np.minimum(rows, height - 1 - rows),
        np.minimum(cols, width - 1 - cols),
    ) + 1


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    height, width = grid.shape
    for col in range(width):
        value = int(grid[0, col])
        if value == bg:
            continue
        if all(int(grid[row, col]) == value for row in range(height)):
            candidates.append((col, value))
    assert len(candidates) == 1, "expected one full-height separator line"
    return candidates[0]


def _find_rectangle(
    grid: np.ndarray, bg: int, separator_col: int
) -> tuple[int, int, int, int, int]:
    line_mask = np.zeros(grid.shape, dtype=bool)
    line_mask[:, separator_col] = True

    candidates: list[tuple[int, int, int, int, int]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        coords = np.argwhere((grid == color) & ~line_mask)
        if coords.size == 0:
            continue
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        area = int(coords.shape[0])
        if area < 9:
            continue
        if c1 >= separator_col:
            continue
        if area != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 : c1 + 1] == color):
            continue
        candidates.append((r0, c0, r1, c1, color))

    assert len(candidates) == 1, "expected one solid rectangle"
    return candidates[0]


def _collect_markers(
    grid: np.ndarray,
    bg: int,
    rect: tuple[int, int, int, int, int],
    separator_col: int,
) -> list[tuple[int, int, int]]:
    height, width = grid.shape
    r0, c0, r1, c1, _rect_color = rect

    forbidden = np.zeros(grid.shape, dtype=bool)
    forbidden[r0 : r1 + 1, c0 : c1 + 1] = True
    forbidden[:, separator_col] = True

    markers = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere((grid != bg) & ~forbidden)]
    assert 3 <= len(markers) <= 5, "expected 3-5 isolated markers"
    assert len({row for row, _col, _color in markers}) == len(markers), "markers must occupy distinct rows"

    expected_foreground = (r1 - r0 + 1) * (c1 - c0 + 1) + height + len(markers)
    assert int(np.count_nonzero(grid != bg)) == expected_foreground, "unexpected foreground structure"

    fg_mask = grid != bg
    for row, col, _color in markers:
        assert r0 <= row <= r1, "marker rows must overlap rectangle rows"
        assert c1 + 1 < col < separator_col, "markers must lie strictly between rectangle and separator"
        for rr in range(max(0, row - 1), min(height, row + 2)):
            for cc in range(max(0, col - 1), min(width, col + 2)):
                if (rr, cc) == (row, col):
                    continue
                assert not fg_mask[rr, cc], "markers must be isolated"

    return markers


def _parse_instance(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = _dominant_color(g)
    separator_col, separator_color = _find_separator(g, bg)
    rect = _find_rectangle(g, bg, separator_col)
    markers = _collect_markers(g, bg, rect, separator_col)

    r0, c0, r1, c1, rect_color = rect
    depths = _depth_grid(r1 - r0 + 1, c1 - c0 + 1)
    row_depths = depths.max(axis=1)

    return {
        "grid": g,
        "bg": int(bg),
        "separator_col": int(separator_col),
        "separator_color": int(separator_color),
        "rect": (int(r0), int(c0), int(r1), int(c1), int(rect_color)),
        "markers": markers,
        "depths": depths,
        "row_depths": row_depths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor a solid rectangle into concentric layers, then cast rightward rays whose lengths come from row-wise layer depth.

    Input structure:
    - The background is a dominant empty color.
    - The left side contains one filled axis-aligned rectangle, and the right side contains one full-height vertical separator line.
    - Between them are 3-5 isolated single-pixel markers on distinct rows, each row lying within the rectangle's vertical span.

    Rule:
    - For each rectangle cell, compute concentric depth as one plus its minimum distance to the rectangle's four sides.
    - Keep depth-1 cells in the rectangle's original color; recolor larger odd depths red and larger even depths green.
    - For each marker, let D be the maximum rectangle depth on that marker's row; paint exactly D cells to its right, truncating at the separator without crossing it.
    - If the D-th painted cell is exactly the separator's left neighbor, recolor the separator on that row to the marker's color; otherwise keep the separator unchanged.

    Key insight:
    - The rectangle's geometry determines a row-wise depth profile, and that profile becomes the control signal for ray length rather than only for recoloring the rectangle itself.
    """
    parsed = _parse_instance(grid)
    g = parsed["grid"]
    separator_col = int(parsed["separator_col"])
    r0, c0, r1, c1, rect_color = parsed["rect"]
    depths = np.asarray(parsed["depths"], dtype=int)
    row_depths = np.asarray(parsed["row_depths"], dtype=int)
    markers = list(parsed["markers"])

    out = g.copy()
    rect_region = out[r0 : r1 + 1, c0 : c1 + 1]
    rect_region[(depths > 1) & (depths % 2 == 1)] = RED
    rect_region[(depths > 1) & (depths % 2 == 0)] = GREEN
    rect_region[depths == 1] = rect_color

    for row, col, color in markers:
        ray_len = int(row_depths[row - r0])
        ray_end = min(col + ray_len, separator_col - 1)
        if col + 1 <= ray_end:
            out[row, col + 1 : ray_end + 1] = color
        if col + ray_len + 1 == separator_col:
            out[row, separator_col] = color

    return out


_DIMENSIONS_BY_MODE = {
    "tall": [(h, w) for h in range(5, 8) for w in range(5, 8) if h > w],
    "wide": [(h, w) for h in range(5, 8) for w in range(5, 8) if w > h],
    "square": [(k, k) for k in range(5, 8)],
}


def _marker_count(rect_height: int) -> int:
    choices = [count for count in (3, 4, 5) if count <= rect_height]
    weights = [1 if count == 3 else 2 for count in choices]
    return int(random.choices(choices, weights=weights, k=1)[0])


def _select_marker_rows(row_depths: list[int], count: int) -> tuple[list[int], int]:
    rect_height = len(row_depths)
    max_depth = max(row_depths)
    center_candidates = [row for row, depth in enumerate(row_depths) if depth == max_depth]
    center_row = center_candidates[len(center_candidates) // 2]

    selected = [0, rect_height - 1, center_row]
    available = [row for row in range(rect_height) if row not in selected]

    intermediate = [row for row in available if 1 < row_depths[row] < max_depth]
    if len(selected) < count and intermediate:
        row = random.choice(intermediate)
        selected.append(row)
        available.remove(row)

    while len(selected) < count:
        used_depths = {row_depths[row] for row in selected}
        weights = [3 if row_depths[row] not in used_depths else 1 for row in available]
        row = int(random.choices(available, weights=weights, k=1)[0])
        selected.append(row)
        available.remove(row)

    random.shuffle(selected)
    return selected, center_row


def _assign_marker_columns(
    rows_local: list[int],
    row_depths: list[int],
    center_row: int,
    max_available_gap: int,
    line_col: int,
) -> list[tuple[int, int, int]]:
    specs = []
    extra_colors = [RED, BLUE]
    random.shuffle(extra_colors)

    for row in rows_local:
        depth = int(row_depths[row])
        if row == center_row:
            gaps = [depth]
            color = BLUE
        else:
            gaps = list(range(depth + 1, max_available_gap + 1))
            color = extra_colors[(row + depth) % 2]
        specs.append({"row": row, "gaps": gaps, "color": color})

    # Force the bottom edge marker to be an exact-hit example and the top edge marker to be a short example.
    for spec in specs:
        if spec["row"] == 0:
            spec["gaps"] = list(range(int(row_depths[0]) + 1, max_available_gap + 1))
        if spec["row"] == len(row_depths) - 1:
            spec["gaps"] = [int(row_depths[-1])]
            spec["color"] = RED

    order = sorted(range(len(specs)), key=lambda idx: len(specs[idx]["gaps"]))
    placed: list[tuple[int, int, int]] = []

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True

        spec = specs[order[pos]]
        gap_options = list(spec["gaps"])
        random.shuffle(gap_options)

        for gap in gap_options:
            col = line_col - int(gap) - 1
            ok = True
            for other_row, other_col, _other_color in placed:
                if abs(spec["row"] - other_row) <= 1 and abs(col - other_col) <= 1:
                    ok = False
                    break
            if not ok:
                continue
            placed.append((int(spec["row"]), int(col), int(spec["color"])))
            if backtrack(pos + 1):
                return True
            placed.pop()
        return False

    assert backtrack(0), "failed to place isolated markers"
    placed.sort()
    return placed


def _build_instance() -> np.ndarray:
    mode = random.choice(tuple(_DIMENSIONS_BY_MODE))
    rect_height, rect_width = random.choice(_DIMENSIONS_BY_MODE[mode])
    depths = _depth_grid(rect_height, rect_width)
    row_depths = [int(value) for value in depths.max(axis=1)]
    rect_max_depth = int(depths.max())

    marker_count = _marker_count(rect_height)
    rows_local, center_row = _select_marker_rows(row_depths, marker_count)

    top_margin = random.randint(1, 5)
    bottom_margin = random.randint(1, 5)
    left_margin = random.randint(1, 4)
    right_margin = random.randint(0, 3)
    max_available_gap = rect_max_depth + random.randint(2, 4)

    grid_height = rect_height + top_margin + bottom_margin
    rect_r0 = top_margin
    rect_c0 = left_margin
    rect_r1 = rect_r0 + rect_height - 1
    rect_c1 = rect_c0 + rect_width - 1
    line_col = rect_c1 + max_available_gap + 3
    grid_width = line_col + 1 + right_margin

    assert grid_height <= 30 and grid_width <= 30

    markers_local = _assign_marker_columns(rows_local, row_depths, center_row, max_available_gap, line_col)

    grid = np.full((grid_height, grid_width), BG, dtype=int)
    grid[:, line_col] = GRAY
    grid[rect_r0 : rect_r1 + 1, rect_c0 : rect_c1 + 1] = YELLOW

    for local_row, col, color in markers_local:
        grid[rect_r0 + local_row, col] = color

    return grid


def _generated_instance_is_good(grid: np.ndarray) -> bool:
    try:
        parsed = _parse_instance(grid)
    except AssertionError:
        return False

    r0, _c0, r1, _c1, _rect_color = parsed["rect"]
    separator_col = int(parsed["separator_col"])
    row_depths = [int(value) for value in parsed["row_depths"]]
    markers = list(parsed["markers"])

    colors = {color for _row, _col, color in markers}
    if colors != {BLUE, RED}:
        return False

    hit_rows = 0
    miss_rows = 0
    depths_seen = set()
    edge_hit = False
    edge_miss = False

    for row, col, _color in markers:
        depth = row_depths[row - r0]
        depths_seen.add(depth)
        gap = separator_col - col - 1
        if gap == depth:
            hit_rows += 1
            if depth == 1:
                edge_hit = True
        elif gap > depth:
            miss_rows += 1
            if depth == 1:
                edge_miss = True
        else:
            return False

    if hit_rows < 2 or miss_rows < 1:
        return False
    if len(depths_seen) < 2:
        return False
    if not edge_hit or not edge_miss:
        return False
    return r1 - r0 + 1 >= 5


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _build_instance()
        if _generated_instance_is_good(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
