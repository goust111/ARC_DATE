from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _horizontal(color: int, row: int, c0: int, c1: int) -> dict[str, int | str]:
    assert c0 <= c1
    return {"color": int(color), "orientation": "h", "row": int(row), "start": int(c0), "end": int(c1)}


def _vertical(color: int, col: int, r0: int, r1: int) -> dict[str, int | str]:
    assert r0 <= r1
    return {"color": int(color), "orientation": "v", "col": int(col), "start": int(r0), "end": int(r1)}


def _covers(path: dict[str, int | str], row: int, col: int) -> bool:
    if path["orientation"] == "h":
        return int(path["row"]) == row and int(path["start"]) <= col <= int(path["end"])
    return int(path["col"]) == col and int(path["start"]) <= row <= int(path["end"])


def _visible_paths(grid: np.ndarray) -> dict[int, dict[str, int | str]]:
    paths: dict[int, dict[str, int | str]] = {}
    colors = [int(color) for color in np.unique(grid) if int(color) not in {BG, GREY}]
    for color in colors:
        coords = np.argwhere(grid == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1:
            paths[color] = _horizontal(color, int(rows[0]), int(cols.min()), int(cols.max()))
        elif len(cols) == 1:
            paths[color] = _vertical(color, int(cols[0]), int(rows.min()), int(rows.max()))
        else:
            raise AssertionError(f"color {color} does not form a single visible straight path")
    return paths


def _hidden_horizontal_endpoint(
    grid: np.ndarray,
    paths: dict[int, dict[str, int | str]],
    row: int,
    cross_col: int,
    marker_col: int,
) -> bool:
    height, width = grid.shape
    if not (0 <= row < height and 0 <= cross_col < width and 0 <= marker_col < width):
        return False
    if int(grid[row, marker_col]) != GREY:
        return False
    other = int(grid[row, cross_col])
    if other in {BG, GREY}:
        return False
    other_path = paths.get(other)
    return bool(
        other_path
        and other_path["orientation"] == "v"
        and int(other_path["start"]) <= row <= int(other_path["end"])
    )


def _hidden_vertical_endpoint(
    grid: np.ndarray,
    paths: dict[int, dict[str, int | str]],
    cross_row: int,
    marker_row: int,
    col: int,
) -> bool:
    height, width = grid.shape
    if not (0 <= cross_row < height and 0 <= marker_row < height and 0 <= col < width):
        return False
    if int(grid[marker_row, col]) != GREY:
        return False
    other = int(grid[cross_row, col])
    if other in {BG, GREY}:
        return False
    other_path = paths.get(other)
    return bool(
        other_path
        and other_path["orientation"] == "h"
        and int(other_path["start"]) <= col <= int(other_path["end"])
    )


def _extend_hidden_endpoints(grid: np.ndarray, paths: dict[int, dict[str, int | str]]) -> None:
    for path in paths.values():
        if path["orientation"] == "h":
            row = int(path["row"])
            left = int(path["start"]) - 1
            if _hidden_horizontal_endpoint(grid, paths, row, left, left - 1):
                path["start"] = left
            right = int(path["end"]) + 1
            if _hidden_horizontal_endpoint(grid, paths, row, right, right + 1):
                path["end"] = right
        else:
            col = int(path["col"])
            top = int(path["start"]) - 1
            if _hidden_vertical_endpoint(grid, paths, top, top - 1, col):
                path["start"] = top
            bottom = int(path["end"]) + 1
            if _hidden_vertical_endpoint(grid, paths, bottom, bottom + 1, col):
                path["end"] = bottom


def _split_paths(paths: dict[int, dict[str, int | str]]) -> tuple[list[dict[str, int | str]], list[dict[str, int | str]]]:
    horizontals = [path for path in paths.values() if path["orientation"] == "h"]
    verticals = [path for path in paths.values() if path["orientation"] == "v"]
    return horizontals, verticals


def _crossings(
    horizontals: list[dict[str, int | str]],
    verticals: list[dict[str, int | str]],
) -> dict[tuple[int, int], tuple[dict[str, int | str], dict[str, int | str]]]:
    out: dict[tuple[int, int], tuple[dict[str, int | str], dict[str, int | str]]] = {}
    for h_path in horizontals:
        row = int(h_path["row"])
        for v_path in verticals:
            col = int(v_path["col"])
            if int(h_path["start"]) <= col <= int(h_path["end"]) and int(v_path["start"]) <= row <= int(v_path["end"]):
                out[(row, col)] = (h_path, v_path)
    return out


def _marker_top_direction(grid: np.ndarray, row: int, col: int) -> str:
    height, width = grid.shape
    directions: list[str] = []
    if row > 0 and int(grid[row - 1, col]) == GREY:
        directions.append("v")
    if row + 1 < height and int(grid[row + 1, col]) == GREY:
        directions.append("v")
    if col > 0 and int(grid[row, col - 1]) == GREY:
        directions.append("h")
    if col + 1 < width and int(grid[row, col + 1]) == GREY:
        directions.append("h")
    if not directions:
        return "h"
    assert len(directions) == 1, "each crossing may have at most one grey overpass marker"
    return directions[0]


def _render(
    shape: tuple[int, int],
    horizontals: list[dict[str, int | str]],
    verticals: list[dict[str, int | str]],
    top_by_cross: dict[tuple[int, int], str],
    markers: set[tuple[int, int]] | None,
) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for path in horizontals:
        row = int(path["row"])
        out[row, int(path["start"]) : int(path["end"]) + 1] = int(path["color"])
    for path in verticals:
        col = int(path["col"])
        out[int(path["start"]) : int(path["end"]) + 1, col] = int(path["color"])
    for (row, col), top in top_by_cross.items():
        h_path, v_path = _crossings(horizontals, verticals)[(row, col)]
        out[row, col] = int(h_path["color"] if top == "h" else v_path["color"])
    if markers:
        for row, col in markers:
            assert out[row, col] == BG
            out[row, col] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden straight paths and redraw every crossing using nearby grey markers.

    Grey above/below a crossing means the vertical path is on top.
    Grey left/right means the horizontal path is on top.
    When there is no grey marker, horizontal wins by default.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert g.shape[0] >= 5 and g.shape[1] >= 5
    assert int(np.min(g)) >= 0

    paths = _visible_paths(g)
    _extend_hidden_endpoints(g, paths)
    horizontals, verticals = _split_paths(paths)
    assert horizontals and verticals, "expected both horizontal and vertical paths"

    for path in paths.values():
        if path["orientation"] == "h":
            assert int(path["start"]) == 0 or int(path["end"]) == g.shape[1] - 1
        else:
            assert int(path["start"]) == 0 or int(path["end"]) == g.shape[0] - 1

    crosses = _crossings(horizontals, verticals)
    top_by_cross = {pos: _marker_top_direction(g, *pos) for pos in crosses}

    for row, col in np.argwhere((g != BG) & (g != GREY)):
        cell = int(g[row, col])
        assert any(int(path["color"]) == cell and _covers(path, int(row), int(col)) for path in paths.values())

    out = np.full_like(g, BG)
    for path in horizontals:
        row = int(path["row"])
        out[row, int(path["start"]) : int(path["end"]) + 1] = int(path["color"])
    for path in verticals:
        col = int(path["col"])
        out[int(path["start"]) : int(path["end"]) + 1, col] = int(path["color"])
    for (row, col), top in top_by_cross.items():
        h_path, v_path = crosses[(row, col)]
        out[row, col] = int(h_path["color"] if top == "h" else v_path["color"])
    return out


# --------------- generator ---------------

SIZE_RANGE = (12, 18)
COUNT_RANGE = (2, 4)
MIN_GAP = 2
MAX_GENERATION_ATTEMPTS = 400


def _sample_spaced_positions(lo: int, hi: int, count: int, gap: int) -> list[int]:
    span = hi - lo + 1
    compressed_span = span - (count - 1) * (gap - 1)
    if compressed_span < count:
        raise ValueError("not enough room for spaced positions")
    picks = sorted(random.sample(range(compressed_span), count))
    return [lo + picks[i] + i * (gap - 1) for i in range(count)]


def _build_scene() -> tuple[
    tuple[int, int],
    list[dict[str, int | str]],
    list[dict[str, int | str]],
    dict[tuple[int, int], str],
    dict[tuple[int, int], str],
    set[tuple[int, int]],
]:
    h_count = random.randint(*COUNT_RANGE)
    v_count = random.randint(*COUNT_RANGE)
    height_min = max(SIZE_RANGE[0], h_count + (h_count - 1) * (MIN_GAP - 1) + 4)
    width_min = max(SIZE_RANGE[0], v_count + (v_count - 1) * (MIN_GAP - 1) + 4)
    height = random.randint(height_min, SIZE_RANGE[1])
    width = random.randint(width_min, SIZE_RANGE[1])

    rows = _sample_spaced_positions(2, height - 3, h_count, MIN_GAP)
    cols = _sample_spaced_positions(2, width - 3, v_count, MIN_GAP)
    colors = random.sample(PATH_COLORS, h_count + v_count)
    h_colors = colors[:h_count]
    v_colors = colors[h_count:]

    h_mark_idx = random.randrange(h_count)
    h_full_idx = random.choice([i for i in range(h_count) if i != h_mark_idx])
    v_full_idx = random.randrange(v_count)
    v_mark_idx = random.choice([i for i in range(v_count) if i != v_full_idx])

    horizontals = [_horizontal(h_colors[i], rows[i], 0, width - 1) for i in range(h_count)]
    verticals = [_vertical(v_colors[i], cols[i], 0, height - 1) for i in range(v_count)]

    h_mark_row = rows[h_mark_idx]
    h_mark_col = cols[v_full_idx]
    v_mark_row = rows[h_full_idx]
    v_mark_col = cols[v_mark_idx]

    if v_mark_col < h_mark_col:
        horizontals[h_mark_idx] = _horizontal(h_colors[h_mark_idx], h_mark_row, h_mark_col, width - 1)
        h_marker = (h_mark_row, h_mark_col - 1)
    else:
        horizontals[h_mark_idx] = _horizontal(h_colors[h_mark_idx], h_mark_row, 0, h_mark_col)
        h_marker = (h_mark_row, h_mark_col + 1)

    if h_mark_row < v_mark_row:
        verticals[v_mark_idx] = _vertical(v_colors[v_mark_idx], v_mark_col, v_mark_row, height - 1)
        v_marker = (v_mark_row - 1, v_mark_col)
    else:
        verticals[v_mark_idx] = _vertical(v_colors[v_mark_idx], v_mark_col, 0, v_mark_row)
        v_marker = (v_mark_row + 1, v_mark_col)

    markers = {h_marker, v_marker}
    if len(markers) != 2:
        raise ValueError("markers overlap")

    crosses = _crossings(horizontals, verticals)
    marked_h = (h_mark_row, h_mark_col)
    marked_v = (v_mark_row, v_mark_col)
    anchor = (v_mark_row, h_mark_col)
    assert marked_h in crosses and marked_v in crosses and anchor in crosses

    final_top = {pos: "h" for pos in crosses}
    final_top[marked_v] = "v"

    input_top = {pos: random.choice(("h", "v")) for pos in crosses}

    for path in verticals:
        pos = (h_mark_row, int(path["col"]))
        if pos in crosses and pos != marked_h:
            input_top[pos] = "h"
    for path in horizontals:
        pos = (int(path["row"]), v_mark_col)
        if pos in crosses and pos != marked_v:
            input_top[pos] = "v"

    input_top[marked_h] = "v"
    input_top[marked_v] = "h"
    input_top[anchor] = "v"

    shape = (height, width)
    input_grid = _render(shape, horizontals, verticals, input_top, markers)
    output_grid = _render(shape, horizontals, verticals, final_top, None)

    if any(input_grid[row, col] != GREY for row, col in markers):
        raise ValueError("marker placement failed")
    if np.array_equal(input_grid, output_grid):
        raise ValueError("scene is degenerate")
    if not np.array_equal(main(input_grid), output_grid):
        raise ValueError("generated scene does not match solver")
    return shape, horizontals, verticals, final_top, input_top, markers


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            shape, horizontals, verticals, _final_top, input_top, markers = _build_scene()
        except (AssertionError, ValueError):
            continue
        return _render(shape, horizontals, verticals, input_top, markers)
    raise RuntimeError("failed to generate a valid input for puzzle 7")
