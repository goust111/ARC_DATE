from __future__ import annotations

import random

import numpy as np


BG = 0
CROSS_COLOR = 5
PATH_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_MIN = 15
GRID_MAX = 28
PATH_COUNT_RANGE = (2, 4)
MIN_POS_GAP = 3
MAX_GENERATION_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    start: int | None = None
    for idx, value in enumerate(mask.tolist()):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(mask) - 1))
    return runs


def _extract_horizontal_paths(grid: np.ndarray, bg: int) -> list[dict]:
    paths: list[dict] = []
    for row in range(grid.shape[0]):
        for c0, c1 in _runs(grid[row] != bg):
            if c1 - c0 + 1 >= 2:
                paths.append({"row": int(row), "c0": int(c0), "c1": int(c1), "crossings": []})
    return paths


def _extract_vertical_paths(grid: np.ndarray, bg: int) -> list[dict]:
    paths: list[dict] = []
    for col in range(grid.shape[1]):
        for r0, r1 in _runs(grid[:, col] != bg):
            if r1 - r0 + 1 >= 2:
                paths.append({"col": int(col), "r0": int(r0), "r1": int(r1), "crossings": []})
    return paths


def _attach_crossings(
    grid: np.ndarray, bg: int, h_paths: list[dict], v_paths: list[dict]
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], dict[tuple[int, int], int], int]:
    crossing_positions: set[tuple[int, int]] = set()
    h_by_cross: dict[tuple[int, int], int] = {}
    v_by_cross: dict[tuple[int, int], int] = {}

    for hi, h_path in enumerate(h_paths):
        row = int(h_path["row"])
        for vi, v_path in enumerate(v_paths):
            col = int(v_path["col"])
            if h_path["c0"] <= col <= h_path["c1"] and v_path["r0"] <= row <= v_path["r1"]:
                pos = (row, col)
                assert int(grid[row, col]) != bg, "crossing cell may not be background"
                crossing_positions.add(pos)
                h_path["crossings"].append(pos)
                v_path["crossings"].append(pos)
                h_by_cross[pos] = hi
                v_by_cross[pos] = vi

    assert crossing_positions, "expected at least one crossing"
    cross_colors = {int(grid[r, c]) for r, c in crossing_positions}
    assert len(cross_colors) == 1, "all crossings must share one grey marker color"
    cross_color = cross_colors.pop()
    all_cross_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == cross_color)}
    assert all_cross_cells == crossing_positions, "grey cells must be exactly the crossings"

    for h_path in h_paths:
        h_path["crossings"].sort(key=lambda pos: pos[1])
    for v_path in v_paths:
        v_path["crossings"].sort(key=lambda pos: pos[0])
    return crossing_positions, h_by_cross, v_by_cross, cross_color


def _assign_path_colors(grid: np.ndarray, cross_color: int, h_paths: list[dict], v_paths: list[dict]) -> None:
    for h_path in h_paths:
        row = int(h_path["row"])
        cross_cols = {c for _r, c in h_path["crossings"]}
        visible = {
            int(grid[row, col])
            for col in range(int(h_path["c0"]), int(h_path["c1"]) + 1)
            if col not in cross_cols
        }
        assert len(visible) == 1, "horizontal path must keep one visible color"
        color = visible.pop()
        assert color != cross_color, "horizontal path color cannot be the crossing marker"
        h_path["color"] = color

    for v_path in v_paths:
        col = int(v_path["col"])
        cross_rows = {r for r, _c in v_path["crossings"]}
        visible = {
            int(grid[row, col])
            for row in range(int(v_path["r0"]), int(v_path["r1"]) + 1)
            if row not in cross_rows
        }
        assert len(visible) == 1, "vertical path must keep one visible color"
        color = visible.pop()
        assert color != cross_color, "vertical path color cannot be the crossing marker"
        v_path["color"] = color


def _decision_maps(h_paths: list[dict], v_paths: list[dict]) -> tuple[list[dict], list[dict]]:
    h_orders = [{pos: idx for idx, pos in enumerate(h_path["crossings"])} for h_path in h_paths]
    v_orders = [{pos: idx for idx, pos in enumerate(v_path["crossings"])} for v_path in v_paths]
    return h_orders, v_orders


def _weave_color(
    h_path: dict,
    v_path: dict,
    pos: tuple[int, int],
    h_orders: list[dict],
    v_orders: list[dict],
    hi: int,
    vi: int,
) -> int:
    h_over = h_orders[hi][pos] % 2 == 0
    v_over = v_orders[vi][pos] % 2 == 0
    if h_over == v_over or h_over:
        return int(h_path["color"])
    return int(v_path["color"])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Weave horizontal and vertical paths by alternating which path is visible at each crossing.

    Input structure
    - Background is black.
    - Several one-pixel-wide straight paths appear in distinct colors.
    - Paths are only horizontal or vertical.
    - Every horizontal/vertical crossing is replaced by a grey marker pixel.

    Rule
    - Read crossings left-to-right on each horizontal path and top-to-bottom on
      each vertical path.
    - Along each path, the 1st, 3rd, 5th, ... crossings are "over"; the 2nd,
      4th, 6th, ... crossings are "under".
    - At a crossing, whichever path is over supplies the output color.
    - If both paths land in the same phase, the horizontal path wins.

    Non-crossing path cells stay unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    h_paths = _extract_horizontal_paths(g, bg)
    v_paths = _extract_vertical_paths(g, bg)
    assert h_paths and v_paths, "expected both horizontal and vertical paths"

    crossing_positions, h_by_cross, v_by_cross, cross_color = _attach_crossings(g, bg, h_paths, v_paths)
    _assign_path_colors(g, cross_color, h_paths, v_paths)
    h_orders, v_orders = _decision_maps(h_paths, v_paths)

    out = g.copy()
    for pos in crossing_positions:
        hi = h_by_cross[pos]
        vi = v_by_cross[pos]
        out[pos] = _weave_color(h_paths[hi], v_paths[vi], pos, h_orders, v_orders, hi, vi)
    return out


def _sample_positions(count: int, lo: int, hi: int, min_gap: int) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * (min_gap - 1)
    if span < needed:
        return None
    compressed = span - (count - 1) * (min_gap - 1)
    picks = sorted(random.sample(range(compressed), count))
    return [lo + picks[i] + i * (min_gap - 1) for i in range(count)]


def _sample_canvas(n_h: int, n_v: int) -> tuple[int, int]:
    min_h = max(GRID_MIN, 3 * n_h + 2)
    min_w = max(GRID_MIN, 3 * n_v + 2)
    return random.randint(min_h, GRID_MAX), random.randint(min_w, GRID_MAX)


def _sample_structure() -> tuple[int, int, list[dict], list[dict]] | None:
    n_h = random.randint(*PATH_COUNT_RANGE)
    n_v = random.randint(*PATH_COUNT_RANGE)
    if n_h + n_v > len(PATH_COLORS):
        return None

    H, W = _sample_canvas(n_h, n_v)
    rows = _sample_positions(n_h, 2, H - 3, MIN_POS_GAP)
    cols = _sample_positions(n_v, 2, W - 3, MIN_POS_GAP)
    if rows is None or cols is None:
        return None

    colors = random.sample(PATH_COLORS, n_h + n_v)
    h_paths: list[dict] = []
    for idx, row in enumerate(rows):
        left_idx = random.randint(0, n_v - 1)
        right_idx = random.randint(left_idx, n_v - 1)
        h_paths.append(
            {
                "row": int(row),
                "c0": int(cols[left_idx] - 1),
                "c1": int(cols[right_idx] + 1),
                "color": int(colors[idx]),
                "crossings": [],
            }
        )

    v_paths: list[dict] = []
    for idx, col in enumerate(cols):
        top_idx = random.randint(0, n_h - 1)
        bottom_idx = random.randint(top_idx, n_h - 1)
        v_paths.append(
            {
                "col": int(col),
                "r0": int(rows[top_idx] - 1),
                "r1": int(rows[bottom_idx] + 1),
                "color": int(colors[n_h + idx]),
                "crossings": [],
            }
        )

    return H, W, h_paths, v_paths


def _compute_sample_crossings(
    h_paths: list[dict], v_paths: list[dict]
) -> tuple[dict[tuple[int, int], tuple[int, int]], bool, bool] | None:
    cross_map: dict[tuple[int, int], tuple[int, int]] = {}
    for h_path in h_paths:
        h_path["crossings"] = []
    for v_path in v_paths:
        v_path["crossings"] = []

    for hi, h_path in enumerate(h_paths):
        row = int(h_path["row"])
        for vi, v_path in enumerate(v_paths):
            col = int(v_path["col"])
            if h_path["c0"] <= col <= h_path["c1"] and v_path["r0"] <= row <= v_path["r1"]:
                pos = (row, col)
                cross_map[pos] = (hi, vi)
                h_path["crossings"].append(pos)
                v_path["crossings"].append(pos)

    if not cross_map:
        return None

    if any(len(h_path["crossings"]) == 0 for h_path in h_paths):
        return None
    if any(len(v_path["crossings"]) == 0 for v_path in v_paths):
        return None
    if max(max(len(h_path["crossings"]) for h_path in h_paths), max(len(v_path["crossings"]) for v_path in v_paths)) < 2:
        return None

    for h_path in h_paths:
        h_path["crossings"].sort(key=lambda pos: pos[1])
    for v_path in v_paths:
        v_path["crossings"].sort(key=lambda pos: pos[0])

    h_orders, v_orders = _decision_maps(h_paths, v_paths)
    odd_odd_tie = False
    vertical_visible = False
    for pos, (hi, vi) in cross_map.items():
        h_over = h_orders[hi][pos] % 2 == 0
        v_over = v_orders[vi][pos] % 2 == 0
        if h_over and v_over:
            odd_odd_tie = True
        if (not h_over) and v_over:
            vertical_visible = True

    if not odd_odd_tie or not vertical_visible:
        return None
    return cross_map, odd_odd_tie, vertical_visible


def _render_input(H: int, W: int, h_paths: list[dict], v_paths: list[dict], cross_map: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for h_path in h_paths:
        grid[int(h_path["row"]), int(h_path["c0"]) : int(h_path["c1"]) + 1] = int(h_path["color"])
    for v_path in v_paths:
        grid[int(v_path["r0"]) : int(v_path["r1"]) + 1, int(v_path["col"])] = int(v_path["color"])
    for row, col in cross_map:
        grid[int(row), int(col)] = CROSS_COLOR
    return grid


def _render_output(inp: np.ndarray, h_paths: list[dict], v_paths: list[dict], cross_map: dict[tuple[int, int], tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    h_orders, v_orders = _decision_maps(h_paths, v_paths)
    for pos, (hi, vi) in cross_map.items():
        out[pos] = _weave_color(h_paths[hi], v_paths[vi], pos, h_orders, v_orders, hi, vi)
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_structure()
    if sampled is None:
        return None
    H, W, h_paths, v_paths = sampled

    crossing_info = _compute_sample_crossings(h_paths, v_paths)
    if crossing_info is None:
        return None
    cross_map, _odd_odd_tie, _vertical_visible = crossing_info

    inp = _render_input(H, W, h_paths, v_paths, cross_map)
    expected = _render_output(inp, h_paths, v_paths, cross_map)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid weaving-path input")
