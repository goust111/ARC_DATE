from __future__ import annotations

import random

import numpy as np


BG = 0
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Generator bounds.
GRID_SIZE_MIN = 10
GRID_SIZE_MAX = 16
PATH_COUNT_MIN = 3
PATH_COUNT_MAX = 5
POSITION_MIN_GAP = 2
MAX_GENERATION_ATTEMPTS = 300
MAX_POSITION_ATTEMPTS = 80


def _extract_paths(g: np.ndarray) -> list[dict[str, int | str]]:
    paths: list[dict[str, int | str]] = []
    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        coords = np.argwhere(g == color)
        assert len(coords) >= 2, "each path must expose at least two visible pixels"

        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])
        if len(rows) == 1 and len(cols) >= 2:
            paths.append(
                {
                    "color": color,
                    "orient": "H",
                    "fixed": int(rows[0]),
                    "start": int(cols.min()),
                    "end": int(cols.max()),
                }
            )
        elif len(cols) == 1 and len(rows) >= 2:
            paths.append(
                {
                    "color": color,
                    "orient": "V",
                    "fixed": int(cols[0]),
                    "start": int(rows.min()),
                    "end": int(rows.max()),
                }
            )
        else:
            raise AssertionError("every color must form a visible straight horizontal or vertical trace")

    assert PATH_COUNT_MIN <= len(paths) <= PATH_COUNT_MAX, "expected 3-5 paths"
    return paths


def _cross_point(a: dict[str, int | str], b: dict[str, int | str]) -> tuple[int, int] | None:
    if a["orient"] == b["orient"]:
        return None

    if a["orient"] == "H":
        h, v = a, b
    else:
        h, v = b, a

    row = int(h["fixed"])
    col = int(v["fixed"])
    if int(h["start"]) <= col <= int(h["end"]) and int(v["start"]) <= row <= int(v["end"]):
        return row, col
    return None


def _build_order_graph(g: np.ndarray, paths: list[dict[str, int | str]]) -> dict[int, set[int]]:
    edges = {int(path["color"]): set() for path in paths}
    for i, a in enumerate(paths):
        for b in paths[i + 1 :]:
            cross = _cross_point(a, b)
            if cross is None:
                continue

            row, col = cross
            visible = int(g[row, col])
            a_color = int(a["color"])
            b_color = int(b["color"])
            assert visible in {a_color, b_color}, "crossing pixel must show one of the two crossing paths"
            if visible == a_color:
                edges[b_color].add(a_color)
            else:
                edges[a_color].add(b_color)
    return edges


def _unique_topological_order(colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in colors}
    for lower in colors:
        for upper in edges[lower]:
            indegree[upper] += 1

    available = sorted(color for color in colors if indegree[color] == 0)
    order: list[int] = []
    while available:
        assert len(available) == 1, "layering order must be uniquely determined"
        color = available.pop()
        order.append(color)
        for upper in sorted(edges[color]):
            indegree[upper] -= 1
            if indegree[upper] == 0:
                available.append(upper)
        available.sort()

    assert len(order) == len(colors), "layering graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the unique draw order from visible crossings, then extend the topmost path to the grid edge."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    paths = _extract_paths(g)
    colors = [int(path["color"]) for path in paths]
    edges = _build_order_graph(g, paths)
    order = _unique_topological_order(colors, edges)
    top_color = order[-1]
    top_path = next(path for path in paths if int(path["color"]) == top_color)

    out = g.copy()
    if top_path["orient"] == "H":
        out[int(top_path["fixed"]), :] = top_color
    else:
        out[:, int(top_path["fixed"])] = top_color
    return out


def _paint_path(grid: np.ndarray, path: dict[str, int | str]) -> None:
    color = int(path["color"])
    fixed = int(path["fixed"])
    start = int(path["start"])
    end = int(path["end"])
    if path["orient"] == "H":
        grid[fixed, start : end + 1] = color
    else:
        grid[start : end + 1, fixed] = color


def _render_scene(shape: tuple[int, int], paths: list[dict[str, int | str]]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for path in sorted(paths, key=lambda item: int(item["rank"])):
        _paint_path(grid, path)
    return grid


def _expected_output(shape: tuple[int, int], paths: list[dict[str, int | str]]) -> np.ndarray:
    out = _render_scene(shape, paths)
    top_path = max(paths, key=lambda item: int(item["rank"]))
    color = int(top_path["color"])
    fixed = int(top_path["fixed"])
    if top_path["orient"] == "H":
        out[fixed, :] = color
    else:
        out[:, fixed] = color
    return out


def _sample_spaced_positions(count: int, low: int, high: int, min_gap: int) -> list[int] | None:
    values = list(range(low, high + 1))
    if len(values) < count:
        return None

    for _ in range(MAX_POSITION_ATTEMPTS):
        picks = sorted(random.sample(values, count))
        if all(b - a >= min_gap for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _make_scene(shape: tuple[int, int], path_count: int, start_orient: str) -> list[dict[str, int | str]] | None:
    h, w = shape
    orients = [start_orient if i % 2 == 0 else ("V" if start_orient == "H" else "H") for i in range(path_count)]
    h_count = sum(orient == "H" for orient in orients)
    v_count = path_count - h_count
    if h_count == 0 or v_count == 0:
        return None

    rows = _sample_spaced_positions(h_count, 2, h - 3, POSITION_MIN_GAP)
    cols = _sample_spaced_positions(v_count, 2, w - 3, POSITION_MIN_GAP)
    if rows is None or cols is None:
        return None

    random.shuffle(rows)
    random.shuffle(cols)
    colors = random.sample(COLOR_POOL, path_count)
    min_row, max_row = min(rows), max(rows)
    min_col, max_col = min(cols), max(cols)

    paths: list[dict[str, int | str]] = []
    row_idx = 0
    col_idx = 0
    for rank, orient in enumerate(orients):
        if orient == "H":
            fixed = rows[row_idx]
            row_idx += 1
            start = random.randint(1, min_col - 1)
            end = random.randint(max_col + 1, w - 2)
        else:
            fixed = cols[col_idx]
            col_idx += 1
            start = random.randint(1, min_row - 1)
            end = random.randint(max_row + 1, h - 2)

        paths.append(
            {
                "color": int(colors[rank]),
                "orient": orient,
                "fixed": int(fixed),
                "start": int(start),
                "end": int(end),
                "rank": int(rank),
            }
        )

    return paths


def _is_unique_order_from_grid(grid: np.ndarray, expected_order: list[int]) -> bool:
    try:
        paths = _extract_paths(grid)
        edges = _build_order_graph(grid, paths)
        order = _unique_topological_order([int(path["color"]) for path in paths], edges)
    except AssertionError:
        return False
    return order == expected_order


def _gen_attempt() -> np.ndarray | None:
    path_count = random.randint(PATH_COUNT_MIN, PATH_COUNT_MAX)
    start_orient = random.choice(["H", "V"])
    height = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    width = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)

    paths = _make_scene((height, width), path_count, start_orient)
    if paths is None:
        return None

    grid = _render_scene((height, width), paths)
    expected = _expected_output((height, width), paths)
    expected_order = [int(path["color"]) for path in sorted(paths, key=lambda item: int(item["rank"]))]

    if np.array_equal(grid, expected):
        return None
    if any(int(np.sum(grid == int(path["color"]))) < 2 for path in paths):
        return None
    if not _is_unique_order_from_grid(grid, expected_order):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid layered-line puzzle")
