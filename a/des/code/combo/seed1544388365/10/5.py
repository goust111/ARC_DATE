import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _hidden_orientation(grid: np.ndarray, color: int, row: int, col: int) -> str | None:
    h, w = grid.shape
    horizontal = 0 < col < w - 1 and int(grid[row, col - 1]) == color and int(grid[row, col + 1]) == color
    vertical = 0 < row < h - 1 and int(grid[row - 1, col]) == color and int(grid[row + 1, col]) == color
    assert not (horizontal and vertical), "a hidden path may not self-cross at one cell"
    if horizontal:
        return "horizontal"
    if vertical:
        return "vertical"
    return None


def _unique_topological_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    followers = {color: set() for color in colors}
    indegree = {color: 0 for color in colors}
    for lower, upper in edges:
        if upper not in followers[lower]:
            followers[lower].add(upper)
            indegree[upper] += 1

    order: list[int] = []
    ready = [color for color in colors if indegree[color] == 0]
    while ready:
        assert len(ready) == 1, "layer order must be uniquely determined"
        color = ready.pop()
        order.append(color)
        for nxt in followers[color]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)

    assert len(order) == len(colors), "layer graph must be acyclic"
    return order


def _solve_lengths_and_order(grid: np.ndarray) -> tuple[dict[int, int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 3 and w >= 3

    bg = _bg_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert len(colors) >= 2

    visible = {color: int(np.count_nonzero(g == color)) for color in colors}
    hidden = {color: 0 for color in colors}
    edges: set[tuple[int, int]] = set()

    for row in range(h):
        for col in range(w):
            top = int(g[row, col])
            if top == bg:
                continue
            under_candidates: list[int] = []
            for color in colors:
                if color == top:
                    continue
                if _hidden_orientation(g, color, row, col) is not None:
                    under_candidates.append(color)
            assert len(under_candidates) <= 1, "each crossing may hide at most one path"
            if under_candidates:
                lower = under_candidates[0]
                hidden[lower] += 1
                edges.add((lower, top))

    lengths = {color: visible[color] + hidden[color] for color in colors}
    for color, length in lengths.items():
        assert 1 <= length <= w, f"path color {color} does not fit output width"

    order = _unique_topological_order(colors, edges)
    return lengths, order, bg


def main(grid: np.ndarray) -> np.ndarray:
    """Recover hidden crossing pixels, infer the global layer order, then flatten each path."""
    g = np.asarray(grid, dtype=int)
    lengths, order, bg = _solve_lengths_and_order(g)
    out = np.full((2 * len(order) - 1, g.shape[1]), bg, dtype=int)
    for idx, color in enumerate(order):
        out[2 * idx, : lengths[color]] = color
    return out


GRID_HEIGHT_RANGE = (9, 14)
GRID_WIDTH_MAX_EXCLUSIVE = 25
PATH_COUNT_RANGE = (3, 7)
BENT_VERTICAL_PROB = 0.5
MAX_GENERATION_ATTEMPTS = 400
PALETTE = tuple(range(10))


def _horizontal_cells(width: int, row: int) -> set[tuple[int, int]]:
    return {(row, col) for col in range(width)}


def _vertical_cells(height: int, base_col: int, turn_row: int | None = None, end_col: int | None = None) -> set[tuple[int, int]]:
    if turn_row is None or end_col is None or end_col == base_col:
        return {(row, base_col) for row in range(height)}

    step = 1 if end_col > base_col else -1
    cells = {(row, base_col) for row in range(turn_row + 1)}
    cells.update((turn_row, col) for col in range(base_col, end_col + step, step))
    cells.update((row, end_col) for row in range(turn_row, height))
    return cells


def _render_paths(shape: tuple[int, int], bg: int, paths: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, bg, dtype=int)
    for path in paths:
        color = int(path["color"])
        for row, col in path["cells"]:
            grid[row, col] = color
    return grid


def _expected_output(width: int, bg: int, paths: list[dict[str, object]]) -> np.ndarray:
    out = np.full((2 * len(paths) - 1, width), bg, dtype=int)
    for idx, path in enumerate(paths):
        out[2 * idx, : int(path["length"])] = int(path["color"])
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height = int(rng.integers(*GRID_HEIGHT_RANGE))
    width_min = max(height + 2, 16)
    if width_min >= GRID_WIDTH_MAX_EXCLUSIVE:
        return None
    width = int(rng.integers(width_min, GRID_WIDTH_MAX_EXCLUSIVE))

    path_count = int(rng.integers(*PATH_COUNT_RANGE))
    start_with_horizontal = bool(rng.integers(0, 2))
    orientations = [
        "H" if ((idx % 2 == 0) == start_with_horizontal) else "V"
        for idx in range(path_count)
    ]
    horizontal_count = orientations.count("H")
    vertical_count = orientations.count("V")

    horizontal_rows = list(range(1, height - 1, 2))
    base_columns = list(range(2, width - 2, 4))
    turn_rows = list(range(2, height - 1, 2))
    if len(horizontal_rows) < horizontal_count or len(base_columns) < vertical_count or not turn_rows:
        return None

    chosen_rows = rng.choice(horizontal_rows, size=horizontal_count, replace=False).tolist()
    chosen_cols = rng.choice(base_columns, size=vertical_count, replace=False).tolist()
    rng.shuffle(chosen_rows)
    rng.shuffle(chosen_cols)

    colors = rng.choice(PALETTE, size=path_count + 1, replace=False).tolist()
    bg = int(colors[0])
    path_colors = [int(color) for color in colors[1:]]

    paths: list[dict[str, object]] = []
    h_idx = 0
    v_idx = 0
    for layer_idx, orientation in enumerate(orientations):
        color = path_colors[layer_idx]
        if orientation == "H":
            row = int(chosen_rows[h_idx])
            h_idx += 1
            cells = _horizontal_cells(width, row)
            paths.append(
                {
                    "color": color,
                    "cells": cells,
                    "length": len(cells),
                }
            )
            continue

        base_col = int(chosen_cols[v_idx])
        v_idx += 1
        bent = bool(rng.random() < BENT_VERTICAL_PROB)
        end_col = base_col
        turn_row = None
        if bent:
            shift = int(rng.choice([-1, 1]))
            if 1 <= base_col + shift <= width - 2:
                end_col = base_col + shift
                turn_row = int(rng.choice(turn_rows))
        cells = _vertical_cells(height, base_col, turn_row, end_col)
        if len(cells) > width:
            return None
        paths.append(
            {
                "color": color,
                "cells": cells,
                "length": len(cells),
            }
        )

    grid = _render_paths((height, width), bg, paths)
    expected = _expected_output(width, bg, paths)
    if not np.array_equal(main(grid), expected):
        return None
    if len({tuple(cell) for path in paths for cell in path["cells"]}) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
