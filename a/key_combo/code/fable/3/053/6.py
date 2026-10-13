from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND_COLOR = 0
SEPARATOR_COLOR = 5
LINE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

MIN_TOTAL_LINES = 4
MAX_TOTAL_LINES = 7
TOP_HEIGHT_MAX = 14
GRID_WIDTH_MAX = 16
BOTTOM_HEIGHT_RANGE = (2, 4)
GEN_RETRIES = 300


def _mode(values: np.ndarray) -> int:
    return int(Counter(map(int, values)).most_common(1)[0][0])


def _find_separator_and_placeholder(grid: np.ndarray) -> tuple[int, int, int]:
    height, width = grid.shape
    for sep_r in range(height):
        row = grid[sep_r]
        if not np.all(row == row[0]):
            continue
        sep_color = int(row[0])
        for placeholder_r in range(sep_r + 1, height):
            count = int(np.sum(grid[placeholder_r] == sep_color))
            if 0 < count < width:
                return int(sep_r), int(placeholder_r), int(sep_color)
    raise AssertionError("expected one full separator row and one partial placeholder row")


def _background_from_bottom(bottom: np.ndarray, separator_color: int) -> int:
    values = bottom[bottom != separator_color]
    assert values.size, "bottom area must contain background cells"
    return _mode(values)


def _unique_topological_order(edges: list[tuple[int, int]], nodes: list[int]) -> list[int]:
    adjacency = {node: set() for node in nodes}
    indegree = {node: 0 for node in nodes}
    for src, dst in edges:
        if dst in adjacency[src]:
            continue
        adjacency[src].add(dst)
        indegree[dst] += 1

    order: list[int] = []
    ready = sorted(node for node in nodes if indegree[node] == 0)
    while ready:
        assert len(ready) == 1, "line order is not uniquely determined"
        node = ready.pop()
        order.append(node)
        for nxt in sorted(adjacency[node]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
        ready.sort()

    assert len(order) == len(nodes), "comparison graph must be acyclic and complete enough"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full drawing order from line crossings and write it into the lower placeholder row.

    Input structure:
    - A full gray separator row splits the canvas into an upper line panel and a lower placeholder panel.
    - The upper panel contains several full-width horizontal lines and full-height vertical lines, each with a distinct color.
    - Each crossing cell shows the color of whichever of the two lines was drawn later.
    - The lower panel contains one row of gray placeholder cells whose count equals the total number of lines.

    Rule:
    - Detect the horizontal and vertical lines from the upper panel.
    - For every crossing, orient the comparison from the hidden earlier line to the visible later line.
    - Recover the unique total drawing order implied by all crossing outcomes.
    - Recolor the placeholder cells from left to right with the line colors from earliest to latest, leaving the upper panel unchanged.

    Key insight:
    - A crossing is a direct "later covers earlier" record, and the missing comparisons between parallel lines are recovered through transitive chains that alternate between horizontal and vertical lines.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    separator_r, placeholder_r, separator_color = _find_separator_and_placeholder(g)
    top = g[:separator_r, :]
    bottom = g[separator_r + 1 :, :]
    assert top.size and bottom.size, "separator must split the grid into two non-empty parts"

    bg = _background_from_bottom(bottom, separator_color)

    horizontal_rows = [int(r) for r in np.flatnonzero(np.all(top != bg, axis=1))]
    vertical_cols = [int(c) for c in np.flatnonzero(np.all(top != bg, axis=0))]
    assert horizontal_rows and vertical_cols, "expected both horizontal and vertical lines"

    horizontal_colors = {}
    for r in horizontal_rows:
        color = _mode(top[r, :])
        assert color not in (bg, separator_color)
        horizontal_colors[r] = int(color)

    vertical_colors = {}
    for c in vertical_cols:
        color = _mode(top[:, c])
        assert color not in (bg, separator_color)
        vertical_colors[c] = int(color)

    line_colors = list(horizontal_colors.values()) + list(vertical_colors.values())
    assert len(line_colors) == len(set(line_colors)), "line colors must be distinct"

    edges: list[tuple[int, int]] = []
    for r, h_color in horizontal_colors.items():
        for c, v_color in vertical_colors.items():
            crossing = int(top[r, c])
            if crossing == h_color:
                edges.append((v_color, h_color))
            elif crossing == v_color:
                edges.append((h_color, v_color))
            else:
                raise AssertionError("crossing must show one of its two line colors")

    order = _unique_topological_order(edges, line_colors)

    placeholder_cols = [int(c) for c in np.flatnonzero(g[placeholder_r] == separator_color)]
    assert len(placeholder_cols) == len(order), "placeholder count must equal line count"

    out = g.copy()
    out[placeholder_r, placeholder_cols] = np.asarray(order, dtype=int)
    return out


def _spaced_positions(count: int, size: int) -> list[int]:
    assert size >= 2 * count - 1
    anchors = sorted(random.sample(range(size - count + 1), count))
    return [int(anchor + i) for i, anchor in enumerate(anchors)]


def _alternating_orientations(total: int, first: str) -> list[str]:
    out = []
    current = first
    for _ in range(total):
        out.append(current)
        current = "V" if current == "H" else "H"
    return out


def _draw_top(height: int, width: int, order_specs: list[dict[str, int | str]]) -> np.ndarray:
    top = np.full((height, width), BACKGROUND_COLOR, dtype=int)
    for spec in order_specs:
        color = int(spec["color"])
        index = int(spec["index"])
        if spec["orientation"] == "H":
            top[index, :] = color
        else:
            top[:, index] = color
    return top


def _make_expected(grid: np.ndarray, placeholder_r: int, start_col: int, colors: list[int]) -> np.ndarray:
    out = grid.copy()
    out[placeholder_r, start_col : start_col + len(colors)] = np.asarray(colors, dtype=int)
    return out


def _gen_attempt() -> np.ndarray | None:
    total_lines = random.randint(MIN_TOTAL_LINES, MAX_TOTAL_LINES)
    orientations = _alternating_orientations(total_lines, random.choice(["H", "V"]))
    n_horizontal = orientations.count("H")
    n_vertical = orientations.count("V")

    min_top_height = max(2 * n_horizontal + 1, 6)
    min_width = max(2 * n_vertical + 1, total_lines + 2)
    if min_top_height > TOP_HEIGHT_MAX or min_width > GRID_WIDTH_MAX:
        return None

    top_height = random.randint(min_top_height, TOP_HEIGHT_MAX)
    width = random.randint(min_width, GRID_WIDTH_MAX)

    horizontal_rows = _spaced_positions(n_horizontal, top_height)
    vertical_cols = _spaced_positions(n_vertical, width)
    random.shuffle(horizontal_rows)
    random.shuffle(vertical_cols)

    colors = random.sample(LINE_COLORS, total_lines)
    horizontal_specs = [
        {"orientation": "H", "index": horizontal_rows[i], "color": colors[i]}
        for i in range(n_horizontal)
    ]
    vertical_specs = [
        {"orientation": "V", "index": vertical_cols[i], "color": colors[n_horizontal + i]}
        for i in range(n_vertical)
    ]
    random.shuffle(horizontal_specs)
    random.shuffle(vertical_specs)

    order_specs = []
    h_idx = 0
    v_idx = 0
    for orientation in orientations:
        if orientation == "H":
            order_specs.append(horizontal_specs[h_idx])
            h_idx += 1
        else:
            order_specs.append(vertical_specs[v_idx])
            v_idx += 1

    top = _draw_top(top_height, width, order_specs)

    bottom_height = random.randint(*BOTTOM_HEIGHT_RANGE)
    placeholder_local_r = random.randrange(bottom_height)
    placeholder_r = top_height + 1 + placeholder_local_r
    start_col = random.randint(0, width - total_lines)

    grid = np.full((top_height + 1 + bottom_height, width), BACKGROUND_COLOR, dtype=int)
    grid[:top_height, :] = top
    grid[top_height, :] = SEPARATOR_COLOR
    grid[placeholder_r, start_col : start_col + total_lines] = SEPARATOR_COLOR

    expected = _make_expected(
        grid,
        placeholder_r,
        start_col,
        [int(spec["color"]) for spec in order_specs],
    )
    try:
        solved = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
