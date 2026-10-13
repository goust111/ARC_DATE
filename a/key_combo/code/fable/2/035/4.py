from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

try:
    from arc_utils import get_bg
except Exception:
    def get_bg(grid: np.ndarray) -> int:
        counts = Counter(map(int, np.asarray(grid).reshape(-1)))
        return int(counts.most_common(1)[0][0])


Node = tuple[str, int]

BG = 0
COLOR_POOL = tuple(range(1, 10))

TOTAL_BAR_RANGE = (5, 7)
GRID_SIDE_MAX = 18
GRID_SIDE_MIN = 10
INTERIOR_MARGIN = 1
MIN_SAME_ORIENTATION_GAP = 2
GEN_RETRY_LIMIT = 400


def _bar_rows(grid: np.ndarray, bg: int) -> list[int]:
    return [int(r) for r in np.flatnonzero(np.all(grid != bg, axis=1))]


def _bar_cols(grid: np.ndarray, bg: int) -> list[int]:
    return [int(c) for c in np.flatnonzero(np.all(grid != bg, axis=0))]


def _recover_bar_colors(
    grid: np.ndarray,
    bg: int,
    row_bars: list[int],
    col_bars: list[int],
) -> tuple[dict[int, int], dict[int, int]]:
    h, w = grid.shape
    non_row_mask = np.ones(h, dtype=bool)
    non_col_mask = np.ones(w, dtype=bool)
    non_row_mask[row_bars] = False
    non_col_mask[col_bars] = False

    assert non_row_mask.any(), "each column bar needs at least one non-crossing cell"
    assert non_col_mask.any(), "each row bar needs at least one non-crossing cell"

    row_colors: dict[int, int] = {}
    for r in row_bars:
        vals = np.unique(grid[int(r), non_col_mask])
        assert vals.size == 1 and int(vals[0]) != bg
        row_colors[int(r)] = int(vals[0])

    col_colors: dict[int, int] = {}
    for c in col_bars:
        vals = np.unique(grid[non_row_mask, int(c)])
        assert vals.size == 1 and int(vals[0]) != bg
        col_colors[int(c)] = int(vals[0])

    all_colors = list(row_colors.values()) + list(col_colors.values())
    assert len(all_colors) == len(set(all_colors)), "bar colors must be pairwise distinct"
    return row_colors, col_colors


def _build_relation_graph(
    grid: np.ndarray,
    row_bars: list[int],
    col_bars: list[int],
    row_colors: dict[int, int],
    col_colors: dict[int, int],
) -> tuple[dict[Node, int], dict[Node, set[Node]]]:
    nodes = [("r", r) for r in row_bars] + [("c", c) for c in col_bars]
    colors = {("r", r): row_colors[r] for r in row_bars}
    colors.update({("c", c): col_colors[c] for c in col_bars})
    edges = {node: set() for node in nodes}

    for r in row_bars:
        for c in col_bars:
            seen = int(grid[int(r), int(c)])
            row_color = int(row_colors[r])
            col_color = int(col_colors[c])
            if seen == row_color:
                edges[("c", c)].add(("r", r))
            elif seen == col_color:
                edges[("r", r)].add(("c", c))
            else:
                raise AssertionError("crossing cell must show one of its two bar colors")

    return colors, edges


def _unique_topological_order(edges: dict[Node, set[Node]]) -> list[Node]:
    indegree = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst in indegree, f"unknown node referenced from {src}"
            indegree[dst] += 1

    frontier = [node for node, deg in indegree.items() if deg == 0]
    order: list[Node] = []
    while frontier:
        assert len(frontier) == 1, "occlusion graph does not determine a unique total order"
        node = frontier.pop()
        order.append(node)
        for dst in edges[node]:
            indegree[dst] -= 1
            if indegree[dst] == 0:
                frontier.append(dst)

    assert len(order) == len(edges), "occlusion graph must be acyclic"
    return order


def _analyze_grid(
    grid: np.ndarray,
) -> tuple[int, list[int], list[int], dict[int, int], dict[int, int], dict[Node, int], dict[Node, set[Node]], list[Node]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))

    row_bars = _bar_rows(g, bg)
    col_bars = _bar_cols(g, bg)
    assert len(row_bars) >= 1 and len(col_bars) >= 1

    row_colors, col_colors = _recover_bar_colors(g, bg, row_bars, col_bars)
    node_colors, edges = _build_relation_graph(g, row_bars, col_bars, row_colors, col_colors)
    order = _unique_topological_order(edges)
    return bg, row_bars, col_bars, row_colors, col_colors, node_colors, edges, order


def _is_monotone(values: list[int]) -> bool:
    if len(values) < 3:
        return True
    return values == sorted(values) or values == sorted(values, reverse=True)


def _sample_spaced_positions(length: int, count: int) -> list[int] | None:
    candidates = list(range(INTERIOR_MARGIN, length - INTERIOR_MARGIN))
    for _ in range(200):
        vals = sorted(random.sample(candidates, count))
        if all(b - a >= MIN_SAME_ORIENTATION_GAP for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _permute_positions(sorted_positions: list[int]) -> list[int] | None:
    vals = list(sorted_positions)
    for _ in range(200):
        random.shuffle(vals)
        if len(vals) < 3 or not _is_monotone(vals):
            return list(vals)
    return None


def _orientation_sequence() -> list[str]:
    total = random.randint(*TOTAL_BAR_RANGE)
    first = random.choice(("r", "c"))
    seq = []
    current = first
    for _ in range(total):
        seq.append(current)
        current = "c" if current == "r" else "r"
    return seq


def _render_grid(shape: tuple[int, int], bars: list[tuple[str, int, int]]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    for orient, index, color in bars:
        if orient == "r":
            out[int(index), :] = int(color)
        else:
            out[:, int(index)] = int(color)
    return out


def _validate_generated_grid(
    grid: np.ndarray,
    expected_order: list[Node],
    expected_colors: list[int],
) -> None:
    bg, row_bars, col_bars, row_colors, col_colors, node_colors, edges, order = _analyze_grid(grid)
    assert bg == BG
    assert order == expected_order
    assert [node_colors[node] for node in order] == expected_colors
    assert len(row_bars) >= 2 and len(col_bars) >= 2
    assert grid.shape[0] > len(row_bars) and grid.shape[1] > len(col_bars)

    row_profiles = {
        r: tuple(int(grid[int(r), int(c)] == row_colors[r]) for c in col_bars)
        for r in row_bars
    }
    col_profiles = {
        c: tuple(int(grid[int(r), int(c)] == col_colors[c]) for r in row_bars)
        for c in col_bars
    }
    assert len(set(row_profiles.values())) == len(row_profiles), "row bars must be totally ordered by transitive clues"
    assert len(set(col_profiles.values())) == len(col_profiles), "column bars must be totally ordered by transitive clues"

    order_orients = [orient for orient, _ in order]
    assert all(a != b for a, b in zip(order_orients, order_orients[1:])), "generator should build an alternating occlusion chain"

    row_order_positions = [idx for orient, idx in order if orient == "r"]
    col_order_positions = [idx for orient, idx in order if orient == "c"]
    if len(row_order_positions) >= 3:
        assert not _is_monotone(row_order_positions), "row order must not collapse to a simple spatial sort"
    if len(col_order_positions) >= 3:
        assert not _is_monotone(col_order_positions), "column order must not collapse to a simple spatial sort"

    same_orientation_order = {
        "r": [node for node in order if node[0] == "r"],
        "c": [node for node in order if node[0] == "c"],
    }
    for orient in ("r", "c"):
        chain = same_orientation_order[orient]
        for a, b in zip(chain, chain[1:]):
            assert b not in edges[a] and a not in edges[b], "same-orientation order must be indirect"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the full drawing order of row bars and column bars from their crossing occlusions.

    Input structure:
    - The grid has a dominant background plus several full-width row bars and several full-height column bars.
    - Each bar has a unique color, and every row-bar/column-bar pair crosses exactly once.
    - A crossing cell shows the color of the bar drawn later, while non-crossing cells still show each bar's own color.

    Rule:
    - Detect which rows are row bars and which columns are column bars.
    - Recover every bar's intrinsic color from its non-crossing cells.
    - At each crossing, read which of the two bar colors is visible to infer that pair's above/below relation.
    - Combine all such pairwise relations, use transitive inference to obtain the unique bottom-to-top total order, and output the bar colors as a 1xN row.

    Key insight:
    - A crossing is an occlusion fossil: it directly orders one row bar against one column bar, and the same-orientation order appears only after chaining those clues transitively.
    """
    _bg, _rows, _cols, _row_colors, _col_colors, node_colors, _edges, order = _analyze_grid(grid)
    return np.array([[node_colors[node] for node in order]], dtype=int)


def _gen_attempt() -> np.ndarray | None:
    orient_seq = _orientation_sequence()
    row_count = orient_seq.count("r")
    col_count = orient_seq.count("c")

    height_min = max(GRID_SIDE_MIN, 2 * row_count + 3)
    width_min = max(GRID_SIDE_MIN, 2 * col_count + 3)
    if height_min > GRID_SIDE_MAX or width_min > GRID_SIDE_MAX:
        return None

    height = random.randint(height_min, GRID_SIDE_MAX)
    width = random.randint(width_min, GRID_SIDE_MAX)

    row_slots = _sample_spaced_positions(height, row_count)
    col_slots = _sample_spaced_positions(width, col_count)
    if row_slots is None or col_slots is None:
        return None

    row_positions = _permute_positions(row_slots)
    col_positions = _permute_positions(col_slots)
    if row_positions is None or col_positions is None:
        return None

    colors = random.sample(COLOR_POOL, len(orient_seq))
    bars: list[tuple[str, int, int]] = []
    expected_order: list[Node] = []
    row_i = 0
    col_i = 0
    for orient, color in zip(orient_seq, colors):
        if orient == "r":
            index = row_positions[row_i]
            row_i += 1
        else:
            index = col_positions[col_i]
            col_i += 1
        bars.append((orient, index, int(color)))
        expected_order.append((orient, int(index)))

    grid = _render_grid((height, width), bars)
    expected = np.array([colors], dtype=int)

    try:
        _validate_generated_grid(grid, expected_order, colors)
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for occlusion-order puzzle")
