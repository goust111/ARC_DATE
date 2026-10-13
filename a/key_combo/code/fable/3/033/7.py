from __future__ import annotations

import itertools
import random

import numpy as np


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _classify_bars(grid: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int], int | None]:
    row_bars: dict[int, int] = {}
    col_bars: dict[int, int] = {}
    leftovers: list[int] = []

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        rows = np.unique(pts[:, 0])
        cols = np.unique(pts[:, 1])
        count = int(len(pts))
        is_row = len(rows) == 1 and count >= 2 and len(cols) >= 2
        is_col = len(cols) == 1 and count >= 2 and len(rows) >= 2

        if is_row and not is_col:
            row = int(rows[0])
            assert row not in row_bars
            row_bars[row] = color
        elif is_col and not is_row:
            col = int(cols[0])
            assert col not in col_bars
            col_bars[col] = color
        else:
            leftovers.append(color)

    assert row_bars and col_bars
    assert len(leftovers) <= 1
    return row_bars, col_bars, (leftovers[0] if leftovers else None)


def _infer_layer_strengths(
    grid: np.ndarray,
    row_bars: dict[int, int],
    col_bars: dict[int, int],
    occluder: int | None,
) -> dict[tuple[str, int], int]:
    nodes = [("h", r) for r in sorted(row_bars)] + [("v", c) for c in sorted(col_bars)]
    edges = {node: set() for node in nodes}

    for row, row_color in row_bars.items():
        for col, col_color in col_bars.items():
            cell = int(grid[row, col])
            if occluder is not None and cell == occluder:
                continue
            row_node = ("h", row)
            col_node = ("v", col)
            if cell == row_color:
                edges[row_node].add(col_node)
            elif cell == col_color:
                edges[col_node].add(row_node)
            else:
                raise AssertionError("intersection must show one of its two bar colors or the occluder")

    reach: dict[tuple[str, int], set[tuple[str, int]]] = {}
    for node in nodes:
        seen: set[tuple[str, int]] = set()
        stack = list(edges[node])
        while stack:
            cur = stack.pop()
            if cur in seen:
                continue
            seen.add(cur)
            stack.extend(edges[cur] - seen)
        reach[node] = seen

    for i, u in enumerate(nodes):
        for v in nodes[i + 1 :]:
            u_above = v in reach[u]
            v_above = u in reach[v]
            assert u_above ^ v_above, "visible crossings must imply a unique total order"

    strengths = {node: len(reach[node]) for node in nodes}
    assert set(strengths.values()) == set(range(len(nodes)))
    return strengths


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the colors of occluded row-column crossings by inferring the bars' unique global stack order.

    Input structure:
    - A dominant background contains several full-width horizontal bars and several full-height vertical bars.
    - Every bar has its own color, and a crossing normally shows whichever of the two bars lies on top.
    - A few crossings are replaced by one uniform occluder color that is not a bar color.

    Rule:
    - Detect the horizontal bars and vertical bars from the colors confined to one row or one column.
    - Read each visible crossing as a pairwise "above/below" relation, then take the transitive closure to recover the unique total order of all bars.
    - Repaint every occluded crossing with the color of the higher-ranked bar at that crossing; leave every other cell unchanged.

    Key insight:
    - Same-orientation bars never touch directly, so their order is only recoverable through alternating visible row-vs-column comparisons.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    row_bars, col_bars, occluder = _classify_bars(g, bg)
    strengths = _infer_layer_strengths(g, row_bars, col_bars, occluder)

    out = g.copy()
    if occluder is None:
        return out

    for row, row_color in row_bars.items():
        for col, col_color in col_bars.items():
            if int(out[row, col]) != occluder:
                continue
            row_node = ("h", row)
            col_node = ("v", col)
            out[row, col] = row_color if strengths[row_node] > strengths[col_node] else col_color
    return out


# --------------- generator ---------------

BG_COLOR = 0
COUNT_OPTIONS = [(2, 2), (2, 3), (3, 2), (3, 3), (3, 4), (4, 3)]
GRID_SIDE_MAX = 20
MAX_RETRIES = 300


def _sample_spaced_positions(length: int, count: int) -> list[int]:
    span = length - (count - 1)
    bases = sorted(random.sample(range(span), count))
    return [base + i for i, base in enumerate(bases)]


def _alternating_kinds(num_rows: int, num_cols: int) -> list[str]:
    assert abs(num_rows - num_cols) <= 1
    if num_rows > num_cols:
        current = "h"
    elif num_cols > num_rows:
        current = "v"
    else:
        current = random.choice(["h", "v"])

    rem_rows, rem_cols = num_rows, num_cols
    kinds: list[str] = []
    while rem_rows or rem_cols:
        if current == "h":
            assert rem_rows > 0
            kinds.append("h")
            rem_rows -= 1
            current = "v"
        else:
            assert rem_cols > 0
            kinds.append("v")
            rem_cols -= 1
            current = "h"
    return kinds


def _build_total_order(num_rows: int, num_cols: int) -> list[tuple[str, int]]:
    kinds = _alternating_kinds(num_rows, num_cols)
    row_ids = list(range(num_rows))
    col_ids = list(range(num_cols))
    random.shuffle(row_ids)
    random.shuffle(col_ids)

    order: list[tuple[str, int]] = []
    for kind in kinds:
        if kind == "h":
            order.append(("h", row_ids.pop()))
        else:
            order.append(("v", col_ids.pop()))
    return order


def _eligible_hidden_pairs(order_pos: dict[tuple[str, int], int], num_rows: int, num_cols: int) -> list[tuple[int, int]]:
    pairs = []
    for row_id in range(num_rows):
        for col_id in range(num_cols):
            if abs(order_pos[("h", row_id)] - order_pos[("v", col_id)]) > 1:
                pairs.append((row_id, col_id))
    return pairs


def _choose_hidden_pairs(eligible: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if len(eligible) == 1:
        return [eligible[0]]
    if len(eligible) > 1 and random.random() < 0.35:
        return [random.choice(eligible)]

    max_hidden = min(4, len(eligible))
    target = random.randint(2, max_hidden)
    for size in range(target, 1, -1):
        good = [
            list(combo)
            for combo in itertools.combinations(eligible, size)
            if len({row_id for row_id, _ in combo}) >= 2 and len({col_id for _, col_id in combo}) >= 2
        ]
        if good:
            return random.choice(good)
    return [random.choice(eligible)]


def _render_complete_grid(
    height: int,
    width: int,
    row_positions: list[int],
    row_colors: list[int],
    col_positions: list[int],
    col_colors: list[int],
    order_pos: dict[tuple[str, int], int],
) -> np.ndarray:
    grid = np.full((height, width), BG_COLOR, dtype=int)
    for row, color in zip(row_positions, row_colors):
        grid[row, :] = color
    for col, color in zip(col_positions, col_colors):
        grid[:, col] = color
    for row_id, row in enumerate(row_positions):
        for col_id, col in enumerate(col_positions):
            if order_pos[("h", row_id)] < order_pos[("v", col_id)]:
                grid[row, col] = row_colors[row_id]
            else:
                grid[row, col] = col_colors[col_id]
    return grid


def _gen_attempt() -> np.ndarray | None:
    num_rows, num_cols = random.choice(COUNT_OPTIONS)
    min_height = max(10, 2 * num_rows + 3)
    min_width = max(10, 2 * num_cols + 3)
    height = random.randint(min_height, GRID_SIDE_MAX)
    width = random.randint(min_width, GRID_SIDE_MAX)

    row_positions = _sample_spaced_positions(height, num_rows)
    col_positions = _sample_spaced_positions(width, num_cols)

    palette = list(range(1, 10))
    random.shuffle(palette)
    occluder = palette[0]
    bar_colors = palette[1 : 1 + num_rows + num_cols]
    random.shuffle(bar_colors)
    row_colors = bar_colors[:num_rows]
    col_colors = bar_colors[num_rows:]

    order = _build_total_order(num_rows, num_cols)
    order_pos = {node: pos for pos, node in enumerate(order)}

    expected = _render_complete_grid(
        height, width, row_positions, row_colors, col_positions, col_colors, order_pos
    )
    eligible = _eligible_hidden_pairs(order_pos, num_rows, num_cols)
    if not eligible:
        return None
    hidden = _choose_hidden_pairs(eligible)

    inp = expected.copy()
    for row_id, col_id in hidden:
        inp[row_positions[row_id], col_positions[col_id]] = occluder

    for color in row_colors + col_colors:
        if int((inp == color).sum()) < 2:
            return None

    if _most_common_color(inp) != BG_COLOR:
        return None

    try:
        if not np.array_equal(main(inp), expected):
            return None
    except AssertionError:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
