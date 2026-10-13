import itertools
import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
BAR_COLORS = (1, 2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400


def _majority_color(values: np.ndarray) -> int:
    counts = Counter(map(int, values))
    color, _ = counts.most_common(1)[0]
    return int(color)


def _full_bar_coords(shape: tuple[int, int], bar: dict) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if bar["orientation"] == "h":
        r = int(bar["pos"])
        return [(r, c) for c in range(w)]
    c = int(bar["pos"])
    return [(r, c) for r in range(h)]


def _render_bars(shape: tuple[int, int], bars: dict[str, dict], order: list[str]) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for bar_id in reversed(order):
        bar = bars[bar_id]
        if bar["orientation"] == "h":
            grid[int(bar["pos"]), :] = int(bar["color"])
        else:
            grid[:, int(bar["pos"])] = int(bar["color"])
    return grid


def _mirror_coords(shape: tuple[int, int], source_bar: dict, axis_bar: dict) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    coords = []
    if axis_bar["orientation"] == "h":
        axis_r = int(axis_bar["pos"])
        for r, c in _full_bar_coords(shape, source_bar):
            rr = 2 * axis_r - int(r)
            if 0 <= rr < h:
                coords.append((rr, int(c)))
    else:
        axis_c = int(axis_bar["pos"])
        for r, c in _full_bar_coords(shape, source_bar):
            cc = 2 * axis_c - int(c)
            if 0 <= cc < w:
                coords.append((int(r), cc))
    return coords


def _apply_rule_from_order(
    shape: tuple[int, int],
    bars: dict[str, dict],
    order: list[str],
    top_id: str,
    second_id: str,
) -> np.ndarray:
    remaining_order = [bar_id for bar_id in order if bar_id != second_id]
    out = _render_bars(shape, bars, remaining_order)
    second_bar = bars[second_id]
    axis_bar = bars[top_id]
    for r, c in _mirror_coords(shape, second_bar, axis_bar):
        out[r, c] = int(second_bar["color"])
    return out


def _all_linear_extensions(nodes: list[str], edges: set[tuple[str, str]]) -> list[list[str]]:
    outgoing: dict[str, set[str]] = {node: set() for node in nodes}
    indegree: dict[str, int] = {node: 0 for node in nodes}
    for upper, lower in edges:
        if lower not in outgoing[upper]:
            outgoing[upper].add(lower)
            indegree[lower] += 1

    orders: list[list[str]] = []

    def rec(order: list[str], indeg: dict[str, int]) -> None:
        if len(order) == len(nodes):
            orders.append(order.copy())
            return
        available = sorted(node for node in nodes if indeg[node] == 0 and node not in order)
        for node in available:
            updated = dict(indeg)
            updated[node] = -1
            for nxt in outgoing[node]:
                updated[nxt] -= 1
            order.append(node)
            rec(order, updated)
            order.pop()

    rec([], dict(indegree))
    return orders


def _detect_bars(grid: np.ndarray) -> dict[str, dict]:
    h, w = map(int, grid.shape)
    row_positions = [r for r in range(h) if np.all(grid[r, :] != BG)]
    col_positions = [c for c in range(w) if np.all(grid[:, c] != BG)]
    assert 1 <= len(row_positions) <= 2, "expected one or two horizontal full-width bars"
    assert 1 <= len(col_positions) <= 2, "expected one or two vertical full-height bars"
    assert 3 <= len(row_positions) + len(col_positions) <= 4, "expected 3-4 total bars"

    bars: dict[str, dict] = {}
    used_colors: set[int] = set()

    for idx, r in enumerate(row_positions):
        color = _majority_color(grid[r, :])
        assert color != BG
        assert color not in used_colors
        used_colors.add(color)
        bars[f"h{idx}"] = {"orientation": "h", "pos": int(r), "color": int(color)}

    for idx, c in enumerate(col_positions):
        color = _majority_color(grid[:, c])
        assert color != BG
        assert color not in used_colors
        used_colors.add(color)
        bars[f"v{idx}"] = {"orientation": "v", "pos": int(c), "color": int(color)}

    return bars


def _intersection_edges(grid: np.ndarray, bars: dict[str, dict]) -> set[tuple[str, str]]:
    edges: set[tuple[str, str]] = set()
    horizontals = [bar for bar in bars.values() if bar["orientation"] == "h"]
    verticals = [bar for bar in bars.values() if bar["orientation"] == "v"]
    h_ids = [bar_id for bar_id, bar in bars.items() if bar["orientation"] == "h"]
    v_ids = [bar_id for bar_id, bar in bars.items() if bar["orientation"] == "v"]

    for h_id, h_bar in zip(h_ids, horizontals):
        for v_id, v_bar in zip(v_ids, verticals):
            visible = int(grid[int(h_bar["pos"]), int(v_bar["pos"])])
            if visible == int(h_bar["color"]):
                edges.add((h_id, v_id))
            elif visible == int(v_bar["color"]):
                edges.add((v_id, h_id))
            else:
                raise AssertionError("intersection color must match one of the two crossing bars")
    return edges


def _analyze_grid(grid: np.ndarray) -> tuple[dict[str, dict], set[tuple[str, str]], list[list[str]], str, str]:
    bars = _detect_bars(grid)
    edges = _intersection_edges(grid, bars)
    nodes = sorted(bars)
    orders = _all_linear_extensions(nodes, edges)
    assert orders, "layer constraints must admit at least one order"
    top_ids = {order[0] for order in orders}
    second_ids = {order[1] for order in orders}
    assert len(top_ids) == 1, "highest bar must be uniquely determined"
    assert len(second_ids) == 1, "second-highest bar must be uniquely determined"
    top_id = next(iter(top_ids))
    second_id = next(iter(second_ids))
    assert bars[top_id]["orientation"] != bars[second_id]["orientation"]
    return bars, edges, orders, top_id, second_id


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the unique top strip from crossing visibility, then mirror the second strip across that top strip's axis and erase its source.

    Input structure:
    - The background is black.
    - There are 3 or 4 full-length 1-pixel bars, each with a distinct non-black color.
    - Horizontal bars fill an entire row; vertical bars fill an entire column.
    - At every horizontal/vertical crossing, the visible color is the color of the higher-layer bar.

    Rule:
    - Detect every full-width row bar and full-height column bar, along with each bar's color.
    - Use the crossing colors to derive the bar layering order and identify the unique highest bar and the unique second-highest bar.
    - If the highest bar is horizontal, use its row as a vertical-flip axis; if it is vertical, use its column as a horizontal-flip axis.
    - Rebuild the scene without the original second-highest bar, then mirror that bar's full hidden-support row/column across the chosen axis and paint the mirrored cells in the second bar's color on top of everything else.

    Key insight:
    - The crossing colors do not directly produce the output; they encode which bar is topmost, and that inferred top bar determines both the mirror axis direction and which second-layer strip must be moved.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bars, _edges, orders, top_id, second_id = _analyze_grid(g)
    return _apply_rule_from_order(g.shape, bars, orders[0], top_id, second_id)


def _sample_axis_position(length: int) -> int:
    outer = [
        pos
        for pos in range(2, length - 2)
        if abs(2 * pos - (length - 1)) >= 4 and (pos <= length // 3 or pos >= length - 1 - length // 3)
    ]
    if not outer:
        outer = [pos for pos in range(1, length - 1) if abs(2 * pos - (length - 1)) >= 2]
    return int(random.choice(outer))


def _sample_more_positions(length: int, count: int, used: list[int]) -> list[int] | None:
    positions = list(used)
    for _ in range(count):
        options = [
            pos
            for pos in range(1, length - 1)
            if pos not in positions and all(abs(pos - prev) >= 2 for prev in positions)
        ]
        if not options:
            return None
        positions.append(int(random.choice(options)))
    return positions


def _make_chain_bars(shape: tuple[int, int], top_orientation: str, n_bars: int) -> tuple[dict[str, dict], list[str]] | None:
    h, w = map(int, shape)
    orientations = [top_orientation if i % 2 == 0 else ("v" if top_orientation == "h" else "h") for i in range(n_bars)]
    h_count = orientations.count("h")
    v_count = orientations.count("v")

    row_positions = []
    col_positions = []
    if top_orientation == "h":
        row_positions = _sample_more_positions(h, h_count, [_sample_axis_position(h)])
        col_positions = _sample_more_positions(w, v_count, [])
    else:
        row_positions = _sample_more_positions(h, h_count, [])
        col_positions = _sample_more_positions(w, v_count, [_sample_axis_position(w)])
    if row_positions is None or col_positions is None:
        return None

    random.shuffle(row_positions[1:] if len(row_positions) > 1 else row_positions)
    random.shuffle(col_positions[1:] if len(col_positions) > 1 else col_positions)

    colors = random.sample(BAR_COLORS, n_bars)
    row_iter = iter(row_positions)
    col_iter = iter(col_positions)
    bars: dict[str, dict] = {}
    order: list[str] = []

    for idx, orientation in enumerate(orientations):
        bar_id = f"b{idx}"
        pos = next(row_iter) if orientation == "h" else next(col_iter)
        bars[bar_id] = {"orientation": orientation, "pos": int(pos), "color": int(colors[idx])}
        order.append(bar_id)

    return bars, order


def _bar_signature(bar: dict) -> tuple[str, int, int]:
    return str(bar["orientation"]), int(bar["pos"]), int(bar["color"])


def _semantic_orders_match(grid: np.ndarray, expected_bars: dict[str, dict], expected_top: str, expected_second: str) -> bool:
    try:
        bars, _edges, _orders, top_id, second_id = _analyze_grid(grid)
    except AssertionError:
        return False
    return (
        _bar_signature(bars[top_id]) == _bar_signature(expected_bars[expected_top])
        and _bar_signature(bars[second_id]) == _bar_signature(expected_bars[expected_second])
    )


def _wrong_choice_collision(shape: tuple[int, int], bars: dict[str, dict], order: list[str], top_id: str, second_id: str, expected: np.ndarray) -> bool:
    for alt_top in order:
        for alt_second in order:
            if alt_top == alt_second:
                continue
            if bars[alt_top]["orientation"] == bars[alt_second]["orientation"]:
                continue
            if alt_top == top_id and alt_second == second_id:
                continue
            alt = _apply_rule_from_order(shape, bars, order, alt_top, alt_second)
            if np.array_equal(alt, expected):
                return True
    return False


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(12, 16)
    w = random.randint(12, 16)
    top_orientation = random.choice(("h", "v"))
    n_bars = random.choice((3, 4))

    made = _make_chain_bars((h, w), top_orientation, n_bars)
    if made is None:
        return None
    bars, order = made
    top_id = order[0]
    second_id = order[1]

    inp = _render_bars((h, w), bars, order)
    expected = _apply_rule_from_order((h, w), bars, order, top_id, second_id)

    mirror_coords = set(_mirror_coords((h, w), bars[second_id], bars[top_id]))
    source_coords = set(_full_bar_coords((h, w), bars[second_id]))

    if mirror_coords == source_coords:
        return None
    if len(mirror_coords) < 3:
        return None
    if np.array_equal(inp, expected):
        return None
    if not _semantic_orders_match(inp, bars, top_id, second_id):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    if _wrong_choice_collision((h, w), bars, order, top_id, second_id, expected):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
