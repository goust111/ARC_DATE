from __future__ import annotations

import random
from itertools import permutations

import numpy as np


BG = 0
MARK = 5
BAR_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _color_positions(grid: np.ndarray, color: int) -> np.ndarray:
    return np.argwhere(grid == int(color))


def _infer_bars(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    bars: list[dict] = []
    seen_colors = sorted(int(color) for color in np.unique(grid) if int(color) != BG)
    for color in seen_colors:
        coords = _color_positions(grid, color)
        rows = coords[:, 0]
        cols = coords[:, 1]
        if np.all(rows == rows[0]) and int(cols.min()) == 0 and int(cols.max()) == w - 1:
            bars.append({"color": int(color), "orient": "H", "pos": int(rows[0])})
        elif np.all(cols == cols[0]) and int(rows.min()) == 0 and int(rows.max()) == h - 1:
            bars.append({"color": int(color), "orient": "V", "pos": int(cols[0])})
    assert 3 <= len(bars) <= 4, "expected 3-4 visible full-span bars"
    assert any(bar["orient"] == "H" for bar in bars), "expected at least one horizontal bar"
    assert any(bar["orient"] == "V" for bar in bars), "expected at least one vertical bar"
    colors = [int(bar["color"]) for bar in bars]
    assert len(colors) == len(set(colors)), "bar colors must be distinct"
    return bars


def _visible_order_constraints(grid: np.ndarray, bars: list[dict]) -> set[tuple[int, int]]:
    horizontal = [bar for bar in bars if bar["orient"] == "H"]
    vertical = [bar for bar in bars if bar["orient"] == "V"]
    edges: set[tuple[int, int]] = set()
    for hbar in horizontal:
        r = int(hbar["pos"])
        hcolor = int(hbar["color"])
        for vbar in vertical:
            c = int(vbar["pos"])
            vcolor = int(vbar["color"])
            visible = int(grid[r, c])
            assert visible in (hcolor, vcolor), "crossing must show one of its two bars"
            if visible == hcolor:
                edges.add((vcolor, hcolor))
            else:
                edges.add((hcolor, vcolor))
    assert edges, "crossings must reveal layer constraints"
    return edges


def _unique_layer_order(colors: list[int], edges: set[tuple[int, int]]) -> list[int]:
    valid_orders: list[tuple[int, ...]] = []
    for order in permutations(colors):
        index = {int(color): pos for pos, color in enumerate(order)}
        if all(index[int(lower)] < index[int(upper)] for lower, upper in edges):
            valid_orders.append(order)
    assert len(valid_orders) == 1, "crossings must determine a unique total order"
    return [int(color) for color in valid_orders[0]]


def _bar_maps(bars: list[dict]) -> tuple[list[int], list[int], dict[int, int], dict[int, int]]:
    row_to_color = {int(bar["pos"]): int(bar["color"]) for bar in bars if bar["orient"] == "H"}
    col_to_color = {int(bar["pos"]): int(bar["color"]) for bar in bars if bar["orient"] == "V"}
    rows = sorted(row_to_color)
    cols = sorted(col_to_color)
    return rows, cols, row_to_color, col_to_color


def _chambers(
    shape: tuple[int, int],
    rows: list[int],
    cols: list[int],
    row_to_color: dict[int, int],
    col_to_color: dict[int, int],
) -> list[dict]:
    h, w = map(int, shape)
    row_bounds = [-1] + [int(r) for r in rows] + [h]
    col_bounds = [-1] + [int(c) for c in cols] + [w]
    chambers: list[dict] = []
    for i in range(len(row_bounds) - 1):
        r0 = int(row_bounds[i] + 1)
        r1 = int(row_bounds[i + 1] - 1)
        if r0 > r1:
            continue
        top_row = None if row_bounds[i] < 0 else int(row_bounds[i])
        bottom_row = None if row_bounds[i + 1] == h else int(row_bounds[i + 1])
        for j in range(len(col_bounds) - 1):
            c0 = int(col_bounds[j] + 1)
            c1 = int(col_bounds[j + 1] - 1)
            if c0 > c1:
                continue
            left_col = None if col_bounds[j] < 0 else int(col_bounds[j])
            right_col = None if col_bounds[j + 1] == w else int(col_bounds[j + 1])
            boundary_colors: list[int] = []
            if top_row is not None:
                boundary_colors.append(int(row_to_color[top_row]))
            if bottom_row is not None:
                boundary_colors.append(int(row_to_color[bottom_row]))
            if left_col is not None:
                boundary_colors.append(int(col_to_color[left_col]))
            if right_col is not None:
                boundary_colors.append(int(col_to_color[right_col]))
            assert 2 <= len(boundary_colors) <= 4
            chambers.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "boundary_colors": boundary_colors,
                    "boundary_count": len(boundary_colors),
                }
            )
    assert len(chambers) >= 4, "bars must form at least four chambers"
    return chambers


def _expected_output_from_structure(
    grid: np.ndarray,
    bars: list[dict],
    layer_order: list[int],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    rows, cols, row_to_color, col_to_color = _bar_maps(bars)
    chambers = _chambers(grid.shape, rows, cols, row_to_color, col_to_color)
    rank = {int(color): idx for idx, color in enumerate(layer_order)}
    strip_rows = set(rows)
    strip_cols = set(cols)

    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bbox"]
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert not any(r in strip_rows for r in range(r0, r1 + 1))
        assert not any(c in strip_cols for c in range(c0, c1 + 1))
        allowed = {BG, MARK}
        assert set(map(int, np.unique(region))).issubset(allowed), "non-strip chamber cells must be bg or markers"
        markers = np.argwhere(region == MARK)
        if markers.size:
            assert markers.shape[0] == 1, "each marked chamber must contain exactly one marker"
            fill_color = MARK
        else:
            fill_color = max(chamber["boundary_colors"], key=lambda color: rank[int(color)])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fill_color)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Fill each strip-bounded chamber using either its marker color or the highest-layer boundary strip color.

    Input structure:
    - Black background contains 3-4 distinct full-span one-pixel strips, each entirely horizontal or vertical.
    - Horizontal and vertical strips cross orthogonally; each crossing shows only the color of the upper strip.
    - The strips partition the remaining non-strip cells into rectangular chambers, and some chambers contain one gray marker pixel.

    Rule:
    - Reconstruct every strip from its visible full-span row or column and treat every horizontal/vertical pair as a crossing.
    - At each crossing, the visible strip color is above the hidden strip, and all such constraints must imply one unique bottom-to-top strip order.
    - For every chamber, keep strip pixels unchanged; if the chamber contains a gray marker then fill all its non-strip cells with gray, otherwise fill the chamber with the color of its highest-layer boundary strip.

    Key insight:
    - The crossing colors do not directly define the output; they first encode a unique strip layer order, and that inferred order then parameterizes the chamber flood-fill color unless a marker overrides it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bars = _infer_bars(g)
    colors = [int(bar["color"]) for bar in bars]
    edges = _visible_order_constraints(g, bars)
    order = _unique_layer_order(colors, edges)
    return _expected_output_from_structure(g, bars, order)


def _sample_segment_sizes(total: int, parts: int, min_size: int) -> list[int] | None:
    remainder = int(total) - int(parts) * int(min_size)
    if remainder < 0:
        return None
    cuts = sorted(random.sample(range(remainder + parts - 1), parts - 1)) if parts > 1 else []
    pieces: list[int] = []
    prev = -1
    for cut in cuts + [remainder + parts - 1]:
        pieces.append(int(cut - prev - 1 + min_size))
        prev = cut
    random.shuffle(pieces)
    return pieces


def _sample_positions(length: int, count: int, min_gap: int = 2) -> list[int] | None:
    segments = _sample_segment_sizes(length - count, count + 1, min_gap)
    if segments is None:
        return None
    positions: list[int] = []
    cursor = int(segments[0])
    for i in range(count):
        positions.append(cursor)
        cursor += 1 + int(segments[i + 1])
    return positions


def _render_bars(
    shape: tuple[int, int],
    bars: list[dict],
    layer_order: list[int],
) -> np.ndarray:
    h, w = map(int, shape)
    color_to_bar = {int(bar["color"]): bar for bar in bars}
    grid = np.full((h, w), BG, dtype=int)
    for color in layer_order:
        bar = color_to_bar[int(color)]
        if bar["orient"] == "H":
            grid[int(bar["pos"]), :] = int(color)
        else:
            grid[:, int(bar["pos"])] = int(color)
    return grid


def _select_marked_chambers(
    chambers: list[dict],
    layer_order: list[int],
) -> set[int] | None:
    top_color = int(layer_order[-1])
    default_colors = []
    rank = {int(color): idx for idx, color in enumerate(layer_order)}
    for chamber in chambers:
        default_colors.append(
            max(chamber["boundary_colors"], key=lambda color: rank[int(color)])
        )

    all_indices = list(range(len(chambers)))
    internal = [idx for idx, chamber in enumerate(chambers) if chamber["boundary_count"] == 4]
    edge = [idx for idx, chamber in enumerate(chambers) if chamber["boundary_count"] < 4]

    for _ in range(100):
        mark_count = random.randint(1, len(chambers) - 1)
        marked = set(random.sample(all_indices, mark_count))
        if internal and random.random() < 0.5:
            marked.add(random.choice(internal))
        if edge and random.random() < 0.5:
            marked.add(random.choice(edge))
        if len(marked) == len(chambers):
            marked.remove(random.choice(tuple(marked)))
        unmarked = [idx for idx in all_indices if idx not in marked]
        if not unmarked:
            continue
        unmarked_colors = {int(default_colors[idx]) for idx in unmarked}
        if len(unmarked_colors) < 2:
            continue
        if not any(int(default_colors[idx]) != top_color for idx in unmarked):
            continue
        return marked
    return None


def _place_markers(grid: np.ndarray, chambers: list[dict], marked: set[int]) -> None:
    for idx in marked:
        r0, c0, r1, c1 = chambers[idx]["bbox"]
        r = random.randint(r0, r1)
        c = random.randint(c0, c1)
        grid[r, c] = MARK


def _truth_output(
    input_grid: np.ndarray,
    chambers: list[dict],
    marked: set[int],
    layer_order: list[int],
) -> np.ndarray:
    out = np.array(input_grid, dtype=int, copy=True)
    rank = {int(color): idx for idx, color in enumerate(layer_order)}
    for idx, chamber in enumerate(chambers):
        r0, c0, r1, c1 = chamber["bbox"]
        if idx in marked:
            fill_color = MARK
        else:
            fill_color = max(chamber["boundary_colors"], key=lambda color: rank[int(color)])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(fill_color)
    return out


def _gen_attempt() -> np.ndarray | None:
    branch = random.choices([(1, 2), (2, 1), (2, 2)], weights=[2, 2, 3], k=1)[0]
    h_count, v_count = map(int, branch)
    h = random.randint(12, 18)
    w = random.randint(12, 18)
    rows = _sample_positions(h, h_count, min_gap=2)
    cols = _sample_positions(w, v_count, min_gap=2)
    if rows is None or cols is None:
        return None

    colors = random.sample(BAR_COLORS, h_count + v_count)
    bars: list[dict] = []
    for row in rows:
        bars.append({"orient": "H", "pos": int(row), "color": int(colors.pop())})
    for col in cols:
        bars.append({"orient": "V", "pos": int(col), "color": int(colors.pop())})

    layer_order = [int(bar["color"]) for bar in bars]
    random.shuffle(layer_order)
    rendered = _render_bars((h, w), bars, layer_order)

    try:
        inferred = _unique_layer_order(
            [int(bar["color"]) for bar in bars],
            _visible_order_constraints(rendered, bars),
        )
    except AssertionError:
        return None
    if inferred != layer_order:
        return None

    row_to_color = {int(bar["pos"]): int(bar["color"]) for bar in bars if bar["orient"] == "H"}
    col_to_color = {int(bar["pos"]): int(bar["color"]) for bar in bars if bar["orient"] == "V"}
    chambers = _chambers(rendered.shape, sorted(rows), sorted(cols), row_to_color, col_to_color)
    marked = _select_marked_chambers(chambers, layer_order)
    if marked is None:
        return None

    inp = np.array(rendered, dtype=int, copy=True)
    _place_markers(inp, chambers, marked)
    expected = _truth_output(inp, chambers, marked, layer_order)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None

    unique_outputs = {int(expected[r0, c0]) for r0, c0, _r1, _c1 in (ch["bbox"] for ch in chambers)}
    if len(unique_outputs) < 3:
        return None
    if np.count_nonzero(inp == MARK) != len(marked):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_RETRIES} attempts")
