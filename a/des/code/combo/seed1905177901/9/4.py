import random

import numpy as np


def _has_h(kind: str) -> bool:
    return kind in ("h", "hv")


def _has_v(kind: str) -> bool:
    return kind in ("v", "hv")


def _infer_structure(coords: np.ndarray) -> tuple[int | None, int | None]:
    """Return the unique (horizontal row, vertical col) footprint for one color."""
    rows = sorted({int(r) for r in coords[:, 0]})
    cols = sorted({int(c) for c in coords[:, 1]})

    if len(rows) == 1 and len(cols) > 1:
        return rows[0], None
    if len(cols) == 1 and len(rows) > 1:
        return None, cols[0]

    candidates: list[tuple[int, int]] = []
    for row in rows:
        for col in cols:
            if not np.all((coords[:, 0] == row) | (coords[:, 1] == col)):
                continue
            has_h = bool(np.any((coords[:, 0] == row) & (coords[:, 1] != col)))
            has_v = bool(np.any((coords[:, 1] == col) & (coords[:, 0] != row)))
            if has_h and has_v:
                candidates.append((row, col))

    assert len(candidates) == 1, "each color must expose a unique row/col skeleton"
    return candidates[0]


def _segments(grid: np.ndarray) -> dict[int, dict[str, tuple[int, int] | int | None]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert int(np.min(g)) >= 0
    bg = 0
    colors = [int(color) for color in np.unique(g) if int(color) != bg]
    assert colors, "expected at least one foreground color"

    segments: dict[int, dict[str, tuple[int, int] | int | None]] = {}
    for color in colors:
        coords = np.argwhere(g == color)
        row, col = _infer_structure(coords)

        hspan = None
        if row is not None:
            row_cols = coords[coords[:, 0] == row, 1]
            assert row_cols.size >= 2, "horizontal segment must reveal both endpoints"
            hspan = (int(np.min(row_cols)), int(np.max(row_cols)))

        vspan = None
        if col is not None:
            col_rows = coords[coords[:, 1] == col, 0]
            assert col_rows.size >= 2, "vertical segment must reveal both endpoints"
            vspan = (int(np.min(col_rows)), int(np.max(col_rows)))

        segments[color] = {"row": row, "col": col, "hspan": hspan, "vspan": vspan}

    return segments


def _layer_graph(grid: np.ndarray, segments: dict[int, dict[str, tuple[int, int] | int | None]]) -> dict[int, set[int]]:
    g = np.asarray(grid, dtype=int)
    edges = {color: set() for color in segments}

    for h_color, h_info in segments.items():
        row = h_info["row"]
        hspan = h_info["hspan"]
        if row is None or hspan is None:
            continue
        h0, h1 = hspan

        for v_color, v_info in segments.items():
            if h_color == v_color:
                continue
            col = v_info["col"]
            vspan = v_info["vspan"]
            if col is None or vspan is None:
                continue
            v0, v1 = vspan
            if not (h0 <= col <= h1 and v0 <= row <= v1):
                continue

            visible = int(g[row, col])
            assert visible in {h_color, v_color}, "crossing must show one of the two line colors"
            if visible == h_color:
                low, high = v_color, h_color
            else:
                low, high = h_color, v_color
            edges[low].add(high)

    return edges


def _unique_topological_order(edges: dict[int, set[int]]) -> list[int]:
    indeg = {node: 0 for node in edges}
    for src, dsts in edges.items():
        for dst in dsts:
            assert dst in indeg and src in indeg
            indeg[dst] += 1

    ready = sorted(node for node, deg in indeg.items() if deg == 0)
    order: list[int] = []
    while ready:
        assert len(ready) == 1, "layer order must be uniquely determined"
        node = ready.pop()
        order.append(node)
        for dst in sorted(edges[node]):
            indeg[dst] -= 1
            if indeg[dst] == 0:
                ready.append(dst)
        ready.sort()

    assert len(order) == len(edges), "layer graph must be acyclic"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the bottom-to-top color order from overlapping horizontal/vertical line layers.

    Input structure
    - Black background.
    - Each foreground color forms at most one horizontal segment and at most one
      vertical segment.
    - Crossings show only the topmost color.

    Rule
    - Infer each color's hidden full segment(s) from its visible row/column footprint.
    - At every reconstructed crossing, the visible color sits above the hidden one.
    - Topologically sort the resulting partial order and output the colors from
      bottom layer to top layer as a 1xK row.
    """
    segments = _segments(np.asarray(grid, dtype=int))
    edges = _layer_graph(grid, segments)
    order = _unique_topological_order(edges)
    return np.asarray([order], dtype=int)


ORIENTATION_TYPES = ("h", "v", "hv")
COLOR_POOL = tuple(range(1, 10))
COLOR_COUNT_RANGE = (3, 6)
EXTRA_EDGE_PROB = 0.35
AXIS_START_CHOICES = (1, 2)
AXIS_GAP_CHOICES = (2, 3)
AXIS_TAIL_CHOICES = (1, 2, 3)
SPAN_MARGIN_CHOICES = (1, 2)
MAX_GENERATION_ATTEMPTS = 400


def _compatible_crossings(low_kind: str, high_kind: str) -> list[str]:
    choices: list[str] = []
    if _has_h(low_kind) and _has_v(high_kind):
        choices.append("low_h__high_v")
    if _has_v(low_kind) and _has_h(high_kind):
        choices.append("high_h__low_v")
    return choices


def _sample_orientations(num_colors: int) -> list[str] | None:
    kinds = [random.choice(ORIENTATION_TYPES) for _ in range(num_colors)]
    if not any(_has_h(kind) for kind in kinds):
        return None
    if not any(_has_v(kind) for kind in kinds):
        return None
    for low_kind, high_kind in zip(kinds, kinds[1:]):
        if not _compatible_crossings(low_kind, high_kind):
            return None
    return kinds


def _sample_axis_positions(count: int) -> tuple[list[int], int]:
    cur = random.choice(AXIS_START_CHOICES)
    positions: list[int] = []
    for _ in range(count):
        positions.append(cur)
        cur += random.choice(AXIS_GAP_CHOICES)
    size = cur + random.choice(AXIS_TAIL_CHOICES)
    return positions, size


def _apply_crossing(
    mode: str,
    low_color: int,
    high_color: int,
    row_of: dict[int, int],
    col_of: dict[int, int],
    h_cross: dict[int, set[int]],
    v_cross: dict[int, set[int]],
) -> None:
    if mode == "low_h__high_v":
        h_cross[low_color].add(col_of[high_color])
        v_cross[high_color].add(row_of[low_color])
    else:
        assert mode == "high_h__low_v"
        h_cross[high_color].add(col_of[low_color])
        v_cross[low_color].add(row_of[high_color])


def _build_spans(
    colors: list[int],
    kinds: dict[int, str],
    row_of: dict[int, int],
    col_of: dict[int, int],
    h_cross: dict[int, set[int]],
    v_cross: dict[int, set[int]],
    height: int,
    width: int,
) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]] | None:
    hspan: dict[int, tuple[int, int]] = {}
    for color in colors:
        if not _has_h(kinds[color]):
            continue
        anchors = set(h_cross[color])
        if _has_v(kinds[color]):
            anchors.add(col_of[color])
        if not anchors:
            anchors.add(random.randint(1, width - 2))
        start = min(anchors) - random.choice(SPAN_MARGIN_CHOICES)
        end = max(anchors) + random.choice(SPAN_MARGIN_CHOICES)
        if start < 0 or end >= width or end - start < 2:
            return None
        hspan[color] = (start, end)

    vspan: dict[int, tuple[int, int]] = {}
    for color in colors:
        if not _has_v(kinds[color]):
            continue
        anchors = set(v_cross[color])
        if _has_h(kinds[color]):
            anchors.add(row_of[color])
        if not anchors:
            anchors.add(random.randint(1, height - 2))
        start = min(anchors) - random.choice(SPAN_MARGIN_CHOICES)
        end = max(anchors) + random.choice(SPAN_MARGIN_CHOICES)
        if start < 0 or end >= height or end - start < 2:
            return None
        vspan[color] = (start, end)

    return hspan, vspan


def _render_grid(
    order: list[int],
    kinds: dict[int, str],
    row_of: dict[int, int],
    col_of: dict[int, int],
    hspan: dict[int, tuple[int, int]],
    vspan: dict[int, tuple[int, int]],
    height: int,
    width: int,
) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)
    for color in order:
        if _has_h(kinds[color]):
            row = row_of[color]
            start, end = hspan[color]
            grid[row, start : end + 1] = color
        if _has_v(kinds[color]):
            col = col_of[color]
            start, end = vspan[color]
            grid[start : end + 1, col] = color
    return grid


def _gen_attempt() -> np.ndarray | None:
    num_colors = random.randint(*COLOR_COUNT_RANGE)
    order = random.sample(COLOR_POOL, num_colors)

    kind_list = _sample_orientations(num_colors)
    if kind_list is None:
        return None
    kinds = {color: kind for color, kind in zip(order, kind_list)}

    h_colors = [color for color in order if _has_h(kinds[color])]
    v_colors = [color for color in order if _has_v(kinds[color])]
    h_positions, height = _sample_axis_positions(len(h_colors))
    v_positions, width = _sample_axis_positions(len(v_colors))
    random.shuffle(h_positions)
    random.shuffle(v_positions)
    row_of = {color: pos for color, pos in zip(h_colors, h_positions)}
    col_of = {color: pos for color, pos in zip(v_colors, v_positions)}

    h_cross = {color: set() for color in h_colors}
    v_cross = {color: set() for color in v_colors}

    for low_color, high_color in zip(order, order[1:]):
        choices = _compatible_crossings(kinds[low_color], kinds[high_color])
        if not choices:
            return None
        _apply_crossing(random.choice(choices), low_color, high_color, row_of, col_of, h_cross, v_cross)

    for i, low_color in enumerate(order):
        for high_color in order[i + 2 :]:
            choices = _compatible_crossings(kinds[low_color], kinds[high_color])
            if choices and random.random() < EXTRA_EDGE_PROB:
                _apply_crossing(random.choice(choices), low_color, high_color, row_of, col_of, h_cross, v_cross)

    spans = _build_spans(order, kinds, row_of, col_of, h_cross, v_cross, height, width)
    if spans is None:
        return None
    hspan, vspan = spans
    grid = _render_grid(order, kinds, row_of, col_of, hspan, vspan, height, width)

    try:
        expected = np.asarray([order], dtype=int)
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid overlapping-line input")
