from itertools import combinations, permutations

import numpy as np

from arc_utils import colors_present, get_bg
import random


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    h, w = shape
    if kind == "horizontal":
        return frozenset((value, c) for c in range(w))
    if kind == "vertical":
        return frozenset((r, value) for r in range(h))
    if kind == "diag_down":
        return frozenset((r, r - value) for r in range(h) if 0 <= r - value < w)
    if kind == "diag_up":
        return frozenset((r, value - r) for r in range(h) if 0 <= value - r < w)
    raise AssertionError(f"unknown line kind: {kind}")


def _infer_lines(g: np.ndarray, bg: int) -> dict[int, frozenset[tuple[int, int]]]:
    lines: dict[int, frozenset[tuple[int, int]]] = {}
    signatures: set[tuple[str, int]] = set()

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == color)
        assert coords.shape[0] >= 2
        rows = coords[:, 0]
        cols = coords[:, 1]

        candidates: list[tuple[str, int]] = []
        if np.all(rows == rows[0]):
            candidates.append(("horizontal", int(rows[0])))
        if np.all(cols == cols[0]):
            candidates.append(("vertical", int(cols[0])))
        down_values = rows - cols
        if np.all(down_values == down_values[0]):
            candidates.append(("diag_down", int(down_values[0])))
        up_values = rows + cols
        if np.all(up_values == up_values[0]):
            candidates.append(("diag_up", int(up_values[0])))

        assert len(candidates) == 1
        signature = candidates[0]
        assert signature not in signatures
        signatures.add(signature)

        cells = _line_cells(g.shape, *signature)
        assert set(map(tuple, coords)).issubset(cells)
        lines[int(color)] = cells

    assert lines
    return lines


def main(grid: np.ndarray) -> np.ndarray:
    """List the colors of overlaid straight lines in their only possible bottom-to-top crossing order.

    Input structure
    The background is the dominant color. Every foreground color is the visible part
    of one full one-cell-thick horizontal, vertical, or 45-degree diagonal line.
    Higher lines may hide pixels of lower lines at crossings.

    Rule
    Recover each full line from its visible collinear pixels. At any cell where
    multiple recovered lines cross, the visible color is above every other line
    through that cell. Return the unique ordering of all line colors that respects
    those above/below constraints and whose consecutive colors are lines that
    geometrically cross.

    Key insight
    The visible crossing colors give a partial layer order, and the intended answer
    is the unique Hamiltonian chain through the line-intersection graph from the
    bottom layer to the top layer.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    lines = _infer_lines(g, bg)
    colors = list(lines)
    below_edges: set[tuple[int, int]] = set()
    intersects: dict[int, set[int]] = {color: set() for color in colors}

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color, cells in lines.items() if (r, c) in cells]
            visible = int(g[r, c])

            if visible == bg:
                assert not covering
                continue

            assert visible in covering
            if len(covering) < 2:
                continue

            for a, b in combinations(covering, 2):
                intersects[a].add(b)
                intersects[b].add(a)
            for color in covering:
                if color != visible:
                    below_edges.add((color, visible))

    def is_valid_order(order: tuple[int, ...]) -> bool:
        positions = {color: i for i, color in enumerate(order)}
        if any(positions[below] >= positions[above] for below, above in below_edges):
            return False
        return all(order[i + 1] in intersects[order[i]] for i in range(len(order) - 1))

    valid_orders = [order for order in permutations(colors) if is_valid_order(order)]
    assert len(valid_orders) == 1
    return np.array([valid_orders[0]], dtype=int)

# --------------- generator (c) ---------------

# Canvas and line count bounds.
MIN_LINE_COUNT = 2
MAX_LINE_COUNT = 6
MIN_GRID_SIZE = 7
MAX_GRID_SIZE = 16
GRID_SIZE_PER_LINE = 3
GRID_SIZE_MARGIN = 2

# Line geometry and rendering.
LINE_KINDS = ['H', 'V', 'D', 'U']
DIAGONAL_MIN_CELLS = 3
BACKGROUND_COLOR = 7
COLOR_VALUES = list(range(10))
VISIBLE_LINE_MIN_PIXELS = 2

# Search and validation limits.
CHAIN_BUILD_ATTEMPTS = 50
GENERATION_RETRIES = 300


def _gen_line_cells(S, kind, value):
    """Return set of (r,c) for a line of given kind/value on an SxS grid."""
    if kind == 'H':
        return {(value, c) for c in range(S)}
    if kind == 'V':
        return {(r, value) for r in range(S)}
    if kind == 'D':  # diag_down: r - c = value
        return {(r, r - value) for r in range(S) if 0 <= r - value < S}
    if kind == 'U':  # diag_up: r + c = value
        return {(r, value - r) for r in range(S) if 0 <= value - r < S}
    raise ValueError(kind)


def _lines_cross(S, k1, v1, k2, v2):
    """Check if two lines cross within an SxS grid."""
    if k1 == k2:
        return False  # same-kind lines are parallel
    c1 = _gen_line_cells(S, k1, v1)
    c2 = _gen_line_cells(S, k2, v2)
    return bool(c1 & c2)


def _valid_params(S, kind):
    """Return parameters whose line has enough raw cells before occlusion."""
    if kind == 'H':
        return list(range(S))
    if kind == 'V':
        return list(range(S))
    if kind == 'D':
        return list(range(-(S - DIAGONAL_MIN_CELLS), S - DIAGONAL_MIN_CELLS + 1))
    if kind == 'U':
        return list(range(DIAGONAL_MIN_CELLS - 1, 2 * S - DIAGONAL_MIN_CELLS))
    raise ValueError(kind)


def _shuffled_unused_params(S, kind, used_params):
    params = [v for v in _valid_params(S, kind) if v not in used_params.get(kind, set())]
    if params:
        random.shuffle(params)
    return params


def _crosses_non_adjacent(S, lines, kind, value):
    """Check earlier chain lines, excluding the immediate predecessor."""
    for prior_kind, prior_value in lines[:-1]:
        if _lines_cross(S, prior_kind, prior_value, kind, value):
            return True
    return False


def _try_extend_chain(S, lines, used_params, candidate_kinds, avoid_non_adjacent):
    """Append one line crossing the current tail, optionally avoiding earlier lines."""
    prev_kind, prev_value = lines[-1]
    for kind in candidate_kinds:
        for value in _shuffled_unused_params(S, kind, used_params):
            if not _lines_cross(S, prev_kind, prev_value, kind, value):
                continue
            if avoid_non_adjacent and _crosses_non_adjacent(S, lines, kind, value):
                continue
            lines.append((kind, value))
            used_params.setdefault(kind, set()).add(value)
            return True
    return False


def _build_line_chain(S, N):
    """Greedily build a bottom-to-top line chain with crossing neighbors."""
    lines = []
    used_params = {}

    first_kind = random.choice(LINE_KINDS)
    first_params = _valid_params(S, first_kind)
    if not first_params:
        return None
    first_value = random.choice(first_params)
    lines.append((first_kind, first_value))
    used_params[first_kind] = {first_value}

    for _ in range(1, N):
        prev_kind, _ = lines[-1]
        candidate_kinds = [kind for kind in LINE_KINDS if kind != prev_kind]
        random.shuffle(candidate_kinds)
        placed = _try_extend_chain(S, lines, used_params, candidate_kinds, True)
        if not placed and not _try_extend_chain(S, lines, used_params, candidate_kinds, False):
            return None

    return lines


def _crossing_graph(S, lines):
    crosses = {i: set() for i in range(len(lines))}
    for i, j in combinations(range(len(lines)), 2):
        kind_i, value_i = lines[i]
        kind_j, value_j = lines[j]
        if _lines_cross(S, kind_i, value_i, kind_j, value_j):
            crosses[i].add(j)
            crosses[j].add(i)
    return crosses


def _valid_layer_orders(crosses):
    """Enumerate Hamiltonian orders that respect bottom-to-top occlusions."""
    N = len(crosses)
    below_edges = set()
    for i, j in combinations(range(N), 2):
        if j in crosses[i]:
            below_edges.add((i, j))

    def is_valid_order(order):
        positions = {idx: pos for pos, idx in enumerate(order)}
        for lower, higher in below_edges:
            if positions[lower] >= positions[higher]:
                return False
        for pos in range(len(order) - 1):
            if order[pos + 1] not in crosses[order[pos]]:
                return False
        return True

    return [order for order in permutations(range(N)) if is_valid_order(order)]


def _choose_line_colors(N):
    available_colors = [color for color in COLOR_VALUES if color != BACKGROUND_COLOR]
    return random.sample(available_colors, N)


def _render_lines(S, lines, colors):
    grid = np.full((S, S), BACKGROUND_COLOR, dtype=int)
    for i in range(len(lines)):
        kind, value = lines[i]
        color = colors[i]
        for r, c in _gen_line_cells(S, kind, value):
            grid[r, c] = color
    return grid


def _all_lines_visible(grid, colors):
    return all(np.sum(grid == color) >= VISIBLE_LINE_MIN_PIXELS for color in colors)


def _gen_attempt():
    N = random.randint(MIN_LINE_COUNT, MAX_LINE_COUNT)
    min_size = max(MIN_GRID_SIZE, GRID_SIZE_PER_LINE * N - GRID_SIZE_MARGIN)
    S = random.randint(min_size, MAX_GRID_SIZE)

    for _ in range(CHAIN_BUILD_ATTEMPTS):
        lines = _build_line_chain(S, N)
        if lines is None or len(lines) != N:
            continue

        crosses = _crossing_graph(S, lines)
        valid_orders = _valid_layer_orders(crosses)
        if len(valid_orders) != 1:
            continue

        assert valid_orders[0] == tuple(range(N))

        colors = _choose_line_colors(N)
        grid = _render_lines(S, lines, colors)
        if not _all_lines_visible(grid, colors):
            continue

        return grid

    return None


def generate_input():
    for _ in range(GENERATION_RETRIES):
        result = _gen_attempt()
        if result is not None:
            try:
                out = main(result)
            except Exception:
                continue
            # Verify the output is a 1xN array
            if out.ndim == 2 and out.shape[0] == 1:
                return result
    raise RuntimeError(f"Failed to generate valid input after {GENERATION_RETRIES} attempts")
