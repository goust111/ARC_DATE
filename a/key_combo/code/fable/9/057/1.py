from __future__ import annotations

from itertools import combinations, permutations
import random

import numpy as np


LINE_KINDS = ("H", "V", "D", "U")
COLOR_POOL = tuple(range(10))
MIN_LINE_COUNT = 4
MAX_LINE_COUNT = 6
MIN_GRID_SIZE = 9
MAX_GRID_SIZE = 16
MIN_RAW_LINE_LEN = 4
MIN_VISIBLE_PIXELS = 2
MAX_GENERATION_ATTEMPTS = 1200
MAX_ORDER_TRIES_PER_SET = 120
RECENT_OUTPUT_CAP = 256

_RECENT_OUTPUT_SIGNATURES: list[tuple[int, ...]] = []


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    colors = sorted(map(int, np.unique(grid)))
    if exclude:
        blocked = set(map(int, exclude))
        colors = [color for color in colors if color not in blocked]
    return colors


def _line_cells(shape: tuple[int, int], kind: str, value: int) -> frozenset[tuple[int, int]]:
    h, w = map(int, shape)
    if kind == "H":
        return frozenset((value, c) for c in range(w))
    if kind == "V":
        return frozenset((r, value) for r in range(h))
    if kind == "D":
        return frozenset((r, r - value) for r in range(h) if 0 <= r - value < w)
    if kind == "U":
        return frozenset((r, value - r) for r in range(h) if 0 <= value - r < w)
    raise AssertionError(f"unknown line kind: {kind}")


def _raw_line_length(size: int, kind: str, value: int) -> int:
    return len(_line_cells((size, size), kind, value))


def _valid_params(size: int, kind: str) -> list[int]:
    params: list[int] = []
    if kind in {"H", "V"}:
        params = list(range(size))
    elif kind == "D":
        params = list(range(-(size - 1), size))
    elif kind == "U":
        params = list(range(2 * size - 1))
    else:
        raise AssertionError(kind)
    return [value for value in params if _raw_line_length(size, kind, value) >= MIN_RAW_LINE_LEN]


def _lines_cross(size: int, a: tuple[str, int], b: tuple[str, int]) -> bool:
    if a[0] == b[0]:
        return False
    return bool(_line_cells((size, size), *a) & _line_cells((size, size), *b))


def _infer_lines(grid: np.ndarray, bg: int) -> dict[int, frozenset[tuple[int, int]]]:
    lines: dict[int, frozenset[tuple[int, int]]] = {}
    seen_signatures: set[tuple[str, int]] = set()
    for color in colors_present(grid, exclude={bg}):
        coords = np.argwhere(grid == color)
        assert coords.shape[0] >= 2, "each line needs at least two visible pixels"
        rows = coords[:, 0]
        cols = coords[:, 1]

        candidates: list[tuple[str, int]] = []
        if np.all(rows == rows[0]):
            candidates.append(("H", int(rows[0])))
        if np.all(cols == cols[0]):
            candidates.append(("V", int(cols[0])))
        down = rows - cols
        if np.all(down == down[0]):
            candidates.append(("D", int(down[0])))
        up = rows + cols
        if np.all(up == up[0]):
            candidates.append(("U", int(up[0])))

        assert len(candidates) == 1, "visible pixels must identify one unique full line"
        signature = candidates[0]
        assert signature not in seen_signatures, "two colors cannot belong to the same geometric line"
        seen_signatures.add(signature)

        full_cells = _line_cells(grid.shape, *signature)
        assert set(map(tuple, coords)).issubset(full_cells)
        lines[int(color)] = full_cells

    assert lines
    return lines


def _extract_relations(
    shape: tuple[int, int],
    lines: dict[int, frozenset[tuple[int, int]]],
    visible_lookup: dict[tuple[int, int], int],
) -> tuple[set[tuple[int, int]], dict[int, set[int]]]:
    below_edges: set[tuple[int, int]] = set()
    intersects: dict[int, set[int]] = {color: set() for color in lines}

    h, w = map(int, shape)
    for r in range(h):
        for c in range(w):
            covering = [color for color, cells in lines.items() if (r, c) in cells]
            if not covering:
                continue

            visible = visible_lookup[(r, c)]
            assert visible in covering
            if len(covering) < 2:
                continue

            for a, b in combinations(covering, 2):
                intersects[a].add(b)
                intersects[b].add(a)
            for color in covering:
                if color != visible:
                    below_edges.add((color, visible))

    return below_edges, intersects


def _enumerate_valid_orders(
    colors: list[int],
    below_edges: set[tuple[int, int]],
    intersects: dict[int, set[int]],
    *,
    require_crossing_chain: bool,
) -> list[tuple[int, ...]]:
    valid: list[tuple[int, ...]] = []
    for order in permutations(colors):
        positions = {color: idx for idx, color in enumerate(order)}
        if any(positions[lower] >= positions[upper] for lower, upper in below_edges):
            continue
        if require_crossing_chain and any(order[idx + 1] not in intersects[order[idx]] for idx in range(len(order) - 1)):
            continue
        valid.append(order)
    return valid


def _ring_grid(order: tuple[int, ...]) -> np.ndarray:
    n = len(order)
    size = 2 * n - 1
    center = n - 1
    out = np.empty((size, size), dtype=int)
    for r in range(size):
        for c in range(size):
            depth = max(abs(r - center), abs(c - center))
            out[r, c] = order[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Rewrite the unique bottom-to-top order of overlaid straight lines as concentric square rings.

    Input structure
    The dominant background surrounds several one-cell-thick straight lines, each
    shown in a different foreground color. Every foreground color belongs to one
    full horizontal, vertical, or 45-degree diagonal line, but crossings only show
    the color of the higher visible line.

    Rule
    Recover each complete line from its visible collinear pixels. At every crossing
    cell, the visible line is above every other recovered line through that cell.
    Among all orders that respect those above/below constraints, keep the unique
    full bottom-to-top order whose neighboring lines really intersect geometrically.
    If that order has n colors, output a (2n-1)x(2n-1) grid whose center is the
    bottom color, whose Chebyshev-distance-1 ring is the next color, and so on
    until the top color fills the outermost square ring.

    Key insight
    The input only reveals local occlusion facts at crossings, so the hidden depth
    chain must be reconstructed first; only then can depth be re-encoded as ring radius.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = get_bg(g)
    lines = _infer_lines(g, bg)
    visible_lookup: dict[tuple[int, int], int] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) != bg:
                visible_lookup[(r, c)] = int(g[r, c])

    below_edges, intersects = _extract_relations(g.shape, lines, visible_lookup)
    colors = list(lines)

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            covering = [color for color, cells in lines.items() if (r, c) in cells]
            if int(g[r, c]) == bg:
                assert not covering
            else:
                assert (r, c) in visible_lookup

    valid_orders = _enumerate_valid_orders(colors, below_edges, intersects, require_crossing_chain=True)
    assert len(valid_orders) == 1, "expected one unique bottom-to-top line order"
    return _ring_grid(valid_orders[0])


def _crossing_graph(size: int, signatures: list[tuple[str, int]]) -> dict[int, set[int]]:
    graph = {idx: set() for idx in range(len(signatures))}
    for i, j in combinations(range(len(signatures)), 2):
        if _lines_cross(size, signatures[i], signatures[j]):
            graph[i].add(j)
            graph[j].add(i)
    return graph


def _hamiltonian_orders(graph: dict[int, set[int]]) -> list[tuple[int, ...]]:
    nodes = sorted(graph)
    return _enumerate_valid_orders(nodes, set(), graph, require_crossing_chain=True)


def _order_relations(
    size: int,
    signatures: list[tuple[str, int]],
    order: tuple[int, ...],
) -> tuple[set[tuple[int, int]], dict[int, set[int]], list[int], int]:
    line_cells = [_line_cells((size, size), *signature) for signature in signatures]
    position = {line_idx: idx for idx, line_idx in enumerate(order)}
    below_edges: set[tuple[int, int]] = set()
    graph = {idx: set() for idx in range(len(signatures))}
    visible_counts = [0] * len(signatures)
    max_covering = 0

    for r in range(size):
        for c in range(size):
            covering = [idx for idx, cells in enumerate(line_cells) if (r, c) in cells]
            if not covering:
                continue
            max_covering = max(max_covering, len(covering))
            top_idx = max(covering, key=lambda idx: position[idx])
            visible_counts[top_idx] += 1
            for a, b in combinations(covering, 2):
                graph[a].add(b)
                graph[b].add(a)
            for idx in covering:
                if idx != top_idx:
                    below_edges.add((idx, top_idx))

    return below_edges, graph, visible_counts, max_covering


def _render_grid(
    size: int,
    signatures: list[tuple[str, int]],
    order: tuple[int, ...],
    bg: int,
    line_colors: list[int],
) -> np.ndarray:
    grid = np.full((size, size), bg, dtype=int)
    line_cells = [_line_cells((size, size), *signature) for signature in signatures]
    for line_idx in order:
        color = line_colors[line_idx]
        for r, c in line_cells[line_idx]:
            grid[r, c] = color
    return grid


def _sample_focus_lines(size: int, count: int) -> list[tuple[str, int]] | None:
    if count < 3 or size < 6:
        return None
    focus_r = random.randint(2, size - 3)
    focus_c = random.randint(2, size - 3)
    hub_kind_count = random.choice([3, 4])
    chosen_kinds = random.sample(LINE_KINDS, hub_kind_count)
    signatures: list[tuple[str, int]] = []
    if "H" in chosen_kinds:
        signatures.append(("H", focus_r))
    if "V" in chosen_kinds:
        signatures.append(("V", focus_c))
    if "D" in chosen_kinds:
        signatures.append(("D", focus_r - focus_c))
    if "U" in chosen_kinds:
        signatures.append(("U", focus_r + focus_c))

    used_by_kind = {kind: {value} for kind, value in signatures}
    all_candidates = [(kind, value) for kind in LINE_KINDS for value in _valid_params(size, kind)]
    random.shuffle(all_candidates)
    for signature in all_candidates:
        if len(signatures) >= count:
            break
        kind, value = signature
        if value in used_by_kind.get(kind, set()):
            continue
        if not any(_lines_cross(size, signature, other) for other in signatures):
            continue
        signatures.append(signature)
        used_by_kind.setdefault(kind, set()).add(value)

    if len(signatures) != count:
        return None
    return signatures


def _sample_chain_lines(size: int, count: int) -> list[tuple[str, int]] | None:
    used: dict[str, set[int]] = {}
    lines: list[tuple[str, int]] = []
    first_kind = random.choice(LINE_KINDS)
    first_value = random.choice(_valid_params(size, first_kind))
    lines.append((first_kind, first_value))
    used[first_kind] = {first_value}

    for _ in range(1, count):
        prev = lines[-1]
        candidates = [(kind, value) for kind in LINE_KINDS for value in _valid_params(size, kind)]
        random.shuffle(candidates)
        placed = False
        for signature in candidates:
            kind, value = signature
            if value in used.get(kind, set()):
                continue
            if not _lines_cross(size, prev, signature):
                continue
            lines.append(signature)
            used.setdefault(kind, set()).add(value)
            placed = True
            break
        if not placed:
            return None

    return lines


def _sample_signatures(size: int, count: int) -> list[tuple[str, int]] | None:
    builders = [_sample_focus_lines, _sample_chain_lines]
    random.shuffle(builders)
    for builder in builders:
        signatures = builder(size, count)
        if signatures is None:
            continue
        if len({signature for signature in signatures}) != count:
            continue
        graph = _crossing_graph(size, signatures)
        if any(not graph[idx] for idx in graph):
            continue
        return signatures
    return None


def _output_signature(grid: np.ndarray) -> tuple[int, ...]:
    flat = tuple(map(int, grid.ravel()))
    return (int(grid.shape[0]), int(grid.shape[1]), *flat)


def _recently_used_output(signature: tuple[int, ...]) -> bool:
    return signature in _RECENT_OUTPUT_SIGNATURES


def _remember_output(signature: tuple[int, ...]) -> None:
    _RECENT_OUTPUT_SIGNATURES.append(signature)
    if len(_RECENT_OUTPUT_SIGNATURES) > RECENT_OUTPUT_CAP:
        del _RECENT_OUTPUT_SIGNATURES[0]


def _gen_attempt() -> np.ndarray | None:
    line_count = random.randint(MIN_LINE_COUNT, MAX_LINE_COUNT)
    size_min = max(MIN_GRID_SIZE, 2 * line_count + 1)
    size = random.randint(size_min, MAX_GRID_SIZE)

    signatures = _sample_signatures(size, line_count)
    if signatures is None:
        return None

    graph = _crossing_graph(size, signatures)
    graph_orders = _hamiltonian_orders(graph)
    if len(graph_orders) <= 1:
        return None

    candidate_orders = graph_orders[:]
    random.shuffle(candidate_orders)
    for order in candidate_orders[:MAX_ORDER_TRIES_PER_SET]:
        below_edges, relation_graph, visible_counts, max_covering = _order_relations(size, signatures, order)
        if max_covering < 3:
            continue
        if min(visible_counts) < MIN_VISIBLE_PIXELS:
            continue

        topo_orders = _enumerate_valid_orders(list(range(line_count)), below_edges, relation_graph, require_crossing_chain=False)
        if len(topo_orders) <= 1:
            continue

        combined_orders = _enumerate_valid_orders(list(range(line_count)), below_edges, relation_graph, require_crossing_chain=True)
        if combined_orders != [order]:
            continue

        bg = random.choice(COLOR_POOL)
        usable_colors = [color for color in COLOR_POOL if color != bg]
        line_colors = random.sample(usable_colors, line_count)
        grid = _render_grid(size, signatures, order, bg, line_colors)

        out = main(grid)
        expected_order = tuple(line_colors[line_idx] for line_idx in order)
        expected_out = _ring_grid(expected_order)
        if not np.array_equal(out, expected_out):
            continue
        if _recently_used_output(_output_signature(out)):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is None:
            continue
        out = main(grid)
        signature = _output_signature(out)
        if _recently_used_output(signature):
            continue
        _remember_output(signature)
        return grid
    raise RuntimeError("failed to generate a valid layered-ring input")
