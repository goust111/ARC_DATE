import random

import numpy as np


RING_SIZE = 5
RING = tuple(
    (r, c)
    for r in range(RING_SIZE)
    for c in range(RING_SIZE)
    if r in (0, RING_SIZE - 1) or c in (0, RING_SIZE - 1)
)
RING_SET = frozenset(RING)


def _background(grid: np.ndarray) -> int:
    vals = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(vals).argmax())


def _ring_cells(offset: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r0, c0 = offset
    return tuple((r0 + dr, c0 + dc) for dr, dc in RING)


def _ring_cell_set(offset: tuple[int, int]) -> set[tuple[int, int]]:
    return set(_ring_cells(offset))


def _legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[int, int], list[int]]:
    h, _w = grid.shape
    top = grid[h - 2]
    bottom = grid[h - 1]
    cols = np.where((top != bg) | (bottom != bg))[0]
    assert cols.size > 0, "legend must occupy the bottom edge"
    c0 = int(cols[0])
    c1 = int(cols[-1])
    assert np.array_equal(cols, np.arange(c0, c1 + 1)), "legend columns must be contiguous"

    order = [int(v) for v in top[c0 : c1 + 1]]
    mapped = [int(v) for v in bottom[c0 : c1 + 1]]
    assert all(v != bg for v in order + mapped), "legend cells must be colored"
    mapping = dict(zip(order, mapped))
    assert len(mapping) == len(order), "legend keys must be unique"
    return mapping, order


def _matching_offsets(
    coords: list[tuple[int, int]],
    grid_shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = grid_shape
    legend_top = h - 2
    max_r0 = legend_top - RING_SIZE - 2
    max_c0 = w - RING_SIZE
    counts: dict[tuple[int, int], int] = {}
    for r, c in coords:
        for dr, dc in RING:
            r0 = r - dr
            c0 = c - dc
            if 0 <= r0 <= max_r0 and 0 <= c0 <= max_c0:
                key = (int(r0), int(c0))
                counts[key] = counts.get(key, 0) + 1
    return [offset for offset, count in counts.items() if count == len(coords)]


def _recover_positions(grid: np.ndarray, ring_colors: list[int]) -> dict[int, tuple[int, int]]:
    scene = grid[:-2]
    positions: dict[int, tuple[int, int]] = {}
    for color in ring_colors:
        coords = [tuple(map(int, cell)) for cell in np.argwhere(scene == color)]
        assert coords, f"ring color {color} must have a visible fragment"
        matches = _matching_offsets(coords, grid.shape)
        assert len(matches) == 1, f"ring color {color} must match exactly one 5x5 ring placement"
        positions[color] = matches[0]
    return positions


def _occlusion_edges(
    grid: np.ndarray,
    ring_colors: list[int],
    positions: dict[int, tuple[int, int]],
    bg: int,
) -> dict[int, set[int]]:
    ring_set = set(ring_colors)
    edges = {color: set() for color in ring_colors}
    for color, offset in positions.items():
        for r, c in _ring_cells(offset):
            seen = int(grid[r, c])
            if seen != bg and seen != color and seen in ring_set:
                edges[color].add(seen)
    return edges


def _bottom_to_top_order(ring_colors: list[int], edges: dict[int, set[int]]) -> list[int]:
    indegree = {color: 0 for color in ring_colors}
    for src in ring_colors:
        for dst in edges[src]:
            indegree[dst] += 1

    remaining = set(ring_colors)
    order: list[int] = []
    while remaining:
        pick = next((color for color in ring_colors if color in remaining and indegree[color] == 0), None)
        assert pick is not None, "occlusion graph must be acyclic"
        order.append(pick)
        remaining.remove(pick)
        for dst in edges[pick]:
            indegree[dst] -= 1
    return order


def _render_recolored(
    shape: tuple[int, int],
    bg: int,
    positions: dict[int, tuple[int, int]],
    layer_order: list[int],
    mapping: dict[int, int],
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for color in layer_order:
        paint = mapping[color]
        for r, c in _ring_cells(positions[color]):
            out[r, c] = paint
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct overlapping 5x5 hollow square rings, infer their stacking order,
    recolor them via the bottom legend, and redraw the complete rings.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background(g)
    mapping, ring_colors = _legend_mapping(g, bg)
    positions = _recover_positions(g, ring_colors)
    edges = _occlusion_edges(g, ring_colors, positions, bg)
    layer_order = _bottom_to_top_order(ring_colors, edges)
    return _render_recolored(g.shape, bg, positions, layer_order, mapping)


# --------------- generator ---------------

GEN_BG = 0
GRID_H_RANGE = (18, 24)
GRID_W_RANGE = (18, 24)
RING_COUNT_OPTIONS = (3, 4)
MIN_VISIBLE_PIXELS = 4
MAX_PAIR_OVERLAP = 6
MAX_BUILD_ATTEMPTS = 120
MAX_GEN_ATTEMPTS = 400

_SHIFT_COUNTS = []
for sr in range(-(RING_SIZE - 1), RING_SIZE):
    for sc in range(-(RING_SIZE - 1), RING_SIZE):
        shifted = {(r + sr, c + sc) for r, c in RING}
        overlap = len(RING_SET & shifted)
        if 2 <= overlap <= MAX_PAIR_OVERLAP:
            _SHIFT_COUNTS.append((sr, sc, overlap))


def _valid_offset(offset: tuple[int, int], h: int, w: int) -> bool:
    r0, c0 = offset
    return 1 <= r0 <= h - 9 and 1 <= c0 <= w - 6


def _pair_overlap(a: tuple[int, int], b: tuple[int, int]) -> int:
    return len(_ring_cell_set(a) & _ring_cell_set(b))


def _overlap_graph(offsets: list[tuple[int, int]]) -> list[set[int]]:
    graph = [set() for _ in offsets]
    for i in range(len(offsets)):
        for j in range(i + 1, len(offsets)):
            if _pair_overlap(offsets[i], offsets[j]) > 0:
                graph[i].add(j)
                graph[j].add(i)
    return graph


def _build_dense_offsets(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    offsets = [(rng.randint(1, h - 9), rng.randint(1, w - 6))]
    for _ in range(1, count):
        candidate = None
        for _ in range(MAX_BUILD_ATTEMPTS):
            base = offsets[rng.randrange(len(offsets))]
            dr, dc, _ov = rng.choice(_SHIFT_COUNTS)
            trial = (base[0] + dr, base[1] + dc)
            if trial in offsets or not _valid_offset(trial, h, w):
                continue
            if any(_pair_overlap(trial, other) > MAX_PAIR_OVERLAP for other in offsets):
                continue
            candidate = trial
            break
        if candidate is None:
            return None
        offsets.append(candidate)

    graph = _overlap_graph(offsets)
    if any(not neighbors for neighbors in graph):
        return None
    return offsets


def _build_isolated_offsets(h: int, w: int, count: int, rng: random.Random) -> list[tuple[int, int]] | None:
    core = _build_dense_offsets(h, w, count - 1, rng)
    if core is None:
        return None

    isolated = None
    for _ in range(MAX_BUILD_ATTEMPTS):
        trial = (rng.randint(1, h - 9), rng.randint(1, w - 6))
        if trial in core or not _valid_offset(trial, h, w):
            continue
        if all(_pair_overlap(trial, other) == 0 for other in core):
            isolated = trial
            break
    if isolated is None:
        return None

    offsets = core + [isolated]
    rng.shuffle(offsets)
    graph = _overlap_graph(offsets)
    edge_count = sum(len(neighbors) for neighbors in graph) // 2
    if edge_count == 0 or not any(not neighbors for neighbors in graph):
        return None
    return offsets


def _render_input(
    h: int,
    w: int,
    positions: dict[int, tuple[int, int]],
    layer_order: list[int],
    legend_order: list[int],
    mapping: dict[int, int],
    legend_col: int,
) -> np.ndarray:
    grid = np.full((h, w), GEN_BG, dtype=int)
    for color in layer_order:
        for r, c in _ring_cells(positions[color]):
            grid[r, c] = color
    for i, color in enumerate(legend_order):
        grid[h - 2, legend_col + i] = color
        grid[h - 1, legend_col + i] = mapping[color]
    return grid


def _visible_coords(grid: np.ndarray, color: int) -> list[tuple[int, int]]:
    return [tuple(map(int, cell)) for cell in np.argwhere(grid[:-2] == color)]


def _input_is_valid(
    grid: np.ndarray,
    positions: dict[int, tuple[int, int]],
    ring_colors: list[int],
    expected: np.ndarray,
    branch: str,
) -> bool:
    offsets = [positions[color] for color in ring_colors]
    graph = _overlap_graph(offsets)
    edge_count = sum(len(neighbors) for neighbors in graph) // 2
    if edge_count == 0:
        return False
    if branch == "dense" and any(not neighbors for neighbors in graph):
        return False
    if branch == "isolated" and not any(not neighbors for neighbors in graph):
        return False

    for color in ring_colors:
        coords = _visible_coords(grid, color)
        if len(coords) < MIN_VISIBLE_PIXELS:
            return False
        matches = _matching_offsets(coords, grid.shape)
        if matches != [positions[color]]:
            return False

    return np.array_equal(main(grid), expected)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    count = rng.choice(RING_COUNT_OPTIONS)
    branch = rng.choice(("dense", "isolated"))

    color_pool = list(range(1, 10))
    rng.shuffle(color_pool)
    ring_colors = color_pool[:count]
    output_colors = color_pool[count : 2 * count]
    if len(output_colors) != count:
        return None

    if branch == "dense":
        offsets = _build_dense_offsets(h, w, count, rng)
    else:
        offsets = _build_isolated_offsets(h, w, count, rng)
    if offsets is None:
        return None

    positions = dict(zip(ring_colors, offsets))
    layer_order = ring_colors[:]
    rng.shuffle(layer_order)

    legend_order = ring_colors[:]
    rng.shuffle(legend_order)
    shuffled_outputs = output_colors[:]
    rng.shuffle(shuffled_outputs)
    mapping = dict(zip(ring_colors, shuffled_outputs))
    legend_col = rng.randint(0, w - count)

    grid = _render_input(h, w, positions, layer_order, legend_order, mapping, legend_col)
    expected = _render_recolored(grid.shape, GEN_BG, positions, layer_order, mapping)
    if not _input_is_valid(grid, positions, ring_colors, expected, branch):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
