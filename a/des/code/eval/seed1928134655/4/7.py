from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
AZURE = 8
MAROON = 9

INPUT_COLORS = {RED, BLUE, YELLOW}
NETWORK_COLORS = INPUT_COLORS | {GREY, AZURE}
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 400

MIX_TO_COLOR = {
    frozenset({RED}): RED,
    frozenset({BLUE}): BLUE,
    frozenset({YELLOW}): YELLOW,
    frozenset({RED, BLUE}): MAGENTA,
    frozenset({RED, YELLOW}): ORANGE,
    frozenset({BLUE, YELLOW}): GREEN,
    frozenset({RED, BLUE, YELLOW}): MAROON,
}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in DIRS4:
        nr = r + dr
        nc = c + dc
        if _in_bounds(shape, nr, nc):
            out.append((nr, nc))
    return out


def _is_border(shape: tuple[int, int], r: int, c: int) -> bool:
    return r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1


def _network_neighbors(grid: np.ndarray, r: int, c: int) -> list[tuple[int, int]]:
    return [
        (nr, nc)
        for nr, nc in _neighbors4(grid.shape, r, c)
        if int(grid[nr, nc]) in NETWORK_COLORS
    ]


def _build_graph(grid: np.ndarray) -> tuple[list[dict], dict[int, list[int]]]:
    g = np.asarray(grid, dtype=int)
    colors = set(map(int, np.unique(g)))
    assert colors <= (NETWORK_COLORS | {BLACK}), "unexpected colors in input"

    network_cells = [tuple(map(int, rc)) for rc in np.argwhere(np.isin(g, list(NETWORK_COLORS)))]
    assert network_cells, "expected a pipe network"

    degrees: dict[tuple[int, int], int] = {}
    for r, c in network_cells:
        degrees[(r, c)] = len(_network_neighbors(g, r, c))

    nodes: list[dict] = []
    coord_to_node: dict[tuple[int, int], int] = {}
    for r, c in network_cells:
        color = int(g[r, c])
        degree = degrees[(r, c)]
        is_node = color in INPUT_COLORS or color == AZURE or degree != 2
        if not is_node:
            continue

        if color in INPUT_COLORS:
            assert _is_border(g.shape, r, c), "inputs must lie on the border"
            assert degree == 1, "input endpoints must be degree-1"
            kind = "source"
        elif color == AZURE:
            assert degree in (3, 4), "merge nodes must have 2-3 upstream arms and one output arm"
            kind = "merge"
        else:
            assert color == GREY
            if _is_border(g.shape, r, c):
                assert degree == 1, "border grey nodes must be outputs"
                kind = "sink"
            else:
                assert degree >= 3, "internal grey nodes with special role must be splitters"
                kind = "split"

        coord_to_node[(r, c)] = len(nodes)
        nodes.append({"coord": (r, c), "kind": kind, "color": color})

    assert any(node["kind"] == "source" for node in nodes), "expected at least one source"
    assert any(node["kind"] == "sink" for node in nodes), "expected at least one sink"

    adjacency_sets: dict[int, set[int]] = {idx: set() for idx in range(len(nodes))}
    for node_id, node in enumerate(nodes):
        r, c = node["coord"]
        for nr, nc in _network_neighbors(g, r, c):
            prev = (r, c)
            cur = (nr, nc)
            while cur not in coord_to_node:
                next_steps = [nxt for nxt in _network_neighbors(g, cur[0], cur[1]) if nxt != prev]
                assert len(next_steps) == 1, "corridors between graph nodes must be unambiguous"
                prev, cur = cur, next_steps[0]
            other_id = coord_to_node[cur]
            if other_id != node_id:
                adjacency_sets[node_id].add(other_id)

    adjacency = {idx: sorted(neighbors) for idx, neighbors in adjacency_sets.items()}
    for idx, node in enumerate(nodes):
        degree = len(adjacency[idx])
        kind = node["kind"]
        if kind == "source":
            assert degree == 1
        elif kind == "sink":
            assert degree == 1
        elif kind == "merge":
            assert degree in (3, 4)
        else:
            assert degree >= 3
    return nodes, adjacency


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate input colors through the grey pipe network, split unchanged colors
    at grey junctions, and mix colors at azure merge nodes before continuing.
    Only output endpoints are recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "grid must be 2D"

    nodes, adjacency = _build_graph(g)
    out = g.copy()

    source_ids = [idx for idx, node in enumerate(nodes) if node["kind"] == "source"]
    sink_ids = [idx for idx, node in enumerate(nodes) if node["kind"] == "sink"]

    queue: deque[tuple[int, int, frozenset[int]]] = deque()
    edge_signals: dict[tuple[int, int], frozenset[int]] = {}
    merge_arrivals: dict[int, dict[int, frozenset[int]]] = defaultdict(dict)
    merge_emitted: dict[int, frozenset[int]] = {}
    sink_signals: dict[int, frozenset[int]] = {}

    def send(src: int, dst: int, signal: frozenset[int]) -> None:
        prev = edge_signals.get((src, dst), frozenset())
        combined = prev | signal
        if combined == prev:
            return
        edge_signals[(src, dst)] = combined
        queue.append((src, dst, combined))

    for source_id in source_ids:
        base_color = int(nodes[source_id]["color"])
        assert base_color in INPUT_COLORS
        for neighbor in adjacency[source_id]:
            send(source_id, neighbor, frozenset({base_color}))

    while queue:
        src, dst, signal = queue.popleft()
        node = nodes[dst]
        kind = node["kind"]

        if kind == "sink":
            sink_signals[dst] = sink_signals.get(dst, frozenset()) | signal
            continue

        if kind == "merge":
            arrivals = merge_arrivals[dst]
            arrivals[src] = arrivals.get(src, frozenset()) | signal
            if len(arrivals) == len(adjacency[dst]) - 1:
                downstream = [neighbor for neighbor in adjacency[dst] if neighbor not in arrivals]
                assert len(downstream) == 1, "merge node must have exactly one downstream arm"
                combined = frozenset().union(*arrivals.values())
                emitted = merge_emitted.get(dst, frozenset())
                updated = emitted | combined
                if updated != emitted:
                    merge_emitted[dst] = updated
                    send(dst, downstream[0], updated)
            continue

        for neighbor in adjacency[dst]:
            if neighbor != src:
                send(dst, neighbor, signal)

    for sink_id in sink_ids:
        assert sink_id in sink_signals, "every output must receive a signal"
        signal = sink_signals[sink_id]
        assert signal in MIX_TO_COLOR, "only the three base colors may participate in mixing"
        r, c = nodes[sink_id]["coord"]
        out[r, c] = MIX_TO_COLOR[signal]

    return out


def _paint_cell(grid: np.ndarray, r: int, c: int, color: int) -> None:
    current = int(grid[r, c])
    assert current in (BLACK, color), "unexpected path overlap"
    grid[r, c] = color


def _paint_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int) -> None:
    start = min(c0, c1)
    end = max(c0, c1)
    for col in range(start, end + 1):
        _paint_cell(grid, row, col, color)


def _paint_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int) -> None:
    start = min(r0, r1)
    end = max(r0, r1)
    for row in range(start, end + 1):
        _paint_cell(grid, row, col, color)


def _transform(grid: np.ndarray, rotations: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotations)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _canonical_geometry(
    height: int,
    width: int,
    use_left: bool,
    use_upstream_split: bool,
    use_downstream_split: bool,
) -> tuple[int, int, int | None, int | None]:
    merge_row = random.randint(3, height - 4)
    merge_col = random.randint(4, width - 5)

    left_split_col = None
    if use_left and use_upstream_split:
        left_split_col = random.randint(2, merge_col - 2)

    downstream_split_col = None
    if use_downstream_split:
        downstream_split_col = random.randint(merge_col + 2, width - 2)

    return merge_row, merge_col, left_split_col, downstream_split_col


def _pair_setup() -> tuple[tuple[str, ...], dict[str, int]]:
    active_sides = random.choice(
        [("left", "top"), ("left", "bottom"), ("top", "bottom")]
    )
    pair = random.choice([(RED, BLUE), (RED, YELLOW), (BLUE, YELLOW)])
    order = list(pair)
    random.shuffle(order)
    colors = {side: color for side, color in zip(active_sides, order)}
    return active_sides, colors


def _triple_setup() -> tuple[tuple[str, ...], dict[str, int]]:
    colors = [RED, BLUE, YELLOW]
    random.shuffle(colors)
    active_sides = ("left", "top", "bottom")
    return active_sides, dict(zip(active_sides, colors))


def _build_canonical_case() -> tuple[np.ndarray, np.ndarray] | None:
    height = random.randint(13, 21)
    width = random.randint(13, 21)

    if random.random() < 0.5:
        active_sides, input_colors = _pair_setup()
    else:
        active_sides, input_colors = _triple_setup()

    use_left = "left" in active_sides
    use_upstream_split = use_left and random.random() < 0.5
    use_downstream_split = random.random() < 0.7

    merge_row, merge_col, left_split_col, downstream_split_col = _canonical_geometry(
        height, width, use_left, use_upstream_split, use_downstream_split
    )

    if use_upstream_split:
        assert left_split_col is not None
        upstream_branch_side = random.choice(["top", "bottom"])
    else:
        upstream_branch_side = None

    if use_downstream_split:
        assert downstream_split_col is not None
        branch_count = random.choice([1, 2])
        downstream_branch_sides = random.sample(["top", "bottom"], branch_count)
    else:
        downstream_branch_sides = []

    mixed_signal = frozenset(input_colors[side] for side in active_sides)
    mixed_color = MIX_TO_COLOR[mixed_signal]

    inp = np.full((height, width), BLACK, dtype=int)
    out = np.full((height, width), BLACK, dtype=int)

    _paint_cell(inp, merge_row, merge_col, AZURE)
    _paint_cell(out, merge_row, merge_col, AZURE)

    if use_left:
        left_color = int(input_colors["left"])
        _paint_cell(inp, merge_row, 0, left_color)
        _paint_cell(out, merge_row, 0, left_color)
        if use_upstream_split:
            assert left_split_col is not None and upstream_branch_side is not None
            _paint_h(inp, merge_row, 1, left_split_col, GREY)
            _paint_h(out, merge_row, 1, left_split_col, GREY)
            _paint_h(inp, merge_row, left_split_col, merge_col - 1, GREY)
            _paint_h(out, merge_row, left_split_col, merge_col - 1, GREY)
            if upstream_branch_side == "top":
                _paint_v(inp, left_split_col, 0, merge_row, GREY)
                _paint_v(out, left_split_col, 0, merge_row, GREY)
                out[0, left_split_col] = left_color
            else:
                _paint_v(inp, left_split_col, merge_row, height - 1, GREY)
                _paint_v(out, left_split_col, merge_row, height - 1, GREY)
                out[height - 1, left_split_col] = left_color
            inp[merge_row, 0] = left_color
            out[merge_row, 0] = left_color
        else:
            _paint_h(inp, merge_row, 1, merge_col - 1, GREY)
            _paint_h(out, merge_row, 1, merge_col - 1, GREY)

    if "top" in active_sides:
        top_color = int(input_colors["top"])
        _paint_cell(inp, 0, merge_col, top_color)
        _paint_cell(out, 0, merge_col, top_color)
        _paint_v(inp, merge_col, 1, merge_row - 1, GREY)
        _paint_v(out, merge_col, 1, merge_row - 1, GREY)

    if "bottom" in active_sides:
        bottom_color = int(input_colors["bottom"])
        _paint_cell(inp, height - 1, merge_col, bottom_color)
        _paint_cell(out, height - 1, merge_col, bottom_color)
        _paint_v(inp, merge_col, merge_row + 1, height - 2, GREY)
        _paint_v(out, merge_col, merge_row + 1, height - 2, GREY)

    if use_downstream_split:
        assert downstream_split_col is not None
        _paint_h(inp, merge_row, merge_col + 1, downstream_split_col, GREY)
        _paint_h(out, merge_row, merge_col + 1, downstream_split_col, GREY)
        _paint_h(inp, merge_row, downstream_split_col, width - 1, GREY)
        _paint_h(out, merge_row, downstream_split_col, width - 1, GREY)
        out[merge_row, width - 1] = mixed_color
        for side in downstream_branch_sides:
            if side == "top":
                _paint_v(inp, downstream_split_col, 0, merge_row, GREY)
                _paint_v(out, downstream_split_col, 0, merge_row, GREY)
                out[0, downstream_split_col] = mixed_color
            else:
                _paint_v(inp, downstream_split_col, merge_row, height - 1, GREY)
                _paint_v(out, downstream_split_col, merge_row, height - 1, GREY)
                out[height - 1, downstream_split_col] = mixed_color
    else:
        _paint_h(inp, merge_row, merge_col + 1, width - 1, GREY)
        _paint_h(out, merge_row, merge_col + 1, width - 1, GREY)
        out[merge_row, width - 1] = mixed_color

    if not np.any(inp == BLACK):
        return None
    if np.array_equal(inp, out):
        return None

    rotations = random.randint(0, 3)
    flip_lr = bool(random.getrandbits(1))
    inp = _transform(inp, rotations, flip_lr)
    out = _transform(out, rotations, flip_lr)
    return inp, out


def _gen_attempt() -> np.ndarray | None:
    built = _build_canonical_case()
    if built is None:
        return None
    inp, expected = built
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
