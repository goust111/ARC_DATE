import random
from collections import deque

import numpy as np


BG = 0
START = 2
PATH = 5

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
DIR_TO_IDX = {d: i for i, d in enumerate(DIRS)}
SIDE_TO_DIR = {
    "top": 0,
    "right": 1,
    "bottom": 2,
    "left": 3,
}
SIDE_NAMES = tuple(SIDE_TO_DIR)

TERMINAL_COLORS = [1, 3, 4, 6, 7, 8, 9]
TERMINAL_SHAPES = [(2, 2), (1, 2), (2, 1), (1, 1)]

MIN_SIZE = 19
MAX_SIZE = 31
MIN_NODES = 6
MAX_NODES = 10
GEN_RETRIES = 400


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if _in_bounds(h, w, nr, nc):
            out.append((nr, nc))
    return out


def _neighbors8(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr = r + dr
            nc = c + dc
            if _in_bounds(h, w, nr, nc):
                out.append((nr, nc))
    return out


def _connected_component(g: np.ndarray, start: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = g.shape
    color = int(g[start])
    todo = deque([start])
    seen = {start}
    cells: list[tuple[int, int]] = []
    while todo:
        r, c = todo.popleft()
        cells.append((r, c))
        for nr, nc in _neighbors4(r, c, h, w):
            if (nr, nc) in seen:
                continue
            if int(g[nr, nc]) != color:
                continue
            seen.add((nr, nc))
            todo.append((nr, nc))
    return cells


def _component_is_rectangle(cells: list[tuple[int, int]]) -> bool:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return len(cells) == (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1)


def _path_neighbors(g: np.ndarray, cell: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = g.shape
    r, c = cell
    return [(nr, nc) for nr, nc in _neighbors4(r, c, h, w) if int(g[nr, nc]) == PATH]


def _find_start(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = g.shape
    starts = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(g[r, c]) == START and (r in (0, h - 1) or c in (0, w - 1))
    ]
    assert len(starts) == 1, "expected exactly one border start marker"
    start = starts[0]
    entry_neighbors = _path_neighbors(g, start)
    assert len(entry_neighbors) == 1, "start marker must touch exactly one path cell"
    return start, entry_neighbors[0]


def _marker_candidates(offset: tuple[int, int]) -> list[int]:
    dr, dc = offset
    candidates: list[int] = []
    if dr < 0:
        candidates.append(0)
    if dc > 0:
        candidates.append(1)
    if dr > 0:
        candidates.append(2)
    if dc < 0:
        candidates.append(3)
    assert candidates, "marker offset must encode at least one direction"
    return candidates


def _junction_preference(
    g: np.ndarray,
    prev: tuple[int, int],
    cur: tuple[int, int],
) -> int:
    h, w = g.shape
    r, c = cur
    marker_cells = [
        (nr, nc)
        for nr, nc in _neighbors8(r, c, h, w)
        if int(g[nr, nc]) not in (BG, PATH, START)
    ]
    assert len(marker_cells) == 1, "each intersection must have one nearby marker"
    mr, mc = marker_cells[0]
    candidates = _marker_candidates((mr - r, mc - c))
    back_dir = DIR_TO_IDX[(prev[0] - r, prev[1] - c)]

    usable = [d for d in candidates if d != back_dir]
    if not usable:
        usable = list(candidates)

    passable = []
    for d in usable:
        dr, dc = DIRS[d]
        nr = r + dr
        nc = c + dc
        if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) == PATH:
            passable.append(d)
    if passable:
        return passable[0]
    return usable[0]


def _terminal_component(g: np.ndarray, endpoint: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = g.shape
    candidates = [
        (nr, nc)
        for nr, nc in _neighbors4(endpoint[0], endpoint[1], h, w)
        if int(g[nr, nc]) not in (BG, PATH, START)
    ]
    assert candidates, "path endpoint must touch a terminal region"
    comp = _connected_component(g, candidates[0])
    assert _component_is_rectangle(comp), "terminal must be a filled rectangle"
    colors = {int(g[r, c]) for r, c in comp}
    assert len(colors) == 1, "terminal must be monochrome"
    return comp


def _trace_signal(
    g: np.ndarray,
) -> tuple[list[tuple[int, int]], list[dict[str, object]]]:
    start, cur = _find_start(g)
    prev = start
    seen_states: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    decisions: list[dict[str, object]] = []

    while True:
        state = (prev, cur)
        assert state not in seen_states, "signal entered a loop"
        seen_states.add(state)

        neighbors = _path_neighbors(g, cur)
        forward = [n for n in neighbors if n != prev]
        if not forward:
            return _terminal_component(g, cur), decisions

        if len(forward) == 1:
            nxt = forward[0]
        else:
            preferred = _junction_preference(g, prev, cur)
            forward_map = {
                DIR_TO_IDX[(nr - cur[0], nc - cur[1])]: (nr, nc)
                for nr, nc in forward
            }
            chosen_dir = None
            nxt = None
            for step in range(4):
                d = (preferred + step) % 4
                if d in forward_map:
                    chosen_dir = d
                    nxt = forward_map[d]
                    break
            assert chosen_dir is not None and nxt is not None
            decisions.append(
                {
                    "junction": cur,
                    "preferred": preferred,
                    "chosen": chosen_dir,
                    "used_fallback": chosen_dir != preferred,
                }
            )

        prev, cur = cur, nxt


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate a signal from the red border marker through the gray path network,
    resolve each T-junction using its nearby direction marker, and recolor the
    reached terminal rectangle to red.
    """
    g = np.asarray(grid, dtype=int)
    terminal, _decisions = _trace_signal(g)
    out = g.copy()
    for r, c in terminal:
        out[r, c] = START
    return out


def _index_neighbors(node: tuple[int, int], nrows: int, ncols: int) -> list[tuple[int, int]]:
    r, c = node
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < nrows and 0 <= nc < ncols:
            out.append((nr, nc))
    return out


def _actual_pos(
    node: tuple[int, int],
    row_coords: list[int],
    col_coords: list[int],
) -> tuple[int, int]:
    return row_coords[node[0]], col_coords[node[1]]


def _dir_between(a: tuple[int, int], b: tuple[int, int]) -> int:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    step = (0 if dr == 0 else dr // abs(dr), 0 if dc == 0 else dc // abs(dc))
    return DIR_TO_IDX[step]


def _paint_segment(g: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    if a[0] == b[0]:
        r = a[0]
        c0, c1 = sorted((a[1], b[1]))
        g[r, c0 : c1 + 1] = PATH
    else:
        c = a[1]
        r0, r1 = sorted((a[0], b[0]))
        g[r0 : r1 + 1, c] = PATH


def _build_tree(
    rng: random.Random,
    nrows: int,
    ncols: int,
    root: tuple[int, int],
    target_nodes: int,
) -> tuple[dict[tuple[int, int], list[tuple[int, int]]], dict[tuple[int, int], tuple[int, int]]]:
    used = {root}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    degree = {root: 0}
    adjacency: dict[tuple[int, int], list[tuple[int, int]]] = {root: []}

    while len(used) < target_nodes:
        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for node in used:
            limit = 1 if node == root else 3
            if degree[node] >= limit:
                continue
            for nb in _index_neighbors(node, nrows, ncols):
                if nb in used:
                    continue
                candidates.append((node, nb))
        if not candidates:
            raise RuntimeError("tree growth stalled")

        parent_node, new_node = rng.choice(candidates)
        used.add(new_node)
        parent[new_node] = parent_node
        degree[parent_node] += 1
        degree[new_node] = 1
        adjacency.setdefault(parent_node, []).append(new_node)
        adjacency[new_node] = [parent_node]

    return adjacency, parent


def _find_first_junction(
    root: tuple[int, int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
) -> tuple[int, int] | None:
    prev = root
    cur = adjacency[root][0]
    while True:
        deg = len(adjacency[cur])
        if deg >= 3:
            return cur
        if deg == 1:
            return None
        nxt = adjacency[cur][0] if adjacency[cur][1] == prev else adjacency[cur][1]
        prev, cur = cur, nxt


def _terminal_bboxes(
    leaf_pos: tuple[int, int],
    outward_dir: int,
    h: int,
    w: int,
) -> list[tuple[int, int, int, int]]:
    r, c = leaf_pos
    bboxes: list[tuple[int, int, int, int]] = []
    if outward_dir == 0:
        r0 = r - h
        col_starts = [c] if w == 1 else [c - 1, c]
        for c0 in col_starts:
            bboxes.append((r0, c0, h, w))
    elif outward_dir == 1:
        c0 = c + 1
        row_starts = [r] if h == 1 else [r - 1, r]
        for r0 in row_starts:
            bboxes.append((r0, c0, h, w))
    elif outward_dir == 2:
        r0 = r + 1
        col_starts = [c] if w == 1 else [c - 1, c]
        for c0 in col_starts:
            bboxes.append((r0, c0, h, w))
    else:
        c0 = c - w
        row_starts = [r] if h == 1 else [r - 1, r]
        for r0 in row_starts:
            bboxes.append((r0, c0, h, w))
    return bboxes


def _bbox_fits(g: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    h, w = g.shape
    r0, c0, bh, bw = bbox
    if r0 < 0 or c0 < 0 or r0 + bh > h or c0 + bw > w:
        return False
    return np.all(g[r0 : r0 + bh, c0 : c0 + bw] == BG)


def _paint_bbox(g: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, h, w = bbox
    g[r0 : r0 + h, c0 : c0 + w] = color


def _preferred_marker_offset(
    back_dir: int,
    blocked_dir: int,
    child_dir: int,
) -> tuple[int, int]:
    companion = blocked_dir if child_dir == (back_dir + 2) % 4 else back_dir
    return (
        DIRS[child_dir][0] + DIRS[companion][0],
        DIRS[child_dir][1] + DIRS[companion][1],
    )


def _place_markers(
    rng: random.Random,
    g: np.ndarray,
    row_coords: list[int],
    col_coords: list[int],
    adjacency: dict[tuple[int, int], list[tuple[int, int]]],
    parent: dict[tuple[int, int], tuple[int, int]],
    marker_colors: list[int],
) -> None:
    root = next(node for node, neighbors in adjacency.items() if len(neighbors) == 1 and node not in parent)
    first_junction = _find_first_junction(root, adjacency)

    for node, neighbors in adjacency.items():
        if len(neighbors) != 3:
            continue

        back_dir = _dir_between(_actual_pos(node, row_coords, col_coords), _actual_pos(parent[node], row_coords, col_coords))
        child_dirs = [
            _dir_between(_actual_pos(node, row_coords, col_coords), _actual_pos(nb, row_coords, col_coords))
            for nb in neighbors
            if nb != parent[node]
        ]
        used_dirs = {back_dir, *child_dirs}
        blocked_dir = next(d for d in range(4) if d not in used_dirs)
        fallback_dir = next((blocked_dir + step) % 4 for step in range(4) if (blocked_dir + step) % 4 in child_dirs)

        if node == first_junction and rng.random() < 0.5:
            offset = DIRS[blocked_dir]
        else:
            chosen_child = child_dirs[0] if rng.random() < 0.5 else child_dirs[1]
            if node == first_junction and rng.random() < 0.5:
                chosen_child = fallback_dir
                if rng.random() < 0.5:
                    offset = DIRS[blocked_dir]
                else:
                    offset = _preferred_marker_offset(back_dir, blocked_dir, chosen_child)
            else:
                offset = _preferred_marker_offset(back_dir, blocked_dir, chosen_child)

        r, c = _actual_pos(node, row_coords, col_coords)
        mr = r + offset[0]
        mc = c + offset[1]
        assert _in_bounds(*g.shape, mr, mc)
        assert int(g[mr, mc]) == BG
        g[mr, mc] = rng.choice(marker_colors)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(MIN_SIZE, MAX_SIZE)
    w = rng.randint(MIN_SIZE, MAX_SIZE)
    row_coords = list(range(1, h - 1, 4))
    col_coords = list(range(1, w - 1, 4))
    if len(row_coords) < 4 or len(col_coords) < 4:
        return None

    nrows = len(row_coords)
    ncols = len(col_coords)
    side = rng.choice(SIDE_NAMES)
    if side in ("top", "bottom"):
        if ncols < 3:
            return None
        root = (0 if side == "top" else nrows - 1, rng.randrange(1, ncols - 1))
    else:
        if nrows < 3:
            return None
        root = (rng.randrange(1, nrows - 1), 0 if side == "left" else ncols - 1)

    max_nodes = min(MAX_NODES, nrows * ncols)
    if max_nodes < MIN_NODES:
        return None
    target_nodes = rng.randint(MIN_NODES, max_nodes)

    try:
        adjacency, parent = _build_tree(rng, nrows, ncols, root, target_nodes)
    except RuntimeError:
        return None

    degrees = {node: len(neighbors) for node, neighbors in adjacency.items()}
    if degrees[root] != 1:
        return None
    if not any(deg == 3 for deg in degrees.values()):
        return None

    leaves = [node for node, deg in degrees.items() if deg == 1]
    terminal_leaves = [node for node in leaves if node != root]
    if not (2 <= len(terminal_leaves) <= 5):
        return None

    first_junction = _find_first_junction(root, adjacency)
    if first_junction is None:
        return None

    g = np.full((h, w), BG, dtype=int)
    for node, neighbors in adjacency.items():
        a = _actual_pos(node, row_coords, col_coords)
        for nb in neighbors:
            if node < nb:
                b = _actual_pos(nb, row_coords, col_coords)
                _paint_segment(g, a, b)

    root_r, root_c = _actual_pos(root, row_coords, col_coords)
    side_dir = SIDE_TO_DIR[side]
    start_r = root_r + DIRS[side_dir][0]
    start_c = root_c + DIRS[side_dir][1]
    if not _in_bounds(h, w, start_r, start_c):
        return None
    if start_r not in (0, h - 1) and start_c not in (0, w - 1):
        return None
    g[start_r, start_c] = START

    terminal_colors = rng.sample(TERMINAL_COLORS, len(terminal_leaves))
    terminal_info: list[tuple[tuple[int, int, int, int], int]] = []
    for leaf, color in zip(terminal_leaves, terminal_colors):
        neighbor = adjacency[leaf][0]
        leaf_pos = _actual_pos(leaf, row_coords, col_coords)
        outward_dir = (_dir_between(leaf_pos, _actual_pos(neighbor, row_coords, col_coords)) + 2) % 4

        shapes = list(TERMINAL_SHAPES)
        rng.shuffle(shapes)
        bbox = None
        for bh, bw in shapes:
            candidates = _terminal_bboxes(leaf_pos, outward_dir, bh, bw)
            rng.shuffle(candidates)
            for candidate in candidates:
                if _bbox_fits(g, candidate):
                    bbox = candidate
                    break
            if bbox is not None:
                break
        if bbox is None:
            return None

        _paint_bbox(g, bbox, color)
        terminal_info.append((bbox, color))

    marker_colors = [c for c in TERMINAL_COLORS if c not in terminal_colors]
    if not marker_colors:
        return None

    try:
        _place_markers(rng, g, row_coords, col_coords, adjacency, parent, marker_colors)
    except AssertionError:
        return None

    try:
        terminal_cells, decisions = _trace_signal(g)
    except AssertionError:
        return None
    if not decisions:
        return None

    terminal_set = set(terminal_cells)
    matched = 0
    for bbox, _color in terminal_info:
        r0, c0, bh, bw = bbox
        bbox_cells = {(r, c) for r in range(r0, r0 + bh) for c in range(c0, c0 + bw)}
        if bbox_cells == terminal_set:
            matched += 1
    if matched != 1:
        return None

    out = main(g)
    red_terminals = 0
    for bbox, color in terminal_info:
        r0, c0, bh, bw = bbox
        patch = out[r0 : r0 + bh, c0 : c0 + bw]
        if np.all(patch == START):
            red_terminals += 1
        elif not np.all(patch == color):
            return None
    if red_terminals != 1:
        return None

    return g


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-path puzzle")
