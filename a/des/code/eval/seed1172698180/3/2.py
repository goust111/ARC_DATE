from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREEN = 3
GREY = 5
MAGENTA = 6
SOURCE_PALETTE = (1, 2, 4, 7, 8, 9)

DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIR_BY_NAME = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        nr = int(r + dr)
        nc = int(c + dc)
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _collect_sources(grid: np.ndarray) -> list[tuple[int, tuple[tuple[int, int], ...]]]:
    h, w = grid.shape
    grey_mask = grid == GREY
    source_mask = (grid != BLACK) & (grid != GREY) & (grid != MAGENTA)
    sources: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for r, c in np.argwhere(source_mask):
        starts = {
            (int(nr), int(nc))
            for nr, nc in _neighbors4(int(r), int(c), h, w)
            if bool(grey_mask[nr, nc])
        }
        if starts:
            sources.append((int(grid[r, c]), tuple(sorted(starts))))
    return sources


def _bfs_dist(grey_mask: np.ndarray, starts: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = grey_mask.shape
    inf = h * w + 1
    dist = np.full((h, w), inf, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in starts:
        if bool(grey_mask[r, c]) and int(dist[r, c]) > 0:
            dist[r, c] = 0
            q.append((int(r), int(c)))
    while q:
        r, c = q.popleft()
        nd = int(dist[r, c] + 1)
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(grey_mask[nr, nc]) and int(dist[nr, nc]) > nd:
                dist[nr, nc] = nd
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate endpoint colors along the grey path network, stopping at magenta valves.

    Grey cells are colored by the set of source colors that reach them at minimum
    graph distance through grey cells. One color wins if unique; distinct tied
    colors turn the cell green.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    grey_mask = g == GREY
    if not bool(grey_mask.any()):
        return out

    sources = _collect_sources(g)
    if not sources:
        return out

    dist_maps = [(color, _bfs_dist(grey_mask, starts)) for color, starts in sources]
    inf = g.shape[0] * g.shape[1] + 1

    for r, c in np.argwhere(grey_mask):
        best = inf
        colors: set[int] = set()
        rr = int(r)
        cc = int(c)
        for color, dist in dist_maps:
            d = int(dist[rr, cc])
            if d < best:
                best = d
                colors = {int(color)}
            elif d == best and d < inf:
                colors.add(int(color))
        if best == inf:
            continue
        out[rr, cc] = GREEN if len(colors) >= 2 else next(iter(colors))

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    h, w = mask.shape
    start = tuple(int(v) for v in pts[0])
    seen = {start}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _rotate_coord(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = coord
    for _ in range(int(turns) % 4):
        r, c = -c, r
    return int(r), int(c)


def _step(coord: tuple[int, int], dir_name: str) -> tuple[int, int]:
    dr, dc = DIR_BY_NAME[dir_name]
    return int(coord[0] + dr), int(coord[1] + dc)


def _add_chain(
    start: tuple[int, int],
    dir_name: str,
    length: int,
    owner: str,
    nodes_owner: dict[tuple[int, int], str],
    grey_edges: list[tuple[tuple[int, int], tuple[int, int]]],
) -> tuple[int, int]:
    assert length >= 1
    cur = start
    for _ in range(length):
        nxt = _step(cur, dir_name)
        prev_owner = nodes_owner.get(nxt)
        assert prev_owner in (None, owner)
        nodes_owner[nxt] = owner
        grey_edges.append((cur, nxt))
        cur = nxt
    return cur


def _build_abstract_case() -> dict[str, object]:
    nodes_owner: dict[tuple[int, int], str] = {}
    grey_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    valve_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []

    def add_node(node: tuple[int, int], owner: str) -> None:
        prev_owner = nodes_owner.get(node)
        assert prev_owner in (None, owner)
        nodes_owner[node] = owner

    # Component A: two same-color sources and one valve-side port.
    a_center = (0, -3)
    add_node(a_center, "A")
    a_port = _add_chain(a_center, "R", 1, "A", nodes_owner, grey_edges)
    a_source_dirs = random.sample(["U", "L", "D"], 2)
    a_extra_dir = next(d for d in ["U", "L", "D"] if d not in a_source_dirs)
    a_sources = []
    for dir_name in a_source_dirs:
        end = _add_chain(a_center, dir_name, random.randint(1, 2), "A", nodes_owner, grey_edges)
        a_sources.append(end)
    if random.random() < 0.6:
        _add_chain(a_center, a_extra_dir, random.randint(1, 2), "A", nodes_owner, grey_edges)

    # Component B: two different-color sources, one valve on each side.
    b_center = (0, 0)
    add_node(b_center, "B")
    b_left = _add_chain(b_center, "L", 1, "B", nodes_owner, grey_edges)
    b_right = _add_chain(b_center, "R", 1, "B", nodes_owner, grey_edges)
    b_up = _add_chain(b_center, "U", random.randint(1, 2), "B", nodes_owner, grey_edges)
    b_down = _add_chain(b_center, "D", random.randint(1, 2), "B", nodes_owner, grey_edges)

    # Component C: optional isolated, single-source, same-source, or diff-source branch set.
    c_center = (0, 3)
    add_node(c_center, "C")
    c_port = _add_chain(c_center, "L", 1, "C", nodes_owner, grey_edges)
    c_mode = random.choices(
        ["isolated", "single", "same", "diff"],
        weights=[2, 2, 1, 1],
        k=1,
    )[0]
    if c_mode in {"same", "diff"}:
        c_dir_count = random.randint(2, 3)
    else:
        c_dir_count = random.randint(1, 3)
    c_dirs = random.sample(["U", "R", "D"], c_dir_count)
    c_sources: list[tuple[int, int]] = []
    if c_mode == "isolated":
        for dir_name in c_dirs[: random.randint(1, min(2, len(c_dirs)))]:
            _add_chain(c_center, dir_name, random.randint(1, 2), "C", nodes_owner, grey_edges)
    elif c_mode == "single":
        src_dir = c_dirs[0]
        c_sources.append(_add_chain(c_center, src_dir, random.randint(1, 2), "C", nodes_owner, grey_edges))
        for dir_name in c_dirs[1:]:
            _add_chain(c_center, dir_name, random.randint(1, 2), "C", nodes_owner, grey_edges)
    else:
        need = 2
        for dir_name in c_dirs[:need]:
            c_sources.append(_add_chain(c_center, dir_name, random.randint(1, 2), "C", nodes_owner, grey_edges))
        if len(c_dirs) > need:
            _add_chain(c_center, c_dirs[need], random.randint(1, 2), "C", nodes_owner, grey_edges)

    valve_edges.append((a_port, b_left))
    valve_edges.append((b_right, c_port))

    palette = list(SOURCE_PALETTE)
    random.shuffle(palette)
    a_color = int(palette.pop())
    b_color_1 = int(palette.pop())
    b_color_2 = int(palette.pop())
    while b_color_2 == b_color_1:
        random.shuffle(palette)
        b_color_2 = int(palette.pop())

    source_nodes: list[tuple[tuple[int, int], int]] = []
    for node in a_sources:
        source_nodes.append((node, a_color))
    source_nodes.append((b_up, b_color_1))
    source_nodes.append((b_down, b_color_2))

    c_expected: dict[str, object] = {"mode": c_mode}
    if c_mode == "single":
        c_color = random.choice([a_color, b_color_1, b_color_2] + palette)
        source_nodes.append((c_sources[0], int(c_color)))
        c_expected["colors"] = (int(c_color),)
    elif c_mode == "same":
        c_color = random.choice([a_color, b_color_1, b_color_2] + palette)
        for node in c_sources:
            source_nodes.append((node, int(c_color)))
        c_expected["colors"] = (int(c_color),)
    elif c_mode == "diff":
        c_pool = [a_color, b_color_1, b_color_2] + palette
        c1, c2 = random.sample(c_pool, 2)
        source_nodes.append((c_sources[0], int(c1)))
        source_nodes.append((c_sources[1], int(c2)))
        c_expected["colors"] = (int(c1), int(c2))
    else:
        c_expected["colors"] = tuple()

    return {
        "nodes_owner": nodes_owner,
        "grey_edges": grey_edges,
        "valve_edges": valve_edges,
        "source_nodes": source_nodes,
        "expectations": {
            "A": {"mode": "same", "colors": (a_color,)},
            "B": {"mode": "diff", "colors": (b_color_1, b_color_2)},
            "C": c_expected,
        },
    }


def _transform_case(case: dict[str, object]) -> dict[str, object]:
    turns = random.randrange(4)
    nodes_owner = {
        _rotate_coord(node, turns): owner
        for node, owner in case["nodes_owner"].items()
    }
    grey_edges = [
        (_rotate_coord(a, turns), _rotate_coord(b, turns))
        for a, b in case["grey_edges"]
    ]
    valve_edges = [
        (_rotate_coord(a, turns), _rotate_coord(b, turns))
        for a, b in case["valve_edges"]
    ]
    source_nodes = [
        (_rotate_coord(node, turns), int(color))
        for node, color in case["source_nodes"]
    ]

    coords = list(nodes_owner.keys())
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)

    pad_top = random.randint(0, 4)
    pad_bottom = random.randint(0, 4)
    pad_left = random.randint(0, 5)
    pad_right = random.randint(0, 5)
    shift_r = int(pad_top - min_r)
    shift_c = int(pad_left - min_c)

    def shift(node: tuple[int, int]) -> tuple[int, int]:
        return int(node[0] + shift_r), int(node[1] + shift_c)

    lattice_h = int(max_r - min_r + 1 + pad_top + pad_bottom)
    lattice_w = int(max_c - min_c + 1 + pad_left + pad_right)

    return {
        "nodes_owner": {shift(node): owner for node, owner in nodes_owner.items()},
        "grey_edges": [(shift(a), shift(b)) for a, b in grey_edges],
        "valve_edges": [(shift(a), shift(b)) for a, b in valve_edges],
        "source_nodes": [(shift(node), int(color)) for node, color in source_nodes],
        "expectations": case["expectations"],
        "lattice_shape": (lattice_h, lattice_w),
    }


def _render_case(case: dict[str, object]) -> tuple[np.ndarray, dict[str, set[tuple[int, int]]], set[tuple[int, int]]]:
    lattice_h, lattice_w = case["lattice_shape"]
    h = int(2 * lattice_h + 5)
    w = int(2 * lattice_w + 5)
    grid = np.full((h, w), BLACK, dtype=int)

    nodes_owner: dict[tuple[int, int], str] = case["nodes_owner"]
    grey_edges: list[tuple[tuple[int, int], tuple[int, int]]] = case["grey_edges"]
    valve_edges: list[tuple[tuple[int, int], tuple[int, int]]] = case["valve_edges"]
    source_nodes: list[tuple[tuple[int, int], int]] = case["source_nodes"]

    component_cells = {"A": set(), "B": set(), "C": set()}
    node_pixels: dict[tuple[int, int], tuple[int, int]] = {}
    grey_adj: dict[tuple[int, int], list[tuple[int, int]]] = {node: [] for node in nodes_owner}

    for node, owner in nodes_owner.items():
        pr = int(2 + 2 * node[0])
        pc = int(2 + 2 * node[1])
        node_pixels[node] = (pr, pc)
        assert int(grid[pr, pc]) == BLACK
        grid[pr, pc] = GREY
        component_cells[owner].add((pr, pc))

    for a, b in grey_edges:
        owner_a = nodes_owner[a]
        owner_b = nodes_owner[b]
        assert owner_a == owner_b
        ar, ac = node_pixels[a]
        br, bc = node_pixels[b]
        mr = int((ar + br) // 2)
        mc = int((ac + bc) // 2)
        assert int(grid[mr, mc]) == BLACK
        grid[mr, mc] = GREY
        component_cells[owner_a].add((mr, mc))
        grey_adj[a].append(b)
        grey_adj[b].append(a)

    valve_pixels: set[tuple[int, int]] = set()
    for a, b in valve_edges:
        ar, ac = node_pixels[a]
        br, bc = node_pixels[b]
        mr = int((ar + br) // 2)
        mc = int((ac + bc) // 2)
        assert int(grid[mr, mc]) == BLACK
        grid[mr, mc] = MAGENTA
        valve_pixels.add((mr, mc))

    for node, color in source_nodes:
        neighbors = grey_adj[node]
        assert len(neighbors) == 1
        pr, pc = node_pixels[node]
        dr = int(neighbors[0][0] - node[0])
        dc = int(neighbors[0][1] - node[1])
        sr = int(pr - dr)
        sc = int(pc - dc)
        assert 0 <= sr < h and 0 <= sc < w
        assert int(grid[sr, sc]) == BLACK
        grid[sr, sc] = int(color)

    return grid, component_cells, valve_pixels


def _validate_generated_case(
    grid: np.ndarray,
    component_cells: dict[str, set[tuple[int, int]]],
    valve_pixels: set[tuple[int, int]],
    expectations: dict[str, dict[str, object]],
) -> bool:
    path_mask = (grid == GREY) | (grid == MAGENTA)
    if not _mask_connected(path_mask):
        return False

    h, w = grid.shape
    for r, c in np.argwhere((grid != BLACK) & (grid != GREY) & (grid != MAGENTA)):
        adj = sum(1 for nr, nc in _neighbors4(int(r), int(c), h, w) if int(grid[nr, nc]) == GREY)
        if adj != 1:
            return False

    out = main(grid)
    if not bool(np.any(out != grid)):
        return False

    for r, c in valve_pixels:
        if int(out[r, c]) != MAGENTA:
            return False

    a_color = int(expectations["A"]["colors"][0])
    if {int(out[r, c]) for r, c in component_cells["A"]} != {a_color}:
        return False

    b_colors = tuple(int(v) for v in expectations["B"]["colors"])
    b_vals = [int(out[r, c]) for r, c in component_cells["B"]]
    if GREEN not in b_vals:
        return False
    if not set(b_vals).issubset(set(b_colors) | {GREEN}):
        return False

    c_mode = str(expectations["C"]["mode"])
    c_colors = tuple(int(v) for v in expectations["C"]["colors"])
    c_vals = [int(out[r, c]) for r, c in component_cells["C"]]
    if c_mode == "isolated":
        if set(c_vals) != {GREY}:
            return False
    elif c_mode in {"single", "same"}:
        if set(c_vals) != {int(c_colors[0])}:
            return False
    else:
        if GREEN not in c_vals:
            return False
        if not set(c_vals).issubset(set(c_colors) | {GREEN}):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            abstract_case = _build_abstract_case()
            transformed_case = _transform_case(abstract_case)
            grid, component_cells, valve_pixels = _render_case(transformed_case)
            if _validate_generated_case(
                grid,
                component_cells,
                valve_pixels,
                transformed_case["expectations"],
            ):
                return grid
        except AssertionError:
            continue
    raise AssertionError("failed to generate a valid signal-propagation puzzle")
