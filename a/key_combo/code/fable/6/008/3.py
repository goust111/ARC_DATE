from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import components_mask, get_bg


BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
BG_CHOICES = (0, 5, 6, 7, 8, 9)

UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

SHAPES = (
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0]], dtype=bool),
)

GRID_RETRIES = 500
PLACEMENT_RETRIES = 60
DISTRACTOR_PROB = 0.55
MAX_BLOBS = 8


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _offset_cells(mask: np.ndarray, origin: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = origin
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _side_cells(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = mask.shape
    out = []
    for r, c in np.argwhere(mask):
        rr = int(r + d[0])
        cc = int(c + d[1])
        if not (0 <= rr < h and 0 <= cc < w and mask[rr, cc]):
            out.append((int(r), int(c)))
    return out


SIDE_CACHE = {
    shape.tobytes(): {d: _side_cells(shape, d) for d in DIRS}
    for shape in SHAPES
}


@dataclass
class Node:
    idx: int
    mask: np.ndarray
    parent: int | None = None
    incoming: tuple[int, int] | None = None
    origin: tuple[int, int] | None = None
    children: list[tuple[tuple[int, int], int]] = field(default_factory=list)
    add_back_edge: bool = False
    port_local_by_dir: dict[tuple[int, int], tuple[int, int]] = field(default_factory=dict)
    entry_local: tuple[int, int] | None = None


def _extract_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    blue_comps = components_mask(g == BLUE, connectivity=8)
    assert blue_comps, "expected at least one blue component"
    cell_to_blob: dict[tuple[int, int], int] = {}
    blob_sets: list[frozenset[tuple[int, int]]] = []
    for idx, comp in enumerate(blue_comps):
        pts = frozenset((int(r), int(c)) for r, c in comp)
        blob_sets.append(pts)
        for pt in pts:
            cell_to_blob[pt] = idx

    yellow = np.argwhere(g == YELLOW)
    assert yellow.shape[0] == 1, "expected exactly one yellow marker"
    yr, yc = map(int, yellow[0])
    start_neighbors = {
        cell_to_blob[(nr, nc)]
        for nr, nc in ((yr - 1, yc), (yr + 1, yc), (yr, yc - 1), (yr, yc + 1))
        if (nr, nc) in cell_to_blob
    }
    assert len(start_neighbors) == 1, "yellow must touch exactly one blue blob"
    start = next(iter(start_neighbors))

    ports: dict[int, list[tuple[tuple[int, int], tuple[int, int]]]] = {i: [] for i in range(len(blob_sets))}
    for gr, gc in np.argwhere(g == GREEN):
        gr = int(gr)
        gc = int(gc)
        owners = set()
        dirs = set()
        for nr, nc in ((gr - 1, gc), (gr + 1, gc), (gr, gc - 1), (gr, gc + 1)):
            if (nr, nc) in cell_to_blob:
                owners.add(cell_to_blob[(nr, nc)])
                dirs.add((int(gr - nr), int(gc - nc)))
        assert len(owners) == 1 and len(dirs) == 1, "each green port must belong to one blob side"
        owner = next(iter(owners))
        d = next(iter(dirs))
        assert d in OPP, "port direction must be cardinal"
        ports[owner].append(((gr, gc), d))

    for idx, blob_ports in ports.items():
        assert blob_ports, f"blob {idx} is missing ports"

    adj: dict[int, set[int]] = {i: set() for i in range(len(blob_sets))}
    ray_targets: list[tuple[int, int]] = []
    for src, blob_ports in ports.items():
        for (r, c), d in blob_ports:
            rr, cc = r, c
            target = None
            while True:
                rr += int(d[0])
                cc += int(d[1])
                if not (0 <= rr < h and 0 <= cc < w):
                    break
                if g[rr, cc] == BLUE:
                    target = cell_to_blob[(rr, cc)]
                    break
            assert target is not None, "every port ray must hit a blob"
            adj[src].add(int(target))
            ray_targets.append((int(src), int(target)))

    return {
        "bg": bg,
        "blobs": blob_sets,
        "start": int(start),
        "adj": adj,
        "ray_targets": ray_targets,
    }


def _wave_layers(adj: dict[int, set[int]], start: int) -> list[list[int]]:
    seen = {int(start)}
    wave = [int(start)]
    layers: list[list[int]] = [wave]
    while True:
        nxt = sorted({
            int(dst)
            for src in wave
            for dst in adj[src]
            if int(dst) not in seen
        })
        if not nxt:
            return layers
        seen.update(nxt)
        layers.append(nxt)
        wave = nxt


def _histogram(bg: int, counts: list[int]) -> np.ndarray:
    out = np.full((max(counts), len(counts)), int(bg), dtype=int)
    for c, count in enumerate(counts):
        if count:
            out[-count:, c] = RED
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count blue blobs activated wave by wave through green port rays and return a red histogram.

    Input structure:
    - The background is the most common color, with several blue 8-connected blobs.
    - Green single-pixel ports sit just outside blob edges and encode cardinal ray directions.
    - Exactly one yellow pixel touches one blob and marks the starting blob.

    Rule:
    - The yellow-touched blob is wave 1.
    - Each active wave emits straight rays from all of its green ports; every not-yet-activated blob first hit by those rays becomes the next wave.
    - Continue breadth-first until no new blobs are reached, then draw one output column per wave and fill each column upward in red by that wave's blob count.

    Key insight:
    - The answer is a time-profile of the cascade, so the solver must recover the directed port network and count BFS layers rather than redraw the scene.
    """
    scene = _extract_scene(np.asarray(grid, dtype=int))
    layers = _wave_layers(scene["adj"], scene["start"])
    counts = [len(layer) for layer in layers]
    return _histogram(scene["bg"], counts)


def _random_shape() -> np.ndarray:
    return np.array(random.choice(SHAPES), dtype=bool)


def _node_side_cells(node: Node, d: tuple[int, int]) -> list[tuple[int, int]]:
    return SIDE_CACHE[node.mask.tobytes()][d]


def _available_forward_dirs(node: Node) -> list[tuple[int, int]]:
    used = {d for d, _child in node.children}
    reserved = {OPP[node.incoming]} if node.incoming is not None else set()
    return [d for d in DIRS if d not in used and d not in reserved]


def _blue_conflict(cells: set[tuple[int, int]], occupied_blue: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if (nr, nc) in occupied_blue:
                    return True
    return False


def _build_reachable_graph() -> tuple[list[Node], list[list[int]]] | None:
    depth = random.choices([2, 3, 4, 5], weights=[1, 3, 3, 2])[0]
    max_blobs = random.randint(depth, MAX_BLOBS)

    nodes = [Node(idx=0, mask=_random_shape())]
    waves = [[0]]
    next_idx = 1

    for layer in range(1, depth):
        frontier = waves[-1]
        capacities = {nid: len(_available_forward_dirs(nodes[nid])) for nid in frontier}
        total_cap = sum(capacities.values())
        remaining_layers = depth - layer - 1
        max_new = min(total_cap, max_blobs - next_idx - remaining_layers)
        if max_new < 1:
            return None

        upper = min(4, max_new)
        choices = list(range(1, upper + 1))
        if layer == 1:
            weights = [1, 4, 3, 2][:upper]
        else:
            weights = [3, 3, 2, 1][:upper]
        need = random.choices(choices, weights=weights, k=1)[0]

        remaining = need
        degree_budget = capacities.copy()
        next_wave: list[int] = []
        while remaining:
            candidates = [nid for nid, cap in degree_budget.items() if cap > 0]
            if not candidates:
                return None
            parent = random.choice(candidates)
            dirs = _available_forward_dirs(nodes[parent])
            random.shuffle(dirs)
            chosen_dir = None
            for d in dirs:
                if d not in {dd for dd, _ in nodes[parent].children}:
                    chosen_dir = d
                    break
            if chosen_dir is None:
                return None
            child = Node(idx=next_idx, mask=_random_shape(), parent=parent, incoming=chosen_dir)
            nodes.append(child)
            nodes[parent].children.append((chosen_dir, next_idx))
            next_wave.append(next_idx)
            degree_budget[parent] -= 1
            next_idx += 1
            remaining -= 1

        waves.append(next_wave)

    for node in nodes[1:]:
        node.add_back_edge = (not node.children) or (random.random() < 0.45)
    return nodes, waves


def _place_child(
    parent: Node,
    child: Node,
    direction: tuple[int, int],
    occupied_blue: set[tuple[int, int]],
) -> bool:
    assert parent.origin is not None
    parent_side = _node_side_cells(parent, direction)[:]
    child_side = _node_side_cells(child, OPP[direction])[:]
    random.shuffle(parent_side)
    random.shuffle(child_side)
    gaps = [2, 3, 4, 5]
    random.shuffle(gaps)

    for p_local in parent_side:
        if direction in parent.port_local_by_dir:
            return False
        p_blue = _add(parent.origin, p_local)
        p_green = _add(p_blue, direction)
        for c_local in child_side:
            for gap in gaps:
                entry = (int(p_green[0] + direction[0] * gap), int(p_green[1] + direction[1] * gap))
                origin = (int(entry[0] - c_local[0]), int(entry[1] - c_local[1]))
                cells = _offset_cells(child.mask, origin)
                if p_green in cells or _blue_conflict(cells, occupied_blue):
                    continue
                child.origin = origin
                child.entry_local = c_local
                parent.port_local_by_dir[direction] = p_local
                occupied_blue.update(cells)
                return True
    return False


def _place_graph(nodes: list[Node], waves: list[list[int]]) -> bool:
    occupied_blue: set[tuple[int, int]] = set()
    nodes[0].origin = (0, 0)
    occupied_blue.update(_offset_cells(nodes[0].mask, nodes[0].origin))

    for wave in waves[:-1]:
        for nid in wave:
            parent = nodes[nid]
            for direction, child_id in parent.children:
                if not _place_child(parent, nodes[child_id], direction, occupied_blue):
                    return False
    return True


def _build_distractor_pair(start_idx: int) -> tuple[list[Node], list[list[int]]] | None:
    a = Node(idx=start_idx, mask=_random_shape())
    b = Node(idx=start_idx + 1, mask=_random_shape(), parent=start_idx, incoming=random.choice(DIRS))
    a.children.append((b.incoming, b.idx))
    b.add_back_edge = True
    return [a, b], [[a.idx], [b.idx]]


def _place_distractor_pair(nodes: list[Node], all_nodes: list[Node]) -> bool:
    occupied_blue = set()
    for node in all_nodes:
        assert node.origin is not None
        occupied_blue.update(_offset_cells(node.mask, node.origin))

    a, b = nodes
    for _ in range(PLACEMENT_RETRIES):
        anchor = (random.randint(-14, 14), random.randint(-14, 14))
        a.origin = anchor
        a_cells = _offset_cells(a.mask, a.origin)
        if _blue_conflict(a_cells, occupied_blue):
            continue
        new_occ = occupied_blue | a_cells
        if _place_child(a, b, b.incoming, new_occ):
            return True
        a.port_local_by_dir.clear()
        b.origin = None
        b.entry_local = None
    return False


def _all_port_specs(nodes: list[Node]) -> list[tuple[int, int, tuple[int, int], tuple[int, int]]]:
    specs = []
    for node in nodes:
        for direction, child_id in node.children:
            assert direction in node.port_local_by_dir and node.origin is not None
            blue_cell = _add(node.origin, node.port_local_by_dir[direction])
            green = _add(blue_cell, direction)
            specs.append((node.idx, child_id, green, direction))
        if node.add_back_edge:
            assert node.parent is not None and node.entry_local is not None and node.origin is not None
            direction = OPP[node.incoming]
            blue_cell = _add(node.origin, node.entry_local)
            green = _add(blue_cell, direction)
            specs.append((node.idx, node.parent, green, direction))
    return specs


def _choose_yellow(root: Node, port_specs: list[tuple[int, int, tuple[int, int], tuple[int, int]]]) -> tuple[int, int] | None:
    assert root.origin is not None
    used_dirs = {direction for src, _dst, _green, direction in port_specs if src == root.idx}
    options = [d for d in DIRS if d not in used_dirs]
    random.shuffle(options)
    occupied_green = {green for _src, _dst, green, _d in port_specs}
    for direction in options:
        side = _node_side_cells(root, direction)[:]
        random.shuffle(side)
        for local in side:
            blue = _add(root.origin, local)
            yellow = _add(blue, direction)
            if yellow not in occupied_green:
                return yellow
    return None


def _render_scene(nodes: list[Node], bg: int) -> tuple[np.ndarray, dict] | None:
    port_specs = _all_port_specs(nodes)
    yellow = _choose_yellow(nodes[0], port_specs)
    if yellow is None:
        return None

    blue_by_node: dict[int, set[tuple[int, int]]] = {
        node.idx: _offset_cells(node.mask, node.origin) for node in nodes
    }

    all_coords = set().union(*blue_by_node.values())
    green_coords = [green for _src, _dst, green, _d in port_specs]
    if len(set(green_coords)) != len(green_coords):
        return None
    if yellow in all_coords or yellow in set(green_coords):
        return None
    all_coords |= set(green_coords)
    all_coords.add(yellow)

    min_r = min(r for r, _c in all_coords)
    max_r = max(r for r, _c in all_coords)
    min_c = min(c for _r, c in all_coords)
    max_c = max(c for _r, c in all_coords)
    shift = (1 - min_r, 1 - min_c)
    h = max_r - min_r + 3
    w = max_c - min_c + 3
    if h > 30 or w > 30:
        return None

    grid = np.full((h, w), int(bg), dtype=int)
    shifted_blobs: dict[int, frozenset[tuple[int, int]]] = {}
    for node in nodes:
        shifted = frozenset((_add(cell, shift)) for cell in blue_by_node[node.idx])
        shifted_blobs[node.idx] = shifted
        for r, c in shifted:
            grid[r, c] = BLUE

    shifted_ports = []
    for src, dst, green, direction in port_specs:
        sgr = _add(green, shift)
        if grid[sgr] == BLUE:
            return None
        grid[sgr] = GREEN
        shifted_ports.append((src, dst, sgr, direction))

    sy = _add(yellow, shift)
    if grid[sy] != bg:
        return None
    grid[sy] = YELLOW

    expected_edges = {(src, dst) for src, dst, _green, _d in shifted_ports}
    reachable_adj: dict[int, set[int]] = {node.idx: set() for node in nodes}
    for src, dst in expected_edges:
        reachable_adj[src].add(dst)

    expected_layers = _wave_layers(reachable_adj, nodes[0].idx)
    expected_counts = [len(layer) for layer in expected_layers]
    expected = {
        "blob_sets": {shifted_blobs[node.idx] for node in nodes},
        "start_blob": shifted_blobs[nodes[0].idx],
        "edges": {
            (shifted_blobs[src], shifted_blobs[dst])
            for src, dst in expected_edges
        },
        "counts": expected_counts,
        "output": _histogram(bg, expected_counts),
    }
    return grid, expected


def _semantic_checks(grid: np.ndarray, expected: dict) -> bool:
    try:
        scene = _extract_scene(grid)
    except Exception:
        return False

    parsed_blobs = set(scene["blobs"])
    if parsed_blobs != expected["blob_sets"]:
        return False
    if scene["blobs"][scene["start"]] != expected["start_blob"]:
        return False

    parsed_edges = {
        (scene["blobs"][src], scene["blobs"][dst])
        for src, dst in scene["ray_targets"]
    }
    if parsed_edges != expected["edges"]:
        return False

    counts = [len(layer) for layer in _wave_layers(scene["adj"], scene["start"])]
    if counts != expected["counts"]:
        return False

    if not any(
        scene["blobs"][src] != expected["start_blob"] and dst in _wave_layers(scene["adj"], scene["start"])[0]
        for src, dst in scene["ray_targets"]
    ):
        start_idx = scene["start"]
        layers = _wave_layers(scene["adj"], start_idx)
        layer_of = {node: depth for depth, layer in enumerate(layers) for node in layer}
        if not any(layer_of[src] >= layer_of[dst] for src, dst in scene["ray_targets"] if src in layer_of and dst in layer_of):
            return False

    return np.array_equal(main(grid), expected["output"])


def _gen_attempt() -> np.ndarray | None:
    built = _build_reachable_graph()
    if built is None:
        return None
    nodes, waves = built
    if not _place_graph(nodes, waves):
        return None

    if random.random() < DISTRACTOR_PROB and len(nodes) <= MAX_BLOBS - 2:
        distractor = _build_distractor_pair(len(nodes))
        if distractor is None:
            return None
        d_nodes, _d_waves = distractor
        if not _place_distractor_pair(d_nodes, nodes):
            return None
        nodes = nodes + d_nodes

    bg = random.choice(BG_CHOICES)
    rendered = _render_scene(nodes, bg)
    if rendered is None:
        return None
    grid, expected = rendered
    if not _semantic_checks(grid, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GRID_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid cascade-histogram input")
