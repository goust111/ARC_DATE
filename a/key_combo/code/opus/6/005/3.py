from __future__ import annotations

import random
from collections import Counter

import numpy as np
from scipy.ndimage import label


BG = 5
CORRIDOR = 0
DEST = 1
MOVER = 2
SPRITE = 3

ROOM_HALF = 2
ROOM_SIZE = 2 * ROOM_HALF + 1
LATTICE_SIZE = 4
STEP = 7
BASE_GRID = 27
STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {d: (-d[0], -d[1]) for d in DIRS}
GEN_ATTEMPTS = 400


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    struct = STRUCT4 if int(connectivity) == 4 else STRUCT8
    lab, n = label(mask.astype(bool), structure=struct)
    return [np.argwhere(lab == k) for k in range(1, int(n) + 1)]


def _add(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    return int(a[0] + b[0]), int(a[1] + b[1])


def _dir(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    dr = int(np.sign(int(b[0]) - int(a[0])))
    dc = int(np.sign(int(b[1]) - int(a[1])))
    return dr, dc


def _center_of_lattice(pos: tuple[int, int]) -> tuple[int, int]:
    return 3 + STEP * int(pos[0]), 3 + STEP * int(pos[1])


def _room_bbox(center: tuple[int, int]) -> tuple[int, int, int, int]:
    r, c = center
    return r - ROOM_HALF, c - ROOM_HALF, r + ROOM_HALF, c + ROOM_HALF


def _paint_room(grid: np.ndarray, center: tuple[int, int]) -> None:
    r0, c0, r1, c1 = _room_bbox(center)
    grid[r0 : r1 + 1, c0 : c1 + 1] = CORRIDOR


def _paint_link(grid: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> None:
    ra, ca = _center_of_lattice(a)
    rb, cb = _center_of_lattice(b)
    if ra == rb:
        c0, c1 = sorted((ca, cb))
        grid[ra - 1 : ra + 1, c0 + ROOM_HALF + 1 : c1 - ROOM_HALF] = CORRIDOR
    else:
        r0, r1 = sorted((ra, rb))
        grid[r0 + ROOM_HALF + 1 : r1 - ROOM_HALF, ca - 1 : ca + 1] = CORRIDOR


def _paint_stub(grid: np.ndarray, center: tuple[int, int], direction: tuple[int, int], length: int) -> None:
    r, c = center
    dr, dc = direction
    if dr == -1:
        grid[r - ROOM_HALF - length : r - ROOM_HALF, c - 1 : c + 1] = CORRIDOR
    elif dr == 1:
        grid[r + ROOM_HALF + 1 : r + ROOM_HALF + 1 + length, c - 1 : c + 1] = CORRIDOR
    elif dc == -1:
        grid[r - 1 : r + 1, c - ROOM_HALF - length : c - ROOM_HALF] = CORRIDOR
    else:
        grid[r - 1 : r + 1, c + ROOM_HALF + 1 : c + ROOM_HALF + 1 + length] = CORRIDOR


BASE_UP_TEMPLATES = [
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)],
]


def _transform_template(cells: list[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    out = []
    for r, c in cells:
        rr = r - 1
        cc = c - 1
        if direction == (-1, 0):
            tr, tc = rr, cc
        elif direction == (1, 0):
            tr, tc = -rr, cc
        elif direction == (0, -1):
            tr, tc = cc, rr
        else:
            tr, tc = cc, -rr
        out.append((tr + 1, tc + 1))
    return out


def _stamp_sprite(grid: np.ndarray, center: tuple[int, int], bias_dir: tuple[int, int]) -> None:
    top = center[0] - 1
    left = center[1] - 1
    cells = _transform_template(random.choice(BASE_UP_TEMPLATES), bias_dir)
    for dr, dc in cells:
        grid[top + dr, left + dc] = SPRITE


def _decode_bias(coords: np.ndarray) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox_coords(coords)
    cent_r = float(coords[:, 0].mean())
    cent_c = float(coords[:, 1].mean())
    box_r = (r0 + r1) / 2.0
    box_c = (c0 + c1) / 2.0
    dr = cent_r - box_r
    dc = cent_c - box_c
    if abs(dr) > abs(dc):
        assert abs(dr) > 1e-6
        return (1, 0) if dr > 0 else (-1, 0)
    assert abs(dc) > 1e-6
    return (0, 1) if dc > 0 else (0, -1)


def _single_coord(grid: np.ndarray, color: int) -> tuple[int, int]:
    pts = np.argwhere(grid == int(color))
    assert pts.shape == (1, 2)
    return int(pts[0, 0]), int(pts[0, 1])


def _node_region(center: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = _room_bbox(center)
    h, w = shape
    return max(0, r0), max(0, c0), min(h - 1, r1), min(w - 1, c1)


def _touch_direction(
    comp: np.ndarray, region: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = region
    up = int(np.count_nonzero((comp[:, 0] == r0 - 1) & (comp[:, 1] >= c0) & (comp[:, 1] <= c1)))
    down = int(np.count_nonzero((comp[:, 0] == r1 + 1) & (comp[:, 1] >= c0) & (comp[:, 1] <= c1)))
    left = int(np.count_nonzero((comp[:, 1] == c0 - 1) & (comp[:, 0] >= r0) & (comp[:, 0] <= r1)))
    right = int(np.count_nonzero((comp[:, 1] == c1 + 1) & (comp[:, 0] >= r0) & (comp[:, 0] <= r1)))
    counts = {(-1, 0): up, (1, 0): down, (0, -1): left, (0, 1): right}
    best = max(counts.items(), key=lambda kv: kv[1])
    if best[1] == 0:
        return None
    winners = [d for d, n in counts.items() if n == best[1] and n > 0]
    assert len(winners) == 1
    return winners[0]


def _extract_graph(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    start = _single_coord(g, MOVER)
    dest = _single_coord(g, DEST)
    nodes = [
        {"kind": "start", "center": start, "coords": np.array([start], dtype=int)},
        {"kind": "dest", "center": dest, "coords": np.array([dest], dtype=int)},
    ]
    junctions = components_mask(g == SPRITE, connectivity=8)
    junctions.sort(key=lambda pts: tuple(map(int, bbox_coords(pts)[:2])))
    for pts in junctions:
        r0, c0, r1, c1 = bbox_coords(pts)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        nodes.append({"kind": "junction", "center": center, "coords": pts})

    walkable = np.isin(g, [CORRIDOR, DEST, MOVER, SPRITE])
    corridor = walkable.copy()
    for node in nodes:
        r0, c0, r1, c1 = _node_region(node["center"], g.shape)
        corridor[r0 : r1 + 1, c0 : c1 + 1] = False

    comps = components_mask(corridor, connectivity=4)
    adjacency = [[] for _ in nodes]
    for comp_id, comp in enumerate(comps):
        touches = []
        for idx, node in enumerate(nodes):
            direction = _touch_direction(comp, _node_region(node["center"], g.shape))
            if direction is not None:
                touches.append((idx, direction))
        assert 1 <= len(touches) <= 2
        if len(touches) == 1:
            idx, direction = touches[0]
            adjacency[idx].append({"dir": direction, "other": None, "comp": comp_id})
        else:
            (a, da), (b, db) = touches
            adjacency[a].append({"dir": da, "other": b, "comp": comp_id})
            adjacency[b].append({"dir": db, "other": a, "comp": comp_id})
    return nodes, comps, adjacency


def _follow_route(grid: np.ndarray):
    nodes, comps, adjacency = _extract_graph(grid)
    start_idx = 0
    dest_idx = 1
    current = start_idx
    prev = None
    visited_junctions: list[int] = []
    final_cell = None

    for _ in range(len(comps) + len(nodes) + 5):
        if current == dest_idx:
            return {
                "reached_dest": True,
                "visited_junctions": visited_junctions,
                "final_cell": tuple(nodes[dest_idx]["center"]),
                "nodes": nodes,
                "comps": comps,
                "adjacency": adjacency,
            }

        node = nodes[current]
        if node["kind"] == "junction":
            visited_junctions.append(current)
            bias = _decode_bias(node["coords"])
            wanted = OPPOSITE[bias]
            options = [edge for edge in adjacency[current] if edge["dir"] == wanted]
            assert len(options) == 1
            edge = options[0]
        else:
            options = [edge for edge in adjacency[current] if edge["other"] != prev]
            if prev is None:
                assert len(options) == 1
            else:
                assert len(options) == 1
            edge = options[0]

        if edge["other"] is None:
            comp = comps[edge["comp"]]
            cr, cc = nodes[current]["center"]
            dists = np.abs(comp[:, 0] - int(cr)) + np.abs(comp[:, 1] - int(cc))
            r, c = map(int, comp[int(np.argmax(dists))])
            final_cell = (r, c)
            return {
                "reached_dest": False,
                "visited_junctions": visited_junctions,
                "final_cell": final_cell,
                "nodes": nodes,
                "comps": comps,
                "adjacency": adjacency,
            }

        prev, current = current, edge["other"]

    raise AssertionError("route traversal did not terminate")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow a branching black corridor and recolor only the visited direction sprites.

    Input structure:
    - Grey (5) is the background, black (0) is a width-2 winding corridor, red (2) is the mover, blue (1) is the destination, and green (3) sprites sit in junction rooms.
    - Each green sprite is a small 8-connected asymmetric body whose centroid is biased toward one cardinal side of its own bbox.
    - Corridor branches meet only at sprite-bearing junctions; dead-end side branches may exist.

    Rule:
    - Remove the red mover from its starting cell by turning that cell black.
    - Reconstruct the corridor graph from the start room, destination room, and green-sprite junction rooms.
    - Whenever the mover reaches a junction, compute that sprite's centroid-bias direction and take the opposite cardinal direction as the chosen outgoing branch.
    - Recolor every visited junction sprite to red, leave unvisited sprites green, and replace the blue destination with red if the route reaches it.

    Key insight:
    - The green sprite is a local control signal: its mass imbalance selects one branch of the maze, so both graph topology and centroid decoding are required to recover the route.
    """
    g = np.asarray(grid, dtype=int)
    route = _follow_route(g)
    out = g.copy()

    sr, sc = _single_coord(g, MOVER)
    out[sr, sc] = CORRIDOR

    for node_idx in route["visited_junctions"]:
        pts = route["nodes"][node_idx]["coords"]
        out[pts[:, 0], pts[:, 1]] = MOVER

    if route["reached_dest"]:
        dr, dc = _single_coord(g, DEST)
        out[dr, dc] = MOVER
    else:
        fr, fc = route["final_cell"]
        out[fr, fc] = MOVER

    return out


def _random_path(length: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        path = [(
            random.randrange(LATTICE_SIZE),
            random.randrange(LATTICE_SIZE),
        )]
        while len(path) < length:
            cur = path[-1]
            candidates = []
            for d in DIRS:
                nxt = _add(cur, d)
                if not (0 <= nxt[0] < LATTICE_SIZE and 0 <= nxt[1] < LATTICE_SIZE):
                    continue
                if nxt in path:
                    continue
                candidates.append(nxt)
            if not candidates:
                break
            path.append(random.choice(candidates))
        if len(path) == length:
            return path
    return None


def _has_turn(path: list[tuple[int, int]]) -> bool:
    for i in range(1, len(path) - 1):
        incoming = _dir(path[i], path[i - 1])
        outgoing = _dir(path[i], path[i + 1])
        if outgoing != OPPOSITE[incoming]:
            return True
    return False


def _free_decoy_dirs(
    path: list[tuple[int, int]], idx: int, blocked: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    cur = path[idx]
    forbidden = {_dir(cur, path[idx - 1]), _dir(cur, path[idx + 1])}
    out = []
    for d in DIRS:
        if d in forbidden:
            continue
        nxt = _add(cur, d)
        if not (0 <= nxt[0] < LATTICE_SIZE and 0 <= nxt[1] < LATTICE_SIZE):
            continue
        if nxt in blocked:
            continue
        out.append(d)
    return out


def _paint_meta_instance(meta: dict) -> np.ndarray:
    grid = np.full((BASE_GRID, BASE_GRID), BG, dtype=int)
    path = meta["path"]
    rooms = set(path)
    if meta["extra_pos"] is not None:
        rooms.add(meta["extra_pos"])

    for pos in rooms:
        _paint_room(grid, _center_of_lattice(pos))
    for a, b in zip(path, path[1:]):
        _paint_link(grid, a, b)
    if meta["extra_pos"] is not None:
        _paint_link(grid, path[meta["extra_parent_idx"]], meta["extra_pos"])

    start_center = _center_of_lattice(path[0])
    dest_center = _center_of_lattice(path[-1])
    grid[start_center] = MOVER
    grid[dest_center] = DEST

    for idx in range(1, len(path) - 1):
        center = _center_of_lattice(path[idx])
        correct_dir = _dir(path[idx], path[idx + 1])
        _stamp_sprite(grid, center, OPPOSITE[correct_dir])
        stub_dir = meta["main_decoys"][idx]
        if meta["extra_parent_idx"] == idx:
            continue
        _paint_stub(grid, center, stub_dir, random.randint(3, 4))

    if meta["extra_pos"] is not None:
        center = _center_of_lattice(meta["extra_pos"])
        correct_dir = random.choice(meta["extra_stub_dirs"])
        _stamp_sprite(grid, center, OPPOSITE[correct_dir])
        for d in meta["extra_stub_dirs"]:
            _paint_stub(grid, center, d, random.randint(3, 4))

    rows, cols = np.where(grid != BG)
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    top = random.randint(0, min(3, r0))
    left = random.randint(0, min(3, c0))
    bottom = random.randint(0, min(3, BASE_GRID - 1 - r1))
    right = random.randint(0, min(3, BASE_GRID - 1 - c1))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _build_meta() -> dict | None:
    total_junctions = random.randint(2, 4)
    extra = 1 if total_junctions >= 3 and random.random() < 0.6 else 0
    main_junctions = total_junctions - extra
    path_len = main_junctions + 2
    path = _random_path(path_len)
    if path is None or not _has_turn(path):
        return None

    blocked = set(path)
    extra_parent_idx = None
    extra_pos = None
    extra_stub_dirs = None

    if extra:
        candidates = []
        for idx in range(1, len(path) - 1):
            for d in _free_decoy_dirs(path, idx, blocked):
                pos = _add(path[idx], d)
                free2 = []
                for d2 in DIRS:
                    if d2 == OPPOSITE[d]:
                        continue
                    nxt = _add(pos, d2)
                    if not (0 <= nxt[0] < LATTICE_SIZE and 0 <= nxt[1] < LATTICE_SIZE):
                        continue
                    if nxt in blocked:
                        continue
                    free2.append(d2)
                if len(free2) >= 2:
                    candidates.append((idx, d, pos, free2))
        if not candidates:
            return None
        extra_parent_idx, extra_dir, extra_pos, free2 = random.choice(candidates)
        blocked.add(extra_pos)
        extra_stub_dirs = random.sample(free2, 2)
    else:
        extra_dir = None

    main_decoys: dict[int, tuple[int, int]] = {}
    for idx in range(1, len(path) - 1):
        if extra_parent_idx == idx:
            main_decoys[idx] = extra_dir
            continue
        choices = _free_decoy_dirs(path, idx, blocked)
        if not choices:
            return None
        main_decoys[idx] = random.choice(choices)

    return {
        "path": path,
        "extra_parent_idx": extra_parent_idx,
        "extra_pos": extra_pos,
        "main_decoys": main_decoys,
        "extra_stub_dirs": extra_stub_dirs,
        "main_junctions": main_junctions,
        "total_junctions": total_junctions,
    }


def _validate_generated(grid: np.ndarray, meta: dict) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if Counter(map(int, grid.ravel()))[MOVER] != 1 or Counter(map(int, grid.ravel()))[DEST] != 1:
        return False
    sprite_comps = components_mask(grid == SPRITE, connectivity=8)
    if len(sprite_comps) != meta["total_junctions"]:
        return False
    for pts in sprite_comps:
        if not (5 <= len(pts) <= 7):
            return False
        bias = _decode_bias(pts)
        if bias not in DIRS:
            return False

    route = _follow_route(grid)
    if not route["reached_dest"]:
        return False
    if len(route["visited_junctions"]) != meta["main_junctions"]:
        return False

    nodes = route["nodes"]
    adjacency = route["adjacency"]
    junction_nodes = [idx for idx, node in enumerate(nodes) if node["kind"] == "junction"]
    if len(junction_nodes) != meta["total_junctions"]:
        return False
    if any(len(adjacency[idx]) != 3 for idx in junction_nodes):
        return False

    turned = False
    path = meta["path"]
    for idx in range(1, len(path) - 1):
        incoming = _dir(path[idx], path[idx - 1])
        outgoing = _dir(path[idx], path[idx + 1])
        if outgoing != OPPOSITE[incoming]:
            turned = True
            break
    if not turned:
        return False

    out = main(grid)
    dr, dc = _single_coord(grid, DEST)
    if int(out[dr, dc]) != MOVER:
        return False
    if meta["extra_pos"] is not None and not np.any(out == SPRITE):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        meta = _build_meta()
        if meta is None:
            continue
        grid = _paint_meta_instance(meta)
        if _validate_generated(grid, meta):
            return grid
    raise RuntimeError("failed to generate valid input")
