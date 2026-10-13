import itertools
from collections import defaultdict

import numpy as np


BG = 0
WALL = 5
LOCK = 8
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KEY_COLORS = [1, 2, 3, 4, 6, 7, 9]
MAX_GEN_RETRIES = 400
_RNG = np.random.default_rng()


def _normalize(coords):
    pts = [(int(r), int(c)) for r, c in coords]
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return tuple(sorted((r - min_r, c - min_c) for r, c in pts))


def _rotate90(coords):
    pts = list(coords)
    height = max(r for r, _ in pts) + 1
    return _normalize((c, height - 1 - r) for r, c in pts)


def _orientations(coords):
    out = []
    cur = _normalize(coords)
    for _ in range(4):
        if cur not in out:
            out.append(cur)
        cur = _rotate90(cur)
    return out


def _component_coords(mask):
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(tuple(pts))
    return comps


def _is_connected(coords):
    pts = set((int(r), int(c)) for r, c in coords)
    if not pts:
        return False
    stack = [next(iter(pts))]
    seen = {stack[0]}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in pts and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(pts)


def _shape_from_component(comp):
    return _normalize(comp)


def _complementary(shape_a, shape_b):
    area = len(shape_a) + len(shape_b)
    rects = [(h, area // h) for h in range(1, area + 1) if area % h == 0]
    for ori_a in _orientations(shape_a):
        pts_a = set(ori_a)
        ah = max(r for r, _ in ori_a) + 1
        aw = max(c for _, c in ori_a) + 1
        for ori_b in _orientations(shape_b):
            pts_b = set(ori_b)
            bh = max(r for r, _ in ori_b) + 1
            bw = max(c for _, c in ori_b) + 1
            for rect_h, rect_w in rects:
                if ah > rect_h or aw > rect_w or bh > rect_h or bw > rect_w:
                    continue
                for ar in range(rect_h - ah + 1):
                    for ac in range(rect_w - aw + 1):
                        placed_a = {(r + ar, c + ac) for r, c in pts_a}
                        for br in range(rect_h - bh + 1):
                            for bc in range(rect_w - bw + 1):
                                placed_b = {(r + br, c + bc) for r, c in pts_b}
                                if placed_a & placed_b:
                                    continue
                                if len(placed_a | placed_b) == rect_h * rect_w:
                                    return True
    return False


def _unique_perfect_matching(keys, locks):
    candidates = []
    for key in keys:
        row = []
        for lock in locks:
            row.append(_complementary(key["shape"], lock["shape"]))
        candidates.append(row)

    assert all(any(row) for row in candidates), "every key must match at least one lock"
    n = len(keys)
    used = [False] * n
    current = {}
    solutions = []

    order = sorted(range(n), key=lambda i: sum(candidates[i]))

    def search(idx):
        if len(solutions) > 1:
            return
        if idx == n:
            solutions.append(dict(current))
            return
        key_idx = order[idx]
        for lock_idx, ok in sorted(enumerate(candidates[key_idx]), key=lambda x: x[0]):
            if not ok or used[lock_idx]:
                continue
            used[lock_idx] = True
            current[key_idx] = lock_idx
            search(idx + 1)
            del current[key_idx]
            used[lock_idx] = False

    search(0)
    assert len(solutions) == 1, "key-lock pairing must be unique"
    return solutions[0]


def _black_neighbors(grid, pos):
    h, w = grid.shape
    r, c = pos
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
            out.append((nr, nc))
    return out


def _segment_from_lock(grid, lock_coords):
    h, w = grid.shape
    touching = set()
    for r, c in lock_coords:
        for dr, dc in DIRS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
                touching.add((nr, nc))
    assert len(touching) == 1, "lock must touch exactly one corridor entrance"

    prev = None
    cur = next(iter(touching))
    segment = []
    while True:
        segment.append(cur)
        forward = [nxt for nxt in _black_neighbors(grid, cur) if nxt != prev]
        if len(forward) >= 2:
            return segment
        if not forward:
            return segment
        prev, cur = cur, forward[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each colored key with the unique complementary azure lock and flood
    the matching key color along the lock's corridor segment until the first
    branch point, coloring the branch cell itself.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    lock_components = _component_coords(g == LOCK)
    key_components = []
    for color in sorted(set(map(int, g.ravel())) - {BG, WALL, LOCK}):
        for comp in _component_coords(g == int(color)):
            key_components.append({"color": int(color), "coords": comp, "shape": _shape_from_component(comp)})
    locks = [{"coords": comp, "shape": _shape_from_component(comp)} for comp in lock_components]

    assert key_components and len(key_components) == len(locks), "expected equal nonzero key and lock counts"
    matching = _unique_perfect_matching(key_components, locks)

    for key_idx, lock_idx in matching.items():
        color = int(key_components[key_idx]["color"])
        for r, c in _segment_from_lock(g, locks[lock_idx]["coords"]):
            out[r, c] = color

    return out


PAIR_LIBRARY = [
    {
        "key": _normalize([(0, 0), (0, 1), (0, 2)]),
        "lock": _normalize([(1, 0), (1, 1), (1, 2)]),
    },
    {
        "key": _normalize([(0, 0), (0, 1), (1, 0)]),
        "lock": _normalize([(0, 2), (1, 1), (1, 2)]),
    },
    {
        "key": _normalize([(0, 0), (0, 1), (0, 2), (1, 0)]),
        "lock": _normalize([(0, 3), (1, 1), (1, 2), (1, 3)]),
    },
    {
        "key": _normalize([(0, 0), (0, 1), (1, 0), (1, 1)]),
        "lock": _normalize([(0, 2), (0, 3), (1, 2), (1, 3)]),
    },
    {
        "key": _normalize([(0, 0), (1, 0), (1, 1), (1, 2)]),
        "lock": _normalize([(0, 1), (0, 2), (0, 3), (1, 3)]),
    },
]


def _build_spine(cell_h, cell_w, length):
    margin = 2
    rows = range(margin, cell_h - margin)
    cols = range(margin, cell_w - margin)
    for _ in range(120):
        start = (int(_RNG.choice(list(rows))), int(_RNG.choice(list(cols))))
        path = [start]
        used = {start}
        prev_dir = None
        while len(path) < length:
            cur_r, cur_c = path[-1]
            options = []
            for dr, dc in DIRS4:
                nr, nc = cur_r + dr, cur_c + dc
                if not (margin <= nr < cell_h - margin and margin <= nc < cell_w - margin):
                    continue
                if (nr, nc) in used:
                    continue
                options.append((dr, dc))
            if not options:
                break
            if prev_dir in options and _RNG.random() < 0.45:
                dr, dc = prev_dir
            else:
                dr, dc = options[int(_RNG.integers(len(options)))]
            nxt = (cur_r + dr, cur_c + dc)
            path.append(nxt)
            used.add(nxt)
            prev_dir = (dr, dc)
        if len(path) == length:
            return path
    return None


def _cell_neighbors(node):
    r, c = node
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _attach_straight_arm(adj, occupied, start_node, direction, max_length, cell_h, cell_w):
    dr, dc = direction
    limit = 0
    cur_r, cur_c = start_node
    while limit < max_length:
        nr = cur_r + dr * (limit + 1)
        nc = cur_c + dc * (limit + 1)
        if not (1 <= nr < cell_h - 1 and 1 <= nc < cell_w - 1):
            break
        if (nr, nc) in occupied:
            break
        limit += 1
    if limit == 0:
        return None

    length = int(_RNG.integers(1, limit + 1))
    prev = start_node
    end = start_node
    for step in range(1, length + 1):
        node = (start_node[0] + dr * step, start_node[1] + dc * step)
        occupied.add(node)
        adj[prev].add(node)
        adj[node].add(prev)
        prev = node
        end = node
    return end


def _render_tree(cell_h, cell_w, adj):
    h = cell_h * 2 + 1
    w = cell_w * 2 + 1
    grid = np.full((h, w), WALL, dtype=int)
    for node, neighbors in adj.items():
        gr, gc = node[0] * 2 + 1, node[1] * 2 + 1
        grid[gr, gc] = BG
        for nr, nc in neighbors:
            ngr, ngc = nr * 2 + 1, nc * 2 + 1
            grid[(gr + ngr) // 2, (gc + ngc) // 2] = BG
    return grid


def _leaf_nodes(adj):
    return [node for node, neighbors in adj.items() if len(neighbors) == 1]


def _grid_leaf_info(adj, leaf_node):
    parent = next(iter(adj[leaf_node]))
    gr, gc = leaf_node[0] * 2 + 1, leaf_node[1] * 2 + 1
    dr = int(np.sign(leaf_node[0] - parent[0]))
    dc = int(np.sign(leaf_node[1] - parent[1]))
    return (gr, gc), (dr, dc)


def _find_shape_placement(grid, leaf_pos, preferred_dir, shape):
    h, w = grid.shape
    available_dirs = []
    lr, lc = leaf_pos
    for dr, dc in DIRS4:
        nr, nc = lr + dr, lc + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
            available_dirs.append((dr, dc))

    directions = []
    if preferred_dir in available_dirs:
        directions.append(preferred_dir)
    directions.extend(d for d in available_dirs if d != preferred_dir)
    if len(directions) > 1:
        tail = directions[1:]
        _RNG.shuffle(tail)
        directions = [directions[0]] + tail

    for direction in directions:
        socket_r = lr + direction[0]
        socket_c = lc + direction[1]
        orientations = _orientations(shape)
        _RNG.shuffle(orientations)
        for ori in orientations:
            cells = list(ori)
            _RNG.shuffle(cells)
            for anchor_r, anchor_c in cells:
                shift_r = socket_r - anchor_r
                shift_c = socket_c - anchor_c
                placed = {(r + shift_r, c + shift_c) for r, c in ori}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in placed):
                    continue
                if any(int(grid[r, c]) != WALL for r, c in placed):
                    continue

                touches = []
                bad_touch = False
                for r, c in placed:
                    for dr, dc in DIRS4:
                        nr, nc = r + dr, c + dc
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        val = int(grid[nr, nc])
                        if val == BG:
                            touches.append((nr, nc))
                        elif val not in (WALL,) and (nr, nc) not in placed:
                            bad_touch = True
                if bad_touch:
                    continue
                if len(touches) != 1 or touches[0] != leaf_pos:
                    continue
                return sorted(placed)
    return None


def _path_to_first_branch(grid, leaf_pos):
    prev = None
    cur = leaf_pos
    segment = []
    while True:
        segment.append(cur)
        forward = [nxt for nxt in _black_neighbors(grid, cur) if nxt != prev]
        if len(forward) >= 2:
            return segment
        if not forward:
            return None
        prev, cur = cur, forward[0]


def _choose_tree_structure(pair_count, cell_h, cell_w):
    extra_key_arms = max(0, pair_count - 2)
    spare_arms = int(_RNG.integers(1, 3))
    spine_len = int(_RNG.integers(max(7, pair_count + 3), min(11, cell_h + cell_w) + 1))

    spine = _build_spine(cell_h, cell_w, spine_len)
    if spine is None:
        return None

    adj = defaultdict(set)
    occupied = set(spine)
    for a, b in zip(spine, spine[1:]):
        adj[a].add(b)
        adj[b].add(a)

    internal = spine[1:-1]
    if len(internal) < pair_count:
        return None

    def free_dirs(node):
        used = {(nbr[0] - node[0], nbr[1] - node[1]) for nbr in adj[node]}
        out = []
        for dr, dc in DIRS4:
            if (dr, dc) in used:
                continue
            out.append((dr, dc))
        return out

    hub_indices = list(range(len(internal)))
    _RNG.shuffle(hub_indices)
    lock_hubs = [internal[i] for i in hub_indices[:pair_count]]
    lock_leaves = []
    for hub in lock_hubs:
        dirs = free_dirs(hub)
        _RNG.shuffle(dirs)
        end = None
        for direction in dirs:
            end = _attach_straight_arm(adj, occupied, hub, direction, 3, cell_h, cell_w)
            if end is not None:
                break
        if end is None:
            return None
        lock_leaves.append(end)

    arms_to_add = extra_key_arms + spare_arms
    arm_endpoints = []
    attachable = list(internal)
    for _ in range(arms_to_add):
        placed = False
        _RNG.shuffle(attachable)
        for node in attachable:
            dirs = free_dirs(node)
            if not dirs:
                continue
            _RNG.shuffle(dirs)
            for direction in dirs:
                end = _attach_straight_arm(adj, occupied, node, direction, 3, cell_h, cell_w)
                if end is not None:
                    arm_endpoints.append(end)
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    return adj, set(lock_leaves)


def _gen_attempt():
    pair_count = int(_RNG.integers(3, 6))
    h = int(_RNG.choice(np.arange(23, 34, 2)))
    w = int(_RNG.choice(np.arange(23, 34, 2)))
    cell_h = (h - 1) // 2
    cell_w = (w - 1) // 2

    structure = _choose_tree_structure(pair_count, cell_h, cell_w)
    if structure is None:
        return None
    adj, lock_leaf_nodes = structure
    grid = _render_tree(cell_h, cell_w, adj)

    leaves = _leaf_nodes(adj)
    non_lock_leaf_nodes = [leaf for leaf in leaves if leaf not in lock_leaf_nodes]
    if len(lock_leaf_nodes) != pair_count or len(non_lock_leaf_nodes) < pair_count:
        return None

    lock_infos = []
    used_segment_cells = set()
    for leaf_node in lock_leaf_nodes:
        leaf_pos, outward = _grid_leaf_info(adj, leaf_node)
        seg = _path_to_first_branch(grid, leaf_pos)
        if seg is None:
            return None
        seg_set = set(seg)
        if used_segment_cells & seg_set:
            return None
        used_segment_cells |= seg_set
        lock_infos.append({"node": leaf_node, "leaf_pos": leaf_pos, "outward": outward})

    key_infos = []
    for leaf_node in non_lock_leaf_nodes:
        leaf_pos, outward = _grid_leaf_info(adj, leaf_node)
        key_infos.append({"node": leaf_node, "leaf_pos": leaf_pos, "outward": outward})

    pair_types = list(_RNG.choice(len(PAIR_LIBRARY), size=pair_count, replace=False))
    colors = list(_RNG.choice(KEY_COLORS, size=pair_count, replace=False))
    instances = []
    for pair_type, color in zip(pair_types, colors):
        spec = PAIR_LIBRARY[int(pair_type)]
        instances.append(
            {
                "color": int(color),
                "key_shape": spec["key"],
                "lock_shape": spec["lock"],
                "size": max(len(spec["key"]), len(spec["lock"])),
            }
        )
    instances.sort(key=lambda item: -item["size"])

    free_lock_infos = lock_infos[:]
    for inst in instances:
        _RNG.shuffle(free_lock_infos)
        placed = False
        for idx, info in enumerate(free_lock_infos):
            coords = _find_shape_placement(grid, info["leaf_pos"], info["outward"], inst["lock_shape"])
            if coords is None:
                continue
            for r, c in coords:
                grid[r, c] = LOCK
            inst["lock_coords"] = coords
            free_lock_infos.pop(idx)
            placed = True
            break
        if not placed:
            return None

    free_key_infos = key_infos[:]
    for inst in instances:
        _RNG.shuffle(free_key_infos)
        placed = False
        for idx, info in enumerate(free_key_infos):
            coords = _find_shape_placement(grid, info["leaf_pos"], info["outward"], inst["key_shape"])
            if coords is None:
                continue
            for r, c in coords:
                grid[r, c] = inst["color"]
            inst["key_coords"] = coords
            free_key_infos.pop(idx)
            placed = True
            break
        if not placed:
            return None

    key_comps = []
    for color in colors:
        comps = _component_coords(grid == int(color))
        if len(comps) != 1:
            return None
        key_comps.append({"color": int(color), "coords": comps[0], "shape": _shape_from_component(comps[0])})
    lock_comps = _component_coords(grid == LOCK)
    if len(lock_comps) != pair_count:
        return None
    locks = [{"coords": comp, "shape": _shape_from_component(comp)} for comp in lock_comps]
    try:
        matching = _unique_perfect_matching(key_comps, locks)
    except AssertionError:
        return None

    used_paths = set()
    for lock_idx in matching.values():
        seg = _segment_from_lock(grid, locks[lock_idx]["coords"])
        seg_set = set(seg)
        if used_paths & seg_set:
            return None
        used_paths |= seg_set

    try:
        solved = main(grid)
    except AssertionError:
        return None
    if solved.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for puzzle 6")
