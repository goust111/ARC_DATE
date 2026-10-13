import random
from collections import Counter

import numpy as np


BG = 0
CORRIDOR = 5
ENTRY_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
REL_PRIORITY = (1, 0, -1)
REL_TO_LABEL = {1: "R", 0: "S", -1: "L"}
SIDE_TO_DIR = {"top": 2, "right": 3, "bottom": 0, "left": 1}
GEN_ATTEMPTS = 400
PATH_ATTEMPTS = 120
BRANCH_ATTEMPTS = 40


def _mode(values: np.ndarray) -> int:
    vals, counts = np.unique(values, return_counts=True)
    return int(vals[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _step(pos: tuple[int, int], direction: int) -> tuple[int, int]:
    dr, dc = DIRS[direction]
    return pos[0] + dr, pos[1] + dc


def _rel_label(prev_dir: int, next_dir: int) -> str:
    delta = (next_dir - prev_dir) % 4
    if delta == 1:
        return "R"
    if delta == 0:
        return "S"
    if delta == 3:
        return "L"
    raise ValueError("reverse moves are not valid path steps")


def _boundary_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _corridor_color(grid: np.ndarray, bg: int) -> int:
    boundary = _boundary_mask(grid.shape)
    interior = grid[(grid != bg) & (~boundary)]
    if interior.size:
        return _mode(interior)
    non_bg = grid[grid != bg]
    if non_bg.size:
        return _mode(non_bg)
    return bg


def _entry_direction(walkable: np.ndarray, start: tuple[int, int]) -> int:
    h, w = walkable.shape
    r, c = start
    preferred: list[int] = []
    if r == 0:
        preferred.append(2)
    if c == w - 1:
        preferred.append(3)
    if r == h - 1:
        preferred.append(0)
    if c == 0:
        preferred.append(1)
    for direction in preferred + [0, 1, 2, 3]:
        nr, nc = _step(start, direction)
        if _in_bounds(walkable.shape, nr, nc) and bool(walkable[nr, nc]):
            return direction
    raise ValueError("entry has no inward walkable neighbor")


def _trace_path(
    walkable: np.ndarray, start: tuple[int, int], initial_dir: int
) -> tuple[list[tuple[int, int]], list[str]]:
    pos = start
    prev: tuple[int, int] | None = None
    heading = initial_dir
    path = [start]
    labels: list[str] = []
    seen_states: set[tuple[int, int, int, int, int]] = set()

    while True:
        prev_r = -1 if prev is None else prev[0]
        prev_c = -1 if prev is None else prev[1]
        state = (pos[0], pos[1], heading, prev_r, prev_c)
        if state in seen_states:
            break
        seen_states.add(state)

        next_dir = None
        next_label = None
        for rel in REL_PRIORITY:
            direction = (heading + rel) % 4
            nr, nc = _step(pos, direction)
            if not _in_bounds(walkable.shape, nr, nc):
                continue
            if not bool(walkable[nr, nc]):
                continue
            if prev is not None and (nr, nc) == prev:
                continue
            next_dir = direction
            next_label = REL_TO_LABEL[rel]
            break
        if next_dir is None:
            break
        prev = pos
        pos = _step(pos, next_dir)
        heading = next_dir
        path.append(pos)
        labels.append(next_label)

    return path, labels


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    bg = int(Counter(corners).most_common(1)[0][0])
    corridor = _corridor_color(g, bg)
    walkable = g != bg
    boundary = _boundary_mask(g.shape)

    entries = []
    for r, c in np.argwhere(boundary & walkable & (g != corridor)):
        entries.append((int(r), int(c), int(g[r, c])))
    entries.sort()

    out = g.copy()
    for r, c, color in entries:
        direction = _entry_direction(walkable, (r, c))
        path, _labels = _trace_path(walkable, (r, c), direction)
        for pr, pc in path:
            out[pr, pc] = color

    return out


def _sector_bounds(name: str, mh: int, mw: int) -> tuple[tuple[int, int], tuple[int, int]]:
    cr = mh // 2
    cc = mw // 2
    if name == "TL":
        return (0, cr - 1), (0, cc - 1)
    if name == "TR":
        return (0, cr - 1), (cc + 1, mw - 1)
    if name == "BL":
        return (cr + 1, mh - 1), (0, cc - 1)
    if name == "BR":
        return (cr + 1, mh - 1), (cc + 1, mw - 1)
    raise ValueError(name)


def _sector_side(name: str, rng: random.Random) -> str:
    if name == "TL":
        return rng.choice(["top", "left"])
    if name == "TR":
        return rng.choice(["top", "right"])
    if name == "BL":
        return rng.choice(["bottom", "left"])
    if name == "BR":
        return rng.choice(["bottom", "right"])
    raise ValueError(name)


def _start_node(
    name: str,
    side: str,
    bounds: tuple[tuple[int, int], tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int]:
    (r0, r1), (c0, c1) = bounds
    if side == "top":
        return r0, rng.randint(c0, c1)
    if side == "bottom":
        return r1, rng.randint(c0, c1)
    if side == "left":
        return rng.randint(r0, r1), c0
    if side == "right":
        return rng.randint(r0, r1), c1
    raise ValueError(side)


def _zone_nodes(bounds: tuple[tuple[int, int], tuple[int, int]]) -> set[tuple[int, int]]:
    (r0, r1), (c0, c1) = bounds
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _free_degree(
    node: tuple[int, int],
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    visited: set[tuple[int, int]],
) -> int:
    count = 0
    for direction in range(4):
        nxt = _step(node, direction)
        if nxt in allowed and nxt not in blocked and nxt not in visited:
            count += 1
    return count


def _walk_path(
    start: tuple[int, int],
    initial_dir: int,
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    target_edges: int,
    required_type: str,
    rng: random.Random,
) -> tuple[list[tuple[int, int]], list[str]] | None:
    visited = {start}
    path = [start]
    labels: list[str] = []

    def dfs(current: tuple[int, int], prev_dir: int) -> bool:
        if len(path) - 1 == target_edges:
            return required_type in labels

        candidates = []
        dirs = list(range(4))
        rng.shuffle(dirs)
        for direction in dirs:
            nxt = _step(current, direction)
            if nxt not in allowed or nxt in blocked or nxt in visited:
                continue
            score = _free_degree(nxt, allowed, blocked, visited)
            inward_bias = abs(nxt[0] - start[0]) + abs(nxt[1] - start[1])
            candidates.append((score, inward_bias, direction))
        candidates.sort(key=lambda item: (-item[0], -item[1]))

        remaining = target_edges - (len(path) - 1)
        need_type = required_type not in labels

        for _score, _bias, direction in candidates:
            rel = _rel_label(prev_dir, direction)
            if need_type and remaining == 1 and rel != required_type:
                continue
            nxt = _step(current, direction)
            visited.add(nxt)
            path.append(nxt)
            labels.append(rel)
            if dfs(nxt, direction):
                return True
            labels.pop()
            path.pop()
            visited.remove(nxt)
        return False

    if dfs(start, initial_dir):
        return path, labels
    return None


def _safe_branch_dirs(prev_dir: int, out_dir: int) -> list[int]:
    priority = [((prev_dir + 1) % 4), prev_dir, ((prev_dir - 1) % 4)]
    out_index = priority.index(out_dir)
    return priority[out_index + 1 :]


def _walk_branch(
    start: tuple[int, int],
    first_dir: int,
    allowed: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    first = _step(start, first_dir)
    if first not in allowed or first in blocked:
        return None

    target_edges = rng.randint(1, 4)
    visited = {start, first}
    branch = [first]

    def dfs(current: tuple[int, int], remaining: int) -> bool:
        if remaining == 0:
            return True
        dirs = list(range(4))
        rng.shuffle(dirs)
        candidates = []
        for direction in dirs:
            nxt = _step(current, direction)
            if nxt not in allowed or nxt in blocked or nxt in visited:
                continue
            score = _free_degree(nxt, allowed, blocked, visited)
            candidates.append((score, direction))
        candidates.sort(key=lambda item: -item[0])
        for _score, direction in candidates:
            nxt = _step(current, direction)
            visited.add(nxt)
            branch.append(nxt)
            if dfs(nxt, remaining - 1):
                return True
            branch.pop()
            visited.remove(nxt)
        return False

    if dfs(first, target_edges - 1):
        return branch
    return None


def _add_edge(
    graph: dict[tuple[int, int], set[tuple[int, int]]],
    a: tuple[int, int],
    b: tuple[int, int],
) -> None:
    graph.setdefault(a, set()).add(b)
    graph.setdefault(b, set()).add(a)


def _add_chain(
    graph: dict[tuple[int, int], set[tuple[int, int]]], nodes: list[tuple[int, int]]
) -> None:
    for node in nodes:
        graph.setdefault(node, set())
    for a, b in zip(nodes, nodes[1:]):
        _add_edge(graph, a, b)


def _node_to_pixel(node: tuple[int, int]) -> tuple[int, int]:
    return 2 * node[0] + 1, 2 * node[1] + 1


def _entry_pixel(
    side: str, node: tuple[int, int], grid_shape: tuple[int, int]
) -> tuple[int, int]:
    nr, nc = _node_to_pixel(node)
    if side == "top":
        return 0, nc
    if side == "bottom":
        return grid_shape[0] - 1, nc
    if side == "left":
        return nr, 0
    if side == "right":
        return nr, grid_shape[1] - 1
    raise ValueError(side)


def _paint_graph(
    graph: dict[tuple[int, int], set[tuple[int, int]]], shape: tuple[int, int]
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    seen_edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for node, nbrs in graph.items():
        pr, pc = _node_to_pixel(node)
        grid[pr, pc] = CORRIDOR
        for nbr in nbrs:
            edge = tuple(sorted((node, nbr)))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            qr, qc = _node_to_pixel(nbr)
            grid[(pr + qr) // 2, (pc + qc) // 2] = CORRIDOR
    return grid


def _paint_actual_path(
    grid: np.ndarray,
    side: str,
    color: int,
    nodes: list[tuple[int, int]],
) -> None:
    er, ec = _entry_pixel(side, nodes[0], grid.shape)
    grid[er, ec] = color
    for i, node in enumerate(nodes):
        pr, pc = _node_to_pixel(node)
        grid[pr, pc] = color
        if i + 1 < len(nodes):
            qr, qc = _node_to_pixel(nodes[i + 1])
            grid[(pr + qr) // 2, (pc + qc) // 2] = color


def _path_pixels(side: str, nodes: list[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    pixels = {_entry_pixel(side, nodes[0], shape)}
    for i, node in enumerate(nodes):
        pixels.add(_node_to_pixel(node))
        if i + 1 < len(nodes):
            a = _node_to_pixel(node)
            b = _node_to_pixel(nodes[i + 1])
            pixels.add(((a[0] + b[0]) // 2, (a[1] + b[1]) // 2))
    return pixels


def _generate_sector_paths(
    mh: int, mw: int, sector_names: list[str], required_types: list[str], rng: random.Random
) -> tuple[
    dict[tuple[int, int], set[tuple[int, int]]],
    list[dict[str, object]],
    bool,
] | None:
    graph: dict[tuple[int, int], set[tuple[int, int]]] = {}
    occupied: set[tuple[int, int]] = set()
    metas: list[dict[str, object]] = []
    any_branch = False

    for sector_name, required_type in zip(sector_names, required_types):
        bounds = _sector_bounds(sector_name, mh, mw)
        allowed = _zone_nodes(bounds)
        side = _sector_side(sector_name, rng)
        start = _start_node(sector_name, side, bounds, rng)
        initial_dir = SIDE_TO_DIR[side]

        zone_area = len(allowed)
        max_edges = min(8, max(4, zone_area - 3))
        min_edges = min(4, max_edges)
        path_result = None

        for _ in range(PATH_ATTEMPTS):
            target_edges = rng.randint(min_edges, max_edges)
            path_result = _walk_path(
                start=start,
                initial_dir=initial_dir,
                allowed=allowed,
                blocked=occupied,
                target_edges=target_edges,
                required_type=required_type,
                rng=rng,
            )
            if path_result is not None:
                break
        if path_result is None:
            return None

        nodes, labels = path_result
        _add_chain(graph, nodes)
        occupied.update(nodes)

        target_branch_count = rng.randint(1, min(3, max(1, len(nodes) - 1)))
        candidate_infos = []
        for i in range(len(nodes) - 1):
            prev_dir = initial_dir if i == 0 else next(
                d for d in range(4) if _step(nodes[i - 1], d) == nodes[i]
            )
            out_dir = next(d for d in range(4) if _step(nodes[i], d) == nodes[i + 1])
            safe_dirs = _safe_branch_dirs(prev_dir, out_dir)
            if safe_dirs:
                candidate_infos.append((labels[i], i, prev_dir, out_dir, safe_dirs))

        rng.shuffle(candidate_infos)
        candidate_infos.sort(key=lambda item: (item[0] != required_type, rng.random()))

        added_here = 0
        for _label, idx, prev_dir, out_dir, safe_dirs in candidate_infos:
            if added_here >= target_branch_count:
                break
            dirs = safe_dirs[:]
            rng.shuffle(dirs)
            for first_dir in dirs:
                branch = None
                for _ in range(BRANCH_ATTEMPTS):
                    branch = _walk_branch(
                        start=nodes[idx],
                        first_dir=first_dir,
                        allowed=allowed,
                        blocked=occupied,
                        rng=rng,
                    )
                    if branch is not None:
                        break
                if branch is None:
                    continue
                _add_chain(graph, [nodes[idx]] + branch)
                occupied.update(branch)
                added_here += 1
                any_branch = True
                break

        metas.append(
            {
                "sector": sector_name,
                "side": side,
                "color": None,
                "nodes": nodes,
                "labels": labels,
                "required_type": required_type,
            }
        )

    return graph, metas, any_branch


def _validate_instance(inp: np.ndarray, expected: np.ndarray, metas: list[dict[str, object]]) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    bg = BG
    grey_remaining = int(np.count_nonzero((inp == CORRIDOR) & (expected == CORRIDOR)))
    if grey_remaining <= 0:
        return False

    boundary = _boundary_mask(inp.shape)
    entry_pixels = np.argwhere(boundary & (inp != bg) & (inp != CORRIDOR))
    if entry_pixels.shape[0] != len(metas):
        return False

    seen: set[tuple[int, int]] = set()
    for meta in metas:
        pixels = _path_pixels(str(meta["side"]), list(meta["nodes"]), inp.shape)
        if seen & pixels:
            return False
        seen |= pixels

    labels = [label for meta in metas for label in list(meta["labels"])]
    return {"R", "S", "L"} & set(labels) == set(labels) or bool(labels)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    mh = rng.randint(8, 12)
    mw = rng.randint(8, 12)
    h, w = 2 * mh + 1, 2 * mw + 1

    sectors = ["TL", "TR", "BL", "BR"]
    rng.shuffle(sectors)
    count = rng.randint(2, 4)
    sector_names = sectors[:count]

    req_pool = ["R", "S", "L"]
    rng.shuffle(req_pool)
    if count == 2:
        required_types = req_pool[:2]
    elif count == 3:
        required_types = req_pool[:]
    else:
        required_types = req_pool + [rng.choice(req_pool)]
    rng.shuffle(required_types)

    result = _generate_sector_paths(mh, mw, sector_names, required_types, rng)
    if result is None:
        return None
    graph, metas, any_branch = result
    if not any_branch:
        return None

    base = _paint_graph(graph, (h, w))
    colors = rng.sample(ENTRY_COLORS, len(metas))
    for meta, color in zip(metas, colors):
        meta["color"] = color
        er, ec = _entry_pixel(str(meta["side"]), list(meta["nodes"])[0], base.shape)
        base[er, ec] = color

    expected = base.copy()
    for meta in metas:
        _paint_actual_path(
            expected,
            side=str(meta["side"]),
            color=int(meta["color"]),
            nodes=list(meta["nodes"]),
        )

    # Require the intended rule branches to appear in the chosen paths.
    labels_by_type = Counter(label for meta in metas for label in list(meta["labels"]))
    if len(labels_by_type) < 2:
        return None
    if count >= 3 and any(labels_by_type[t] == 0 for t in ("R", "S", "L")):
        return None

    if not _validate_instance(base, expected, metas):
        return None
    return base


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
