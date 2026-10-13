from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_PAD = 3
GEN_COARSE_MIN = 5
GEN_COARSE_MAX = 12
GEN_ATTEMPTS = 500
GEN_MIN_EDGE_DISTANCE = 5
GEN_MIN_TURNS = 3
GEN_MIN_ASSIGNED = 10
GEN_MIN_TIES = 2


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int):
    r = int(r)
    c = int(c)
    for dr, dc in DIR4:
        yield r + dr, c + dc


def bfs_distances(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    dist = np.full(passable.shape, -1, dtype=int)
    sr, sc = map(int, start)
    dist[sr, sc] = 0
    q: deque[tuple[int, int]] = deque([(sr, sc)])
    while q:
        r, c = q.popleft()
        step = int(dist[r, c]) + 1
        for nr, nc in neighbors4(r, c):
            if not in_bounds(passable.shape, nr, nc):
                continue
            if dist[nr, nc] != -1 or not bool(passable[nr, nc]):
                continue
            dist[nr, nc] = step
            q.append((nr, nc))
    return dist


def _parse_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    counts = Counter(map(int, g.flat))
    source_colors = sorted(color for color, n in counts.items() if int(n) == 1)
    assert len(source_colors) == 2, "expected exactly two singleton source colors"

    sources = []
    for color in source_colors:
        coords = np.argwhere(g == int(color))
        assert coords.shape == (1, 2)
        sources.append(
            {
                "color": int(color),
                "coord": (int(coords[0, 0]), int(coords[0, 1])),
            }
        )

    base_colors = [int(color) for color in counts if int(color) not in source_colors]
    assert len(base_colors) == 2, "expected floor and wall colors besides the two sources"

    floor_candidates: list[tuple[int, np.ndarray]] = []
    for floor_color in base_colors:
        passable = g == int(floor_color)
        if not bool(passable.any()):
            continue
        if not all(
            any(in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == int(floor_color) for nr, nc in neighbors4(*src["coord"]))
            for src in sources
        ):
            continue
        reachable = np.zeros(g.shape, dtype=bool)
        for src in sources:
            dist = bfs_distances(passable, src["coord"])
            reachable |= dist >= 0
        if int(np.count_nonzero(reachable & passable)) == int(np.count_nonzero(passable)):
            floor_candidates.append((int(floor_color), passable))

    assert len(floor_candidates) == 1, "floor color must be uniquely determined by reachability"
    floor_color, passable = floor_candidates[0]
    wall_color = int(next(color for color in base_colors if int(color) != int(floor_color)))

    return {
        "grid": g,
        "floor_color": int(floor_color),
        "wall_color": int(wall_color),
        "sources": sources,
        "passable": passable,
    }


def _expected_output(scene: dict) -> np.ndarray:
    g = scene["grid"]
    out = g.copy()
    floor_mask = scene["passable"]
    dist0 = bfs_distances(floor_mask, scene["sources"][0]["coord"])
    dist1 = bfs_distances(floor_mask, scene["sources"][1]["coord"])

    nearer0 = floor_mask & (dist0 >= 0) & ((dist1 < 0) | (dist0 < dist1))
    nearer1 = floor_mask & (dist1 >= 0) & ((dist0 < 0) | (dist1 < dist0))
    out[nearer0] = int(scene["sources"][0]["color"])
    out[nearer1] = int(scene["sources"][1]["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Partition every floor cell by true around-the-wall distance to the nearer source, leaving ties as boundary floor.

    Input structure:
    - The grid uses exactly four colors: one floor color, one wall color, and two distinct one-pixel source colors.
    - Walls are 4-neighbor barriers that bend the walkable corridors; every floor cell is connected to at least one source.

    Rule:
    - Measure 4-neighbor shortest-path distance from each source while walking only on floor cells and never through walls.
    - Recolor each floor cell with the color of the strictly nearer source.
    - If both sources are equally near, keep that cell as floor; keep all wall cells and both source pixels unchanged.

    Key insight:
    - Ownership is decided by geodesic distance through the corridor system, not by straight-line proximity, so walls can pull territory deep around bends.
    """
    scene = _parse_scene(grid)
    return _expected_output(scene)


def _coarse_neighbors(node: tuple[int, int], h: int, w: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIR4:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((int(nr), int(nc)))
    return out


def _add_edge(tree: dict[tuple[int, int], set[tuple[int, int]]], a: tuple[int, int], b: tuple[int, int]) -> None:
    tree.setdefault(a, set()).add(b)
    tree.setdefault(b, set()).add(a)


def _dfs_simple_path(
    h: int,
    w: int,
    target_len: int,
    path: list[tuple[int, int]],
    used: set[tuple[int, int]],
) -> bool:
    if len(path) == target_len:
        return True

    cur = path[-1]
    choices = [nb for nb in _coarse_neighbors(cur, h, w) if nb not in used]
    random.shuffle(choices)
    choices.sort(
        key=lambda nb: (
            sum(1 for nn in _coarse_neighbors(nb, h, w) if nn not in used),
            random.random(),
        ),
        reverse=True,
    )
    for nxt in choices:
        used.add(nxt)
        path.append(nxt)
        if _dfs_simple_path(h, w, target_len, path, used):
            return True
        path.pop()
        used.remove(nxt)
    return False


def _simple_path(h: int, w: int, target_len: int) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(h) for c in range(w)]
    random.shuffle(starts)
    for start in starts[: min(len(starts), 30)]:
        path = [start]
        used = {start}
        if _dfs_simple_path(h, w, target_len, path, used):
            return path
    return None


def _grow_branch(
    tree: dict[tuple[int, int], set[tuple[int, int]]],
    used: set[tuple[int, int]],
    h: int,
    w: int,
    anchor: tuple[int, int],
    *,
    first_step: tuple[int, int] | None = None,
    min_len: int = 1,
    max_len: int = 3,
) -> bool:
    cur = anchor
    new_nodes: list[tuple[int, int]] = []
    target = random.randint(min_len, max_len)
    for step in range(target):
        if step == 0 and first_step is not None:
            choices = [first_step] if first_step not in used and first_step in _coarse_neighbors(cur, h, w) else []
        else:
            choices = [nb for nb in _coarse_neighbors(cur, h, w) if nb not in used]
        if not choices:
            break
        nxt = random.choice(choices)
        _add_edge(tree, cur, nxt)
        used.add(nxt)
        new_nodes.append(nxt)
        cur = nxt
    if len(new_nodes) < min_len:
        for node in reversed(new_nodes):
            parent = next(iter(tree[node]))
            tree[parent].remove(node)
            tree[node].remove(parent)
            if not tree[node]:
                tree.pop(node, None)
            used.remove(node)
        return False
    return True


def _build_sparse_tree(
    h: int,
    w: int,
    expect_ties: bool,
) -> tuple[dict[tuple[int, int], set[tuple[int, int]]], list[tuple[int, int]]] | None:
    max_nodes = min(max(9, (h * w) // 3), 18)
    min_nodes = min(8, max_nodes)
    target_nodes = random.randint(min_nodes, max_nodes)
    if expect_ties and (target_nodes - 1) % 2 == 1:
        target_nodes += 1
    if target_nodes > h * w:
        return None

    main_path = _simple_path(h, w, target_nodes)
    if main_path is None or _path_turns(main_path) < GEN_MIN_TURNS:
        return None
    if expect_ties and (len(main_path) - 1) % 2 == 1:
        return None

    tree: dict[tuple[int, int], set[tuple[int, int]]] = {}
    used = {main_path[0]}
    tree[main_path[0]] = set()
    for a, b in zip(main_path, main_path[1:]):
        _add_edge(tree, a, b)
        used.add(b)

    if expect_ties:
        mid = main_path[(len(main_path) - 1) // 2]
        mid_choices = [nb for nb in _coarse_neighbors(mid, h, w) if nb not in used]
        if not mid_choices:
            return None
        random.shuffle(mid_choices)
        if not any(_grow_branch(tree, used, h, w, mid, first_step=choice, min_len=1, max_len=3) for choice in mid_choices):
            return None

    extra_branches = random.randint(2, 5)
    anchors = main_path[1:-1]
    if not anchors:
        return None
    for _ in range(extra_branches):
        random.shuffle(anchors)
        for anchor in anchors:
            if _grow_branch(tree, used, h, w, anchor, min_len=1, max_len=3):
                break

    return tree, main_path


def _node_coord(node: tuple[int, int]) -> tuple[int, int]:
    return GEN_PAD + 2 * int(node[0]), GEN_PAD + 2 * int(node[1])


def _render_tree(
    tree: dict[tuple[int, int], set[tuple[int, int]]],
    h: int,
    w: int,
    floor_color: int,
    wall_color: int,
) -> np.ndarray:
    grid_h = 2 * GEN_PAD + 2 * h - 1
    grid_w = 2 * GEN_PAD + 2 * w - 1
    grid = np.full((grid_h, grid_w), int(wall_color), dtype=int)
    for node, neighbors in tree.items():
        r, c = _node_coord(node)
        grid[r, c] = int(floor_color)
        for nb in neighbors:
            if nb < node:
                continue
            nr, nc = _node_coord(nb)
            grid[(r + nr) // 2, (c + nc) // 2] = int(floor_color)
    return grid


def _tree_path(
    tree: dict[tuple[int, int], set[tuple[int, int]]],
    start: tuple[int, int],
    end: tuple[int, int],
) -> list[tuple[int, int]]:
    parent = {start: None}
    q: deque[tuple[int, int]] = deque([start])
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for nb in tree[cur]:
            if nb in parent:
                continue
            parent[nb] = cur
            q.append(nb)
    assert end in parent
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path


def _path_turns(path: list[tuple[int, int]]) -> int:
    turns = 0
    for i in range(1, len(path) - 1):
        prev_step = (int(path[i][0] - path[i - 1][0]), int(path[i][1] - path[i - 1][1]))
        next_step = (int(path[i + 1][0] - path[i][0]), int(path[i + 1][1] - path[i][1]))
        if prev_step != next_step:
            turns += 1
    return turns


def _pocket_cells(leaf_rc: tuple[int, int], outward: tuple[int, int], depth: int) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    lr, lc = map(int, leaf_rc)
    or_, oc = map(int, outward)
    source = (lr + depth * or_, lc + depth * oc)
    cells = {source}
    for dr, dc in DIR4:
        cells.add((source[0] + dr, source[1] + dc))
    if depth == 2:
        cells.add((source[0] - 2 * or_, source[1] - 2 * oc))
    return source, cells


def _attach_source(
    grid: np.ndarray,
    leaf_spec: dict,
    depth: int,
    source_color: int,
    floor_color: int,
    wall_color: int,
) -> tuple[int, int] | None:
    leaf = _node_coord(leaf_spec["node"])
    neighbor = _node_coord(leaf_spec["neighbor"])
    source, cells = _pocket_cells(leaf, leaf_spec["outward"], depth)
    if not all(in_bounds(grid.shape, r, c) for r, c in cells):
        return None

    connector = ((leaf[0] + neighbor[0]) // 2, (leaf[1] + neighbor[1]) // 2)
    for r, c in cells:
        value = int(grid[r, c])
        if (r, c) == leaf:
            if value != int(floor_color):
                return None
        elif value != int(wall_color):
            return None

    for r, c in cells:
        for nr, nc in neighbors4(r, c):
            if not in_bounds(grid.shape, nr, nc):
                continue
            if int(grid[nr, nc]) not in {int(floor_color), int(wall_color)} and (nr, nc) not in cells:
                return None
            if int(grid[nr, nc]) == int(floor_color) and (nr, nc) not in cells:
                if (r, c) == leaf and (nr, nc) == connector:
                    continue
                return None

    for r, c in cells:
        grid[r, c] = int(floor_color)
    grid[source] = int(source_color)
    return source


def _bfs_with_target(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> np.ndarray:
    walkable = passable.copy()
    walkable[start] = True
    walkable[end] = True
    return bfs_distances(walkable, start)


def _extract_unique_path(passable: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]] | None:
    dist_start = _bfs_with_target(passable, start, end)
    total = int(dist_start[end])
    if total <= 0:
        return None
    dist_end = _bfs_with_target(passable, end, start)

    path = [start]
    cur = start
    while cur != end:
        r, c = cur
        candidates = []
        for nr, nc in neighbors4(r, c):
            if not in_bounds(passable.shape, nr, nc):
                continue
            if dist_start[nr, nc] != dist_start[r, c] + 1:
                continue
            if dist_start[nr, nc] + dist_end[nr, nc] != total:
                continue
            if (nr, nc) != end and not bool(passable[nr, nc]):
                continue
            candidates.append((int(nr), int(nc)))
        if len(candidates) != 1:
            return None
        cur = candidates[0]
        path.append(cur)
    return path


def _ownership_signature(scene: dict, dist0: np.ndarray, dist1: np.ndarray) -> np.ndarray:
    floor = scene["passable"]
    sig = np.full(scene["grid"].shape, -1, dtype=int)
    sig[floor & (dist0 == dist1)] = 2
    sig[floor & (dist0 >= 0) & ((dist1 < 0) | (dist0 < dist1))] = 0
    sig[floor & (dist1 >= 0) & ((dist0 < 0) | (dist1 < dist0))] = 1
    return sig


def _semantic_ok(grid: np.ndarray, expect_ties: bool) -> bool:
    try:
        scene = _parse_scene(grid)
    except AssertionError:
        return False

    floor = scene["passable"]
    for src in scene["sources"]:
        if sum(1 for nr, nc in neighbors4(*src["coord"]) if in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == int(scene["floor_color"])) != 4:
            return False

    dist0 = bfs_distances(floor, scene["sources"][0]["coord"])
    dist1 = bfs_distances(floor, scene["sources"][1]["coord"])
    if not np.all((dist0 >= 0) | (dist1 >= 0) | ~floor):
        return False

    expected = _expected_output(scene)
    geo_sig = _ownership_signature(scene, dist0, dist1)
    tie_count = int(np.count_nonzero(geo_sig == 2))
    if expect_ties:
        if tie_count < GEN_MIN_TIES:
            return False
    elif tie_count != 0:
        return False

    assigned0 = int(np.count_nonzero(floor & (expected == int(scene["sources"][0]["color"]))))
    assigned1 = int(np.count_nonzero(floor & (expected == int(scene["sources"][1]["color"]))))
    if min(assigned0, assigned1) < GEN_MIN_ASSIGNED:
        return False

    path = _extract_unique_path(floor, scene["sources"][0]["coord"], scene["sources"][1]["coord"])
    if path is None or _path_turns(path) < GEN_MIN_TURNS:
        return False

    rr, cc = np.indices(grid.shape)
    src0 = scene["sources"][0]["coord"]
    src1 = scene["sources"][1]["coord"]
    md0 = np.abs(rr - int(src0[0])) + np.abs(cc - int(src0[1]))
    md1 = np.abs(rr - int(src1[0])) + np.abs(cc - int(src1[1]))
    man_sig = np.full(grid.shape, -1, dtype=int)
    man_sig[floor & (md0 == md1)] = 2
    man_sig[floor & (md0 < md1)] = 0
    man_sig[floor & (md1 < md0)] = 1
    if not bool(np.any(floor & (geo_sig != man_sig))):
        return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    coarse_h = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)
    coarse_w = random.randint(GEN_COARSE_MIN, GEN_COARSE_MAX)

    colors = random.sample(list(range(10)), 4)
    floor_color, wall_color, source_a, source_b = map(int, colors)

    expect_ties = random.random() < 0.5
    built = _build_sparse_tree(coarse_h, coarse_w, expect_ties)
    if built is None:
        return None
    tree, main_path = built
    if len(main_path) - 1 < GEN_MIN_EDGE_DISTANCE:
        return None

    left = {
        "node": main_path[0],
        "neighbor": main_path[1],
        "outward": (int(main_path[0][0] - main_path[1][0]), int(main_path[0][1] - main_path[1][1])),
    }
    right = {
        "node": main_path[-1],
        "neighbor": main_path[-2],
        "outward": (int(main_path[-1][0] - main_path[-2][0]), int(main_path[-1][1] - main_path[-2][1])),
    }

    base = _render_tree(tree, coarse_h, coarse_w, floor_color, wall_color)
    offset_patterns = [(1, 1), (2, 2)] if expect_ties else [(1, 2), (2, 1)]
    for depth_a, depth_b in random.sample(offset_patterns, k=len(offset_patterns)):
        grid = base.copy()
        src_a = _attach_source(grid, left, depth_a, source_a, floor_color, wall_color)
        if src_a is None:
            continue
        src_b = _attach_source(grid, right, depth_b, source_b, floor_color, wall_color)
        if src_b is None or src_a == src_b:
            continue
        if _semantic_ok(grid, expect_ties):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mountain-boundary example")
