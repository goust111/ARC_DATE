from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bfs_dist(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque([start])
    dist[start] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] == -1:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _find_legend(g: np.ndarray) -> tuple[int, int, int, np.ndarray]:
    h, w = g.shape
    corners = [
        (0, 0),
        (0, w - 2),
        (h - 2, 0),
        (h - 2, w - 2),
    ]
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for r0, c0 in corners:
        if r0 < 0 or c0 < 0:
            continue
        block = g[r0 : r0 + 2, c0 : c0 + 2]
        if block.shape != (2, 2):
            continue
        pts = [tuple(map(int, rc)) for rc in np.argwhere(block != BLACK)]
        if len(pts) != 3:
            continue
        colors = {int(block[r, c]) for r, c in pts}
        if len(colors) != 3:
            continue
        deg = {}
        for r, c in pts:
            deg[(r, c)] = sum(abs(r - rr) + abs(c - cc) == 1 for rr, cc in pts if (rr, cc) != (r, c))
        elbows = [p for p, d in deg.items() if d == 2]
        ends = [p for p, d in deg.items() if d == 1]
        if len(elbows) != 1 or len(ends) != 2:
            continue
        mix = int(block[elbows[0]])
        color_a = int(block[ends[0]])
        color_b = int(block[ends[1]])
        mask = np.zeros_like(g, dtype=bool)
        mask[r0 : r0 + 2, c0 : c0 + 2] = block != BLACK
        candidates.append((color_a, color_b, mix, mask))
    assert len(candidates) == 1, "expected exactly one corner legend"
    return candidates[0]


def _render_output(
    shape: tuple[int, int],
    skeleton_mask: np.ndarray,
    src_a: tuple[int, int],
    src_b: tuple[int, int],
    color_a: int,
    color_b: int,
    mix: int,
) -> np.ndarray:
    dist_a = _bfs_dist(skeleton_mask, src_a)
    dist_b = _bfs_dist(skeleton_mask, src_b)
    out = np.full(shape, BLACK, dtype=int)
    for r, c in map(tuple, np.argwhere(skeleton_mask)):
        rr, cc = int(r), int(c)
        if (rr, cc) == src_a:
            out[rr, cc] = color_a
        elif (rr, cc) == src_b:
            out[rr, cc] = color_b
        elif dist_a[rr, cc] == dist_b[rr, cc]:
            out[rr, cc] = mix
        elif dist_a[rr, cc] < dist_b[rr, cc]:
            out[rr, cc] = color_a
        else:
            out[rr, cc] = color_b
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    color_a, color_b, mix, legend_mask = _find_legend(g)
    skeleton_mask = (g != BLACK) & (~legend_mask)
    comps = _components(skeleton_mask)
    assert len(comps) == 1, "expected one skeleton component"

    for r in range(g.shape[0] - 1):
        for c in range(g.shape[1] - 1):
            assert not np.all(skeleton_mask[r : r + 2, c : c + 2]), "skeleton must stay 1-pixel wide"

    payload = {int(v) for v in np.unique(g[skeleton_mask]) if int(v) != GREY}
    assert payload == {color_a, color_b}, "skeleton should contain exactly the two injection colors"

    src_a_pts = np.argwhere((g == color_a) & skeleton_mask)
    src_b_pts = np.argwhere((g == color_b) & skeleton_mask)
    assert src_a_pts.shape == (1, 2) and src_b_pts.shape == (1, 2), "each injection color must occur once on the skeleton"
    src_a = tuple(map(int, src_a_pts[0]))
    src_b = tuple(map(int, src_b_pts[0]))
    return _render_output(g.shape, skeleton_mask, src_a, src_b, color_a, color_b, mix)


def _abstract_neighbors(node: tuple[int, int], gh: int, gw: int) -> list[tuple[int, int]]:
    r, c = node
    out = []
    for dr, dc in DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < gh and 0 <= nc < gw:
            out.append((nr, nc))
    return out


def _random_path(gh: int, gw: int, min_nodes: int, max_nodes: int) -> list[tuple[int, int]] | None:
    for _ in range(100):
        target = random.randint(min_nodes, min(max_nodes, gh * gw))
        path = [(random.randrange(gh), random.randrange(gw))]
        used = {path[0]}
        while len(path) < target:
            cur = path[-1]
            options = [nb for nb in _abstract_neighbors(cur, gh, gw) if nb not in used]
            if not options:
                break
            nxt = random.choice(options)
            path.append(nxt)
            used.add(nxt)
        if len(path) >= min_nodes:
            return path
    return None


def _path_graph(gh: int, gw: int):
    path = _random_path(gh, gw, 4, 8)
    if path is None:
        return None
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    return set(path), edges


def _branch_graph(gh: int, gw: int):
    path = _random_path(gh, gw, 5, 8)
    if path is None or len(path) < 4:
        return None
    nodes = set(path)
    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
    roots = path[1:-1]
    random.shuffle(roots)
    target_branches = random.randint(1, min(2, len(roots)))
    added = 0
    for root in roots:
        first_options = [nb for nb in _abstract_neighbors(root, gh, gw) if nb not in nodes]
        if not first_options:
            continue
        cur = random.choice(first_options)
        branch_len = random.randint(1, 3)
        branch = [root, cur]
        used = set(nodes)
        used.add(cur)
        for _ in range(branch_len - 1):
            options = [nb for nb in _abstract_neighbors(cur, gh, gw) if nb not in used]
            if not options:
                break
            cur = random.choice(options)
            branch.append(cur)
            used.add(cur)
        for a, b in zip(branch, branch[1:], strict=False):
            edges.append((a, b))
            nodes.add(b)
        added += 1
        if added >= target_branches:
            break
    if added == 0:
        return None
    return nodes, edges


def _cross_graph(gh: int, gw: int):
    if gh < 4 or gw < 4:
        return None
    for _ in range(100):
        center = (random.randrange(1, gh - 1), random.randrange(1, gw - 1))
        dirs = DIRS4[:]
        random.shuffle(dirs)
        arm_dirs = dirs[: random.choice([3, 4])]
        nodes = {center}
        edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
        ok = True
        leaves: list[tuple[int, int]] = []
        for dr, dc in arm_dirs:
            limit = 0
            r, c = center
            while 0 <= r + dr * (limit + 1) < gh and 0 <= c + dc * (limit + 1) < gw and limit < 3:
                limit += 1
            if limit == 0:
                ok = False
                break
            arm_len = random.randint(1, limit)
            prev = center
            for step in range(1, arm_len + 1):
                node = (center[0] + dr * step, center[1] + dc * step)
                edges.append((prev, node))
                nodes.add(node)
                prev = node
            leaves.append(prev)
        if not ok:
            continue
        if random.random() < 0.5 and leaves:
            leaf = random.choice(leaves)
            options = [nb for nb in _abstract_neighbors(leaf, gh, gw) if nb not in nodes]
            if options:
                extra = random.choice(options)
                edges.append((leaf, extra))
                nodes.add(extra)
        return nodes, edges
    return None


def _adjacency(nodes: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]):
    adj = {node: set() for node in nodes}
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    return adj


def _embed_graph(nodes: set[tuple[int, int]], edges: list[tuple[tuple[int, int], tuple[int, int]]]):
    rows = [r for r, _ in nodes]
    cols = [c for _, c in nodes]
    rmin, rmax = min(rows), max(rows)
    cmin, cmax = min(cols), max(cols)
    node_pixels = {node: (2 * (node[0] - rmin), 2 * (node[1] - cmin)) for node in nodes}
    pixels = set(node_pixels.values())
    for a, b in edges:
        ra, ca = node_pixels[a]
        rb, cb = node_pixels[b]
        pixels.add(((ra + rb) // 2, (ca + cb) // 2))
    height = 2 * (rmax - rmin) + 1
    width = 2 * (cmax - cmin) + 1
    return pixels, node_pixels, height, width


def _choose_sources(adj: dict[tuple[int, int], set[tuple[int, int]]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    dist_map: dict[tuple[tuple[int, int], tuple[int, int]], int] = {}
    nodes = list(adj)
    for start in nodes:
        q = deque([start])
        dist = {start: 0}
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in dist:
                    dist[nb] = dist[cur] + 1
                    q.append(nb)
        for end, d in dist.items():
            if start < end:
                dist_map[(start, end)] = d

    leaves = [node for node, nbs in adj.items() if len(nbs) == 1]
    if random.random() < 0.5 and len(leaves) >= 2:
        pool = [(a, b, d) for (a, b), d in dist_map.items() if a in leaves and b in leaves and d >= 2]
    else:
        pool = [(a, b, d) for (a, b), d in dist_map.items() if d >= 2]
    if not pool:
        pool = [(a, b, d) for (a, b), d in dist_map.items() if d >= 1]
    if not pool:
        return None
    if random.random() < 0.5:
        best = max(d for _, _, d in pool)
        pool = [item for item in pool if item[2] == best]
    a, b, _ = random.choice(pool)
    return a, b


def _place_legend(g: np.ndarray, color_a: int, color_b: int, mix: int) -> None:
    block = np.zeros((2, 2), dtype=int)
    missing = random.choice([(0, 0), (0, 1), (1, 0), (1, 1)])
    pts = [(r, c) for r in range(2) for c in range(2) if (r, c) != missing]
    deg = {}
    for r, c in pts:
        deg[(r, c)] = sum(abs(r - rr) + abs(c - cc) == 1 for rr, cc in pts if (rr, cc) != (r, c))
    elbow = next(p for p, d in deg.items() if d == 2)
    ends = [p for p in pts if p != elbow]
    random.shuffle(ends)
    block[elbow] = mix
    block[ends[0]] = color_a
    block[ends[1]] = color_b

    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        g[0:2, 0:2] = block
    elif corner == "tr":
        g[0:2, -2:] = block
    elif corner == "bl":
        g[-2:, 0:2] = block
    else:
        g[-2:, -2:] = block


def _gen_attempt() -> np.ndarray | None:
    mode = random.choice(["path", "branch", "cross"])
    gh = random.randint(4, 7)
    gw = random.randint(4, 7)
    if mode == "path":
        graph = _path_graph(gh, gw)
    elif mode == "branch":
        graph = _branch_graph(gh, gw)
    else:
        graph = _cross_graph(max(gh, 4), max(gw, 4))
    if graph is None:
        return None

    nodes, edges = graph
    adj = _adjacency(nodes, edges)
    chosen = _choose_sources(adj)
    if chosen is None:
        return None
    src_node_a, src_node_b = chosen

    pixels, node_pixels, ph, pw = _embed_graph(nodes, edges)
    top = random.randint(2, 5)
    left = random.randint(2, 5)
    bottom = random.randint(2, 5)
    right = random.randint(2, 5)
    h = ph + top + bottom
    w = pw + left + right
    g = np.full((h, w), BLACK, dtype=int)

    skeleton_mask = np.zeros((h, w), dtype=bool)
    for r, c in pixels:
        skeleton_mask[top + r, left + c] = True
        g[top + r, left + c] = GREY

    color_a, color_b, mix = random.sample(PALETTE, 3)
    src_a = (top + node_pixels[src_node_a][0], left + node_pixels[src_node_a][1])
    src_b = (top + node_pixels[src_node_b][0], left + node_pixels[src_node_b][1])
    g[src_a] = color_a
    g[src_b] = color_b
    _place_legend(g, color_a, color_b, mix)

    expected = _render_output(g.shape, skeleton_mask, src_a, src_b, color_a, color_b, mix)
    if np.count_nonzero(expected == mix) < 1:
        return None
    if np.count_nonzero(expected == color_a) < 2 or np.count_nonzero(expected == color_b) < 2:
        return None
    if len(_components(skeleton_mask)) != 1:
        return None
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(skeleton_mask[r : r + 2, c : c + 2]):
                return None
    try:
        solved = main(g)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return g


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
