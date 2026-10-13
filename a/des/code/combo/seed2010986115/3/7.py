from __future__ import annotations

import numpy as np


BG = 0
MAGENTA = 6
ORANGE = 7
AZURE = 8
NODE_COLORS = (1, 2, 3, 4, 5, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Generator config.
MIN_NODES = 3
MAX_NODES = 6
LATTICE_MIN = 3
LATTICE_MAX = 5
CELL_STEP = 5
MARGIN_MIN = 2
MARGIN_MAX = 4
MAX_GENERATION_ATTEMPTS = 300


def _neighbors4(r: int, c: int, h: int, w: int):
    for dr, dc in DIR4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            yield rr, cc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _min_manhattan(pixel: tuple[int, int], points: list[tuple[int, int]]) -> int:
    pr, pc = pixel
    return min(abs(pr - r) + abs(pc - c) for r, c in points)


def _node_points(center: tuple[int, int], size: int) -> list[tuple[int, int]]:
    cr, cc = center
    r0 = cr - size // 2
    c0 = cc - size // 2
    return [(r, c) for r in range(r0, r0 + size) for c in range(c0, c0 + size)]


def _outside_port(center: tuple[int, int], size: int, direction: tuple[int, int]) -> tuple[int, int]:
    cr, cc = center
    dr, dc = direction
    if dr == 0 and dc == 1:
        return cr, cc + size - size // 2
    if dr == 0 and dc == -1:
        return cr, cc - size // 2 - 1
    if dr == 1 and dc == 0:
        return cr + size - size // 2, cc
    if dr == -1 and dc == 0:
        return cr - size // 2 - 1, cc
    raise AssertionError("invalid direction")


def _corridor_cells(
    center_a: tuple[int, int],
    size_a: int,
    center_b: tuple[int, int],
    size_b: int,
) -> list[tuple[int, int]]:
    ra, ca = center_a
    rb, cb = center_b
    if ra == rb:
        if cb > ca:
            start = _outside_port(center_a, size_a, (0, 1))
            end = _outside_port(center_b, size_b, (0, -1))
        else:
            start = _outside_port(center_a, size_a, (0, -1))
            end = _outside_port(center_b, size_b, (0, 1))
        assert start[0] == end[0]
        c0, c1 = sorted((start[1], end[1]))
        return [(start[0], c) for c in range(c0, c1 + 1)]

    assert ca == cb
    if rb > ra:
        start = _outside_port(center_a, size_a, (1, 0))
        end = _outside_port(center_b, size_b, (-1, 0))
    else:
        start = _outside_port(center_a, size_a, (-1, 0))
        end = _outside_port(center_b, size_b, (1, 0))
    assert start[1] == end[1]
    r0, r1 = sorted((start[0], end[0]))
    return [(r, start[1]) for r in range(r0, r1 + 1)]


def _extract_nodes(grid: np.ndarray) -> list[dict]:
    node_mask = (grid != BG) & (grid != ORANGE) & (grid != MAGENTA)
    components = _components(node_mask)
    nodes: list[dict] = []
    for comp in components:
        colors = {int(grid[r, c]) for r, c in comp}
        assert len(colors) == 1
        color = next(iter(colors))
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h == w and h in (2, 3)
        assert len(comp) == h * w
        nodes.append(
            {
                "color": color,
                "points": comp,
                "point_set": set(comp),
                "bbox": (r0, c0, r1, c1),
            }
        )
    assert len(nodes) >= 2
    return nodes


def _follow_orange_to_node(
    grid: np.ndarray,
    node_id_grid: np.ndarray,
    start: tuple[int, int],
    source_id: int,
) -> int:
    h, w = grid.shape
    prev: tuple[int, int] | None = None
    cur = start
    seen: set[tuple[int, int]] = set()

    while True:
        if cur in seen:
            raise AssertionError("orange path loops before reaching another node")
        seen.add(cur)

        hit_nodes = {
            int(node_id_grid[nr, nc])
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if node_id_grid[nr, nc] >= 0 and int(node_id_grid[nr, nc]) != source_id
        }
        if hit_nodes:
            assert len(hit_nodes) == 1
            return next(iter(hit_nodes))

        next_pixels = [
            (nr, nc)
            for nr, nc in _neighbors4(cur[0], cur[1], h, w)
            if int(grid[nr, nc]) == ORANGE and (nr, nc) != prev
        ]
        assert len(next_pixels) == 1
        prev, cur = cur, next_pixels[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow the orange-linked node chain, orient it using the magenta edge marker,
    keep odd-indexed nodes unchanged, recolor even-indexed nodes to azure, and
    remove the orange path plus the marker.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    marker_pixels = np.argwhere(g == MAGENTA)
    assert len(marker_pixels) == 1
    marker = (int(marker_pixels[0, 0]), int(marker_pixels[0, 1]))
    assert marker[0] in (0, h - 1) or marker[1] in (0, w - 1)

    nodes = _extract_nodes(g)
    node_id_grid = np.full((h, w), -1, dtype=int)
    for idx, node in enumerate(nodes):
        for r, c in node["points"]:
            node_id_grid[r, c] = idx

    adjacency = [set() for _ in nodes]
    for idx, node in enumerate(nodes):
        contacts = set()
        for r, c in node["points"]:
            for nr, nc in _neighbors4(r, c, h, w):
                if int(g[nr, nc]) == ORANGE:
                    contacts.add((nr, nc))
        assert contacts
        for contact in contacts:
            adjacency[idx].add(_follow_orange_to_node(g, node_id_grid, contact, idx))

    degrees = [len(neighbors) for neighbors in adjacency]
    assert all(degree in (1, 2) for degree in degrees)
    endpoints = [idx for idx, degree in enumerate(degrees) if degree == 1]
    assert len(endpoints) == 2

    dists = [_min_manhattan(marker, nodes[idx]["points"]) for idx in endpoints]
    assert dists[0] != dists[1]
    start = endpoints[0] if dists[0] < dists[1] else endpoints[1]

    order: list[int] = []
    prev = -1
    cur = start
    while True:
        order.append(cur)
        next_nodes = [nxt for nxt in adjacency[cur] if nxt != prev]
        if not next_nodes:
            break
        assert len(next_nodes) == 1
        prev, cur = cur, next_nodes[0]

    assert len(order) == len(nodes)

    out = np.full_like(g, BG)
    for pos, node_idx in enumerate(order, start=1):
        paint_color = nodes[node_idx]["color"] if pos % 2 == 1 else AZURE
        for r, c in nodes[node_idx]["points"]:
            out[r, c] = paint_color
    return out


def _sample_self_avoiding_path(
    rng: np.random.Generator,
    rows: int,
    cols: int,
    length: int,
) -> list[tuple[int, int]] | None:
    start = (int(rng.integers(rows)), int(rng.integers(cols)))
    path = [start]
    seen = {start}

    def dfs() -> bool:
        if len(path) == length:
            return True
        r, c = path[-1]
        candidates = []
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            nxt = (nr, nc)
            if 0 <= nr < rows and 0 <= nc < cols and nxt not in seen:
                freedom = 0
                for dr2, dc2 in DIR4:
                    rr = nr + dr2
                    cc = nc + dc2
                    if 0 <= rr < rows and 0 <= cc < cols and (rr, cc) not in seen:
                        freedom += 1
                candidates.append((freedom, nxt))
        if not candidates:
            return False
        order = rng.permutation(len(candidates))
        shuffled = [candidates[i] for i in order]
        shuffled.sort(key=lambda item: item[0], reverse=True)
        for _, nxt in shuffled:
            seen.add(nxt)
            path.append(nxt)
            if dfs():
                return True
            path.pop()
            seen.remove(nxt)
        return False

    if dfs():
        return path.copy()
    return None


def _marker_candidates(
    grid: np.ndarray,
    near_points: list[tuple[int, int]],
    far_points: list[tuple[int, int]],
) -> list[tuple[int, int, int, tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if int(grid[r, c]) != BG:
                continue
            if any(int(grid[nr, nc]) != BG for nr, nc in _neighbors4(r, c, h, w)):
                continue
            near_dist = _min_manhattan((r, c), near_points)
            far_dist = _min_manhattan((r, c), far_points)
            if near_dist < far_dist:
                candidates.append((near_dist, far_dist - near_dist, r + c, (r, c)))
    return candidates


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_nodes = int(rng.integers(MIN_NODES, MAX_NODES + 1))
    lattice_rows = int(rng.integers(LATTICE_MIN, LATTICE_MAX + 1))
    lattice_cols = int(rng.integers(LATTICE_MIN, LATTICE_MAX + 1))
    if lattice_rows * lattice_cols < n_nodes:
        return None

    path = _sample_self_avoiding_path(rng, lattice_rows, lattice_cols, n_nodes)
    if path is None:
        return None

    top_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    left_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    bottom_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))
    right_margin = int(rng.integers(MARGIN_MIN, MARGIN_MAX + 1))

    h = top_margin + (lattice_rows - 1) * CELL_STEP + bottom_margin + 1
    w = left_margin + (lattice_cols - 1) * CELL_STEP + right_margin + 1
    grid = np.full((h, w), BG, dtype=int)

    sizes = [int(rng.integers(2, 4)) for _ in range(n_nodes)]
    colors = [int(v) for v in rng.permutation(np.array(NODE_COLORS, dtype=int))[:n_nodes]]
    centers = [
        (top_margin + cell_r * CELL_STEP, left_margin + cell_c * CELL_STEP)
        for cell_r, cell_c in path
    ]

    node_pixels: list[list[tuple[int, int]]] = []
    for center, size, color in zip(centers, sizes, colors):
        pts = _node_points(center, size)
        if any(not (0 <= r < h and 0 <= c < w) for r, c in pts):
            return None
        if any(int(grid[r, c]) != BG for r, c in pts):
            return None
        for r, c in pts:
            grid[r, c] = color
        node_pixels.append(pts)

    for i in range(n_nodes - 1):
        corridor = _corridor_cells(centers[i], sizes[i], centers[i + 1], sizes[i + 1])
        if not corridor:
            return None
        for r, c in corridor:
            if not (0 <= r < h and 0 <= c < w):
                return None
            if int(grid[r, c]) != BG:
                return None
        for r, c in corridor:
            grid[r, c] = ORANGE

    use_first_endpoint = bool(rng.integers(0, 2) == 0)
    near_points = node_pixels[0] if use_first_endpoint else node_pixels[-1]
    far_points = node_pixels[-1] if use_first_endpoint else node_pixels[0]
    candidates = _marker_candidates(grid, near_points, far_points)
    if not candidates:
        return None

    best_near = min(item[0] for item in candidates)
    shortlisted = [item for item in candidates if item[0] <= best_near + 1]
    shortlisted.sort(key=lambda item: (-item[1], item[2]))
    marker_r, marker_c = shortlisted[int(rng.integers(len(shortlisted)))][3]
    grid[marker_r, marker_c] = MAGENTA

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == ORANGE) or np.any(out == MAGENTA):
        return None
    if not np.any(out == AZURE):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
