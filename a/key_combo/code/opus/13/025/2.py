from __future__ import annotations

import math
import random
from collections import deque

import numpy as np


BG = 0
WALL = 5
INDICATOR_COLORS = (1, 2, 3, 4, 7)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400
_NEXT_LOOP_COUNT = random.choice([4, 5])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_non_wall(grid: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = grid.shape
    mask = np.zeros((h, w), dtype=bool)
    sr, sc = start
    q = deque([(sr, sc)])
    mask[sr, sc] = True
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds(grid.shape, nr, nc) and not mask[nr, nc] and grid[nr, nc] != WALL:
                mask[nr, nc] = True
                q.append((nr, nc))
    return mask


def _enclosed_mask_from_wall(wall_mask: np.ndarray) -> np.ndarray:
    h, w = wall_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not wall_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not wall_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if _in_bounds((h, w), nr, nc) and not wall_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~wall_mask) & (~outside)


def _centroid(coords: np.ndarray) -> tuple[float, float]:
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _extract_regions(grid: np.ndarray) -> list[dict]:
    indicators = np.argwhere((grid != BG) & (grid != WALL))
    assert 4 <= len(indicators) <= 5, "expected 4 or 5 indicator pixels"

    regions: list[dict] = []
    covered = np.zeros(grid.shape, dtype=bool)
    for r, c in indicators:
        region = _flood_non_wall(grid, (int(r), int(c)))
        assert not np.any(covered & region), "regions must be disjoint"
        covered |= region

        indicator_cells = np.argwhere(region & ((grid != BG) & (grid != WALL)))
        assert len(indicator_cells) == 1, "each region must contain exactly one indicator"

        region_coords = np.argwhere(region)
        boundary = np.zeros(grid.shape, dtype=bool)
        for rr, cc in region_coords:
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = int(rr) + dr, int(cc) + dc
                    if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == WALL:
                        boundary[nr, nc] = True
        boundary_coords = np.argwhere(boundary)
        assert boundary_coords.size > 0, "region must have a wall boundary"

        regions.append(
            {
                "mask": region,
                "coords": region_coords,
                "boundary": boundary,
                "boundary_coords": boundary_coords,
                "color": int(grid[int(r), int(c)]),
                "center": _centroid(region_coords),
            }
        )
    return regions


def _connector_edges(grid: np.ndarray, regions: list[dict]) -> set[tuple[int, int]]:
    boundary_owner = np.full(grid.shape, -1, dtype=int)
    all_boundaries = np.zeros(grid.shape, dtype=bool)
    for idx, region in enumerate(regions):
        boundary_owner[region["boundary"]] = idx
        all_boundaries |= region["boundary"]

    connector_mask = (grid == WALL) & (~all_boundaries)
    edges: set[tuple[int, int]] = set()
    for comp in _components(connector_mask):
        touched: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = int(r) + dr, int(c) + dc
                if _in_bounds(grid.shape, nr, nc) and boundary_owner[nr, nc] >= 0:
                    touched.add(int(boundary_owner[nr, nc]))
        assert len(touched) == 2, f"connector must touch exactly two loops, got {touched}"
        a, b = sorted(touched)
        edges.add((a, b))

    assert len(edges) == len(regions), "connectors must form one Hamiltonian cycle"
    degrees = [0] * len(regions)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    assert all(deg == 2 for deg in degrees), f"expected degree-2 cycle, got {degrees}"
    return edges


def _clockwise_order(regions: list[dict], edges: set[tuple[int, int]]) -> list[int]:
    centers = np.array([region["center"] for region in regions], dtype=float)
    global_center = centers.mean(axis=0)
    angles = [math.atan2(r - global_center[0], c - global_center[1]) for r, c in centers]
    order = sorted(range(len(regions)), key=lambda i: angles[i])

    edge_set = {tuple(sorted(edge)) for edge in edges}
    cyclic_pairs = {
        tuple(sorted((order[i], order[(i + 1) % len(order)])))
        for i in range(len(order))
    }
    assert cyclic_pairs == edge_set, "geometric clockwise order must match connector cycle"
    return order


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate the fill color of each closed wall-bounded region to the next region clockwise around the connector cycle.

    Input structure:
    - The grid has black background, several grey closed wall loops, and grey connector lines joining the loops into one cycle.
    - Each wall loop encloses one irregular interior region, and each interior contains exactly one single-pixel indicator color.
    - Wall pixels and connector pixels share the same grey color, while all non-wall non-background pixels are indicators.

    Rule:
    - Flood-fill from each indicator through non-grey cells to recover the exact interior region bounded by its wall loop.
    - Remove those boundary loops from the grey network, trace the remaining connector components, and use them to recover the loop-cycle adjacency.
    - Order the loop centers clockwise and recolor every pixel in each interior region with the indicator color from the next clockwise region, leaving all grey pixels unchanged.

    Key insight:
    - The pixels to repaint come from local flood-fill geometry inside each closed wall, but the color source comes from the external connector-cycle topology, so both must be inferred and combined.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    regions = _extract_regions(g)
    edges = _connector_edges(g, regions)
    order = _clockwise_order(regions, edges)

    out = g.copy()
    for idx, region_id in enumerate(order):
        next_region_id = order[(idx + 1) % len(order)]
        out[regions[region_id]["mask"]] = regions[next_region_id]["color"]
    out[g == WALL] = WALL
    return out


def _template_arrays() -> list[np.ndarray]:
    rows = [
        ["####", "#..#", "#..#", "####"],
        ["#####", "#...#", "#...#", "#####"],
        ["#####", "#...#", "#...#", "#...#", "#####"],
        ["#####", "#...#", "###.#", "#...#", "#####"],
        ["######", "#....#", "###..#", "#....#", "######"],
    ]
    out = []
    for tmpl in rows:
        arr = np.array([[ch == "#" for ch in row] for row in tmpl], dtype=bool)
        enclosed = _enclosed_mask_from_wall(arr)
        assert 4 <= int(enclosed.sum()) <= 10
        assert len(_components(enclosed)) == 1
        out.append(arr)
    return out


LOOP_TEMPLATES = _template_arrays()


def _random_loop_template(n_loops: int) -> tuple[np.ndarray, np.ndarray, bool]:
    pool = LOOP_TEMPLATES if n_loops == 4 else [tmpl for tmpl in LOOP_TEMPLATES if max(tmpl.shape) <= 5]
    wall = random.choice(pool).copy()
    wall = np.rot90(wall, random.randrange(4))
    if random.random() < 0.5:
        wall = np.fliplr(wall)
    interior = _enclosed_mask_from_wall(wall)
    irregular = bool(np.any(wall[1:-1, 1:-1]))
    return wall, interior, irregular


def _bbox_with_margin(cells: np.ndarray, margin: int = 0) -> tuple[int, int, int, int]:
    r0 = int(cells[:, 0].min()) - margin
    c0 = int(cells[:, 1].min()) - margin
    r1 = int(cells[:, 0].max()) + margin
    c1 = int(cells[:, 1].max()) + margin
    return r0, c0, r1, c1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _layout_centers(h: int, w: int, n: int) -> list[tuple[int, int]]:
    if n == 4:
        anchors = [(0.22, 0.24), (0.22, 0.76), (0.78, 0.76), (0.78, 0.24)]
    else:
        anchors = [(0.17, 0.50), (0.31, 0.81), (0.74, 0.72), (0.76, 0.28), (0.31, 0.19)]
    centers = []
    for rf, cf in anchors:
        jr = random.randint(-1, 1)
        jc = random.randint(-1, 1)
        centers.append((int(round(rf * (h - 1))) + jr, int(round(cf * (w - 1))) + jc))
    return centers


def _candidate_ports(loop: dict, side: str) -> list[tuple[int, int]]:
    coords = loop["boundary_coords"]
    if side == "N":
        extreme = int(coords[:, 0].min())
        pts = [tuple(map(int, p)) for p in coords if int(p[0]) == extreme]
        pts.sort(key=lambda rc: rc[1])
    elif side == "S":
        extreme = int(coords[:, 0].max())
        pts = [tuple(map(int, p)) for p in coords if int(p[0]) == extreme]
        pts.sort(key=lambda rc: rc[1])
    elif side == "W":
        extreme = int(coords[:, 1].min())
        pts = [tuple(map(int, p)) for p in coords if int(p[1]) == extreme]
        pts.sort(key=lambda rc: rc[0])
    else:
        extreme = int(coords[:, 1].max())
        pts = [tuple(map(int, p)) for p in coords if int(p[1]) == extreme]
        pts.sort(key=lambda rc: rc[0])
    return pts


def _side_toward(src: tuple[float, float], dst: tuple[float, float]) -> str:
    dr = dst[0] - src[0]
    dc = dst[1] - src[1]
    if abs(dc) >= abs(dr):
        return "E" if dc > 0 else "W"
    return "S" if dr > 0 else "N"


def _outside_neighbors(grid: np.ndarray, port: tuple[int, int], region_mask: np.ndarray) -> list[tuple[int, int]]:
    r, c = port
    nbrs = []
    for dr, dc in DIR4:
        nr, nc = r + dr, c + dc
        if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BG and not region_mask[nr, nc]:
            nbrs.append((nr, nc))
    return nbrs


def _bfs_path(
    grid: np.ndarray,
    starts: list[tuple[int, int]],
    goals: set[tuple[int, int]],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    q = deque(starts)
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None for start in starts}
    while q:
        cur = q.popleft()
        if cur in goals:
            path: list[tuple[int, int]] = []
            node: tuple[int, int] | None = cur
            while node is not None:
                path.append(node)
                node = prev[node]
            path.reverse()
            return path
        r, c = cur
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            nxt = (nr, nc)
            if not _in_bounds(grid.shape, nr, nc):
                continue
            if blocked[nr, nc] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    return None


def _connect_loops(grid: np.ndarray, loops: list[dict], order: list[int]) -> bool:
    boundary_union = np.zeros(grid.shape, dtype=bool)
    region_union = np.zeros(grid.shape, dtype=bool)
    for loop in loops:
        boundary_union |= loop["boundary"]
        region_union |= loop["region_mask"]

    boundary_clearance = np.zeros(grid.shape, dtype=bool)
    for r, c in np.argwhere(boundary_union):
        for dr, dc in DIR4:
            nr, nc = int(r) + dr, int(c) + dc
            if _in_bounds(grid.shape, nr, nc) and not boundary_union[nr, nc]:
                boundary_clearance[nr, nc] = True

    used_ports: list[set[tuple[int, int]]] = [set() for _ in loops]
    for pos, loop_id in enumerate(order):
        next_id = order[(pos + 1) % len(order)]
        src = loops[loop_id]
        dst = loops[next_id]
        src_side = _side_toward(src["center"], dst["center"])
        dst_side = _side_toward(dst["center"], src["center"])

        src_ports = _candidate_ports(src, src_side)
        dst_ports = _candidate_ports(dst, dst_side)
        src_ports.sort(key=lambda rc: abs(rc[0] - dst["center"][0]) + abs(rc[1] - dst["center"][1]))
        dst_ports.sort(key=lambda rc: abs(rc[0] - src["center"][0]) + abs(rc[1] - src["center"][1]))

        placed = False
        for src_port in src_ports:
            if src_port in used_ports[loop_id]:
                continue
            src_starts = _outside_neighbors(grid, src_port, src["region_mask"])
            if not src_starts:
                continue
            for dst_port in dst_ports:
                if dst_port in used_ports[next_id]:
                    continue
                dst_starts = _outside_neighbors(grid, dst_port, dst["region_mask"])
                if not dst_starts:
                    continue

                blocked = (grid != BG) | region_union | boundary_clearance
                for cell in src_starts + dst_starts:
                    blocked[cell] = False
                path = _bfs_path(grid, src_starts, set(dst_starts), blocked)
                if path is None:
                    continue

                for r, c in path:
                    grid[r, c] = WALL
                used_ports[loop_id].add(src_port)
                used_ports[next_id].add(dst_port)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _angles_clockwise(centers: list[tuple[float, float]]) -> list[int]:
    pts = np.array(centers, dtype=float)
    ctr = pts.mean(axis=0)
    angles = [math.atan2(r - ctr[0], c - ctr[1]) for r, c in pts]
    return sorted(range(len(centers)), key=lambda i: angles[i])


def _build_example(n_loops: int | None = None) -> np.ndarray | None:
    if n_loops is None:
        n_loops = random.choice([4, 5])
    h = random.randint(18, 26) if n_loops == 4 else random.randint(23, 29)
    w = random.randint(18, 26) if n_loops == 4 else random.randint(23, 29)
    grid = np.zeros((h, w), dtype=int)

    templates = [_random_loop_template(n_loops) for _ in range(n_loops)]
    centers = _layout_centers(h, w, n_loops)
    loops: list[dict] = []
    occupied_boxes: list[tuple[int, int, int, int]] = []

    for idx, ((wall, interior, irregular), center) in enumerate(zip(templates, centers)):
        ph, pw = wall.shape
        r0 = center[0] - ph // 2
        c0 = center[1] - pw // 2
        if r0 < 1 or c0 < 1 or r0 + ph >= h - 1 or c0 + pw >= w - 1:
            return None

        local_boundary = np.argwhere(wall)
        global_boundary = local_boundary + np.array([r0, c0])
        box = _bbox_with_margin(global_boundary, margin=1)
        if box[0] < 0 or box[1] < 0 or box[2] >= h or box[3] >= w:
            return None
        if any(_boxes_overlap(box, prev) for prev in occupied_boxes):
            return None
        occupied_boxes.append(box)

        local_region = np.argwhere(interior)
        global_region = local_region + np.array([r0, c0])
        boundary_mask = np.zeros_like(grid, dtype=bool)
        region_mask = np.zeros_like(grid, dtype=bool)
        boundary_mask[global_boundary[:, 0], global_boundary[:, 1]] = True
        region_mask[global_region[:, 0], global_region[:, 1]] = True
        grid[boundary_mask] = WALL

        loops.append(
            {
                "boundary": boundary_mask,
                "boundary_coords": global_boundary,
                "region_mask": region_mask,
                "center": _centroid(global_region),
                "irregular": irregular,
                "id": idx,
            }
        )

    order = _angles_clockwise([loop["center"] for loop in loops])
    expected_edges = {
        tuple(sorted((order[i], order[(i + 1) % len(order)])))
        for i in range(len(order))
    }

    if not _connect_loops(grid, loops, order):
        return None

    colors = random.sample(list(INDICATOR_COLORS), n_loops)
    irregular_count = 0
    for loop, color, (_wall, _interior, _irregular) in zip(loops, colors, templates):
        region_coords = np.argwhere(loop["region_mask"])
        if len(region_coords) < 4:
            return None
        if loop["irregular"]:
            irregular_count += 1
        indicator = tuple(map(int, region_coords[random.randrange(len(region_coords))]))
        grid[indicator] = color

    if irregular_count == 0 and n_loops >= 4:
        return None

    try:
        regions = _extract_regions(grid)
        edges = _connector_edges(grid, regions)
        if edges != expected_edges:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _NEXT_LOOP_COUNT
    target_loop_count = _NEXT_LOOP_COUNT
    _NEXT_LOOP_COUNT = 4 if _NEXT_LOOP_COUNT == 5 else 5

    attempts = MAX_GEN_ATTEMPTS if target_loop_count == 4 else MAX_GEN_ATTEMPTS * 4
    for _ in range(attempts):
        grid = _build_example(target_loop_count)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
