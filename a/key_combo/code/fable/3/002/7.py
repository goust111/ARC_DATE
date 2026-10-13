from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
ARC_COLORS = tuple(range(10))

Rect = tuple[int, int, int, int]

_GEN_RETRY_LIMIT = 900
_GEN_CALL_COUNT = 0
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _get_bg(g: np.ndarray) -> int:
    return int(Counter(map(int, g.flat)).most_common(1)[0][0])


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    H, W = mask.shape
    labels = np.zeros((H, W), dtype=int)
    label = 0
    for sr_raw, sc_raw in np.argwhere(mask):
        sr = int(sr_raw)
        sc = int(sc_raw)
        if labels[sr, sc] != 0:
            continue

        label += 1
        queue: deque[tuple[int, int]] = deque([(sr, sc)])
        labels[sr, sc] = label
        while queue:
            r, c = queue.popleft()
            for rr, cc in _neighbors4(r, c):
                if 0 <= rr < H and 0 <= cc < W and bool(mask[rr, cc]) and labels[rr, cc] == 0:
                    labels[rr, cc] = label
                    queue.append((rr, cc))
    return labels, label


def _legend_origin(shape: tuple[int, int], corner: int, k: int) -> tuple[int, int]:
    H, W = map(int, shape)
    width = int(k) + 2
    if corner == 0:
        return 0, 0
    if corner == 1:
        return 0, W - width
    if corner == 2:
        return H - 3, 0
    return H - 3, W - width


def _find_palette_box(g: np.ndarray, bg: int) -> tuple[list[int], int, np.ndarray, Rect]:
    H, W = g.shape
    candidates: list[tuple[list[int], int, np.ndarray, Rect]] = []
    max_k = min(7, W - 2)
    for k in range(2, max_k + 1):
        width = k + 2
        origins = {
            (0, 0),
            (0, W - width),
            (H - 3, 0),
            (H - 3, W - width),
        }
        for r0, c0 in origins:
            if r0 < 0 or c0 < 0 or r0 + 3 > H or c0 + width > W:
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + width]
            wall = int(patch[0, 0])
            if wall == int(bg):
                continue
            border = np.zeros(patch.shape, dtype=bool)
            border[0, :] = True
            border[2, :] = True
            border[1, 0] = True
            border[1, -1] = True
            if not bool(np.all(patch[border] == wall)):
                continue

            strip = [int(x) for x in patch[1, 1:-1]]
            if len(set(strip)) != k:
                continue
            if any(color in (int(bg), wall) for color in strip):
                continue

            mask = np.zeros(g.shape, dtype=bool)
            mask[r0 + 1, c0 + 1 : c0 + 1 + k] = True
            candidates.append((strip, wall, mask, (r0, c0, r0 + 2, c0 + width - 1)))

    assert len(candidates) == 1, "expected one corner palette strip enclosed by a wall frame"
    return candidates[0]


def _build_region_graph(
    g: np.ndarray, wall: int, palette_mask: np.ndarray
) -> tuple[np.ndarray, int, dict[int, set[int]], set[tuple[int, int]]]:
    passable = (g != int(wall)) & ~palette_mask
    labels, n_regions = _label_components(passable)
    assert n_regions >= 2, "wall lines must split the background into multiple regions"

    H, W = g.shape
    edges: set[tuple[int, int]] = set()
    for r_raw, c_raw in np.argwhere(g == int(wall)):
        r = int(r_raw)
        c = int(c_raw)
        for (a, b) in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
            ar, ac = a
            br, bc = b
            if not (0 <= ar < H and 0 <= ac < W and 0 <= br < H and 0 <= bc < W):
                continue
            la = int(labels[ar, ac])
            lb = int(labels[br, bc])
            if la > 0 and lb > 0 and la != lb:
                edges.add(tuple(sorted((la, lb))))

    graph = {region: set() for region in range(1, n_regions + 1)}
    for a, b in edges:
        graph[a].add(b)
        graph[b].add(a)
    return labels, n_regions, graph, edges


def _tree_distances(graph: dict[int, set[int]], start: int, n_regions: int) -> dict[int, int]:
    assert len(sum((list(v) for v in graph.values()), [])) // 2 == n_regions - 1, (
        "region adjacency graph must be a tree"
    )
    distances = {int(start): 0}
    queue: deque[int] = deque([int(start)])
    while queue:
        region = queue.popleft()
        for neighbor in graph[region]:
            if neighbor in distances:
                continue
            distances[neighbor] = distances[region] + 1
            queue.append(neighbor)
    assert set(distances) == set(range(1, n_regions + 1)), "all regions must be connected through walls"
    return distances


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every wall-separated background region by its wall distance from the seed region along the palette cycle.

    Input structure:
    - One visible wall color separates the background into regions whose wall-adjacency graph is a tree.
    - In one corner, the same wall color frames a one-row palette strip of k distinct colors; the strip order is read left to right and cycles after the last color.
    - Outside the palette strip there is exactly one seed pixel, and its color is one of the palette colors.

    Rule:
    - Keep all wall pixels and all palette-strip pixels unchanged.
    - Treat the seed pixel as part of its background region and recolor that whole region with the seed color.
    - For every other region, count the number of crossed wall edges from the seed region in the region-adjacency tree.
    - Advance that many steps from the seed color's position in the palette strip, wrapping around the strip, and fill the whole region with that color.

    Key insight:
    - The familiar two-color alternation is only the k=2 case; the true rule is distance in the wall-defined region tree, read modulo the visible palette cycle and anchored by the seed color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _get_bg(g)
    palette, wall, palette_mask, _legend_bbox = _find_palette_box(g, bg)
    palette_set = set(palette)

    labels, n_regions, graph, edges = _build_region_graph(g, wall, palette_mask)
    assert len(edges) == n_regions - 1, "region adjacency graph must be a tree"

    non_bg_region = (g != int(bg)) & (g != int(wall)) & ~palette_mask
    seed_cells = [(int(r), int(c)) for r, c in np.argwhere(non_bg_region)]
    assert len(seed_cells) == 1, "expected exactly one seed pixel outside the palette"
    sr, sc = seed_cells[0]
    seed_color = int(g[sr, sc])
    assert seed_color in palette_set, "seed pixel color must appear in the palette strip"
    seed_region = int(labels[sr, sc])
    assert seed_region > 0, "seed pixel must lie inside a background region"
    seed_index = palette.index(seed_color)

    distances = _tree_distances(graph, seed_region, n_regions)
    out = g.copy()
    k = len(palette)
    for region, distance in distances.items():
        out[labels == int(region)] = int(palette[(seed_index + int(distance)) % k])
    return out


# --------------- generator ---------------


def _legend_bbox(shape: tuple[int, int], corner: int, k: int) -> Rect:
    r0, c0 = _legend_origin(shape, corner, k)
    return r0, c0, r0 + 2, c0 + k + 1


def _expand(rect: Rect, amount: int = 1) -> Rect:
    r0, c0, r1, c1 = rect
    return r0 - amount, c0 - amount, r1 + amount, c1 + amount


def _intersects(a: Rect, b: Rect) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar0 <= br1 and br0 <= ar1 and ac0 <= bc1 and bc0 <= ac1


def _rect_size(rect: Rect) -> tuple[int, int]:
    r0, c0, r1, c1 = rect
    return r1 - r0 + 1, c1 - c0 + 1


def _draw_rect_wall(grid: np.ndarray, rect: Rect, wall: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = int(wall)
    grid[r1, c0 : c1 + 1] = int(wall)
    grid[r0 : r1 + 1, c0] = int(wall)
    grid[r0 : r1 + 1, c1] = int(wall)


def _paint_legend(grid: np.ndarray, corner: int, palette: list[int], wall: int) -> None:
    k = len(palette)
    r0, c0 = _legend_origin(grid.shape, corner, k)
    grid[r0 : r0 + 3, c0 : c0 + k + 2] = int(wall)
    for idx, color in enumerate(palette):
        grid[r0 + 1, c0 + 1 + idx] = int(color)


def _container_bounds(parent: int, rects: list[Rect], shape: tuple[int, int]) -> Rect | None:
    H, W = shape
    if parent == -1:
        return 1, 1, H - 2, W - 2

    r0, c0, r1, c1 = rects[parent]
    bounds = (r0 + 2, c0 + 2, r1 - 2, c1 - 2)
    h, w = _rect_size(bounds)
    if h < 5 or w < 5:
        return None
    return bounds


def _random_rect(
    bounds: Rect,
    rng: np.random.Generator,
    *,
    min_h: int = 5,
    min_w: int = 5,
    prefer: str = "any",
) -> Rect | None:
    br0, bc0, br1, bc1 = bounds
    max_h = br1 - br0 + 1
    max_w = bc1 - bc0 + 1
    if max_h < min_h or max_w < min_w:
        return None

    if prefer == "large":
        lo_h = max(min_h, int(np.ceil(max_h * 0.62)))
        lo_w = max(min_w, int(np.ceil(max_w * 0.62)))
        hi_h, hi_w = max_h, max_w
    elif prefer == "small":
        lo_h, lo_w = min_h, min_w
        hi_h = max(min_h, min(max_h, min_h + 5))
        hi_w = max(min_w, min(max_w, min_w + 5))
    else:
        lo_h, lo_w = min_h, min_w
        hi_h, hi_w = max_h, max_w

    h = int(rng.integers(lo_h, hi_h + 1))
    w = int(rng.integers(lo_w, hi_w + 1))
    r0 = int(rng.integers(br0, br1 - h + 2))
    c0 = int(rng.integers(bc0, bc1 - w + 2))
    return r0, c0, r0 + h - 1, c0 + w - 1


def _add_loop(
    parent: int,
    rects: list[Rect],
    children: dict[int, list[int]],
    shape: tuple[int, int],
    forbidden: list[Rect],
    rng: np.random.Generator,
    *,
    prefer: str = "any",
    tries: int = 180,
) -> int | None:
    bounds = _container_bounds(parent, rects, shape)
    if bounds is None:
        return None

    for _ in range(tries):
        rect = _random_rect(bounds, rng, prefer=prefer)
        if rect is None:
            return None

        padded = _expand(rect, 1)
        bad = False
        for sibling in children.get(parent, []):
            if _intersects(padded, _expand(rects[sibling], 1)):
                bad = True
                break
        if bad:
            continue
        if parent == -1 and any(_intersects(padded, f) for f in forbidden):
            continue

        idx = len(rects)
        rects.append(rect)
        children.setdefault(parent, []).append(idx)
        children.setdefault(idx, [])
        return idx
    return None


def _build_star_layout(
    shape: tuple[int, int], forbidden: list[Rect], rng: np.random.Generator
) -> list[Rect] | None:
    rects: list[Rect] = []
    children: dict[int, list[int]] = {-1: []}
    target = int(rng.integers(3, 6))
    for _ in range(500):
        if len(rects) >= target:
            return rects
        _add_loop(-1, rects, children, shape, forbidden, rng, prefer="small", tries=20)
    return rects if len(rects) >= 3 else None


def _build_chain_layout(
    shape: tuple[int, int], forbidden: list[Rect], rng: np.random.Generator
) -> list[Rect] | None:
    rects: list[Rect] = []
    children: dict[int, list[int]] = {-1: []}
    parent = _add_loop(-1, rects, children, shape, forbidden, rng, prefer="large", tries=260)
    if parent is None:
        return None

    target_depth = int(rng.integers(3, 6))
    while len(rects) < target_depth:
        child = _add_loop(parent, rects, children, shape, forbidden, rng, prefer="large", tries=240)
        if child is None:
            break
        parent = child
    return rects if len(rects) >= 3 else None


def _build_mixed_layout(
    shape: tuple[int, int], forbidden: list[Rect], rng: np.random.Generator
) -> list[Rect] | None:
    rects: list[Rect] = []
    children: dict[int, list[int]] = {-1: []}
    outer = _add_loop(-1, rects, children, shape, forbidden, rng, prefer="large", tries=280)
    if outer is None:
        return None

    inner_targets = int(rng.integers(2, 4))
    inner_ids: list[int] = []
    for _ in range(inner_targets):
        child = _add_loop(outer, rects, children, shape, forbidden, rng, prefer="small", tries=220)
        if child is not None:
            inner_ids.append(child)
    if len(inner_ids) < 2:
        return None

    rng.shuffle(inner_ids)
    _add_loop(inner_ids[0], rects, children, shape, forbidden, rng, prefer="small", tries=180)
    _add_loop(-1, rects, children, shape, forbidden, rng, prefer="small", tries=120)
    return rects if len(rects) >= 4 else None


def _build_random_layout(
    shape: tuple[int, int], forbidden: list[Rect], rng: np.random.Generator
) -> list[Rect] | None:
    rects: list[Rect] = []
    children: dict[int, list[int]] = {-1: []}
    target = int(rng.integers(4, 8))
    for step in range(700):
        if len(rects) >= target:
            break
        parents = [-1] + list(range(len(rects)))
        rng.shuffle(parents)
        placed = False
        for parent in parents:
            prefer = "large" if parent == -1 and len(rects) == 0 else ("small" if rng.random() < 0.6 else "any")
            child = _add_loop(parent, rects, children, shape, forbidden, rng, prefer=prefer, tries=24)
            if child is not None:
                placed = True
                break
        if not placed and step > 120:
            break
    return rects if len(rects) >= 4 else None


def _choose_shape(mode: int, rng: np.random.Generator) -> tuple[int, int]:
    if mode in (1, 2):
        return int(rng.integers(23, 31)), int(rng.integers(23, 31))
    if mode == 0:
        return int(rng.integers(17, 29)), int(rng.integers(18, 31))
    return int(rng.integers(20, 31)), int(rng.integers(20, 31))


def _choose_k(mode: int, rng: np.random.Generator) -> int:
    if mode == 0:
        return int(rng.choice(np.array([2, 3], dtype=int)))
    if mode == 1:
        return int(rng.choice(np.array([3, 4, 5], dtype=int)))
    if mode == 2:
        return int(rng.choice(np.array([4, 5], dtype=int)))
    return int(rng.choice(np.array([2, 3, 4, 5], dtype=int)))


def _sample_colors(k: int, rng: np.random.Generator) -> tuple[int, list[int]]:
    wall = int(rng.choice(np.array([1, 2, 5, 8], dtype=int)))
    available = [color for color in ARC_COLORS if color not in (BLACK, wall)]
    palette = rng.choice(np.array(available, dtype=int), size=int(k), replace=False).astype(int).tolist()
    rng.shuffle(palette)
    return wall, [int(c) for c in palette]


def _all_distances(graph: dict[int, set[int]], n_regions: int) -> dict[int, dict[int, int]]:
    all_dist: dict[int, dict[int, int]] = {}
    for start in range(1, n_regions + 1):
        try:
            all_dist[start] = _tree_distances(graph, start, n_regions)
        except AssertionError:
            return {}
    return all_dist


def _choose_seed_region(
    graph: dict[int, set[int]], n_regions: int, k: int, rng: np.random.Generator
) -> tuple[int, dict[int, int]] | None:
    all_dist = _all_distances(graph, n_regions)
    if not all_dist:
        return None

    candidates: list[tuple[int, dict[int, int]]] = []
    for region, distances in all_dist.items():
        residues = {int(d) % int(k) for d in distances.values()}
        if len(residues) == min(int(k), int(n_regions)):
            candidates.append((region, distances))

    if not candidates:
        return None

    max_depth = max(max(dist.values()) for _region, dist in candidates)
    deep = [(region, dist) for region, dist in candidates if max(dist.values()) >= max_depth - 1]
    return deep[int(rng.integers(0, len(deep)))]


def _safe_seed_cells(labels: np.ndarray, region: int, grid: np.ndarray, bg: int) -> np.ndarray:
    cells = np.argwhere((labels == int(region)) & (grid == int(bg)))
    if len(cells) == 0:
        return cells

    H, W = labels.shape
    safe: list[tuple[int, int]] = []
    for r_raw, c_raw in cells:
        r = int(r_raw)
        c = int(c_raw)
        same_neighbors = 0
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < H and 0 <= cc < W and int(labels[rr, cc]) == int(region):
                same_neighbors += 1
        if same_neighbors >= 2:
            safe.append((r, c))
    return np.array(safe if safe else [(int(r), int(c)) for r, c in cells], dtype=int)


def _expected_from_semantics(
    grid: np.ndarray,
    labels: np.ndarray,
    distances: dict[int, int],
    palette: list[int],
    seed_index: int,
) -> np.ndarray:
    out = grid.copy()
    k = len(palette)
    for region, distance in distances.items():
        out[labels == int(region)] = int(palette[(int(seed_index) + int(distance)) % k])
    return out


def _semantic_valid(
    grid: np.ndarray,
    expected: np.ndarray,
    k: int,
    n_regions: int,
    distances: dict[int, int],
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if n_regions < 4:
        return False
    residues = {int(d) % int(k) for d in distances.values()}
    if len(residues) < min(int(k), int(n_regions)):
        return False
    if max(distances.values()) < 2:
        return False
    try:
        actual = main(grid)
    except AssertionError:
        return False
    return bool(np.array_equal(actual, expected))


def _gen_attempt(mode: int) -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()
    H, W = _choose_shape(mode, rng)
    k = _choose_k(mode, rng)
    corner = int(rng.integers(0, 4))
    wall, palette = _sample_colors(k, rng)
    bg = BLACK

    legend = _legend_bbox((H, W), corner, k)
    forbidden = [_expand(legend, 1)]

    if mode == 0:
        rects = _build_star_layout((H, W), forbidden, rng)
    elif mode == 1:
        rects = _build_chain_layout((H, W), forbidden, rng)
    elif mode == 2:
        rects = _build_mixed_layout((H, W), forbidden, rng)
    else:
        rects = _build_random_layout((H, W), forbidden, rng)
    if rects is None:
        return None

    grid = np.full((H, W), bg, dtype=int)
    _paint_legend(grid, corner, palette, wall)
    for rect in rects:
        _draw_rect_wall(grid, rect, wall)

    try:
        found_palette, found_wall, palette_mask, _ = _find_palette_box(grid, bg)
    except AssertionError:
        return None
    if found_palette != palette or found_wall != wall:
        return None

    labels, n_regions, graph, edges = _build_region_graph(grid, wall, palette_mask)
    if n_regions != len(rects) + 1 or len(edges) != n_regions - 1:
        return None

    seed_choice = _choose_seed_region(graph, n_regions, k, rng)
    if seed_choice is None:
        return None
    seed_region, distances = seed_choice

    seed_cells = _safe_seed_cells(labels, seed_region, grid, bg)
    if len(seed_cells) == 0:
        return None
    sr, sc = map(int, seed_cells[int(rng.integers(0, len(seed_cells)))])

    if mode == 3 and rng.random() < 0.35:
        seed_index = 0
    else:
        seed_index = int(rng.integers(1, k))
    grid[sr, sc] = int(palette[seed_index])

    expected = _expected_from_semantics(grid, labels, distances, palette, seed_index)
    if not _semantic_valid(grid, expected, k, n_regions, distances):
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT

    for _ in range(_GEN_RETRY_LIMIT):
        mode = _GEN_CALL_COUNT % 4
        result = _gen_attempt(mode)
        if result is None:
            continue

        grid, expected = result
        signature = (tuple(map(int, expected.shape)), expected.tobytes())
        if signature in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(signature)
        _GEN_CALL_COUNT += 1
        return grid

    raise RuntimeError("failed to generate a valid wall-distance palette-cycle input")
