import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8]
GEN_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, *, conn8: bool) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    neigh = neighbors8 if conn8 else neighbors4
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for nr, nc in neigh(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _rect_border_ok(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    expect = 2 * (h + w) - 4
    if int(mask[r0 : r1 + 1, c0 : c1 + 1].sum()) != expect:
        return False
    if not np.all(mask[r0, c0 : c1 + 1]):
        return False
    if not np.all(mask[r1, c0 : c1 + 1]):
        return False
    if not np.all(mask[r0 : r1 + 1, c0]):
        return False
    if not np.all(mask[r0 : r1 + 1, c1]):
        return False
    return True


def _find_legend_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    grey_mask = grid == GREY
    cands = []
    for comp in _components(grey_mask, conn8=False):
        box = _bbox(comp)
        if not _rect_border_ok(grey_mask, box):
            continue
        r0, c0, r1, c1 = box
        cands.append((r1 + c1, r0, c0, r1, c1))
    assert cands, "legend frame not found"
    _score, r0, c0, r1, c1 = max(cands)
    return r0, c0, r1, c1


def _runs(indices: list[int]) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans = []
    start = prev = indices[0]
    for x in indices[1:]:
        if x == prev + 1:
            prev = x
            continue
        spans.append((start, prev))
        start = prev = x
    spans.append((start, prev))
    return spans


def _decode_pattern(patch: np.ndarray) -> tuple[str, int, int]:
    colors = [int(x) for x in np.unique(patch) if int(x) != BG]
    assert len(colors) == 2, "legend patch must use exactly two colors"
    if all(len(set(map(int, row))) == 1 for row in patch):
        return "h", int(patch[0, 0]), int(patch[1, 0])
    if all(len(set(map(int, patch[:, c]))) == 1 for c in range(patch.shape[1])):
        return "v", int(patch[0, 0]), int(patch[0, 1])
    return "d", int(patch[0, 0]), int(patch[0, 1])


def _read_legend(grid: np.ndarray) -> tuple[dict[int, tuple[str, int, int]], tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _find_legend_bbox(grid)
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    nonempty_rows = [r for r in range(inner.shape[0]) if np.any(inner[r] != BG)]
    mapping: dict[int, tuple[str, int, int]] = {}
    for rr0, rr1 in _runs(nonempty_rows):
        band = inner[rr0 : rr1 + 1]
        maroon = band == MAROON
        n_sides = int(maroon.sum())
        colored = (band != BG) & (band != MAROON)
        pts = np.argwhere(colored)
        assert pts.size > 0, "legend row missing sample patch"
        pr0, pc0 = pts.min(axis=0)
        pr1, pc1 = pts.max(axis=0)
        patch = band[pr0 : pr1 + 1, pc0 : pc1 + 1]
        assert patch.shape == (3, 3), "legend sample patch must be 3x3"
        mapping[n_sides] = _decode_pattern(patch)
    assert mapping, "legend mapping must be non-empty"
    return mapping, (r0, c0, r1, c1)


def _trace_cycle(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in coords:
        nbrs = []
        for nr, nc in neighbors8(r, c):
            if (nr, nc) not in pts:
                continue
            dr = nr - r
            dc = nc - c
            if abs(dr) + abs(dc) == 1:
                nbrs.append((nr, nc))
                continue
            if abs(dr) == 1 and abs(dc) == 1:
                if (r, c + dc) not in pts and (r + dr, c) not in pts:
                    nbrs.append((nr, nc))
        assert len(nbrs) == 2, "polygon outline must be a simple cycle"
        adj[(r, c)] = sorted(nbrs)
    start = min(coords)
    ordered = [start]
    prev = None
    cur = start
    while True:
        a, b = adj[cur]
        nxt = a if a != prev else b
        prev, cur = cur, nxt
        if cur == start:
            break
        ordered.append(cur)
        assert len(ordered) <= len(coords), "cycle traversal overflow"
    assert len(ordered) == len(coords), "cycle traversal must visit every outline pixel"
    return ordered


def _count_sides(coords: list[tuple[int, int]]) -> int:
    points = sorted({(c, r) for r, c in coords})
    assert len(points) >= 3

    def cross(o: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> int:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[int, int]] = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper: list[tuple[int, int]] = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    return len(hull)


def _interior_cells(
    grid: np.ndarray, color: int, coords: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox(coords)
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    wall = sub == color
    open_mask = sub == BG
    h, w = sub.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if open_mask[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and open_mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    inside = [(r0 + r, c0 + c) for r in range(h) for c in range(w) if open_mask[r, c] and not seen[r, c]]
    return (r0, c0, r1, c1), inside


def _fill_pattern(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    cells: list[tuple[int, int]],
    pattern: tuple[str, int, int],
) -> None:
    r0, c0, _r1, _c1 = bbox
    orient, a, b = pattern
    for r, c in cells:
        rr = r - r0
        cc = c - c0
        if orient == "h":
            out[r, c] = a if rr % 2 == 0 else b
        elif orient == "v":
            out[r, c] = a if cc % 2 == 0 else b
        else:
            out[r, c] = a if (rr + cc) % 2 == 0 else b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill polygon interiors using a legend keyed by polygon side count.

    Input structure:
    - The background is black.
    - Several single-pixel-wide closed convex polygon outlines of different colors appear on the grid.
    - A grey rectangular legend sits near the lower-right corner; each legend row shows a maroon pixel count on the left and a 3x3 stripe sample on the right.

    Rule:
    - Read the grey-framed legend and map each maroon count N to the stripe style shown in that row's 3x3 sample.
    - For every colored polygon outline, trace its cycle and count direction runs to recover its number of sides N.
    - Fill the black pixels strictly inside the outline with the looked-up stripe pattern, aligned from the polygon's bounding-box top-left corner, while keeping the outline unchanged.

    Key insight:
    - Color does not determine the fill; the important attribute is each polygon's geometric side count, which indexes a visual rule stored in the legend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_map, legend_bbox = _read_legend(g)
    lr0, lc0, lr1, lc1 = legend_bbox
    usable = np.ones_like(g, dtype=bool)
    usable[lr0 : lr1 + 1, lc0 : lc1 + 1] = False

    out = g.copy()
    colors = [int(x) for x in np.unique(g[usable]) if int(x) != BG]
    for color in colors:
        mask = (g == color) & usable
        for comp in _components(mask, conn8=True):
            n_sides = _count_sides(comp)
            assert n_sides in legend_map, f"no legend entry for {n_sides}-gon"
            bbox, inside = _interior_cells(g, color, comp)
            _fill_pattern(out, bbox, inside, legend_map[n_sides])
    return out


def _segment_points(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return [a]
    sr = 0 if dr == 0 else dr // abs(dr)
    sc = 0 if dc == 0 else dc // abs(dc)
    assert dr == 0 or dc == 0 or abs(dr) == abs(dc)
    return [(a[0] + i * sr, a[1] + i * sc) for i in range(steps + 1)]


def _normalize_vertices(vertices: list[tuple[int, int]]) -> list[tuple[int, int]]:
    min_r = min(r for r, _ in vertices)
    min_c = min(c for _, c in vertices)
    return [(r - min_r, c - min_c) for r, c in vertices]


def _transform_vertices(vertices: list[tuple[int, int]], mode: int) -> list[tuple[int, int]]:
    def tf(r: int, c: int) -> tuple[int, int]:
        if mode == 0:
            return r, c
        if mode == 1:
            return c, -r
        if mode == 2:
            return -r, -c
        if mode == 3:
            return -c, r
        if mode == 4:
            return r, -c
        if mode == 5:
            return -r, c
        if mode == 6:
            return c, r
        return -c, -r

    return _normalize_vertices([tf(r, c) for r, c in vertices])


def _rasterize_outline(vertices: list[tuple[int, int]]) -> np.ndarray:
    verts = _normalize_vertices(vertices)
    pts: set[tuple[int, int]] = set()
    for i in range(len(verts)):
        a = verts[i]
        b = verts[(i + 1) % len(verts)]
        pts.update(_segment_points(a, b))
    h = max(r for r, _ in pts) + 1
    w = max(c for _, c in pts) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return mask


def _triangle_vertices(rng: random.Random) -> list[tuple[int, int]]:
    side = rng.randint(4, 7)
    return [(0, 0), (0, side), (side, 0)]


def _quad_vertices(rng: random.Random) -> list[tuple[int, int]]:
    if rng.random() < 0.5:
        h = rng.randint(4, 7)
        w = rng.randint(4, 7)
        return [(0, 0), (0, w), (h, w), (h, 0)]
    a = rng.randint(4, 7)
    b = rng.randint(2, 4)
    return [(0, 0), (0, a), (b, a + b), (b, b)]


def _pentagon_vertices(rng: random.Random) -> list[tuple[int, int]]:
    b = rng.randint(1, 3)
    c = rng.randint(3, 5)
    d = rng.randint(1, 3)
    a = c + d - b
    return [(0, 0), (0, a), (b, a + b), (b + c, d), (b + c, 0)]


def _hexagon_vertices(rng: random.Random) -> list[tuple[int, int]]:
    a = rng.randint(3, 5)
    b = rng.randint(1, 3)
    c = rng.randint(3, 5)
    return [(0, 0), (0, a), (b, a + b), (b + c, a + b), (b + c, b), (c, 0)]


def _make_polygon_mask(rng: random.Random, n_sides: int) -> np.ndarray:
    for _ in range(50):
        if n_sides == 3:
            vertices = _triangle_vertices(rng)
        elif n_sides == 4:
            vertices = _quad_vertices(rng)
        elif n_sides == 5:
            vertices = _pentagon_vertices(rng)
        else:
            vertices = _hexagon_vertices(rng)
        vertices = _transform_vertices(vertices, rng.randrange(8))
        mask = _rasterize_outline(vertices)
        temp = np.zeros(mask.shape, dtype=int)
        temp[mask] = 1
        coords = list(map(tuple, np.argwhere(mask)))
        counted = _count_sides(coords)
        _bbox0, inside = _interior_cells(temp, 1, coords)
        if counted == n_sides and len(inside) >= 2:
            return mask
    raise RuntimeError(f"failed to synthesize a valid {n_sides}-gon")


def _pattern_patch(kind: str, a: int, b: int) -> np.ndarray:
    patch = np.zeros((3, 3), dtype=int)
    for r in range(3):
        for c in range(3):
            if kind == "h":
                patch[r, c] = a if r % 2 == 0 else b
            elif kind == "v":
                patch[r, c] = a if c % 2 == 0 else b
            else:
                patch[r, c] = a if (r + c) % 2 == 0 else b
    return patch


def _legend_grid(rng: random.Random) -> tuple[np.ndarray, dict[int, tuple[str, int, int]]]:
    orientations = ["h", "v", "d", rng.choice(["h", "v", "d"])]
    rng.shuffle(orientations)
    pairs = rng.sample([(a, b) for i, a in enumerate(COLOR_POOL) for b in COLOR_POOL[i + 1 :]], 4)
    mapping: dict[int, tuple[str, int, int]] = {}

    row_h = 3
    gap = 1
    inner_h = row_h * 4 + gap * 3
    inner_w = 11
    legend = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    legend[0, :] = GREY
    legend[-1, :] = GREY
    legend[:, 0] = GREY
    legend[:, -1] = GREY

    order = [3, 4, 5, 6]
    rng.shuffle(order)
    for idx, n_sides in enumerate(order):
        r0 = 1 + idx * (row_h + gap)
        kind = orientations[idx]
        a, b = pairs[idx]
        mapping[n_sides] = (kind, a, b)
        legend[r0 + 1, 1 : 1 + n_sides] = MAROON
        legend[r0 : r0 + 3, 7:10] = _pattern_patch(kind, a, b)
    return legend, mapping


def _choose_side_counts(rng: random.Random, n_polys: int) -> list[int]:
    base = [3, 4, 5, 6]
    if n_polys <= 4:
        side_counts = rng.sample(base, n_polys)
        if n_polys == 3 and rng.random() < 0.35:
            dup = rng.choice(side_counts)
            side_counts[rng.randrange(3)] = dup
    else:
        side_counts = base + [rng.choice(base)]
    rng.shuffle(side_counts)
    return side_counts


def _place_mask(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    spots = [(r, c) for r in range(1, h - mh) for c in range(1, w - mw)]
    rng.shuffle(spots)
    for r0, c0 in spots:
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + mh + 1)
        cc1 = min(w, c0 + mw + 1)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        pts = np.argwhere(mask)
        for dr, dc in pts:
            grid[r0 + dr, c0 + dc] = color
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(24, 30)
    w = rng.randint(24, 30)
    grid = np.zeros((h, w), dtype=int)

    legend, expected_map = _legend_grid(rng)
    lh, lw = legend.shape
    if lh + 2 > h or lw + 2 > w:
        return None
    bottom_margin = rng.randint(0, 1)
    right_margin = rng.randint(0, 1)
    lr0 = h - lh - bottom_margin
    lc0 = w - lw - right_margin
    grid[lr0 : lr0 + lh, lc0 : lc0 + lw] = legend

    reserved = np.zeros_like(grid, dtype=bool)
    reserved[max(0, lr0 - 1) : min(h, lr0 + lh + 1), max(0, lc0 - 1) : min(w, lc0 + lw + 1)] = True

    n_polys = rng.randint(3, 5)
    side_counts = _choose_side_counts(rng, n_polys)
    colors = rng.sample(COLOR_POOL, n_polys)

    for color, n_sides in zip(colors, side_counts):
        mask = _make_polygon_mask(rng, n_sides)
        if not _place_mask(grid, reserved, mask, color, rng):
            return None

    try:
        out = main(grid)
        got_map, _legend_box = _read_legend(grid)
    except Exception:
        return None
    if got_map != expected_map:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
