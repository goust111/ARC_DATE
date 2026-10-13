import random
from collections import deque

import numpy as np


BG = 0
RING = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 300
SHAPE_ATTEMPTS = 200
GRID_MIN = 22
GRID_MAX = 30
MARGIN = 1
GAP = 2
K_CHOICES = (2, 3, 4)
K_WEIGHTS = (3, 4, 2)


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _flood_from_border(passable: np.ndarray) -> np.ndarray:
    h, w = passable.shape
    seen = np.zeros_like(passable, dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if passable[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and passable[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return seen


def _ring_inside_mask(ring_coords: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(ring_coords)
    local_h = r1 - r0 + 3
    local_w = c1 - c0 + 3
    wall = np.zeros((local_h, local_w), dtype=bool)
    for r, c in ring_coords:
        wall[int(r) - r0 + 1, int(c) - c0 + 1] = True
    outside = _flood_from_border(~wall)
    inside_local = (~wall) & (~outside)
    if not inside_local.any():
        raise AssertionError("expected a closed ring with a non-empty interior")
    inside = np.zeros(shape, dtype=bool)
    inside[r0 : r1 + 1, c0 : c1 + 1] = inside_local[1:-1, 1:-1]
    return inside


def _analyze_scene(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_coords = [(int(r), int(c)) for r, c in np.argwhere((g != BG) & (g != RING))]
    assert marker_coords, "expected at least one colored marker"
    marker_colors = [int(g[r, c]) for r, c in marker_coords]
    assert len(set(marker_colors)) == len(marker_colors), "marker colors must be unique"

    ring_comps = _components_from_mask(g == RING)
    assert ring_comps, "expected gray closed rings"

    rings = []
    for coords in ring_comps:
        inside_mask = _ring_inside_mask(coords, g.shape)
        sample = tuple(map(int, np.argwhere(inside_mask)[0]))
        rings.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
                "inside_mask": inside_mask,
                "sample": sample,
            }
        )

    n = len(rings)
    parents = [-1] * n
    for j in range(n):
        containers = []
        sr, sc = rings[j]["sample"]
        for i in range(n):
            if i == j:
                continue
            if bool(rings[i]["inside_mask"][sr, sc]):
                containers.append(i)
        if containers:
            parents[j] = min(containers, key=lambda i: rings[i]["area"])

    depths = [0] * n

    def ring_depth(idx: int) -> int:
        if depths[idx]:
            return depths[idx]
        parent = parents[idx]
        depths[idx] = 1 if parent == -1 else ring_depth(parent) + 1
        return depths[idx]

    for i in range(n):
        ring_depth(i)

    markers = []
    for r, c in marker_coords:
        containers = [i for i, ring in enumerate(rings) if bool(ring["inside_mask"][r, c])]
        assert containers, "every marker must be enclosed by at least one ring"
        chain_depths = sorted(depths[i] for i in containers)
        deepest = max(containers, key=lambda i: depths[i])
        d = depths[deepest]
        assert chain_depths == list(range(1, d + 1)), "ring containers must form a proper nesting chain"
        markers.append(
            {
                "coord": (r, c),
                "color": int(g[r, c]),
                "depth": int(d),
                "deepest_ring": int(deepest),
            }
        )

    marker_depths = sorted(m["depth"] for m in markers)
    assert marker_depths == list(range(1, len(markers) + 1)), "marker depths must be exactly 1..K"
    assert len({m["deepest_ring"] for m in markers}) == len(markers), "each group must have one marker"

    return {"rings": rings, "markers": markers, "ring_depths": depths, "shape": (h, w)}


def _build_output_from_markers(markers: list[dict]) -> np.ndarray:
    depth_to_color = {int(m["depth"]): int(m["color"]) for m in markers}
    dmax = len(markers)
    out = np.zeros((2 * dmax + 1, 2 * dmax + 1), dtype=int)
    center = dmax
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            dist = max(abs(r - center), abs(c - center))
            if dist == dmax:
                continue
            depth = dmax - dist
            out[r, c] = depth_to_color[depth]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each marker's nesting depth and redraw the depth order as concentric square rings.

    Input structure:
    - The grid uses black background and several spatially separated groups of gray one-pixel closed rings.
    - Each group is a strictly nested family of 4-connected contour loops.
    - Inside the innermost enclosed region of each group there is exactly one non-black, non-gray marker pixel.

    Rule:
    - Build the containment relation of all gray rings and determine, for each marker, how many ring boundaries enclose its region.
    - Let that nesting count be the marker depth, with the marker depths forming exactly 1..K across the scene.
    - Create a square output of size (2K+1) x (2K+1); keep the outermost Chebyshev-distance-K border black.
    - Fill every inner Chebyshev ring at distance K-d with the color of the marker whose nesting depth is d, so the deepest marker becomes the center.

    Key insight:
    - The input geometry is disposable: only the topological nesting depth of each marker-bearing ring group survives, and that depth order becomes the color order of concentric square rings.
    """
    analysis = _analyze_scene(np.asarray(grid, dtype=int))
    return _build_output_from_markers(analysis["markers"])


def _paint_rect_ring(grid: np.ndarray, r0: int, c0: int, h: int, w: int) -> None:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    grid[r0, c0 : c1 + 1] = RING
    grid[r1, c0 : c1 + 1] = RING
    grid[r0 : r1 + 1, c0] = RING
    grid[r0 : r1 + 1, c1] = RING


def _shape_pixel_count(depth: int, inner_h: int, inner_w: int) -> int:
    total = 0
    for layer in range(depth):
        h = inner_h + 4 * (depth - layer - 1)
        w = inner_w + 4 * (depth - layer - 1)
        total += 2 * h + 2 * w - 4
    return total


def _monotone_depth_order(values: list[int]) -> bool:
    if len(values) < 3:
        return False
    inc = all(values[i] < values[i + 1] for i in range(len(values) - 1))
    dec = all(values[i] > values[i + 1] for i in range(len(values) - 1))
    return inc or dec


def _choose_specs(k: int) -> list[dict] | None:
    depths = list(range(1, k + 1))
    colors = random.sample(MARKER_COLORS, k)
    random.shuffle(colors)
    for _ in range(SHAPE_ATTEMPTS):
        specs = []
        for depth, color in zip(depths, colors):
            inner_h = random.randint(3, 6)
            inner_w = random.randint(3, 7)
            if random.random() < 0.65:
                while inner_w == inner_h:
                    inner_w = random.randint(3, 7)
            outer_h = inner_h + 4 * (depth - 1)
            outer_w = inner_w + 4 * (depth - 1)
            if outer_h > 19 or outer_w > 19:
                break
            specs.append(
                {
                    "depth": depth,
                    "color": color,
                    "inner_h": inner_h,
                    "inner_w": inner_w,
                    "outer_h": outer_h,
                    "outer_w": outer_w,
                    "outer_area": outer_h * outer_w,
                    "ring_pixels": _shape_pixel_count(depth, inner_h, inner_w),
                }
            )
        if len(specs) != k:
            continue
        area_by_depth = [spec["outer_area"] for spec in specs]
        pixels_by_depth = [spec["ring_pixels"] for spec in specs]
        if _monotone_depth_order(area_by_depth) and _monotone_depth_order(pixels_by_depth):
            continue
        return specs
    return None


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_specs(specs: list[dict], h: int, w: int) -> bool:
    placed_boxes: list[tuple[int, int, int, int]] = []
    random.shuffle(specs)
    for spec in specs:
        gh = spec["outer_h"]
        gw = spec["outer_w"]
        ok = False
        for _ in range(PLACE_ATTEMPTS):
            r0 = random.randint(MARGIN, h - gh - MARGIN)
            c0 = random.randint(MARGIN, w - gw - MARGIN)
            box = (r0, c0, r0 + gh - 1, c0 + gw - 1)
            if all(_separated(box, other, GAP) for other in placed_boxes):
                spec["top"] = r0
                spec["left"] = c0
                placed_boxes.append(box)
                ok = True
                break
        if not ok:
            return False
    return True


def _paint_scene(specs: list[dict], h: int, w: int) -> tuple[np.ndarray, list[dict]]:
    grid = np.zeros((h, w), dtype=int)
    meta = []
    for spec in specs:
        r0 = spec["top"]
        c0 = spec["left"]
        for layer in range(spec["depth"]):
            rr = r0 + 2 * layer
            cc = c0 + 2 * layer
            hh = spec["outer_h"] - 4 * layer
            ww = spec["outer_w"] - 4 * layer
            _paint_rect_ring(grid, rr, cc, hh, ww)
        ir0 = r0 + 2 * (spec["depth"] - 1) + 1
        ic0 = c0 + 2 * (spec["depth"] - 1) + 1
        ih = spec["inner_h"] - 2
        iw = spec["inner_w"] - 2
        mr = random.randint(ir0, ir0 + ih - 1)
        mc = random.randint(ic0, ic0 + iw - 1)
        grid[mr, mc] = spec["color"]
        meta.append(
            {
                "depth": spec["depth"],
                "color": spec["color"],
                "marker": (mr, mc),
                "outer_box": (r0, c0, r0 + spec["outer_h"] - 1, c0 + spec["outer_w"] - 1),
                "outer_area": spec["outer_area"],
            }
        )
    return grid, meta


def _placement_not_spatial_shortcut(meta: list[dict]) -> bool:
    if len(meta) < 3:
        return True
    by_depth = sorted(meta, key=lambda x: x["depth"])
    row_centers = [m["outer_box"][0] + m["outer_box"][2] for m in by_depth]
    col_centers = [m["outer_box"][1] + m["outer_box"][3] for m in by_depth]
    row_mono = all(row_centers[i] < row_centers[i + 1] for i in range(len(row_centers) - 1)) or all(
        row_centers[i] > row_centers[i + 1] for i in range(len(row_centers) - 1)
    )
    col_mono = all(col_centers[i] < col_centers[i + 1] for i in range(len(col_centers) - 1)) or all(
        col_centers[i] > col_centers[i + 1] for i in range(len(col_centers) - 1)
    )
    return not (row_mono or col_mono)


def _validate_generated(grid: np.ndarray, meta: list[dict]) -> bool:
    try:
        analysis = _analyze_scene(grid)
    except AssertionError:
        return False

    got = sorted((m["depth"], m["color"], m["coord"]) for m in analysis["markers"])
    want = sorted((m["depth"], m["color"], m["marker"]) for m in meta)
    if got != want:
        return False

    expected = _build_output_from_markers(analysis["markers"])
    if not np.array_equal(main(grid), expected):
        return False

    out = expected
    dmax = len(meta)
    if out.shape != (2 * dmax + 1, 2 * dmax + 1):
        return False
    center = dmax
    depth_to_color = {m["depth"]: m["color"] for m in meta}
    for r in range(out.shape[0]):
        for c in range(out.shape[1]):
            dist = max(abs(r - center), abs(c - center))
            if dist == dmax:
                if out[r, c] != BG:
                    return False
            else:
                if out[r, c] != depth_to_color[dmax - dist]:
                    return False

    return True


def _generate_attempt() -> np.ndarray | None:
    k = random.choices(K_CHOICES, weights=K_WEIGHTS, k=1)[0]
    specs = _choose_specs(k)
    if specs is None:
        return None

    max_h = max(spec["outer_h"] for spec in specs)
    max_w = max(spec["outer_w"] for spec in specs)
    min_h = max(GRID_MIN, max_h + 2 * MARGIN + 1)
    min_w = max(GRID_MIN, max_w + 2 * MARGIN + 1)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None

    for _ in range(PLACE_ATTEMPTS):
        h = random.randint(min_h, GRID_MAX)
        w = random.randint(min_w, GRID_MAX)
        trial_specs = [dict(spec) for spec in specs]
        if not _place_specs(trial_specs, h, w):
            continue
        grid, meta = _paint_scene(trial_specs, h, w)
        if not _placement_not_spatial_shortcut(meta):
            continue
        if not _validate_generated(grid, meta):
            continue
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth ARC input")
