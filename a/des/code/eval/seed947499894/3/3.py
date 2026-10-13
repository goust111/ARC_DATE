import random
from collections import Counter, deque

import numpy as np


# --------------- shared helpers ---------------

BG_COLOR = 0
GRID_H_RANGE = (18, 38)
GRID_W_RANGE = (18, 38)
DEPTH_OPTIONS = (2, 3, 4)
DEPTH_WEIGHTS = (4, 4, 2)
GROUP_COUNT_OPTIONS = (1, 2, 3)
GROUP_COUNT_WEIGHTS = (3, 3, 2)
COLOR_POOL = tuple(range(1, 10))
PLACEMENT_TRIES = 120
GEN_RETRIES = 300

PLUS_BASE = np.array(
    [
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
    ],
    dtype=bool,
)

_PLUS_BOUNDARY_CACHE: dict[int, np.ndarray] = {}


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(h: int, w: int, r: int, c: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    if connectivity == 4:
        neighbor_offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        neighbor_offsets = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]
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
                for dr, dc in neighbor_offsets:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    eroded = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            keep = True
            for nr, nc in _neighbors4(h, w, r, c):
                if not mask[nr, nc]:
                    keep = False
                    break
            if keep and r not in (0, h - 1) and c not in (0, w - 1):
                eroded[r, c] = True
    return eroded


def _boundary_from_filled(filled: np.ndarray) -> np.ndarray:
    return filled & (~_erode4(filled))


def _interior_from_boundary(boundary: np.ndarray) -> np.ndarray:
    h, w = boundary.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()

    def maybe_push(r: int, c: int) -> None:
        if not boundary[r, c] and not outside[r, c]:
            outside[r, c] = True
            q.append((r, c))

    for r in range(h):
        maybe_push(r, 0)
        maybe_push(r, w - 1)
    for c in range(w):
        maybe_push(0, c)
        maybe_push(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(h, w, r, c):
            if not boundary[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~outside) & (~boundary)


def _legend_mask_and_colors(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    h, w = grid.shape
    candidates: list[tuple[np.ndarray, list[int]]] = []
    specs = [
        (range(0, h), 0),
        (range(h - 1, -1, -1), 0),
        (range(0, h), w - 1),
        (range(h - 1, -1, -1), w - 1),
    ]

    for row_iter, col in specs:
        rows = list(row_iter)
        if int(grid[rows[0], col]) == bg:
            continue

        seg = []
        for r in rows:
            if int(grid[r, col]) == bg:
                break
            seg.append(r)
        if len(seg) < 2:
            continue

        top = min(seg)
        bottom = max(seg)
        inward_col = 1 if col == 0 else w - 2
        ok = True

        for r in range(h):
            if r < top or r > bottom:
                if int(grid[r, col]) != bg:
                    ok = False
                    break
        if not ok:
            continue

        for rr in range(max(0, top - 1), min(h, bottom + 2)):
            if int(grid[rr, inward_col]) != bg:
                ok = False
                break
        if not ok:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r in seg:
            mask[r, col] = True
        colors = [int(grid[r, col]) for r in sorted(seg)]
        candidates.append((mask, colors))

    assert len(candidates) == 1, f"expected one corner legend, got {len(candidates)}"
    return candidates[0]


def _global_inside_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    rows = coords[:, 0]
    cols = coords[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())

    local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    local[rows - r0, cols - c0] = True
    inside_local = _interior_from_boundary(local)
    assert inside_local.any(), "each contour must enclose a non-empty interior"

    inside = np.zeros(shape, dtype=bool)
    inside[r0 : r1 + 1, c0 : c1 + 1] = inside_local
    return inside


# --------------- solver ---------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested single-color contour groups by depth and fill each innermost cavity.

    The solver is contour-generic:
    - detect the corner legend structurally,
    - find all remaining closed outline components,
    - infer nesting depth from contour containment,
    - recolor each outline by its depth,
    - fill the cavity enclosed by contours with no deeper descendants.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend_mask, legend_colors = _legend_mask_and_colors(g, bg)

    contour_mask = (g != bg) & (~legend_mask)
    contours = _components(contour_mask, connectivity=8)
    assert contours, "expected at least one contour"

    inside_masks = [_global_inside_mask(g.shape, comp) for comp in contours]

    depths = []
    for i, comp in enumerate(contours):
        depth = 1
        for j, inside in enumerate(inside_masks):
            if i == j:
                continue
            if np.all(inside[comp[:, 0], comp[:, 1]]):
                depth += 1
        depths.append(depth)

    max_depth = max(depths)
    assert max_depth == len(legend_colors), "legend length must match maximum nesting depth"

    out = g.copy()
    for comp, depth in zip(contours, depths):
        assert 1 <= depth <= len(legend_colors)
        out[comp[:, 0], comp[:, 1]] = legend_colors[depth - 1]

    for i, (inside, depth) in enumerate(zip(inside_masks, depths)):
        has_descendant = False
        for j, other in enumerate(contours):
            if i == j:
                continue
            if np.all(inside[other[:, 0], other[:, 1]]):
                has_descendant = True
                break
        if not has_descendant:
            fill_mask = inside & (g == bg)
            out[fill_mask] = legend_colors[depth - 1]

    return out


# --------------- generator ---------------

def _plus_boundary(scale: int) -> np.ndarray:
    cached = _PLUS_BOUNDARY_CACHE.get(scale)
    if cached is not None:
        return cached.copy()

    scaled = np.kron(PLUS_BASE, np.ones((scale, scale), dtype=bool))
    boundary = _boundary_from_filled(scaled)
    _PLUS_BOUNDARY_CACHE[scale] = boundary
    return boundary.copy()


def _make_rect_group(depth: int, rng: random.Random) -> dict:
    inner_h = rng.randint(3, 7)
    inner_w = rng.randint(3, 7)
    h = inner_h + 4 * (depth - 1)
    w = inner_w + 4 * (depth - 1)
    layers: list[np.ndarray] = []

    for layer in range(depth):
        top = 2 * layer
        bottom = h - 1 - 2 * layer
        left = 2 * layer
        right = w - 1 - 2 * layer
        mask = np.zeros((h, w), dtype=bool)
        mask[top, left : right + 1] = True
        mask[bottom, left : right + 1] = True
        mask[top : bottom + 1, left] = True
        mask[top : bottom + 1, right] = True
        layers.append(mask)

    fill = _interior_from_boundary(layers[-1])
    return {"family": "rect", "h": h, "w": w, "layers": layers, "fill": fill}


def _make_plus_group(depth: int, rng: random.Random) -> dict:
    assert depth in (2, 3)
    if depth == 3:
        inner_scale = rng.choice((1, 2))
    else:
        inner_scale = rng.choice((1, 2, 3))

    outer_scale = inner_scale + 2 * (depth - 1)
    size = 5 * outer_scale
    layers: list[np.ndarray] = []

    for layer in range(depth):
        scale = outer_scale - 2 * layer
        boundary = _plus_boundary(scale)
        h, w = boundary.shape
        top = (size - h) // 2
        left = (size - w) // 2
        mask = np.zeros((size, size), dtype=bool)
        mask[top : top + h, left : left + w] = boundary
        layers.append(mask)

    fill = _interior_from_boundary(layers[-1])
    return {"family": "plus", "h": size, "w": size, "layers": layers, "fill": fill}


def _legend_layout(h: int, w: int, depth: int, corner: str) -> tuple[np.ndarray, list[tuple[int, int]]]:
    mask = np.zeros((h, w), dtype=bool)
    reserve = np.zeros((h, w), dtype=bool)

    if "top" in corner:
        rows = list(range(depth))
    else:
        rows = list(range(h - depth, h))
    col = 0 if "left" in corner else w - 1

    cells = [(r, col) for r in rows]
    for r, c in cells:
        mask[r, c] = True
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                reserve[rr, cc] = True
    return reserve, cells


def _choose_grid(specs: list[dict], depth: int, rng: random.Random) -> tuple[int, int] | None:
    max_h = max(spec["h"] for spec in specs)
    max_w = max(spec["w"] for spec in specs)
    area_hint = int(sum((spec["h"] + 2) * (spec["w"] + 2) for spec in specs) ** 0.5) + 4
    min_h = max(GRID_H_RANGE[0], max_h + 2, area_hint)
    min_w = max(GRID_W_RANGE[0], max_w + 2, area_hint)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    if min(h, w) <= depth + 2:
        return None
    return h, w


def _place_groups(
    h: int,
    w: int,
    specs: list[dict],
    reserve: np.ndarray,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    occupied = reserve.copy()
    placements: list[tuple[int, int] | None] = [None] * len(specs)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["h"] * specs[i]["w"], reverse=True)

    for idx in order:
        spec = specs[idx]
        gh = spec["h"]
        gw = spec["w"]
        low_r = 1
        high_r = h - gh - 1
        low_c = 1
        high_c = w - gw - 1
        if high_r < low_r or high_c < low_c:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = rng.randint(low_r, high_r)
            left = rng.randint(low_c, high_c)
            rr0 = max(0, top - 1)
            rr1 = min(h, top + gh + 1)
            cc0 = max(0, left - 1)
            cc1 = min(w, left + gw + 1)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            occupied[rr0:rr1, cc0:cc1] = True
            placements[idx] = (top, left)
            placed = True
            break
        if not placed:
            return None

    return [p for p in placements if p is not None]


def _paint_group(target: np.ndarray, spec: dict, top: int, left: int, colors: list[int]) -> None:
    sub = target[top : top + spec["h"], left : left + spec["w"]]
    for layer, color in zip(spec["layers"], colors):
        sub[layer] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    depth = rng.choices(DEPTH_OPTIONS, weights=DEPTH_WEIGHTS, k=1)[0]
    group_count = rng.choices(GROUP_COUNT_OPTIONS, weights=GROUP_COUNT_WEIGHTS, k=1)[0]

    outline_color = rng.choice(COLOR_POOL)
    available = [c for c in COLOR_POOL if c != outline_color]
    legend_colors = rng.sample(available, depth)

    specs: list[dict] = []
    for _ in range(group_count):
        use_plus = depth <= 3 and (group_count == 1 or depth == 2) and rng.random() < 0.45
        if use_plus:
            specs.append(_make_plus_group(depth, rng))
        else:
            specs.append(_make_rect_group(depth, rng))

    grid_shape = _choose_grid(specs, depth, rng)
    if grid_shape is None:
        return None
    h, w = grid_shape

    corner = rng.choice(("top_left", "top_right", "bottom_left", "bottom_right"))
    legend_reserve, legend_cells = _legend_layout(h, w, depth, corner)
    placements = _place_groups(h, w, specs, legend_reserve, rng)
    if placements is None:
        return None

    grid = np.full((h, w), BG_COLOR, dtype=int)
    expected = grid.copy()

    for spec, (top, left) in zip(specs, placements):
        _paint_group(grid, spec, top, left, [outline_color] * depth)
        _paint_group(expected, spec, top, left, legend_colors)
        fill = spec["fill"]
        expected[top : top + spec["h"], left : left + spec["w"]][fill] = legend_colors[-1]

    for (r, c), color in zip(sorted(legend_cells), legend_colors):
        grid[r, c] = color
        expected[r, c] = color

    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
