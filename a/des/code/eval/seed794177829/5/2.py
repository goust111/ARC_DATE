import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG_COLOR = 8
GEN_RETRIES = 300
GRID_MIN = 16
GRID_MAX = 40
SCALE_OPTIONS = {
    2: [(1, 2), (1, 3), (1, 4), (1, 5)],
    3: [(1, 2, 3), (1, 2, 4), (1, 3, 4), (1, 2, 5)],
    4: [(1, 2, 3, 4)],
}


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _scale_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=int)
    return np.kron(mask.astype(int), block).astype(bool)


def _scale_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    block = np.ones((factor, factor), dtype=int)
    return np.kron(grid, block)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _touches_all_sides(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _mask_connected(mask: np.ndarray) -> bool:
    if not mask.any():
        return False
    return len(_components_mask(mask)) == 1


def _mask_has_hole(mask: np.ndarray) -> bool:
    free = ~mask
    h, w = free.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return bool((free & ~outside).any())


def _enclosed_region_mask(wall_mask: np.ndarray) -> np.ndarray:
    free = ~wall_mask
    h, w = wall_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if free[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return free & ~outside


def _extract_outline_objects_for_color(
    grid: np.ndarray, border_color: int, bg: int
) -> list[dict]:
    objects: list[dict] = []
    for comp in _components_mask(grid == border_color):
        r0, c0, r1, c1 = _bbox(comp)
        wall = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            wall[r - r0, c - c0] = True

        hole_mask_full = _enclosed_region_mask(wall)
        if not hole_mask_full.any():
            continue
        hole_comps = _components_mask(hole_mask_full)
        if len(hole_comps) != 1:
            continue

        hr0, hc0, hr1, hc1 = _bbox(hole_comps[0])
        hole_mask = hole_mask_full[hr0 : hr1 + 1, hc0 : hc1 + 1].copy()
        hole_patch = grid[r0 + hr0 : r0 + hr1 + 1, c0 + hc0 : c0 + hc1 + 1].copy()
        hole_patch[~hole_mask] = bg
        interior_colors = sorted(int(v) for v in np.unique(hole_patch[hole_mask]) if int(v) != bg)

        objects.append(
            {
                "bbox": (r0, c0, r1, c1),
                "outer_h": int(r1 - r0 + 1),
                "outer_w": int(c1 - c0 + 1),
                "border_size": int(len(comp)),
                "hole_bbox_abs": (r0 + hr0, c0 + hc0, r0 + hr1, c0 + hc1),
                "hole_mask": hole_mask,
                "hole_patch": hole_patch,
                "interior_colors": interior_colors,
            }
        )
    return objects


def _infer_scale(source_mask: np.ndarray, target_mask: np.ndarray) -> int | None:
    sh, sw = source_mask.shape
    th, tw = target_mask.shape
    if th % sh != 0 or tw % sw != 0:
        return None
    mr, mc = th // sh, tw // sw
    if mr != mc or mr < 1:
        return None
    factor = int(mr)
    if not np.array_equal(target_mask, _scale_mask(source_mask, factor)):
        return None
    return factor


def _select_outline_family(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    best: tuple[tuple[int, int, int], int, list[dict]] | None = None
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        objects = _extract_outline_objects_for_color(grid, color, bg)
        if len(objects) < 2:
            continue
        objects.sort(key=lambda obj: (obj["outer_h"] * obj["outer_w"], obj["outer_h"], obj["outer_w"], obj["bbox"]))
        source = objects[0]
        if not source["interior_colors"]:
            continue
        if any(obj["interior_colors"] for obj in objects[1:]):
            continue

        scales: list[int] = []
        ok = True
        for obj in objects:
            factor = _infer_scale(source["hole_mask"], obj["hole_mask"])
            if factor is None:
                ok = False
                break
            scales.append(int(factor))
        if not ok or scales[0] != 1 or scales != sorted(scales) or len(set(scales)) != len(scales):
            continue

        score = (len(objects), sum(obj["border_size"] for obj in objects), -source["outer_h"] * source["outer_w"])
        if best is None or score > best[0]:
            best = (score, color, objects)

    assert best is not None, "expected one outline family with a single colored source and larger empty copies"
    return best[1], best[2]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find several same-shape outline objects of different sizes and propagate the
    smallest object's interior pattern into the larger ones by integer nearest-
    neighbor scaling.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    _border_color, objects = _select_outline_family(g, bg)
    source = objects[0]
    source_mask = source["hole_mask"]
    source_pattern = source["hole_patch"]

    out = g.copy()
    for obj in objects[1:]:
        factor = _infer_scale(source_mask, obj["hole_mask"])
        assert factor is not None and factor > 1
        scaled_pattern = _scale_grid(source_pattern, factor)
        r0, c0, r1, c1 = obj["hole_bbox_abs"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[obj["hole_mask"]] = scaled_pattern[obj["hole_mask"]]
    return out


def _random_interior_mask(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    mask = np.ones((h, w), dtype=bool)
    max_remove = max(0, (h * w) // 3)
    remove_goal = 0 if max_remove == 0 or rng.random() < 0.35 else rng.randint(1, max_remove)
    attempts = 0
    removed = 0
    cells = [(r, c) for r in range(h) for c in range(w)]
    while removed < remove_goal and attempts < h * w * 10:
        attempts += 1
        r, c = rng.choice(cells)
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if int(trial.sum()) < 4:
            continue
        if not _touches_all_sides(trial):
            continue
        if not _mask_connected(trial):
            continue
        if _mask_has_hole(trial):
            continue
        mask = trial
        removed += 1

    k = rng.randint(0, 3)
    if k:
        mask = np.rot90(mask, k)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _random_pattern(mask: np.ndarray, bg: int, border_color: int, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    patch = np.full((h, w), bg, dtype=int)
    cells = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    colors_pool = [c for c in range(10) if c not in (bg, border_color)]
    n_colors = min(len(cells), rng.randint(2, min(4, len(cells))))
    colors = rng.sample(colors_pool, n_colors)
    rng.shuffle(cells)
    for (r, c), color in zip(cells, colors):
        patch[r, c] = int(color)
    for r, c in cells[n_colors:]:
        patch[r, c] = int(rng.choice(colors))
    return patch


def _build_object_patch(
    base_mask: np.ndarray,
    base_pattern: np.ndarray,
    scale: int,
    bg: int,
    border_color: int,
    fill_interior: bool,
) -> np.ndarray:
    inner = _scale_mask(base_mask, scale)
    inner_canvas = np.pad(inner, 1, constant_values=False)
    border = _dilate8(inner_canvas) & ~inner_canvas
    patch = np.full(inner_canvas.shape, bg, dtype=int)
    patch[border] = border_color
    if fill_interior:
        scaled_pattern = _scale_grid(base_pattern, scale)
        inner_view = patch[1 : 1 + inner.shape[0], 1 : 1 + inner.shape[1]]
        inner_view[inner] = scaled_pattern[inner]
    return patch


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_patches(
    dims: list[tuple[int, int]], h: int, w: int, rng: random.Random, gap: int = 1
) -> list[tuple[int, int]] | None:
    placements: list[tuple[int, int] | None] = [None] * len(dims)
    blocked: list[tuple[int, int, int, int]] = []
    order = sorted(range(len(dims)), key=lambda i: dims[i][0] * dims[i][1], reverse=True)
    for idx in order:
        ph, pw = dims[idx]
        r_lo, r_hi = 1, h - ph - 1
        c_lo, c_hi = 1, w - pw - 1
        if r_hi < r_lo or c_hi < c_lo:
            return None
        placed = False
        for _ in range(300):
            r0 = rng.randint(r_lo, r_hi)
            c0 = rng.randint(c_lo, c_hi)
            rect = (r0 - gap, c0 - gap, r0 + ph - 1 + gap, c0 + pw - 1 + gap)
            if any(_rectangles_overlap(rect, other) for other in blocked):
                continue
            placements[idx] = (r0, c0)
            blocked.append(rect)
            placed = True
            break
        if not placed:
            return None
    return [pos for pos in placements if pos is not None]


def _choose_scales(base_mask: np.ndarray, rng: random.Random) -> tuple[int, ...]:
    n_objects = rng.choice([2, 3, 4])
    options = list(SCALE_OPTIONS[n_objects])
    rng.shuffle(options)
    max_dim = max(base_mask.shape)
    for scales in options:
        largest_patch = max_dim * max(scales) + 2
        if largest_patch <= 28 or n_objects == 2:
            return scales
    return options[0]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = BG_COLOR
    base_mask = _random_interior_mask(rng)
    scales = _choose_scales(base_mask, rng)
    border_color = rng.choice([c for c in range(10) if c != bg])
    base_pattern = _random_pattern(base_mask, bg, border_color, rng)

    patches = [
        _build_object_patch(base_mask, base_pattern, scale, bg, border_color, fill_interior=(i == 0))
        for i, scale in enumerate(scales)
    ]
    dims = [patch.shape for patch in patches]
    max_h = max(h for h, _ in dims)
    max_w = max(w for _, w in dims)
    padded_area = sum((h + 2) * (w + 2) for h, w in dims)
    side = int(np.ceil(np.sqrt(padded_area)))
    h_low = max(GRID_MIN, max_h + 2, side)
    w_low = max(GRID_MIN, max_w + 2, side)
    h_high = min(GRID_MAX, max(h_low, side + 10))
    w_high = min(GRID_MAX, max(w_low, side + 10))
    if h_low > GRID_MAX or w_low > GRID_MAX:
        return None

    h = rng.randint(h_low, h_high)
    w = rng.randint(w_low, w_high)
    placements = _place_patches(dims, h, w, rng, gap=1)
    if placements is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for patch, (r0, c0) in zip(patches, placements):
        ph, pw = patch.shape
        view = grid[r0 : r0 + ph, c0 : c0 + pw]
        mask = patch != bg
        if np.any(view[mask] != bg):
            return None
        view[mask] = patch[mask]

    try:
        detected_color, objects = _select_outline_family(grid, bg)
        out = main(grid)
    except AssertionError:
        return None

    if detected_color != border_color or len(objects) != len(scales):
        return None
    if np.array_equal(out, grid):
        return None
    for obj in objects[1:]:
        r0, c0, r1, c1 = obj["hole_bbox_abs"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        if np.any(patch[obj["hole_mask"]] == bg):
            return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
