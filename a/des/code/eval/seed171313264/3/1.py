import random
from collections import deque

import numpy as np


CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in CARDINALS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _dilate8(mask: np.ndarray) -> np.ndarray:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    padded = np.pad(mask, 1)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + h, dc : dc + w]
    return out


def _nonzero_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return _bbox(coords)


def _pattern_for_coords(coords: np.ndarray, pattern: np.ndarray, color: int) -> tuple[bool, bool]:
    ph, pw = pattern.shape
    values = pattern[coords[:, 0] % ph, coords[:, 1] % pw]
    return bool(np.any(values == color)), bool(np.any(values == 0))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each grey-walled region with the pattern from the same-colored legend block.

    The legend's interior pattern is tiled with global alignment, so region cells
    at (r, c) read pattern[r % ph, c % pw].
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = int(np.bincount(g.reshape(-1)).argmax())
    colors = [int(c) for c in np.unique(g) if int(c) != bg]

    seed_colors: list[int] = []
    color_components: dict[int, list[np.ndarray]] = {}
    for color in colors:
        comps = _components(g == color)
        color_components[color] = comps
        if any(len(comp) == 1 for comp in comps):
            seed_colors.append(color)

    wall_candidates = [color for color in colors if color not in seed_colors]
    assert len(wall_candidates) == 1, "expected exactly one wall color"
    wall = wall_candidates[0]

    legends: dict[int, np.ndarray] = {}
    legend_mask = np.zeros_like(g, dtype=bool)
    for color in seed_colors:
        comps = color_components[color]
        seeds = [comp for comp in comps if len(comp) == 1]
        legend_parts = [comp for comp in comps if len(comp) > 1]
        assert len(seeds) == 1, "each seed color must appear as one singleton"
        assert len(legend_parts) == 1, "each seed color must have one legend block"

        r0, c0, r1, c1 = _bbox(legend_parts[0])
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = sub.shape
        assert h == w and h in (4, 5), "legend block must be 4x4 or 5x5"
        assert np.all(sub[0, :] == color) and np.all(sub[-1, :] == color)
        assert np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color)
        assert set(map(int, np.unique(sub))) <= {bg, color}

        legends[color] = sub[1:-1, 1:-1].copy()
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    assert legends, "expected at least one legend block"

    seed_mask = np.isin(g, seed_colors) & ~legend_mask
    passable = ((g == bg) | seed_mask) & ~legend_mask
    out = g.copy()

    for comp in _components(passable):
        rows = comp[:, 0]
        cols = comp[:, 1]
        if np.any(rows == 0) or np.any(cols == 0) or np.any(rows == g.shape[0] - 1) or np.any(cols == g.shape[1] - 1):
            continue

        comp_colors = sorted({int(g[r, c]) for r, c in comp if int(g[r, c]) != bg})
        assert len(comp_colors) == 1, "each enclosed region must contain exactly one seed"
        color = comp_colors[0]
        pattern = legends[color]
        ph, pw = pattern.shape
        out[rows, cols] = pattern[rows % ph, cols % pw]

    return out


# --------------- generator ---------------

GEN_BG = 0
GEN_WALL = 5
GEN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
LEGEND_RESERVE = 6
MAX_REGIONS = 4

REGION_CORE_H_RANGE = (3, 6)
REGION_CORE_W_RANGE = (3, 6)
REGION_BLOB_ATTEMPTS = 80
REGION_SHAPE_ATTEMPTS = 120
REGION_PLACEMENT_ATTEMPTS = 160
GEN_ATTEMPTS = 400

LEGEND_GAPS = (1, 2)

PATTERN_LIBRARY = {
    2: (
        np.array([[1, 0], [0, 1]], dtype=int),
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 0]], dtype=int),
        np.array([[1, 0], [1, 0]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [1, 1]], dtype=int),
    ),
    3: (
        np.array([[1, 1, 1], [1, 0, 0], [1, 0, 0]], dtype=int),
        np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 1], [1, 0, 1], [1, 0, 1]], dtype=int),
        np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 1, 0], [1, 0, 0], [1, 1, 1]], dtype=int),
        np.array([[1, 0, 0], [1, 1, 0], [1, 0, 0]], dtype=int),
    ),
}


def _random_blob(height: int, width: int, target: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (rng.randrange(height), rng.randrange(width))
    mask[start] = True
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in CARDINALS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                frontier.add((nr, nc))

    add_frontier(*start)
    while int(mask.sum()) < target and frontier:
        cells = list(frontier)
        weights = []
        for r, c in cells:
            deg = 0
            for dr, dc in CARDINALS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and mask[nr, nc]:
                    deg += 1
            weights.append(max(1, deg * deg))
        r, c = rng.choices(cells, weights=weights, k=1)[0]
        frontier.remove((r, c))
        mask[r, c] = True
        add_frontier(r, c)
    return mask


def _make_region_patch(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(REGION_SHAPE_ATTEMPTS):
        height = rng.randint(*REGION_CORE_H_RANGE)
        width = rng.randint(*REGION_CORE_W_RANGE)
        area_min = max(4, (height * width) // 3)
        area_max = max(area_min, height * width - 1)
        target = rng.randint(area_min, area_max)

        core = None
        for _ in range(REGION_BLOB_ATTEMPTS):
            blob = _random_blob(height, width, target, rng)
            bbox = _nonzero_bbox(blob)
            if bbox is None:
                continue
            r0, c0, r1, c1 = bbox
            blob = blob[r0 : r1 + 1, c0 : c1 + 1]
            if min(blob.shape) < 2 or int(blob.sum()) < 4:
                continue
            bbox_area = blob.shape[0] * blob.shape[1]
            if bbox_area == int(blob.sum()) and rng.random() < 0.8:
                continue
            core = blob
            break
        if core is None:
            continue

        padded = np.pad(core, 1)
        outer = _dilate8(padded)
        boundary = outer & ~padded
        if len(_components(boundary)) != 1:
            continue

        rr0, cc0, rr1, cc1 = _bbox(np.argwhere(outer))
        return padded[rr0 : rr1 + 1, cc0 : cc1 + 1], boundary[rr0 : rr1 + 1, cc0 : cc1 + 1]
    return None


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_regions(specs: list[dict], region_h: int, region_w: int, rng: random.Random) -> bool:
    occupied: list[tuple[int, int, int, int]] = []
    for spec in sorted(specs, key=lambda item: item["area"], reverse=True):
        patch_h, patch_w = spec["core"].shape
        max_top = region_h - patch_h - 1
        max_left = region_w - patch_w - 1
        if max_top < 1 or max_left < 1:
            return False

        placed = False
        for _ in range(REGION_PLACEMENT_ATTEMPTS):
            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            padded_rect = (top - 1, left - 1, top + patch_h, left + patch_w)
            if any(_rects_overlap(padded_rect, other) for other in occupied):
                continue
            spec["top"] = top
            spec["left"] = left
            occupied.append(padded_rect)
            placed = True
            break
        if not placed:
            return False
    return True


def _choose_pattern(coords: np.ndarray, color: int, rng: random.Random) -> np.ndarray | None:
    candidates: list[np.ndarray] = []
    sizes = [2, 3]
    rng.shuffle(sizes)
    for size in sizes:
        bases = list(PATTERN_LIBRARY[size])
        rng.shuffle(bases)
        for base in bases:
            pattern = np.where(base == 1, color, GEN_BG).astype(int)
            has_color, has_bg = _pattern_for_coords(coords, pattern, color)
            if has_color and has_bg:
                candidates.append(pattern)
    if not candidates:
        return None
    return rng.choice(candidates).copy()


def _legend_block(color: int, pattern: np.ndarray) -> np.ndarray:
    ph, pw = pattern.shape
    block = np.full((ph + 2, pw + 2), color, dtype=int)
    block[1:-1, 1:-1] = pattern
    return block


def _place_legends(grid: np.ndarray, legends: list[dict], side: str, rng: random.Random) -> bool:
    order = legends[:]
    rng.shuffle(order)
    if side == "bottom":
        gaps = [rng.choice(LEGEND_GAPS) for _ in range(max(0, len(order) - 1))]
        total_w = sum(item["block"].shape[1] for item in order) + sum(gaps)
        if total_w > grid.shape[1]:
            return False
        col = rng.randint(0, grid.shape[1] - total_w)
        for idx, item in enumerate(order):
            block = item["block"]
            bh, bw = block.shape
            top = grid.shape[0] - bh
            if np.any(grid[top : top + bh, col : col + bw] != GEN_BG):
                return False
            grid[top : top + bh, col : col + bw] = block
            if idx < len(gaps):
                col += bw + gaps[idx]
    else:
        gaps = [rng.choice(LEGEND_GAPS) for _ in range(max(0, len(order) - 1))]
        total_h = sum(item["block"].shape[0] for item in order) + sum(gaps)
        if total_h > grid.shape[0]:
            return False
        row = rng.randint(0, grid.shape[0] - total_h)
        for idx, item in enumerate(order):
            block = item["block"]
            bh, bw = block.shape
            left = grid.shape[1] - bw
            if np.any(grid[row : row + bh, left : left + bw] != GEN_BG):
                return False
            grid[row : row + bh, left : left + bw] = block
            if idx < len(gaps):
                row += bh + gaps[idx]
    return True


def _build_expected(grid: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = grid.copy()
    for spec in specs:
        coords = spec["coords"]
        pattern = spec["pattern"]
        ph, pw = pattern.shape
        rows = coords[:, 0]
        cols = coords[:, 1]
        out[rows, cols] = pattern[rows % ph, cols % pw]
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_H_RANGE)
    width = rng.randint(*GRID_W_RANGE)
    side = rng.choice(("bottom", "right"))

    legend_capacity = width if side == "bottom" else height
    max_regions = min(MAX_REGIONS, max(1, (legend_capacity + 1) // 6))
    n_regions = rng.randint(1, max_regions)
    colors = rng.sample(list(GEN_COLORS), n_regions)

    specs: list[dict] = []
    for color in colors:
        patch = _make_region_patch(rng)
        if patch is None:
            return None
        core, boundary = patch
        specs.append(
            {
                "color": color,
                "core": core,
                "boundary": boundary,
                "area": int(core.sum()),
            }
        )

    region_h = height - LEGEND_RESERVE if side == "bottom" else height
    region_w = width - LEGEND_RESERVE if side == "right" else width
    if not _place_regions(specs, region_h, region_w, rng):
        return None

    grid = np.full((height, width), GEN_BG, dtype=int)
    legends: list[dict] = []
    for spec in specs:
        core = spec["core"]
        boundary = spec["boundary"]
        top = spec["top"]
        left = spec["left"]
        patch_h, patch_w = core.shape

        region_view = grid[top : top + patch_h, left : left + patch_w]
        region_view[boundary] = GEN_WALL

        coords = np.argwhere(core) + np.array([top, left], dtype=int)
        pattern = _choose_pattern(coords, spec["color"], rng)
        if pattern is None:
            return None

        seed_r, seed_c = map(int, coords[rng.randrange(len(coords))])
        grid[seed_r, seed_c] = spec["color"]

        spec["coords"] = coords
        spec["seed"] = (seed_r, seed_c)
        spec["pattern"] = pattern
        legends.append({"color": spec["color"], "block": _legend_block(spec["color"], pattern)})

    if not _place_legends(grid, legends, side, rng):
        return None

    expected = _build_expected(grid, specs)
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None
    if np.array_equal(grid, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
