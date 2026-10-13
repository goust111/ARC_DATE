import random
from collections import deque

import numpy as np


GREY = 5
GEN_RETRIES = 300
GRID_MIN = 18
GRID_MAX = 30
REGION_COUNT_CHOICES = (1, 2, 3)
SHAPE_H_RANGE = (6, 10)
SHAPE_W_RANGE = (6, 10)
TEMPLATE_SIZES = (3, 4)
EDGE_SIDES = ("top", "bottom", "left", "right")
N4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mode_color(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
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
                for dr, dc in N4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _shape_boundary(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            for dr, dc in N4:
                nr, nc = r + dr, c + dc
                if not shape[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _fence_region_from_component(
    grid: np.ndarray, wall_pts: list[tuple[int, int]], bg: int
) -> tuple[list[tuple[int, int]], int] | None:
    h, w = grid.shape
    wall_color = int(grid[wall_pts[0]])
    r0, c0, r1, c1 = _bbox(wall_pts)
    pr0 = max(0, r0 - 1)
    pc0 = max(0, c0 - 1)
    pr1 = min(h - 1, r1 + 1)
    pc1 = min(w - 1, c1 + 1)

    wall = np.zeros((pr1 - pr0 + 1, pc1 - pc0 + 1), dtype=bool)
    for r, c in wall_pts:
        wall[r - pr0, c - pc0] = True

    seen = np.zeros_like(wall, dtype=bool)
    q = deque()
    hh, ww = wall.shape
    for r in range(hh):
        for c in range(ww):
            if r not in (0, hh - 1) and c not in (0, ww - 1):
                continue
            if wall[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in N4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < hh and 0 <= nc < ww and not wall[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    enclosed = (~wall) & (~seen)
    matches: list[tuple[list[tuple[int, int]], int]] = []
    for comp in _components(enclosed):
        pts = [(pr0 + r, pc0 + c) for r, c in comp]
        vals = [int(grid[r, c]) for r, c in pts]
        non_bg = [v for v in vals if v != bg]
        if len(non_bg) != 1:
            continue
        seed_color = int(non_bg[0])
        if seed_color == wall_color:
            continue
        if sum(int(grid[r, c]) == seed_color for r, c in pts) != 1:
            continue
        if any(int(grid[r, c]) not in (bg, seed_color) for r, c in pts):
            continue
        matches.append((pts, seed_color))

    if len(matches) != 1:
        return None
    return matches[0]


def _find_fence_regions(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    regions: list[dict[str, object]] = []
    for color in [int(v) for v in np.unique(grid) if int(v) != bg]:
        for comp in _components(grid == color):
            result = _fence_region_from_component(grid, comp, bg)
            if result is None:
                continue
            region_pts, seed_color = result
            regions.append(
                {
                    "wall_color": color,
                    "wall_pts": comp,
                    "region_pts": region_pts,
                    "seed_color": seed_color,
                }
            )
    assert regions, "expected at least one enclosed fence region"
    return regions


def _extract_templates(
    grid: np.ndarray, bg: int, regions: list[dict[str, object]]
) -> dict[int, np.ndarray]:
    interior_mask = np.zeros(grid.shape, dtype=bool)
    for item in regions:
        for r, c in item["region_pts"]:
            interior_mask[r, c] = True

    templates: dict[int, np.ndarray] = {}
    for item in regions:
        seed_color = int(item["seed_color"])
        if seed_color in templates:
            continue
        coords = [(int(r), int(c)) for r, c in np.argwhere((grid == seed_color) & (~interior_mask))]
        assert coords, f"missing template cells for seed color {seed_color}"
        r0, c0, r1, c1 = _bbox(coords)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert sub.shape[0] == sub.shape[1] and sub.shape[0] in TEMPLATE_SIZES, (
            f"template bbox for color {seed_color} must be 3x3 or 4x4, got {sub.shape}"
        )
        assert np.all((sub == bg) | (sub == seed_color)), "template bbox must contain only bg and its foreground color"
        pattern = sub == seed_color
        assert np.any(pattern) and np.any(~pattern), "template pattern should contain both foreground and background"
        templates[seed_color] = pattern
    return templates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior by tiling the edge template whose foreground color matches the seed pixel.

    Input structure:
    - bg is the most common color in the grid.
    - Several single-pixel-wide grey fence loops enclose irregular interior regions.
    - Each enclosed region contains exactly one non-bg, non-grey seed pixel, and the rest of the region is bg.
    - Outside the fenced regions, the grid edge holds 3x3 or 4x4 texture templates drawn with one foreground color on bg.

    Rule:
    - Detect each closed fence and recover its enclosed interior region by flood filling the non-fence cells that are not connected to the local outside.
    - Read the unique seed color inside that region and find the outside template that uses the same foreground color.
    - Tile that template from the interior region's top-left cell across the whole enclosed region, writing either the seed color or bg according to the template pattern.
    - Leave the fence pixels, the template blocks, and every exterior pixel unchanged.

    Key insight:
    - The seed pixel is not the fill itself; it is a lookup key that selects a small texture, and that texture must then be periodically repeated over an irregular enclosed area.
    """
    g = np.asarray(grid, dtype=int)
    bg = _mode_color(g)
    regions = _find_fence_regions(g, bg)
    templates = _extract_templates(g, bg, regions)

    out = g.copy()
    for item in regions:
        seed_color = int(item["seed_color"])
        pattern = templates[seed_color]
        ph, pw = pattern.shape
        region_pts = item["region_pts"]
        rr0, rc0, _, _ = _bbox(region_pts)
        for r, c in region_pts:
            out[r, c] = seed_color if pattern[(r - rr0) % ph, (c - rc0) % pw] else bg
    return out


def _random_pattern(size: int, rng: random.Random) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mandatory = {
        (0, rng.randrange(size)),
        (size - 1, rng.randrange(size)),
        (rng.randrange(size), 0),
        (rng.randrange(size), size - 1),
    }
    for r, c in mandatory:
        mask[r, c] = True

    low = max(size + 1, len(mandatory) + 1)
    high = size * size - 1
    target = rng.randint(low, high)
    while int(mask.sum()) < target:
        mask[rng.randrange(size), rng.randrange(size)] = True
    if np.all(mask):
        candidates = [(r, c) for r in range(size) for c in range(size) if (r, c) not in mandatory]
        r, c = rng.choice(candidates)
        mask[r, c] = False
    return mask


def _random_shape(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(200):
        h = rng.randint(*SHAPE_H_RANGE)
        w = rng.randint(*SHAPE_W_RANGE)
        shape = np.ones((h, w), dtype=bool)
        notch_count = rng.choice((0, 1, 1, 2, 2, 3))
        for _ in range(notch_count):
            side = rng.choice(EDGE_SIDES)
            if side in ("top", "bottom"):
                if w < 4 or h < 5:
                    continue
                width = rng.randint(1, w - 2)
                start = rng.randint(1, w - width)
                depth = rng.randint(1, h - 3)
                rows = slice(0, depth) if side == "top" else slice(h - depth, h)
                shape[rows, start : start + width] = False
            else:
                if h < 4 or w < 5:
                    continue
                height = rng.randint(1, h - 2)
                start = rng.randint(1, h - height)
                depth = rng.randint(1, w - 3)
                cols = slice(0, depth) if side == "left" else slice(w - depth, w)
                shape[start : start + height, cols] = False

        if len(_components(shape)) != 1:
            continue
        boundary = _shape_boundary(shape)
        interior = shape & (~boundary)
        comps = _components(interior)
        if len(comps) != 1 or interior.sum() < 4:
            continue
        return shape, boundary
    raise RuntimeError("failed to sample a valid fence shape")


def _fits_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> bool:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occupied.shape[0], r0 + h + gap)
    cc1 = min(occupied.shape[1], c0 + w + gap)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _occupy_bbox(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, gap: int) -> None:
    rr0 = max(0, r0 - gap)
    cc0 = max(0, c0 - gap)
    rr1 = min(occupied.shape[0], r0 + h + gap)
    cc1 = min(occupied.shape[1], c0 + w + gap)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_edge_template(
    grid: np.ndarray, occupied: np.ndarray, color: int, pattern: np.ndarray, rng: random.Random
) -> bool:
    h, w = pattern.shape
    gh, gw = grid.shape
    candidates: list[tuple[int, int]] = []
    for _ in range(50):
        side = rng.choice(EDGE_SIDES)
        if side == "top":
            r0, c0 = 0, rng.randint(0, gw - w)
        elif side == "bottom":
            r0, c0 = gh - h, rng.randint(0, gw - w)
        elif side == "left":
            r0, c0 = rng.randint(0, gh - h), 0
        else:
            r0, c0 = rng.randint(0, gh - h), gw - w
        candidates.append((r0, c0))

    for r0, c0 in candidates:
        if not _fits_bbox(occupied, r0, c0, h, w, gap=1):
            continue
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        sub[pattern] = color
        _occupy_bbox(occupied, r0, c0, h, w, gap=1)
        return True
    return False


def _place_fence(
    grid: np.ndarray,
    occupied: np.ndarray,
    seed_color: int,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    shape, boundary = _random_shape(rng)
    h, w = shape.shape
    interior = shape & (~boundary)
    candidates: list[tuple[int, int]] = []
    for _ in range(80):
        r0 = rng.randint(2, gh - h - 2)
        c0 = rng.randint(2, gw - w - 2)
        candidates.append((r0, c0))

    for r0, c0 in candidates:
        if not _fits_bbox(occupied, r0, c0, h, w, gap=2):
            continue
        sub = grid[r0 : r0 + h, c0 : c0 + w]
        sub[boundary] = GREY
        seed_candidates = [(r, c) for r, c in np.argwhere(interior)]
        sr, sc = rng.choice(seed_candidates)
        sub[sr, sc] = seed_color
        _occupy_bbox(occupied, r0, c0, h, w, gap=2)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    palette = [c for c in range(10) if c != GREY]
    bg = rng.choice(palette)
    colors = [c for c in palette if c != bg]
    region_count = rng.choice(REGION_COUNT_CHOICES)
    if len(colors) < region_count:
        return None
    seed_colors = rng.sample(colors, region_count)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    patterns: dict[int, np.ndarray] = {}
    for color in seed_colors:
        size = rng.choice(TEMPLATE_SIZES)
        patterns[color] = _random_pattern(size, rng)

    order = seed_colors[:]
    rng.shuffle(order)
    for color in order:
        if not _place_edge_template(grid, occupied, color, patterns[color], rng):
            return None

    rng.shuffle(order)
    for color in order:
        if not _place_fence(grid, occupied, color, rng):
            return None

    try:
        regions = _find_fence_regions(grid, bg)
        if len(regions) != region_count:
            return None
        found_colors = sorted(int(item["seed_color"]) for item in regions)
        if found_colors != sorted(seed_colors):
            return None
        out = main(grid)
        if out.shape != grid.shape or not np.any(out != grid):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
