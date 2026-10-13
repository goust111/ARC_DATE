from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_MIN = 12
GRID_MAX = 30
OBJECT_COUNT_CHOICES = [1, 2, 2, 3]
REGION_RETRIES = 100
GEN_RETRIES = 400
COLOR_REUSE_PROB = 0.35


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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _distance_from_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    dist = np.zeros((h, w), dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if any(
                not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
                for dr, dc in DIR4
            ):
                dist[r, c] = 1
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] == 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def _analyze_regions(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    regions: list[dict] = []
    for coords in _components(grid != GREY):
        if _touches_border(coords, h, w):
            continue
        seed_coords = [(r, c) for r, c in coords if int(grid[r, c]) not in (BLACK, GREY)]
        if not seed_coords:
            continue
        seed_color = Counter(int(grid[r, c]) for r, c in seed_coords).most_common(1)[0][0]
        r0, c0, r1, c1 = _bbox(coords)
        local_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in coords:
            local_mask[r - r0, c - c0] = True
        dist = _distance_from_boundary(local_mask)
        regions.append(
            {
                "coords": coords,
                "seed_count": len(seed_coords),
                "seed_color": seed_color,
                "bbox": (r0, c0, r1, c1),
                "dist": dist,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill enclosed regions with inward concentric rings whose depth equals the seed count.

    Input structure:
    - Background is black.
    - One or more grey closed walls surround interior regions.
    - Each enclosed region contains several same-colored seed pixels scattered on black interior cells.

    Rule:
    - Process each non-grey connected component that does not touch the grid border as one enclosed region.
    - Count that region's seed pixels; this count N determines how many inward layers to fill.
    - Starting from the cells adjacent to the grey wall, fill the first N 4-connected boundary-distance layers with the seed color.
    - Leave the grey wall unchanged and turn all deeper interior cells black, even if they originally contained seeds.

    Key insight:
    - Seed positions are irrelevant. Only the common seed color and the number of seed pixels matter, and the fill grows from the wall inward rather than from the seeds outward.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _analyze_regions(g):
        r0, c0, _r1, _c1 = region["bbox"]
        dist = region["dist"]
        seed_count = int(region["seed_count"])
        seed_color = int(region["seed_color"])

        for r, c in region["coords"]:
            out[r, c] = BLACK
        for r, c in region["coords"]:
            if int(dist[r - r0, c - c0]) <= seed_count:
                out[r, c] = seed_color

    return out


def _random_polyomino(rng: random.Random, rows: int, cols: int, area: int) -> np.ndarray | None:
    start = (rng.randrange(rows), rng.randrange(cols))
    cells = {start}
    frontier = set()
    for dr, dc in DIR4:
        nr = start[0] + dr
        nc = start[1] + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            frontier.add((nr, nc))

    while len(cells) < area and frontier:
        cell = rng.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        for dr, dc in DIR4:
            nr = cell[0] + dr
            nc = cell[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != area:
        return None

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if min(rs) != 0 or max(rs) != rows - 1 or min(cs) != 0 or max(cs) != cols - 1:
        return None

    mask = np.zeros((rows, cols), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _coarse_shape(rng: random.Random) -> np.ndarray | None:
    rows = rng.randint(2, 4)
    cols = rng.randint(2, 4)
    if rng.random() < 0.4:
        return np.ones((rows, cols), dtype=bool)

    min_area = max(3, (rows * cols + 1) // 2)
    for _ in range(40):
        area = rng.randint(min_area, rows * cols)
        mask = _random_polyomino(rng, rows, cols, area)
        if mask is not None:
            return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return out


def _pick_color(rng: random.Random, used_colors: list[int]) -> int:
    if used_colors and rng.random() < COLOR_REUSE_PROB:
        return rng.choice(used_colors)
    fresh = [color for color in PALETTE if color not in used_colors]
    return rng.choice(fresh if fresh else PALETTE)


def _choose_seed_positions(
    rng: random.Random,
    dist: np.ndarray,
    seed_count: int,
    mode: str,
) -> list[tuple[int, int]]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(dist > 0)]
    deep = [cell for cell in cells if int(dist[cell]) > seed_count]
    shallow = [cell for cell in cells if int(dist[cell]) <= seed_count]

    chosen: list[tuple[int, int]] = []
    remaining = set(cells)

    if mode == "partial" and deep:
        seed = rng.choice(deep)
        chosen.append(seed)
        remaining.remove(seed)
    if mode == "partial" and shallow and len(chosen) < seed_count and rng.random() < 0.75:
        candidates = [cell for cell in shallow if cell in remaining]
        if candidates:
            seed = rng.choice(candidates)
            chosen.append(seed)
            remaining.remove(seed)

    need = seed_count - len(chosen)
    if need > 0:
        chosen.extend(rng.sample(list(remaining), need))
    return chosen


def _make_region(
    rng: random.Random,
    used_colors: list[int],
    preferred_mode: str,
) -> dict | None:
    for _ in range(REGION_RETRIES):
        coarse = _coarse_shape(rng)
        if coarse is None:
            continue
        scale = rng.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]
        interior = np.kron(coarse, np.ones((scale, scale), dtype=bool))
        padded_interior = np.pad(interior, 1)
        wall = _dilate8(padded_interior) & ~padded_interior
        if len(_components(padded_interior)) != 1 or len(_components(wall)) != 1:
            continue

        dist = _distance_from_boundary(padded_interior)
        max_depth = int(dist.max())
        if max_depth < 1:
            continue

        mode = preferred_mode
        if mode == "partial" and max_depth < 2:
            mode = "full"

        interior_size = int(padded_interior.sum())
        if mode == "partial":
            seed_count = rng.randint(1, max_depth - 1)
        else:
            upper = min(interior_size, max_depth + rng.randint(1, 4))
            seed_count = rng.randint(max_depth, upper)

        color = _pick_color(rng, used_colors)
        seeds = _choose_seed_positions(rng, dist, seed_count, mode)

        local = np.zeros_like(padded_interior, dtype=int)
        local[wall] = GREY
        for r, c in seeds:
            local[r, c] = color

        return {
            "grid": local,
            "mode": mode,
            "color": color,
            "seed_count": seed_count,
        }
    return None


def _place_region(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    local: np.ndarray,
) -> bool:
    h, w = canvas.shape
    lh, lw = local.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(1, h - lh):
        for c0 in range(1, w - lw):
            if occupied[r0 - 1 : r0 + lh + 1, c0 - 1 : c0 + lw + 1].any():
                continue
            positions.append((r0, c0))
    if not positions:
        return False

    r0, c0 = rng.choice(positions)
    patch = canvas[r0 : r0 + lh, c0 : c0 + lw]
    patch[:, :] = local
    occupied[r0 - 1 : r0 + lh + 1, c0 - 1 : c0 + lw + 1] = True
    return True


def _valid_generated_grid(grid: np.ndarray) -> bool:
    regions = _analyze_regions(grid)
    if not regions:
        return False
    if any(len({int(grid[r, c]) for r, c in region["coords"] if int(grid[r, c]) not in (BLACK, GREY)}) != 1 for region in regions):
        return False
    out = main(grid)
    if not np.any((grid == BLACK) & (out != grid)):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    canvas = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    target_objects = rng.choice(OBJECT_COUNT_CHOICES)

    modes = [rng.choice(["partial", "full"]) for _ in range(target_objects)]
    if target_objects >= 2 and rng.random() < 0.6:
        modes[0] = "partial"
        modes[1] = "full"
        rng.shuffle(modes)

    used_colors: list[int] = []
    for mode in modes:
        region = _make_region(rng, used_colors, mode)
        if region is None:
            return None
        if not _place_region(rng, canvas, occupied, region["grid"]):
            return None
        used_colors.append(int(region["color"]))

    return canvas if _valid_generated_grid(canvas) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid enclosed-ring ARC input")
