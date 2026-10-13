import random
from collections import Counter

import numpy as np


# --------------- helpers ---------------

MAX_GENERATION_ATTEMPTS = 400
GRID_H_RANGE = (16, 28)
GRID_W_RANGE = (16, 28)
REGION_COUNT_RANGE = (1, 4)
INTERIOR_H_RANGE = (3, 10)
INTERIOR_W_RANGE = (3, 11)
MAX_PALETTE_SIZE = 4
PLACEMENT_RETRIES = 120
CLEARANCE = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _boundary_mode(grid: np.ndarray) -> int:
    h, w = grid.shape
    boundary = np.concatenate(
        [
            grid[0, :],
            grid[h - 1, :],
            grid[1 : h - 1, 0],
            grid[1 : h - 1, w - 1],
        ]
    )
    return Counter(map(int, boundary)).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _ring_depth(r: int, c: int, r0: int, c0: int, r1: int, c1: int) -> int:
    return min(r - r0, c - c0, r1 - r, c1 - c)


def _regions_for_wall(grid: np.ndarray, wall: int) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    regions: list[tuple[int, int, int, int]] = []
    for comp in _components(grid != wall):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if len(comp) != (r1 - r0 + 1) * (c1 - c0 + 1):
            continue
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if not np.all(grid[r0 - 1, c0 : c1 + 1] == wall):
            continue
        if not np.all(grid[r1 + 1, c0 : c1 + 1] == wall):
            continue
        if not np.all(grid[r0 : r1 + 1, c0 - 1] == wall):
            continue
        if not np.all(grid[r0 : r1 + 1, c1 + 1] == wall):
            continue
        regions.append((r0, c0, r1, c1))
    regions.sort()
    return regions


def _find_wall_and_regions(grid: np.ndarray) -> tuple[int, list[tuple[int, int, int, int]]]:
    bg = _boundary_mode(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) != bg]
    best_wall = None
    best_regions: list[tuple[int, int, int, int]] = []
    best_score = (-1, -1, -1)
    for wall in colors:
        regions = _regions_for_wall(grid, wall)
        if not regions:
            continue
        total_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in regions)
        score = (len(regions), total_area, int(np.sum(grid == wall)))
        if score > best_score:
            best_score = score
            best_wall = wall
            best_regions = regions
    assert best_wall is not None, "main: failed to identify the wall color"
    return int(best_wall), best_regions


def _region_palette(
    grid: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, list[int]]:
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    main_color = Counter(map(int, patch.reshape(-1))).most_common(1)[0][0]

    votes: dict[int, Counter[int]] = {}
    first_seen: dict[tuple[int, int], int] = {}
    order = 0
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            color = int(grid[r, c])
            if color == main_color:
                continue
            depth = _ring_depth(r, c, r0, c0, r1, c1)
            if depth not in votes:
                votes[depth] = Counter()
            votes[depth][color] += 1
            first_seen.setdefault((depth, color), order)
            order += 1

    if not votes:
        return main_color, [main_color]

    depth_colors = []
    for depth in sorted(votes):
        counter = votes[depth]
        chosen = max(counter, key=lambda color: (counter[color], -first_seen[(depth, color)]))
        depth_colors.append(chosen)

    palette = depth_colors if min(votes) == 0 else [main_color] + depth_colors
    return main_color, [int(color) for color in palette]


# --------------- main ---------------

def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a depth-ordered palette from marker pixels in each enclosed rectangle and
    repaint the region as concentric rectangular rings.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _wall, regions = _find_wall_and_regions(g)

    for r0, c0, r1, c1 in regions:
        _main_color, palette = _region_palette(g, r0, c0, r1, c1)
        period = len(palette)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                depth = _ring_depth(r, c, r0, c0, r1, c1)
                out[r, c] = palette[depth % period]

    return out


# --------------- generator ---------------

def _interior_max_depth(height: int, width: int) -> int:
    return min((height - 1) // 2, (width - 1) // 2)


def _sample_colors(rng: random.Random, banned: set[int], count: int) -> list[int]:
    choices = [c for c in range(10) if c not in banned]
    rng.shuffle(choices)
    if count <= len(choices):
        return choices[:count]
    out = []
    while len(out) < count:
        pool = choices[:]
        rng.shuffle(pool)
        out.extend(pool)
    return out[:count]


def _ring_cells(r0: int, c0: int, r1: int, c1: int, depth: int) -> list[tuple[int, int]]:
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _ring_depth(r, c, r0, c0, r1, c1) == depth:
                cells.append((r, c))
    return cells


def _pick_branch_mode(rng: random.Random, region_count: int) -> str:
    modes = ["boundary", "no_boundary"]
    if region_count >= 2:
        modes.append("mixed")
    return rng.choice(modes)


def _place_region(
    grid: np.ndarray,
    occupied: np.ndarray,
    rng: random.Random,
    outer_h: int,
    outer_w: int,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    if outer_h + 2 >= h or outer_w + 2 >= w:
        return None
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, h - outer_h - 1)
        c0 = rng.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        er0 = max(0, r0 - CLEARANCE)
        ec0 = max(0, c0 - CLEARANCE)
        er1 = min(h - 1, r1 + CLEARANCE)
        ec1 = min(w - 1, c1 + CLEARANCE)
        if occupied[er0 : er1 + 1, ec0 : ec1 + 1].any():
            continue
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True
        return r0, c0, r1, c1
    return None


def _decorate_region(
    grid: np.ndarray,
    inner_box: tuple[int, int, int, int],
    wall: int,
    bg: int,
    rng: random.Random,
    branch: str,
) -> None:
    r0, c0, r1, c1 = inner_box
    inner_h = r1 - r0 + 1
    inner_w = c1 - c0 + 1
    max_depth = _interior_max_depth(inner_h, inner_w)

    main_color = _sample_colors(rng, {bg, wall}, 1)[0]
    grid[r0 : r1 + 1, c0 : c1 + 1] = main_color

    if branch == "boundary":
        max_palette = min(MAX_PALETTE_SIZE, max_depth + 1)
        min_palette = 1 if max_palette == 1 else 2
        palette_len = rng.randint(min_palette, max_palette)
        marker_depths = list(range(palette_len))
    else:
        max_palette = min(MAX_PALETTE_SIZE, max_depth + 1)
        if max_palette < 2:
            marker_depths = [0]
        else:
            palette_len = rng.randint(2, max_palette)
            marker_depths = list(range(1, palette_len))

    marker_colors = _sample_colors(rng, {bg, wall, main_color}, len(marker_depths))
    for depth, color in zip(marker_depths, marker_colors):
        cells = _ring_cells(r0, c0, r1, c1, depth)
        rng.shuffle(cells)
        grid[cells[0]] = color
        if len(cells) > 1 and rng.random() < 0.35:
            grid[cells[1]] = color


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    grid = np.zeros((h, w), dtype=int)

    bg, wall = rng.sample(list(range(10)), 2)
    grid[:, :] = bg

    region_count = rng.randint(*REGION_COUNT_RANGE)
    branch_mode = _pick_branch_mode(rng, region_count)
    branches = []
    if branch_mode == "mixed":
        branches = ["boundary", "no_boundary"]
        while len(branches) < region_count:
            branches.append(rng.choice(["boundary", "no_boundary"]))
        rng.shuffle(branches)
    else:
        branches = [branch_mode] * region_count

    occupied = np.zeros_like(grid, dtype=bool)
    placed_regions: list[tuple[int, int, int, int]] = []

    for branch in branches:
        inner_h = rng.randint(*INTERIOR_H_RANGE)
        inner_w = rng.randint(*INTERIOR_W_RANGE)
        if branch == "no_boundary":
            inner_h = max(inner_h, 3)
            inner_w = max(inner_w, 3)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        box = _place_region(grid, occupied, rng, outer_h, outer_w)
        if box is None:
            return None
        or0, oc0, or1, oc1 = box
        grid[or0 : or1 + 1, oc0 : oc1 + 1] = wall
        inner_box = (or0 + 1, oc0 + 1, or1 - 1, oc1 - 1)
        _decorate_region(grid, inner_box, wall, bg, rng, branch)
        placed_regions.append(inner_box)

    counts = Counter(map(int, grid.reshape(-1)))
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return None

    detected_wall, detected_regions = _find_wall_and_regions(grid)
    if detected_wall != wall:
        return None
    if sorted(detected_regions) != sorted(placed_regions):
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid sample")
