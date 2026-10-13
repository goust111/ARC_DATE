from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
YELLOW = 4
GREY = 5

DIAGONALS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
TEMPLATE_DIRS = {
    "top": [(-1, -1), (-1, 1)],
    "bottom": [(1, -1), (1, 1)],
    "left": [(-1, -1), (1, -1)],
    "right": [(-1, 1), (1, 1)],
    "nw": [(-1, -1)],
    "ne": [(-1, 1)],
    "sw": [(1, -1)],
    "se": [(1, 1)],
}

GENERATE_ATTEMPTS = 600


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _seed_from_square(square_r: int, square_c: int, dr: int, dc: int) -> tuple[int, int]:
    corner_r = square_r if dr < 0 else square_r + 1
    corner_c = square_c if dc < 0 else square_c + 1
    return corner_r + dr, corner_c + dc


def _ray_path(shape: tuple[int, int], seed_r: int, seed_c: int, dr: int, dc: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    r = seed_r + dr
    c = seed_c + dc
    while _in_bounds(shape, r, c):
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _component_regions(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    regions: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            min_r = r
            min_c = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr < min_r or (rr == min_r and cc < min_c):
                    min_r, min_c = rr, cc
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            regions.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "top_left": (min_r, min_c),
                }
            )
    return regions


def _sorted_regions(regions: list[dict]) -> list[dict]:
    return sorted(regions, key=lambda region: (region["area"], region["top_left"][0], region["top_left"][1]))


def _detect_squares_and_seeds(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int, int, int]]]:
    h, w = grid.shape
    squares: list[tuple[int, int]] = []
    for r in range(h - 1):
        for c in range(w - 1):
            if np.all(grid[r : r + 2, c : c + 2] == BLUE):
                squares.append((r, c))

    assert 2 <= len(squares) <= 4, "expected 2-4 blue 2x2 squares"
    for i, (r0, c0) in enumerate(squares):
        rows0 = {r0, r0 + 1}
        cols0 = {c0, c0 + 1}
        for r1, c1 in squares[i + 1 :]:
            assert rows0.isdisjoint({r1, r1 + 1}), "squares must not share rows"
            assert cols0.isdisjoint({c1, c1 + 1}), "squares must not share cols"

    square_cells: set[tuple[int, int]] = set()
    for r, c in squares:
        for rr in [r, r + 1]:
            for cc in [c, c + 1]:
                square_cells.add((rr, cc))

    seeds: list[tuple[int, int, int, int]] = []
    seed_cells: set[tuple[int, int]] = set()
    for square_r, square_c in squares:
        square_seeds: list[tuple[int, int, int, int]] = []
        for dr, dc in DIAGONALS:
            seed_r, seed_c = _seed_from_square(square_r, square_c, dr, dc)
            if _in_bounds(grid.shape, seed_r, seed_c) and grid[seed_r, seed_c] == BLUE:
                square_seeds.append((seed_r, seed_c, dr, dc))
        assert 1 <= len(square_seeds) <= 2, "each square needs 1-2 diagonal markers"
        for seed in square_seeds:
            assert (seed[0], seed[1]) not in seed_cells, "seed cannot belong to two squares"
            seed_cells.add((seed[0], seed[1]))
            seeds.append(seed)

    blue_cells = {tuple(map(int, coord)) for coord in np.argwhere(grid == BLUE)}
    assert blue_cells == square_cells | seed_cells, "blue cells must be squares plus diagonal markers"
    return squares, seeds


def _solve_with_regions(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    squares, seeds = _detect_squares_and_seeds(g)
    del squares

    out = g.copy()
    for seed_r, seed_c, dr, dc in seeds:
        for r, c in _ray_path(g.shape, seed_r, seed_c, dr, dc):
            assert g[r, c] == BLACK, "ray continuation must cross black background only"
            out[r, c] = GREY

    regions = _component_regions(out == BLACK)
    assert len(regions) >= 2, "ray walls must partition the background"
    ordered = _sorted_regions(regions)
    for index, region in enumerate(ordered):
        if index % 3 == 0:
            for r, c in region["coords"]:
                out[r, c] = YELLOW
    return out, ordered


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend blue diagonal markers into grey partition rays, then fill every third black region by area.

    Input structure:
    - A black background contains 2-4 blue 2x2 anchor squares whose row sets and column sets are pairwise disjoint.
    - Each square has 1-2 blue single-pixel markers on diagonal corner positions, with no other blue cells present.
    - The markers indicate diagonal directions that continue away from their parent square toward the grid boundary.

    Rule:
    - For each diagonal marker, extend a grey ray along the same diagonal direction away from the square until it leaves the grid.
    - Treat blue squares, blue markers, and grey ray cells as blockers; find the remaining black 4-connected regions.
    - Sort those black regions by ascending area, then by the row of their topmost pixel, then by the column of the leftmost pixel on that row.
    - Fill regions at sorted indices 0, 3, 6, 9, ... with yellow, leaving all other cells unchanged.

    Key insight:
    - The diagonal markers do not merely decorate the squares: their ray geometry reshapes the black regions, and the yellow targets are chosen only after those regions are area-sorted rather than spatially ordered.
    """
    out, _ = _solve_with_regions(grid)
    return out


def _core_positions(height: int, width: int, profile: str) -> list[tuple[int, int, str]]:
    mid_c = width // 2
    if profile == "diag_lr":
        top_c = random.randint(3, max(3, mid_c - 2))
        bot_c = random.randint(min(width - 5, mid_c + 1), width - 5)
        top_r = random.randint(2, height - 5)
        bot_r = random.randint(2, height - 5)
        return [(top_r, top_c, "top"), (bot_r, bot_c, "bottom")]

    top_c = random.randint(min(width - 5, mid_c + 1), width - 5)
    bot_c = random.randint(3, max(3, mid_c - 2))
    top_r = random.randint(2, height - 5)
    bot_r = random.randint(2, height - 5)
    return [(top_r, top_c, "top"), (bot_r, bot_c, "bottom")]


def _extra_position(height: int, width: int, template_name: str) -> tuple[int, int]:
    mid_r = height // 2
    mid_c = width // 2
    if template_name == "left":
        return random.randint(3, height - 5), random.randint(2, max(2, mid_c - 3))
    if template_name == "right":
        return random.randint(3, height - 5), random.randint(min(width - 5, mid_c + 1), width - 5)
    if template_name == "nw":
        return random.randint(mid_r - 1, height - 5), random.randint(mid_c - 1, width - 5)
    if template_name == "ne":
        return random.randint(mid_r - 1, height - 5), random.randint(2, max(2, mid_c - 2))
    if template_name == "sw":
        return random.randint(2, max(2, mid_r - 2)), random.randint(mid_c - 1, width - 5)
    if template_name == "se":
        return random.randint(2, max(2, mid_r - 2)), random.randint(2, max(2, mid_c - 2))
    raise ValueError(template_name)


def _squares_clear(items: list[tuple[int, int, str]], square_r: int, square_c: int) -> bool:
    rows = {square_r, square_r + 1}
    cols = {square_c, square_c + 1}
    for other_r, other_c, _ in items:
        if not rows.isdisjoint({other_r, other_r + 1}):
            return False
        if not cols.isdisjoint({other_c, other_c + 1}):
            return False
        if abs(square_r - other_r) < 3 and abs(square_c - other_c) < 3:
            return False
    return True


def _build_input_from_layout(
    height: int, width: int, items: list[tuple[int, int, str]]
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    occupied: set[tuple[int, int]] = set()
    ray_keys: set[tuple[str, int]] = set()
    rays: list[tuple[int, int, int, int]] = []
    ray_cells_seen: set[tuple[int, int]] = set()

    for square_r, square_c, _template_name in items:
        for rr in [square_r, square_r + 1]:
            for cc in [square_c, square_c + 1]:
                occupied.add((rr, cc))

    for square_r, square_c, template_name in items:
        for dr, dc in TEMPLATE_DIRS[template_name]:
            seed_r, seed_c = _seed_from_square(square_r, square_c, dr, dc)
            if not _in_bounds((height, width), seed_r, seed_c):
                return None
            if not _in_bounds((height, width), seed_r + dr, seed_c + dc):
                return None
            if (seed_r, seed_c) in occupied:
                return None

            key = ("main", seed_r - seed_c) if dr == dc else ("anti", seed_r + seed_c)
            if key in ray_keys:
                return None
            ray_keys.add(key)

            path = _ray_path((height, width), seed_r, seed_c, dr, dc)
            ray_cells = [(seed_r, seed_c)] + path
            if any(cell in occupied for cell in ray_cells):
                return None
            if any(cell in ray_cells_seen for cell in ray_cells):
                return None

            occupied.add((seed_r, seed_c))
            ray_cells_seen.update(ray_cells)
            rays.append((seed_r, seed_c, dr, dc))

    grid = np.zeros((height, width), dtype=int)
    for square_r, square_c, _template_name in items:
        grid[square_r : square_r + 2, square_c : square_c + 2] = BLUE
    for seed_r, seed_c, _dr, _dc in rays:
        grid[seed_r, seed_c] = BLUE
    return grid, rays


def _region_mask(shape: tuple[int, int], regions: list[dict], selected_indices: set[int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for idx, region in enumerate(regions):
        if idx in selected_indices:
            for r, c in region["coords"]:
                mask[r, c] = True
    return mask


def _validate_sample(inp: np.ndarray, out: np.ndarray, items: list[tuple[int, int, str]], rays: list[tuple[int, int, int, int]]) -> bool:
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    if np.count_nonzero(out == GREY) < 8:
        return False
    if not (2 <= len(items) <= 4):
        return False
    seed_counts = [len(TEMPLATE_DIRS[name]) for _r, _c, name in items]
    if not all(1 <= count <= 2 for count in seed_counts):
        return False
    if len(set(seed_counts)) == 1 and len(items) >= 3:
        return False
    if len(rays) < 3:
        return False

    regions = _component_regions(out == BLACK)
    if not (4 <= len(regions) <= 9):
        return False

    ordered = _sorted_regions(regions)
    actual_selected = {idx for idx in range(len(ordered)) if idx % 3 == 0}
    if len(actual_selected) < 2:
        return False
    actual_mask = _region_mask(inp.shape, ordered, actual_selected)

    row_major = sorted(ordered, key=lambda region: (region["top_left"][0], region["top_left"][1], region["area"]))
    row_major_mask = _region_mask(inp.shape, row_major, actual_selected)
    if np.array_equal(actual_mask, row_major_mask):
        return False

    col_major = sorted(ordered, key=lambda region: (region["top_left"][1], region["top_left"][0], region["area"]))
    col_major_mask = _region_mask(inp.shape, col_major, actual_selected)
    if np.array_equal(actual_mask, col_major_mask):
        return False

    yellow_regions = _component_regions(out == YELLOW)
    if len(yellow_regions) < 2:
        return False

    areas = [region["area"] for region in ordered]
    if max(areas) - min(areas) < 10:
        return False

    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(14, 18)
    width = random.randint(14, 18)
    core = _core_positions(height, width, random.choice(["diag_lr", "diag_rl"]))
    items = core[:]

    target_count = random.choices([2, 3, 4], weights=[2, 4, 3], k=1)[0]
    extras_pool = ["left", "right", "nw", "ne", "sw", "se"]
    random.shuffle(extras_pool)

    for template_name in extras_pool:
        if len(items) >= target_count:
            break
        for _ in range(30):
            square_r, square_c = _extra_position(height, width, template_name)
            if _squares_clear(items, square_r, square_c):
                items.append((square_r, square_c, template_name))
                break

    if len(items) != target_count:
        return None

    built = _build_input_from_layout(height, width, items)
    if built is None:
        return None
    inp, rays = built

    try:
        out = main(inp)
    except Exception:
        return None

    if not _validate_sample(inp, out, items, rays):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray partition sample")
