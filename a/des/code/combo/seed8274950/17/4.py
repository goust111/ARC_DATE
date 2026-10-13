from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


SOIL_COLOR = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_ORDER = {
    "top": [(1, 0), (0, -1), (0, 1), (-1, 0)],
    "bottom": [(-1, 0), (0, -1), (0, 1), (1, 0)],
    "left": [(0, 1), (-1, 0), (1, 0), (0, -1)],
    "right": [(0, -1), (-1, 0), (1, 0), (0, 1)],
}

GEN_ATTEMPTS = 300
REGION_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 80
GRID_H_RANGE = (11, 22)
GRID_W_RANGE = (11, 24)
REGION_COUNT_RANGE = (1, 4)
SHAPE_H_RANGE = (3, 7)
SHAPE_W_RANGE = (3, 8)
AREA_RANGE = (5, 18)


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


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
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    comps.sort(key=lambda comp: min(comp))
    return comps


def _find_seeded_soil_regions(
    grid: np.ndarray, bg: int
) -> list[tuple[set[tuple[int, int]], tuple[int, int]]]:
    h, w = grid.shape
    soil_mask = grid == SOIL_COLOR
    comps = _components(soil_mask)
    assert comps, "expected at least one soil region"

    seed_pixels = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != bg) & (grid != SOIL_COLOR))
    }
    assert seed_pixels, "expected at least one seed pixel"

    regions: list[tuple[set[tuple[int, int]], tuple[int, int]]] = []
    used_seeds: set[tuple[int, int]] = set()
    for comp in comps:
        comp_set = set(comp)
        seeds: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    value = int(grid[nr, nc])
                    if value != bg and value != SOIL_COLOR:
                        seeds.add((int(nr), int(nc)))
        assert len(seeds) == 1, "each soil region must touch exactly one seed"
        seed = next(iter(seeds))
        used_seeds.add(seed)
        regions.append((comp_set, seed))

    assert used_seeds == seed_pixels, "all non-background, non-soil pixels must be seeds"
    return regions


def _component_side(seed: tuple[int, int], comp: set[tuple[int, int]]) -> str:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    sr, sc = seed
    if sr < min_r:
        return "top"
    if sr > max_r:
        return "bottom"
    if sc < min_c:
        return "left"
    if sc > max_c:
        return "right"

    distances = {
        "top": abs(sr - min_r),
        "bottom": abs(sr - max_r),
        "left": abs(sc - min_c),
        "right": abs(sc - max_c),
    }
    return min(distances, key=distances.get)


def _growth_order(
    comp: set[tuple[int, int]], seed: tuple[int, int], side: str
) -> list[tuple[int, int]]:
    neighbor_order = SIDE_ORDER.get(side, DIR4)
    visited: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque()
    for dr, dc in neighbor_order:
        cell = (seed[0] + dr, seed[1] + dc)
        if cell in comp and cell not in visited:
            visited.add(cell)
            q.append(cell)

    order: list[tuple[int, int]] = []
    while q:
        cell = q.popleft()
        order.append(cell)
        r, c = cell
        for dr, dc in neighbor_order:
            nxt = (r + dr, c + dc)
            if nxt in comp and nxt not in visited:
                visited.add(nxt)
                q.append(nxt)

    assert len(order) == len(comp), "soil region must be reachable from its seed"
    return order


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background_color(g)
    assert bg != SOIL_COLOR, "background must differ from soil color"

    out = g.copy()
    for comp, seed in _find_seeded_soil_regions(g, bg):
        quota = len(comp) // 2
        if quota == 0:
            continue
        side = _component_side(seed, comp)
        order = _growth_order(comp, seed, side)
        seed_color = int(g[seed])
        for r, c in order[:quota]:
            out[r, c] = seed_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each boundary seed into its adjacent grey soil region, coloring exactly
    half of the soil cells by 4-neighbor breadth-first expansion from the seed.
    """
    return _solve(grid)


def _normalize_shape(
    cells: set[tuple[int, int]], seed: tuple[int, int]
) -> tuple[set[tuple[int, int]], tuple[int, int], int, int]:
    all_cells = set(cells)
    all_cells.add(seed)
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    shifted_cells = {(r - min_r, c - min_c) for r, c in cells}
    shifted_seed = (seed[0] - min_r, seed[1] - min_c)
    max_r = max(r for r, _ in shifted_cells | {shifted_seed})
    max_c = max(c for _, c in shifted_cells | {shifted_seed})
    return shifted_cells, shifted_seed, max_r + 1, max_c + 1


def _choose_target_area(rng: random.Random, box_h: int, box_w: int) -> int:
    low = max(AREA_RANGE[0], box_h + box_w - 2)
    high = min(AREA_RANGE[1], box_h * box_w)
    if low > high:
        low = min(AREA_RANGE[0], high)
    parity = rng.randrange(2)
    choices = [area for area in range(low, high + 1) if area % 2 == parity]
    if not choices:
        choices = list(range(low, high + 1))
    return int(rng.choice(choices))


def _generate_region_shape(
    rng: random.Random,
) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    box_h = rng.randint(*SHAPE_H_RANGE)
    box_w = rng.randint(*SHAPE_W_RANGE)
    target = _choose_target_area(rng, box_h, box_w)
    compact_mode = rng.random() < 0.5

    cells = {(rng.randrange(box_h), rng.randrange(box_w))}
    while len(cells) < target:
        frontier: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in cells:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < box_h and 0 <= nc < box_w and (nr, nc) not in cells and (nr, nc) not in seen:
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
        if not frontier:
            break

        weights = []
        for r, c in frontier:
            adjacent = sum((r + dr, c + dc) in cells for dr, dc in DIR4)
            if compact_mode:
                weights.append(1 + 3 * adjacent)
            else:
                exposed = sum(
                    0 <= r + dr < box_h
                    and 0 <= c + dc < box_w
                    and (r + dr, c + dc) not in cells
                    for dr, dc in DIR4
                )
                weights.append(1 + adjacent + exposed)
        cells.add(rng.choices(frontier, weights=weights, k=1)[0])

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    if max(rows) == min(rows) or max(cols) == min(cols):
        raise ValueError("degenerate region")

    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    side_to_candidates = {
        "top": [(min_r - 1, c) for c in range(min_c, max_c + 1) if (min_r, c) in cells],
        "bottom": [(max_r + 1, c) for c in range(min_c, max_c + 1) if (max_r, c) in cells],
        "left": [(r, min_c - 1) for r in range(min_r, max_r + 1) if (r, min_c) in cells],
        "right": [(r, max_c + 1) for r in range(min_r, max_r + 1) if (r, max_c) in cells],
    }
    side = rng.choice(["top", "bottom", "left", "right"])
    seed = rng.choice(side_to_candidates[side])
    return cells, seed


def _reserve(blocked: np.ndarray, coords: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    soil_cells: set[tuple[int, int]],
    seed: tuple[int, int],
    seed_color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    soil_cells, seed, shape_h, shape_w = _normalize_shape(soil_cells, seed)
    if h - shape_h < 3 or w - shape_w < 3:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        top = rng.randint(1, h - shape_h - 1)
        left = rng.randint(1, w - shape_w - 1)
        abs_cells = {(top + r, left + c) for r, c in soil_cells}
        seed_abs = (top + seed[0], left + seed[1])
        occupied = set(abs_cells)
        occupied.add(seed_abs)
        if any(blocked[r, c] for r, c in occupied):
            continue

        for r, c in abs_cells:
            grid[r, c] = SOIL_COLOR
        grid[seed_abs] = seed_color
        _reserve(blocked, occupied)
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_regions = rng.randint(*REGION_COUNT_RANGE)
    min_h = max(GRID_H_RANGE[0], 5 + 3 * n_regions)
    min_w = max(GRID_W_RANGE[0], 6 + 4 * n_regions)
    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])

    colors = list(range(10))
    bg_choices = [color for color in colors if color != SOIL_COLOR]
    bg = rng.choice(bg_choices)
    seed_colors = [color for color in colors if color not in (bg, SOIL_COLOR)]
    rng.shuffle(seed_colors)
    seed_colors = seed_colors[:n_regions]

    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    for seed_color in seed_colors:
        placed = False
        for _ in range(REGION_ATTEMPTS):
            try:
                soil_cells, seed = _generate_region_shape(rng)
            except ValueError:
                continue
            if _place_region(grid, blocked, soil_cells, seed, seed_color, rng):
                placed = True
                break
        if not placed:
            return None

    if not _is_valid_generated_grid(grid):
        return None
    return grid


def _is_valid_generated_grid(grid: np.ndarray) -> bool:
    try:
        bg = _background_color(grid)
        regions = _find_seeded_soil_regions(grid, bg)
        if len(regions) != int(np.count_nonzero((grid != bg) & (grid != SOIL_COLOR))):
            return False
        out = _solve(grid)
        if np.array_equal(out, grid):
            return False
        return True
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
