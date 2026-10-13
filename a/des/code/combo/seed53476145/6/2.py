from __future__ import annotations

import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
INNER_COLORS = tuple(range(2, 10))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
REGION_COUNT_RANGE = (1, 4)
PLACEMENT_TRIES = 80
PARTITION_TRIES = 80
GEN_RETRIES = 300
EXTRA_LAYER_MAX = 3
MAX_SHORT_DIM = 12
MAX_LONG_EXTRA = 5
MIN_AREA_PER_COLOR = 2


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))

    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _boundary_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _sorted_colors_by_area(inner: np.ndarray) -> list[int]:
    counts = Counter(map(int, inner.reshape(-1)))
    anchors: dict[int, tuple[int, int]] = {}
    for r in range(inner.shape[0]):
        for c in range(inner.shape[1]):
            color = int(inner[r, c])
            anchors.setdefault(color, (r, c))
    return sorted(
        counts,
        key=lambda color: (-counts[color], anchors[color][0], anchors[color][1]),
    )


def _paint_concentric(shape: tuple[int, int], ordered_colors: list[int]) -> np.ndarray:
    h, w = map(int, shape)
    last = len(ordered_colors) - 1
    out = np.empty((h, w), dtype=int)
    for r in range(h):
        for c in range(w):
            depth = min(r, c, h - 1 - r, w - 1 - c)
            out[r, c] = int(ordered_colors[min(depth, last)])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort the filled color blocks inside each blue-walled rectangle by area and redraw them as concentric rectangular rings.

    Input structure
    - Background is black.
    - Each object is a non-overlapping rectangle whose 1-pixel wall is blue.
    - The wall encloses a fully tiled interior of 2-4 differently colored 4-connected blocks.

    Rule
    - For each enclosed region, count the interior pixels of each color.
    - Sort colors by descending area.
    - Replace the original interior with concentric rectangular rings from outermost to innermost.
    - If there are more ring depths than colors, keep extending the smallest-area rank into the remaining center.
    - Leave the blue wall and the outside background unchanged.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = map(int, g.shape)

    out = g.copy()
    regions = _components(g != BLACK)
    assert regions, "expected at least one enclosed region"

    for coords in regions:
        r0, c0, r1, c1 = _bbox(coords)
        assert 0 < r0 and 0 < c0 and r1 < h - 1 and c1 < w - 1, "region must be enclosed by background"

        window = g[r0 : r1 + 1, c0 : c1 + 1]
        assert coords.shape[0] == window.size, "each region should fill its bounding box"
        assert window.shape[0] >= 3 and window.shape[1] >= 3, "wall must enclose a non-empty interior"

        border = _boundary_mask(*window.shape)
        assert bool(np.all(window[border] == BLUE)), "expected a blue rectangular wall"

        inner = window[1:-1, 1:-1]
        assert inner.size > 0 and bool(np.all(inner != BLACK)), "interior must be fully tiled"

        colors = np.unique(inner)
        assert 2 <= len(colors) <= 4, "expected 2-4 interior colors"
        ring_capacity = (min(inner.shape) + 1) // 2
        assert len(colors) <= ring_capacity, "interior cannot represent more color ranks than ring depths"

        ordered_colors = _sorted_colors_by_area(inner)
        out[r0 + 1 : r1, c0 + 1 : c1] = _paint_concentric(inner.shape, ordered_colors)

    return out


def _sample_region_shape(n_colors: int) -> tuple[int, int]:
    extra_layers = random.randint(0, EXTRA_LAYER_MAX)
    ring_depths = n_colors if extra_layers == 0 else n_colors + extra_layers
    short_dim = random.choice((2 * ring_depths - 1, 2 * ring_depths))
    short_dim = min(short_dim, MAX_SHORT_DIM)
    long_dim = random.randint(short_dim, short_dim + MAX_LONG_EXTRA)
    if random.random() < 0.5:
        return short_dim, long_dim
    return long_dim, short_dim


def _sample_distinct_areas(total: int, n_parts: int) -> list[int] | None:
    if total < n_parts * MIN_AREA_PER_COLOR:
        return None

    for _ in range(200):
        cuts = sorted(random.sample(range(1, total), n_parts - 1))
        parts = [cuts[0], *[b - a for a, b in zip(cuts, cuts[1:])], total - cuts[-1]]
        if min(parts) < MIN_AREA_PER_COLOR:
            continue
        if len(set(parts)) != n_parts:
            continue
        return parts
    return None


def _sample_seeds(h: int, w: int, n_regions: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(h) for c in range(w)]
    min_gap = 3 if min(h, w) >= 6 else 2
    for _ in range(50):
        random.shuffle(cells)
        seeds: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - sr) + abs(cell[1] - sc) >= min_gap for sr, sc in seeds):
                seeds.append(cell)
                if len(seeds) == n_regions:
                    return seeds
    return random.sample(cells, n_regions)


def _is_solid_rectangle(cells: set[tuple[int, int]]) -> bool:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return len(cells) == (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _grow_partition(h: int, w: int, target_areas: list[int]) -> np.ndarray | None:
    n_regions = len(target_areas)
    total = h * w
    assert sum(target_areas) == total

    for _ in range(PARTITION_TRIES):
        owner = np.full((h, w), -1, dtype=int)
        seeds = _sample_seeds(h, w, n_regions)
        region_cells = [set() for _ in range(n_regions)]
        frontiers = [set() for _ in range(n_regions)]

        for idx, (r, c) in enumerate(seeds):
            owner[r, c] = idx
            region_cells[idx].add((r, c))

        for idx, (r, c) in enumerate(seeds):
            for nr, nc in _neighbors4(r, c, h, w):
                if owner[nr, nc] == -1:
                    frontiers[idx].add((nr, nc))

        assigned = n_regions
        while assigned < total:
            candidates = [
                idx
                for idx in range(n_regions)
                if len(region_cells[idx]) < target_areas[idx] and frontiers[idx]
            ]
            if not candidates:
                break

            weights = [target_areas[idx] - len(region_cells[idx]) for idx in candidates]
            idx = random.choices(candidates, weights=weights, k=1)[0]
            cells = [cell for cell in frontiers[idx] if owner[cell] == -1]
            if not cells:
                frontiers[idx].clear()
                continue

            cell_weights = []
            for r, c in cells:
                open_degree = sum(1 for nr, nc in _neighbors4(r, c, h, w) if owner[nr, nc] == -1)
                cell_weights.append(open_degree + 1)
            r, c = random.choices(cells, weights=cell_weights, k=1)[0]

            for frontier in frontiers:
                frontier.discard((r, c))
            owner[r, c] = idx
            region_cells[idx].add((r, c))
            assigned += 1

            for nr, nc in _neighbors4(r, c, h, w):
                if owner[nr, nc] == -1:
                    frontiers[idx].add((nr, nc))

        if assigned != total or bool(np.any(owner == -1)):
            continue
        if all(_is_solid_rectangle(cells) for cells in region_cells):
            continue
        return owner

    return None


def _build_region() -> np.ndarray | None:
    n_colors = random.randint(2, 4)
    inner_h, inner_w = _sample_region_shape(n_colors)
    total = inner_h * inner_w
    areas = _sample_distinct_areas(total, n_colors)
    if areas is None:
        return None

    partition = _grow_partition(inner_h, inner_w, areas)
    if partition is None:
        return None

    palette = random.sample(INNER_COLORS, n_colors)
    inner = np.empty((inner_h, inner_w), dtype=int)
    for idx, color in enumerate(palette):
        inner[partition == idx] = color

    region = np.full((inner_h + 2, inner_w + 2), BLUE, dtype=int)
    region[1:-1, 1:-1] = inner

    unique_counts = sorted(Counter(map(int, inner.reshape(-1))).values(), reverse=True)
    if len(set(unique_counts)) != len(unique_counts):
        return None
    if np.array_equal(region[1:-1, 1:-1], _paint_concentric(inner.shape, _sorted_colors_by_area(inner))):
        return None
    return region


def _can_place(occupied: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    rr0 = top - 1
    cc0 = left - 1
    rr1 = top + height + 1
    cc1 = left + width + 1
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_region(grid: np.ndarray, occupied: np.ndarray, region: np.ndarray) -> bool:
    h, w = grid.shape
    rh, rw = region.shape
    if rh + 2 > h or rw + 2 > w:
        return False

    for _ in range(PLACEMENT_TRIES):
        top = random.randint(1, h - rh - 1)
        left = random.randint(1, w - rw - 1)
        if not _can_place(occupied, top, left, rh, rw):
            continue

        grid[top : top + rh, left : left + rw] = region
        occupied[top - 1 : top + rh + 1, left - 1 : left + rw + 1] = True
        return True

    return False


def _attempt_grid() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    n_regions = random.randint(*REGION_COUNT_RANGE)
    placed = 0
    for _ in range(n_regions):
        region = _build_region()
        if region is None:
            return None
        if not _place_region(grid, occupied, region):
            return None
        placed += 1

    if placed != n_regions:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_grid()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
