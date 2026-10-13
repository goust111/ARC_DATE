import random
from collections import deque

import numpy as np


BG = 0
GRAY = 5
RING_PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
GENERATE_RETRIES = 400
GRID_RANGE = (18, 29)
RING_COUNT_RANGE = (2, 4)
OBJECT_COUNT_RANGE = (3, 7)


BASE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
    ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 1), (2, 1)),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _normalize_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in shape)
    min_c = min(c for _, c in shape)
    return tuple(sorted((r - min_r, c - min_c) for r, c in shape))


def _rotate_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((c, -r) for r, c in shape)


def _flip_shape(shape: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple((r, -c) for r, c in shape)


def _all_shapes() -> tuple[tuple[tuple[int, int], ...], ...]:
    variants = set()
    for base in BASE_SHAPES:
        cur = tuple(base)
        for _ in range(4):
            variants.add(_normalize_shape(cur))
            variants.add(_normalize_shape(_flip_shape(cur)))
            cur = _rotate_shape(cur)
    return tuple(sorted(variants))


SHAPES = _all_shapes()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _draw_ring(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = rect
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _ring_points_by_color(grid: np.ndarray) -> dict[int, np.ndarray]:
    ring_points: dict[int, np.ndarray] = {}
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BG, GRAY)):
        ring_points[color] = np.argwhere(grid == color).astype(int)
    return ring_points


def _distance_profile(cells: list[tuple[int, int]], ring_points: np.ndarray) -> tuple[int, int, int]:
    cell_arr = np.asarray(cells, dtype=int)
    dists = np.abs(cell_arr[:, None, :] - ring_points[None, :, :]).sum(axis=2)
    nearest = dists.min(axis=1)
    return int(nearest.min()), int(nearest.sum()), int(ring_points.shape[0])


def _ray_tiebreak(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    candidate_colors: list[int],
) -> int:
    h, w = grid.shape
    seed_r, seed_c = min(cells)
    candidate_set = set(candidate_colors)
    for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        r, c = seed_r + dr, seed_c + dc
        while 0 <= r < h and 0 <= c < w:
            value = int(grid[r, c])
            if value not in (BG, GRAY):
                if value in candidate_set:
                    return value
                break
            r += dr
            c += dc
    return candidate_colors[0]


def _nearest_ring_color(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    ring_points: dict[int, np.ndarray],
) -> int:
    scored: list[tuple[tuple[int, int, int], int]] = []
    for color, points in ring_points.items():
        scored.append((_distance_profile(cells, points), int(color)))
    best_score = min(score for score, _color in scored)
    best_colors = [color for score, color in scored if score == best_score]
    if len(best_colors) == 1:
        return int(best_colors[0])
    return _ray_tiebreak(grid, cells, best_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray object to the color of the nearest surrounding rectangular ring.

    Input structure:
    - The background is black.
    - Several differently colored one-pixel-thick closed rectangular rings are nested with black gaps between them.
    - Small gray 4-connected objects sit on black cells inside the ring system.

    Rule:
    - Keep every colored ring unchanged.
    - For each gray object, compare how close that object is to each ring boundary.
    - The ring whose boundary is nearest to the object is the ring that directly surrounds that local black zone; a ray cast from the object reaches that ring before any farther ring in that zone.
    - Recolor the whole gray object to that ring color without moving or reshaping it.

    Key insight:
    - The target color comes from the spatial nesting geometry of the rings, not from the object's own shape or from a simple global left/right partition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    gray_mask = g == GRAY
    if not gray_mask.any():
        return out

    ring_points = _ring_points_by_color(g)
    assert ring_points, "expected at least one colored ring"

    for comp in _components(gray_mask):
        color = _nearest_ring_color(g, comp, ring_points)
        rows, cols = zip(*comp)
        out[rows, cols] = color
    return out


def _sample_rings() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]] | None:
    n_rings = random.randint(*RING_COUNT_RANGE)
    min_size = 5 + 4 * (n_rings - 1)
    h = random.randint(max(GRID_RANGE[0], min_size + 2), GRID_RANGE[1])
    w = random.randint(max(GRID_RANGE[0], min_size + 2), GRID_RANGE[1])

    outer_h = random.randint(min_size, h - 2)
    outer_w = random.randint(min_size, w - 2)
    r0 = random.randint(1, h - outer_h - 1)
    c0 = random.randint(1, w - outer_w - 1)
    rects = [(r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)]

    for depth in range(1, n_rings):
        prev_r0, prev_c0, prev_r1, prev_c1 = rects[-1]
        remaining_inner = n_rings - depth - 1
        min_inner = 5 + 4 * remaining_inner
        prev_h = prev_r1 - prev_r0 + 1
        prev_w = prev_c1 - prev_c0 + 1
        max_h = prev_h - 4
        max_w = prev_w - 4
        if max_h < min_inner or max_w < min_inner:
            return None
        inner_h = random.randint(min_inner, max_h)
        inner_w = random.randint(min_inner, max_w)
        row_lo = prev_r0 + 2
        row_hi = prev_r1 - 2 - inner_h + 1
        col_lo = prev_c0 + 2
        col_hi = prev_c1 - 2 - inner_w + 1
        if row_lo > row_hi or col_lo > col_hi:
            return None
        inner_r0 = random.randint(row_lo, row_hi)
        inner_c0 = random.randint(col_lo, col_hi)
        rects.append((inner_r0, inner_c0, inner_r0 + inner_h - 1, inner_c0 + inner_w - 1))

    colors = random.sample(RING_PALETTE, n_rings)
    grid = np.full((h, w), BG, dtype=int)
    for rect, color in zip(rects, colors):
        _draw_ring(grid, rect, color)
    return grid, rects, colors


def _cell_owner(
    r: int,
    c: int,
    ring_points: dict[int, np.ndarray],
) -> int | None:
    scored: list[tuple[tuple[int, int, int], int]] = []
    for color, points in ring_points.items():
        dists = np.abs(points[:, 0] - r) + np.abs(points[:, 1] - c)
        best = int(dists.min())
        scored.append(((best, best, int(points.shape[0])), int(color)))
    best_score = min(score for score, _color in scored)
    winners = [color for score, color in scored if score == best_score]
    if len(winners) == 1:
        return int(winners[0])
    return None


def _zone_masks(
    grid: np.ndarray,
    outer_rect: tuple[int, int, int, int],
    colors: list[int],
) -> dict[int, np.ndarray]:
    ring_points = _ring_points_by_color(grid)
    r0, c0, r1, c1 = outer_rect
    masks = {int(color): np.zeros(grid.shape, dtype=bool) for color in colors}
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if int(grid[r, c]) != BG:
                continue
            owner = _cell_owner(r, c, ring_points)
            if owner is not None:
                masks[owner][r, c] = True
    return masks


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr0 = max(0, -dr)
            rr1 = min(h, h - dr)
            cc0 = max(0, -dc)
            cc1 = min(w, w - dc)
            out[rr0:rr1, cc0:cc1] |= mask[rr0 + dr : rr1 + dr, cc0 + dc : cc1 + dc]
    return out


def _random_shape() -> tuple[tuple[int, int], ...]:
    return random.choice(SHAPES)


def _shape_cells(
    anchor_r: int,
    anchor_c: int,
    shape: tuple[tuple[int, int], ...],
) -> list[tuple[int, int]]:
    return [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]


def _candidate_placements(
    zone_mask: np.ndarray,
    blocked: np.ndarray,
    shape: tuple[tuple[int, int], ...],
) -> list[list[tuple[int, int]]]:
    h, w = zone_mask.shape
    shape_h = max(r for r, _ in shape) + 1
    shape_w = max(c for _, c in shape) + 1
    placements: list[list[tuple[int, int]]] = []
    for r0 in range(h - shape_h + 1):
        for c0 in range(w - shape_w + 1):
            cells = _shape_cells(r0, c0, shape)
            if all(zone_mask[r, c] and not blocked[r, c] for r, c in cells):
                placements.append(cells)
    random.shuffle(placements)
    return placements


def _place_objects(
    base_grid: np.ndarray,
    zone_masks: dict[int, np.ndarray],
    colors: list[int],
) -> tuple[np.ndarray, list[tuple[list[tuple[int, int]], int]]] | None:
    usable_colors = [color for color in colors if int(zone_masks[color].sum()) >= 2]
    if len(usable_colors) < 2:
        return None

    cover_count = random.randint(2, len(usable_colors))
    mandatory = random.sample(usable_colors, cover_count)
    object_count = random.randint(max(OBJECT_COUNT_RANGE[0], cover_count), OBJECT_COUNT_RANGE[1])
    desired_colors = mandatory + [random.choice(usable_colors) for _ in range(object_count - cover_count)]
    random.shuffle(desired_colors)

    grid = base_grid.copy()
    blocked = np.zeros(grid.shape, dtype=bool)
    ring_points = _ring_points_by_color(base_grid)
    placements: list[tuple[list[tuple[int, int]], int]] = []

    for target_color in desired_colors:
        placed = False
        for _ in range(40):
            shape = _random_shape()
            for cells in _candidate_placements(zone_masks[target_color], blocked, shape):
                if _nearest_ring_color(base_grid, cells, ring_points) != target_color:
                    continue
                rows, cols = zip(*cells)
                grid[rows, cols] = GRAY
                obj_mask = np.zeros(grid.shape, dtype=bool)
                obj_mask[rows, cols] = True
                blocked |= _dilate8(obj_mask)
                placements.append((cells, target_color))
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return grid, placements


def _expected_output(
    grid: np.ndarray,
    placements: list[tuple[list[tuple[int, int]], int]],
) -> np.ndarray:
    out = grid.copy()
    for cells, color in placements:
        rows, cols = zip(*cells)
        out[rows, cols] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    sampled = _sample_rings()
    if sampled is None:
        return None
    base_grid, rects, colors = sampled
    zones = _zone_masks(base_grid, rects[0], colors)
    placed = _place_objects(base_grid, zones, colors)
    if placed is None:
        return None
    grid, placements = placed
    expected = _expected_output(grid, placements)
    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATE_RETRIES} attempts")
