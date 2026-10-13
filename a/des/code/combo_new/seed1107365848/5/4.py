from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
WALL = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

# Generator bounds.
GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
CHAMBER_COUNT_CHOICES = (1, 2, 3, 4)
CHAMBER_COUNT_WEIGHTS = (1, 3, 3, 2)
PATCH_HEIGHT_RANGE = (7, 12)
PATCH_WIDTH_RANGE = (7, 12)
INNER_AREA_RATIO_RANGE = (0.25, 0.62)
MIN_CHAMBER_AREA = 6
SHAPE_ATTEMPTS = 100
GENERATION_ATTEMPTS = 400


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    offsets = NEIGHBORS4 if connectivity == 4 else NEIGHBORS8
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in offsets:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            comps.append(np.asarray(cells, dtype=int))

    return comps


def _touches_border(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return bool(
        np.any(
            (coords[:, 0] == 0)
            | (coords[:, 0] == h - 1)
            | (coords[:, 1] == 0)
            | (coords[:, 1] == w - 1)
        )
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return mask | (~mask & ~outside)


def _distance_map(mask: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = start
    assert mask[sr, sc]
    dist[sr, sc] = 0
    q = deque([start])

    while q:
        r, c = q.popleft()
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    assert np.all(dist[mask] >= 0)
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify each sealed chamber's seed and color only the odd BFS-distance layers inside it.

    Input structure:
    - The background is black.
    - Grey pixels form wall structures that partition the grid into an outside
      region plus one or more enclosed chambers.
    - Each enclosed chamber is a 4-connected non-grey region containing black
      cells and exactly one non-black, non-grey seed pixel.
    - Different chambers use different seed colors, and the walls remain fixed.

    Rule:
    - Treat grey pixels as barriers and find every 4-connected non-grey region
      that does not touch the grid border; each such region is one chamber.
    - Inside each chamber, run a 4-connected breadth-first expansion from the
      unique colored seed: the seed is layer 0, its black neighbors are layer 1,
      their unvisited black neighbors are layer 2, and so on.
    - Paint the seed and every odd-numbered layer in the seed's color, while
      keeping every even-numbered non-seed layer black.
    - Leave all grey wall pixels unchanged.

    Key insight:
    - The fill center is the chamber's unique seed pixel, not the chamber's
      geometric center, and the visible pattern depends on distance parity
      rather than on a plain flood fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape

    walkable = g != WALL
    seed_mask = (g != BLACK) & (g != WALL)
    total_seeds = int(seed_mask.sum())
    assert total_seeds > 0, "expected at least one colored seed"

    out = g.copy()
    chamber_count = 0
    processed_seeds = 0

    for comp in _components(walkable, connectivity=4):
        if _touches_border(comp, (h, w)):
            continue

        chamber_count += 1
        chamber_mask = np.zeros((h, w), dtype=bool)
        chamber_mask[comp[:, 0], comp[:, 1]] = True

        chamber_seeds = comp[seed_mask[comp[:, 0], comp[:, 1]]]
        assert len(chamber_seeds) == 1, "each chamber must contain exactly one seed"
        sr, sc = map(int, chamber_seeds[0])
        seed_color = int(g[sr, sc])
        processed_seeds += 1

        dist = _distance_map(chamber_mask, (sr, sc))
        for r, c in comp:
            rr = int(r)
            cc = int(c)
            d = int(dist[rr, cc])
            if d == 0 or d % 2 == 1:
                out[rr, cc] = seed_color
            else:
                out[rr, cc] = BLACK

    assert chamber_count > 0, "expected at least one enclosed chamber"
    assert processed_seeds == total_seeds, "all colored seeds must lie inside enclosed chambers"
    return out


def _grow_connected_blob(height: int, width: int, target_area: int) -> np.ndarray | None:
    if target_area <= 0 or target_area > height * width:
        return None

    start = (random.randrange(height), random.randrange(width))
    blob = {start}
    frontier: set[tuple[int, int]] = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in NEIGHBORS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in blob:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(blob) < target_area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        add_frontier(*cell)

    mask = np.zeros((height, width), dtype=bool)
    for r, c in blob:
        mask[r, c] = True
    return mask


def _shape_looks_irregular(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) < MIN_CHAMBER_AREA:
        return False

    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False

    area = len(coords)
    bbox_area = height * width
    if area == bbox_area:
        return False

    row_counts = [int(mask[r].sum()) for r in range(r0, r1 + 1) if mask[r].any()]
    col_counts = [int(mask[:, c].sum()) for c in range(c0, c1 + 1) if mask[:, c].any()]
    return len(set(row_counts)) > 1 or len(set(col_counts)) > 1


def _wall_from_interior(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    wall = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        for dr, dc in NEIGHBORS8:
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < h and 0 <= nc < w and not interior[nr, nc]:
                wall[nr, nc] = True
    return wall


def _trim_patch(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    assert len(coords) > 0
    r0, c0, r1, c1 = _bbox(coords)
    return patch[r0 : r1 + 1, c0 : c1 + 1].copy()


def _bbox_center_score(coords: np.ndarray, cell: tuple[int, int]) -> int:
    r0, c0, r1, c1 = _bbox(coords)
    r, c = cell
    return abs(2 * r - (r0 + r1)) + abs(2 * c - (c0 + c1))


def _generate_chamber_patch(color: int) -> tuple[np.ndarray, int, bool] | None:
    for _ in range(SHAPE_ATTEMPTS):
        ph = random.randint(*PATCH_HEIGHT_RANGE)
        pw = random.randint(*PATCH_WIDTH_RANGE)
        inner_h = ph - 2
        inner_w = pw - 2
        inner_area = inner_h * inner_w

        min_area = max(MIN_CHAMBER_AREA, int(np.ceil(inner_area * INNER_AREA_RATIO_RANGE[0])))
        max_area = min(inner_area - 1, int(np.floor(inner_area * INNER_AREA_RATIO_RANGE[1])))
        if min_area > max_area:
            continue

        interior_small = _grow_connected_blob(inner_h, inner_w, random.randint(min_area, max_area))
        if interior_small is None:
            continue
        interior_small = _fill_holes(interior_small)
        if not _shape_looks_irregular(interior_small):
            continue

        interior = np.zeros((ph, pw), dtype=bool)
        interior[1:-1, 1:-1] = interior_small
        wall = _wall_from_interior(interior)
        if len(_components(wall, connectivity=4)) != 1:
            continue

        interior_coords = np.argwhere(interior)
        cells = [tuple(map(int, rc)) for rc in interior_coords]
        eccentric_cells = [cell for cell in cells if _bbox_center_score(interior_coords, cell) >= 2]
        if eccentric_cells and random.random() < 0.75:
            seed_r, seed_c = random.choice(eccentric_cells)
        else:
            seed_r, seed_c = random.choice(cells)

        patch = np.full((ph, pw), BLACK, dtype=int)
        patch[wall] = WALL
        patch[seed_r, seed_c] = int(color)
        patch = _trim_patch(patch)

        walkable = patch != WALL
        chambers = [comp for comp in _components(walkable, connectivity=4) if not _touches_border(comp, patch.shape)]
        if len(chambers) != 1:
            continue
        chamber = chambers[0]

        chamber_mask = np.zeros(patch.shape, dtype=bool)
        chamber_mask[chamber[:, 0], chamber[:, 1]] = True
        chamber_seeds = chamber[patch[chamber[:, 0], chamber[:, 1]] != BLACK]
        if len(chamber_seeds) != 1:
            continue
        sr, sc = map(int, chamber_seeds[0])

        dist = _distance_map(chamber_mask, (sr, sc))
        max_dist = int(dist[chamber_mask].max())
        if max_dist < 2:
            continue

        off_center = _bbox_center_score(chamber, (sr, sc)) >= 2
        return patch, max_dist, off_center

    return None


def _place_patch(grid: np.ndarray, occupied: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - ph):
        for c0 in range(1, w - pw):
            rs = max(0, r0 - 1)
            re = min(h, r0 + ph + 1)
            cs = max(0, c0 - 1)
            ce = min(w, c0 + pw + 1)
            if occupied[rs:re, cs:ce].any():
                continue
            candidates.append((r0, c0))

    if not candidates:
        return False

    r0, c0 = random.choice(candidates)
    mask = patch != BLACK
    grid[r0 : r0 + ph, c0 : c0 + pw][mask] = patch[mask]
    occupied[r0 : r0 + ph, c0 : c0 + pw] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    chamber_count = random.choices(CHAMBER_COUNT_CHOICES, weights=CHAMBER_COUNT_WEIGHTS, k=1)[0]

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    colors = random.sample(SEED_COLORS, chamber_count)

    max_dists: list[int] = []
    off_center_flags: list[bool] = []
    for color in colors:
        chamber = _generate_chamber_patch(int(color))
        if chamber is None:
            return None
        patch, max_dist, off_center = chamber
        if not _place_patch(grid, occupied, patch):
            return None
        max_dists.append(max_dist)
        off_center_flags.append(off_center)

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if max(max_dists) < 3:
        return None
    if not any(off_center_flags):
        return None
    if int(np.count_nonzero(out != grid)) <= chamber_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-seed ARC input")
