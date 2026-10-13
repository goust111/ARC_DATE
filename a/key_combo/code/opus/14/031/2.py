from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _outside_black_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BLACK and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and grid[nr, nc] == BLACK and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))
    return outside


def _interior_black_mask(grid: np.ndarray) -> np.ndarray:
    outside = _outside_black_mask(grid)
    return (grid == BLACK) & ~outside


def _seed_regions(grid: np.ndarray, interior: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(grid == RED)]
    regions: list[tuple[tuple[int, int], np.ndarray]] = []
    assigned = np.zeros_like(interior, dtype=bool)
    for seed in seeds:
        start = None
        for nr, nc in _neighbors4(*seed):
            if _in_bounds(grid.shape, nr, nc) and interior[nr, nc]:
                start = (nr, nc)
                break
        assert start is not None, "seed must touch its enclosed black interior"
        q = deque([start])
        mask = np.zeros_like(interior, dtype=bool)
        mask[start] = True
        assigned[start] = True
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if _in_bounds(grid.shape, nr, nc) and interior[nr, nc] and not mask[nr, nc]:
                    mask[nr, nc] = True
                    assigned[nr, nc] = True
                    q.append((nr, nc))
        regions.append((seed, mask))
    assert np.array_equal(assigned, interior), "every enclosed black cell must belong to one seeded region"
    return regions


def _ray_cells_for_seed(grid: np.ndarray, seed: tuple[int, int]) -> list[tuple[int, int]]:
    cells = []
    for dr, dc in DIRS4:
        r, c = seed[0] + dr, seed[1] + dc
        advanced = False
        while _in_bounds(grid.shape, r, c) and grid[r, c] != GREEN:
            assert grid[r, c] == BLACK, "rays may only pass through black interior cells"
            cells.append((r, c))
            r += dr
            c += dc
            advanced = True
        assert advanced, "each direction must have a non-empty ray"
        assert _in_bounds(grid.shape, r, c) and grid[r, c] == GREEN, "each ray must stop at a green boundary"
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each red seed as four rays inside its green enclosure, then flood-fill only the even-area ray-split subregions.

    Input structure:
    - Black is the background.
    - Green pixels form 1-pixel-wide closed boundaries around 2-3 separate enclosed regions.
    - Each enclosed region contains exactly one red seed pixel on black interior cells.

    Rule:
    - Flood-fill black from the outer canvas border; black cells not reached are enclosed interior cells.
    - For each enclosed region, extend the red seed up, down, left, and right through black cells until the ray reaches the green boundary, without repainting the boundary.
    - The rays split the remaining enclosed black cells into 4-connected subregions; recolor each even-area subregion yellow and leave each odd-area subregion black.
    - Keep all original green boundary pixels, the red seeds, and the red ray pixels unchanged.

    Key insight:
    - The green outline first defines which black cells are truly inside, and the red cross then changes the topology of that interior so area parity can be tested on the ray-cut pieces rather than on the whole region.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    interior = _interior_black_mask(g)
    regions = _seed_regions(g, interior)

    out = g.copy()
    for seed, _region in regions:
        for r, c in _ray_cells_for_seed(g, seed):
            out[r, c] = RED

    remaining = (out == BLACK) & interior
    for comp in _components(remaining):
        if len(comp) % 2 == 0:
            out[comp[:, 0], comp[:, 1]] = YELLOW
    return out


def _shape_boundary(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    boundary = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        for nr, nc in _neighbors4(int(r), int(c)):
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                boundary[r, c] = True
                break
    return boundary


def _random_walk_extents(length: int, anchor: int, lo: int, hi: int) -> list[int]:
    values = [0] * length
    values[anchor] = random.randint(lo, hi)
    for r in range(anchor - 1, -1, -1):
        values[r] = min(hi, max(lo, values[r + 1] + random.choice([-1, 0, 1])))
    for r in range(anchor + 1, length):
        values[r] = min(hi, max(lo, values[r - 1] + random.choice([-1, 0, 1])))
    return values


def _build_shape() -> dict | None:
    h = random.randint(7, 9)
    sr = random.randint(2, h - 3)
    left_ext = _random_walk_extents(h, sr, 2, 4)
    right_ext = _random_walk_extents(h, sr, 2, 4)
    if len(set(left_ext)) == 1 and len(set(right_ext)) == 1:
        return None

    max_left = max(left_ext)
    max_right = max(right_ext)
    w = max_left + max_right + 1
    sc = max_left

    mask = np.zeros((h, w), dtype=bool)
    for r in range(h):
        c0 = sc - left_ext[r]
        c1 = sc + right_ext[r]
        mask[r, c0 : c1 + 1] = True

    boundary = _shape_boundary(mask)
    interior = mask & ~boundary
    interior_area = int(interior.sum())
    if not (15 <= interior_area <= 40):
        return None
    if not interior[sr, sc]:
        return None
    if sr == (h - 1) // 2 and sc == (w - 1) // 2:
        return None
    if np.all(mask):
        return None

    ray_mask = np.zeros_like(mask, dtype=bool)
    ray_mask[sr, sc] = True
    for r in range(sr - 1, -1, -1):
        if boundary[r, sc]:
            break
        ray_mask[r, sc] = True
    else:
        return None
    if not boundary[0, sc]:
        return None
    for r in range(sr + 1, h):
        if boundary[r, sc]:
            break
        ray_mask[r, sc] = True
    else:
        return None
    if not boundary[h - 1, sc]:
        return None
    for c in range(sc - 1, -1, -1):
        if boundary[sr, c]:
            break
        ray_mask[sr, c] = True
    else:
        return None
    if not boundary[sr, 0]:
        return None
    for c in range(sc + 1, w):
        if boundary[sr, c]:
            break
        ray_mask[sr, c] = True
    else:
        return None
    if not boundary[sr, w - 1]:
        return None

    remaining = interior & ~ray_mask
    comps = _components(remaining)
    if len(comps) != 4:
        return None
    areas = [len(comp) for comp in comps]
    if not any(area % 2 == 0 for area in areas):
        return None
    if not any(area % 2 == 1 for area in areas):
        return None

    return {
        "mask": mask,
        "boundary": boundary,
        "interior": interior,
        "seed": (sr, sc),
        "ray_mask": ray_mask,
        "areas": areas,
    }


def _placeable(occupied: np.ndarray, shape_mask: np.ndarray, top: int, left: int) -> bool:
    h, w = shape_mask.shape
    oh, ow = occupied.shape
    if top < 0 or left < 0 or top + h > oh or left + w > ow:
        return False
    for r in range(h):
        for c in range(w):
            if not shape_mask[r, c]:
                continue
            for nr in range(top + r - 1, top + r + 2):
                for nc in range(left + c - 1, left + c + 2):
                    if 0 <= nr < oh and 0 <= nc < ow and occupied[nr, nc]:
                        return False
    return True


def _compose_scene(shapes: list[dict], grid_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    grid = np.zeros(grid_shape, dtype=int)
    expected = np.zeros(grid_shape, dtype=int)
    occupied = np.zeros(grid_shape, dtype=bool)
    placed = []

    for shape in shapes:
        mask = shape["mask"]
        h, w = mask.shape
        options = []
        for top in range(1, grid_shape[0] - h):
            for left in range(1, grid_shape[1] - w):
                if _placeable(occupied, mask, top, left):
                    options.append((top, left))
        if not options:
            raise ValueError("failed to place generated shape")
        top, left = random.choice(options)
        placed.append((shape, top, left))
        occupied[top : top + h, left : left + w] |= mask

    for shape, top, left in placed:
        boundary = shape["boundary"]
        interior = shape["interior"]
        seed_r, seed_c = shape["seed"]
        ray_mask = shape["ray_mask"]
        grid[top : top + boundary.shape[0], left : left + boundary.shape[1]][boundary] = GREEN
        grid[top + seed_r, left + seed_c] = RED

        local_out = np.zeros_like(boundary, dtype=int)
        local_out[boundary] = GREEN
        local_out[ray_mask] = RED
        remaining = interior & ~ray_mask
        for comp in _components(remaining):
            color = YELLOW if len(comp) % 2 == 0 else BLACK
            local_out[comp[:, 0], comp[:, 1]] = color
        expected[top : top + local_out.shape[0], left : left + local_out.shape[1]] = np.maximum(
            expected[top : top + local_out.shape[0], left : left + local_out.shape[1]], local_out
        )
    return grid, expected


def _no_ray_shortcut_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    interior = _interior_black_mask(grid)
    for comp in _components(interior):
        if len(comp) % 2 == 0:
            out[comp[:, 0], comp[:, 1]] = YELLOW
    return out


def _fill_all_subregions_output(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    interior = _interior_black_mask(grid)
    regions = _seed_regions(grid, interior)
    for seed, _region in regions:
        for r, c in _ray_cells_for_seed(grid, seed):
            out[r, c] = RED
    out[(out == BLACK) & interior] = YELLOW
    return out


def _scene_valid(grid: np.ndarray, expected: np.ndarray) -> bool:
    if not (grid.ndim == 2 and max(grid.shape) <= 30):
        return False
    if not (2 <= np.sum(grid == RED) <= 3):
        return False
    if np.array_equal(grid, expected):
        return False
    if np.array_equal(expected, _no_ray_shortcut_output(grid)):
        return False
    if np.array_equal(expected, _fill_all_subregions_output(grid)):
        return False
    return np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        n_shapes = random.choice([2, 2, 3])
        shapes = []
        for _ in range(n_shapes):
            shape = None
            for _ in range(120):
                shape = _build_shape()
                if shape is not None:
                    break
            if shape is None:
                break
            shapes.append(shape)
        if len(shapes) != n_shapes:
            continue

        h = random.randint(18, 26)
        w = random.randint(18, 26)
        try:
            grid, expected = _compose_scene(shapes, (h, w))
        except ValueError:
            continue
        if _scene_valid(grid, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
