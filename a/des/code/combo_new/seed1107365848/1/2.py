from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GEN_ATTEMPTS = 400
REGION_TRIES = 160
PLACEMENT_PAD = 1


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _filled_boundary(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    boundary = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w and filled[nr, nc]):
                    boundary[r, c] = True
                    break
    return boundary


def _inner_wall_mask(interior: np.ndarray, boundary: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    inner = np.zeros_like(interior, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not interior[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and boundary[nr, nc]:
                    inner[r, c] = True
                    break
    return inner


def _analyze_regions(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    regions: list[dict[str, object]] = []
    for cells in _components(grid != GREY):
        if _touches_border(cells, h, w):
            continue
        seed_cells = [(r, c) for r, c in cells if int(grid[r, c]) not in (BLACK, GREY)]
        if not seed_cells:
            continue
        colors = {int(grid[r, c]) for r, c in seed_cells}
        assert len(colors) == 1, "each closed region must have a unique seed color"
        assert 1 <= len(seed_cells) <= 2, "each closed region must contain one or two seeds"
        color = next(iter(colors))
        cell_set = set(cells)
        inner_wall = [
            (r, c)
            for r, c in cells
            if any(
                0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == GREY
                for nr, nc in _neighbors4(r, c)
            )
        ]
        core = [cell for cell in cells if cell not in inner_wall]
        regions.append(
            {
                "cells": cells,
                "cell_set": cell_set,
                "seed_cells": seed_cells,
                "seed_count": len(seed_cells),
                "color": color,
                "inner_wall": inner_wall,
                "core": core,
            }
        )
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Paint each closed grey-bounded region with a seed-colored inner wall, and fully fill only the two-seed regions.

    Input structure:
    - The background is black.
    - Several simple closed regions are outlined by one-cell-thick grey orthogonal polygons.
    - Each closed region contains one or two isolated seed pixels of a single non-black, non-grey color.

    Rule:
    - Keep every grey boundary pixel unchanged.
    - For each enclosed region, read the unique seed color C inside it.
    - Recolor every interior cell that is 4-adjacent to the grey boundary to C, forming a one-cell inner wall.
    - If the region contains two seed pixels, also recolor every remaining interior black cell to C; if it contains one seed pixel, leave the remaining interior cells black.

    Key insight:
    - The seed color determines which color to paint, while the number of seeds independently determines whether to paint only the inner border or the whole enclosed interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for region in _analyze_regions(g):
        color = int(region["color"])
        for r, c in region["inner_wall"]:
            out[r, c] = color
        if int(region["seed_count"]) == 2:
            for r, c in region["cells"]:
                out[r, c] = color
    return out


def _rectangle_mask(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    h = rng.randint(5, max(5, min(10, max_h)))
    w = rng.randint(5, max(5, min(10, max_w)))
    return np.ones((h, w), dtype=bool)


def _l_mask(rng: random.Random, max_h: int, max_w: int) -> np.ndarray:
    h = rng.randint(8, max(8, min(12, max_h)))
    w = rng.randint(8, max(8, min(12, max_w)))
    vert_thickness = rng.randint(4, w - 3)
    horiz_thickness = rng.randint(4, h - 3)
    mask = np.zeros((h, w), dtype=bool)
    mask[:, :vert_thickness] = True
    mask[h - horiz_thickness :, :] = True
    return np.rot90(mask, rng.randint(0, 3))


def _random_shape(
    rng: random.Random,
    shape_kind: str,
    max_h: int,
    max_w: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    if shape_kind == "rectangle":
        filled = _rectangle_mask(rng, max_h, max_w)
    else:
        if max_h < 8 or max_w < 8:
            return None
        filled = _l_mask(rng, max_h, max_w)
    boundary = _filled_boundary(filled)
    interior = filled & ~boundary
    inner_wall = _inner_wall_mask(interior, boundary)
    core = interior & ~inner_wall
    if interior.sum() < 3 or core.sum() == 0:
        return None
    return filled, boundary, interior, core


def _positions_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r, c in np.argwhere(mask)]


def _choose_seed_positions(
    rng: random.Random,
    interior: np.ndarray,
    core: np.ndarray,
    seed_count: int,
) -> list[tuple[int, int]] | None:
    interior_cells = _positions_from_mask(interior)
    core_cells = _positions_from_mask(core)
    core_set = set(core_cells)
    inner_wall_cells = [cell for cell in interior_cells if cell not in core_set]
    if seed_count == 1:
        if len(core_cells) >= 2:
            return [rng.choice(core_cells)]
        if inner_wall_cells and core_cells:
            return [rng.choice(inner_wall_cells)]
        return None

    valid_pairs = [
        (a, b)
        for i, a in enumerate(interior_cells)
        for b in interior_cells[i + 1 :]
        if max(abs(a[0] - b[0]), abs(a[1] - b[1])) > 1
    ]
    if not valid_pairs:
        return None
    pair = rng.choice(valid_pairs)
    return [pair[0], pair[1]]


def _shape_kinds(rng: random.Random, count: int) -> list[str]:
    kinds = [rng.choice(["rectangle", "lshape"]) for _ in range(count)]
    if count >= 2 and len(set(kinds)) == 1:
        kinds[0] = "rectangle"
        kinds[1] = "lshape"
    rng.shuffle(kinds)
    return kinds


def _seed_counts(rng: random.Random, count: int) -> list[int]:
    counts = [1, 2] * (count // 2)
    if count % 2:
        counts.append(rng.choice([1, 2]))
    rng.shuffle(counts)
    return counts


def _can_place(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> bool:
    gh, gw = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    return not reserved[r0:r1, c0:c1].any()


def _reserve(reserved: np.ndarray, top: int, left: int, h: int, w: int, pad: int) -> None:
    gh, gw = reserved.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(gh, top + h + pad)
    c1 = min(gw, left + w + pad)
    reserved[r0:r1, c0:c1] = True


def _place_region(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    color: int,
    seed_count: int,
    shape_kind: str,
) -> bool:
    gh, gw = grid.shape
    max_h = min(12 if shape_kind == "lshape" else 10, gh - 2)
    max_w = min(12 if shape_kind == "lshape" else 10, gw - 2)
    for _ in range(REGION_TRIES):
        shape = _random_shape(rng, "rectangle" if shape_kind == "rectangle" else "lshape", max_h, max_w)
        if shape is None:
            continue
        filled, boundary, interior, core = shape
        sh, sw = filled.shape
        if gh - sh - 1 < 1 or gw - sw - 1 < 1:
            continue
        seeds = _choose_seed_positions(rng, interior, core, seed_count)
        if seeds is None:
            continue
        top = rng.randint(1, gh - sh - 1)
        left = rng.randint(1, gw - sw - 1)
        if not _can_place(reserved, top, left, sh, sw, PLACEMENT_PAD):
            continue

        rows, cols = np.where(boundary)
        grid[top + rows, left + cols] = GREY
        for sr, sc in seeds:
            grid[top + sr, left + sc] = color
        _reserve(reserved, top, left, sh, sw, PLACEMENT_PAD)
        return True
    return False


def _validate_generated_grid(grid: np.ndarray, expected_regions: int) -> bool:
    regions = _analyze_regions(grid)
    if len(regions) != expected_regions:
        return False
    seed_counts = {int(region["seed_count"]) for region in regions}
    if seed_counts != {1, 2}:
        return False
    if any(not region["core"] for region in regions):
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    for region in regions:
        black_left = sum(int(out[r, c]) == BLACK for r, c in region["cells"])
        if int(region["seed_count"]) == 1 and black_left == 0:
            return False
        if int(region["seed_count"]) == 2 and black_left != 0:
            return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    gh = rng.randint(*GRID_H_RANGE)
    gw = rng.randint(*GRID_W_RANGE)
    grid = np.full((gh, gw), BLACK, dtype=int)
    reserved = np.zeros((gh, gw), dtype=bool)

    max_regions = min(5, max(2, (gh * gw) // 170 + 1))
    region_count = rng.randint(2, max_regions)
    colors = rng.sample(COLOR_POOL, region_count)
    seed_counts = _seed_counts(rng, region_count)
    shape_kinds = _shape_kinds(rng, region_count)

    for color, seed_count, shape_kind in zip(colors, seed_counts, shape_kinds):
        if not _place_region(grid, reserved, rng, color, seed_count, shape_kind):
            return None

    if not _validate_generated_grid(grid, region_count):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to produce a valid puzzle instance")
