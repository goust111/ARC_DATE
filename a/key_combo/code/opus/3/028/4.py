from __future__ import annotations

import itertools
import math
import random

import numpy as np


BG_COLOR = 0
FRACTAL_COLOR = 7
RAY_COLOR = 3
MASK_SIDE = 3
TILE_SIDE = 3
REGION_SIDE = MASK_SIDE * TILE_SIDE
COMPASS_DIRS = (
    (-1, 0),   # N
    (-1, 1),   # NE
    (0, 1),    # E
    (1, 1),    # SE
    (1, 0),    # S
    (1, -1),   # SW
    (0, -1),   # W
    (-1, -1),  # NW
)
CARDINAL_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
GEN_MAX_ATTEMPTS = 300
ACTIVE_COUNT_WEIGHTS = {
    3: 3,
    4: 4,
    5: 4,
    6: 3,
    7: 2,
}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _compress_block_mask(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert g.shape == (REGION_SIDE, REGION_SIDE), "expected a 9x9 input grid"
    assert set(map(int, np.unique(g))) <= {BG_COLOR, FRACTAL_COLOR}, "unexpected input colors"

    mask = np.zeros((MASK_SIDE, MASK_SIDE), dtype=bool)
    for br in range(MASK_SIDE):
        for bc in range(MASK_SIDE):
            tile = g[br * TILE_SIDE : (br + 1) * TILE_SIDE, bc * TILE_SIDE : (bc + 1) * TILE_SIDE]
            on = bool(tile[0, 0] == FRACTAL_COLOR)
            expected = FRACTAL_COLOR if on else BG_COLOR
            assert np.all(tile == expected), "each 3x3 source tile must be uniform"
            mask[br, bc] = on
    return mask


def _render_input(mask: np.ndarray) -> np.ndarray:
    tile = np.full((TILE_SIDE, TILE_SIDE), FRACTAL_COLOR, dtype=int)
    return np.where(np.kron(mask.astype(int), tile), FRACTAL_COLOR, BG_COLOR).astype(int)


def _kronecker_self(mask: np.ndarray) -> np.ndarray:
    return np.kron(mask.astype(bool), mask.astype(bool)).astype(bool)


def _centroid(mask: np.ndarray) -> tuple[float, float]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one active pixel"
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _four_neighbor_degree(mask: np.ndarray, r: int, c: int) -> int:
    return sum(
        1
        for dr, dc in CARDINAL_DIRS
        if _in_bounds(mask.shape, r + dr, c + dc) and bool(mask[r + dr, c + dc])
    )


def _corner_pixels(mask: np.ndarray) -> list[tuple[int, int]]:
    corners: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        if _four_neighbor_degree(mask, rr, cc) <= 1:
            corners.append((rr, cc))
    corners.sort()
    return corners


def _compass_index(delta_r: float, delta_c: float) -> int:
    assert delta_r != 0.0 or delta_c != 0.0, "direction from centroid must be nonzero"
    angle = (math.degrees(math.atan2(delta_c, -delta_r)) + 360.0) % 360.0
    return int(math.floor((angle + 22.5) / 45.0)) % 8


def _trace_ray(mask: np.ndarray, start: tuple[int, int], direction: tuple[int, int]) -> tuple[list[tuple[int, int]], bool]:
    h, w = map(int, mask.shape)
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w and not bool(mask[r, c]):
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells, 0 <= r < h and 0 <= c < w


def _ray_specs(mask: np.ndarray, center: tuple[float, float] | None = None) -> list[dict]:
    if center is None:
        center = _centroid(mask)
    center_r, center_c = center

    specs: list[dict] = []
    for r, c in _corner_pixels(mask):
        idx = _compass_index(r - center_r, c - center_c)
        cells, stops_on_hit = _trace_ray(mask, (r, c), COMPASS_DIRS[idx])
        specs.append(
            {
                "start": (r, c),
                "dir_index": idx,
                "cells": tuple(cells),
                "stops_on_hit": bool(stops_on_hit),
            }
        )
    return specs


def _paint_rays(mask: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = np.where(mask, FRACTAL_COLOR, BG_COLOR).astype(int)
    for spec in specs:
        for r, c in spec["cells"]:
            out[r, c] = RAY_COLOR
    return out


def _rotate_forms(mask: np.ndarray) -> set[tuple[int, ...]]:
    return {tuple(np.rot90(mask, k).astype(int).reshape(-1)) for k in range(4)}


def _bbox_center(mask: np.ndarray) -> tuple[float, float]:
    box = _bbox(mask)
    assert box is not None
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _analyze_mask(mask: np.ndarray) -> dict | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (MASK_SIDE, MASK_SIDE):
        return None

    active = int(mask.sum())
    if active < 3 or active >= MASK_SIDE * MASK_SIDE:
        return None
    if len(_rotate_forms(mask)) < 2:
        return None

    fractal = _kronecker_self(mask)
    try:
        specs = _ray_specs(fractal)
    except AssertionError:
        return None
    if not specs:
        return None

    used_cells: set[tuple[int, int]] = set()
    ray_cell_count = 0
    hit_count = 0
    edge_count = 0
    for spec in specs:
        cells = spec["cells"]
        if not used_cells.isdisjoint(cells):
            return None
        used_cells.update(cells)
        ray_cell_count += len(cells)
        if spec["stops_on_hit"]:
            hit_count += 1
        else:
            edge_count += 1

    if ray_cell_count < 4:
        return None
    if hit_count == 0 or edge_count == 0:
        return None

    output = _paint_rays(fractal, specs)
    try:
        fixed_center_output = _paint_rays(fractal, _ray_specs(fractal, center=(4.0, 4.0)))
        bbox_center_output = _paint_rays(fractal, _ray_specs(fractal, center=_bbox_center(fractal)))
    except AssertionError:
        return None
    if np.array_equal(output, fixed_center_output):
        return None

    if np.array_equal(output, bbox_center_output):
        return None

    dir_indices = {int(spec["dir_index"]) for spec in specs}
    has_cardinal = any(idx % 2 == 0 for idx in dir_indices)
    has_diagonal = any(idx % 2 == 1 for idx in dir_indices)
    if not (has_cardinal and has_diagonal):
        return None

    return {
        "mask": mask.copy(),
        "input": _render_input(mask),
        "fractal": fractal.copy(),
        "output": output,
        "active": active,
        "ray_cells": ray_cell_count,
        "hit_count": hit_count,
        "edge_count": edge_count,
    }


def _build_catalog() -> tuple[list[dict], dict[int, list[dict]]]:
    catalog: list[dict] = []
    by_active: dict[int, list[dict]] = {}
    for bits in itertools.product((0, 1), repeat=MASK_SIDE * MASK_SIDE):
        mask = np.array(bits, dtype=bool).reshape(MASK_SIDE, MASK_SIDE)
        record = _analyze_mask(mask)
        if record is None:
            continue
        catalog.append(record)
        by_active.setdefault(int(record["active"]), []).append(record)

    assert catalog, "expected at least one valid generator candidate"
    return catalog, by_active


_CATALOG, _CATALOG_BY_ACTIVE = _build_catalog()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the block-coded 3x3 mask into its Kronecker self-pattern and add centroid-directed corner rays.

    Input structure:
    - The grid is a 9x9 black background with orange foreground only.
    - The 9x9 area is partitioned into a 3x3 block grid of uniform 3x3 tiles.
    - Each tile is either fully orange or fully black, so the input encodes a 3x3 binary mask.

    Rule:
    - Compress the 9x9 block grid into its 3x3 on/off mask.
    - Replace that mask by its Kronecker self-product: each on cell becomes a copy of the whole 3x3 mask, and each off cell becomes a blank 3x3 block.
    - In the 9x9 Kronecker result, find every orange pixel whose 4-neighbor orange degree is at most 1.
    - Compute the geometric centroid of all orange pixels, quantize each corner pixel's centroid-relative vector to the nearest compass direction with clockwise tie-breaking, and extend a green ray away from the centroid until the next step would leave the grid or hit an orange pixel.

    Key insight:
    - The Kronecker fractal is both the source of the ray seeds and the obstacle field that truncates those rays, so the self-product must be constructed before the ray directions and stopping points are knowable.
    """
    mask = _compress_block_mask(grid)
    fractal = _kronecker_self(mask)
    return _paint_rays(fractal, _ray_specs(fractal))


def _sample_candidate() -> dict:
    active_counts = sorted(_CATALOG_BY_ACTIVE)
    weights = [ACTIVE_COUNT_WEIGHTS.get(count, 1) for count in active_counts]
    active = random.choices(active_counts, weights=weights, k=1)[0]
    return random.choice(_CATALOG_BY_ACTIVE[active])


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        record = _sample_candidate()
        inp = record["input"].copy()
        expected = record["output"]

        assert inp.shape == (REGION_SIDE, REGION_SIDE)
        assert set(map(int, np.unique(inp))) <= {BG_COLOR, FRACTAL_COLOR}
        assert np.array_equal(_compress_block_mask(inp), record["mask"])
        if not np.array_equal(main(inp), expected):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
