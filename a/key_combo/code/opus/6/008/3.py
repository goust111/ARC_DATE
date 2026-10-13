from __future__ import annotations

import random
from typing import Iterable

import numpy as np


GRID_SIZE = 16
QUAD_SIZE = 8
ANCHOR_SIZE = 3
BG = 0

QUADRANT_NAMES = ("TL", "TR", "BL", "BR")
QUADRANT_OFFSETS = {
    "TL": (0, 0),
    "TR": (0, QUAD_SIZE),
    "BL": (QUAD_SIZE, 0),
    "BR": (QUAD_SIZE, QUAD_SIZE),
}
ROTATIONS = {
    "TL": 2,
    "TR": -1,
    "BL": 1,
    "BR": 0,
}

RAY_STEPS = {
    "N": (-1, 0),
    "S": (1, 0),
    "W": (0, -1),
    "E": (0, 1),
    "NW": (-1, -1),
    "NE": (-1, 1),
    "SW": (1, -1),
    "SE": (1, 1),
}
RAY_START_OFFSETS = {
    "N": (0, 1),
    "S": (2, 1),
    "W": (1, 0),
    "E": (1, 2),
    "NW": (0, 0),
    "NE": (0, 2),
    "SW": (2, 0),
    "SE": (2, 2),
}

GEN_INPUT_RAYS = {
    "TL": ("N", "W", "NW"),
    "TR": ("E", "S", "SE"),
    "BL": ("E", "S", "SE"),
    "BR": ("N", "W", "NW"),
}
GEN_OUTPUT_RAYS = {
    "TL": {"E", "S", "SE"},
    "TR": {"W", "S", "SW"},
    "BL": {"E", "N", "NE"},
    "BR": {"W", "N", "NW"},
}

GEN_ANCHOR_TOPLEFTS = ((2, 2), (2, 3), (3, 2), (3, 3))
GEN_COLORS = (1, 2, 3, 4)
GEN_RETRIES = 500


def _nonzero_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(v) for v in np.unique(grid) if int(v) != BG)


def _ray_cells(anchor_rc: tuple[int, int]) -> dict[str, list[tuple[int, int]]]:
    """Return per-ray cells in order from nearest to farthest from the anchor block."""
    r0, c0 = map(int, anchor_rc)
    rays: dict[str, list[tuple[int, int]]] = {}
    for name, (dr, dc) in RAY_STEPS.items():
        sr, sc = RAY_START_OFFSETS[name]
        r = r0 + sr + dr
        c = c0 + sc + dc
        cells: list[tuple[int, int]] = []
        while 0 < r < QUAD_SIZE - 1 and 0 < c < QUAD_SIZE - 1:
            cells.append((int(r), int(c)))
            r += dr
            c += dc
        rays[name] = cells
    return rays


def _find_anchor(quadrant: np.ndarray, color: int) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    for r0 in range(QUAD_SIZE - ANCHOR_SIZE + 1):
        for c0 in range(QUAD_SIZE - ANCHOR_SIZE + 1):
            if np.all(quadrant[r0 : r0 + 3, c0 : c0 + 3] == color):
                candidates.append((r0, c0))
    assert len(candidates) == 1, f"expected one 3x3 anchor, got {candidates}"
    return candidates[0]


def _occupied_ray_map(
    quadrant: np.ndarray, color: int, anchor_rc: tuple[int, int]
) -> dict[str, list[tuple[int, int]]]:
    rays = _ray_cells(anchor_rc)
    ray_by_cell: dict[tuple[int, int], str] = {}
    for ray_name, cells in rays.items():
        for cell in cells:
            assert cell not in ray_by_cell
            ray_by_cell[cell] = ray_name

    occupied: dict[str, list[tuple[int, int]]] = {name: [] for name in rays}
    for r, c in np.argwhere(quadrant == color):
        rr = int(r)
        cc = int(c)
        if anchor_rc[0] <= rr < anchor_rc[0] + 3 and anchor_rc[1] <= cc < anchor_rc[1] + 3:
            continue
        cell = (rr, cc)
        assert cell in ray_by_cell, f"pixel {cell} is not on a valid anchor ray"
        occupied[ray_by_cell[cell]].append(cell)
    for cells in occupied.values():
        cells.sort()
    return occupied


def _compress_quadrant(quadrant: np.ndarray) -> np.ndarray:
    q = np.asarray(quadrant, dtype=int)
    colors = _nonzero_colors(q)
    assert len(colors) == 1, f"quadrant must contain exactly one foreground color, got {colors}"
    color = colors[0]

    anchor_rc = _find_anchor(q, color)
    occupied = _occupied_ray_map(q, color, anchor_rc)
    rays = _ray_cells(anchor_rc)

    out = q.copy()
    mask = (out == color)
    mask[anchor_rc[0] : anchor_rc[0] + 3, anchor_rc[1] : anchor_rc[1] + 3] = False
    out[mask] = BG

    for ray_name, old_cells in occupied.items():
        if not old_cells:
            continue
        ray_cells = rays[ray_name]
        count = len(old_cells)
        assert count <= len(ray_cells), f"ray {ray_name} overfilled"
        for r, c in ray_cells[:count]:
            out[r, c] = color
    return out


def _rotate_quadrant(quadrant: np.ndarray, quadrant_name: str) -> np.ndarray:
    k = ROTATIONS[quadrant_name]
    return np.rot90(quadrant, k=k)


def _quadrant_slices(name: str) -> tuple[slice, slice]:
    r0, c0 = QUADRANT_OFFSETS[name]
    return slice(r0, r0 + QUAD_SIZE), slice(c0, c0 + QUAD_SIZE)


def _transform(grid: np.ndarray, *, do_compress: bool, do_rotate: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE), f"expected 16x16 grid, got {g.shape}"

    out = np.zeros_like(g)
    for name in QUADRANT_NAMES:
        rs, cs = _quadrant_slices(name)
        quad = g[rs, cs]
        work = quad.copy()
        if do_compress:
            work = _compress_quadrant(work)
        if do_rotate:
            work = _rotate_quadrant(work, name)
        out[rs, cs] = work
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress each quadrant's ray pixels against its anchor block, then rotate the compressed quadrant.

    Input structure:
    - The input is a 16x16 grid split into four 8x8 quadrants with black background.
    - Each quadrant contains exactly one monochrome 3x3 solid anchor block and only same-color scatter pixels besides it.
    - Every scatter pixel lies on one of the anchor's row, column, or diagonal rays and stays strictly inside the quadrant border.

    Rule:
    - In each quadrant, slide every same-color scatter pixel toward the anchor along its own ray until the ray becomes a contiguous arm touching the anchor edge or corner.
    - Clear the original scattered positions, keeping the 3x3 anchor block fixed during this compression.
    - After compression, rotate the whole quadrant in place: upper-left 180 degrees, upper-right 90 degrees clockwise, lower-left 90 degrees counterclockwise, lower-right unchanged.

    Key insight:
    - The scattered pixels are interpreted relative to the local 3x3 anchor, not the whole grid; compression creates short directional arms, and the quadrant-specific rotations reorient those arms into the final symmetric pattern.
    """
    return _transform(grid, do_compress=True, do_rotate=True)


def _sample_counts(capacity_by_ray: dict[str, int]) -> dict[str, int] | None:
    rays = list(capacity_by_ray)
    usable = [ray for ray in rays if capacity_by_ray[ray] > 0]
    if len(usable) < 2:
        return None

    chosen_count = random.randint(2, len(usable))
    chosen = random.sample(usable, chosen_count)
    max_total = min(5, sum(capacity_by_ray[ray] for ray in chosen))
    if max_total < chosen_count:
        return None

    total = random.randint(chosen_count, max_total)
    counts = {ray: 0 for ray in rays}
    for ray in chosen:
        counts[ray] = 1
    remaining = total - chosen_count
    while remaining > 0:
        options = [ray for ray in chosen if counts[ray] < capacity_by_ray[ray]]
        if not options:
            return None
        pick = random.choice(options)
        counts[pick] += 1
        remaining -= 1
    return counts


def _sample_ray_positions(ray_cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    if count == 0:
        return []
    indices = list(range(len(ray_cells)))
    if count == len(indices):
        chosen = indices
    else:
        chosen = sorted(random.sample(indices, count))
    return [ray_cells[i] for i in chosen]


def _build_quadrant(
    quadrant_name: str, color: int
) -> tuple[np.ndarray, tuple[int, int], dict[str, list[tuple[int, int]]]]:
    q = np.zeros((QUAD_SIZE, QUAD_SIZE), dtype=int)
    anchor_rc = random.choice(GEN_ANCHOR_TOPLEFTS)
    r0, c0 = anchor_rc
    q[r0 : r0 + 3, c0 : c0 + 3] = color

    rays = _ray_cells(anchor_rc)
    allowed = GEN_INPUT_RAYS[quadrant_name]
    capacity_by_ray = {ray: len(rays[ray]) if ray in allowed else 0 for ray in RAY_STEPS}
    counts = _sample_counts(capacity_by_ray)
    assert counts is not None

    chosen_positions: dict[str, list[tuple[int, int]]] = {}
    for ray_name, count in counts.items():
        positions = _sample_ray_positions(rays[ray_name], count)
        chosen_positions[ray_name] = positions
        for r, c in positions:
            q[r, c] = color
    return q, anchor_rc, chosen_positions


def _prefix_like(positions: Iterable[tuple[int, int]], ray_cells: list[tuple[int, int]]) -> bool:
    positions = list(positions)
    if not positions:
        return True
    return positions == ray_cells[: len(positions)]


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
        rotate_only = _transform(grid, do_compress=False, do_rotate=True)
        compress_only = _transform(grid, do_compress=True, do_rotate=False)
    except Exception:
        return False

    if np.array_equal(out, rotate_only):
        return False
    if np.array_equal(out, compress_only):
        return False
    if np.array_equal(out, grid):
        return False

    moved_by_compression = False
    for name in QUADRANT_NAMES:
        rs, cs = _quadrant_slices(name)
        quad = grid[rs, cs]
        colors = _nonzero_colors(quad)
        if len(colors) != 1:
            return False
        color = colors[0]
        anchor_rc = _find_anchor(quad, color)
        occupied = _occupied_ray_map(quad, color, anchor_rc)
        active_rays = [ray for ray, cells in occupied.items() if cells]
        if not (2 <= sum(len(cells) for cells in occupied.values()) <= 5):
            return False
        if len(active_rays) < 2:
            return False
        if any(ray not in GEN_INPUT_RAYS[name] for ray in active_rays):
            return False
        rays = _ray_cells(anchor_rc)
        if any(not _prefix_like(occupied[ray], rays[ray]) for ray in active_rays):
            moved_by_compression = True

    if not moved_by_compression:
        return False

    for name in QUADRANT_NAMES:
        rs, cs = _quadrant_slices(name)
        quad = out[rs, cs]
        colors = _nonzero_colors(quad)
        if len(colors) != 1:
            return False
        color = colors[0]
        anchor_rc = _find_anchor(quad, color)
        occupied = _occupied_ray_map(quad, color, anchor_rc)
        active_rays = {ray for ray, cells in occupied.items() if cells}
        if not active_rays:
            return False
        if not active_rays.issubset(GEN_OUTPUT_RAYS[name]):
            return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
        colors = random.sample(GEN_COLORS, len(GEN_COLORS))
        for name, color in zip(QUADRANT_NAMES, colors):
            quad, _anchor_rc, _positions = _build_quadrant(name, color)
            rs, cs = _quadrant_slices(name)
            grid[rs, cs] = quad
        if _semantic_checks(grid):
            return grid
    raise RuntimeError("failed to generate a valid compressed-quadrant-rotation input")
