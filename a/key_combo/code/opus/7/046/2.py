from __future__ import annotations

from collections import deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
ORANGE = 7

BARRIER_COLORS = (BLUE, GREEN)
PASSABLE_COLORS = (BLACK, RED, YELLOW, ORANGE)
ORTHOGONAL_STEPS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_STEPS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_NEIGHBOR_STEPS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]

GEN_HEIGHT_RANGE = (14, 21)
GEN_WIDTH_RANGE = (14, 21)
GEN_RING_COUNT_CHOICES = (2, 3)
GEN_RING_OUTER_H_RANGE = (5, 9)
GEN_RING_OUTER_W_RANGE = (5, 9)
GEN_RING_MARGIN = 1
GEN_RING_SEPARATION = 2
GEN_STUB_COUNT_RANGE = (0, 3)
GEN_STUB_LENGTH_RANGE = (1, 3)
GEN_MAX_ATTEMPTS = 500


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    groups: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            groups.append(np.array(coords, dtype=int))
    return groups


def _touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(np.any((rows == 0) | (rows == h - 1) | (cols == 0) | (cols == w - 1)))


def _passable_mask(grid: np.ndarray) -> np.ndarray:
    return np.isin(grid, PASSABLE_COLORS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each red marker with a topology-selected halo based on whether its surrounding region reaches the grid edge.

    Input structure:
    - A black background is partitioned by blue and green barrier walls into several 4-connected non-barrier regions.
    - Some of those regions touch the grid boundary while others are fully enclosed holes.
    - Each marked region contains a red pixel that sits away from the barrier.

    Rule:
    - Treat blue and green cells as barriers, and classify every 4-connected non-barrier region by whether it touches the outer grid edge.
    - For each red pixel in an edge-touching region, color its orthogonal black neighbors yellow.
    - For each red pixel in an enclosed region, color its diagonal black neighbors orange.
    - Preserve all existing non-black cells, and never overwrite the red markers themselves.

    Key insight:
    - All markers have the same color, so the halo geometry is determined by the topology of the containing background region rather than by marker identity.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    out = g.copy()
    marker_count = 0
    for coords in _components(_passable_mask(g)):
        is_external = _touches_edge(coords, g.shape)
        halo_color = YELLOW if is_external else ORANGE
        deltas = ORTHOGONAL_STEPS if is_external else DIAGONAL_STEPS
        for r, c in coords:
            if g[r, c] != RED:
                continue
            marker_count += 1
            for dr, dc in deltas:
                rr, cc = int(r + dr), int(c + dc)
                if _in_bounds(g.shape, rr, cc) and out[rr, cc] == BLACK:
                    out[rr, cc] = halo_color

    assert marker_count > 0
    return out


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _paint_ring(mask: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True


def _place_rings(h: int, w: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    ring_count = rng.choice(GEN_RING_COUNT_CHOICES)
    boxes: list[tuple[int, int, int, int]] = []
    for _ in range(ring_count):
        placed = False
        for _ in range(120):
            outer_h = rng.randint(*GEN_RING_OUTER_H_RANGE)
            outer_w = rng.randint(*GEN_RING_OUTER_W_RANGE)
            if outer_h >= h - 2 * GEN_RING_MARGIN or outer_w >= w - 2 * GEN_RING_MARGIN:
                continue
            max_r0 = h - outer_h - GEN_RING_MARGIN
            max_c0 = w - outer_w - GEN_RING_MARGIN
            if max_r0 < GEN_RING_MARGIN or max_c0 < GEN_RING_MARGIN:
                continue
            r0 = rng.randint(GEN_RING_MARGIN, max_r0)
            c0 = rng.randint(GEN_RING_MARGIN, max_c0)
            box = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(box, other, GEN_RING_SEPARATION) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _try_add_stub(mask: np.ndarray, box: tuple[int, int, int, int], rng: random.Random) -> None:
    r0, c0, r1, c1 = box
    side = rng.choice(("top", "bottom", "left", "right"))
    length = rng.randint(*GEN_STUB_LENGTH_RANGE)

    if side == "top":
        if c1 - c0 < 2:
            return
        start = (r0, rng.randint(c0 + 1, c1 - 1))
        step = (-1, 0)
    elif side == "bottom":
        if c1 - c0 < 2:
            return
        start = (r1, rng.randint(c0 + 1, c1 - 1))
        step = (1, 0)
    elif side == "left":
        if r1 - r0 < 2:
            return
        start = (rng.randint(r0 + 1, r1 - 1), c0)
        step = (0, -1)
    else:
        if r1 - r0 < 2:
            return
        start = (rng.randint(r0 + 1, r1 - 1), c1)
        step = (0, 1)

    cells: list[tuple[int, int]] = []
    rr, cc = start
    for _ in range(length):
        rr += step[0]
        cc += step[1]
        if not _in_bounds(mask.shape, rr, cc) or mask[rr, cc]:
            return
        cells.append((rr, cc))
    for rr, cc in cells:
        mask[rr, cc] = True


def _build_barrier_mask(h: int, w: int, rng: random.Random) -> tuple[np.ndarray, list[tuple[int, int, int, int]]] | None:
    boxes = _place_rings(h, w, rng)
    if boxes is None:
        return None
    mask = np.zeros((h, w), dtype=bool)
    for box in boxes:
        _paint_ring(mask, box)
    for box in boxes:
        for _ in range(rng.randint(*GEN_STUB_COUNT_RANGE)):
            _try_add_stub(mask, box, rng)
    return mask, boxes


def _full_patch_candidates(coords: np.ndarray, region_mask: np.ndarray) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        ok = True
        for dr, dc in ALL_NEIGHBOR_STEPS:
            nr, nc = rr + dr, cc + dc
            if not _in_bounds(region_mask.shape, nr, nc) or not region_mask[nr, nc]:
                ok = False
                break
        if ok:
            candidates.append((rr, cc))
    return candidates


def _colorize_barriers(barrier_mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = barrier_mask.shape
    grid = np.zeros((h, w), dtype=int)
    phase = rng.randint(0, 1)
    for r, c in np.argwhere(barrier_mask):
        color = BLUE if (int(r) + int(c) + phase) % 2 == 0 else GREEN
        if rng.random() < 0.25:
            color = GREEN if color == BLUE else BLUE
        grid[int(r), int(c)] = color

    if not np.any(grid == BLUE):
        rr, cc = np.argwhere(barrier_mask)[0]
        grid[int(rr), int(cc)] = BLUE
    if not np.any(grid == GREEN):
        rr, cc = np.argwhere(barrier_mask)[-1]
        grid[int(rr), int(cc)] = GREEN
    return grid


def _place_markers(
    barrier_mask: np.ndarray,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = barrier_mask.shape
    grid = _colorize_barriers(barrier_mask, rng)
    passable = ~barrier_mask
    components = _components(passable)

    internal_regions: list[np.ndarray] = []
    external_regions: list[np.ndarray] = []
    for coords in components:
        if _touches_edge(coords, (h, w)):
            external_regions.append(coords)
        else:
            internal_regions.append(coords)

    if len(internal_regions) < 2 or len(external_regions) != 1:
        return None

    region_mask = np.zeros((h, w), dtype=int)
    used: set[tuple[int, int]] = set()
    for coords in internal_regions:
        region_mask.fill(0)
        region_mask[coords[:, 0], coords[:, 1]] = 1
        candidates = _full_patch_candidates(coords, region_mask.astype(bool))
        if not candidates:
            return None
        r, c = rng.choice(candidates)
        grid[r, c] = RED
        used.add((r, c))

    external = external_regions[0]
    region_mask.fill(0)
    region_mask[external[:, 0], external[:, 1]] = 1
    candidates = [cell for cell in _full_patch_candidates(external, region_mask.astype(bool)) if cell not in used]
    if not candidates:
        return None
    r, c = rng.choice(candidates)
    grid[r, c] = RED

    return grid


def _valid_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    colors = set(map(int, np.unique(grid)))
    if not colors.issubset({BLACK, BLUE, RED, GREEN}):
        return False
    if BLUE not in colors or GREEN not in colors or RED not in colors:
        return False

    barrier = np.isin(grid, BARRIER_COLORS)
    components = _components(~barrier)
    internal_count = 0
    external_red_regions = 0

    for coords in components:
        reds = int(np.sum(grid[coords[:, 0], coords[:, 1]] == RED))
        if _touches_edge(coords, grid.shape):
            if reds > 1:
                return False
            external_red_regions += int(reds == 1)
        else:
            internal_count += 1
            if reds != 1:
                return False

    if internal_count < 2 or external_red_regions != 1:
        return False

    out = main(grid)
    return bool(np.any(out == YELLOW) and np.any(out == ORANGE) and not np.array_equal(out, grid))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GEN_HEIGHT_RANGE)
    w = rng.randint(*GEN_WIDTH_RANGE)

    layout = _build_barrier_mask(h, w, rng)
    if layout is None:
        return None
    barrier_mask, _boxes = layout

    grid = _place_markers(barrier_mask, rng)
    if grid is None:
        return None
    if not _valid_generated_input(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle-2 input")
