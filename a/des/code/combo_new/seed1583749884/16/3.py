from __future__ import annotations

from collections import deque

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

GEN_RETRIES = 400
OBJECT_TRIES = 200
MIN_OBJECTS = 1
MAX_OBJECTS = 4


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _marker_direction(mask: np.ndarray, bbox: tuple[int, int, int, int], r: int, c: int) -> tuple[int, int]:
    h, w = mask.shape
    local_dirs: list[tuple[int, int]] = []
    if r == 0 or not mask[r - 1, c]:
        local_dirs.append((-1, 0))
    if r == h - 1 or not mask[r + 1, c]:
        local_dirs.append((1, 0))
    if c == 0 or not mask[r, c - 1]:
        local_dirs.append((0, -1))
    if c == w - 1 or not mask[r, c + 1]:
        local_dirs.append((0, 1))

    r0, c0, r1, c1 = bbox
    bbox_dirs: list[tuple[int, int]] = []
    if r == r0:
        bbox_dirs.append((-1, 0))
    if r == r1:
        bbox_dirs.append((1, 0))
    if c == c0:
        bbox_dirs.append((0, -1))
    if c == c1:
        bbox_dirs.append((0, 1))

    if len(local_dirs) == 1:
        return local_dirs[0]
    overlap = [direction for direction in bbox_dirs if direction in local_dirs]
    if len(overlap) == 1:
        return overlap[0]
    if len(bbox_dirs) == 1:
        return bbox_dirs[0]
    raise AssertionError(f"ambiguous marker side at {(r, c)}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each yellow boundary marker outward by a ray whose length equals the object's marker count.

    Input structure:
    - Background is black.
    - Foreground consists of one or more 4-connected solid grey objects.
    - Each object contains 1-4 yellow marker pixels on distinct top/bottom/left/right boundary sides, replacing grey boundary cells.

    Rule:
    - Treat each non-black 4-connected component as one object and count its yellow marker pixels.
    - For every yellow marker, determine which object side it belongs to and use the outward normal of that side as the ray direction.
    - Draw a yellow ray of that common object-specific length into background cells only, stopping if the ray would leave the grid or hit pre-existing foreground.
    - Keep the original grey object cells unchanged and retain the yellow marker pixels.

    Key insight:
    - Each marker's side encodes a direction, while the number of markers on its own object simultaneously encodes the ray length for all of that object's rays.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for coords in _components(g != BG):
        bbox = _bbox(coords)
        mask = np.zeros_like(g, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True

        markers = [(int(r), int(c)) for r, c in coords if int(g[r, c]) == YELLOW]
        assert 1 <= len(markers) <= 4, "each object must have 1-4 yellow markers"
        ray_len = len(markers)

        for r, c in markers:
            dr, dc = _marker_direction(mask, bbox, r, c)
            for step in range(1, ray_len + 1):
                nr = r + dr * step
                nc = c + dc * step
                if not (0 <= nr < g.shape[0] and 0 <= nc < g.shape[1]):
                    break
                if int(g[nr, nc]) != BG:
                    break
                out[nr, nc] = YELLOW
    return out


def _sample_marker_sides(rng: np.random.Generator) -> list[int]:
    ray_len = int(rng.integers(1, 5))
    return sorted(int(side) for side in rng.choice(4, size=ray_len, replace=False))


def _marker_cell(rng: np.random.Generator, side: int, r0: int, c0: int, r1: int, c1: int) -> tuple[int, int]:
    if side == 0:
        return r0, int(rng.integers(c0 + 1, c1))
    if side == 1:
        return r1, int(rng.integers(c0 + 1, c1))
    if side == 2:
        return int(rng.integers(r0 + 1, r1)), c0
    return int(rng.integers(r0 + 1, r1)), c1


def _place_object(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    sides = _sample_marker_sides(rng)
    ray_len = len(sides)

    top_margin = ray_len if 0 in sides else 1
    bottom_margin = ray_len if 1 in sides else 1
    left_margin = ray_len if 2 in sides else 1
    right_margin = ray_len if 3 in sides else 1

    max_obj_h = min(7, h - top_margin - bottom_margin)
    max_obj_w = min(7, w - left_margin - right_margin)
    if max_obj_h < 3 or max_obj_w < 3:
        return False

    for _ in range(OBJECT_TRIES):
        obj_h = int(rng.integers(3, max_obj_h + 1))
        obj_w = int(rng.integers(3, max_obj_w + 1))

        r0_low = top_margin
        r0_high = h - bottom_margin - obj_h
        c0_low = left_margin
        c0_high = w - right_margin - obj_w
        if r0_low > r0_high or c0_low > c0_high:
            continue

        r0 = int(rng.integers(r0_low, r0_high + 1))
        c0 = int(rng.integers(c0_low, c0_high + 1))
        r1 = r0 + obj_h - 1
        c1 = c0 + obj_w - 1

        pad = ray_len + 1
        rr0 = max(0, r0 - pad)
        cc0 = max(0, c0 - pad)
        rr1 = min(h - 1, r1 + pad)
        cc1 = min(w - 1, c1 + pad)
        if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
            continue

        grid[r0 : r1 + 1, c0 : c1 + 1] = GREY
        markers = {_marker_cell(rng, side, r0, c0, r1, c1) for side in sides}
        if len(markers) != ray_len:
            grid[r0 : r1 + 1, c0 : c1 + 1] = BG
            continue
        for mr, mc in markers:
            grid[mr, mc] = YELLOW
        reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        return True
    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_objects = int(rng.integers(MIN_OBJECTS, MAX_OBJECTS + 1))
    min_side = 8 + 4 * n_objects
    h = int(rng.integers(min_side, 31))
    w = int(rng.integers(min_side, 31))

    grid = np.full((h, w), BG, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    for _ in range(n_objects):
        if not _place_object(grid, reserved, rng):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    if len(_components(grid != BG)) != n_objects:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
