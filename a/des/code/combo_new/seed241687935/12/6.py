import random
from collections import deque

import numpy as np


BLACK = 0
ORANGE = 7
SEED_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_MIN = 12
GRID_MAX = 30
REGION_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
SHAPE_SIZE_MIN = 8
SHAPE_SIZE_MAX = 14
MAX_NOTCHES = 4
GEN_ATTEMPTS = 300
SHAPE_ATTEMPTS = 120
PLACEMENT_ATTEMPTS = 120
MIN_INTERIOR_CELLS = 10


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            keep = True
            for dr, dc in NEIGHBORS4:
                if not mask[r + dr, c + dc]:
                    keep = False
                    break
            if keep:
                out[r, c] = True
    return out


def _touches_border(comp: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)


def _carve_notch(mask: np.ndarray, side: str, start: int, span: int, depth: int) -> None:
    h, w = mask.shape
    if side == "top":
        mask[:depth, start : start + span] = False
    elif side == "bottom":
        mask[h - depth :, start : start + span] = False
    elif side == "left":
        mask[start : start + span, :depth] = False
    else:
        mask[start : start + span, w - depth :] = False


def _region_layers(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interior = _erode4(mask)
    boundary = mask & ~interior
    seed_zone = _erode4(interior)
    return boundary, interior, seed_zone


def _make_region_mask(rng: random.Random) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    for _ in range(SHAPE_ATTEMPTS):
        h = rng.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        w = rng.randint(SHAPE_SIZE_MIN, SHAPE_SIZE_MAX)
        mask = np.ones((h, w), dtype=bool)

        if rng.random() < 0.2:
            notch_count = 0
        else:
            notch_count = rng.randint(1, MAX_NOTCHES)

        for _ in range(notch_count):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                max_depth = min(max(1, h // 3), h - 4)
                if max_depth < 1 or w <= 3:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(1, w - 2)
                start = rng.randint(1, w - span - 1)
            else:
                max_depth = min(max(1, w // 3), w - 4)
                if max_depth < 1 or h <= 3:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(1, h - 2)
                start = rng.randint(1, h - span - 1)
            _carve_notch(mask, side, start, span, depth)

        if rng.random() < 0.5:
            mask = np.flip(mask, axis=0)
        if rng.random() < 0.5:
            mask = np.flip(mask, axis=1)
        if rng.random() < 0.35:
            mask = mask.T.copy()

        if len(_components(mask)) != 1:
            continue

        boundary, interior, seed_zone = _region_layers(mask)
        if np.count_nonzero(interior) < MIN_INTERIOR_CELLS:
            continue
        if np.count_nonzero(seed_zone) == 0:
            continue
        if len(_components(interior)) != 1:
            continue
        if len(_components(boundary)) != 1:
            continue
        return mask, boundary, interior, seed_zone

    raise RuntimeError("failed to build a valid region mask")


def _can_place(mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> bool:
    h, w = occupied.shape
    for r, c in np.argwhere(mask):
        gr = top + int(r)
        gc = left + int(c)
        for nr in range(max(0, gr - 1), min(h, gr + 2)):
            for nc in range(max(0, gc - 1), min(w, gc + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _mark_occupied(mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> None:
    h, w = occupied.shape
    for r, c in np.argwhere(mask):
        gr = top + int(r)
        gc = left + int(c)
        for nr in range(max(0, gr - 1), min(h, gr + 2)):
            for nc in range(max(0, gc - 1), min(w, gc + 2)):
                occupied[nr, nc] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.choice(REGION_COUNT_CHOICES)
    min_dim = max(GRID_MIN, 10 + 4 * region_count)
    h = rng.randint(min_dim, GRID_MAX)
    w = rng.randint(min_dim, GRID_MAX)

    grid = np.full((h, w), BLACK, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    seed_colors = rng.sample(SEED_COLORS, region_count)

    for color in seed_colors:
        mask, boundary, _interior, seed_zone = _make_region_mask(rng)
        mh, mw = mask.shape
        if mh + 2 > h or mw + 2 > w:
            return None

        max_top = h - mh - 1
        max_left = w - mw - 1
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(1, max_top)
            left = rng.randint(1, max_left)
            if not _can_place(mask, occupied, top, left):
                continue

            for r, c in np.argwhere(boundary):
                grid[top + int(r), left + int(c)] = ORANGE

            seed_candidates = np.argwhere(seed_zone)
            sr, sc = seed_candidates[rng.randrange(len(seed_candidates))]
            grid[top + int(sr), left + int(sc)] = int(color)

            _mark_occupied(mask, occupied, top, left)
            placed = True
            break

        if not placed:
            return None

    enclosed = 0
    for comp in _components(grid != ORANGE):
        if _touches_border(comp, grid.shape):
            continue
        colors = {int(grid[r, c]) for r, c in comp if int(grid[r, c]) != BLACK}
        if len(colors) != 1:
            return None
        enclosed += 1
    if enclosed != region_count:
        return None

    return grid


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill each orange-enclosed region with the color of its interior seed pixel.

    Input structure:
    - The background is black.
    - One or more irregular orange 4-connected closed boundaries appear in the grid.
    - Each enclosed area contains exactly one non-black seed pixel and otherwise black empty cells.

    Rule:
    - Treat orange pixels as walls and find every 4-connected component of non-orange cells.
    - Any such component that does not touch the grid border is an enclosed region interior together with its seed.
    - Replace all black pixels in that enclosed component with the unique seed color, while keeping the orange boundary and the outside black background unchanged.

    Key insight:
    - The apparent inner-outline-plus-core effect is just a full flood fill of every orange-bounded interior using its single seed color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    found = False

    for comp in _components(g != ORANGE):
        if _touches_border(comp, g.shape):
            continue
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BLACK}
        assert len(colors) == 1, "each enclosed region must contain exactly one seed color"
        fill_color = next(iter(colors))
        for r, c in comp:
            if int(g[r, c]) == BLACK:
                out[r, c] = fill_color
        found = True

    assert found, "expected at least one enclosed region"
    return out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
