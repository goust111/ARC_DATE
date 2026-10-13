from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
SPRITE = 1
CONTOUR = 2
MARKER = 7

GEN_MAX_ATTEMPTS = 400
GRID_MIN = 18
GRID_MAX = 24
INTERIOR_MIN = 8
INTERIOR_MAX = 12


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _components(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

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
                for dr, dc in offsets:
                    nr, nc = cr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _enclosed_region(red_mask: np.ndarray) -> np.ndarray:
    h, w = red_mask.shape
    outside = np.zeros_like(red_mask, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not red_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not red_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(red_mask.shape, nr, nc) and not red_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~red_mask) & (~outside)


def _outer_ring(grid: np.ndarray, red_mask: np.ndarray, interior: np.ndarray) -> np.ndarray:
    ring = np.zeros_like(red_mask, dtype=bool)
    for r, c in np.argwhere(red_mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors4(rr, cc):
            if (
                _in_bounds(grid.shape, nr, nc)
                and not interior[nr, nc]
                and grid[nr, nc] == BACKGROUND
            ):
                ring[nr, nc] = True
    return ring


def main(grid: np.ndarray) -> np.ndarray:
    """
    Tile the blue sprite inside the closed contour and repaint the contour's outside-adjacent ring.

    Input structure:
    - The grid uses black background with one red single-cell-thick closed contour.
    - A single blue 4-connected non-rectangular sprite sits strictly inside that contour.
    - One or two orange marker pixels sit outside the contour and do not touch it.

    Rule:
    - Find the region enclosed by the red contour.
    - Extract the blue sprite's tight bounding box and use that bbox as a periodic tile, anchored at the sprite's original top-left corner; draw only blue cells whose tiled positions fall inside the enclosed region.
    - Paint orange on black cells that are 4-adjacent to the red contour and lie outside the enclosed region, then erase the red contour and the original orange markers.

    Key insight:
    - The red contour is not an object to preserve; it is a temporary fence that defines where the sprite pattern can repeat, while the orange output marks only the contour's exterior trace.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    red_mask = g == CONTOUR
    blue_mask = g == SPRITE
    assert red_mask.any()
    assert blue_mask.any()

    interior = _enclosed_region(red_mask)
    blue_components = _components(blue_mask, connectivity=4)
    assert len(blue_components) == 1
    sprite_coords = blue_components[0]
    r0, c0, r1, c1 = _bbox(sprite_coords)
    patch = g[r0 : r1 + 1, c0 : c1 + 1] == SPRITE
    tile_h, tile_w = patch.shape

    out = np.full_like(g, BACKGROUND)
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        if patch[(rr - r0) % tile_h, (cc - c0) % tile_w]:
            out[rr, cc] = SPRITE

    out[_outer_ring(g, red_mask, interior)] = MARKER
    return out


def _random_row_interval_shape(rng: random.Random, h: int, w: int) -> np.ndarray | None:
    widths = []
    width = rng.randint(max(5, w - 3), w)
    for _ in range(h):
        widths.append(width)
        width += rng.choice([-1, 0, 0, 1])
        width = max(5, min(w, width))

    lefts = [rng.randint(0, max(0, w - widths[0]))]
    rights = [lefts[0] + widths[0] - 1]
    for i in range(1, h):
        prev_l, prev_r = lefts[-1], rights[-1]
        cand = []
        for shift_l in (-1, 0, 1):
            for shift_r in (-1, 0, 1):
                l = prev_l + shift_l
                r = prev_r + shift_r
                if r - l + 1 != widths[i]:
                    continue
                if l < 0 or r >= w:
                    continue
                if l > prev_r or r < prev_l:
                    continue
                cand.append((l, r))
        if not cand:
            return None
        l, r = rng.choice(cand)
        lefts.append(l)
        rights.append(r)

    shape = np.zeros((h, w), dtype=bool)
    for r in range(h):
        shape[r, lefts[r] : rights[r] + 1] = True

    if np.all(shape):
        return None
    if len(_components(shape, connectivity=4)) != 1:
        return None
    return shape


def _build_contour(interior: np.ndarray) -> np.ndarray:
    h, w = interior.shape
    contour = np.zeros_like(interior, dtype=bool)
    for r, c in np.argwhere(interior):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(interior.shape, nr, nc) and not interior[nr, nc]:
                contour[nr, nc] = True
    return contour


def _erode8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        keep = True
        for nr, nc in _neighbors8(rr, cc):
            if not _in_bounds(mask.shape, nr, nc) or not mask[nr, nc]:
                keep = False
                break
        if keep:
            out[rr, cc] = True
    return out


def _random_sprite(rng: random.Random) -> np.ndarray | None:
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        area = rng.randint(3, min(6, h * w - 1))
        mask = np.zeros((h, w), dtype=bool)
        start = (rng.randrange(h), rng.randrange(w))
        mask[start] = True
        cells = [start]
        while int(mask.sum()) < area:
            frontier = []
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.append((nr, nc))
            frontier = list(dict.fromkeys(frontier))
            if not frontier:
                break
            chosen = rng.choice(frontier)
            mask[chosen] = True
            cells.append(chosen)

        coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
        if len(coords) != area:
            continue
        r0, c0, r1, c1 = _bbox(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.all():
            continue
        if len(_components(cropped, connectivity=4)) != 1:
            continue
        return cropped
    return None


def _place_sprite(
    rng: random.Random,
    grid: np.ndarray,
    interior: np.ndarray,
    sprite: np.ndarray,
) -> bool:
    safe = _erode8(interior)
    sh, sw = sprite.shape
    candidates = []
    h, w = grid.shape
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            bbox_ok = True
            for dr in range(sh):
                for dc in range(sw):
                    if not interior[r0 + dr, c0 + dc]:
                        bbox_ok = False
                        break
                if not bbox_ok:
                    break
            if not bbox_ok:
                continue
            sprite_ok = True
            for dr in range(sh):
                for dc in range(sw):
                    if sprite[dr, dc] and not safe[r0 + dr, c0 + dc]:
                        sprite_ok = False
                        break
                if not sprite_ok:
                    break
            if sprite_ok:
                candidates.append((r0, c0))
    if not candidates:
        return False
    r0, c0 = rng.choice(candidates)
    for dr in range(sh):
        for dc in range(sw):
            if sprite[dr, dc]:
                grid[r0 + dr, c0 + dc] = SPRITE
    return True


def _far_from_contour(mask: np.ndarray, contour: np.ndarray) -> np.ndarray:
    out = mask.copy()
    for r, c in np.argwhere(mask):
        rr = int(r)
        cc = int(c)
        for nr, nc in _neighbors8(rr, cc):
            if _in_bounds(mask.shape, nr, nc) and contour[nr, nc]:
                out[rr, cc] = False
                break
    return out


def _place_markers(
    rng: random.Random,
    grid: np.ndarray,
    contour: np.ndarray,
    interior: np.ndarray,
    count: int,
) -> bool:
    outside = (grid == BACKGROUND) & (~interior)
    candidates = np.argwhere(_far_from_contour(outside, contour))
    if len(candidates) < count:
        return False
    picks = rng.sample([tuple(map(int, coord)) for coord in candidates], count)
    for r, c in picks:
        grid[r, c] = MARKER
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    ih = rng.randint(INTERIOR_MIN, min(INTERIOR_MAX, h - 6))
    iw = rng.randint(INTERIOR_MIN, min(INTERIOR_MAX, w - 6))

    local_interior = _random_row_interval_shape(rng, ih, iw)
    if local_interior is None:
        return None

    top = rng.randint(2, h - ih - 3)
    left = rng.randint(2, w - iw - 3)
    grid = np.zeros((h, w), dtype=int)
    interior = np.zeros((h, w), dtype=bool)
    interior[top : top + ih, left : left + iw] = local_interior
    contour = _build_contour(interior)
    if not contour.any():
        return None
    if len(_components(contour, connectivity=8)) != 1:
        return None

    if np.any(interior & contour):
        return None

    grid[contour] = CONTOUR
    recovered = _enclosed_region(contour)
    if not np.array_equal(recovered, interior):
        return None
    if len(_components(recovered, connectivity=4)) != 1:
        return None

    sprite = _random_sprite(rng)
    if sprite is None or not _place_sprite(rng, grid, interior, sprite):
        return None

    marker_count = rng.choice([1, 2])
    if not _place_markers(rng, grid, contour, interior, marker_count):
        return None

    blue_mask = grid == SPRITE
    if len(_components(blue_mask, connectivity=4)) != 1:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    if not np.any(out == SPRITE):
        return None
    if not np.any(out == MARKER):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(random)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
