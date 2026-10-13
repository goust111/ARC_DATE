import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
AZURE = 8
SEED_COLORS = (1, 2, 3, 4, 6, 7, 9)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def _components(mask: np.ndarray, conn8: bool = False) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = _neighbors8 if conn8 else _neighbors4
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
                for nr, nc in dirs(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if (
                mask[r - 1, c]
                and mask[r + 1, c]
                and mask[r, c - 1]
                and mask[r, c + 1]
            ):
                out[r, c] = True
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _grow_region(
    grid: np.ndarray,
    region_cells: list[tuple[int, int]],
    seed: tuple[int, int],
    layers: int,
) -> np.ndarray:
    out = grid.copy()
    region = set(region_cells)
    color = int(grid[seed])
    colored = {seed}
    frontier = {seed}

    for _ in range(layers):
        nxt: set[tuple[int, int]] = set()
        for r, c in frontier:
            for nr, nc in _neighbors8(r, c):
                if (nr, nc) not in region or (nr, nc) in colored:
                    continue
                if int(grid[nr, nc]) not in (BLACK, AZURE):
                    continue
                nxt.add((nr, nc))
        for r, c in nxt:
            out[r, c] = color
        if not nxt:
            break
        colored |= nxt
        frontier = nxt
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each colored seed inside its grey fence for as many 8-neighborhood
    layers as there are azure markers in that fenced interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for cells in _components(g != GREY, conn8=False):
        if _touches_border(cells, h, w):
            continue

        seeds = [(r, c) for r, c in cells if int(g[r, c]) not in (BLACK, AZURE)]
        assert len(seeds) == 1, "each fenced interior must contain exactly one seed"
        seed = seeds[0]
        layers = sum(int(g[r, c]) == AZURE for r, c in cells)
        out = _grow_region(out, cells, seed, layers)
    return out


# --------------- generator ---------------

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
GEN_RETRIES = 400

LOCAL_CANVAS = 20
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 120
CONTENT_RETRIES = 80
PLACEMENT_GAP = 2
BORDER_MARGIN = 1
MIN_INSIDE_CELLS = 9

BASE_H_RANGE = (6, 9)
BASE_W_RANGE = (6, 9)
BULGE_H_RANGE = (3, 5)
BULGE_W_RANGE = (3, 5)


def _draw_rect(mask: np.ndarray, r0: int, c0: int, height: int, width: int) -> None:
    mask[r0 : r0 + height, c0 : c0 + width] = True


def _valid_ring(wall: np.ndarray, inside: np.ndarray) -> bool:
    if int(inside.sum()) < MIN_INSIDE_CELLS:
        return False
    if len(_components(inside, conn8=False)) != 1:
        return False
    wall_comps = _components(wall, conn8=False)
    if len(wall_comps) != 1:
        return False
    for r, c in wall_comps[0]:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < wall.shape[0] and 0 <= nc < wall.shape[1] and wall[nr, nc]:
                deg += 1
        if deg != 2:
            return False
    return True


def _sample_bulge(mask: np.ndarray, base_r: int, base_c: int, base_h: int, base_w: int, side: str) -> bool:
    if side in ("top", "bottom"):
        max_w = min(BULGE_W_RANGE[1], base_w - 2)
        if max_w < BULGE_W_RANGE[0]:
            return False
        width = random.randint(BULGE_W_RANGE[0], max_w)
        start_c = base_c + random.randint(1, base_w - width - 1)
        height = random.randint(BULGE_H_RANGE[0], BULGE_H_RANGE[1])
        r0 = base_r - height if side == "top" else base_r + base_h
        c0 = start_c
    else:
        max_h = min(BULGE_H_RANGE[1], base_h - 2)
        if max_h < BULGE_H_RANGE[0]:
            return False
        height = random.randint(BULGE_H_RANGE[0], max_h)
        start_r = base_r + random.randint(1, base_h - height - 1)
        width = random.randint(BULGE_W_RANGE[0], BULGE_W_RANGE[1])
        r0 = start_r
        c0 = base_c - width if side == "left" else base_c + base_w

    if r0 < 0 or c0 < 0 or r0 + height > mask.shape[0] or c0 + width > mask.shape[1]:
        return False
    _draw_rect(mask, r0, c0, height, width)
    return True


def _random_shape() -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_RETRIES):
        filled = np.zeros((LOCAL_CANVAS, LOCAL_CANVAS), dtype=bool)
        base_h = random.randint(*BASE_H_RANGE)
        base_w = random.randint(*BASE_W_RANGE)
        base_r = (LOCAL_CANVAS - base_h) // 2
        base_c = (LOCAL_CANVAS - base_w) // 2
        _draw_rect(filled, base_r, base_c, base_h, base_w)

        mode = random.choices(("rect", "one", "two"), weights=(2, 4, 3))[0]
        bulge_count = {"rect": 0, "one": 1, "two": 2}[mode]
        ok = True
        if bulge_count:
            for side in random.sample(("top", "bottom", "left", "right"), bulge_count):
                ok &= _sample_bulge(filled, base_r, base_c, base_h, base_w, side)
                if not ok:
                    break
        if not ok:
            continue

        filled = _crop(filled)
        if random.random() < 0.5:
            filled = np.flipud(filled)
        if random.random() < 0.5:
            filled = np.fliplr(filled)
        if random.random() < 0.4:
            filled = filled.T

        inside = _erode4(filled)
        wall = filled & ~inside
        if _valid_ring(wall, inside):
            return wall, inside
    return None


def _reserve_ok(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> bool:
    r0 = max(0, top - PLACEMENT_GAP)
    c0 = max(0, left - PLACEMENT_GAP)
    r1 = min(reserved.shape[0], top + height + PLACEMENT_GAP)
    c1 = min(reserved.shape[1], left + width + PLACEMENT_GAP)
    return not reserved[r0:r1, c0:c1].any()


def _reserve_mark(reserved: np.ndarray, top: int, left: int, height: int, width: int) -> None:
    r0 = max(0, top - PLACEMENT_GAP)
    c0 = max(0, left - PLACEMENT_GAP)
    r1 = min(reserved.shape[0], top + height + PLACEMENT_GAP)
    c1 = min(reserved.shape[1], left + width + PLACEMENT_GAP)
    reserved[r0:r1, c0:c1] = True


def _place_contents(
    grid: np.ndarray,
    inside_cells: list[tuple[int, int]],
    seed_color: int,
    marker_count: int,
) -> bool:
    inside_set = set(inside_cells)
    for _ in range(CONTENT_RETRIES):
        seed = random.choice(inside_cells)
        forbidden = {seed}
        sr, sc = seed
        for nr, nc in _neighbors8(sr, sc):
            if (nr, nc) in inside_set:
                forbidden.add((nr, nc))
        marker_candidates = [cell for cell in inside_cells if cell not in forbidden]
        if len(marker_candidates) < marker_count:
            continue

        grid[sr, sc] = seed_color
        for r, c in random.sample(marker_candidates, marker_count):
            grid[r, c] = AZURE
        return True
    return False


def _build_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    grid = np.full((height, width), BLACK, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    area = height * width
    max_fences = 2 if area < 420 else 3 if area < 650 else 4
    fence_options = list(range(1, max_fences + 1))
    fence_weights = [2, 4, 3, 1][: len(fence_options)]
    fence_count = random.choices(fence_options, weights=fence_weights)[0]

    seed_colors = random.sample(SEED_COLORS, fence_count)
    marker_counts = [random.randint(0, 4) for _ in range(fence_count)]
    if all(count == 0 for count in marker_counts):
        idx = random.randrange(fence_count)
        marker_counts[idx] = random.randint(1, 4)

    for seed_color, marker_count in zip(seed_colors, marker_counts):
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            shape = _random_shape()
            if shape is None:
                break
            wall, inside = shape
            sh, sw = wall.shape
            if sh + 2 * BORDER_MARGIN > height or sw + 2 * BORDER_MARGIN > width:
                continue

            top = random.randint(BORDER_MARGIN, height - sh - BORDER_MARGIN)
            left = random.randint(BORDER_MARGIN, width - sw - BORDER_MARGIN)
            if not _reserve_ok(reserved, top, left, sh, sw):
                continue

            inside_cells = [(top + r, left + c) for r, c in np.argwhere(inside)]
            if not _place_contents(grid, inside_cells, seed_color, marker_count):
                continue

            wall_cells = np.argwhere(wall)
            for r, c in wall_cells:
                grid[top + int(r), left + int(c)] = GREY
            _reserve_mark(reserved, top, left, sh, sw)
            placed = True
            break
        if not placed:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid fenced-growth puzzle")
