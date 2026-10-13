import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _touches_edge(coords: np.ndarray, h: int, w: int) -> bool:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return bool(
        np.any(rows == 0)
        or np.any(rows == h - 1)
        or np.any(cols == 0)
        or np.any(cols == w - 1)
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Periodically tile each enclosed region's top-left seed pattern across the whole region interior.

    Input structure:
    - The background is black.
    - One or more blue 1-pixel-wide closed borders partition the grid into enclosed non-blue interior regions.
    - In each enclosed region, all non-black non-blue pixels appear only in a small seed pattern near the region's top-left corner.

    Rule:
    - Find every 4-connected non-blue component that does not touch the grid edge; this is one region interior.
    - Inside each region, take the minimum bounding box of its non-black non-blue pixels as the seed tile.
    - Repeat that tile periodically in both row and column directions from the seed tile's top-left corner, filling only black cells of the same enclosed region.

    Key insight:
    - The colored pixels inside each border are not separate objects to move; they are a sample tile whose bounding-box size defines the repetition period over the entire enclosed shape.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    enclosed_regions = [
        comp for comp in _components(g != BLUE) if not _touches_edge(comp, *g.shape)
    ]
    assert enclosed_regions, "expected at least one blue-enclosed region"

    for region in enclosed_regions:
        seed_coords = np.array(
            [(int(r), int(c)) for r, c in region if int(g[int(r), int(c)]) not in (BLACK, BLUE)],
            dtype=int,
        )
        assert seed_coords.size > 0, "each region must contain a seed pattern"

        r0, c0, r1, c1 = _bbox(seed_coords)
        tile = g[r0 : r1 + 1, c0 : c1 + 1]
        th, tw = tile.shape

        for r, c in region:
            r = int(r)
            c = int(c)
            if int(out[r, c]) != BLACK:
                continue
            color = int(tile[(r - r0) % th, (c - c0) % tw])
            if color != BLACK:
                out[r, c] = color

    return out


def _shape_rect(rng: random.Random, h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _shape_corner_cut(rng: random.Random, h: int, w: int) -> np.ndarray:
    shape = np.ones((h, w), dtype=bool)
    cut_h = rng.randint(1, max(1, h - 2))
    cut_w = rng.randint(1, max(1, w - 2))
    shape[h - cut_h :, w - cut_w :] = False
    return shape


def _shape_bottom_cut(rng: random.Random, h: int, w: int) -> np.ndarray:
    shape = np.ones((h, w), dtype=bool)
    cut_h = rng.randint(1, max(1, h // 2))
    start_c = rng.randint(1, w - 2)
    shape[h - cut_h :, start_c:] = False
    return shape


def _shape_right_cut(rng: random.Random, h: int, w: int) -> np.ndarray:
    shape = np.ones((h, w), dtype=bool)
    cut_w = rng.randint(1, max(1, w // 2))
    start_r = rng.randint(1, h - 2)
    shape[start_r:, w - cut_w :] = False
    return shape


def _shape_double_cut(rng: random.Random, h: int, w: int) -> np.ndarray:
    shape = np.ones((h, w), dtype=bool)
    cut_h = rng.randint(1, max(1, h // 2))
    start_c = rng.randint(2, w - 2)
    shape[h - cut_h :, start_c:] = False
    cut_w = rng.randint(1, max(1, w // 2))
    start_r = rng.randint(2, h - 2)
    shape[start_r:, w - cut_w :] = False
    return shape


def _make_shape(rng: random.Random, h: int, w: int) -> np.ndarray:
    builders = [
        _shape_rect,
        _shape_corner_cut,
        _shape_bottom_cut,
        _shape_right_cut,
        _shape_double_cut,
    ]
    for _ in range(20):
        shape = rng.choice(builders)(rng, h, w)
        if not shape[0, 0]:
            continue
        if shape.sum() < max(12, (h * w) // 2):
            continue
        if len(_components(shape)) != 1:
            continue
        return shape
    return np.ones((h, w), dtype=bool)


def _pattern_bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.array(list(coords), dtype=int)
    return _bbox(arr)


def _make_pattern(rng: random.Random, h: int, w: int, palette: list[int]) -> np.ndarray:
    for _ in range(60):
        pattern = np.zeros((h, w), dtype=int)
        colors = rng.sample(palette, k=rng.randint(1, min(3, len(palette))))
        for r in range(h):
            for c in range(w):
                if rng.random() < rng.uniform(0.35, 0.8):
                    pattern[r, c] = rng.choice(colors)

        if h == 1 and w == 1:
            anchors = [(0, 0)]
        elif h == 1:
            anchors = [(0, 0), (0, w - 1)]
        elif w == 1:
            anchors = [(0, 0), (h - 1, 0)]
        else:
            anchors = [(0, 0), (h - 1, w - 1)]
            if rng.random() < 0.5:
                anchors.append((0, w - 1))
            if rng.random() < 0.5:
                anchors.append((h - 1, 0))

        for r, c in anchors:
            pattern[r, c] = rng.choice(colors)

        nz = np.argwhere(pattern != BLACK)
        if nz.size == 0:
            continue
        if _pattern_bbox((tuple(map(int, pt)) for pt in nz)) != (0, 0, h - 1, w - 1):
            continue
        if h * w >= 4 and np.count_nonzero(pattern) == h * w and rng.random() < 0.6:
            rr = rng.randrange(h)
            cc = rng.randrange(w)
            if (rr, cc) not in anchors:
                pattern[rr, cc] = BLACK
        return pattern
    pattern = np.zeros((h, w), dtype=int)
    color = rng.choice(palette)
    pattern[0, 0] = color
    pattern[h - 1, w - 1] = color
    return pattern


def _seed_candidates(shape: np.ndarray) -> list[tuple[int, int]]:
    h, w = shape.shape
    limit_h = min(5, h - 1)
    limit_w = min(5, w - 1)
    candidates: list[tuple[int, int]] = []
    total_area = int(shape.sum())
    for sh in range(1, limit_h + 1):
        for sw in range(1, limit_w + 1):
            if 1 + sh > h or 1 + sw > w:
                continue
            if not bool(shape[1 : 1 + sh, 1 : 1 + sw].all()):
                continue
            if total_area <= sh * sw:
                continue
            candidates.append((sh, sw))
    return candidates


def _build_region(rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    solid_h = rng.randint(7, 12)
    solid_w = rng.randint(7, 14)
    solid = _make_shape(rng, solid_h, solid_w)

    border = np.zeros((solid_h, solid_w), dtype=bool)
    for r in range(solid_h):
        for c in range(solid_w):
            if not solid[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < solid_h and 0 <= nc < solid_w and solid[nr, nc]):
                    border[r, c] = True
                    break

    interior = solid & (~border)
    if not interior[1, 1]:
        return None
    if len(_components(border)) != 1:
        return None
    if len(_components(interior)) != 1:
        return None

    candidates = _seed_candidates(interior)
    if not candidates:
        return None

    seed_h, seed_w = rng.choice(candidates)
    if not bool(interior[1 : 1 + seed_h, 1 : 1 + seed_w].all()):
        return None

    region = np.zeros((solid_h, solid_w), dtype=int)
    region[border] = BLUE

    palette = [color for color in range(2, 10)]
    pattern = _make_pattern(rng, seed_h, seed_w, palette)
    region[1 : 1 + seed_h, 1 : 1 + seed_w] = pattern

    footprint = solid
    if not np.any(region == BLUE):
        return None
    if np.count_nonzero(region[interior] > BLACK) == 0:
        return None
    return region, footprint


def _place_region(
    grid: np.ndarray,
    blocked: np.ndarray,
    region: np.ndarray,
    top: int,
    left: int,
) -> None:
    h, w = region.shape
    target = grid[top : top + h, left : left + w]
    target[region == BLUE] = BLUE
    target[region > BLUE] = region[region > BLUE]

    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(grid.shape[0], top + h + 1)
    c1 = min(grid.shape[1], left + w + 1)
    blocked[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    grid = np.zeros((h, w), dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    max_regions = 4 if h * w >= 450 else 3
    region_count = rng.randint(1, max_regions)
    placed = 0

    for _ in range(region_count):
        success = False
        for _ in range(100):
            built = _build_region(rng)
            if built is None:
                continue
            region, footprint = built
            rh, rw = region.shape
            if rh > h - 2 or rw > w - 2:
                continue
            top = rng.randint(1, h - rh - 1)
            left = rng.randint(1, w - rw - 1)
            if blocked[top - 1 : top + rh + 1, left - 1 : left + rw + 1].any():
                continue
            _place_region(grid, blocked, region, top, left)
            placed += 1
            success = True
            break
        if not success and placed == 0:
            return None

    if placed == 0:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    enclosed = [comp for comp in _components(grid != BLUE) if not _touches_edge(comp, h, w)]
    if len(enclosed) != placed:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
