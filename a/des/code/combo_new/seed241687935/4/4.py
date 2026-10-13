from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 400
SHAPE_RETRIES = 120
PLACEMENT_RETRIES = 160


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _boundary_of_shape(shape: np.ndarray) -> np.ndarray:
    h, w = shape.shape
    boundary = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not shape[r, c]:
                continue
            if r == 0 or c == 0 or r == h - 1 or c == w - 1:
                boundary[r, c] = True
                continue
            for nr, nc in _neighbors4(r, c):
                if not shape[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _background_color(grid: np.ndarray) -> int:
    return BG if np.any(grid == BG) else int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _wall_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.ravel()))
    others = [color for color in counts if color != bg]
    assert others, "expected at least one non-background color"
    return int(max(others, key=lambda color: counts[color]))


def _direction_for_seed(seed_r: int, seed_c: int, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    row_bias = 2 * seed_r - (r0 + r1)
    col_bias = 2 * seed_c - (c0 + c1)
    if abs(row_bias) > abs(col_bias):
        return "down" if row_bias < 0 else "up"
    return "right" if col_bias < 0 else "left"


def _paint_region(
    out: np.ndarray,
    region: list[tuple[int, int]],
    seed_r: int,
    seed_c: int,
    seed_color: int,
    direction: str,
    bg: int,
) -> None:
    for r, c in region:
        out[r, c] = bg

    for r, c in region:
        if direction == "down":
            delta = r - seed_r
        elif direction == "up":
            delta = seed_r - r
        elif direction == "right":
            delta = c - seed_c
        else:
            delta = seed_c - c
        if delta >= 0 and delta % 2 == 0:
            out[r, c] = seed_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed region with directional stripes keyed by its off-center seed pixel.

    Input structure:
    - Background is black.
    - One wall color draws several closed 4-connected outline regions.
    - Each enclosed interior contains exactly one uniquely colored seed pixel.

    Rule:
    - Find every non-wall connected component that is enclosed by an outline and locate its single seed.
    - Compare the seed position with the enclosing region's bounding-box center; the dominant offset chooses one of four stripe directions.
    - Starting from the seed row or seed column, repaint the enclosed interior in alternating seed-color and background stripes while keeping the outline unchanged.

    Key insight:
    - The seed's color gives the stripe color, and the seed's offset from the region center gives both the stripe orientation and which way the stripes advance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    wall = _wall_color(g, bg)

    out = g.copy()
    passable = g != wall

    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue

        seeds = [(r, c, int(g[r, c])) for r, c in comp if int(g[r, c]) != bg]
        if len(seeds) != 1:
            continue

        boundary_neighbors: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == wall:
                    boundary_neighbors.add((nr, nc))
        if not boundary_neighbors:
            continue

        rows = [r for r, _c in boundary_neighbors]
        cols = [c for _r, c in boundary_neighbors]
        bbox = (min(rows), min(cols), max(rows), max(cols))

        seed_r, seed_c, seed_color = seeds[0]
        direction = _direction_for_seed(seed_r, seed_c, bbox)
        _paint_region(out, comp, seed_r, seed_c, seed_color, direction, bg)

    return out


def _has_required_candidates(interior: np.ndarray, boundary: np.ndarray, direction: str) -> bool:
    cells = np.argwhere(interior)
    if len(cells) == 0:
        return False
    rows = boundary[:, :].any(axis=1).nonzero()[0]
    cols = boundary[:, :].any(axis=0).nonzero()[0]
    bbox = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
    min_r = int(cells[:, 0].min())
    max_r = int(cells[:, 0].max())
    min_c = int(cells[:, 1].min())
    max_c = int(cells[:, 1].max())
    for r, c in cells:
        d = _direction_for_seed(int(r), int(c), bbox)
        if d != direction:
            continue
        if direction == "down" and max_r - int(r) >= 2:
            return True
        if direction == "up" and int(r) - min_r >= 2:
            return True
        if direction == "right" and max_c - int(c) >= 2:
            return True
        if direction == "left" and int(c) - min_c >= 2:
            return True
    return False


def _pick_seed(interior: np.ndarray, boundary: np.ndarray, direction: str, rng: random.Random) -> tuple[int, int] | None:
    cells = np.argwhere(interior)
    rows = boundary[:, :].any(axis=1).nonzero()[0]
    cols = boundary[:, :].any(axis=0).nonzero()[0]
    bbox = (int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1]))
    min_r = int(cells[:, 0].min())
    max_r = int(cells[:, 0].max())
    min_c = int(cells[:, 1].min())
    max_c = int(cells[:, 1].max())

    candidates: list[tuple[int, int]] = []
    for r, c in cells:
        rr = int(r)
        cc = int(c)
        if _direction_for_seed(rr, cc, bbox) != direction:
            continue
        if direction == "down" and max_r - rr < 2:
            continue
        if direction == "up" and rr - min_r < 2:
            continue
        if direction == "right" and max_c - cc < 2:
            continue
        if direction == "left" and cc - min_c < 2:
            continue
        candidates.append((rr, cc))
    if not candidates:
        return None
    return rng.choice(candidates)


def _sample_shape(direction: str, rng: random.Random) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(6, 11)
        w = rng.randint(6, 11)
        shape = np.ones((h, w), dtype=bool)

        notch_count = rng.randint(0, 3)
        for _ in range(notch_count):
            side = rng.choice(["top", "bottom", "left", "right"])
            if side in ("top", "bottom"):
                if h <= 5 or w <= 4:
                    continue
                depth = rng.randint(1, max(1, h - 4))
                width = rng.randint(1, max(1, w - 2))
                c0 = rng.randint(1, w - width - 1)
                if side == "top":
                    shape[:depth, c0 : c0 + width] = False
                else:
                    shape[h - depth :, c0 : c0 + width] = False
            else:
                if w <= 5 or h <= 4:
                    continue
                depth = rng.randint(1, max(1, w - 4))
                height = rng.randint(1, max(1, h - 2))
                r0 = rng.randint(1, h - height - 1)
                if side == "left":
                    shape[r0 : r0 + height, :depth] = False
                else:
                    shape[r0 : r0 + height, w - depth :] = False

        boundary = _boundary_of_shape(shape)
        interior = shape & (~boundary)
        if interior.sum() < 6:
            continue
        if len(_components(shape)) != 1:
            continue
        if len(_components(boundary)) != 1:
            continue
        interior_comps = _components(interior)
        if len(interior_comps) != 1:
            continue
        if not _has_required_candidates(interior, boundary, direction):
            continue
        return boundary, interior
    return None


def _stamp_region(
    grid: np.ndarray,
    used: np.ndarray,
    boundary: np.ndarray,
    seed: tuple[int, int],
    border_color: int,
    seed_color: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = boundary.shape
    max_r = h - ph - 1
    max_c = w - pw - 1
    if max_r < 1 or max_c < 1:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(1, max_r)
        c0 = rng.randint(1, max_c)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(h, r0 + ph + 1)
        cc1 = min(w, c0 + pw + 1)
        if used[rr0:rr1, cc0:cc1].any():
            continue

        grid[r0 : r0 + ph, c0 : c0 + pw][boundary] = border_color
        sr, sc = seed
        grid[r0 + sr, c0 + sc] = seed_color
        used[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _count_enclosed_regions(grid: np.ndarray, wall: int) -> int:
    h, w = grid.shape
    passable = grid != wall
    count = 0
    for comp in _components(passable):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp):
            continue
        if sum(int(grid[r, c] != BG) for r, c in comp) == 1:
            count += 1
    return count


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    region_count = rng.randint(1, 4)
    min_side = 14 if region_count <= 2 else 18
    h = rng.randint(min_side, 30)
    w = rng.randint(min_side, 30)
    grid = np.full((h, w), BG, dtype=int)
    used = np.zeros((h, w), dtype=bool)

    border_color = rng.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    seed_palette = [color for color in range(1, 10) if color != border_color]
    if len(seed_palette) < region_count:
        return None
    seed_colors = rng.sample(seed_palette, region_count)

    directions = [rng.choice(["up", "down", "left", "right"]) for _ in range(region_count)]

    for direction, seed_color in zip(directions, seed_colors):
        sample = _sample_shape(direction, rng)
        if sample is None:
            return None
        boundary, interior = sample
        seed = _pick_seed(interior, boundary, direction, rng)
        if seed is None:
            return None
        if not _stamp_region(grid, used, boundary, seed, border_color, seed_color, rng):
            return None

    if _count_enclosed_regions(grid, border_color) != region_count:
        return None

    out = main(grid)
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid directional-stripe input")
