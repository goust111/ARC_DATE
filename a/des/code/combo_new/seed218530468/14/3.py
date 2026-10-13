from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


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
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _enclosed_regions(grid: np.ndarray) -> list[list[tuple[int, int]]]:
    return [
        comp
        for comp in _components(grid != GREY)
        if not _touches_border(comp, grid.shape)
    ]


def _distance_from_grey(grid: np.ndarray, region: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    h, w = grid.shape
    region_set = set(region)
    dist: dict[tuple[int, int], int] = {}
    q: deque[tuple[int, int]] = deque()
    for r, c in region:
        if any(
            0 <= r + dr < h and 0 <= c + dc < w and grid[r + dr, c + dc] == GREY
            for dr, dc in DIR4
        ):
            dist[(r, c)] = 1
            q.append((r, c))
    assert q, "each enclosed region must touch a grey boundary"
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in region_set and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow same-thickness colored border bands inside each seeded grey-bounded region.

    Input structure:
    - The background is black.
    - Grey 1-pixel lines form one or more closed boundaries, leaving black enclosed regions.
    - Some enclosed regions contain 1 to 4 scattered seed pixels of a single non-black,
      non-grey color; other regions contain no seeds.

    Rule:
    - Treat every non-grey 4-connected component that does not touch the grid border as one
      enclosed region.
    - For each seeded region, count its seed pixels as N and compute each cell's 4-neighbor
      distance from the surrounding grey boundary.
    - Recolor every black cell in that region whose boundary distance is between 1 and N,
      using the region's seed color. Keep seeds unchanged and leave unseeded regions unchanged.

    Key insight:
    - The color does not spread outward from the seed locations; the seeds only choose the
      color and the band thickness, while the actual growth is a uniform inward offset from
      the grey boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _enclosed_regions(g):
        seed_cells = [(r, c) for r, c in region if g[r, c] not in (BG, GREY)]
        if not seed_cells:
            continue
        seed_colors = {int(g[r, c]) for r, c in seed_cells}
        assert len(seed_colors) == 1, "each region must use a single seed color"
        color = seed_colors.pop()
        layers = len(seed_cells)
        dist = _distance_from_grey(g, region)
        for r, c in region:
            if g[r, c] == BG and dist[(r, c)] <= layers:
                out[r, c] = color
    return out


def _rectangle_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 3 or max_w < 3:
        return None
    h = random.randint(3, min(9, max_h))
    w = random.randint(3, min(9, max_w))
    return np.ones((h, w), dtype=bool)


def _l_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    h = random.randint(5, min(10, max_h))
    w = random.randint(5, min(10, max_w))
    cut_h = random.randint(2, h - 2)
    cut_w = random.randint(2, w - 2)
    mask = np.ones((h, w), dtype=bool)
    corner = random.randrange(4)
    if corner == 0:
        mask[:cut_h, :cut_w] = False
    elif corner == 1:
        mask[:cut_h, w - cut_w :] = False
    elif corner == 2:
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _t_mask(max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 5 or max_w < 5:
        return None
    h = random.randint(5, min(10, max_h))
    w = random.randint(5, min(10, max_w))
    bar = random.randint(2, min(3, h - 2))
    stem = random.randint(2, min(3, w - 2))
    start = random.randint(0, w - stem)
    mask = np.zeros((h, w), dtype=bool)
    mask[:bar, :] = True
    mask[:, start : start + stem] = True
    return np.rot90(mask, random.randrange(4)).copy()


def _shape_mask(max_h: int, max_w: int) -> np.ndarray | None:
    builders = [_rectangle_mask, _l_mask, _t_mask]
    random.shuffle(builders)
    for builder in builders:
        mask = builder(max_h, max_w)
        if mask is not None:
            return mask
    return None


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = src_r0 + dr
            dst_r1 = src_r1 + dr
            dst_c0 = src_c0 + dc
            dst_c1 = src_c1 + dc
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _build_region(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    padded = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=bool)
    padded[1:-1, 1:-1] = mask
    wall = _dilate8(padded) & ~padded
    return wall, padded


def _sample_seed_cells(cells: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    coords = cells[:]
    random.shuffle(coords)
    for gap in (2, 1, 0):
        chosen: list[tuple[int, int]] = []
        for r, c in coords:
            if all(max(abs(r - rr), abs(c - cc)) > gap for rr, cc in chosen):
                chosen.append((r, c))
                if len(chosen) == count:
                    return chosen
    return random.sample(cells, count)


def _place_regions(rows: int, cols: int, region_count: int) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((rows, cols), BG, dtype=int)
    blocked = np.zeros((rows, cols), dtype=bool)
    regions: list[dict[str, object]] = []
    max_inner_h = min(10, rows - 4)
    max_inner_w = min(10, cols - 4)

    for _ in range(region_count):
        placed = False
        for _attempt in range(120):
            interior_mask = _shape_mask(max_inner_h, max_inner_w)
            if interior_mask is None:
                return None
            wall_mask, interior_local = _build_region(interior_mask)
            h, w = wall_mask.shape
            if h + 2 > rows or w + 2 > cols:
                continue
            r0 = random.randint(1, rows - h - 1)
            c0 = random.randint(1, cols - w - 1)
            if blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1].any():
                continue

            for rr, cc in np.argwhere(wall_mask):
                grid[r0 + int(rr), c0 + int(cc)] = GREY
            blocked[r0 - 1 : r0 + h + 1, c0 - 1 : c0 + w + 1] = True

            interior_cells = [
                (r0 + int(rr), c0 + int(cc))
                for rr, cc in np.argwhere(interior_local)
            ]
            regions.append({"cells": interior_cells})
            placed = True
            break
        if not placed:
            return None
    return grid, regions


def _assign_seeds(grid: np.ndarray, regions: list[dict[str, object]]) -> bool:
    region_count = len(regions)
    seeded = [random.random() < 0.55 for _ in range(region_count)]
    if not any(seeded):
        seeded[random.randrange(region_count)] = True
    if region_count > 1 and all(seeded):
        seeded[random.randrange(region_count)] = False

    palette = [1, 2, 3, 4, 6, 7, 8, 9]
    seeded_count = sum(seeded)
    pool_size = random.randint(1, min(4, seeded_count, len(palette)))
    color_pool = random.sample(palette, pool_size)

    for idx, info in enumerate(regions):
        cells = list(info["cells"])
        if not seeded[idx]:
            info["seed_count"] = 0
            continue
        seed_count = random.randint(1, min(4, len(cells)))
        info["seed_count"] = seed_count
        color = random.choice(color_pool)
        info["color"] = color
        for r, c in _sample_seed_cells(cells, seed_count):
            grid[r, c] = color
    return True


def _valid_attempt(grid: np.ndarray, expected_regions: int) -> bool:
    enclosed = _enclosed_regions(grid)
    if len(enclosed) != expected_regions:
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(300):
        rows = random.randint(14, 28)
        cols = random.randint(14, 28)
        area = rows * cols
        max_regions = min(6, max(2, area // 120 + 1))
        region_count = random.randint(2, max_regions)

        placed = _place_regions(rows, cols, region_count)
        if placed is None:
            continue
        grid, regions = placed
        if not _assign_seeds(grid, regions):
            continue
        if _valid_attempt(grid, region_count):
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
