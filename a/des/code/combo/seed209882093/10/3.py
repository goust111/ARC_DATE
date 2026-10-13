import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (2, 3, 3, 2)
OBJECT_SIZE_RANGE = (6, 12)
MIN_INTERIOR_CELLS = 6
SHAPE_TRIES = 80
OBJECT_TRIES = 80
PLACE_TRIES = 120
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


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
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _perimeter(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    border = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    border[r, c] = True
                    break
    return border


def _distance_to_boundary(region_mask: np.ndarray, boundary_mask: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = region_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q = deque()

    for r in range(h):
        for c in range(w):
            if not region_mask[r, c]:
                continue
            if any(0 <= nr < h and 0 <= nc < w and boundary_mask[nr, nc] for nr, nc in _neighbors4(r, c)):
                dist[r, c] = 1
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and region_mask[nr, nc] and dist[nr, nc] < 0:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))

    return {
        (r, c): int(dist[r, c])
        for r in range(h)
        for c in range(w)
        if region_mask[r, c] and dist[r, c] > 0
    }


def _extract_regions(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    regions: list[dict] = []
    nongrey = g != GREY

    for cells in _components(nongrey):
        if _touches_border(cells, h, w):
            continue
        seed_cells = [(r, c) for r, c in cells if int(g[r, c]) not in (BLACK, GREY)]
        if not seed_cells:
            continue
        seed_colors = {int(g[r, c]) for r, c in seed_cells}
        if len(seed_colors) != 1:
            continue

        region_mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            region_mask[r, c] = True
        distances = _distance_to_boundary(region_mask, g == GREY)
        if len(distances) != len(cells):
            continue

        regions.append(
            {
                "cells": cells,
                "seed_cells": seed_cells,
                "seed_color": next(iter(seed_colors)),
                "seed_count": len(seed_cells),
                "distances": distances,
            }
        )

    regions.sort(key=lambda reg: min(reg["cells"]))
    return regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-colored seed pixels inside each grey closed outline and paint
    that many inner layers along the outline's interior wall.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for region in _extract_regions(g):
        fill_color = int(region["seed_color"])
        depth = int(region["seed_count"])
        for (r, c), dist in region["distances"].items():
            if dist <= depth:
                out[r, c] = fill_color

    return out


def _build_rectangle(h: int, w: int) -> np.ndarray:
    return np.ones((h, w), dtype=bool)


def _build_corner_notch(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    cut_h = rng.randint(2, h - 3)
    cut_w = rng.randint(2, w - 3)
    corner = rng.choice(("tl", "tr", "bl", "br"))
    if corner == "tl":
        mask[:cut_h, :cut_w] = False
    elif corner == "tr":
        mask[:cut_h, w - cut_w :] = False
    elif corner == "bl":
        mask[h - cut_h :, :cut_w] = False
    else:
        mask[h - cut_h :, w - cut_w :] = False
    return mask


def _build_double_corner_notch(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    corners = rng.sample(("tl", "tr", "bl", "br"), 2)
    max_h = min(4, h - 3)
    max_w = min(4, w - 3)
    for corner in corners:
        cut_h = rng.randint(2, max_h)
        cut_w = rng.randint(2, max_w)
        if corner == "tl":
            mask[:cut_h, :cut_w] = False
        elif corner == "tr":
            mask[:cut_h, w - cut_w :] = False
        elif corner == "bl":
            mask[h - cut_h :, :cut_w] = False
        else:
            mask[h - cut_h :, w - cut_w :] = False
    return mask


def _build_side_notch(rng: random.Random, h: int, w: int) -> np.ndarray:
    mask = np.ones((h, w), dtype=bool)
    side = rng.choice(("top", "bottom", "left", "right"))
    if side in ("top", "bottom"):
        depth = rng.randint(2, h - 3)
        span = rng.randint(2, w - 4)
        start = rng.randint(1, w - span - 1)
        if side == "top":
            mask[:depth, start : start + span] = False
        else:
            mask[h - depth :, start : start + span] = False
    else:
        depth = rng.randint(2, w - 3)
        span = rng.randint(2, h - 4)
        start = rng.randint(1, h - span - 1)
        if side == "left":
            mask[start : start + span, :depth] = False
        else:
            mask[start : start + span, w - depth :] = False
    return mask


def _shape_ok(mask: np.ndarray) -> bool:
    if int(mask.sum()) < MIN_INTERIOR_CELLS + 12:
        return False

    if len(_components(mask)) != 1:
        return False

    boundary = _perimeter(mask)
    if not boundary.any() or len(_components(boundary)) != 1:
        return False

    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not boundary[r, c]:
                continue
            degree = 0
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and boundary[nr, nc]:
                    degree += 1
            if degree != 2:
                return False

    interior = mask & (~boundary)
    if int(interior.sum()) < MIN_INTERIOR_CELLS:
        return False
    if len(_components(interior)) != 1:
        return False

    distances = _distance_to_boundary(interior, boundary)
    return len(distances) == int(interior.sum())


def _random_outer_shape(rng: random.Random, size_lo: int, size_hi: int) -> np.ndarray | None:
    builders = (
        ("rect", 3),
        ("corner", 3),
        ("side", 2),
        ("double", 2),
    )
    labels = [name for name, _ in builders]
    weights = [weight for _, weight in builders]

    for _ in range(SHAPE_TRIES):
        h = rng.randint(size_lo, size_hi)
        w = rng.randint(size_lo, size_hi)
        kind = rng.choices(labels, weights=weights, k=1)[0]

        if kind == "rect":
            mask = _build_rectangle(h, w)
        elif kind == "corner":
            mask = _build_corner_notch(rng, h, w)
        elif kind == "side":
            mask = _build_side_notch(rng, h, w)
        else:
            mask = _build_double_corner_notch(rng, h, w)

        if _shape_ok(mask):
            return mask
    return None


def _make_object_patch(rng: random.Random, seed_color: int, size_lo: int, size_hi: int) -> np.ndarray | None:
    for _ in range(OBJECT_TRIES):
        outer = _random_outer_shape(rng, size_lo, size_hi)
        if outer is None:
            continue

        boundary = _perimeter(outer)
        interior = outer & (~boundary)
        distances = _distance_to_boundary(interior, boundary)
        if len(distances) != int(interior.sum()):
            continue

        max_depth = max(distances.values())
        seed_count = rng.randint(1, min(3, max_depth))
        seeds = rng.sample(list(distances.keys()), seed_count)

        patch = np.zeros(outer.shape, dtype=int)
        patch[boundary] = GREY
        for r, c in seeds:
            patch[r, c] = int(seed_color)
        return patch

    return None


def _can_place(reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> bool:
    h, w = reserved.shape
    ph, pw = patch.shape
    if top < 1 or left < 1 or top + ph >= h or left + pw >= w:
        return False
    r0 = top - 1
    c0 = left - 1
    r1 = top + ph + 1
    c1 = left + pw + 1
    return not reserved[r0:r1, c0:c1].any()


def _place_patch(grid: np.ndarray, reserved: np.ndarray, top: int, left: int, patch: np.ndarray) -> None:
    ph, pw = patch.shape
    grid[top : top + ph, left : left + pw] = patch
    reserved[top - 1 : top + ph + 1, left - 1 : left + pw + 1] = True


def _generated_grid_ok(grid: np.ndarray, expected_regions: int) -> bool:
    if expected_regions < 1:
        return False
    if np.any(grid[0, :] == GREY) or np.any(grid[-1, :] == GREY):
        return False
    if np.any(grid[:, 0] == GREY) or np.any(grid[:, -1] == GREY):
        return False

    grey_components = _components(grid == GREY)
    if len(grey_components) != expected_regions:
        return False

    regions = _extract_regions(grid)
    if len(regions) != expected_regions:
        return False
    if len({reg["seed_color"] for reg in regions}) != expected_regions:
        return False
    if any(not (1 <= reg["seed_count"] <= 3) for reg in regions):
        return False
    if any(reg["seed_count"] > max(reg["distances"].values()) for reg in regions):
        return False

    out = main(grid)
    return not np.array_equal(out, grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    target_objects = rng.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]

    min_h = max(GRID_H_RANGE[0], 15 + 3 * target_objects)
    min_w = max(GRID_W_RANGE[0], 15 + 3 * target_objects)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    h = rng.randint(min_h, GRID_H_RANGE[1])
    w = rng.randint(min_w, GRID_W_RANGE[1])
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    if target_objects >= len(SEED_COLORS):
        return None
    colors = rng.sample(SEED_COLORS, target_objects)

    size_hi = OBJECT_SIZE_RANGE[1]
    if target_objects == 3:
        size_hi = 10
    elif target_objects == 4:
        size_hi = 9

    for seed_color in colors:
        patch = _make_object_patch(rng, seed_color, OBJECT_SIZE_RANGE[0], size_hi)
        if patch is None:
            return None

        ph, pw = patch.shape
        if ph + 2 >= h or pw + 2 >= w:
            return None

        placed = False
        for _ in range(PLACE_TRIES):
            top = rng.randint(1, h - ph - 1)
            left = rng.randint(1, w - pw - 1)
            if not _can_place(reserved, top, left, patch):
                continue
            _place_patch(grid, reserved, top, left, patch)
            placed = True
            break
        if not placed:
            return None

    if not _generated_grid_ok(grid, target_objects):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed to build a valid random instance")
