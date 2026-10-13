from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
BG_COLOR = 0
BOUNDARY_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MAX_GRID_SIZE = 30
MAX_GEN_ATTEMPTS = 400
LOCAL_SHAPE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 120
MIN_GAP = 2


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIR4]


def _background_color(grid: np.ndarray) -> int:
    corners = [int(grid[0, 0]), int(grid[0, -1]), int(grid[-1, 0]), int(grid[-1, -1])]
    return Counter(corners).most_common(1)[0][0]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _boundary_mask(region: np.ndarray) -> np.ndarray:
    h, w = region.shape
    out = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if not bool(region[r, c]):
                continue
            for nr, nc in neighbors4(r, c):
                if not (0 <= nr < h and 0 <= nc < w) or not bool(region[nr, nc]):
                    out[r, c] = True
                    break
    return out


def _interior_from_boundary(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    q = deque()
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if bool(boundary_mask[r, c]) or bool(outside[r, c]):
                continue
            outside[r, c] = True
            q.append((r, c))
    while q:
        cr, cc = q.popleft()
        for nr, nc in neighbors4(cr, cc):
            if 0 <= nr < h and 0 <= nc < w and not bool(boundary_mask[nr, nc]) and not bool(outside[nr, nc]):
                outside[nr, nc] = True
                q.append((nr, nc))
    return (~boundary_mask) & (~outside)


def _layer_depth(boundary: set[tuple[int, int]], bg_cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    depth: dict[tuple[int, int], int] = {}
    q = deque()
    for cell in bg_cells:
        if any(nb in boundary for nb in neighbors4(*cell)):
            depth[cell] = 1
            q.append(cell)
    while q:
        cell = q.popleft()
        cur = depth[cell]
        for nb in neighbors4(*cell):
            if nb in bg_cells and nb not in depth:
                depth[nb] = cur + 1
                q.append(nb)
    return depth


def _extract_shapes(grid: np.ndarray, bg: int, boundary_color: int) -> list[dict]:
    shapes: list[dict] = []
    for comp in _components(grid == boundary_color):
        r0, c0, r1, c1 = _bbox(comp)
        local_boundary = (grid[r0 : r1 + 1, c0 : c1 + 1] == boundary_color)
        interior_local = _interior_from_boundary(local_boundary)
        if not bool(interior_local.any()):
            continue

        boundary = {(r, c) for r, c in comp}
        interior: set[tuple[int, int]] = set()
        seeds: list[tuple[int, int]] = []
        bg_cells: set[tuple[int, int]] = set()
        seed_colors: set[int] = set()

        for lr, lc in np.argwhere(interior_local):
            gr = int(r0 + int(lr))
            gc = int(c0 + int(lc))
            interior.add((gr, gc))
            val = int(grid[gr, gc])
            if val == bg:
                bg_cells.add((gr, gc))
            elif val != boundary_color:
                seeds.append((gr, gc))
                seed_colors.add(val)

        shapes.append(
            {
                "boundary": boundary,
                "interior": interior,
                "bg_cells": bg_cells,
                "seeds": seeds,
                "seed_colors": seed_colors,
            }
        )
    return shapes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw inward contour layers inside each closed outline using the seed color and seed count.

    Input structure:
    - The background is the dominant color.
    - Several one-pixel-thick closed irregular outlines share one common border color.
    - Each outline encloses background cells plus 1 to 3 isolated seed pixels of one shared non-border color.

    Rule:
    - Keep every outline pixel unchanged.
    - For each closed outline, count the seeds inside it; that count gives how many inward contour layers to paint.
    - Paint layer 1 on interior background cells that are 4-neighbor-adjacent to the outline, then layer 2 from layer 1, and layer 3 from layer 2 when needed.
    - Use the seed color for all painted layers, and erase the seed pixels back to background.

    Key insight:
    - The seeds are control signals rather than final content: their color selects the contour color, and their number selects the inward depth.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    non_bg = [int(v) for v in g.ravel() if int(v) != bg]
    if not non_bg:
        return g.copy()
    boundary_color = Counter(non_bg).most_common(1)[0][0]

    out = g.copy()
    for shape in _extract_shapes(g, bg, boundary_color):
        seeds = shape["seeds"]
        if not seeds:
            continue

        seed_colors = shape["seed_colors"]
        assert len(seed_colors) == 1, "all seeds inside one shape must share a color"
        fill_color = next(iter(seed_colors))
        layer_count = len(seeds)

        for r, c in seeds:
            out[r, c] = bg

        depth = _layer_depth(shape["boundary"], shape["bg_cells"])
        for cell, dist in depth.items():
            if dist <= layer_count:
                out[cell] = fill_color

    return out


def _choose_spaced_cells(
    candidates: list[tuple[int, int]],
    depth: dict[tuple[int, int], int],
    count: int,
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    ranked = candidates[:]
    rng.shuffle(ranked)
    ranked.sort(key=lambda cell: depth[cell], reverse=True)
    chosen: list[tuple[int, int]] = []
    for cell in ranked:
        if all(abs(cell[0] - rr) + abs(cell[1] - cc) > 1 for rr, cc in chosen):
            chosen.append(cell)
        if len(chosen) == count:
            return chosen
    return None


def _make_region(rng: random.Random, seed_count: int, dim_hi: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    dim_lo = max(8, seed_count * 2 + 4)
    for _ in range(LOCAL_SHAPE_ATTEMPTS):
        h = rng.randint(dim_lo, dim_hi)
        w = rng.randint(dim_lo, dim_hi)
        region = np.ones((h, w), dtype=bool)

        notch_count = rng.randint(1, 2 if seed_count == 3 else 4)
        for _ in range(notch_count):
            side = rng.choice(("top", "bottom", "left", "right"))
            if side in ("top", "bottom"):
                max_depth = min(2 if seed_count == 3 else 3, h - 5)
                if max_depth < 1 or w < 5:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(2, max(2, w - 2))
                span = min(span, w - 2)
                c0 = rng.randint(1, w - span - 1)
                if side == "top":
                    region[:depth, c0 : c0 + span] = False
                else:
                    region[h - depth :, c0 : c0 + span] = False
            else:
                max_depth = min(2 if seed_count == 3 else 3, w - 5)
                if max_depth < 1 or h < 5:
                    continue
                depth = rng.randint(1, max_depth)
                span = rng.randint(2, max(2, h - 2))
                span = min(span, h - 2)
                r0 = rng.randint(1, h - span - 1)
                if side == "left":
                    region[r0 : r0 + span, :depth] = False
                else:
                    region[r0 : r0 + span, w - depth :] = False

        if len(_components(region)) != 1:
            continue

        boundary_mask = _boundary_mask(region)
        if len(_components(boundary_mask)) != 1:
            continue

        interior_mask = region & (~boundary_mask)
        if not bool(interior_mask.any()):
            continue
        if len(_components(interior_mask)) != 1:
            continue

        boundary = {(int(r), int(c)) for r, c in np.argwhere(boundary_mask)}
        bg_cells = {(int(r), int(c)) for r, c in np.argwhere(interior_mask)}
        depth = _layer_depth(boundary, bg_cells)
        if not depth:
            continue
        if not all(any(dist == layer for dist in depth.values()) for layer in range(1, seed_count + 1)):
            continue

        candidates = [cell for cell, dist in depth.items() if dist > seed_count]
        seeds = _choose_spaced_cells(candidates, depth, seed_count, rng)
        if seeds is None:
            continue

        return region, seeds

    raise RuntimeError("failed to create a valid local region")


def _build_patch(rng: random.Random, seed_count: int, seed_color: int, dim_hi: int) -> np.ndarray:
    region, seeds = _make_region(rng, seed_count, dim_hi)
    boundary_mask = _boundary_mask(region)
    patch = np.full(region.shape, BG_COLOR, dtype=int)
    patch[boundary_mask] = BOUNDARY_COLOR
    for r, c in seeds:
        patch[r, c] = seed_color

    rot = rng.randint(0, 3)
    if rot:
        patch = np.rot90(patch, rot)
    if rng.random() < 0.5:
        patch = np.flipud(patch)
    if rng.random() < 0.5:
        patch = np.fliplr(patch)
    return patch.copy()


def _place_patches(rng: random.Random, patches: list[np.ndarray]) -> np.ndarray | None:
    canvas = np.full((MAX_GRID_SIZE, MAX_GRID_SIZE), BG_COLOR, dtype=int)
    blocked = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=bool)
    placements: list[tuple[np.ndarray, int, int]] = []

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, MAX_GRID_SIZE - ph - 2)
            c0 = rng.randint(1, MAX_GRID_SIZE - pw - 2)
            rs = max(0, r0 - MIN_GAP)
            re = min(MAX_GRID_SIZE, r0 + ph + MIN_GAP)
            cs = max(0, c0 - MIN_GAP)
            ce = min(MAX_GRID_SIZE, c0 + pw + MIN_GAP)
            if bool(blocked[rs:re, cs:ce].any()):
                continue
            blocked[rs:re, cs:ce] = True
            placements.append((patch, r0, c0))
            placed = True
            break
        if not placed:
            return None

    for patch, r0, c0 in placements:
        ph, pw = patch.shape
        mask = patch != BG_COLOR
        view = canvas[r0 : r0 + ph, c0 : c0 + pw]
        view[mask] = patch[mask]

    used = np.argwhere(canvas != BG_COLOR)
    if used.size == 0:
        return None

    rmin, cmin = used.min(axis=0)
    rmax, cmax = used.max(axis=0)
    top = rng.randint(1, min(6, int(rmin)))
    left = rng.randint(1, min(6, int(cmin)))
    bottom = rng.randint(1, min(6, int(MAX_GRID_SIZE - 1 - rmax)))
    right = rng.randint(1, min(6, int(MAX_GRID_SIZE - 1 - cmax)))
    return canvas[rmin - top : rmax + bottom + 1, cmin - left : cmax + right + 1]


def _validate_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    if max(grid.shape) > MAX_GRID_SIZE:
        return False
    if int(grid[0, 0]) != BG_COLOR or int(grid[0, -1]) != BG_COLOR or int(grid[-1, 0]) != BG_COLOR or int(grid[-1, -1]) != BG_COLOR:
        return False

    bg = _background_color(grid)
    non_bg = [int(v) for v in grid.ravel() if int(v) != bg]
    if not non_bg:
        return False
    boundary_color = Counter(non_bg).most_common(1)[0][0]
    if boundary_color != BOUNDARY_COLOR:
        return False

    shapes = _extract_shapes(grid, bg, boundary_color)
    if not (1 <= len(shapes) <= 3):
        return False

    seen_seeds: set[tuple[int, int]] = set()
    for shape in shapes:
        seeds = shape["seeds"]
        if len(seeds) not in (1, 2, 3):
            return False
        if len(shape["seed_colors"]) != 1:
            return False
        depth = _layer_depth(shape["boundary"], shape["bg_cells"])
        if not all(any(dist == layer for dist in depth.values()) for layer in range(1, len(seeds) + 1)):
            return False
        if any(seed in seen_seeds for seed in seeds):
            return False
        seen_seeds.update(seeds)

    expected_seed_cells = {(r, c) for shape in shapes for r, c in shape["seeds"]}
    actual_seed_cells = {
        (int(r), int(c))
        for r, c in np.argwhere((grid != bg) & (grid != boundary_color))
    }
    return expected_seed_cells == actual_seed_cells


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        rng = random.Random()
        object_count = rng.randint(1, 3)
        dim_hi = {1: 15, 2: 12, 3: 10}[object_count]

        seed_counts = [rng.randint(1, 3) for _ in range(object_count)]
        rng.shuffle(seed_counts)
        colors = list(rng.sample(SEED_COLORS, object_count))

        try:
            patches = [
                _build_patch(rng, seed_count=seed_counts[i], seed_color=colors[i], dim_hi=dim_hi)
                for i in range(object_count)
            ]
        except RuntimeError:
            continue

        grid = _place_patches(rng, patches)
        if grid is None:
            continue
        if _validate_grid(grid):
            return grid

    raise RuntimeError(f"failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
