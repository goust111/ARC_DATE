from collections import deque
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
LOOP_COUNT_RANGE = (3, 5)
GEN_ATTEMPTS = 300
PLACE_TRIES = 200
SHAPE_TRIES = 80
NOTCH_PROB = 0.45
PAD = 1


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _inside_area(boundary_mask: np.ndarray) -> np.ndarray:
    h, w = boundary_mask.shape
    reachable = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not boundary_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not boundary_mask[r, c] and not reachable[r, c]:
                reachable[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if boundary_mask[nr, nc] or reachable[nr, nc]:
                continue
            reachable[nr, nc] = True
            q.append((nr, nc))

    return (~reachable) & (~boundary_mask)


def _erode4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    inner = np.zeros((h, w), dtype=bool)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if not mask[r, c]:
                continue
            if mask[r - 1, c] and mask[r + 1, c] and mask[r, c - 1] and mask[r, c + 1]:
                inner[r, c] = True
    return inner


def _ring_from_outer(outer: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    interior = _erode4(outer)
    if not interior.any():
        return None
    boundary = outer & (~interior)
    if len(_components_mask(boundary)) != 1:
        return None
    if len(_components_mask(interior)) != 1:
        return None
    return boundary, interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill single-seed cavities and recolor each enclosing loop by the number of seeds inside it.

    Input structure:
    - The background is black.
    - Several one-color 4-connected closed boundary loops appear on the grid, and different loops may use different colors.
    - Each loop encloses a cavity containing black cells plus zero or more colored seed pixels.
    - Seed pixels are isolated colored cells inside cavities and are not part of the boundary loop itself.

    Rule:
    - For each monochrome closed loop, find the region inside that loop.
    - If the interior contains exactly one non-black seed pixel, recolor every black interior cell to that seed color.
    - Recolor the boundary according to the same seed count: keep its original color for 0 seeds, change it to the seed color for 1 seed, and change it to grey for 2 or more seeds.

    Key insight:
    The same per-cavity seed count controls two coupled effects: only single-seed cavities get filled, while the boundary color follows a three-way count rule (unchanged, seed color, or grey).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for color in [int(c) for c in np.unique(g) if int(c) != BG]:
        for comp in _components_mask(g == color):
            if len(comp) <= 1:
                continue
            boundary_mask = np.zeros_like(g, dtype=bool)
            rows, cols = zip(*comp)
            boundary_mask[rows, cols] = True
            inside = _inside_area(boundary_mask)
            if not inside.any():
                continue

            seed_coords = np.argwhere(inside & (g != BG))
            seed_count = int(seed_coords.shape[0])
            if seed_count == 0:
                continue
            if seed_count == 1:
                sr, sc = map(int, seed_coords[0])
                fill_color = int(g[sr, sc])
                out[inside & (g == BG)] = fill_color
                out[boundary_mask] = fill_color
            else:
                out[boundary_mask] = GREY

    return out


def _sample_seed_cells(
    interior_coords: list[tuple[int, int]], seed_count: int, rng: random.Random
) -> list[tuple[int, int]] | None:
    if seed_count == 0:
        return []
    coords = interior_coords[:]
    rng.shuffle(coords)
    chosen: list[tuple[int, int]] = []
    for r, c in coords:
        if all(abs(r - rr) + abs(c - cc) >= 2 for rr, cc in chosen):
            chosen.append((r, c))
        if len(chosen) == seed_count:
            return chosen
    return None


def _maybe_add_notch(outer: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = outer.shape
    if h < 7 or w < 7 or rng.random() >= NOTCH_PROB:
        return outer

    trial = outer.copy()
    side = rng.choice(["top", "bottom", "left", "right"])
    if side in ("top", "bottom"):
        notch_width = rng.randint(2, w - 4)
        start = rng.randint(2, w - notch_width - 2)
        depth = rng.randint(1, h - 4)
        if side == "top":
            trial[:depth, start : start + notch_width] = False
        else:
            trial[h - depth :, start : start + notch_width] = False
    else:
        notch_height = rng.randint(2, h - 4)
        start = rng.randint(2, h - notch_height - 2)
        depth = rng.randint(1, w - 4)
        if side == "left":
            trial[start : start + notch_height, :depth] = False
        else:
            trial[start : start + notch_height, w - depth :] = False
    return trial


def _make_ring_shape(
    seed_count: int, max_h: int, max_w: int, rng: random.Random
) -> tuple[np.ndarray, np.ndarray, list[tuple[int, int]]] | None:
    min_cells = max(4, seed_count * 3)
    for _ in range(SHAPE_TRIES):
        h = rng.randint(5, max_h)
        w = rng.randint(5, max_w)
        outer = np.ones((h, w), dtype=bool)
        outer = _maybe_add_notch(outer, rng)
        ring = _ring_from_outer(outer)
        if ring is None:
            continue
        boundary, interior = ring
        interior_coords = [(int(r), int(c)) for r, c in np.argwhere(interior)]
        if len(interior_coords) < min_cells:
            continue
        seeds = _sample_seed_cells(interior_coords, seed_count, rng)
        if seeds is None:
            continue
        return boundary, interior, seeds
    return None


def _shape_limits(loop_count: int) -> tuple[int, int]:
    if loop_count == 3:
        return 10, 11
    if loop_count == 4:
        return 9, 10
    return 8, 9


def _place_spec(
    grid: np.ndarray, occ: np.ndarray, spec: dict, rng: random.Random
) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = grid.shape
    boundary = spec["boundary"]
    interior = spec["interior"]
    box_h, box_w = boundary.shape

    for _ in range(PLACE_TRIES):
        r0 = rng.randint(0, h - box_h)
        c0 = rng.randint(0, w - box_w)
        rr0 = max(0, r0 - PAD)
        rr1 = min(h, r0 + box_h + PAD)
        cc0 = max(0, c0 - PAD)
        cc1 = min(w, c0 + box_w + PAD)
        if occ[rr0:rr1, cc0:cc1].any():
            continue

        boundary_mask = np.zeros_like(grid, dtype=bool)
        interior_mask = np.zeros_like(grid, dtype=bool)
        boundary_mask[r0 : r0 + box_h, c0 : c0 + box_w] = boundary
        interior_mask[r0 : r0 + box_h, c0 : c0 + box_w] = interior
        occ[rr0:rr1, cc0:cc1] = True
        return boundary_mask, interior_mask
    return None


def _branch_seed_counts(loop_count: int, rng: random.Random) -> list[int]:
    counts = [0, 1, rng.randint(2, 3)]
    for _ in range(loop_count - 3):
        branch = rng.choice([0, 1, 2])
        counts.append(branch if branch < 2 else rng.randint(2, 3))
    rng.shuffle(counts)
    return counts


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    loop_count = rng.randint(*LOOP_COUNT_RANGE)
    grid_h = rng.randint(*GRID_H_RANGE)
    grid_w = rng.randint(*GRID_W_RANGE)
    max_h, max_w = _shape_limits(loop_count)

    seed_counts = _branch_seed_counts(loop_count, rng)
    boundary_colors = rng.sample(PALETTE, loop_count)
    specs: list[dict] = []

    for seed_count, boundary_color in zip(seed_counts, boundary_colors):
        shape = _make_ring_shape(seed_count, max_h, max_w, rng)
        if shape is None:
            return None
        boundary, interior, local_seed_cells = shape
        seed_colors = rng.sample([c for c in PALETTE if c != boundary_color], seed_count)
        specs.append(
            {
                "boundary": boundary,
                "interior": interior,
                "seed_cells": local_seed_cells,
                "boundary_color": int(boundary_color),
                "seed_colors": [int(c) for c in seed_colors],
            }
        )

    specs.sort(key=lambda spec: spec["boundary"].shape[0] * spec["boundary"].shape[1], reverse=True)

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    expected = grid.copy()
    occ = np.zeros((grid_h, grid_w), dtype=bool)

    for spec in specs:
        placed = _place_spec(grid, occ, spec, rng)
        if placed is None:
            return None
        boundary_mask, interior_mask = placed

        grid[boundary_mask] = spec["boundary_color"]
        expected[boundary_mask] = spec["boundary_color"]

        seed_globals: list[tuple[int, int, int]] = []
        box_rows, box_cols = np.where(boundary_mask | interior_mask)
        r0 = int(box_rows.min())
        c0 = int(box_cols.min())
        for (lr, lc), color in zip(spec["seed_cells"], spec["seed_colors"]):
            gr = r0 + lr
            gc = c0 + lc
            seed_globals.append((gr, gc, color))
            grid[gr, gc] = color
            expected[gr, gc] = color

        seed_count = len(seed_globals)
        if seed_count == 1:
            fill_color = seed_globals[0][2]
            expected[interior_mask] = fill_color
            expected[boundary_mask] = fill_color
        elif seed_count >= 2:
            expected[boundary_mask] = GREY

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
