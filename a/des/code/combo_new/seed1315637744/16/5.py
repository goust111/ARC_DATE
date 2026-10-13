from __future__ import annotations

from collections import Counter
import random

import numpy as np


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
QUADRANT_TO_GROWTH = {
    "ul": (-1, 0),
    "ur": (0, 1),
    "ll": (0, -1),
    "lr": (1, 0),
}

PALETTE = tuple(range(10))
GEN_ATTEMPTS = 300
PATCH_ATTEMPTS = 80
PLACE_ATTEMPTS = 80


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def get_bg(grid: np.ndarray) -> int:
    return Counter(map(int, grid.reshape(-1))).most_common(1)[0][0]


def _components_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _wall_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(int(v) for v in grid.reshape(-1) if int(v) != int(bg))
    assert counts, "expected at least one non-background color"
    return max(counts, key=counts.get)


def _quadrant(r: int, c: int, center_r: float, center_c: float) -> str:
    if r < center_r and c < center_c:
        return "ul"
    if r < center_r and c >= center_c:
        return "ur"
    if r >= center_r and c < center_c:
        return "ll"
    return "lr"


def _growth_capacity(region_mask: np.ndarray, r: int, c: int, dr: int, dc: int) -> int:
    steps = 0
    rr, cc = int(r), int(c)
    h, w = region_mask.shape
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if not region_mask[rr, cc]:
            break
        steps += 1
    return steps


def _analyze_regions(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    wall = int(_wall_color(g, bg))
    non_wall = g != wall
    h, w = g.shape

    regions: list[dict[str, object]] = []
    for comp in _components_mask(non_wall):
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in comp):
            continue
        seeds = [(r, c) for r, c in comp if int(g[r, c]) not in (bg, wall)]
        if not seeds:
            continue
        rows = np.array([r for r, _ in comp], dtype=float)
        cols = np.array([c for _, c in comp], dtype=float)
        region_mask = np.zeros_like(g, dtype=bool)
        for r, c in comp:
            region_mask[r, c] = True
        regions.append(
            {
                "cells": comp,
                "seeds": sorted(seeds),
                "seed_count": len(seeds),
                "center_r": float(rows.mean()),
                "center_c": float(cols.mean()),
                "mask": region_mask,
            }
        )
    return bg, wall, regions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each seed along its room wall-facing axis determined by its quadrant inside an enclosed region.

    Input structure:
    - The background color fills the canvas, while one wall color draws thin closed boundaries.
    - Each enclosed non-wall component is a room made of background cells plus a few colored singleton seeds.
    - Seeds sit strictly inside their room and rooms are separated from each other by the wall color.

    Rule:
    - Identify every enclosed room formed by the wall color and count its seed pixels; call that count N for the room.
    - Compute the room's visual center from all non-wall cells in the room, then classify each seed as upper-left, upper-right, lower-left, or lower-right relative to that center.
    - Map quadrants to growth directions: upper-left grows up, upper-right grows right, lower-left grows left, lower-right grows down.
    - From each seed, paint up to N additional pixels of the same color in that direction, excluding the seed itself and stopping early at a wall cell or the grid boundary.

    Key insight:
    - The seed count is shared by the whole room, but the growth direction is local to each seed's position relative to the room center.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg, wall, regions = _analyze_regions(g)

    for region in regions:
        seed_count = int(region["seed_count"])
        center_r = float(region["center_r"])
        center_c = float(region["center_c"])
        for r, c in region["seeds"]:
            quadrant = _quadrant(int(r), int(c), center_r, center_c)
            dr, dc = QUADRANT_TO_GROWTH[quadrant]
            rr, cc = int(r), int(c)
            for _ in range(seed_count):
                rr += dr
                cc += dc
                if not (0 <= rr < g.shape[0] and 0 <= cc < g.shape[1]):
                    break
                if int(g[rr, cc]) == wall:
                    break
                out[rr, cc] = int(g[r, c])

    assert int(bg) != int(wall)
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    comps = _components_mask(mask)
    return len(comps) == 1


def _make_shape(rng: random.Random) -> np.ndarray | None:
    inner_h = rng.randint(5, 9)
    inner_w = rng.randint(5, 10)
    shape = np.ones((inner_h, inner_w), dtype=bool)

    notch_count = rng.choices([0, 1, 2], weights=[1, 4, 3])[0]
    for _ in range(notch_count):
        candidate = shape.copy()
        side = rng.choice(["top", "bottom", "left", "right"])
        if side in ("top", "bottom"):
            max_depth = max(1, inner_h // 2)
            depth = rng.randint(1, max_depth)
            width = rng.randint(1, max(1, inner_w - 2))
            start = rng.randint(0, inner_w - width)
            rows = slice(0, depth) if side == "top" else slice(inner_h - depth, inner_h)
            candidate[rows, start : start + width] = False
        else:
            max_depth = max(1, inner_w // 2)
            depth = rng.randint(1, max_depth)
            height = rng.randint(1, max(1, inner_h - 2))
            start = rng.randint(0, inner_h - height)
            cols = slice(0, depth) if side == "left" else slice(inner_w - depth, inner_w)
            candidate[start : start + height, cols] = False

        if candidate.sum() < 12 or not candidate.any():
            continue
        if _mask_connected(candidate):
            shape = candidate

    rows = np.flatnonzero(shape.any(axis=1))
    cols = np.flatnonzero(shape.any(axis=0))
    if len(rows) < 4 or len(cols) < 4:
        return None
    return shape


def _shape_candidates(shape: np.ndarray) -> dict[str, list[dict[str, object]]]:
    rows, cols = np.where(shape)
    center_r = float(np.mean(rows))
    center_c = float(np.mean(cols))
    candidates = {quad: [] for quad in QUADRANT_TO_GROWTH}

    for r, c in zip(rows, cols):
        if abs(float(r) - center_r) < 1e-9 or abs(float(c) - center_c) < 1e-9:
            continue
        quad = _quadrant(int(r), int(c), center_r, center_c)
        dr, dc = QUADRANT_TO_GROWTH[quad]
        capacity = _growth_capacity(shape, int(r), int(c), dr, dc)
        if capacity < 1:
            continue
        path = []
        rr, cc = int(r), int(c)
        for _ in range(capacity):
            rr += dr
            cc += dc
            path.append((rr, cc))
        candidates[quad].append(
            {
                "seed": (int(r), int(c)),
                "capacity": int(capacity),
                "path": tuple(path),
                "quadrant": quad,
            }
        )
    return candidates


def _choose_seed_specs(
    shape: np.ndarray, seed_count: int, rng: random.Random
) -> list[dict[str, object]] | None:
    candidates = _shape_candidates(shape)
    usable_quads = [quad for quad, items in candidates.items() if items]
    if len(usable_quads) < seed_count:
        return None

    chosen_quads = rng.sample(usable_quads, seed_count)
    desired_modes: dict[str, str] = {}
    for quad in chosen_quads:
        desired_modes[quad] = rng.choice(["full", "short"])

    if seed_count >= 2:
        desired_modes[chosen_quads[0]] = "full"
        desired_modes[chosen_quads[1]] = "short"

    ordered_quads = sorted(chosen_quads, key=lambda quad: len(candidates[quad]))
    reserved: set[tuple[int, int]] = set()
    chosen: list[dict[str, object]] = []

    def backtrack(idx: int) -> bool:
        if idx == len(ordered_quads):
            return True

        quad = ordered_quads[idx]
        want = desired_modes[quad]
        preferred = [
            item
            for item in candidates[quad]
            if (item["capacity"] >= seed_count) == (want == "full")
        ]
        fallback = [item for item in candidates[quad] if item not in preferred]
        rng.shuffle(preferred)
        rng.shuffle(fallback)

        for item in preferred + fallback:
            seed = item["seed"]
            draw_len = min(int(item["capacity"]), seed_count)
            path_cells = set(item["path"][:draw_len])
            occupied = path_cells | {seed}
            if occupied & reserved:
                continue
            reserved.update(occupied)
            chosen.append(
                {
                    "seed": seed,
                    "quadrant": quad,
                    "capacity": int(item["capacity"]),
                    "draw_len": int(draw_len),
                }
            )
            if backtrack(idx + 1):
                return True
            chosen.pop()
            reserved.difference_update(occupied)
        return False

    if not backtrack(0):
        return None
    return chosen


def _render_patch(
    shape: np.ndarray,
    seed_specs: list[dict[str, object]],
    bg: int,
    wall: int,
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    patch_h = shape.shape[0] + 2
    patch_w = shape.shape[1] + 2
    patch = np.full((patch_h, patch_w), bg, dtype=int)
    region = np.zeros((patch_h, patch_w), dtype=bool)
    region[1:-1, 1:-1] = shape

    wall_mask = np.zeros_like(region)
    for r, c in np.argwhere(region):
        rr, cc = int(r), int(c)
        for nr, nc in neighbors4(rr, cc):
            if 0 <= nr < patch_h and 0 <= nc < patch_w and not region[nr, nc]:
                wall_mask[nr, nc] = True

    patch[wall_mask] = wall
    for color, spec in zip(colors, seed_specs):
        sr, sc = spec["seed"]
        patch[sr + 1, sc + 1] = int(color)
    return patch, wall_mask


def _place_patch(
    grid: np.ndarray,
    reserve: np.ndarray,
    patch: np.ndarray,
    bg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if ph + 2 > h or pw + 2 > w:
        return False

    for _ in range(PLACE_ATTEMPTS):
        top = rng.randint(1, h - ph - 1)
        left = rng.randint(1, w - pw - 1)
        r0 = top - 1
        c0 = left - 1
        r1 = top + ph + 1
        c1 = left + pw + 1
        if reserve[r0:r1, c0:c1].any():
            continue
        sub = grid[top : top + ph, left : left + pw]
        if np.any((patch != bg) & (sub != bg)):
            continue
        reserve[r0:r1, c0:c1] = True
        non_bg = patch != bg
        sub[non_bg] = patch[non_bg]
        return True
    return False


def _validate_generated(grid: np.ndarray) -> bool:
    bg, wall, regions = _analyze_regions(grid)
    if bg == wall or len(regions) < 2:
        return False

    directions: list[str] = []
    total_growth = 0

    for region in regions:
        seed_count = int(region["seed_count"])
        if not (1 <= seed_count <= 4):
            return False
        region_mask = np.asarray(region["mask"], dtype=bool)
        center_r = float(region["center_r"])
        center_c = float(region["center_c"])
        for r, c in region["seeds"]:
            if abs(float(r) - center_r) < 1e-9 or abs(float(c) - center_c) < 1e-9:
                return False
            quad = _quadrant(int(r), int(c), center_r, center_c)
            directions.append(quad)
            dr, dc = QUADRANT_TO_GROWTH[quad]
            local_mask = region_mask
            capacity = _growth_capacity(local_mask, int(r), int(c), dr, dc)
            if capacity < 1:
                return False
            total_growth += min(capacity, seed_count)

    if len(set(directions)) < 2:
        return False
    if total_growth < sum(int(region["seed_count"]) for region in regions):
        return False
    return not np.array_equal(main(grid), grid)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(18, 30)
    w = rng.randint(18, 30)
    bg = rng.choice(PALETTE)
    wall_choices = [c for c in PALETTE if c != bg]
    wall = rng.choice(wall_choices)
    seed_palette = [c for c in PALETTE if c not in (bg, wall)]

    grid = np.full((h, w), bg, dtype=int)
    reserve = np.zeros((h, w), dtype=bool)
    region_target = rng.randint(2, 4)
    placed = 0

    for _ in range(region_target):
        success = False
        for _ in range(PATCH_ATTEMPTS):
            shape = _make_shape(rng)
            if shape is None:
                continue
            seed_count = rng.randint(1, 4)
            seed_specs = _choose_seed_specs(shape, seed_count, rng)
            if seed_specs is None:
                continue
            colors = rng.sample(seed_palette, seed_count)
            patch, _wall_mask = _render_patch(shape, seed_specs, bg, wall, colors)
            if _place_patch(grid, reserve, patch, bg, rng):
                placed += 1
                success = True
                break
        if not success:
            return None

    if placed != region_target:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 5")
