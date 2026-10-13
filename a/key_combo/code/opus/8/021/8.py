from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
WALL = 3
RAY_COLORS = {BLUE, RED}
DIR_RIGHT = (0, 1)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _mirror_walls(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int, copy=True)
    h, w = g.shape
    mid = w // 2
    mirrored = g.copy()
    mirrored[:, mid:] = g[:, mid:]
    for r in range(h):
        for c in range(mid):
            if int(g[r, c]) == WALL:
                mirrored[r, w - 1 - c] = WALL
    return mirrored


def _wall_lookup(structure: np.ndarray) -> dict[tuple[int, int], dict]:
    lookup: dict[tuple[int, int], dict] = {}
    for comp in _components(structure == WALL):
        rows = {r for r, _ in comp}
        cols = {c for _, c in comp}
        if len(rows) == 1:
            row = next(iter(rows))
            lo = min(cols)
            hi = max(cols)
            openings = ((row, lo - 1), (row, hi + 1))
            orientation = "h"
        else:
            col = next(iter(cols))
            lo = min(rows)
            hi = max(rows)
            openings = ((lo - 1, col), (hi + 1, col))
            orientation = "v"
        info = {
            "coords": tuple(comp),
            "openings": openings,
            "orientation": orientation,
        }
        for cell in comp:
            lookup[cell] = info
    return lookup


def _seed_cells(grid: np.ndarray) -> list[tuple[int, int, int]]:
    h, _ = grid.shape
    seeds: list[tuple[int, int, int]] = []
    for r in range(h):
        color = int(grid[r, 0])
        if color in RAY_COLORS:
            seeds.append((r, 0, color))
    return sorted(seeds)


def _trace_rays(
    structure: np.ndarray,
    *,
    return_meta: bool,
) -> tuple[np.ndarray, dict]:
    out = structure.copy()
    wall_lookup = _wall_lookup(structure)
    stack = _seed_cells(structure)[::-1]
    seen_starts: set[tuple[int, int, int]] = set()
    branch_count = 0
    wall_hits = 0
    horizontal_hits = 0
    ray_cells: set[tuple[int, int]] = set()

    while stack:
        r, c, color = stack.pop()
        state = (r, c, color)
        if state in seen_starts:
            continue
        seen_starts.add(state)
        if not _in_bounds(out.shape, r, c):
            continue
        if int(structure[r, c]) == WALL:
            continue
        if int(out[r, c]) in RAY_COLORS and int(out[r, c]) != color:
            continue
        if int(out[r, c]) == BG:
            out[r, c] = color
            ray_cells.add((r, c))
        cr, cc = r, c
        while True:
            nr, nc = cr + DIR_RIGHT[0], cc + DIR_RIGHT[1]
            if not _in_bounds(out.shape, nr, nc):
                break
            nv = int(structure[nr, nc])
            ov = int(out[nr, nc])
            if nv == WALL:
                info = wall_lookup[(nr, nc)]
                wall_hits += 1
                if info["orientation"] == "h":
                    horizontal_hits += 1
                for orow, ocol in info["openings"]:
                    if _in_bounds(out.shape, orow, ocol) and int(structure[orow, ocol]) == BG:
                        stack.append((orow, ocol, color))
                        branch_count += 1
                break
            if ov in RAY_COLORS:
                break
            out[nr, nc] = color
            ray_cells.add((nr, nc))
            cr, cc = nr, nc

    meta = {
        "branch_count": branch_count,
        "wall_hits": wall_hits,
        "horizontal_hits": horizontal_hits,
        "ray_cells": ray_cells,
    }
    return (out, meta) if return_meta else (out, {})


def _fill_regions(grid: np.ndarray, ray_cells: set[tuple[int, int]]) -> tuple[np.ndarray, dict]:
    out = grid.copy()
    h, w = out.shape
    filled_regions = 0
    filled_cells = 0
    for comp in _components(out == BG):
        region = set(comp)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in region):
            continue
        boundary: dict[int, set[tuple[int, int]]] = {BLUE: set(), RED: set()}
        for r, c in region:
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(out.shape, nr, nc) or (nr, nc) in region:
                    continue
                if (nr, nc) in ray_cells and int(out[nr, nc]) in boundary:
                    boundary[int(out[nr, nc])].add((nr, nc))
        counts = {color: len(cells) for color, cells in boundary.items()}
        if max(counts.values()) == 0:
            continue
        fill_color = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
        for r, c in region:
            out[r, c] = fill_color
        filled_regions += 1
        filled_cells += len(region)
    return out, {"filled_regions": filled_regions, "filled_cells": filled_cells}


def _solve(
    grid: np.ndarray,
    *,
    mirror_walls: bool = True,
    do_fill: bool = True,
    return_meta: bool = False,
) -> tuple[np.ndarray, dict]:
    structure = _mirror_walls(grid) if mirror_walls else np.array(grid, dtype=int, copy=True)
    rays, meta = _trace_rays(structure, return_meta=True)
    if do_fill:
        out, fill_meta = _fill_regions(rays, meta["ray_cells"])
        meta.update(fill_meta)
    else:
        out = rays
        meta.update({"filled_regions": 0, "filled_cells": 0})
    return (out, meta) if return_meta else (out, {})


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror the left-half wall drawing, branch rightward rays at wall openings, and fill the enclosed cavities.

    Input structure:
    - The grid has black background, green wall segments confined to the left half, and one or two single-pixel seeds on the left edge.
    - The right half starts empty; only the wall color participates in the hidden reflection step.
    - Wall components are one-pixel-thick straight horizontal or vertical segments separated in the input.

    Rule:
    - Reflect every green wall pixel from the left half across the vertical midline into the right half, while leaving the edge seeds unchanged.
    - From each seed, cast a ray to the right through background; when the next cell is a wall segment, stop and branch from the background cells immediately beyond that segment's two endpoints.
    - Repeat the same rightward ray rule for every new branch until a ray exits the grid or would enter an already painted ray cell, then flood-fill each non-border background cavity enclosed by walls and rays with the majority ray color on its boundary.

    Key insight:
    - The visible input omits the right-half walls, so the solver must infer the mirror first; that mirrored topology determines the branch openings, and the resulting ray network determines which cavities become fillable.
    """
    g = np.asarray(grid, dtype=int)
    out, _ = _solve(g, mirror_walls=True, do_fill=True, return_meta=False)
    return out


def _paint_segment(grid: np.ndarray, orientation: str, fixed: int, start: int, end: int) -> None:
    if orientation == "v":
        grid[start : end + 1, fixed] = WALL
    else:
        grid[fixed, start : end + 1] = WALL


def _gen_extra_seed_row(h: int, forbidden: set[int]) -> int | None:
    choices = [r for r in range(1, h - 1) if r not in forbidden]
    return random.choice(choices) if choices else None


def _build_candidate() -> np.ndarray | None:
    h = random.randint(13, 18)
    w = random.choice([14, 16, 18, 20])
    mid = w // 2
    grid = np.zeros((h, w), dtype=int)

    wall_c = random.randint(2, max(2, mid - 5))
    wall_len = random.randint(3, 5)
    top = random.randint(3, h - wall_len - 4)
    bottom = top + wall_len - 1
    upper_row = top - 1
    lower_row = bottom + 1
    seed_row = random.randint(top, bottom)

    _paint_segment(grid, "v", wall_c, top, bottom)

    rows_with_horizontal: list[int] = []
    for branch_row in (upper_row, lower_row):
        if random.random() < 0.78 or not rows_with_horizontal:
            start = random.randint(wall_c + 2, mid - 3)
            mode = random.choice(["join", "gap"])
            end = mid - 1 if mode == "join" else mid - 2
            if start > end:
                return None
            _paint_segment(grid, "h", branch_row, start, end)
            rows_with_horizontal.append(branch_row)

    if not rows_with_horizontal:
        return None

    seed_color = random.choice([BLUE, RED])
    grid[seed_row, 0] = seed_color

    if random.random() < 0.45:
        other_row = _gen_extra_seed_row(h, {seed_row, top, upper_row, lower_row, bottom})
        if other_row is not None:
            grid[other_row, 0] = RED if seed_color == BLUE else BLUE

    return grid


def _validate_candidate(grid: np.ndarray) -> bool:
    out, meta = _solve(grid, mirror_walls=True, do_fill=True, return_meta=True)
    no_mirror, no_mirror_meta = _solve(grid, mirror_walls=False, do_fill=True, return_meta=True)
    no_fill, _ = _solve(grid, mirror_walls=True, do_fill=False, return_meta=True)

    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, no_mirror):
        return False
    if np.array_equal(out, no_fill):
        return False
    if meta["branch_count"] < 2:
        return False
    if meta["horizontal_hits"] < 1:
        return False
    if meta["filled_regions"] < 1 or meta["filled_cells"] < 2:
        return False
    if no_mirror_meta["filled_regions"] >= meta["filled_regions"]:
        return False

    left_wall = int(np.count_nonzero(grid[:, : grid.shape[1] // 2] == WALL))
    mirrored_wall = int(np.count_nonzero(_mirror_walls(grid) == WALL))
    if mirrored_wall <= left_wall:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _build_candidate()
        if grid is not None and _validate_candidate(grid):
            return grid
    raise RuntimeError("failed to generate a valid mirrored-ray puzzle instance")
