from __future__ import annotations

from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
REGION_COLORS = (1, 2, 3, 4, 6, 7, 8)

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SIDE_TO_DIR = {"top": 0, "right": 1, "bottom": 2, "left": 3}

GRID_H_RANGE = (18, 29)
GRID_W_RANGE = (18, 29)
MODULE_COUNT_CHOICES = (1, 2, 3)
MODULE_COUNT_WEIGHTS = np.array([0.3, 0.5, 0.2], dtype=float)
MAX_GENERATION_ATTEMPTS = 400
MAX_MODULE_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in pts]
    cols = [c for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        cells.append((r0, c))
        cells.append((r1, c))
    for r in range(r0 + 1, r1):
        cells.append((r, c0))
        cells.append((r, c1))
    return cells


def _detect_fences(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    fences: list[dict] = []
    for comp in _components(g == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 4 or c1 - c0 + 1 < 4:
            continue
        comp_set = set(comp)
        perimeter = set(_perimeter_cells(r0, c0, r1, c1))
        if not comp_set <= perimeter:
            continue
        if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GREY):
            continue
        missing = list(perimeter - comp_set)
        if len(missing) != 1:
            continue
        gap = missing[0]
        gr, gc = gap
        if (gr, gc) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
            continue
        if int(g[gr, gc]) != BLACK:
            continue

        if gr == r0:
            side = "top"
        elif gr == r1:
            side = "bottom"
        elif gc == c0:
            side = "left"
        else:
            side = "right"
        dir_idx = SIDE_TO_DIR[side]

        segments = []
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        for color in sorted(set(int(v) for v in np.unique(interior)) - {BLACK}):
            locs = np.argwhere(interior == color)
            area = int(locs.shape[0])
            if area == 0:
                continue
            abs_pts = [(r0 + 1 + int(rr), c0 + 1 + int(cc)) for rr, cc in locs]
            dist = min(max(abs(rr - gr), abs(cc - gc)) for rr, cc in abs_pts)
            anchor = min(abs_pts)
            segments.append((int(dist), anchor[0], anchor[1], int(color), area))
        if not segments:
            continue
        segments.sort()
        fences.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "dir_idx": dir_idx,
                "segments": [(color, area) for _d, _ar, _ac, color, area in segments],
            }
        )

    fences.sort(key=lambda f: (f["gap"][0], f["gap"][1], f["bbox"]))
    return fences


def _paint_branch(
    out: np.ndarray,
    reserved: np.ndarray,
    gap: tuple[int, int],
    dir_idx: int,
    segments: list[tuple[int, int]],
) -> None:
    h, w = out.shape
    current = gap
    current_dir = dir_idx
    for color, length in segments:
        for _ in range(length):
            for _turns in range(4):
                dr, dc = DIRS[current_dir]
                nr = current[0] + dr
                nc = current[1] + dc
                if 0 <= nr < h and 0 <= nc < w and not reserved[nr, nc] and int(out[nr, nc]) == BLACK:
                    out[nr, nc] = int(color)
                    current = (nr, nc)
                    break
                current_dir = (current_dir + 1) % 4
            else:
                return


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow color segments out of each one-gap grey fence.

    Segment order is determined by the nearest region-to-gap Chebyshev distance,
    segment length is the region area, and the branch turns right whenever the
    next cell would hit an obstacle or the grid boundary.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    fences = _detect_fences(g)
    reserved = np.zeros_like(g, dtype=bool)
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    for fence in fences:
        _paint_branch(out, reserved, fence["gap"], fence["dir_idx"], fence["segments"])
    return out


def _spaced_positions(count: int, limit: int, rng: np.random.Generator) -> list[int]:
    usable = limit - (count - 1)
    base = sorted(int(v) for v in rng.choice(np.arange(1, usable + 1), size=count, replace=False))
    return [base[i] + i for i in range(count)]


def _sample_regions(
    side: str,
    ih: int,
    iw: int,
    gap_offset: int,
    num_colors: int,
    rng: np.random.Generator,
) -> list[dict]:
    colors = [int(v) for v in rng.choice(np.array(REGION_COLORS), size=num_colors, replace=False)]
    if side in ("top", "bottom"):
        depths = _spaced_positions(num_colors, ih, rng)
    else:
        depths = _spaced_positions(num_colors, iw, rng)

    regions: list[dict] = []
    for depth, color in zip(depths, colors, strict=False):
        if side in ("top", "bottom"):
            length = int(rng.integers(2, iw + 1))
            left_min = max(1, gap_offset - length + 1)
            left_max = min(gap_offset, iw - length + 1)
            left = int(rng.integers(left_min, left_max + 1))
            row = depth - 1 if side == "top" else ih - depth
            cells = [(row, col) for col in range(left - 1, left - 1 + length)]
        else:
            length = int(rng.integers(2, ih + 1))
            top_min = max(1, gap_offset - length + 1)
            top_max = min(gap_offset, ih - length + 1)
            top = int(rng.integers(top_min, top_max + 1))
            col = depth - 1 if side == "left" else iw - depth
            cells = [(row, col) for row in range(top - 1, top - 1 + length)]
        regions.append({"depth": int(depth), "color": color, "cells": cells, "area": int(length)})
    regions.sort(key=lambda region: region["depth"])
    return regions


def _eligible_profiles(total_len: int) -> list[tuple[str, ...]]:
    profiles = [
        tuple(),
        ("obstacle",),
        ("boundary",),
        ("obstacle", "obstacle"),
        ("boundary", "obstacle"),
        ("obstacle", "boundary"),
    ]
    return [profile for profile in profiles if total_len >= 2 * len(profile) + 1]


def _module_geometry(
    r0: int,
    c0: int,
    ih: int,
    iw: int,
    side: str,
    gap_offset: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int], int]:
    r1 = r0 + ih + 1
    c1 = c0 + iw + 1
    if side == "top":
        gap = (r0, c0 + gap_offset)
    elif side == "bottom":
        gap = (r1, c0 + gap_offset)
    elif side == "left":
        gap = (r0 + gap_offset, c0)
    else:
        gap = (r0 + gap_offset, c1)
    return (r0, c0, r1, c1), gap, SIDE_TO_DIR[side]


def _forward_free(
    current: tuple[int, int],
    dir_idx: int,
    occupied: np.ndarray,
) -> tuple[list[tuple[int, int]], bool]:
    h, w = occupied.shape
    dr, dc = DIRS[dir_idx]
    cells: list[tuple[int, int]] = []
    rr = current[0] + dr
    cc = current[1] + dc
    while 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    blocked_by_boundary = not (0 <= rr < h and 0 <= cc < w)
    return cells, blocked_by_boundary


def _plan_path_recursive(
    current: tuple[int, int],
    dir_idx: int,
    remaining: int,
    causes: tuple[str, ...],
    occupied: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    free_cells, blocked_by_boundary = _forward_free(current, dir_idx, occupied)

    if not causes:
        if remaining <= len(free_cells):
            return free_cells[:remaining], []
        return None

    cause = causes[0]
    min_rest = 2 * (len(causes) - 1) + 1
    if cause == "boundary":
        step_count = len(free_cells)
        if not blocked_by_boundary or step_count < 2 or remaining - step_count < min_rest:
            return None
        path_part = free_cells[:step_count]
        new_occ = occupied.copy()
        for rr, cc in path_part:
            new_occ[rr, cc] = True
        tail = _plan_path_recursive(
            path_part[-1],
            (dir_idx + 1) % 4,
            remaining - step_count,
            causes[1:],
            new_occ,
            rng,
        )
        if tail is None:
            return None
        tail_path, tail_obs = tail
        return path_part + tail_path, tail_obs

    max_step = min(len(free_cells) - 1, remaining - min_rest)
    if max_step < 2:
        return None
    choices = list(range(2, max_step + 1))
    rng.shuffle(choices)
    for step_count in choices:
        path_part = free_cells[:step_count]
        obstacle = free_cells[step_count]
        new_occ = occupied.copy()
        for rr, cc in path_part:
            new_occ[rr, cc] = True
        new_occ[obstacle[0], obstacle[1]] = True
        tail = _plan_path_recursive(
            path_part[-1],
            (dir_idx + 1) % 4,
            remaining - step_count,
            causes[1:],
            new_occ,
            rng,
        )
        if tail is None:
            continue
        tail_path, tail_obs = tail
        return path_part + tail_path, [obstacle] + tail_obs
    return None


def _try_place_module(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: np.random.Generator,
) -> dict | None:
    h, w = grid.shape
    side = str(rng.choice(np.array(["top", "right", "bottom", "left"])))
    num_colors = int(rng.integers(2, 4))

    if side in ("top", "bottom"):
        ih = int(rng.integers(max(5, 2 * num_colors - 1), 8))
        iw = int(rng.integers(4, 8))
        gap_offset = int(rng.integers(1, iw + 1))
    else:
        ih = int(rng.integers(4, 8))
        iw = int(rng.integers(max(5, 2 * num_colors - 1), 8))
        gap_offset = int(rng.integers(1, ih + 1))

    regions = _sample_regions(side, ih, iw, gap_offset, num_colors, rng)
    total_len = sum(int(region["area"]) for region in regions)
    profiles = _eligible_profiles(total_len)
    rng.shuffle(profiles)

    fh = ih + 2
    fw = iw + 2
    positions = [(r0, c0) for r0 in range(1, h - fh) for c0 in range(1, w - fw)]
    rng.shuffle(positions)

    for profile in profiles:
        for r0, c0 in positions:
            r1 = r0 + fh - 1
            c1 = c0 + fw - 1
            rs0 = max(0, r0 - 1)
            cs0 = max(0, c0 - 1)
            rs1 = min(h, r1 + 2)
            cs1 = min(w, c1 + 2)
            if np.any(reserved[rs0:rs1, cs0:cs1]):
                continue

            bbox, gap, dir_idx = _module_geometry(r0, c0, ih, iw, side, gap_offset)
            occupied = reserved.copy()
            occupied[r0 : r1 + 1, c0 : c1 + 1] = True
            plan = _plan_path_recursive(gap, dir_idx, total_len, profile, occupied, rng)
            if plan is None:
                continue
            path_cells, obstacle_cells = plan
            return {
                "bbox": bbox,
                "gap": gap,
                "dir_idx": dir_idx,
                "regions": regions,
                "path": path_cells,
                "obstacles": obstacle_cells,
                "segments": [(int(region["color"]), int(region["area"])) for region in regions],
            }
    return None


def _paint_module_input(grid: np.ndarray, module: dict) -> None:
    r0, c0, r1, c1 = module["bbox"]
    gap = module["gap"]
    for rr, cc in _perimeter_cells(r0, c0, r1, c1):
        if (rr, cc) != gap:
            grid[rr, cc] = GREY
    for region in module["regions"]:
        color = int(region["color"])
        for rr, cc in region["cells"]:
            grid[r0 + 1 + rr, c0 + 1 + cc] = color
    for rr, cc in module["obstacles"]:
        grid[rr, cc] = MAROON


def _expected_from_modules(grid: np.ndarray, modules: list[dict]) -> np.ndarray:
    out = grid.copy()
    for module in modules:
        idx = 0
        for color, length in module["segments"]:
            for _ in range(length):
                rr, cc = module["path"][idx]
                out[rr, cc] = int(color)
                idx += 1
    return out


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        h = int(rng.integers(*GRID_H_RANGE))
        w = int(rng.integers(*GRID_W_RANGE))
        grid = np.full((h, w), BLACK, dtype=int)
        reserved = np.zeros((h, w), dtype=bool)
        modules: list[dict] = []

        module_count = int(rng.choice(np.array(MODULE_COUNT_CHOICES), p=MODULE_COUNT_WEIGHTS))
        ok = True
        for _module_idx in range(module_count):
            module = None
            for _ in range(MAX_MODULE_ATTEMPTS):
                module = _try_place_module(grid, reserved, rng)
                if module is not None:
                    break
            if module is None:
                ok = False
                break

            _paint_module_input(grid, module)
            r0, c0, r1, c1 = module["bbox"]
            reserved[r0 : r1 + 1, c0 : c1 + 1] = True
            for rr, cc in module["path"]:
                reserved[rr, cc] = True
            for rr, cc in module["obstacles"]:
                reserved[rr, cc] = True
            modules.append(module)

        if not ok:
            continue

        if len(_detect_fences(grid)) != len(modules):
            continue

        expected = _expected_from_modules(grid, modules)
        produced = main(grid)
        if not np.array_equal(produced, expected):
            continue

        return grid

    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
