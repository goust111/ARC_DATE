from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, in_bounds, neighbors4, paint_coords

Coord = tuple[int, int]

BLACK = Color.BLACK
BLUE = Color.BLUE
YELLOW = Color.YELLOW
TEAL = Color.TEAL
DIR4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))

PRIMARY_RECT_CHOICES = ((3, 4), (3, 5), (4, 4))
SECONDARY_RECT_CHOICES = ((3, 3), (3, 4), (4, 3))
OBSTACLE_SIZES = ((2, 2), (2, 3), (3, 2))
GEN_RETRIES = 400
FOREGROUND_RETRIES = 120
PLACEMENT_RETRIES = 120


def _detect_cross_centers(grid: np.ndarray) -> list[Coord]:
    centers: list[Coord] = []
    h, w = map(int, grid.shape)
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(grid[r, c]) != YELLOW:
                continue
            if all(int(grid[r + dr, c + dc]) == YELLOW for dr, dc in DIR4):
                centers.append((int(r), int(c)))
    return centers


def _cross_pixels(centers: list[Coord]) -> set[Coord]:
    pixels: set[Coord] = set()
    for r, c in centers:
        pixels.add((int(r), int(c)))
        for nr, nc in neighbors4(r, c):
            pixels.add((int(nr), int(nc)))
    return pixels


def _ray_trace(grid: np.ndarray, center: Coord, direction: Coord) -> tuple[list[tuple[int, int, int]], str]:
    h, w = map(int, grid.shape)
    r, c = center
    dr, dc = direction
    rr, cc = int(r + dr), int(c + dc)
    cells: list[tuple[int, int, int]] = []
    while 0 <= rr < h and 0 <= cc < w:
        value = int(grid[rr, cc])
        if value == BLUE:
            return cells, "wall"
        cells.append((int(rr), int(cc), value))
        rr += dr
        cc += dc
    return cells, "edge"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded plus-crosses in the yellow foreground, recolor them, and cast teal rays until blue walls.

    Input structure:
    - The grid uses black background, yellow foreground shapes, and blue solid-rectangle obstacles.
    - A cross center is any yellow cell whose four orthogonal neighbors are also yellow.
    - The yellow foreground may contain large solid blocks, thin lines, and multiple disconnected regions.

    Rule:
    - Detect all yellow cross centers from the original input.
    - Recolor every detected center and its four orthogonal neighbors to teal.
    - From each center, extend a teal ray up, down, left, and right until the next step would leave the grid or hit a blue obstacle.
    - Rays pass through yellow foreground without changing it, keep already-teal cells teal, and recolor only black background cells along their path.

    Key insight:
    - The ray sources are not explicitly marked: they emerge from the topology of the yellow foreground, so the solver must identify every interior plus-pattern before tracing the obstacle-truncated rays.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    centers = _detect_cross_centers(g)
    paint_coords(out, _cross_pixels(centers), TEAL)

    for center in centers:
        for direction in DIR4:
            for rr, cc, _value in _ray_trace(g, center, direction)[0]:
                if int(out[rr, cc]) == BLACK:
                    out[rr, cc] = TEAL
    return out


def _rect_cells(h: int, w: int) -> list[Coord]:
    return [(int(r), int(c)) for r in range(h) for c in range(w)]


def _line_cells(length: int, horizontal: bool) -> list[Coord]:
    if horizontal:
        return [(0, int(c)) for c in range(length)]
    return [(int(r), 0) for r in range(length)]


def _elbow_cells(h: int, w: int, thickness: int) -> list[Coord]:
    cells: set[Coord] = set()
    for r in range(h):
        for c in range(thickness):
            cells.add((int(r), int(c)))
    for r in range(thickness):
        for c in range(w):
            cells.add((int(r), int(c)))
    return sorted(cells)


def _translate_cells(cells: list[Coord], top: int, left: int) -> list[Coord]:
    return [(int(top + r), int(left + c)) for r, c in cells]


def _shape_size(cells: list[Coord]) -> tuple[int, int]:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    return int(max_r + 1), int(max_c + 1)


def _can_place_shape(grid: np.ndarray, cells: list[Coord], top: int, left: int, padding: int) -> bool:
    h, w = map(int, grid.shape)
    translated = set(_translate_cells(cells, top, left))
    for r, c in translated:
        if not in_bounds(grid.shape, r, c):
            return False
        if int(grid[r, c]) != BLACK:
            return False
        for rr in range(r - padding, r + padding + 1):
            for cc in range(c - padding, c + padding + 1):
                if not in_bounds(grid.shape, rr, cc):
                    continue
                if (rr, cc) in translated:
                    continue
                if int(grid[rr, cc]) == YELLOW:
                    return False
    return True


def _place_shape(grid: np.ndarray, cells: list[Coord], *, padding: int, margin: int, tries: int = PLACEMENT_RETRIES) -> bool:
    h, w = map(int, grid.shape)
    shape_h, shape_w = _shape_size(cells)
    if h - shape_h - margin < margin or w - shape_w - margin < margin:
        return False

    for _ in range(tries):
        top = random.randint(margin, h - shape_h - margin)
        left = random.randint(margin, w - shape_w - margin)
        if not _can_place_shape(grid, cells, top, left, padding):
            continue
        for r, c in _translate_cells(cells, top, left):
            grid[r, c] = YELLOW
        return True
    return False


def _sample_extra_shape() -> list[Coord]:
    chooser = random.random()
    if chooser < 0.4:
        return _rect_cells(*random.choice(SECONDARY_RECT_CHOICES))
    if chooser < 0.72:
        return _line_cells(random.randint(3, 6), horizontal=random.choice([True, False]))
    return _elbow_cells(
        h=random.randint(3, 5),
        w=random.randint(3, 5),
        thickness=random.choice([2, 3]),
    )


def _build_foreground(height: int, width: int) -> np.ndarray | None:
    for _ in range(FOREGROUND_RETRIES):
        grid = np.zeros((height, width), dtype=int)
        target_yellow = random.randint(31, 45)

        if not _place_shape(
            grid,
            _rect_cells(*random.choice(PRIMARY_RECT_CHOICES)),
            padding=1,
            margin=2,
        ):
            continue

        for _ in range(8):
            yellow_count = int(np.sum(grid == YELLOW))
            if yellow_count >= target_yellow:
                break
            shape = _sample_extra_shape()
            if yellow_count + len(shape) > 50:
                continue
            _place_shape(grid, shape, padding=1, margin=1)

        yellow_count = int(np.sum(grid == YELLOW))
        centers = _detect_cross_centers(grid)
        if not (30 <= yellow_count <= 50):
            continue
        if not (2 <= len(centers) <= 6):
            continue

        pass_through_rays = 0
        for center in centers:
            for direction in DIR4:
                ray_cells, _stop_kind = _ray_trace(grid, center, direction)
                if any(value == YELLOW for idx, (_r, _c, value) in enumerate(ray_cells) if idx >= 1):
                    pass_through_rays += 1
        if pass_through_rays == 0:
            continue
        return grid
    return None


def _obstacle_touching_yellow(yellow_grid: np.ndarray, cells: list[Coord]) -> bool:
    shape_set = set(cells)
    for r, c in cells:
        for rr in range(r - 1, r + 2):
            for cc in range(c - 1, c + 2):
                if not in_bounds(yellow_grid.shape, rr, cc):
                    continue
                if (rr, cc) in shape_set:
                    continue
                if int(yellow_grid[rr, cc]) == YELLOW:
                    return True
    return False


def _obstacle_candidates(yellow_grid: np.ndarray) -> list[dict]:
    height, width = map(int, yellow_grid.shape)
    centers = _detect_cross_centers(yellow_grid)
    rays: list[tuple[tuple[Coord, Coord], set[Coord]]] = []
    for center in centers:
        for direction in DIR4:
            ray_cells, _stop_kind = _ray_trace(yellow_grid, center, direction)
            rays.append(((center, direction), {(r, c) for r, c, _ in ray_cells}))

    candidates: list[dict] = []
    for obs_h, obs_w in OBSTACLE_SIZES:
        for top in range(height - obs_h + 1):
            for left in range(width - obs_w + 1):
                cells = [(int(top + r), int(left + c)) for r in range(obs_h) for c in range(obs_w)]
                if any(int(yellow_grid[r, c]) != BLACK for r, c in cells):
                    continue
                if _obstacle_touching_yellow(yellow_grid, cells):
                    continue
                hits = []
                cell_set = set(cells)
                for ray_id, ray_set in rays:
                    if ray_set & cell_set:
                        hits.append(ray_id)
                candidates.append(
                    {
                        "top": int(top),
                        "left": int(left),
                        "h": int(obs_h),
                        "w": int(obs_w),
                        "cells": cells,
                        "hits": hits,
                    }
                )
    return candidates


def _obstacles_compatible(candidate: dict, chosen: list[dict]) -> bool:
    cell_set = set(candidate["cells"])
    for other in chosen:
        other_set = set(other["cells"])
        for r, c in cell_set:
            for rr, cc in other_set:
                if abs(int(r) - int(rr)) <= 1 and abs(int(c) - int(cc)) <= 1:
                    return False
    return True


def _choose_obstacles(yellow_grid: np.ndarray) -> list[dict] | None:
    candidates = _obstacle_candidates(yellow_grid)
    blocking = [cand for cand in candidates if cand["hits"]]
    neutral = [cand for cand in candidates if not cand["hits"]]
    if not blocking:
        return None

    n_obstacles = random.randint(2, 4)
    chosen: list[dict] = []
    used_rays: set[tuple[Coord, Coord]] = set()

    blocking_pool = blocking[:]
    random.shuffle(blocking_pool)
    preferred_blockers = 1
    if n_obstacles >= 3 and len(blocking_pool) >= 2 and random.random() < 0.6:
        preferred_blockers = 2

    for candidate in blocking_pool:
        if len(chosen) >= preferred_blockers:
            break
        if not _obstacles_compatible(candidate, chosen):
            continue
        if used_rays and all(ray in used_rays for ray in candidate["hits"]):
            continue
        chosen.append(candidate)
        used_rays.update(candidate["hits"])

    if not chosen:
        return None

    pool = candidates[:]
    random.shuffle(pool)
    for candidate in pool:
        if len(chosen) >= n_obstacles:
            break
        if candidate in chosen:
            continue
        if not _obstacles_compatible(candidate, chosen):
            continue
        chosen.append(candidate)

    if len(chosen) < 2:
        return None
    return chosen


def _scene_metrics(inp: np.ndarray) -> dict:
    centers = _detect_cross_centers(inp)
    out = main(inp)

    blocked_rays = 0
    edge_rays = 0
    pass_through_yellow = 0
    painted_ray_count = 0
    for center in centers:
        for direction in DIR4:
            ray_cells, stop_kind = _ray_trace(inp, center, direction)
            if stop_kind == "wall":
                blocked_rays += 1
            else:
                edge_rays += 1
            if any(value == YELLOW for idx, (_r, _c, value) in enumerate(ray_cells) if idx >= 1):
                pass_through_yellow += 1
            if any(int(inp[r, c]) == BLACK and int(out[r, c]) == TEAL for r, c, _value in ray_cells):
                painted_ray_count += 1

    return {
        "centers": centers,
        "out": out,
        "blocked_rays": int(blocked_rays),
        "edge_rays": int(edge_rays),
        "pass_through_yellow": int(pass_through_yellow),
        "painted_ray_count": int(painted_ray_count),
        "painted_cells": int(np.sum((inp == BLACK) & (out == TEAL))),
        "recolored_cross_pixels": int(np.sum((inp == YELLOW) & (out == TEAL))),
        "remaining_yellow": int(np.sum((inp == YELLOW) & (out == YELLOW))),
    }


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        height = random.randint(14, 19)
        width = random.randint(14, 19)
        yellow_grid = _build_foreground(height, width)
        if yellow_grid is None:
            continue

        obstacles = _choose_obstacles(yellow_grid)
        if obstacles is None:
            continue

        inp = yellow_grid.copy()
        for obstacle in obstacles:
            inp[
                int(obstacle["top"]) : int(obstacle["top"] + obstacle["h"]),
                int(obstacle["left"]) : int(obstacle["left"] + obstacle["w"]),
            ] = BLUE

        metrics = _scene_metrics(inp)
        if not (2 <= len(metrics["centers"]) <= 6):
            continue
        if metrics["blocked_rays"] < 1:
            continue
        if metrics["edge_rays"] < 1:
            continue
        if metrics["pass_through_yellow"] < 1:
            continue
        if metrics["painted_ray_count"] < 2:
            continue
        if metrics["painted_cells"] < 8:
            continue
        if metrics["recolored_cross_pixels"] < 8:
            continue
        if metrics["remaining_yellow"] < 6:
            continue
        return inp

    raise RuntimeError("failed to generate a valid cross-ray ARC input")
