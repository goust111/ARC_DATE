import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _obstacle_components(grid: np.ndarray) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    index: dict[tuple[int, int], int] = {}
    for r in range(h):
        for c in range(w):
            if grid[r, c] != GREY or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                index[(rr, cc)] = len(comps)
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds((h, w), nr, nc) and not seen[nr, nc] and grid[nr, nc] == GREY:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps, index


def _marker_rows(grid: np.ndarray) -> list[tuple[int, int]]:
    rows = []
    for r in range(grid.shape[0]):
        color = int(grid[r, 0])
        if color not in (BG, GREY):
            rows.append((r, color))
    return rows


def _cast_shadow(shadow: np.ndarray, obstacle: list[tuple[int, int]]) -> None:
    h, w = shadow.shape
    for r, c in obstacle:
        rr, cc = r, c
        while 0 <= rr < h and 0 <= cc < w:
            shadow[rr, cc] = True
            rr += 1
            cc += 1


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    shadow = np.zeros((h, w), dtype=bool)
    obstacles, obstacle_index = _obstacle_components(g)

    for start_r, color in sorted(_marker_rows(g)):
        step = 0
        while True:
            r = start_r + step
            c = step
            if not _in_bounds((h, w), r, c):
                break
            if shadow[r, c]:
                break
            if g[r, c] == GREY:
                _cast_shadow(shadow, obstacles[obstacle_index[(r, c)]])
                break
            if out[r, c] == BG:
                out[r, c] = color
            step += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend left-edge marker rays along a down-right shear and let grey obstacles cast matching diagonal shadows.

    Input structure:
    - The background is black, and column 0 contains 3-5 isolated colored marker pixels on distinct rows.
    - Elsewhere the only foreground objects are 2-4 grey 1x2 or 2x1 obstacle rectangles that do not touch column 0.
    - Each marker row defines one diagonal band because the intended ray advances one row down for every column right.

    Rule:
    - Process the markers from top to bottom. From marker (r, 0), follow the cells (r, 0), (r+1, 1), (r+2, 2), ... and color background cells with that marker's color.
    - Stop a ray when it would leave the grid, when it reaches a grey obstacle cell, or when it reaches an already-shadowed cell.
    - When a ray hits any cell of a grey obstacle rectangle, that whole obstacle casts a shadow: every obstacle cell keeps extending down-right by repeated (+1, +1) steps, and later rays may not color any shadowed cell.
    - Obstacles stay grey, and previously colored cells are never overwritten.

    Key insight:
    - The markers do not grow horizontally; they trace parallel diagonal shear paths, and an obstacle converts from a local blocker into a diagonal no-paint band for subsequent rays.
    """
    return _solve(grid)


def _row_spacing_ok(rows: list[int]) -> bool:
    return all(b - a >= 2 for a, b in zip(rows, rows[1:]))


def _choose_marker_rows(rng: random.Random, h: int, count: int) -> list[int] | None:
    candidates = list(range(0, h - 4))
    for _ in range(200):
        rows = sorted(rng.sample(candidates, count))
        if _row_spacing_ok(rows):
            if min(h - 1 - r for r in rows) >= 4 and rows[-1] - rows[0] >= 4:
                return rows
    return None


def _ray_cells(h: int, w: int, start_r: int) -> list[tuple[int, int]]:
    cells = []
    step = 0
    while True:
        r = start_r + step
        c = step
        if not _in_bounds((h, w), r, c):
            return cells
        cells.append((r, c))
        step += 1


def _cell_on_any_ray(marker_rows: list[int], h: int, w: int, r: int, c: int) -> bool:
    if c < 0 or c >= w:
        return False
    start_r = r - c
    return start_r in marker_rows and start_r >= 0 and (r, c) in set(_ray_cells(h, w, start_r))


def _touches_existing(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in candidate:
        if (r, c) in occupied:
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                if (r + dr, c + dc) in occupied:
                    return True
    return False


def _candidate_obstacles_for_hit(h: int, w: int, hit_r: int, hit_c: int, orientation: str) -> list[list[tuple[int, int]]]:
    if orientation == "h":
        return [
            [(hit_r, hit_c), (hit_r, hit_c + 1)],
            [(hit_r, hit_c - 1), (hit_r, hit_c)],
        ]
    return [
        [(hit_r, hit_c), (hit_r + 1, hit_c)],
        [(hit_r - 1, hit_c), (hit_r, hit_c)],
    ]


def _valid_obstacle(candidate: list[tuple[int, int]], h: int, w: int) -> bool:
    rows = {r for r, _ in candidate}
    cols = {c for _, c in candidate}
    if any(not _in_bounds((h, w), r, c) for r, c in candidate):
        return False
    if any(c == 0 for _, c in candidate):
        return False
    return (len(rows), len(cols)) in {(1, 2), (2, 1)}


def _place_blocking_obstacle(
    rng: random.Random,
    grid: np.ndarray,
    start_r: int,
    orientation: str,
    occupied: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    ray = _ray_cells(h, w, start_r)
    if len(ray) < 5:
        return False
    steps = list(range(2, len(ray) - 1))
    rng.shuffle(steps)
    for step in steps:
        hit_r, hit_c = ray[step]
        variants = _candidate_obstacles_for_hit(h, w, hit_r, hit_c, orientation)
        rng.shuffle(variants)
        for candidate in variants:
            if not _valid_obstacle(candidate, h, w):
                continue
            if _touches_existing(candidate, occupied):
                continue
            if any(grid[r, c] != BG for r, c in candidate):
                continue
            for r, c in candidate:
                grid[r, c] = GREY
                occupied.add((r, c))
            return True
    return False


def _place_distractor_obstacle(
    rng: random.Random,
    grid: np.ndarray,
    orientation: str,
    occupied: set[tuple[int, int]],
    marker_rows: list[int],
) -> bool:
    h, w = grid.shape
    cells = [(r, c) for r in range(h) for c in range(1, w)]
    rng.shuffle(cells)
    for r, c in cells:
        if orientation == "h":
            candidate = [(r, c), (r, c + 1)]
        else:
            candidate = [(r, c), (r + 1, c)]
        if not _valid_obstacle(candidate, h, w):
            continue
        if _touches_existing(candidate, occupied):
            continue
        if any(_cell_on_any_ray(marker_rows, h, w, rr, cc) for rr, cc in candidate):
            continue
        if any(grid[rr, cc] != BG for rr, cc in candidate):
            continue
        for rr, cc in candidate:
            grid[rr, cc] = GREY
            occupied.add((rr, cc))
        return True
    return False


def _obstacle_shapes(grid: np.ndarray) -> list[tuple[int, int]]:
    comps, _ = _obstacle_components(grid)
    shapes = []
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        shapes.append((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1))
    return shapes


def _analyze_instance(grid: np.ndarray) -> dict:
    h, w = grid.shape
    markers = _marker_rows(grid)
    out = _solve(grid)
    blocked = 0
    unblocked = 0
    ray_lengths: list[int] = []
    painted_lengths: list[int] = []
    for start_r, color in markers:
        path = _ray_cells(h, w, start_r)
        ray_lengths.append(len(path))
        painted = 0
        hit = False
        for r, c in path:
            if grid[r, c] == GREY:
                hit = True
                break
            if out[r, c] == color:
                painted += 1
            else:
                break
        painted_lengths.append(painted)
        if hit:
            blocked += 1
        else:
            unblocked += 1
    return {
        "marker_rows": [r for r, _ in markers],
        "marker_count": len(markers),
        "obstacle_shapes": _obstacle_shapes(grid),
        "blocked": blocked,
        "unblocked": unblocked,
        "painted_lengths": painted_lengths,
        "new_cells": int(np.count_nonzero((out != grid) & (grid == BG))),
    }


def _validate_generated(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2:
        return False
    h, w = grid.shape
    if max(h, w) > 30:
        return False
    if np.any(grid[:, 0] == GREY):
        return False

    markers = _marker_rows(grid)
    rows = [r for r, _ in markers]
    colors = [c for _, c in markers]
    if not (3 <= len(markers) <= 5):
        return False
    if len(colors) != len(set(colors)):
        return False
    if not _row_spacing_ok(rows):
        return False
    if any(grid[r, c] != BG for r in range(h) for c in range(w) if c != 0 and grid[r, c] not in (BG, GREY)):
        return False

    comps, _ = _obstacle_components(grid)
    if not (2 <= len(comps) <= 4):
        return False
    shapes = []
    for comp in comps:
        rs = [r for r, _ in comp]
        cs = [c for _, c in comp]
        bbox_shape = (max(rs) - min(rs) + 1, max(cs) - min(cs) + 1)
        if bbox_shape not in {(1, 2), (2, 1)} or len(comp) != 2:
            return False
        if min(cs) == 0:
            return False
        shapes.append(bbox_shape)
    if (1, 2) not in shapes or (2, 1) not in shapes:
        return False

    info = _analyze_instance(grid)
    if info["blocked"] < 1 or info["unblocked"] < 1:
        return False
    if info["new_cells"] < 8:
        return False
    if max(info["painted_lengths"]) < 4:
        return False
    if min(info["painted_lengths"]) < 1:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(12, 17)
    w = rng.randint(12, 20)
    max_markers = min(5, (h - 3) // 2)
    if max_markers < 3:
        return None
    marker_count = rng.randint(3, max_markers)
    marker_rows = _choose_marker_rows(rng, h, marker_count)
    if marker_rows is None:
        return None

    grid = np.zeros((h, w), dtype=int)
    marker_colors = rng.sample(MARKER_COLORS, marker_count)
    for r, color in zip(marker_rows, marker_colors):
        grid[r, 0] = color

    occupied = {(r, 0) for r in marker_rows}
    total_obstacles = rng.randint(2, 4)
    blocked_count = rng.randint(1, min(marker_count - 1, total_obstacles))
    blocked_rows = rng.sample(marker_rows, blocked_count)

    required_orientations = ["h", "v"]
    while len(required_orientations) < total_obstacles:
        required_orientations.append(rng.choice(["h", "v"]))
    rng.shuffle(required_orientations)

    for start_r in blocked_rows:
        orientation = required_orientations.pop()
        if not _place_blocking_obstacle(rng, grid, start_r, orientation, occupied):
            return None

    while required_orientations:
        orientation = required_orientations.pop()
        if not _place_distractor_obstacle(rng, grid, orientation, occupied, marker_rows):
            return None

    return grid if _validate_generated(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid diagonal-ray instance")
