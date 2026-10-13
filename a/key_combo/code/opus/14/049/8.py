from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 7
OBSTACLE = 1
SEED = 2
MARKER = 5
EDGE = 8

GRID_MIN = 14
GRID_MAX = 22
SIDE_MIN = 6
SIDE_MAX = 10
MAX_OBSTACLES = 5
MAX_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _marker_spec(grid: np.ndarray) -> dict:
    coords = np.argwhere(grid == MARKER)
    assert coords.shape == (4, 2), "expected exactly four grey markers"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 == c1 - c0, "marker bbox must be a square"
    side = int(r1 - r0)
    assert side >= 2

    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    marker_set = {tuple(map(int, rc)) for rc in coords}
    present_corners = marker_set & corners
    assert len(present_corners) == 3, "expected three corner markers"
    missing = next(iter(corners - present_corners))

    inside_markers = list(marker_set - present_corners)
    assert len(inside_markers) == 1, "expected one interior marker"
    inner = inside_markers[0]
    assert r0 < inner[0] < r1 and c0 < inner[1] < c1, "interior marker must be inside the square"

    adjacent = [
        corner
        for corner in corners
        if corner != missing and (corner[0] == missing[0] or corner[1] == missing[1])
    ]
    assert len(adjacent) == 2
    directions = [(_sign(ar - missing[0]), _sign(ac - missing[1])) for ar, ac in adjacent]

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "missing": missing,
        "adjacent": adjacent,
        "directions": directions,
        "inner_marker": inner,
    }


def _spec_from_bbox_missing(
    r0: int, c0: int, r1: int, c1: int, missing: tuple[int, int]
) -> dict:
    side = int(r1 - r0)
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    adjacent = [
        corner
        for corner in corners
        if corner != missing and (corner[0] == missing[0] or corner[1] == missing[1])
    ]
    directions = [(_sign(ar - missing[0]), _sign(ac - missing[1])) for ar, ac in adjacent]
    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "missing": missing,
        "adjacent": adjacent,
        "directions": directions,
        "inner_marker": None,
    }


def _triangle_coords(spec: dict, r: int, c: int) -> tuple[int, int]:
    r0, c0, r1, c1 = spec["bbox"]
    mr, mc = spec["missing"]
    if (mr, mc) == (r0, c0):
        return r - r0, c - c0
    if (mr, mc) == (r0, c1):
        return r - r0, c1 - c
    if (mr, mc) == (r1, c0):
        return r1 - r, c - c0
    return r1 - r, c1 - c


def _in_triangle(spec: dict, r: int, c: int) -> bool:
    x, y = _triangle_coords(spec, r, c)
    n = spec["side"]
    return 0 <= x <= n and 0 <= y <= n and x + y <= n


def _on_boundary(spec: dict, r: int, c: int) -> bool:
    if not _in_triangle(spec, r, c):
        return False
    x, y = _triangle_coords(spec, r, c)
    n = spec["side"]
    return x == 0 or y == 0 or x + y == n


def _strict_interior(spec: dict, r: int, c: int) -> bool:
    x, y = _triangle_coords(spec, r, c)
    n = spec["side"]
    return 0 < x < n and 0 < y < n and x + y < n


def _boundary_mask(shape: tuple[int, int], spec: dict) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = spec["bbox"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _on_boundary(spec, r, c):
                mask[r, c] = True
    return mask


def _strict_interior_cells(spec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells = []
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _strict_interior(spec, r, c):
                cells.append((r, c))
    return cells


def _square_nontriangle_cells(spec: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = spec["bbox"]
    cells = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            if not _in_triangle(spec, r, c):
                cells.append((r, c))
    return cells


def _shadow_path(
    shape: tuple[int, int],
    boundary_mask: np.ndarray,
    start: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    dr, dc = direction
    rr, cc = start
    cells: list[tuple[int, int]] = []
    while True:
        rr += dr
        cc += dc
        if not (0 <= rr < h and 0 <= cc < w):
            return cells
        if boundary_mask[rr, cc]:
            return cells
        cells.append((rr, cc))


def _solve(grid: np.ndarray, *, return_details: bool = False):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    spec = _marker_spec(g)
    boundary_mask = _boundary_mask(g.shape, spec)
    strict_mask = np.zeros_like(g, dtype=bool)
    r0, c0, r1, c1 = spec["bbox"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if _strict_interior(spec, r, c):
                strict_mask[r, c] = True

    seeds = np.argwhere(g == SEED)
    assert seeds.shape == (1, 2), "expected exactly one red seed"
    seed = tuple(map(int, seeds[0]))
    assert strict_mask[seed], "seed must lie strictly inside the triangle"
    assert not boundary_mask[seed]

    passable = strict_mask & (g != MARKER) & (g != OBSTACLE) & ~boundary_mask
    assert passable[seed], "seed must be on a passable interior cell"

    visited = np.zeros_like(g, dtype=bool)
    visited[seed] = True
    queue = deque([seed])
    while queue:
        r, c = queue.popleft()
        for rr, cc in _neighbors4(r, c):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and passable[rr, cc] and not visited[rr, cc]:
                visited[rr, cc] = True
                queue.append((rr, cc))

    fill_before_shadow = visited & (g == BG)
    out = g.copy()
    out[boundary_mask & (out == BG)] = EDGE
    out[fill_before_shadow] = SEED

    shadowed = np.zeros_like(g, dtype=bool)
    for obstacle in [tuple(map(int, rc)) for rc in np.argwhere(g == OBSTACLE)]:
        for direction in spec["directions"]:
            for cell in _shadow_path(g.shape, boundary_mask, obstacle, direction):
                shadowed[cell] = True

    for r, c in np.argwhere(shadowed):
        rr, cc = int(r), int(c)
        if g[rr, cc] in (MARKER, OBSTACLE) or boundary_mask[rr, cc]:
            continue
        out[rr, cc] = BG

    if return_details:
        red_before_shadow = fill_before_shadow.copy()
        red_before_shadow[seed] = True
        return out, {
            "spec": spec,
            "boundary_mask": boundary_mask,
            "strict_mask": strict_mask,
            "seed": seed,
            "fill_before_shadow": fill_before_shadow,
            "shadowed": shadowed,
            "red_before_shadow": red_before_shadow,
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Draw the missing-corner right triangle, flood-fill its interior, then erase obstacle shadows cast along the right-angle legs.

    Input structure:
    - The background is orange, with exactly four grey markers whose bounding box is a square.
    - Three grey markers occupy square corners, one grey marker is inside the square, blue obstacle pixels sit on other square cells, and one red seed lies inside the triangle implied by the missing corner.

    Rule:
    - Find the square corner that lacks a grey marker; that missing corner is the right-angle vertex.
    - Draw the teal outline of the right triangle formed by that missing corner and its two adjacent square corners, then 4-neighbor flood-fill the triangle interior from the red seed while treating teal edges, grey markers, blue obstacles, and the grid boundary as blockers.
    - Each blue obstacle casts two axis-aligned shadow rays in the directions from the missing corner toward the adjacent corners; every traversed non-marker, non-obstacle cell is restored to orange until the ray hits a teal edge or exits the grid.

    Key insight:
    - The missing corner determines both the triangle geometry and the shadow directions, so the same geometric deduction controls where filling happens and which filled cells are later erased.
    """
    return _solve(grid, return_details=False)


def _primary_obstacle_candidates(
    spec: dict, boundary_mask: np.ndarray, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    candidates = []
    for cell in _strict_interior_cells(spec):
        paths = [_shadow_path(shape, boundary_mask, cell, d) for d in spec["directions"]]
        if min(len(path) for path in paths) >= 1 and len(set(paths[0]) | set(paths[1])) >= 3:
            candidates.append(cell)
    return candidates


def _sample_extra_obstacles(
    spec: dict,
    occupied: set[tuple[int, int]],
    total_target: int,
) -> list[tuple[int, int]]:
    extras: list[tuple[int, int]] = []
    interior = _strict_interior_cells(spec)
    exterior = _square_nontriangle_cells(spec)
    while len(extras) + len(occupied) < total_target and len(extras) < MAX_OBSTACLES - 1:
        pool = interior if random.random() < 0.7 else exterior
        pool = [cell for cell in pool if cell not in occupied and cell not in extras]
        if not pool:
            break
        cell = random.choice(pool)
        if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in extras):
            continue
        extras.append(cell)
    return extras


def _build_candidate() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    max_side = min(SIDE_MAX, h - 4, w - 4)
    if max_side < SIDE_MIN:
        return None

    side = random.randint(SIDE_MIN, max_side)
    r0 = random.randint(1, h - side - 2)
    c0 = random.randint(1, w - side - 2)
    r1, c1 = r0 + side, c0 + side

    corners = [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]
    missing = random.choice(corners)

    grid = np.full((h, w), BG, dtype=int)
    for corner in corners:
        if corner != missing:
            grid[corner] = MARKER

    spec = _spec_from_bbox_missing(r0, c0, r1, c1, missing)
    boundary_mask = _boundary_mask(grid.shape, spec)
    primary_candidates = _primary_obstacle_candidates(spec, boundary_mask, grid.shape)
    if not primary_candidates:
        return None

    primary = random.choice(primary_candidates)
    obstacles = [primary]
    target_count = random.randint(1, MAX_OBSTACLES)
    obstacles.extend(_sample_extra_obstacles(spec, set(obstacles), target_count))
    for r, c in obstacles:
        grid[r, c] = OBSTACLE

    shadow_set = set()
    for obstacle in obstacles:
        for direction in spec["directions"]:
            shadow_set.update(_shadow_path(grid.shape, boundary_mask, obstacle, direction))

    seed_shadowed = random.random() < 0.35
    interior_cells = [cell for cell in _strict_interior_cells(spec) if grid[cell] == BG]
    if seed_shadowed:
        seed_pool = [cell for cell in shadow_set if cell in interior_cells]
    else:
        seed_pool = [cell for cell in interior_cells if cell not in shadow_set]
    if not seed_pool:
        return None
    seed = random.choice(seed_pool)
    grid[seed] = SEED

    marker_inside_triangle = random.random() < 0.45
    outside_cells = [cell for cell in _square_nontriangle_cells(spec) if grid[cell] == BG and cell not in shadow_set]
    inside_cells = [cell for cell in _strict_interior_cells(spec) if grid[cell] == BG and cell not in shadow_set]
    if marker_inside_triangle and inside_cells:
        inner_marker = random.choice(inside_cells)
    elif outside_cells:
        inner_marker = random.choice(outside_cells)
    elif inside_cells:
        inner_marker = random.choice(inside_cells)
    else:
        return None
    grid[inner_marker] = MARKER

    try:
        out, details = _solve(grid, return_details=True)
    except Exception:
        return None

    if not np.array_equal(main(grid), out):
        return None
    if np.count_nonzero(out == EDGE) < side * 2:
        return None
    if int(details["fill_before_shadow"].sum()) < 8:
        return None

    shadow_effect = details["shadowed"] & details["red_before_shadow"]
    if int(shadow_effect.sum()) < 2:
        return None
    if seed_shadowed != bool(details["shadowed"][details["seed"]]):
        return None
    if not np.any((grid == OBSTACLE) & details["strict_mask"]):
        return None

    remaining_red = int(np.count_nonzero(out == SEED))
    if remaining_red < 5:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _build_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid missing-corner triangle shadow puzzle")
