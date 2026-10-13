from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
OBSTACLE = 7
SEED_COLORS = (1, 2, 3, 4, 6, 8, 9)
DIRS = ("up", "down", "left", "right")


def _sign(x: int) -> int:
    return (x > 0) - (x < 0)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _opening_midpoint(open_cells: list[tuple[int, int]], side: str, alt_even_midpoint: bool) -> tuple[int, int]:
    ordered = sorted(open_cells)
    if side in ("up", "down"):
        ordered = sorted(open_cells, key=lambda rc: rc[1])
    else:
        ordered = sorted(open_cells, key=lambda rc: rc[0])
    idx = len(ordered) // 2
    if len(ordered) % 2 == 0:
        idx = idx if alt_even_midpoint else idx - 1
    return ordered[idx]


def _analyze_wall(
    grid: np.ndarray,
    comp: list[tuple[int, int]],
    *,
    alt_even_midpoint: bool = False,
) -> dict:
    r0, c0, r1, c1 = _bbox(comp)
    comp_set = set(comp)
    if r1 - r0 < 2 or c1 - c0 < 2:
        raise AssertionError("wall bbox too small")
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if not corners.issubset(comp_set):
        raise AssertionError("wall must keep all bbox corners")

    side_cells = {
        "up": [(r0, c) for c in range(c0 + 1, c1)],
        "down": [(r1, c) for c in range(c0 + 1, c1)],
        "left": [(r, c0) for r in range(r0 + 1, r1)],
        "right": [(r, c1) for r in range(r0 + 1, r1)],
    }
    open_side = None
    for side, cells in side_cells.items():
        if not cells:
            continue
        if any(cell in comp_set for cell in cells):
            continue
        if all(cell in comp_set for name, cells2 in side_cells.items() if name != side for cell in cells2):
            open_side = side
            break
    if open_side is None:
        raise AssertionError("could not determine wall opening")

    inner_r0, inner_c0 = r0 + 1, c0 + 1
    inner_r1, inner_c1 = r1 - 1, c1 - 1
    interior = [(r, c) for r in range(inner_r0, inner_r1 + 1) for c in range(inner_c0, inner_c1 + 1)]
    if not interior:
        raise AssertionError("wall interior empty")

    seed_colors = sorted({int(grid[r, c]) for r, c in interior if int(grid[r, c]) not in (BG, WALL)})
    if len(seed_colors) != 1:
        raise AssertionError("expected exactly one seed color inside each wall")
    seed_color = seed_colors[0]
    seed_cells = [(r, c) for r, c in interior if int(grid[r, c]) == seed_color]
    if len(seed_cells) != 1:
        raise AssertionError("expected exactly one seed pixel inside each wall")

    open_cells = side_cells[open_side]
    origin = _opening_midpoint(open_cells, open_side, alt_even_midpoint)
    return {
        "bbox": (r0, c0, r1, c1),
        "cells": comp_set,
        "interior": interior,
        "open_side": open_side,
        "open_cells": open_cells,
        "origin": origin,
        "seed_color": seed_color,
        "seed_pos": seed_cells[0],
        "opening_width": len(open_cells),
    }


def _in_wedge(info: dict, r: int, c: int) -> bool:
    r0, c0, r1, c1 = info["bbox"]
    open_side = info["open_side"]
    if open_side == "up":
        depth = r0 - r
        if depth < 0:
            return False
        left = c0 + 1 - depth
        right = c1 - 1 + depth
        return left <= c <= right
    if open_side == "down":
        depth = r - r1
        if depth < 0:
            return False
        left = c0 + 1 - depth
        right = c1 - 1 + depth
        return left <= c <= right
    if open_side == "left":
        depth = c0 - c
        if depth < 0:
            return False
        top = r0 + 1 - depth
        bottom = r1 - 1 + depth
        return top <= r <= bottom
    depth = c - c1
    if depth < 0:
        return False
    top = r0 + 1 - depth
    bottom = r1 - 1 + depth
    return top <= r <= bottom


def _shadow_cells(shape: tuple[int, int], origin: tuple[int, int], obstacle: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    dr = _sign(obstacle[0] - origin[0])
    dc = _sign(obstacle[1] - origin[1])
    if dr == 0 and dc == 0:
        return set()
    rr, cc = obstacle[0] + dr, obstacle[1] + dc
    out = set()
    while 0 <= rr < h and 0 <= cc < w:
        out.add((rr, cc))
        rr += dr
        cc += dc
    return out


def _transform(
    grid: np.ndarray,
    *,
    ignore_shadows: bool = False,
    fill_all_wedge: bool = False,
    alt_even_midpoint: bool = False,
    collect_meta: bool = False,
):
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    h, w = g.shape

    out = g.copy()
    wall_comps = _components_of_color(g, WALL)
    infos = sorted(
        [_analyze_wall(g, comp, alt_even_midpoint=alt_even_midpoint) for comp in wall_comps],
        key=lambda info: info["bbox"],
    )
    shadow_meta = []

    for info in infos:
        seed = info["seed_color"]
        for r, c in info["interior"]:
            if int(out[r, c]) == BG:
                out[r, c] = seed

        wedge = {(r, c) for r in range(h) for c in range(w) if _in_wedge(info, r, c)}
        obstacles = [(r, c) for r, c in wedge if int(g[r, c]) == OBSTACLE]
        blocked: set[tuple[int, int]] = set()
        if not ignore_shadows:
            for obs in obstacles:
                blocked |= _shadow_cells(g.shape, info["origin"], obs)

        filled = 0
        odd_black = 0
        blocked_even = 0
        for r, c in wedge:
            if int(g[r, c]) != BG:
                continue
            dist = max(abs(r - info["origin"][0]), abs(c - info["origin"][1]))
            even = dist % 2 == 0
            if (r, c) in blocked and even:
                blocked_even += 1
            if (r, c) in blocked:
                continue
            if not fill_all_wedge and not even:
                odd_black += 1
                continue
            out[r, c] = seed
            if (r, c) not in info["interior"]:
                filled += 1

        if collect_meta:
            shadow_meta.append(
                {
                    "direction": info["open_side"],
                    "opening_width": info["opening_width"],
                    "origin": info["origin"],
                    "seed_color": seed,
                    "obstacles": obstacles,
                    "wedge_size": len(wedge),
                    "exterior_fill": filled,
                    "odd_black": odd_black,
                    "blocked_even": blocked_even,
                }
            )

    if not collect_meta:
        return out
    return out, {"walls": shadow_meta, "wall_count": len(infos)}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each half-enclosed wall interior, then radiate alternating Chebyshev rings through the wall opening.

    Input structure:
    - The grid uses black background, grey one-pixel-thick U/C walls, one colored seed pixel inside each wall, and optional orange obstacle pixels outside.
    - Every grey wall is a single three-sided rectangular enclosure with one cardinal opening; its interior is the bbox interior rectangle.
    - Different walls have different seed colors, and obstacle pixels are separate from all walls.

    Rule:
    - For each grey wall, detect the unique missing side, fill all black interior cells with that wall's seed color, and keep the wall and seed unchanged.
    - Treat the opening-side midpoint as the Chebyshev origin, using the left/up midpoint when the opening width is even.
    - On the opening side only, consider the 45-degree-expanding trapezoid defined by the opening; black cells at even Chebyshev distance from the opening midpoint are filled with the seed color, while odd-distance cells stay black.
    - Orange obstacles stay orange and cast a shadow along the ray from the opening midpoint through the obstacle, preventing any fill on cells farther along that ray.

    Key insight:
    - The same seed color behaves differently in two contexts: it makes a solid interior flood fill inside the wall, but outside the opening it appears only on even-distance Chebyshev rings unless an obstacle shadow removes those ring cells.
    """
    return _transform(grid)


def _wall_from_spec(
    shape: tuple[int, int],
    direction: str,
    r0: int,
    c0: int,
    inner_h: int,
    inner_w: int,
) -> dict:
    h, w = shape
    r1 = r0 + inner_h + 1
    c1 = c0 + inner_w + 1
    if not (0 <= r0 < r1 < h and 0 <= c0 < c1 < w):
        raise AssertionError("wall bbox out of bounds")

    cells: set[tuple[int, int]] = set()
    if direction == "up":
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r1, c))
    elif direction == "down":
        for r in range(r0, r1 + 1):
            cells.add((r, c0))
            cells.add((r, c1))
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
    elif direction == "left":
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c1))
    else:
        for c in range(c0, c1 + 1):
            cells.add((r0, c))
            cells.add((r1, c))
        for r in range(r0, r1 + 1):
            cells.add((r, c0))

    info = _analyze_wall(
        _paint_wall_only(shape, cells, direction, r0, c0, r1, c1),
        list(cells),
    )
    return {
        "direction": direction,
        "bbox": (r0, c0, r1, c1),
        "wall_cells": cells,
        "interior": set(info["interior"]),
        "open_cells": info["open_cells"],
        "origin": info["origin"],
        "opening_width": info["opening_width"],
        "wedge": {(r, c) for r in range(h) for c in range(w) if _in_wedge(info, r, c)},
    }


def _paint_wall_only(
    shape: tuple[int, int],
    wall_cells: set[tuple[int, int]],
    direction: str,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for r, c in wall_cells:
        grid[r, c] = WALL
    if direction == "up":
        grid[r0 + 1, c0 + 1] = 1
    elif direction == "down":
        grid[r1 - 1, c0 + 1] = 1
    elif direction == "left":
        grid[r0 + 1, c0 + 1] = 1
    else:
        grid[r0 + 1, c1 - 1] = 1
    return grid


def _candidate_positions(
    shape: tuple[int, int],
    direction: str,
    inner_h: int,
    inner_w: int,
) -> list[tuple[int, int]]:
    h, w = shape
    box_h = inner_h + 2
    box_w = inner_w + 2
    out: list[tuple[int, int]] = []

    if direction == "up":
        row_lo = max(3, h // 2 - 1)
        row_hi = h - box_h - 1
        col_lo, col_hi = 1, w - box_w - 1
    elif direction == "down":
        row_lo = 1
        row_hi = min(h // 2 - box_h + 1, h - box_h - 4)
        col_lo, col_hi = 1, w - box_w - 1
    elif direction == "left":
        row_lo, row_hi = 1, h - box_h - 1
        col_lo = max(3, w // 2 - 1)
        col_hi = w - box_w - 1
    else:
        row_lo, row_hi = 1, h - box_h - 1
        col_lo = 1
        col_hi = min(w // 2 - box_w + 1, w - box_w - 4)

    if row_lo > row_hi or col_lo > col_hi:
        return out
    for r0 in range(row_lo, row_hi + 1):
        for c0 in range(col_lo, col_hi + 1):
            out.append((r0, c0))
    random.shuffle(out)
    return out


def _obstacle_styles(direction: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if direction == "up":
        return ((-1, 0), (-1, -1), (-1, 1))
    if direction == "down":
        return ((1, 0), (1, -1), (1, 1))
    if direction == "left":
        return ((0, -1), (-1, -1), (1, -1))
    return ((0, 1), (-1, 1), (1, 1))


def _candidate_obstacles(
    shape: tuple[int, int],
    wall: dict,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    origin = wall["origin"]
    wedge = wall["wedge"]
    out = []
    for step in _obstacle_styles(wall["direction"]):
        rr, cc = origin[0] + step[0], origin[1] + step[1]
        while 0 <= rr < h and 0 <= cc < w and (rr, cc) in wedge:
            if (rr, cc) not in occupied:
                shadow = _shadow_cells(shape, origin, (rr, cc))
                useful = any(
                    cell in wedge
                    and cell not in occupied
                    and max(abs(cell[0] - origin[0]), abs(cell[1] - origin[1])) % 2 == 0
                    for cell in shadow
                )
                if useful:
                    out.append((rr, cc))
            rr += step[0]
            cc += step[1]
    random.shuffle(out)
    dedup = []
    seen = set()
    for cell in out:
        if cell not in seen:
            seen.add(cell)
            dedup.append(cell)
    return dedup


def _build_attempt() -> np.ndarray | None:
    h = random.randint(16, 22)
    w = random.randint(16, 22)
    grid = np.full((h, w), BG, dtype=int)

    wall_count = random.choice((2, 2, 3))
    need_even = random.random() < 0.6
    directions = random.sample(DIRS, wall_count)
    seed_colors = random.sample(SEED_COLORS, wall_count)

    walls: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    used_wedges: set[tuple[int, int]] = set()

    for idx, direction in enumerate(directions):
        if direction in ("up", "down"):
            inner_h = random.randint(3, 4)
            inner_w = 4 if need_even and idx == 0 else random.randint(3, 4)
        else:
            inner_h = 4 if need_even and idx == 0 else random.randint(3, 4)
            inner_w = random.randint(3, 4)

        placed = None
        for r0, c0 in _candidate_positions((h, w), direction, inner_h, inner_w):
            wall = _wall_from_spec((h, w), direction, r0, c0, inner_h, inner_w)
            footprint = wall["wall_cells"] | wall["interior"]
            if footprint & occupied:
                continue
            if wall["wedge"] & occupied:
                continue
            if wall["wedge"] & used_wedges:
                continue
            placed = wall
            break
        if placed is None:
            return None

        seed_pos = random.choice(sorted(placed["interior"]))
        seed_color = seed_colors[idx]
        for r, c in placed["wall_cells"]:
            grid[r, c] = WALL
        grid[seed_pos] = seed_color
        placed["seed_pos"] = seed_pos
        placed["seed_color"] = seed_color
        walls.append(placed)
        occupied |= placed["wall_cells"] | placed["interior"]
        used_wedges |= placed["wedge"]

    obstacle_count = random.choice((1, 2))
    obstacle_cells: list[tuple[int, int]] = []
    wall_order = walls[:]
    random.shuffle(wall_order)
    for wall in wall_order:
        if len(obstacle_cells) >= obstacle_count:
            break
        candidates = _candidate_obstacles((h, w), wall, occupied | set(obstacle_cells))
        if not candidates:
            continue
        obstacle = candidates[0]
        obstacle_cells.append(obstacle)
        grid[obstacle] = OBSTACLE

    if len(obstacle_cells) != obstacle_count:
        return None

    return grid


def _valid_sample(grid: np.ndarray) -> bool:
    try:
        out, meta = _transform(grid, collect_meta=True)
        no_shadows = _transform(grid, ignore_shadows=True)
        no_parity = _transform(grid, fill_all_wedge=True)
        alt_mid = _transform(grid, alt_even_midpoint=True)
    except AssertionError:
        return False

    if meta["wall_count"] not in (2, 3):
        return False
    if np.array_equal(out, grid):
        return False
    if np.array_equal(out, no_shadows):
        return False
    if np.array_equal(out, no_parity):
        return False

    walls = meta["walls"]
    if sum(len(w["obstacles"]) for w in walls) not in (1, 2):
        return False
    if sum(w["exterior_fill"] for w in walls) < 12:
        return False
    if not any(w["blocked_even"] > 0 for w in walls):
        return False
    if not all(w["exterior_fill"] > 0 and w["odd_black"] > 0 for w in walls):
        return False

    if any(w["opening_width"] % 2 == 0 for w in walls) and np.array_equal(out, alt_mid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _build_attempt()
        if grid is not None and _valid_sample(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
