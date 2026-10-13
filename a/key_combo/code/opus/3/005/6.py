from __future__ import annotations

import random

import numpy as np


BG = 0
MARKER = 1
WALL = 5
MAROON = 9

CARDINAL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ALL_PROJECTILE_DIRS = DIAGONAL_DIRS + CARDINAL_DIRS
SPRITE_COLORS = (2, 3, 4, 7)

MAX_GENERATION_ATTEMPTS = 400


Coord = tuple[int, int]


def _background(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _neighbors8(r: int, c: int) -> list[Coord]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _components_of_color(grid: np.ndarray, color: int, connectivity: int = 8) -> list[list[Coord]]:
    h, w = grid.shape
    coords = {(int(r), int(c)) for r, c in np.argwhere(grid == int(color))}
    comps: list[list[Coord]] = []
    nbrs = _neighbors4 if connectivity == 4 else _neighbors8
    while coords:
        start = coords.pop()
        stack = [start]
        comp = [start]
        while stack:
            r, c = stack.pop()
            for nr, nc in nbrs(r, c):
                if (nr, nc) in coords:
                    coords.remove((nr, nc))
                    stack.append((nr, nc))
                    comp.append((nr, nc))
        comps.append(comp)
    return comps


def _edge_direction(pos: Coord, shape: tuple[int, int]) -> Coord:
    r, c = pos
    h, w = shape
    dirs: list[Coord] = []
    if r == 0:
        dirs.append((1, 0))
    if r == h - 1:
        dirs.append((-1, 0))
    if c == 0:
        dirs.append((0, 1))
    if c == w - 1:
        dirs.append((0, -1))
    assert len(dirs) == 1, "expected one edge marker"
    return dirs[0]


def _triangle_apex_and_dir(comp: list[Coord]) -> tuple[Coord, Coord] | None:
    if len(comp) != 3:
        return None
    cells = set(comp)
    deg4 = {(r, c): sum((nr, nc) in cells for nr, nc in _neighbors4(r, c)) for r, c in comp}
    deg8 = {(r, c): sum((nr, nc) in cells for nr, nc in _neighbors8(r, c)) for r, c in comp}

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms: list[Coord] = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = arms[0][0] + arms[1][0]
        dc = arms[0][1] + arms[1][1]
        fire = (0 if dr == 0 else -1 if dr > 0 else 1, 0 if dc == 0 else -1 if dc > 0 else 1)
        return corner, fire

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    dr = -((others[0][0] - ar) + (others[1][0] - ar)) // 2
    dc = -((others[0][1] - ac) + (others[1][1] - ac)) // 2
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (int(dr), int(dc))


def _find_sprites(grid: np.ndarray) -> list[dict]:
    bg = _background(grid)
    colors = [int(c) for c in np.unique(grid) if int(c) not in {bg, MARKER, WALL, MAROON}]
    sprites: list[dict] = []
    for color in colors:
        for comp in _components_of_color(grid, color, connectivity=8):
            ad = _triangle_apex_and_dir(comp)
            assert ad is not None, "expected all non-wall foreground objects to be 3-pixel triangles"
            apex, direction = ad
            sprites.append(
                {
                    "color": color,
                    "coords": tuple(sorted(comp)),
                    "apex": apex,
                    "dir": direction,
                }
            )
    return sprites


def _erase_sprites(grid: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = BG
    return out


def _settle_projectiles(grid: np.ndarray, sprites: list[dict] | None = None) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    sprites = _find_sprites(g) if sprites is None else sprites
    out = _erase_sprites(g, sprites)

    new_walls: set[Coord] = set()
    maroons: set[Coord] = set()
    closed_gaps: list[Coord] = []
    wall_hits: list[Coord] = []
    sprite_hits: list[Coord] = []
    projectile_info: list[dict] = []

    for sprite in sprites:
        ar, ac = sprite["apex"]
        dr, dc = sprite["dir"]
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        info = {"dest": (rr, cc), "in_bounds": _in_bounds(g.shape, rr, cc), "kind": "oob"}
        if _in_bounds(g.shape, rr, cc):
            cell = int(g[rr, cc])
            if cell == BG:
                new_walls.add((rr, cc))
                info["kind"] = "new_wall"
                # In this task all intentional background landings are wall gaps or open space.
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(g.shape, nr, nc) and g[nr, nc] == WALL:
                        closed_gaps.append((rr, cc))
                        break
            else:
                maroons.add((rr, cc))
                info["kind"] = "maroon"
                if cell == WALL:
                    wall_hits.append((rr, cc))
                else:
                    sprite_hits.append((rr, cc))
        projectile_info.append(info)

    for r, c in new_walls:
        out[r, c] = WALL
    for r, c in maroons:
        out[r, c] = MAROON

    return out, {
        "sprites": sprites,
        "projectiles": projectile_info,
        "new_walls": new_walls,
        "maroons": maroons,
        "closed_gaps": closed_gaps,
        "wall_hits": wall_hits,
        "sprite_hits": sprite_hits,
    }


def _trace_branches(env: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(env, dtype=int)
    marker_cells = np.argwhere(g == MARKER)
    assert len(marker_cells) == 1, "expected a single blue edge marker"
    start = tuple(map(int, marker_cells[0]))
    dr, dc = _edge_direction(start, g.shape)

    h, w = g.shape
    out = g.copy()
    stack: list[Coord] = [start]
    seen: set[Coord] = set()
    wall_hits = 0
    openings_used = 0

    while stack:
        r, c = stack.pop()
        if (r, c) in seen:
            continue
        seen.add((r, c))
        out[r, c] = MARKER
        cr, cc = r, c

        while True:
            nr, nc = cr + dr, cc + dc
            if not _in_bounds(g.shape, nr, nc):
                break
            nxt = int(g[nr, nc])
            if nxt == MAROON:
                break
            if nxt != WALL:
                out[nr, nc] = MARKER
                cr, cc = nr, nc
                continue

            wall_hits += 1
            if dr != 0:
                wall_r = nr
                lo = hi = nc
                while lo - 1 >= 0 and g[wall_r, lo - 1] == WALL:
                    lo -= 1
                while hi + 1 < w and g[wall_r, hi + 1] == WALL:
                    hi += 1

                for x in range(max(0, lo - 1), min(w - 1, hi + 1) + 1):
                    if g[cr, x] != WALL and g[cr, x] != MAROON:
                        out[cr, x] = MARKER
                for x in (lo - 1, hi + 1):
                    if _in_bounds(g.shape, wall_r, x) and g[wall_r, x] not in {WALL, MAROON}:
                        out[wall_r, x] = MARKER
                        stack.append((wall_r, x))
                        openings_used += 1
            else:
                wall_c = nc
                lo = hi = nr
                while lo - 1 >= 0 and g[lo - 1, wall_c] == WALL:
                    lo -= 1
                while hi + 1 < h and g[hi + 1, wall_c] == WALL:
                    hi += 1

                for y in range(max(0, lo - 1), min(h - 1, hi + 1) + 1):
                    if g[y, cc] != WALL and g[y, cc] != MAROON:
                        out[y, cc] = MARKER
                for y in (lo - 1, hi + 1):
                    if _in_bounds(g.shape, y, wall_c) and g[y, wall_c] not in {WALL, MAROON}:
                        out[y, wall_c] = MARKER
                        stack.append((y, wall_c))
                        openings_used += 1
            break

    return out, {"seen_states": len(seen), "wall_hits": wall_hits, "openings_used": openings_used}


def _solve_details(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    sprites = _find_sprites(g)
    projectile_env, projectile_info = _settle_projectiles(g, sprites)
    baseline_env = _erase_sprites(g, sprites)
    out, branch_info = _trace_branches(projectile_env)
    baseline_out, baseline_branch_info = _trace_branches(baseline_env)
    return {
        "sprites": sprites,
        "projectile_env": projectile_env,
        "projectile_info": projectile_info,
        "baseline_env": baseline_env,
        "output": out,
        "branch_info": branch_info,
        "baseline_output": baseline_out,
        "baseline_branch_info": baseline_branch_info,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve triangle projectiles first, then trace the blue edge ray through the wall layout they modify.

    Input structure:
    - Black background with one blue single-cell edge marker on exactly one border cell.
    - Two or three parallel grey wall lines perpendicular to the marker's inward direction; each line is one cell thick and contains one or two single-cell gaps.
    - Two to four colored 3-pixel triangle sprites, each either an axis-aligned L or a diagonal V whose orientation encodes a firing direction.

    Rule:
    - For every triangle, find its apex and fire a projectile exactly 5 cells along its encoded direction.
    - A projectile landing on background creates a new grey wall pixel; a projectile landing on any existing non-background pixel recolors that landing cell to maroon.
    - After all projectiles settle, erase the original triangle sprites but keep the new grey pixels and maroon marks.
    - Then advance blue rays inward from the edge marker: fill background cells straight ahead, stop at maroon obstacles, and when the next cell is grey, paint the near side of that contiguous wall run plus its adjacent openings and continue inward from each in-bounds opening.

    Key insight:
    - The ray phase does not run on the original walls: triangle projectiles first rewrite the wall topology by sealing gaps or inserting maroon blockers, and only that modified environment determines the later branching path.
    """
    return _solve_details(np.asarray(grid, dtype=int))["output"]


def _sprite_cells(apex: Coord, direction: Coord) -> tuple[Coord, ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _halo(cells: tuple[Coord, ...], shape: tuple[int, int]) -> set[Coord]:
    h, w = shape
    out: set[Coord] = set()
    for r, c in cells:
        for rr in range(max(0, r - 1), min(h - 1, r + 1) + 1):
            for cc in range(max(0, c - 1), min(w - 1, c + 1) + 1):
                out.add((rr, cc))
    return out


def _candidate_sprites_for_dest(grid: np.ndarray, dest: Coord) -> list[dict]:
    h, w = grid.shape
    candidates: list[dict] = []
    for dr, dc in ALL_PROJECTILE_DIRS:
        ar = dest[0] - 5 * dr
        ac = dest[1] - 5 * dc
        cells = _sprite_cells((ar, ac), (dr, dc))
        if all(0 <= r < h and 0 <= c < w for r, c in cells):
            if all(grid[r, c] == BG for r, c in cells):
                candidates.append({"apex": (ar, ac), "dir": (dr, dc), "cells": cells, "dest": dest})
    random.shuffle(candidates)
    return candidates


def _draw_horizontal_wall(
    grid: np.ndarray,
    row: int,
    start: int,
    end: int,
    gaps: tuple[int, ...],
) -> None:
    grid[row, start : end + 1] = WALL
    for gap in gaps:
        grid[row, gap] = BG


def _build_top_canonical() -> tuple[np.ndarray, dict] | None:
    wall_count = random.choices([2, 3], weights=[2, 5])[0]
    if wall_count == 2:
        increment_choices = [(1, 1), (1, 2), (2, 1), (2, 2)]
    else:
        increment_choices = [(1, 1, 1), (1, 1, 2), (1, 2, 1), (2, 1, 1)]

    g0 = random.choice([2, 3])
    incs = list(random.choice(increment_choices))
    gaps = [g0]
    for inc in incs:
        gaps.append(gaps[-1] + inc)

    width_min = max(13, gaps[-1] + 5)
    width_max = max(width_min, 16)
    width = random.randint(width_min, width_max)

    row1 = random.randint(3, 4)
    rows = [row1]
    for _ in range(wall_count - 1):
        rows.append(rows[-1] + random.randint(3, 4))
    height_min = rows[-1] + 4
    height_max = max(height_min, min(17, height_min + 2))
    height = random.randint(height_min, height_max)

    grid = np.zeros((height, width), dtype=int)
    grid[0, g0] = MARKER

    spans: list[tuple[int, int, tuple[int, ...]]] = []
    for i, row in enumerate(rows):
        if i == 0:
            start = max(0, g0 - 2)
            end = gaps[1] + 2
            wall_gaps = (gaps[0], gaps[1])
        else:
            start = max(0, g0 - 1)
            end = gaps[i + 1] + 2
            wall_gaps = (gaps[i + 1],)
        if end - start + 1 < 4 or end - start + 1 > 8 or end >= width:
            return None
        _draw_horizontal_wall(grid, row, start, end, wall_gaps)
        spans.append((start, end, wall_gaps))

    effects = [
        {"name": "close_first_gap", "dest": (rows[0], gaps[0])},
        {"name": "close_last_gap", "dest": (rows[-1], gaps[-1])},
    ]

    extra_pool = [{"name": "block_active_wall", "dest": (rows[min(1, wall_count - 1)], gaps[1])}]

    start, end, wall_gaps = spans[-1]
    wall_cells = [c for c in range(start, end + 1) if c not in wall_gaps and c != gaps[-2]]
    if wall_cells:
        extra_pool.append({"name": "mark_far_wall", "dest": (rows[-1], random.choice(wall_cells))})

    extra_r = min(height - 2, rows[-1] + 1)
    extra_c = min(width - 2, gaps[-1] + 3)
    if grid[extra_r, extra_c] == BG:
        extra_pool.append({"name": "drop_extra_wall", "dest": (extra_r, extra_c)})

    target_count = random.choices([2, 3, 4], weights=[3, 5, 2])[0]
    random.shuffle(extra_pool)
    for effect in extra_pool:
        if len(effects) >= target_count:
            break
        if all(effect["dest"] != existing["dest"] for existing in effects):
            effects.append(effect)
    if len(effects) != target_count:
        return None

    colors = random.sample(SPRITE_COLORS, len(effects))
    placements: list[dict] = []
    blocked_halo: set[Coord] = set()

    for effect, color in zip(effects, colors):
        candidates = _candidate_sprites_for_dest(grid, effect["dest"])
        chosen = None
        for cand in candidates:
            if any(cell in blocked_halo for cell in cand["cells"]):
                continue
            chosen = cand.copy()
            chosen["color"] = color
            chosen["effect"] = effect["name"]
            placements.append(chosen)
            blocked_halo |= _halo(cand["cells"], grid.shape)
            break
        if chosen is None:
            return None

    for placement in placements:
        for r, c in placement["cells"]:
            grid[r, c] = placement["color"]

    return grid, {
        "rows": rows,
        "gaps": gaps,
        "spans": spans,
        "placements": placements,
        "effects": effects,
    }


def _random_transform(grid: np.ndarray) -> np.ndarray:
    out = grid
    if random.random() < 0.5:
        out = np.fliplr(out)
    k = random.randrange(4)
    if k:
        out = np.rot90(out, k)
    return np.asarray(out, dtype=int).copy()


def _validate_generated(grid: np.ndarray) -> bool:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2 or max(g.shape) > 30:
        return False
    if np.count_nonzero(g == MARKER) != 1:
        return False

    try:
        details = _solve_details(g)
    except Exception:
        return False

    sprites = details["sprites"]
    if not (2 <= len(sprites) <= 4):
        return False
    if len({sprite["color"] for sprite in sprites}) != len(sprites):
        return False

    projectile_info = details["projectile_info"]
    if not projectile_info["closed_gaps"]:
        return False
    if not projectile_info["new_walls"]:
        return False

    baseline_out = details["baseline_output"]
    full_out = details["output"]
    if np.array_equal(baseline_out, full_out):
        return False
    if np.count_nonzero(baseline_out != full_out) < 6:
        return False

    branch_info = details["branch_info"]
    if branch_info["wall_hits"] < 2:
        return False
    if branch_info["seen_states"] < 2:
        return False

    expected = main(g)
    if not np.array_equal(expected, full_out):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    built = _build_top_canonical()
    if built is None:
        return None
    grid, _meta = built
    transformed = _random_transform(grid)
    if not _validate_generated(transformed):
        return None
    return transformed


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projectile-branch ARC input")
