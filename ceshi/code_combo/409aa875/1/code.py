# Combo variant of 409aa875 (index 1)
# combo_spec: D1/2+D13/0
# Name: 反射路径+静态墙壁

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the combo variant
# -------------------------------------
# a) concretization / behavior trace
#    - Keep the original 16x16 canvas, orange background, and monochrome 3-pixel triangle
#      sprites. Add a fixed wall color (5). Walls are input-only obstacles, not sprites.
#    - Shape reading is unchanged: L-shapes launch diagonally away from the missing 2x2
#      corner; V-shapes launch axially away from the opening.
#    - The old solver jumped directly to `apex + 5*dir`. The fused rule instead traces 5
#      one-cell moves. Whenever the next cell is a wall, the projectile does not enter the
#      wall; it turns by 45 degrees into one of the adjacent compass directions and uses
#      that step immediately.
#    - Concrete trace: a V-sprite at apex (10, 6) firing north would normally visit
#      (9,6),(8,6),(7,6),(6,6),(5,6). If a wall sits at (8,6), then on move 2 the shot
#      turns to either north-west or north-east. In generated samples a second wall blocks
#      one branch, so the turn is forced. Example forced path:
#      (9,6) -> wall ahead at (8,6), guard wall at (8,5), therefore turn to (8,7) and
#      continue to (7,8),(6,9),(5,10). The landing cell changes from (5,6) to (5,10).
#
# b) complexity calibration
#    - Reflection is driven only by visible wall pixels; sprite colors never encode hidden
#      priorities. To avoid unsupported ambiguity, the generator always builds a wall pair:
#      one wall straight ahead and one wall on exactly one of the two 45-degree branches.
#    - Each sample contains at least one reflected projectile and at least one untouched
#      straight projectile, so the human solver can compare “bent” and “unchanged” rays in
#      the same grid. The generator also rejects samples whose combo output matches either
#      the original straight-line solver or the wall-truncation-only solver.
#
# c) implicit-assumption audit against the original main()
#    - Original component scan assumed every non-background component was a triangle. That
#      breaks once walls are introduced, so wall color is excluded from sprite detection.
#    - Original landing arithmetic assumed the direction never changes. The fused solver
#      replaces that with stepwise simulation and turn handling.
#    - Original hit lookup only depended on the final destination; that still holds after
#      the path is simulated, so recolor/collision logic can stay the same.
#    - A new assumption appears: projectiles are only meant to touch sprites on their final
#      landing step, not during intermediate travel. The generator enforces this by
#      rejecting any layout where a simulated path crosses a sprite before move 5.
#
# d) placement legality
#    - Triangle sprites keep Chebyshev separation >= 2 so 8-connected components never
#      merge.
#    - Walls are kept off sprite halos, except for the intended reflected path geometry.
#      Additional sprites are accepted only if the full set of simulated trajectories stays
#      in-bounds, unambiguous, and free of mid-flight sprite crossings.
#
# e) visual observability
#    - The new cue is spatial and explicit: grey walls create a bent 5-step path, and the
#      output marker appears at the end of that bend rather than at the old straight or
#      truncated landing cell. Because at least one other projectile remains straight, a
#      human can infer that walls alter only the shots they intersect.
#
# f) visual cleanliness
#    - The generator uses a single forced reflection cluster (two nearby wall pixels) and
#      otherwise fills the board with straight, non-overlapping shooters. This keeps the
#      wall evidence local and prevents the grid from becoming cluttered with many crossing
#      bends.
#
# g) change inferability
#    - Output colors preserve the original semantics exactly: maroon (9) for a lone
#      background landing or a hit sprite, blue (1) for multi-projectile background
#      collisions. Wall color (5) never changes role. The only new inference is the wall-
#      induced 45-degree reflection.


H = W = 16
BG = 7
WALL_COLOR = 5
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 6, 8]
PROJECTILE_DIST = 5

DIR_CYCLE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIR_CYCLE)}
DIAG_DIRS = [d for d in DIR_CYCLE if abs(d[0]) + abs(d[1]) == 2]
AXIS_DIRS = [d for d in DIR_CYCLE if abs(d[0]) + abs(d[1]) == 1]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 8)
MAX_ATTEMPTS = 700
GENERATION_MODES = ("plain", "collision", "sprite_hit")


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms = []
        for r, c in comp:
            if (r, c) == corner:
                continue
            dr, dc = r - cr, c - cc
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return corner, (dr, dc)

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr = -((d1[0] + d2[0]) // 2)
    dc = -((d1[1] + d2[1]) // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _in_bounds(r: int, c: int, h: int, w: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _turn_options(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    idx = DIR_INDEX[direction]
    return DIR_CYCLE[(idx - 1) % 8], DIR_CYCLE[(idx + 1) % 8]


def _landing_straight(
    apex: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]
) -> tuple[int, int] | None:
    h, w = shape
    rr = int(apex[0] + PROJECTILE_DIST * direction[0])
    cc = int(apex[1] + PROJECTILE_DIST * direction[1])
    if not _in_bounds(rr, cc, h, w):
        return None
    return rr, cc


def _landing_truncate(
    apex: tuple[int, int],
    direction: tuple[int, int],
    walls: set[tuple[int, int]],
    shape: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = shape
    prev = (int(apex[0]), int(apex[1]))
    for step in range(1, PROJECTILE_DIST + 1):
        cell = (int(apex[0] + step * direction[0]), int(apex[1] + step * direction[1]))
        if not _in_bounds(cell[0], cell[1], h, w):
            return None
        if cell in walls:
            return prev
        prev = cell
    return prev


def _simulate_projectile(
    apex: tuple[int, int],
    direction: tuple[int, int],
    walls: set[tuple[int, int]],
    shape: tuple[int, int],
) -> dict | None:
    h, w = shape
    cur = (int(apex[0]), int(apex[1]))
    dr, dc = int(direction[0]), int(direction[1])
    path: list[tuple[int, int]] = []
    reflections = 0
    ambiguous_turns = 0

    for _ in range(PROJECTILE_DIST):
        nr = cur[0] + dr
        nc = cur[1] + dc
        if not _in_bounds(nr, nc, h, w):
            return None

        if (nr, nc) in walls:
            open_turns = []
            for cand in _turn_options((dr, dc)):
                tr = cur[0] + cand[0]
                tc = cur[1] + cand[1]
                if _in_bounds(tr, tc, h, w) and (tr, tc) not in walls:
                    open_turns.append(cand)
            if not open_turns:
                return None
            if len(open_turns) == 2:
                ambiguous_turns += 1
                dr, dc = open_turns[1]
            else:
                dr, dc = open_turns[0]
            reflections += 1
            nr = cur[0] + dr
            nc = cur[1] + dc
            if not _in_bounds(nr, nc, h, w) or (nr, nc) in walls:
                return None

        cur = (int(nr), int(nc))
        path.append(cur)

    return {
        "dest": cur,
        "path": path,
        "reflections": reflections,
        "ambiguous_turns": ambiguous_turns,
    }


def _collect_sprites(grid: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg, WALL_COLOR}):
        for comp in _cc8(grid == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _render_from_destinations(
    grid: np.ndarray,
    comps: list[list[tuple[int, int]]],
    cell_to_comp: dict[tuple[int, int], int],
    destinations: list[tuple[int, int]],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, dest in enumerate(destinations):
        dest_to_shooters.setdefault(dest, []).append(cid)

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR
    return out


def _solve_grid(grid: np.ndarray, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    walls = {(int(r), int(c)) for r, c in np.argwhere(g == WALL_COLOR)} if mode != "straight" else set()

    comps, cell_to_comp = _collect_sprites(g, bg)
    destinations: list[tuple[int, int]] = []
    for comp in comps:
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        if mode == "straight":
            dest = _landing_straight(ad[0], ad[1], (h, w))
        elif mode == "truncate":
            dest = _landing_truncate(ad[0], ad[1], walls, (h, w))
        elif mode == "combo":
            sim = _simulate_projectile(ad[0], ad[1], walls, (h, w))
            dest = None if sim is None else sim["dest"]
        else:
            raise ValueError(mode)
        if dest is not None:
            destinations.append(dest)
    return _render_from_destinations(g, comps, cell_to_comp, destinations)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch 5-step projectiles from triangle sprites, reflecting them off grey wall pixels.

    Input structure:
    - The background is the most common color.
    - Non-wall foreground components are still monochrome 3-pixel triangles: axis-aligned
      L-shapes and diagonal V-shapes.
    - Grey cells (color 5) are static walls. They are not sprites and they never recolor.

    Rule:
    - Each triangle still determines its starting direction from its geometry: L-shapes
      start diagonally away from the missing corner, while V-shapes start along the axis
      opposite their opening.
    - A projectile now moves one cell at a time for 5 total moves. If the next cell is a
      wall, the shot does not enter that wall; instead it turns 45 degrees into an
      adjacent compass direction and uses that step immediately.
    - After all 5 moves, landing on a sprite recolors that entire sprite to maroon (9); a
      lone background landing paints maroon (9); and two or more projectiles landing on
      the same empty background cell paint blue (1).

    Key insight:
    - The triangles still encode only the initial direction. The visible target must be
      found by identifying grey wall pixels and tracing the bent 5-step trajectory they
      induce; neither the original straight jump nor a simple “stop before wall” rule is
      sufficient.
    """
    return _solve_grid(grid, mode="combo")


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], list[dict]]:
    placements: list[dict] = []
    reflection_options: list[dict] = []
    pid = 0
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                straight_dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not _in_bounds(straight_dest[0], straight_dest[1], H, W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "id": pid,
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "straight_dest": straight_dest,
                }
                placements.append(placement)
                pid += 1

                for block_step in (2, 3, 4):
                    cur = (ar + (block_step - 1) * dr, ac + (block_step - 1) * dc)
                    main_wall = (cur[0] + dr, cur[1] + dc)
                    if not _in_bounds(main_wall[0], main_wall[1], H, W):
                        continue
                    turns = _turn_options((dr, dc))
                    for blocked_turn in turns:
                        guard_wall = (cur[0] + blocked_turn[0], cur[1] + blocked_turn[1])
                        if not _in_bounds(guard_wall[0], guard_wall[1], H, W):
                            continue
                        walls = {main_wall, guard_wall}
                        if walls & placement["halo"]:
                            continue
                        sim = _simulate_projectile((ar, ac), (dr, dc), walls, (H, W))
                        if sim is None:
                            continue
                        if sim["reflections"] != 1 or sim["ambiguous_turns"] != 0:
                            continue
                        if sim["dest"] == straight_dest:
                            continue
                        reflection_options.append(
                            {
                                "placement": placement,
                                "dest": sim["dest"],
                                "walls": frozenset(walls),
                                "straight_dest": straight_dest,
                            }
                        )
    return placements, reflection_options


PLACEMENTS, REFLECTION_OPTIONS = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _placement_fits(placement: dict, occupied: set[tuple[int, int]], walls: set[tuple[int, int]]) -> bool:
    blocked = occupied | walls
    return not (placement["halo"] & blocked)


def _occupied_and_index(shots: list[dict]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    occupied: set[tuple[int, int]] = set()
    cell_to_idx: dict[tuple[int, int], int] = {}
    for idx, shot in enumerate(shots):
        for cell in shot["placement"]["cells"]:
            occupied.add(cell)
            cell_to_idx[cell] = idx
    return occupied, cell_to_idx


def _simulate_all(shots: list[dict], walls: set[tuple[int, int]]) -> list[dict] | None:
    occupied, _ = _occupied_and_index(shots)
    results: list[dict] = []
    for shot in shots:
        sim = _simulate_projectile(shot["placement"]["apex"], shot["placement"]["dir"], walls, (H, W))
        if sim is None:
            return None
        if sim["ambiguous_turns"] != 0:
            return None
        if any(cell in occupied for cell in sim["path"][:-1]):
            return None
        if sim["dest"] in walls:
            return None
        results.append(sim)
    return results


def _layout_profile(shots: list[dict], walls: set[tuple[int, int]]) -> dict | None:
    results = _simulate_all(shots, walls)
    if results is None:
        return None
    occupied, _ = _occupied_and_index(shots)
    dest_counts = Counter(sim["dest"] for sim in results)
    if any(shot["dest"] != sim["dest"] for shot, sim in zip(shots, results)):
        return None
    return {
        "results": results,
        "occupied": occupied,
        "dest_counts": dest_counts,
        "reflected": sum(sim["reflections"] > 0 for sim in results),
        "straight": sum(sim["reflections"] == 0 for sim in results),
        "sprite_hit_shots": sum(sim["dest"] in occupied for sim in results),
        "bg_collision_cells": sum(count >= 2 and dest not in occupied for dest, count in dest_counts.items()),
        "bg_collision_shots": sum(count for dest, count in dest_counts.items() if count >= 2 and dest not in occupied),
    }


def _all_shots_stable(shots: list[dict], walls: set[tuple[int, int]]) -> list[dict] | None:
    profile = _layout_profile(shots, walls)
    if profile is None:
        return None
    results = profile["results"]
    occupied = profile["occupied"]
    if any(sim["dest"] in occupied for sim in results):
        return None
    if len(profile["dest_counts"]) != len(results):
        return None
    return results


def _candidate_shot(placement: dict, shots: list[dict], walls: set[tuple[int, int]]) -> dict | None:
    occupied, _ = _occupied_and_index(shots)
    if not _placement_fits(placement, occupied, walls):
        return None
    sim = _simulate_projectile(placement["apex"], placement["dir"], walls, (H, W))
    if sim is None or sim["ambiguous_turns"] != 0:
        return None
    if any(cell in occupied for cell in sim["path"][:-1]):
        return None
    if sim["dest"] in walls:
        return None
    return {
        "placement": placement,
        "dest": sim["dest"],
        "reflected": sim["reflections"] > 0,
        "straight_dest": placement["straight_dest"],
    }


def _choose_reflection_seed(rng: random.Random) -> tuple[list[dict], set[tuple[int, int]]] | None:
    options = REFLECTION_OPTIONS[:]
    rng.shuffle(options)
    for option in options:
        walls = set(option["walls"])
        shot = {
            "placement": option["placement"],
            "dest": option["dest"],
            "reflected": True,
            "straight_dest": option["straight_dest"],
        }
        if not _placement_fits(option["placement"], set(), walls):
            continue
        if _all_shots_stable([shot], walls) is not None:
            return [shot], walls
    return None


def _choose_unique_shot(
    rng: random.Random,
    shots: list[dict],
    walls: set[tuple[int, int]],
    require_unreflected: bool,
) -> dict | None:
    occupied, _ = _occupied_and_index(shots)
    reserved = {shot["dest"] for shot in shots}
    used_ids = {shot["placement"]["id"] for shot in shots}
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["id"] in used_ids:
            continue
        shot = _candidate_shot(placement, shots, walls)
        if shot is None:
            continue
        if require_unreflected and shot["reflected"]:
            continue
        if shot["dest"] in reserved or shot["dest"] in occupied:
            continue
        trial = shots + [shot]
        if _all_shots_stable(trial, walls) is not None:
            return shot
    return None


def _choose_collision_pair(rng: random.Random, shots: list[dict], walls: set[tuple[int, int]]) -> list[dict] | None:
    occupied, _ = _occupied_and_index(shots)
    reserved = {shot["dest"] for shot in shots}
    used_ids = {shot["placement"]["id"] for shot in shots}
    grouped: dict[tuple[int, int], list[dict]] = {}
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["id"] in used_ids:
            continue
        shot = _candidate_shot(placement, shots, walls)
        if shot is None or shot["reflected"]:
            continue
        if shot["dest"] in occupied or shot["dest"] in reserved:
            continue
        grouped.setdefault(shot["dest"], []).append(shot)

    dests = list(grouped)
    rng.shuffle(dests)
    for dest in dests:
        group = grouped[dest]
        if len(group) < 2:
            continue
        rng.shuffle(group)
        for i, first in enumerate(group):
            trial = shots + [first]
            for second in group[i + 1 :]:
                second_shot = _candidate_shot(second["placement"], trial, walls)
                if second_shot is None or second_shot["reflected"] or second_shot["dest"] != dest:
                    continue
                final_shots = trial + [second_shot]
                profile = _layout_profile(final_shots, walls)
                if profile is None:
                    continue
                if profile["sprite_hit_shots"] != 0:
                    continue
                if profile["bg_collision_cells"] != 1 or profile["bg_collision_shots"] != 2:
                    continue
                return [first, second_shot]
    return None


def _choose_sprite_hit_shot(rng: random.Random, shots: list[dict], walls: set[tuple[int, int]]) -> dict | None:
    occupied, _ = _occupied_and_index(shots)
    used_ids = {shot["placement"]["id"] for shot in shots}
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["id"] in used_ids:
            continue
        shot = _candidate_shot(placement, shots, walls)
        if shot is None or shot["reflected"]:
            continue
        if shot["dest"] not in occupied or shot["dest"] in placement["cell_set"]:
            continue
        trial = shots + [shot]
        profile = _layout_profile(trial, walls)
        if profile is None:
            continue
        if profile["sprite_hit_shots"] != 1:
            continue
        if profile["bg_collision_cells"] != 0:
            continue
        return shot
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, n_walls: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    if np.count_nonzero(grid == WALL_COLOR) != n_walls:
        return False
    sprite_colors = sorted(set(int(v) for v in np.unique(grid)) - {BG, WALL_COLOR})
    if len(sprite_colors) != 1:
        return False
    comps = []
    for col in sprite_colors:
        comps.extend(_cc8(grid == col))
    if len(comps) != n_sprites:
        return False
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


def _render_output(inp: np.ndarray, shots: list[dict], results: list[dict]) -> np.ndarray:
    comps, cell_to_comp = _collect_sprites(inp, _bg_color(inp))
    return _render_from_destinations(inp, comps, cell_to_comp, [sim["dest"] for sim in results])


def _gen_attempt(mode: str) -> np.ndarray | None:
    rng = random.Random()
    seeded = _choose_reflection_seed(rng)
    if seeded is None:
        return None
    shots, walls = seeded

    contrast = _choose_unique_shot(rng, shots, walls, require_unreflected=True)
    if contrast is None:
        return None
    shots.append(contrast)

    if mode == "collision":
        pair = _choose_collision_pair(rng, shots, walls)
        if pair is None:
            return None
        shots.extend(pair)
    elif mode == "sprite_hit":
        hitter = _choose_sprite_hit_shot(rng, shots, walls)
        if hitter is None:
            return None
        shots.append(hitter)
    elif mode != "plain":
        raise ValueError(mode)

    n_sprites = max(len(shots), rng.randint(*SPRITE_COUNT_RANGE))
    while len(shots) < n_sprites:
        extra = _choose_unique_shot(rng, shots, walls, require_unreflected=True)
        if extra is None:
            if mode == "plain":
                return None
            break
        shots.append(extra)

    profile = _layout_profile(shots, walls)
    if profile is None:
        return None
    results = profile["results"]

    inp = np.full((H, W), BG, dtype=int)
    for r, c in walls:
        inp[r, c] = WALL_COLOR
    sprite_color = rng.choice(SPRITE_COLORS)
    for shot in shots:
        _paint_sprite(inp, shot["placement"], sprite_color)

    if not _validate_input(inp, len(shots), len(walls)):
        return None

    out = _render_output(inp, shots, results)
    straight_out = _solve_grid(inp, mode="straight")
    trunc_out = _solve_grid(inp, mode="truncate")

    if profile["reflected"] < 1:
        return None
    if profile["straight"] < 1:
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if np.array_equal(out, straight_out):
        return None
    if np.array_equal(out, trunc_out):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if mode == "plain" and (profile["sprite_hit_shots"] != 0 or profile["bg_collision_cells"] != 0):
        return None
    if mode == "collision" and (profile["sprite_hit_shots"] != 0 or profile["bg_collision_cells"] != 1):
        return None
    if mode == "sprite_hit" and profile["sprite_hit_shots"] != 1:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    counter = getattr(generate_input, "_counter", 0)
    preferred = GENERATION_MODES[counter % len(GENERATION_MODES)]
    generate_input._counter = counter + 1

    ordered_modes = (preferred,) + tuple(mode for mode in GENERATION_MODES if mode != preferred)
    attempts_per_mode = max(1, MAX_ATTEMPTS // len(ordered_modes))
    for mode in ordered_modes:
        for _ in range(attempts_per_mode):
            grid = _gen_attempt(mode)
            if grid is not None:
                return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
