# Variant of 409aa875
# Dimension: D1 — 发射路径类型（对角线 vs 轴对齐） (path)
# Change: L形精灵沿对角线发射，V形精灵沿轴方向发射；方向由形状的缺口/开口方向决定... → 反射路径（遇障碍弹射）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D1/path variant
# ---------------------------------------
# concretization:
# - The input still contains only one family of 3-pixel monochrome triangle sprites on a
#   16x16 orange background. L-shapes still define diagonal launch directions; V-shapes
#   still define axis launch directions.
# - The changed behavior is not in shape recognition, but in how the 5-cell projectile
#   travel is simulated: instead of jumping directly to apex + 5*dir, the projectile
#   advances one cell at a time for 5 moves.
# - On moves 1..4, if the next cell in the current direction is occupied by any sprite,
#   the projectile does not enter that cell. It pivots by 45 degrees to one of the two
#   neighboring compass directions and then takes that step instead. Example:
#   a V-sprite firing east would normally land 5 cells to the right; with a blocker two
#   cells ahead, the east step is replaced by either north-east or south-east. The
#   generator only uses layouts where exactly one of those two 45-degree escape cells is
#   free, so the bend is visually attributable to obstacle geometry rather than to a
#   hidden tiebreak.
# - The 5th move is still the landing move. If that final next cell is a sprite cell, the
#   projectile lands on it and recolors that whole sprite to maroon exactly as in the
#   original task. Background landings still paint maroon, and multi-projectile background
#   collisions still paint blue.
#
# complexity calibration:
# - The visible novelty is the displaced landing marker: at least one shooter per sample
#   must bend around a blocker and end somewhere different from its straight-line
#   apex+5*dir cell. The generator rejects layouts where the new solver output matches the
#   original straight-line output.
# - Reflection choices are intentionally limited to the two adjacent 45-degree directions.
#   This is enough to create bent paths without introducing many visually unsupported
#   branches.
#
# implicit-assumption audit against the original main():
# - Original `rr = ar + 5*dr; cc = ac + 5*dc` assumed direction never changes. This is the
#   main broken assumption and is replaced by stepwise simulation.
# - Original `dest_to_shooters` remained correct once destinations were known, so that part
#   stays unchanged.
# - Original hit detection assumed the only sprite interaction happened at the final
#   destination. In the variant, intermediate sprite encounters are blockers rather than
#   hits; only the final landing cell can trigger a sprite recolor.
# - Original code was robust to ambiguous geometry because no intermediate branching
#   existed. The new solver includes a deterministic clockwise fallback for robustness, but
#   the generator guarantees zero ambiguous turns in intended samples.
#
# placement legality:
# - Sprites remain pairwise separated by Chebyshev distance >= 2 so components never merge.
# - Generator first places a forced reflection pair (shooter + blocker), then incrementally
#   adds extra sprites only if all simulated 5-step trajectories remain in-bounds and every
#   reflection stays unambiguous.
#
# observability / cleanliness:
# - No color-id-driven logic is introduced.
# - The only new cue is spatial: reflected projectiles land off the original straight ray,
#   and the blocking sprite sits exactly where that straight path would have continued.
# - Generator requires at least one reflected projectile to land on background, so the bent
#   result is visible as a new marker rather than being hidden inside a recolored sprite.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
PROJECTILE_DIST = 5
DIR_CYCLE = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
DIR_INDEX = {d: i for i, d in enumerate(DIR_CYCLE)}
DIAG_DIRS = [d for d in DIR_CYCLE if abs(d[0]) + abs(d[1]) == 2]
AXIS_DIRS = [d for d in DIR_CYCLE if abs(d[0]) + abs(d[1]) == 1]
ALL_DIRS = DIR_CYCLE[:]
SPRITE_COUNT_RANGE = (4, 9)
MAX_ATTEMPTS = 400


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


def _simulate_projectile(
    apex: tuple[int, int],
    direction: tuple[int, int],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> dict | None:
    cur = (int(apex[0]), int(apex[1]))
    dr, dc = int(direction[0]), int(direction[1])
    path: list[tuple[int, int]] = []
    reflections = 0
    ambiguous_turns = 0

    for step_idx in range(PROJECTILE_DIST):
        nr = cur[0] + dr
        nc = cur[1] + dc
        if not _in_bounds(nr, nc, h, w):
            return None

        if step_idx < PROJECTILE_DIST - 1 and (nr, nc) in occupied:
            open_turns = []
            for cand in _turn_options((dr, dc)):
                tr = cur[0] + cand[0]
                tc = cur[1] + cand[1]
                if _in_bounds(tr, tc, h, w) and (tr, tc) not in occupied:
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
            if not _in_bounds(nr, nc, h, w) or (nr, nc) in occupied:
                return None

        cur = (int(nr), int(nc))
        path.append(cur)

    return {
        "dest": cur,
        "path": path,
        "reflections": reflections,
        "ambiguous_turns": ambiguous_turns,
    }


def _collect_grid_components(grid: np.ndarray, bg: int) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch 5-step projectiles from triangle sprites, letting other sprites deflect the path.

    Input structure:
    - The background is the most common color. Every foreground component is a 3-pixel
      triangle sprite: either an axis-aligned L or a diagonal V.

    Rule:
    - Each sprite still determines its launch direction from its geometry: L-shapes start
      on a diagonal away from the missing corner, and V-shapes start along the axis away
      from the opening.
    - The projectile now moves one cell at a time for 5 moves. During moves 1..4, if the
      next cell is occupied by any sprite, the projectile reflects by turning 45 degrees
      into an adjacent compass direction and continues from there.
    - After the 5th move, landing on a sprite recolors that whole sprite to maroon (9); a
      lone background landing paints maroon (9); and two or more projectiles landing on
      the same background cell paint blue (1).

    Key insight:
    - Direction is still encoded by the 3-pixel triangle, but the visible destination is
      no longer `apex + 5*dir`. The solver must trace the bent path around blocking
      sprites and only apply the original hit/collision coloring at the final landing cell.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps, cell_to_comp = _collect_grid_components(g, bg)
    occupied = set(cell_to_comp)

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        sim = _simulate_projectile(ad[0], ad[1], occupied, h, w)
        if sim is None:
            continue
        dest_to_shooters.setdefault(sim["dest"], []).append(cid)

    out = g.copy()
    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_comp:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
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
                for cell in cells:
                    by_cell[cell].append(placement)
                pid += 1
    return placements, by_cell


PLACEMENTS, PLACEMENTS_BY_CELL = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _occupied_and_index(sprites: list[dict]) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]]:
    occupied: set[tuple[int, int]] = set()
    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, placement in enumerate(sprites):
        for cell in placement["cells"]:
            occupied.add(cell)
            cell_to_comp[cell] = idx
    return occupied, cell_to_comp


def _simulate_all(sprites: list[dict]) -> list[dict] | None:
    occupied, _ = _occupied_and_index(sprites)
    results = []
    for placement in sprites:
        sim = _simulate_projectile(placement["apex"], placement["dir"], occupied, H, W)
        if sim is None:
            return None
        results.append(sim)
    return results


def _render_from_results(inp: np.ndarray, sprites: list[dict], results: list[dict]) -> np.ndarray:
    out = inp.copy()
    occupied, cell_to_comp = _occupied_and_index(sprites)
    bg_targets = Counter(sim["dest"] for sim in results if sim["dest"] not in occupied)
    hit_sprite_ids = {cell_to_comp[sim["dest"]] for sim in results if sim["dest"] in cell_to_comp}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _render_old_straight(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    straight_results = [{"dest": placement["straight_dest"]} for placement in sprites]
    return _render_from_results(inp, sprites, straight_results)


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


def _choose_reflection_pair(rng: random.Random) -> list[dict] | None:
    shooters = PLACEMENTS[:]
    rng.shuffle(shooters)
    for shooter in shooters:
        occupied = set(shooter["cells"])
        steps = [2, 3, 4]
        rng.shuffle(steps)
        for step in steps:
            cur = (
                shooter["apex"][0] + (step - 1) * shooter["dir"][0],
                shooter["apex"][1] + (step - 1) * shooter["dir"][1],
            )
            forward = (cur[0] + shooter["dir"][0], cur[1] + shooter["dir"][1])
            if not _in_bounds(forward[0], forward[1], H, W):
                continue
            turns = list(_turn_options(shooter["dir"]))
            rng.shuffle(turns)
            for blocked_turn in turns:
                free_turn = turns[1] if blocked_turn == turns[0] else turns[0]
                blocked_cell = (cur[0] + blocked_turn[0], cur[1] + blocked_turn[1])
                free_cell = (cur[0] + free_turn[0], cur[1] + free_turn[1])
                if not (_in_bounds(blocked_cell[0], blocked_cell[1], H, W) and _in_bounds(free_cell[0], free_cell[1], H, W)):
                    continue
                candidates = [p for p in PLACEMENTS_BY_CELL[forward] if blocked_cell in p["cell_set"]]
                rng.shuffle(candidates)
                for blocker in candidates:
                    if blocker["id"] == shooter["id"]:
                        continue
                    if free_cell in blocker["cell_set"]:
                        continue
                    if not _fits(blocker, occupied):
                        continue
                    sim = _simulate_all([shooter, blocker])
                    if sim is None:
                        continue
                    if sim[0]["reflections"] < 1 or sim[0]["ambiguous_turns"] != 0:
                        continue
                    if sim[0]["dest"] == shooter["straight_dest"]:
                        continue
                    return [shooter, blocker]
    return None


def _extend_sprites(rng: random.Random, sprites: list[dict], target_n: int) -> bool:
    occupied, _ = _occupied_and_index(sprites)
    pool = PLACEMENTS[:]
    rng.shuffle(pool)
    for placement in pool:
        if len(sprites) >= target_n:
            return True
        if placement["id"] in {s["id"] for s in sprites}:
            continue
        if not _fits(placement, occupied):
            continue
        trial = sprites + [placement]
        results = _simulate_all(trial)
        if results is None:
            continue
        if sum(sim["ambiguous_turns"] for sim in results) != 0:
            continue
        sprites.append(placement)
        occupied.update(placement["cells"])
    return len(sprites) == target_n


def _reflected_bg_marker_exists(sprites: list[dict], results: list[dict]) -> bool:
    occupied, _ = _occupied_and_index(sprites)
    return any(
        sim["reflections"] > 0 and sim["dest"] not in occupied and sim["dest"] != placement["straight_dest"]
        for placement, sim in zip(sprites, results)
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    pair = _choose_reflection_pair(rng)
    if pair is None:
        return None

    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    sprites = pair[:]
    if not _extend_sprites(rng, sprites, n_sprites):
        return None

    results = _simulate_all(sprites)
    if results is None:
        return None
    if sum(sim["ambiguous_turns"] for sim in results) != 0:
        return None
    if sum(sim["reflections"] for sim in results) < 1:
        return None
    if not _reflected_bg_marker_exists(sprites, results):
        return None

    inp = np.full((H, W), BG, dtype=int)
    sprite_color = rng.choice(SPRITE_COLORS)
    for placement in sprites:
        _paint_sprite(inp, placement, sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_from_results(inp, sprites, results)
    old_out = _render_old_straight(inp, sprites)

    if np.array_equal(out, old_out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
