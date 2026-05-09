# Combo variant of 409aa875 (index 4)
# combo_spec: D5/0+D6/0
# Name: 碰撞爆炸+连续轨迹

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    - The input stays in the original object language: a 16x16 grid with bg=7 and
#      4-9 monochrome 3-pixel triangle sprites, each either an axis-aligned L or a
#      diagonal V. Shape still determines direction exactly as in the base task.
#    - The fused change is not "draw trails" plus "change endpoint marker" as two
#      unrelated edits. Instead, every projectile now occupies its whole 5-step path:
#      steps 1..5 from the apex are the projectile's visible trajectory, with normal
#      path cells shown in maroon 9.
#    - Collision detection therefore moves from "shared empty landing cell only" to
#      "any shared empty path cell". Example: a vertical V-shot and a horizontal V-shot
#      may both pass through (8,8) at step 3 and continue to different endpoints.
#      In the base task, nothing special happened at (8,8) because it was not a landing
#      cell. In this combo, their paths intersect there, so the output paints a blue
#      cross centered at (8,8) while the rest of both trajectories stay maroon.
#    - If a projectile ends on a sprite cell, the hit sprite is still recolored to 9.
#      A lone empty path/landing cell is 9, while a repeated empty path cell becomes the
#      center of a blue explosion cross.
#
# b) Complexity calibration
#    - The generator always includes at least one genuine mid-flight intersection, not
#      just a shared destination. This makes the fusion visually different from both the
#      original task and the standalone D6 variant.
#    - Only two effect colors are used: maroon 9 for ordinary trail / lone landing /
#      hit sprite, and blue 1 for collision explosions. The trail length remains fixed
#      at 5 from the original geometry, so the new freedom comes from where trajectories
#      intersect, not from variable ray lengths or extra color categories.
#    - Each accepted sample also contains at least one non-collision projectile, so the
#      learner sees both maroon-only trajectories and blue collision crosses together.
#
# c) Hidden assumptions broken by the fusion
#    - The base solver assumed only destinations mattered. That fails once intermediate
#      path cells become visible and can collide.
#    - The D5 standalone variant assumed collision centers were endpoints. With full
#      trails, a collision can happen before the destination, so the solver must count
#      visits over the whole path, not only over `apex + 5*dir`.
#    - The old generator could ignore path occupancy. That no longer works: if a trail
#      crosses a sprite, the output would become ambiguous. The combo generator therefore
#      reserves all non-collision trail cells as background-only cells.
#    - Explosion crosses now coexist with trails. Write order matters: maroon trails are
#      painted first, hit sprites next, and blue crosses last so collisions override the
#      underlying trail line locally.
#
# d) Placement legality
#    - Every source sprite remains a separate 3-pixel 8-connected component.
#    - The generator constructs one deliberate orthogonal mid-air collision between two
#      axis shots whose trails share exactly one background cell. Their source sprites
#      stay separated by the original Chebyshev>=2 rule.
#    - All other placed sprites must avoid every reserved trail cell, endpoint, and
#      explosion footprint, so later objects never overwrite the deliberate collision.
#
# e) Visual observability
#    - A human can read direction from the maroon rays and can distinguish collision from
#      non-collision because only shared path cells create a 5-cell blue cross.
#    - The generator forces the key fused cue: at least one blue cross appears at a cell
#      that is not a shared landing in the base task, proving that the new rule depends
#      on trajectory overlap rather than destination coincidence alone.
#
# f) Visual cleanliness
#    - Outside the deliberate collision center, trails are kept disjoint. This avoids a
#      messy board full of accidental overlaps and makes the blue cross easy to locate.
#    - Explosion footprints are kept away from all source sprites and unrelated trails.
#      The colliding pair may run through cells adjacent to the center, which is fine:
#      those path cells simply get overwritten by the blue cross in the output.
#
# g) Color inferability
#    - No hidden numeric ordering of colors is used. The rule is structural:
#      one projectile on a background path cell -> maroon; multiple projectiles on the
#      same background path cell -> blue explosion cross; landing on a sprite -> recolor
#      that sprite maroon.
#
# Post-change main() assumptions checked against generate_input():
#    - main() assumes each non-bg input component is still a valid 3-pixel triangle:
#      `_validate_input()` enforces that.
#    - main() assumes every non-collision trail cell is background in the input:
#      placement code reserves those cells before adding later sprites.
#    - main() assumes blue explosion footprints do not need to preserve source sprites:
#      the generator keeps collision crosses away from all sprite cells.
#    - main() assumes any repeated background path cell is intentional and readable:
#      the generator allows exactly one shared path cell cluster per collision event and
#      forbids other trail overlaps.


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
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = -dr_sum
        dc = -dc_sum
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
    dr_sum = d1[0] + d2[0]
    dc_sum = d1[1] + d2[1]
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _trail_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, PROJECTILE_DIST + 1))


def _collision_cross(center: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = center
    return ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _paint_collision_cross(grid: np.ndarray, center: tuple[int, int], color: int) -> None:
    h, w = grid.shape
    for r, c in _collision_cross(center):
        if 0 <= r < h and 0 <= c < w:
            grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites, draw their full paths, and explode shared path cells.

    Input structure:
    - The most common color is background, and every foreground component is a monochrome 3-cell triangle sprite: either an axis-aligned L or a diagonal V.

    Rule:
    - Each triangle determines an apex and firing direction from its geometry: L-shapes shoot diagonally away from the missing 2x2 corner, while V-shapes shoot along the axis opposite their opening.
    - The projectile occupies the whole 5-cell ray from the apex. Any path cell on background is painted maroon (9), so the full straight trajectory becomes visible.
    - If the step-5 destination lies on a sprite, that entire struck sprite is recolored maroon (9).
    - If two or more projectiles pass through the same background cell anywhere along their paths, that shared cell becomes the center of a blue (1) explosion cross extending one step in the four cardinal directions.

    Key insight: The triangles no longer encode only a hidden landing point; they encode full rays, and collisions are inferred from where those visible rays coincide on background cells.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    shots: list[tuple[tuple[int, int], ...]] = []
    bg_visit_counts: Counter[tuple[int, int]] = Counter()
    hit_components: set[int] = set()

    for comp in comps:
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        trail = _trail_cells(*ad)
        rr, cc = trail[-1]
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        shots.append(trail)
        if (rr, cc) in cell_to_comp:
            hit_components.add(cell_to_comp[(rr, cc)])
        for r, c in trail:
            if 0 <= r < h and 0 <= c < w and g[r, c] == bg:
                bg_visit_counts[(r, c)] += 1

    out = g.copy()

    for trail in shots:
        for r, c in trail:
            if g[r, c] == bg:
                out[r, c] = TRAIL_COLOR

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for cell, count in bg_visit_counts.items():
        if count >= 2:
            _paint_collision_cross(out, cell, COLLISION_COLOR)

    return out


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
TRAIL_COLOR = 9
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 800


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


def _cross_is_full_in_bounds(center: tuple[int, int]) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in _collision_cross(center))


def _build_placements() -> tuple[
    list[dict],
    dict[tuple[int, int], list[dict]],
    dict[tuple[int, int], list[dict]],
]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_trail: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                trail = _trail_cells((ar, ac), (dr, dc))
                step_map = {cell: step for step, cell in enumerate(trail, start=1)}
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "trail": trail,
                    "trail_set": frozenset(trail),
                    "trail_mid": trail[:-1],
                    "trail_mid_set": frozenset(trail[:-1]),
                    "step_map": step_map,
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                for cell in trail:
                    by_trail[cell].append(placement)
    return placements, by_dest, by_trail


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_TRAIL = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _sprite_fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _fits_unique_bg(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if placement["trail_set"] & occupied:
        return False
    if placement["trail_set"] & reserved_bg:
        return False
    return True


def _fits_hit(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if placement["trail_mid_set"] & occupied:
        return False
    if placement["trail_mid_set"] & reserved_bg:
        return False
    return True


def _reserve_sprite(sprite: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> None:
    placement = sprite["placement"]
    occupied.update(placement["cells"])
    reserved_bg.update(placement["trail_mid"])
    if sprite["target_kind"] == "bg":
        reserved_bg.add(sprite["target"])


def _choose_midair_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    centers = [cell for cell in PLACEMENTS_BY_TRAIL if _cross_is_full_in_bounds(cell)]
    rng.shuffle(centers)
    for center in centers:
        if center in occupied or center in reserved_bg:
            continue
        center_footprint = set(_collision_cross(center))
        candidates = [
            p
            for p in PLACEMENTS_BY_TRAIL[center]
            if abs(p["dir"][0]) + abs(p["dir"][1]) == 1
            and 2 <= p["step_map"][center] <= 4
            and _sprite_fits(p, occupied, reserved_bg)
            and not (p["trail_set"] & occupied)
            and not ((p["trail_set"] - {center}) & reserved_bg)
        ]
        rng.shuffle(candidates)
        for first in candidates:
            if center_footprint & first["cell_set"]:
                continue
            occ2 = occupied | set(first["cells"])
            for second in candidates:
                if second is first:
                    continue
                if second["dir"][0] == first["dir"][0] or second["dir"][1] == first["dir"][1]:
                    continue
                if center_footprint & second["cell_set"]:
                    continue
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                if (first["trail_set"] & second["trail_set"]) != {center}:
                    continue
                if (first["trail_set"] - {center}) & second["cell_set"]:
                    continue
                if (second["trail_set"] - {center}) & first["cell_set"]:
                    continue
                if (second["trail_set"] - {center}) & reserved_bg:
                    continue
                if second["trail_set"] & occupied:
                    continue
                if first["dest"] == second["dest"]:
                    continue
                return center, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if _fits_unique_bg(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits_hit(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for placement in candidates:
            return placement, cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int] | None:
    options = []
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 3) + 1):
        unique = n_sprites - 2 - e_hit
        if unique >= 1:
            options.append((e_hit, unique))
    return rng.choice(options) if options else None


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for comp in _cc8(inp != BG):
        cid = len(comps)
        comps.append(comp)
        for rc in comp:
            cell_to_comp[rc] = cid

    bg_visit_counts: Counter[tuple[int, int]] = Counter()
    for sprite in sprites:
        for cell in sprite["placement"]["trail"]:
            if inp[cell] == BG:
                bg_visit_counts[cell] += 1

    for cell in bg_visit_counts:
        out[cell] = TRAIL_COLOR

    hit_components: set[int] = set()
    for sprite in sprites:
        dest = sprite["placement"]["dest"]
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for cell, count in bg_visit_counts.items():
        if count >= 2:
            _paint_collision_cross(out, cell, COLLISION_COLOR)

    return out


def _render_endpoint_only_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for comp in _cc8(inp != BG):
        cid = len(comps)
        comps.append(comp)
        for rc in comp:
            cell_to_comp[rc] = cid

    bg_targets = Counter()
    hit_components: set[int] = set()
    for sprite in sprites:
        dest = sprite["placement"]["dest"]
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])
        elif inp[dest] == BG:
            bg_targets[dest] += 1

    for sprite in sprites:
        for cell in sprite["placement"]["trail"]:
            if inp[cell] == BG:
                out[cell] = TRAIL_COLOR

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, count in bg_targets.items():
        if count >= 2:
            _paint_collision_cross(out, dest, COLLISION_COLOR)

    return out


def _count_midair_collisions(inp: np.ndarray, sprites: list[dict]) -> int:
    visits: defaultdict[tuple[int, int], list[int]] = defaultdict(list)
    for sprite in sprites:
        for step, cell in enumerate(sprite["placement"]["trail"], start=1):
            if inp[cell] == BG:
                visits[cell].append(step)
    total = 0
    for steps in visits.values():
        if len(steps) >= 2 and any(step < PROJECTILE_DIST for step in steps):
            total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    pair = _choose_midair_collision_pair(rng, occupied, reserved_bg)
    if pair is None:
        return None
    collision_center, placements = pair
    for placement in placements:
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
    reserved_bg.update(_collision_cross(collision_center))

    pre_hit_unique = 0
    if unique_total >= 2 and e_hit >= 1 and rng.random() < 0.6:
        pre_hit_unique = 1
    for _ in range(pre_hit_unique):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return None
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
    unique_total -= pre_hit_unique

    used_target_cells: set[tuple[int, int]] = set()
    if e_hit > 0:
        if len(sprites) < e_hit:
            return None
        hit_targets = rng.sample(range(len(sprites)), e_hit)
        for idx in hit_targets:
            choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
            if choice is None:
                return None
            placement, cell = choice
            sprite = {"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx}
            sprites.append(sprite)
            _reserve_sprite(sprite, occupied, reserved_bg)
            used_target_cells.add(cell)

    for _ in range(unique_total):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return None
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)

    if len(sprites) != n_sprites:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    endpoint_only = _render_endpoint_only_output(inp, sprites)
    midair_collisions = _count_midair_collisions(inp, sprites)
    extra_pixels = int(np.count_nonzero(out != endpoint_only))

    if midair_collisions < 1:
        return None
    if extra_pixels < 3:
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if np.array_equal(inp, out) or np.array_equal(out, endpoint_only):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
