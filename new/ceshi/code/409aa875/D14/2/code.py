# Variant of 409aa875
# Dimension: D14 — 精灵间最小间距 (meta)
# Change: 任意两个精灵之间 Chebyshev 距离 >= 2（不可8-连通相邻）... → 间距编码优先级

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


"""
Design analysis for the D14 variant
===================================

spec:
  preserved_visual_atoms:
    - 16x16 grid, background 7
    - every sprite is still a 3-cell monochrome triangle:
      axis-aligned L -> diagonal shot, diagonal V -> axis shot
    - every projectile still lands at apex + 5 * direction
    - hit sprite -> whole sprite becomes 9
    - two shots reaching the same empty background cell in the same firing step -> 1
    - one shot reaching an empty background cell in that step -> 9
  changed_dimension:
    old: sprite spacing only guaranteed separability during generation
    new: sprite-to-sprite Chebyshev distance determines firing priority
  concrete_temporal_rule:
    - extract all triangle sprites from the input
    - repeatedly choose the closest pair among the remaining sprites
      using minimum Chebyshev distance between their occupied cells
    - that pair fires simultaneously, and its result is written to the grid
    - then remove that pair from the queue and continue with the next closest pair
    - if one sprite remains unpaired, it fires alone at the end

behavior_trace:
  setup:
    - pair A has two sprites whose shots both land on background cell X
    - pair B has one sprite shooting to the same X and one sprite shooting to Y
  original_logic:
    - all four sprites fire together
    - X receives three projectiles total, so X becomes blue (1)
  variant_logic_if_B_is_closer:
    - pair B fires first, so X is still background and receives only one shot -> X becomes 9
    - later pair A reaches the already painted X, which is no longer background, so it does not upgrade to 1
  variant_logic_if_A_is_closer:
    - pair A fires first, so its same-step double hit makes X blue (1)
    - later pair B cannot overwrite that marker
  visible_difference:
    - the same shared target cell can finish as 9 or 1 depending on which pair is closer

complexity_calibration:
  - keep exactly four sprites so the distance ordering is legible by eye
  - use exactly two intended pairs; one is an internal collision pair, the other contributes a lone competing shot
  - vary which pair is closer across generated samples, so the priority branch is not fixed
  - sprite colors exclude 1 and 9 to keep output markers visually distinct

implicit_assumption_review:
  - old `dest_to_shooters` batching assumed order never matters; this is no longer valid
  - old `hit_components` was collected once from the input and applied after global aggregation; now hits and empty-cell painting must be applied per firing step
  - old empty-cell test used only the input occupancy; now a later pair must check the current output cell, because an earlier pair may already have written a marker there
  - original code tolerated using spacing only as a generation convenience; here component separability is still required structurally, but distance magnitude now carries the puzzle signal

placement_legality:
  - sprites still remain 8-disconnected in the input so the solver can decompose them into distinct 3-cell components
  - the shared target X and unique target Y are kept off all sprite cells, so they are true background landings
  - the closest pair is forced to have strictly smaller cell-distance than any cross-pair distance, which makes the priority cue visible rather than tie-broken accidentally

visual_observability:
  - the decisive cue is geometric and visible: the nearest pair acts first
  - the generator intentionally creates a repeated target cell X so that order changes the final color there
  - no sorting or branching depends on color ids or invisible metadata

main_assumption_audit:
  - `_extract_sprites` assumes every relevant object is an isolated 3-cell L or V component; the generator guarantees that
  - `_firing_schedule` assumes pair distances suffice to create a deterministic order; the generator enforces a unique closest pair
  - `_render_sequential` assumes markers never need to be interpreted as sprites; target-vs-sprite membership is checked against the original sprite geometry only
  - the generator validates every candidate by running `main()` and by checking the intended priority branch explicitly
"""


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
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
        dr_sum = arms[0][0] + arms[1][0]
        dc_sum = arms[0][1] + arms[1][1]
        dr = 0 if dr_sum == 0 else (-1 if dr_sum > 0 else 1)
        dc = 0 if dc_sum == 0 else (-1 if dc_sum > 0 else 1)
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


def _sprite_key(sprite: dict) -> tuple:
    return tuple(sprite["apex"]) + tuple(sorted(sprite["cells"]))


def _sprite_distance(cells_a: tuple[tuple[int, int], ...], cells_b: tuple[tuple[int, int], ...]) -> int:
    return min(max(abs(ar - br), abs(ac - bc)) for ar, ac in cells_a for br, bc in cells_b)


def _firing_schedule(sprites: list[dict]) -> list[tuple[int, ...]]:
    remaining = set(range(len(sprites)))
    groups: list[tuple[int, ...]] = []

    while len(remaining) >= 2:
        ordered = sorted(remaining, key=lambda idx: _sprite_key(sprites[idx]))
        best_key = None
        best_pair = None
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                pa, pb = sorted((a, b), key=lambda idx: _sprite_key(sprites[idx]))
                dist = _sprite_distance(sprites[pa]["cells"], sprites[pb]["cells"])
                key = (dist, _sprite_key(sprites[pa]), _sprite_key(sprites[pb]))
                if best_key is None or key < best_key:
                    best_key = key
                    best_pair = (pa, pb)
        assert best_pair is not None
        groups.append(best_pair)
        remaining.remove(best_pair[0])
        remaining.remove(best_pair[1])

    if remaining:
        groups.append((min(remaining, key=lambda idx: _sprite_key(sprites[idx])),))
    return groups


def _render_sequential(base: np.ndarray, sprites: list[dict], bg: int) -> np.ndarray:
    out = base.copy()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for sid, sprite in enumerate(sprites):
        for rc in sprite["cells"]:
            cell_to_sprite[rc] = sid

    for group in _firing_schedule(sprites):
        bg_counts: Counter[tuple[int, int]] = Counter()
        hit_ids: set[int] = set()
        for sid in group:
            r, c = sprites[sid]["dest"]
            if (r, c) in cell_to_sprite:
                hit_ids.add(cell_to_sprite[(r, c)])
            elif out[r, c] == bg:
                bg_counts[(r, c)] += 1

        for hit_id in hit_ids:
            for r, c in sprites[hit_id]["cells"]:
                out[r, c] = HIT_COLOR
        for (r, c), count in bg_counts.items():
            out[r, c] = COLLISION_COLOR if count >= 2 else HIT_COLOR

    return out


def _render_simultaneous(base: np.ndarray, sprites: list[dict], bg: int) -> np.ndarray:
    out = base.copy()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)

    for sid, sprite in enumerate(sprites):
        for rc in sprite["cells"]:
            cell_to_sprite[rc] = sid
        dest_to_shooters[sprite["dest"]].append(sid)

    hit_ids = {cell_to_sprite[dest] for dest in dest_to_shooters if dest in cell_to_sprite}
    for hit_id in hit_ids:
        for r, c in sprites[hit_id]["cells"]:
            out[r, c] = HIT_COLOR
    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite:
            continue
        r, c = dest
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def _extract_sprites(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    sprites: list[dict] = []

    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            (ar, ac), (dr, dc) = ad
            rr = ar + PROJECTILE_DIST * dr
            cc = ac + PROJECTILE_DIST * dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            cells = tuple(sorted(comp))
            sprites.append(
                {
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": (rr, cc),
                    "color": col,
                }
            )
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve triangle projectiles in distance-priority firing pairs.

    Input structure:
    - The most common color is background. Every foreground object is an isolated 3-cell triangle sprite: either an L made from orthogonal cells or a V made from diagonal legs.

    Rule:
    - Each sprite still determines its projectile direction from its shape, and its projectile still lands exactly 5 cells from the apex.
    - Sprites do not all fire together anymore. Instead, repeatedly take the closest pair of remaining sprites by minimum Chebyshev distance between their occupied cells; that pair fires simultaneously, the result is written to the grid, then the next pair is processed. A final unpaired sprite, if any, fires alone.
    - Within one firing step, hitting a sprite recolors that whole sprite to maroon (9); two shots that reach the same currently empty background cell make that cell blue (1); one shot on a currently empty background cell makes it maroon (9). Later steps do not repaint cells that are no longer background.

    Key insight:
    - Orientation still tells you where each projectile wants to go, but spacing now tells you when each projectile gets to matter; only same-step collisions can create blue.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sprites = _extract_sprites(g)
    return _render_sequential(g, sprites, bg)


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
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": tuple(sorted(cells)),
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _placements_compatible(selected: list[dict], placement: dict, reserved_bg: set[tuple[int, int]]) -> bool:
    if placement["cell_set"] & reserved_bg:
        return False
    for other in selected:
        if placement["halo"] & other["cell_set"]:
            return False
    return True


def _cross_distances(scene: list[dict], left: tuple[int, int], right: tuple[int, int]) -> list[int]:
    vals = []
    for i in left:
        for j in right:
            vals.append(_sprite_distance(scene[i]["cells"], scene[j]["cells"]))
    return vals


def _build_scene_sprites(placements: list[dict]) -> list[dict]:
    sprites = []
    for placement in placements:
        sprites.append(
            {
                "cells": placement["cells"],
                "cell_set": placement["cell_set"],
                "apex": placement["apex"],
                "dir": placement["dir"],
                "dest": placement["dest"],
            }
        )
    return sprites


def _scene_valid_for_branch(placements: list[dict], collision_first: bool) -> bool:
    sprites = _build_scene_sprites(placements)
    groups = [set(group) for group in _firing_schedule(sprites)]
    if groups != ([{0, 1}, {2, 3}] if collision_first else [{2, 3}, {0, 1}]):
        return False

    d_collision = _sprite_distance(placements[0]["cells"], placements[1]["cells"])
    d_single = _sprite_distance(placements[2]["cells"], placements[3]["cells"])
    cross = _cross_distances(placements, (0, 1), (2, 3))
    if not cross:
        return False
    if collision_first:
        return d_collision < d_single and d_collision < min(cross)
    return d_single < d_collision and d_single < min(cross)


def _paint_input(placements: list[dict], color: int) -> np.ndarray:
    grid = np.full((H, W), BG, dtype=int)
    for placement in placements:
        for r, c in placement["cells"]:
            grid[r, c] = color
    return grid


def _choose_scene(rng: random.Random, collision_first: bool) -> tuple[list[dict], tuple[int, int], tuple[int, int]] | None:
    shared_dests = [dest for dest, ps in PLACEMENTS_BY_DEST.items() if len(ps) >= 3]
    rng.shuffle(shared_dests)

    for shared in shared_dests:
        shared_candidates = PLACEMENTS_BY_DEST[shared][:]
        rng.shuffle(shared_candidates)
        reserved = {shared}

        for i, first in enumerate(shared_candidates):
            if first["dest"] != shared:
                continue
            if not _placements_compatible([], first, reserved):
                continue

            for second in shared_candidates[i + 1 :]:
                if not _placements_compatible([first], second, reserved):
                    continue

                pair_a = [first, second]
                third_choices = shared_candidates[:]
                rng.shuffle(third_choices)
                for third in third_choices:
                    if third in pair_a:
                        continue
                    if not _placements_compatible(pair_a, third, reserved):
                        continue

                    selected3 = pair_a + [third]
                    occupied3 = set().union(*(p["cell_set"] for p in selected3))
                    fourth_choices = PLACEMENTS[:]
                    rng.shuffle(fourth_choices)
                    for fourth in fourth_choices:
                        if fourth["dest"] == shared:
                            continue
                        if fourth["dest"] in occupied3:
                            continue
                        if not _placements_compatible(selected3, fourth, reserved):
                            continue

                        placements = [first, second, third, fourth]
                        if not _scene_valid_for_branch(placements, collision_first):
                            continue
                        return placements, shared, fourth["dest"]
    return None


def _validate_input(grid: np.ndarray) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    sprites = _extract_sprites(grid)
    if len(sprites) != 4:
        return False
    return all(len(sprite["cells"]) == 3 for sprite in sprites)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    collision_first = rng.choice([False, True])
    choice = _choose_scene(rng, collision_first)
    if choice is None:
        return None
    placements, shared, unique = choice

    color = rng.choice(SPRITE_COLORS)
    inp = _paint_input(placements, color)
    if not _validate_input(inp):
        return None

    sprites = _build_scene_sprites(placements)
    sequential = _render_sequential(inp, sprites, BG)
    simultaneous = _render_simultaneous(inp, sprites, BG)

    expected_shared = COLLISION_COLOR if collision_first else HIT_COLOR
    if sequential[shared] != expected_shared:
        return None
    if sequential[unique] != HIT_COLOR:
        return None
    if not np.any(sequential == HIT_COLOR):
        return None
    if collision_first and not np.any(sequential == COLLISION_COLOR):
        return None
    if not collision_first and simultaneous[shared] == sequential[shared]:
        return None
    if np.count_nonzero(sequential != BG) <= np.count_nonzero(inp != BG):
        return None
    if np.array_equal(inp, sequential):
        return None

    solved = main(inp)
    if not np.array_equal(solved, sequential):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
