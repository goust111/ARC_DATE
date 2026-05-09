# Combo variant of 409aa875 (index 25)
# combo_spec: D6/2+D10/1
# Name: 颜色渐变轨迹+碰撞色由方向决定
#
# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   summary:
#     - Keep the original 16x16 orange background and the original 3-pixel triangle
#       sprites whose orientation determines one 5-step projectile direction.
#     - Fuse two signals:
#         1. the full trajectory is drawn with a 2-2-1 distance gradient;
#         2. only an empty-cell multi-projectile collision changes the endpoint color,
#            and that color is chosen by the shared direction family.
#   concretization_and_behavior_trace:
#     local_example:
#       - Suppose one L-sprite fires diagonally down-right from apex A.
#       - Original task: only A+5d changed, with empty-single -> 9 and empty-double -> 1.
#       - D6 alone: A+d,A+2d become 4; A+3d,A+4d become 6; A+5d becomes 9.
#       - This combo: steps 1-4 still show 4,4,6,6. At step 5:
#           * lone empty landing -> 9
#           * hit another sprite -> that whole sprite -> 9
#           * 2+ shooters on the same empty cell -> color depends on family:
#               horizontal -> 1, vertical -> 2, diagonal -> 3
#     difference_from_source:
#       - The source task encoded only endpoint type.
#       - The combo exposes distance on every ray segment and direction family on
#         shared empty endpoints, so the solver must combine both geometry channels.
#   complexity_calibration:
#     - The distance palette uses only three endpoint-adjacent bands: 4 for steps 1-2,
#       6 for steps 3-4, and a special step-5 color from {9,1,2,3}. This is small
#       enough to infer from a few examples.
#     - Sprite colors are restricted away from {1,2,3,4,6,9} so trail colors and
#       collision colors never blend with sprite bodies.
#     - Every generated sample includes exactly two empty-cell collisions from two
#       different direction families, so the direction-coded endpoint behavior is
#       always visible and branch coverage is balanced across small ARC train sets.
#   implicit_assumptions_review:
#     - _triangle_apex_and_dir still assumes every meaningful component is exactly one
#       canonical 3-cell L or V sprite; generation preserves that.
#     - The source solver only needed destination multiplicity. The combo solver also
#       needs each shooter's direction family, because the endpoint color now depends on
#       whether the shared family is horizontal, vertical, or diagonal.
#     - Once trajectories are painted, main() implicitly assumes steps 1-4 do not cross
#       existing sprites or other paths in generated examples; otherwise overpainting
#       would make the signal ambiguous. The generator enforces disjoint interiors.
#     - For multi-hit empty collisions, main() assumes all shooters sharing that empty
#       endpoint belong to one family. The generator constructs collisions from one
#       prefiltered family only, so the mapping stays human-inferable.
#   placement_legality:
#     - All sprite cells and all 5-step paths stay in-bounds.
#     - Sprites remain Chebyshev-separated by at least 2, so input components do not
#       merge under 8-connectivity.
#     - Path interiors are kept off sprites, off reserved endpoints, and off other path
#       cells. Collision pairs may overlap only at their common destination.
#     - Hit events target an already placed sprite cell exactly at step 5; the earlier
#       four path cells remain background.
#   visual_observability:
#     - The repeated 4,4,6,6 banding makes the projectile length rule readable at a
#       glance across several orientations.
#     - The collision endpoint color is tied to a visible geometric family already
#       present in the input: horizontal/vertical V-shots versus diagonal L-shots.
#     - Because each sample shows two distinct collision families, horizontal blue(1)
#       is regularly paired with red(2) or green(3), and some samples show the
#       vertical/diagonal pair directly.
#   visual_cleanliness:
#     - The generator forbids trajectory crossings except at intentional shared endpoints.
#     - Collision markers stay isolated single cells, while hit sprites remain compact
#       3-cell recolors, so the two output mechanisms do not visually merge.
#   change_inferability:
#     - Humans only need infer two small mappings:
#         * distance bands: 1-2 -> 4, 3-4 -> 6
#         * shared-endpoint family: horizontal -> 1, vertical -> 2, diagonal -> 3
#       Both mappings repeat within each dataset and are reinforced by sprite geometry.

from __future__ import annotations

import random
import sys
from collections import defaultdict

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4: dict[tuple[int, int], int] = {}
    deg8: dict[tuple[int, int], int] = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        corner = corners[0]
        cr, cc = corner
        arms: list[tuple[int, int]] = []
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


H = W = 16
BG = 7
SPRITE_COLORS = [5, 8]

PROJECTILE_DIST = 5
NEAR_COLOR = 4
MID_COLOR = 6
HIT_COLOR = 9
STEP_TO_COLOR = {1: NEAR_COLOR, 2: NEAR_COLOR, 3: MID_COLOR, 4: MID_COLOR}
COLLISION_COLORS = {
    "horizontal": 1,
    "vertical": 2,
    "diagonal": 3,
}

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
COLLISION_KINDS = ("horizontal", "vertical", "diagonal")
NON_HORIZONTAL_COLLISION_KINDS = ("vertical", "diagonal")

SPRITE_COUNT_RANGE = (5, 8)
MAX_HIT_EVENTS = 2
MIN_COLL_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 500
COLLISION_PAIR_OPTIONS = (
    ("horizontal", "vertical"),
    ("horizontal", "diagonal"),
    ("vertical", "diagonal"),
)


def _dir_kind(direction: tuple[int, int]) -> str:
    dr, dc = direction
    if dr == 0:
        return "horizontal"
    if dc == 0:
        return "vertical"
    return "diagonal"


def _path_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, PROJECTILE_DIST + 1))


def _empty_endpoint_color(directions: list[tuple[int, int]]) -> int:
    if len(directions) == 1:
        return HIT_COLOR
    kinds = {_dir_kind(direction) for direction in directions}
    if len(kinds) == 1:
        return COLLISION_COLORS[next(iter(kinds))]
    return HIT_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites and reveal both distance bands and direction-coded collisions.

    Input structure:
    - The background is the most common color.
    - Every non-background 8-connected 3-pixel component is a monochrome triangle sprite,
      either an axis-aligned L or a diagonal V.

    Rule:
    - Each sprite contributes one 5-step projectile determined by its geometry: L-shapes fire
      diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite
      their opening.
    - Steps 1-2 of every trajectory are painted color 4 and steps 3-4 are painted color 6.
    - Step 5 is the collision cell. If it hits a sprite, that whole sprite is recolored to 9.
      If it lands on empty background alone, the endpoint becomes 9. If multiple projectiles
      share the same empty endpoint, that cell is colored by their common direction family:
      horizontal -> 1, vertical -> 2, diagonal -> 3.

    Key insight: The sprite orientation still determines the ray, but the output now uses two
    coupled color codes: distance bands on the first four cells and direction-family coding only
    at shared empty endpoints.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    dest_to_shots: dict[tuple[int, int], list[tuple[int, tuple[int, int]]]] = defaultdict(list)
    rays: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        apex, direction = ad
        path = _path_cells(apex, direction)
        rr, cc = path[-1]
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        rays.append((cid, path))
        dest_to_shots[path[-1]].append((cid, direction))

    out = g.copy()
    for _cid, path in rays:
        for step, (r, c) in enumerate(path[:-1], start=1):
            out[r, c] = STEP_TO_COLOR[step]

    hit_components: set[int] = set()
    for dest in dest_to_shots:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for dest, shots in dest_to_shots.items():
        if dest in cell_to_comp:
            continue
        out[dest] = _empty_endpoint_color([direction for _, direction in shots])

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


def _build_placements() -> tuple[
    list[dict],
    dict[tuple[int, int], list[dict]],
    dict[tuple[tuple[int, int], str], list[dict]],
    dict[str, list[tuple[int, int]]],
]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_dest_kind: dict[tuple[tuple[int, int], str], list[dict]] = defaultdict(list)
    dests_by_kind: dict[str, set[tuple[int, int]]] = {kind: set() for kind in COLLISION_KINDS}
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                path = _path_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(path):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dir_kind": _dir_kind((dr, dc)),
                    "dest": path[-1],
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "path": path,
                    "path_set": frozenset(path),
                    "interior": path[:-1],
                    "interior_set": frozenset(path[:-1]),
                }
                placements.append(placement)
                by_dest[path[-1]].append(placement)
                by_dest_kind[(path[-1], placement["dir_kind"])].append(placement)
                dests_by_kind[placement["dir_kind"]].add(path[-1])
    return placements, by_dest, by_dest_kind, {kind: list(dests) for kind, dests in dests_by_kind.items()}


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_DEST_KIND, DESTS_BY_KIND = _build_placements()


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _sprite_fits(
    placement: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
) -> bool:
    if placement["halo"] & occupied_sprites:
        return False
    if placement["cell_set"] & occupied_paths:
        return False
    return True


def _placement_fits_bg(
    placement: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if not _sprite_fits(placement, occupied_sprites, occupied_paths):
        return False
    if placement["path_set"] & occupied_paths:
        return False
    if placement["path_set"] & occupied_sprites:
        return False
    if placement["path_set"] & reserved_bg:
        return False
    return True


def _placement_fits_collision(
    placement: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if not _sprite_fits(placement, occupied_sprites, occupied_paths):
        return False
    if placement["dest"] in occupied_sprites or placement["dest"] in occupied_paths or placement["dest"] in reserved_bg:
        return False
    if placement["interior_set"] & occupied_paths:
        return False
    if placement["interior_set"] & occupied_sprites:
        return False
    if placement["interior_set"] & reserved_bg:
        return False
    return True


def _placement_fits_hit(
    placement: dict,
    target_cell: tuple[int, int],
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if placement["dest"] != target_cell:
        return False
    if not _sprite_fits(placement, occupied_sprites, occupied_paths):
        return False
    if placement["interior_set"] & occupied_paths:
        return False
    if placement["interior_set"] & occupied_sprites:
        return False
    if placement["interior_set"] & reserved_bg:
        return False
    if target_cell in occupied_paths:
        return False
    return True


def _choose_unique(
    rng: random.Random,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg:
            continue
        if _placement_fits_bg(placement, occupied_sprites, occupied_paths, reserved_bg):
            return placement
    return None


def _choose_collision_pair(
    rng: random.Random,
    collision_kind: str,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = DESTS_BY_KIND[collision_kind][:]
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied_sprites or dest in occupied_paths:
            continue
        candidates = [
            placement
            for placement in PLACEMENTS_BY_DEST_KIND[(dest, collision_kind)]
            if _placement_fits_collision(placement, occupied_sprites, occupied_paths, reserved_bg)
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & first["cell_set"]:
                    continue
                if first["halo"] & second["cell_set"]:
                    continue
                if second["interior_set"] & first["cell_set"]:
                    continue
                if first["interior_set"] & second["cell_set"]:
                    continue
                overlap = first["path_set"] & second["path_set"]
                if overlap not in (set(), {dest}):
                    continue
                return dest, [first, second]
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [
            placement
            for placement in PLACEMENTS_BY_DEST[cell]
            if _placement_fits_hit(placement, cell, occupied_sprites, occupied_paths, reserved_bg)
        ]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
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


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    e_coll = 2
    if n_sprites < 2 * e_coll:
        return None
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 2 * e_coll) + 1):
        unique = n_sprites - e_hit - 2 * e_coll
        if unique < 0:
            continue
        if unique + e_hit < 1:
            continue
        options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _plan_collision_kinds(rng: random.Random, e_coll: int) -> list[str]:
    if e_coll == 2:
        return list(rng.choice(COLLISION_PAIR_OPTIONS))
    kinds = list(rng.choice(COLLISION_PAIR_OPTIONS))
    while len(kinds) < e_coll:
        unused = [kind for kind in COLLISION_KINDS if kind not in kinds]
        if unused:
            kinds.append(rng.choice(unused))
        else:
            kinds.append(rng.choice(COLLISION_KINDS))
    rng.shuffle(kinds)
    return kinds


def _place_collisions(
    rng: random.Random,
    collision_kinds: list[str],
    sprites: list[dict],
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for collision_kind in collision_kinds:
        choice = _choose_collision_pair(rng, collision_kind, occupied_sprites, occupied_paths, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append(
                {
                    "placement": placement,
                    "target": dest,
                    "target_kind": "bg",
                    "dir_kind": placement["dir_kind"],
                }
            )
            occupied_sprites.update(placement["cells"])
            occupied_paths.update(placement["path"])
    return True


def _place_unique_landings(
    rng: random.Random,
    n: int,
    sprites: list[dict],
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied_sprites, occupied_paths, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append(
            {
                "placement": placement,
                "target": placement["dest"],
                "target_kind": "bg",
                "dir_kind": placement["dir_kind"],
            }
        )
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
    return True


def _place_hits(
    rng: random.Random,
    e_hit: int,
    sprites: list[dict],
    occupied_sprites: set[tuple[int, int]],
    occupied_paths: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied_sprites, occupied_paths, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append(
            {
                "placement": placement,
                "target": cell,
                "target_kind": "hit",
                "target_sprite": idx,
            }
        )
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    hit_sprite_ids = {sprite["target_sprite"] for sprite in sprites if sprite["target_kind"] == "hit"}
    for sprite in sprites:
        placement = sprite["placement"]
        for step, (r, c) in enumerate(placement["interior"], start=1):
            out[r, c] = STEP_TO_COLOR[step]
        if sprite["target_kind"] == "bg":
            bg_targets[sprite["target"]].append(placement["dir"])
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, directions in bg_targets.items():
        out[dest] = _empty_endpoint_color(directions)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts
    collision_kinds = _plan_collision_kinds(rng, e_coll)

    sprites: list[dict] = []
    occupied_sprites: set[tuple[int, int]] = set()
    occupied_paths: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, collision_kinds, sprites, occupied_sprites, occupied_paths, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied_sprites, occupied_paths, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied_sprites, occupied_paths, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(rng, remaining, sprites, occupied_sprites, occupied_paths, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)

    if np.array_equal(inp, out):
        return None
    if not np.any(out == NEAR_COLOR) or not np.any(out == MID_COLOR):
        return None
    if not np.any((inp == BG) & np.isin(out, [COLLISION_COLORS["vertical"], COLLISION_COLORS["diagonal"]])):
        return None
    if not np.any(out == HIT_COLOR):
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
