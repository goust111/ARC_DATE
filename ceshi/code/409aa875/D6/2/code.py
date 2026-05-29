# Variant of 409aa875
# Dimension: D6 — 轨迹涂色模式 (trajectory)
# Change: 仅在落点涂色（无飞行轨迹），落点颜色取决于碰撞类型... → 颜色渐变轨迹

# Design analysis for the D6 variant
# ----------------------------------
# spec_variant:
#   core_change:
#     old: Only the landing cell was painted, and the landing color depended on collision type.
#     new: Every projectile paints its whole 5-cell flight path. Distances 1-2 use color 4,
#       distances 3-4 use color 6, and distance 5 uses color 9.
#   concrete_behavior:
#     input:
#       - Same 16x16 orange background (7).
#       - Same 3-pixel triangle sprites, all sharing one non-background color.
#       - No explicit trajectory hints are drawn in the input.
#     output:
#       - From each apex, a straight 5-step ray appears in the inferred firing direction.
#       - Cells at steps 1 and 2 become yellow (4), steps 3 and 4 become pink (6),
#         and the last step becomes maroon (9).
#       - If the last step lands on another sprite, that whole hit sprite becomes maroon (9),
#         which matches the step-5 color.
#   behavior_trace:
#     original_local_example:
#       - A shooter with apex A and direction d only modified A + 5d.
#       - If A + 5d was empty, exactly one cell changed.
#     variant_local_example:
#       - The same shooter now modifies A + d, A + 2d, A + 3d, A + 4d, A + 5d.
#       - The first two cells share one color band, the next two another band, and the
#         endpoint is a third color.
#     difference:
#       - The informative signal moves from endpoint-only collision color to a visible
#         distance-coded stripe pattern along the full ray.
#   complexity_calibration:
#     - The palette is intentionally limited to three fixed trail colors so the distance rule
#       is inferable from 2-4 examples.
#     - Sprite colors are restricted away from {4, 6, 9}; otherwise trail colors could visually
#       blend with sprites and make the new mechanism hard to see.
#     - The generator keeps both "hit sprite" and "land on empty bg" endpoints, but removes the
#       old blue collision branch because it would break the clean distance->color mapping.
#   implicit_assumptions_review:
#     - Original main() assumed the only visible consequence of a shot was at the destination.
#       That is no longer true; every step along the ray must now be materialized.
#     - Original dest_to_shooters counting supported a special blue multi-hit endpoint. The
#       variant no longer needs endpoint multiplicity because distance alone determines color.
#     - Recoloring a whole hit sprite is still safe because the generator guarantees that only
#       step 5 can touch a sprite; steps 1-4 stay on background.
#   placement_legality:
#     - Sprites still obey the original Chebyshev>=2 separation.
#     - The generator additionally prevents trajectory interiors from crossing sprites or other
#       trajectories, so the solver's straight-line interpretation remains unambiguous.
#     - Hit events are constructed so a shooter targets one of an existing sprite's three cells
#       exactly at step 5 and nowhere earlier.
#   visual_observability:
#     - Every generated grid contains several 5-step rays, so the repeated 2-2-1 color rhythm
#       is visible multiple times in different orientations.
#     - Because the stripe colors are fixed and the sprite palette excludes them, a human can
#       infer the distance mapping directly from examples.
#   cleanliness:
#     - Path overlap is disallowed in generation, which avoids messy overpainting and keeps each
#       ray visually legible as one independent trajectory.

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
SPRITE_COLORS = [0, 2, 3, 5, 8]

PROJECTILE_DIST = 5
NEAR_COLOR = 4
MID_COLOR = 6
END_COLOR = 9
STEP_TO_COLOR = {1: NEAR_COLOR, 2: NEAR_COLOR, 3: MID_COLOR, 4: MID_COLOR, 5: END_COLOR}

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 8)
MAX_HIT_EVENTS = 2
MAX_ATTEMPTS = 400


def _path_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, PROJECTILE_DIST + 1))


def _color_for_step(step: int) -> int:
    return STEP_TO_COLOR[int(step)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites and reveal a distance-coded color trail.

    Input structure:
    - The background is the most common color.
    - Every non-background 8-connected 3-pixel component is a monochrome triangle sprite,
      either an axis-aligned L or a diagonal V.

    Rule:
    - Each triangle determines an apex and a firing direction exactly as in the base task:
      L-shapes fire diagonally away from the missing 2x2 corner, and V-shapes fire along the
      axis opposite their opening.
    - The projectile still travels exactly 5 cells from the apex.
    - Instead of marking only the landing site, the whole 5-cell trajectory is painted:
      steps 1-2 use color 4, steps 3-4 use color 6, and step 5 uses color 9.
    - If step 5 lands on a sprite, that entire hit sprite is recolored to 9.

    Key insight: Orientation still determines the ray, but the visible evidence is now the
    repeated 2-2-1 color gradient along distance, not a collision-dependent endpoint marker.
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

    rays: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    hit_components: set[int] = set()
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
        if path[-1] in cell_to_comp:
            hit_components.add(cell_to_comp[path[-1]])

    out = g.copy()
    for _cid, path in rays:
        for step, (r, c) in enumerate(path, start=1):
            out[r, c] = _color_for_step(step)

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = END_COLOR

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
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
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
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _placement_fits_hit(p, cell, occupied_sprites, occupied_paths, reserved_bg)]
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


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites:
        for step, (r, c) in enumerate(sprite["placement"]["path"], start=1):
            out[r, c] = _color_for_step(step)
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], END_COLOR)
    return out


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int]:
    max_hits = min(MAX_HIT_EVENTS, n_sprites // 2)
    e_hit = rng.randint(1, max_hits)
    base_unique = rng.randint(e_hit, n_sprites - e_hit)
    return e_hit, base_unique


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    e_hit, base_unique = _counts_for_n(rng, n_sprites)

    sprites: list[dict] = []
    occupied_sprites: set[tuple[int, int]] = set()
    occupied_paths: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    for _ in range(base_unique):
        placement = _choose_unique(rng, occupied_sprites, occupied_paths, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        reserved_bg.add(placement["dest"])

    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(base_unique), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied_sprites, occupied_paths, reserved_bg, used_target_cells)
        if choice is None:
            return None
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        used_target_cells.add(cell)

    while len(sprites) < n_sprites:
        placement = _choose_unique(rng, occupied_sprites, occupied_paths, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied_sprites.update(placement["cells"])
        occupied_paths.update(placement["path"])
        reserved_bg.add(placement["dest"])

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)

    if np.array_equal(inp, out):
        return None
    if not np.any(out == NEAR_COLOR) or not np.any(out == MID_COLOR) or not np.any(out == END_COLOR):
        return None
    if np.count_nonzero(out == NEAR_COLOR) < 4 or np.count_nonzero(out == MID_COLOR) < 4:
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
