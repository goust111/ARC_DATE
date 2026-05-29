# Variant of 409aa875
# Dimension: D6 — 轨迹涂色模式 (trajectory)
# Change: 仅在落点涂色（无飞行轨迹），落点颜色取决于碰撞类型... → 间歇轨迹（每隔一格涂色）

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, get_bg, neighbors4, neighbors8


# Design analysis for the D6 trajectory variant
# ---------------------------------------------
# spec:
#   variant_core:
#     base_rule_kept:
#       - Triangle detection is unchanged: every 3-cell monochrome 8-connected object is either
#         an axis-aligned L or a diagonal V.
#       - The apex and firing direction are still inferred from that shape.
#       - The projectile still lands exactly 5 cells away from the apex.
#       - The landing semantics are preserved:
#         sprite hit -> recolor the whole hit sprite to 9
#         repeated empty landing -> color the landing cell 1
#         single empty landing -> color the landing cell 9
#     new_trajectory_rule:
#       - The projectile path is now partially visible.
#       - For the 5-cell ray from apex to landing, odd-distance cells are the visible samples.
#       - Distances 1 and 3 are painted as maroon trail dots when they are background.
#       - Distance 5 is still the landing cell, so it is colored by the original impact rule rather
#         than by a generic trail overwrite.
#   behavior_trace:
#     original:
#       - An L with apex at (8,8) firing up-left only changed cell (3,3), or recolored a sprite
#         containing (3,3). The 4 cells between them were invisible.
#     variant:
#       - The same shooter now also paints (7,7) and (5,5), creating a dashed diagonal. The cell
#         (3,3) remains the impact site, so it is maroon for a lone landing or sprite hit, but blue
#         if multiple projectiles share that empty destination.
#     difference:
#       - The rule change happens in output rendering, not in sprite parsing or direction inference.
#       - Generator legality must therefore be extended from "destination is feasible" to
#         "destination and visible trail samples are feasible and readable".
#   complexity_calibration:
#     - Using odd steps 1,3,5 yields three visible samples per projectile, which is enough for a
#       human to notice a dashed line after only a few examples.
#     - The color vocabulary stays small and structured: maroon for trail / single-hit effects,
#       blue only for repeated empty destinations.
#     - The solver does not introduce new hidden categories based on color ids or ordering.
#   implicit_assumptions_review:
#     - Original main() only cared about the destination map; that was safe because intermediate
#       cells were invisible. In this variant, blindly painting every odd-step cell would be wrong
#       if those cells overlapped sprites or special destinations.
#     - The updated main() therefore paints mid-trail dots only on background cells and lets the
#       terminal landing logic run afterwards so blue destinations can override a generic maroon
#       trail mark if needed.
#     - Original generator only reserved the landing cell. That is no longer enough because a later
#       sprite could occupy a visible trail dot, making the new D6 behavior disappear. The updated
#       generator reserves every visible background mark in advance.
#   placement_legality:
#     - Every sprite must stay in-bounds and remain a separate 3-cell 8-connected component.
#     - Every visible mid-trail cell (steps 1 and 3) must remain background in the input.
#     - For background landings, the destination cell must also remain background.
#     - Shared background destinations are allowed only for explicit collision events; other trail
#       overlaps are rejected to keep the outputs legible.
#   visual_observability:
#     - The new signal is a dashed segment with fixed stride 2, which is directly visible in both
#       diagonal and axis-aligned shots.
#     - Because each grid still contains several shooters, repeated examples expose the "odd cells
#       along a 5-step ray" pattern without needing hidden bookkeeping.
#   cleanliness:
#     - The generator avoids using sprite cells as trail cells and avoids accidental overlap between
#       unrelated trail marks, so the output does not devolve into a dense maroon cloud.
#   color_inference:
#     - Only two effect colors remain in play: 9 for normal trail / lone landing / hit sprites, 1
#       for repeated empty landing cells. This preserves the original color semantics and makes the
#       D6 change purely spatial.
#   new_main_assumptions_checked_against_generator:
#     - main() assumes any visible mid-trail dot should be background in the input for the intended
#       samples. generate_input() enforces that by reserving those cells before later placements.
#     - main() assumes landing destinations are still exactly 5 steps away. generate_input() uses
#       the same precomputed placements, so the solver and generator stay aligned.


TRAIL_STEPS = (1, 3, 5)
MID_TRAIL_STEPS = TRAIL_STEPS[:-1]


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


def _trajectory_cells(apex: tuple[int, int], direction: tuple[int, int], steps: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in steps)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit dashed projectiles from 3-cell triangles while keeping the original impact rules.

    Input structure:
    - The background is the most frequent color; every non-background monochrome 8-connected component is a 3-cell triangle, either an L or a diagonal V.

    Rule:
    - Infer each triangle's apex and firing direction from its shape exactly as in the base task: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite their opening.
    - Each projectile still travels 5 cells from the apex, but the odd-distance samples of that flight become visible: steps 1 and 3 are painted maroon (9) whenever those cells are background.
    - Step 5 remains the impact cell, so it keeps the original terminal behavior: a hit on a sprite recolors the whole sprite to maroon, repeated landings on the same empty background cell color that cell blue (1), and a lone empty landing colors it maroon.
    - If a terminal landing coincides with a generic trail sample, the terminal color wins.

    Key insight: The triangle orientation still determines the full ray, but the output now reveals that ray as a dashed odd-step trail instead of showing only the endpoint.
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

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    mid_trail_cells: set[tuple[int, int]] = set()
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        apex, direction = ad
        trail = _trajectory_cells(apex, direction, TRAIL_STEPS)
        (rr, cc) = trail[-1]
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        dest_to_shooters[(rr, cc)].append(cid)
        for tr, tc in trail[:-1]:
            if 0 <= tr < h and 0 <= tc < w and int(g[tr, tc]) == int(bg):
                mid_trail_cells.add((tr, tc))

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for r, c in mid_trail_cells:
        out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


# --------------- generator ---------------

H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 600


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
                apex = (ar, ac)
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells(apex, (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                trail = _trajectory_cells(apex, (dr, dc), TRAIL_STEPS)
                placement = {
                    "apex": apex,
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "mid_trail": frozenset(trail[:-1]),
                    "bg_marks": frozenset(trail),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    bg_marks: set[tuple[int, int]],
    shared_bg: set[tuple[int, int]] | None = None,
) -> bool:
    shared_bg = shared_bg or set()
    if placement["halo"] & occupied:
        return False
    if placement["cell_set"] & reserved_bg:
        return False
    effective_marks = set(bg_marks) - set(shared_bg)
    if effective_marks & occupied:
        return False
    if effective_marks & reserved_bg:
        return False
    return True


def _bg_marks_for(sprite: dict) -> set[tuple[int, int]]:
    if sprite["target_kind"] == "bg":
        return set(sprite["placement"]["bg_marks"])
    return set(sprite["placement"]["mid_trail"])


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [
            p for p in PLACEMENTS_BY_DEST[dest]
            if _fits(p, occupied, reserved_bg, set(p["bg_marks"]))
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            reserved2 = reserved_bg | set(first["mid_trail"]) | {dest}
            for second in candidates[i + 1:]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if not _fits(second, occ2, reserved2, set(second["bg_marks"]), shared_bg={dest}):
                    continue
                return dest, [first, second]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg, set(placement["bg_marks"])):
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
        candidates = [
            p for p in PLACEMENTS_BY_DEST[cell]
            if _fits(p, occupied, reserved_bg, set(p["mid_trail"]))
        ]
        if candidates:
            return rng.choice(candidates), cell
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
    for e_hit in range(0, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    return rng.choice(options) if options else None


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprite = {"placement": placement, "target": dest, "target_kind": "bg"}
            sprites.append(sprite)
            occupied.update(placement["cells"])
            reserved_bg.update(_bg_marks_for(sprite) - {dest})
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        occupied.update(placement["cells"])
        reserved_bg.update(_bg_marks_for(sprite))
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprite = {"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx}
        sprites.append(sprite)
        occupied.update(placement["cells"])
        reserved_bg.update(_bg_marks_for(sprite))
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}

    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)

    trail_cells: set[tuple[int, int]] = set()
    for sprite in sprites:
        trail_cells.update(sprite["placement"]["mid_trail"])
    for r, c in trail_cells:
        if out[r, c] == BG:
            out[r, c] = HIT_COLOR

    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg):
        return None

    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None

    extra_marks = np.count_nonzero((out == HIT_COLOR) & (inp == BG))
    if extra_marks < 2:
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
