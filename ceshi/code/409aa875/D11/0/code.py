# Variant of 409aa875
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 单轮同时处理：所有精灵同时发射，落点同时计算碰撞结果... → 按距离排序的顺序发射

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D11 variant
# -----------------------------------
# variant_spec:
#   core_change:
#     - Keep the same 16x16 canvas, the same 3-pixel triangle sprites, and the same
#       "apex + 5 * direction" projectile geometry.
#     - Replace simultaneous resolution with an ordered simulation:
#       sprites fire from near the grid center to far from the center.
#     - A sprite that is hit before its own turn is recolored to 9 immediately and
#       loses its future shot. This is the stateful step that makes order matter.
#   center_distance:
#     - Use the apex as the sprite's firing position.
#     - Distance key is squared Euclidean distance from the apex to the center
#       point (7.5, 7.5), encoded as:
#         (2*r - 15)^2 + (2*c - 15)^2
#       This avoids floats and preserves the true center ordering.
#     - Ties are broken by (row, col), but the generator avoids equal distance keys
#       so the user mainly sees a clean near-to-far ordering.
#   concrete_behavior_trace:
#     - Original behavior:
#       A hits B, B hits C, and all shooters fire anyway, so both B and C end up maroon.
#     - Variant behavior:
#       A is nearer to the center than B. A fires first, hits B, and B turns maroon.
#       When B's turn arrives, B is already disabled, so B does not fire and C stays
#       in the original sprite color. The missing effect on C is the visible D11 change.
#   complexity_calibration:
#     - Only one extra state bit is added: a sprite may become disabled before its turn.
#     - The generator always creates at least one A->B->C chain so the ordered rule is
#       visible, but it still keeps normal bg landings and sometimes shared bg landings
#       so the rest of the original task remains recognizable.
#   implicit_assumption_review:
#     - The original solver collected all destinations first, which assumed later shots
#       were unaffected by earlier hits. That assumption is no longer valid.
#     - A set of "hit components" was enough before; now we must process one actor at a
#       time and maintain a disabled set.
#     - The old bg collision logic was order-independent. It still works here, but only
#       for sprites that remain active long enough to shoot.
#   placement_legality:
#     - Sprites remain 8-connected 3-pixel components with Chebyshev separation >= 2.
#     - Every placement keeps its own sprite cells and landing cell in-bounds.
#     - Extra shooters are prevented from accidentally targeting chain sprites, so the
#       A->B suppression pattern survives generation.
#   visual_observability:
#     - Input sprite colors exclude 9, so "hit / disabled" sprites are visually distinct.
#     - The generator avoids equal center distances among sprites, so the ordering cue
#       can be read from geometry rather than hidden tie rules.
#     - Each sample contains at least one suppressed shot, and validation rejects inputs
#       whose ordered output matches the simultaneous baseline.
#   cleanliness_and_color_inference:
#     - Only the original sprite color, maroon 9, blue 1, and bg 7 appear.
#     - The output remains sparse: sprite recolors plus a few impact marks, with no
#       overlapping decorative structure that would obscure the rule.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 8)
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


def _apex_dist4(apex: tuple[int, int]) -> int:
    ar, ac = apex
    dr = 2 * ar - (H - 1)
    dc = 2 * ac - (W - 1)
    return dr * dr + dc * dc


def _parse_sprites(grid: np.ndarray) -> tuple[int, list[list[tuple[int, int]]], dict[tuple[int, int], int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    actors: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            (ar, ac), (dr, dc) = ad
            rr = ar + PROJECTILE_DIST * dr
            cc = ac + PROJECTILE_DIST * dc
            if 0 <= rr < h and 0 <= cc < w:
                actors.append(
                    {
                        "cid": cid,
                        "apex": (ar, ac),
                        "dest": (rr, cc),
                        "order_key": (_apex_dist4((ar, ac)), ar, ac),
                    }
                )
    actors.sort(key=lambda actor: actor["order_key"])
    return bg, comps, cell_to_comp, actors


def _run_ordered(grid: np.ndarray, *, return_meta: bool = False):
    g = np.asarray(grid, dtype=int)
    bg, comps, cell_to_comp, actors = _parse_sprites(g)
    out = g.copy()
    disabled: set[int] = set()
    bg_counts: dict[tuple[int, int], int] = defaultdict(int)
    meta = {"fired": 0, "skipped": 0, "sprite_hits": 0, "bg_repeat_hits": 0}

    for actor in actors:
        cid = actor["cid"]
        if cid in disabled:
            meta["skipped"] += 1
            continue
        meta["fired"] += 1
        dest = actor["dest"]
        if dest in cell_to_comp:
            target_cid = cell_to_comp[dest]
            meta["sprite_hits"] += 1
            disabled.add(target_cid)
            for r, c in comps[target_cid]:
                out[r, c] = HIT_COLOR
            continue
        bg_counts[dest] += 1
        if bg_counts[dest] == 1:
            out[dest] = HIT_COLOR
        else:
            out[dest] = COLLISION_COLOR
            meta["bg_repeat_hits"] += 1

    return (out, meta) if return_meta else out


def _run_simultaneous(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _bg, comps, cell_to_comp, actors = _parse_sprites(g)
    out = g.copy()
    dest_to_count: dict[tuple[int, int], int] = defaultdict(int)
    hit_components: set[int] = set()

    for actor in actors:
        dest = actor["dest"]
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])
        else:
            dest_to_count[dest] += 1

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR
    for dest, count in dest_to_count.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle-sprite projectiles in center-out order, with immediate state updates.

    Input structure:
    - bg is the most common color. Every non-bg 3-pixel component is an axis-aligned L or a diagonal V, and all such components are candidate shooters.

    Rule:
    - Read each triangle's apex and firing direction from its shape exactly as in the base task, and compute the landing cell 5 steps away.
    - Sort sprites by apex distance to the grid center from nearest to farthest, then process them one at a time.
    - When a sprite is hit before its own turn, it is recolored to maroon (9) immediately and loses its future shot.
    - An active shot that lands on a sprite recolors that whole sprite to 9; an active shot that lands on empty bg paints 9 on first impact and 1 after a repeated impact on that same bg cell.

    Key insight: The geometry still determines where every projectile wants to go, but now the early inner sprites can disable outer shooters before those outer shooters ever act.
    """
    return _run_ordered(grid)


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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    by_cell: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(H):
        for ac in range(W):
            apex = (ar, ac)
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells(apex, (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                placement = {
                    "apex": apex,
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "dist4": _apex_dist4(apex),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                for cell in cells:
                    by_cell[cell].append(placement)
    return placements, by_dest, by_cell


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_BY_CELL = _build_placements()


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dist4: set[int],
) -> bool:
    if placement["dist4"] in used_dist4:
        return False
    if placement["halo"] & occupied:
        return False
    if placement["cell_set"] & reserved_bg:
        return False
    return True


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_chain(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dist4: set[int],
) -> list[dict] | None:
    b_candidates = PLACEMENTS[:]
    rng.shuffle(b_candidates)
    for b in b_candidates:
        if not _fits(b, occupied, reserved_bg, used_dist4):
            continue
        bc_occ = occupied | set(b["cells"])
        bc_dist = used_dist4 | {b["dist4"]}
        c_candidates = [
            c
            for c in PLACEMENTS_BY_CELL[b["dest"]]
            if c["dist4"] > b["dist4"]
            and _fits(c, bc_occ, reserved_bg, bc_dist)
            and c["dest"] not in bc_occ
            and c["dest"] not in reserved_bg
        ]
        rng.shuffle(c_candidates)
        for c in c_candidates:
            bcc_occ = bc_occ | set(c["cells"])
            bcc_dist = bc_dist | {c["dist4"]}
            a_candidates = []
            b_cells = list(b["cells"])
            rng.shuffle(b_cells)
            for cell in b_cells:
                for a in PLACEMENTS_BY_DEST[cell]:
                    if a["dist4"] >= b["dist4"]:
                        continue
                    if _fits(a, bcc_occ, reserved_bg, bcc_dist):
                        a_candidates.append(a)
            rng.shuffle(a_candidates)
            if a_candidates:
                return [a_candidates[0], b, c]
    return None


def _choose_unique(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dist4: set[int],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in occupied or placement["dest"] in reserved_bg:
            continue
        if _fits(placement, occupied, reserved_bg, used_dist4):
            return placement
    return None


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    used_dist4: set[int],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST.keys())
    rng.shuffle(dests)
    for dest in dests:
        if dest in occupied or dest in reserved_bg:
            continue
        candidates = [
            p
            for p in PLACEMENTS_BY_DEST[dest]
            if _fits(p, occupied, reserved_bg, used_dist4)
        ]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            dist2 = used_dist4 | {first["dist4"]}
            for second in candidates[i + 1 :]:
                if second["dist4"] in dist2:
                    continue
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    cols = sorted(set(int(v) for v in np.unique(grid)) - {BG})
    if len(cols) != 1 or cols[0] == HIT_COLOR:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    used_dist4: set[int] = set()

    chain = _choose_chain(rng, occupied, reserved_bg, used_dist4)
    if chain is None:
        return None
    for placement in chain:
        sprites.append({"placement": placement})
        occupied.update(placement["cells"])
        used_dist4.add(placement["dist4"])
    reserved_bg.add(chain[2]["dest"])

    remaining = n_sprites - len(sprites)
    if remaining < 1:
        return None

    if remaining >= 2 and rng.random() < 0.55:
        choice = _choose_collision_pair(rng, occupied, reserved_bg, used_dist4)
        if choice is not None:
            dest, pair = choice
            reserved_bg.add(dest)
            for placement in pair:
                sprites.append({"placement": placement})
                occupied.update(placement["cells"])
                used_dist4.add(placement["dist4"])
            remaining -= 2

    while remaining > 0:
        placement = _choose_unique(rng, occupied, reserved_bg, used_dist4)
        if placement is None:
            return None
        sprites.append({"placement": placement})
        occupied.update(placement["cells"])
        used_dist4.add(placement["dist4"])
        reserved_bg.add(placement["dest"])
        remaining -= 1

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out, meta = _run_ordered(inp, return_meta=True)
    sim = _run_simultaneous(inp)

    if meta["skipped"] < 1:
        return None
    if np.array_equal(out, inp):
        return None
    if np.array_equal(out, sim):
        return None
    if np.count_nonzero(out != sim) < 3:
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
