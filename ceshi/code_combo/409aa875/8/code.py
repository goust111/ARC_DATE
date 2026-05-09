# Combo variant of 409aa875 (index 8)
# combo_spec: D1/0+D2/1+D3/0
# Name: 统一轴向射线+形状决定距离+箭头精灵

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg

_bg_color = get_bg


# Design analysis
# ---------------
# concretization_and_trace:
#   - The fused task keeps the original canvas/event logic but changes the source objects.
#   - Every sprite is now a 3-pixel arrow that points directly along one of the 4 cardinal
#     directions, so direction is read from the visible tip instead of from an L/V "gap".
#   - Distance is no longer global. We keep the old two-way range split, but it is now
#     encoded by arrow family:
#       * kind "L": skew hook arrow, range 4
#       * kind "V": symmetric chevron arrow, range 6
#   - Local trace:
#       * original cells {(5,5),(6,5),(5,6)} formed an L, fired diagonally to (0,0).
#       * fused analogue can be a short-range up arrow {(5,5),(6,4),(6,5)}.
#         It points upward directly, so it lands at (1,5) = tip + 4*(-1,0).
#       * a long-range up arrow {(5,5),(6,4),(6,6)} points the same way but lands at
#         (-1,5) conceptually, so the generator only places such arrows where tip+6*dir stays
#         in bounds. The visible difference between the two families is therefore not color but
#         arrow silhouette plus the induced landing distance.
# complexity_calibration:
#   - The fusion is not simplified into a single arrow template: there are two reusable shape
#     families and four directions, so solvers must infer both orientation and family-specific
#     range.
#   - The family split has only two distances (4 and 6), which is small enough to learn from
#     examples without introducing arbitrary numeric freedom.
#   - Generator examples force both arrow families to appear and reject cases whose output is
#     identical to a fixed-5 interpretation, so the distance dimension is actually exercised.
# implicit_assumptions_review:
#   - Component parsing: main() assumes every actionable foreground object is a 3-cell,
#     monochrome, 8-connected arrow matching exactly one template. The generator uses only
#     those templates and validates them with _arrow_info(...).
#   - Component separation: cell_to_comp assumes components are disjoint. Halo spacing keeps
#     Chebyshev distance >= 2, so no two sprites merge under 8-connectivity.
#   - Range lookup: main() assumes parsed kinds are only {"L","V"} before applying 4 or 6.
#     _arrow_info returns only those labels, and generator placements are precomputed by kind.
#   - Landing logic: hit detection still uses destinations computed from the untouched input.
#     That remains sound because recoloring never changes target geometry during resolution.
# placement_legality:
#   - Placements are precomputed only when both the arrow cells and tip+range*dir destination
#     are in bounds.
#   - Reserved background landing cells prevent accidental bg-target reuse except for deliberate
#     collision events, and occupied/halo checks prevent overlap and near-touching sprites.
# visual_observability:
#   - All arrows are axis-aligned, so the D1 change is visible directly in the input.
#   - The two arrow silhouettes are visually distinct, and both appear repeatedly, so the D2
#     distance split has support instead of depending on a one-off exception.
#   - Input colors avoid 1 and 9 so blue collision marks and maroon hit marks stay legible.
# cleanliness_and_inference:
#   - Output markers remain single cells and hit recoloring remains whole-sprite, so the grid
#     stays sparse and readable.
#   - No rule depends on hidden ordering or color ranking; only local shape family, arrow tip,
#     and landing multiplicity matter.


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
KINDS = ("L", "V")
RANGE_BY_KIND = {"L": 4, "V": 6}
FIXED5_BY_KIND = {"L": 5, "V": 5}

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 600
MIN_L_SPRITES = 2
MIN_V_SPRITES = 2
MIN_VERTICAL_SPRITES = 1
MIN_HORIZONTAL_SPRITES = 1


ARROW_OFFSETS_BY_KIND_DIR: dict[str, dict[tuple[int, int], tuple[tuple[int, int], ...]]] = {
    "L": {
        (-1, 0): ((0, 0), (1, -1), (1, 0)),
        (1, 0): ((0, 0), (-1, 0), (-1, 1)),
        (0, -1): ((0, 0), (0, 1), (1, 1)),
        (0, 1): ((0, 0), (-1, -1), (0, -1)),
    },
    "V": {
        (-1, 0): ((0, 0), (1, -1), (1, 1)),
        (1, 0): ((0, 0), (-1, -1), (-1, 1)),
        (0, -1): ((0, 0), (-1, 1), (1, 1)),
        (0, 1): ((0, 0), (-1, -1), (1, -1)),
    },
}


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _arrow_info(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    """Return (tip_rc, dir_rc, kind) for a 3-cell axis arrow sprite, else None."""
    if len(comp) != 3:
        return None
    cell_set = set(comp)
    matches: list[tuple[tuple[int, int], tuple[int, int], str]] = []
    for tip in comp:
        tr, tc = tip
        for kind in KINDS:
            for direction, offsets in ARROW_OFFSETS_BY_KIND_DIR[kind].items():
                shaped = {(tr + dr, tc + dc) for dr, dc in offsets}
                if shaped == cell_set:
                    matches.append((tip, direction, kind))
    if len(matches) != 1:
        return None
    return matches[0]


def _solve_with_ranges(grid: np.ndarray, range_by_kind: dict[str, int]) -> np.ndarray:
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

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        info = _arrow_info(comp)
        if info is None:
            continue
        (tr, tc), (dr, dc), kind = info
        dist = int(range_by_kind[kind])
        rr = tr + dist * dr
        cc = tc + dist * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = HIT_COLOR

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire straight projectiles from 3-pixel arrow sprites, with range determined by arrow family.

    Input structure:
    - The background is the most common color, and each foreground component is a 3-cell monochrome arrow.
    - Every arrow points directly up, down, left, or right from a single tip cell.
    - There are two arrow silhouettes: a skew hook family and a symmetric chevron family.

    Rule:
    - Read the firing direction from the arrow tip itself; all projectiles travel along cardinal rays only.
    - The arrow family sets the distance: hook arrows (the former L-coded family) travel 4 cells, while chevron arrows (the former V-coded family) travel 6 cells.
    - If a landing cell lies on a sprite, recolor that entire hit sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, color that cell blue (1); otherwise a lone empty-cell landing becomes maroon (9).

    Key insight:
    - Direction has become visually explicit, so the remaining inference is to separate the two arrow families and notice that they keep different fixed ranges even though they shoot along the same four axes.
    """
    return _solve_with_ranges(grid, RANGE_BY_KIND)


def _sprite_cells(kind: str, tip: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    tr, tc = tip
    return tuple((tr + dr, tc + dc) for dr, dc in ARROW_OFFSETS_BY_KIND_DIR[kind][direction])


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
    for kind in KINDS:
        dist = RANGE_BY_KIND[kind]
        for tr in range(H):
            for tc in range(W):
                for dr, dc in AXIS_DIRS:
                    dest = (tr + dist * dr, tc + dist * dc)
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                        continue
                    cells = _sprite_cells(kind, (tr, tc), (dr, dc))
                    if not _in_bounds_cells(cells):
                        continue
                    placement = {
                        "kind": kind,
                        "range": dist,
                        "tip": (tr, tc),
                        "dir": (dr, dc),
                        "dest": dest,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                    }
                    placements.append(placement)
                    by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


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
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
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
        if _fits(placement, occupied, reserved_bg):
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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(_arrow_info(comp) is not None for comp in comps)


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
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _family_mix_ok(sprites: list[dict]) -> bool:
    counts = Counter(s["placement"]["kind"] for s in sprites)
    return counts["L"] >= MIN_L_SPRITES and counts["V"] >= MIN_V_SPRITES


def _axis_mix_ok(sprites: list[dict]) -> bool:
    vertical = sum(1 for s in sprites if s["placement"]["dir"][0] != 0)
    horizontal = len(sprites) - vertical
    return vertical >= MIN_VERTICAL_SPRITES and horizontal >= MIN_HORIZONTAL_SPRITES


def _build_case() -> dict | None:
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

    if not _family_mix_ok(sprites):
        return None
    if not _axis_mix_ok(sprites):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    fixed5_out = _solve_with_ranges(inp, FIXED5_BY_KIND)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, fixed5_out):
        return None

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    return {
        "input": inp,
        "output": out,
        "sprites": sprites,
        "fixed5_output": fixed5_out,
    }


def _gen_attempt() -> np.ndarray | None:
    case = _build_case()
    return None if case is None else case["input"]


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
