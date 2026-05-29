# Combo variant of 409aa875 (index 15)
# combo_spec: D2/1+D11/3
# Name: 形状决定距离+L先V后发

import random
from collections import defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the fused D2/1 + D11/3 variant
# ---------------------------------------------------
# a_concretization_and_trace:
#   - Visible objects are unchanged from the source task: each foreground object is a monochrome
#     3-cell triangle sprite, either an axis-aligned L or a diagonal V.
#   - The fusion changes two linked stages:
#       1. shape decides range: L -> 4 cells, V -> 6 cells
#       2. shape decides time order: all L-shots resolve first, then all V-shots resolve on the
#          already-updated board
#   - Concrete trace:
#       * L sprite with apex (5,5) and diagonal dir (-1,-1):
#         original dest = (0,0), fused dest = (1,1), and it resolves in phase 1
#       * V sprite with apex (7,6) and axis dir (-1,0):
#         original dest = (2,6), fused dest = (1,6), and it resolves in phase 2
#       * If one L and one V share the same fused empty destination X, phase 1 paints X=9, then
#         the V-shot sees X as occupied rather than empty background. Final X stays 9 instead of
#         becoming a simultaneous blue collision.
# b_complexity_calibration:
#   - The rule still has only two shape classes and two range values, so the lookup burden is small.
#   - To keep both dimensions learnable, the generator always includes:
#       * one mixed L/V shared destination that witnesses phased resolution, and
#       * one same-phase shared destination that still becomes blue, showing collisions did not vanish.
#   - Samples are rejected if the fused output matches either:
#       * a simultaneous solver using the same shape-based ranges, or
#       * a phased solver using the old fixed distance 5.
# c_implicit_assumptions_review:
#   - Original assumption: every shooter can be summarized into one global dest->shooters map.
#     Broken by D11/3, because V-shots must inspect the board after L results are written.
#   - Original assumption: every target uses `apex + 5 * dir`.
#     Broken by D2/1, because distance must be selected after classifying L vs V.
#   - Original assumption: hit detection only needs components from the untouched input.
#     Broken in the fused variant, because phase-1 marks become valid occupied components for phase 2.
#   - New solver assumption: actionable input objects are still 3-cell L/V sprites parseable by
#     `_triangle_info(...)`. The generator guarantees this and keeps sprite separation >= 2.
# d_placement_legality:
#   - Placements are precomputed only when both sprite cells and the fused landing cell are in bounds.
#   - Sprite halos enforce Chebyshev separation >= 2 so sprites never merge under 8-connectivity.
#   - Background impact cells reserve their own 8-neighborhood so phase-created 9/1 markers stay isolated.
# e_visual_observability:
#   - Ordering depends only on visible shape kind, never on hidden properties such as color ids.
#   - Both L and V are mandatory in every sample, and the mixed L/V destination is mandatory as a
#     direct witness that L fires first and V fires later.
#   - Sprite colors exclude 1 and 9 so output marks remain visually distinct.
# f_visual_cleanliness:
#   - Collision markers are kept apart from sprite halos and from each other, avoiding accidental
#     merged 9-components that would obscure which marks came from which event.
#   - The base construction uses only a few event types, keeping the output readable.
# g_change_inferability:
#   - Color semantics are unchanged from the source task: 9 marks lone hits or recolored hit
#     components, and 1 marks same-phase multi-hit empty cells.
#   - The new information to infer is structural: shape jointly determines direction family,
#     travel distance, and firing order.
# new_solver_assumptions_checked_against_generator:
#   - `_extract_sprites(...)` assumes the input grid contains only original sprites, not pre-painted
#     markers. The generator constructs exactly that.
#   - `_apply_batches(...)` assumes all shooters in a batch use destinations computed from the input
#     sprites, while occupancy is recomputed from the evolving board before each batch. This matches
#     the fused rule and is exercised by the mixed L/V witness.
#   - `_choose_hit(...)` assumes at least one existing sprite cell can be targeted by another legal
#     shooter when hit events are requested. If not, generation retries.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_info(
    comp: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], str] | None:
    """Return (apex_rc, dir_rc, kind) for a 3-cell triangle sprite."""
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
        return corner, (dr, dc), "L"

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
    return apex, (dr, dc), "V"


def _range_for_kind(kind: str, range_by_kind: dict[str, int]) -> int:
    return int(range_by_kind[kind])


def _current_components(
    grid: np.ndarray,
    bg: int,
) -> tuple[list[list[tuple[int, int]]], dict[tuple[int, int], int]]:
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid
    return comps, cell_to_comp


def _extract_sprites(grid: np.ndarray, bg: int, range_by_kind: dict[str, int]) -> list[dict]:
    h, w = grid.shape
    sprites: list[dict] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg}):
        for comp in _cc8(grid == col):
            info = _triangle_info(comp)
            if info is None:
                continue
            apex, direction, kind = info
            ar, ac = apex
            dr, dc = direction
            dist = _range_for_kind(kind, range_by_kind)
            dest = (ar + dist * dr, ac + dist * dc)
            if not (0 <= dest[0] < h and 0 <= dest[1] < w):
                continue
            sprites.append(
                {
                    "cells": tuple(comp),
                    "kind": kind,
                    "apex": apex,
                    "dir": direction,
                    "dest": dest,
                }
            )
    return sprites


def _apply_batches(grid: np.ndarray, batches: list[list[dict]]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = _bg_color(out)
    for shooters in batches:
        if not shooters:
            continue
        comps, cell_to_comp = _current_components(out, bg)
        dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
        for sid, sprite in enumerate(shooters):
            dest_to_shooters[sprite["dest"]].append(sid)

        hit_components: set[int] = set()
        for dest in dest_to_shooters:
            if dest in cell_to_comp:
                hit_components.add(cell_to_comp[dest])

        for cid in hit_components:
            for r, c in comps[cid]:
                out[r, c] = 9

        for (r, c), batch_shooters in dest_to_shooters.items():
            if (r, c) in cell_to_comp:
                continue
            out[r, c] = 1 if len(batch_shooters) >= 2 else 9
    return out


def _solve_simultaneous(grid: np.ndarray, range_by_kind: dict[str, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sprites = _extract_sprites(g, _bg_color(g), range_by_kind)
    return _apply_batches(g, [sprites])


def _solve_phased(grid: np.ndarray, range_by_kind: dict[str, int]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sprites = _extract_sprites(g, bg, range_by_kind)
    l_batch = [sprite for sprite in sprites if sprite["kind"] == "L"]
    v_batch = [sprite for sprite in sprites if sprite["kind"] == "V"]
    return _apply_batches(g, [l_batch, v_batch])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from tiny triangle sprites, with both range and timing determined by shape.

    Input structure:
    - The background is the most common color, and each foreground object is a monochrome 3-cell triangle sprite.
    - Sprites come in two visible families: axis-aligned L-shapes and diagonal V-shapes.

    Rule:
    - Shape still determines apex and firing direction, but now it also selects travel distance: L-shapes fire 4 cells along their diagonal direction, while V-shapes fire 6 cells along their axis direction.
    - Resolution is split into two batches: all L-shots are applied first and their sprite hits or background marks are written immediately; only then do all V-shots resolve against the updated board.
    - In either batch, landing on any current non-background component recolors that whole component to maroon (9); landing on empty background paints maroon (9) for one shooter or blue (1) for multiple shooters in that same batch.

    Key insight: Shape now encodes three things at once: direction family, shot range, and firing order. Mixed L/V overlaps no longer behave like simultaneous collisions because V-shots see the marks left behind by the L phase.
    """
    return _solve_phased(grid, {"L": 4, "V": 6})


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

RANGE_BY_KIND = {"L": 4, "V": 6}
FIXED5_BY_KIND = {"L": 5, "V": 5}
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS_BY_KIND = {"L": DIAG_DIRS, "V": AXIS_DIRS}

SPRITE_COUNT_RANGE = (5, 9)
MAX_HIT_EVENTS = 2
MAX_ATTEMPTS = 700


def _sprite_cells(kind: str, apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if kind == "L":
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


def _cell_halo(cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r, c = cell
    return frozenset(
        (rr, cc)
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for kind, dirs in DIRS_BY_KIND.items():
        dist = RANGE_BY_KIND[kind]
        for ar in range(H):
            for ac in range(W):
                for dr, dc in dirs:
                    dest = (ar + dist * dr, ac + dist * dc)
                    if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                        continue
                    cells = _sprite_cells(kind, (ar, ac), (dr, dc))
                    if not _in_bounds_cells(cells):
                        continue
                    fixed5_dest = (ar + 5 * dr, ac + 5 * dc)
                    placement = {
                        "kind": kind,
                        "range": dist,
                        "apex": (ar, ac),
                        "dir": (dr, dc),
                        "dest": dest,
                        "fixed5_dest": fixed5_dest if 0 <= fixed5_dest[0] < H and 0 <= fixed5_dest[1] < W else None,
                        "cells": cells,
                        "cell_set": frozenset(cells),
                        "halo": _halo(cells),
                    }
                    placements.append(placement)
                    by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg_halo: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg_halo)


def _bg_dest_is_clear(
    dest: tuple[int, int],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> bool:
    return dest not in occupied_halo and dest not in reserved_bg_halo


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _place_sprite(
    sprites: list[dict],
    placement: dict,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
) -> None:
    sprites.append({"placement": placement})
    occupied.update(placement["cells"])
    occupied_halo.update(placement["halo"])


def _reserve_bg_dest(
    dest: tuple[int, int],
    reserved_bg: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> None:
    reserved_bg.add(dest)
    reserved_bg_halo.update(_cell_halo(dest))


def _choose_mixed_collision(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if not _bg_dest_is_clear(dest, occupied_halo, reserved_bg_halo):
            continue
        l_candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == "L" and _fits(p, occupied, reserved_bg_halo)]
        v_candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == "V" and _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(l_candidates)
        rng.shuffle(v_candidates)
        for first in l_candidates:
            occ2 = occupied | set(first["cells"])
            for second in v_candidates:
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_same_kind_collision(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
    kind: str,
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if not _bg_dest_is_clear(dest, occupied_halo, reserved_bg_halo):
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if p["kind"] == kind and _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1 :]:
                if second["halo"] & occ2:
                    continue
                return dest, [first, second]
    return None


def _choose_unique_bg(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if not _bg_dest_is_clear(placement["dest"], occupied_halo, reserved_bg_halo):
            continue
        if _fits(placement, occupied, reserved_bg_halo):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    target_sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg_halo: set[tuple[int, int]],
    used_target_cells: set[tuple[int, int]],
) -> tuple[dict, tuple[int, int]] | None:
    cells = list(target_sprite["placement"]["cells"])
    rng.shuffle(cells)
    for cell in cells:
        if cell in used_target_cells:
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _fits(p, occupied, reserved_bg_halo)]
        rng.shuffle(candidates)
        if candidates:
            return candidates[0], cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    kinds = set()
    for comp in comps:
        info = _triangle_info(comp)
        if info is None:
            return False
        kinds.add(info[2])
    return kinds == {"L", "V"}


def _build_case() -> dict | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    # Always include at least one sprite-hit example so the recolor branch is visible in every task.
    hit_events = rng.randint(1, min(MAX_HIT_EVENTS, n_sprites - 4))

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    occupied_halo: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()
    reserved_bg_halo: set[tuple[int, int]] = set()

    mixed = _choose_mixed_collision(rng, occupied, occupied_halo, reserved_bg_halo)
    if mixed is None:
        return None
    mixed_dest, mixed_pair = mixed
    _reserve_bg_dest(mixed_dest, reserved_bg, reserved_bg_halo)
    for placement in mixed_pair:
        _place_sprite(sprites, placement, occupied, occupied_halo)

    same_kind = rng.choice(["L", "V"])
    same_phase = _choose_same_kind_collision(rng, occupied, occupied_halo, reserved_bg_halo, same_kind)
    if same_phase is None:
        return None
    same_phase_dest, same_phase_pair = same_phase
    _reserve_bg_dest(same_phase_dest, reserved_bg, reserved_bg_halo)
    for placement in same_phase_pair:
        _place_sprite(sprites, placement, occupied, occupied_halo)

    used_target_cells: set[tuple[int, int]] = set()
    for _ in range(hit_events):
        target_idx = rng.randrange(len(sprites))
        choice = _choose_hit(rng, sprites[target_idx], occupied, reserved_bg_halo, used_target_cells)
        if choice is None:
            return None
        placement, cell = choice
        _place_sprite(sprites, placement, occupied, occupied_halo)
        used_target_cells.add(cell)

    while len(sprites) < n_sprites:
        placement = _choose_unique_bg(rng, occupied, occupied_halo, reserved_bg_halo)
        if placement is None:
            return None
        _reserve_bg_dest(placement["dest"], reserved_bg, reserved_bg_halo)
        _place_sprite(sprites, placement, occupied, occupied_halo)

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = main(inp)
    simultaneous_same_range = _solve_simultaneous(inp, RANGE_BY_KIND)
    phased_fixed5 = _solve_phased(inp, FIXED5_BY_KIND)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(out == COLLISION_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, simultaneous_same_range):
        return None
    if np.array_equal(out, phased_fixed5):
        return None
    if out[mixed_dest] != HIT_COLOR:
        return None
    if out[same_phase_dest] != COLLISION_COLOR:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    return {
        "input": inp,
        "output": out,
        "mixed_dest": mixed_dest,
        "same_phase_dest": same_phase_dest,
        "hit_events": hit_events,
        "phase_diff_cells": int(np.count_nonzero(out != simultaneous_same_range)),
        "range_diff_cells": int(np.count_nonzero(out != phased_fixed5)),
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
