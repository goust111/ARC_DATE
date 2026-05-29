# Variant of 409aa875
# Dimension: D2 — 投射距离 (path)
# Change: 固定距离5格（apex + 5 * direction）→ 距离由形状类型决定：L形射程4格，V形射程6格

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D2 variant
# ----------------------------------
# variant:
#   core_change:
#     - The original task used a single landing rule: destination = apex + 5 * direction.
#     - This variant keeps shape parsing and hit/collision semantics unchanged, but replaces
#       the fixed distance with a shape-dependent distance:
#         L-shape -> 4 cells
#         V-shape -> 6 cells
#   source_task:
#     - None. No extra reference solver is required beyond the original 409aa875 code/spec.
# concretization_and_trace:
#   - L example:
#       cells {(5,5),(5,6),(6,5)} form an L with apex/corner at (5,5), direction (-1,-1).
#       Old rule: landing cell = (0,0).
#       New rule: landing cell = (1,1).
#       The projectile still behaves the same after landing; only the destination moved.
#   - V example:
#       cells {(7,6),(8,5),(8,7)} form a V with apex at (7,6), direction (-1,0).
#       Old rule: landing cell = (2,6).
#       New rule: landing cell = (1,6).
#   - Therefore the workflow change is localized but consequential:
#       shape detection -> direction inference -> choose distance from shape kind -> resolve hit/bg mark.
# complexity_calibration:
#   - Only two distance values are introduced. This is a small, learnable lookup rather than a free
#     numeric parameter, so the task stays inferable from a few examples.
#   - To keep the change visually observable, the generator requires both shape kinds to appear and
#     rejects any sample whose output would be identical under the old fixed-5 solver.
# implicit_assumptions_review:
#   - `_triangle_info(...)` still assumes every actionable object is a 3-cell monochrome component.
#     The generator preserves this.
#   - The original `rr = ar + 5 * dr` / `cc = ac + 5 * dc` hard-coded a global invariant.
#     That is no longer valid; range lookup must happen per component after shape classification.
#   - Hit resolution still assumes targets are computed from the original input before any recoloring.
#     This remains correct because the new variant changes only destination selection, not update order.
# placement_legality:
#   - Generator placements are precomputed with the correct distance for their shape kind, and only
#     in-bounds destinations are retained.
#   - Sprite halos still enforce Chebyshev distance >= 2 so components do not merge under 8-connectivity.
#   - Reserved background targets still prevent placing a sprite on top of a previously chosen bg impact cell.
# observability_and_cleanliness:
#   - There is no hidden use of color ids. Shape alone determines direction and range.
#   - All sprites share one input color; only the standard output marks appear: maroon 9 and optional blue 1.
#   - The grid is kept sparse enough that emitters, hit sprites, and bg impact points remain visually separable.
# new_solver_assumptions_checked_against_generator:
#   - Every non-bg component in generated inputs is an L or V that `_triangle_info` can parse.
#   - Both L and V appear in every generated sample, so the distance split has repeated support.
#   - The generated output differs from the old fixed-5 interpretation, so the D2 rule is actually exercised.


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

    # L-shape: the corner pixel has two 4-neighbors.
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
        return corner, (dr, dc), "L"

    # V-shape: the apex has two 8-neighbors, one on each diagonal arm.
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
        info = _triangle_info(comp)
        if info is None:
            continue
        (ar, ac), (dr, dc), kind = info
        dist = _range_for_kind(kind, range_by_kind)
        rr = ar + dist * dr
        cc = ac + dist * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from tiny triangle sprites, with range determined by sprite type.

    Input structure:
    - The background is the most common color, and every non-background object is a 3-cell monochrome triangle sprite.
    - Sprites come in two forms: axis-aligned L-shapes and diagonal V-shapes; the same input may contain both.

    Rule:
    - Each sprite still uses its shape to determine its apex and firing direction: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite their opening.
    - The landing distance is no longer global: L-shapes travel 4 cells from the apex, and V-shapes travel 6 cells from the apex.
    - A projectile that lands on a sprite recolors that entire hit sprite to maroon (9). If multiple projectiles land on the same empty background cell, that cell becomes blue (1); otherwise a lone empty-cell landing becomes maroon (9).

    Key insight: Shape now controls two linked parameters at once: orientation still gives direction, but shape family also selects the projectile's range.
    """
    return _solve_with_ranges(grid, {"L": 4, "V": 6})


# --------------- generator ---------------

H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

RANGE_BY_KIND = {"L": 4, "V": 6}
FIXED5_BY_KIND = {"L": 5, "V": 5}
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIRS_BY_KIND = {"L": DIAG_DIRS, "V": AXIS_DIRS}

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 400


def _sprite_cells(kind: str, apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3 cells of a sprite with the given kind, apex, and firing direction."""
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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    """Precompute all sprite placements whose shape-specific landing cell is in-bounds."""
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
                    placement = {
                        "kind": kind,
                        "range": dist,
                        "apex": (ar, ac),
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
    return all(_triangle_info(comp) is not None for comp in comps)


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


def _shape_counts(sprites: list[dict]) -> Counter:
    return Counter(s["placement"]["kind"] for s in sprites)


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

    kinds_present = set(_shape_counts(sprites))
    if kinds_present != {"L", "V"}:
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    old_out = _solve_with_ranges(inp, FIXED5_BY_KIND)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if np.array_equal(inp, out):
        return None
    if np.array_equal(out, old_out):
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
        "sprites": sprites,
        "shape_counts": _shape_counts(sprites),
        "old_output": old_out,
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
