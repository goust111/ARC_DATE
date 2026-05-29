# Variant of 409aa875
# Dimension: D12 — 方向确定规则（形状语义） (path)
# Change: L形：对角线方向远离缺失的2x2角；V形：轴方向远离开口侧。方向完全由像素排列的几何结构决定... → 方向翻转（沿缺口/开口方向发射）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D12 inversion variant
# =============================================
# a) Concrete behavior and trace
# - Objects stay exactly the same as in the source task: 16x16 grid, bg=7, 4-9 monochrome
#   3-pixel sprites, each an 8-connected component shaped either as an axis-aligned L or a
#   diagonal V. The projectile distance also stays fixed at 5 cells from the apex.
# - Only the meaning of orientation changes:
#   - L-shape: the corner pixel still acts as apex, but the shot now travels toward the missing
#     corner of the 2x2 box instead of away from it.
#   - V-shape: the apex still is the unique pixel with two diagonal neighbors, but the shot now
#     travels toward the open/spread side instead of away from it.
# - Small trace:
#   - Original L example cells {(0,0),(0,1),(1,0)} have apex (0,0), missing corner (1,1).
#     Old solver fired to (-1,-1); new solver fires to (1,1). The landing cell flips from
#     apex+5*(-1,-1) to apex+5*(1,1), so every downstream event may move to the opposite side.
#   - Original V example cells {(0,1),(1,0),(1,2)} have apex (0,1), diagonal-leg average (1,0).
#     Old solver fired up; new solver fires down. Again, the landing cell is mirrored through the
#     apex along the same axis.
# - Therefore the only stage that changes is "shape -> direction". Input construction, component
#   detection, 5-step travel, hit recolor, and collision painting remain unchanged.
#
# b) Complexity calibration
# - The variant is not simplified into arbitrary per-color or per-instance exceptions. Direction
#   still comes from geometry alone and stays limited to the same 8 canonical directions.
# - To keep the change visually recoverable from a few examples, the generator rejects degenerate
#   samples where the original rule and the inverted rule would accidentally produce the same
#   output. That forces the flipped semantics to have observable consequences.
# - The generator keeps both original shape families available; it does not collapse everything to
#   only L or only V logic.
#
# c) Hidden-assumption audit of the original solver
# - `_triangle_apex_and_dir(...)` was the only place encoding the old "away from gap/open side"
#   assumption. The apex-identification logic itself still holds; only the final sign of the
#   direction vector must flip.
# - The loop building `dest_to_shooters` assumes each valid component maps to one apex and one
#   in-bounds direction. That remains valid because the generator only emits canonical 3-pixel
#   shapes whose new-direction landing cell is inside the grid.
# - `hit_components` assumes landing on any occupied cell of a component recolors that whole
#   component. This stays correct because the variant changes destination selection, not hit
#   semantics.
# - The final painting loop assumes bg markers never need to share a cell with a sprite cell.
#   That still holds because the generator reserves hit events separately from bg landings.
#
# d) Placement legality
# - The generator keeps the same halo rule: sprites must be at Chebyshev distance >= 2 so 8-way
#   components never merge.
# - Placements are precomputed only when both sprite cells and the new-rule landing cell are
#   in-bounds.
# - Collision events reserve an empty destination and choose two compatible shooters for it.
# - Hit events pick an existing sprite cell first, then place a later shooter whose new-rule
#   landing cell matches that exact occupied target cell.
#
# e) Visual observability
# - Humans can see the flipped rule because every sprite still points, but the new markers appear
#   on the gap/opening side instead of the opposite side. This is a visible spatial reversal,
#   not a hidden numeric change.
# - The generator explicitly rejects samples whose output is identical under the old rule, so the
#   training pairs cannot silently teach the original task.
#
# f) Visual cleanliness
# - Only one shared sprite color is used per grid, preserving the clean original layout.
# - New marks remain sparse: at most 2 hit events and 2 bg-collision events, with other shots
#   landing on distinct bg cells. This avoids dense clutter.
#
# g) Color-change inferability
# - Color behavior is unchanged and still simple: hit sprite or lone bg landing -> 9, repeated bg
#   landing -> 1. No extra color remapping is introduced.
#
# New solver assumptions introduced by the variant and how the generator guarantees them
# - The solver still assumes every non-bg component meant to matter is a canonical 3-pixel L or V.
#   `_validate_input(...)` checks component count/size and that every component is recognized.
# - The solver assumes the meaningful landing cells are those produced by the inverted direction
#   rule; `_build_placements()` and all event-placement helpers now use that same rule.
# - The solver has no ambiguity resolution for malformed 3-pixel components with both an L corner
#   and a V apex signature. The canonical placement formulas never generate such ambiguous shapes.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _normalize_diag(dr: int, dc: int) -> tuple[int, int]:
    return (1 if dr > 0 else -1), (1 if dc > 0 else -1)


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Returns (apex_rc, dir_rc). Direction is one of 4-neigh (axis) or diagonal.
    if len(comp) != 3:
        return None
    s = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    # L-shape: fire toward the missing 2x2 corner.
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
        return corner, _normalize_diag(dr_sum, dc_sum)

    # V-shape: fire toward the open/spread side.
    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    d1 = (others[0][0] - ar, others[0][1] - ac)
    d2 = (others[1][0] - ar, others[1][1] - ac)
    dr = (d1[0] + d2[0]) // 2
    dc = (d1[1] + d2[1]) // 2
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _triangle_apex_and_dir_original(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Reference helper for generator anti-degeneracy checks against the source task."""
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
        return corner, _normalize_diag(-dr_sum, -dc_sum)

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


def _solve_with_direction_rule(
    grid: np.ndarray,
    direction_fn,
) -> np.ndarray:
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
        ad = direction_fn(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
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
    Three-pixel triangle sprites launch inward-facing shots and paint the impact results.

    Input structure:
    - The background is the majority color, and every non-background object is a monochrome
      3-cell component shaped either like an axis-aligned L or a diagonal V.

    Rule:
    - Each sprite first reveals an apex from its geometry. L-shapes shoot diagonally toward the
      missing corner of their 2x2 box, while V-shapes shoot along the axis toward their open side.
    - The landing cell is exactly 5 steps from the apex in that direction.
    - Landing on a sprite recolors the whole hit sprite to maroon (9). Landing on empty background
      paints maroon (9), except that two or more shots sharing the same empty cell paint blue (1).

    Key insight:
    - The puzzle is unchanged except for one semantic inversion: every triangle still encodes a
      unique direction, but now the encoded shot goes into the gap/opening instead of away from it.
    """
    return _solve_with_direction_rule(grid, _triangle_apex_and_dir)


# --------------- generator ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2

# ---- generation retries ----
MAX_ATTEMPTS = 300


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Canonical sprite cells for the new direction semantics.

    `direction` is the actual projectile direction under the variant:
    - diagonal dir -> L-shape whose missing 2x2 corner lies in `direction`
    - axis dir -> V-shape whose open/spread side lies in `direction`
    """
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar + dr, ac), (ar, ac + dc))
    if dr != 0:
        return ((ar, ac), (ar + dr, ac - 1), (ar + dr, ac + 1))
    return ((ar, ac), (ar - 1, ac + dc), (ar + 1, ac + dc))


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
            for second in candidates[i + 1:]:
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
    if _bg_color(grid) != BG:
        return False
    comps = _cc8(grid != BG)
    if len(comps) != n_sprites:
        return False
    return all(len(comp) == 3 and _triangle_apex_and_dir(comp) is not None for comp in comps)


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


def _old_rule_output(inp: np.ndarray) -> np.ndarray:
    return _solve_with_direction_rule(inp, _triangle_apex_and_dir_original)


def _variant_is_observable(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> bool:
    old_out = _old_rule_output(inp)
    if np.array_equal(old_out, out):
        return False

    diag_count = sum(1 for s in sprites if abs(s["placement"]["dir"][0]) + abs(s["placement"]["dir"][1]) == 2)
    axis_count = len(sprites) - diag_count
    if diag_count < 1 or axis_count < 1:
        return False

    changed_new = set(map(tuple, np.argwhere(out != inp)))
    changed_old = set(map(tuple, np.argwhere(old_out != inp)))
    return len(changed_new.symmetric_difference(changed_old)) >= 1


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
    if not _variant_is_observable(inp, out, sprites):
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
