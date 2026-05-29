# Variant of 409aa875
# Dimension: D18 — 终止/目标条件 (dst)
# Change: 无显式目标；投射物飞行固定距离后落地，效果取决于落点内容... → 目标标记点（投射物必须命中指定位置）

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


"""
design_analysis:
  source_task: none
  preserved_core:
    - 16x16 grid, background 7, 3-pixel triangle shooters, and the same apex+direction decoding.
    - Projectile destination is still apex + 5 * dir.
  variant_concretization:
    - Add explicit single-pixel target markers in a fixed color (TARGET_COLOR = 2).
    - Shooters remain monochrome 3-pixel L-shapes or V-shapes in one shared non-target color.
    - A shot only matters if its landing cell is exactly one of those target markers.
    - A target hit by one projectile becomes 9; a target hit by two or more projectiles becomes 1.
    - Misses do nothing, and untouched targets stay color 2.
  behavior_trace:
    - Original flow: a shooter landing on any empty bg cell painted that cell, and a shared landing painted blue.
    - New flow: the same shooter direction is computed, but if its landing cell is not a pre-drawn target pixel, nothing is painted.
    - Example: if two shooters both land on (8, 10), original logic would paint that bg cell 1; new logic paints 1 only if input[8,10] is already a target marker, otherwise output keeps (8,10) as background.
  complexity_calibration:
    - To make the gating visible, every generated grid contains:
      * at least one hit target,
      * at least one miss shooter whose shot lands on unmarked background,
      * at least one decoy target that remains untouched.
    - Some grids also contain a shared target hit so the blue branch is exercised.
    - Input uses at most three colors: bg 7, sprite color, target color 2. This keeps the task legible.
  implicit_assumptions_review:
    - Original solver treated every non-bg component as a possible sprite; that fails now because target pixels are also non-bg, so new code filters shooters to 3-pixel triangle components and reads targets separately.
    - Original solver activated every in-bounds destination; that assumption is invalid in this variant, so new code checks membership in the explicit target set before applying any effect.
    - Original hit-sprite recoloring branch depended on destinations being allowed to land on sprite cells. New targets are isolated single pixels, so that branch is removed rather than left as dead logic.
  placement_legality:
    - Sprites keep Chebyshev separation >= 2 via halo checks.
    - Target markers are placed only on empty background cells and are kept out of every sprite halo so they remain visually isolated.
    - Shooter destinations are unique except intentional shared-target collisions.
  observability_and_cleanliness:
    - Humans can compare shooters to visible target dots and see that only matching shots matter.
    - Unhit targets staying red while missed shots leave no mark is the main visual cue differentiating this from the base task.
  solver_generator_contract:
    - All target markers are singleton pixels of color 2.
    - All shooters are 3-pixel monochrome L/V components in colors other than 2 and 7.
"""


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


def _triangle_components(grid: np.ndarray, bg: int, target_color: int) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    for col in sorted(set(int(v) for v in np.unique(grid)) - {bg, target_color}):
        for comp in _cc8(grid == col):
            if _triangle_apex_and_dir(comp) is not None:
                comps.append(comp)
    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve target-gated projectile landings fired by 3-pixel triangle sprites.

    Input structure:
    - Background is the most common color. Shooters are monochrome 3-pixel L/V triangles, and target markers are fixed-color single pixels.

    Rule:
    - Each triangle still determines an apex and firing direction from its shape, and its projectile still lands exactly 5 cells from that apex.
    - A landing only counts when that exact destination already contains a target marker.
    - One projectile on a target recolors that marker to maroon (9); two or more projectiles on the same target recolor it to blue (1).
    - Missed shots do nothing, and untouched targets keep their original color.

    Key insight: The orientation-to-direction logic is unchanged, but explicit target dots now gate the effect, so you only recolor destinations that are both geometrically reachable and pre-marked.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    targets = {(int(r), int(c)) for r, c in np.argwhere(g == TARGET_COLOR)}
    shooters = _triangle_components(g, bg, TARGET_COLOR)

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(shooters):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w and (rr, cc) in targets:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    out = g.copy()
    for (r, c), shooters_here in dest_to_shooters.items():
        out[r, c] = COLLISION_COLOR if len(shooters_here) >= 2 else HIT_COLOR
    return out


# ---- canvas / palette ----
H = W = 16
BG = 7
TARGET_COLOR = 2
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 3, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)
MAX_SINGLE_HITS = 3
MAX_COLLISION_TARGETS = 2
MAX_DECOY_TARGETS = 3

# ---- generation retries ----
MAX_ATTEMPTS = 400


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


def _point_halo(cell: tuple[int, int]) -> frozenset[tuple[int, int]]:
    r, c = cell
    return frozenset(
        (rr, cc)
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
ALL_CELLS = [(r, c) for r in range(H) for c in range(W)]


def _fits(
    placement: dict,
    occupied: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
) -> bool:
    return (
        not (placement["halo"] & occupied)
        and not (placement["cell_set"] & reserved_dest)
        and not (placement["halo"] & target_cells)
    )


def _can_place_target(
    cell: tuple[int, int],
    occupied_halo: set[tuple[int, int]],
    target_halo: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
) -> bool:
    return cell not in occupied_halo and cell not in target_halo and cell not in reserved_dest


def _add_sprite_state(
    placement: dict,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
) -> None:
    occupied.update(placement["cells"])
    occupied_halo.update(placement["halo"])


def _add_target_state(
    cell: tuple[int, int],
    target_cells: set[tuple[int, int]],
    target_halo: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
) -> None:
    target_cells.add(cell)
    target_halo.update(_point_halo(cell))
    reserved_dest.add(cell)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int, int] | None:
    options = []
    for single_hits in range(1, min(MAX_SINGLE_HITS, n_sprites - 1) + 1):
        max_coll = min(MAX_COLLISION_TARGETS, (n_sprites - single_hits - 1) // 2)
        for coll_targets in range(0, max_coll + 1):
            misses = n_sprites - single_hits - 2 * coll_targets
            if misses < 1:
                continue
            for decoys in range(1, MAX_DECOY_TARGETS + 1):
                options.append((single_hits, coll_targets, misses, decoys))
    return rng.choice(options) if options else None


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
    target_halo: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(PLACEMENTS_BY_DEST)
    rng.shuffle(dests)
    for dest in dests:
        if not _can_place_target(dest, occupied_halo, target_halo, reserved_dest):
            continue
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _fits(p, occupied, reserved_dest, target_cells)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            for second in candidates[i + 1:]:
                if second["cell_set"] & first["cell_set"]:
                    continue
                if second["halo"] & occ2:
                    continue
                if second["halo"] & target_cells:
                    continue
                return dest, [first, second]
    return None


def _choose_single_hit(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    occupied_halo: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
    target_halo: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        dest = placement["dest"]
        if not _can_place_target(dest, occupied_halo, target_halo, reserved_dest):
            continue
        if _fits(placement, occupied, reserved_dest, target_cells):
            return placement
    return None


def _choose_miss(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
    target_cells: set[tuple[int, int]],
) -> dict | None:
    candidates = PLACEMENTS[:]
    rng.shuffle(candidates)
    for placement in candidates:
        dest = placement["dest"]
        if dest in reserved_dest or dest in target_cells or dest in occupied:
            continue
        if _fits(placement, occupied, reserved_dest, target_cells):
            return placement
    return None


def _choose_decoy_target(
    rng: random.Random,
    occupied_halo: set[tuple[int, int]],
    target_halo: set[tuple[int, int]],
    reserved_dest: set[tuple[int, int]],
) -> tuple[int, int] | None:
    cells = ALL_CELLS[:]
    rng.shuffle(cells)
    for cell in cells:
        if _can_place_target(cell, occupied_halo, target_halo, reserved_dest):
            return cell
    return None


def _place_collision_hits(rng, n, sprites, occupied, occupied_halo, target_cells, target_halo, reserved_dest) -> bool:
    for _ in range(n):
        choice = _choose_collision_pair(rng, occupied, occupied_halo, target_cells, target_halo, reserved_dest)
        if choice is None:
            return False
        dest, pair = choice
        _add_target_state(dest, target_cells, target_halo, reserved_dest)
        for placement in pair:
            sprites.append({"placement": placement, "kind": "targeted", "target": dest})
            _add_sprite_state(placement, occupied, occupied_halo)
    return True


def _place_single_hits(rng, n, sprites, occupied, occupied_halo, target_cells, target_halo, reserved_dest) -> bool:
    for _ in range(n):
        placement = _choose_single_hit(rng, occupied, occupied_halo, target_cells, target_halo, reserved_dest)
        if placement is None:
            return False
        dest = placement["dest"]
        _add_target_state(dest, target_cells, target_halo, reserved_dest)
        sprites.append({"placement": placement, "kind": "targeted", "target": dest})
        _add_sprite_state(placement, occupied, occupied_halo)
    return True


def _place_misses(rng, n, sprites, occupied, occupied_halo, reserved_dest, target_cells) -> bool:
    for _ in range(n):
        placement = _choose_miss(rng, occupied, reserved_dest, target_cells)
        if placement is None:
            return False
        reserved_dest.add(placement["dest"])
        sprites.append({"placement": placement, "kind": "miss", "target": placement["dest"]})
        _add_sprite_state(placement, occupied, occupied_halo)
    return True


def _place_decoys(rng, n, target_cells, target_halo, occupied_halo, reserved_dest) -> bool:
    for _ in range(n):
        cell = _choose_decoy_target(rng, occupied_halo, target_halo, reserved_dest)
        if cell is None:
            return False
        _add_target_state(cell, target_cells, target_halo, reserved_dest)
    return True


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    target_comps = _cc8(grid == TARGET_COLOR)
    if not target_comps or not all(len(comp) == 1 for comp in target_comps):
        return False

    sprite_count = 0
    sprite_halo: set[tuple[int, int]] = set()
    for col in sorted(set(int(v) for v in np.unique(grid)) - {BG, TARGET_COLOR}):
        comps = _cc8(grid == col)
        sprite_count += len(comps)
        for comp in comps:
            if len(comp) != 3 or _triangle_apex_and_dir(comp) is None:
                return False
            sprite_halo.update(_halo(tuple(comp)))
    if any(cell in sprite_halo for comp in target_comps for cell in comp):
        return False
    return sprite_count == n_sprites


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    target_counts = Counter(s["target"] for s in sprites if s["kind"] == "targeted")
    for dest, count in target_counts.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    single_hits, coll_targets, misses, decoys = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    occupied_halo: set[tuple[int, int]] = set()
    target_cells: set[tuple[int, int]] = set()
    target_halo: set[tuple[int, int]] = set()
    reserved_dest: set[tuple[int, int]] = set()

    if not _place_collision_hits(rng, coll_targets, sprites, occupied, occupied_halo, target_cells, target_halo, reserved_dest):
        return None
    if not _place_single_hits(rng, single_hits, sprites, occupied, occupied_halo, target_cells, target_halo, reserved_dest):
        return None
    if not _place_misses(rng, misses, sprites, occupied, occupied_halo, reserved_dest, target_cells):
        return None
    if not _place_decoys(rng, decoys, target_cells, target_halo, occupied_halo, reserved_dest):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)
    for r, c in target_cells:
        inp[r, c] = TARGET_COLOR

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    if np.array_equal(inp, out):
        return None
    if not np.any(out == HIT_COLOR):
        return None
    if not np.any(inp == TARGET_COLOR):
        return None
    if not np.any(out == TARGET_COLOR):
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
