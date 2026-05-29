# Variant of 409aa875
# Dimension: D4 — 碰撞规则（投射物-精灵） (transform)
# Change: 投射物命中精灵时，整个被命中精灵重着色为栗色(9)... → 命中精灵被擦除（消失）

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D4 variant
# ----------------------------------
# spec:
#   unchanged_core:
#     - Grid is still 16x16 with fixed background color 7 in generated samples.
#     - Every non-background object is still a monochrome 3-cell triangle sprite:
#       either an axis-aligned L or a diagonal V.
#     - Each sprite still emits exactly one projectile whose landing cell is
#       apex + 5 * direction, where direction is decoded from shape orientation.
#     - Empty-cell impacts still paint maroon (9) for a single projectile and
#       blue (1) for a multi-projectile collision.
#   D4_change:
#     - Only the projectile-versus-sprite collision rule changes.
#     - Original behavior: if a landing cell lies on a sprite, recolor that
#       whole hit sprite to maroon.
#     - Variant behavior: if a landing cell lies on a sprite, erase that whole
#       hit sprite to the background color, so the component disappears.
#   concrete_pixel_effect:
#     - Input still shows only the triangle sprites.
#     - Output may contain:
#       1. missing 3-pixel sprites where impacts landed on sprite cells,
#       2. isolated maroon single-impact marks on empty background,
#       3. isolated blue marks where two projectiles share an empty destination.
#     - The erased sprite is identified using the input occupancy, not by
#       checking the partially modified output grid.
#   traced_example:
#     - Suppose an L-sprite fires to cell (6, 6), and (6, 6) belongs to another
#       3-cell sprite.
#     - Original task: the target sprite remains present but all three of its
#       cells become 9.
#     - Variant: the same target sprite's three cells are set to background 7,
#       so a whole object vanishes.
#     - Empty-cell impacts are unchanged in both tasks.
#   complexity_calibration:
#     - The rule change is visually strong but still single-branch and local.
#     - No new shape classes, distances, colors, or matching heuristics are
#       introduced, so the task stays at the same structural complexity.
#     - To keep the variant observable, generation must force at least one
#       sprite-hit event; collision-only samples would look identical to the
#       original task and would therefore hide the D4 change.
#   implicit_assumption_audit:
#     - Component detection by monochrome 8-connectivity still holds because
#       inputs remain unchanged.
#     - Mapping each input cell to its component id must be computed before any
#       erasure; otherwise deleting one target could incorrectly affect later
#       hit checks.
#     - Empty-cell marking must still be based on whether the landing cell was
#       occupied in the input, not whether it becomes empty after erasing a hit
#       sprite.
#     - Generator assertions from the original code that required the output to
#       have more non-background cells are no longer valid; erasure can reduce
#       the count, so validation has to target "at least one erased sprite"
#       instead of "pixel count increases".
#   placement_legality:
#     - Sprites are still placed with Chebyshev separation >= 2 so components do
#       not merge and every 3-cell object remains solver-readable.
#     - Hit shooters are still chosen from placements whose landing cells are on
#       an existing target sprite and whose own cells do not overlap other
#       sprites or reserved empty-impact cells.
#     - Background impact cells are reserved so empty-cell markers do not stack
#       on top of sprite cells.
#   visual_observability:
#     - Each generated sample has at least one erased target sprite and at least
#       one empty-cell impact marker, so examples show both "where shots land"
#       and the new "sprite disappears" consequence.
#     - The output stays tidy because only one layer of marks is added to empty
#       cells, while hit sprites are removed instead of recolored on top of
#       other structures.
#   color_reasoning:
#     - No invisible ordering by color id is used.
#     - Colors still have only a small functional vocabulary: sprite color,
#       background 7, maroon 9 for lone empty impacts, blue 1 for shared empty
#       impacts. The erased case is conveyed structurally by absence, not by a
#       new arbitrary recoloring map.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


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

    # L-shape (axis-aligned): one pixel has two 4-neighbors (the corner).
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
        # Projectile goes in the diagonal direction opposite the missing 2x2 corner.
        dr = -dr_sum
        dc = -dc_sum
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return corner, (dr, dc)

    # Diagonal V-shape: one pixel has two 8-neighbors (diagonal legs).
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
    # The direction is the axis direction opposite the average of the two diagonal legs.
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; erase hit sprites and mark empty impacts.

    Input structure:
    - bg = most common color. Non-bg cells form monochrome 3-pixel sprites shaped as axis-aligned L's or diagonal V's.

    Rule:
    - Each triangle has an apex and an emission direction determined by its shape: L-shapes fire diagonally away from the missing corner; V-shapes fire along the axis away from the open side.
    - A projectile lands exactly 5 cells from the apex in the emission direction.
    - If that landing cell lies on an existing sprite, erase the entire hit sprite to the background color.
    - If two projectiles land on the same empty bg cell, that cell becomes blue (1); a lone projectile on empty bg paints maroon (9).

    Key insight: The triangle orientation still encodes the landing cell, but sprite-hit impacts are read as disappearance of the whole struck component rather than a recolor.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    # Find all monochrome components for each non-bg color.
    comps: list[list[tuple[int, int]]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            cid = len(comps)
            comps.append(comp)
            for rc in comp:
                cell_to_comp[rc] = cid

    # Each triangle emits a projectile to apex + 5 * dir, measured on the input grid.
    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
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

    # Variant D4: projectile hits erase the struck sprite instead of recoloring it.
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = bg

    # Projectile collision locations on empty background are unchanged.
    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


# --------------- generator (variant-specific output-first generator) ---------------

# ---- canvas / palette ----
H = W = 16
BG = 7
LONE_BG_COLOR = 9
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
MAX_ATTEMPTS = 400


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Return the 3 pixel coords of a triangle sprite with given apex and firing direction."""
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
    """Cell + 8-neighborhood, clipped to grid. Used to enforce Chebyshev>=2 separation."""
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    """Precompute all (apex, dir) placements whose sprite and landing cell are in-bounds."""
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
    """Placement is OK if its halo doesn't touch any occupied cell and its cells are free."""
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    """Pick a bg cell and two compatible shooters whose projectiles both land on it."""
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
    """Pick a shooter whose landing cell is a fresh, empty bg cell."""
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
    """Pick a shooter that lands on one of target_sprite's 3 pixels."""
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
    """Sanity check: correct shape, bg, and exactly n_sprites 3-pixel components."""
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    """Pick (e_hit, e_coll, unique) counts, forcing visible erase events."""
    options = []
    for e_hit in range(1, min(MAX_HIT_EVENTS, n_sprites - 1) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
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
    """Add e_hit shooters, each targeting an existing sprite's pixels."""
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
    """Apply the variant transform rule to produce the expected output grid."""
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        for r, c in sprites[idx]["placement"]["cells"]:
            out[r, c] = BG
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else LONE_BG_COLOR
    return out


def _count_erased_components(inp: np.ndarray, out: np.ndarray) -> int:
    """Count input 3-cell components that become entirely background in the output."""
    erased = 0
    for comp in components_mask(inp != BG, connectivity=8):
        cells = [(int(r), int(c)) for r, c in comp]
        if len(cells) == 3 and all(int(out[r, c]) == BG for r, c in cells):
            erased += 1
    return erased


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

    # 1. Collision events first (most constrained).
    if not _place_collisions(rng, e_coll, sprites, occupied, reserved_bg):
        return None

    # 2. Base sprites that hit-shooters will target. Need enough existing sprites for e_hit picks.
    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, base_needed, sprites, occupied, reserved_bg):
        return None

    # 3. Hit-shooters aimed at existing sprites.
    if not _place_hits(rng, e_hit, sprites, occupied, reserved_bg):
        return None

    # 4. Remaining shooters with fresh bg landing cells.
    remaining = n_sprites - len(sprites)
    if remaining < 0 or not _place_unique_landings(rng, remaining, sprites, occupied, reserved_bg):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    erased_components = _count_erased_components(inp, out)

    if erased_components < 1:
        return None
    if not np.any(out == LONE_BG_COLOR) and not np.any(out == COLLISION_COLOR):
        return None
    if np.array_equal(inp, out):
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
