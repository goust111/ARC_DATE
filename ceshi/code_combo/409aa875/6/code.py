# Combo variant of 409aa875 (index 6)
# combo_spec: D4/0+D9/0
# Name: 多色精灵+命中消失

from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for combo D4/0 + D9/0
# --------------------------------------
# spec_anchor:
#   - Keep the original geometry: 16x16 grid, fixed bg 7 in generated samples,
#     3-cell triangle sprites only, same apex/direction decoding, same
#     projectile distance 5, same empty-bg markers (single -> 9, shared -> 1).
#   - Fusion changes only two axes:
#       1. hit sprite outcome: erase the whole struck sprite instead of painting 9
#       2. sprite color policy: each sprite has its own color instead of a global color
#
# concrete_behavior_and_trace:
#   - Small trace:
#       * Input has a red L-sprite, a green V-sprite, and a yellow V-sprite.
#       * Suppose the red sprite's landing cell lies on one pixel of the green sprite.
#       * Original task: the green 3-cell sprite stays present but becomes maroon 9.
#       * D9 alone: the green sprite would become red.
#       * This combo: the green sprite disappears entirely, so the only direct sign
#         of the hit is the missing green component in the output.
#       * The red color still matters because among multiple surviving shooters, the
#         colors make each emitter visually distinguishable while the solver traces
#         shape-driven trajectories.
#   - Empty-background impacts are unchanged, so outputs can show both missing
#     components and isolated 9/1 landing markers in the same sample.
#
# complexity_calibration:
#   - If inputs stayed monochrome, the D9 half would be invisible and the combo
#     would collapse to the plain D4 erase variant.
#   - If samples had no sprite-hit events, the erase rule would be invisible.
#   - To keep the fusion observable without adding extra mechanics:
#       * every generated sample has at least one sprite-hit event
#       * every sprite gets a unique color within the sample
#       * all sprite colors avoid {1, 9} so they do not visually blend with
#         blue/maroon empty-impact markers
#       * each sample also keeps at least one empty-background landing event
#
# implicit_assumption_audit_against_original_main:
#   - Original code grouped components by color first. That is fragile here because
#     colors vary per sprite; the robust invariant is simply "each non-bg 8-component
#     is one triangle sprite". The combo solver therefore extracts all non-bg
#     components directly.
#   - Original code only needed the set of hit components because all hits recolored
#     to a constant 9. That remains sufficient again after fusing with erase, since
#     the output no longer depends on shooter color at the hit site.
#   - The occupancy map must still be built from the input before any erasure. Empty
#     markers are decided from original occupancy, not from the partially erased output.
#   - Multiple shooters hitting the same sprite would still erase that sprite only once,
#     but such cases add little evidence and can make human tracing messy, so the
#     generator keeps hit targets unique.
#
# placement_legality:
#   - Sprites are chosen from placements whose 3 cells and landing cell are in bounds.
#   - Halo checks keep pairwise Chebyshev distance >= 2, so 8-connected components do
#     not merge even when colors differ.
#   - Background landing cells are reserved before later placements, preventing a later
#     sprite from occupying a cell that should become an empty-impact marker.
#   - Hit shooters are added only after their target sprites already exist, and the
#     chosen hit shooters are never themselves selected as hit targets.
#
# visual_observability:
#   - A human comparing 2-4 samples can see that outputs sometimes lose whole colored
#     triangles rather than recolor them.
#   - Because each input sprite has a unique color, a disappeared target can be paired
#     with a specific surviving shooter by tracing orientation and landing geometry.
#   - The generator rejects monochrome or low-diversity constructions by enforcing one
#     distinct color per sprite.
#
# visual_cleanliness:
#   - Empty-impact markers remain single cells only; there is no extra overlay on top
#     of surviving sprites.
#   - Hit count is capped at 2 and collision count at 2, matching the original task's
#     density, so the board stays readable.
#   - No sprite uses colors 1 or 9, so the single-cell output marks stay visually distinct.
#
# change_inference:
#   - Color changes are not numeric rules here; colors are identity tags for sprites.
#   - The inferable pattern is: orientation determines the projectile path, a struck
#     target vanishes, and unique sprite colors help identify which visible shooter
#     explains that disappearance.
#   - This is why the generator tracks both erased targets and distinct shooter colors
#     as explicit validation metrics.
#
# new_main_assumption_audit:
#   - Each non-bg component must still be a single 3-cell triangle. Generator validation
#     enforces exactly n such components.
#   - Each sprite is monochrome internally, so its visible identity color is well-defined.
#   - Each erased target is chosen from sprites that existed before hit shooters were added,
#     which guarantees at least one surviving shooter remains available for visual tracing.


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from multicolored 3-pixel triangle sprites; erase hit sprites and mark empty impacts.

    Input structure:
    - The background is the most common color. Every non-background 8-connected component is a 3-cell monochrome triangle sprite, and different sprites may use different colors.

    Rule:
    - Each triangle's shape determines its firing direction: L-shapes fire diagonally away from the missing corner, while V-shapes fire along the axis opposite their opening.
    - Its projectile lands exactly 5 cells from the apex in that direction.
    - If the landing cell lies on another sprite, erase that entire struck sprite to the background color.
    - If the landing cell is empty background, a lone projectile paints 9 and a shared landing by 2 or more projectiles paints 1.

    Key insight:
    - Shape still determines the landing cell, but sprite-hit evidence is now a missing component rather than a recolor; the multiple input colors help distinguish which surviving shooter accounts for that disappearance.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _cc8(g != bg)
    cell_to_comp: dict[tuple[int, int], int] = {}
    for cid, comp in enumerate(comps):
        for rc in comp:
            cell_to_comp[rc] = cid

    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for cid, comp in enumerate(comps):
        ad = _triangle_apex_and_dir(comp)
        if ad is None:
            continue
        (ar, ac), (dr, dc) = ad
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    hit_components = {cell_to_comp[dest] for dest in dest_to_shooters if dest in cell_to_comp}

    out = g.copy()
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = bg

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


H = W = 16
BG = 7
LONE_BG_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]
TRACEABLE_HIT_SHOOTER_COLORS = SPRITE_COLORS[:]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 7)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 500


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
            rng.shuffle(candidates)
            return candidates[0], cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    if not all(comp.shape == (3, 2) for comp in comps):
        return False
    return len(set(int(v) for v in np.unique(grid)) - {BG}) == n_sprites


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
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
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
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
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _assign_sprite_colors(rng: random.Random, sprites: list[dict]) -> bool:
    hit_shooter_ids = [idx for idx, s in enumerate(sprites) if s["target_kind"] == "hit"]
    if len(hit_shooter_ids) > len(TRACEABLE_HIT_SHOOTER_COLORS):
        return False

    assigned: dict[int, int] = {}
    for idx, color in zip(hit_shooter_ids, rng.sample(TRACEABLE_HIT_SHOOTER_COLORS, len(hit_shooter_ids))):
        assigned[idx] = color

    remaining_colors = [c for c in SPRITE_COLORS if c not in assigned.values()]
    remaining_ids = [idx for idx in range(len(sprites)) if idx not in assigned]
    if len(remaining_ids) > len(remaining_colors):
        return False
    for idx, color in zip(remaining_ids, rng.sample(remaining_colors, len(remaining_ids))):
        assigned[idx] = color

    for idx, sprite in enumerate(sprites):
        sprite["color"] = assigned[idx]
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        for r, c in sprites[idx]["placement"]["cells"]:
            out[r, c] = BG
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else LONE_BG_COLOR
    return out


def _render_original_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        for r, c in sprites[idx]["placement"]["cells"]:
            out[r, c] = 9
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else LONE_BG_COLOR
    return out


def _count_erased_components(inp: np.ndarray, out: np.ndarray) -> int:
    erased = 0
    for comp in components_mask(inp != BG, connectivity=8):
        cells = [(int(r), int(c)) for r, c in comp]
        if len(cells) == 3 and all(int(out[r, c]) == BG for r, c in cells):
            erased += 1
    return erased


def _variant_metrics(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> dict[str, int]:
    sprite_colors = [sprite["color"] for sprite in sprites]
    hit_shooter_colors = [sprite["color"] for sprite in sprites if sprite["target_kind"] == "hit"]
    bg_counter = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    return {
        "sprite_count": len(sprites),
        "distinct_sprite_colors": len(set(sprite_colors)),
        "hit_events": len(hit_shooter_colors),
        "traceable_hit_shooters": sum(color not in {1, 9} for color in hit_shooter_colors),
        "erased_components": _count_erased_components(inp, out),
        "bg_marker_events": sum(1 for count in bg_counter.values() if count >= 1),
        "blue_cells": int(np.count_nonzero(out == COLLISION_COLOR)),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
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

    if not _assign_sprite_colors(rng, sprites):
        return None

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite["color"])

    if not _validate_input(inp, n_sprites):
        return None

    out = _render_output(inp, sprites)
    legacy_out = _render_original_output(inp, sprites)
    metrics = _variant_metrics(inp, out, sprites)

    if np.array_equal(out, legacy_out):
        return None
    if metrics["erased_components"] < 1:
        return None
    if metrics["distinct_sprite_colors"] != n_sprites:
        return None
    if metrics["traceable_hit_shooters"] < 1:
        return None
    if metrics["bg_marker_events"] < 1:
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
