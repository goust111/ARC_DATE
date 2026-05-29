# Combo variant of 409aa875 (index 24)
# combo_spec: D7/0+D15/0
# Name: 裁剪活动区域+可变网格尺寸
#
# Design analysis
# a) Concrete behavior and trace:
#    - The base puzzle logic is unchanged locally: every 3-pixel triangle sprite decodes
#      to an apex plus one of 8 firing directions, and the landing cell is apex + 5 * dir.
#    - The fused change is global and happens in two places:
#      1. Inputs are no longer fixed 16x16; each example uses a board whose height and
#         width are chosen independently from [10, 20], excluding 16x16 so the size change
#         is visible in training.
#      2. Outputs are no longer returned on the full board. After recoloring hit sprites
#         and painting empty landing cells, keep only the minimal rectangle containing all
#         non-background pixels.
#    - Example trace: on a 12x18 board, suppose one L sprite fires to (2, 8), another V
#      sprite fires to the same empty cell, and a third shot recolors a target sprite at
#      rows 5..6, cols 10..12. The old logic would return a full 12x18 board with a blue
#      pixel at (2, 8) and the recolored sprite in place. The fused variant first produces
#      that same full result, then crops to the bbox spanning the leftmost/rightmost active
#      pixels and topmost/bottommost active pixels. The projectile reasoning is identical,
#      but the visible frame is now smaller and depends on both board size and activity.
# b) Complexity calibration:
#    - Only two dimensions move: input frame size and output frame size. Sprite vocabulary,
#      colors, and projectile distance stay fixed so the task remains recognizably the same.
#    - To keep the crop perceptible without introducing extra symbolic clutter, the generator
#      places every sprite cell and every landing cell at least one cell away from the border.
#      This makes the crop strict while still allowing many legal placements on 10..20 boards.
#    - Small boards use slightly fewer sprites and fewer special events so the same interaction
#      pattern stays feasible instead of turning into random congestion near the edges.
# c) Hidden-assumption audit on the original solver:
#    - `h, w = g.shape` already supports variable board sizes, so the main projectile loop is
#      structurally safe once the generator stops assuming a global 16x16 placement table.
#    - Component discovery by monochrome 8-connectivity still works because all foreground
#      objects remain isolated 3-pixel L or V sprites.
#    - The original code assumed `out` and the return value had the same shape as `grid`.
#      That assumption breaks under D7/0, so cropping must happen only after all hits and
#      empty landing marks have been resolved on the full-size board.
#    - The original generator cached a single placement table tied to fixed `H = W = 16`.
#      Under D15/0 this would silently reject legal placements on some shapes and accept
#      stale bounds logic on others, so placement tables are now cached by `(h, w)`.
# d) Placement legality:
#    - Every candidate placement must satisfy three constraints simultaneously:
#      sprite cells in bounds, landing cell in bounds, and all of those cells inside the
#      crop-safe interior margin so the final bbox stays away from the border.
#    - Sprites keep Chebyshev separation >= 2 via halo checks, preventing accidental 8-connected
#      merges or ambiguous component parsing.
#    - Empty landing cells are reserved once used, except when a deliberate >=2-shooter collision
#      is being constructed for a blue output cell.
#    - Hit events only target already placed sprites, so every projectile-to-sprite interaction
#      visible in the output corresponds to an actually present target component.
# e) Visual observability:
#    - Across a few examples, a solver can see both fusion axes directly: inputs arrive on
#      different-sized boards, and outputs are compact windows rather than full canvases.
#    - The local triangle-to-direction rule remains the only way to explain where the maroon
#      and blue marks appear inside those compact outputs, so the crop does not obscure the
#      original reasoning challenge.
# f) Visual cleanliness:
#    - One sprite color per input keeps the scene sparse. The only extra output colors remain
#      the original maroon hit/lone-impact color and blue shared-impact color.
#    - Border margins are removed only after painting, so no new overlapping glyph system is
#      introduced; the crop simply discards dead space around the already interpretable scene.
# g) Change inferability:
#    - No new hidden color mapping is introduced. Blue still means repeated empty impact and
#      maroon still means either a hit sprite or a lone empty impact.
#    - Because the crop is geometric rather than color-dependent, the fused variation is
#      inferable from shape differences between input and output without relying on palette ids.
# h) Fusion-specific non-degeneracy target:
#    - A generated sample is considered healthy only if the active bbox of the resolved full
#      output does not touch any input border. That guarantees `main(input)` differs from the
#      uncropped base-task output on every accepted sample.
# i) New solver assumptions and generator guarantees:
#    - `main()` assumes every meaningful foreground component is a 3-pixel triangle; malformed
#      components are ignored instead of crashing. The generator guarantees all components are valid.
#    - `main()` assumes there is at least one non-background pixel after the transform when a
#      strict crop is expected. The generator guarantees several sprites, so the active bbox exists.
#    - `main()` treats all projectile effects as simultaneous on the full board before cropping.
#      The generator mirrors that exact sequencing when it builds expected outputs for verification.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg as _bg_color, neighbors4, neighbors8


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
    dr_sum = d1[0] + d2[0]
    dc_sum = d1[1] + d2[1]
    dr = -(dr_sum // 2)
    dc = -(dc_sum // 2)
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return grid.copy()
    r0, c0, r1, c1 = bbox_coords(coords)
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve triangle-shot interactions on a variable-size board, then return the active crop only.

    Input structure:
    - The board size can vary from example to example, but the most common color is still the background.
    - Every foreground object is a monochrome 3-pixel triangle sprite: either an axis-aligned L or a
      diagonal V, separated so components stay distinct under 8-connectivity.

    Rule:
    - Decode each sprite's apex and firing direction from its geometry. L sprites shoot diagonally away
      from the missing corner of their 2x2 box, while V sprites shoot straight away from their opening.
    - The landing cell is exactly 5 steps from the apex in that decoded direction.
    - If a landing cell lies on a sprite, recolor that whole sprite to maroon (9).
    - If a landing cell is empty background, paint it maroon (9), except when two or more projectiles
      share that same empty landing cell, in which case that cell becomes blue (1).
    - After all of those effects are resolved on the full board, crop away every all-background border
      row and column so the output is the minimal non-background bounding box.

    Key insight:
    - The projectile reasoning is unchanged from the base task, but the board frame is no longer fixed:
      inputs can have different sizes, and outputs keep only the compact region that contains the final
      active pixels produced by those 5-step shots.
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
    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return _tight_crop_non_bg(out, bg)


MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 20
ORIGINAL_SHAPE = (16, 16)
ACTIVE_MARGIN = 1
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

MAX_ATTEMPTS = 600
PLACEMENT_CACHE: dict[tuple[int, int], tuple[list[dict], dict[tuple[int, int], list[dict]]]] = {}
EVENT_MODE_CYCLE = ("hit_only", "collision_only", "mixed")
_GEN_COUNTER = 0


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _cell_in_active_window(cell: tuple[int, int], h: int, w: int, margin: int = ACTIVE_MARGIN) -> bool:
    r, c = cell
    return margin <= r < h - margin and margin <= c < w - margin


def _cells_in_active_window(
    cells: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    margin: int = ACTIVE_MARGIN,
) -> bool:
    return all(_cell_in_active_window(cell, h, w, margin) for cell in cells)


def _in_bounds_cells(cells: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in cells)


def _halo(cells: tuple[tuple[int, int], ...], h: int, w: int) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(h, r + 2))
        for cc in range(max(0, c - 1), min(w, c + 2))
    )


def _build_placements(h: int, w: int) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    for ar in range(h):
        for ac in range(w):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not _cell_in_active_window(dest, h, w):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells, h, w):
                    continue
                if not _cells_in_active_window(cells, h, w):
                    continue
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells, h, w),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


def _get_placements(h: int, w: int) -> tuple[list[dict], dict[tuple[int, int], list[dict]]]:
    key = (h, w)
    if key not in PLACEMENT_CACHE:
        PLACEMENT_CACHE[key] = _build_placements(h, w)
    return PLACEMENT_CACHE[key]


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _choose_collision_pair(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> tuple[tuple[int, int], list[dict]] | None:
    dests = list(placements_by_dest)
    rng.shuffle(dests)
    for dest in dests:
        if dest in reserved_bg or dest in occupied:
            continue
        candidates = [p for p in placements_by_dest[dest] if _fits(p, occupied, reserved_bg)]
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
    placements: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> dict | None:
    candidates = placements[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if placement["dest"] in reserved_bg or placement["dest"] in occupied:
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


def _choose_hit(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
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
        candidates = [p for p in placements_by_dest.get(cell, []) if _fits(p, occupied, reserved_bg)]
        if candidates:
            return rng.choice(candidates), cell
    return None


def _validate_input(grid: np.ndarray, n_sprites: int, shape: tuple[int, int]) -> bool:
    if grid.shape != shape:
        return False
    if _bg_color(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    for comp in comps:
        if len(comp) != 3:
            return False
        points = [(int(r), int(c)) for r, c in comp]
        if _triangle_apex_and_dir(points) is None:
            return False
    return True


def _shape_event_caps(h: int, w: int) -> tuple[int, int]:
    if min(h, w) <= 11:
        return 1, 1
    return 2, 2


def _sprite_count_range(h: int, w: int) -> tuple[int, int]:
    area = h * w
    low = 3 if min(h, w) <= 11 else 4
    high = min(9, max(low + 1, int(round(area / 32))))
    return low, high


def _counts_match_mode(e_hit: int, e_coll: int, preferred_mode: str | None) -> bool:
    if preferred_mode is None:
        return True
    if preferred_mode == "hit_only":
        return e_hit >= 1 and e_coll == 0
    if preferred_mode == "collision_only":
        return e_coll >= 1 and e_hit == 0
    if preferred_mode == "mixed":
        return e_hit >= 1 and e_coll >= 1
    raise ValueError(f"Unknown preferred_mode: {preferred_mode}")


def _counts_for_n(
    rng: random.Random,
    n_sprites: int,
    max_hit_events: int,
    max_coll_events: int,
    preferred_mode: str | None = None,
) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(0, min(max_hit_events, n_sprites - 1) + 1):
        for e_coll in range(0, min(max_coll_events, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 0:
                continue
            if unique + e_coll < 1:
                continue
            if e_hit + e_coll < 1:
                continue
            options.append((e_hit, e_coll, unique))
    preferred = [opt for opt in options if _counts_match_mode(opt[0], opt[1], preferred_mode)]
    if preferred:
        return rng.choice(preferred)
    return rng.choice(options) if options else None


def _has_both_shape_families(sprites: list[dict]) -> bool:
    has_diag = False
    has_axis = False
    for sprite in sprites:
        dr, dc = sprite["placement"]["dir"]
        if abs(dr) + abs(dc) == 2:
            has_diag = True
        else:
            has_axis = True
        if has_diag and has_axis:
            return True
    return False


def _place_collisions(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    e_coll: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, placements_by_dest, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target": dest, "target_kind": "bg"})
            occupied.update(placement["cells"])
    return True


def _place_unique_landings(
    rng: random.Random,
    placements: list[dict],
    n: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, placements, occupied, reserved_bg)
        if placement is None:
            return False
        reserved_bg.add(placement["dest"])
        sprites.append({"placement": placement, "target": placement["dest"], "target_kind": "bg"})
        occupied.update(placement["cells"])
    return True


def _place_hits(
    rng: random.Random,
    placements_by_dest: dict[tuple[int, int], list[dict]],
    e_hit: int,
    sprites: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, placements_by_dest, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprites.append({"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)
    return True


def _render_output_full(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _bbox_touches_border(grid: np.ndarray, bg: int) -> bool:
    coords = np.argwhere(grid != bg)
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    return r0 == 0 or c0 == 0 or r1 == grid.shape[0] - 1 or c1 == grid.shape[1] - 1


def _sample_shape(rng: random.Random) -> tuple[int, int]:
    while True:
        h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        if (h, w) != ORIGINAL_SHAPE:
            return h, w


def _gen_attempt(rng: random.Random, preferred_mode: str | None = None) -> np.ndarray | None:
    h, w = _sample_shape(rng)
    placements, placements_by_dest = _get_placements(h, w)
    if not placements:
        return None

    sprite_color = rng.choice(SPRITE_COLORS)
    n_low, n_high = _sprite_count_range(h, w)
    n_sprites = rng.randint(n_low, n_high)
    max_hit_events, max_coll_events = _shape_event_caps(h, w)
    counts = _counts_for_n(rng, n_sprites, max_hit_events, max_coll_events, preferred_mode)
    if counts is None:
        return None
    e_hit, e_coll, unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    if not _place_collisions(rng, placements_by_dest, e_coll, sprites, occupied, reserved_bg):
        return None

    base_needed = max(0, e_hit - len(sprites))
    if base_needed > unique_total:
        return None
    if not _place_unique_landings(rng, placements, base_needed, sprites, occupied, reserved_bg):
        return None

    if not _place_hits(rng, placements_by_dest, e_hit, sprites, occupied, reserved_bg):
        return None

    remaining = n_sprites - len(sprites)
    if remaining < 0:
        return None
    if not _place_unique_landings(rng, placements, remaining, sprites, occupied, reserved_bg):
        return None
    if not _has_both_shape_families(sprites):
        return None

    inp = np.full((h, w), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, (h, w)):
        return None

    out_full = _render_output_full(inp, sprites)
    if _bbox_touches_border(out_full, BG):
        return None
    out = _tight_crop_non_bg(out_full, BG)

    if np.count_nonzero(out_full != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out_full == HIT_COLOR):
        return None
    if np.array_equal(inp, out_full):
        return None
    if out.shape == (h, w):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    global _GEN_COUNTER
    rng = random.Random()
    preferred_mode = EVENT_MODE_CYCLE[_GEN_COUNTER % len(EVENT_MODE_CYCLE)]
    fallback_after = MAX_ATTEMPTS // 2
    for attempt in range(MAX_ATTEMPTS):
        mode = preferred_mode if attempt < fallback_after else None
        grid = _gen_attempt(rng, preferred_mode=mode)
        if grid is not None:
            _GEN_COUNTER += 1
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
