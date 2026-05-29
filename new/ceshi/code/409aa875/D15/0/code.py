# Variant of 409aa875
# Dimension: D15 — 网格尺寸 (meta)
# Change: 固定16x16正方形网格... → 可变尺寸（10x10到20x20）
#
# Design analysis
# spec:
#   core_change:
#     - The firing rule stays the same: each 3-pixel triangle emits toward apex + 5 * dir.
#     - Only the canvas size changes. Each generated board is rectangular with both sides in [10, 20].
#     - To make the variation visible in every example, the generator never emits the original 16x16 size.
#   concrete_behavior:
#     - Inputs still contain monochrome 3-cell L sprites and diagonal V sprites on background 7.
#     - Outputs still recolor hit sprites / lone empty impacts to 9 and shared empty impacts to 1.
#     - What changes visually is the frame itself: the same local projectile logic must now be read on
#       different board sizes, including non-square boards.
#   flow_trace:
#     - Example: on a 12x18 board, an L sprite with apex (6, 6) and dir (-1, -1) lands at (1, 1).
#       In the original generator, legality was checked against one global 16x16 placement table.
#       In this variant, legality must be recomputed for 12x18; otherwise the generator would either
#       miss valid placements or reuse stale 16x16 bounds near the edges.
#   complexity_calibration:
#     - Keep the object vocabulary, colors, and distance fixed so D15 is the only conceptual change.
#     - Small boards get slightly fewer sprites/events than large boards; this is an implementation
#       adaptation so the original interaction pattern remains feasible instead of becoming overcrowded.
#   hidden_assumption_audit:
#     - main(): `h, w = g.shape` is already generic, and the in-bounds projectile guard still holds.
#     - The old generator relied on fixed globals `H`, `W`, `PLACEMENTS`, `PLACEMENTS_BY_DEST`,
#       `_halo`, `_validate_input`, and `np.full((H, W), BG)`. Those assumptions break once size varies.
#     - Caching placement tables is still safe, but only when the cache key is the current `(h, w)`.
#   placement_legality:
#     - Every sprite cell and every landing cell must stay in bounds for the sampled shape.
#     - Sprites remain Chebyshev-separated by at least 2 so components never merge under 8-connectivity.
#     - Reserved empty landing cells cannot later be occupied by sprites, and hit events only target
#       already placed sprites.
#   visual_observability:
#     - Excluding 16x16 prevents accidental regression to the original presentation.
#     - Allowing both square and rectangular boards makes the size variation visible from a few examples.
#   cleanliness:
#     - A single sprite color per board and the original sparse spacing keep resized boards readable.
#   color_inference:
#     - No new color mapping is introduced; blue still means shared empty impact and maroon still means
#       either a hit sprite or a lone empty impact.
#   new_generator_guarantees_for_main:
#     - Every non-background component is still a valid 3-pixel L or V sprite.
#     - All intended projectile destinations are in bounds for the sampled board, so solver verification
#       remains exact on every generated example.

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, get_bg, neighbors4, neighbors8


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve 5-step projectile landings from tiny triangle sprites on variable-size boards.

    Input structure:
    - The background is the most common color, and every foreground component is a monochrome 3-pixel
      triangle sprite of one of two types: an axis-aligned L or a diagonal V. In this variant the board
      size may change from example to example, but the sprite vocabulary does not.

    Rule:
    - Infer each sprite's apex and firing direction from its shape: L sprites shoot diagonally away from
      the missing corner of their 2x2 box, while V sprites shoot straight away from their opening.
    - The landing cell is exactly 5 steps from the apex in that direction.
    - If that landing cell lies on another sprite, recolor the entire hit sprite to maroon (9).
    - If multiple projectiles land on the same empty background cell, color that cell blue (1); a single
      projectile landing on empty background colors it maroon (9).

    Key insight:
    - The only meta change is board size. The local orientation-to-direction decoding is unchanged, so the
      solver must apply the same 5-step targeting rule consistently on whatever grid dimensions appear.
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

    return out


MIN_GRID_SIZE = 10
MAX_GRID_SIZE = 20
ORIGINAL_SHAPE = (16, 16)
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

MAX_ATTEMPTS = 400
PLACEMENT_CACHE: dict[tuple[int, int], tuple[list[dict], dict[tuple[int, int], list[dict]]]] = {}


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if abs(dr) + abs(dc) == 2:
        return ((ar, ac), (ar - dr, ac), (ar, ac - dc))
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


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
                if not (0 <= dest[0] < h and 0 <= dest[1] < w):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells, h, w):
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
    if get_bg(grid) != BG:
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


def _counts_for_n(
    rng: random.Random,
    n_sprites: int,
    max_hit_events: int,
    max_coll_events: int,
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
    return rng.choice(options) if options else None


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


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _sample_shape(rng: random.Random) -> tuple[int, int]:
    while True:
        h = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        w = rng.randint(MIN_GRID_SIZE, MAX_GRID_SIZE)
        if (h, w) != ORIGINAL_SHAPE:
            return h, w


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = _sample_shape(rng)
    placements, placements_by_dest = _get_placements(h, w)
    if not placements:
        return None

    sprite_color = rng.choice(SPRITE_COLORS)
    n_low, n_high = _sprite_count_range(h, w)
    n_sprites = rng.randint(n_low, n_high)
    max_hit_events, max_coll_events = _shape_event_caps(h, w)
    counts = _counts_for_n(rng, n_sprites, max_hit_events, max_coll_events)
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

    inp = np.full((h, w), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites, (h, w)):
        return None

    out = _render_output(inp, sprites)
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not np.any(out == HIT_COLOR):
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
    rng = random.Random()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
