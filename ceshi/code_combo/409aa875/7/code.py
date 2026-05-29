# Combo variant of 409aa875 (index 7)
# combo_spec: D6/0+D12/1
# Name: 连续轨迹+方向翻转

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The canvas stays 16x16 with bg=7 and 4-9 monochrome 3-pixel triangle sprites.
#      The two legal sprite families are unchanged: axis-aligned L components and
#      diagonal V components.
#    - The fused rule changes two stages at once:
#      1) direction semantics flips relative to the source task
#         * L: shoot toward the missing 2x2 corner
#         * V: shoot toward the open side
#      2) the whole 5-step flight becomes visible; steps 1..4 are painted maroon 9,
#         and step 5 keeps the original landing semantics (hit sprite -> recolor to 9,
#         lone bg landing -> 9, shared bg landing -> 1).
#    - Small trace:
#      * L example cells {(6,6),(7,6),(6,7)} have apex (6,6) and missing corner (7,7).
#        Source task: direction (-1,-1), only landing (1,1) was marked.
#        Combo: direction flips to (1,1), trail paints (7,7),(8,8),(9,9),(10,10), and
#        landing logic is applied at (11,11).
#      * V example cells {(6,6),(7,5),(7,7)} have apex (6,6) and open side downward.
#        Source task fired upward; combo fires downward, paints a 4-cell vertical trail,
#        then resolves the landing 5 cells below the apex.
#    - The important difference is not just "more paint": the trail now points in the
#      reversed direction, so it visually teaches the corrected semantic rule.
#
# b) Complexity calibration
#    - The variant still uses only the original eight canonical directions. There is no
#      per-color or per-instance exception, so the rule remains inferable from geometry.
#    - Continuous trails make the direction change observable without introducing new
#      object types. Trail color is fixed to 9, while only empty shared landings use 1.
#    - To avoid trivial or misleading samples, the generator rejects cases where
#      "inverted+trail" matches either "original-direction+trail" or "inverted-no-trail".
#      That forces both fused dimensions to matter.
#
# c) Hidden-assumption audit of the original solver
#    - The original `_triangle_apex_and_dir(...)` encoded "shoot away from the gap/opening".
#      That assumption no longer holds; only apex detection is preserved.
#    - The original painting loop assumed only the destination cell changed. The combo
#      needs a two-phase write order: paint trail cells first, then overwrite the landing
#      cell or hit sprite result so endpoint semantics stay dominant.
#    - `cell_to_comp` must still come from the input grid, not from partially painted
#      output, otherwise trail pixels could be mistaken for sprite occupancy.
#    - The original solver was indifferent to intermediate path crossings. The combo is
#      not: if trails crossed sprite cells in the input, the visible ray would become
#      ambiguous. The generator therefore reserves all mid-trail cells as background-only.
#
# d) Placement legality
#    - Every placement must keep the sprite itself, the 5-step landing cell, and all
#      trail cells in bounds.
#    - Sprites keep Chebyshev separation >= 2 so components never merge.
#    - Mid-trail cells may not overlap any sprite cell or previously reserved trail cell.
#      Two-shot collisions are allowed only at the shared empty destination cell.
#    - Hit events still target an already placed sprite cell, but only with a trail whose
#      first four cells are all background in the input.
#
# e) Visual observability
#    - A solver can now read direction from the straight maroon ray, then match that ray
#      against the L/V geometry and notice the shot goes into the gap/opening rather than
#      away from it.
#    - The generator requires at least one L shooter and one V shooter, so both halves of
#      the flipped semantic rule are represented in every sample.
#
# f) Visual cleanliness
#    - Trails are the only dense marks, so the generator forbids trail overlaps except at
#      an intentional shared destination. This keeps the output legible.
#    - Only two output colors are used for new markings: 9 for trails, lone landings, and
#      hit sprites; 1 for repeated empty-cell landings.
#
# g) New solver assumptions and generator guarantees
#    - `main()` assumes each non-bg component is a canonical 3-pixel L or V. The generator
#      validates every component with the same direction parser.
#    - `main()` assumes trail cells are background in the input whenever it paints them.
#      Placement helpers reserve those cells before later sprites are added.
#    - `main()` assumes the direction flip has visible consequences. The generator measures
#      both trail gain and direction delta and rejects degenerate boards.

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
TRAIL_COLOR = 9
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

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
MAX_ATTEMPTS = 800


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _normalize_diag(dr: int, dc: int) -> tuple[int, int]:
    return (1 if dr > 0 else -1), (1 if dc > 0 else -1)


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Variant semantics: L shoots toward the missing corner; V shoots toward the opening."""
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
        return corner, _normalize_diag(dr_sum, dc_sum)

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
    """Reference semantics from the source task, used only for anti-degeneracy checks."""
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


def _trail_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    return tuple((ar + step * dr, ac + step * dc) for step in range(1, PROJECTILE_DIST + 1))


def _solve_with_direction_rule(
    grid: np.ndarray,
    direction_fn,
    draw_trails: bool,
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

    shots: list[tuple[tuple[int, int], ...]] = []
    dest_to_shooters: dict[tuple[int, int], list[int]] = defaultdict(list)
    for comp in comps:
        ad = direction_fn(comp)
        if ad is None:
            continue
        trail = _trail_cells(*ad)
        rr, cc = trail[-1]
        if 0 <= rr < h and 0 <= cc < w:
            sid = len(shots)
            shots.append(trail)
            dest_to_shooters[(rr, cc)].append(sid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    out = g.copy()

    if draw_trails:
        for trail in shots:
            for r, c in trail[:-1]:
                if 0 <= r < h and 0 <= c < w and g[r, c] == bg:
                    out[r, c] = TRAIL_COLOR

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
    Three-pixel triangle sprites fire inward-facing shots and leave full straight trails.

    Input structure:
    - The majority color is background. Every non-background object is a monochrome 3-cell
      component shaped either as an axis-aligned L or as a diagonal V.

    Rule:
    - Infer one apex and one firing direction from each sprite. L-shapes fire diagonally
      toward the missing 2x2 corner, while V-shapes fire along the axis toward their open side.
    - The projectile travels exactly 5 cells from the apex. Steps 1..4 on that path are painted
      maroon (9) when they pass through background.
    - At step 5, the landing rule is resolved: landing on a sprite recolors that whole sprite to
      maroon (9); landing on empty background paints maroon (9); if multiple shots share an empty
      landing cell, that cell becomes blue (1).

    Key insight:
    - The same geometry now controls both a visible straight ray and a reversed directional
      meaning, so the trail serves as evidence that shots travel into each sprite's gap/opening.
    """
    return _solve_with_direction_rule(grid, _triangle_apex_and_dir, draw_trails=True)


def _sprite_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    """Canonical sprite cells for the combo semantics."""
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
                trail = _trail_cells((ar, ac), (dr, dc))
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                    "trail": trail,
                    "trail_mid": trail[:-1],
                    "trail_mid_set": frozenset(trail[:-1]),
                    "trail_set": frozenset(trail),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
    return placements, by_dest


PLACEMENTS, PLACEMENTS_BY_DEST = _build_placements()


def _sprite_fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _trail_fits_bg(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if placement["dest"] in occupied or placement["dest"] in reserved_bg:
        return False
    if placement["trail_mid_set"] & occupied:
        return False
    if placement["trail_set"] & reserved_bg:
        return False
    return True


def _trail_fits_shared_dest(
    placement: dict,
    dest: tuple[int, int],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> bool:
    if placement["dest"] != dest:
        return False
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if placement["trail_mid_set"] & occupied:
        return False
    if placement["trail_mid_set"] & reserved_bg:
        return False
    return True


def _trail_fits_hit(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    if not _sprite_fits(placement, occupied, reserved_bg):
        return False
    if placement["trail_mid_set"] & occupied:
        return False
    if placement["trail_mid_set"] & reserved_bg:
        return False
    return True


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _reserve_sprite(
    sprite: dict,
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> None:
    placement = sprite["placement"]
    occupied.update(placement["cells"])
    reserved_bg.update(placement["trail_mid"])
    if sprite["target_kind"] == "bg":
        reserved_bg.add(sprite["target"])


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
        candidates = [p for p in PLACEMENTS_BY_DEST[dest] if _trail_fits_bg(p, occupied, reserved_bg)]
        rng.shuffle(candidates)
        for i, first in enumerate(candidates):
            occ2 = occupied | set(first["cells"])
            res2 = reserved_bg | set(first["trail_mid"]) | {dest}
            for second in candidates[i + 1:]:
                if _trail_fits_shared_dest(second, dest, occ2, res2):
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
        if _trail_fits_bg(placement, occupied, reserved_bg):
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
        candidates = [p for p in PLACEMENTS_BY_DEST[cell] if _trail_fits_hit(p, occupied, reserved_bg)]
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
    return all(_triangle_apex_and_dir(comp) is not None for comp in comps)


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
    if not options:
        return None

    preferred = [
        counts
        for counts in options
        if counts[0] >= 1 and counts[1] >= 1 and counts[2] >= 1
    ]
    return rng.choice(preferred or options)


def _place_collisions(rng, e_coll, sprites, occupied, reserved_bg) -> bool:
    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return False
        dest, pair = choice
        for placement in pair:
            sprite = {"placement": placement, "target": dest, "target_kind": "bg"}
            sprites.append(sprite)
            _reserve_sprite(sprite, occupied, reserved_bg)
    return True


def _place_unique_landings(rng, n, sprites, occupied, reserved_bg) -> bool:
    for _ in range(n):
        placement = _choose_unique(rng, occupied, reserved_bg)
        if placement is None:
            return False
        sprite = {"placement": placement, "target": placement["dest"], "target_kind": "bg"}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
    return True


def _place_hits(rng, e_hit, sprites, occupied, reserved_bg) -> bool:
    used_target_cells: set[tuple[int, int]] = set()
    hit_targets = rng.sample(range(len(sprites)), e_hit)
    for idx in hit_targets:
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return False
        placement, cell = choice
        sprite = {"placement": placement, "target": cell, "target_kind": "hit", "target_sprite": idx}
        sprites.append(sprite)
        _reserve_sprite(sprite, occupied, reserved_bg)
        used_target_cells.add(cell)
    return True


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}

    for sprite in sprites:
        for r, c in sprite["placement"]["trail_mid"]:
            out[r, c] = TRAIL_COLOR

    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)

    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR

    return out


def _fusion_metrics(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> dict[str, int]:
    old_trail_out = _solve_with_direction_rule(inp, _triangle_apex_and_dir_original, draw_trails=True)
    no_trail_out = _solve_with_direction_rule(inp, _triangle_apex_and_dir, draw_trails=False)
    diag_count = sum(1 for s in sprites if abs(s["placement"]["dir"][0]) + abs(s["placement"]["dir"][1]) == 2)
    axis_count = len(sprites) - diag_count
    trail_gain = int(np.count_nonzero((out != no_trail_out) & (inp == BG)))
    direction_delta = int(np.count_nonzero(out != old_trail_out))
    return {
        "trail_gain": trail_gain,
        "direction_delta": direction_delta,
        "diag_count": diag_count,
        "axis_count": axis_count,
    }


def _variant_is_observable(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> bool:
    metrics = _fusion_metrics(inp, out, sprites)
    if metrics["diag_count"] < 1 or metrics["axis_count"] < 1:
        return False
    if metrics["trail_gain"] < 16:
        return False
    if metrics["direction_delta"] < 1:
        return False
    return True


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
