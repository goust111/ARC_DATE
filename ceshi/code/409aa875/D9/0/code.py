# Variant of 409aa875
# Dimension: D9 — 精灵颜色策略 (meta)
# Change: 所有精灵共享同一随机颜色（从9种非背景色中选1种）... → 多色精灵（每个精灵独立随机颜色）

"""
design_analysis:
  variant_goal:
    keep_fixed:
      - 16x16 canvas, bg=7
      - every sprite is still a 3-cell monochrome L or V triangle
      - each triangle still fires exactly 5 cells from its apex along the same direction logic
      - empty-bg outcomes stay the same: lone landing -> 9, multi-landing -> 1
    change_only:
      - sprites are no longer globally one color
      - when a projectile lands on a sprite, the whole hit sprite changes to the shooter's color
        instead of always changing to 9

  concrete_behavior_and_trace:
    input_view:
      - the input is still only triangles on a plain background; there are no visible projectile markers
      - each triangle is monochrome, but different triangles use different colors
    output_view:
      - background landing cells still get 9 or 1
      - a hit sprite copies the color of the shooter that landed on it
    traced_example:
      - suppose a green shooter fires into a yellow target sprite
      - original rule: yellow target becomes maroon 9
      - variant rule: yellow target becomes green, because the solver must identify which sprite fired there
      - difference: the solver can no longer mark "hit happened" with a constant color; it must carry shooter identity

  complexity_calibration:
    risks_if_too_simple:
      - if a sample has no sprite-hit event, the new color-transfer rule is invisible
      - if the shooter color is 9, the hit sprite would still become 9 and could look identical to the old task
    generator_controls:
      - every sample contains at least one sprite-hit event
      - hit-shooters are assigned colors from a safe subset excluding 1 and 9, so at least one recolored target visibly differs from the old task
      - sprite colors are unique per sample, which makes "who hit whom" visually traceable
    why_not_more_freedom:
      - geometry, projectile distance, and bg marker colors stay unchanged, so D9 remains the only conceptual novelty

  implicit_assumption_audit_against_original_main:
    original_dependency_1:
      code_pattern: "recolor every hit component to constant 9"
      old_reason: all hit sprites shared the same forced output color
      variant_status: invalid; must instead look up each shooter's own sprite color
    original_dependency_2:
      code_pattern: "group components by color and then by connectivity"
      old_reason: all sprites were monochrome and usually same-colored
      variant_status: still workable, but weaker than necessary; new solver instead extracts all non-bg 8-components directly
    original_dependency_3:
      code_pattern: "set of hit components is enough"
      old_reason: only hit/not-hit mattered
      variant_status: invalid; new solver needs a mapping target_component -> shooter_color
    original_dependency_4:
      code_pattern: "multiple shooters hitting one sprite do not matter"
      old_reason: they all produced the same 9 anyway
      variant_status: now ambiguous if colors differ, so generator guarantees at most one hit-shooter per target sprite

  placement_legality:
    guarantees:
      - sprites remain pairwise Chebyshev-separated by at least 2 via halo checks
      - every projectile landing cell is in bounds
      - hit-shooters are added only after their target sprites already exist
      - background landing cells are reserved so later sprite placements do not occupy them
      - non-hit shooters never land on an occupied sprite cell

  visual_observability:
    thought_experiment:
      - in 2-4 examples, a human can match a recolored target to the only triangle whose line-of-fire reaches it
      - because the copied color is already present on the shooter, the mapping is visually grounded and not color-id arithmetic
    anti-degeneracy:
      - generator rejects any sample whose output matches the old fixed-9 output

  visual_cleanliness:
    choices:
      - keep original spacing rules, so differently colored triangles do not merge
      - keep bg markers as single cells only, avoiding extra overlays
      - limit sprite-hit count to at most 2, so recolor relationships stay readable

  color_change_inference:
    structure:
      - recolored targets always copy an existing sprite color already visible in the input
      - copied colors come from shooter sprites, not from palette order or hidden numeric rules
      - every sample has at least one non-9 transferred color, so the change is inferable from examples

  new_main_assumption_audit:
    assumption_1:
      - each sprite is a single 8-connected 3-cell component
      - generator enforces this with placement geometry and validation
    assumption_2:
      - each hit target is hit by at most one shooter
      - generator enforces unique target-sprite selection for hit events
    assumption_3:
      - each sprite is monochrome, so its shooter color is well-defined
      - generator paints one solid color per sprite
"""

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


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
    Fire projectiles from 3-pixel triangle sprites; hit sprites inherit the shooter's color.

    Input structure:
    - Background is the most common color. Every non-background 8-connected component is a 3-cell monochrome triangle sprite, and different sprites may use different colors.

    Rule:
    - Each triangle's shape determines its firing direction: L-shapes fire diagonally away from the missing corner, while V-shapes fire along the axis opposite their opening.
    - The projectile lands exactly 5 cells from the apex in that direction.
    - If the landing cell lies on another sprite, the entire hit sprite is recolored to the firing sprite's color.
    - If the landing cell is empty background, a lone projectile paints 9 and a shared landing by 2 or more projectiles paints 1.

    Key insight: Direction is still encoded by shape alone, but the visible color change on a hit sprite now depends on tracing which specific colored shooter reached it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps = _cc8(g != bg)
    cell_to_comp: dict[tuple[int, int], int] = {}
    comp_colors: list[int] = []
    for cid, comp in enumerate(comps):
        for rc in comp:
            cell_to_comp[rc] = cid
        comp_colors.append(int(g[comp[0]]))

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

    out = g.copy()

    hit_color_by_comp: dict[int, int] = {}
    for dest, shooters in dest_to_shooters.items():
        target_cid = cell_to_comp.get(dest)
        if target_cid is None:
            continue
        if shooters:
            # Samples are generated so a target sprite has at most one incoming shooter.
            hit_color_by_comp[target_cid] = comp_colors[shooters[0]]

    for cid, new_color in hit_color_by_comp.items():
        for r, c in comps[cid]:
            out[r, c] = new_color

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 1, 2, 3, 4, 5, 6, 8, 9]
TRACEABLE_HIT_SHOOTER_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
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
            return rng.choice(candidates), cell
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
    hit_colors = rng.sample(TRACEABLE_HIT_SHOOTER_COLORS, len(hit_shooter_ids))
    for idx, color in zip(hit_shooter_ids, hit_colors):
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
    for sprite in sprites:
        if sprite["target_kind"] == "hit":
            _paint_sprite(out, sprites[sprite["target_sprite"]]["placement"], sprite["color"])
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _render_legacy_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    bg_targets = Counter(s["target"] for s in sprites if s["target_kind"] == "bg")
    hit_sprite_ids = {s["target_sprite"] for s in sprites if s["target_kind"] == "hit"}
    for idx in hit_sprite_ids:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in bg_targets.items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out


def _variant_metrics(sprites: list[dict], out: np.ndarray) -> dict[str, int]:
    recolored_non9 = sum(1 for s in sprites if s["target_kind"] == "hit" and s["color"] != 9)
    return {
        "hit_events": sum(1 for s in sprites if s["target_kind"] == "hit"),
        "collision_events": sum(1 for dest, count in Counter(s["target"] for s in sprites if s["target_kind"] == "bg").items() if count >= 2),
        "transferred_non9_hits": recolored_non9,
        "distinct_sprite_colors": len({s["color"] for s in sprites}),
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
    legacy_out = _render_legacy_output(inp, sprites)
    metrics = _variant_metrics(sprites, out)

    if np.array_equal(out, legacy_out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if metrics["hit_events"] < 1 or metrics["transferred_non9_hits"] < 1:
        return None
    if metrics["distinct_sprite_colors"] != n_sprites:
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
