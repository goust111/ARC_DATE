# Combo variant of 409aa875 (index 14)
# combo_spec: D9/0+D10/0
# Name: 多色精灵+标记色追溯发射者

"""
design_analysis:
  fusion_goal:
    keep_fixed:
      - 16x16 canvas, bg=7, 3-cell L/V triangle sprites, and projectile distance 5
      - shape still determines firing direction exactly as in the base task
      - input still shows only sprites; landing markers appear only in the output
    change_only:
      - sprites are multicolor: each sprite is independently assigned one source color
      - any visible recolor must be traced back to the shooter color
      - a lone background landing copies the shooter color
      - a shared background landing uses a predefined mix color for the involved shooter colors

  concrete_behavior_and_trace:
    traced_example:
      - suppose a red sprite (2) fires into a green sprite (3)
      - original task: the green target becomes constant maroon 9
      - D9 change: the green target becomes red 2, so the solver must identify the shooter
      - now add D10: if a teal sprite (8) lands alone on empty bg, that empty cell becomes teal 8,
        not maroon 9; if red (2) and teal (8) both land on the same empty bg cell, that cell
        becomes the predefined mixed color 6
      - difference from the base task: every output color now answers "which shooter(s) caused
        this mark?" rather than "which event type happened?"

  complexity_calibration:
    small_palette_choice:
      - all sprites draw from exactly three source colors {2,3,8}
      - pairwise background collisions use a tiny symmetric mix table:
          {2,3}->4, {2,8}->6, {3,8}->9
      - a 3-color fallback {2,3,8}->5 is defined for solver completeness, but generation only
        uses 2-shooter collisions so humans only need the pair table
    why_this_is_not_too_simple:
      - every generated grid contains at least one sprite hit, one lone bg landing, and one
        shared bg collision, so all three fused effects are visible together
      - the source palette is small enough for repeated evidence across 2-4 samples, but not so
        small that the task collapses back to the monochrome original
    why_this_is_not_too_free:
      - only three source colors and three pair-mix outputs are used; there is no arbitrary
        many-to-many color arithmetic

  implicit_assumption_audit_against_original_main:
    original_dependency_1:
      code_pattern: "group components by color and then recolor hits to constant 9"
      variant_status: invalid; components must be extracted from all non-bg cells directly, and
        hit targets need the incoming shooter color
    original_dependency_2:
      code_pattern: "hit_components is a set"
      variant_status: invalid; hit handling now needs target_component -> shooter_color
    original_dependency_3:
      code_pattern: "all empty-bg collisions become 1 and lone bg hits become 9"
      variant_status: invalid; bg marking now depends on shooter colors and the mix table
    original_dependency_4:
      code_pattern: "multiple shooters hitting one sprite do not matter"
      variant_status: unsafe once colors differ; generator enforces at most one incoming hit
        shooter per target sprite
    original_dependency_5:
      code_pattern: "all shooters sharing a destination can be summarized by count only"
      variant_status: invalid; solver must preserve the shooter colors that contributed

  placement_legality:
    guarantees:
      - every sprite placement and every landing cell is in bounds
      - halo spacing keeps sprites at Chebyshev distance >= 2, so 8-connected components stay
        separate and target cells never ambiguously belong to two sprites
      - collision destinations are reserved as empty background so no later sprite occupies them
      - hit shooters are added only after target sprites already exist
      - collision events use exactly two shooters with distinct source colors

  visual_observability:
    thought_experiment:
      - in a sample pair, a recolored target copies a visible shooter color already present in
        the input, so the human can trace "who hit whom" geometrically
      - lone bg marks match an existing shooter sprite exactly, while shared bg marks use one of
        only three repeated mix colors, giving direct support for the source-tracing rule
    anti_degeneracy:
      - generation rejects any sample whose output matches the original fixed-(9,1) marker task
      - generation also rejects any sample whose output matches the D9-only variant where bg
        markers are still fixed to 9/1

  visual_cleanliness:
    choices:
      - keep the original sparse spacing and single-cell landing markers
      - use a small, repeated source/mix palette so the grid does not become a noisy rainbow
      - cap sprite-hit and collision counts to preserve readable causal relationships

  color_inferability:
    structure:
      - every visible output color is either a source color already seen on a shooter or one of
        three repeated pair-mix colors
      - no rule depends on numeric ordering of colors; the only cue is geometric causality
      - accepted samples must show at least one visible hit transfer and at least one visible
        mixed collision cell

  new_main_assumption_audit:
    assumption_1:
      - each non-bg component is exactly one monochrome 3-cell triangle
      - generator validates this after painting
    assumption_2:
      - every hit target has at most one incoming shooter
      - generator enforces distinct target sprites for hit events
    assumption_3:
      - every collision cell is caused by at most the three allowed source colors
      - generator uses only the fixed source palette and only 2-shooter collisions
    assumption_4:
      - shooter color means the input color of the shooting sprite, even if that shooter is
        recolored later by another incoming shot
      - solver reads all shooter colors from the unchanged input before writing output
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


H = W = 16
BG = 7
SOURCE_COLORS = [2, 3, 8]
PAIR_MIX_COLOR = {
    frozenset((2, 3)): 4,
    frozenset((2, 8)): 6,
    frozenset((3, 8)): 9,
}
SET_MIX_COLOR = {
    **PAIR_MIX_COLOR,
    frozenset((2, 3, 8)): 5,
}
MIX_OUTPUT_COLORS = set(SET_MIX_COLOR.values())

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 9)
MAX_HIT_EVENTS = 2
MAX_COLL_EVENTS = 2
MAX_ATTEMPTS = 800


def _landing_color_from_shooter_colors(shooter_colors: list[int]) -> int:
    unique = frozenset(int(c) for c in shooter_colors)
    if len(unique) == 1:
        return next(iter(unique))
    if unique in SET_MIX_COLOR:
        return SET_MIX_COLOR[unique]
    raise ValueError(f"Unsupported shooter color set for collision: {sorted(unique)}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from multicolor 3-pixel triangle sprites; every visible mark inherits shooter color information.

    Input structure:
    - The background is the most common color.
    - Every non-background 8-connected component is a monochrome 3-cell triangle sprite, either an axis-aligned L or a diagonal V.
    - Different sprites may use different source colors, and no landing markers are drawn in the input.

    Rule:
    - Recover each triangle's apex and firing direction from its shape: L-shapes fire diagonally away from the missing 2x2 corner, while V-shapes fire along the axis opposite their opening.
    - The projectile lands exactly 5 cells from the apex in that direction.
    - If the landing cell lies on another sprite, recolor that entire hit sprite to the firing sprite's input color.
    - If the landing cell is empty background, paint it with the shooter's color when only one projectile lands there; if multiple projectiles share that bg cell, paint it with the predefined mixed color for the contributing shooter colors.

    Key insight: Geometry still tells you where each shot goes, but the output colors now encode shooter identity: single effects copy one shooter color, and shared background collisions combine the two shooter colors instead of using fixed event colors.
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
        rr = ar + PROJECTILE_DIST * dr
        cc = ac + PROJECTILE_DIST * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters[(rr, cc)].append(cid)

    out = g.copy()

    hit_colors_by_comp: dict[int, list[int]] = defaultdict(list)
    for dest, shooters in dest_to_shooters.items():
        target_cid = cell_to_comp.get(dest)
        if target_cid is None or not shooters:
            continue
        hit_colors_by_comp[target_cid].extend(comp_colors[sid] for sid in shooters)

    for cid, shooter_colors in hit_colors_by_comp.items():
        new_color = _landing_color_from_shooter_colors(shooter_colors)
        for r, c in comps[cid]:
            out[r, c] = new_color

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = _landing_color_from_shooter_colors([comp_colors[sid] for sid in shooters])

    return out


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
    colors = set(int(v) for v in np.unique(grid)) - {BG}
    return colors <= set(SOURCE_COLORS)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for e_hit in range(1, min(MAX_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(1, min(MAX_COLL_EVENTS, n_sprites // 2) + 1):
            unique = n_sprites - e_hit - 2 * e_coll
            if unique < 1:
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


def _constraint_edges(sprites: list[dict]) -> dict[int, set[int]]:
    neq: dict[int, set[int]] = {idx: set() for idx in range(len(sprites))}
    for sprite in sprites:
        if sprite["target_kind"] == "hit":
            shooter_idx = int(sprite["id"])
            target_idx = int(sprite["target_sprite"])
            neq[shooter_idx].add(target_idx)
            neq[target_idx].add(shooter_idx)

    for dest, ids in _bg_groups(sprites).items():
        if len(ids) < 2:
            continue
        for i, left in enumerate(ids):
            for right in ids[i + 1 :]:
                neq[left].add(right)
                neq[right].add(left)
    return neq


def _assign_sprite_colors(rng: random.Random, sprites: list[dict]) -> bool:
    for idx, sprite in enumerate(sprites):
        sprite["id"] = idx

    neq = _constraint_edges(sprites)
    order = sorted(range(len(sprites)), key=lambda i: (-len(neq[i]), i))
    assigned: dict[int, int] = {}
    target_palette = set(SOURCE_COLORS)

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return set(assigned.values()) == target_palette

        idx = order[pos]
        domain = SOURCE_COLORS[:]
        rng.shuffle(domain)
        for color in domain:
            if any(assigned.get(nb) == color for nb in neq[idx]):
                continue
            assigned[idx] = color
            used = set(assigned.values())
            missing = len(target_palette - used)
            remaining = len(order) - pos - 1
            if missing <= remaining and backtrack(pos + 1):
                return True
            del assigned[idx]
        return False

    if not backtrack(0):
        return False

    for idx, sprite in enumerate(sprites):
        sprite["color"] = assigned[idx]
    return True


def _bg_groups(sprites: list[dict]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for idx, sprite in enumerate(sprites):
        if sprite["target_kind"] == "bg":
            groups[sprite["target"]].append(idx)
    return groups


def _render_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()

    for sprite in sprites:
        if sprite["target_kind"] == "hit":
            _paint_sprite(out, sprites[sprite["target_sprite"]]["placement"], sprite["color"])

    for dest, ids in _bg_groups(sprites).items():
        out[dest] = _landing_color_from_shooter_colors([sprites[idx]["color"] for idx in ids])

    return out


def _render_d9_only_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    for sprite in sprites:
        if sprite["target_kind"] == "hit":
            _paint_sprite(out, sprites[sprite["target_sprite"]]["placement"], sprite["color"])
    for dest, ids in _bg_groups(sprites).items():
        out[dest] = 1 if len(ids) >= 2 else 9
    return out


def _render_original_output(inp: np.ndarray, sprites: list[dict]) -> np.ndarray:
    out = inp.copy()
    hit_targets = {sprite["target_sprite"] for sprite in sprites if sprite["target_kind"] == "hit"}
    for idx in hit_targets:
        _paint_sprite(out, sprites[idx]["placement"], 9)
    for dest, ids in _bg_groups(sprites).items():
        out[dest] = 1 if len(ids) >= 2 else 9
    return out


def _variant_metrics(inp: np.ndarray, out: np.ndarray, sprites: list[dict]) -> dict[str, int]:
    bg_groups = _bg_groups(sprites)
    hit_events = [s for s in sprites if s["target_kind"] == "hit"]
    visible_hit_transfers = sum(
        1
        for s in hit_events
        if s["color"] != sprites[s["target_sprite"]]["color"]
    )
    collision_events = []
    lone_events = []
    mixed_colors = set()
    lone_colors = set()
    for dest, ids in bg_groups.items():
        if len(ids) >= 2:
            collision_events.append(dest)
            mixed_colors.add(int(out[dest]))
        else:
            lone_events.append(dest)
            lone_colors.add(int(out[dest]))
    d9_out = _render_d9_only_output(inp, sprites)
    original_out = _render_original_output(inp, sprites)
    return {
        "hit_events": len(hit_events),
        "visible_hit_transfers": visible_hit_transfers,
        "collision_events": len(collision_events),
        "lone_bg_events": len(lone_events),
        "distinct_source_colors": len({s["color"] for s in sprites}),
        "distinct_mixed_colors": len(mixed_colors),
        "distinct_lone_colors": len(lone_colors),
        "diff_vs_d9": int(np.count_nonzero(out != d9_out)),
        "diff_vs_original": int(np.count_nonzero(out != original_out)),
        "mixed_collision_cells": sum(1 for dest in collision_events if int(out[dest]) in MIX_OUTPUT_COLORS),
    }


def _shape_branch_metrics(sprites: list[dict]) -> dict[str, int]:
    diag = sum(1 for sprite in sprites if abs(sprite["placement"]["dir"][0]) + abs(sprite["placement"]["dir"][1]) == 2)
    axis = len(sprites) - diag
    return {"diag_sprites": diag, "axis_sprites": axis}


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
    metrics = _variant_metrics(inp, out, sprites)
    shape_metrics = _shape_branch_metrics(sprites)

    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if metrics["hit_events"] < 1 or metrics["visible_hit_transfers"] < 1:
        return None
    if metrics["collision_events"] < 1 or metrics["mixed_collision_cells"] < 1:
        return None
    if metrics["lone_bg_events"] < 1:
        return None
    if metrics["distinct_source_colors"] != len(SOURCE_COLORS):
        return None
    if metrics["diff_vs_d9"] == 0 or metrics["diff_vs_original"] == 0:
        return None
    if shape_metrics["diag_sprites"] < 1 or shape_metrics["axis_sprites"] < 1:
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
