# Variant of 409aa875
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 单轮同时处理：所有精灵同时发射，落点同时计算碰撞结果... → 两轮迭代：第一轮发射后被命中精灵在第二轮反向发射

import random
from collections import Counter, defaultdict

import numpy as np

from arc_utils import components_mask, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis
# ---------------
# spec:
#   variant_goal:
#     Keep the original 3-pixel triangle vocabulary, 16x16 board, and 5-cell
#     projectile distance, but replace the single simultaneous firing pass with
#     exactly two passes. Round 1 is identical to the source task. Round 2 is
#     new: every sprite that was hit in round 1 fires once more in the opposite
#     direction (180-degree flip of its intrinsic direction). Round-2 hits do
#     not trigger any further rounds.
#   concretization_and_trace:
#     Input still contains only monochrome 3-cell L/V triangles on orange bg.
#     Example trace:
#       - Sprite A is an L pointing up-left, so round-1 dest is apex+5*(-1,-1).
#       - Sprite B's round-1 projectile lands on a cell of A, so A is marked as
#         "hit in round 1".
#       - In the original task, processing would stop here apart from bg marks.
#       - In this variant, A also shoots in round 2 toward apex+5*(+1,+1).
#       - If that reverse dest is an empty bg cell, that cell gains a new maroon
#         marker that never appears in the original task; if multiple retaliators
#         land there, it turns blue by the same collision rule.
#   complexity_calibration:
#     The variant stays within one extra binary stage rather than introducing an
#     unbounded cascade. Generator samples 1-3 round-1 hit targets and 0-1 bg
#     collision points so outputs are richer than a trivial one-shot puzzle but
#     still visually readable in 2-4 examples.
#   implicit_assumption_audit:
#     Original main() assumed a single dest_to_shooters map was sufficient and
#     that recoloring targets could be decided from one pass only. That no
#     longer holds because round-1 hit status now controls which shooters exist
#     in round 2. The updated solver therefore separates:
#       1. all round-1 destinations
#       2. the subset of sprites hit in round 1
#       3. round-2 reversed destinations for that subset
#       4. final recolor/bg counting after aggregating both rounds
#     Another source-task assumption was that sprite color could equal marker
#     colors 1 or 9 without harming clarity. For this variant that weakens the
#     observability of second-round effects, so the generator excludes sprite
#     base colors 1 and 9 while leaving solver logic color-agnostic.
#   placement_legality:
#     Sprites remain 8-connected 3-cell components separated by Chebyshev >= 2.
#     Every placed sprite must have its forward landing cell in bounds, and the
#     generator additionally guarantees at least one round-1 hit target whose
#     reverse landing cell is also in bounds and kept empty in the input so the
#     second round becomes visible.
#   visual_observability:
#     Human-solvable clue: some output marks cannot be explained by a single
#     forward shot because they appear five cells behind a sprite that itself
#     turns maroon. Across samples, the same "hit then shoot backwards once"
#     pattern repeats on both L and V orientations.
#   visual_cleanliness:
#     Event counts are capped and bg destinations are reserved so the board does
#     not become saturated with overlapping markers. Two-round interactions are
#     present but typically confined to a few isolated cells.
#   color_change_inference:
#     Colors keep the original meaning only: maroon=single landed shot or hit
#     sprite, blue=2+ shots onto the same originally empty cell. No hidden
#     ordering or color-id-based behavior is introduced.
#   post_change_main_assumptions:
#     - Only original sprites (input components) can be shooters/targets.
#     - Round-2 bg marks are judged against the original sprite occupancy, not
#       against first-round painted markers.
#     - Exactly two rounds are simulated; round-2 hits never recurse.
#     Generator construction enforces these assumptions.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _triangle_apex_and_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(comp) != 3:
        return None
    cells = set(comp)

    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in cells)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in cells)

    corners = [p for p, d in deg4.items() if d == 2]
    if corners:
        apex = corners[0]
        ar, ac = apex
        arms = []
        for r, c in comp:
            if (r, c) == apex:
                continue
            dr, dc = r - ar, c - ac
            if abs(dr) + abs(dc) == 1:
                arms.append((dr, dc))
        if len(arms) != 2:
            return None
        dr = -(arms[0][0] + arms[1][0])
        dc = -(arms[0][1] + arms[1][1])
        dr = 0 if dr == 0 else (1 if dr > 0 else -1)
        dc = 0 if dc == 0 else (1 if dc > 0 else -1)
        return apex, (dr, dc)

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


def _landing(apex: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int] | None:
    h, w = shape
    rr = apex[0] + 5 * direction[0]
    cc = apex[1] + 5 * direction[1]
    if 0 <= rr < h and 0 <= cc < w:
        return rr, cc
    return None


def _simulate_entities(
    entities: list[dict],
    cell_to_entity: dict[tuple[int, int], int],
    shape: tuple[int, int],
) -> dict:
    round1: dict[tuple[int, int], list[int]] = defaultdict(list)
    for eid, ent in enumerate(entities):
        dest = _landing(ent["apex"], ent["dir"], shape)
        if dest is not None:
            round1[dest].append(eid)

    round1_hit_ids = {cell_to_entity[dest] for dest in round1 if dest in cell_to_entity}

    round2: dict[tuple[int, int], list[int]] = defaultdict(list)
    for eid in sorted(round1_hit_ids):
        dr, dc = entities[eid]["dir"]
        dest = _landing(entities[eid]["apex"], (-dr, -dc), shape)
        if dest is not None:
            round2[dest].append(eid)

    hit_ids = set(round1_hit_ids)
    for dest in round2:
        if dest in cell_to_entity:
            hit_ids.add(cell_to_entity[dest])

    bg_counts: Counter[tuple[int, int]] = Counter()
    for dest_map in (round1, round2):
        for dest, shooters in dest_map.items():
            if dest in cell_to_entity:
                continue
            bg_counts[dest] += len(shooters)

    return {
        "round1": round1,
        "round2": round2,
        "round1_hit_ids": round1_hit_ids,
        "hit_ids": hit_ids,
        "bg_counts": bg_counts,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire triangle projectiles in two passes; hit sprites counterfire backward once.

    Input structure:
    - bg = most common color. Non-bg objects are 3-pixel monochrome triangle sprites:
      axis-aligned L-shapes or diagonal V-shapes.

    Rule:
    - Each sprite has an apex and a built-in direction from its geometry. Round 1 uses
      the same forward direction as the source task, landing exactly 5 cells from the apex.
    - Any sprite struck in round 1 is recolored maroon (9) and also shoots once in round 2
      from the same apex but with its direction flipped 180 degrees.
    - Across both rounds, any projectile landing on a sprite recolors that entire sprite to
      maroon (9). Projectiles landing on originally empty bg cells are counted together:
      2+ shots make the cell blue (1), while a single shot makes it maroon (9).

    Key insight: First identify which sprites get hit by the forward volley; only that subset
    becomes the second-wave shooters, and their reverse shots contribute to the same final
    maroon/blue collision bookkeeping.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)

    entities: list[dict] = []
    cell_to_entity: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            ad = _triangle_apex_and_dir(comp)
            if ad is None:
                continue
            apex, direction = ad
            eid = len(entities)
            entities.append({"cells": comp, "apex": apex, "dir": direction})
            for rc in comp:
                cell_to_entity[rc] = eid

    sim = _simulate_entities(entities, cell_to_entity, g.shape)
    out = g.copy()

    for eid in sim["hit_ids"]:
        for r, c in entities[eid]["cells"]:
            out[r, c] = 9

    for (r, c), count in sim["bg_counts"].items():
        out[r, c] = 1 if count >= 2 else 9

    return out


H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS

SPRITE_COUNT_RANGE = (4, 8)
MAX_EXTRA_HIT_EVENTS = 2
MAX_COLL_EVENTS = 1
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


def _build_placements() -> tuple[list[dict], dict[tuple[int, int], list[dict]], list[dict]]:
    placements: list[dict] = []
    by_dest: dict[tuple[int, int], list[dict]] = defaultdict(list)
    placements_with_rev: list[dict] = []
    for ar in range(H):
        for ac in range(W):
            for dr, dc in ALL_DIRS:
                dest = (ar + PROJECTILE_DIST * dr, ac + PROJECTILE_DIST * dc)
                if not (0 <= dest[0] < H and 0 <= dest[1] < W):
                    continue
                cells = _sprite_cells((ar, ac), (dr, dc))
                if not _in_bounds_cells(cells):
                    continue
                rev = (ar - PROJECTILE_DIST * dr, ac - PROJECTILE_DIST * dc)
                rev_dest = rev if 0 <= rev[0] < H and 0 <= rev[1] < W else None
                placement = {
                    "apex": (ar, ac),
                    "dir": (dr, dc),
                    "dest": dest,
                    "rev_dest": rev_dest,
                    "cells": cells,
                    "cell_set": frozenset(cells),
                    "halo": _halo(cells),
                }
                placements.append(placement)
                by_dest[dest].append(placement)
                if rev_dest is not None:
                    placements_with_rev.append(placement)
    return placements, by_dest, placements_with_rev


PLACEMENTS, PLACEMENTS_BY_DEST, PLACEMENTS_WITH_REV = _build_placements()


def _fits(placement: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (placement["halo"] & occupied) and not (placement["cell_set"] & reserved_bg)


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _placement_to_entity(placement: dict) -> dict:
    return {"cells": list(placement["cells"]), "apex": placement["apex"], "dir": placement["dir"]}


def _choose_from_pool(
    rng: random.Random,
    pool: list[dict],
    occupied: set[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
    *,
    require_fresh_dest: bool = True,
) -> dict | None:
    candidates = pool[:]
    rng.shuffle(candidates)
    for placement in candidates:
        if require_fresh_dest and (placement["dest"] in reserved_bg or placement["dest"] in occupied):
            continue
        if _fits(placement, occupied, reserved_bg):
            return placement
    return None


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
    return all(comp.shape == (3, 2) for comp in comps)


def _counts_for_n(rng: random.Random, n_sprites: int) -> tuple[int, int, int] | None:
    options = []
    for extra_hits in range(0, min(MAX_EXTRA_HIT_EVENTS, n_sprites - 2) + 1):
        for e_coll in range(0, min(MAX_COLL_EVENTS, (n_sprites - 2 - extra_hits) // 2) + 1):
            unique = n_sprites - 2 - extra_hits - 2 * e_coll
            if unique < 0:
                continue
            options.append((extra_hits, e_coll, unique))
    return rng.choice(options) if options else None


def _render_output(inp: np.ndarray, sprites: list[dict]) -> tuple[np.ndarray, dict]:
    entities = [_placement_to_entity(sprite["placement"]) for sprite in sprites]
    cell_to_entity = {}
    for idx, ent in enumerate(entities):
        for cell in ent["cells"]:
            cell_to_entity[cell] = idx
    sim = _simulate_entities(entities, cell_to_entity, inp.shape)

    out = inp.copy()
    for idx in sim["hit_ids"]:
        _paint_sprite(out, sprites[idx]["placement"], HIT_COLOR)
    for dest, count in sim["bg_counts"].items():
        out[dest] = COLLISION_COLOR if count >= 2 else HIT_COLOR
    return out, sim


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)
    counts = _counts_for_n(rng, n_sprites)
    if counts is None:
        return None
    extra_hits, e_coll, _unique_total = counts

    sprites: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    reserved_bg: set[tuple[int, int]] = set()

    target = _choose_from_pool(rng, PLACEMENTS_WITH_REV, occupied, reserved_bg)
    if target is None or target["rev_dest"] is None:
        return None
    sprites.append({"placement": target, "target_kind": "base"})
    occupied.update(target["cells"])
    reserved_bg.add(target["rev_dest"])

    first_hit = _choose_hit(rng, sprites[0], occupied, reserved_bg, set())
    if first_hit is None:
        return None
    shooter, hit_cell = first_hit
    sprites.append({"placement": shooter, "target_kind": "hit", "target_cell": hit_cell, "target_sprite": 0})
    occupied.update(shooter["cells"])

    for _ in range(e_coll):
        choice = _choose_collision_pair(rng, occupied, reserved_bg)
        if choice is None:
            return None
        dest, pair = choice
        reserved_bg.add(dest)
        for placement in pair:
            sprites.append({"placement": placement, "target_kind": "bg", "target_cell": dest})
            occupied.update(placement["cells"])

    eligible_targets = [idx for idx, sprite in enumerate(sprites) if sprite["placement"]["rev_dest"] is not None]
    extra_bases_needed = max(0, extra_hits - len(eligible_targets))
    for _ in range(extra_bases_needed):
        placement = _choose_from_pool(rng, PLACEMENTS_WITH_REV, occupied, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "target_kind": "bg", "target_cell": placement["dest"]})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])

    eligible_targets = [idx for idx, sprite in enumerate(sprites) if sprite["placement"]["rev_dest"] is not None]
    if len(eligible_targets) < extra_hits:
        return None

    used_target_cells = {hit_cell}
    for idx in rng.sample(eligible_targets, extra_hits):
        choice = _choose_hit(rng, sprites[idx], occupied, reserved_bg, used_target_cells)
        if choice is None:
            return None
        placement, cell = choice
        sprites.append({"placement": placement, "target_kind": "hit", "target_cell": cell, "target_sprite": idx})
        occupied.update(placement["cells"])
        used_target_cells.add(cell)

    while len(sprites) < n_sprites:
        placement = _choose_from_pool(rng, PLACEMENTS, occupied, reserved_bg)
        if placement is None:
            return None
        sprites.append({"placement": placement, "target_kind": "bg", "target_cell": placement["dest"]})
        occupied.update(placement["cells"])
        reserved_bg.add(placement["dest"])

    inp = np.full((H, W), BG, dtype=int)
    for sprite in sprites:
        _paint_sprite(inp, sprite["placement"], sprite_color)

    if not _validate_input(inp, n_sprites):
        return None

    out, sim = _render_output(inp, sprites)

    if np.array_equal(inp, out):
        return None
    if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
        return None
    if not sim["round1_hit_ids"]:
        return None

    round1_bg_cells = {dest for dest in sim["round1"] if dest not in {(r, c) for sprite in sprites for (r, c) in sprite["placement"]["cells"]}}
    if not any(dest not in round1_bg_cells for dest in sim["round2"]):
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
