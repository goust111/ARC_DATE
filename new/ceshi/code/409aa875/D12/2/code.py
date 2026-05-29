# Variant of 409aa875
# Dimension: D12 — 方向确定规则（形状语义） (path)
# Change: L形：对角线方向远离缺失的2x2角；V形：轴方向远离开口侧。方向完全由像素排列的几何结构决定... → 方向由邻近精灵决定（朝向最近精灵射击）

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, get_bg as _bg_color, neighbors4, neighbors8


# Design analysis for the D12 variant
# a) Concretization / behavior trace:
#    - The input still contains 4-9 monochrome 3-pixel triangle sprites on a 16x16 orange background.
#      The two visible sprite shapes stay the same: axis-aligned L and diagonal V.
#    - The changed behavior is only how the firing direction is inferred. The apex is still the structural
#      corner / tip of the 3-pixel sprite, but the projectile direction now comes from global layout:
#      find the nearest other sprite by apex-to-apex squared Euclidean distance, then move one step toward
#      that apex using the sign of row/col deltas. The landing cell is still apex + 5 * dir.
#    - Small trace: if two apexes sit at (4,3) and (4,13), the left sprite's nearest apex is to the right,
#      so it fires east and lands at (4,8). In the original task an L at (4,3) might have fired diagonally
#      up-left or down-right depending on its missing corner; here its local shape is irrelevant.
# b) Complexity calibration:
#    - The rule uses only one global relation: nearest neighbor. That is richer than a local shape lookup,
#      but still human-trackable because "closest sprite" is visually salient.
#    - Generator layouts use pair-distance motifs (5 for apex-hit, 10 for midpoint collision, 7/8/9/11 for
#      unique background landings) so the same nearest-neighbor logic repeats within each sample.
# c) Implicit assumptions audit against the old main():
#    - Old `_triangle_apex_and_dir` bundled apex extraction with direction extraction. That is invalid now:
#      apex still comes from shape geometry, but direction must ignore shape orientation completely.
#    - Old generator stored `placement["dir"]` and trusted it as the emitted direction. That assumption also
#      breaks; generator must instead validate directions from the final multi-sprite apex layout.
#    - Old hit detection by target cell membership remains valid, because only the direction source changes.
# d) Placement legality:
#    - Apexes are placed first, shapes second. Apex layouts are accepted only when every sprite has a unique
#      nearest neighbor, every resulting landing cell is in bounds, and there is at least one nontrivial
#      event (sprite hit or multi-projectile background collision).
#    - Shape assignment uses backtracking with the original Chebyshev>=2 separation rule, and reserves every
#      intended background landing cell so no sprite accidentally occupies it.
# e) Visual observability:
#    - The generator rejects grids whose outputs match the original shape-based solver, and also rejects
#      shape assignments where fewer than two sprites change firing direction relative to the old rule.
#    - This makes the D12 change visible across samples: identical shapes can fire differently, and different
#      shapes can share the same firing direction because proximity, not silhouette, controls shooting.
# f) Visual cleanliness:
#    - Motifs are built from isolated apex pairs / tails, which keeps nearest-neighbor relations local and
#      prevents cluttered cross-interference. Background impact cells are reserved so markers do not overlap
#      sprite bodies except for deliberate apex hits.
# g) Color inference:
#    - Sprite source colors exclude 1 and 9 so both result colors remain visually diagnostic:
#      maroon(9) means a hit recolor or lone landing, blue(1) means a multi-projectile background collision.


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def _dir_toward(src: tuple[int, int], dst: tuple[int, int]) -> tuple[int, int]:
    return _sign(dst[0] - src[0]), _sign(dst[1] - src[1])


def _triangle_apex(comp: list[tuple[int, int]]) -> tuple[int, int] | None:
    if len(comp) != 3:
        return None
    s = set(comp)
    deg4 = {}
    deg8 = {}
    for r, c in comp:
        deg4[(r, c)] = sum(1 for n in neighbors4(r, c) if n in s)
        deg8[(r, c)] = sum(1 for n in neighbors8(r, c) if n in s)

    corners = [p for p, d in deg4.items() if d == 2]
    if len(corners) == 1:
        return corners[0]

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) == 1:
        return apexes[0]
    return None


def _triangle_apex_and_shape_dir(comp: list[tuple[int, int]]) -> tuple[tuple[int, int], tuple[int, int]] | None:
    # Legacy shape-based direction, kept only for generator anti-degeneracy checks.
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
        dr = -arms[0][0] - arms[1][0]
        dc = -arms[0][1] - arms[1][1]
        return corner, (_sign(dr), _sign(dc))

    apexes = [p for p, d in deg8.items() if d == 2]
    if len(apexes) != 1:
        return None
    apex = apexes[0]
    ar, ac = apex
    others = [p for p in comp if p != apex]
    dr = -((others[0][0] - ar) + (others[1][0] - ar)) // 2
    dc = -((others[0][1] - ac) + (others[1][1] - ac)) // 2
    if abs(dr) + abs(dc) != 1:
        return None
    return apex, (dr, dc)


def _nearest_indices(apexes: list[tuple[int, int]]) -> list[int] | None:
    nearest: list[int] = []
    for i, (ar, ac) in enumerate(apexes):
        best_idx = -1
        best_dist = None
        tied = False
        for j, (br, bc) in enumerate(apexes):
            if i == j:
                continue
            dist = (ar - br) ** 2 + (ac - bc) ** 2
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_idx = j
                tied = False
            elif dist == best_dist:
                tied = True
        if best_idx < 0 or tied:
            return None
        nearest.append(best_idx)
    return nearest


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire projectiles from 3-pixel triangle sprites; mark collisions with maroon or blue.

    Input structure:
    - bg = most common color. Non-bg objects are still monochrome 3-pixel L or V triangle sprites, each with a unique geometric apex.

    Rule:
    - Each sprite first finds the nearest other sprite using apex-to-apex distance. Its projectile direction is the 8-way sign direction from its own apex toward that nearest apex.
    - A projectile lands exactly 5 cells from the apex in that nearest-neighbor direction.
    - If the landing cell lies on a sprite, the entire hit sprite is recolored to maroon (9).
    - If two or more projectiles land on the same empty background cell, that cell becomes blue (1); a lone empty-cell landing becomes maroon (9).

    Key insight: The local triangle shape only identifies the apex; the actual firing direction is a global nearest-neighbor relation, so two identical sprites can shoot in different directions.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _bg_color(g)

    comps: list[list[tuple[int, int]]] = []
    apexes: list[tuple[int, int]] = []
    cell_to_comp: dict[tuple[int, int], int] = {}
    for col in sorted(set(int(v) for v in np.unique(g)) - {bg}):
        for comp in _cc8(g == col):
            apex = _triangle_apex(comp)
            if apex is None:
                continue
            cid = len(comps)
            comps.append(comp)
            apexes.append(apex)
            for rc in comp:
                cell_to_comp[rc] = cid

    nearest = _nearest_indices(apexes)
    out = g.copy()
    if nearest is None:
        return out

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for cid, other_idx in enumerate(nearest):
        ar, ac = apexes[cid]
        dr, dc = _dir_toward((ar, ac), apexes[other_idx])
        rr = ar + 5 * dr
        cc = ac + 5 * dc
        if 0 <= rr < h and 0 <= cc < w:
            dest_to_shooters.setdefault((rr, cc), []).append(cid)

    hit_components: set[int] = set()
    for dest in dest_to_shooters:
        if dest in cell_to_comp:
            hit_components.add(cell_to_comp[dest])

    for cid in hit_components:
        for r, c in comps[cid]:
            out[r, c] = 9

    for (r, c), shooters in dest_to_shooters.items():
        if (r, c) in cell_to_comp:
            continue
        out[r, c] = 1 if len(shooters) >= 2 else 9

    return out


# ---- canvas / palette ----
H = W = 16
BG = 7
HIT_COLOR = 9
COLLISION_COLOR = 1
SPRITE_COLORS = [0, 2, 3, 4, 5, 6, 8]

# ---- geometry ----
PROJECTILE_DIST = 5
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
AXIS_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_DIRS = DIAG_DIRS + AXIS_DIRS
L_ARM_PAIRS = [
    ((-1, 0), (0, -1)),
    ((-1, 0), (0, 1)),
    ((1, 0), (0, -1)),
    ((1, 0), (0, 1)),
]
UNIQUE_PAIR_DISTANCES = [7, 8, 9, 11]
APEX_MIN = 1
APEX_MAX = 14

# ---- structural counts per grid ----
SPRITE_COUNT_RANGE = (4, 9)

# ---- generation retries ----
MAX_ATTEMPTS = 1200
SHAPE_ASSIGN_ATTEMPTS = 50


def _v_cells(apex: tuple[int, int], direction: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = apex
    dr, dc = direction
    if dr != 0:
        return ((ar, ac), (ar - dr, ac - 1), (ar - dr, ac + 1))
    return ((ar, ac), (ar - 1, ac - dc), (ar + 1, ac - dc))


def _in_bounds_cell(cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < H and 0 <= c < W


def _in_bounds_cells(cells: tuple[tuple[int, int], ...]) -> bool:
    return all(_in_bounds_cell(cell) for cell in cells)


def _halo(cells: tuple[tuple[int, int], ...]) -> frozenset[tuple[int, int]]:
    return frozenset(
        (rr, cc)
        for r, c in cells
        for rr in range(max(0, r - 1), min(H, r + 2))
        for cc in range(max(0, c - 1), min(W, c + 2))
    )


def _variants_for_apex(apex: tuple[int, int]) -> list[dict]:
    ar, ac = apex
    variants: list[dict] = []
    for arm_a, arm_b in L_ARM_PAIRS:
        cells = ((ar, ac), (ar + arm_a[0], ac + arm_a[1]), (ar + arm_b[0], ac + arm_b[1]))
        if not _in_bounds_cells(cells):
            continue
        old_dir = _dir_toward((0, 0), (-arm_a[0] - arm_b[0], -arm_a[1] - arm_b[1]))
        variants.append(
            {
                "apex": apex,
                "kind": "L",
                "old_dir": old_dir,
                "cells": cells,
                "cell_set": frozenset(cells),
                "halo": _halo(cells),
            }
        )
    for direction in AXIS_DIRS:
        cells = _v_cells(apex, direction)
        if not _in_bounds_cells(cells):
            continue
        variants.append(
            {
                "apex": apex,
                "kind": "V",
                "old_dir": direction,
                "cells": cells,
                "cell_set": frozenset(cells),
                "halo": _halo(cells),
            }
        )
    return variants


VARIANTS_BY_APEX = {
    (r, c): _variants_for_apex((r, c)) for r in range(APEX_MIN, APEX_MAX + 1) for c in range(APEX_MIN, APEX_MAX + 1)
}


def _build_pair_candidates(distances: list[int]) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for dist in distances:
        for dr, dc in ALL_DIRS:
            for ar in range(APEX_MIN, APEX_MAX + 1):
                for ac in range(APEX_MIN, APEX_MAX + 1):
                    br = ar + dist * dr
                    bc = ac + dist * dc
                    if APEX_MIN <= br <= APEX_MAX and APEX_MIN <= bc <= APEX_MAX:
                        candidates.append(((ar, ac), (br, bc)))
    return candidates


PAIR_CANDIDATES = {
    "hit": _build_pair_candidates([5]),
    "collision": _build_pair_candidates([10]),
    "unique": _build_pair_candidates(UNIQUE_PAIR_DISTANCES),
}


def _apex_chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _apex_fits(apex: tuple[int, int], existing: list[tuple[int, int]]) -> bool:
    return all(_apex_chebyshev(apex, other) >= 3 for other in existing)


def _choose_pair(
    rng: random.Random,
    pair_type: str,
    existing: list[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    candidates = PAIR_CANDIDATES[pair_type][:]
    rng.shuffle(candidates)
    for a, b in candidates:
        if a in existing or b in existing:
            continue
        if not _apex_fits(a, existing) or not _apex_fits(b, existing):
            continue
        return a, b
    return None


def _choose_single_apex(rng: random.Random, existing: list[tuple[int, int]]) -> tuple[int, int] | None:
    candidates = [(r, c) for r in range(APEX_MIN, APEX_MAX + 1) for c in range(APEX_MIN, APEX_MAX + 1)]
    rng.shuffle(candidates)
    for apex in candidates:
        if apex in existing:
            continue
        if _apex_fits(apex, existing):
            return apex
    return None


def _analyze_apex_layout(apexes: list[tuple[int, int]]) -> dict | None:
    nearest = _nearest_indices(apexes)
    if nearest is None:
        return None

    dirs: list[tuple[int, int]] = []
    dests: list[tuple[int, int]] = []
    apex_set = set(apexes)
    for i, j in enumerate(nearest):
        direction = _dir_toward(apexes[i], apexes[j])
        dest = (
            apexes[i][0] + PROJECTILE_DIST * direction[0],
            apexes[i][1] + PROJECTILE_DIST * direction[1],
        )
        if not _in_bounds_cell(dest):
            return None
        dirs.append(direction)
        dests.append(dest)

    bg_dests = [dest for dest in dests if dest not in apex_set]
    bg_counter = Counter(bg_dests)
    hit_count = sum(dest in apex_set for dest in dests)
    collision_count = sum(1 for count in bg_counter.values() if count >= 2)
    if not bg_dests:
        return None
    if hit_count + collision_count < 1:
        return None
    if len(set(dirs)) < 2:
        return None

    return {
        "nearest": nearest,
        "dirs": dirs,
        "dests": dests,
        "bg_reserved": set(bg_dests),
        "hit_count": hit_count,
        "collision_count": collision_count,
    }


def _sample_apex_layout(rng: random.Random, n_sprites: int) -> tuple[list[tuple[int, int]], dict] | None:
    pair_count = n_sprites // 2
    pair_types = [rng.choice(["hit", "collision"])]
    while len(pair_types) < pair_count:
        pair_types.append(rng.choice(["unique", "unique", "unique", "collision", "hit"]))
    if all(pair_type == "hit" for pair_type in pair_types):
        pair_types[-1] = "unique"
    rng.shuffle(pair_types)

    apexes: list[tuple[int, int]] = []
    for pair_type in pair_types:
        pair = _choose_pair(rng, pair_type, apexes)
        if pair is None:
            return None
        apexes.extend(pair)

    if n_sprites % 2 == 1:
        extra = _choose_single_apex(rng, apexes)
        if extra is None:
            return None
        apexes.append(extra)

    analysis = _analyze_apex_layout(apexes)
    if analysis is None:
        return None
    return apexes, analysis


def _fits_variant(variant: dict, occupied: set[tuple[int, int]], reserved_bg: set[tuple[int, int]]) -> bool:
    return not (variant["halo"] & occupied) and not (variant["cell_set"] & reserved_bg)


def _assign_shapes(
    rng: random.Random,
    apexes: list[tuple[int, int]],
    reserved_bg: set[tuple[int, int]],
) -> list[dict] | None:
    order = sorted(range(len(apexes)), key=lambda idx: apexes[idx])
    placed: dict[int, dict] = {}
    occupied: set[tuple[int, int]] = set()

    def dfs(pos: int) -> bool:
        if pos == len(order):
            kinds = {placed[idx]["kind"] for idx in placed}
            return len(kinds) >= 2

        idx = order[pos]
        candidates = VARIANTS_BY_APEX[apexes[idx]][:]
        rng.shuffle(candidates)
        for variant in candidates:
            if not _fits_variant(variant, occupied, reserved_bg):
                continue
            placed[idx] = variant
            occupied.update(variant["cells"])
            if dfs(pos + 1):
                return True
            for cell in variant["cells"]:
                occupied.remove(cell)
            del placed[idx]
        return False

    if not dfs(0):
        return None
    return [placed[idx] for idx in range(len(apexes))]


def _paint_sprite(grid: np.ndarray, placement: dict, color: int) -> None:
    for r, c in placement["cells"]:
        grid[r, c] = color


def _render_output(inp: np.ndarray, placements: list[dict], directions: list[tuple[int, int]]) -> np.ndarray:
    out = inp.copy()
    cell_to_sprite: dict[tuple[int, int], int] = {}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_sprite[cell] = idx

    dest_to_shooters: dict[tuple[int, int], list[int]] = {}
    for idx, (placement, direction) in enumerate(zip(placements, directions)):
        ar, ac = placement["apex"]
        rr = ar + PROJECTILE_DIST * direction[0]
        cc = ac + PROJECTILE_DIST * direction[1]
        if 0 <= rr < H and 0 <= cc < W:
            dest_to_shooters.setdefault((rr, cc), []).append(idx)

    hit_ids = {cell_to_sprite[dest] for dest in dest_to_shooters if dest in cell_to_sprite}
    for idx in hit_ids:
        _paint_sprite(out, placements[idx], HIT_COLOR)

    for dest, shooters in dest_to_shooters.items():
        if dest in cell_to_sprite:
            continue
        out[dest] = COLLISION_COLOR if len(shooters) >= 2 else HIT_COLOR
    return out


def _validate_input(grid: np.ndarray, n_sprites: int) -> bool:
    if grid.shape != (H, W):
        return False
    if get_bg(grid) != BG:
        return False
    comps = components_mask(grid != BG, connectivity=8)
    if len(comps) != n_sprites:
        return False
    return all(comp.shape == (3, 2) and _triangle_apex([(int(r), int(c)) for r, c in comp]) is not None for comp in comps)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    sprite_color = rng.choice(SPRITE_COLORS)
    n_sprites = rng.randint(*SPRITE_COUNT_RANGE)

    layout = _sample_apex_layout(rng, n_sprites)
    if layout is None:
        return None
    apexes, layout_info = layout

    for _ in range(SHAPE_ASSIGN_ATTEMPTS):
        placements = _assign_shapes(rng, apexes, layout_info["bg_reserved"])
        if placements is None:
            return None

        inp = np.full((H, W), BG, dtype=int)
        for placement in placements:
            _paint_sprite(inp, placement, sprite_color)
        if not _validate_input(inp, n_sprites):
            continue

        new_dirs = layout_info["dirs"]
        old_dirs = [placement["old_dir"] for placement in placements]
        mismatch_count = sum(old_dir != new_dir for old_dir, new_dir in zip(old_dirs, new_dirs))
        if mismatch_count < 2:
            continue

        out = _render_output(inp, placements, new_dirs)
        old_out = _render_output(inp, placements, old_dirs)
        if np.array_equal(out, old_out):
            continue
        if np.array_equal(inp, out):
            continue
        if np.count_nonzero(out != BG) <= np.count_nonzero(inp != BG):
            continue
        if not np.any(out == HIT_COLOR):
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if np.array_equal(solved, out):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {MAX_ATTEMPTS} attempts")
