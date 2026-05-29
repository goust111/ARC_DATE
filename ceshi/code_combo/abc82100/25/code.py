# Combo variant of abc82100 (index 25)
# combo_spec: D4/2+D16/2
# Name: 颜色对方向编码旋转+模板旋转/翻转

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused variant
# ------------------------------------
# a) concretization and behavior trace:
# - The legend keeps the original three-step cue: source -> target -> anchor,
#   with a nearby teal stencil. The teal stencil is shown in one fixed grid
#   orientation, so the pair direction can independently encode an extra quarter
#   turn:
#     east=0, south=1, west=2, north=3.
# - Each scattered occurrence is no longer a lone source pixel. It is a 3-cell
#   seed:
#     * center = true source pixel,
#     * one same-color arm = base forward direction,
#     * one target-color arm = left/right mirror bit.
# - Solver flow on a seed:
#     1. read the canonical stencil from the legend,
#     2. rotate that stencil by the legend-coded extra turn,
#     3. orient it so the seed's same-color arm becomes "forward",
#     4. optionally mirror across that forward axis depending on which side the
#        target-colored arm occupies,
#     5. recolor to the legend target color and stamp at the seed center.
# - Compared with the original task, geometry is no longer fixed per color:
#   one rule color pair can produce multiple output orientations via the seed,
#   while two rules with similar seeds can still differ because their legends
#   encode different extra rotations.
#
# b) complexity calibration:
# - To keep both dimensions visible, the generator uses only teal stencils whose
#   8 dihedral transforms are all distinct. A symmetric stencil would hide the
#   D4 flip/rotation logic and could also make some D16 turns look identical.
# - Each rule gets repeated seed structures with both mirror states and multiple
#   forward directions. This prevents the task from collapsing to a per-color
#   fixed transform.
# - Multiple rules prefer distinct extra rotation codes so a single grid usually
#   exhibits more than one legend-direction case.
#
# c) implicit-assumption audit for main():
# - Original `_discover_rules` could store raw absolute stencil cells because the
#   output copied the stencil unchanged. That no longer works: the solver now
#   converts the teal component into anchor-relative local coordinates.
# - Original stamping treated every non-legend source-colored pixel as a source
#   center. That is invalid after introducing 3-cell seeds, so the solver now
#   accepts only source pixels with exactly one same-color arm and one
#   perpendicular target-color arm.
# - Original bystander preservation only removed legend source pixels and teal.
#   The fused task must also erase the seed arms, including the target-colored
#   side arm, otherwise garbage markers would leak into the output.
# - The new solver assumes the legend stencil is the only teal object type and
#   that valid seeds are 4-isolated from other colored neighbors. The generator
#   enforces both constraints.
#
# d) placement legality:
# - Legends, seeds, bystanders, and future output stamps are all placed under
#   occupancy/reservation checks, so they do not overlap or create accidental
#   extra legends/seeds.
# - Every seed is accepted only when its fully transformed stamp fits in-bounds
#   and avoids prior output cells. This keeps outputs readable and prevents
#   hidden clipping from masking the fused behavior.
#
# e) visual observability:
# - The legend still teaches the color mapping locally and visibly. The same
#   three-cell seed repeats around scattered sources, so a human can compare
#   seeds with the resulting copied teal shapes and infer the D4 arm semantics.
# - The legend pair direction varies independently of the seed shapes, making the
#   extra quarter-turn observable without relying on color ids or coordinate
#   conventions invisible to humans.
#
# f) cleanliness / inferability:
# - Only three visible object families remain: legends with teal stencils, seed
#   triplets, and unrelated bystanders. No teal appears outside legends.
# - Bystanders never reuse source colors, so any source-colored object belongs to
#   a rule. This keeps the color change inferable from a small number of
#   examples instead of mixing "consumed" and "preserved" uses of the same color.


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
_STEP_BY_ROT = {
    0: (0, 1),
    1: (1, 0),
    2: (0, -1),
    3: (-1, 0),
}
_ROT_BY_STEP = {step: rot for rot, step in _STEP_BY_ROT.items()}


def _left(vec: tuple[int, int]) -> tuple[int, int]:
    return (-int(vec[1]), int(vec[0]))


def _right(vec: tuple[int, int]) -> tuple[int, int]:
    lv = _left(vec)
    return (-lv[0], -lv[1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _rotate_local_coord(coord: tuple[int, int], turns: int) -> tuple[int, int]:
    a, b = int(coord[0]), int(coord[1])
    for _ in range(int(turns) % 4):
        a, b = b, -a
    return int(a), int(b)


def _rotate_local_pattern(local_pattern: np.ndarray, turns: int) -> np.ndarray:
    return np.asarray([_rotate_local_coord((int(a), int(b)), turns) for a, b in local_pattern], dtype=int)


def _mirror_local(coord: tuple[int, int]) -> tuple[int, int]:
    return int(coord[0]), -int(coord[1])


def _variant_signatures(local_pattern: list[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    base = tuple(sorted((int(a), int(b)) for a, b in local_pattern))
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = base
    for _ in range(4):
        seen.add(cur)
        seen.add(tuple(sorted(_mirror_local(coord) for coord in cur)))
        cur = tuple(sorted(_rotate_local_coord(coord, 1) for coord in cur))
    return seen


def _apply_local_pattern(
    local_pattern: np.ndarray,
    anchor: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    anchor_r, anchor_c = int(anchor[0]), int(anchor[1])
    forward_r, forward_c = int(forward[0]), int(forward[1])
    left_r, left_c = _left(forward)
    side_sign = -1 if flip else 1

    rows = anchor_r + local_pattern[:, 0] * forward_r + side_sign * local_pattern[:, 1] * left_r
    cols = anchor_c + local_pattern[:, 0] * forward_c + side_sign * local_pattern[:, 1] * left_c
    return np.stack([rows, cols], axis=1).astype(int)


def _stamp_pattern(
    local_pattern: np.ndarray,
    anchor: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
    extra_rot: int,
) -> np.ndarray:
    rotated = _rotate_local_pattern(local_pattern, extra_rot)
    return _apply_local_pattern(rotated, anchor=anchor, forward=forward, flip=flip)


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                anchor = (tr + dr, tc + dc)
                candidates: list[tuple[int, int, np.ndarray]] = []
                for i, pattern in enumerate(patterns):
                    dist = _anchor_distance(pattern, anchor)
                    if dist <= 1:
                        candidates.append((int(dist), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], len(item[2])))
                _dist, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                local_pattern: list[tuple[int, int]] = []
                for pr, pc in map(tuple, pattern):
                    delta_r = int(pr) - int(anchor[0])
                    delta_c = int(pc) - int(anchor[1])
                    local_pattern.append((int(delta_c), int(-delta_r)))

                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "extra_rot": int(_ROT_BY_STEP[(int(dr), int(dc))]),
                        "local_pattern": np.asarray(local_pattern, dtype=int),
                    }
                )

    return rules


def _find_rule_seeds(
    grid: np.ndarray,
    rule: dict,
    legend_pixels: set[tuple[int, int]],
    bg: int,
) -> list[dict]:
    h, w = grid.shape
    src = int(rule["source_color"])
    tgt = int(rule["target_color"])
    seeds: list[dict] = []
    consumed: set[tuple[int, int]] = set()

    for sr, sc in map(tuple, np.argwhere(grid == src)):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels:
            continue

        same_dirs: list[tuple[int, int]] = []
        tgt_dirs: list[tuple[int, int]] = []
        extra_neighbor = False
        for dr, dc in _DIRS4:
            rr = sr + dr
            cc = sc + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            color = int(grid[rr, cc])
            if color == src:
                same_dirs.append((int(dr), int(dc)))
            elif color == tgt:
                tgt_dirs.append((int(dr), int(dc)))
            elif color != int(bg):
                extra_neighbor = True

        if extra_neighbor or len(same_dirs) != 1 or len(tgt_dirs) != 1:
            continue

        forward = same_dirs[0]
        side = tgt_dirs[0]
        left_dir = _left(forward)
        if side == left_dir:
            flip = False
        elif side == _right(forward):
            flip = True
        else:
            continue

        cells = {
            (sr, sc),
            (sr + forward[0], sc + forward[1]),
            (sr + side[0], sc + side[1]),
        }
        if cells & legend_pixels or cells & consumed:
            continue

        seeds.append(
            {
                "center": (sr, sc),
                "forward": forward,
                "flip": bool(flip),
                "cells": cells,
            }
        )
        consumed.update(cells)

    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use local legends plus 3-cell seeds to stamp compounded stencil transforms.

    Input structure:
    - Black background with teal (8) legend stencils, source->target color pairs,
      and scattered 3-cell source seeds.
    - A legend is a 4-neighbor source/target pair followed by an anchor in the
      same direction, with a teal stencil touching that anchor.
    - Each seed has a source-colored center, a same-color forward arm, and a
      target-colored side arm.

    Rule:
    - Learn the source->target recoloring from the legend pair and read an extra
      quarter-turn from that pair's direction.
    - For every matching seed, first rotate the canonical teal stencil by that
      legend-coded amount, then orient it by the seed's same-color arm and mirror
      it when the target-colored side arm is on the opposite side.
    - Stamp the transformed stencil at the seed center in the target color, and
      erase all legend pixels, teal pixels, and seed-marker pixels.

    Key insight: Two independent visible cues must be fused: the legend direction
    adds a global quarter-turn, while the little L-shaped seed around each source
    pixel supplies the local D4 orientation and flip before the stencil is copied.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(rule["source_color"])

    seeds_by_rule: list[list[dict]] = []
    seed_pixels: set[tuple[int, int]] = set()
    for rule in rules:
        seeds = _find_rule_seeds(g, rule=rule, legend_pixels=legend_pixels, bg=bg)
        seeds_by_rule.append(seeds)
        for seed in seeds:
            seed_pixels.update(seed["cells"])

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in seed_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule, seeds in zip(rules, seeds_by_rule):
        for seed in seeds:
            placed = _stamp_pattern(
                rule["local_pattern"],
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
                extra_rot=int(rule["extra_rot"]),
            )
            rows = placed[:, 0]
            cols = placed[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 19}

RULE_COUNT_WEIGHTS = [1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 7
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 45

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_SEED = 160
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500
STENCIL_RETRIES = 400


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _make_stencil(min_pixels: int, max_pixels: int) -> tuple[np.ndarray, np.ndarray]:
    lo = max(4, int(min_pixels))
    hi = max(lo, int(max_pixels))
    for _ in range(STENCIL_RETRIES):
        n = random.randint(lo, hi)
        raw_coords = {(0, 0)}
        while len(raw_coords) < n:
            frontier = set()
            for r, c in raw_coords:
                for dr, dc in _DIRS8:
                    cand = (int(r) + dr, int(c) + dc)
                    if cand not in raw_coords:
                        frontier.add(cand)
            raw_coords.add(random.choice(list(frontier)))

        raw_pattern = np.asarray(sorted(raw_coords), dtype=int)
        local_pattern = np.asarray([(int(c), int(-r)) for r, c in raw_pattern], dtype=int)
        if len(_variant_signatures(local_pattern.tolist())) != 8:
            continue
        if len({int(r) for r, _c in raw_pattern}) < 2:
            continue
        if len({int(c) for _r, c in raw_pattern}) < 2:
            continue
        return raw_pattern, local_pattern

    raw_pattern = np.asarray([(0, 0), (0, 1), (1, 1), (2, 1)], dtype=int)
    local_pattern = np.asarray([(int(c), int(-r)) for r, c in raw_pattern], dtype=int)
    return raw_pattern, local_pattern


def _choose_extra_rots(num_rules: int) -> list[int]:
    if num_rules == 1:
        return [random.choice([1, 2, 3])]
    rots = [0, 1, 2, 3]
    random.shuffle(rots)
    return rots[:num_rules]


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(4, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(4, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    extra_rots = _choose_extra_rots(num_rules)

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        raw_pattern, local_pattern = _make_stencil(min_s, max_s)
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "raw_pattern": raw_pattern,
                "local_pattern": local_pattern,
                "extra_rot": int(extra_rots[i]),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _seed_variants(n_sc: int) -> list[tuple[tuple[int, int], bool]]:
    primary = random.choice(_DIRS4)
    secondary_choices = [d for d in _DIRS4 if d != primary and d != (-primary[0], -primary[1])]
    secondary = random.choice(secondary_choices)
    variants = [(primary, False), (primary, True), (secondary, False)]

    pool = [(d, f) for d in _DIRS4 for f in (False, True)]
    random.shuffle(pool)
    for item in pool:
        if item not in variants:
            variants.append(item)
        if len(variants) >= n_sc:
            break
    return variants[:n_sc]


def _place_legends(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        dr, dc = _STEP_BY_ROT[int(rule["extra_rot"])]
        placed = False
        for _ in range(LEGEND_RETRIES):
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            teal_cells = {(int(ar) + int(rr), int(ac) + int(cc)) for rr, cc in rule["raw_pattern"]}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue
            reverse_anchor = (int(sr - dr), int(sc - dc))
            if min(_chebyshev(cell, reverse_anchor) for cell in teal_cells) <= 1:
                continue

            all_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_scattered(
    rules: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    future_output: set[tuple[int, int]],
    legend_teal: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        seeds: list[dict] = []
        for forward, flip in _seed_variants(int(rule["n_sc"])):
            side = _right(forward) if flip else _left(forward)
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_SEED):
                sr = random.randint(0, h - 1)
                sc = random.randint(0, w - 1)
                center = (int(sr), int(sc))
                src_arm = (int(sr + forward[0]), int(sc + forward[1]))
                tgt_arm = (int(sr + side[0]), int(sc + side[1]))
                cells = {center, src_arm, tgt_arm}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if cells & (occupied | reserved):
                    continue
                false_anchor = (int(sr + 2 * side[0]), int(sc + 2 * side[1]))
                if min(_chebyshev(cell, false_anchor) for cell in legend_teal) <= 1:
                    continue

                stamp = {
                    tuple(cell)
                    for cell in _stamp_pattern(
                        rule["local_pattern"],
                        anchor=center,
                        forward=forward,
                        flip=flip,
                        extra_rot=int(rule["extra_rot"]),
                    )
                }
                if any(not (0 <= r < h and 0 <= c < w) for r, c in stamp):
                    continue
                if stamp & future_output:
                    continue
                if stamp & occupied:
                    continue

                inp[center] = int(rule["src"])
                inp[src_arm] = int(rule["src"])
                inp[tgt_arm] = int(rule["tgt"])
                occupied.update(cells)
                _reserve_4neighbors(reserved, cells)
                future_output.update(stamp)
                seeds.append(
                    {
                        "center": center,
                        "forward": forward,
                        "flip": bool(flip),
                    }
                )
                placed = True
                break
            if not placed:
                return False
        rule["seeds"] = seeds
    return True


def _place_bystanders(
    used_colors: set[int],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
    future_output: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    placed: list[tuple[int, int, int]] = []
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in future_output:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            placed.append((int(r), int(c), int(color)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, {(int(r), int(c))})
            break
    return placed


def _build_output(
    rules: list[dict],
    bystanders: list[tuple[int, int, int]],
    h: int,
    w: int,
) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)

    for rule in rules:
        for seed in rule["seeds"]:
            stamp = _stamp_pattern(
                rule["local_pattern"],
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
                extra_rot=int(rule["extra_rot"]),
            )
            out[stamp[:, 0], stamp[:, 1]] = int(rule["tgt"])
    return out


def _validate_instance(inp: np.ndarray, rules: list[dict]) -> bool:
    bg = get_bg(inp)
    discovered = _discover_rules(inp, bg=bg, pattern_color=8)
    expected_by_src = {int(rule["src"]): rule for rule in rules}
    if len(discovered) != len(rules):
        return False
    if {int(rule["source_color"]) for rule in discovered} != set(expected_by_src):
        return False

    legend_pixels = {rule["source_pos"] for rule in discovered} | {rule["target_pos"] for rule in discovered}
    for disc in discovered:
        exp = expected_by_src[int(disc["source_color"])]
        if int(disc["target_color"]) != int(exp["tgt"]):
            return False
        if int(disc["extra_rot"]) != int(exp["extra_rot"]):
            return False
        disc_pat = tuple(sorted(map(tuple, np.asarray(disc["local_pattern"], dtype=int).tolist())))
        exp_pat = tuple(sorted(map(tuple, np.asarray(exp["local_pattern"], dtype=int).tolist())))
        if disc_pat != exp_pat:
            return False

        seeds = _find_rule_seeds(inp, rule=disc, legend_pixels=legend_pixels, bg=bg)
        disc_seeds = {
            (tuple(seed["center"]), tuple(seed["forward"]), bool(seed["flip"]))
            for seed in seeds
        }
        exp_seeds = {
            (tuple(seed["center"]), tuple(seed["forward"]), bool(seed["flip"]))
            for seed in exp["seeds"]
        }
        if disc_seeds != exp_seeds:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    future_output: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    legend_teal = {tuple(cell) for cell in np.argwhere(inp == 8)}
    if not _place_scattered(rules, inp, occupied, reserved, future_output, legend_teal):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, future_output)
    if not _validate_instance(inp, rules):
        return None

    output = _build_output(rules, bystanders, h, w)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
