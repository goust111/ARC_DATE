# Variant of abc82100
# Dimension: D4 — teal模板的形状类型 (dst)
# Change: teal模板是由teal像素构成的任意连通图案（菱形、线段等），定义了要复制的形状... → 模板需旋转/翻转后使用

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the D4 variant
# ----------------------------------
# concretization:
# - The legend still uses the original "src -> tgt -> teal stencil" triple, so each
#   rule continues to teach a color substitution plus one canonical teal shape.
# - The changed part is each scattered source occurrence. Instead of a lone source
#   pixel, the input now contains a 3-cell "seed":
#   * the center is the real source pixel,
#   * one adjacent same-color arm marks the forward direction,
#   * one adjacent target-color arm sits on the left or right side of the center.
# - The output erases the whole seed and stamps the legend stencil at the center,
#   recolored to the rule target color. The same-color arm determines the 90-degree
#   rotation; the side on which the target-color arm sits determines whether the
#   stencil is mirrored across that forward axis.
#
# behavior trace:
# - Original task: a scattered source at (r,c) always receives the teal component in
#   the legend orientation, translated so the legend anchor lands on (r,c).
# - Variant task: if the seed around (r,c) points east and its side arm is on the
#   north side, the legend stencil is first rotated so its "source side" points east;
#   if the side arm is on the south side, that rotated stencil is mirrored before
#   placement. The same legend can therefore produce multiple output orientations.
#
# complexity calibration:
# - A purely symmetric stencil would hide the D4 change, so the generator only emits
#   8-connected stencil shapes whose 8 dihedral transforms are all distinct.
# - Each rule gets multiple scattered seeds, and every rule uses both mirror states
#   plus multiple forward directions. This keeps the transform rule visible instead
#   of collapsing to a per-color fixed orientation.
#
# implicit-assumption audit for main():
# - `_discover_rules` still assumes teal components belong only to legends. The
#   variant therefore does not introduce any extra teal markers outside legends.
# - Rule discovery used to store absolute stencil coordinates; now it must convert
#   them into a local coordinate frame derived from the legend line direction,
#   otherwise later rotations/flips would have no reference frame.
# - The old stamping loop treated every non-legend source-colored pixel as a true
#   source center. That is invalid now because each seed contains an extra same-color
#   arm. We therefore parse only source pixels with exactly one same-color cardinal
#   neighbor and one perpendicular target-color neighbor.
# - The old bystander-preservation logic erased only legend pixels and source-colored
#   scatter points. That would incorrectly preserve the new target-color side arms, so
#   all seed cells are collected and removed before copying ordinary bystanders.
#
# placement legality:
# - The generator reserves 4-neighbor space around legends and around each seed so no
#   accidental extra adjacencies create false seeds or false legends.
# - Every seed is placed only when the transformed stencil fits fully in-bounds and
#   does not overlap previously reserved output cells, which keeps the output clean.
#
# observability / cleanliness:
# - The input now contains repeated L-shaped local cues around the source centers, so
#   a human can compare several examples and see that those cues control orientation.
# - Only two nearby colors are involved in the cue, and their meaning comes from
#   position, not from numeric color id ordering. The output remains uncluttered
#   because seed arms disappear and transformed stencils do not overlap.


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


def _left(vec: tuple[int, int]) -> tuple[int, int]:
    return (-int(vec[1]), int(vec[0]))


def _right(vec: tuple[int, int]) -> tuple[int, int]:
    lv = _left(vec)
    return (-lv[0], -lv[1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


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
                forward = (-int(dr), -int(dc))
                lateral = _left(forward)
                local_pattern: list[tuple[int, int]] = []
                for pr, pc in map(tuple, pattern):
                    delta_r = int(pr) - int(anchor[0])
                    delta_c = int(pc) - int(anchor[1])
                    a = delta_r * forward[0] + delta_c * forward[1]
                    b = delta_r * lateral[0] + delta_c * lateral[1]
                    local_pattern.append((int(a), int(b)))

                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
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
    Use local source-target-pattern examples to stamp rotated or mirrored copies.

    Input structure:
    - Black background with teal (8) legend stencils, color-substitution examples,
      and scattered 3-cell source seeds.
    - A rule example is still a 4-neighbor source->target pair whose continued
      direction reaches an anchor touching a teal stencil component.
    - Each scattered seed has a source-colored center, a same-color forward arm,
      and a target-colored side arm.

    Rule:
    - Learn the stencil shape from the nearby teal component and the source/target
      color mapping from the legend pair.
    - For every scattered seed of that source color, rotate the stencil so its
      legend-facing side points along the seed's same-color arm; if the target-colored
      arm is on the opposite side, mirror the rotated stencil before stamping.
    - Recolor the stamped stencil to the rule target color, and remove all legend,
      teal, and seed-marker pixels.

    Key insight: The legend still teaches the canonical stencil, but the two short
    arms around each scattered source define a local frame that selects one dihedral
    transform before the stencil is copied.
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

    # Preserve only unrelated colored pixels; legends, seed markers, and teal disappear.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in seed_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule, seeds in zip(rules, seeds_by_rule):
        local_pattern = rule["local_pattern"]
        for seed in seeds:
            placed = _apply_local_pattern(
                local_pattern,
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
            )
            rows = placed[:, 0]
            cols = placed[:, 1]
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (c) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 19}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 7
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 45

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_SEED = 120
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400
STENCIL_RETRIES = 300


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _rot90_local(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((-int(b), int(a)) for a, b in coords))


def _flip_local(coords: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(a), -int(b)) for a, b in coords))


def _variant_signatures(local_coords: list[tuple[int, int]]) -> set[tuple[tuple[int, int], ...]]:
    base = tuple(sorted((int(a), int(b)) for a, b in local_coords))
    seen: set[tuple[tuple[int, int], ...]] = set()
    cur = base
    for _ in range(4):
        seen.add(cur)
        seen.add(_flip_local(cur))
        cur = _rot90_local(cur)
    return seen


def _make_stencil(min_pixels: int, max_pixels: int) -> np.ndarray:
    for _ in range(STENCIL_RETRIES):
        n = random.randint(max(4, min_pixels), max(max(4, min_pixels), max_pixels))
        coords = {(0, 0)}
        while len(coords) < n:
            frontier = set()
            for a, b in coords:
                for da, db in _DIRS8:
                    cand = (int(a) + da, int(b) + db)
                    if cand not in coords:
                        frontier.add(cand)
            coords.add(random.choice(list(frontier)))

        local_coords = sorted(coords)
        if len(_variant_signatures(local_coords)) != 8:
            continue
        if not any(a != 0 for a, _b in local_coords):
            continue
        if not any(b != 0 for _a, b in local_coords):
            continue
        return np.asarray(local_coords, dtype=int)
    raise RuntimeError("failed to create asymmetric stencil")


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(4, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(4, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "local_pattern": _make_stencil(min_s, max_s),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _seed_variants(n_sc: int) -> list[tuple[tuple[int, int], bool]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    variants = [(dirs[0], False), (dirs[1], True)]
    pool = [(d, f) for d in _DIRS4 for f in (False, True)]
    random.shuffle(pool)
    for item in pool:
        if item not in variants:
            variants.append(item)
        if len(variants) >= n_sc:
            break
    return variants[:n_sc]


def _place_legends(rules: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    for rule in rules:
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            forward = (-dr, -dc)
            teal_cells = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    rule["local_pattern"],
                    anchor=(ar, ac),
                    forward=forward,
                    flip=False,
                )
            }
            all_cells = {(sr, sc), (tr, tc)} | teal_cells
            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
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
) -> bool:
    h, w = inp.shape
    for rule in rules:
        seeds: list[dict] = []
        for forward, flip in _seed_variants(rule["n_sc"]):
            side = _right(forward) if flip else _left(forward)
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_SEED):
                sr = random.randint(0, h - 1)
                sc = random.randint(0, w - 1)
                center = (sr, sc)
                src_arm = (sr + forward[0], sc + forward[1])
                tgt_arm = (sr + side[0], sc + side[1])
                cells = {center, src_arm, tgt_arm}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
                    continue
                if cells & (occupied | reserved):
                    continue

                stamp = {
                    tuple(cell)
                    for cell in _apply_local_pattern(
                        rule["local_pattern"],
                        anchor=center,
                        forward=forward,
                        flip=flip,
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
            placed.append((r, c, int(color)))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, {(r, c)})
            break
    return placed


def _build_output(rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = int(color)

    for rule in rules:
        for seed in rule["seeds"]:
            stamp = _apply_local_pattern(
                rule["local_pattern"],
                anchor=seed["center"],
                forward=seed["forward"],
                flip=seed["flip"],
            )
            out[stamp[:, 0], stamp[:, 1]] = int(rule["tgt"])
    return out


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
    if not _place_scattered(rules, inp, occupied, reserved, future_output):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved, future_output)

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
