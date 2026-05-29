# Combo variant of abc82100 (index 22)
# combo_spec: D4/2+D11/1
# Name: 分散teal片段模板+模板旋转/翻转

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) concretization and behavior trace
# - The legend keeps the original `source -> target -> anchor` line, but the teal
#   stencil is no longer a single component. Instead, each legend owns a compact
#   5x5 anchor window containing 2-4 disconnected teal islands. The solver must
#   group those islands into one template before doing anything else.
# - Scattered occurrences are no longer lone source pixels. Each one is a 3-cell
#   seed: source-colored center, one same-color arm giving a forward direction, and
#   one target-colored side arm giving left-vs-right mirror state.
# - Example trace:
#   * Original task: translate one connected teal component from the legend anchor
#     onto every scattered source pixel with no orientation change.
#   * Fused task: first merge several nearby teal islands into one fragmented local
#     stencil; then, for a seed whose same-color arm points east and whose side arm
#     sits south, rotate that whole fragmented stencil to point east and mirror it
#     across the forward axis before stamping it at the seed center.
#   * Difference: both "what counts as the template" and "which orientation to use"
#     changed, and the second step is defined on the merged union, not on each
#     fragment independently.
#
# b) complexity calibration
# - A fragmented but symmetric stencil would visually collapse the rotation signal,
#   so the generator only emits unions whose 8 dihedral transforms are all distinct.
# - Every rule has at least 2 teal fragments and at least 3 scattered seeds; the
#   seed variants deliberately include both mirror states plus multiple directions.
# - The fragment count stays small and local (radius-2 window) so the grouping task
#   is visible but not noisy.
#
# c) implicit-assumption audit of the original main()
# - Original `_discover_rules` assumed "one nearby teal component = one rule
#   template". That breaks here; discovery must gather all teal components inside
#   the anchor's local window and reject legends with fewer than 2 fragments.
# - Original rules stored absolute template coordinates. That is insufficient once
#   per-seed rotation/flip is needed; the template must be re-expressed in a local
#   frame aligned to the legend direction.
# - Original stamping treated every non-legend source-colored pixel as a true source.
#   That fails because seeds now contain an extra same-color arm. We therefore parse
#   only source centers with exactly one same-color cardinal neighbor and one
#   perpendicular target-color neighbor, while rejecting extra colored neighbors.
# - Original bystander preservation removed legend cells and teal, but not the new
#   target-colored seed arms. We therefore collect all seed cells and erase them.
#
# d) placement legality
# - Generator reserves each legend's whole radius-2 window so teal islands from
#   different legends cannot mix.
# - Fragments within one legend are pairwise 8-disconnected. Scattered seeds are
#   separated by 4-neighbor reservation so they do not merge into accidental new
#   seeds or false legends.
# - Output stamps are required to fit fully in-bounds and not overlap each other.
#   This is stricter than the original task but keeps the fused behavior legible.
#
# e) visual observability
# - A human sees repeated local patterns of three colored cells for scattered
#   occurrences, and repeated legends where one color pair points at several teal
#   islands. Across 2-4 examples, that supports the inference "group the islands,
#   then orient the grouped shape according to the seed arms".
# - The transform cue depends only on relative position of the arms, never on color
#   id ordering or any non-visual metadata.
#
# f) visual cleanliness
# - Legends stay compact in a 5x5 zone. Seeds are only 3 cells each. Bystanders are
#   kept out of future output cells, so the grid does not become visually tangled.
#
# g) inference and new guarantees
# - Source/target recoloring still comes from the legend pair, not from arbitrary
#   color arithmetic.
# - New solver assumptions: each valid legend has anchor cell occupied by teal,
#   at least two teal components inside the radius-2 window, and each scattered seed
#   follows the exact 3-cell cue format.
# - Generator guarantees those assumptions directly.


LOCAL_RADIUS = 2

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 12, 3: 16, 4: 20}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_TOTAL_MIN = 5
STENCIL_TOTAL_MAX_CAP = 8
STENCIL_MIN_AREA_DIV = 90
STENCIL_MAX_AREA_DIV = 45
FRAGMENT_COUNT_MIN = 2
FRAGMENT_COUNT_MAX = 4
FRAGMENT_SIZE_CAP = 3
FRAGMENT_RETRIES = 400

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 6
SCATTERED_AREA_DIV = 45

BYSTANDER_MAX = 4

LEGEND_RETRIES = 600
SCATTERED_RETRIES_PER_SEED = 160
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 500

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
_LOCAL_CELLS = [
    (r, c)
    for r in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
    for c in range(-LOCAL_RADIUS, LOCAL_RADIUS + 1)
]


def _left(vec: tuple[int, int]) -> tuple[int, int]:
    return (-int(vec[1]), int(vec[0]))


def _right(vec: tuple[int, int]) -> tuple[int, int]:
    lv = _left(vec)
    return (-lv[0], -lv[1])


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _max_anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return max(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells: set[tuple[int, int]]) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _reserve_window(reserved: set[tuple[int, int]], anchor: tuple[int, int], radius: int) -> None:
    ar, ac = int(anchor[0]), int(anchor[1])
    for rr in range(ar - radius, ar + radius + 1):
        for cc in range(ac - radius, ac + radius + 1):
            reserved.add((int(rr), int(cc)))


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


def _apply_local_pattern(
    local_pattern: np.ndarray,
    anchor: tuple[int, int],
    forward: tuple[int, int],
    flip: bool,
) -> np.ndarray:
    ar, ac = int(anchor[0]), int(anchor[1])
    fr, fc = int(forward[0]), int(forward[1])
    lr, lc = _left(forward)
    side_sign = -1 if flip else 1
    rows = ar + local_pattern[:, 0] * fr + side_sign * local_pattern[:, 1] * lr
    cols = ac + local_pattern[:, 0] * fc + side_sign * local_pattern[:, 1] * lc
    return np.stack([rows, cols], axis=1).astype(int)


def _collect_local_pattern(
    patterns: list[np.ndarray],
    anchor: tuple[int, int],
    radius: int,
    forward: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, ...]] | None:
    component_ids: list[int] = []
    coords: set[tuple[int, int]] = set()
    anchor_is_teal = False

    for i, pattern in enumerate(patterns):
        if _max_anchor_distance(pattern, anchor) > radius:
            continue
        component_ids.append(int(i))
        for r, c in pattern:
            cell = (int(r), int(c))
            coords.add(cell)
            if cell == anchor:
                anchor_is_teal = True

    if not anchor_is_teal or len(component_ids) < 2:
        return None

    lateral = _left(forward)
    local_coords: list[tuple[int, int]] = []
    for pr, pc in sorted(coords):
        delta_r = int(pr) - int(anchor[0])
        delta_c = int(pc) - int(anchor[1])
        a = delta_r * forward[0] + delta_c * forward[1]
        b = delta_r * lateral[0] + delta_c * lateral[1]
        local_coords.append((int(a), int(b)))
    return np.asarray(local_coords, dtype=int), tuple(component_ids)


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = 8,
    local_radius: int = LOCAL_RADIUS,
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], tuple[int, ...]]] = set()

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
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                    continue

                legend_forward = (-int(dr), -int(dc))
                pattern_info = _collect_local_pattern(
                    patterns,
                    anchor=anchor,
                    radius=local_radius,
                    forward=legend_forward,
                )
                if pattern_info is None:
                    continue
                local_pattern, component_ids = pattern_info

                key = ((int(r), int(c)), (int(tr), int(tc)), component_ids)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "legend_forward": legend_forward,
                        "local_pattern": local_pattern,
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
    Reconstruct fragmented teal legends and stamp their rotated or mirrored unions.

    Input structure:
    - The background is black, with local rule legends made from a source->target
      color pair and a nearby teal(8) anchor window.
    - Inside each legend window, the teal template is split into several detached
      fragments rather than one connected component.
    - Each scattered occurrence is a 3-cell seed: a source-colored center, a
      same-color forward arm, and a target-colored side arm.

    Rule:
    - For every legend, merge all teal fragments in the anchor's compact local
      window into one template and express that template in the legend's local frame.
    - For every matching scattered seed, rotate the merged template so its forward
      axis matches the seed's same-color arm; if the target-colored arm is on the
      opposite side, mirror the rotated template before stamping.
    - Recolor every stamped pixel to the legend target color, and erase legends,
      teal fragments, and seed markers from the output.

    Key insight:
    - The solver must perform two linked spatial steps in order: first group several
      nearby teal islands into one stencil, then apply a dihedral transform chosen
      by the seed's two short arms before copying that full fragmented stencil.
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


def _far_from_cells(cell: tuple[int, int], cells: set[tuple[int, int]]) -> bool:
    return all(_chebyshev(cell, other) >= 2 for other in cells)


def _random_partition(total: int, parts: int, piece_cap: int) -> list[int]:
    pieces = [1] * parts
    remaining = total - parts
    while remaining > 0:
        candidates = [i for i, size in enumerate(pieces) if size < piece_cap]
        if not candidates:
            return []
        idx = random.choice(candidates)
        pieces[idx] += 1
        remaining -= 1
    random.shuffle(pieces)
    return pieces


def _grow_fragment(
    start: tuple[int, int],
    size: int,
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    fragment = {start}
    while len(fragment) < size:
        frontier: set[tuple[int, int]] = set()
        for r, c in fragment:
            for dr, dc in _DIRS8:
                nr, nc = int(r) + dr, int(c) + dc
                cell = (int(nr), int(nc))
                if cell in fragment:
                    continue
                if max(abs(cell[0]), abs(cell[1])) > LOCAL_RADIUS:
                    continue
                if not _far_from_cells(cell, blocked):
                    continue
                frontier.add(cell)
        if not frontier:
            return None
        fragment.add(random.choice(sorted(frontier)))
    return fragment


def _make_fragmented_stencil(min_pixels: int, max_pixels: int) -> np.ndarray:
    for _ in range(FRAGMENT_RETRIES):
        total = random.randint(max(STENCIL_TOTAL_MIN, min_pixels), max_pixels)
        n_fragments = random.randint(FRAGMENT_COUNT_MIN, min(FRAGMENT_COUNT_MAX, total))
        sizes = _random_partition(total, n_fragments, FRAGMENT_SIZE_CAP)
        if not sizes:
            continue

        fragments: list[set[tuple[int, int]]] = []
        first = _grow_fragment((0, 0), sizes[0], blocked=set())
        if first is None:
            continue
        fragments.append(first)
        used = set(first)

        ok = True
        for size in sizes[1:]:
            starts = [
                cell
                for cell in _LOCAL_CELLS
                if cell not in used and _far_from_cells(cell, used)
            ]
            if not starts:
                ok = False
                break
            start = random.choice(starts)
            fragment = _grow_fragment(start, size, blocked=used)
            if fragment is None:
                ok = False
                break
            fragments.append(fragment)
            used.update(fragment)
        if not ok or len(fragments) < FRAGMENT_COUNT_MIN:
            continue

        local_coords = sorted(used)
        if len(_variant_signatures(local_coords)) != 8:
            continue
        if not any(a != 0 for a, _b in local_coords):
            continue
        if not any(b != 0 for _a, b in local_coords):
            continue
        return np.asarray(local_coords, dtype=int)

    raise RuntimeError("failed to create asymmetric fragmented stencil")


def _build_rules(num_rules: int, area: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_TOTAL_MAX_CAP, max(STENCIL_TOTAL_MIN, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_TOTAL_MIN, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules: list[dict] = []
    used: set[int] = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((int(src_c), int(tgt_c)))
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "local_pattern": _make_fragmented_stencil(min_s, max_s),
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _seed_variants(n_sc: int) -> list[tuple[tuple[int, int], bool]]:
    dirs = _DIRS4[:]
    random.shuffle(dirs)
    variants: list[tuple[tuple[int, int], bool]] = []
    if n_sc >= 1:
        variants.append((dirs[0], False))
    if n_sc >= 2:
        variants.append((dirs[0], True))
    if n_sc >= 3:
        variants.append((dirs[1], False))
    if n_sc >= 4:
        variants.append((dirs[1], True))
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
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr = random.randint(0, h - 1)
            sc = random.randint(0, w - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
                continue

            legend_forward = (-int(dr), -int(dc))
            teal_cells = {
                tuple(cell)
                for cell in _apply_local_pattern(
                    rule["local_pattern"],
                    anchor=(ar, ac),
                    forward=legend_forward,
                    flip=False,
                )
            }
            own_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
            if any(not (0 <= r < h and 0 <= c < w) for r, c in own_cells):
                continue

            window_cells = {
                (rr, cc)
                for rr in range(ar - LOCAL_RADIUS, ar + LOCAL_RADIUS + 1)
                for cc in range(ac - LOCAL_RADIUS, ac + LOCAL_RADIUS + 1)
            }
            if own_cells & (occupied | reserved):
                continue
            if (window_cells - own_cells) & occupied:
                continue

            inp[sr, sc] = int(rule["src"])
            inp[tr, tc] = int(rule["tgt"])
            for r, c in teal_cells:
                inp[r, c] = 8
            occupied.update(own_cells)
            _reserve_4neighbors(reserved, own_cells)
            _reserve_window(reserved, (ar, ac), LOCAL_RADIUS)
            rule["legend_forward"] = legend_forward
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
                center = (int(sr), int(sc))
                src_arm = (int(sr + forward[0]), int(sc + forward[1]))
                tgt_arm = (int(sr + side[0]), int(sc + side[1]))
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
