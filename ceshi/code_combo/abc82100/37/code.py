# Combo variant of abc82100 (index 37)
# combo_spec: D2/3+D9/1
# Name: 堆叠优先级编码+同色条件分支

from __future__ import annotations

from collections import defaultdict
import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for this combo variant
# --------------------------------------
# fused_behavior:
#   - Each same-source family is a nested legend pair: a large outer teal ring and a
#     smaller inner teal ring inside it. The two legends reuse the same source color
#     but have different target colors and different stencils.
#   - A scattered source pixel does not choose by nearest legend and does not always
#     choose the deepest legend. Instead it chooses the smallest same-source legend
#     whose stencil bbox contains the pixel. If the pixel lies in the inner bbox, the
#     inner rule wins; if it lies in the outer-only annulus, the outer rule wins.
# concrete_trace:
#   - Example family: blue->red on an outer 9x9 ring, blue->yellow on an inner 5x5
#     ring nested inside it.
#   - Original task: every blue scatter would stamp exactly one fixed blue rule.
#   - D2-only behavior: every blue scatter would always use the inner 5x5 rule.
#   - D9-only nearest behavior: a blue scatter just below the inner box would often
#     choose the inner legend because that legend source cell is nearer.
#   - Combo behavior here: that same blue scatter is outside the inner bbox but still
#     inside the outer bbox, so it must use the outer rule. A blue scatter inside the
#     inner bbox uses the inner rule. The nesting is the branch condition itself.
# complexity_calibration:
#   - Only two layers per family are used. That is enough to expose priority and
#     conditional branching without making the legend logic unreadable.
#   - Each generated grid contains 1-2 nested families, all with unique source colors.
#     This keeps the hypothesis space small while repeating the combo behavior across
#     samples.
# implicit_assumptions_audit:
#   - The base solver assumed "one source color -> one rule". This is false now, so
#     rule application must happen per scattered source pixel, not per color.
#   - The D2 single variant assumed "same color always chooses deepest". This is also
#     false now; depth matters only after filtering to legends whose bboxes contain the
#     source pixel.
#   - The D9 single variant assumed geometric distance alone selects the branch. This
#     generator intentionally places outer-branch scatters that are nearer to the inner
#     legend, so nearest-legend decoding is provably wrong on valid samples.
#   - The preserve pass still assumes bystander colors never reuse a source color; the
#     generator keeps that guarantee.
# placement_legality:
#   - Nested rings use sizes 9 and 5 with the inner ring offset by (2,2), leaving a
#     one-cell moat so teal components stay disconnected and anchor matching is stable.
#   - Groups reserve a one-cell halo around the outer bbox, preventing cross-group
#     collisions and accidental anchor ambiguity.
#   - Scattered pixels are placed only on empty cells inside their intended zones, are
#     kept away from colored legend cells, and their output footprints are required to
#     stay in bounds. Inner/outer footprints within one family may overlap slightly,
#     but only under a tight cap so both branch effects remain visible.
# visual_observability_and_inferability:
#   - The same source color appears in two nested legends with two target colors; that
#     already tells the solver that color alone is insufficient.
#   - The input source pixels visibly split into "inside the small box" and "inside the
#     large-only shell" cases. The output confirms that the small case gets the small
#     stencil/target while the shell case gets the large stencil/target.
#   - No hidden ordering by numeric color id is used. The only control signal is a
#     human-visible spatial relation: membership in nested teal boxes.


_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ["top", "bottom", "left", "right"]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

# Grid / legend layout
GRID_DIM_MAX = 20
MIN_DIM_BY_GROUPS = {1: 14, 2: 18}
GROUP_COUNT_WEIGHTS = [1, 1, 2]
OUTER_SIZE = 9
INNER_SIZE = 5
INNER_OFFSET = 2
GROUP_MARGIN = 1

# Placement budgets
GROUP_RETRIES = 500
SCATTER_PAIR_RETRIES = 120
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 400

# Output cleanliness
MAX_LOCAL_FOOTPRINT_OVERLAP = 2
BYSTANDER_MAX = 2


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _pattern_bbox(pattern: np.ndarray) -> tuple[int, int, int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _strictly_contains(
    outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]
) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _point_in_bbox(point: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = int(point[0]), int(point[1])
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (int(bbox[2]) - int(bbox[0]) + 1) * (int(bbox[3]) - int(bbox[1]) + 1)


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
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                    continue

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
                pattern_array = np.asarray(pattern, dtype=int)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": pattern_array,
                        "bbox": _pattern_bbox(pattern_array),
                    }
                )

    return rules


def _annotate_depths(rules_by_source: dict[int, list[dict]]) -> None:
    for same_source in rules_by_source.values():
        for rule in same_source:
            rule["depth"] = sum(
                1 for other in same_source if other is not rule and _strictly_contains(other["bbox"], rule["bbox"])
            )


def _collect_rule_state(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    rules_by_source: dict[int, list[dict]] = defaultdict(list)
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        rules_by_source[int(rule["source_color"])].append(rule)
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(int(rule["source_color"]))
    _annotate_depths(rules_by_source)
    return g, int(bg), int(pattern_color), rules_by_source, legend_pixels, source_colors


def _preserve_bystanders(
    g: np.ndarray,
    bg: int,
    pattern_color: int,
    legend_pixels: set[tuple[int, int]],
    source_colors: set[int],
) -> np.ndarray:
    out = np.full_like(g, int(bg))
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(pattern_color)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color
    return out


def _stamp_rule(out: np.ndarray, rule: dict, source_pos: tuple[int, int]) -> None:
    h, w = out.shape
    sr, sc = int(source_pos[0]), int(source_pos[1])
    ar, ac = rule["anchor"]
    pattern = rule["pattern"]
    dr = sr - int(ar)
    dc = sc - int(ac)
    rows = pattern[:, 0] + dr
    cols = pattern[:, 1] + dc
    ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
    out[rows[ok], cols[ok]] = int(rule["target_color"])


def _choose_combo_rule(rules_for_color: list[dict], source_pos: tuple[int, int]) -> dict:
    containing = [rule for rule in rules_for_color if _point_in_bbox(source_pos, rule["bbox"])]
    pool = containing if containing else rules_for_color
    return min(
        pool,
        key=lambda rule: (
            -int(rule["depth"]),
            _bbox_area(rule["bbox"]),
            int(rule["bbox"][0]),
            int(rule["bbox"][1]),
            int(rule["target_color"]),
        ),
    )


def _choose_deepest_rule(rules_for_color: list[dict], _source_pos: tuple[int, int]) -> dict:
    return min(
        rules_for_color,
        key=lambda rule: (
            -int(rule["depth"]),
            _bbox_area(rule["bbox"]),
            int(rule["bbox"][0]),
            int(rule["bbox"][1]),
            int(rule["target_color"]),
        ),
    )


def _choose_nearest_rule(rules_for_color: list[dict], source_pos: tuple[int, int]) -> dict:
    sr, sc = int(source_pos[0]), int(source_pos[1])
    return min(
        rules_for_color,
        key=lambda rule: (
            _chebyshev((sr, sc), rule["source_pos"]),
            _chebyshev((sr, sc), rule["anchor"]),
            int(rule["source_pos"][0]),
            int(rule["source_pos"][1]),
            int(rule["target_color"]),
            len(rule["pattern"]),
        ),
    )


def _solve_with_selector(grid: np.ndarray, chooser) -> np.ndarray:
    g, bg, pattern_color, rules_by_source, legend_pixels, source_colors = _collect_rule_state(grid)
    out = _preserve_bystanders(g, bg, pattern_color, legend_pixels, source_colors)

    if not source_colors:
        return out

    for sr, sc in map(tuple, np.argwhere(np.isin(g, list(source_colors)))):
        sr = int(sr)
        sc = int(sc)
        if (sr, sc) in legend_pixels:
            continue
        source_color = int(g[sr, sc])
        chosen_rule = chooser(rules_by_source[source_color], (sr, sc))
        _stamp_rule(out, chosen_rule, (sr, sc))

    return out


def _main_deepest_only(grid: np.ndarray) -> np.ndarray:
    return _solve_with_selector(grid, _choose_deepest_rule)


def _main_nearest_same_color(grid: np.ndarray) -> np.ndarray:
    return _solve_with_selector(grid, _choose_nearest_rule)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode nested legend families, then branch by which same-color box contains each source pixel.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A legend is still a 4-neighbor source->target pair whose direction continues to an anchor
      touching a teal stencil.
    - Some source colors appear twice as a nested family: a larger outer teal ring contains a
      smaller inner teal ring, and the two legends use different target colors.

    Rule:
    - Recover all local legends and group them by source color.
    - For each non-legend source pixel, only compare legends of the same source color, keep the
      legends whose stencil bbox contains that pixel, and if several layers contain it choose the
      deepest one.
    - Translate the chosen legend's teal stencil so its anchor lands on the source pixel, recolor
      it to that legend's target color, and erase every legend cell plus every teal stencil.

    Key insight: Same-color ambiguity is resolved by visible nesting. The source pixel asks "which
    box am I inside?", and inner boxes override outer ones only when they also contain the pixel.
    """
    return _solve_with_selector(grid, _choose_combo_rule)


def _rect_ring(size: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if r in (0, size - 1) or c in (0, size - 1):
                cells.add((r, c))
    return cells


def _legend_triplet(size: int, side: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    k = (size - 3) // 2
    if side == "top":
        return (0, k), (0, k + 1), (0, k + 2)
    if side == "bottom":
        return (size - 1, k), (size - 1, k + 1), (size - 1, k + 2)
    if side == "left":
        return (k, 0), (k + 1, 0), (k + 2, 0)
    if side == "right":
        return (k, size - 1), (k + 1, size - 1), (k + 2, size - 1)
    raise ValueError(f"Unsupported side: {side}")


def _make_rule_shape(size: int, side: str) -> dict:
    ring = _rect_ring(size)
    source_cell, target_cell, anchor_cell = _legend_triplet(size, side)
    pattern_cells = sorted(ring - {source_cell, target_cell})
    ar, ac = anchor_cell
    rel = [(int(r - ar), int(c - ac)) for r, c in pattern_cells]
    return {
        "size": int(size),
        "side": side,
        "source_cell": source_cell,
        "target_cell": target_cell,
        "anchor_cell": anchor_cell,
        "pattern_cells": pattern_cells,
        "rel": rel,
    }


def _expanded_bbox_cells(top: int, left: int, size: int, shape: tuple[int, int], margin: int) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r in range(max(0, top - margin), min(h, top + size + margin)):
        for c in range(max(0, left - margin), min(w, left + size + margin)):
            cells.add((int(r), int(c)))
    return cells


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _build_groups(num_groups: int) -> tuple[list[dict], set[int]]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)
    groups: list[dict] = []
    used_colors: set[int] = set()

    for i in range(num_groups):
        side = random.choice(SIDES)
        src = colors[3 * i]
        tgt_outer = colors[3 * i + 1]
        tgt_inner = colors[3 * i + 2]
        used_colors.update((src, tgt_outer, tgt_inner))
        groups.append(
            {
                "src": int(src),
                "rules": [
                    {
                        "branch": "outer",
                        "src": int(src),
                        "tgt": int(tgt_outer),
                        "shape": _make_rule_shape(OUTER_SIZE, side),
                        "offset": (0, 0),
                    },
                    {
                        "branch": "inner",
                        "src": int(src),
                        "tgt": int(tgt_inner),
                        "shape": _make_rule_shape(INNER_SIZE, side),
                        "offset": (INNER_OFFSET, INNER_OFFSET),
                    },
                ],
            }
        )

    return groups, used_colors


def _place_groups(groups: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape

    for group in groups:
        placed = False
        for _ in range(GROUP_RETRIES):
            top = random.randint(0, h - OUTER_SIZE)
            left = random.randint(0, w - OUTER_SIZE)
            concrete_rules: list[dict] = []
            group_cells: set[tuple[int, int]] = set()
            ok = True

            for spec in group["rules"]:
                off_r, off_c = spec["offset"]
                base_r = top + int(off_r)
                base_c = left + int(off_c)
                shape = spec["shape"]
                source_pos = (base_r + shape["source_cell"][0], base_c + shape["source_cell"][1])
                target_pos = (base_r + shape["target_cell"][0], base_c + shape["target_cell"][1])
                anchor_pos = (base_r + shape["anchor_cell"][0], base_c + shape["anchor_cell"][1])
                pattern_cells = {(base_r + r, base_c + c) for r, c in shape["pattern_cells"]}
                rule_cells = {source_pos, target_pos} | pattern_cells
                if rule_cells & occupied or rule_cells & reserved:
                    ok = False
                    break
                concrete_rules.append(
                    {
                        "branch": spec["branch"],
                        "src": int(spec["src"]),
                        "tgt": int(spec["tgt"]),
                        "rel": [tuple(map(int, rc)) for rc in shape["rel"]],
                        "source_pos": source_pos,
                        "target_pos": target_pos,
                        "anchor": anchor_pos,
                        "pattern_cells": pattern_cells,
                        "bbox": (base_r, base_c, base_r + shape["size"] - 1, base_c + shape["size"] - 1),
                        "depth": 0 if spec["branch"] == "outer" else 1,
                        "scattered": [],
                    }
                )
                group_cells |= rule_cells

            if not ok:
                continue

            for rule in concrete_rules:
                sr, sc = rule["source_pos"]
                tr, tc = rule["target_pos"]
                inp[sr, sc] = int(rule["src"])
                inp[tr, tc] = int(rule["tgt"])
                for r, c in rule["pattern_cells"]:
                    inp[r, c] = 8

            occupied.update(group_cells)
            reserved.update(_expanded_bbox_cells(top, left, OUTER_SIZE, inp.shape, GROUP_MARGIN))
            group["rules"] = concrete_rules
            group["top_left"] = (int(top), int(left))
            group["outer_bbox"] = concrete_rules[0]["bbox"]
            group["inner_bbox"] = concrete_rules[1]["bbox"]
            placed = True
            break

        if not placed:
            return False

    return True


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = int(cell[0]), int(cell[1])
    return [(r + dr, c + dc) for dr, dc in _DIRS4]


def _candidate_footprint(rule: dict, pos: tuple[int, int], shape: tuple[int, int]) -> set[tuple[int, int]] | None:
    h, w = shape
    sr, sc = int(pos[0]), int(pos[1])
    cells = {(sr + rr, sc + cc) for rr, cc in rule["rel"]}
    if any(not (0 <= r < h and 0 <= c < w) for r, c in cells):
        return None
    return cells


def _candidate_cells_for_group(group: dict, inp: np.ndarray):
    outer_rule, inner_rule = group["rules"]
    colored_legend_cells = {
        outer_rule["source_pos"],
        outer_rule["target_pos"],
        inner_rule["source_pos"],
        inner_rule["target_pos"],
    }

    def valid_empty_cell(cell: tuple[int, int]) -> bool:
        r, c = cell
        if inp[r, c] != 0:
            return False
        return not any(nbr in colored_legend_cells for nbr in _neighbors4(cell))

    outer_candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []
    inner_candidates: list[tuple[tuple[int, int], set[tuple[int, int]]]] = []

    def nearest_generator_rule(pos: tuple[int, int]) -> dict:
        sr, sc = int(pos[0]), int(pos[1])
        return min(
            [outer_rule, inner_rule],
            key=lambda rule: (
                _chebyshev((sr, sc), rule["source_pos"]),
                _chebyshev((sr, sc), rule["anchor"]),
                int(rule["source_pos"][0]),
                int(rule["source_pos"][1]),
                int(rule["tgt"]),
            ),
        )

    r0, c0, r1, c1 = outer_rule["bbox"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            pos = (int(r), int(c))
            if not valid_empty_cell(pos):
                continue
            if _point_in_bbox(pos, inner_rule["bbox"]):
                continue
            footprint = _candidate_footprint(outer_rule, pos, inp.shape)
            if footprint is None:
                continue
            nearest = nearest_generator_rule(pos)
            if nearest["branch"] != "inner":
                continue
            outer_candidates.append((pos, footprint))

    r0, c0, r1, c1 = inner_rule["bbox"]
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            pos = (int(r), int(c))
            if not valid_empty_cell(pos):
                continue
            footprint = _candidate_footprint(inner_rule, pos, inp.shape)
            if footprint is None:
                continue
            inner_candidates.append((pos, footprint))

    random.shuffle(outer_candidates)
    random.shuffle(inner_candidates)
    return outer_candidates, inner_candidates


def _place_scattered(
    groups: list[dict],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> tuple[bool, set[tuple[int, int]], dict]:
    output_occupied: set[tuple[int, int]] = set()
    meta = {
        "outer_branch_count": 0,
        "inner_branch_count": 0,
        "nearest_mismatch_count": 0,
        "local_overlap_counts": [],
    }

    for group in groups:
        outer_rule, inner_rule = group["rules"]
        outer_candidates, inner_candidates = _candidate_cells_for_group(group, inp)
        if not outer_candidates or not inner_candidates:
            return False, set(), meta

        best_pair = None
        for _ in range(SCATTER_PAIR_RETRIES):
            for outer_pos, outer_fp in outer_candidates:
                if outer_fp & output_occupied:
                    continue
                for inner_pos, inner_fp in inner_candidates:
                    if inner_fp & output_occupied:
                        continue
                    if outer_pos == inner_pos or _chebyshev(outer_pos, inner_pos) < 2:
                        continue
                    overlap = len(outer_fp & inner_fp)
                    if overlap > MAX_LOCAL_FOOTPRINT_OVERLAP:
                        continue
                    score = (overlap, -_chebyshev(outer_pos, inner_pos))
                    if best_pair is None or score < best_pair[0]:
                        best_pair = (score, outer_pos, outer_fp, inner_pos, inner_fp)
            if best_pair is not None:
                break
            random.shuffle(outer_candidates)
            random.shuffle(inner_candidates)

        if best_pair is None:
            return False, set(), meta

        _score, outer_pos, outer_fp, inner_pos, inner_fp = best_pair
        inp[outer_pos] = int(group["src"])
        inp[inner_pos] = int(group["src"])
        occupied.add(outer_pos)
        occupied.add(inner_pos)
        _reserve_4neighbors(reserved, [outer_pos, inner_pos])

        outer_rule["scattered"] = [outer_pos]
        inner_rule["scattered"] = [inner_pos]
        output_occupied.update(outer_fp)
        output_occupied.update(inner_fp)

        meta["outer_branch_count"] += 1
        meta["inner_branch_count"] += 1
        meta["nearest_mismatch_count"] += 1
        meta["local_overlap_counts"].append(len(outer_fp & inner_fp))

    return True, output_occupied, meta


def _place_bystanders(
    used_colors: set[int],
    output_occupied: set[tuple[int, int]],
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> list[tuple[int, int, int]]:
    colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not colors:
        return []

    h, w = inp.shape
    count = random.randint(0, min(BYSTANDER_MAX, len(colors)))
    bystanders: list[tuple[int, int, int]] = []
    for _ in range(count):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in output_occupied:
                continue
            color = random.choice(colors)
            inp[r, c] = int(color)
            bystanders.append((int(r), int(c), int(color)))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return bystanders


def _build_output(groups: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = int(color)

    stamps: list[tuple[tuple[int, int], dict]] = []
    for group in groups:
        for rule in group["rules"]:
            for pos in rule["scattered"]:
                stamps.append((pos, rule))

    for (sr, sc), rule in sorted(stamps, key=lambda item: (item[0][0], item[0][1])):
        for rr, cc in rule["rel"]:
            out[sr + rr, sc + cc] = int(rule["tgt"])

    return out


def _gen_attempt(return_meta: bool = False):
    num_groups = random.choice(GROUP_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_GROUPS[num_groups]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    groups, used_colors = _build_groups(num_groups)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_groups(groups, inp, occupied, reserved):
        return None

    ok, output_occupied, meta = _place_scattered(groups, inp, occupied, reserved)
    if not ok:
        return None

    bystanders = _place_bystanders(used_colors, output_occupied, inp, occupied, reserved)
    out = _build_output(groups, bystanders, h, w)

    try:
        solved = main(inp)
        deepest = _main_deepest_only(inp)
        nearest = _main_nearest_same_color(inp)
    except Exception:
        return None

    if not np.array_equal(solved, out):
        return None
    if np.array_equal(deepest, out):
        return None
    if np.array_equal(nearest, out):
        return None

    if return_meta:
        meta.update(
            {
                "groups": int(num_groups),
                "bystanders": int(len(bystanders)),
                "deepest_diff_cells": int(np.count_nonzero(deepest != out)),
                "nearest_diff_cells": int(np.count_nonzero(nearest != out)),
            }
        )
        return inp, meta
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
