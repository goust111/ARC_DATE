# Variant of abc82100
# Dimension: D2 — 规则示例的编码格式 (symbol_instructions)
# Change: 规则示例由相邻的源色-目标色像素对加上附近的teal模板图案构成，颜色对直接编码映射关系... → 规则示例按堆叠层次编码优先级

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for this D2 variant
# -----------------------------------
# spec_variant:
#   core_change:
#     original: each source color had one direct local legend, so color alone chose the rule
#     variant: a source color may have several legends with different targets/stencils; the
#       active one is selected by spatial nesting depth, with the innermost legend winning
#   concrete_visual_form:
#     - Each legend still contains a 4-neighbor source->target pair and a nearby teal stencil.
#     - The stencil is now a rectangular teal ring with two boundary cells replaced by the
#       source/target pair, so the whole legend sits inside one bounding box.
#     - At least one source color appears twice: an outer ring legend and a smaller inner ring
#       legend nested inside it. Their target colors differ.
#     - Scattered source pixels of that color only use the inner legend's stencil/target.
#   behavior_trace:
#     - Original logic on a nested same-source pair would stamp both learned stencils onto every
#       scattered source pixel of that color, producing an unwanted union of outer+inner shapes.
#     - Variant logic first groups rules by source color, measures stencil-bbox containment, and
#       keeps only the deepest rule in each same-source group before stamping.
#     - Difference is visible because outer and inner rings have different sizes, so applying both
#       cannot accidentally match the inner-only output.
#   complexity_calibration:
#     - Nesting depth is limited to 2 to keep the hierarchy readable.
#     - Total legend count stays within 2-4, close to the original task scale.
#     - Colors are globally unique except for the intentionally repeated source color inside a
#       nested group, which gives repeated support for the priority rule without causing clutter.
#   implicit_assumptions_audit:
#     - Original main() assumed "one source color -> one rule"; in the variant this is false, so
#       rule application must choose one active rule per source color.
#     - Original discovery used the nearest teal component to the anchor; generation must keep the
#       inner anchor at Chebyshev distance >= 2 from the outer ring, so discovery remains stable.
#     - Original preserve-pass removed every non-legend pixel whose color was any source color;
#       that still holds because all scattered pixels of such colors are consumable.
#   placement_legality:
#     - Nested rings use sizes (7,3) or (9,5), so the inner bbox is strictly inside the outer bbox
#       with a one-cell empty moat. This prevents 8-connected merging of teal components.
#     - Group bboxes are separated by a reserved one-cell margin, and scattered/bystander pixels
#       reserve 4-neighbors, preventing spurious colored pairs.
#   visual_observability:
#     - The repeated source color plus different target colors inside concentric teal rings makes
#       the hierarchy visible in 2-3 examples; humans can compare it with the output, where only
#       the inner target/ring survives on scattered instances.
#   neatness:
#     - Only one nested group is required, with at most one extra nested group or up to two single
#       legends. This keeps the legend region interpretable instead of saturating the board.


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

            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
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
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                        "bbox": _pattern_bbox(np.asarray(pattern, dtype=int)),
                    }
                )

    return rules


def _select_active_rules(rules: list[dict]) -> list[dict]:
    by_source: dict[int, list[dict]] = {}
    for rule in rules:
        by_source.setdefault(int(rule["source_color"]), []).append(rule)

    active: list[dict] = []
    for source_color, same_source in by_source.items():
        for rule in same_source:
            rule["depth"] = sum(
                1
                for other in same_source
                if other is not rule and _strictly_contains(other["bbox"], rule["bbox"])
            )
        best = max(
            same_source,
            key=lambda rule: (int(rule["depth"]), -len(rule["pattern"]), -rule["bbox"][0], -rule["bbox"][1]),
        )
        best["source_color"] = int(source_color)
        active.append(best)
    return active


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn stencil-copy rules from local legends, but resolve duplicate source colors by nesting priority.

    Input structure:
    - Black background with isolated colored pixels and teal (8) stencil components.
    - A legend is still a 4-neighbor source->target pair whose continued direction reaches a teal stencil.
    - Some legends reuse the same source color and appear as nested teal rings, with one legend enclosed
      inside another.

    Rule:
    - Discover all local legends exactly as before, then group them by source color.
    - If a source color has multiple candidate legends, keep the most deeply nested stencil (innermost bbox).
    - Stamp only that winning stencil onto every non-legend pixel of the same source color, recolored to the
      winning target color, and erase all legend/stencil pixels.

    Key insight: Color alone no longer identifies the rule. When the same source color appears in concentric
    legends, the inner stencil has higher priority and suppresses the outer one.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    pattern_color = 8
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, pattern_color=pattern_color)
    active_rules = _select_active_rules(rules)

    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        source_colors.add(int(rule["source_color"]))

    out = np.full_like(g, int(bg))

    # Preserve colored bystanders that are neither legend cells nor consumable source pixels.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in active_rules:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == int(rule["source_color"]))):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


# --------------- generator (c) ---------------

# Grid dimensions
GRID_DIM_MAX = 20
GRID_DIM_MIN = 12
MIN_DIM_BY_EXAMPLES = {2: 12, 3: 14, 4: 16}

# Rule / group sampling
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
SIDES = ["top", "bottom", "left", "right"]

# Scattered source pixels
SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 36

# Bystanders
BYSTANDER_MAX = 5

# Placement retry budgets
GROUP_RETRIES = 300
SCATTERED_RETRIES_PER_PIXEL = 60
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 300

# 4-connected directions
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _rect_ring(size: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if r in (0, size - 1) or c in (0, size - 1):
                coords.add((r, c))
    return coords


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
    pattern = sorted(ring - {source_cell, target_cell})
    ar, ac = anchor_cell
    rel = [(r - ar, c - ac) for r, c in pattern]
    return {
        "size": int(size),
        "side": side,
        "source_cell": source_cell,
        "target_cell": target_cell,
        "anchor_cell": anchor_cell,
        "pattern_cells": pattern,
        "rel": rel,
    }


def _expanded_bbox_cells(top: int, left: int, size: int, H: int, W: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, top - 1), min(H, top + size + 1)):
        for c in range(max(0, left - 1), min(W, left + size + 1)):
            cells.add((r, c))
    return cells


def _build_groups() -> tuple[list[dict], set[int], int]:
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    nested_count = random.choice([1, 1, 1, 2])
    example_count = nested_count * 2
    singleton_count = random.randint(0, 4 - example_count)

    groups: list[dict] = []
    used_colors: set[int] = set()
    color_index = 0

    for _ in range(nested_count):
        side = random.choice(SIDES)
        outer_size = random.choice([7, 9])
        inner_size = outer_size - 4
        src = colors[color_index]
        tgt_outer = colors[color_index + 1]
        tgt_inner = colors[color_index + 2]
        color_index += 3
        used_colors.update((src, tgt_outer, tgt_inner))
        groups.append(
            {
                "kind": "nested",
                "bbox_size": outer_size,
                "rules": [
                    {"src": src, "tgt": tgt_outer, "shape": _make_rule_shape(outer_size, side), "offset": (0, 0)},
                    {"src": src, "tgt": tgt_inner, "shape": _make_rule_shape(inner_size, side), "offset": (2, 2)},
                ],
                "active_index": 1,
            }
        )

    for _ in range(singleton_count):
        side = random.choice(SIDES)
        size = random.choice([3, 5, 7])
        src = colors[color_index]
        tgt = colors[color_index + 1]
        color_index += 2
        used_colors.update((src, tgt))
        groups.append(
            {
                "kind": "single",
                "bbox_size": size,
                "rules": [{"src": src, "tgt": tgt, "shape": _make_rule_shape(size, side), "offset": (0, 0)}],
                "active_index": 0,
            }
        )

    return groups, used_colors, example_count + singleton_count


def _place_groups(groups: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    H, W = inp.shape
    all_rules: list[dict] = []
    active_rules: list[dict] = []

    for group_id, group in enumerate(groups):
        size = int(group["bbox_size"])
        placed = False
        for _ in range(GROUP_RETRIES):
            top = random.randint(0, H - size)
            left = random.randint(0, W - size)
            group_cells: set[tuple[int, int]] = set()
            concrete_rules: list[dict] = []
            ok = True

            for spec in group["rules"]:
                off_r, off_c = spec["offset"]
                base_r = top + off_r
                base_c = left + off_c
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
                        "group_id": group_id,
                        "src": int(spec["src"]),
                        "tgt": int(spec["tgt"]),
                        "rel": [tuple(map(int, rc)) for rc in shape["rel"]],
                        "source_pos": source_pos,
                        "target_pos": target_pos,
                        "anchor_pos": anchor_pos,
                        "pattern_cells": pattern_cells,
                    }
                )
                group_cells |= rule_cells

            if not ok:
                continue

            for rule in concrete_rules:
                sr, sc = rule["source_pos"]
                tr, tc = rule["target_pos"]
                inp[sr, sc] = rule["src"]
                inp[tr, tc] = rule["tgt"]
                for r, c in rule["pattern_cells"]:
                    inp[r, c] = 8

            occupied.update(group_cells)
            reserved.update(_expanded_bbox_cells(top, left, size, H, W))
            all_rules.extend(concrete_rules)
            active_rules.append(concrete_rules[group["active_index"]])
            placed = True
            break

        if not placed:
            return None, None

    return all_rules, active_rules


def _place_scattered(active_rules: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    H, W = inp.shape
    area = H * W
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    for rule in active_rules:
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        positions: list[tuple[int, int]] = []
        for _ in range(n_sc * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= n_sc:
                break
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            inp[r, c] = int(rule["src"])
            for dr, dc in _DIRS4:
                reserved.add((r + dr, c + dc))
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions

    return True


def _place_bystanders(used_colors: set[int], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not colors:
        return []

    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(colors) * 2))
    bystanders: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(colors)
            inp[r, c] = color
            bystanders.append((r, c, color))
            occupied.add((r, c))
            for dr, dc in _DIRS4:
                reserved.add((r + dr, c + dc))
            break
    return bystanders


def _build_output(active_rules: list[dict], bystanders: list[tuple[int, int, int]], H: int, W: int) -> np.ndarray:
    out = np.zeros((H, W), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = int(color)
    for rule in active_rules:
        for sr, sc in rule["scattered"]:
            for dr, dc in rule["rel"]:
                nr = sr + dr
                nc = sc + dc
                if 0 <= nr < H and 0 <= nc < W:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _gen_attempt(return_meta: bool = False):
    groups, used_colors, example_count = _build_groups()
    min_dim = max(GRID_DIM_MIN, MIN_DIM_BY_EXAMPLES[example_count])
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    inp = np.zeros((H, W), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    all_rules, active_rules = _place_groups(groups, inp, occupied, reserved)
    if all_rules is None:
        return None
    if not _place_scattered(active_rules, inp, occupied, reserved):
        return None
    bystanders = _place_bystanders(used_colors, inp, occupied, reserved)
    output = _build_output(active_rules, bystanders, H, W)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, output):
        return None

    if return_meta:
        nested_groups = sum(1 for group in groups if group["kind"] == "nested")
        repeated_source_colors = len({rule["src"] for rule in active_rules}) < len(all_rules)
        return inp, {
            "nested_groups": nested_groups,
            "example_count": example_count,
            "repeated_source_colors": repeated_source_colors,
        }
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
