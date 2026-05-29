# Combo variant of abc82100 (index 33)
# combo_spec: D2/3+D11/2
# Name: 模板不同颜色区分+堆叠优先级

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the fused variant
# -------------------------------------
# a) concretization_and_behavior_trace:
#    - Visual form:
#      Each legend is a rectangular ring stencil whose boundary contains a 4-neighbor
#      source->target pair and the next boundary cell acts as the anchor. Unlike the
#      base task, every rule's ring uses its own template color.
#    - Fusion core:
#      One nested group is mandatory. Its outer and inner rings have different template
#      colors, different target colors, but the same source color. This makes rule
#      identity visible by color while priority is decided by spatial containment.
#    - Example trace:
#      Suppose blue->red is drawn on a large grey ring and blue->yellow is drawn on a
#      smaller teal ring nested inside it. In the original task, source color uniquely
#      selected one rule and every stencil was the same color, so neither repeated
#      source colors nor colored-template identity existed. In this combo variant, rule
#      discovery finds both local legends, but only the inner one is active for blue
#      because its stencil bbox is strictly inside the outer one. Scattered blue pixels
#      therefore stamp only the smaller inner ring, recolored yellow. A solver that
#      ignores priority would stamp both rings and leave a larger outer boundary behind.
# b) complexity_calibration:
#    - The palette only has 9 non-background colors, so the generator uses exactly one
#      mandatory nested group (5 colors) and optionally one singleton rule (3 more).
#      This keeps the board readable and guarantees at least two differently colored
#      templates in every sample.
#    - Nesting depth is capped at 2. That is enough to make priority visible without
#      turning legend parsing into arbitrary recursion.
# c) implicit_assumptions_audit:
#    - Original assumption "all templates are color 8" is false; rule discovery must
#      scan connected components for every non-background color.
#    - Original assumption "one source color implies one rule" is false; rules must be
#      grouped by source color and filtered by containment depth before stamping.
#    - Original cleanup by excluding only color 8 is false; template pixels now have
#      multiple colors, so cleanup must remove the exact recovered template cells.
#    - Stamping still depends only on recovered relative coordinates, so that part of
#      the original mechanism remains valid.
# d) placement_legality:
#    - Nested rings use sizes (7,3) or (9,5) with a one-cell moat, so the two template
#      components never touch, even diagonally.
#    - Group placement reserves the whole bounding box plus one halo cell, preventing
#      accidental anchor matches from scattered pixels or bystanders.
#    - Scattered source pixels and bystanders reserve 8-neighborhoods, which blocks
#      spurious adjacent color pairs and stray near-anchor components.
# e) visual_observability:
#    - Different template colors are directly visible inside the legends.
#    - Reused source color plus concentric placement makes the priority cue human-
#      visible: the inner legend is literally enclosed by the outer one.
#    - The rule does not depend on numeric color ordering; only equality and spatial
#      containment matter, which are visible from examples.
# f) visual_cleanliness:
#    - Legends are compact ring groups instead of arbitrary shapes scattered around the
#      board, so the training examples stay legible.
#    - Optional singleton legends use the same visual language as the nested group and
#      are placed with bbox halos to avoid clutter.
# g) inferability_of_change:
#    - Template color is only an identity tag for "this stencil belongs to this legend".
#    - Priority is supported by repeated scattered copies of the repeated source color:
#      humans can see that only the inner nested stencil survives on those copies.
# solver_assumptions_post_change:
#    - _discover_rules assumes legend source/target cells are singleton components and
#      template components are multi-pixel; the generator enforces this by using ring
#      templates plus isolated colored dots.
#    - _discover_rules also assumes each anchor has exactly one nearest candidate
#      template within Chebyshev distance 1 after excluding source/target colors; the
#      generator enforces this with halo spacing.
#    - _select_active_rules assumes containment priority only needs to be resolved
#      among legends that share a source color; the generator creates one such nested
#      same-source group and keeps other source colors unique.
#    - _render_output assumes all non-legend pixels with an active source color are
#      consumable scattered sources; the generator never places bystanders in source
#      colors and never repeats source colors outside their intended rule groups.
#    - main() also assumes every real source color appears multiple times in the grid,
#      while target colors do not; the generator guarantees this via 3+ scattered
#      copies per active rule.


ALL_NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SIDES = ["top", "bottom", "left", "right"]
_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


# Generator config
GRID_DIM_MAX = 20
GRID_DIM_MIN = 12
MIN_DIM_BY_EXAMPLES = {2: 12, 3: 15}

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 36

BYSTANDER_MAX = 4

GROUP_RETRIES = 300
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 80
OUTER_RETRIES = 300


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


def _pattern_components_by_color(grid: np.ndarray, bg: int) -> dict[int, list[np.ndarray]]:
    components: dict[int, list[np.ndarray]] = {}
    for color in sorted(set(map(int, grid.flatten()))):
        if color == int(bg):
            continue
        parts = components_mask(grid == color, connectivity=8)
        if parts:
            components[int(color)] = [np.asarray(part, dtype=int) for part in parts]
    return components


def _discover_rules(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    color_counts = {int(color): int(np.sum(grid == color)) for color in set(map(int, grid.flatten())) if color != int(bg)}
    patterns_by_color = _pattern_components_by_color(grid, bg)
    singleton_cells: set[tuple[int, int]] = set()
    for parts in patterns_by_color.values():
        for part in parts:
            if len(part) == 1:
                singleton_cells.add((int(part[0, 0]), int(part[0, 1])))
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int, int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color == int(bg):
                continue
            if color_counts.get(source_color, 0) <= 1:
                continue
            if (int(r), int(c)) not in singleton_cells:
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color == int(bg) or target_color == source_color:
                    continue
                if (int(tr), int(tc)) not in singleton_cells:
                    continue

                anchor = (tr + dr, tc + dc)
                if not (0 <= anchor[0] < h and 0 <= anchor[1] < w):
                    continue

                candidates: list[tuple[int, int, int, int, np.ndarray]] = []
                for pattern_color, patterns in patterns_by_color.items():
                    if pattern_color in (source_color, target_color):
                        continue
                    for i, pattern in enumerate(patterns):
                        if len(pattern) < 3:
                            continue
                        dist = _anchor_distance(pattern, anchor)
                        if dist <= 1:
                            candidates.append((int(dist), len(pattern), int(pattern_color), int(i), pattern))

                if not candidates:
                    continue

                candidates.sort(key=lambda item: (item[0], item[1]))
                _dist, _size, pattern_color, pattern_index, pattern = candidates[0]
                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_color), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "pattern_color": int(pattern_color),
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
            key=lambda rule: (
                int(rule["depth"]),
                -len(rule["pattern"]),
                -rule["bbox"][0],
                -rule["bbox"][1],
            ),
        )
        best["source_color"] = int(source_color)
        active.append(best)
    return active


def _render_output(grid: np.ndarray, bg: int, all_rules: list[dict], rules_to_apply: list[dict]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    legend_pixels: set[tuple[int, int]] = set()
    template_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in all_rules:
        legend_pixels.add(rule["source_pos"])
        legend_pixels.add(rule["target_pos"])
        template_pixels.update(map(tuple, rule["pattern"]))
        source_colors.add(int(rule["source_color"]))

    out = np.full_like(g, int(bg))

    for r, c in map(tuple, np.argwhere(g != int(bg))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or (r, c) in template_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules_to_apply:
        ar, ac = rule["anchor"]
        pattern = np.asarray(rule["pattern"], dtype=int)
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


def _solve_without_priority(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg)
    return _render_output(g, bg, rules, rules)


def combo_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg)
    active_rules = _select_active_rules(rules)
    combo_out = _render_output(g, bg, rules, active_rules)
    naive_out = _render_output(g, bg, rules, rules)
    repeated_sources = len(rules) - len({int(rule["source_color"]) for rule in rules})
    return {
        "discovered_rules": len(rules),
        "active_rules": len(active_rules),
        "template_color_count": len({int(rule["pattern_color"]) for rule in rules}),
        "repeated_sources": int(repeated_sources),
        "priority_changes_output": not np.array_equal(combo_out, naive_out),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover colored local stencil legends, but apply only the highest-priority legend for each source color.

    Input structure:
    - Black background with isolated colored source pixels and several stencil components whose colors differ
      from rule to rule instead of all being teal.
    - A legend is still a 4-neighbor source->target pair whose forward direction reaches an anchor touching a
      same-component stencil.
    - At least one source color appears in nested legends: an outer stencil encloses an inner stencil.

    Rule:
    - Discover every local legend by pairing each source->target example with the nearby stencil component,
      regardless of that stencil's color.
    - Group discovered legends by source color. If a source color has several legends, keep only the one with
      greatest nesting depth, i.e. the stencil whose bbox is contained inside the others.
    - Stamp that winning stencil onto every non-legend pixel of the same source color, recolored to the winning
      target color, and erase all legend/template pixels.

    Key insight: The stencil color tells you which pixels belong to the same template, but it does not decide
    the output color. When the same source color appears in nested colored templates, the inner enclosed legend
    overrides the outer one.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _discover_rules(g, bg=bg)
    active_rules = _select_active_rules(rules)
    return _render_output(g, bg, rules, active_rules)


def _reserve_8neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS8:
            reserved.add((r + dr, c + dc))


def _rect_ring(size: int) -> set[tuple[int, int]]:
    coords: set[tuple[int, int]] = set()
    for r in range(size):
        for c in range(size):
            if r in (0, size - 1) or c in (0, size - 1):
                coords.add((r, c))
    return coords


def _legend_triplet(size: int, side: str) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if side == "top":
        return (0, 0), (0, 1), (0, 2)
    if side == "bottom":
        return (size - 1, size - 1), (size - 1, size - 2), (size - 1, size - 3)
    if side == "left":
        return (0, 0), (1, 0), (2, 0)
    if side == "right":
        return (size - 1, size - 1), (size - 2, size - 1), (size - 3, size - 1)
    raise ValueError(f"Unsupported side: {side}")


def _make_rule_shape(size: int, side: str) -> dict:
    ring = _rect_ring(size)
    source_cell, target_cell, anchor_cell = _legend_triplet(size, side)
    pattern_cells = sorted(ring - {source_cell, target_cell})
    ar, ac = anchor_cell
    rel = [(r - ar, c - ac) for r, c in pattern_cells]
    return {
        "size": int(size),
        "side": side,
        "source_cell": source_cell,
        "target_cell": target_cell,
        "anchor_cell": anchor_cell,
        "pattern_cells": pattern_cells,
        "rel": rel,
    }


def _expanded_bbox_cells(top: int, left: int, size: int, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for r in range(max(0, top - 1), min(h, top + size + 1)):
        for c in range(max(0, left - 1), min(w, left + size + 1)):
            cells.add((r, c))
    return cells


def _build_groups() -> tuple[list[dict], set[int], int]:
    colors = ALL_NON_BG_COLORS[:]
    random.shuffle(colors)

    side = random.choice(SIDES)
    outer_size = random.choice([7, 9])
    inner_size = outer_size - 4

    src = colors[0]
    tgt_outer = colors[1]
    tgt_inner = colors[2]
    pattern_outer = colors[3]
    pattern_inner = colors[4]

    groups = [
        {
            "kind": "nested",
            "bbox_size": outer_size,
            "rules": [
                {
                    "src": src,
                    "tgt": tgt_outer,
                    "pattern_color": pattern_outer,
                    "shape": _make_rule_shape(outer_size, side),
                    "offset": (0, 0),
                },
                {
                    "src": src,
                    "tgt": tgt_inner,
                    "pattern_color": pattern_inner,
                    "shape": _make_rule_shape(inner_size, side),
                    "offset": (2, 2),
                },
            ],
            "active_index": 1,
        }
    ]
    used_colors: set[int] = {src, tgt_outer, tgt_inner, pattern_outer, pattern_inner}

    example_count = 2
    if random.random() < 0.65:
        singleton_size = random.choice([3, 5, 7])
        singleton_side = random.choice(SIDES)
        singleton = {
            "kind": "single",
            "bbox_size": singleton_size,
            "rules": [
                {
                    "src": colors[5],
                    "tgt": colors[6],
                    "pattern_color": colors[7],
                    "shape": _make_rule_shape(singleton_size, singleton_side),
                    "offset": (0, 0),
                }
            ],
            "active_index": 0,
        }
        groups.append(singleton)
        used_colors.update({colors[5], colors[6], colors[7]})
        example_count += 1

    return groups, used_colors, example_count


def _place_groups(
    groups: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]
):
    h, w = inp.shape
    all_rules: list[dict] = []
    active_rules: list[dict] = []

    for group_id, group in enumerate(groups):
        size = int(group["bbox_size"])
        placed = False
        for _ in range(GROUP_RETRIES):
            top = random.randint(0, h - size)
            left = random.randint(0, w - size)
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

                if rule_cells & group_cells:
                    ok = False
                    break
                if rule_cells & occupied or rule_cells & reserved:
                    ok = False
                    break

                concrete_rules.append(
                    {
                        "group_id": group_id,
                        "src": int(spec["src"]),
                        "tgt": int(spec["tgt"]),
                        "pattern_color": int(spec["pattern_color"]),
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
                inp[sr, sc] = int(rule["src"])
                inp[tr, tc] = int(rule["tgt"])
                for r, c in rule["pattern_cells"]:
                    inp[r, c] = int(rule["pattern_color"])

            occupied.update(group_cells)
            reserved.update(_expanded_bbox_cells(top, left, size, h, w))
            all_rules.extend(concrete_rules)
            active_rules.append(concrete_rules[group["active_index"]])
            placed = True
            break

        if not placed:
            return None, None

    return all_rules, active_rules


def _place_scattered(
    active_rules: list[dict], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]
) -> bool:
    h, w = inp.shape
    area = h * w
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    for rule in active_rules:
        n_sc = random.randint(SCATTERED_MIN, sc_max)
        positions: list[tuple[int, int]] = []
        for _ in range(n_sc * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= n_sc:
                break
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            inp[r, c] = int(rule["src"])
            _reserve_8neighbors(reserved, [(r, c)])
        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions

    return True


def _place_bystanders(
    used_colors: set[int], inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]
):
    bystander_colors = [c for c in ALL_NON_BG_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    bystanders: list[tuple[int, int, int]] = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = int(color)
            bystanders.append((r, c, int(color)))
            occupied.add((r, c))
            _reserve_8neighbors(reserved, [(r, c)])
            break
    return bystanders


def _build_output(active_rules: list[dict], bystanders: list[tuple[int, int, int]], h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[r, c] = int(color)
    for rule in active_rules:
        for sr, sc in rule["scattered"]:
            for dr, dc in rule["rel"]:
                nr = sr + dr
                nc = sc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _gen_attempt():
    groups, used_colors, example_count = _build_groups()
    min_dim = max(GRID_DIM_MIN, MIN_DIM_BY_EXAMPLES[example_count])
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    all_rules, active_rules = _place_groups(groups, inp, occupied, reserved)
    if all_rules is None:
        return None
    if not _place_scattered(active_rules, inp, occupied, reserved):
        return None
    _place_bystanders(used_colors, inp, occupied, reserved)

    try:
        metrics = combo_metrics(inp)
        solved = main(inp)
    except Exception:
        return None
    if metrics["discovered_rules"] != len(all_rules):
        return None
    if metrics["active_rules"] != len(active_rules):
        return None
    if metrics["template_color_count"] != len(all_rules):
        return None
    if metrics["repeated_sources"] < 1:
        return None
    if not bool(metrics["priority_changes_output"]):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
