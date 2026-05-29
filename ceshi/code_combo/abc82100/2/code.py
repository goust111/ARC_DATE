# Combo variant of abc82100 (index 2)
# combo_spec: D1/2+D3/0
# Name: 多属性联合匹配+多像素色块源

# Design analysis for this combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The original task learns a rule from a local legend made of
#      source-pixel -> target-pixel -> teal-stencil, then stamps that stencil on
#      every scattered source pixel of the same color.
#    - In this combo, the free-standing sources are no longer isolated pixels.
#      They are 4-connected monochrome blocks of size >= 2. Each block has one
#      visible geometric anchor: its top-left occupied cell.
#    - Matching is no longer color-only. The learned key is
#      (source color, anchor 4-neighbor non-background count). For a source
#      block, this count is measured at the block anchor, so the count depends
#      jointly on the block's own same-color body and on nearby gray support
#      markers.
#    - Each legend source is also a multi-pixel block. Its top-left cell is the
#      designated source cell touching the target pixel. Example trace:
#      a red legend block whose anchor touches one same-color block cell, one
#      blue target cell, and one gray helper has count 3. Elsewhere, a red block
#      whose top-left cell touches one same-color cell and two gray helpers also
#      has count 3, so it receives that legend's blue stencil. Another red block
#      with only one gray helper has count 2, so it must follow a different red
#      rule instead.
#    - Compared with the original task, the solver must first segment source
#      blobs, extract one anchor per blob, and only then decide which same-color
#      rule applies by local anchor context.
#
# b) Complexity calibration
#    - The context vocabulary is restricted to counts {2, 3}. This is large
#      enough to force multi-attribute matching, but still small enough to be
#      visible and learnable from 2-4 ARC examples.
#    - Every generated puzzle contains at least two rules sharing one source
#      color but split across the two counts, so color-only reasoning is
#      insufficient.
#    - All source blocks are multi-cell and all chosen source shapes have anchor
#      internal degree 1. This keeps the count logic stable and readable.
#
# c) Hidden assumptions that break from the original solver
#    - Iterating over every pixel of a source color is invalid: a whole block
#      should trigger one stamp, not one stamp per cell.
#    - Treating source color as a unique rule key is invalid: the same color can
#      map to different targets/stencils under counts 2 and 3.
#    - Removing only the single legend source pixel is invalid: each legend now
#      contains a whole source block plus gray support markers.
#    - Preserving every non-source non-teal pixel is invalid: gray markers are
#      input-only context and must disappear when attached to a matched or legend
#      anchor.
#
# d) Placement legality
#    - Legend blocks, free source blocks, teal stencils, and bystanders are
#      separated by reserved 4-neighborhood padding so no accidental adjacencies
#      create false legends or alter anchor counts.
#    - Legend source cells are always the top-left cell of their block, and the
#      target direction is chosen so neither the target cell nor the continuation
#      anchor overlaps the source block.
#    - Gray support markers are placed only on sides of the anchor cell that are
#      not occupied by the source block or the target cell.
#
# e) Visual observability
#    - The fusion is visible in both directions: the input plainly contains
#      source blobs rather than pixels, and the same source color can appear with
#      two different gray-support patterns that lead to different outputs.
#    - The decisive local feature is human-visible because it is just the number
#      of non-black 4-neighbors around the blob's top-left cell.
#
# f) Visual cleanliness
#    - Source blocks come from a small shape vocabulary and are separated by
#      padding, so the grids do not collapse into touching color masses.
#    - Gray is reserved for context markers and teal is reserved for stencils,
#      keeping object roles easy to distinguish.
#
# g) Change inferability
#    - Color changes are always taught by a nearby legend. The additional split
#      between same-color rules is not hidden metadata: it is backed by an
#      explicit visible difference in anchor neighborhood count.
#    - The solver never uses numeric color ordering or invisible tie-breakers;
#      only visible color identity, connectedness, and local adjacency matter.
#
# 9) New solver assumptions and generator guarantees
#    - Assumption: every learned key (source color, neighbor count) is unique.
#      Guarantee: the generator samples unique keys and validates them from the
#      discovered legends.
#    - Assumption: every free source block has a recoverable top-left anchor with
#      internal degree 1 and no extra accidental neighbors.
#      Guarantee: source shapes are filtered to anchor degree 1 and surrounding
#      cells are reserved during placement.
#    - Assumption: gray cells adjacent to matched anchors are input-only support
#      and not bystanders.
#      Guarantee: the generator uses gray only for support markers and never as a
#      preserved bystander color.
#    - Assumption: the fusion is behaviorally necessary, not decorative.
#      Guarantee: the generator rejects samples where a color-only blob baseline
#      would produce the same output as the full combo solver.

from __future__ import annotations

import random
import sys

import numpy as np

SOLVER_UTILS = "/home/lixinyu/snap/new/solvers"
if SOLVER_UTILS not in sys.path:
    sys.path.append(SOLVER_UTILS)

from arc_utils import components_mask, get_bg


SOURCE_COLORS = [1, 2, 3]
TARGET_COLORS = [4, 6, 7, 9]
MARKER_COLOR = 5
PATTERN_COLOR = 8
MATCH_COUNTS = [2, 3]

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 11, 3: 15, 4: 18}
RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]

STENCIL_MAX_CAP = 12
STENCIL_MIN_AREA_DIV = 100
STENCIL_MAX_AREA_DIV = 25

SOURCE_BLOCK_MIN = 2
SOURCE_BLOCK_MAX_CAP = 5
SOURCE_BLOCK_AREA_DIV = 55

BYSTANDER_MAX = 3

LEGEND_RETRIES = 600
SOURCE_BLOCK_RETRIES_PER_BLOCK = 100
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

_BASE_SOURCE_SHAPES = [
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (0, 2), (1, 2)],
    [(0, 0), (1, 0), (2, 0), (3, 0)],
    [(0, 0), (0, 1), (0, 2), (0, 3)],
]


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _count_non_bg4(grid: np.ndarray, r: int, c: int, bg: int) -> int:
    h, w = grid.shape
    total = 0
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            total += 1
    return total


def _non_bg_neighbors4(grid: np.ndarray, r: int, c: int, bg: int) -> list[tuple[int, int]]:
    h, w = grid.shape
    out: list[tuple[int, int]] = []
    for dr, dc in _DIRS4:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != int(bg):
            out.append((int(nr), int(nc)))
    return out


def _normalize_shape(cells) -> list[tuple[int, int]]:
    pts = [(int(r), int(c)) for r, c in cells]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return sorted((r - r0, c - c0) for r, c in pts)


def _rotate_reflect_shape(cells) -> list[tuple[int, int]]:
    transformed = []
    flip = random.choice([False, True])
    turns = random.randint(0, 3)
    for r, c in cells:
        rr, cc = int(r), int(c)
        if flip:
            cc = -cc
        for _ in range(turns):
            rr, cc = cc, -rr
        transformed.append((rr, cc))
    return _normalize_shape(transformed)


def _shape_anchor_degree(shape: list[tuple[int, int]]) -> int:
    pts = set((int(r), int(c)) for r, c in shape)
    total = 0
    for dr, dc in _DIRS4:
        if (dr, dc) in pts:
            total += 1
    return total


def _make_source_shape() -> list[tuple[int, int]]:
    for _ in range(200):
        shape = _rotate_reflect_shape(random.choice(_BASE_SOURCE_SHAPES))
        if len(shape) >= 2 and _shape_anchor_degree(shape) == 1:
            return shape
    raise RuntimeError("Failed to generate source shape with anchor degree 1")


def _component_anchor(component: np.ndarray) -> tuple[int, int]:
    cells = sorted((int(r), int(c)) for r, c in component)
    return cells[0]


def _component_set(component: np.ndarray) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r, c in component}


def _source_components(
    grid: np.ndarray, source_color: int, excluded: set[tuple[int, int]]
) -> list[np.ndarray]:
    mask = grid == int(source_color)
    if excluded:
        mask = mask.copy()
        for r, c in excluded:
            if 0 <= r < mask.shape[0] and 0 <= c < mask.shape[1]:
                mask[r, c] = False
    return components_mask(mask, connectivity=4)


def _component_containing(
    grid: np.ndarray, source_color: int, cell: tuple[int, int]
) -> np.ndarray | None:
    for component in components_mask(grid == int(source_color), connectivity=4):
        if any(int(r) == int(cell[0]) and int(c) == int(cell[1]) for r, c in component):
            return component
    return None


def _translated_cells(anchor: tuple[int, int], rel_cells) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(int(ar + rr), int(ac + cc)) for rr, cc in rel_cells}


def _eligible_target_dirs(shape: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(shape)
    out = []
    for dr, dc in _DIRS4:
        if (dr, dc) in pts:
            continue
        if (2 * dr, 2 * dc) in pts:
            continue
        out.append((int(dr), int(dc)))
    return out


def _available_support_dirs(
    shape: list[tuple[int, int]], forbidden: set[tuple[int, int]]
) -> list[tuple[int, int]]:
    pts = set(shape)
    out = []
    for dr, dc in _DIRS4:
        if (int(dr), int(dc)) in pts:
            continue
        if (int(dr), int(dc)) in forbidden:
            continue
        out.append((int(dr), int(dc)))
    return out


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        reserved.add((int(r), int(c)))
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    n = random.randint(min_pixels, max_pixels)
    coords = {(0, 0)}
    for _ in range(n - 1):
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                p = (r + dr, c + dc)
                if p not in coords:
                    frontier.add(p)
        coords.add(random.choice(list(frontier)))
    coords_list = sorted(coords)
    return coords_list, random.choice(coords_list)


def _discover_rules(
    grid: np.ndarray,
    bg: int,
    pattern_color: int = PATTERN_COLOR,
    marker_color: int = MARKER_COLOR,
) -> list[dict]:
    h, w = grid.shape
    patterns = components_mask(grid == pattern_color, connectivity=8)
    rules: list[dict] = []
    seen: set[tuple[tuple[int, int], tuple[int, int], int]] = set()

    for r in range(h):
        for c in range(w):
            source_color = int(grid[r, c])
            if source_color in (int(bg), int(pattern_color), int(marker_color)):
                continue

            for dr, dc in _DIRS4:
                tr = r + dr
                tc = c + dc
                if not (0 <= tr < h and 0 <= tc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color), int(marker_color)):
                    continue
                if target_color == source_color:
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
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "neighbor_count": _count_non_bg4(grid, int(r), int(c), int(bg)),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "anchor": (int(anchor[0]), int(anchor[1])),
                        "pattern": np.asarray(pattern, dtype=int),
                    }
                )

    return rules


def _baseline_color_only_blob_output(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape
    rules = _discover_rules(g, bg=bg)

    legend_cells: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    first_rule_by_color: dict[int, dict] = {}
    for rule in rules:
        component = _component_containing(g, int(rule["source_color"]), rule["source_pos"])
        if component is not None:
            legend_cells.update(_component_set(component))
        legend_cells.add(rule["target_pos"])
        source_colors.add(int(rule["source_color"]))
        first_rule_by_color.setdefault(int(rule["source_color"]), rule)

    out = np.full_like(g, int(bg))
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_cells or color in source_colors or color == MARKER_COLOR:
            continue
        out[r, c] = color

    for source_color, rule in first_rule_by_color.items():
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        for component in _source_components(g, int(source_color), legend_cells):
            sr, sc = _component_anchor(component)
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local block-to-stencil rules, then apply them to matching source blobs.

    Input structure:
    - Black background with teal (8) stencil components, gray (5) support
      markers, multi-pixel same-color source blocks, and local legend blocks.
    - A rule example is a source block whose top-left cell touches a target
      pixel, with the same direction continuing to an anchor that touches the
      teal stencil. The source block anchor's non-background 4-neighbor count is
      part of the rule.

    Rule:
    - For each learned legend, find every non-legend 4-connected component of
      the same source color whose top-left cell has the same non-background
      4-neighbor count as the legend source cell.
    - Copy that legend's teal stencil once per matching block, aligning the
      legend anchor with the block's top-left cell, and recolor the copied
      stencil to the legend target color.
    - Remove the legend blocks, legend/support markers, teal stencils, and all
      source blocks after stamping.

    Key insight: The decisive object is not an isolated pixel anymore but a
    whole colored blob. Its top-left cell simultaneously provides the geometric
    placement anchor and the local context count that disambiguates same-color
    rules.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg)
    legend_cells: set[tuple[int, int]] = set()
    consumed_support: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    rule_by_key: dict[tuple[int, int], dict] = {}

    for rule in rules:
        source_color = int(rule["source_color"])
        source_pos = rule["source_pos"]
        target_pos = rule["target_pos"]
        source_colors.add(source_color)
        rule_by_key[(source_color, int(rule["neighbor_count"]))] = rule

        component = _component_containing(g, source_color, source_pos)
        if component is not None:
            component_cells = _component_set(component)
            legend_cells.update(component_cells)
        else:
            component_cells = {source_pos}
            legend_cells.add(source_pos)
        legend_cells.add(target_pos)

        for nr, nc in _non_bg_neighbors4(g, source_pos[0], source_pos[1], int(bg)):
            if (nr, nc) == target_pos or (nr, nc) in component_cells:
                continue
            consumed_support.add((int(nr), int(nc)))

    matched_blocks: list[tuple[dict, tuple[int, int]]] = []
    for source_color in source_colors:
        for component in _source_components(g, int(source_color), legend_cells):
            component_cells = _component_set(component)
            sr, sc = _component_anchor(component)
            count = _count_non_bg4(g, sr, sc, int(bg))
            rule = rule_by_key.get((int(source_color), int(count)))
            if rule is None:
                continue
            matched_blocks.append((rule, (int(sr), int(sc))))
            for nr, nc in _non_bg_neighbors4(g, sr, sc, int(bg)):
                if (nr, nc) in component_cells:
                    continue
                consumed_support.add((int(nr), int(nc)))

    out = np.full_like(g, int(bg))
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != PATTERN_COLOR))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_cells or (r, c) in consumed_support:
            continue
        if color in source_colors or color == MARKER_COLOR:
            continue
        out[r, c] = color

    for rule, (sr, sc) in matched_blocks:
        ar, ac = rule["anchor"]
        pattern = rule["pattern"]
        dr = sr - ar
        dc = sc - ac
        rows = pattern[:, 0] + dr
        cols = pattern[:, 1] + dc
        ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[ok], cols[ok]] = int(rule["target_color"])

    return out


def _build_rules(num_rules: int, area: int):
    max_s = min(STENCIL_MAX_CAP, max(2, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(1, area // STENCIL_MIN_AREA_DIV), max_s)
    block_max = min(SOURCE_BLOCK_MAX_CAP, max(SOURCE_BLOCK_MIN, area // SOURCE_BLOCK_AREA_DIV))

    shared_src = random.choice(SOURCE_COLORS)
    keys = [(shared_src, MATCH_COUNTS[0]), (shared_src, MATCH_COUNTS[1])]
    remaining = [
        (src, cnt)
        for src in SOURCE_COLORS
        for cnt in MATCH_COUNTS
        if (src, cnt) not in keys
    ]
    random.shuffle(remaining)
    keys.extend(remaining[: num_rules - 2])
    random.shuffle(keys)

    targets = TARGET_COLORS[:]
    random.shuffle(targets)

    rules: list[dict] = []
    used_colors = {MARKER_COLOR, PATTERN_COLOR}
    for i, (src_c, count) in enumerate(keys):
        tgt_c = int(targets[i])
        used_colors.update((int(src_c), int(tgt_c)))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rules.append(
            {
                "src": int(src_c),
                "tgt": tgt_c,
                "count": int(count),
                "rel": rel,
                "legend_shape": _make_source_shape(),
                "n_blocks": random.randint(SOURCE_BLOCK_MIN, block_max),
            }
        )
    return rules, used_colors


def _place_legends(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        shape = rule["legend_shape"]
        target_dirs = _eligible_target_dirs(shape)
        marker_needed = int(rule["count"]) - 2
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(target_dirs)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            source_cells = _translated_cells((sr, sc), shape)
            target_cell = (int(sr + dr), int(sc + dc))
            anchor_cell = (int(sr + 2 * dr), int(sc + 2 * dc))

            if any(not (0 <= r < h and 0 <= c < w) for r, c in source_cells):
                continue
            if not (0 <= target_cell[0] < h and 0 <= target_cell[1] < w):
                continue
            if not (0 <= anchor_cell[0] < h and 0 <= anchor_cell[1] < w):
                continue

            support_dirs = _available_support_dirs(shape, {(int(dr), int(dc))})
            if len(support_dirs) < marker_needed:
                continue
            chosen_support = random.sample(support_dirs, marker_needed)
            support_cells = {(int(sr + rr), int(sc + cc)) for rr, cc in chosen_support}
            if any(not (0 <= r < h and 0 <= c < w) for r, c in support_cells):
                continue

            teal_cells = _translated_cells(anchor_cell, rule["rel"])
            if any(not (0 <= r < h and 0 <= c < w) for r, c in teal_cells):
                continue

            all_cells = source_cells | {target_cell, anchor_cell} | support_cells | teal_cells
            if all_cells & occupied or all_cells & reserved:
                continue

            for r, c in source_cells:
                inp[r, c] = int(rule["src"])
            inp[target_cell] = int(rule["tgt"])
            for r, c in support_cells:
                inp[r, c] = MARKER_COLOR
            for r, c in teal_cells:
                inp[r, c] = PATTERN_COLOR

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            rule["legend"] = {
                "source": (int(sr), int(sc)),
                "target": target_cell,
                "anchor": anchor_cell,
                "cells": sorted(source_cells),
                "supports": sorted(support_cells),
            }
            placed = True
            break
        if not placed:
            return False
    return True


def _place_source_blocks(rules, inp, occupied, reserved):
    h, w = inp.shape
    for rule in rules:
        blocks = []
        marker_needed = int(rule["count"]) - 1
        target_count = int(rule["n_blocks"])
        for _ in range(target_count * SOURCE_BLOCK_RETRIES_PER_BLOCK):
            if len(blocks) >= target_count:
                break
            shape = _make_source_shape()
            support_dirs = _available_support_dirs(shape, set())
            if len(support_dirs) < marker_needed:
                continue
            chosen_support = random.sample(support_dirs, marker_needed)
            sr, sc = random.randint(0, h - 1), random.randint(0, w - 1)
            source_cells = _translated_cells((sr, sc), shape)
            support_cells = {(int(sr + rr), int(sc + cc)) for rr, cc in chosen_support}
            all_cells = source_cells | support_cells

            if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                continue
            if all_cells & occupied or all_cells & reserved:
                continue

            for r, c in source_cells:
                inp[r, c] = int(rule["src"])
            for r, c in support_cells:
                inp[r, c] = MARKER_COLOR

            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            blocks.append(
                {
                    "anchor": (int(sr), int(sc)),
                    "cells": sorted(source_cells),
                    "supports": sorted(support_cells),
                }
            )
        if len(blocks) < SOURCE_BLOCK_MIN:
            return False
        rule["blocks"] = blocks
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    choices = [c for c in SOURCE_COLORS + TARGET_COLORS if c not in used_colors]
    if not choices:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(choices) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, h - 1), random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = int(random.choice(choices))
            inp[r, c] = color
            placed.append((int(r), int(c), color))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h, w):
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        out[r, c] = int(color)
    for rule in rules:
        for block in rule["blocks"]:
            sr, sc = block["anchor"]
            for rr, cc in rule["rel"]:
                nr, nc = sr + rr, sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["tgt"])
    return out


def _variant_metrics(inp: np.ndarray) -> dict[str, int]:
    bg = get_bg(inp)
    rules = _discover_rules(inp, bg=bg)
    legend_cells: set[tuple[int, int]] = set()
    for rule in rules:
        component = _component_containing(inp, int(rule["source_color"]), rule["source_pos"])
        if component is not None:
            legend_cells.update(_component_set(component))
        legend_cells.add(rule["target_pos"])

    source_components = 0
    multi_cell_components = 0
    for color in {int(rule["source_color"]) for rule in rules}:
        for component in _source_components(inp, int(color), legend_cells):
            source_components += 1
            if len(component) > 1:
                multi_cell_components += 1

    repeated_source = len(rules) - len({int(rule["source_color"]) for rule in rules})
    unique_keys = len({(int(rule["source_color"]), int(rule["neighbor_count"])) for rule in rules})
    color_only_same = int(np.array_equal(main(inp), _baseline_color_only_blob_output(inp)))
    return {
        "rules": len(rules),
        "unique_keys": unique_keys,
        "repeated_source_rules": repeated_source,
        "source_components": source_components,
        "multi_cell_components": multi_cell_components,
        "color_only_same": color_only_same,
    }


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_source_blocks(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    metrics = _variant_metrics(inp)
    if metrics["rules"] != num_rules:
        return None
    if metrics["unique_keys"] != num_rules:
        return None
    if metrics["repeated_source_rules"] < 1:
        return None
    if metrics["source_components"] != metrics["multi_cell_components"]:
        return None
    if metrics["color_only_same"] != 0:
        return None

    output = _build_output(rules, bystander_pos, h, w)
    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {OUTER_RETRIES} attempts")
