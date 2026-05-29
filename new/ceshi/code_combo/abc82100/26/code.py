# Combo variant of abc82100 (index 26)
# combo_spec: D1/0+D2/1+D3/0
# Name: 形状匹配+训练对编码+多像素色块

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, find_components, get_bg


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    - The original task encoded each rule as a local source-pixel -> target-pixel
#      pair plus a nearby teal stencil, then applied that stencil to every other
#      pixel of the same source color.
#    - In this combo, the legend is no longer a local pair. Each rule is a teal
#      framed training pair: the left 3x3 board shows a multi-cell source sprite,
#      the right 3x3 board shows the target-colored output stencil produced from
#      that sprite.
#    - Free-standing sources are also multi-cell monochrome components, not pixels.
#      They are matched to rules by exact 4-connected shape equality, ignoring color.
#    - Concrete trace:
#      old: a blue pixel near a red pixel plus teal shape T means "all other blue
#      pixels become T in red".
#      new: a framed input board contains an L-like 3-cell sprite, and the framed
#      output board contains an orange stamp S. Elsewhere, every other L-like blob
#      becomes S in orange, even if one blob is green and another is yellow.
#    - The source anchor is the sprite's top-left occupied cell. The right training
#      board is drawn relative to that same anchor, so the solver must infer both
#      the shape class and the alignment rule before applying the transformation.
#
# b) Complexity calibration
#    - Source sprites are 3-5 cells, 4-connected, and forced to span at least a
#      2x2 bbox, so the input visibly differs from the original single-pixel task.
#    - Rule legends use compact 5x11 framed pairs like D2, keeping the instruction
#      format human-readable while still requiring analogy-style parsing.
#    - All rules share a small source-color palette in the application area. The
#      same color is deliberately reused across different source shapes, so color
#      equality is provably insufficient and D1's shape-matching change stays active.
#
# c) Implicit assumption audit against the original main()
#    - Original rule discovery scanned adjacent colored pixels and searched for teal
#      components nearby. That fails here because teal only appears as hollow frames.
#      The new solver must detect frame pairs and parse their inner boards.
#    - Original application iterated over `g == source_color`, assuming each source
#      object was a pixel and color identified the rule. Both assumptions break:
#      sources are multi-cell components and colors are intentionally ambiguous.
#    - Original preservation logic removed every pixel of any source color. That
#      would destroy unrelated shapes here. Preservation must be component-based,
#      and only components whose normalized shape matches a learned rule are consumed.
#
# d) Placement legality
#    - Training pairs are stacked vertically as non-overlapping 5x11 clusters. Their
#      teal frames are separated by at least a 4-neighbor moat from all free objects.
#    - Free source blobs are placed as non-touching 4-connected components, so the
#      solver recovers the intended connected components exactly.
#    - Future output stamps may overlap each other or clip at the border, but input
#      objects never overlap. Bystanders are excluded from future stamp footprints.
#
# e) Visual observability
#    - A human can read each rule from a visible input/output analogy panel, then
#      look for repeated source shapes elsewhere. Because multiple different shapes
#      share source colors, the rule cannot be inferred from color alone.
#    - The anchor is not a hidden numeric convention: it is the top-left occupied
#      cell of the source sprite, which is visually recoverable from the left panel.
#
# f) Visual cleanliness
#    - Legends live in a dedicated vertical strip, while the application objects are
#      separated by padding, so the framed instruction area and the free blobs do not
#      visually merge.
#    - Bystanders remain single pixels. This keeps unmatched leftovers easy to spot
#      and avoids accidental ambiguity with the rule-bearing multi-cell shapes.
#
# g) Color-change inferability
#    - Target colors are learned directly from the right training boards, not from a
#      free-floating target pixel. The number of target colors equals the number of
#      rules, so each shape class has one visible target example.
#    - Source colors are intentionally low-cardinality and reused. This gives repeated
#      support for the fact that shape, not color, determines which rule applies.
#
# h) New solver assumptions and generator guarantees
#    - Assumption: a rule legend is exactly two same-sized teal hollow frames with
#      one blank column between them. Guarantee: generator only emits that pattern.
#    - Assumption: each left frame contains exactly one monochrome 4-connected source
#      sprite. Guarantee: generator samples source shapes from a connected library.
#    - Assumption: each right frame contains exactly one monochrome output stencil.
#      Guarantee: generator samples one connected target pattern per rule.
#    - Assumption: each learned source shape is unique across rules. Guarantee:
#      generator samples source shapes without replacement.
#    - Assumption: the source anchor can be recovered as the source sprite's top-left
#      occupied cell. Guarantee: every sampled source shape includes its bbox corner
#      cell (0,0), so the anchor is stable under translation.


FRAME_COLOR = 8
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

INNER_SIZE = 3
FRAME_SIZE = INNER_SIZE + 2
PAIR_GAP = 1
CLUSTER_H = FRAME_SIZE
CLUSTER_W = FRAME_SIZE * 2 + PAIR_GAP

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {2: 11, 3: 16, 4: 20}
RULE_COUNT_WEIGHTS = [2, 2, 3, 3, 4]

SOURCE_BLOCK_MIN = 3
SOURCE_BLOCK_MAX_CAP = 6
SOURCE_BLOCK_AREA_DIV = 45
BYSTANDER_MAX = 3

SCATTERED_RETRIES_PER_OBJECT = 100
BYSTANDER_RETRIES = 100
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


def _is_connected(cells: set[tuple[int, int]], connectivity: int) -> bool:
    dirs = _DIRS4 if int(connectivity) == 4 else _DIRS8
    todo = [next(iter(cells))]
    seen = set()
    while todo:
        cur = todo.pop()
        if cur in seen:
            continue
        seen.add(cur)
        r, c = cur
        for dr, dc in dirs:
            nxt = (int(r + dr), int(c + dc))
            if nxt in cells and nxt not in seen:
                todo.append(nxt)
    return seen == cells


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    coords = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in coords)
    c0 = min(c for _, c in coords)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _shape_info(coords: np.ndarray) -> dict:
    coords = np.asarray(coords, dtype=int)
    rel = _normalize_coords(coords)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    return {
        "bbox": (r0, c0, r1, c1),
        "rel": rel,
        "coord_set": set((int(r), int(c)) for r, c in coords),
        "area": int(coords.shape[0]),
    }


def _translate(rel: tuple[tuple[int, int], ...], top: int, left: int) -> set[tuple[int, int]]:
    return {(int(top + r), int(left + c)) for r, c in rel}


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r + dr), int(c + dc)))


def _rect_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for r in range(r0, r0 + h) for c in range(c0, c0 + w)}


def _frame_border_cells(r0: int, c0: int, r1: int, c1: int) -> set[tuple[int, int]]:
    out = set()
    for c in range(c0, c1 + 1):
        out.add((int(r0), int(c)))
        out.add((int(r1), int(c)))
    for r in range(r0, r1 + 1):
        out.add((int(r), int(c0)))
        out.add((int(r), int(c1)))
    return out


def _sample_starts(total: int, block: int, count: int) -> list[int] | None:
    slack = int(total - block * count)
    if slack < 0:
        return None
    parts = [0] * (count + 1)
    for _ in range(slack):
        parts[random.randint(0, count)] += 1
    starts = []
    cur = parts[0]
    for i in range(count):
        starts.append(int(cur))
        cur += block + parts[i + 1]
    return starts


def _find_frames(grid: np.ndarray, frame_color: int = FRAME_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    candidates: list[dict] = []
    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if int(g[r0, c0]) != int(frame_color):
                continue
            for r1 in range(r0 + 2, h):
                if int(g[r1, c0]) != int(frame_color):
                    continue
                for c1 in range(c0 + 2, w):
                    if int(g[r0, c1]) != int(frame_color) or int(g[r1, c1]) != int(frame_color):
                        continue
                    if not (
                        np.all(g[r0, c0 : c1 + 1] == int(frame_color))
                        and np.all(g[r1, c0 : c1 + 1] == int(frame_color))
                        and np.all(g[r0 : r1 + 1, c0] == int(frame_color))
                        and np.all(g[r0 : r1 + 1, c1] == int(frame_color))
                    ):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == int(frame_color)):
                        continue
                    candidates.append(
                        {
                            "r0": int(r0),
                            "c0": int(c0),
                            "r1": int(r1),
                            "c1": int(c1),
                            "h": int(r1 - r0 + 1),
                            "w": int(c1 - c0 + 1),
                        }
                    )
    candidates.sort(key=lambda fr: (-(fr["h"] * fr["w"]), fr["r0"], fr["c0"]))
    picked: list[dict] = []
    used: set[tuple[int, int]] = set()
    for fr in candidates:
        area = _rect_cells(fr["r0"], fr["c0"], fr["h"], fr["w"])
        if area & used:
            continue
        fr["area_cells"] = area
        picked.append(fr)
        used.update(area)
    picked.sort(key=lambda fr: (fr["r0"], fr["c0"]))
    return picked


def _colored_components(grid: np.ndarray, bg: int, pattern_color: int = FRAME_COLOR) -> list[dict]:
    raw = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    comps: list[dict] = []
    for comp in raw:
        color = int(comp["color"])
        if color in (int(bg), int(pattern_color)):
            continue
        coords = np.asarray(comp["coords"], dtype=int)
        info = _shape_info(coords)
        comps.append(
            {
                "index": len(comps),
                "color": color,
                "coords": coords,
                "bbox": info["bbox"],
                "rel": info["rel"],
                "coord_set": info["coord_set"],
                "area": info["area"],
            }
        )
    return comps


def _discover_rules(grid: np.ndarray, bg: int, frame_color: int = FRAME_COLOR) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g, frame_color=frame_color)
    by_key = {(fr["r0"], fr["c0"], fr["r1"], fr["c1"]): fr for fr in frames}
    rules: list[dict] = []

    for fr in frames:
        right_key = (fr["r0"], fr["c1"] + PAIR_GAP + 1, fr["r1"], fr["c1"] + PAIR_GAP + fr["w"])
        right = by_key.get(right_key)
        if right is None or right["h"] != fr["h"] or right["w"] != fr["w"]:
            continue

        left_inner = g[fr["r0"] + 1 : fr["r1"], fr["c0"] + 1 : fr["c1"]]
        right_inner = g[right["r0"] + 1 : right["r1"], right["c0"] + 1 : right["c1"]]

        left_mask = (left_inner != int(bg)) & (left_inner != int(frame_color))
        left_colors = sorted(set(int(x) for x in left_inner[left_mask]))
        left_parts = components_mask(left_mask, connectivity=4)
        if len(left_colors) != 1 or len(left_parts) != 1:
            continue
        source_coords = np.asarray(left_parts[0], dtype=int)
        source_shape = _normalize_coords(source_coords)
        source_anchor_local = min((int(r), int(c)) for r, c in source_coords)
        source_anchor_rel = (
            int(source_anchor_local[0] - source_coords[:, 0].min()),
            int(source_anchor_local[1] - source_coords[:, 1].min()),
        )

        right_colors = sorted(set(int(x) for x in right_inner.ravel() if int(x) not in (int(bg), int(frame_color))))
        if len(right_colors) != 1:
            continue
        target_color = int(right_colors[0])
        target_coords = np.argwhere(right_inner == int(target_color))
        if target_coords.size == 0:
            continue
        output_rel = tuple(
            sorted((int(r - source_anchor_local[0]), int(c - source_anchor_local[1])) for r, c in target_coords)
        )

        legend_cells = set(fr["area_cells"]) | set(right["area_cells"]) | {
            (int(r), int(right["c0"] - 1)) for r in range(fr["r0"], fr["r1"] + 1)
        }
        rules.append(
            {
                "source_shape": source_shape,
                "source_anchor_rel": source_anchor_rel,
                "target_color": target_color,
                "output_rel": output_rel,
                "legend_cells": legend_cells,
            }
        )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn shape-indexed rewrite rules from framed training pairs, then stamp them onto matching source blobs.

    Input structure:
    - Black background with teal (8) hollow frames and several free-standing monochrome source blobs.
    - Each rule legend is a horizontal teal training pair: the left mini-board contains one multi-pixel
      source sprite, and the right mini-board contains the corresponding target-colored output stencil.

    Rule:
    - Parse each framed pair as an input/output example. Record the source sprite's exact 4-connected shape,
      its top-left occupied cell as the anchor, and the target stencil drawn relative to that anchor.
    - Find every other non-legend component whose normalized shape matches that learned source shape, even if
      its color is different from the legend color, and stamp the learned target stencil at that blob's anchor.
    - Remove all legend frames, legend contents, and matched source blobs. Preserve unrelated bystanders.

    Key insight: The legend teaches a transformation by analogy rather than by an adjacent color pair, and the
    reusable key is the source blob's geometry. Color is deliberately unreliable in the application area.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    h, w = g.shape

    rules = _discover_rules(g, bg=bg, frame_color=FRAME_COLOR)
    comps = _colored_components(g, bg=bg, pattern_color=FRAME_COLOR)

    legend_cells: set[tuple[int, int]] = set()
    shape_to_rules: dict[tuple[tuple[int, int], ...], list[dict]] = {}
    for rule in rules:
        legend_cells.update(rule["legend_cells"])
        shape_to_rules.setdefault(rule["source_shape"], []).append(rule)

    out = np.full_like(g, int(bg))

    for comp in comps:
        if comp["coord_set"] & legend_cells:
            continue

        candidates = shape_to_rules.get(comp["rel"], [])
        if len(candidates) == 1:
            rule = candidates[0]
            cr0, cc0, _cr1, _cc1 = comp["bbox"]
            anchor_r = int(cr0 + rule["source_anchor_rel"][0])
            anchor_c = int(cc0 + rule["source_anchor_rel"][1])
            rows = np.array([anchor_r + rr for rr, _ in rule["output_rel"]], dtype=int)
            cols = np.array([anchor_c + cc for _, cc in rule["output_rel"]], dtype=int)
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = int(rule["target_color"])
            continue

        for r, c in comp["coord_set"]:
            out[int(r), int(c)] = int(comp["color"])

    return out


def _build_source_library() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE) if (r, c) != (0, 0)]
    patterns: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1 << len(cells)):
        chosen = {(0, 0)}
        for i, pos in enumerate(cells):
            if (mask >> i) & 1:
                chosen.add(pos)
        if not (3 <= len(chosen) <= 5):
            continue
        if not _is_connected(chosen, connectivity=4):
            continue
        rows = {r for r, _ in chosen}
        cols = {c for _, c in chosen}
        if len(rows) < 2 or len(cols) < 2:
            continue
        rel = tuple(sorted((int(r), int(c)) for r, c in chosen))
        patterns.append(rel)
    patterns = sorted(set(patterns), key=lambda rel: (len(rel), rel))
    return patterns


def _build_output_library() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(INNER_SIZE) for c in range(INNER_SIZE) if (r, c) != (0, 0)]
    patterns: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1 << len(cells)):
        chosen = {(0, 0)}
        for i, pos in enumerate(cells):
            if (mask >> i) & 1:
                chosen.add(pos)
        if not (2 <= len(chosen) <= 5):
            continue
        if not _is_connected(chosen, connectivity=8):
            continue
        rel = tuple(sorted((int(r), int(c)) for r, c in chosen))
        patterns.append(rel)
    patterns = sorted(set(patterns), key=lambda rel: (len(rel), rel))
    return patterns


_SOURCE_LIBRARY = _build_source_library()
_OUTPUT_LIBRARY = _build_output_library()


def _build_rules(num_rules: int, area: int):
    target_colors = random.sample(NON_BG_NON_TEAL_COLORS, k=num_rules)
    remaining = [c for c in NON_BG_NON_TEAL_COLORS if c not in target_colors]
    palette_size = min(len(remaining), 2 if num_rules <= 2 else 3)
    source_palette = random.sample(remaining, k=palette_size) if palette_size > 0 else [random.choice(NON_BG_NON_TEAL_COLORS)]
    source_shapes = random.sample(_SOURCE_LIBRARY, k=num_rules)
    output_shapes = random.sample(_OUTPUT_LIBRARY, k=num_rules)

    block_budget_cap = max(SOURCE_BLOCK_MIN, 5 - max(0, num_rules - 2))
    block_max = min(
        SOURCE_BLOCK_MAX_CAP,
        max(SOURCE_BLOCK_MIN, area // SOURCE_BLOCK_AREA_DIV),
        block_budget_cap,
    )
    rules = []
    for i in range(num_rules):
        rules.append(
            {
                "legend_color": int(source_palette[i % len(source_palette)]),
                "target_color": int(target_colors[i]),
                "source_rel": source_shapes[i],
                "output_rel": output_shapes[i],
                "n_blocks": random.randint(SOURCE_BLOCK_MIN, block_max),
                "source_palette": [int(c) for c in source_palette],
            }
        )
    used_colors = set(target_colors) | set(source_palette)
    return rules, used_colors


def _draw_legend(rule: dict, inp: np.ndarray, top: int, left: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    border_cells = set()
    occupied_cells = set()

    left_r0, left_c0 = int(top), int(left)
    left_r1, left_c1 = int(top + FRAME_SIZE - 1), int(left + FRAME_SIZE - 1)
    right_r0, right_c0 = int(top), int(left + FRAME_SIZE + PAIR_GAP)
    right_r1, right_c1 = int(right_r0 + FRAME_SIZE - 1), int(right_c0 + FRAME_SIZE - 1)

    for r, c in _frame_border_cells(left_r0, left_c0, left_r1, left_c1):
        inp[r, c] = int(FRAME_COLOR)
        border_cells.add((r, c))
        occupied_cells.add((r, c))
    for r, c in _frame_border_cells(right_r0, right_c0, right_r1, right_c1):
        inp[r, c] = int(FRAME_COLOR)
        border_cells.add((r, c))
        occupied_cells.add((r, c))

    for rr, cc in rule["source_rel"]:
        nr = int(left_r0 + 1 + rr)
        nc = int(left_c0 + 1 + cc)
        inp[nr, nc] = int(rule["legend_color"])
        occupied_cells.add((nr, nc))

    for rr, cc in rule["output_rel"]:
        nr = int(right_r0 + 1 + rr)
        nc = int(right_c0 + 1 + cc)
        inp[nr, nc] = int(rule["target_color"])
        occupied_cells.add((nr, nc))

    cluster_cells = _rect_cells(top, left, CLUSTER_H, CLUSTER_W)
    gap_col = int(left + FRAME_SIZE)
    cluster_cells.update((int(r), gap_col) for r in range(top, top + CLUSTER_H))
    return border_cells, cluster_cells | occupied_cells


def _place_legends(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    if h < CLUSTER_H or w < CLUSTER_W:
        return False
    starts = _sample_starts(h, CLUSTER_H, len(rules))
    if starts is None:
        return False
    left = random.randint(0, w - CLUSTER_W)
    for rule, top in zip(rules, starts):
        cluster_cells = _rect_cells(top, left, CLUSTER_H, CLUSTER_W)
        if cluster_cells & occupied:
            return False
        border_cells, all_cells = _draw_legend(rule, inp, top, left)
        occupied.update(all_cells)
        _reserve_4neighbors(reserved, border_cells)
        rule["legend_top_left"] = (int(top), int(left))
    return True


def _scatter_color_sequence(rule: dict) -> list[int]:
    palette = list(rule["source_palette"])
    n = int(rule["n_blocks"])
    seq = [int(palette[0]) for _ in range(n)]
    if len(palette) >= 2 and n >= 2:
        seq[1] = int(palette[1])
    for i in range(2, n):
        seq[i] = int(random.choice(palette))
    return seq


def _place_source_blocks(rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]) -> bool:
    h, w = inp.shape
    for rule in rules:
        blocks = []
        colors = _scatter_color_sequence(rule)
        for color in colors:
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_OBJECT):
                top = random.randint(0, h - 1)
                left = random.randint(0, w - 1)
                cells = _translate(rule["source_rel"], top, left)
                if min(r for r, _ in cells) < 0 or max(r for r, _ in cells) >= h:
                    continue
                if min(c for _, c in cells) < 0 or max(c for _, c in cells) >= w:
                    continue
                if cells & (occupied | reserved):
                    continue
                for r, c in cells:
                    inp[r, c] = int(color)
                occupied.update(cells)
                _reserve_4neighbors(reserved, cells)
                blocks.append({"anchor": (int(top), int(left)), "cells": sorted(cells), "color": int(color)})
                placed = True
                break
            if not placed:
                return False
        rule["blocks"] = blocks
    return True


def _stamp_footprint(rules, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for rule in rules:
        for block in rule.get("blocks", []):
            anchor_r, anchor_c = block["anchor"]
            for rr, cc in rule["output_rel"]:
                nr = int(anchor_r + rr)
                nc = int(anchor_c + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    cells.add((nr, nc))
    return cells


def _place_bystanders(used_colors, rules, inp: np.ndarray, occupied: set[tuple[int, int]], reserved: set[tuple[int, int]]):
    h, w = inp.shape
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    forbidden = _stamp_footprint(rules, h, w)
    placed = []
    for _ in range(random.randint(0, BYSTANDER_MAX)):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved or (r, c) in forbidden:
                continue
            color = int(random.choice(bystander_colors))
            inp[r, c] = color
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            placed.append((int(r), int(c), color))
            break
    return placed


def _build_output(rules, bystanders, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    for r, c, color in bystanders:
        out[int(r), int(c)] = int(color)
    for rule in rules:
        for block in rule["blocks"]:
            anchor_r, anchor_c = block["anchor"]
            for rr, cc in rule["output_rel"]:
                nr = int(anchor_r + rr)
                nc = int(anchor_c + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    out[nr, nc] = int(rule["target_color"])
    return out


def _gen_attempt():
    num_rules = int(random.choice(RULE_COUNT_WEIGHTS))
    min_dim = int(MIN_DIM_BY_RULES[num_rules])
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
    bystanders = _place_bystanders(used_colors, rules, inp, occupied, reserved)

    expected = _build_output(rules, bystanders, h, w)
    try:
        if np.array_equal(main(inp), expected):
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
