# Combo variant of abc82100 (index 32)
# combo_spec: D7/1+D8/0+D10/0
# Name: 规则移至边缘+紧凑裁剪+交错分布

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

try:
    from arc_utils import bbox_coords, components_mask, get_bg
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "solvers"))
    from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis for combo D7/1 + D8/0 + D10/0
# ------------------------------------------------
# concrete_behavior_trace:
# - Original abc82100:
#   1. Detect a local legend: source pixel next to target pixel, then one more
#      step in the same direction reaches an anchor touching a teal stencil.
#   2. Stamp that stencil onto every other source-colored pixel, recolored to
#      the legend's target color.
#   3. Delete the legend and teal stencil, keep same canvas size.
# - This combo:
#   1. Detection is still entirely local, but legends are mixed into the same
#      region as the ordinary source pixels, so the solver cannot rely on a
#      separate "legend corner".
#   2. The painted content is formed exactly as before from the scattered
#      source pixels and preserved bystanders.
#   3. Instead of erasing the legends forever, rebuild each legend as a compact
#      reference block and move those blocks to the bottom edge of the output.
#   4. Tight-crop the combined result (painted content + moved edge legends).
# - Small example:
#   Suppose a blue->red legend with a 3-cell teal shape sits near the middle of
#   the input, and lone blue pixels are scattered around it. The old task would
#   replace the lone blue pixels with red copies of the teal shape and leave the
#   old legend area blank on an input-sized board. The combo variant does the
#   same painting, but then packs the original blue->red+teal legend onto the
#   bottom edge and returns only the smallest bounding rectangle containing both
#   the painted content and that edge legend strip.
#
# complexity_calibration:
# - The rule itself is unchanged, so the task is still inferable from 2-4
#   examples. The added difficulty is a short reasoning chain:
#   separate mixed-in legends from ordinary source pixels -> apply the stencil
#   substitution -> notice legends are preserved but relocated to an edge ->
#   predict the post-relocation crop.
# - To keep this readable, each rule still has a unique source color, bystanders
#   never reuse a source color, and the moved edge legends keep the exact local
#   geometry of source/target/stencil. This avoids ambiguous recoloring.
# - Stencils are kept compact so the bottom legend strip remains visually neat
#   and typically fits in one wrapped band near the bottom-right.
#
# implicit_assumptions_audited:
# - In _discover_rules, every adjacent non-bg/non-teal color pair whose forward
#   continuation sees teal is treated as a rule. Therefore the generator must
#   keep a 4-neighbor exclusion halo around objects and keep teal components
#   8-disconnected, or false legends would appear.
# - In the original main(), `out = full_like(input)` assumed same-size output.
#   That assumption is invalid here: same-size is used only as an intermediate
#   canvas, then moved legends are added on the bottom edge and the result is
#   cropped.
# - The original preservation rule "drop any non-legend pixel whose color is a
#   learned source color" still holds. Therefore bystanders must continue to use
#   colors outside all learned source colors.
# - New assumption introduced here:
#   moved legends are packed into a bottom-edge gallery. The generator therefore
#   reserves a bottom band in the transformed content area so stamps do not
#   collide with that gallery.
#
# placement_legality:
# - Input placement uses a 4-neighbor blocked set for every occupied cell. This
#   prevents accidental source-target pairs between unrelated objects.
# - Teal cells also reserve their 8-neighborhood to stop separate stencils from
#   merging into one component or becoming simultaneously reachable from one
#   anchor.
# - Ordinary source pixels are placed only where their recolored stencil stays
#   fully inside an interior content window above the reserved bottom legend
#   band, so the final crop is smaller than the input and the output gallery has
#   room.
#
# visual_observability_and_neatness:
# - Interleaving is made visible by forcing nearby lone source pixels around
#   every legend, often from different rules on a second pass.
# - The moved legends are rendered as intact mini-blocks on the bottom edge with
#   1-cell spacing, so a human can recognize that the same local examples from
#   the input have reappeared as a reference strip instead of being deleted.
# - Tight crop is visible because transformed content is kept away from the top
#   and left borders, while the legend strip is anchored to the bottom-right.
#
# new_main_assumptions_checked_against_generator:
# - main() assumes legends can be packed into the bottom edge of the input-sized
#   intermediate canvas. generate_input() computes that packing up front and
#   rejects samples that do not fit.
# - main() assumes moved legends do not overwrite painted content. The generator
#   guarantees this by reserving the output bottom band and only placing ordinary
#   source pixels whose stamped results stay above it.
# - main() sorts moved legends by their input source positions (row-major), a
#   human-visible property. The generator places legends in increasing row bands
#   so that this ordering is stable and predictable.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(grid != int(bg))
    if nz.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    r0, c0, r1, c1 = bbox_coords(nz)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], dtype=int)


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
                    }
                )

    return rules


def _sort_rules_by_source_pos(rules: list[dict]) -> list[dict]:
    return sorted(
        rules,
        key=lambda rule: (
            int(rule["source_pos"][0]),
            int(rule["source_pos"][1]),
            int(rule["target_pos"][0]),
            int(rule["target_pos"][1]),
        ),
    )


def _legend_pixels_from_rules(rules: list[dict]) -> tuple[set[tuple[int, int]], set[int]]:
    legend_pixels: set[tuple[int, int]] = set()
    source_colors: set[int] = set()
    for rule in rules:
        legend_pixels.add((int(rule["source_pos"][0]), int(rule["source_pos"][1])))
        legend_pixels.add((int(rule["target_pos"][0]), int(rule["target_pos"][1])))
        source_colors.add(int(rule["source_color"]))
    return legend_pixels, source_colors


def _render_content_without_legends(grid: np.ndarray, rules: list[dict], bg: int, pattern_color: int = 8) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legend_pixels, source_colors = _legend_pixels_from_rules(rules)
    out = np.full((h, w), int(bg), dtype=int)

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != int(pattern_color)))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        ar, ac = int(rule["anchor"][0]), int(rule["anchor"][1])
        pattern = np.asarray(rule["pattern"], dtype=int)
        source_color = int(rule["source_color"])
        target_color = int(rule["target_color"])
        for sr, sc in map(tuple, np.argwhere(g == source_color)):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            dr = sr - ar
            dc = sc - ac
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = target_color

    return out


def _extract_legend_block(rule: dict, bg: int, pattern_color: int = 8) -> np.ndarray:
    source_pos = (int(rule["source_pos"][0]), int(rule["source_pos"][1]))
    target_pos = (int(rule["target_pos"][0]), int(rule["target_pos"][1]))
    anchor = (int(rule["anchor"][0]), int(rule["anchor"][1]))
    pattern = np.asarray(rule["pattern"], dtype=int)
    coords = np.vstack(
        [
            np.array([source_pos, target_pos, anchor], dtype=int),
            pattern.astype(int),
        ]
    )
    r0, c0, r1, c1 = bbox_coords(coords)
    block = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    block[int(source_pos[0] - r0), int(source_pos[1] - c0)] = int(rule["source_color"])
    block[int(target_pos[0] - r0), int(target_pos[1] - c0)] = int(rule["target_color"])
    for pr, pc in pattern:
        block[int(pr - r0), int(pc - c0)] = int(pattern_color)
    return block


def _pack_blocks_bottom_right(
    blocks: list[np.ndarray],
    h: int,
    w: int,
    *,
    gap: int = 1,
    row_gap: int = 1,
) -> dict | None:
    rows: list[tuple[list[dict], int]] = []
    current: list[dict] = []
    cursor = int(w)
    row_height = 0

    for idx, block in enumerate(blocks):
        bh, bw = map(int, block.shape)
        need = bw if not current else gap + bw
        if current and cursor - need < 0:
            rows.append((current, int(row_height)))
            current = []
            cursor = int(w)
            row_height = 0
            need = bw
        if cursor - need < 0:
            return None
        x0 = int(cursor - bw)
        current.append({"index": int(idx), "x0": int(x0), "block": block})
        cursor = int(x0 - gap)
        row_height = max(int(row_height), int(bh))

    if current:
        rows.append((current, int(row_height)))

    total_height = sum(row_h for _entries, row_h in rows) + row_gap * max(0, len(rows) - 1)
    if total_height > h:
        return None

    placements: list[tuple[int, int] | None] = [None] * len(blocks)
    y_bottom = int(h)
    for entries, row_h in rows:
        for entry in entries:
            block = entry["block"]
            bh, _bw = map(int, block.shape)
            placements[int(entry["index"])] = (int(y_bottom - bh), int(entry["x0"]))
        y_bottom = int(y_bottom - row_h - row_gap)

    return {
        "placements": [pos for pos in placements if pos is not None],
        "band_height": int(total_height),
        "row_count": int(len(rows)),
    }


def _compose_uncropped_output(content: np.ndarray, rules: list[dict], bg: int, pattern_color: int = 8) -> np.ndarray:
    out = np.array(content, dtype=int)
    h, w = out.shape
    ordered_rules = _sort_rules_by_source_pos(rules)
    blocks = [_extract_legend_block(rule, bg=int(bg), pattern_color=int(pattern_color)) for rule in ordered_rules]
    packed = _pack_blocks_bottom_right(blocks, h, w)
    if packed is None:
        return out
    for block, (r0, c0) in zip(blocks, packed["placements"]):
        bh, bw = map(int, block.shape)
        sub = out[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw]
        mask = block != int(bg)
        sub[mask] = block[mask]
        out[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw] = sub
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn mixed-in source-target-stencil legends, move those legends to an edge, then crop.

    Input structure:
    - A black-background grid contains isolated colored pixels and teal (8) stencil components.
    - Rule examples are interleaved with the ordinary source pixels instead of living in one
      dedicated legend zone.
    - A valid rule example is still a 4-neighbor source->target color pair whose forward
      continuation reaches an anchor touching that rule's teal stencil.

    Rule:
    - For each discovered rule, stamp its teal stencil onto every other pixel of the same
      source color, recoloring the stencil to the rule's target color.
    - Preserve unrelated bystander pixels, but do not leave the original legends in place.
    - Rebuild each legend as a compact reference block and pack those blocks onto the bottom
      edge of the output canvas, then tight-crop around all non-background output pixels.

    Key insight: The local cue for learning a rule is unchanged even though legends are mixed
    among the source pixels. After applying the usual stencil substitution, the extra step is
    noticing that the legends reappear as an edge gallery rather than being erased.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    rules = _sort_rules_by_source_pos(_discover_rules(g, bg=int(bg), pattern_color=8))
    content = _render_content_without_legends(g, rules, bg=int(bg), pattern_color=8)
    combined = _compose_uncropped_output(content, rules, bg=int(bg), pattern_color=8)
    return _tight_crop_non_bg(combined, bg=int(bg))


# --------------- generator ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 11, 2: 13, 3: 16, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MIN_PIXELS = 2
STENCIL_MAX_CAP = 7
STENCIL_MIN_AREA_DIV = 120
STENCIL_MAX_AREA_DIV = 40
STENCIL_MAX_BBOX = 4
STENCIL_RETRIES = 120

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 7
SCATTERED_AREA_DIV = 32

BYSTANDER_MAX = 4

CONTENT_PAD_MIN = 1
CONTENT_PAD_MAX = 3
CONTENT_LEGEND_GAP = 1
PAD_RETRIES = 100
MIN_CONTENT_HEIGHT = 5
MIN_CONTENT_WIDTH = 6

LEGEND_RETRIES = 700
OUTER_RETRIES = 500

NEARBY_SOURCE_PASSES = 2
NEARBY_RING_MIN = 2
NEARBY_RING_MAX = 4
LEGEND_RING_REQUIREMENT = 3

BLOCK_GAP = 1
BLOCK_ROW_GAP = 1

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _reserve_neighbors(blocked: set[tuple[int, int]], cells, dirs) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        blocked.add((r, c))
        for dr, dc in dirs:
            blocked.add((int(r + dr), int(c + dc)))


def _make_stencil(min_pixels: int, max_pixels: int):
    for _ in range(STENCIL_RETRIES):
        n = random.randint(min_pixels, max_pixels)
        coords = {(0, 0)}
        for _j in range(n - 1):
            frontier = set()
            for r, c in coords:
                for dr, dc in _DIRS8:
                    p = (int(r + dr), int(c + dc))
                    if p not in coords:
                        frontier.add(p)
            coords.add(random.choice(list(frontier)))
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        if max(rows) - min(rows) + 1 > STENCIL_MAX_BBOX:
            continue
        if max(cols) - min(cols) + 1 > STENCIL_MAX_BBOX:
            continue
        coords_list = sorted(coords)
        return coords_list, random.choice(coords_list)
    return [(0, 0), (0, 1), (1, 1)], (0, 1)


def _rel_bounds(rel: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _c in rel]
    cs = [int(c) for _r, c in rel]
    return min(rs), max(rs), min(cs), max(cs)


def _make_local_legend_block(rule: dict, bg: int = 0, pattern_color: int = 8) -> np.ndarray:
    dr, dc = rule["dir"]
    source = (0, 0)
    target = (int(dr), int(dc))
    anchor = (int(2 * dr), int(2 * dc))
    teal = [(int(anchor[0] + rr), int(anchor[1] + cc)) for rr, cc in rule["rel"]]
    coords = np.array([source, target, anchor, *teal], dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    block = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    block[int(source[0] - r0), int(source[1] - c0)] = int(rule["src"])
    block[int(target[0] - r0), int(target[1] - c0)] = int(rule["tgt"])
    for tr, tc in teal:
        block[int(tr - r0), int(tc - c0)] = int(pattern_color)
    return block


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    max_s = min(STENCIL_MAX_CAP, max(STENCIL_MIN_PIXELS, area // STENCIL_MAX_AREA_DIV))
    min_s = min(max(STENCIL_MIN_PIXELS, area // STENCIL_MIN_AREA_DIV), max_s)
    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used_colors = set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used_colors.update((src_c, tgt_c))
        stencil, anchor = _make_stencil(min_s, max_s)
        rel = [(int(r - anchor[0]), int(c - anchor[1])) for r, c in stencil]
        rule = {
            "src": int(src_c),
            "tgt": int(tgt_c),
            "rel": rel,
            "bounds": _rel_bounds(rel),
            "dir": random.choice(_DIRS4),
            "n_sc": random.randint(SCATTERED_MIN, sc_max),
            "remaining": 0,
            "scattered": [],
        }
        rule["legend_block"] = _make_local_legend_block(rule)
        rules.append(rule)
    return rules, used_colors


def _choose_content_padding(rules, h: int, w: int, bottom_pad: int):
    if bottom_pad >= h:
        return None
    row_cap = max(CONTENT_PAD_MIN, min(CONTENT_PAD_MAX, max(1, (h - bottom_pad - 1) // 3)))
    col_cap = max(CONTENT_PAD_MIN, min(CONTENT_PAD_MAX, max(1, (w - 2) // 3)))
    for _ in range(PAD_RETRIES):
        top = random.randint(CONTENT_PAD_MIN, row_cap)
        left = random.randint(CONTENT_PAD_MIN, col_cap)
        right = random.randint(CONTENT_PAD_MIN, col_cap)
        if top + bottom_pad >= h or left + right >= w:
            continue
        if h - top - bottom_pad < MIN_CONTENT_HEIGHT:
            continue
        if w - left - right < MIN_CONTENT_WIDTH:
            continue
        ok = True
        for rule in rules:
            min_r, max_r, min_c, max_c = rule["bounds"]
            if top - min_r > (h - 1 - bottom_pad - max_r):
                ok = False
                break
            if left - min_c > (w - 1 - right - max_c):
                ok = False
                break
        if ok:
            return {"top": int(top), "bottom": int(bottom_pad), "left": int(left), "right": int(right)}
    return None


def _zone_bounds(pads: dict, h: int, w: int) -> tuple[int, int, int, int]:
    return int(pads["top"]), int(h - 1 - pads["bottom"]), int(pads["left"]), int(w - 1 - pads["right"])


def _valid_source_box(rule: dict, pads: dict, h: int, w: int) -> tuple[int, int, int, int] | None:
    min_r, max_r, min_c, max_c = rule["bounds"]
    r0 = int(pads["top"] - min_r)
    r1 = int(h - 1 - pads["bottom"] - max_r)
    c0 = int(pads["left"] - min_c)
    c1 = int(w - 1 - pads["right"] - max_c)
    if r0 > r1 or c0 > c1:
        return None
    return r0, r1, c0, c1


def _legend_layout(rule: dict, sr: int, sc: int, h: int, w: int, zone: tuple[int, int, int, int]):
    dr, dc = rule["dir"]
    tr, tc = int(sr + dr), int(sc + dc)
    ar, ac = int(sr + 2 * dr), int(sc + 2 * dc)
    zt, zb, zl, zr = zone
    if not (0 <= tr < h and 0 <= tc < w and 0 <= ar < h and 0 <= ac < w):
        return None

    teal_cells = set()
    for rr, cc in rule["rel"]:
        nr, nc = int(ar + rr), int(ac + cc)
        if not (0 <= nr < h and 0 <= nc < w):
            return None
        if not (zt <= nr <= zb and zl <= nc <= zr):
            return None
        teal_cells.add((nr, nc))

    if not (zt <= sr <= zb and zl <= sc <= zr and zt <= tr <= zb and zl <= tc <= zr):
        return None

    visible_cells = {(int(sr), int(sc)), (int(tr), int(tc))} | teal_cells
    all_cells = visible_cells | {(int(ar), int(ac))}
    return {
        "source_pos": (int(sr), int(sc)),
        "target_pos": (int(tr), int(tc)),
        "anchor": (int(ar), int(ac)),
        "teal_cells": teal_cells,
        "visible_cells": visible_cells,
        "all_cells": all_cells,
    }


def _ring_candidates(
    host_cells,
    zone: tuple[int, int, int, int],
    occupied: set[tuple[int, int]],
    blocked4: set[tuple[int, int]],
    valid_box: tuple[int, int, int, int] | None = None,
):
    zt, zb, zl, zr = zone
    rows = [r for r, _c in host_cells]
    cols = [c for _r, c in host_cells]
    candidates = []
    for r in range(max(zt, min(rows) - NEARBY_RING_MAX), min(zb, max(rows) + NEARBY_RING_MAX) + 1):
        for c in range(max(zl, min(cols) - NEARBY_RING_MAX), min(zr, max(cols) + NEARBY_RING_MAX) + 1):
            pos = (int(r), int(c))
            if pos in occupied or pos in blocked4:
                continue
            if valid_box is not None:
                r0, r1, c0, c1 = valid_box
                if not (r0 <= r <= r1 and c0 <= c <= c1):
                    continue
            dist = min(_chebyshev(pos, cell) for cell in host_cells)
            if NEARBY_RING_MIN <= dist <= NEARBY_RING_MAX:
                candidates.append(pos)
    return candidates


def _commit_legend(rule: dict, layout: dict, inp, occupied, blocked4, blocked_teal8) -> None:
    sr, sc = layout["source_pos"]
    tr, tc = layout["target_pos"]
    inp[int(sr), int(sc)] = int(rule["src"])
    inp[int(tr), int(tc)] = int(rule["tgt"])
    for r, c in layout["teal_cells"]:
        inp[int(r), int(c)] = 8

    occupied.update(layout["all_cells"])
    _reserve_neighbors(blocked4, layout["all_cells"], _DIRS4)
    _reserve_neighbors(blocked_teal8, layout["teal_cells"], _DIRS8)
    rule["legend"] = layout
    rule["remaining"] = int(rule["n_sc"])
    rule["scattered"] = []


def _place_legends(rules, inp, occupied, blocked4, blocked_teal8, pads):
    h, w = inp.shape
    zone = _zone_bounds(pads, h, w)
    zt, zb, zl, zr = zone
    usable_h = zb - zt + 1
    placed_layouts: list[dict] = []

    for i, rule in enumerate(rules):
        valid_box = _valid_source_box(rule, pads, h, w)
        if valid_box is None:
            return False
        band_r0 = int(zt + (i * usable_h) // len(rules))
        band_r1 = int(zt + ((i + 1) * usable_h) // len(rules) - 1)
        band_r1 = max(band_r0, min(band_r1, zb))

        best_layout = None
        best_score = None
        for _ in range(LEGEND_RETRIES):
            sr = random.randint(band_r0, band_r1)
            sc = random.randint(zl, zr)
            layout = _legend_layout(rule, sr, sc, h, w, zone)
            if layout is None:
                continue
            if layout["all_cells"] & occupied:
                continue
            if layout["all_cells"] & blocked4:
                continue
            if layout["teal_cells"] & blocked_teal8:
                continue

            ring = _ring_candidates(layout["all_cells"], zone, occupied, blocked4, valid_box=valid_box)
            if len(ring) < LEGEND_RING_REQUIREMENT:
                continue

            anchor = layout["anchor"]
            spread = min(
                (_chebyshev(anchor, other["anchor"]) for other in placed_layouts),
                default=max(h, w),
            )
            openness = len(ring)
            score = (spread, openness, -abs(sr - (band_r0 + band_r1) // 2))
            if best_score is None or score > best_score:
                best_score = score
                best_layout = layout

        if best_layout is None:
            return False

        _commit_legend(rule, best_layout, inp, occupied, blocked4, blocked_teal8)
        placed_layouts.append(best_layout)

    return True


def _place_source_pixel(rule: dict, pos: tuple[int, int], inp, occupied, blocked4) -> None:
    r, c = pos
    inp[int(r), int(c)] = int(rule["src"])
    rule["scattered"].append((int(r), int(c)))
    rule["remaining"] -= 1
    occupied.add((int(r), int(c)))
    _reserve_neighbors(blocked4, [(int(r), int(c))], _DIRS4)


def _choose_rule_index(rules, preferred_indices):
    candidates = [i for i in preferred_indices if rules[i]["remaining"] > 0]
    if not candidates:
        candidates = [i for i, rule in enumerate(rules) if rule["remaining"] > 0]
    if not candidates:
        return None
    max_remaining = max(rules[i]["remaining"] for i in candidates)
    best = [i for i in candidates if rules[i]["remaining"] == max_remaining]
    return int(random.choice(best))


def _place_nearby_sources(rules, inp, occupied, blocked4, pads):
    h, w = inp.shape
    zone = _zone_bounds(pads, h, w)
    valid_boxes = [_valid_source_box(rule, pads, h, w) for rule in rules]
    if any(box is None for box in valid_boxes):
        return False

    legend_indices = list(range(len(rules)))
    random.shuffle(legend_indices)
    for pass_id in range(NEARBY_SOURCE_PASSES):
        for host_idx in legend_indices:
            if pass_id == 0:
                preferred = [host_idx]
            else:
                preferred = [i for i in legend_indices if i != host_idx] + [host_idx]
            target_idx = _choose_rule_index(rules, preferred)
            if target_idx is None:
                return False
            candidates = _ring_candidates(
                rules[host_idx]["legend"]["all_cells"],
                zone,
                occupied,
                blocked4,
                valid_box=valid_boxes[target_idx],
            )
            random.shuffle(candidates)
            if not candidates:
                return False
            _place_source_pixel(rules[target_idx], candidates[0], inp, occupied, blocked4)
    return True


def _place_remaining_sources(rules, inp, occupied, blocked4, pads):
    h, w = inp.shape
    for rule in rules:
        box = _valid_source_box(rule, pads, h, w)
        if box is None:
            return False
        r0, r1, c0, c1 = box
        candidates = [(int(r), int(c)) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        random.shuffle(candidates)
        idx = 0
        while rule["remaining"] > 0 and idx < len(candidates):
            pos = candidates[idx]
            idx += 1
            if pos in occupied or pos in blocked4:
                continue
            _place_source_pixel(rule, pos, inp, occupied, blocked4)
        if rule["remaining"] > 0 or len(rule["scattered"]) < SCATTERED_MIN:
            return False
    return True


def _place_bystanders(used_colors, inp, occupied, blocked4, pads):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []

    zt, zb, zl, zr = _zone_bounds(pads, *inp.shape)
    cells = [(int(r), int(c)) for r in range(zt, zb + 1) for c in range(zl, zr + 1)]
    random.shuffle(cells)
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for r, c in cells:
        if len(placed) >= n:
            break
        if (r, c) in occupied or (r, c) in blocked4:
            continue
        color = int(random.choice(bystander_colors))
        inp[int(r), int(c)] = color
        placed.append((int(r), int(c), color))
        occupied.add((int(r), int(c)))
        _reserve_neighbors(blocked4, [(int(r), int(c))], _DIRS4)
    return placed


def _build_expected_output(rules, bystander_pos, h: int, w: int):
    content = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        content[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc in rule["rel"]:
                nr, nc = int(sr + rr), int(sc + cc)
                if 0 <= nr < h and 0 <= nc < w:
                    content[nr, nc] = int(rule["tgt"])

    ordered_rules = sorted(rules, key=lambda rule: tuple(rule["legend"]["source_pos"]))
    blocks = [rule["legend_block"] for rule in ordered_rules]
    packed = _pack_blocks_bottom_right(blocks, h, w, gap=BLOCK_GAP, row_gap=BLOCK_ROW_GAP)
    if packed is None:
        return None

    out = np.array(content, dtype=int)
    for block, (r0, c0) in zip(blocks, packed["placements"]):
        bh, bw = map(int, block.shape)
        sub = out[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw]
        mask = block != 0
        sub[mask] = block[mask]
        out[int(r0) : int(r0) + bh, int(c0) : int(c0) + bw] = sub
    return _tight_crop_non_bg(out, 0)


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    packed = _pack_blocks_bottom_right(
        [rule["legend_block"] for rule in rules],
        h,
        w,
        gap=BLOCK_GAP,
        row_gap=BLOCK_ROW_GAP,
    )
    if packed is None:
        return None

    pads = _choose_content_padding(rules, h, w, bottom_pad=int(packed["band_height"]) + CONTENT_LEGEND_GAP)
    if pads is None:
        return None

    inp = np.zeros((h, w), dtype=int)
    occupied: set[tuple[int, int]] = set()
    blocked4: set[tuple[int, int]] = set()
    blocked_teal8: set[tuple[int, int]] = set()

    if not _place_legends(rules, inp, occupied, blocked4, blocked_teal8, pads):
        return None
    if [tuple(rule["legend"]["source_pos"]) for rule in rules] != sorted(
        (tuple(rule["legend"]["source_pos"]) for rule in rules)
    ):
        return None
    if not _place_nearby_sources(rules, inp, occupied, blocked4, pads):
        return None
    if not _place_remaining_sources(rules, inp, occupied, blocked4, pads):
        return None

    bystander_pos = _place_bystanders(used_colors, inp, occupied, blocked4, pads)
    expected = _build_expected_output(rules, bystander_pos, h, w)
    if expected is None:
        return None
    if expected.shape[0] >= h and expected.shape[1] >= w:
        return None

    try:
        actual = main(inp)
        if np.array_equal(actual, expected):
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
