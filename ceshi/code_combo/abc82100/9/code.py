# Combo variant of abc82100 (index 9)
# combo_spec: D3/2+D5/2
# Name: 边缘源像素+象限对齐

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the combo variant
# source_task: abc82100
#
# spec_variant:
#   concreteization_and_trace:
#     baseline_example:
#       a legend teaches source color A, target color B, and a teal stencil with a local anchor.
#       every scattered A pixel receives the same stencil by aligning the legend anchor onto that pixel.
#     combo_example:
#       only edge A pixels are actionable sources; interior A pixels belong only to legends.
#       if an actionable source sits on the top edge in the left half, it is in the upper-left quadrant, so the stencil's
#       upper-left bbox corner lands on that source; a bottom-right edge source instead receives the same stencil with its
#       lower-right bbox corner aligned to the source.
#     visible_difference:
#       the same color can now produce the same target color but with different absolute placement offsets depending on which
#       border quadrant the source occupies. This is not explainable by a single legend-anchor translation.
#   complexity_calibration:
#     keep_branching_small:
#       there are only four placement cases, one per quadrant.
#     repeated_support:
#       each rule gets at least one edge source in every quadrant so the quadrant logic is repeated within each sample.
#     anti_degeneracy:
#       every stencil occupies all four bbox corners and the legend anchor is chosen from a non-corner cell, preventing the
#       quadrant corner from collapsing to the original anchor-based behavior.
#   implicit_assumption_audit:
#     preserved:
#       _discover_rules still relies on an interior 4-neighbor source->target legend whose continuation touches a teal
#       component, so legend geometry is kept unchanged.
#     changed_in_main:
#       the baseline preservation loop erased all pixels with a learned source color because every such pixel was actionable.
#       that is no longer true logically, so the new loop removes only edge pixels of learned source colors.
#     changed_in_stamping:
#       the baseline stamping loop assumed the legend anchor is the placement reference for every source.
#       the combo invalidates that assumption; the reference must be recomputed from the source pixel's quadrant each time.
#     generator_obligations:
#       legends and teal components stay off the border, actionable sources stay on the border, source pixels remain isolated,
#       and bystanders avoid future output stamps so preservation remains visually true.
#   placement_legality:
#     legends:
#       placed fully inside the non-border interior to keep border semantics unambiguous.
#     edge_sources:
#       chosen from border cells that belong to a specific quadrant and also keep the full quadrant-aligned stencil in bounds.
#     bystanders:
#       placed only in the interior and outside projected output cells to avoid accidental overwrites.
#   visual_observability:
#     human_cue:
#       the border itself highlights which pixels are special sources, and the quadrant determines which corner of the output
#       object coincides with the vanished input pixel.
#     readability:
#       legends live in the interior, edge sources live on the frame, and bystanders avoid the frame; this separation keeps
#       training examples clean enough to infer the fused rule from a few samples.
#   color_inference:
#     unchanged:
#       colors still only encode explicit source->target substitutions taught by the legends; quadrant choice depends on
#       location, not on color ids.
#   main_assumption_review:
#     rule_discovery_loop:
#       assumes each rule has one nearby teal component distinguishable from others by continuation direction; legend placement
#       and reservation preserve that.
#     colored_pixel_preservation_loop:
#       assumes non-legend, non-teal, non-edge-source pixels should survive unchanged; generator only uses such pixels for
#       bystanders, so this remains valid.
#     source_iteration_loop:
#       assumes every actionable source of a learned color lies on the border and its quadrant-selected corner is an occupied
#       stencil cell; generator guarantees both conditions.


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _anchor_distance(pattern: np.ndarray, anchor: tuple[int, int]) -> int:
    return min(_chebyshev((int(r), int(c)), anchor) for r, c in pattern)


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


def _quadrant_key(r: int, c: int, h: int, w: int) -> str:
    top = 2 * int(r) < int(h)
    left = 2 * int(c) < int(w)
    if top and left:
        return "tl"
    if top and not left:
        return "tr"
    if not top and left:
        return "bl"
    return "br"


def _pattern_corner(pattern: np.ndarray, quadrant: str) -> tuple[int, int]:
    rows = pattern[:, 0]
    cols = pattern[:, 1]
    r0 = int(rows.min())
    r1 = int(rows.max())
    c0 = int(cols.min())
    c1 = int(cols.max())
    if quadrant == "tl":
        return r0, c0
    if quadrant == "tr":
        return r0, c1
    if quadrant == "bl":
        return r1, c0
    return r1, c1


def _is_edge(r: int, c: int, h: int, w: int) -> bool:
    return int(r) in (0, int(h) - 1) or int(c) in (0, int(w) - 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn interior source-target-teal legends, then recolor the teal stencil from edge sources using quadrant-selected corners.

    Input structure:
    - The grid has a black background, several interior legend examples, and extra source-colored pixels restricted to the
      outer border.
    - Each legend is a 4-neighbor source->target pair whose continued direction reaches an anchor touching one teal (8)
      component, which defines the stencil for that color rule.
    - Non-source bystander pixels may also appear, but they are separate from both the legends and the border sources.

    Rule:
    - Each legend teaches a source color, a target color, and the teal stencil to reuse.
    - Only non-legend source pixels on the border are actionable. Their quadrant picks the stencil corner used for alignment:
      upper-left, upper-right, lower-left, or lower-right bbox corner respectively.
    - Copy the whole stencil so that the chosen corner lands on the source pixel, recolor it to the target color, and erase
      every legend pixel plus every teal stencil from the output.

    Key insight:
    - The border source pixel now carries two pieces of information at once: its color selects which legend rule to use, and
      its global quadrant selects which stencil corner becomes the placement reference instead of the legend anchor.
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

    out = np.full_like(g, int(bg))

    # Preserve non-legend colored pixels, except for actionable edge sources that will be replaced by stencils.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels:
            continue
        if color in source_colors and _is_edge(r, c, h, w):
            continue
        out[r, c] = color

    for rule in rules:
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels or not _is_edge(sr, sc, h, w):
                continue
            corner = _pattern_corner(pattern, _quadrant_key(sr, sc, h, w))
            dr = sr - int(corner[0])
            dc = sc - int(corner[1])
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (combo) ---------------

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 8, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]
NON_BG_NON_TEAL_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]

STENCIL_MAX_CAP = 12
STENCIL_MIN_PIXELS = 6
STENCIL_EXTRA_PIXELS_MAX = 4

SCATTERED_MIN = 4
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 28

BYSTANDER_MAX = 5

LEGEND_RETRIES = 700
SCATTERED_RETRIES_PER_QUADRANT = 100
SCATTERED_EXTRA_RETRIES = 240
BYSTANDER_RETRIES = 120
OUTER_RETRIES = 500

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_QUADRANTS = ("tl", "tr", "bl", "br")


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((int(r) + dr, int(c) + dc))


def _king_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    path = [(int(r), int(c))]
    while (r, c) != (tr, tc):
        if r < tr:
            r += 1
        elif r > tr:
            r -= 1
        if c < tc:
            c += 1
        elif c > tc:
            c -= 1
        path.append((int(r), int(c)))
    return path


def _make_stencil(min_pixels: int, max_pixels: int):
    """Generate a connected stencil whose four bbox corners are occupied."""
    dim_options = [(3, 3), (3, 4), (4, 3)]
    ph, pw = random.choice(dim_options)
    center = (random.randint(1, ph - 2), random.randint(1, pw - 2))
    corners = {(0, 0), (0, pw - 1), (ph - 1, 0), (ph - 1, pw - 1)}

    coords = set()
    for corner in corners:
        coords.update(_king_path(center, corner))

    lo = max(min_pixels, len(coords) + 1)
    hi = min(max_pixels, ph * pw, len(coords) + STENCIL_EXTRA_PIXELS_MAX)
    target = random.randint(lo, hi)
    while len(coords) < target:
        frontier = set()
        for r, c in coords:
            for dr, dc in _DIRS8:
                nr, nc = r + dr, c + dc
                if 0 <= nr < ph and 0 <= nc < pw and (nr, nc) not in coords:
                    frontier.add((nr, nc))
        coords.add(random.choice(sorted(frontier)))

    non_corner_cells = sorted(coords - corners)
    anchor = random.choice(non_corner_cells)
    return sorted(coords), anchor, (ph, pw)


def _build_rules(num_rules: int, area: int):
    colors = NON_BG_NON_TEAL_COLORS[:]
    random.shuffle(colors)

    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))
    rules, used = [], set()
    for i in range(num_rules):
        src_c, tgt_c = colors[2 * i], colors[2 * i + 1]
        used.update((src_c, tgt_c))
        stencil, anchor, size = _make_stencil(STENCIL_MIN_PIXELS, STENCIL_MAX_CAP)
        rules.append(
            {
                "src": src_c,
                "tgt": tgt_c,
                "cells": stencil,
                "anchor": anchor,
                "size": size,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
            }
        )
    return rules, used


def _quadrant_ranges(h: int, w: int) -> dict[str, tuple[range, range]]:
    r_split = (int(h) + 1) // 2
    c_split = (int(w) + 1) // 2
    return {
        "tl": (range(0, r_split), range(0, c_split)),
        "tr": (range(0, r_split), range(c_split, int(w))),
        "bl": (range(r_split, int(h)), range(0, c_split)),
        "br": (range(r_split, int(h)), range(c_split, int(w))),
    }


def _border_cells_for_quadrant(h: int, w: int, quadrant: str) -> list[tuple[int, int]]:
    row_range, col_range = _quadrant_ranges(h, w)[quadrant]
    cells = {
        (int(r), int(c))
        for r in row_range
        for c in col_range
        if _is_edge(int(r), int(c), h, w)
    }
    return sorted(cells)


def _stamp_origin(rule: dict, sr: int, sc: int, h: int, w: int) -> tuple[int, int]:
    ph, pw = rule["size"]
    quadrant = _quadrant_key(sr, sc, h, w)
    if quadrant == "tl":
        return int(sr), int(sc)
    if quadrant == "tr":
        return int(sr), int(sc) - (int(pw) - 1)
    if quadrant == "bl":
        return int(sr) - (int(ph) - 1), int(sc)
    return int(sr) - (int(ph) - 1), int(sc) - (int(pw) - 1)


def _anchor_origin(rule: dict, sr: int, sc: int) -> tuple[int, int]:
    ar, ac = rule["anchor"]
    return int(sr) - int(ar), int(sc) - int(ac)


def _source_fits_quadrant(rule: dict, r: int, c: int, h: int, w: int) -> bool:
    top, left = _stamp_origin(rule, r, c, h, w)
    ph, pw = rule["size"]
    return 0 <= top and 0 <= left and top + int(ph) <= int(h) and left + int(pw) <= int(w)


def _stamped_cells(rule: dict, sr: int, sc: int, top_left: tuple[int, int], h: int, w: int) -> set[tuple[int, int]]:
    del sr, sc
    top, left = top_left
    cells = set()
    for rr, cc in rule["cells"]:
        nr = int(top) + int(rr)
        nc = int(left) + int(cc)
        if 0 <= nr < int(h) and 0 <= nc < int(w):
            cells.add((nr, nc))
    return cells


def _place_legends(rules, inp, occupied, reserved):
    """Place source->target->anchor legends fully inside the interior."""
    h, w = inp.shape
    for rule in rules:
        cells = rule["cells"]
        ar_local, ac_local = rule["anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(1, h - 2), random.randint(1, w - 2)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 < tr < h - 1 and 0 < tc < w - 1 and 0 < ar < h - 1 and 0 < ac < w - 1):
                continue

            top = ar - ar_local
            left = ac - ac_local
            teal_cells = {(top + rr, left + cc) for rr, cc in cells}
            if any(not (0 < nr < h - 1 and 0 < nc < w - 1) for nr, nc in teal_cells):
                continue
            reverse_anchor = (sr - dr, sc - dc)
            if any(
                max(abs(int(nr) - int(reverse_anchor[0])), abs(int(nc) - int(reverse_anchor[1]))) <= 1
                for nr, nc in teal_cells
            ):
                continue

            all_cells = {(sr, sc), (tr, tc), (ar, ac)} | teal_cells
            if all_cells & (occupied | reserved):
                continue

            inp[sr, sc] = rule["src"]
            inp[tr, tc] = rule["tgt"]
            for nr, nc in teal_cells:
                inp[nr, nc] = 8
            rule["legend"] = {"source": (sr, sc), "target": (tr, tc), "anchor_abs": (ar, ac)}
            occupied.update(all_cells)
            _reserve_4neighbors(reserved, all_cells)
            placed = True
            break
        if not placed:
            return False
    return True


def _place_edge_sources(rules, inp, occupied, reserved):
    """Place isolated actionable source pixels on the border, with support in all four quadrants."""
    h, w = inp.shape
    border_by_quadrant = {q: _border_cells_for_quadrant(h, w, q) for q in _QUADRANTS}

    for rule in sorted(rules, key=lambda item: int(item["n_sc"]), reverse=True):
        positions: list[tuple[int, int]] = []

        for quadrant in _QUADRANTS:
            candidates = [
                (r, c)
                for r, c in border_by_quadrant[quadrant]
                if _source_fits_quadrant(rule, r, c, h, w)
            ]
            if not candidates:
                return False
            placed = False
            for r, c in random.sample(candidates, len(candidates)):
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                positions.append((int(r), int(c)))
                occupied.add((int(r), int(c)))
                _reserve_4neighbors(reserved, [(int(r), int(c))])
                placed = True
                break
            if not placed:
                return False

        extra_pool = [
            (r, c)
            for quadrant in _QUADRANTS
            for r, c in border_by_quadrant[quadrant]
            if _source_fits_quadrant(rule, r, c, h, w)
        ]
        for r, c in random.sample(extra_pool, len(extra_pool)):
            if len(positions) >= int(rule["n_sc"]):
                break
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((int(r), int(c)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[int(r), int(c)] = rule["src"]
    return True


def _project_output_cells(rules, h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for rule in rules:
        for sr, sc in rule["scattered"]:
            cells |= _stamped_cells(rule, sr, sc, _stamp_origin(rule, sr, sc, h, w), h, w)
    return cells


def _place_bystanders(used_colors, rules, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    blocked_output = _project_output_cells(rules, h, w)
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(1, h - 2), random.randint(1, w - 2)
            if (r, c) in blocked_output:
                continue
            if (r, c) in occupied or (r, c) in reserved:
                continue
            bc = random.choice(bystander_colors)
            inp[r, c] = bc
            placed.append((int(r), int(c), int(bc)))
            occupied.add((int(r), int(c)))
            _reserve_4neighbors(reserved, [(int(r), int(c))])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            top_left = _stamp_origin(rule, sr, sc, h, w)
            for nr, nc in _stamped_cells(rule, sr, sc, top_left, h, w):
                output[int(nr), int(nc)] = int(rule["tgt"])
    return output


def _build_anchor_output(rules, bystander_pos, h: int, w: int):
    output = np.zeros((h, w), dtype=int)
    for r, c, bc in bystander_pos:
        output[int(r), int(c)] = int(bc)
    for rule in rules:
        for sr, sc in rule["scattered"]:
            top_left = _anchor_origin(rule, sr, sc)
            for nr, nc in _stamped_cells(rule, sr, sc, top_left, h, w):
                output[int(nr), int(nc)] = int(rule["tgt"])
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied, reserved = set(), set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_edge_sources(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, rules, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, h, w)
    anchor_output = _build_anchor_output(rules, bystander_pos, h, w)
    if np.array_equal(output, anchor_output):
        return None

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
