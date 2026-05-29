# Variant of abc82100
# Dimension: D5 — 模板放置的对齐方式 (transform)
# Change: d.anchor（规则示例中的对齐参考点）对齐到源像素s的位置，模板整体平移... → 对齐方向由源像素在网格中的象限决定

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg

# Design analysis for the D5 variant
# source_task: None, so the baseline is the original abc82100 solver/generator only.
#
# spec_variant:
#   core_change:
#     old_behavior: place each recolored stencil by translating the legend anchor onto a scattered source pixel
#     new_behavior: ignore the legend anchor during stamping; choose one of the stencil's four bbox corners from the scattered
#       source pixel's global quadrant, then align that chosen corner to the source pixel
#   concrete_pixel_behavior:
#     legend:
#       unchanged_visual_role: a 4-neighbor source->target pair still points at a teal component and teaches the color mapping
#       retained_solver_role: the continued direction still identifies which teal component belongs to the rule
#     stamping:
#       upper_left_quadrant: align stencil bbox upper-left corner to the source pixel
#       upper_right_quadrant: align stencil bbox upper-right corner to the source pixel
#       lower_left_quadrant: align stencil bbox lower-left corner to the source pixel
#       lower_right_quadrant: align stencil bbox lower-right corner to the source pixel
#     visibility_requirement:
#       because the source pixel disappears in the output, the chosen corner should itself be a filled stencil cell
#       therefore the generator only emits teal stencils whose four bbox corners are occupied
#   behavior_trace:
#     original_local_step:
#       a scattered source at (sr, sc) uses dr = (sr, sc) - legend_anchor, then adds that offset to every stencil cell
#     variant_step:
#       if (sr, sc) is in the upper-right quadrant, use the stencil bbox upper-right corner as the reference cell instead of
#       legend_anchor; translate so that upper-right corner lands at (sr, sc)
#     difference:
#       two scattered source pixels of the same color but in different quadrants now stamp the same teal shape from different
#       reference corners, so their output positions cannot be explained by a single anchor translation
#   complexity_calibration:
#     keep_low_branching: only four alignment cases, tied to visible global position categories rather than hidden ids
#     avoid_degenerate_templates:
#       stencils are at least 3x3 in bbox, occupy all four bbox corners, and include a non-corner anchor cell
#       this prevents the new corner choice from collapsing to the old anchor or to an invisible empty bbox corner
#     repeated_support:
#       each rule gets at least one scattered source in every quadrant, so the samples expose repeated examples of the new logic
#   implicit_assumption_audit:
#     preserved_code_paths:
#       rule discovery still assumes source/target are 4-neighbors and a teal component touches the continued anchor cell
#       preservation of bystanders still assumes source colors are consumed globally and other colors are safe to copy through
#     changed_assumption:
#       the original stamping loop assumed the legend anchor is the placement reference for every scattered source
#       that assumption is false in this variant, so the stamping reference must be recomputed per source pixel from its quadrant
#     new_needed_assumptions:
#       the selected bbox corner exists as a teal cell
#       each quadrant receives at least one scattered source often enough for the rule to be visually inferable
#       the legend anchor is not one of the four bbox corners, so the new behavior differs from the original baseline
#   placement_legality:
#     legend placement:
#       still uses the original collinear source->target->anchor geometry
#       the generator computes stencil top-left from the chosen local anchor and rejects out-of-bounds legends
#     scattered placement:
#       scattered source pixels remain isolated via 4-neighbor reservation to avoid spurious source-target pseudo-rules
#       quadrant-aware stamping may clip at borders, but the chosen quadrant naturally tends to push the stencil inward
#   visual_observability:
#     human-facing cue:
#       compare a scattered source pixel's location in the input with the same-color output object; the occupied output corner
#       that lands exactly on the former source position changes with quadrant
#     anti-noise:
#       templates stay small, connected, and re-used within a rule; bystanders use unrelated colors and remain isolated
#   cleanliness:
#     overlap policy:
#       output stamps may overlap as in the original task, but the generator still spaces input objects apart and keeps legends
#       away from scattered pixels so the training examples remain readable
#   color_inference:
#     unchanged:
#       colors still serve only as explicit source->target substitutions taught by the local legends
#       no sorting or branching depends on numeric color ids


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-teal examples, then stamp the teal pattern with quadrant-dependent corner alignment.

    Input structure:
    - The grid has a black background, isolated colored pixels, and one teal (8) stencil component per rule.
    - A rule example is still a 4-neighbor source->target pair whose continued direction reaches an anchor touching that teal
      stencil.
    - Extra scattered pixels reuse the source colors, and their global positions can fall into different quadrants of the grid.

    Rule:
    - Each rule example teaches a source color, a target color, and which teal component is the stencil.
    - For every non-legend source pixel, choose the stencil corner from the pixel's quadrant: upper-left uses the stencil's
      upper-left bbox corner, upper-right uses upper-right, lower-left uses lower-left, and lower-right uses lower-right.
    - Translate the whole stencil so that the chosen corner lands on the source pixel, recolor it to the target color, and
      erase every legend pixel plus every teal stencil from the output.

    Key insight:
    - The legend still identifies the color substitution and the stencil itself, but placement is no longer tied to the legend
      anchor. The source pixel's global quadrant selects which stencil corner acts as the reference point.
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

    # Preserve ordinary colored pixels that are not consumed by any learned rule.
    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        pattern = rule["pattern"]
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            corner = _pattern_corner(pattern, _quadrant_key(sr, sc, h, w))
            dr = sr - int(corner[0])
            dc = sc - int(corner[1])
            rows = pattern[:, 0] + dr
            cols = pattern[:, 1] + dc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = rule["target_color"]

    return out


# --------------- generator (c) ---------------

# ── CFG: tunable generation parameters ──────────────────────────────
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

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_QUADRANT = 60
SCATTERED_EXTRA_RETRIES = 120
BYSTANDER_RETRIES = 60
OUTER_RETRIES = 400

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIRS8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
_QUADRANTS = ("tl", "tr", "bl", "br")


def _reserve_4neighbors(reserved: set, cells) -> None:
    for r, c in cells:
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


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
    dim_options = [dims for dims in dim_options if dims[0] * dims[1] <= max(STENCIL_MAX_CAP, max_pixels)]
    ph, pw = random.choice(dim_options)
    center = (random.randint(1, ph - 2), random.randint(1, pw - 2))
    corners = {(0, 0), (0, pw - 1), (ph - 1, 0), (ph - 1, pw - 1)}

    coords = set()
    for corner in corners:
        coords.update(_king_path(center, corner))

    target = random.randint(max(min_pixels, len(coords) + 1), min(max_pixels, ph * pw, len(coords) + STENCIL_EXTRA_PIXELS_MAX))
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


def _stamp_origin(rule: dict, sr: int, sc: int, H: int, W: int) -> tuple[int, int]:
    ph, pw = rule["size"]
    quadrant = _quadrant_key(sr, sc, H, W)
    if quadrant == "tl":
        return sr, sc
    if quadrant == "tr":
        return sr, sc - (pw - 1)
    if quadrant == "bl":
        return sr - (ph - 1), sc
    return sr - (ph - 1), sc - (pw - 1)


def _place_legends(rules, inp, occupied, reserved):
    """Place original-style legend triplets while keeping stencils fully in bounds."""
    H, W = inp.shape
    for rule in rules:
        cells = rule["cells"]
        ar_local, ac_local = rule["anchor"]
        placed = False
        for _ in range(LEGEND_RETRIES):
            dr, dc = random.choice(_DIRS4)
            sr, sc = random.randint(0, H - 1), random.randint(0, W - 1)
            tr, tc = sr + dr, sc + dc
            ar, ac = sr + 2 * dr, sc + 2 * dc
            if not (0 <= tr < H and 0 <= tc < W and 0 <= ar < H and 0 <= ac < W):
                continue

            top = ar - ar_local
            left = ac - ac_local
            teal_cells = {(top + rr, left + cc) for rr, cc in cells}
            if any(not (0 <= nr < H and 0 <= nc < W) for nr, nc in teal_cells):
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


def _quadrant_ranges(H: int, W: int) -> dict[str, tuple[range, range]]:
    r_split = (H + 1) // 2
    c_split = (W + 1) // 2
    return {
        "tl": (range(0, r_split), range(0, c_split)),
        "tr": (range(0, r_split), range(c_split, W)),
        "bl": (range(r_split, H), range(0, c_split)),
        "br": (range(r_split, H), range(c_split, W)),
    }


def _place_scattered(rules, inp, occupied, reserved):
    """Place isolated source pixels with explicit support in all four quadrants."""
    H, W = inp.shape
    ranges = _quadrant_ranges(H, W)
    for rule in rules:
        positions: list[tuple[int, int]] = []

        for quadrant in _QUADRANTS:
            row_range, col_range = ranges[quadrant]
            if len(row_range) == 0 or len(col_range) == 0:
                return False
            placed = False
            for _ in range(SCATTERED_RETRIES_PER_QUADRANT):
                r = random.choice(list(row_range))
                c = random.choice(list(col_range))
                if (r, c) in occupied or (r, c) in reserved:
                    continue
                positions.append((r, c))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                placed = True
                break
            if not placed:
                return False

        for _ in range(SCATTERED_EXTRA_RETRIES):
            if len(positions) >= rule["n_sc"]:
                break
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            positions.append((r, c))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(used_colors, inp, occupied, reserved):
    bystander_colors = [c for c in NON_BG_NON_TEAL_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    H, W = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r, c = random.randint(0, H - 1), random.randint(0, W - 1)
            if (r, c) not in occupied and (r, c) not in reserved:
                bc = random.choice(bystander_colors)
                inp[r, c] = bc
                placed.append((r, c, bc))
                occupied.add((r, c))
                _reserve_4neighbors(reserved, [(r, c)])
                break
    return placed


def _build_output(rules, bystander_pos, H, W):
    output = np.zeros((H, W), dtype=int)
    for r, c, bc in bystander_pos:
        output[r, c] = bc
    for rule in rules:
        for sr, sc in rule["scattered"]:
            top, left = _stamp_origin(rule, sr, sc, H, W)
            for rr, cc in rule["cells"]:
                nr, nc = top + rr, left + cc
                if 0 <= nr < H and 0 <= nc < W:
                    output[nr, nc] = rule["tgt"]
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    H = random.randint(min_dim, GRID_DIM_MAX)
    W = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, H * W)
    occupied, reserved = set(), set()
    inp = np.zeros((H, W), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

    output = _build_output(rules, bystander_pos, H, W)
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

