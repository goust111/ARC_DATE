# Combo variant of abc82100 (index 14)
# combo_spec: D6/2+D14/0
# Name: 渐变着色+几何中心锚点

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


# Design analysis for the combo variant
# -------------------------------------
# design_analysis:
#   concretization:
#     - The original task learns a local legend `source -> target -> stencil-touch`
#       and stamps that teal stencil onto scattered source pixels.
#     - This combo keeps the same local legend shape so the rule is still readable,
#       but the continued cell is now only a contact point used to identify which
#       teal component belongs to the legend. The true stamping anchor is the
#       stencil's geometric center: the center of its bounding box.
#     - Stamped coloring also changes. Instead of a flat target-color fill, each
#       translated stencil cell is recolored by Manhattan distance from that
#       geometric center:
#         distance 0 -> target color
#         distance 1 -> middle gradient color
#         distance >=2 -> outer gradient color
#     - Concrete trace:
#         suppose a legend uses pink source 6, maroon target 9, and a 3x3 teal
#         stencil whose continued contact cell is the top edge but whose bbox
#         center is the middle cell. In the original task, stamping onto a pink
#         scattered pixel would align the contact cell there and fill everything
#         maroon. In this combo, the same scattered pixel becomes the stencil's
#         center; its center cell stays maroon(9), edge cells become red(2), and
#         corner cells become orange(7). The same legend therefore teaches both
#         "which stencil" and "ignore the contact point when aligning it".
#   complexity_calibration:
#     - To prevent collapse back to the original task, every generated stencil:
#         1. has an odd 3x3 bounding box with center cell present,
#         2. contains at least one distance-1 cell and one distance-2 cell,
#         3. is placed with a legend contact cell that is not the center.
#     - To keep the gradient visually inferable, target colors are restricted to
#       four dark hues with a small explicit visible palette:
#         blue(1)   -> teal(8)   -> yellow(4)
#         red(2)    -> orange(7) -> yellow(4)
#         green(3)  -> teal(8)   -> yellow(4)
#         maroon(9) -> red(2)    -> orange(7)
#     - Stamped footprints are kept fully in-bounds and mutually disjoint, so no
#       later overwrite can destroy the center-outward gradient evidence.
#   implicit_assumption_audit:
#     - Original code assumed the forward-continued cell was the translation
#       anchor. That is no longer valid; solver-side translation must use pattern
#       coordinates centered on the stencil bbox center instead.
#     - Original code could accept any teal component touching the continued cell.
#       Here the component must have an integer geometric center and include that
#       center cell, otherwise no distance-0 pixel would exist. The generator
#       guarantees both properties by construction.
#     - Original code could allow clipping and overlap without changing the color
#       logic because stamps were uniform. Here that would obscure the gradient,
#       so generation reserves each whole footprint and its 4-neighborhood.
#     - Preserving non-legend colored pixels still works, but bystanders must use
#       colors outside the gradient palettes to avoid accidental false cues.
#   placement_legality:
#     - Legend placement chooses a real non-center stencil cell as the continued
#       contact point, then places `target` one step outward and `source` another
#       step outward on the same line. Source/target may not overlap the stencil.
#     - Scattered source pixels are only placed where the entire centered stencil
#       footprint fits inside the grid and does not intersect any occupied or
#       reserved region. This simultaneously prevents overlap and false-rule
#       adjacency.
#   visual_observability:
#     - The same source color appears on 3-8 scattered pixels, so each training
#       sample repeats the exact center-anchored gradient multiple times.
#     - Because the legend contact point is visibly off-center while the output
#       target-color pixel sits at the stencil center, a solver must reconcile the
#       two clues; simple "continue the pair direction" alignment fails.
#   cleanliness:
#     - Bystanders are drawn only from clean colors that do not appear inside the
#       gradient ramps for active rules, keeping the output visually separable.
#   inferability:
#     - The color changes are structured by position, not by hidden numeric order.
#       Across samples, the target hue always appears at the center, the same
#       middle hue appears on distance-1 cells, and the same outer hue appears on
#       distance-2 cells for each target family. This is repeatable from 2-4 ARC
#       examples and does not rely on invisible metadata.
#   solver_generator_contract:
#     - main() assumes each legend's continued cell belongs to exactly one teal
#       component and that this component has an integer bbox center included in
#       the component itself.
#     - generate_input() enforces those assumptions, validates every sample with
#       `main(input) == expected_output`, and also forces `contact != center` so
#       the D14 change is materially active instead of decorative.


TARGET_GRADIENTS: dict[int, tuple[int, int, int]] = {
    1: (1, 8, 4),
    2: (2, 7, 4),
    3: (3, 8, 4),
    9: (9, 2, 7),
}
TARGET_COLORS = list(TARGET_GRADIENTS.keys())
SOURCE_COLORS = [4, 5, 6, 7]
CLEAN_BYSTANDER_COLORS = [5, 6]

GRID_DIM_MAX = 20
MIN_DIM_BY_RULES = {1: 5, 2: 10, 3: 14, 4: 18}

RULE_COUNT_WEIGHTS = [1, 1, 2, 2, 3, 3, 4]

SCATTERED_MIN = 3
SCATTERED_MAX_CAP = 8
SCATTERED_AREA_DIV = 30

BYSTANDER_MAX = 5

LEGEND_RETRIES = 500
SCATTERED_RETRIES_PER_PIXEL = 80
BYSTANDER_RETRIES = 50
OUTER_RETRIES = 300

_DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# Every stencil is 8-connected, uses a 3x3 bbox centered at (0,0), contains the
# center cell, and contains both distance-1 and distance-2 cells.
STENCIL_BANK: list[tuple[tuple[int, int], ...]] = [
    ((-1, -1), (-1, 0), (0, -1), (0, 0), (0, 1), (1, 0)),
    ((-1, 0), (-1, 1), (0, -1), (0, 0), (1, -1), (1, 0), (1, 1)),
    ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, 0), (1, 1)),
    ((-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0)),
    ((-1, -1), (-1, 0), (0, 0), (0, 1), (1, -1), (1, 0)),
    ((-1, 0), (0, -1), (0, 0), (0, 1), (1, 0), (1, 1)),
    ((-1, -1), (-1, 1), (0, -1), (0, 0), (1, 0), (1, 1)),
    ((-1, 0), (0, -1), (0, 0), (1, 0), (1, 1), (0, 1), (-1, -1)),
]


def _gradient_color(target_color: int, distance: int) -> int:
    palette = TARGET_GRADIENTS[int(target_color)]
    return int(palette[min(max(int(distance), 0), 2)])


def _pattern_center(pattern: np.ndarray) -> tuple[int, int] | None:
    min_r = int(pattern[:, 0].min())
    max_r = int(pattern[:, 0].max())
    min_c = int(pattern[:, 1].min())
    max_c = int(pattern[:, 1].max())
    if (min_r + max_r) % 2 != 0 or (min_c + max_c) % 2 != 0:
        return None
    return ((min_r + max_r) // 2, (min_c + max_c) // 2)


def _reserve_4neighbors(reserved: set[tuple[int, int]], cells) -> None:
    for r, c in cells:
        r = int(r)
        c = int(c)
        for dr, dc in _DIRS4:
            reserved.add((r + dr, c + dc))


def _footprint_cells(rel: list[tuple[int, int]], anchor_r: int, anchor_c: int) -> set[tuple[int, int]]:
    return {(int(anchor_r) + rr, int(anchor_c) + cc) for rr, cc in rel}


def _contact_options(rel: list[tuple[int, int]]) -> list[tuple[int, int, int, int]]:
    stencil = set(rel)
    options: list[tuple[int, int, int, int]] = []
    for rr, cc in rel:
        if (rr, cc) == (0, 0):
            continue
        for dr, dc in _DIRS4:
            target = (rr - dr, cc - dc)
            source = (rr - 2 * dr, cc - 2 * dc)
            if target in stencil or source in stencil:
                continue
            options.append((rr, cc, dr, dc))
    return options


def _discover_rules(grid: np.ndarray, bg: int, pattern_color: int = 8) -> list[dict]:
    h, w = grid.shape
    patterns = [np.asarray(pattern, dtype=int) for pattern in components_mask(grid == pattern_color, connectivity=8)]
    cell_to_pattern: dict[tuple[int, int], int] = {}
    for i, pattern in enumerate(patterns):
        for r, c in map(tuple, pattern):
            cell_to_pattern[(int(r), int(c))] = int(i)

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
                cr = tr + dr
                cc = tc + dc
                if not (0 <= tr < h and 0 <= tc < w and 0 <= cr < h and 0 <= cc < w):
                    continue

                target_color = int(grid[tr, tc])
                if target_color in (int(bg), int(pattern_color)) or target_color == source_color:
                    continue

                pattern_index = cell_to_pattern.get((int(cr), int(cc)))
                if pattern_index is None:
                    continue

                key = ((int(r), int(c)), (int(tr), int(tc)), int(pattern_index))
                if key in seen:
                    continue
                seen.add(key)

                pattern = patterns[pattern_index]
                center = _pattern_center(pattern)
                if center is None:
                    continue
                center_r, center_c = center
                if not np.any((pattern[:, 0] == int(center_r)) & (pattern[:, 1] == int(center_c))):
                    continue

                rel_pattern = np.column_stack(
                    (pattern[:, 0] - int(center_r), pattern[:, 1] - int(center_c))
                ).astype(int)
                distance_buckets = np.minimum(np.abs(rel_pattern[:, 0]) + np.abs(rel_pattern[:, 1]), 2).astype(int)
                rules.append(
                    {
                        "source_color": int(source_color),
                        "target_color": int(target_color),
                        "source_pos": (int(r), int(c)),
                        "target_pos": (int(tr), int(tc)),
                        "contact_pos": (int(cr), int(cc)),
                        "center_pos": (int(center_r), int(center_c)),
                        "rel_pattern": rel_pattern,
                        "distance_buckets": distance_buckets,
                    }
                )

    return rules


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn local source-target-stencil examples, then stamp centered gradients elsewhere.

    Input structure:
    - The grid has a dominant background, isolated colored pixels, and teal(8)
      stencil components attached to local legend examples.
    - Each legend is a 4-neighbor source->target pair whose forward continuation
      reaches one teal cell of the corresponding stencil, but that touched teal
      cell is not necessarily the placement anchor.

    Rule:
    - Every non-legend pixel of a learned source color becomes a translated copy
      of that legend's teal stencil, aligned so the stencil's geometric center
      (the center of its bounding box) lands on the source pixel.
    - The copied stencil is recolored by Manhattan distance from that center:
      distance 0 uses the target color, distance 1 uses the middle gradient color,
      and distance >=2 uses the outer gradient color for that target family.
    - All legend pixels and all teal template pixels are erased from the output.

    Key insight: The adjacent colored pair still identifies which stencil belongs
    to which source color, but the forward-touched teal cell is only a locator.
    The actual alignment and recoloring both depend on the stencil's geometric
    center, so the solver must separate "which component" from "where its origin is".
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

    for r, c in map(tuple, np.argwhere((g != int(bg)) & (g != pattern_color))):
        r = int(r)
        c = int(c)
        color = int(g[r, c])
        if (r, c) in legend_pixels or color in source_colors:
            continue
        out[r, c] = color

    for rule in rules:
        rel_pattern = rule["rel_pattern"]
        distance_buckets = rule["distance_buckets"]
        pattern_colors = np.array(
            [_gradient_color(rule["target_color"], int(dist)) for dist in distance_buckets],
            dtype=int,
        )
        for sr, sc in map(tuple, np.argwhere(g == rule["source_color"])):
            sr = int(sr)
            sc = int(sc)
            if (sr, sc) in legend_pixels:
                continue
            rows = rel_pattern[:, 0] + sr
            cols = rel_pattern[:, 1] + sc
            ok = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
            out[rows[ok], cols[ok]] = pattern_colors[ok]

    return out


def _build_rules(num_rules: int, area: int):
    target_colors = TARGET_COLORS[:]
    source_colors = SOURCE_COLORS[:]
    stencils = STENCIL_BANK[:]
    random.shuffle(target_colors)
    random.shuffle(source_colors)
    random.shuffle(stencils)

    sc_max = min(SCATTERED_MAX_CAP, max(SCATTERED_MIN, area // SCATTERED_AREA_DIV))

    rules = []
    used = set()
    for i in range(num_rules):
        src_c = source_colors[i]
        tgt_c = target_colors[i]
        rel = [tuple(cell) for cell in stencils[i]]
        rel_with_dist = [(rr, cc, min(abs(rr) + abs(cc), 2)) for rr, cc in rel]
        options = _contact_options(rel)
        min_rr = min(rr for rr, _cc in rel)
        max_rr = max(rr for rr, _cc in rel)
        min_cc = min(cc for _rr, cc in rel)
        max_cc = max(cc for _rr, cc in rel)
        used.update((src_c, tgt_c))
        rules.append(
            {
                "src": int(src_c),
                "tgt": int(tgt_c),
                "rel": rel,
                "rel_with_dist": rel_with_dist,
                "contact_options": options,
                "n_sc": random.randint(SCATTERED_MIN, sc_max),
                "row_bounds": (-min_rr, max_rr),
                "col_bounds": (-min_cc, max_cc),
            }
        )
    return rules, used


def _place_legends(
    rules,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        rel = rule["rel"]
        placed = False
        options = rule["contact_options"][:]
        random.shuffle(options)
        for _ in range(LEGEND_RETRIES):
            center_r = random.randint(1, h - 2)
            center_c = random.randint(1, w - 2)
            for contact_rr, contact_cc, dr, dc in options:
                source = (center_r + contact_rr - 2 * dr, center_c + contact_cc - 2 * dc)
                target = (center_r + contact_rr - dr, center_c + contact_cc - dc)
                teal_cells = _footprint_cells(rel, center_r, center_c)
                all_cells = teal_cells | {source, target}
                if any(not (0 <= r < h and 0 <= c < w) for r, c in all_cells):
                    continue
                if source in teal_cells or target in teal_cells:
                    continue
                if all_cells & occupied or all_cells & reserved:
                    continue

                sr, sc = source
                tr, tc = target
                inp[sr, sc] = rule["src"]
                inp[tr, tc] = rule["tgt"]
                for nr, nc in teal_cells:
                    inp[nr, nc] = 8
                occupied.update(all_cells)
                _reserve_4neighbors(reserved, all_cells)
                rule["legend_contact"] = (center_r + contact_rr, center_c + contact_cc)
                rule["legend_center"] = (center_r, center_c)
                placed = True
                break
            if placed:
                break
        if not placed:
            return False
    return True


def _place_scattered(
    rules,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    h, w = inp.shape
    for rule in rules:
        positions = []
        row_margin_lo, row_margin_hi = rule["row_bounds"]
        col_margin_lo, col_margin_hi = rule["col_bounds"]
        row_min = int(row_margin_lo)
        row_max = h - 1 - int(row_margin_hi)
        col_min = int(col_margin_lo)
        col_max = w - 1 - int(col_margin_hi)
        if row_min > row_max or col_min > col_max:
            return False

        for _ in range(rule["n_sc"] * SCATTERED_RETRIES_PER_PIXEL):
            if len(positions) >= rule["n_sc"]:
                break
            r = random.randint(row_min, row_max)
            c = random.randint(col_min, col_max)
            footprint = _footprint_cells(rule["rel"], r, c)
            if footprint & occupied or footprint & reserved:
                continue

            positions.append((r, c))
            occupied.add((r, c))
            reserved.update(footprint)
            _reserve_4neighbors(reserved, footprint)

        if len(positions) < SCATTERED_MIN:
            return False
        rule["scattered"] = positions
        for r, c in positions:
            inp[r, c] = rule["src"]
    return True


def _place_bystanders(
    used_colors,
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
):
    bystander_colors = [c for c in CLEAN_BYSTANDER_COLORS if c not in used_colors]
    if not bystander_colors:
        return []
    h, w = inp.shape
    n = random.randint(0, min(BYSTANDER_MAX, len(bystander_colors) * 2))
    placed = []
    for _ in range(n):
        for _ in range(BYSTANDER_RETRIES):
            r = random.randint(0, h - 1)
            c = random.randint(0, w - 1)
            if (r, c) in occupied or (r, c) in reserved:
                continue
            color = random.choice(bystander_colors)
            inp[r, c] = color
            placed.append((r, c, color))
            occupied.add((r, c))
            _reserve_4neighbors(reserved, [(r, c)])
            break
    return placed


def _build_output(rules, bystander_pos, h: int, w: int) -> np.ndarray:
    output = np.zeros((h, w), dtype=int)
    for r, c, color in bystander_pos:
        output[r, c] = color
    for rule in rules:
        for sr, sc in rule["scattered"]:
            for rr, cc, dist in rule["rel_with_dist"]:
                nr = sr + rr
                nc = sc + cc
                if 0 <= nr < h and 0 <= nc < w:
                    output[nr, nc] = _gradient_color(rule["tgt"], dist)
    return output


def _gen_attempt():
    num_rules = random.choice(RULE_COUNT_WEIGHTS)
    min_dim = MIN_DIM_BY_RULES[num_rules]
    h = random.randint(min_dim, GRID_DIM_MAX)
    w = random.randint(min_dim, GRID_DIM_MAX)

    rules, used_colors = _build_rules(num_rules, h * w)
    occupied: set[tuple[int, int]] = set()
    reserved: set[tuple[int, int]] = set()
    inp = np.zeros((h, w), dtype=int)

    if not _place_legends(rules, inp, occupied, reserved):
        return None
    if not _place_scattered(rules, inp, occupied, reserved):
        return None
    bystander_pos = _place_bystanders(used_colors, inp, occupied, reserved)

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
