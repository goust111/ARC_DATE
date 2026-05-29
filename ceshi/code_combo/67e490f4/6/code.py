# Combo variant of 67e490f4 (index 6)
# combo_spec: D3/0+D16/0
# Name: 迭代填充+最少频率
#
# Design analysis
# ---------------
# spec:
#   fused_core:
#     - Keep the original scene grammar: one large monochrome frame with bg-colored
#       holes inside, plus scattered outside sprites whose shapes match the holes up
#       to rotation/flip.
#     - Fuse two changes into one decision rule:
#         1. D3/0: choose the least frequent color, not the most frequent one.
#         2. D16/0: fill holes iteratively from larger area to smaller area, and each
#            fill consumes one matching sprite.
#     - To make the two changes genuinely interact, frequency is evaluated on the
#       GLOBAL multiset of remaining sprite colors, while shape still gates which
#       colors are eligible for a particular hole. A color can be chosen for a hole
#       only if at least one remaining sprite of that hole's shape has that color.
#   concretization:
#     - Example:
#         large hole H1 matches shape S1 with candidate sprite colors {2, 7}
#         medium hole H2 matches shape S2 with candidate sprite colors {2, 8}
#         small hole H3 matches shape S3 with candidate sprite colors {7, 8}
#         initial global counts: color2=2, color7=4, color8=5
#       Then:
#         H1 picks color2 (globally rarest eligible), consuming one S1/color2 sprite.
#         counts become color2=1, color7=4, color8=5
#         H2 still picks color2, consuming it again.
#         counts become color2=0, color7=4, color8=5
#         H3 cannot use color2 at all; among {7,8} it now picks color7.
#       Original task would have used each shape's local majority independently and
#       would never let early large holes exhaust a rare color for later shapes.
#   behavior_trace:
#     - Original:
#         each hole family is solved independently by per-shape plurality color.
#     - D3 only:
#         each hole family is still independent, but uses per-shape minority color.
#     - Fused combo here:
#         holes are processed in descending area order; each chosen sprite reduces the
#         remaining global supply of that color, so later holes may be forced to fall
#         back to a different eligible color even when their local shape-specific
#         counts never changed in isolation.
#   complexity_calibration:
#     - The generator uses only 3-4 active sprite colors and gives each hole shape
#       exactly two eligible colors. This keeps the counting problem human-trackable.
#     - Colors are deliberately reused across multiple shapes, otherwise D16 would
#       degenerate into independent per-shape bookkeeping.
#     - At least three distinct hole areas are required, so "fill larger holes first"
#       is visible in the examples instead of hiding behind ties.
#   implicit_assumption_audit:
#     - Frame detection as the unique largest monochrome non-bg component still holds.
#     - Hole extraction as 4-connected non-frame cells inside the frame bbox still holds.
#     - Rotation/flip matching still holds, but the old `sprites_by_key[key][color]`
#       tally is no longer sufficient by itself: the solver must also maintain a
#       global color counter that shrinks after each hole is filled.
#     - The old assumption "all holes of one shape receive the same color" is invalid.
#       Under the fused rule they often still do, but the solver must not rely on it.
#     - Area ties are resolved deterministically by top-left scan order. The generator
#       reduces reliance on this by demanding several distinct hole areas.
#   placement_legality:
#     - Holes remain border-separated, in bounds, and 4-disconnected inside the frame.
#     - Sprites are scattered fully outside the frame bbox with Chebyshev padding, so
#       connected-component parsing remains stable.
#     - Each shape gets at least as many matching sprites as there are holes of that
#       shape, ensuring the iterative consumer never runs out mid-solve.
#   visual_observability:
#     - The same color visibly appears on multiple different sprite shapes, so the
#       solver cannot treat each shape as an isolated vote.
#     - Samples are accepted only if the combo output differs from:
#         * the original per-shape majority baseline,
#         * the D3-only per-shape minority baseline,
#         * a "least frequent but non-consuming" baseline,
#         * and a "consume but choose most frequent" baseline.
#       These checks ensure both the minority inversion and the iterative consumption
#       materially affect the generated examples.
#   visual_cleanliness:
#     - No extra legend or marker layer is introduced.
#     - Outside objects remain sparse monochrome sprites on a plain background.
#     - The output is still just the cropped frame filled with colors, so the task
#       stays readable despite the more involved counting rule.
#   new_solver_assumptions:
#     - main() now assumes that every hole shape has at least one remaining matching
#       sprite when it is reached in descending-area order.
#     - generate_input() guarantees this by allocating per-shape sprite totals
#       strictly greater than or equal to hole multiplicity.

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)


NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
MIN_DISTINCT_AREAS = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering / fused behavior
ACTIVE_COLOR_OPTIONS = (3,)
EXTRA_SPRITES_RANGE = (1, 2)
SPRITE_PADDING = 1
MAX_PLAN_ATTEMPTS = 12

# Generation retry budget
MAX_GEN_ATTEMPTS = 500


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    base = mask.astype(bool)
    for flip in (False, True):
        flipped = np.flipud(base) if flip else base
        for k in range(4):
            oriented = np.rot90(flipped, k)
            yield (oriented.shape, oriented.tobytes()), oriented


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen: set[tuple[tuple[int, int], bytes]] = set()
    result: list[np.ndarray] = []
    for key, oriented in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(oriented)
    return result


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 0], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 1, 1, 1]],
        [[1, 0], [1, 1], [0, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1], [0, 1], [0, 1]],
    ]
    seen: set[tuple[tuple[int, int], bytes]] = set()
    dedup: list[np.ndarray] = []
    for shape in raw:
        mask = np.array(shape, dtype=bool)
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4-fold symmetric mirror positions (deduplicated)."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _try_place_quadrant(occ, shape, th, tw):
    """Place shape in the top-left quadrant, mirrored into all quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on the horizontal center line and mirror left-right."""
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
    """Place on the vertical center line and mirror up-down."""
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    """Place exactly at the center. Requires odd height and odd width."""
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap, 4-disconnected from existing holes."""
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc]:
                        if (nr, nc) not in new_cells:
                            return None
    return positions


def _try_place_symmetric(occ, shape, th, tw):
    """Try a random symmetry-preserving placement mode."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    """Mark hole cells as False in the frame occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place symmetric holes. Returns (occupancy_mask, hole_placements) or None."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda shape: int(np.sum(shape)))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for shape in available:
        if len(hole_placements) >= k_target:
            break
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed.append(positions)
                    break
        if placed:
            hole_placements.append((shape, placed))

    if len(hole_placements) < MIN_SHAPES:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(pos_group) for _, placed_groups in hole_placements for pos_group in placed_groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    if len({int(np.sum(shape)) for shape, _ in hole_placements}) < MIN_DISTINCT_AREAS:
        return None

    return occ, hole_placements


def _extract_hole_records_from_sub(sub: np.ndarray, frame_col: int):
    """Return holes sorted by descending area, then top-left."""
    hole_mask = sub != int(frame_col)
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(
            {
                "key": _canonical_key(mask),
                "pos": (int(hr0), int(hc0)),
                "mask": mask,
                "area": int(mask.sum()),
            }
        )
    holes.sort(key=lambda hole: (-int(hole["area"]), hole["pos"][0], hole["pos"][1]))
    return holes


def _extract_sprite_counts(grid: np.ndarray, bg: int, frame_bbox):
    """Return per-shape color counts and global color totals for sprites outside the frame."""
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = np.array(grid, dtype=int).copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    per_shape: dict[tuple[tuple[int, int], bytes], Counter] = {}
    global_counts: Counter[int] = Counter()
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in grid[coords[:, 0], coords[:, 1]]]
        color = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = _canonical_key(mask)
        per_shape.setdefault(key, Counter())[color] += 1
        global_counts[color] += 1

    return per_shape, global_counts


def _simulate_fill_sequence(
    holes,
    per_shape_counts,
    global_counts,
    *,
    rule: str,
    scope: str,
    consume: bool,
):
    """
    Simulate one policy over the ordered holes.

    rule:
      - "least" or "most"
    scope:
      - "global": compare candidate colors by current global counts
      - "shape": compare candidate colors by current per-shape counts
    consume:
      - if True, remove one chosen sprite after each hole
    """
    local_remaining = {key: Counter(counts) for key, counts in per_shape_counts.items()}
    global_remaining = Counter(global_counts)
    choices: list[int] = []

    for hole in holes:
        key = hole["key"]
        cmap = local_remaining.get(key)
        assert cmap is not None
        candidates = [color for color, count in cmap.items() if count > 0]
        assert candidates

        if scope == "global":
            if rule == "least":
                chosen = min(candidates, key=lambda color: (int(global_remaining[color]), int(color)))
            else:
                chosen = max(candidates, key=lambda color: (int(global_remaining[color]), -int(color)))
        else:
            if rule == "least":
                chosen = min(candidates, key=lambda color: (int(cmap[color]), int(color)))
            else:
                chosen = max(candidates, key=lambda color: (int(cmap[color]), -int(color)))

        choices.append(int(chosen))
        if consume:
            cmap[chosen] -= 1
            global_remaining[chosen] -= 1

    return choices


def _render_from_choices(holes, th: int, tw: int, frame_col: int, choices):
    """Render the cropped output canvas from ordered hole choices."""
    out = np.full((th, tw), int(frame_col), dtype=int)
    for hole, color in zip(holes, choices):
        patch = np.full(hole["mask"].shape, int(color), dtype=int)
        paste(out, patch, hole["pos"][0], hole["pos"][1], mask=hole["mask"])
    return out


def _extract_scene(grid: np.ndarray):
    """Shared parsing used by main() and validation helpers."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    frame_bbox = tuple(map(int, frame["bbox"]))
    sub = crop_bbox(g, frame_bbox)
    th, tw = map(int, sub.shape)

    holes = _extract_hole_records_from_sub(sub, frame_col)
    per_shape_counts, global_counts = _extract_sprite_counts(g, bg, frame_bbox)
    return g, bg, frame_col, frame_bbox, th, tw, holes, per_shape_counts, global_counts


def _behavior_report(grid: np.ndarray):
    """Diagnostic helper used for anti-degeneracy tests."""
    _g, _bg, frame_col, _fb, th, tw, holes, per_shape_counts, global_counts = _extract_scene(grid)

    combo = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="global", consume=True
    )
    orig = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="most", scope="shape", consume=False
    )
    d3_only = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="shape", consume=False
    )
    no_consume = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="global", consume=False
    )
    global_max = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="most", scope="global", consume=True
    )

    combo_out = _render_from_choices(holes, th, tw, frame_col, combo)
    orig_out = _render_from_choices(holes, th, tw, frame_col, orig)
    d3_out = _render_from_choices(holes, th, tw, frame_col, d3_only)
    no_consume_out = _render_from_choices(holes, th, tw, frame_col, no_consume)
    global_max_out = _render_from_choices(holes, th, tw, frame_col, global_max)

    color_to_shapes: dict[int, set[tuple[tuple[int, int], bytes]]] = {}
    for key, cmap in per_shape_counts.items():
        for color, count in cmap.items():
            if count > 0:
                color_to_shapes.setdefault(int(color), set()).add(key)

    return {
        "hole_count": len(holes),
        "distinct_hole_areas": len({int(hole["area"]) for hole in holes}),
        "diff_vs_original": int(np.sum(combo_out != orig_out)),
        "diff_vs_d3_only": int(np.sum(combo_out != d3_out)),
        "diff_vs_no_consume": int(np.sum(combo_out != no_consume_out)),
        "diff_vs_global_max": int(np.sum(combo_out != global_max_out)),
        "shared_color_count": int(sum(len(keys) >= 2 for keys in color_to_shapes.values())),
        "combo_fill_colors": int(len(set(combo))),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame interior by matching hole shapes to outside sprites under an iterative rarity rule.

    Input structure:
    - The background is the most common color.
    - One large monochrome frame encloses several bg-colored holes of different shapes.
    - Outside the frame are scattered monochrome sprites; each sprite shape matches one hole family up to rotation or reflection, and colors are reused across multiple families.

    Rule:
    - Crop the output to the frame's bounding box and start from a solid fill of the frame color.
    - Process holes from larger area to smaller area, breaking equal-area ties by top-left position.
    - For the current hole, look only at remaining sprites whose shape matches that hole. Among their colors, choose the one whose GLOBAL remaining sprite count is smallest, then consume one matching sprite of that chosen color.

    Key insight:
    - Shape still determines which sprites are eligible, but color is resolved by a global shrinking inventory. Early large holes can spend a rare color and force later smaller holes to switch to a different eligible color.
    """
    _g, _bg, frame_col, _frame_bbox, th, tw, holes, per_shape_counts, global_counts = _extract_scene(grid)
    choices = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="global", consume=True
    )
    return _render_from_choices(holes, th, tw, frame_col, choices)


def _build_group_records(hole_placements):
    """Convert generator placements into per-shape group records and ordered hole records."""
    groups = []
    holes = []
    for shape, placed_groups in hole_placements:
        positions: list[tuple[int, int]] = []
        for pos_group in placed_groups:
            for pr, pc in pos_group:
                positions.append((int(pr), int(pc)))
        positions.sort()
        key = _canonical_key(shape)
        area = int(np.sum(shape))
        group = {
            "key": key,
            "shape": shape,
            "positions": positions,
            "area": area,
            "hole_count": len(positions),
        }
        groups.append(group)
        for pos in positions:
            holes.append({"key": key, "mask": shape, "pos": pos, "area": area})
    holes.sort(key=lambda hole: (-int(hole["area"]), hole["pos"][0], hole["pos"][1]))
    return groups, holes


def _sample_shape_counts(groups, bg: int, frame_col: int):
    """Construct a shared-color plan that almost always exercises iterative depletion."""
    palette = [color for color in range(NUM_COLORS) if color not in {bg, frame_col}]
    n_active = min(random.choice(ACTIVE_COLOR_OPTIONS), len(palette))
    if n_active < 3:
        return None
    active = random.sample(palette, n_active)
    a, b, c = active[:3]
    pair_cycle = [((a, c), a, c), ((a, b), a, b), ((b, c), b, c)]

    ordered_groups = sorted(groups, key=lambda group: (-int(group["area"]), -int(group["hole_count"])))

    per_shape_counts: dict[tuple[tuple[int, int], bytes], Counter] = {}
    for idx, group in enumerate(ordered_groups):
        pair, rare_color, common_color = pair_cycle[idx % len(pair_cycle)]
        hole_count = int(group["hole_count"])
        rare_count = 1
        common_count = max(2, hole_count) + random.randint(*EXTRA_SPRITES_RANGE) - 1
        if common_count <= rare_count:
            common_count = rare_count + 1
        if random.random() < 0.25:
            rare_color, common_color = common_color, rare_color
            rare_count, common_count = rare_count, common_count
        per_shape_counts[group["key"]] = Counter(
            {int(rare_color): int(rare_count), int(common_color): int(common_count)}
        )

    global_counts: Counter[int] = Counter()
    for cmap in per_shape_counts.values():
        global_counts.update(cmap)
    return per_shape_counts, global_counts


def _plan_metrics(groups, holes, occ, per_shape_counts, global_counts, frame_col: int):
    """Score whether a sampled plan visibly exercises the fused behavior."""
    th, tw = occ.shape
    combo = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="global", consume=True
    )
    orig = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="most", scope="shape", consume=False
    )
    d3_only = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="shape", consume=False
    )
    no_consume = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="least", scope="global", consume=False
    )
    global_max = _simulate_fill_sequence(
        holes, per_shape_counts, global_counts, rule="most", scope="global", consume=True
    )

    combo_out = _render_from_choices(holes, th, tw, frame_col, combo)
    orig_out = _render_from_choices(holes, th, tw, frame_col, orig)
    d3_out = _render_from_choices(holes, th, tw, frame_col, d3_only)
    no_consume_out = _render_from_choices(holes, th, tw, frame_col, no_consume)
    global_max_out = _render_from_choices(holes, th, tw, frame_col, global_max)

    color_to_shapes: dict[int, set[tuple[tuple[int, int], bytes]]] = {}
    for key, cmap in per_shape_counts.items():
        for color, count in cmap.items():
            if count > 0:
                color_to_shapes.setdefault(int(color), set()).add(key)

    metrics = {
        "diff_vs_original": int(np.sum(combo_out != orig_out)),
        "diff_vs_d3_only": int(np.sum(combo_out != d3_out)),
        "diff_vs_no_consume": int(np.sum(combo_out != no_consume_out)),
        "diff_vs_global_max": int(np.sum(combo_out != global_max_out)),
        "shared_color_count": int(sum(len(keys) >= 2 for keys in color_to_shapes.values())),
        "combo_fill_colors": int(len(set(combo))),
    }

    ok = (
        metrics["diff_vs_original"] > 0
        and metrics["diff_vs_d3_only"] > 0
        and metrics["diff_vs_no_consume"] > 0
        and metrics["diff_vs_global_max"] > 0
        and metrics["shared_color_count"] >= 1
        and metrics["combo_fill_colors"] >= 2
    )
    return ok, combo_out, metrics


def _place_sprite(inp, placement, ori, color, bg):
    """Place one sprite on the input canvas and occupancy tracker."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, placement, groups, per_shape_counts, bg):
    """Scatter all planned sprites outside the frame."""
    for group in groups:
        orientations = _all_orientations(group["shape"])
        cmap = per_shape_counts[group["key"]]
        for color, count in cmap.items():
            for _ in range(int(count)):
                if not _place_sprite(inp, placement, random.choice(orientations), int(color), bg):
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt validated by the solver."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([color for color in range(NUM_COLORS) if color != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout
    groups, holes = _build_group_records(hole_placements)

    for _ in range(MAX_PLAN_ATTEMPTS):
        sampled = _sample_shape_counts(groups, bg, frame_col)
        if sampled is None:
            continue
        per_shape_counts, global_counts = sampled
        ok, combo_out, _metrics = _plan_metrics(groups, holes, occ, per_shape_counts, global_counts, frame_col)
        if not ok:
            continue

        fr = random.randint(0, GRID_SIZE - th)
        fc = random.randint(0, GRID_SIZE - tw)
        inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        frame_patch = np.where(occ, frame_col, bg).astype(int)
        paste(inp, frame_patch, fr, fc)

        placement = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        placement[fr : fr + th, fc : fc + tw] = int(frame_col)
        if not _scatter_sprites(inp, placement, groups, per_shape_counts, bg):
            continue

        try:
            if np.array_equal(main(inp), combo_out):
                return inp
        except Exception:
            pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
