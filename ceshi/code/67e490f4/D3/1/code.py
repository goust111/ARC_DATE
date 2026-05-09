# Variant of 67e490f4
# Dimension: D3 — 匹配条件：颜色消歧规则 (match_cond)
# Change: frequency_tiebreak — 当多个不同颜色的精灵匹配同一个洞形状时，选择出现频率最高的颜色填充；频率相同时... → 唯一颜色填充（无需消歧）

from __future__ import annotations

from collections import Counter
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


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# variant_analysis:
#   core_change:
#     old_rule: A hole shape could have several matching sprite colors outside the frame.
#       The solver aggregated counts per color and selected the plurality color
#       (ties fell back to smaller color id).
#     new_rule: Every hole shape is paired with exactly one sprite color.
#       Multiple sprite instances may still exist for that shape, but all of them
#       share the same color, so the solver only needs shape lookup.
#
#   a_concretization_and_trace:
#     scene:
#       - One large monochrome frame sits on the background.
#       - The frame bbox contains several 4-connected holes.
#       - Outside the bbox, small monochrome sprites act as keys.
#     old_local_trace:
#       - Suppose a hole has L-shape.
#       - Outside sprites include L-shapes in red, red, yellow.
#       - The original solver builds {red: 2, yellow: 1} for that shape and fills
#         the hole red.
#     new_local_trace:
#       - The same L-shape hole now only has, for example, blue L-sprites outside.
#       - The new solver records "L-shape -> blue" directly and fills all matching
#         L-holes blue.
#     visible_difference:
#       - Inputs no longer contain same-shape sprites in competing colors.
#       - The task becomes pure shape-to-color association under rotation/flip.
#
#   b_complexity_calibration:
#     - Keep 3-6 distinct hole shapes and 5+ total holes so the frame still needs
#       multiple correspondences rather than trivial one-off matching.
#     - Keep 2-5 sprites per shape so each color/shape pairing has repeated support.
#     - Keep distinct fill colors across shapes to make correspondences legible.
#
#   c_implicit_assumption_audit_for_main:
#     - Largest monochrome non-bg component is still the frame. Generator preserves
#       this by keeping sprites small and disconnected from the frame.
#     - Hole extraction still treats every non-frame cell in the bbox as hole area.
#       Generator still carves holes as bg within the frame.
#     - Exact hole keys remain orientation-specific, so sprite extraction must still
#       register every rotation/flip variant of each sprite mask.
#     - The changed assumption is the map built from sprite shapes:
#         old code assumed "shape -> color counts".
#         new code assumes "shape -> unique color".
#       Any second color for the same shape would violate the variant and should be
#       rejected by the solver's assertion.
#
#   d_placement_legality:
#     - Hole placement remains symmetric, inside the frame border, and 4-disconnected.
#     - Sprite placement still uses random_free_location with padding=1, so sprites
#       do not overlap each other and do not touch the frame bbox.
#     - Because only one color per shape is emitted, no post-placement ambiguity can
#       arise among matching sprites.
#
#   e_visual_observability:
#     - Across 2-4 examples, a human can see that every repeated sprite shape always
#       keeps one color, while different shapes use different colors.
#     - The removed frequency rule is observable through absence of mixed-color
#       versions of the same shape.
#
#   f_visual_cleanliness:
#     - No extra marks are added inside the frame.
#     - Outside sprites remain separated by background, so each key shape reads
#       clearly and does not merge into accidental compounds.
#
#   g_color_inference:
#     - Colors are not transformed numerically and are never chosen by visible-id
#       ordering at solve time.
#     - Each shape gets one stable color with repeated sprite instances, which gives
#       enough evidence for human inference.
#
#   h_variant_metric_used_after_implementation:
#     - For each canonical hole shape, count the number of distinct sprite colors
#       appearing outside the frame.
#     - Target distribution for this variant: always exactly 1.
#
#   i_new_main_assumptions_checked_against_generator:
#     - main() now asserts that all matching sprites for a shape agree on color.
#     - generate_input() guarantees this by assigning one fill color per shape and
#       scattering only that color for all instances of that shape.
# ---------------------------------------------------------------------------


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped frame by filling each interior hole from uniquely colored shape keys.

    Input structure:
    - The background is the most common color. One large monochrome frame sits on it.
    - Inside the frame bbox, holes appear as non-frame-color regions of various shapes.
    - Outside the frame bbox, small monochrome sprites are scattered on the background.

    Rule:
    - Crop to the frame's bounding box and start from a solid patch of the frame color.
    - Match each hole shape to an outside sprite shape, allowing rotation and reflection.
    - Every hole shape corresponds to exactly one sprite color, so fill the hole directly with that color.

    Key insight:
    - The outside sprites are shape labels with a unique attached color: shape identifies the target hole,
      and color is copied once the shape match is found.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: largest connected component of any non-bg color (8-connected).
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = frame["bbox"]
    r0, c0, r1, c1 = map(int, fb)

    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    # STEP 2) Extract hole components inside the bbox (anything not equal to frame color), 4-connected.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        key = (mask.shape, mask.tobytes())
        holes.append((key, (int(hr0), int(hc0)), mask))

    # STEP 3) Extract sprites outside the bbox and map each shape key to its unique color.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    def _all_mask_keys(mask: np.ndarray):
        """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

    sprites_by_key: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(mask):
            prev = sprites_by_key.get(key)
            if prev is None:
                sprites_by_key[key] = scol
            else:
                assert prev == scol

    # STEP 4) Render output: solid frame color with each hole filled by its uniquely matched sprite color.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        scol = sprites_by_key.get(key)
        assert scol is not None
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

# ── Configuration ──────────────────────────────────────────────────────
NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)  # distinct canonical hole shapes
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
SPRITES_PER_SHAPE_RANGE = (2, 5)  # repeated evidence for the unique shape->color mapping
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants (canonical representative)."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
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
    seen, dedup = set(), []
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
    """Place shape in top-left quadrant, mirror to all 4 quadrants."""
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
    """Place on horizontal center line, mirror LR only (2 copies)."""
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
    """Place on vertical center line, mirror UD only (2 copies)."""
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
    """Place at dead center (1 copy). Requires both dims odd."""
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap with existing holes, and 4-disconnected."""
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
    """Try a random symmetric placement type."""
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    """Mark hole cells as unoccupied in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th, tw):
    """Place holes with LR+UD+rot180 symmetry. Returns (occ, hole_placements) or None."""
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

    total_holes = sum(len(positions) for _, placement_groups in hole_placements for positions in placement_groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """Render the target output canvas and assign one distinct fill color per shape."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, placement_groups), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in placement_groups:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    """Place a single sprite on inp and placement grids. Returns True on success."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """Place repeated monochrome sprites so each hole shape appears in exactly one color."""
    placement = inp.copy()
    _ = frame_col  # kept for signature parity with the original generator

    for shape, fill_color, _positions in shape_info:
        orientations = _all_orientations(shape)
        n_sprites = random.randint(*SPRITES_PER_SHAPE_RANGE)
        for _ in range(n_sprites):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build symmetric frame, scatter unique-color sprites, validate."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    result = _build_output(occ, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    try:
        if np.array_equal(main(inp), output):
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
