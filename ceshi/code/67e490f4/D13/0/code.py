# Variant of 67e490f4
# Dimension: D13 — 精灵与框架的空间关系 (meta)
# Change: 精灵完全位于框架 bbox 外部，与框架和其他精灵之间保持至少 1 像素的切比雪夫距离间隔... → 精灵可紧贴框架边界

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
# ---------------------------------------------------------------------------
# spec_variant:
#   core_change:
#     dimension: D13 (sprite-to-frame spacing)
#     old_behavior:
#       - every outside sprite stayed at least one Chebyshev pixel away from
#         the frame bbox
#       - solver could rely on visual whitespace, even if it did not state that
#     new_behavior:
#       - sprites still stay fully outside the frame bbox
#       - sprites still stay mutually separated by >=1 Chebyshev pixel
#       - but some sprites are now placed flush against a frame side, with zero
#         gap between the sprite bbox and the frame bbox
#   concrete_effect:
#     - a 2x2 sprite may sit immediately above the top border, or a 3x1 sprite
#       may sit immediately to the left of the frame, sharing an edge but never
#       entering the bbox
#     - output rule is unchanged: matched sprite shape chooses the color for the
#       corresponding hole shape
#   behavior_trace:
#     old_pipeline:
#       - generator uses padding=1 around the frame, so a sprite can never be at
#         row r0-1..r0 if the frame bbox starts at r0
#     new_pipeline:
#       - generator explicitly forbids bbox overlap with the frame but does not
#         reserve empty space around it
#       - e.g. if frame bbox is rows 8..16, a 2x3 sprite may be placed at rows
#         6..7 and cols 10..12, so its bottom edge touches the frame top edge
#   complexity_calibration:
#     - only the frame-gap constraint changes; hole shapes, color plurality, and
#       rotation/flip matching stay the same
#     - at least 2 shapes per sample get a winning sprite that must touch a
#       frame side, making the variant visible in a few examples
#     - sprite-sprite padding is preserved, so touching sprites do not merge
#   implicit_assumptions_review:
#     - main/find frame: largest component alone is slightly brittle; the new
#       solver therefore filters for a monochrome component whose bbox has a full
#       border of one color and contains interior non-frame cells
#     - main/erase frame bbox before sprite extraction: this only remains valid
#       if every sprite cell is outside the frame bbox; generator enforces bbox
#       disjointness
#     - main/find sprite components with monochromatic=False: this assumes
#       sprites do not touch each other; generator keeps the old sprite-sprite
#       Chebyshev padding
#   placement_legality:
#     - frame is kept at least one cell away from the outer canvas edge so all
#       four sides can potentially host touching sprites
#     - a sprite candidate is legal iff:
#         1) its tight bbox is fully in-grid
#         2) its bbox is disjoint from the frame bbox
#         3) if required, its bbox side-touches the frame bbox
#         4) every sprite pixel is >=1 Chebyshev cell away from every existing
#            sprite pixel
#   visual_observability:
#     - touching is implemented as side contact, not only corner contact, so the
#       difference from the original task is easy to see
#     - colors still encode plurality only; no invisible ordering rule is added
#   new_solver_assumptions_to_match_in_generator:
#     - exactly one large frame-like monochrome component exists
#     - frame border is intact and holes stay inside the border
#     - every hole shape has at least one matching sprite component outside the
#       bbox


def _all_mask_keys(mask: np.ndarray):
    """Generate keys for all rotations and vertical-flip variants of a mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _is_frame_component(grid: np.ndarray, comp: dict) -> bool:
    """Frame = monochrome component with a solid rectangular border and holes inside."""
    color = int(comp["color"])
    sub = crop_bbox(grid, comp["bbox"])
    if sub.shape[0] < 3 or sub.shape[1] < 3:
        return False
    border_ok = (
        np.all(sub[0, :] == color)
        and np.all(sub[-1, :] == color)
        and np.all(sub[:, 0] == color)
        and np.all(sub[:, -1] == color)
    )
    return bool(border_ok and np.any(sub != color))


def _find_frame(grid: np.ndarray, bg: int) -> dict:
    comps = find_components(grid, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    candidates = [comp for comp in comps if _is_frame_component(grid, comp)]
    assert candidates
    return max(candidates, key=lambda comp: (int(comp["area"]), _bbox_area(comp["bbox"])))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the frame interior by using nearby outside sprites as shape-coded color keys.

    Input structure:
    - The background is the most common color, with one large monochrome frame whose border is intact.
    - Inside that frame bbox, holes appear as non-frame-color regions.
    - Outside the bbox, small sprites may now sit flush against the frame boundary, although they never enter the bbox or touch each other.

    Rule:
    - Detect the frame by its solid rectangular border and crop to its bounding box.
    - Treat each hole shape inside the bbox as a key, then search outside sprites for the same shape under rotation or flip.
    - Fill each hole with the most frequent color among its matching sprites and output only the cropped frame rectangle.

    Key insight:
    - The hard part is separating "frame border" from "sprite touching the border"; the bbox itself is still the clean divider, so sprite shapes remain color labels for hole shapes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the frame by structure, not by empty margin around it.
    frame = _find_frame(g, bg)
    frame_col = int(frame["color"])
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
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

    # STEP 3) Extract sprites from outside the frame bbox only.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
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
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output: solid frame color, then fill each hole by matched sprite plurality.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)
FRAME_MARGIN = 1

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_SPRITE_PADDING = 1
TOUCHING_SHAPE_RANGE = (2, 4)

MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
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
    for s in raw:
        mask = np.array(s, dtype=bool)
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
    """Check all positions: in bounds, no overlap with existing holes, 4-disconnected."""
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
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        available.sort(key=lambda s: int(np.sum(s)))

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

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    """Render the target output canvas and assign fill colors."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, placements), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in placements:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _interval_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _bboxes_disjoint(a, b) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0


def _bbox_side_touches(a, b) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return bool(
        (ar1 + 1 == br0 and _interval_overlap(ac0, ac1, bc0, bc1))
        or (br1 + 1 == ar0 and _interval_overlap(ac0, ac1, bc0, bc1))
        or (ac1 + 1 == bc0 and _interval_overlap(ar0, ar1, br0, br1))
        or (bc1 + 1 == ac0 and _interval_overlap(ar0, ar1, br0, br1))
    )


def _can_place_sprite(placement, mask_coords, patch_shape, top, left, frame_bbox, bg, require_touch):
    h, w = map(int, patch_shape)
    H, W = placement.shape
    if top < 0 or left < 0 or top + h > H or left + w > W:
        return False

    sprite_bbox = (top, left, top + h - 1, left + w - 1)
    if not _bboxes_disjoint(sprite_bbox, frame_bbox):
        return False
    if require_touch and not _bbox_side_touches(sprite_bbox, frame_bbox):
        return False

    for dr, dc in mask_coords:
        rr = int(top + dr)
        cc = int(left + dc)
        for nr in range(rr - SPRITE_SPRITE_PADDING, rr + SPRITE_SPRITE_PADDING + 1):
            for nc in range(cc - SPRITE_SPRITE_PADDING, cc + SPRITE_SPRITE_PADDING + 1):
                if 0 <= nr < H and 0 <= nc < W and int(placement[nr, nc]) != int(bg):
                    return False
    return True


def _find_sprite_location(placement, patch, frame_bbox, bg, require_touch):
    mask_coords = np.argwhere(patch != int(bg))
    h, w = patch.shape
    candidates = []
    for top in range(0, placement.shape[0] - h + 1):
        for left in range(0, placement.shape[1] - w + 1):
            if _can_place_sprite(placement, mask_coords, patch.shape, top, left, frame_bbox, bg, require_touch):
                candidates.append((int(top), int(left)))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_sprite(inp, placement, ori, color, bg, frame_bbox, require_touch):
    """Place one monochrome sprite while keeping it outside the frame bbox."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    loc = _find_sprite_location(placement, patch, frame_bbox, bg, require_touch)
    if loc is None:
        return False
    r, c = loc
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col, frame_bbox):
    """Place sprites outside the frame bbox, forcing several to side-touch the frame."""
    placement = np.full_like(inp, bg)
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    plans = []
    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, min(n_other, len(losers)))
        loser_counts = {oc: random.randint(1, max(1, n_winning - 1)) for oc in oc_list}
        plans.append(
            {
                "shape": shape,
                "fill_color": fill_color,
                "orientations": orientations,
                "n_winning": n_winning,
                "loser_counts": loser_counts,
            }
        )

    max_touch = min(TOUCHING_SHAPE_RANGE[1], len(plans))
    min_touch = min(TOUCHING_SHAPE_RANGE[0], max_touch)
    n_touch = random.randint(min_touch, max_touch)
    touch_indices = set(random.sample(range(len(plans)), n_touch))

    touch_order = list(touch_indices)
    random.shuffle(touch_order)
    for idx in touch_order:
        plan = plans[idx]
        oris = random.sample(plan["orientations"], len(plan["orientations"]))
        if not any(
            _place_sprite(inp, placement, ori, plan["fill_color"], bg, frame_bbox, require_touch=True)
            for ori in oris
        ):
            return False
        plan["n_winning"] -= 1

    plan_order = list(range(len(plans)))
    random.shuffle(plan_order)
    for idx in plan_order:
        plan = plans[idx]
        for _ in range(plan["n_winning"]):
            oris = random.sample(plan["orientations"], len(plan["orientations"]))
            if not any(
                _place_sprite(inp, placement, ori, plan["fill_color"], bg, frame_bbox, require_touch=False)
                for ori in oris
            ):
                return False
        for oc, count in plan["loser_counts"].items():
            for _ in range(count):
                oris = random.sample(plan["orientations"], len(plan["orientations"]))
                if not any(
                    _place_sprite(inp, placement, ori, oc, bg, frame_bbox, require_touch=False)
                    for ori in oris
                ):
                    break

    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build frame, place touching sprites, validate."""
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

    fr = random.randint(FRAME_MARGIN, GRID_SIZE - th - FRAME_MARGIN)
    fc_pos = random.randint(FRAME_MARGIN, GRID_SIZE - tw - FRAME_MARGIN)
    frame_bbox = (fr, fc_pos, fr + th - 1, fc_pos + tw - 1)

    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc_pos)

    if not _scatter_sprites(inp, shape_info, bg, frame_col, frame_bbox):
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
