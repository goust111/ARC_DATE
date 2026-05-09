# Combo variant of 67e490f4 (index 8)
# combo_spec: D2/1+D7/2
# Name: 仅旋转匹配+对称洞

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


# Design analysis for combo D2/1 + D7/2
# --------------------------------------
# spec_variant:
#   fused_rule:
#     - Matching now uses rotations only; reflections are forbidden.
#     - Every hole shape must have non-trivial rotational symmetry, but some of those shapes are still
#       chiral under reflection, such as S/Z-like zigzags with 180-degree symmetry.
#   concrete_pixel_behavior:
#     - Example pair:
#         A = [[0,1,1],
#              [1,1,0]]
#         B = [[1,1,0],
#              [0,1,1]]
#       Both A and B are invariant under 180-degree rotation, so both satisfy the new "symmetric hole"
#       constraint. They are mirror images of each other but not rotation-equivalent.
#     - Old solver behavior: A and B sprites would be pooled because flip-aware matching treated the two
#       handedness classes as equivalent. If color 2 appears on A-sprites twice and color 7 appears on
#       B-sprites twice, both A- and B-holes collapse to the same pooled plurality outcome.
#     - New combo behavior: A-holes only look at A-sprites up to rotation; B-holes only look at B-sprites
#       up to rotation. The two hole families can therefore receive different fill colors even though each
#       family is individually rotationally symmetric.
#   complexity_calibration:
#     - Restrict the library to a small set of rotationally symmetric shapes so the new D7 constraint is
#       visible and samples stay readable.
#     - Force at least one chiral rotational-symmetry pair per sample; otherwise the "no reflection"
#       change could become observationally inert.
#     - Keep 3-6 distinct shapes with repeated hole instances so humans can induce the geometry-to-color
#       mapping from a few examples instead of memorizing one-off objects.
#   implicit_assumption_review:
#     - main() step 1 still assumes the frame is the unique largest monochrome non-background component;
#       the generator preserves that.
#     - main() step 2 relies on exact hole masks inside the frame bbox. This remains valid because the
#       holes are generated directly from the intended rotationally symmetric masks and stay 4-disconnected.
#     - main() step 3 used to depend on flip augmentation. That assumption is intentionally broken here:
#       sprites are indexed only by their 4 rotations, so mirror partners stay distinct.
#     - main() step 4 assumes each hole key has at least one outside match. The generator guarantees this
#       shape-by-shape and also guarantees distinct winning colors for the required chiral pair.
#   placement_legality:
#     - Holes stay strictly inside the frame border, never overlap, and remain 4-disconnected.
#     - Sprites are placed fully outside the frame bbox with Chebyshev padding 1, so outside components
#       do not merge and monochrome sprite colors remain visually separable.
#   visual_observability:
#     - Every sample contains at least one rotationally symmetric mirror pair such as S/Z with different
#       winning colors and equal winning counts. A flip-aware interpretation would merge them, but the
#       intended rotation-only rule keeps them separate.
#     - The remaining shapes are also rotationally symmetric, so the D7 restriction is visible across the
#       whole frame rather than on a single special case.
#   cleanliness:
#     - No extra markers are introduced; all evidence lives in the holes and in the scattered sprite keys.
#     - The palette is still limited to the standard ARC colors, with one distinct fill color per shape.


def _all_rotations(mask: np.ndarray):
    """Yield (key, oriented_mask) for the 4 rotational variants of a bool mask."""
    m0 = mask.astype(bool)
    for k in range(4):
        m = np.rot90(m0, k)
        yield (m.shape, m.tobytes()), m


def _rotation_key(mask: np.ndarray):
    """Canonical representative under rotations only."""
    return min(key for key, _ in _all_rotations(mask))


def _all_rotation_keys(mask: np.ndarray):
    """Generate keys for all rotations of a boolean mask, without any flip."""
    for key, _ in _all_rotations(mask):
        yield key


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotations of a mask."""
    seen, result = set(), []
    for key, m in _all_rotations(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _mirror_mask(mask: np.ndarray) -> np.ndarray:
    """One mirror operator is enough to span the reflected handedness class."""
    return np.fliplr(mask.astype(bool))


def _mirror_key(mask: np.ndarray):
    """Canonical key of the mirrored partner under rotation-only equivalence."""
    return _rotation_key(_mirror_mask(mask))


def _has_nontrivial_rotation_symmetry(mask: np.ndarray) -> bool:
    """True iff the mask equals one of its 90/180/270 degree rotations."""
    m = mask.astype(bool)
    return any(np.array_equal(m, np.rot90(m, k)) for k in (1, 2, 3))


def _is_chiral(mask: np.ndarray) -> bool:
    """True iff the mask is not equivalent to its mirror under rotations alone."""
    return _rotation_key(mask) != _mirror_key(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the holes inside one frame by using outside sprites as rotation-only shape keys.

    Input structure:
    - The background is the most common color. One large monochrome frame sits on that background.
    - Inside the frame bbox are several holes, each shown as a non-frame-color region.
    - Outside the frame are small monochrome sprites whose shapes encode which hole family they refer to.

    Rule:
    - Crop to the frame bounding box and start from a solid block of the frame color.
    - For each hole, gather outside sprites whose masks match that hole under rotation only.
    - Fill the hole with the most frequent matching sprite color; mirror-image sprites do not count unless
      they are already identical by rotation.

    Key insight:
    - All holes are rotationally symmetric, but some are still chiral under reflection. The outside sprites
      therefore act as handed shape labels: two mirror-related symmetric holes can receive different colors
      because flipping is forbidden while rotating is allowed.
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

    # STEP 3) Extract sprites outside the bbox and index them by rotation only.
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

        for key in _all_rotation_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Render output: start from solid frame color, then fill each hole by shape match.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (rotation-only + rotationally symmetric holes) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (8, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 80
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Sprite scattering
WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 500


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[1, 1, 1, 1], [1, 1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1, 1], [1, 1, 1, 0]],
        [[1, 1, 1, 0], [0, 1, 1, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        mask = np.array(s, dtype=bool)
        if int(np.sum(mask)) > SHAPE_MAX_AREA:
            continue
        if not _has_nontrivial_rotation_symmetry(mask):
            continue
        key = _rotation_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()
SHAPE_BY_KEY = {_rotation_key(shape): shape for shape in SHAPE_LIBRARY}


def _build_chiral_pairs():
    pairs = []
    seen = set()
    for shape in SHAPE_LIBRARY:
        if not _is_chiral(shape):
            continue
        key = _rotation_key(shape)
        mirror = _mirror_key(shape)
        if mirror not in SHAPE_BY_KEY:
            continue
        pair_id = tuple(sorted((key, mirror)))
        if pair_id in seen:
            continue
        seen.add(pair_id)
        pairs.append((SHAPE_BY_KEY[pair_id[0]], SHAPE_BY_KEY[pair_id[1]]))
    return pairs


CHIRAL_PAIRS = _build_chiral_pairs()


def _mirror_pos(r, c, sh, sw, th, tw):
    """Return the 4 mirror positions used by the placement scheme."""
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ, shape, th, tw):
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
    c_m = tw - sw - c
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ, shape, th, tw):
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
    r_m = th - sh - r
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ, shape, th, tw):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r, c = th // 2 - sh // 2, tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _validate_positions(occ, shape, positions, th, tw):
    """Check all positions: in bounds, no overlap with existing holes, 4-disconnected from them."""
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
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ, shape, positions):
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _multiplicity_weights(th, tw):
    return MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS


def _build_hole_layout(th, tw):
    """Place only rotationally symmetric hole shapes and force one chiral mirror pair."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    available_by_key = {_rotation_key(shape): shape for shape in available}
    pair_candidates = []
    seen_pairs = set()
    for shape in available:
        key = _rotation_key(shape)
        mirror = _mirror_key(shape)
        if not _is_chiral(shape) or mirror not in available_by_key:
            continue
        pair_id = tuple(sorted((key, mirror)))
        if pair_id in seen_pairs:
            continue
        seen_pairs.add(pair_id)
        pair_candidates.append((available_by_key[pair_id[0]], available_by_key[pair_id[1]]))
    if not pair_candidates:
        return None

    required_pair = random.choice(pair_candidates)
    required_keys = {_rotation_key(required_pair[0]), _rotation_key(required_pair[1])}

    others = [shape for shape in available if _rotation_key(shape) not in required_keys]
    random.shuffle(others)
    max_target = min(SHAPE_COUNT_RANGE[1], len(available))
    if max_target < MIN_SHAPES:
        return None
    k_target = random.randint(MIN_SHAPES, max_target)
    selection_order = [required_pair[0], required_pair[1], *others]

    occ = np.ones((th, tw), dtype=bool)
    hole_records = []
    placed_keys = set()
    mult_vals, mult_wts = _multiplicity_weights(th, tw)

    for shape in selection_order:
        key = _rotation_key(shape)
        must_place = key in required_keys
        if len(hole_records) >= k_target and not must_place:
            continue
        if key in placed_keys:
            continue

        mult = random.choices(mult_vals, weights=mult_wts)[0]
        if must_place:
            mult = max(1, min(mult, 2))

        placed_groups = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed_groups.append(positions)
                    break
        if must_place and not placed_groups:
            return None
        if placed_groups:
            hole_records.append({"shape": shape, "groups": placed_groups, "key": key})
            placed_keys.add(key)

    if len(hole_records) < MIN_SHAPES:
        return None
    if not required_keys.issubset(placed_keys):
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group) for rec in hole_records for group in rec["groups"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_records, required_keys


def _build_output(occ, hole_records, frame_col, bg, required_keys):
    """Render the target output canvas and assign a distinct fill color to each shape family."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_records):
        return None
    shape_colors = random.sample(avail_colors, len(hole_records))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for rec, fill_col in zip(hole_records, shape_colors):
        shape = rec["shape"]
        all_pos = []
        for positions in rec["groups"]:
            for pr, pc in positions:
                patch = np.full(shape.shape, fill_col, dtype=int)
                paste(output, patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "shape": shape,
                "fill_color": fill_col,
                "positions": all_pos,
                "key": rec["key"],
                "required_pair": rec["key"] in required_keys,
                "is_chiral": _is_chiral(shape),
            }
        )
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
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
    """
    Place sprites outside the frame.

    The required chiral pair gets equal winning counts and no losing colors, making reflection-forbidden
    matching operational: a flip-aware solver would pool both handedness classes and collapse the pair.
    """
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    paired_winning = random.randint(2, 4)

    for info in shape_info:
        shape = info["shape"]
        fill_color = info["fill_color"]
        orientations = _all_orientations(shape)

        if info["required_pair"]:
            n_winning = paired_winning
            oc_list = []
        else:
            n_winning = random.randint(*WINNING_SPRITE_RANGE)
            losers = [c for c in other_pool if c != fill_color]
            n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
            oc_list = random.sample(losers, min(n_other, len(losers)))

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_records, required_keys = layout

    result = _build_output(occ, hole_records, frame_col, bg, required_keys)
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
