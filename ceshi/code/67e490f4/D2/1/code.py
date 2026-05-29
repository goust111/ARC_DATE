# Variant of 67e490f4
# Dimension: D2 — 匹配条件：旋转/翻转的允许范围 (match_cond)
# Change: rotation — 匹配时允许 8 种旋转/翻转变体（0/90/180/270度旋转 x 是否垂直翻转）... → 仅允许旋转不允许翻转

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


# Design analysis for the D2 variant
# ----------------------------------
# spec_variant:
#   unchanged_backbone:
#     - 30x30 input grid with one large monochrome frame and scattered outside sprites.
#     - Output is still cropped to the frame bbox and starts as a solid frame-color block.
#     - Each hole is still recolored from the plurality color of shape-matching outside sprites.
#   core_change:
#     old_match_cond: rotations + reflections were treated as equivalent.
#     new_match_cond: only 4 rotations are equivalent; mirror images are distinct.
#   concrete_pixel_behavior:
#     - A left-handed L hole can be filled only by left-handed L sprites under some rotation.
#     - A right-handed L sprite no longer matches that hole unless it becomes identical by rotation alone.
#     - To make this visible, the generator must include at least one chiral pair:
#       two hole shapes that are mirror images of each other but not rotation-equivalent.
#   traced_example:
#     - Hole A = [[1,0],[1,1]], Hole B = [[0,1],[1,1]], with colors 2 and 7 outside.
#     - Original logic pooled both sprite families under one canonical flip-aware key, so both holes
#       could inherit the same plurality color.
#     - Variant logic keeps A and B under different rotation-only keys, so the two hole families can
#       receive different colors and chirality becomes task-relevant.
#   complexity_calibration:
#     - Keep 3-6 distinct hole shapes so repetition still supports induction.
#     - Require at least one chiral mirror pair per sample; otherwise some samples would collapse
#       back to the original behavior on symmetric/non-chiral shapes.
#     - Keep sprite colors distinct per hole shape to avoid accidental ambiguity.
#   implicit_assumption_review:
#     - main() step 1 still assumes the frame is the unique largest monochrome non-bg component;
#       generator preserves that.
#     - main() step 2 still keys holes by their exact mask orientation inside the frame; this is now
#       essential because mirrored holes must stay distinguishable.
#     - main() step 3 was the old dependency on the legacy property: augmenting each sprite with
#       reflected variants. That assumption is invalid now and must be removed.
#     - main() plurality voting still assumes every hole shape has at least one matching sprite;
#       generator guarantees that shape-by-shape.
#   placement_legality:
#     - Holes remain 4-disconnected and away from the frame border.
#     - Sprites remain fully outside the frame bbox and separated by >=1 bg cell in Chebyshev distance,
#       so outside components do not merge.
#   visual_observability:
#     - Samples always contain a mirror pair such as L/J or S/Z with different fill colors.
#     - Outside sprites visibly repeat both handednesses, so a solver must notice that reflection no
#       longer preserves identity.
#   cleanliness:
#     - No extra markings are added; the only visible difference is the handedness-sensitive matching,
#       which keeps the puzzle readable.


def _all_rotations(mask: np.ndarray):
    """Yield (key, oriented_mask) for the 4 rotational variants of a bool mask."""
    m0 = mask.astype(bool)
    for k in range(4):
        m = np.rot90(m0, k)
        yield (m.shape, m.tobytes()), m


def _rotation_key(mask: np.ndarray):
    """Canonical representative under rotations only."""
    return min(key for key, _ in _all_rotations(mask))


def _mirror_mask(mask: np.ndarray) -> np.ndarray:
    """A single mirror operator; with rotations, this spans the two handedness classes."""
    return np.fliplr(mask.astype(bool))


def _mirror_key(mask: np.ndarray):
    """Canonical key of the mirrored partner under rotation-only equivalence."""
    return _rotation_key(_mirror_mask(mask))


def _all_rotation_keys(mask: np.ndarray):
    """Generate keys for all rotations of a boolean mask, without any flip."""
    for key, _ in _all_rotations(mask):
        yield key


def _is_chiral(mask: np.ndarray) -> bool:
    """True iff the mask is not equivalent to its mirror under rotations alone."""
    return _rotation_key(mask) != _mirror_key(mask)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor holes inside a large frame by matching them to outside sprites of the same handed shape.

    Input structure:
    - bg is the most common color. One large monochrome frame sits on that background.
    - Inside the frame bbox are several holes, seen as non-frame-color cells.
    - Outside the frame are small sprites on the background, usually monochrome and shape-coded.

    Rule:
    - Crop to the frame's bounding box and begin with a solid block of the frame color.
    - For each hole, collect outside sprites whose masks match that hole up to rotation only.
    - Fill the hole with the most frequent matching sprite color; mirrored sprites do not count as matches.

    Key insight:
    - The outside sprites are color keys indexed by geometry, and in this variant chirality matters:
      two mirror-image shapes are distinct unless one can be turned into the other by rotation alone.
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
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        key = (m.shape, m.tobytes())
        holes.append((key, (int(hr0), int(hc0)), m))

    # STEP 3) Extract sprites outside the bbox.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> color -> count
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    for comp in spr_comps:
        coords = comp["coords"]
        bb = comp["bbox"]
        sr0, sc0, sr1, sc1 = map(int, bb)
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_rotation_keys(m):
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


# --------------- generator (rotation-only chirality variant) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

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
MAX_GEN_ATTEMPTS = 400


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotations of a mask."""
    seen, result = set(), []
    for key, m in _all_rotations(mask):
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
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
        [[1, 1, 1], [0, 1, 0], [0, 1, 0]],
        [[1, 0], [1, 1]],
        [[0, 1], [1, 1]],
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1], [0, 1], [1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 1], [0, 1], [0, 1]],
        [[1, 1], [1, 0], [1, 0]],
        [[1, 0], [1, 1], [0, 1]],
        [[0, 1], [1, 1], [1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1, 1]],
        [[1, 1, 1, 1, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        k = _rotation_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()
SHAPE_BY_KEY = {_rotation_key(shape): shape for shape in SHAPE_LIBRARY}


def _build_chiral_pairs():
    pairs = []
    seen = set()
    for shape in SHAPE_LIBRARY:
        key = _rotation_key(shape)
        mirror = _mirror_key(shape)
        if mirror == key or mirror not in SHAPE_BY_KEY:
            continue
        pair_id = tuple(sorted((key, mirror)))
        if pair_id in seen:
            continue
        seen.add(pair_id)
        pairs.append((SHAPE_BY_KEY[pair_id[0]], SHAPE_BY_KEY[pair_id[1]]))
    return pairs


CHIRAL_PAIRS = _build_chiral_pairs()


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
    c_m = tw - 1 - c - sw + 1
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
    r_m = th - 1 - r - sh + 1
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
    """Place holes and force at least one mirror-distinct pair to appear."""
    max_sh, max_sw = max(1, (th - 2) // 2), max(1, (tw - 2) // 2)
    available = [
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    available_by_key = {_rotation_key(shape): shape for shape in available}
    pair_candidates = []
    seen_pairs = set()
    for shape in available:
        key = _rotation_key(shape)
        mirror = _mirror_key(shape)
        if mirror == key or mirror not in available_by_key:
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

    others = [s for s in available if _rotation_key(s) not in required_keys]
    random.shuffle(others)
    max_target = min(SHAPE_COUNT_RANGE[1], len(available))
    if max_target < MIN_SHAPES:
        return None
    k_target = random.randint(max(MIN_SHAPES, 2), max_target)
    selection_order = [required_pair[0], required_pair[1], *others]

    occ = np.ones((th, tw), dtype=bool)
    hole_records = []
    mult_vals, mult_wts = _multiplicity_weights(th, tw)
    placed_keys = set()

    for shape in selection_order:
        key = _rotation_key(shape)
        must_place = key in required_keys
        if len(hole_records) >= k_target and not must_place:
            continue
        if len(placed_keys) >= k_target and key not in placed_keys and not must_place:
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

    return occ, hole_records, tuple(required_keys)


def _build_output(occ, hole_records, frame_col, bg):
    """Render the target output canvas and assign fill colors."""
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
                "mirror_key": _mirror_key(shape),
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


def _scatter_sprites(inp, shape_info, bg, frame_col, required_pair_keys):
    """
    Place sprites outside the frame.

    For the required chiral pair, use equal winning counts and no distractor colors.
    This makes the mirror distinction operational: a flip-aware matcher would pool the
    two handedness classes, while the rotation-only matcher keeps them separate.
    """
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    paired_winning = random.randint(2, 4)

    for info in shape_info:
        shape = info["shape"]
        fill_color = info["fill_color"]
        orientations = _all_orientations(shape)
        is_required_pair = info["key"] in required_pair_keys

        if is_required_pair:
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
    occ, hole_records, required_pair_keys = layout

    result = _build_output(occ, hole_records, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc_pos = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc_pos)

    if not _scatter_sprites(inp, shape_info, bg, frame_col, required_pair_keys):
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
