# Combo variant of 67e490f4 (index 15)
# combo_spec: D2/1+D4/0+D9/0
# Name: 多色图案+仅旋转搬运

from __future__ import annotations

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
# Design analysis for the combo variant
#
# a) Concrete behavior and trace:
#    The scaffold is still one large monochrome frame with bg holes inside and
#    separated source sprites outside. The fused change is that outside sprites
#    are now multicolor mini-stamps, they may rotate but may not flip to match
#    a hole, and the output copies the whole aligned stamp rather than one fill
#    color.
#
#    Small trace:
#    - Hole A uses the left-handed L mask [[1,0],[1,1]]; Hole B uses its mirror
#      [[0,1],[1,1]]. Outside the frame there are several left-handed sprites
#      carrying pattern P = [[2,0],[3,2]] and several right-handed sprites
#      carrying pattern Q = [[4,5],[0,4]].
#    - Original solver behavior:
#      1. reflection was allowed, so A and B were pooled as one shape family.
#      2. each sprite was reduced to one dominant color.
#      3. each hole was filled uniformly.
#    - Combo behavior:
#      1. A only matches rotated left-handed sprites; B only matches rotated
#         right-handed sprites.
#      2. the repeated winning object is the full colored patch, not one color.
#      3. the winning patch is rotated into the hole and stamped cell-by-cell.
#    - Visual difference:
#      a mirrored family can now produce a different interior motif, so chirality
#      is visible directly in the output rather than being hidden behind a flat
#      recolor.
#
# b) Complexity calibration:
#    - Use 3-5 hole families so examples still have repetition but remain legible.
#    - Restrict the library to chiral masks with 4 distinct rotations and 8
#      distinct dihedral transforms. This makes rotation alignment unique and
#      reflection genuinely meaningful.
#    - Winning patches always use 2-3 colors and are non-monochrome, so the
#      D9/D4 changes are visible in every valid sample.
#    - Every sample contains at least one mirror pair of hole families with
#      different winning patterns and equal winning counts, so D2 is not optional.
#
# c) Hidden-assumption review of the original main():
#    - Step 1 assumed the frame is the unique largest monochrome non-bg object.
#      Still valid; the generator keeps all sprite color-components much smaller.
#    - Step 2 keyed holes by their exact in-frame orientation. This becomes
#      essential because mirrored holes must remain distinguishable.
#    - Step 3 used two assumptions that are now false:
#      1. reflections were added to the match set;
#      2. a sprite could be summarized by one dominant color.
#      The combo solver must instead preserve the whole patch and generate only
#      rotational variants.
#    - Step 4 used a solid-color patch. That would erase the multi-color motif,
#      so the new solver pastes the winning aligned patch itself.
#
# d) Placement legality:
#    - Holes stay 4-disconnected and away from the frame border.
#    - Sprites are fully outside the frame bbox and separated by Chebyshev
#      padding >= 1, so multicolor sprites do not merge accidentally.
#    - Output-first construction guarantees each scattered sprite family has a
#      corresponding hole family and a unique plurality-winning patch.
#
# e) Visual observability:
#    - Required mirror-pair families make chirality visible.
#    - Winning patches are asymmetric and multi-color, so humans can see that
#      the outside objects are literal stamps, not mere color keys.
#    - Equal winning counts across the required mirror pair ensure a hypothetical
#      flip-aware matcher would pool evidence incorrectly; the control variable
#      is operational, not cosmetic.
#
# f) Visual cleanliness:
#    - No extra markers are added. The scene remains "frame + holes + outside
#      sprites", just with small multicolor stamps.
#    - Shared palettes are kept small so the grids do not turn into noise.
#
# g) New solver assumptions and generator guarantees:
#    - Assumption: each hole family has at least one outside sprite match.
#      Guarantee: every chosen family scatters multiple winning sprites.
#    - Assumption: for one sprite and one hole orientation, the aligned patch is
#      unique under the allowed transforms.
#      Guarantee: only masks with four distinct rotations are used.
#    - Assumption: each family has a strict plurality-winning patch.
#      Guarantee: losing patterns are fewer than the winning count, and the
#      required mirror pair uses distinct non-dihedrally-equivalent winners.
# ---------------------------------------------------------------------------


MaskKey = tuple[tuple[int, int], bytes]
PatchSig = tuple[tuple[int, int], bytes]


def _all_rotations(mask: np.ndarray):
    """Yield (key, rotated_mask) for the 4 rotational variants of a bool mask."""
    m0 = mask.astype(bool)
    for k in range(4):
        m = np.rot90(m0, k)
        yield (m.shape, m.tobytes()), m


def _all_patch_rotations(patch: np.ndarray):
    """Yield deduplicated 4-rotation transforms of a full colored patch."""
    seen = set()
    p0 = np.array(patch, dtype=int)
    for k in range(4):
        p = np.rot90(p0, k)
        sig = (p.shape, p.tobytes())
        if sig in seen:
            continue
        seen.add(sig)
        yield sig, p


def _all_dihedral_patch_transforms(patch: np.ndarray):
    """Yield deduplicated 8 dihedral transforms of a full colored patch."""
    seen = set()
    p0 = np.array(patch, dtype=int)
    for flip in (False, True):
        p1 = np.fliplr(p0) if flip else p0
        for k in range(4):
            p2 = np.rot90(p1, k)
            sig = (p2.shape, p2.tobytes())
            if sig in seen:
                continue
            seen.add(sig)
            yield sig, p2


def _rotation_key(mask: np.ndarray) -> MaskKey:
    return min(key for key, _ in _all_rotations(mask))


def _mirror_key(mask: np.ndarray) -> MaskKey:
    return _rotation_key(np.fliplr(mask.astype(bool)))


def _rotation_patch_signature(patch: np.ndarray) -> PatchSig:
    return min(sig for sig, _ in _all_patch_rotations(patch))


def _dihedral_patch_signature(patch: np.ndarray) -> PatchSig:
    return min(sig for sig, _ in _all_dihedral_patch_transforms(patch))


def _rotation_count(mask: np.ndarray) -> int:
    return len({key for key, _ in _all_rotations(mask)})


def _dihedral_count(mask: np.ndarray) -> int:
    seen = set()
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.fliplr(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            seen.add((m2.shape, m2.tobytes()))
    return len(seen)


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotations of a mask."""
    seen, result = set(), []
    for key, m in _all_rotations(mask):
        if key in seen:
            continue
        seen.add(key)
        result.append(m)
    return result


def _all_patch_orientations(patch: np.ndarray):
    return [p for _sig, p in _all_patch_rotations(patch)]


def _shape_dims(shape: np.ndarray) -> tuple[int, int]:
    return int(shape.shape[0]), int(shape.shape[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp multi-color hole interiors inside a frame using rotation-matched outside sprites.

    Input structure:
    - The most common color is the background. One large monochrome frame sits on it.
    - Inside the frame bbox are several bg-colored holes of small chiral shapes.
    - Outside the frame are separated multi-color sprites whose silhouettes match hole
      shapes under rotation only; mirrored sprites are a different family.

    Rule:
    - Crop to the frame's bounding box and begin with a solid block of the frame color.
    - For each hole, gather outside sprites whose foreground mask matches that exact hole
      orientation up to rotation, but never by reflection.
    - Among those matches, find the most frequent aligned full patch and paste that whole
      multi-color pattern into the hole.

    Key insight:
    - The outside objects are not color labels anymore; they are literal stamps. Shape
      chooses the destination hole family, chirality separates mirror families, and the
      repeated interior motif tells you exactly what pattern to copy after rotation.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the template frame: the unique largest monochrome non-bg component.
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

    # STEP 2) Extract holes as exact 4-connected non-frame regions inside the bbox.
    hole_mask = sub != int(frame_col)
    holes: list[tuple[MaskKey, tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((((m.shape, m.tobytes())), (int(hr0), int(hc0)), m))

    # STEP 3) Extract outside sprites and preserve their full colored patches.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    # key -> patch_signature -> [count, aligned_patch]
    sprites_by_key: dict[MaskKey, dict[PatchSig, list[object]]] = {}

    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        rel = coords - np.array([sr0, sc0])

        patch = np.full((hh, ww), int(bg), dtype=int)
        patch[rel[:, 0], rel[:, 1]] = g[coords[:, 0], coords[:, 1]]

        variants_by_key: dict[MaskKey, dict[PatchSig, np.ndarray]] = {}
        for sig, oriented_patch in _all_patch_rotations(patch):
            oriented_mask = oriented_patch != int(bg)
            key = (oriented_mask.shape, oriented_mask.tobytes())
            variants_by_key.setdefault(key, {})[sig] = oriented_patch

        for key, sig_map in variants_by_key.items():
            # The generator only uses masks whose 4 rotations are all distinct.
            assert len(sig_map) == 1
            sig, aligned_patch = next(iter(sig_map.items()))
            bucket = sprites_by_key.setdefault(key, {})
            if sig not in bucket:
                bucket[sig] = [0, aligned_patch]
            bucket[sig][0] = int(bucket[sig][0]) + 1

    # STEP 4) Render output: solid frame first, then stamp the winning aligned patch.
    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        patch_map = sprites_by_key.get(key)
        assert patch_map is not None
        best_count = max(int(info[0]) for info in patch_map.values())
        winners = [info[1] for info in patch_map.values() if int(info[0]) == best_count]
        assert len(winners) == 1
        paste(out, winners[0], int(hr0), int(hc0), mask=hmask)

    return out


# --------------- generator (multicolor rotation-only stamp variant) ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Frame bbox dimensions
FRAME_H_RANGE = (8, 13)
FRAME_W_RANGE = (8, 20)

# Hole layout
SHAPE_COUNT_RANGE = (3, 5)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 7
FRAME_DENSITY_RANGE = (0.55, 0.90)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 90
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

# Patterned sprite scattering
PATTERN_POOL_RANGE = (4, 5)
WINNING_SPRITE_RANGE = (3, 5)
LOSING_PATTERN_MAX = 2
LOSING_SPRITE_RANGE = (1, 2)
REQUIRED_PAIR_WIN_COUNT = (3, 4)
SPRITE_PADDING = 1
PATTERN_RETRIES = 100

# Generation retry budget
MAX_GEN_ATTEMPTS = 500


RAW_CHIRAL_PAIRS = [
    (
        [[1, 0], [1, 0], [1, 1]],
        [[0, 1], [0, 1], [1, 1]],
    ),
    (
        [[1, 0, 0], [1, 1, 1], [0, 1, 0]],
        [[0, 0, 1], [1, 1, 1], [0, 1, 0]],
    ),
    (
        [[1, 0, 0], [1, 1, 1], [1, 1, 0]],
        [[0, 0, 1], [1, 1, 1], [0, 1, 1]],
    ),
    (
        [[1, 0, 0], [1, 1, 0], [1, 0, 0], [1, 1, 1]],
        [[0, 0, 1], [0, 1, 1], [0, 0, 1], [1, 1, 1]],
    ),
]


def _build_pair_library():
    pairs = []
    for left_raw, right_raw in RAW_CHIRAL_PAIRS:
        left = np.array(left_raw, dtype=bool)
        right = np.array(right_raw, dtype=bool)
        assert int(np.sum(left)) <= SHAPE_MAX_AREA
        assert int(np.sum(right)) <= SHAPE_MAX_AREA
        assert _rotation_count(left) == 4 and _rotation_count(right) == 4
        assert _dihedral_count(left) == 8 and _dihedral_count(right) == 8
        assert _mirror_key(left) == _rotation_key(right)
        assert _mirror_key(right) == _rotation_key(left)
        pairs.append((left, right))
    return pairs


CHIRAL_PAIRS = _build_pair_library()
SHAPE_LIBRARY = [shape for pair in CHIRAL_PAIRS for shape in pair]


def _fit_rotations(mask: np.ndarray, max_h: int, max_w: int):
    result = []
    seen = set()
    for key, m in _all_rotations(mask):
        if key in seen:
            continue
        seen.add(key)
        h, w = _shape_dims(m)
        if h <= max_h and w <= max_w:
            result.append(m)
    return result


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int):
    """Return the 4-way mirrored placement set for a bbox shape."""
    return list(
        {
            (r, c),
            (r, tw - 1 - c - sw + 1),
            (th - 1 - r - sh + 1, c),
            (th - 1 - r - sh + 1, tw - 1 - c - sw + 1),
        }
    )


def _validate_positions(occ: np.ndarray, shape: np.ndarray, positions, th: int, tw: int):
    """Check bounds, overlap, and 4-disconnection from previously carved holes."""
    sh, sw = _shape_dims(shape)
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


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = _shape_dims(shape)
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r, c = random.randint(1, r_max), random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0:
        return None
    sh, sw = _shape_dims(shape)
    r = th // 2 - sh // 2
    if r < 1 or r + sh > th - 1:
        return None
    c_max = tw // 2 - sw
    if c_max < 1:
        return None
    c = random.randint(1, c_max)
    c_m = tw - 1 - c - sw + 1
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if tw % 2 == 0:
        return None
    sh, sw = _shape_dims(shape)
    c = tw // 2 - sw // 2
    if c < 1 or c + sw > tw - 1:
        return None
    r_max = th // 2 - sh
    if r_max < 1:
        return None
    r = random.randint(1, r_max)
    r_m = th - 1 - r - sh + 1
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = _shape_dims(shape)
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < 1 or c < 1 or r + sh > th - 1 or c + sw > tw - 1:
        return None
    return _validate_positions(occ, shape, [(r, c)], th, tw)


def _try_place_symmetric(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    placers = [_try_place_quadrant]
    if th % 2 == 1:
        placers.append(_try_place_hcenter)
    if tw % 2 == 1:
        placers.append(_try_place_vcenter)
    if th % 2 == 1 and tw % 2 == 1:
        placers.append(_try_place_center)
    return random.choice(placers)(occ, shape, th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions):
    sh, sw = _shape_dims(shape)
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _multiplicity_weights(th: int, tw: int):
    return MULT_EVEN_WEIGHTS if (th % 2 == 0 or tw % 2 == 0) else MULT_ODD_WEIGHTS


def _available_bases(th: int, tw: int):
    max_h = max(1, (th - 2) // 2)
    max_w = max(1, (tw - 2) // 2)

    pair_candidates = []
    available_by_key: dict[MaskKey, np.ndarray] = {}
    for left, right in CHIRAL_PAIRS:
        left_rots = _fit_rotations(left, max_h, max_w)
        right_rots = _fit_rotations(right, max_h, max_w)
        if left_rots:
            available_by_key[_rotation_key(left)] = left
        if right_rots:
            available_by_key[_rotation_key(right)] = right
        if left_rots and right_rots:
            pair_candidates.append((left, right))
    return max_h, max_w, pair_candidates, available_by_key


def _build_hole_layout(th: int, tw: int):
    """
    Place hole families and force one mirror-distinct pair to appear.

    The pair guarantee keeps the rotation-only rule operational, while the
    restricted shape library guarantees unambiguous rotational alignment for
    pattern stamping.
    """
    max_h, max_w, pair_candidates, available_by_key = _available_bases(th, tw)
    if not pair_candidates or len(available_by_key) < MIN_SHAPES:
        return None

    required_pair = random.choice(pair_candidates)
    required_keys = {_rotation_key(required_pair[0]), _rotation_key(required_pair[1])}

    others = [shape for key, shape in available_by_key.items() if key not in required_keys]
    random.shuffle(others)
    max_target = min(SHAPE_COUNT_RANGE[1], len(others) + 2)
    if max_target < MIN_SHAPES:
        return None
    k_target = random.randint(MIN_SHAPES, max_target)

    occ = np.ones((th, tw), dtype=bool)
    records = []
    placed_keys = set()
    selection_order = [required_pair[0], required_pair[1], *others]
    mult_vals, mult_wts = _multiplicity_weights(th, tw)

    for base_shape in selection_order:
        key = _rotation_key(base_shape)
        must_place = key in required_keys
        if len(records) >= k_target and not must_place:
            continue
        if len(placed_keys) >= k_target and key not in placed_keys and not must_place:
            continue

        rotations = _fit_rotations(base_shape, max_h, max_w)
        if not rotations:
            if must_place:
                return None
            continue
        random.shuffle(rotations)

        mult = random.choices(mult_vals, weights=mult_wts)[0]
        if must_place:
            mult = max(1, min(mult, 2))

        family_record = None
        for shape in rotations:
            trial_occ = occ.copy()
            placed_groups = []
            for _ in range(mult):
                success = False
                for _ in range(PLACEMENT_RETRIES):
                    positions = _try_place_symmetric(trial_occ, shape, th, tw)
                    if positions is None:
                        continue
                    _carve_hole(trial_occ, shape, positions)
                    placed_groups.append(positions)
                    success = True
                    break
                if not success:
                    break
            if placed_groups:
                family_record = {"shape": shape, "groups": placed_groups, "key": key}
                occ = trial_occ
                break

        if must_place and family_record is None:
            return None
        if family_record is not None:
            records.append(family_record)
            placed_keys.add(key)

    if len(records) < MIN_SHAPES or not required_keys.issubset(placed_keys):
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group) for rec in records for group in rec["groups"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, records, tuple(required_keys)


def _random_pattern(shape: np.ndarray, palette: list[int], bg: int, banned: set[PatchSig]):
    """
    Build a 2-3 color patch on one hole shape.

    Extra guard: the colored patch itself must have four distinct rotations.
    This keeps the stamped orientation visibly meaningful for humans.
    """
    cells = [tuple(map(int, rc)) for rc in np.argwhere(shape)]
    max_colors = min(3, len(cells), len(palette))
    if max_colors < 2:
        return None

    for _ in range(PATTERN_RETRIES):
        n_cols = random.randint(2, max_colors)
        cols = random.sample(palette, n_cols)
        patch = np.full(shape.shape, int(bg), dtype=int)

        random.shuffle(cells)
        for (rr, cc), col in zip(cells[:n_cols], cols):
            patch[rr, cc] = int(col)
        for rr, cc in cells[n_cols:]:
            patch[rr, cc] = int(random.choice(cols))

        fg_vals = [int(patch[rr, cc]) for rr, cc in cells]
        if len(set(fg_vals)) < 2:
            continue
        if len({sig for sig, _ in _all_patch_rotations(patch)}) != 4:
            continue

        sig = _rotation_patch_signature(patch)
        if sig in banned:
            continue
        return patch, sig
    return None


def _build_output(occ: np.ndarray, records, frame_col: int, bg: int, required_pair_keys):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    max_palette = min(PATTERN_POOL_RANGE[1], len(avail_colors))
    if max_palette < PATTERN_POOL_RANGE[0]:
        return None
    palette_size = random.randint(PATTERN_POOL_RANGE[0], max_palette)
    pattern_palette = random.sample(avail_colors, palette_size)

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    required_winner_dihedral: dict[MaskKey, PatchSig] = {}

    for rec in records:
        shape = rec["shape"]
        built = _random_pattern(shape, pattern_palette, bg, set())
        if built is None:
            return None
        winning_patch, winning_sig = built

        if rec["key"] in required_pair_keys:
            winner_dsig = _dihedral_patch_signature(winning_patch)
            partner_sigs = set(required_winner_dihedral.values())
            if winner_dsig in partner_sigs:
                banned = {winning_sig}
                alt_found = False
                for _ in range(PATTERN_RETRIES):
                    alt = _random_pattern(shape, pattern_palette, bg, banned)
                    if alt is None:
                        break
                    alt_patch, alt_sig = alt
                    alt_dsig = _dihedral_patch_signature(alt_patch)
                    banned.add(alt_sig)
                    if alt_dsig not in partner_sigs:
                        winning_patch, winning_sig, winner_dsig = alt_patch, alt_sig, alt_dsig
                        alt_found = True
                        break
                if not alt_found:
                    return None
            required_winner_dihedral[rec["key"]] = winner_dsig

        banned = {winning_sig}
        losing_patches = []
        if rec["key"] not in required_pair_keys:
            n_losing = random.randint(1, LOSING_PATTERN_MAX)
            for _ in range(n_losing):
                alt = _random_pattern(shape, pattern_palette, bg, banned)
                if alt is None:
                    return None
                losing_patch, losing_sig = alt
                banned.add(losing_sig)
                losing_patches.append(losing_patch)

        all_pos = []
        for positions in rec["groups"]:
            for pr, pc in positions:
                paste(output, winning_patch, pr, pc, mask=shape)
                all_pos.append((pr, pc))

        shape_info.append(
            {
                "shape": shape,
                "key": rec["key"],
                "winning_patch": winning_patch,
                "losing_patches": losing_patches,
                "positions": all_pos,
            }
        )

    if set(required_winner_dihedral) != set(required_pair_keys):
        return None
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, patch: np.ndarray, bg: int):
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, required_pair_keys):
    """
    Scatter outside sprite copies.

    Required mirror-pair families get equal winning counts and no losing patterns.
    This makes reflection sensitivity measurable: a flip-aware matcher would pool
    those two handedness classes even though their stamped outputs differ.
    """
    placement = inp.copy()
    paired_winning = random.randint(*REQUIRED_PAIR_WIN_COUNT)

    for info in shape_info:
        winning_orients = _all_patch_orientations(info["winning_patch"])
        is_required_pair = info["key"] in required_pair_keys

        if is_required_pair:
            n_winning = paired_winning
        else:
            n_winning = random.randint(*WINNING_SPRITE_RANGE)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(winning_orients), bg):
                return False

        if is_required_pair:
            continue

        for losing_patch in info["losing_patches"]:
            losing_orients = _all_patch_orientations(losing_patch)
            n_losing = random.randint(LOSING_SPRITE_RANGE[0], min(LOSING_SPRITE_RANGE[1], n_winning - 1))
            for _ in range(n_losing):
                if not _place_sprite(inp, placement, random.choice(losing_orients), bg):
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
    occ, records, required_pair_keys = layout

    built = _build_output(occ, records, frame_col, bg, required_pair_keys)
    if built is None:
        return None
    output, shape_info = built

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc)

    if not _scatter_sprites(inp, shape_info, bg, required_pair_keys):
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
