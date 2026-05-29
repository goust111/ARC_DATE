# Variant of 67e490f4
# Dimension: D12 — 干扰项设计 (meta)
# Change: 存在 0-1 个干扰精灵（形状不匹配任何洞的精灵），以及同形状不同颜色的'输'精灵作为颜色干扰... → 大量干扰精灵

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

# Design analysis for the D12 variant
#
# a) Concrete behavior and trace
# The large frame and its holes stay unchanged: one monochrome frame contains several
# bg-colored holes, and the output is still the frame bbox with each hole filled by the
# color implied by outside sprites of the same shape. The D12 change happens only in the
# outside sprite field. Instead of at most one obviously unrelated distractor, we now add
# 3-5 distractor sprites whose masks differ from real hole shapes by exactly one cell
# (one extra cell or one missing cell), while keeping them connected and visually close.
#
# Small trace:
# - Original: hole shape is an L triomino. Outside there are several exact L triomino
#   sprites in different colors, plus maybe one unrelated blob. Matching is easy because
#   almost every nearby-looking sprite is actually a valid key.
# - Variant: the same L hole still has exact L sprites outside, but there are also
#   multiple near-miss sprites such as the L with one arm shortened or with one extra
#   adjacent cell. Those distractors must be ignored because shape matching is exact
#   under rotation/flip, not approximate.
# - Difference: the solver logic remains exact-match based, but the generator now creates
#   many more visually tempting false candidates so the exactness becomes observable.
#
# b) Complexity calibration
# The distractor count is fixed to a visible range of 3-5, not an open-ended number.
# Each distractor is derived from a real shape so it resembles the true vocabulary, yet
# it is rejected unless its canonical mask matches exactly. This keeps the pattern
# learnable from a few examples: humans can notice that some outside sprites are "almost"
# the right shape but still never drive any fill.
#
# c) Original implicit assumptions touched by the variant
# - main(): extracting hole keys as exact masks assumes the useful difference lies in
#   exact occupancy, not in rough bbox size. That remains valid and is now essential.
# - main(): the sprite tally loop assumes any non-matching sprite can safely be ignored
#   because only keys present in holes are queried later. This still holds, but only if
#   the generator guarantees distractors never canonicalize to a real hole shape.
# - main(): taking the majority color per exact key assumes there are enough exact-match
#   sprites for every hole shape. The generator still guarantees that and additionally
#   keeps near-miss distractors separate.
#
# d) Placement legality
# All sprites, including distractors, are still scattered with `random_free_location`
# using Chebyshev padding 1. They stay outside the frame bbox because the placement canvas
# already contains the frame. Distractor masks are cropped before placement, kept small
# (area <= SHAPE_MAX_AREA + 1), and remain 4-connected so they render as clean objects.
#
# e) Visual observability
# The new behavior is visible because each sample contains several close-but-invalid
# shapes. A human can compare a hole with outside sprites and see that some candidates
# are off by one cell. Those shapes never influence the output, so the learnable rule is:
# compare silhouettes exactly after rotation/flip; almost-matching sprites are decoys.
#
# f) Visual cleanliness
# The frame region is unchanged, and outside sprites still have background gaps between
# them. Distractors are individual monochrome components, not multi-layer overlays, so
# the input remains readable despite the higher outside density.
#
# g) Color inferability
# Color logic is unchanged: among exact-match sprites, the most frequent color wins.
# Distractors do not add a new color mapping rule; they only increase shape confusion.
# The generator keeps winner counts strictly above any losing color count, so the solver
# never needs invisible tie-breaking to determine the intended answer.
#
# h) Generator-side behavioral guarantee
# After assembling an input, `_analyze_instance` re-extracts the frame and outside
# components and checks that there are 3-5 non-matching sprites and that every one of
# them is edit-distance 1 from some real hole shape. If not, the sample is rejected.
#
# i) New solver assumptions introduced here
# The solver now implicitly relies on distractor shapes being ignorable by exact key
# lookup. The generator guarantees this by filtering every distractor candidate against
# the set of canonical hole keys before placement.


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _all_mask_keys(mask: np.ndarray):
    """Yield deduplicated keys for all rotation/flip variants of a bool mask."""
    seen = set()
    for key, _ in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            yield key


def _canonical_key(mask: np.ndarray):
    """Smallest orientation key used as the canonical representative."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    """Crop a bool mask to its tight bbox."""
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].astype(bool)


def _shape_edit_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Minimum XOR cell count between `a` and any oriented version of `b`."""
    best = None
    aa = a.astype(bool)
    for bb in _all_orientations(b):
        h = max(aa.shape[0], bb.shape[0])
        w = max(aa.shape[1], bb.shape[1])
        pa = np.zeros((h, w), dtype=bool)
        pb = np.zeros((h, w), dtype=bool)
        pa[: aa.shape[0], : aa.shape[1]] = aa
        pb[: bb.shape[0], : bb.shape[1]] = bb
        dist = int(np.logical_xor(pa, pb).sum())
        if best is None or dist < best:
            best = dist
    assert best is not None
    return best


def _extract_frame(g: np.ndarray, bg: int):
    """Return the unique largest non-bg monochrome component as the frame."""
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    assert comps
    frame_area = max(int(comp["area"]) for comp in comps)
    frame_candidates = [comp for comp in comps if int(comp["area"]) == frame_area]
    assert len(frame_candidates) == 1
    frame = frame_candidates[0]
    frame_col = int(frame["color"])
    fb = tuple(map(int, frame["bbox"]))
    return frame_col, fb


def _extract_holes(sub: np.ndarray, frame_col: int):
    """Return hole records as (exact_key, (r0,c0), mask)."""
    hole_mask = (sub != int(frame_col))
    holes = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append(((m.shape, m.tobytes()), (int(hr0), int(hc0)), m))
    return holes


def _extract_sprite_records(g: np.ndarray, frame_bbox, bg: int):
    """Return outside sprite records with their majority color and tight mask."""
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    records = []
    for comp in comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)
        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        scol = int(Counter(vals).most_common(1)[0][0])
        rel = coords - np.array([sr0, sc0])
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        records.append({"color": scol, "mask": m})
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the cropped frame by using only exact sprite-to-hole shape matches.

    Input structure:
    - The background is the most common color, with one large monochrome frame elsewhere.
    - Inside the frame bbox are several holes cut out of the frame color.
    - Outside the frame bbox are many small sprites, including exact matches and several
      near-miss distractors whose silhouettes differ by a cell.

    Rule:
    - Crop to the frame's bounding box and start from a solid canvas of the frame color.
    - For each hole, collect outside sprites whose mask matches that hole exactly after
      rotation or flip; ignore all almost-matching distractors.
    - Fill the hole with the most frequent color among those exact matches.

    Key insight:
    - Outside sprites are shape-coded color keys, but the coding is strict: a sprite that
      is only "close" in shape does nothing, so the task is solved by exact binary-mask
      comparison under rotation/flip, followed by a color plurality vote.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    frame_col, fb = _extract_frame(g, bg)
    sub = crop_bbox(g, fb)
    th, tw = sub.shape

    holes = _extract_holes(sub, frame_col)
    sprite_records = _extract_sprite_records(g, fb, bg)

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for rec in sprite_records:
        for key in _all_mask_keys(rec["mask"]):
            bucket = sprites_by_key.setdefault(key, {})
            scol = int(rec["color"])
            bucket[scol] = int(bucket.get(scol, 0)) + 1

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
DISTRACTOR_RANGE = (3, 5)
DISTRACTOR_AREA_MAX = SHAPE_MAX_AREA + 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 500


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
        m = np.array(s, dtype=bool)
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
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


def _validate_positions(occ, shape, positions, th, tw):
    """Check bounds, overlap, and 4-disconnection from existing holes."""
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


def _build_hole_layout(th, tw):
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
    """Return (output, shape_info) where shape_info is [(shape, fill_color, [positions])]."""
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, fill_col, all_pos))
    return output, shape_info


def _place_sprite(inp, placement, ori, color, bg):
    """Place one monochrome sprite; return True on success."""
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _near_miss_variants(mask: np.ndarray):
    """Return connected masks differing from `mask` by exactly one cell."""
    base = mask.astype(bool)
    h, w = base.shape
    out: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}

    for rr, cc in np.argwhere(base):
        cand = base.copy()
        cand[rr, cc] = False
        if not cand.any():
            continue
        if len(components_mask(cand, connectivity=4)) != 1:
            continue
        trimmed = _trim_mask(cand)
        if int(trimmed.sum()) <= DISTRACTOR_AREA_MAX:
            out[_canonical_key(trimmed)] = trimmed

    ext = np.zeros((h + 2, w + 2), dtype=bool)
    ext[1 : 1 + h, 1 : 1 + w] = base
    frontier = set()
    for rr, cc in np.argwhere(ext):
        for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
            if 0 <= nr < ext.shape[0] and 0 <= nc < ext.shape[1] and not ext[nr, nc]:
                frontier.add((nr, nc))
    for rr, cc in frontier:
        cand = ext.copy()
        cand[rr, cc] = True
        trimmed = _trim_mask(cand)
        if int(trimmed.sum()) <= DISTRACTOR_AREA_MAX:
            out[_canonical_key(trimmed)] = trimmed

    out.pop(_canonical_key(base), None)
    return list(out.values())


def _choose_distractor_shapes(shape_info):
    """Pick 3-5 unique near-miss shapes derived from real shapes."""
    hole_keys = {_canonical_key(shape) for shape, _, _ in shape_info}
    per_base = []
    all_unique = {}

    for shape, _, _ in shape_info:
        local = []
        seen = set()
        for cand in _near_miss_variants(shape):
            key = _canonical_key(cand)
            if key in hole_keys or key in seen:
                continue
            seen.add(key)
            local.append(cand)
            all_unique[key] = cand
        if local:
            random.shuffle(local)
            per_base.append(local)

    if len(all_unique) < DISTRACTOR_RANGE[0]:
        return None

    target = random.randint(DISTRACTOR_RANGE[0], min(DISTRACTOR_RANGE[1], len(all_unique)))
    selected = []
    used = set()

    while len(selected) < target:
        progress = False
        order = list(range(len(per_base)))
        random.shuffle(order)
        for idx in order:
            while per_base[idx] and _canonical_key(per_base[idx][-1]) in used:
                per_base[idx].pop()
            if not per_base[idx]:
                continue
            cand = per_base[idx].pop()
            key = _canonical_key(cand)
            if key in used:
                continue
            selected.append(cand)
            used.add(key)
            progress = True
            if len(selected) >= target:
                break
        if not progress:
            break

    if len(selected) < target:
        leftovers = list(all_unique.items())
        random.shuffle(leftovers)
        for key, cand in leftovers:
            if key in used:
                continue
            selected.append(cand)
            used.add(key)
            if len(selected) >= target:
                break

    return selected if len(selected) >= DISTRACTOR_RANGE[0] else None


def _scatter_sprites(inp, shape_info, bg, frame_col):
    """Place exact-match sprites, losing-color sprites, and many near-miss distractors."""
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
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

    distractor_shapes = _choose_distractor_shapes(shape_info)
    if distractor_shapes is None:
        return False

    preferred_colors = [fill_color for _, fill_color, _ in shape_info]
    color_choices = preferred_colors + other_pool
    for shape in distractor_shapes:
        if not _place_sprite(inp, placement, random.choice(_all_orientations(shape)), random.choice(color_choices), bg):
            return False
    return True


def _analyze_instance(grid: np.ndarray):
    """Return stats used to validate that the D12 behavior is actually present."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frame_col, fb = _extract_frame(g, bg)
    sub = crop_bbox(g, fb)

    holes = _extract_holes(sub, frame_col)
    hole_reps = {}
    for _key, _pos, mask in holes:
        hole_reps.setdefault(_canonical_key(mask), mask)

    sprites = _extract_sprite_records(g, fb, bg)
    distractor_count = 0
    near_miss_count = 0
    edit_distances = []
    hole_keys = set(hole_reps)
    real_shapes = list(hole_reps.values())

    for rec in sprites:
        key = _canonical_key(rec["mask"])
        if key in hole_keys:
            continue
        distractor_count += 1
        dist = min(_shape_edit_distance(rec["mask"], real_shape) for real_shape in real_shapes)
        edit_distances.append(dist)
        if dist == 1:
            near_miss_count += 1

    return {
        "distractor_count": distractor_count,
        "near_miss_count": near_miss_count,
        "min_distractor_edit": min(edit_distances) if edit_distances else None,
        "max_distractor_edit": max(edit_distances) if edit_distances else None,
    }


def _gen_attempt() -> np.ndarray | None:
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
    fc_pos = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), fr, fc_pos)

    if not _scatter_sprites(inp, shape_info, bg, frame_col):
        return None

    try:
        if not np.array_equal(main(inp), output):
            return None
        stats = _analyze_instance(inp)
        if not (DISTRACTOR_RANGE[0] <= stats["distractor_count"] <= DISTRACTOR_RANGE[1]):
            return None
        if stats["near_miss_count"] != stats["distractor_count"]:
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
