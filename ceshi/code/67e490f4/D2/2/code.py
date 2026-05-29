# Variant of 67e490f4
# Dimension: D2 — 匹配条件：旋转/翻转的允许范围 (match_cond)
# Change: rotation — 匹配时允许 8 种旋转/翻转变体（0/90/180/270度旋转 x 是否垂直翻转）... → 仅允许翻转不允许旋转

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
# a) Concrete behavior:
#    The input still has one large frame with bg-colored holes inside and many small sprites outside.
#    The changed behavior is only in shape matching: a hole can be matched by the exact sprite shape,
#    its left-right mirror, or its up-down mirror, but never by a 90/270-degree rotation.
#    Small example:
#    - Hole A uses a 2x3 "┐"-like mask.
#    - Outside sprite B is the 3x2 rotated version of that mask.
#    - Old logic: B would match A after a rotation.
#    - New logic: B does not match A, because rotation is forbidden and flips keep bbox dimensions.
# b) Complexity calibration:
#    If all shapes were squares or plus-signs, removing rotation would be invisible. The generator therefore
#    guarantees one rotational family with two flip-distinct orientations (for example 1x4 vs 4x1 or 2x3 vs 3x2),
#    and assigns those two orientations different majority colors.
# c) Hidden assumptions from the original solver:
#    - Hole keys were exact bbox masks already; that is still valid.
#    - Sprite indexing over all 8 transforms is no longer valid and must be reduced to flip-only transforms.
#    - The old generator canonicalized shapes under rotation+flip; that would collapse horizontal and vertical
#      variants into one category, so it must now canonicalize under flips only.
#    - Sprite orientation sampling must also avoid rotation, otherwise generated samples could fail to expose
#      the new constraint.
# d) Placement legality:
#    Holes remain 4-disconnected, stay off the frame border, and sprites are placed outside the frame bbox with
#    Chebyshev padding via random_free_location, so the solver still sees one clear frame and many isolated keys.
# e) Visual observability:
#    Each sample forces at least one pair of hole shapes that are related only by 90-degree rotation and gives
#    them different fill colors. Under the new rule, the two holes are filled differently; under the old rule,
#    they would be pooled into one rotation-family vote.
# f) Visual cleanliness:
#    The frame stays monochrome, holes stay separated, and sprites remain isolated on bg, so the orientation
#    contrast is readable without layered clutter.
# g) Color inferability:
#    Colors are not transformed; they are still chosen by plurality among matching sprites. The generator avoids
#    ties, so no invisible color-id ordering is needed to infer the intended answer from examples.
# h) New solver assumptions:
#    main() now assumes every hole has at least one outside sprite in the same flip-class. The generator enforces
#    this by constructing holes from a selected shape set and scattering sprites for every selected shape.
# i) Variant metric used in validation:
#    After implementation, we measure whether new-solver output differs from the original rotation-enabled solver
#    on generated samples. A healthy variant should frequently produce such disagreements, proving that the
#    rotation restriction is active rather than dormant.


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return mask.shape, mask.tobytes()


def _flip_variants(mask: np.ndarray):
    """Yield distinct identity / LR-flip / UD-flip variants."""
    seen = set()
    for variant in (mask.astype(bool), np.fliplr(mask), np.flipud(mask)):
        key = _mask_key(variant)
        if key not in seen:
            seen.add(key)
            yield key, variant


def _rot_flip_variants(mask: np.ndarray):
    """Yield distinct rotation+flip variants for library grouping only."""
    seen = set()
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            key = _mask_key(m2)
            if key not in seen:
                seen.add(key)
                yield key, m2


def _flip_canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(key for key, _ in _flip_variants(mask))


def _rot_canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min(key for key, _ in _rot_flip_variants(mask))


def _flip_orientations(mask: np.ndarray) -> list[np.ndarray]:
    return [m for _, m in _flip_variants(mask)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the frame holes by matching them with outside sprites under flip-only equivalence.

    Input structure:
    - The background is the most common color.
    - One large monochrome component is the frame; its interior contains bg-colored holes.
    - Outside the frame sit small isolated sprites that serve as color keys for the hole shapes.

    Rule:
    - Crop to the frame bounding box and start from a solid frame-colored rectangle.
    - Extract each hole shape exactly as drawn inside the frame.
    - Match that hole only against outside sprites of the same shape, or its horizontal/vertical mirror.
      Rotated versions are treated as different shapes.
    - Fill the hole with the plurality color among all matching sprites.

    Key insight:
    The decisive cue is not just the silhouette family, but the silhouette's axis orientation:
    mirrors still count as the same key, while quarter-turn rotations no longer do.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

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

    hole_mask = sub != frame_col
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((_mask_key(mask), (int(hr0), int(hc0)), mask))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = bg
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}
    for comp in spr_comps:
        coords = comp["coords"]
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        hh = int(sr1 - sr0 + 1)
        ww = int(sc1 - sc0 + 1)

        vals = [int(v) for v in g[coords[:, 0], coords[:, 1]]]
        sprite_col = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key, _ in _flip_variants(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[sprite_col] = int(bucket.get(sprite_col, 0)) + 1

    out = np.full((th, tw), frame_col, dtype=int)
    for key, (hr0, hc0), hole in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        fill_col = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hole.shape, int(fill_col), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hole)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

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
SPRITE_PADDING = 1

SPECIAL_DOMINANT_RANGE = (4, 6)
SPECIAL_SECONDARY_RANGE = (2, 3)

MAX_GEN_ATTEMPTS = 400


def _build_shape_records():
    """Build flip-class records while preserving rotation relatives as separate entries."""
    bases = [
        [[1, 1, 1]],
        [[1, 1, 1, 1]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[1, 1, 1, 1], [0, 1, 0, 0]],
    ]
    extras = [
        [[1]],
        [[1, 1]],
        [[1, 1], [1, 1]],
        [[1, 1, 1], [1, 1, 1]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],
    ]

    by_flip: dict[tuple[tuple[int, int], bytes], np.ndarray] = {}
    for shape in bases:
        m0 = np.array(shape, dtype=bool)
        for _, rot in _rot_flip_variants(m0):
            key = _flip_canonical_key(rot)
            by_flip.setdefault(key, rot)
    for shape in extras:
        m0 = np.array(shape, dtype=bool)
        key = _flip_canonical_key(m0)
        by_flip.setdefault(key, m0)

    records = []
    families: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for flip_key, mask in by_flip.items():
        rec = {
            "mask": mask,
            "flip_key": flip_key,
            "rot_key": _rot_canonical_key(mask),
            "area": int(np.sum(mask)),
        }
        records.append(rec)
        families.setdefault(rec["rot_key"], []).append(rec)
    return records, families


SHAPE_RECORDS, ROTATION_FAMILIES = _build_shape_records()


def _mirror_pos(r, c, sh, sw, th, tw):
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


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
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


def _try_place_quadrant(occ, shape, th, tw):
    sh, sw = shape.shape
    r_max, c_max = th // 2 - sh, tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
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
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
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
    max_sh = max(1, (th - 2) // 2)
    max_sw = max(1, (tw - 2) // 2)
    available = [
        rec
        for rec in SHAPE_RECORDS
        if rec["mask"].shape[0] <= max_sh and rec["mask"].shape[1] <= max_sw and rec["area"] <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    fam_to_recs: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for rec in available:
        fam_to_recs.setdefault(rec["rot_key"], []).append(rec)
    visible_families = [recs for recs in fam_to_recs.values() if len(recs) >= 2]
    if not visible_families:
        return None

    required_pair = random.sample(random.choice(visible_families), 2)
    required_keys = {rec["flip_key"] for rec in required_pair}

    others = [rec for rec in available if rec["flip_key"] not in required_keys]
    random.shuffle(others)
    max_shapes = min(SHAPE_COUNT_RANGE[1], len(available))
    k_target = random.randint(max(MIN_SHAPES, len(required_pair)), max_shapes)
    chosen = required_pair[:]

    even_dims = th % 2 == 0 or tw % 2 == 0
    if even_dims:
        others.sort(key=lambda rec: rec["area"])
    while len(chosen) < k_target and others:
        chosen.append(others.pop(0))

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []
    mult_vals, mult_wts = MULT_EVEN_WEIGHTS if even_dims else MULT_ODD_WEIGHTS

    for rec in chosen:
        mult = random.choices(mult_vals, weights=mult_wts)[0]
        placed_groups = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_symmetric(occ, rec["mask"], th, tw)
                if positions is not None:
                    _carve_hole(occ, rec["mask"], positions)
                    placed_groups.append(positions)
                    break
        if placed_groups:
            hole_placements.append(
                {
                    "record": rec,
                    "placed_groups": placed_groups,
                    "required": rec["flip_key"] in required_keys,
                }
            )
        elif rec["flip_key"] in required_keys:
            return None

    if len(hole_placements) < MIN_SHAPES:
        return None
    if sum(1 for item in hole_placements if item["required"]) != 2:
        return None

    frame_density = 1.0 - int(np.sum(~occ)) / (th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    total_holes = sum(len(group) for item in hole_placements for group in item["placed_groups"])
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _build_output(occ, hole_placements, frame_col, bg):
    th, tw = occ.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), frame_col, dtype=int)
    shape_info = []
    for item, fill_col in zip(hole_placements, shape_colors):
        rec = item["record"]
        all_pos = []
        for positions in item["placed_groups"]:
            for pr, pc in positions:
                patch = np.full(rec["mask"].shape, fill_col, dtype=int)
                paste(output, patch, pr, pc, mask=rec["mask"])
                all_pos.append((pr, pc))
        shape_info.append(
            {
                "shape": rec["mask"],
                "fill_color": int(fill_col),
                "positions": all_pos,
                "flip_key": rec["flip_key"],
                "rot_key": rec["rot_key"],
                "required": item["required"],
            }
        )
    return output, shape_info


def _place_sprite(inp, placement, shape, color, bg):
    patch = np.full(shape.shape, bg, dtype=int)
    patch[shape] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg, frame_col):
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    required = [info for info in shape_info if info["required"]]
    assert len(required) == 2
    assert required[0]["rot_key"] == required[1]["rot_key"]
    dominant_idx = random.randrange(2)
    dominant_n = random.randint(*SPECIAL_DOMINANT_RANGE)
    secondary_n = random.randint(*SPECIAL_SECONDARY_RANGE)
    if secondary_n >= dominant_n:
        secondary_n = dominant_n - 1

    special_counts = {
        required[dominant_idx]["flip_key"]: dominant_n,
        required[1 - dominant_idx]["flip_key"]: secondary_n,
    }

    for info in shape_info:
        orientations = _flip_orientations(info["shape"])
        if info["required"]:
            n_winning = special_counts[info["flip_key"]]
        else:
            n_winning = random.randint(*WINNING_SPRITE_RANGE)

        losers = [c for c in other_pool if c != info["fill_color"]]
        if info["required"]:
            n_other = random.randint(0, min(1, len(losers)))
        else:
            n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        other_colors = random.sample(losers, n_other) if n_other else []

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), info["fill_color"], bg):
                return False

        for other_col in other_colors:
            max_other = max(1, n_winning - 1)
            n_other_sprites = random.randint(1, max_other)
            for _ in range(n_other_sprites):
                if not _place_sprite(inp, placement, random.choice(orientations), other_col, bg):
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
