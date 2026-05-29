# Variant of 67e490f4
# Dimension: D12 — 干扰项设计 (meta)
# Change: 存在 0-1 个干扰精灵（形状不匹配任何洞的精灵），以及同形状不同颜色的'输'精灵作为颜色干扰... → 无干扰项（纯净匹配）

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

# Design analysis for the D12 pure-matching variant
# --------------------------------------------------
# a) Concrete behavior:
#    - The frame still contains several holes, and the outside still contains small sprites.
#    - Every outside sprite now matches some hole shape under rotation/flip.
#    - For a given canonical shape, all outside copies use the same color as the output fill.
#    - Small example:
#      old rule: L-shaped hole + outside L sprites in red/red/green -> fill red by plurality.
#      new rule: L-shaped hole + outside L sprites all red -> fill red directly.
#      difference: matching no longer requires filtering distractor shapes or counting colors.
# b) Complexity calibration:
#    - The task stays nontrivial because holes still require shape matching with rotation/flip.
#    - It is not over-simplified into position matching because the sprites remain scattered and
#      can appear in multiple orientations.
#    - Repeated copies per shape are kept so the "all are valid clues" pattern is visible.
# c) Hidden-assumption audit of the original main():
#    - Frame detection as largest monochrome non-bg component still holds.
#    - Hole extraction as 4-connected regions inside the frame bbox still holds.
#    - Old sprite logic assumed multiple colors might share one shape and used counts/tie-breaks.
#      That assumption is removed; new solver instead assumes each canonical shape maps to one color.
#    - Old solver tolerated unmatched sprite keys by simply never querying them. That is unsafe for
#      this variant because it would allow generator regression back to distractors, so the new
#      solver explicitly checks that every sprite shape belongs to the hole-shape set.
# d) Placement legality:
#    - Holes remain 4-disconnected, inside the frame border, and symmetric as in the source task.
#    - Sprites are still placed fully outside the frame bbox with Chebyshev padding >= 1, so no
#      accidental touching or merging occurs.
#    - Because only matching sprites are emitted, generator consistency reduces to:
#      canonical(sprite_shape) ∈ canonical(hole_shapes) and one color per canonical shape.
# e) Visual observability:
#    - Across samples, the solver no longer needs "most frequent color" evidence.
#    - A human sees that every external token is relevant and all copies of one shape agree in color.
#    - Rotation/flip variability remains visible, so shape matching is still the operative cue.
# f) Visual cleanliness:
#    - Removing losing-color sprites and unmatched distractors reduces clutter around the frame.
#    - Padding is preserved to keep frame and sprites visually separable.
# g) Color inferability:
#    - Colors are assigned one-per-shape from the non-bg, non-frame palette.
#    - No invisible ordering or color-id arithmetic is used; the mapping is inferred only from
#      visible shape-color pairings.
# h) Variant metric used in validation:
#    - distractor_sprite_count = number of outside sprite canonical shapes absent from the holes.
#    - multicolor_shape_count = number of canonical sprite shapes carrying more than one color.
#    - The intended distribution for this variant is always 0 for both metrics.
# i) New solver assumptions to re-check against generator:
#    - Each hole canonical key appears in at least one outside sprite.
#    - Each canonical sprite shape uses exactly one color.
#    - Every outside sprite is monochrome and separated from the frame and other sprites.
#    The generator below constructs inputs to guarantee all three conditions.


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Canonical representative across all rotations/flips."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _mask_from_coords(coords: np.ndarray):
    """Return a tight boolean mask for a component."""
    r0, c0, r1, c1 = bbox_coords(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    rel = coords - np.array([r0, c0])
    mask = np.zeros((h, w), dtype=bool)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask, (int(r0), int(c0))


def _extract_holes(sub: np.ndarray, frame_col: int):
    holes = []
    for coords in components_mask(sub != int(frame_col), connectivity=4):
        mask, (r0, c0) = _mask_from_coords(coords)
        holes.append((_canonical_key(mask), (r0, c0), mask))
    return holes


def _extract_sprite_color_map(g: np.ndarray, bg: int, frame_bbox, hole_keys: set | None = None):
    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    key_to_color: dict[tuple[tuple[int, int], bytes], int] = {}
    key_to_count: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in spr_comps:
        mask, _ = _mask_from_coords(comp["coords"])
        key = _canonical_key(mask)
        color = int(comp["color"])
        if hole_keys is not None:
            assert key in hole_keys
        prev = key_to_color.get(key)
        if prev is not None:
            assert prev == color
        else:
            key_to_color[key] = color
        key_to_count[key] = int(key_to_count.get(key, 0)) + 1
    return key_to_color, key_to_count


def analyze_variant_metrics(grid: np.ndarray) -> dict[str, int]:
    """
    Return simple variant metrics used in validation.

    - distractor_sprite_count should be 0 in this D12 variant.
    - multicolor_shape_count should be 0 in this D12 variant.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=True, return_format="dict")
    frame = max(comps, key=lambda comp: int(comp["area"]))
    frame_bbox = frame["bbox"]
    frame_col = int(frame["color"])
    holes = _extract_holes(crop_bbox(g, frame_bbox), frame_col)
    hole_keys = {key for key, _, _ in holes}

    r0, c0, r1, c1 = map(int, frame_bbox)
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=True, return_format="dict")

    colors_by_key: dict[tuple[tuple[int, int], bytes], set[int]] = {}
    distractor_sprite_count = 0
    for comp in spr_comps:
        mask, _ = _mask_from_coords(comp["coords"])
        key = _canonical_key(mask)
        if key not in hole_keys:
            distractor_sprite_count += 1
        colors_by_key.setdefault(key, set()).add(int(comp["color"]))

    multicolor_shape_count = sum(1 for cols in colors_by_key.values() if len(cols) > 1)
    return {
        "hole_shape_count": len(hole_keys),
        "sprite_shape_count": len(colors_by_key),
        "sprite_count": len(spr_comps),
        "distractor_sprite_count": int(distractor_sprite_count),
        "multicolor_shape_count": int(multicolor_shape_count),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the framed pattern by using outside sprites as direct shape-to-color keys.

    Input structure:
    - The background is the most common color.
    - One large monochrome frame contains several holes of different small shapes.
    - Outside the frame, every small monochrome sprite matches one of the hole shapes, possibly after rotation or flip.

    Rule:
    - Crop to the frame bounding box and begin with a solid rectangle of the frame color.
    - Match each hole with the outside sprites by canonical shape, ignoring orientation.
    - Fill every hole with the unique color carried by its matching sprite shape.

    Key insight:
    - The outside objects are a clean dictionary: shape identifies which hole family to fill, and color is read directly because there are no distractors or competing colors to filter out.
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
    frame_bbox = frame["bbox"]

    sub = crop_bbox(g, frame_bbox)
    th, tw = sub.shape
    holes = _extract_holes(sub, frame_col)
    hole_keys = {key for key, _, _ in holes}

    key_to_color, _ = _extract_sprite_color_map(g, bg, frame_bbox, hole_keys)

    out = np.full((th, tw), int(frame_col), dtype=int)
    for key, (hr0, hc0), hmask in holes:
        color = key_to_color.get(key)
        assert color is not None
        patch = np.full(hmask.shape, int(color), dtype=int)
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

# Sprite scattering: pure matching only
SPRITE_COPY_RANGE = (1, 3)
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 300


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
    """Check all positions: in bounds, no overlap with existing holes, 4-disconnected from others."""
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
    """Render the target output and assign one visible fill color per shape."""
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
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = color
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp, shape_info, bg):
    """Place only matching sprites; no losing colors and no unmatched shapes."""
    placement = inp.copy()
    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_copies = random.randint(*SPRITE_COPY_RANGE)
        for _ in range(n_copies):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False
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

    if not _scatter_sprites(inp, shape_info, bg):
        return None

    try:
        if np.array_equal(main(inp), output):
            metrics = analyze_variant_metrics(inp)
            if metrics["distractor_sprite_count"] == 0 and metrics["multicolor_shape_count"] == 0:
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
