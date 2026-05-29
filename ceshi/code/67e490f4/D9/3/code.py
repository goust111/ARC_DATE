# Variant of 67e490f4
# Dimension: D9 — 变换：填充操作方式 (transform)
# Change: hole_fill — 每个洞的所有像素统一填充为匹配精灵的最高频颜色，生成纯色填充... → 条件性填充（仅填充部分洞）

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

# Design analysis for the D9 variant
# a) Concrete behavior:
#    - The large monochrome frame is still the only target region and the outside sprites still
#      encode hole-shape -> fill-color matches.
#    - The change is at render time: a hole is colored only if at least one outside sprite has
#      the same shape up to rotation/flip. Holes whose shapes never appear outside stay as bg in
#      the cropped output, so the output visibly mixes filled holes and still-open holes.
#    - Small trace example: if the frame contains shapes A, B, C and sprites exist only for A
#      and C, then the old task would require A/B/C all to become solid colors; the new task
#      instead keeps B as a background-colored hole while A and C are filled.
# b) Complexity calibration:
#    - Matching still happens by geometric shape class, not by position or color id, so the core
#      reasoning remains the original task's shape lookup.
#    - To make the variant visible, each generated grid contains both matched and unmatched hole
#      shapes, with several repeated hole instances under the original symmetry constraints.
#    - We keep the original plurality-of-colors mechanism for matched shapes so the task does not
#      collapse into mere presence/absence detection.
# c) Implicit assumption audit of the original main():
#    - `out = full(frame_col)` assumed every hole would eventually be overwritten; that is false
#      now, because unmatched holes must stay bg. The new solver starts from the cropped frame
#      pattern (`frame_col` plus bg holes) instead.
#    - `assert cmap is not None` assumed every hole shape appears among sprites; that is also
#      false now. The new solver treats missing matches as a valid branch and leaves the hole.
#    - The sprite plurality code still assumes a unique winning frequency for human-facing tasks;
#      the generator guarantees that by construction and never relies on tie-breaking by color id.
# d) Placement legality:
#    - Hole placement is unchanged: holes stay inside the frame border, remain 4-disconnected,
#      and inherit the original LR+UD+180 symmetry.
#    - Sprite placement is unchanged except that unmatched hole shapes simply receive no sprites.
#      Sprites are still scattered outside the frame bbox with padding, so solver component
#      extraction remains valid.
# e) Visual observability:
#    - Every sample forces both branches to appear: at least one matched shape and at least one
#      unmatched shape. Therefore the user can compare "holes with outside witnesses become
#      colored" against "holes with no witness remain empty" within the same example.
#    - Match status is assigned per shape class, not per individual location, so the pattern is
#      repeated and easier to infer from 2-4 examples.
# f) Visual cleanliness:
#    - No extra markers are drawn inside the frame. The only change is whether a hole remains bg
#      or receives a solid fill, which keeps the frame readable.
#    - External sprites keep the original spacing rule, so no merged clutter appears outside.
# g) Color inferability:
#    - Matched shapes still use a unique plurality color among their sprite copies.
#    - Unmatched shapes have no outside copies at all, so their unchanged bg state is explained by
#      visible absence rather than an invisible numeric rule.


def _all_mask_keys(mask: np.ndarray):
    """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for key in _all_mask_keys(m0):
        h, w = key[0]
        oriented = np.frombuffer(key[1], dtype=bool).reshape(h, w).copy()
        yield key, oriented


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants (canonical representative)."""
    return min(key for key in _all_mask_keys(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only those frame holes whose shapes are witnessed by outside sprites.

    Input structure:
    - The dominant color is the background. One large monochrome frame sits somewhere in the grid.
    - Inside that frame are several background-colored holes of different shapes.
    - Outside the frame are small separated sprites whose shapes may or may not correspond to the holes.

    Rule:
    - Crop to the frame's bounding box, preserving the frame/background pattern already present there.
    - For each hole, look for outside sprites with the same shape up to rotation or flip.
    - If matching sprites exist, fill the hole with the most frequent matching sprite color; otherwise leave the hole as background.

    Key insight:
    - Outside sprites still act as shape-coded color hints, but now their absence is informative too:
    - a hole is filled only when its shape class is represented outside the frame.
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
    r0, c0, r1, c1 = map(int, frame["bbox"])

    sub = crop_bbox(g, frame["bbox"])
    th, tw = sub.shape

    hole_mask = sub != int(frame_col)
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append(((mask.shape, mask.tobytes()), (int(hr0), int(hc0)), mask))

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

    out = np.where(sub == int(frame_col), int(frame_col), int(bg)).astype(int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        if cmap is None:
            continue
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

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

MIN_MATCHED_SHAPES = 1
MIN_UNMATCHED_SHAPES = 1
MIN_MATCHED_HOLES = 2
MIN_UNMATCHED_HOLES = 2

MAX_GEN_ATTEMPTS = 400


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
    return _validate_positions(occ, shape, list({(r, c), (r, tw - 1 - c - sw + 1)}), th, tw)


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
    return _validate_positions(occ, shape, list({(r, c), (th - 1 - r - sh + 1, c)}), th, tw)


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

    total_holes = sum(len(group) for _shape, groups in hole_placements for group in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _choose_match_flags(hole_placements):
    """Choose which shape classes are represented by sprites."""
    n_shapes = len(hole_placements)
    if n_shapes < MIN_MATCHED_SHAPES + MIN_UNMATCHED_SHAPES:
        return None

    counts = [sum(len(group) for group in groups) for _shape, groups in hole_placements]
    indices = list(range(n_shapes))
    for _ in range(80):
        matched_count = random.randint(MIN_MATCHED_SHAPES, n_shapes - MIN_UNMATCHED_SHAPES)
        matched = set(random.sample(indices, matched_count))
        matched_holes = sum(counts[i] for i in matched)
        unmatched_holes = sum(counts[i] for i in indices if i not in matched)
        if matched_holes >= MIN_MATCHED_HOLES and unmatched_holes >= MIN_UNMATCHED_HOLES:
            return [i in matched for i in indices]

    order = indices[:]
    random.shuffle(order)
    order.sort(key=lambda i: counts[i], reverse=True)
    matched = set(order[: max(MIN_MATCHED_SHAPES, n_shapes // 2)])
    matched_holes = sum(counts[i] for i in matched)
    unmatched_holes = sum(counts[i] for i in indices if i not in matched)
    if matched_holes >= MIN_MATCHED_HOLES and unmatched_holes >= MIN_UNMATCHED_HOLES:
        return [i in matched for i in indices]
    return None


def _build_output(occ, hole_placements, frame_col, bg):
    """
    Render the target output canvas.
    Returns (output, shape_info) where shape_info is [(shape, fill_color_or_none, [positions])].
    """
    match_flags = _choose_match_flags(hole_placements)
    if match_flags is None:
        return None

    matched_shapes = sum(1 for flag in match_flags if flag)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < matched_shapes:
        return None
    matched_colors = iter(random.sample(avail_colors, matched_shapes))

    output = np.where(occ, frame_col, bg).astype(int)
    shape_info = []
    for (shape, groups), is_matched in zip(hole_placements, match_flags):
        fill_col = next(matched_colors) if is_matched else None
        all_pos = []
        for positions in groups:
            for pr, pc in positions:
                all_pos.append((pr, pc))
                if fill_col is not None:
                    patch = np.full(shape.shape, int(fill_col), dtype=int)
                    paste(output, patch, pr, pc, mask=shape)
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
    """Place sprites only for matched shape classes."""
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _positions in shape_info:
        if fill_color is None:
            continue

        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        n_other = random.randint(1, min(OTHER_COLOR_MAX, len(losers))) if losers else 0
        oc_list = random.sample(losers, n_other)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for oc in oc_list:
            n_oc = random.randint(1, n_winning - 1)
            for _ in range(n_oc):
                if not _place_sprite(inp, placement, random.choice(orientations), oc, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build symmetric frame, scatter sprites, validate."""
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
