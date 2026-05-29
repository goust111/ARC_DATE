# Combo variant of 67e490f4 (index 22)
# combo_spec: D9/3+D10/0
# Name: 条件填充+完整网格输出

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


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - The original task always solved every frame hole and then returned only the frame crop.
#    - This combo changes both stages together: only hole shapes that have at least one outside
#      sprite match get filled, and the answer is redrawn on the full 30x30 canvas with all
#      sprites erased.
#    - Small trace: if a frame contains shape classes A/B/C and outside sprites exist only for
#      A and C, then the old task outputs a cropped frame where A/B/C all become solid colors.
#      Here the output keeps the full board, removes all outer sprites, fills all A and C holes
#      by plurality color, and leaves all B holes as background-colored openings in the frame.
# b) Complexity calibration:
#    - The main reasoning burden remains shape matching under rotation/flip plus per-shape color
#      plurality. The task is not reduced to mere erasure because matched holes still require
#      selecting the correct color from repeated sprite evidence.
#    - The generator forces both matched and unmatched shape classes, with repeated placements per
#      class under the original symmetry, so the conditional branch is visible and supported by
#      multiple examples rather than a one-off exception.
# c) Implicit-assumption audit of the original solver:
#    - `assert cmap is not None` is invalid now because some hole shapes must legitimately have no
#      witnesses outside the frame.
#    - `out = full(frame_col)` is invalid because unmatched holes should stay background-colored.
#      We must preserve the frame/hole pattern before selectively filling.
#    - Returning only `(th, tw)` is invalid because D10 requires the full 30x30 scene with the
#      solved frame pasted back at its original location and sprites removed.
# d) Placement legality:
#    - Hole placement keeps the original constraints: holes stay inside the frame border,
#      remain 4-disconnected, and preserve LR+UD+180 symmetry.
#    - Sprite placement still occurs fully outside the frame with Chebyshev padding, so sprite
#      components stay separate and never merge with the frame.
# e) Visual observability:
#    - Each generated sample contains both filled and unfilled holes inside one frame, making the
#      D9 effect directly visible.
#    - Each output is full-size and removes all outer sprites, making the D10 effect directly
#      visible at the same time.
# f) Visual cleanliness:
#    - Only two frame-interior states exist in the output: matched holes become solid colors,
#      unmatched holes remain plain background. Nothing extra is overprinted.
#    - A one-cell frame margin is enforced so sprite disappearance is visible as background area,
#      not confused with edge cropping.
# g) Color inferability:
#    - Matched shape classes receive one winning color by repeated sprite plurality.
#    - Unmatched classes receive no sprites at all, so their unchanged background state is
#      explained by visible absence rather than a hidden numeric rule.
# h) New solver assumptions introduced by the combo:
#    - The solver now assumes the frame position must be preserved when redrawing the full board.
#      The generator guarantees exactly one large frame and does not move it between input/output.
#    - The solver treats missing sprite matches as meaningful. The generator guarantees every
#      sample has at least one matched and one unmatched hole shape so that branch is exercised.


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
SPRITE_PADDING = 1

MIN_MATCHED_SHAPES = 1
MIN_UNMATCHED_SHAPES = 1
MIN_MATCHED_HOLES = 2
MIN_UNMATCHED_HOLES = 2
MIN_BRANCH_INSTANCE_RATIO = 0.25

MAX_GEN_ATTEMPTS = 400


def _all_mask_keys(mask: np.ndarray):
    """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Smallest key across all rotation/flip variants."""
    return min(key for key in _all_mask_keys(mask))


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve shape-keyed hole colors, but only where outside evidence exists, and keep the full canvas.

    Input structure:
    - The dominant color is the background on a 30x30 board.
    - One large monochrome component is a frame; inside its bbox, hole cells appear as non-frame background gaps.
    - Outside the frame are separated sprites whose shapes may or may not match hole shapes, up to rotation or reflection.

    Rule:
    - Locate the frame and read each hole mask inside its bounding box.
    - For each hole shape, collect all outside sprites with the same shape class and, if any exist, choose the most frequent sprite color for that class.
    - Return a full-size grid: erase all sprites, keep the frame in place, fill only matched holes, and leave unmatched holes as background.

    Key insight:
    - Outside sprites are shape-indexed color votes, and their absence is also informative: a hole changes only when its shape is represented outside the frame.
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
        sprite_col = int(Counter(vals).most_common(1)[0][0])

        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[sprite_col] = int(bucket.get(sprite_col, 0)) + 1

    solved_frame = np.where(sub == int(frame_col), int(frame_col), int(bg)).astype(int)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        if cmap is None:
            continue
        fill_col = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        paste(
            solved_frame,
            np.full(hmask.shape, int(fill_col), dtype=int),
            int(hr0),
            int(hc0),
            mask=hmask,
        )

    out = np.full(g.shape, int(bg), dtype=int)
    paste(out, solved_frame, int(r0), int(c0))
    return out


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
    """Pick which shape classes receive outside sprite evidence."""
    n_shapes = len(hole_placements)
    if n_shapes < MIN_MATCHED_SHAPES + MIN_UNMATCHED_SHAPES:
        return None

    counts = [sum(len(group) for group in groups) for _shape, groups in hole_placements]
    total_holes = sum(counts)
    indices = list(range(n_shapes))

    def _valid_partition(matched):
        matched_holes = sum(counts[i] for i in matched)
        unmatched_holes = total_holes - matched_holes
        if matched_holes < MIN_MATCHED_HOLES or unmatched_holes < MIN_UNMATCHED_HOLES:
            return False
        return min(matched_holes, unmatched_holes) / total_holes >= MIN_BRANCH_INSTANCE_RATIO

    for _ in range(80):
        matched_count = random.randint(MIN_MATCHED_SHAPES, n_shapes - MIN_UNMATCHED_SHAPES)
        matched = set(random.sample(indices, matched_count))
        if _valid_partition(matched):
            return [i in matched for i in indices]

    order = indices[:]
    random.shuffle(order)
    best = None
    best_gap = None
    for mask in range(1, 1 << n_shapes):
        matched = {indices[i] for i in range(n_shapes) if (mask >> i) & 1}
        if not (MIN_MATCHED_SHAPES <= len(matched) <= n_shapes - MIN_UNMATCHED_SHAPES):
            continue
        matched_holes = sum(counts[i] for i in matched)
        unmatched_holes = total_holes - matched_holes
        gap = abs(matched_holes - unmatched_holes)
        if _valid_partition(matched):
            if best is None or gap < best_gap:
                best = matched
                best_gap = gap
    if best is not None:
        return [i in best for i in indices]
    return None


def _build_full_output(occ, hole_placements, frame_col, bg, fr, fc):
    """
    Render the full-grid target output.
    Returns (output, shape_info), where shape_info is [(shape, fill_color_or_none, positions)].
    """
    match_flags = _choose_match_flags(hole_placements)
    if match_flags is None:
        return None

    matched_shape_count = sum(1 for flag in match_flags if flag)
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < matched_shape_count:
        return None
    matched_colors = iter(random.sample(avail_colors, matched_shape_count))

    solved_frame = np.where(occ, frame_col, bg).astype(int)
    shape_info = []
    for (shape, groups), is_matched in zip(hole_placements, match_flags):
        fill_col = next(matched_colors) if is_matched else None
        positions_flat = []
        for positions in groups:
            for pr, pc in positions:
                positions_flat.append((pr, pc))
                if fill_col is not None:
                    paste(
                        solved_frame,
                        np.full(shape.shape, int(fill_col), dtype=int),
                        int(pr),
                        int(pc),
                        mask=shape,
                    )
        shape_info.append((shape, fill_col, positions_flat))

    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    paste(out, solved_frame, int(fr), int(fc))
    return out, shape_info


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
        other_colors = random.sample(losers, n_other)

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        for other_color in other_colors:
            n_other_sprites = random.randint(1, max(1, n_winning - 1))
            for _ in range(n_other_sprites):
                if not _place_sprite(inp, placement, random.choice(orientations), other_color, bg):
                    break
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt for the combo variant."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_col = random.choice([c for c in range(NUM_COLORS) if c != bg])
    th = random.randint(*FRAME_H_RANGE)
    tw = random.randint(*FRAME_W_RANGE)

    layout = _build_hole_layout(th, tw)
    if layout is None:
        return None
    occ, hole_placements = layout

    fr = random.randint(FRAME_MARGIN, GRID_SIZE - th - FRAME_MARGIN)
    fc = random.randint(FRAME_MARGIN, GRID_SIZE - tw - FRAME_MARGIN)

    built = _build_full_output(occ, hole_placements, frame_col, bg, fr, fc)
    if built is None:
        return None
    output, shape_info = built

    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    paste(inp, np.where(occ, frame_col, bg).astype(int), int(fr), int(fc))

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
