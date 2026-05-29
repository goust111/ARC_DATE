# Combo variant of 67e490f4 (index 12)
# combo_spec: D6/2+D14/1
# Name: 缺口框架+框架色反转

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

# Design analysis for the fused D6/2 + D14/1 variant.
# a) Concrete behavior with a traced example:
#    - Start from the original task: a large frame bbox contains several cavities, and outside sprites give
#      the fill color keyed by cavity shape.
#    - D6/2 changes the frame silhouette: some bg cells lie on the bbox border as shallow notches. Example:
#      in a 9x11 frame crop, a 1x2 gap on the top edge and a 2x1 gap on the right edge are part of the frame
#      design, while a 2x2 enclosed island near the center is a real hole. The original solver would treat all
#      three non-frame regions as fill targets; the fused solver must ignore the edge-touching gaps.
#    - D14/1 then changes the rendering target: after determining the true holes, the output does not preserve
#      the frame color at all. Using the same example, the old output would be a mostly frame-colored crop with
#      colored fillings. The fused output is a bg crop of the same bbox size where only the enclosed holes are
#      painted in their inferred colors. Border notches and former frame cells are both bg there.
# b) Complexity calibration:
#    - The frame still occupies a large bbox and contains 5+ hole instances, so there is enough repeated
#      evidence for shape-to-color matching.
#    - Notches are shallow and repeated on multiple sides, making the "edge-connected gap vs enclosed hole"
#      distinction visible without creating deep tunnels that would blur the structure.
#    - Output inversion is strong because a large visible frame disappears; this is much more obvious than a
#      subtle per-color remap, yet the hole colors stay unchanged to keep the mapping stable.
# c) Implicit assumptions from the base solver that break:
#    - `hole_mask = (sub != frame_col)` no longer means "all these cells are fillable"; boundary-connected
#      components are preserved notches.
#    - `out = full(frame_col)` is now wrong even for valid frame cells because the frame color becomes bg.
#    - The frame can still be found as the largest monochrome non-bg component, so that assumption is retained.
# d) Placement legality required from the generator:
#    - Enclosed holes stay at least 2 cells away from the bbox border, so a 1-cell-deep notch cannot merge with
#      them and every fillable cavity is strictly interior.
#    - Notches are carved only on the outer border, with bounded segment lengths and no overlap conflicts.
#    - Sprites remain fully outside the frame bbox with Chebyshev padding, so the frame detector and sprite
#      detector do not interfere.
# e) Visual observability:
#    - Across examples, humans can see that some bg gaps on the frame edge remain blank while inner cavities get
#      colored, so the notch/hole distinction is inferable from 2-3 samples.
#    - They can also see that the cropped output keeps the hole layout but the frame itself vanishes into the bg.
#      This is structural, not dependent on hidden color ordering.
# f) Visual cleanliness:
#    - Hole symmetry and sprite spacing are preserved from the base task.
#    - Notches are shallow "bites" from the border rather than large tears, so the input still clearly reads as
#      one frame object with a few designed gaps.
#    - The output is sparse but organized: only recolored hole islands remain on a uniform bg crop.
# g) New solver assumptions and generator guarantees:
#    - Solver assumes every fillable target is a 4-connected non-frame component that does not touch the crop
#      border; generator enforces this with `HOLE_MARGIN = 2` and 1-cell-deep notches.
#    - Solver assumes every such hole shape has at least one matching outside sprite and a strict plurality
#      winning color; generator constructs sprites directly from the chosen hole shapes and caps loser counts.
#    - Solver assumes the crop bbox is still the frame bbox even though the output erases the frame; generator
#      keeps a large connected frame component in the input so that bbox is recoverable.
# Assumption review of the modified `main()`:
#    - Frame selection loop assumes the largest monochrome non-bg component is unique. Generator uses exactly one
#      large frame and only tiny outside sprites, so this remains safe.
#    - Hole extraction loop assumes edge-touching non-frame components are all notches, never true holes.
#      Generator forbids holes within 2 cells of the border and makes notches only 1 cell deep.
#    - Sprite matching assumes canonical equality under rotation/flip is sufficient and no distractor shape needs
#      to be handled. Generator only places shapes that correspond to actual holes.
#    - Rendering assumes the entire target background is `bg`. That is the fused D14 rule, and generator builds
#      outputs with no surviving frame-colored cells.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color only the enclosed cavities of a notched frame, while the frame itself disappears in the crop.

    Input structure:
    - The dominant color is bg. One large monochrome frame component occupies a rectangular bbox, but some bbox-edge cells are bg notches rather than solid border.
    - Inside that bbox, enclosed non-frame regions are the real holes; edge-touching gaps are just frame cutouts.
    - Outside the bbox, scattered sprites on bg encode hole colors by shape, allowing rotation and mirror matches.

    Rule:
    - Find the unique largest monochrome frame component and crop to its bbox.
    - Separate bbox-edge-connected non-frame regions from enclosed ones; only the enclosed regions count as holes.
    - For each enclosed hole, choose the most frequent color among shape-matching outside sprites and paint that hole, but render every non-hole cell in the cropped output as bg.

    Key insight: The transformation has two coupled parts: first identify which gaps are true interior holes rather than border notches, then apply the color-key fill while inverting the frame's role so the whole frame turns into background.
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

    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(sub != int(frame_col), connectivity=4):
        touches_border = bool(
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == th - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == tw - 1)
        )
        if touches_border:
            continue
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        hh = int(hr1 - hr0 + 1)
        ww = int(hc1 - hc0 + 1)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((hh, ww), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((((hh, ww), mask.tobytes()), (int(hr0), int(hc0)), mask))

    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    spr_comps = find_components(outside, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    def _all_mask_keys(mask: np.ndarray):
        m0 = mask.astype(bool)
        for flip in (False, True):
            m1 = np.flipud(m0) if flip else m0
            for k in range(4):
                m2 = np.rot90(m1, k)
                yield (m2.shape, m2.tobytes())

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

    out = np.full((th, tw), int(bg), dtype=int)
    for key, (hr0, hc0), hole_mask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hole_mask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hole_mask)

    return out


NUM_COLORS = 10
GRID_SIZE = 30

FRAME_H_RANGE = (7, 13)
FRAME_W_RANGE = (7, 20)

SHAPE_COUNT_RANGE = (3, 6)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 8
FRAME_DENSITY_RANGE = (0.55, 0.92)
MIN_TOTAL_HOLES = 5
PLACEMENT_RETRIES = 90
MULT_EVEN_WEIGHTS = ([1, 2], [4, 1])
MULT_ODD_WEIGHTS = ([1, 2, 3], [3, 2, 1])

HOLE_MARGIN = 2

TOP_BOTTOM_NOTCH_COUNT_RANGE = (1, 2)
LEFT_RIGHT_NOTCH_COUNT_RANGE = (1, 2)
NOTCH_LEN_RANGE = (1, 3)

WINNING_SPRITE_RANGE = (2, 5)
OTHER_COLOR_MAX = 2
SPRITE_PADDING = 1

MAX_GEN_ATTEMPTS = 400


def _all_rot_flip(mask: np.ndarray):
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    seen, result = set(), []
    for key, mask2 in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(mask2)
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
        if int(mask.sum()) > SHAPE_MAX_AREA:
            continue
        key = _canonical_key(mask)
        if key not in seen:
            seen.add(key)
            dedup.append(mask)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int):
    return list(
        {
            (r, c),
            (r, tw - c - sw),
            (th - r - sh, c),
            (th - r - sh, tw - c - sw),
        }
    )


def _validate_positions(occ: np.ndarray, shape: np.ndarray, positions, th: int, tw: int):
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < HOLE_MARGIN or pc < HOLE_MARGIN:
            return None
        if pr + sh > th - HOLE_MARGIN or pc + sw > tw - HOLE_MARGIN:
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


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    sh, sw = shape.shape
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
    if r_max < HOLE_MARGIN or c_max < HOLE_MARGIN:
        return None
    r = random.randint(HOLE_MARGIN, r_max)
    c = random.randint(HOLE_MARGIN, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _try_place_hcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    if r < HOLE_MARGIN or r + sh > th - HOLE_MARGIN:
        return None
    c_max = tw // 2 - sw
    if c_max < HOLE_MARGIN:
        return None
    c = random.randint(HOLE_MARGIN, c_max)
    c_m = tw - c - sw
    return _validate_positions(occ, shape, list({(r, c), (r, c_m)}), th, tw)


def _try_place_vcenter(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if tw % 2 == 0:
        return None
    sh, sw = shape.shape
    c = tw // 2 - sw // 2
    if c < HOLE_MARGIN or c + sw > tw - HOLE_MARGIN:
        return None
    r_max = th // 2 - sh
    if r_max < HOLE_MARGIN:
        return None
    r = random.randint(HOLE_MARGIN, r_max)
    r_m = th - r - sh
    return _validate_positions(occ, shape, list({(r, c), (r_m, c)}), th, tw)


def _try_place_center(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    if th % 2 == 0 or tw % 2 == 0:
        return None
    sh, sw = shape.shape
    r = th // 2 - sh // 2
    c = tw // 2 - sw // 2
    if r < HOLE_MARGIN or c < HOLE_MARGIN:
        return None
    if r + sh > th - HOLE_MARGIN or c + sw > tw - HOLE_MARGIN:
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
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th: int, tw: int):
    max_sh = max(1, (th - 2 * HOLE_MARGIN) // 2)
    max_sw = max(1, (tw - 2 * HOLE_MARGIN) // 2)
    available = [
        shape
        for shape in SHAPE_LIBRARY
        if shape.shape[0] <= max_sh and shape.shape[1] <= max_sw and int(np.sum(shape)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = (th % 2 == 0 or tw % 2 == 0)
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

    total_holes = sum(len(pos) for _, pls in hole_placements for pos in pls)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _add_notches(occ: np.ndarray):
    th, tw = occ.shape
    notch = np.zeros((th, tw), dtype=bool)

    def carve_top_bottom(c0: int, length: int):
        notch[0, c0 : c0 + length] = True
        notch[th - 1, c0 : c0 + length] = True

    def carve_left_right(r0: int, length: int):
        notch[r0 : r0 + length, 0] = True
        notch[r0 : r0 + length, tw - 1] = True

    if tw >= 5:
        max_len = min(NOTCH_LEN_RANGE[1], max(1, tw - 4))
        n_tb = random.randint(*TOP_BOTTOM_NOTCH_COUNT_RANGE)
        used_cols: list[tuple[int, int]] = []
        for _ in range(n_tb):
            for _ in range(40):
                length = random.randint(NOTCH_LEN_RANGE[0], max_len)
                if tw - length - 1 <= 1:
                    continue
                c0 = random.randint(1, tw - length - 1)
                if any(not (c0 + length <= a or b <= c0) for a, b in used_cols):
                    continue
                carve_top_bottom(c0, length)
                used_cols.append((c0, c0 + length))
                break

    if th >= 5:
        max_len = min(NOTCH_LEN_RANGE[1], max(1, th - 4))
        n_lr = random.randint(*LEFT_RIGHT_NOTCH_COUNT_RANGE)
        used_rows: list[tuple[int, int]] = []
        for _ in range(n_lr):
            for _ in range(40):
                length = random.randint(NOTCH_LEN_RANGE[0], max_len)
                if th - length - 1 <= 1:
                    continue
                r0 = random.randint(1, th - length - 1)
                if any(not (r0 + length <= a or b <= r0) for a, b in used_rows):
                    continue
                carve_left_right(r0, length)
                used_rows.append((r0, r0 + length))
                break

    if not notch.any():
        return None

    occ2 = occ.copy()
    occ2[notch] = False

    frame_density = float(occ2.sum()) / float(th * tw)
    if not (FRAME_DENSITY_RANGE[0] <= frame_density <= FRAME_DENSITY_RANGE[1]):
        return None

    return occ2, notch


def _build_output(frame_mask: np.ndarray, hole_placements, frame_col: int, bg: int):
    th, tw = frame_mask.shape
    avail_colors = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]
    if len(avail_colors) < len(hole_placements):
        return None
    shape_colors = random.sample(avail_colors, len(hole_placements))

    output = np.full((th, tw), bg, dtype=int)
    shape_info = []
    for (shape, pls), fill_col in zip(hole_placements, shape_colors):
        all_pos = []
        for positions in pls:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, fill_col, dtype=int), pr, pc, mask=shape)
                all_pos.append((pr, pc))
        shape_info.append((shape, int(fill_col), all_pos))
    return output, shape_info


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int):
    patch = np.full(ori.shape, bg, dtype=int)
    patch[ori] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, frame_col: int):
    placement = inp.copy()
    other_pool = [c for c in range(NUM_COLORS) if c != bg and c != frame_col]

    for shape, fill_color, _ in shape_info:
        orientations = _all_orientations(shape)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)
        losers = [c for c in other_pool if c != fill_color]
        max_other = min(OTHER_COLOR_MAX, len(losers), max(1, n_winning - 1))
        n_other = random.randint(1, max_other) if max_other > 0 else 0
        oc_list = random.sample(losers, n_other) if n_other > 0 else []

        for _ in range(n_winning):
            if not _place_sprite(inp, placement, random.choice(orientations), fill_color, bg):
                return False

        remaining_budget = n_winning - 1
        for idx, oc in enumerate(oc_list):
            max_for_color = max(1, remaining_budget - max(0, len(oc_list) - idx - 1))
            n_oc = random.randint(1, max_for_color)
            remaining_budget -= n_oc
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
    occ, hole_placements = layout

    notched = _add_notches(occ)
    if notched is None:
        return None
    frame_mask, notch_mask = notched
    if int(notch_mask.sum()) == 0:
        return None

    result = _build_output(frame_mask, hole_placements, frame_col, bg)
    if result is None:
        return None
    output, shape_info = result

    fr = random.randint(0, GRID_SIZE - th)
    fc = random.randint(0, GRID_SIZE - tw)
    inp = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    paste(inp, np.where(frame_mask, frame_col, bg).astype(int), fr, fc)

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
