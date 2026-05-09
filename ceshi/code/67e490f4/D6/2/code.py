# Variant of 67e490f4
# Dimension: D6 — 目标结构：框架的形态 (dst)
# Change: hole_fill — 单个大型单色矩形区域作为框架，bbox 为 7-13行 x 7-20列，框架色密度 0.55-0.92 → 框架边界不为实心（有缺口）

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

# Design analysis for the D6 variant.
# a) Concrete behavior:
#    - The input still contains one large monochrome frame component plus outside sprites.
#    - The frame bbox is still rectangular, but some cells on the bbox boundary are bg notches.
#    - True fill targets are only the enclosed non-frame regions that do NOT touch the bbox edge.
#    - Border notches stay bg in the output. Example:
#      Original: every non-frame component inside the bbox is treated as a hole.
#      Variant: a 1x2 bg gap on the top edge is part of the frame silhouette, while a 1x2 bg island in the
#      interior is a fillable hole. The old logic would try to fill both; the new logic fills only the island.
# b) Complexity calibration:
#    - Notches are repeated on multiple sides and preserved in the cropped output, so the silhouette change is
#      visually obvious across examples.
#    - Hole shapes remain small and repeated, and each shape gets a strict plurality sprite color, so the color
#      inference remains stable instead of depending on tie-breaks.
# c) Old implicit assumptions that break here:
#    - `hole_mask = (sub != frame_col)` used to be safe because every such region was an enclosed hole.
#      In the variant that assumption is false: boundary-touching bg regions are design notches, not holes.
#    - `out = full(frame_col)` used to be safe because the frame bbox had no intended bg cells in the output.
#      Now it would wrongly paint over the preserved notches.
#    - Largest monochrome component detection still works, so the frame-finding step is intentionally unchanged.
# d) Placement legality:
#    - Internal holes are forced to stay at least 2 cells away from the bbox border.
#    - Border notches are only 1 cell deep, so they cannot merge with real holes.
#    - Sprites remain fully outside the frame bbox with Chebyshev padding.
# e) Visual observability:
#    - Humans can see two kinds of bg inside the frame bbox: edge-attached gaps that remain bg, and enclosed
#      islands that become colored in the output.
#    - No color-id-based branch is required for the visible rule; generator enforces a unique winning color per
#      shape so the solver's tie fallback never matters in practice.
# f) Visual cleanliness:
#    - Notches are shallow boundary bites rather than deep tunnels, so the frame stays readable.
#    - Hole placement symmetry and border padding keep the frame uncluttered.
# g) New solver assumptions and generator guarantees:
#    - Solver assumes every fillable hole is a 4-connected non-frame component that does not touch the bbox edge.
#      Generator guarantees that with the 2-cell hole margin and 1-cell-deep notches.
#    - Solver assumes each enclosed hole shape has at least one outside sprite match and a strict color winner.
#      Generator builds sprites directly from the chosen hole shapes and winning-color counts.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only enclosed cavities inside a large notched frame using outside sprites.

    Input structure:
    - bg is the dominant color. One large monochrome frame component spans a rectangular bbox, but its boundary can contain bg notches.
    - Inside that bbox, some non-frame regions touch the bbox edge (design gaps) while others are enclosed cavities.
    - Outside the bbox, small sprites are scattered on bg; their shapes identify which enclosed cavity they control.

    Rule:
    - Find the largest monochrome frame component and crop to its bbox.
    - Ignore bbox-edge-connected gaps, because those are border notches that stay bg in the output.
    - For each enclosed cavity, match its shape to outside sprites up to rotation/flip and fill it with the most frequent matching sprite color.

    Key insight: The important split is not frame-color vs non-frame-color, but enclosed cavity vs boundary-connected notch; only enclosed shapes should be recolored.
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

    # Separate non-frame regions into preserved border notches and enclosed fillable holes.
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
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True
        holes.append((((hh, ww), m.tobytes()), (int(hr0), int(hc0)), m))

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
        m = np.zeros((hh, ww), dtype=bool)
        m[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(m):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    out = np.full((th, tw), int(bg), dtype=int)
    out[sub == int(frame_col)] = int(frame_col)
    for key, (hr0, hc0), hmask in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        scol = max(cmap.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        patch = np.full(hmask.shape, int(scol), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hmask)

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
        m = np.array(s, dtype=bool)
        if int(m.sum()) > SHAPE_MAX_AREA:
            continue
        k = _canonical_key(m)
        if k not in seen:
            seen.add(k)
            dedup.append(m)
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
        s
        for s in SHAPE_LIBRARY
        if s.shape[0] <= max_sh and s.shape[1] <= max_sw and int(np.sum(s)) <= SHAPE_MAX_AREA
    ]
    if len(available) < MIN_SHAPES:
        return None

    k_target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)
    even_dims = (th % 2 == 0 or tw % 2 == 0)
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
    output[frame_mask] = int(frame_col)
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
