# Combo variant of b10624e5 (index 25)
# combo_spec: D4/1+D9/0+D16/0
# Name: 非均匀缩放 + 矩形锚块 + 象限面积比缩放

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace:
#    - The original task uses a centered 13/13 split, square anchors, and one scalar resize ratio
#      taken from anchor size. Here the divider is off-center: only (11,15) or (15,11) are used.
#      This creates three stable target cases in every grid:
#        TR changes only width, BL changes only height, BR swaps both axes.
#    - Example with divider (11,15):
#        TL quadrant = 11x15, TR = 11x11, BL = 15x15, BR = 15x11.
#        If the TL red anchor is 3x2, then the copied red rectangles become:
#          TR: 3x1   because rows scale by 11/11 = 1 and cols by 11/15.
#          BL: 4x2   because rows scale by 15/11 and cols by 15/15 = 1.
#          BR: 4x1   because rows scale by 15/11 and cols by 11/15.
#      The old centered task would treat all of these with one scalar scale; the new task must keep
#      the two axes separate. BR is the important support case: its area ratio is 1, but the motif is
#      still distorted because the quadrant aspect flips, so "overall amount" and "axis split" are no
#      longer the same piece of information.
# b) Complexity calibration:
#    - Divider choices are restricted to two mirrored patterns. That keeps the rule readable:
#      one quadrant is smaller than TL, one is larger, and one has equal area but different aspect.
#    - The source anchor is always a non-square rectangle, chosen as 3x2 in wide TL quadrants and
#      2x3 in tall TL quadrants. This keeps every target anchor non-square after resizing.
#    - The motif is built on a small logical grid and expanded by the source anchor rectangle size.
#      That gives visibly blocky structure and keeps anisotropic resize clean enough for humans.
# c) Implicit-assumption audit:
#    - Old code assumed `size = max(h,w)` captured anchor information. That breaks because anchors are
#      rectangles and the task requires separate row/col scales.
#    - Old code assumed the divider was centered, so all quadrant areas matched. That breaks because
#      scale is now tied to the off-center quadrant geometry.
#    - Old code used one ratio for both axes. That breaks because TR/BL/BR deliberately realize
#      horizontal-only, vertical-only, and two-axis cases.
#    - Old roundtrip-resize assertions are no longer valid: area-driven non-integer ratios such as
#      15/11 and 11/15 are intentionally lossy for general motifs.
# d) Placement legality:
#    - The generator computes each scaled patch first, reads back the scaled red bbox, and uses that
#      exact rectangle as the target anchor shape. That makes anchor alignment unambiguous.
#    - Anchor placement prefers positions where the full copied motif stays inside the target quadrant,
#      but still falls back to any legal anchor placement if needed. Divider cells are never painted.
# e) Visual observability:
#    - Every sample contains all three support cases at once: width-only change, height-only change,
#      and same-area-but-anisotropic change. This makes the fused rule inferable from a few examples.
#    - The logical motif is rejected if it is symmetric under left-right or up-down reflection, and it
#      must contain non-red pixels in the anchor's logical row and column. That keeps both mirroring
#      and one-axis stretching visible.
# f) Visual cleanliness:
#    - Divider color is excluded from motif colors so the off-center split remains visually obvious.
#    - Non-TL input quadrants contain only a single red rectangle; all extra colored structure appears
#      only in the output, preserving a clean "given anchor -> reconstructed motif" reading.
# g) Change inferability:
#    - No color remapping is introduced. Colors are copied directly, so the only variables to infer are
#      quadrant-relative reflection and non-uniform geometry.
# h) Quantified anti-degeneracy target:
#    - BR always satisfies `target_area == source_area` but `(scale_h, scale_w) != (1,1)`.
#      Therefore any generator sample that behaves like uniform area scaling would be wrong.
# i) New solver assumptions introduced here:
#    - Exactly one filled red rectangle exists in each quadrant.
#    - The TL motif is the only TL non-background component containing red.
#    - The generator guarantees that the scaled red bbox of each copied patch matches the target anchor.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
DIVIDER_CHOICES = [(11, 15), (15, 11)]
LOGICAL_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 500


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _quadrants(h: int, w: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Return inclusive bounds for the four quadrants."""
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _quad_shape(bounds: tuple[int, int, int, int]) -> tuple[int, int]:
    """Return quadrant shape as (height, width)."""
    r0, r1, c0, c1 = bounds
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _quad_area(bounds: tuple[int, int, int, int]) -> int:
    """Return the area of a quadrant."""
    qh, qw = _quad_shape(bounds)
    return qh * qw


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize with independent row/column targets."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resized patch must keep positive shape"
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _red_bbox_dims(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return the red bounding box as (r0, c0, h, w)."""
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "patch must contain red pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    return r0, c0, int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the top-left motif beside the other red rectangles using quadrant-shaped anisotropic scaling.

    Input structure:
    - One full colored row and one full colored column split the board into four quadrants, and the split
      is intentionally off-center so quadrant heights and widths differ.
    - Each quadrant contains exactly one filled red rectangle; the TL rectangle is attached to the only
      full colored motif, while the other quadrants contain only their red anchor.
    - The TL motif is a single non-background connected component containing that TL red rectangle.

    Rule:
    - Extract the TL component containing red as the source patch.
    - Reflect it left-right for right-side quadrants and up-down for bottom quadrants.
    - Resize rows by `(target_quadrant_height / source_quadrant_height)` and columns by
      `(target_quadrant_width / source_quadrant_width)`, then align the scaled red bbox onto the target
      red rectangle and stamp non-background cells without overwriting existing red anchor pixels.

    Key insight:
    - The off-center quadrants encode the overall size change through their areas, but the rectangular
      anchor and quadrant aspect together reveal how that change splits across height and width. One
      target keeps equal area to TL yet still distorts, so the task cannot be solved with one scalar.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    quads = _quadrants(h, w, mid_r, mid_c)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        ah = int(r1 - r0 + 1)
        aw = int(c1 - c0 + 1)
        assert ah != aw, "anchors must be non-square rectangles in this combo"
        assert len(pts) == ah * aw, "red anchor must be a filled rectangle"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), ah, aw)

    assert set(anchors) == set(quads), "need exactly one red anchor in each quadrant"

    tl_r0, tl_r1, tl_c0, tl_c1 = quads[(0, 0)]
    tl = g[tl_r0 : tl_r1 + 1, tl_c0 : tl_c1 + 1]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with red"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    src_qh, src_qw = _quad_shape(quads[(0, 0)])
    src_area = _quad_area(quads[(0, 0)])

    for (qb, qr), (tr0, tc0, target_h, target_w) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        tgt_qh, tgt_qw = _quad_shape(quads[(qb, qr)])
        scale_h = float(tgt_qh) / float(src_qh)
        scale_w = float(tgt_qw) / float(src_qw)
        new_h = max(1, int(round(patch_t.shape[0] * scale_h)))
        new_w = max(1, int(round(patch_t.shape[1] * scale_w)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

        red_r0, red_c0, red_h, red_w = _red_bbox_dims(patch_s)
        assert red_h == target_h and red_w == target_w, "scaled red bbox must match target anchor"
        place_r0 = tr0 - red_r0
        place_c0 = tc0 - red_c0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

        assert (tgt_qh * tgt_qw) == _quad_area(quads[(qb, qr)])
        assert abs(scale_h * scale_w - (_quad_area(quads[(qb, qr)]) / src_area)) < 1e-9

    return out


def _choose_dividers() -> tuple[int, int]:
    """Use only the two asymmetric divider patterns that realize the three support cases cleanly."""
    return random.choice(DIVIDER_CHOICES)


def _source_anchor_shape(tl_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Pick a non-square source anchor whose longer side points along TL's shorter axis.
    This avoids square targets after anisotropic scaling.
    """
    tl_h, tl_w = tl_shape
    return (3, 2) if tl_w > tl_h else (2, 3)


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it contains n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        filled.add(cell)
        frontier.discard(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
    """Assign motif colors while keeping the bbox tight and the palette small."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    non_red = [p for p in filled if p not in red_cells]
    random.shuffle(non_red)
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None

    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = random.sample(COLOR_POOL, n_colors - 1)
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _logical_patch_is_observable(logical: np.ndarray, ar: int, ac: int) -> bool:
    """Reject motifs where reflection or one-axis scaling would be hard to notice."""
    non_red = logical != BG
    same_row = any(non_red[ar, c] and logical[ar, c] != RED for c in range(logical.shape[1]))
    same_col = any(non_red[r, ac] and logical[r, ac] != RED for r in range(logical.shape[0]))
    asym_h = not np.array_equal(logical, np.fliplr(logical))
    asym_v = not np.array_equal(logical, np.flipud(logical))
    return same_row and same_col and asym_h and asym_v


def _source_patch(
    source_shape: tuple[int, int],
    tl_shape: tuple[int, int],
) -> tuple[np.ndarray, int, int] | None:
    """
    Build the TL motif on a logical grid, then expand each logical cell by the source anchor rectangle.
    The red anchor occupies exactly one logical cell, so its scaled bbox remains easy to predict.
    """
    ah, aw = source_shape
    tl_h, tl_w = tl_shape
    candidates = [(lh, lw) for lh, lw in LOGICAL_DIMS if lh * ah <= tl_h and lw * aw <= tl_w]
    if not candidates:
        return None

    lh, lw = random.choice(candidates)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
    if logical is None or not _logical_patch_is_observable(logical, ar, ac):
        return None
    patch = np.repeat(np.repeat(logical, ah, axis=0), aw, axis=1)
    return patch, ar * ah, ac * aw


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    source_quad_shape: tuple[int, int],
    target_quad_shape: tuple[int, int],
) -> np.ndarray:
    """Reflect and resize a patch according to quadrant height/width ratios."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    src_qh, src_qw = source_quad_shape
    tgt_qh, tgt_qw = target_quad_shape
    out_h = max(1, int(round(p.shape[0] * tgt_qh / src_qh)))
    out_w = max(1, int(round(p.shape[1] * tgt_qw / src_qw)))
    return _resize_nn(p, out_h, out_w)


def _choose_anchor(
    bounds: tuple[int, int, int, int],
    anchor_shape: tuple[int, int],
    patch_s: np.ndarray,
) -> tuple[int, int]:
    """Pick a target anchor top-left, preferring placements where the full motif fits."""
    q_r0, q_r1, q_c0, q_c1 = bounds
    ah, aw = anchor_shape
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - ah + 1
    anch_c0, anch_c1 = q_c0, q_c1 - aw + 1
    assert anch_r0 <= anch_r1 and anch_c0 <= anch_c1, "anchor rectangle must fit inside its quadrant"

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, mid_r: int, mid_c: int) -> None:
    """Stamp the scaled patch onto the output, aligned by the red bbox top-left."""
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    h, w = out.shape
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < h)
        & (tgt_cc >= 0)
        & (tgt_cc < w)
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    """Build output first, erase non-TL motifs, and keep only cases that the solver reproduces exactly."""
    mid_r, mid_c = _choose_dividers()
    quads = _quadrants(GRID_SIZE, GRID_SIZE, mid_r, mid_c)
    source_quad_shape = _quad_shape(quads[(0, 0)])
    source_shape = _source_anchor_shape(source_quad_shape)

    source = _source_patch(source_shape, source_quad_shape)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled: dict[tuple[int, int], np.ndarray] = {}
    target_shapes: dict[tuple[int, int], tuple[int, int]] = {(0, 0): source_shape}
    for key, bounds in quads.items():
        if key == (0, 0):
            continue
        patch_s = _scaled_patch(patch, key[0], key[1], source_quad_shape, _quad_shape(bounds))
        red_r0, red_c0, red_h, red_w = _red_bbox_dims(patch_s)
        if red_h == red_w:
            return None
        if red_r0 < 0 or red_c0 < 0:
            return None
        scaled[key] = patch_s
        target_shapes[key] = (red_h, red_w)

    br_area_ratio = _quad_area(quads[(1, 1)]) / _quad_area(quads[(0, 0)])
    br_shape = target_shapes[(1, 1)]
    if not (abs(br_area_ratio - 1.0) < 1e-9 and br_shape != source_shape):
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_bounds = quads[(0, 0)]
    tl_patch_r0 = random.randint(tl_bounds[0], tl_bounds[1] - patch.shape[0] + 1)
    tl_patch_c0 = random.randint(tl_bounds[2], tl_bounds[3] - patch.shape[1] + 1)
    anchors[(0, 0)] = (tl_patch_r0 + anchor_r_in_patch, tl_patch_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(quads[key], target_shapes[key], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    paste(out, patch, tl_patch_r0, tl_patch_c0)
    for key, shape in target_shapes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full(shape, RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key], mid_r, mid_c)

    inp = out.copy()
    for key, shape in target_shapes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = quads[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full(shape, RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
