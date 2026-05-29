# Combo variant of b10624e5 (index 9)
# combo_spec: D3/1+D13/0
# Name: 标记像素精确定位 + 图案与锚块固定间距

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused D3/1 + D13/0 variant
# ---------------------------------------------------
# a) Concrete behavior and trace
#    - Original task:
#      TL gives one red-connected source patch. In each other quadrant the copied patch is
#      scaled by the target red size, mirrored by quadrant, and translated so the transformed
#      red sub-bbox lands directly on the target red anchor.
#    - Fused task:
#      TL no longer has the motif touching the anchor. Instead it shows one detached non-red
#      motif component sitting on exactly one side of the 2x2 red anchor with a visible blank
#      gap between them. Non-TL quadrants contain the red square plus one singleton marker.
#    - The copied object is now the detached motif sprite itself, not the old red-connected
#      component. The red anchor still determines the scale, and the source-side relation still
#      determines whether the motif belongs to the left/right/up/down side after reflection.
#      But the exact slide position along that side is no longer inherited from TL: the marker
#      fixes the motif center there.
#    - Small concrete trace:
#      suppose the TL source motif is to the right of the 2x2 anchor with gap = 1, and after
#      scaling for a 4x4 target anchor the transformed motif becomes 6x4 with visible center
#      at local (2,1). In TR the horizontal reflection moves the motif to the left side of the
#      anchor and the scaled gap becomes 2. If the target anchor top-left is (4,21), then gap
#      alone fixes motif_col = 21 - 2 - 4 = 15. If the marker is at (9,16), center alignment
#      fixes motif_row = 9 - 2 = 7. Both cues are needed to recover the final top-left (7,15).
#      Old anchor-only logic would preserve the TL vertical offset instead of using the marker.
#    - Difference from the original is therefore visible in two places:
#      1. TL source extraction must ignore red-connectivity because the motif is detached.
#      2. Target placement is solved from "side + gap" and "marker center", not from red overlap.
#
# b) Complexity calibration
#    - The source motif is forced onto one side of the anchor with a gap of one background cell.
#      This keeps the gap rule visually obvious without introducing many relation classes.
#    - The motif itself is asymmetric under both LR and UD flips, so quadrant reflections stay
#      readable across 2-4 examples.
#    - Target sizes are restricted to 2..4. Allowing 1x1 would collapse the visible gap and make
#      the fused relation hard to observe.
#    - Marker color is fixed and excluded from the motif palette, so the new placement cue is
#      easy to isolate by eye.
#
# c) Hidden assumptions from the original solver that break here
#    - Original `components_mask(tl != bg)` + "pick the component containing red" assumed the
#      source motif and the red anchor were one connected component. That is false now.
#    - Original alignment by transformed red bbox assumed the motif should land wherever the red
#      region lands. In this variant the red anchor only gives scale and the side-strip for the
#      motif; the exact slide along that strip comes from the marker.
#    - Original stamping could copy red pixels from the transformed patch. That would create a
#      second red block and erase the "anchor is separate" cue, so the new solver stamps only
#      non-red motif pixels in non-TL quadrants.
#    - Original roundtrip-resize assertion depended on the old generator's exact-lossless scaling
#      constraint. This fused variant does not need that; nearest-neighbor scaling is sufficient.
#
# d) Placement legality
#    - The generator picks anchor columns so the motif always fits on the required reflected side
#      with the scaled gap.
#    - It then picks a motif row inside the quadrant and derives the marker from the transformed
#      motif center, so the marker is always in-bounds and consistent with the output rule.
#    - Motif and anchor never overlap because the side-axis translation is derived from the gap.
#
# e) Visual observability
#    - In TL the detached motif plus blank corridor makes the new gap rule explicit.
#    - In non-TL inputs the red square clearly no longer touches the future motif, and the marker
#      visibly sits at the eventual motif center in the output.
#    - Across examples, the same quadrant reflection rule still applies, while the marker changes
#      the motif's slide position relative to the anchor. That is enough evidence for a human to
#      infer both cues.
#
# f) Visual cleanliness
#    - Each non-TL quadrant contains exactly three conceptually separate objects at most:
#      the red anchor, one singleton marker, and one detached motif. There are no layered
#      overlaps except the marker remaining visible at the motif center.
#    - Divider color, marker color, and motif colors are all disjoint.
#
# g) Color inference
#    - No color remapping is used. Colors are copied directly from the TL motif after reflection
#      and scaling.
#    - The palette is deliberately small, and color ids are not used for ranking or ordering.
#
# h) Degeneration metric used after implementation
#    - For each generated sample we can compare the actual target motif row/col against the row/col
#      that old "anchor-aligned full-layout" logic would have produced. The generator forces a
#      noticeable disagreement on the slide axis in every non-TL quadrant, so marker control is
#      not redundant.
#
# i) New solver assumptions audited against the generator
#    - TL contains exactly one detached non-red motif component and one red source anchor.
#    - The detached motif lies entirely on one side of the red anchor with a positive blank gap.
#    - Every non-TL quadrant contains exactly one red square anchor and one singleton marker.
#    - For each target, the marker coordinate is consistent with the reflected/scaled motif center
#      after the side+gap constraint is applied.


BG = 4
DIV = 1
RED = 2
MARKER = 8
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_OPTIONS = (2, 3, 4)
SOURCE_GAP = 1
MARKER_SHIFT_MIN = 2
MAX_RETRIES = 500

MOTIF_H_RANGE = (2, 4)
MOTIF_W_RANGE = (2, 3)
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.9
MOTIF_COLORS_MIN = 2
MOTIF_COLORS_MAX = 4
COLOR_POOL = [0, 3, 5, 6, 7, 9]


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """
    Return the unique full-row/full-col quadrant divider: (mid_row, mid_col).
    """
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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr:
        out = np.fliplr(out)
    if qb:
        out = np.flipud(out)
    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    p = _reflect_patch(patch, qb, qr)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _visible_info(patch: np.ndarray, bg: int) -> tuple[int, int, int, int, int, int]:
    rr, cc = np.where(patch != bg)
    assert rr.size > 0, "transformed motif must contain visible pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    r1, c1 = int(rr.max()), int(cc.max())
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    return r0, c0, r1, c1, center_r, center_c


def _relation_from_boxes(
    anchor_box: tuple[int, int, int, int], motif_box: tuple[int, int, int, int]
) -> tuple[str, int, int]:
    ar0, ac0, ar1, ac1 = anchor_box
    mr0, mc0, mr1, mc1 = motif_box
    if mc0 > ac1:
        return "h", 1, int(mc0 - ac1 - 1)
    if mc1 < ac0:
        return "h", -1, int(ac0 - mc1 - 1)
    if mr0 > ar1:
        return "v", 1, int(mr0 - ar1 - 1)
    if mr1 < ar0:
        return "v", -1, int(ar0 - mr1 - 1)
    raise AssertionError("motif must lie fully on one side of the anchor")


def _transform_relation(axis: str, sign: int, qb: int, qr: int) -> tuple[str, int]:
    if axis == "h" and qr == 1:
        sign *= -1
    if axis == "v" and qb == 1:
        sign *= -1
    return axis, sign


def _scaled_gap(src_gap: int, target_size: int) -> int:
    ratio = target_size / SOURCE_ANCHOR_SIZE
    return max(1, int(round(src_gap * ratio)))


def _stamp_motif(
    out: np.ndarray,
    motif: np.ndarray,
    place_r0: int,
    place_c0: int,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    h, w = out.shape
    rr_p, cc_p = np.where(motif != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < h and 0 <= cc_t < w):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] in (RED, MARKER):
            continue
        out[rr_t, cc_t] = int(motif[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants using marker-centered sliding plus anchor-gap constraints.

    Input structure:
    - A single full-row/full-column divider splits the canvas into four quadrants.
    - Every quadrant contains one red square anchor; the TL quadrant also contains the source motif.
    - In TL the source motif is detached from the red anchor and sits on one side of it with a visible blank gap.
    - Each non-TL quadrant contains one singleton marker pixel in addition to its red anchor.

    Rule:
    - Extract the detached TL motif component and infer which side of the TL anchor it occupies and how large the blank gap is.
    - For each other quadrant, reflect the motif by quadrant side and scale it by the target-anchor-size / source-anchor-size ratio.
    - Keep the transformed motif on the correspondingly reflected side of the target anchor with the scaled gap.
    - Also place it so the transformed motif's visible center lands on the marker pixel; use the gap rule to fix the side-axis translation and the marker to fix the slide-axis translation.

    Key insight: The red anchor no longer tells the full destination by itself; it gives scale and the allowed offset strip, while the singleton marker tells where along that strip the detached motif must be centered.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == set(QUADS), "there must be one red anchor in every quadrant"
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left red anchor must stay 2x2 in this variant"

    markers: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (q_r0, q_r1, q_c0, q_c1) in QUADS.items():
        if key == (0, 0):
            continue
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        pts = np.argwhere((quad != bg) & (quad != RED))
        assert pts.shape[0] == 1, "each non-TL quadrant must contain exactly one marker pixel"
        mr, mc = map(int, pts[0])
        markers[key] = (q_r0 + mr, q_c0 + mc)

    tl = g[0:mid_r, 0:mid_c]
    motif_comps = components_mask((tl != bg) & (tl != RED), connectivity=4)
    assert len(motif_comps) == 1, "top-left quadrant must contain exactly one detached non-red motif component"
    mr0, mc0, mr1, mc1 = bbox_coords(motif_comps[0])
    motif = tl[mr0 : mr1 + 1, mc0 : mc1 + 1].copy()

    src_anchor_box = (sr0, sc0, sr0 + sL - 1, sc0 + sL - 1)
    src_motif_box = (mr0, mc0, mr1, mc1)
    axis, sign, src_gap = _relation_from_boxes(src_anchor_box, src_motif_box)
    assert src_gap >= 1, "source motif must have a visible positive gap from the anchor"

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        motif_s = _scaled_patch(motif, qb, qr, tL)
        _vr0, _vc0, _vr1, _vc1, center_r, center_c = _visible_info(motif_s, bg)
        axis_t, sign_t = _transform_relation(axis, sign, qb, qr)
        gap_t = _scaled_gap(src_gap, tL)

        marker_r, marker_c = markers[(qb, qr)]
        if axis_t == "h":
            place_c0 = tc0 + tL + gap_t if sign_t > 0 else tc0 - gap_t - motif_s.shape[1]
            place_r0 = marker_r - center_r
            assert marker_c - center_c == place_c0, "marker column must agree with the fixed-gap placement"
        else:
            place_r0 = tr0 + tL + gap_t if sign_t > 0 else tr0 - gap_t - motif_s.shape[0]
            place_c0 = marker_c - center_c
            assert marker_r - center_r == place_r0, "marker row must agree with the fixed-gap placement"

        _stamp_motif(out, motif_s, place_r0, place_c0, bg, mid_r, mid_c)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
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


def _assign_motif_colors(filled: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    cells = list(filled)
    random.shuffle(cells)
    max_colors = min(MOTIF_COLORS_MAX, len(cells))
    if max_colors < MOTIF_COLORS_MIN:
        return None
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    palette = COLOR_POOL[:]
    random.shuffle(palette)
    chosen = palette[:n_colors]
    motif = np.full((h, w), BG, dtype=int)
    for i, (r, c) in enumerate(cells):
        motif[r, c] = chosen[i] if i < len(chosen) else random.choice(chosen)
    return motif


def _make_source_motif() -> np.ndarray | None:
    h = random.randint(*MOTIF_H_RANGE)
    w = random.randint(*MOTIF_W_RANGE)
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * h * w)))
    seed = {(random.randrange(h), random.randrange(w))}
    filled = _grow_4conn(h, w, seed, random.randint(min_fill, max_fill))
    motif = _assign_motif_colors(filled, h, w)
    if motif is None:
        return None
    if np.array_equal(motif, np.fliplr(motif)):
        return None
    if np.array_equal(motif, np.flipud(motif)):
        return None
    return motif


def _crop_non_bg(arr: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    pts = np.argwhere(arr != BG)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))


def _build_source_layout(motif: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int]]:
    motif_h, motif_w = motif.shape
    total_h = max(SOURCE_ANCHOR_SIZE, motif_h) + random.randint(1, 3)
    anchor_r0 = random.randint(0, total_h - SOURCE_ANCHOR_SIZE)
    motif_r0 = random.randint(0, total_h - motif_h)
    total_w = SOURCE_ANCHOR_SIZE + SOURCE_GAP + motif_w

    layout = np.full((total_h, total_w), BG, dtype=int)
    layout[anchor_r0 : anchor_r0 + SOURCE_ANCHOR_SIZE, 0:SOURCE_ANCHOR_SIZE] = RED
    paste(layout, motif, motif_r0, SOURCE_ANCHOR_SIZE + SOURCE_GAP)

    cropped, (r0, c0, _r1, _c1) = _crop_non_bg(layout)
    return cropped, (anchor_r0 - r0, 0 - c0), (motif_r0 - r0, SOURCE_ANCHOR_SIZE + SOURCE_GAP - c0)


def _choose_target_sizes() -> dict[tuple[int, int], int]:
    while True:
        vals = [random.choice(TARGET_SIZE_OPTIONS) for _ in range(3)]
        if len(set(vals)) >= 2 and max(vals) > SOURCE_ANCHOR_SIZE:
            return {
                (0, 0): SOURCE_ANCHOR_SIZE,
                (0, 1): vals[0],
                (1, 0): vals[1],
                (1, 1): vals[2],
            }


def _old_like_motif_position(
    layout: np.ndarray, qb: int, qr: int, target_size: int, anchor_r: int, anchor_c: int
) -> tuple[int, int]:
    layout_s = _scaled_patch(layout, qb, qr, target_size)
    rr_red, cc_red = np.where(layout_s == RED)
    assert rr_red.size > 0, "scaled source layout must retain red anchor pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_vis, cc_vis = np.where((layout_s != BG) & (layout_s != RED))
    assert rr_vis.size > 0, "scaled source layout must retain motif pixels"
    return place_r0 + int(rr_vis.min()), place_c0 + int(cc_vis.min())


def _choose_anchor_for_side(
    qb: int, qr: int, size: int, motif_s: np.ndarray, axis: str, sign: int, gap_t: int
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    motif_h, motif_w = motif_s.shape
    anchor_r_lo, anchor_r_hi = q_r0, q_r1 - size + 1
    anchor_c_lo, anchor_c_hi = q_c0, q_c1 - size + 1
    if axis == "h":
        if sign > 0:
            anchor_c_hi -= gap_t + motif_w
        else:
            anchor_c_lo += gap_t + motif_w
    else:
        if sign > 0:
            anchor_r_hi -= gap_t + motif_h
        else:
            anchor_r_lo += gap_t + motif_h
    if anchor_r_lo > anchor_r_hi or anchor_c_lo > anchor_c_hi:
        return None
    return random.randint(anchor_r_lo, anchor_r_hi), random.randint(anchor_c_lo, anchor_c_hi)


def _choose_marker_position(
    motif_s: np.ndarray,
    qb: int,
    qr: int,
    anchor_r: int,
    anchor_c: int,
    target_size: int,
    axis: str,
    sign: int,
    gap_t: int,
    old_like_top: tuple[int, int],
) -> tuple[int, int, int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    motif_h, motif_w = motif_s.shape
    _vr0, _vc0, _vr1, _vc1, center_r, center_c = _visible_info(motif_s, BG)

    if axis == "h":
        place_c0 = anchor_c + target_size + gap_t if sign > 0 else anchor_c - gap_t - motif_w
        row_lo = q_r0
        row_hi = q_r1 - motif_h + 1
        if row_lo > row_hi:
            return None
        candidates = list(range(row_lo, row_hi + 1))
        random.shuffle(candidates)
        for place_r0 in candidates:
            if abs(place_r0 - old_like_top[0]) < MARKER_SHIFT_MIN:
                continue
            marker_r = place_r0 + center_r
            marker_c = place_c0 + center_c
            return place_r0, place_c0, marker_r, marker_c
        return None

    place_r0 = anchor_r + target_size + gap_t if sign > 0 else anchor_r - gap_t - motif_h
    col_lo = q_c0
    col_hi = q_c1 - motif_w + 1
    if col_lo > col_hi:
        return None
    candidates = list(range(col_lo, col_hi + 1))
    random.shuffle(candidates)
    for place_c0 in candidates:
        if abs(place_c0 - old_like_top[1]) < MARKER_SHIFT_MIN:
            continue
        marker_r = place_r0 + center_r
        marker_c = place_c0 + center_c
        return place_r0, place_c0, marker_r, marker_c
    return None


def _gen_attempt() -> np.ndarray | None:
    target_sizes = _choose_target_sizes()
    motif = _make_source_motif()
    if motif is None:
        return None

    source_layout, source_anchor_local, source_motif_local = _build_source_layout(motif)
    sar, sac = source_anchor_local
    smr, smc = source_motif_local
    src_axis, src_sign, src_gap = _relation_from_boxes(
        (sar, sac, sar + SOURCE_ANCHOR_SIZE - 1, sac + SOURCE_ANCHOR_SIZE - 1),
        (smr, smc, smr + motif.shape[0] - 1, smc + motif.shape[1] - 1),
    )
    if src_axis != "h" or src_sign != 1 or src_gap != SOURCE_GAP:
        return None

    tl_h, tl_w = source_layout.shape
    if tl_h > MID or tl_w > MID:
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    tl_r0 = random.randint(0, MID - tl_h)
    tl_c0 = random.randint(0, MID - tl_w)
    paste(out, source_layout, tl_r0, tl_c0)

    anchors: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (tl_r0 + sar, tl_c0 + sac)
    }
    markers: dict[tuple[int, int], tuple[int, int]] = {}

    for key, target_size in target_sizes.items():
        if key == (0, 0):
            continue

        qb, qr = key
        motif_s = _scaled_patch(motif, qb, qr, target_size)
        axis_t, sign_t = _transform_relation(src_axis, src_sign, qb, qr)
        gap_t = _scaled_gap(src_gap, target_size)
        anchor_pos = _choose_anchor_for_side(qb, qr, target_size, motif_s, axis_t, sign_t, gap_t)
        if anchor_pos is None:
            return None
        anchor_r, anchor_c = anchor_pos
        old_like_top = _old_like_motif_position(source_layout, qb, qr, target_size, anchor_r, anchor_c)
        marker_pack = _choose_marker_position(
            motif_s, qb, qr, anchor_r, anchor_c, target_size, axis_t, sign_t, gap_t, old_like_top
        )
        if marker_pack is None:
            return None
        place_r0, place_c0, marker_r, marker_c = marker_pack

        anchors[key] = (anchor_r, anchor_c)
        markers[key] = (marker_r, marker_c)
        paste(out, np.full((target_size, target_size), RED, dtype=int), anchor_r, anchor_c)
        out[marker_r, marker_c] = MARKER
        _stamp_motif(out, motif_s, place_r0, place_c0, BG, MID, MID)

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, source_layout, tl_r0, tl_c0)
    for key, target_size in target_sizes.items():
        if key == (0, 0):
            continue
        anchor_r, anchor_c = anchors[key]
        marker_r, marker_c = markers[key]
        paste(inp, np.full((target_size, target_size), RED, dtype=int), anchor_r, anchor_c)
        inp[marker_r, marker_c] = MARKER

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
