# Variant of b10624e5
# Dimension: D3 — 目标对象：其他象限的红色锚块定位 (dst)
# Change: 其他象限中红色锚块的相邻位置作为目标，图案根据锚块尺寸和象限关系进行缩放和反射后放置... → 目标位置由标记像素精确指定

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D3 marker-alignment variant
# ---------------------------------------------------
# spec:
#   core_change:
#     old: non-TL quadrants used the red anchor's top-left as the placement key.
#     new: non-TL quadrants still contain the red square, but it now provides scale only.
#          A separate singleton marker pixel gives the destination. The copied motif is
#          centered on that marker, so the solver must read both objects.
#   concrete_objects:
#     divider: same full row/full column cross as the original task.
#     TL quadrant: unchanged; one connected motif containing the 2x2 red source anchor.
#     other quadrants in input: exactly one red square anchor and exactly one singleton
#       marker pixel of a dedicated color. No other foreground pixels appear there.
#     other quadrants in output: the scaled/reflected non-red part of the TL motif is
#       pasted around the marker; the marker and the red anchor both remain visible.
#   behavior_trace:
#     example:
#       - transformed target motif has visible bbox rows 1..5, cols 0..4 inside its local
#         canvas; its discrete visible center is ((1+5)//2, (0+4)//2) = (3,2).
#       - old logic with anchor at (18,20) would place the whole patch by red-top-left.
#       - new logic with marker at (22,16) instead sets top-left = (22-3, 16-2), so the
#         visible motif is centered on the marker regardless of the anchor position.
#       - because target quadrants only paste non-red pixels, the red square is preserved
#         as an independent scale cue rather than a placement cue.
#   complexity_calibration:
#     - The marker color is fixed and excluded from the TL motif palette so the new cue is
#       visually obvious.
#     - The copied motif keeps the original reflection/scale logic, so D3 changes only the
#       destination mechanism, not the transformation family.
#     - Marker/anchor placements are forced to disagree with the old anchor-based placement,
#       preventing degeneration back into the original task.
#   implicit_assumptions_checked:
#     - Original code assumed "red bbox in transformed patch must overlap target red bbox".
#       That assumption is removed. The new solver uses red only to infer target scale.
#     - Original code could stamp red pixels from the transformed patch. That would create a
#       second red block when anchor and marker disagree, so the new solver stamps only
#       non-red foreground in non-TL quadrants.
#     - Marker alignment must be human-visible. Therefore the center is defined on the
#       visible non-red transformed motif bbox, not on hidden red cells.
#   placement_legality:
#     - The generator chooses marker positions so the full visible motif fits inside the
#       quadrant and does not overlap the target red square.
#     - Marker pixels are kept at least one cell away from the anchor, reducing clutter.
#   observability:
#     - Across 2-4 examples, a human can see that quadrant-side still controls mirroring,
#       red-square size still controls scale, and the new singleton marker controls where
#       the motif appears because the motif is centered on it rather than attached to red.


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
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 5)
FREEFORM_W_RANGE = (3, 5)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400
MARKER_SHIFT_MIN = 2
VISIBLE_BBOX_MAX = 9


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
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    return p


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    p = _reflect_patch(patch, qb, qr)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _visible_info(patch: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, int, int, int, int, int, int]:
    mask = (patch != bg) & (patch != RED)
    rr, cc = np.where(mask)
    assert rr.size > 0, "transformed patch must contain visible non-red pixels"
    vr0, vc0 = int(rr.min()), int(cc.min())
    vr1, vc1 = int(rr.max()), int(cc.max())
    center_r = (vr0 + vr1) // 2
    center_c = (vc0 + vc1) // 2
    return rr, cc, vr0, vc0, vr1, vc1, center_r, center_c


def _stamp_visible(out: np.ndarray, patch: np.ndarray, place_r0: int, place_c0: int, bg: int,
                   mid_r: int, mid_c: int) -> None:
    h, w = out.shape
    rr_p, cc_p = np.where((patch != bg) & (patch != RED))
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < h and 0 <= cc_t < w):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] in (RED, MARKER):
            continue
        out[rr_t, cc_t] = int(patch[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a reflected/scaled motif across quadrants, but center it on marker pixels.

    Input structure:
    - A single full-row/full-column divider splits the canvas into four quadrants.
    - Every quadrant contains one red square anchor; the TL quadrant also contains the full source motif.
    - Each non-TL quadrant contains one extra singleton marker pixel that specifies the destination point.

    Rule:
    - Extract the TL motif component that contains the TL red anchor.
    - For every other quadrant, reflect by quadrant side and scale by the ratio of target red size to source red size.
    - Ignore transformed red pixels when copying; instead paste only the visible non-red part so that its discrete center aligns to the marker pixel.

    Key insight: Red anchors still encode scale, but the singleton marker now controls placement, so the solver must decouple "how big" from "where".
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

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

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    _sr0, _sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

    markers: dict[tuple[int, int], tuple[int, int]] = {}
    for key, (q_r0, q_r1, q_c0, q_c1) in QUADS.items():
        if key == (0, 0):
            continue
        if key not in anchors:
            raise AssertionError("each non-TL quadrant must contain a red anchor block")
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        pts = np.argwhere((quad != bg) & (quad != RED))
        assert pts.shape[0] == 1, "each non-TL quadrant must contain exactly one marker pixel"
        mr, mc = map(int, pts[0])
        markers[key] = (q_r0 + mr, q_c0 + mc)

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (_tr0, _tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_s = _scaled_patch(patch, qb, qr, tL)
        roundtrip = _resize_nn(patch_s, patch.shape[0], patch.shape[1])
        assert np.array_equal(roundtrip, _reflect_patch(patch, qb, qr)), "scaling must preserve motif information"

        rr_vis, cc_vis, _vr0, _vc0, _vr1, _vc1, center_r, center_c = _visible_info(patch_s, bg)
        mr, mc = markers[(qb, qr)]
        place_r0 = mr - center_r
        place_c0 = mc - center_c

        _ = rr_vis, cc_vis
        _stamp_visible(out, patch_s, place_r0, place_c0, bg, mid_r, mid_c)

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


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]],
                   h: int, w: int) -> np.ndarray | None:
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
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]
    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    s = SOURCE_ANCHOR_SIZE
    use_block = needs_block or random.random() < BLOCK_PROB
    if use_block:
        lh, lw = random.choice(BLOCK_DIMS)
        n_cells = lh * lw
        min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
        max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
        ar, ac = random.randrange(lh), random.randrange(lw)
        filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
        logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _choose_anchor(qb: int, qr: int, size: int) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    return random.randint(q_r0, q_r1 - size + 1), random.randint(q_c0, q_c1 - size + 1)


def _anchor_mask_cells(anchor_r: int, anchor_c: int, size: int) -> set[tuple[int, int]]:
    return {
        (anchor_r + dr, anchor_c + dc)
        for dr in range(size)
        for dc in range(size)
    }


def _choose_marker(qb: int, qr: int, patch_s: np.ndarray, anchor_r: int, anchor_c: int,
                   size: int) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr_vis, cc_vis, vr0, vc0, vr1, vc1, center_r, center_c = _visible_info(patch_s, BG)
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must still contain a red block for scale inference"
    red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())

    row_lo = q_r0 + (center_r - vr0)
    row_hi = q_r1 - (vr1 - center_r)
    col_lo = q_c0 + (center_c - vc0)
    col_hi = q_c1 - (vc1 - center_c)
    if row_lo > row_hi or col_lo > col_hi:
        return None

    anchor_cells = _anchor_mask_cells(anchor_r, anchor_c, size)
    candidates = [(r, c) for r in range(row_lo, row_hi + 1) for c in range(col_lo, col_hi + 1)]
    random.shuffle(candidates)

    old_place_r0 = anchor_r - red_r0
    old_place_c0 = anchor_c - red_c0
    for mr, mc in candidates:
        if anchor_r - 1 <= mr <= anchor_r + size and anchor_c - 1 <= mc <= anchor_c + size:
            continue
        new_place_r0 = mr - center_r
        new_place_c0 = mc - center_c
        if max(abs(new_place_r0 - old_place_r0), abs(new_place_c0 - old_place_c0)) < MARKER_SHIFT_MIN:
            continue

        conflict = False
        for rr0, cc0 in zip(rr_vis.tolist(), cc_vis.tolist(), strict=False):
            tgt = (new_place_r0 + rr0, new_place_c0 + cc0)
            if tgt in anchor_cells:
                conflict = True
                break
        if not conflict:
            return mr, mc
    return None


def _stamp_expected(out: np.ndarray, patch_s: np.ndarray, marker_r: int, marker_c: int) -> None:
    _rr_vis, _cc_vis, _vr0, _vc0, _vr1, _vc1, center_r, center_c = _visible_info(patch_s, BG)
    place_r0 = marker_r - center_r
    place_c0 = marker_c - center_c
    _stamp_visible(out, patch_s, place_r0, place_c0, BG, MID, MID)


def _patches_are_usable(scaled: dict[tuple[int, int], np.ndarray]) -> bool:
    for patch_s in scaled.values():
        rr_vis, cc_vis, vr0, vc0, vr1, vc1, _cr, _cc = _visible_info(patch_s, BG)
        _ = rr_vis, cc_vis, vr0, vc0
        vis_h = vr1 - vr0 + 1
        vis_w = vc1 - vc0 + 1
        if vis_h > VISIBLE_BBOX_MAX or vis_w > VISIBLE_BBOX_MAX:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    needs_block = any(s % SOURCE_ANCHOR_SIZE != 0 for k, s in target_sizes.items() if k != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }
    if not _patches_are_usable(scaled):
        return None

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV

    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    paste(out, patch, tl_r0, tl_c0)

    anchors: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    }
    markers: dict[tuple[int, int], tuple[int, int]] = {}

    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        patch_s = scaled[key]
        anchor_r, anchor_c = _choose_anchor(key[0], key[1], size)
        marker = _choose_marker(key[0], key[1], patch_s, anchor_r, anchor_c, size)
        if marker is None:
            return None
        anchors[key] = (anchor_r, anchor_c)
        markers[key] = marker
        paste(out, np.full((size, size), RED, dtype=int), anchor_r, anchor_c)
        out[marker[0], marker[1]] = MARKER
        _stamp_expected(out, patch_s, marker[0], marker[1])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
        mr, mc = markers[key]
        inp[mr, mc] = MARKER

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D3 input")
