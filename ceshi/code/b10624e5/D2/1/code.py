# Variant of b10624e5
# Dimension: D2 — 源对象：左上象限锚块旁的彩色图案 (src)
# Change: 左上象限中与红色锚块相邻的彩色像素构成源图案... → 控制像素指定源象限

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# a) Concrete behavior and trace
# - The base task still has one divider row and one divider column splitting a
#   27x27 board into four 13x13 quadrants, with one red square anchor in each
#   quadrant.
# - The changed part is only the source selection. The colorful motif is no
#   longer forced to live in TL. Instead, the input contains exactly two special
#   control pixels of one reserved color on the outer border.
# - One control pixel lies on the top or bottom edge and fixes the source row
#   half. One control pixel lies on the left or right edge and fixes the source
#   column half. Their half-position also agrees with the same quadrant, so the
#   cue is visually redundant rather than numeric.
# - Example trace: if controls are at the top edge above the right half and at
#   the right edge beside the top half, the source is TR. Old logic would still
#   read TL as the source and mirror relative to TL. New logic must extract the
#   TR component containing the TR red anchor, then copy it to TL/BL/BR using
#   scale ratios to each target anchor and flips relative to TR.
#
# b) Complexity calibration
# - The variant adds one new inference step only: decode source quadrant from
#   border controls. Everything else stays identical: one motif, four anchors,
#   nearest-neighbor resizing, and quadrant-based flips.
# - To keep the new rule observable, controls always appear as a pair and the
#   source is sampled from all four quadrants, so the dataset does not collapse
#   into the original TL-only behavior.
#
# c) Hidden assumptions in the original solver that break here
# - `assert (0, 0) in anchors` no longer holds; the source anchor can be in any
#   quadrant.
# - Extracting the source component from `g[0:mid_r, 0:mid_c]` is invalid once
#   the motif may be elsewhere.
# - Flips based only on the target quadrant (`qr == 1`, `qb == 1`) assume TL is
#   the reference orientation. The new rule needs relative flips:
#   horizontal iff target column half differs from source column half,
#   vertical iff target row half differs from source row half.
# - Border scans now have meaningful non-divider, non-background pixels, so the
#   generator must keep anchors and the motif away from the outer frame; without
#   that, control detection would become ambiguous.
#
# d) Placement legality
# - Generator guarantees exactly one red square per quadrant.
# - The source motif stays at least one cell away from the outer border, so the
#   border controls are isolated components and never fuse with the motif.
# - Non-source anchors also stay one cell away from the outer border, so input
#   border colors are: background, divider endpoints, and the two control
#   pixels only.
# - Target motifs may clip at bounds as in the base task, but stamping skips the
#   divider and the control pixels, so the markers remain readable.
#
# e) Visual observability
# - The source motif itself is still asymmetric and multi-colored, so flips are
#   visible.
# - The new behavior is human-visible from a few examples because the source
#   quadrant changes while the two control pixels move with it.
#
# f) Visual cleanliness
# - Only two control pixels are added, both on the border.
# - They use a reserved color not used inside motifs, which prevents visual
#   confusion with motif colors and makes the border cue stand out.
#
# g) Color-rule audit
# - No ordering by color id is used. The control color is just a fixed special
#   marker, visibly distinct from bg/divider/red and excluded from motif colors.
#
# Post-change solver assumptions and generator guarantees
# - Solver assumes there is exactly one control color on the border and exactly
#   two border control pixels, one horizontal-edge control and one vertical-edge
#   control, both decoding to the same source quadrant.
# - Solver assumes the source quadrant contains exactly one non-background
#   component with red pixels: the motif+anchor component.
# - Generator enforces all of those conditions directly before oracle-checking
#   `main(input) == output`.
# ---------------------------------------------------------------------------


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


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(h - 1, c) for c in range(w)]
    coords += [(r, 0) for r in range(1, h - 1)]
    coords += [(r, w - 1) for r in range(1, h - 1)]
    return coords


def _find_source_quadrant(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> tuple[tuple[int, int], int]:
    h, w = g.shape
    div = int(g[mid_r, mid_c])

    border_colors = {int(g[r, c]) for r, c in _border_coords(h, w)}
    border_colors.discard(bg)
    border_colors.discard(div)
    assert len(border_colors) == 1, "border must contain exactly one control color besides bg/divider"
    ctrl = int(next(iter(border_colors)))

    ctrl_pts = [(int(r), int(c)) for r, c in _border_coords(h, w) if int(g[r, c]) == ctrl]
    assert len(ctrl_pts) == 2, "must have exactly two border control pixels"

    src_keys = set()
    seen_row_edge = False
    seen_col_edge = False
    for r, c in ctrl_pts:
        if r == 0 or r == h - 1:
            seen_row_edge = True
            qb = 0 if r == 0 else 1
            qr = 0 if c < mid_c else 1
        else:
            assert c == 0 or c == w - 1, "control pixels must lie on the outer border"
            seen_col_edge = True
            qb = 0 if r < mid_r else 1
            qr = 0 if c == 0 else 1
        src_keys.add((qb, qr))

    assert seen_row_edge and seen_col_edge, "controls must include one horizontal-edge and one vertical-edge mark"
    assert len(src_keys) == 1, "the two control pixels must point to the same source quadrant"
    return next(iter(src_keys)), ctrl


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a quadrant motif using border controls to choose the source anchor+pattern pair.

    Input structure:
    - A full-row/full-column divider splits the board into four quadrants.
    - Each quadrant contains one red square anchor block, but only one quadrant also contains
      the attached colorful motif.
    - Two special control pixels on the outer border identify which quadrant is the source.

    Rule:
    - Decode the border control pixels to find the source quadrant, then extract the non-bg
      component in that quadrant that contains the red anchor.
    - For every other quadrant, scale that source component by the ratio
      `(target anchor size / source anchor size)`.
    - Flip left-right when the target is in the opposite column half from the source, flip
      up-down when it is in the opposite row half, align the scaled red region to the target
      anchor, and stamp non-bg pixels.

    Key insight: The red square still encodes scale, but the border control pixels now decide
    which quadrant supplies the motif orientation and contents.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    src_key, ctrl = _find_source_quadrant(g, bg, mid_r, mid_c)

    red = components_mask(g == 2, connectivity=4)
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

    assert len(anchors) == 4, "there must be exactly four quadrant anchors"
    assert src_key in anchors, "source quadrant must contain a red anchor"
    sr0, sc0, sL = anchors[src_key]
    assert sL > 0, "source red anchor must have positive size"

    quad_bounds = {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }
    q_r0, q_r1, q_c0, q_c1 = quad_bounds[src_key]
    src_quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]

    comps = components_mask(src_quad != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(src_quad[r, c] == 2 for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "source quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = src_quad[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    def resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        src_h, src_w = img.shape
        if new_h <= 0 or new_w <= 0:
            return img.copy()
        rr = (np.floor(np.arange(new_h) * src_h / new_h)).astype(int)
        cc = (np.floor(np.arange(new_w) * src_w / new_w)).astype(int)
        rr = np.clip(rr, 0, src_h - 1)
        cc = np.clip(cc, 0, src_w - 1)
        return img[rr[:, None], cc[None, :]]

    for dst_key, (tr0, tc0, tL) in anchors.items():
        if dst_key == src_key:
            continue

        patch_t = patch
        if dst_key[1] != src_key[1]:
            patch_t = np.fliplr(patch_t)
        if dst_key[0] != src_key[0]:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = resize_nn(patch_t, new_h, new_w)
        roundtrip = resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == 2)
        assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == 2 or out[rr_t, cc_t] == ctrl:
                continue
            out[rr_t, cc_t] = val

    return out


# --------------- generator ---------------

BG = 4
DIV = 1
RED = 2
CTRL = 6
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400

HALF_CENTER = {0: 6, 1: 20}


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _scaled_patch(patch: np.ndarray, src_key: tuple[int, int], dst_key: tuple[int, int], target_size: int) -> np.ndarray:
    p = patch
    if dst_key[1] != src_key[1]:
        p = np.fliplr(p)
    if dst_key[0] != src_key[0]:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _roundtrip_safe(patch: np.ndarray, src_key: tuple[int, int], target_sizes: dict[tuple[int, int], int]) -> bool:
    for dst_key, size in target_sizes.items():
        if dst_key == src_key:
            continue
        patch_t = patch
        if dst_key[1] != src_key[1]:
            patch_t = np.fliplr(patch_t)
        if dst_key[0] != src_key[0]:
            patch_t = np.flipud(patch_t)
        ratio = size / SOURCE_ANCHOR_SIZE
        h = max(1, int(round(patch_t.shape[0] * ratio)))
        w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, h, w)
        if not np.array_equal(_resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1]), patch_t):
            return False
    return True


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


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    h: int,
    w: int,
) -> np.ndarray | None:
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


def _control_points(src_key: tuple[int, int]) -> list[tuple[int, int]]:
    qb, qr = src_key
    return [
        (0 if qb == 0 else GRID_SIZE - 1, HALF_CENTER[qr]),
        (HALF_CENTER[qb], 0 if qr == 0 else GRID_SIZE - 1),
    ]


def _source_top_left_range(src_key: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[range, range]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[src_key]
    ph, pw = map(int, patch_shape)

    top_margin = 1 if q_r0 == 0 else 0
    bottom_margin = 1 if q_r1 == GRID_SIZE - 1 else 0
    left_margin = 1 if q_c0 == 0 else 0
    right_margin = 1 if q_c1 == GRID_SIZE - 1 else 0

    r_lo = q_r0 + top_margin
    r_hi = q_r1 - ph + 1 - bottom_margin
    c_lo = q_c0 + left_margin
    c_hi = q_c1 - pw + 1 - right_margin
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _patch_hits_controls(patch_s: np.ndarray, anchor_r: int, anchor_c: int, ctrl_pts: set[tuple[int, int]]) -> bool:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    src_rr, src_cc = np.where(patch_s != BG)
    for rr0, cc0 in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
        if (place_r0 + rr0, place_c0 + cc0) in ctrl_pts:
            return True
    return False


def _choose_anchor(
    key: tuple[int, int],
    size: int,
    patch_s: np.ndarray,
    ctrl_pts: set[tuple[int, int]],
) -> tuple[int, int]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    top_margin = 1 if q_r0 == 0 else 0
    bottom_margin = 1 if q_r1 == GRID_SIZE - 1 else 0
    left_margin = 1 if q_c0 == 0 else 0
    right_margin = 1 if q_c1 == GRID_SIZE - 1 else 0

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0 = q_r0 + top_margin
    anch_r1 = q_r1 - size + 1 - bottom_margin
    anch_c0 = q_c0 + left_margin
    anch_c1 = q_c1 - size + 1 - right_margin

    if anch_r0 > anch_r1 or anch_c0 > anch_c1:
        raise AssertionError("anchor margins make placement impossible")

    def candidates(r_lo: int, r_hi: int, c_lo: int, c_hi: int) -> list[tuple[int, int]]:
        if r_lo > r_hi or c_lo > c_hi:
            return []
        return [(r, c) for r in range(r_lo, r_hi + 1) for c in range(c_lo, c_hi + 1)]

    full_fit = candidates(max(fit_r0, anch_r0), min(fit_r1, anch_r1), max(fit_c0, anch_c0), min(fit_c1, anch_c1))
    full_fit_safe = [rc for rc in full_fit if not _patch_hits_controls(patch_s, rc[0], rc[1], ctrl_pts)]
    if full_fit_safe:
        return random.choice(full_fit_safe)
    if full_fit:
        return random.choice(full_fit)

    anchor_only = candidates(anch_r0, anch_r1, anch_c0, anch_c1)
    anchor_only_safe = [rc for rc in anchor_only if not _patch_hits_controls(patch_s, rc[0], rc[1], ctrl_pts)]
    if anchor_only_safe:
        return random.choice(anchor_only_safe)
    return random.choice(anchor_only)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < GRID_SIZE)
        & (tgt_cc >= 0)
        & (tgt_cc < GRID_SIZE)
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    keep = (out[tgt_rr, tgt_cc] != RED) & (out[tgt_rr, tgt_cc] != CTRL)
    out[tgt_rr[keep], tgt_cc[keep]] = patch_s[src_rr[keep], src_cc[keep]]


def _gen_attempt() -> np.ndarray | None:
    src_key = random.choice(list(QUADS))
    ctrl_pts = set(_control_points(src_key))

    target_sizes = {key: random.randint(*TARGET_SIZE_RANGE) for key in QUADS}
    target_sizes[src_key] = SOURCE_ANCHOR_SIZE

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != src_key)
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if not _roundtrip_safe(patch, src_key, target_sizes):
        return None

    source_rr, source_cc = _source_top_left_range(src_key, patch.shape)
    if not source_rr or not source_cc:
        return None
    src_patch_r0 = random.choice(list(source_rr))
    src_patch_c0 = random.choice(list(source_cc))

    scaled = {
        key: _scaled_patch(patch, src_key, key, target_sizes[key])
        for key in QUADS
        if key != src_key
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[src_key] = (src_patch_r0 + anchor_r_in_patch, src_patch_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key, target_sizes[key], patch_s, ctrl_pts)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    for r, c in ctrl_pts:
        out[r, c] = CTRL
    paste(out, patch, src_patch_r0, src_patch_c0)
    for key, size in target_sizes.items():
        if key == src_key:
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])
    for r, c in ctrl_pts:
        out[r, c] = CTRL

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == src_key:
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
    for r, c in ctrl_pts:
        inp[r, c] = CTRL

    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D2-variant input")
