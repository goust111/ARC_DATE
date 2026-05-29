# Combo variant of b10624e5 (index 39)
# combo_spec: D5/0+D11/2
# Name: 旋转替代反射 + 先反射后缩放顺序
#
# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace:
#    - Keep the original 27x27 board, the cross divider at row/col 13, one red square
#      anchor per quadrant, and a single TL source motif attached to the TL 2x2 red anchor.
#    - Fusion changes two linked steps at once. D5/0 replaces quadrant-based mirroring with
#      fixed rotations: TR=90 deg clockwise, BL=270 deg clockwise, BR=180 deg. D11/2 then
#      reinterprets "mirror first, then scale" inside that new transform family as
#      "rotate first, then scale".
#    - Small trace: suppose the TL patch is a 3x5 asymmetric component with a 2x2 red block
#      near its upper-left and a colored tail stretching to the right/bottom. For TR with
#      target size 1, the fused rule rotates to 5x3 first and only then downsamples, so the
#      kept pixels are sampled from the rotated frame. A counterfactual D5-only pipeline
#      would first downsample the 3x5 source and only then rotate the smaller image; nearest
#      neighbor keeps different source columns/rows, so the visible non-red pixels move and
#      may even change color. Compared with the original task, TR is also no longer a mirror:
#      its width/height orientation and mass placement follow a quarter-turn instead.
# b) Complexity calibration:
#    - The object vocabulary stays minimal: one divider, four anchors, one source motif, and
#      three copied motifs. That keeps the puzzle in the same family instead of introducing
#      new markers or extra object types.
#    - To make the order change observable, the generator forces TR and BL to use odd target
#      sizes {1, 3}. Those are the ratios where nearest-neighbor sampling most often exposes
#      rotate-then-scale versus scale-then-rotate differences.
#    - To make the rotation change observable, generated samples are rejected unless the fused
#      output also differs from the legacy reflect-then-scale behavior on visible non-red
#      pixels in at least one non-TL quadrant, usually TR or BL.
# c) Implicit assumptions audited in main():
#    - Divider detection still assumes exactly one matching full row/full column pair.
#    - Red connected components are still assumed to be one square anchor per quadrant.
#    - The TL source is still assumed to be the unique TL non-background 4-connected
#      component containing red pixels.
#    - Alignment still uses the top-left cell of the transformed red region; this requires
#      the transformed patch to retain at least one red pixel. The generator preserves that by
#      using a solid 2x2 TL anchor and target sizes in 1..4.
#    - The original roundtrip-resize assertion is intentionally removed. Order-sensitive freeform
#      motifs are exactly the cases that need not be lossless under resize.
# d) Placement legality:
#    - The generator computes each fused transformed patch before anchor placement and chooses
#      anchor positions from that concrete footprint, preferring placements where the whole
#      transformed motif fits inside its quadrant.
#    - Output building still skips divider cells and existing red anchor cells, so copied motifs
#      cannot overwrite the separator or destroy anchors.
# e) Visual observability:
#    - Order sensitivity is checked on visible signatures relative to the red anchor, so hidden
#      red-only differences do not count.
#    - A second check compares full rendered outputs from the fused rule against both baselines:
#      scale-then-rotate and reflect-then-scale. Samples are kept only if those differences
#      survive anchor placement and clipping in the final grid.
# f) Visual cleanliness:
#    - Non-TL input quadrants contain only one red square each. No quadrant contains multiple
#      candidate motifs, so the copied object is easy to isolate.
#    - Anchor placement prefers full-fit placements, which reduces clipping noise and keeps the
#      transformed motif legible.
# g) Color inference:
#    - Colors are arbitrary motif decoration only. No ordering, branching, or indexing depends
#      on color ids beyond the human-visible facts that red is the anchor color and the divider
#      is uniform.
# h) New assumptions rechecked against generate_input():
#    - TR and BL must remain order-sensitive under rotation; the generator explicitly tests that.
#    - The fused rule must differ from the old reflection rule on the same sample; the generator
#      explicitly tests that too.
#    - Each transformed patch must still expose a non-empty red region for alignment; attempts
#      that violate this are rejected before building the final grid.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


ROTATION_K = {
    (0, 0): 0,
    (0, 1): 3,  # 90 degrees clockwise; np.rot90 uses CCW turns
    (1, 0): 1,  # 270 degrees clockwise
    (1, 1): 2,  # 180 degrees
}


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col quadrant divider: (mid_row, mid_col)."""
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
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _rotate_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Rotate a patch according to the target quadrant."""
    return np.rot90(patch, ROTATION_K[(qb, qr)])


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Legacy baseline used only for generator rejection checks."""
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _resize_to_anchor(patch: np.ndarray, target_size: int, source_size: int) -> np.ndarray:
    """Resize a patch so the red anchor scales from source_size to target_size."""
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(patch.shape[0] * ratio)))
    new_w = max(1, int(round(patch.shape[1] * ratio)))
    return _resize_nn(patch, new_h, new_w)


def _transform_rotate_then_scale(
    patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int
) -> np.ndarray:
    """Fused rule: quadrant rotation first, nearest-neighbor scaling second."""
    return _resize_to_anchor(_rotate_patch(patch, qb, qr), target_size, source_size)


def _transform_scale_then_rotate(
    patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int
) -> np.ndarray:
    """Counterfactual D5-only baseline used for order-sensitivity checks."""
    return _rotate_patch(_resize_to_anchor(patch, target_size, source_size), qb, qr)


def _transform_reflect_then_scale(
    patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int
) -> np.ndarray:
    """Legacy original-task baseline used for fusion observability checks."""
    return _resize_to_anchor(_reflect_patch(patch, qb, qr), target_size, source_size)


def _visible_signature(patch_s: np.ndarray, bg: int, red: int) -> dict[tuple[int, int], int] | None:
    """
    Canonical visible motif pixels relative to the transformed red anchor.
    Red cells are omitted because stamping never overwrites the target red anchor.
    """
    rr_red, cc_red = np.where(patch_s == red)
    if rr_red.size == 0:
        return None
    ar0, ac0 = int(rr_red.min()), int(cc_red.min())
    sig: dict[tuple[int, int], int] = {}
    rr, cc = np.where((patch_s != bg) & (patch_s != red))
    for r, c in zip(rr.tolist(), cc.tolist(), strict=False):
        sig[(r - ar0, c - ac0)] = int(patch_s[r, c])
    return sig


def _extract_source_patch(grid: np.ndarray, mid_r: int, mid_c: int, bg: int) -> np.ndarray:
    """Extract the TL non-background connected component containing the red anchor."""
    tl = grid[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left anchored motif into the other quadrants using rotate-first resizing.

    Input structure:
    - A full row and full column of one non-background color split the board into four quadrants.
    - Each quadrant contains one red square anchor, and the top-left quadrant also contains the full
      colored motif as a single 4-connected non-background component attached to its red anchor.
    - The other three quadrants show only their red anchors in the input.

    Rule:
    - Extract the top-left source component that contains the red anchor.
    - For each non-top-left quadrant, replace the old mirror rule with a fixed rotation:
      top-right = 90 degrees clockwise, bottom-left = 270 degrees clockwise, bottom-right = 180 degrees.
    - Apply that rotation before resizing so the transformed patch is sampled in its rotated frame,
      then scale it by target_anchor_size / source_anchor_size, align the transformed red area to the
      target anchor, and stamp non-background pixels while leaving divider cells and existing red cells untouched.

    Key insight: Quadrant location now chooses a rotation instead of a reflection, and because rotation happens
    before nearest-neighbor scaling, odd target sizes preserve different pixels than a scale-then-rotate pipeline.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red_parts = components_mask(g == RED, connectivity=4)
    assert red_parts, "grid must contain red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red_parts:
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
    _sr0, _sc0, source_size = anchors[(0, 0)]
    assert source_size > 0, "top-left red anchor must have positive size"

    patch = _extract_source_patch(g, mid_r, mid_c, bg)

    for (qb, qr), (tr0, tc0, target_size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_s = _transform_rotate_then_scale(patch, qb, qr, target_size, source_size)
        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "transformed source patch must contain red anchor pixels"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


# --------------- generator ---------------

BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
ORDER_SENSITIVE_SIZES = (1, 3)
MAX_RETRIES = 500


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it has n_fill cells."""
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
    """Assign random colors to filled cells; return None if the bbox is not tight enough."""
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


def _source_patch(_needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """
    Build an asymmetric freeform TL motif around a 2x2 red anchor.
    Freeform patches are used because the fused order change must stay visible after resize.
    """
    s = SOURCE_ANCHOR_SIZE
    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    if h == w and random.random() < 0.8:
        return None
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 3, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a random anchor position within the quadrant, preferring full motif fit."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    assert rr.size > 0, "transformed patch must contain red cells"
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp non-bg pixels from a transformed patch onto the output grid."""
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
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _build_output(
    patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
    transformed: dict[tuple[int, int], np.ndarray],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), *anchors[key])
    for key, patch_s in transformed.items():
        _stamp(out, patch_s, *anchors[key])
    return out


def _build_input(
    out: np.ndarray,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
) -> np.ndarray:
    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])
    return inp


def _quadrant_diff_stats(a: np.ndarray, b: np.ndarray) -> dict[tuple[int, int], int]:
    """Count visible non-red disagreements between two rendered outputs per non-TL quadrant."""
    stats: dict[tuple[int, int], int] = {}
    for key, (r0, r1, c0, c1) in QUADS.items():
        if key == (0, 0):
            continue
        qa = a[r0 : r1 + 1, c0 : c1 + 1]
        qb = b[r0 : r1 + 1, c0 : c1 + 1]
        visible = ((qa != RED) | (qb != RED)) & (qa != qb)
        stats[key] = int(visible.sum())
    return stats


def _choose_target_sizes() -> dict[tuple[int, int], int]:
    """Force order-sensitive sizes in TR and BL; keep BR flexible."""
    return {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.choice(ORDER_SENSITIVE_SIZES),
        (1, 0): random.choice(ORDER_SENSITIVE_SIZES),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }


def _visible_order_difference(patch: np.ndarray, qb: int, qr: int, target_size: int) -> bool:
    """Return True iff rotate-then-scale and scale-then-rotate disagree visibly."""
    fused_patch = _transform_rotate_then_scale(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    baseline_patch = _transform_scale_then_rotate(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    fused_sig = _visible_signature(fused_patch, BG, RED)
    baseline_sig = _visible_signature(baseline_patch, BG, RED)
    if fused_sig is None or baseline_sig is None:
        return False
    return fused_sig != baseline_sig


def _visible_rotation_difference(patch: np.ndarray, qb: int, qr: int, target_size: int) -> bool:
    """Return True iff the fused rule disagrees visibly with legacy reflect-then-scale."""
    fused_patch = _transform_rotate_then_scale(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    legacy_patch = _transform_reflect_then_scale(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    fused_sig = _visible_signature(fused_patch, BG, RED)
    legacy_sig = _visible_signature(legacy_patch, BG, RED)
    if fused_sig is None or legacy_sig is None:
        return False
    return fused_sig != legacy_sig


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, and oracle-check."""
    target_sizes = _choose_target_sizes()
    source = _source_patch(False)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    if not _visible_order_difference(patch, 0, 1, target_sizes[(0, 1)]):
        return None
    if not _visible_order_difference(patch, 1, 0, target_sizes[(1, 0)]):
        return None
    if not (
        _visible_rotation_difference(patch, 0, 1, target_sizes[(0, 1)])
        or _visible_rotation_difference(patch, 1, 0, target_sizes[(1, 0)])
        or _visible_rotation_difference(patch, 1, 1, target_sizes[(1, 1)])
    ):
        return None

    fused = {
        key: _transform_rotate_then_scale(patch, key[0], key[1], target_sizes[key], SOURCE_ANCHOR_SIZE)
        for key in target_sizes
        if key != (0, 0)
    }
    d5_only = {
        key: _transform_scale_then_rotate(patch, key[0], key[1], target_sizes[key], SOURCE_ANCHOR_SIZE)
        for key in target_sizes
        if key != (0, 0)
    }
    legacy = {
        key: _transform_reflect_then_scale(patch, key[0], key[1], target_sizes[key], SOURCE_ANCHOR_SIZE)
        for key in target_sizes
        if key != (0, 0)
    }
    all_transformed = [*fused.values(), *d5_only.values(), *legacy.values()]
    if any(np.count_nonzero(patch_s == RED) == 0 for patch_s in all_transformed):
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in fused.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)

    out_fused = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, fused)
    out_d5_only = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, d5_only)
    out_legacy = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, legacy)

    order_stats = _quadrant_diff_stats(out_fused, out_d5_only)
    rotation_stats = _quadrant_diff_stats(out_fused, out_legacy)
    if order_stats[(0, 1)] == 0 or order_stats[(1, 0)] == 0:
        return None
    if max(rotation_stats.values()) == 0:
        return None

    inp = _build_input(out_fused, target_sizes, anchors)
    return inp if np.array_equal(main(inp), out_fused) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo b10624e5 input")
