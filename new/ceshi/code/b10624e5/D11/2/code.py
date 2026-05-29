# Variant of b10624e5
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 四步线性流程：识别锚块 -> 计算缩放因子 -> 确定反射轴 -> 复制缩放反射后的图案... → 先缩放后反射 vs 先反射后缩放
#
# Design analysis
# a) Concrete behavior trace:
#    - The grid is still a 27x27 board split into four 13x13 quadrants by one full
#      divider row and one full divider column.
#    - Every quadrant still has one red square anchor. The TL quadrant still contains
#      the full colored motif attached to its 2x2 red anchor, while the other three
#      quadrants only show their red anchors in the input.
#    - The variant makes the pixel pipeline explicit: mirror first, then resize.
#      With nearest-neighbor sampling this differs from resize first, then mirror
#      whenever the target anchor size is 1 or 3 (ratio 1/2 or 3/2) and the motif
#      is asymmetric. Small example: a 2x2 row [A,B] mirrored first becomes [B,A],
#      and downscaling to width 1 samples B; downscaling first samples A, and
#      mirroring afterwards still leaves A. The visible copied color changes.
# b) Complexity calibration:
#    - The task still has one motif source, four anchors, and fixed quadrant-based
#      mirror rules, so the main reasoning structure stays close to the original.
#    - To make the D11 change observable without adding unrelated ambiguity, the
#      generator forces order-sensitive target sizes in TR and BL and rejects source
#      motifs where the two operation orders collapse to the same visible result.
# c) Hidden assumptions audited in main():
#    - Divider detection still assumes exactly one matching full row/column pair.
#    - Red components are still assumed to be one square anchor per quadrant.
#    - The TL motif is still assumed to be the unique non-background 4-connected
#      component in TL that contains red pixels.
#    - Aligning by the minimum red row/col still assumes the transformed red region
#      stays contiguous and non-empty after mirror+resize; the generator keeps the
#      TL anchor as a solid 2x2 block and only uses valid target sizes 1..4.
#    - The original roundtrip scaling assertion cannot be kept unchanged: the old
#      lossless-block construction makes mirror-then-resize and resize-then-mirror
#      collapse to the same visible result. The variant therefore drops that exact
#      reversibility check and allows asymmetric freeform motifs.
# d) Placement legality:
#    - The generator computes each transformed patch before choosing an anchor
#      position, then prefers anchor placements where the whole transformed motif
#      fits inside its quadrant. This prevents overlap with the divider and avoids
#      invisible full clipping.
# e) Visual observability:
#    - Order sensitivity is checked on the visible non-red stamped pixels after
#      anchor alignment, not merely on raw transformed arrays. This avoids false
#      positives where only hidden red pixels differ.
#    - TR is guaranteed to show a horizontal order-sensitive case and BL a vertical
#      one, so a solver can infer that quadrant orientation still controls mirroring
#      while the sampling order has changed.
# f) Visual cleanliness:
#    - Each non-TL quadrant still contains only one anchored copy of the motif.
#      No extra markers or competing overlays are introduced.
# g) Color inference:
#    - No color ids are used for ordering or control flow beyond the visible fact
#      that red is the anchor color and the divider is uniform. Extra motif colors
#      remain arbitrary but structurally tied to the copied shape.
# h) New main() assumptions rechecked against generate_input():
#    - TR and BL must use odd target sizes so the order difference is actually
#      exercised; generate_input() enforces that.
#    - The motif must be order-sensitive in both LR and UD directions after anchor
#      alignment; generate_input() rejects all motifs that fail those checks.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Mirror a patch according to quadrant position."""
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _resize_to_anchor(patch: np.ndarray, target_size: int, source_size: int) -> np.ndarray:
    """Resize a patch so its red anchor scales from source_size to target_size."""
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(patch.shape[0] * ratio)))
    new_w = max(1, int(round(patch.shape[1] * ratio)))
    return _resize_nn(patch, new_h, new_w)


def _transform_reflect_then_scale(patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int) -> np.ndarray:
    """Variant rule: mirror first, then resize."""
    return _resize_to_anchor(_reflect_patch(patch, qb, qr), target_size, source_size)


def _transform_scale_then_reflect(patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int) -> np.ndarray:
    """Counterfactual old-order baseline used only for generator rejection checks."""
    return _reflect_patch(_resize_to_anchor(patch, target_size, source_size), qb, qr)


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left anchored motif into the other quadrants using mirror-first resizing.

    Input structure:
    - A full row and full column of one non-background color split the board into four quadrants.
    - Each quadrant contains one red square anchor, and the top-left quadrant additionally contains
      the full colored motif attached to that anchor.
    - The other three quadrants show only their red anchors in the input.

    Rule:
    - Extract the top-left connected motif that contains the red anchor.
    - For each non-top-left quadrant, choose horizontal and/or vertical mirroring from its quadrant position.
    - Apply that mirroring to the source motif first, then resize the mirrored motif so the red block matches
      the target anchor size, and finally align the transformed red area onto the target anchor.
    - Stamp every non-background pixel except divider cells and existing red anchor cells.

    Key insight: Quadrant location determines which mirror to apply, anchor size determines the resize ratio,
    and the order matters because nearest-neighbor sampling after mirroring need not match mirroring after sampling.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red_parts = components_mask(g == 2, connectivity=4)
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

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == 2 for r, c in pts):
            patch_bb = bbox_coords(pts)
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, target_size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_m = _reflect_patch(patch, qb, qr)
        patch_s = _resize_to_anchor(patch_m, target_size, source_size)

        rr_red, cc_red = np.where(patch_s == 2)
        assert rr_red.size > 0, "transformed source patch must contain red anchor pixels"
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
            if out[rr_t, cc_t] == 2:
                continue
            out[rr_t, cc_t] = val

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
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
ORDER_SENSITIVE_SIZES = (1, 3)
MAX_RETRIES = 400


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
    """Assign random colors to filled cells; returns None if coverage is incomplete."""
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
    """Build the TL motif patch as a freeform asymmetric component around a 2x2 red anchor."""
    S = SOURCE_ANCHOR_SIZE
    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - S), random.randint(0, w - S)
    red_cells = {(ar + dr, ac + dc) for dr in range(S) for dc in range(S)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a random anchor position for the red block within the quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
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


def _visible_order_difference(patch: np.ndarray, qb: int, qr: int, target_size: int) -> bool:
    """Return True iff the two operation orders yield different visible stamped motifs."""
    new_patch = _transform_reflect_then_scale(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    old_patch = _transform_scale_then_reflect(patch, qb, qr, target_size, SOURCE_ANCHOR_SIZE)
    new_sig = _visible_signature(new_patch, BG, RED)
    old_sig = _visible_signature(old_patch, BG, RED)
    if new_sig is None or old_sig is None:
        return False
    return new_sig != old_sig


def _choose_target_sizes() -> dict[tuple[int, int], int]:
    """Force visible order-sensitive cases in TR and BL; keep BR flexible."""
    return {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.choice(ORDER_SENSITIVE_SIZES),
        (1, 0): random.choice(ORDER_SENSITIVE_SIZES),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    target_sizes = _choose_target_sizes()
    needs_block = any(target_sizes[key] % SOURCE_ANCHOR_SIZE != 0 for key in target_sizes if key != (0, 0))

    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    if not _visible_order_difference(patch, 0, 1, target_sizes[(0, 1)]):
        return None
    if not _visible_order_difference(patch, 1, 0, target_sizes[(1, 0)]):
        return None

    transformed = {
        key: _transform_reflect_then_scale(patch, key[0], key[1], target_sizes[key], SOURCE_ANCHOR_SIZE)
        for key in target_sizes
        if key != (0, 0)
    }
    if any(np.count_nonzero(patch_s == RED) == 0 for patch_s in transformed.values()):
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in transformed.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)

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

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        paste(inp, np.full((size, size), RED, dtype=int), *anchors[key])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D11 b10624e5 input")
