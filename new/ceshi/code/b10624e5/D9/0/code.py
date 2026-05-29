# Variant of b10624e5
# Dimension: D9 — 锚块形状与结构 (meta)
# Change: 每个象限有一个红色正方形色块作为锚块，不同象限锚块尺寸不同（尺寸比决定缩放因子）... → 锚块为矩形（非正方形）

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D9 variant
# ----------------------------------
# spec_variant:
#   grid:
#     same as source task: 27x27, divider row/col at index 13, four 13x13 quadrants
#   anchors:
#     exactly 4 red anchors, one per quadrant, still the only red component in each quadrant
#     changed from filled squares to filled rectangles with height != width
#     top-left anchor is the source scale unit; its height and width are both needed
#   motif:
#     still a single 4-connected non-bg component in TL containing the red anchor
#     still copied into the other quadrants with LR/UD reflections by quadrant
#     but scaling is now anisotropic: rows use target_h / source_h, cols use target_w / source_w
#
# a) Concrete behavior trace:
#   Example source anchor = 2x3 and a target anchor = 4x3.
#   Old task: the code reduced the anchor to one number `size`, so 2x3 and 4x3 would both look
#   like size 3 or 4 depending on the reduction, forcing one uniform ratio for both axes.
#   New task: height ratio = 4/2 = 2 while width ratio = 3/3 = 1, so the copied motif stretches
#   only vertically. A second target such as 2x4 stretches only horizontally. This is the visible
#   place where the variant differs from the original single-ratio pipeline.
#
# b) Complexity calibration:
#   The generator deliberately uses three target categories in each sample:
#   one target keeps width but changes height, one keeps height but changes width, and one changes both.
#   That avoids degenerate samples where every rectangle is accidentally proportional to the source and
#   the task would visually collapse back to a square-style single scale factor.
#
# c) Old implicit assumptions that break under D9:
#   - `size = max(height, width)` no longer preserves the anchor information.
#   - `ratio = target_size / source_size` no longer works because row and col scales can differ.
#   - Anchor placement ranges based on one scalar extent are no longer sufficient.
#   - Pasting a square red block when reconstructing the input/output would corrupt the new variant.
#   Every one of those paths is replaced by `(anchor_h, anchor_w)` handling.
#
# d) Placement legality:
#   The generator chooses anchor top-lefts, not centers, and checks fit ranges with the scaled patch's
#   red bbox plus the target rectangle size. Full-fit positions are preferred; anchor-only fit is the fallback.
#   Divider cells are never overwritten, and clipping is still allowed at the outer grid edge.
#
# e) Visual observability:
#   The source motif is built from logical cells repeated by the TL anchor rectangle dimensions.
#   This makes anisotropic nearest-neighbor scaling exact and visible. The motif is rejected if it is
#   horizontally or vertically symmetric, so the quadrant reflections remain readable to a human.
#
# f) Visual cleanliness:
#   Non-TL quadrants in the input still contain only the red anchor rectangles. The copied motifs appear
#   only in the output, with one connected component per quadrant, so object roles stay visually separated.
#
# g) Color-rule audit:
#   No invisible color-id ordering is used. Colors remain purely decorative labels on the copied motif.
#
# h) Generator anti-regression checks:
#   Every candidate grid is validated with `main(input) == output`.
#   The generator also forces all anchors to be non-square and ensures at least one vertical-only,
#   one horizontal-only, and one fully anisotropic target in every sample.
#
# i) New solver assumptions introduced by this variant:
#   - Each red component is a solid filled rectangle.
#   - The TL motif contains exactly one red rectangle and no extra disconnected red pixels.
#   - The source motif is constructed so anisotropic resize round-trips exactly.
#   The generator guarantees these conditions.


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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize for arbitrary row/column scale ratios."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recreate the top-left motif beside the other anchors using per-axis scaling and quadrant reflections.

    Input structure:
    - The board is split into four quadrants by one full divider row and one full divider column.
    - Each quadrant contains exactly one filled red rectangle, and every red anchor is non-square.
    - The top-left quadrant also contains one colored component attached to its red anchor rectangle.

    Rule:
    - Extract the top-left component containing the red rectangle as the source patch.
    - For each other quadrant, reflect that patch horizontally on the right side and vertically on the bottom side.
    - Scale rows by target_anchor_height / source_anchor_height and columns by target_anchor_width / source_anchor_width, then align the scaled red rectangle to the target anchor and stamp non-background cells.

    Key insight:
    - The anchor no longer gives one scalar size. Its height and width separately encode the vertical and horizontal resize factors, while the quadrant still decides which reflections to apply.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == 2, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        ah = int(r1 - r0 + 1)
        aw = int(c1 - c0 + 1)
        assert ah != aw, "anchors must be non-square rectangles in this variant"
        assert len(pts) == ah * aw, "red anchor must be a filled rectangle"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), ah, aw)

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one red anchor per quadrant"
    _sr0, _sc0, src_h, src_w = anchors[(0, 0)]

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == 2 for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0:pr1 + 1, pc0:pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, tgt_h, tgt_w) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        new_h = max(1, int(round(patch_t.shape[0] * tgt_h / src_h)))
        new_w = max(1, int(round(patch_t.shape[1] * tgt_w / src_w)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "anisotropic scaling must preserve motif information"

        rr_red, cc_red = np.where(patch_s == 2)
        assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        ar1, ac1 = int(rr_red.max()), int(cc_red.max())
        assert ar1 - ar0 + 1 == tgt_h, "scaled red bbox height must match target anchor height"
        assert ac1 - ac0 + 1 == tgt_w, "scaled red bbox width must match target anchor width"
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

SOURCE_ANCHOR_SHAPES = [(2, 3), (3, 2)]
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_RECT_CHOICES = [
    (1, 2), (1, 3), (1, 4),
    (2, 1), (2, 3), (2, 4),
    (3, 1), (3, 2), (3, 4),
    (4, 1), (4, 2), (4, 3),
]
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
    """Assign colors to a logical motif while keeping full bbox coverage."""
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


def _is_asymmetric(patch: np.ndarray) -> bool:
    """Keep the source motif visually informative under quadrant reflections."""
    return (not np.array_equal(patch, np.fliplr(patch))) and (not np.array_equal(patch, np.flipud(patch)))


def _source_patch(source_shape: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """
    Build the TL motif on a logical grid, then expand each logical cell by the
    source anchor rectangle dimensions so anisotropic scaling round-trips exactly.
    """
    ah, aw = source_shape
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    logical = _assign_colors(filled, {(ar, ac)}, lh, lw)
    if logical is None:
        return None
    patch = np.repeat(np.repeat(logical, ah, axis=0), aw, axis=1)
    if not _is_asymmetric(patch):
        return None
    return patch, ar * ah, ac * aw


def _is_anisotropic(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> bool:
    """Whether the target requires different row/col scale factors."""
    sh, sw = source_shape
    th, tw = target_shape
    return th * sw != tw * sh


def _choose_target_shapes(source_shape: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Pick one vertical-only, one horizontal-only, and one fully anisotropic target."""
    sh, sw = source_shape
    vertical_only = [rc for rc in TARGET_RECT_CHOICES if rc[1] == sw and rc[0] != sh]
    horizontal_only = [rc for rc in TARGET_RECT_CHOICES if rc[0] == sh and rc[1] != sw]
    full_aniso = [
        rc for rc in TARGET_RECT_CHOICES
        if rc[0] != sh and rc[1] != sw and _is_anisotropic(source_shape, rc)
    ]
    if not vertical_only or not horizontal_only or not full_aniso:
        return None
    targets = [random.choice(vertical_only), random.choice(horizontal_only), random.choice(full_aniso)]
    random.shuffle(targets)
    keys = [(0, 1), (1, 0), (1, 1)]
    return {(0, 0): source_shape, **{k: v for k, v in zip(keys, targets, strict=False)}}


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
) -> np.ndarray:
    """Reflect and anisotropically scale the source patch for one quadrant."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    sh, sw = source_shape
    th, tw = target_shape
    h = max(1, int(round(p.shape[0] * th / sh)))
    w = max(1, int(round(p.shape[1] * tw / sw)))
    return _resize_nn(p, h, w)


def _choose_anchor(
    qb: int,
    qr: int,
    anchor_shape: tuple[int, int],
    patch_s: np.ndarray,
) -> tuple[int, int]:
    """Pick a target anchor top-left, preferring placements where the full motif fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    ah, aw = anchor_shape
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - ah + 1
    anch_c0, anch_c1 = q_c0, q_c1 - aw + 1
    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp a scaled patch onto the output, aligned by the red rectangle."""
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0) & (tgt_rr < GRID_SIZE) &
        (tgt_cc >= 0) & (tgt_cc < GRID_SIZE) &
        (tgt_rr != MID) & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    """Build output first, then erase non-TL motifs to form a valid input."""
    source_shape = random.choice(SOURCE_ANCHOR_SHAPES)
    target_shapes = _choose_target_shapes(source_shape)
    if target_shapes is None:
        return None

    source = _source_patch(source_shape)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], source_shape, target_shapes[key])
        for key in target_shapes
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_shapes[key], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, shape in target_shapes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full(shape, RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, shape in target_shapes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0:q_r1 + 1, q_c0:q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full(shape, RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D9 input")
