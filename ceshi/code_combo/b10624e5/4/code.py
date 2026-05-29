# Combo variant of b10624e5 (index 4)
# combo_spec: D4/1+D9/0
# Name: 非均匀缩放 + 矩形锚块
#
# Design analysis
# ---------------
# spec_variant:
#   preserved_backbone:
#     - 27x27 grid, background 4, one full divider row and one full divider column of color 1 at index 13
#     - four quadrants, one red anchor per quadrant, TL is the only quadrant with a full colored motif in the input
#     - non-TL quadrants in the input contain only the red anchor; the output restores the reflected/scaled motifs
#   fused_change:
#     - D9/0 changes every red anchor from a square to a non-square filled rectangle
#     - D4/1 then uses that rectangle directly: vertical scale = target_anchor_height / source_anchor_height,
#       horizontal scale = target_anchor_width / source_anchor_width
#     - the rectangle's aspect ratio is therefore not decorative; it is the visible carrier of the two scale factors
#
# a) Concrete behavior trace:
#   Suppose the TL anchor is 2x3 and the TL motif bbox is 6x9 because it comes from a 3x3 logical motif whose
#   logical cells are expanded by 2 rows and 3 cols each. If TR anchor is 2x4, the old solver's single `size`
#   logic would collapse the anchor to one number and scale both axes together. The fused rule instead keeps
#   height ratio = 2/2 = 1 and width ratio = 4/3, so TR becomes a horizontally stretched mirror. If BL anchor
#   is 4x3, only the vertical ratio changes, producing a vertically stretched mirror. If BR anchor is 1x4, both
#   ratios differ and the motif is simultaneously compressed vertically and stretched horizontally. Those three
#   outcomes are visibly distinct, which is the intended fusion effect.
#
# b) Complexity calibration:
#   The generator restricts source anchors to a small non-square family: 2x3 or 3x2. That keeps the visual rule
#   stable across samples and leaves enough room in a 13x13 quadrant for the motif. Each sample forces exactly one
#   target that changes only height, one that changes only width, and one that changes both. This avoids degenerate
#   samples where all targets accidentally look like one scalar resize or where the fusion becomes too subtle.
#
# c) Implicit assumption audit:
#   - Original `size = max(height, width)` is invalid because it discards one scaling degree of freedom.
#   - Original `ratio = target_size / source_size` must split into `target_h / source_h` and `target_w / source_w`.
#   - Original square-anchor placement and reconstruction no longer apply; every path must carry `(anchor_h, anchor_w)`.
#   - Alignment by red bbox top-left still works, but only if each red component is a solid filled rectangle and the
#     scaled red bbox exactly matches the target rectangle. The generator guarantees both.
#   - Roundtrip-resize validity becomes stricter under anisotropic scaling, so the source motif is built from a logical
#     grid whose cells are expanded by the TL anchor shape. That makes all target resizes pixel-exact.
#
# d) Placement legality:
#   The generator places each target anchor by top-left coordinate. It prefers positions where the entire transformed
#   motif fits inside the quadrant, but it can fall back to anchor-only fit while still respecting quadrant bounds.
#   Divider cells are never overwritten, red anchor cells are never overwritten by stamping, and outer-edge clipping is
#   allowed exactly like the base task.
#
# e) Visual observability:
#   The source motif is rejected unless there are non-red pixels sharing the anchor's logical row and logical column,
#   so one-axis stretching is visible rather than hidden in an isolated corner. The motif is also rejected if it is
#   both left-right and up-down symmetric, because that would make quadrant reflections hard to infer from 2-4 samples.
#
# f) Visual cleanliness:
#   There is still only one connected motif per quadrant, anchored by one solid red rectangle. Non-TL inputs remain
#   sparse, so the copied structure does not get confused with unrelated clutter. The divider remains a single clean
#   separator and no extra color rules are introduced.
#
# g) Fusion anti-degeneration policy:
#   Samples are only accepted if the three non-TL targets realize three distinct scale signatures:
#   vertical-only, horizontal-only, and fully anisotropic. This makes the combined behavior observable and ensures the
#   dataset does not silently collapse back to the original square/uniform-scaling task.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    """Nearest-neighbor resize for independent row/column targets."""
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resized patch must stay positive"
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _red_bbox_dims(img: np.ndarray) -> tuple[int, int, int, int]:
    """Return the red bbox as (r0, c0, h, w)."""
    rr, cc = np.where(img == RED)
    assert rr.size > 0, "patch must contain red pixels"
    r0, c0 = int(rr.min()), int(cc.min())
    h = int(rr.max() - rr.min() + 1)
    w = int(cc.max() - cc.min() + 1)
    return r0, c0, h, w


def _is_anisotropic(source_shape: tuple[int, int], target_shape: tuple[int, int]) -> bool:
    """Whether the resize factors differ between axes."""
    sh, sw = source_shape
    th, tw = target_shape
    return th * sw != tw * sh


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the top-left motif beside the other anchors, with each red rectangle encoding two separate resize factors.

    Input structure:
    - A single non-background divider row and divider column split the 27x27 grid into four 13x13 quadrants.
    - Each quadrant contains exactly one filled red rectangle, and all four anchors are non-square.
    - The top-left quadrant also contains one connected colored motif attached to its red rectangle; the other quadrants show only their red anchors.

    Rule:
    - Extract the top-left connected component containing the red rectangle as the source patch.
    - Reflect that patch left-right for right-side quadrants and up-down for bottom quadrants.
    - Resize rows by target_anchor_height / source_anchor_height and columns by target_anchor_width / source_anchor_width, then align the scaled red bbox to the target anchor and stamp non-background cells.

    Key insight:
    - The quadrant still determines the mirror direction, but the red anchor is no longer a one-number scale cue: its height and width separately encode the vertical and horizontal resize amounts.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        ah = int(r1 - r0 + 1)
        aw = int(c1 - c0 + 1)
        assert ah != aw, "anchors must be non-square rectangles"
        assert len(pts) == ah * aw, "each anchor must be a filled rectangle"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), ah, aw)

    assert set(anchors) == {(0, 0), (0, 1), (1, 0), (1, 1)}, "need one red anchor per quadrant"
    _, _, source_h, source_w = anchors[(0, 0)]

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with red pixels"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, target_h, target_w) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        new_h = max(1, int(round(patch_t.shape[0] * target_h / source_h)))
        new_w = max(1, int(round(patch_t.shape[1] * target_w / source_w)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "anisotropic scaling must preserve the motif exactly"

        red_r0, red_c0, red_h, red_w = _red_bbox_dims(patch_s)
        assert red_h == target_h and red_w == target_w, "scaled red bbox must match the target anchor rectangle"
        place_r0 = tr0 - red_r0
        place_c0 = tc0 - red_c0

        src_rr, src_cc = np.where(patch_s != bg)
        for rr0, cc0 in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
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


def _logical_patch_is_observable(logical: np.ndarray, ar: int, ac: int) -> bool:
    """Reject motifs where one-axis stretching or reflection would be too subtle."""
    non_red = logical != BG
    same_row = any(non_red[ar, c] and logical[ar, c] != RED for c in range(logical.shape[1]))
    same_col = any(non_red[r, ac] and logical[r, ac] != RED for r in range(logical.shape[0]))
    asym_h = not np.array_equal(logical, np.fliplr(logical))
    asym_v = not np.array_equal(logical, np.flipud(logical))
    return same_row and same_col and (asym_h or asym_v)


def _source_patch(source_shape: tuple[int, int]) -> tuple[np.ndarray, int, int] | None:
    """
    Build the TL motif on a logical grid, then expand each logical cell by the
    TL anchor rectangle dimensions so anisotropic resizing remains lossless.
    """
    ah, aw = source_shape
    lh, lw = random.choice(BLOCK_DIMS)
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


def _choose_target_shapes(source_shape: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int]] | None:
    """Force one vertical-only, one horizontal-only, and one fully anisotropic target."""
    sh, sw = source_shape
    vertical_only = [shape for shape in TARGET_RECT_CHOICES if shape[1] == sw and shape[0] != sh]
    horizontal_only = [shape for shape in TARGET_RECT_CHOICES if shape[0] == sh and shape[1] != sw]
    full_aniso = [
        shape
        for shape in TARGET_RECT_CHOICES
        if shape[0] != sh and shape[1] != sw and _is_anisotropic(source_shape, shape)
    ]
    if not vertical_only or not horizontal_only or not full_aniso:
        return None
    targets = [random.choice(vertical_only), random.choice(horizontal_only), random.choice(full_aniso)]
    random.shuffle(targets)
    return {(0, 0): source_shape, (0, 1): targets[0], (1, 0): targets[1], (1, 1): targets[2]}


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
    out_h = max(1, int(round(p.shape[0] * th / sh)))
    out_w = max(1, int(round(p.shape[1] * tw / sw)))
    return _resize_nn(p, out_h, out_w)


def _choose_anchor(
    qb: int,
    qr: int,
    anchor_shape: tuple[int, int],
    patch_s: np.ndarray,
) -> tuple[int, int]:
    """Pick a target anchor top-left, preferring placements where the whole motif fits."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    ah, aw = anchor_shape
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)

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
    """Stamp a transformed patch onto the output, aligned by the red rectangle."""
    red_r0, red_c0, _, _ = _red_bbox_dims(patch_s)
    place_r0 = anchor_r - red_r0
    place_c0 = anchor_c - red_c0

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

    for key, patch_s in scaled.items():
        _, _, red_h, red_w = _red_bbox_dims(patch_s)
        if (red_h, red_w) != target_shapes[key]:
            return None

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
