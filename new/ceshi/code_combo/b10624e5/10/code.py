# Combo variant of b10624e5 (index 10)
# combo_spec: D6/0+D7/1
# Name: 仅输出目标象限 + 擦除源图案

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant:
# - The original task has one source quadrant (TL) and three destination quadrants.
#   A literal "keep only the target quadrants" crop would become an L-shape, which is
#   not a valid ARC output. To keep D7/1 as a true crop without introducing a new
#   selector rule, this variant always keeps the diagonal destination quadrant: BR.
# - D6/0 is interpreted as move semantics on the full scene: after transferring the
#   TL motif to the target quadrants, clear the TL source component except for its red
#   anchor square. The final output then crops to BR, so the source vanishes both
#   because it was erased and because the source quadrant is outside the crop.
# - Concrete behavior trace:
#   1. Original task: TL motif stays, and mirrored/scaled copies appear in TR/BL/BR.
#   2. Fused task: construct those target copies the same way, erase TL non-red source
#      pixels, then return only the BR quadrant as a 13x13 canvas.
#   3. Visible difference: output shrinks from 27x27 to 13x13, divider disappears, and
#      only one reflected/scaled target result remains.
# - Complexity calibration:
#   BR is fixed across all samples, so the crop is inferable from 2-4 examples instead
#   of depending on a hidden per-instance selector. TR/BL anchors remain in the input
#   to preserve the original multi-target structure, but generator placement keeps the
#   transformed motifs fully inside each quadrant so the scene stays legible.
# - Main-side assumptions audited and matched by the generator:
#   1. Exactly one full-row/full-col divider of one non-bg color exists.
#   2. Each quadrant has exactly one red square anchor.
#   3. TL contains exactly one non-bg component with the TL red anchor.
#   4. The TL component has non-red pixels, so D6/0 is not a no-op.
#   5. The BR transformed motif fits inside BR and therefore yields a clear 13x13 crop.


# --- Grid constants ---
BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
MID = 13
TARGET_KEY = (1, 1)
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# --- Source motif parameters ---
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

# --- Target anchor parameters ---
TARGET_SIZE_RANGE = (1, 4)

# --- Generation retry limit ---
MAX_RETRIES = 300


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique matching full-row/full-col divider."""
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


def _extract_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Return per-quadrant red anchor info as {(qb, qr): (r0, c0, size)}."""
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))
    assert set(anchors) == set(QUADS), "each quadrant must contain one red anchor block"
    return anchors


def _extract_source_patch(tl: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Return the TL connected component containing the red anchor and its local bbox."""
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(int(tl[r, c]) == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    assert np.any((patch != bg) & (patch != RED)), "source motif must contain non-red pixels"
    return patch, patch_bb


def _transform_patch(patch: np.ndarray, qb: int, qr: int, source_size: int, target_size: int) -> np.ndarray:
    """Reflect and scale the source patch for a destination quadrant."""
    patch_t = patch
    if qr == 1:
        patch_t = np.fliplr(patch_t)
    if qb == 1:
        patch_t = np.flipud(patch_t)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(patch_t.shape[0] * ratio)))
    new_w = max(1, int(round(patch_t.shape[1] * ratio)))
    patch_s = _resize_nn(patch_t, new_h, new_w)
    roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
    assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"
    return patch_s


def _stamp_patch(dst: np.ndarray, patch_s: np.ndarray, anchor_r0: int, anchor_c0: int, bg: int) -> None:
    """Stamp a scaled patch onto dst, aligning its red bbox to the destination anchor."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
    red_r0, red_c0 = int(rr_red.min()), int(cc_red.min())
    place_r0 = int(anchor_r0) - red_r0
    place_c0 = int(anchor_c0) - red_c0
    rr_p, cc_p = np.where(patch_s != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < dst.shape[0] and 0 <= cc_t < dst.shape[1]):
            continue
        if dst[rr_t, cc_t] == RED:
            continue
        dst[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _erase_source_nonred(out: np.ndarray, patch: np.ndarray, patch_bb: tuple[int, int, int, int], bg: int) -> None:
    """Clear the TL source motif while preserving the red anchor block."""
    pr0, pc0, pr1, pc1 = patch_bb
    window = out[pr0 : pr1 + 1, pc0 : pc1 + 1]
    mask = (patch != bg) & (patch != RED)
    window[mask] = bg


def _solve_full(grid: np.ndarray, *, erase_source: bool) -> np.ndarray:
    """Construct the full 27x27 moved/cropped precursor before final cropping."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    assert (mid_r, mid_c) == (MID, MID), "this variant expects the canonical 27x27 quadrant layout"

    anchors = _extract_anchors(g, mid_r, mid_c)
    _sr0, _sc0, source_size = anchors[(0, 0)]
    tl = g[0:mid_r, 0:mid_c]
    patch, patch_bb = _extract_source_patch(tl, bg)

    for (qb, qr), (tr0, tc0, target_size) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_s = _transform_patch(patch, qb, qr, source_size, target_size)
        _stamp_patch(out, patch_s, tr0, tc0, bg)

    if erase_source:
        _erase_source_nonred(out, patch, patch_bb, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the transferred motif scene, erase the source drawing, then keep only BR.

    Input structure:
    - A 27x27 grid is split by one full divider row and one full divider column into
      four 13x13 quadrants.
    - Each quadrant has one red square anchor block, and only the TL quadrant also
      contains the colored source motif attached to its red anchor.

    Rule:
    - Use the TL motif as the template, and create mirrored/scaled versions at the
      other three red anchors exactly as in the original task.
    - Treat the source as moved rather than preserved: once the transfers are built,
      delete the TL motif pixels other than the TL red anchor.
    - Return only the bottom-right target quadrant, without the divider or any other
      quadrant content.

    Key insight: Anchor size still determines scale and quadrant position still
    determines reflections, but the observable answer is the cropped BR destination
    after the TL source has conceptually disappeared.
    """
    full = _solve_full(grid, erase_source=True)
    return full[MID + 1 :, MID + 1 :].copy()


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    return _transform_patch(patch, qb, qr, SOURCE_ANCHOR_SIZE, target_size)


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
    filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int
) -> np.ndarray | None:
    """Assign random colors to a filled region; return None if bbox coverage is incomplete."""
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
    """Build the TL source patch and return it with the anchor's top-left offset."""
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


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int] | None:
    """Pick an anchor position whose transformed motif fully fits within its quadrant."""
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
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _stamp_full(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp onto the full grid, skipping divider cells and existing red anchors."""
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())
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


def _build_full_output(
    patch: np.ndarray,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
    tl_r0: int,
    tl_c0: int,
) -> np.ndarray:
    """Build the full original-style target scene before BR cropping."""
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        patch_s = _scaled_patch(patch, key[0], key[1], size)
        _stamp_full(out, patch_s, *anchors[key])
    return out


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build a valid input whose output is the BR crop."""
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }
    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if not np.any((patch != BG) & (patch != RED)):
        return None

    scaled = {key: _scaled_patch(patch, key[0], key[1], size) for key, size in target_sizes.items() if key != (0, 0)}

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        chosen = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)
        if chosen is None:
            return None
        anchors[key] = chosen

    full_out = _build_full_output(patch, target_sizes, anchors, tl_r0, tl_c0)
    expected = full_out[MID + 1 :, MID + 1 :].copy()
    if np.count_nonzero(expected != BG) <= target_sizes[TARGET_KEY] ** 2:
        return None

    inp = full_out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
