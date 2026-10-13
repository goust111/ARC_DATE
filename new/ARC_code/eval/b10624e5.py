import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste
import random


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across quadrants with scaling and reflection keyed to red anchor blocks.

    Input structure:
    - Grid divided into four quadrants by a full-row and full-column divider of a single non-bg color.
    - Each quadrant contains a red (2) square anchor block of varying size.
    - The top-left quadrant also has a colored motif attached to its red anchor.

    Rule:
    - Use the top-left quadrant's motif as the source pattern.
    - For each other quadrant, scale the motif by the ratio of that quadrant's red block size to the source red block size.
    - Reflect horizontally for right-side quadrants, vertically for bottom quadrants (both for bottom-right).
    - Align the scaled/reflected motif so its red region overlaps the target anchor, then stamp non-bg pixels.

    Key insight: Red block size encodes the scale factor; quadrant position encodes the reflection axes.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    # Find red anchor blocks (color 2) in each quadrant.
    red = components_mask(g == 2, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    # Quadrant key: (top/bottom, left/right)
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    # value: (r0,c0,size)
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
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

    # Build a source patch around the TL motif: take the non-bg connected component (4-neigh)
    # in the TL quadrant that contains the TL red anchor.
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        # component contains any red pixel?
        if any(tl[r, c] == 2 for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    def resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
        src_h, src_w = img.shape
        if new_h <= 0 or new_w <= 0:
            return img.copy()
        rr = (np.floor(np.arange(new_h) * src_h / new_h)).astype(int)
        cc = (np.floor(np.arange(new_w) * src_w / new_w)).astype(int)
        rr = np.clip(rr, 0, src_h - 1)
        cc = np.clip(cc, 0, src_w - 1)
        return img[rr[:, None], cc[None, :]]

    # For each quadrant anchor, paste a scaled/reflected version of the patch.
    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = int(round(patch_t.shape[0] * ratio))
        new_w = int(round(patch_t.shape[1] * ratio))
        new_h = max(1, new_h)
        new_w = max(1, new_w)
        patch_s = resize_nn(patch_t, new_h, new_w)
        roundtrip = resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        # Align scaled red bbox in patch_s to the target red bbox.
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
            if out[rr_t, cc_t] == 2:
                continue
            out[rr_t, cc_t] = val

    return out

# --------------- generator (o5x) ---------------

# --- Grid constants ---
BG = 4                  # background color
DIV = 1                 # divider color
RED = 2                 # anchor color
GRID_SIZE = 27          # full grid dimension
MID = 13                # divider row/col index
QUADS = {               # (row_lo, row_hi, col_lo, col_hi) inclusive per quadrant
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# --- Source motif parameters ---
SOURCE_ANCHOR_SIZE = 2                  # TL anchor is always 2x2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]  # logical block sizes for block method
BLOCK_PROB = 0.5                        # prob of block method when not forced
FILL_RATIO_MIN = 0.5                    # minimum motif fill ratio
FILL_RATIO_MAX = 0.8                    # maximum motif fill ratio
FREEFORM_H_RANGE = (3, 6)              # freeform motif height range
FREEFORM_W_RANGE = (3, 6)              # freeform motif width range
MOTIF_COLORS_MIN = 3                    # min distinct non-bg colors (including red)
MOTIF_COLORS_MAX = 5                    # max distinct non-bg colors
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]  # non-red, non-bg candidate colors

# --- Target anchor parameters ---
TARGET_SIZE_RANGE = (1, 4)              # anchor size range for non-TL quadrants

# --- Generation retry limit ---
MAX_RETRIES = 300


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Reflect and scale the source patch for a given quadrant and target anchor size."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


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


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]],
                   h: int, w: int) -> np.ndarray | None:
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


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif patch. Block method upscales a logical grid by SOURCE_ANCHOR_SIZE;
    freeform method grows directly at pixel resolution with a 2x2 red anchor."""
    S = SOURCE_ANCHOR_SIZE
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
        patch = np.repeat(np.repeat(logical, S, axis=0), S, axis=1)
        return patch, ar * S, ac * S
    else:
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
    """Pick a random anchor position for the red block within the quadrant,
    preferring positions where the full scaled motif fits inside the quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    # Range where the full motif fits inside the quadrant
    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    # Range where the anchor block itself fits inside the quadrant
    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1

    # Intersect: prefer full-fit, fall back to anchor-only fit
    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp non-bg pixels from a scaled patch onto the output grid, aligned by red anchor.
    Skips divider row/col, out-of-bounds pixels, and existing red pixels."""
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    # Source pixel coordinates and their target positions
    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    # Mask: in-bounds, not on divider, target is not already red
    valid = ((tgt_rr >= 0) & (tgt_rr < GRID_SIZE) &
             (tgt_cc >= 0) & (tgt_cc < GRID_SIZE) &
             (tgt_rr != MID) & (tgt_cc != MID))
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
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

    # Place anchors
    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)

    # Build output grid
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    # Build input: clear non-TL quadrants, keep only red anchors
    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 input")
