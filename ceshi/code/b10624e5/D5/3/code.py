# Variant of b10624e5
# Dimension: D5 — 变换：反射规则 (transform)
# Change: 反射轴由源与目标的象限相对位置决定——左右关系用水平轴反射，上下关系用垂直轴反射，对角关系用双轴反射... → 对角线反射（45度轴）

"""
design_analysis:
  variant_goal:
    keep: 27x27 four-quadrant layout, divider detection, red square anchors, TL-only source motif,
      nearest-neighbor scaling by anchor-size ratio, output-first generation.
    change: every non-TL copy uses one fixed mirror across the main diagonal (row/col swap),
      instead of choosing horizontal / vertical / double reflection from quadrant position.
  concrete_behavior_trace:
    source_patch:
      description: a single 4-connected non-bg component in TL that contains one 2x2 red anchor.
      visibility_requirement: the motif must be non-symmetric under transpose so the diagonal flip
        is actually visible after copying.
    old_vs_new_example:
      source_bbox: assume TL patch bbox is 4x6 and the red anchor sits near its left side.
      old_TR: original code applies left-right flip, so a colored cell at (0,5) moves to (0,0).
      new_TR: variant applies transpose, so that same cell moves to (5,0); height/width swap.
      difference: the copied patch is no longer determined by target quadrant. TR, BL, and BR all
        receive the same diagonal reflection before scaling; only scale and placement differ.
  complexity_calibration:
    preserved_freedom: anchor sizes still vary independently in 1..4, motifs still use 3..5 colors,
      and scaling remains lossless by the original roundtrip check.
    added_constraints:
      - reject TL motifs that equal their transpose, because diagonal reflection would be invisible.
      - require at least one large enough target copy and at least two target quadrants whose
        transpose-based result differs from the legacy H/V reflection result.
    reason: without these filters, tiny or symmetric motifs can make the D5 change visually
      indistinguishable from the base task even though the code path changed.
  implicit_assumption_audit:
    - _find_dividers still assumes exactly one full non-bg row/col pair with the same color; unchanged.
    - red-component parsing still assumes exactly one red square per quadrant; generator preserves that.
    - source extraction still assumes the only red pixels belong to the anchor inside the TL component;
      generator keeps non-anchor colors away from red.
    - alignment by scaled red bbox min corner still works after transpose because a square anchor stays
      square under transpose, only its offset inside the patch changes.
    - roundtrip scaling assert remains valid because transpose does not change nearest-neighbor
      losslessness; the generator still prefers block-built motifs when odd target sizes appear.
  placement_legality:
    - max source bbox side is 6, scale factor max is 2, so any transposed scaled patch stays within
      12x12 and can fit inside a 13x13 quadrant.
    - anchor placement prefers full-fit positions computed from the transformed patch's red bbox, so
      targets are usually unclipped and remain easy to read.
    - stamping still skips divider cells, out-of-bounds pixels, and existing red anchor pixels.
  visual_observability:
    - no invisible metadata such as color-id ordering is used.
    - diagonal reflection is communicated by shape relocation, often with swapped height/width.
    - generator enforces asymmetric source motifs and target sizes that preserve enough detail.
  cleanliness:
    - each quadrant contains only one motif plus one anchor, so there is no multi-object ambiguity.
    - the divider stays untouched and separates the four copies clearly.
  color_reasoning:
    - colors are copied unchanged; there is no new color mapping to infer.
  generator_validation:
    - final sample is accepted only if main(input) exactly reconstructs the synthesized output.
    - additional degeneration checks compare the new diagonal reflection against the legacy reflection
      on the same source patch to ensure the new D5 behavior is actually exercised.
"""

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
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _diag_reflect(img: np.ndarray) -> np.ndarray:
    """Reflect across the main diagonal (fixed 45-degree axis)."""
    return np.transpose(img).copy()


def _legacy_reflect(img: np.ndarray, qb: int, qr: int) -> np.ndarray:
    """Original quadrant-dependent reflection, kept only for degeneration checks."""
    out = img
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out.copy()


def _same_grid(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif to the other quadrants using anchor-guided scaling and a fixed diagonal flip.

    Input structure:
    - A single-color full row and full column split the 27x27 grid into four 13x13 quadrants.
    - Each quadrant contains exactly one red square anchor; the TL anchor belongs to a larger colored motif.
    - Outside TL, the input shows only the target anchors, so the missing motif must be reconstructed.

    Rule:
    - Extract the TL connected non-background component that contains the red anchor.
    - For each non-TL anchor, first reflect that source patch across the main diagonal, then scale it by
      target_anchor_size / source_anchor_size using nearest-neighbor resize.
    - Align the scaled red sub-block to the target anchor and stamp every non-background pixel, skipping
      divider cells and already-red target-anchor cells.

    Key insight: red anchor size still determines scale, but quadrant location no longer chooses the flip;
    every target uses the same 45-degree diagonal reflection before placement.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

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

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    _, _, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

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
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = _diag_reflect(patch)
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
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
            if out[rr_t, cc_t] == 2:
                continue
            out[rr_t, cc_t] = val

    return out


# --------------- generator (o5x, diagonal-reflection variant) ---------------

# --- Grid constants ---
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
MAX_RETRIES = 400


def _scaled_patch(patch: np.ndarray, target_size: int) -> np.ndarray:
    """Diagonal-reflect and scale the source patch for a target anchor size."""
    p = _diag_reflect(patch)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _legacy_scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int) -> np.ndarray:
    """Legacy quadrant-based transform, used only to reject degenerate samples."""
    p = _legacy_reflect(patch, qb, qr)
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


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif patch. Block mode is preferred for lossless odd target scales."""
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


def _sample_target_sizes() -> dict[tuple[int, int], int]:
    """Favor samples where the reflected structure remains visually legible."""
    while True:
        target_sizes = {
            (0, 0): SOURCE_ANCHOR_SIZE,
            (0, 1): random.randint(*TARGET_SIZE_RANGE),
            (1, 0): random.randint(*TARGET_SIZE_RANGE),
            (1, 1): random.randint(*TARGET_SIZE_RANGE),
        }
        non_tl = [target_sizes[(0, 1)], target_sizes[(1, 0)], target_sizes[(1, 1)]]
        if max(non_tl) >= 3 and sum(size >= 2 for size in non_tl) >= 2:
            return target_sizes


def _variant_change_count(patch: np.ndarray, target_sizes: dict[tuple[int, int], int]) -> int:
    """Count non-TL quadrants where the new transform differs from the original task."""
    changed = 0
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        new_patch = _scaled_patch(patch, size)
        old_patch = _legacy_scaled_patch(patch, key[0], key[1], size)
        if not _same_grid(new_patch, old_patch):
            changed += 1
    return changed


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick an anchor position, preferring placements where the full transformed motif fits."""
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
    """Stamp non-bg pixels from a scaled patch onto the output grid, aligned by red anchor."""
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
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    target_sizes = _sample_target_sizes()

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    patch_t = _diag_reflect(patch)
    if _same_grid(patch, patch_t):
        return None
    if _variant_change_count(patch, target_sizes) < 2:
        return None

    scaled = {
        key: _scaled_patch(patch, size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)

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
    raise RuntimeError("failed to generate a valid b10624e5 D5 input")
