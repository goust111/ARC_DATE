# Variant of b10624e5
# Dimension: D5 — 变换：反射规则 (transform)
# Change: 反射轴由源与目标的象限相对位置决定——左右关系用水平轴反射，上下关系用垂直轴反射，对角关系用双轴反射... → 反射轴由锚块内部特征决定

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D5 variant
# ----------------------------------
# design:
#   concrete_change:
#     - Keep the original 27x27 four-quadrant layout, divider, scaling rule, and
#       "copy the TL motif to the other quadrants" behavior.
#     - Replace the old solid red target anchors with decorated anchors: each
#       anchor is still an NxN red square in bbox/scale terms, but one corner of
#       that square is recolored with a visible marker color.
#     - The TL source anchor is always the same 2x2 template:
#         [[marker, red],
#          [red,    red]]
#       so its asymmetry is visible to the solver.
#     - In non-TL quadrants, the input shows only the scaled anchor template,
#       already reflected in one of four ways: none / LR / UD / both.
#       The quadrant location no longer tells the solver which reflection to use.
#       Instead, the solver must compare the target anchor's internal marker
#       position against the TL anchor's marker position to recover the transform.
#   flow_trace:
#     - Old logic in a right quadrant: always `np.fliplr(source_patch)`.
#     - New logic in the same right quadrant:
#         1. extract the small non-bg target anchor component,
#         2. test the four reflected versions of the TL anchor template,
#         3. pick the one whose scaled result exactly matches the observed anchor,
#         4. apply that same reflection to the full source motif before scaling.
#     - Difference: quadrant coordinates are still used only to locate anchors,
#       not to choose the reflection.
#   complexity_calibration:
#     - There are exactly four reflection categories, which is enough variety to
#       show the new mechanism without introducing a large search space.
#     - The marker sits in a corner, so the four reflected anchor templates are
#       visually distinct and do not require color-id reasoning.
#     - To keep the marker stable under nearest-neighbor resizing, non-TL anchor
#       sizes are restricted to {2, 4}. This preserves the original scaling idea
#       while avoiding ambiguous 2->3 aliasing of the internal marker.
#   implicit_assumptions_review:
#     - Original code assumed `qr == 1` implies LR reflection and `qb == 1`
#       implies UD reflection. That is no longer valid and is replaced by an
#       explicit template match on the observed target anchor.
#     - Original input generator cleared non-TL quadrants down to a plain red
#       square. That would destroy the new control signal, so the generator must
#       now leave the decorated anchor template visible in the input.
#     - Original stamping skipped existing red cells only. That remains safe:
#       the target anchor's marker colors either already match the transformed
#       motif or get written with the same value.
#   placement_legality:
#     - The decorated anchor is the same bbox as the old anchor, so all anchor
#       placement math based on the red bbox size stays valid.
#     - The generator still chooses anchor positions from the full transformed
#       motif footprint, preventing accidental overlap with quadrant borders.
#   visual_observability:
#     - A human can compare the TL anchor's marked corner with the marked corner
#       in each target anchor and see that the copied motif follows that cue.
#     - The generator enforces at least one non-TL quadrant whose transform is
#       different from the old quadrant-based default, preventing silent
#       regression back to the original task.
#   cleanliness:
#     - Each non-TL quadrant contains only one compact decorated anchor in input,
#       so the control signal is isolated and easy to read.
#   color_reasoning:
#     - No ordering or branching uses hidden numeric properties of colors.
#       The dedicated marker color is just a visible structural cue.
# solver_assumptions_added:
#   - Every quadrant contains exactly one red component belonging to the anchor.
#   - In each non-TL quadrant, the only non-bg component containing red is the
#     decorated anchor template.
#   - The TL source anchor template is asymmetric, so exactly one reflected
#     version matches each target anchor.


BG = 4
DIV = 1
RED = 2
MARK = 8
GRID_SIZE = 27
MID = 13

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZES = (2, 4)
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
MAX_RETRIES = 300

TRANSFORMS = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)
OLD_DEFAULT = {
    (0, 1): (False, True),
    (1, 0): (True, False),
    (1, 1): (True, True),
}


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
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect(img: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = img
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return out


def _component_patch(quadrant: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(quadrant != bg, connectivity=4)
    for pts in comps:
        if any(quadrant[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            return quadrant[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))
    raise AssertionError("quadrant must contain a non-bg component with a red anchor")


def _detect_transform(source_anchor: np.ndarray, target_anchor: np.ndarray) -> tuple[bool, bool]:
    for flip_ud, flip_lr in TRANSFORMS:
        cand = _reflect(source_anchor, flip_ud, flip_lr)
        cand = _resize_nn(cand, target_anchor.shape[0], target_anchor.shape[1])
        if np.array_equal(cand, target_anchor):
            return flip_ud, flip_lr
    raise AssertionError("target anchor does not match any reflected source-anchor template")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across quadrants with scaling and anchor-driven reflection.

    Input structure:
    - A single-color row divider and column divider split the board into four quadrants.
    - Every quadrant contains one red square anchor; the non-TL anchors also expose an internal marker pattern.
    - The top-left quadrant contains the full colored motif, and its 2x2 anchor already shows the source marker template.

    Rule:
    - Treat the TL connected component containing the red anchor as the source motif.
    - For each other quadrant, first read the target anchor's internal marker pattern and determine which reflection of the TL anchor template matches it.
    - Apply that reflection to the whole source motif, scale it by the target/source anchor-size ratio, align the reflected anchor onto the target anchor, and stamp the non-background pixels.

    Key insight: Anchor size still tells you the scale, but reflection is no longer tied to quadrant position; it is encoded by the anchor's own internal asymmetry.
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

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left red anchor must stay 2x2 in this variant"

    tl = g[0:mid_r, 0:mid_c]
    patch, _ = _component_patch(tl, bg)

    source_anchor = tl[sr0 : sr0 + sL, sc0 : sc0 + sL].copy()
    assert source_anchor.shape == (SOURCE_ANCHOR_SIZE, SOURCE_ANCHOR_SIZE), "source anchor template must be 2x2"

    for key, (tr0, tc0, tL) in anchors.items():
        if key == (0, 0):
            continue

        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        target_anchor, _ = _component_patch(quad, bg)
        flip_ud, flip_lr = _detect_transform(source_anchor, target_anchor)

        patch_t = _reflect(patch, flip_ud, flip_lr)
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == RED)
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
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

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


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    marker_cell: tuple[int, int],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    free_cells = [p for p in filled if p not in red_cells and p != marker_cell]
    max_extra = min(MOTIF_COLORS_MAX - 2, len(free_cells))
    if max_extra < 1:
        return None
    n_extra = random.randint(1, max_extra)

    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[:n_extra]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    mr, mc = marker_cell
    grid[mr, mc] = MARK

    random.shuffle(free_cells)
    for i, (r, c) in enumerate(free_cells):
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
        logical = _assign_colors(filled, {(ar, ac)}, (ar, ac), lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        # Make the scaled anchor asymmetric inside its 2x2 source block.
        patch[ar * s : ar * s + s, ac * s : ac * s + s] = RED
        patch[ar * s, ac * s] = MARK
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    anchor_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    marker_cell = (ar, ac)
    red_cells = anchor_cells - {marker_cell}
    min_fill = max(len(anchor_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, anchor_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, marker_cell, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _scaled_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool, target_size: int) -> np.ndarray:
    p = _reflect(patch, flip_ud, flip_lr)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
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


def _choose_transforms() -> dict[tuple[int, int], tuple[bool, bool]]:
    while True:
        chosen = {key: random.choice(TRANSFORMS) for key in ((0, 1), (1, 0), (1, 1))}
        if len(set(chosen.values())) < 2:
            continue
        if all(chosen[key] == OLD_DEFAULT[key] for key in chosen):
            continue
        return chosen


def _gen_attempt() -> np.ndarray | None:
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.choice(TARGET_SIZES),
        (1, 0): random.choice(TARGET_SIZES),
        (1, 1): random.choice(TARGET_SIZES),
    }
    if all(size == SOURCE_ANCHOR_SIZE for key, size in target_sizes.items() if key != (0, 0)):
        target_sizes[random.choice([(0, 1), (1, 0), (1, 1)])] = 4

    source = _source_patch(needs_block=False)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    anchor_patch = patch[
        anchor_r_in_patch : anchor_r_in_patch + SOURCE_ANCHOR_SIZE,
        anchor_c_in_patch : anchor_c_in_patch + SOURCE_ANCHOR_SIZE,
    ].copy()

    chosen_transforms = _choose_transforms()
    scaled_full = {}
    scaled_anchor = {}
    for key in ((0, 1), (1, 0), (1, 1)):
        flip_ud, flip_lr = chosen_transforms[key]
        size = target_sizes[key]
        scaled_full[key] = _scaled_patch(patch, flip_ud, flip_lr, size)
        scaled_anchor[key] = _scaled_patch(anchor_patch, flip_ud, flip_lr, size)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key in ((0, 1), (1, 0), (1, 1)):
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], scaled_full[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key in ((0, 1), (1, 0), (1, 1)):
        ar, ac = anchors[key]
        anchor_part = scaled_anchor[key]
        rr_red, cc_red = np.where(anchor_part == RED)
        paste(out, anchor_part, ar - int(rr_red.min()), ac - int(cc_red.min()))
    for key in ((0, 1), (1, 0), (1, 1)):
        _stamp(out, scaled_full[key], *anchors[key])

    inp = out.copy()
    for key in ((0, 1), (1, 0), (1, 1)):
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        anchor_part = scaled_anchor[key]
        rr_red, cc_red = np.where(anchor_part == RED)
        paste(inp, anchor_part, ar - int(rr_red.min()), ac - int(cc_red.min()))

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D5-variant input")
