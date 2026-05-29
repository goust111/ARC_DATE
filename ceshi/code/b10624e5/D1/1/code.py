# Variant of b10624e5
# Dimension: D1 — 匹配条件：红色锚块颜色匹配 (match_cond)
# Change: 所有象限的锚块共享同一颜色（红色），通过 s.anchor_color == d.anchor_color 配对；缩放因... → 锚块形状匹配而非颜色匹配

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   preserved_backbone:
#     - 27x27 grid, bg=4, divider row/col at index 13 with one shared non-bg color.
#     - TL quadrant contains the full colored motif component.
#     - Other quadrants contain only their red anchor in the input.
#     - Output still pastes reflected and scaled copies of the TL motif.
#   d1_change:
#     - All anchors keep the same red color, so color is no longer discriminative.
#     - Each anchor is instead a scaled copy of one 3x3 prototype shape:
#       L / T / plus.
#     - A target receives the motif iff its normalized 3x3 anchor signature matches
#       the TL anchor signature.
#     - Scale factor is the anchor bbox side-length ratio, which equals the logical
#       scale ratio because every anchor bbox is 3k x 3k.
#   concrete_behavior:
#     - TL has a 4x4 asymmetric source patch whose red pixels form one 3x3 prototype.
#     - Non-TL quadrants show a mix of matching and non-matching red shapes.
#     - Matching targets get the reflected/scaled motif; distractors stay as bare anchors.
#   behavior_trace:
#     - Original logic: detect all red components, treat every one as a square, and
#       replicate into every non-TL quadrant using square-size ratio.
#     - Variant logic: detect all red components, normalize each into a 3x3 binary
#       signature, filter to targets with the same signature as TL, and scale by bbox
#       ratio before stamping.
#     - Difference: the solver can no longer assume one universal target class.
#   hidden-assumption_audit:
#     - Original `size = max(...)` assumed anchors were filled squares; that would
#       merge shape identity into one scalar and lose the new matching signal.
#     - Original loop pasted into every non-TL quadrant; in this variant that would
#       incorrectly fill distractor anchors, so matching must be explicit.
#     - Original red-bbox alignment remains valid only because generator guarantees
#       every anchor is a uniform integer scaling of a fixed 3x3 prototype.
#   placement_legality:
#     - Source patch is at most 4x4 logical cells; after scale 3 it is 12x12, so it
#       fits inside a 13x13 quadrant.
#     - Each quadrant has exactly one red component, and TL's red component is the
#       only red inside its full non-bg motif component.
#   observability:
#     - At least one target matches TL and at least one target is a non-matching
#       distractor, so shape classification is necessary.
#     - The source patch is intentionally asymmetric so quadrant reflections are visible
#       even when the anchor prototype itself is symmetric (for example, the plus).
#   cleanliness:
#     - Distractor quadrants contain only a single red anchor and no extra clutter.
#     - Colors are limited and do not encode hidden ordering or matching rules.


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

SOURCE_PATCH_SIZE = 4
TARGET_SCALE_RANGE = (1, 3)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400

SHAPES = {
    "L": np.array(
        [
            [1, 0, 0],
            [1, 0, 0],
            [1, 1, 1],
        ],
        dtype=int,
    ),
    "T": np.array(
        [
            [1, 1, 1],
            [0, 1, 0],
            [0, 1, 0],
        ],
        dtype=int,
    ),
    "P": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=int,
    ),
}

EXTRA_PATTERNS = [
    [(0, 3), (1, 3), (3, 1)],
    [(2, 3), (3, 0), (3, 1)],
    [(1, 3), (3, 2), (3, 3)],
    [(0, 3), (3, 0), (3, 2)],
]


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
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_signature(pts: np.ndarray) -> tuple[tuple[int, ...], ...]:
    """Normalize a red anchor component into its 3x3 prototype signature."""
    r0, c0, r1, c1 = bbox_coords(pts)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "anchor bbox must be square with side divisible by 3"
    k = h // 3
    mask = np.zeros((h, w), dtype=bool)
    for r, c in pts.tolist():
        mask[int(r - r0), int(c - c0)] = True

    rows = []
    for br in range(3):
        row = []
        for bc in range(3):
            block = mask[br * k : (br + 1) * k, bc * k : (bc + 1) * k]
            full = bool(np.all(block))
            empty = not bool(np.any(block))
            assert full or empty, "anchor blocks must be uniform at the logical-cell level"
            row.append(1 if full else 0)
        rows.append(tuple(row))
    return tuple(rows)


def _anchor_patch(signature: tuple[tuple[int, ...], ...], scale: int) -> np.ndarray:
    logical = np.array(signature, dtype=int)
    return np.kron(logical, np.full((scale, scale), RED, dtype=int))


def _build_source_patch(signature: tuple[tuple[int, ...], ...]) -> np.ndarray:
    """Create a 4x4 logical source motif whose red region is the given 3x3 prototype."""
    patch = np.full((SOURCE_PATCH_SIZE, SOURCE_PATCH_SIZE), BG, dtype=int)
    anchor = np.array(signature, dtype=int)
    patch[:3, :3][anchor == 1] = RED

    extra_cells = random.choice(EXTRA_PATTERNS)
    palette = COLOR_POOL[:]
    random.shuffle(palette)
    n_extra_colors = random.randint(2, min(3, len(extra_cells)))
    extra_colors = palette[:n_extra_colors]
    for idx, (r, c) in enumerate(extra_cells):
        patch[r, c] = extra_colors[idx % len(extra_colors)]
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif only onto quadrants whose red anchor has the same shape class.

    Input structure:
    - A single-color full-row/full-column divider splits the board into four 13x13 quadrants.
    - Every quadrant contains exactly one red anchor component, but anchors are no longer all squares:
      they are scaled copies of a few 3x3 prototype shapes.
    - The top-left quadrant additionally contains one colored motif component attached to its red anchor.

    Rule:
    - Use the top-left motif as the only source patch.
    - Normalize each red anchor into a 3x3 binary signature and keep only targets whose signature
      matches the top-left anchor signature.
    - Reflect the source patch by quadrant (left/right, top/bottom) and scale it by the target/source
      anchor bbox ratio before aligning the scaled red region onto the target anchor.

    Key insight: Red color no longer identifies the destination set; the anchor's shape class does.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

    anchors: dict[tuple[int, int], dict[str, object]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        signature = _anchor_signature(pts)
        anchors[key] = {
            "r0": int(r0),
            "c0": int(c0),
            "side": int(r1 - r0 + 1),
            "signature": signature,
        }

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor"
    source_anchor = anchors[(0, 0)]
    source_side = int(source_anchor["side"])
    source_signature = source_anchor["signature"]

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts.tolist()):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red anchor"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for (qb, qr), anchor in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        if anchor["signature"] != source_signature:
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        ratio = float(anchor["side"]) / float(source_side)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve the motif information"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled patch must retain the red anchor"
        place_r0 = int(anchor["r0"]) - int(rr_red.min())
        place_c0 = int(anchor["c0"]) - int(cc_red.min())

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


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_scale: int) -> np.ndarray:
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    return _resize_nn(p, patch.shape[0] * target_scale, patch.shape[1] * target_scale)


def _choose_anchor_for_patch(qb: int, qr: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Choose the red-anchor bbox top-left inside the quadrant, preferring full motif fit."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())
    red_r1, red_c1 = int(rr.max()), int(cc.max())
    anchor_h = red_r1 - red_r0 + 1
    anchor_w = red_c1 - red_c0 + 1

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - anchor_h + 1
    anch_c0, anch_c1 = q_c0, q_c1 - anchor_w + 1

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
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    source_name = random.choice(list(SHAPES))
    source_signature = tuple(tuple(int(v) for v in row) for row in SHAPES[source_name].tolist())
    source_patch = _build_source_patch(source_signature)

    other_keys = [(0, 1), (1, 0), (1, 1)]
    matched_count = random.choice([1, 2])
    matched_keys = set(random.sample(other_keys, matched_count))
    nonmatched_keys = [key for key in other_keys if key not in matched_keys]

    other_shape_names = [name for name in SHAPES if name != source_name]
    quadrant_shapes: dict[tuple[int, int], tuple[tuple[int, ...], ...]] = {(0, 0): source_signature}
    for key in matched_keys:
        quadrant_shapes[key] = source_signature
    for idx, key in enumerate(nonmatched_keys):
        name = other_shape_names[idx % len(other_shape_names)]
        quadrant_shapes[key] = tuple(tuple(int(v) for v in row) for row in SHAPES[name].tolist())

    scales = {(0, 0): 1}
    for key in other_keys:
        scales[key] = random.randint(*TARGET_SCALE_RANGE)
    if all(scales[key] == 1 for key in matched_keys):
        bump = random.choice(list(matched_keys))
        scales[bump] = random.choice([2, 3])

    scaled_patches = {
        key: _scaled_patch(source_patch, key[0], key[1], scales[key])
        for key in matched_keys
    }
    anchor_imgs = {
        key: _anchor_patch(quadrant_shapes[key], scales[key])
        for key in [(0, 0), *other_keys]
    }

    tl_r0 = random.randint(0, MID - source_patch.shape[0])
    tl_c0 = random.randint(0, MID - source_patch.shape[1])
    anchors: dict[tuple[int, int], tuple[int, int]] = {(0, 0): (tl_r0, tl_c0)}
    for key in other_keys:
        if key in matched_keys:
            anchors[key] = _choose_anchor_for_patch(key[0], key[1], scaled_patches[key])
        else:
            anchors[key] = _choose_anchor_for_patch(key[0], key[1], anchor_imgs[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, source_patch, tl_r0, tl_c0)
    for key in other_keys:
        ar, ac = anchors[key]
        paste(out, anchor_imgs[key], ar, ac)
    for key in matched_keys:
        _stamp(out, scaled_patches[key], *anchors[key])

    inp = out.copy()
    for key in other_keys:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, anchor_imgs[key], ar, ac)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D1 input")
