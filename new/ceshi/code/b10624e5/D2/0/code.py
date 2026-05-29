# Variant of b10624e5
# Dimension: D2 — 源对象：左上象限锚块旁的彩色图案 (src)
# Change: 左上象限中与红色锚块相邻的彩色像素构成源图案... → 每个象限都有独立的源图案
#
# Design analysis for the D2 variant
# ----------------------------------
# spec:
#   grid:
#     size: 27x27
#     bg: 4
#     divider: one full row + one full col of color 1 at row=13, col=13
#   anchors:
#     count: exactly 4, one red square per quadrant
#     sizes:
#       TL: fixed 2x2 to preserve the original anchor convention
#       TR/BL/BR: random in 1..4
#   local_sources:
#     each quadrant contains one connected non-bg component with its own red anchor
#     each source patch is a 2x2 logical motif upscaled by that quadrant's anchor size
#     logical layout is:
#       [a b]
#       [c 2]
#     where a/b/c are non-red colors and the three-color arrangement is unique per quadrant
#   pairing:
#     TL <-> BR
#     TR <-> BL
#     each quadrant is both a source and a destination
#   output_rule:
#     keep all four input-local motifs
#     additionally copy each quadrant's local motif to the diagonally opposite quadrant
#     reflect across both axes (because source and destination are diagonal opposites)
#     scale by destination_anchor_size / source_anchor_size
#     align the scaled red region to the destination anchor and stamp non-bg pixels
#
# a) Concrete behavior and trace:
#    Original: only TL had a colored component, so `main()` extracted one TL patch and
#    propagated it outward. Variant: all four quadrants now contain a source component.
#    Example: if TL has logical colors [[7,3],[5,2]] and BR has size 4 while TL has size 2,
#    the BR output gains a 2x-scaled flipped copy of TL near its BR anchor. Meanwhile BR's
#    own local motif is still present in the input and TL receives BR's flipped-downscaled copy.
#    The changed processing stage is source extraction: instead of one TL-only patch, solver
#    must extract one patch per quadrant, then route each patch to its opposite quadrant.
#
# b) Complexity calibration:
#    The variant is not reduced to a trivial recolor. Every input already contains four motifs,
#    and every output contains eight visible motif lobes around four anchors (four local NW lobes
#    plus four imported SE lobes). Pairing is fixed to two diagonal exchanges, so the rule remains
#    learnable from a few examples without introducing too many branches.
#
# c) Hidden-assumption audit against the original solver:
#    - Original assumption: only TL has a non-red source component. No longer valid.
#    - Original code extracted one patch from `tl = g[0:mid_r, 0:mid_c]`; variant must do this
#      independently inside every quadrant.
#    - Original code used TL anchor size as the only source scale denominator; variant needs a
#      per-source denominator because each quadrant can now originate a copy.
#    - Original code could stamp into empty non-TL quadrants. Variant output stamps into quadrants
#      that already contain their own source motif, so generator must ensure imported pixels land
#      in a disjoint region around the same anchor.
#
# d) Placement legality:
#    Each local motif is anchored on the NW side of its red square: the patch bbox ends exactly at
#    the anchor's bottom-right corner. After diagonal reflection, the imported motif lands on the SE
#    side of the destination anchor. This makes the local and imported lobes disjoint except for the
#    shared red anchor area. Anchors are sampled so both lobes fit fully inside the 13x13 quadrant.
#
# e) Visual observability:
#    The core variant is directly visible in the input: non-TL quadrants now contain local colored
#    motifs instead of bare red anchors. In the output, each anchor gains a second lobe whose colors
#    match the opposite quadrant rather than TL, so the "four independent sources" rule is observable.
#
# f) Visual cleanliness:
#    Using one fixed 2x2 logical motif per quadrant keeps the scene readable. The two motif lobes in
#    each output quadrant occupy opposite diagonal sides of the anchor, avoiding cluttered overlap.
#
# g) Color-rule inferability:
#    No hidden color arithmetic is used. Colors are copied verbatim with only geometric transforms.
#    Quadrants are distinguished by spatial position, not by color id ordering or remapping.
#
# h) New solver assumptions and generator guarantees:
#    - Solver assumes exactly one red anchor component per quadrant; generator enforces this.
#    - Solver assumes each quadrant has exactly one connected non-bg component containing red in the
#      input; generator creates exactly one such component per quadrant.
#    - Solver assumes the copied motif can be aligned by the scaled red bbox; generator builds motifs
#      by integer-block upscaling, so nearest-neighbor scaling is lossless under every chosen ratio.

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
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _quad_bounds(shape: tuple[int, int], mid_r: int, mid_c: int, key: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = shape
    qb, qr = key
    r0 = 0 if qb == 0 else mid_r + 1
    r1 = mid_r - 1 if qb == 0 else h - 1
    c0 = 0 if qr == 0 else mid_c + 1
    c1 = mid_c - 1 if qr == 0 else w - 1
    return int(r0), int(r1), int(c0), int(c1)


def _opposite_quadrant(key: tuple[int, int]) -> tuple[int, int]:
    return 1 - key[0], 1 - key[1]


def _reflect_between_quadrants(patch: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> np.ndarray:
    out = patch
    if src[1] != dst[1]:
        out = np.fliplr(out)
    if src[0] != dst[0]:
        out = np.flipud(out)
    return out


def _extract_source_patch(
    g: np.ndarray,
    bg: int,
    mid_r: int,
    mid_c: int,
    key: tuple[int, int],
) -> np.ndarray:
    q_r0, q_r1, q_c0, q_c1 = _quad_bounds(g.shape, mid_r, mid_c, key)
    quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    comps = components_mask(quad != bg, connectivity=4)
    matches = [pts for pts in comps if any(quad[r, c] == 2 for r, c in pts)]
    assert len(matches) == 1, "each quadrant must contain exactly one non-bg component with red"
    r0, c0, r1, c1 = bbox_coords(matches[0])
    return quad[r0 : r1 + 1, c0 : c1 + 1].copy()


def _stamp_patch(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    h, w = out.shape
    rr_red, cc_red = np.where(patch_s == 2)
    assert rr_red.size > 0, "scaled patch must contain red anchor pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where(patch_s != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < h and 0 <= cc_t < w):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == 2:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange quadrant-local motifs across diagonal quadrants using anchor-sized scaling.

    Input structure:
    - Grid divided into four quadrants by a full-row and full-column divider of one non-bg color.
    - Each quadrant contains one red (2) square anchor block.
    - Unlike the original task, every quadrant already contains its own colored motif attached to that anchor.

    Rule:
    - Treat each quadrant's local anchor-containing component as an independent source pattern.
    - Copy each source only to the diagonally opposite quadrant.
    - Reflect across left/right and up/down as required by the source/destination quadrant relation, then scale by
      destination_anchor_size / source_anchor_size and align the scaled red region to the destination anchor.

    Key insight: The source is no longer uniquely TL; the solver must identify four local sources and pair each one
    with its diagonal counterpart, using anchor size for scaling and quadrant opposition for reflection.
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

    assert len(anchors) == 4, "all four quadrants must contain red anchors"

    patches = {key: _extract_source_patch(g, bg, mid_r, mid_c, key) for key in anchors}

    for src_key, patch in patches.items():
        dst_key = _opposite_quadrant(src_key)
        sr0, sc0, sL = anchors[src_key]
        tr0, tc0, tL = anchors[dst_key]
        assert sL > 0 and tL > 0, "anchor sizes must be positive"

        patch_t = _reflect_between_quadrants(patch, src_key, dst_key)
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        _stamp_patch(out, patch_s, tr0, tc0, bg, mid_r, mid_c)

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
PAIR_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _logical_signature(logical: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in logical.flatten())


def _make_logical_motif(forbidden: set[tuple[int, ...]]) -> np.ndarray:
    for _ in range(200):
        cells = random.sample(COLOR_POOL, 3)
        random.shuffle(cells)
        logical = np.array([[cells[0], cells[1]], [cells[2], RED]], dtype=int)
        sig = _logical_signature(logical)
        if sig not in forbidden:
            return logical
    return None  # type: ignore[return-value]


def _source_patch(size: int, logical: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(logical, size, axis=0), size, axis=1)


def _choose_anchor(key: tuple[int, int], size: int) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    r_lo = q_r0 + size
    r_hi = q_r1 - (2 * size) + 1
    c_lo = q_c0 + size
    c_hi = q_c1 - (2 * size) + 1
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _place_local_patch(out: np.ndarray, patch: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(patch == RED)
    assert rr_red.size > 0
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    paste(out, patch, place_r0, place_c0)


def _scaled_patch_for_pair(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    src_size: int,
    dst_size: int,
) -> np.ndarray:
    reflected = _reflect_between_quadrants(patch, src_key, dst_key)
    ratio = float(dst_size) / float(src_size)
    new_h = max(1, int(round(reflected.shape[0] * ratio)))
    new_w = max(1, int(round(reflected.shape[1] * ratio)))
    return _resize_nn(reflected, new_h, new_w)


def _gen_attempt() -> np.ndarray | None:
    sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }
    if len(set(sizes.values())) < 2:
        return None

    forbidden: set[tuple[int, ...]] = set()
    logicals: dict[tuple[int, int], np.ndarray] = {}
    for key in PAIR_ORDER:
        logical = _make_logical_motif(forbidden)
        if logical is None:
            return None
        logicals[key] = logical
        forbidden.add(_logical_signature(logical))

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    for key in PAIR_ORDER:
        anchor = _choose_anchor(key, sizes[key])
        if anchor is None:
            return None
        anchors[key] = anchor

    patches = {key: _source_patch(sizes[key], logicals[key]) for key in PAIR_ORDER}

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    for key in PAIR_ORDER:
        _place_local_patch(inp, patches[key], *anchors[key])

    out = inp.copy()
    for src_key in PAIR_ORDER:
        dst_key = _opposite_quadrant(src_key)
        patch_s = _scaled_patch_for_pair(patches[src_key], src_key, dst_key, sizes[src_key], sizes[dst_key])
        roundtrip = _resize_nn(patch_s, patches[src_key].shape[0], patches[src_key].shape[1])
        reflected = _reflect_between_quadrants(patches[src_key], src_key, dst_key)
        if not np.array_equal(roundtrip, reflected):
            return None
        _stamp_patch(out, patch_s, anchors[dst_key][0], anchors[dst_key][1], BG, MID, MID)

    solved = main(inp)
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D2 variant input")
