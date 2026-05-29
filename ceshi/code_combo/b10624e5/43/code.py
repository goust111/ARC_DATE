# Combo variant of b10624e5 (index 43)
# combo_spec: D2/0+D5/2+D15/1
# Name: 每象限独立源 + 互为源目标 + 无反射仅平移缩放
#
# Design analysis for the combo variant
# -------------------------------------
# spec adaptation:
#   grid:
#     size: 27x27
#     bg: 4
#     divider: one full row + one full col of color 1 at row=13, col=13
#   anchors:
#     count: exactly 4, one red square per quadrant
#     sizes:
#       TL: fixed 2x2
#       TR/BL/BR: random in 1..4
#   local_sources:
#     every quadrant contains one local non-bg component with its own red anchor
#     each local component is a 2x2 logical motif upscaled by that quadrant's anchor size
#     the red logical cell sits in a quadrant-specific corner:
#       TL -> bottom-right  => colored lobe lies to the NW of the anchor
#       TR -> bottom-left   => colored lobe lies to the NE of the anchor
#       BL -> top-right     => colored lobe lies to the SW of the anchor
#       BR -> top-left      => colored lobe lies to the SE of the anchor
#   pairing:
#     TL <-> BR
#     TR <-> BL
#     every quadrant is both a source and a destination
#   output_rule:
#     keep all four local motifs
#     additionally copy each source motif to the diagonally opposite quadrant
#     do not reflect
#     scale only by destination_anchor_size / source_anchor_size
#     align the scaled red region to the destination anchor and stamp non-bg pixels
#
# a) Concrete behavior and trace:
#    Original behavior: only TL owned a motif, and the solver mirrored it into the other
#    three quadrants. This combo changes three stages at once.
#    - Source extraction: there are now four local sources, not one TL-only source.
#    - Routing: each source goes only to its diagonal opposite, so all four quadrants are
#      active as both source and target.
#    - Geometry: the transferred motif is not flipped; only nearest-neighbor scaling remains.
#    Example: TL may contain logical [[7, 5], [3, 2]] with red at bottom-right, so its
#    colored lobe sits NW of the TL anchor. If BR has anchor size 4, BR receives a 2x
#    enlarged copy of that exact orientation, still NW of the BR anchor. BR's own local
#    source sits SE of its anchor, so the output shows two disjoint lobes around BR: local
#    SE colors from BR itself and imported NW colors from TL.
#
# b) Complexity calibration:
#    The task is still visibly richer than the original. Inputs already show four different
#    local sources, and outputs show four imported additions. The no-reflection rule is
#    observable because imported lobes stay in the source orientation instead of landing in
#    the mirror position. The pairing space stays small and repeated across all examples:
#    only diagonal exchange exists, so the puzzle remains learnable from 2-4 samples.
#
# c) Hidden-assumption audit against the original solver:
#    - "Only TL has a motif" is no longer true; the solver must extract one anchor-containing
#      component per quadrant.
#    - "TL anchor size is the only source denominator" is no longer true; scaling must use
#      the source quadrant's own anchor size.
#    - "Quadrant relation determines reflection axes" is removed; destination is still chosen
#      by quadrant symmetry, but orientation is preserved.
#    - "Targets are empty except for red anchors" is no longer true; every target quadrant
#      already contains its own local motif, so generator placement must keep local and
#      imported lobes disjoint except for the shared red anchor region.
#
# d) Placement legality:
#    The quadrant-specific red-corner design makes the local and imported lobes occupy opposite
#    diagonal sides of the same anchor after diagonal exchange without reflection:
#      TL output: local NW + imported SE
#      TR output: local NE + imported SW
#      BL output: local SW + imported NE
#      BR output: local SE + imported NW
#    Therefore each destination quadrant receives exactly one non-overlapping imported lobe.
#    Anchors are sampled with a one-anchor-size margin on all four sides, which guarantees that
#    both the local 2Sx2S patch and the imported 2Sx2S patch fit completely inside the 13x13
#    quadrant for every size S in {1,2,3,4}.
#
# e) Visual observability:
#    A human can see in the input that all four quadrants already contain colored motifs, so
#    the puzzle cannot be solved by a TL-only copy rule. In the output, each anchor gains a
#    second lobe whose colors match the diagonal opposite quadrant. Because the imported lobe
#    keeps the source orientation, the no-reflection change is also directly visible.
#
# f) Visual cleanliness:
#    Each motif is a small 2x2 logical design with exactly three non-red cells and one red
#    cell, upscaled uniformly. This keeps the scene sparse, prevents crowded overlaps, and
#    makes it easy to trace which imported lobe came from which opposite quadrant.
#
# g) Change inferability:
#    Colors are copied verbatim; there is no hidden remapping or color arithmetic. Distinct
#    local signatures are enforced so that opposite-quadrant matching is driven by visible
#    shape-plus-color patterns, not by color id ordering or any non-visual convention.
#
# h) New solver assumptions and generator guarantees:
#    - Solver assumes exactly one red anchor component per quadrant; generator enforces that.
#    - Solver assumes each quadrant contains exactly one connected non-bg component containing
#      red in the input; generator constructs exactly one such local component per quadrant.
#    - Solver assumes diagonal pairing; generator always uses TL<->BR and TR<->BL.
#    - Solver assumes nearest-neighbor scaling is lossless; generator builds each motif as an
#      integer-block upscale of a 2x2 logical template, so roundtrip resize is exact.

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
PAIR_ORDER = [(0, 0), (0, 1), (1, 0), (1, 1)]
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400
RED_POS_BY_QUAD = {
    (0, 0): (1, 1),
    (0, 1): (1, 0),
    (1, 0): (0, 1),
    (1, 1): (0, 0),
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
    matches = [pts for pts in comps if any(quad[r, c] == RED for r, c in pts)]
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
    rr_red, cc_red = np.where(patch_s == RED)
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
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange four quadrant-local motifs across diagonal quadrants using scaling only.

    Input structure:
    - Grid is split into four 13x13 quadrants by one full divider row and one full divider column.
    - Each quadrant contains exactly one red square anchor block.
    - Every quadrant already contains its own colored motif attached to that anchor, so there are
      four independent source components rather than one TL-only source.

    Rule:
    - Extract the anchor-containing component from each quadrant as that quadrant's local source.
    - Send each source to the diagonally opposite quadrant: TL<->BR and TR<->BL.
    - Keep the motif orientation unchanged, resize it by destination_anchor_size / source_anchor_size,
      align the scaled red region to the destination anchor, and stamp non-background pixels.

    Key insight: The puzzle now combines per-quadrant source discovery, diagonal source-target
    exchange, and anchor-size-based scaling. Quadrant relation chooses the opposite destination,
    but unlike the original task it no longer chooses any reflection axis.
    """
    g = np.asarray(grid, dtype=int)
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
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert len(anchors) == 4, "all four quadrants must contain red anchors"

    patches = {key: _extract_source_patch(g, bg, mid_r, mid_c, key) for key in PAIR_ORDER}

    for src_key in PAIR_ORDER:
        dst_key = _opposite_quadrant(src_key)
        patch = patches[src_key]
        _, _, src_size = anchors[src_key]
        dst_r0, dst_c0, dst_size = anchors[dst_key]
        assert src_size > 0 and dst_size > 0, "anchor sizes must be positive"

        ratio = float(dst_size) / float(src_size)
        new_h = max(1, int(round(patch.shape[0] * ratio)))
        new_w = max(1, int(round(patch.shape[1] * ratio)))
        patch_s = _resize_nn(patch, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch.shape[0], patch.shape[1])
        assert np.array_equal(roundtrip, patch), "scaling must preserve all motif information"

        _stamp_patch(out, patch_s, dst_r0, dst_c0, bg, mid_r, mid_c)

    return out


def _logical_signature(logical: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in logical.flatten())


def _make_logical_motif(forbidden: set[tuple[int, ...]], key: tuple[int, int]) -> np.ndarray | None:
    red_r, red_c = RED_POS_BY_QUAD[key]
    for _ in range(200):
        colors = random.sample(COLOR_POOL, 3)
        random.shuffle(colors)
        logical = np.full((2, 2), BG, dtype=int)
        idx = 0
        for r in range(2):
            for c in range(2):
                if (r, c) == (red_r, red_c):
                    logical[r, c] = RED
                else:
                    logical[r, c] = colors[idx]
                    idx += 1
        sig = _logical_signature(logical)
        if sig not in forbidden:
            return logical
    return None


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
    assert rr_red.size > 0, "local patch must contain red anchor pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())
    paste(out, patch, place_r0, place_c0)


def _scaled_patch_for_pair(
    patch: np.ndarray,
    src_size: int,
    dst_size: int,
) -> np.ndarray:
    ratio = float(dst_size) / float(src_size)
    new_h = max(1, int(round(patch.shape[0] * ratio)))
    new_w = max(1, int(round(patch.shape[1] * ratio)))
    return _resize_nn(patch, new_h, new_w)


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
        logical = _make_logical_motif(forbidden, key)
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
        patch_s = _scaled_patch_for_pair(patches[src_key], sizes[src_key], sizes[dst_key])
        roundtrip = _resize_nn(patch_s, patches[src_key].shape[0], patches[src_key].shape[1])
        if not np.array_equal(roundtrip, patches[src_key]):
            return None
        _stamp_patch(out, patch_s, anchors[dst_key][0], anchors[dst_key][1], BG, MID, MID)

    solved = main(inp)
    return inp if np.array_equal(solved, out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
