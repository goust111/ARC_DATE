# Variant of b10624e5
# Dimension: D16 — 缩放因子的确定方式 (meta)
# Change: 缩放因子 = d.red_block.size / s.red_block.size，由锚块的边长比直接计算... → 缩放因子由象限面积比决定

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D16 variant
# ----------------------------------
# a) Concrete behavior:
#    - The divider row/col is no longer fixed at the center. It still forms four visible quadrants,
#      but their widths/heights differ.
#    - The TL quadrant still contains the only full motif component in the input.
#    - The other quadrants still contain only a red anchor block in the input, but the anchor size is
#      deliberately decoupled from scale: generated examples keep those anchors at 2x2 so the solver
#      must read scale from quadrant area, not from anchor size.
#    - Small trace: if TL is 11x15 (area 165), TR is 11x11 (area 121), BL is 15x15 (area 225),
#      BR is 15x11 (area 165), then the three target scale factors are 121/165, 225/165, 165/165.
#      Under the old logic all three targets would stay at scale 1 because their anchors are all 2x2.
#      Under the new logic TR shrinks, BL enlarges, BR keeps the source scale.
# b) Complexity calibration:
#    - Divider positions are restricted to the near-center asymmetric pairs (11/12 with 14/15, or the
#      transpose). This keeps area ratios visible but moderate, avoiding extreme clipping.
#    - One target quadrant is smaller, one is larger, and one has the same area as TL. That repeated
#      pattern gives the human solver multiple support cases inside one grid.
# c) Implicit-assumption audit from the old solver:
#    - Old `ratio = tL / sL` relied on anchor size encoding scale. That is removed.
#    - Old roundtrip-resize assert relied on specially constructed lossless scale factors. Arbitrary
#      area ratios do not preserve that property, so the assert must be dropped.
#    - Alignment by the scaled red subregion's top-left remains valid as long as the scaled patch still
#      contains at least one red pixel. The generator guarantees that by keeping ratios in ~[0.67, 1.46]
#      and using a 2x2 source anchor.
# d) Placement legality:
#    - The generator computes quadrant bounds from the chosen divider, scales each target motif with the
#      area ratio, and chooses anchor locations that prefer full motif fit inside the quadrant.
#    - Clearing non-TL quadrants when building the input restores the intended input format exactly.
# e) Visual observability:
#    - Reflection remains the same as the original task, but motifs are forced to be asymmetric under
#      LR and UD reflection so the mirrored relation stays visible.
#    - Equal-sized non-TL anchors combined with unequal output motif sizes make the D16 change explicit.
# f) Visual cleanliness:
#    - Divider color is kept out of the random motif color pool so the off-center split remains easy to
#      read and does not get confused with motif content.
#    - Source motifs stay compact (at most 6x6 before scaling), limiting clutter and clipping.
# g) Color inference:
#    - No color remapping is introduced. Colors are copied directly from the TL motif, so the only new
#      concept the user needs to infer is "scale comes from quadrant area".
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes exactly one red component per quadrant in the input; the generator enforces that.
#    - `main()` assumes the TL non-background component containing red is the source motif; the generator
#      places no competing TL component.
#    - `main()` assumes every scaled target patch retains at least one red pixel; the divider range and
#      patch size limits are chosen to guarantee that for generated samples.


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


def _quadrants(h: int, w: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _quad_area(bounds: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = bounds
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants, but let quadrant area determine the scale.

    Input structure:
    - A single full row and a single full column of the same non-background color split the grid into
      four quadrants; the split may be off-center, so the quadrant areas can differ.
    - Each quadrant contains one red anchor square.
    - The top-left quadrant also contains one colored non-background component attached to its anchor.

    Rule:
    - Extract the top-left component that contains the red anchor and use it as the source motif.
    - Reflect it left-right for right quadrants and up-down for bottom quadrants, as in the original task.
    - Scale each reflected copy by `target_quadrant_area / source_quadrant_area`, then align the scaled
      red subregion to the target anchor and stamp non-background pixels.

    Key insight: the visible off-center divider changes the sizes of the four quadrants, and those area
    ratios now control motif scale; the red anchors are only placement markers.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    quads = _quadrants(h, w, mid_r, mid_c)

    red = components_mask(g == 2, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert set(anchors) == set(quads), "each quadrant must contain exactly one red anchor block"

    tl_r0, tl_r1, tl_c0, tl_c1 = quads[(0, 0)]
    tl = g[tl_r0 : tl_r1 + 1, tl_c0 : tl_c1 + 1]
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
    source_area = _quad_area(quads[(0, 0)])

    for (qb, qr), (tr0, tc0, _tr1, _tc1) in anchors.items():
        if (qb, qr) == (0, 0):
            continue

        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        target_area = _quad_area(quads[(qb, qr)])
        ratio = float(target_area) / float(source_area)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)

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

BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27
SOURCE_ANCHOR_SIZE = 2
TARGET_ANCHOR_SIZE = 2

SMALL_DIVS = (11, 12)
LARGE_DIVS = (14, 15)

BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 5)
FREEFORM_W_RANGE = (3, 5)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]

MAX_RETRIES = 400


def _choose_dividers() -> tuple[int, int]:
    if random.random() < 0.5:
        return random.choice(SMALL_DIVS), random.choice(LARGE_DIVS)
    return random.choice(LARGE_DIVS), random.choice(SMALL_DIVS)


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
    h: int,
    w: int,
) -> np.ndarray | None:
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


def _motif_is_asymmetric(patch: np.ndarray) -> bool:
    return not np.array_equal(patch, np.fliplr(patch)) and not np.array_equal(patch, np.flipud(patch))


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    s = SOURCE_ANCHOR_SIZE
    use_block = random.random() < BLOCK_PROB
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
        if not _motif_is_asymmetric(patch):
            return None
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar = random.randint(0, h - s)
    ac = random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None or not _motif_is_asymmetric(patch):
        return None
    return patch, ar, ac


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    source_area: int,
    target_area: int,
) -> np.ndarray:
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = target_area / source_area
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_anchor(
    bounds: tuple[int, int, int, int],
    size: int,
    patch_s: np.ndarray,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = bounds
    rr, cc = np.where(patch_s == RED)
    if rr.size == 0:
        return None
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - size + 1
    anch_c0, anch_c1 = q_c0, q_c1 - size + 1
    if anch_r0 > anch_r1 or anch_c0 > anch_c1:
        return None

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, mid_r: int, mid_c: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    src_rr, src_cc = np.where(patch_s != BG)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0

    h, w = out.shape
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < h)
        & (tgt_cc >= 0)
        & (tgt_cc < w)
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr, tgt_cc, src_rr, src_cc = tgt_rr[valid], tgt_cc[valid], src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    mid_r, mid_c = _choose_dividers()
    quads = _quadrants(GRID_SIZE, GRID_SIZE, mid_r, mid_c)

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    tl_bounds = quads[(0, 0)]
    tl_h = tl_bounds[1] - tl_bounds[0] + 1
    tl_w = tl_bounds[3] - tl_bounds[2] + 1
    if patch.shape[0] > tl_h or patch.shape[1] > tl_w:
        return None

    source_area = _quad_area(tl_bounds)
    scaled: dict[tuple[int, int], np.ndarray] = {}
    for key in quads:
        if key == (0, 0):
            continue
        patch_s = _scaled_patch(patch, key[0], key[1], source_area, _quad_area(quads[key]))
        if not np.any(patch_s == RED):
            return None
        scaled[key] = patch_s

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_patch_r0 = random.randint(tl_bounds[0], tl_bounds[1] - patch.shape[0] + 1)
    tl_patch_c0 = random.randint(tl_bounds[2], tl_bounds[3] - patch.shape[1] + 1)
    anchors[(0, 0)] = (tl_patch_r0 + anchor_r_in_patch, tl_patch_c0 + anchor_c_in_patch)

    for key, patch_s in scaled.items():
        pos = _choose_anchor(quads[key], TARGET_ANCHOR_SIZE, patch_s)
        if pos is None:
            return None
        anchors[key] = pos

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    paste(out, patch, tl_patch_r0, tl_patch_c0)

    target_anchor = np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int)
    for key in quads:
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, target_anchor, ar, ac)

    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key], mid_r, mid_c)

    inp = out.copy()
    for key in quads:
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = quads[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, target_anchor, ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D16 input")
