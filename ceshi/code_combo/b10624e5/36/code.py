# Combo variant of b10624e5 (index 36)
# combo_spec: D12/0+D16/2
# Name: 边缘缩放因子图例 + 缩放因子固定为2的幂次

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for combo D12/0 + D16/2
# ---------------------------------------
# a) Concrete behavior and trace
#    - The original task used target red-square size to imply the scale ratio.
#      Here that cue is removed: every non-TL anchor is fixed at 2x2, and scale is
#      written explicitly as a straight legend bar on the grid edge.
#    - The D16 restriction is fused directly into the legend alphabet: bar lengths are
#      not arbitrary small integers any more, only powers of two {1, 2, 4}. The solver
#      must both find the legend and respect that only these discrete values are valid.
#    - Small trace:
#        original: TL anchor is 2x2, BR anchor is 4x4 -> ratio 2, so the reflected TL
#        patch is enlarged by 2.
#        combo: BR anchor is still only 2x2, but the bottom-edge legend bar has length 4
#        -> the BR patch is reflected and enlarged by 4 while the anchor stays 2x2.
#      The anchor now means "place the copied motif here"; the edge legend alone means
#      "how much to enlarge it".
#    - As in the single D12 variant, the enlarged red region inside the scaled patch is
#      not repainted over the fixed target anchor. Only non-red motif pixels are stamped.
# b) Complexity calibration
#    - Restricting scales to {1, 2, 4} keeps the symbolic search space tiny and visually
#      regular. Three values are enough to show a discrete code without making the puzzle
#      numerically noisy.
#    - Every generated sample uses both 2 and 4 somewhere, and the scale triple is never
#      uniform. That prevents regression to a same-size copy task and makes the power-of-two
#      restriction observable across a small training set.
#    - Source motifs are forced to be asymmetric under both LR and UD flips so that the
#      quadrant-dependent reflections remain recoverable by eye.
# c) Hidden assumptions from the original solver that must change
#    - Original anchor parsing stored a target size and used size/source_size as the scale.
#      That premise is invalid once all target anchors are fixed 2x2. The new solver still
#      locates anchors by quadrant, but obtains scale only from legends.
#    - Original placement assumed the scaled red footprint should match the target anchor.
#      In this combo the scaled red footprint may be 2x2, 4x4, or 8x8 while the visible
#      target anchor stays 2x2, so overlap-by-size is no longer meaningful.
#    - Original generation allowed target sizes 1..4. Doing that here would accidentally
#      reintroduce a second scale cue and dilute the legend; therefore generator fixes all
#      non-TL anchors at 2x2 and varies only legend length.
#    - The new legend decoder assumes exactly three thin straight border-touching legend
#      components of color 8, one per target quadrant. The generator enforces that.
# d) Placement legality
#    - Legends occupy reserved strips: row 0 for TR, column 0 for BL, row 26 for BR.
#      Target motifs are sampled inside quadrant interiors that exclude those strips, so
#      copied content never overwrites the legends.
#    - Source motif bbox is capped at 3x3. This is necessary because 4x scaling must still
#      fit in the 12-cell-tall usable interior next to each legend edge.
#    - Anchor positions are chosen only from placements where the fully transformed patch
#      fits inside the target usable region, avoiding clipping, overlap with the divider,
#      and accidental dependence on out-of-bounds writes.
# e) Visual observability
#    - The key visible departure from the original task is that all target anchors have the
#      same size, yet copied motifs in the output appear at three discrete magnifications
#      controlled only by edge bars.
#    - Because legend lengths come only from {1, 2, 4}, the repeated samples show a clean
#      doubling progression instead of arbitrary lengths. A human can notice that 3 never
#      appears and that 4 is a doubled version of 2.
# f) Visual cleanliness
#    - Legends use a dedicated teal color 8 that is excluded from motif colors, so symbols
#      remain isolated from copied content.
#    - Target motifs are placed away from the legend-bearing border and remain fully inside
#      their quadrants, which keeps anchor, legend, and copied patch visually separable.
# g) Change inferability audit
#    - No behavior depends on color-id ordering or hidden metadata.
#    - The only symbolic code is bar length, which is spatially attached to a single target
#      quadrant. Reflection still comes from quadrant position, so the two control signals
#      (legend for scale, quadrant for flip) are separable and inferable from 2-3 examples.
#
# New solver assumptions guaranteed by generate_input()
# - Divider color is 1, anchor color is 2, legend color is 8, and background is 4.
# - There are exactly three legend bars: top edge in TR, left edge in BL, bottom edge in BR.
# - Every legend bar is one-cell thick, straight, and has length in {1, 2, 4}.
# - TL contains exactly one non-background connected component with the source 2x2 red anchor.
# - All non-TL anchors are 2x2 red squares, and each transformed patch fits entirely inside
#   the usable interior of its target quadrant.


BG = 4
DIV = 1
RED = 2
LEGEND = 8
GRID_SIZE = 27
MID = 13
TARGET_ANCHOR_SIZE = 2
TARGET_KEYS = ((0, 1), (1, 0), (1, 1))
POWER_FACTORS = (1, 2, 4)

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# Exclude the legend-bearing border cell strip from each target quadrant.
USABLE_QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (1, 12, 14, 26),
    (1, 0): (14, 26, 1, 12),
    (1, 1): (14, 25, 14, 26),
}

SOURCE_DIMS = ((2, 3), (3, 2), (3, 3))
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
MAX_RETRIES = 500


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


def _scale_integer(img: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    assert factor in POWER_FACTORS, "scale factor must be a supported power of two"
    if factor == 1:
        return img.copy()
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _read_legends(g: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """Decode the three border legend bars into target quadrant scale factors."""
    h, w = g.shape
    legend_mask = np.zeros_like(g, dtype=bool)
    factors: dict[tuple[int, int], int] = {}
    for pts in components_mask(g == LEGEND, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        touches = [r0 == 0, c0 == 0, r1 == h - 1]
        assert sum(touches) == 1, "legend bars must touch exactly one designated outer edge"
        assert r0 == r1 or c0 == c1, "legend bars must be straight"
        factor = max(r1 - r0 + 1, c1 - c0 + 1)
        assert factor in POWER_FACTORS, "legend length must encode 1, 2, or 4"
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if r0 == 0:
            key = (0, 1)
        elif c0 == 0:
            key = (1, 0)
        else:
            key = (1, 1)
        assert key not in factors, "each target quadrant must have exactly one legend bar"
        factors[key] = int(factor)
    assert set(factors) == set(TARGET_KEYS), "all target quadrants must have legend bars"
    return factors, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants using edge legends and discrete doubling scales.

    Input structure:
    - A single-color full row and full column split the board into four quadrants.
    - Each quadrant contains one red 2x2 anchor; only the TL quadrant also contains the
      source motif attached to that anchor.
    - Three teal legend bars sit on the outer border: top edge for TR, left edge for BL,
      and bottom edge for BR. Their lengths encode the target scale.

    Rule:
    - Extract the TL non-background component containing red as the source patch.
    - Reflect it horizontally for right quadrants and vertically for bottom quadrants.
    - Read the legend length for each target quadrant, enlarge the reflected patch by that
      factor, and stamp only its non-red pixels around the fixed target anchor.

    Key insight: Reflection still comes from quadrant position, but magnification no longer
    comes from anchor size. The edge legend is the sole scale cue, and valid factors are the
    discrete powers of two {1, 2, 4}.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    legend_factors, legend_mask = _read_legends(g)

    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert r1 - r0 + 1 == TARGET_ANCHOR_SIZE and c1 - c0 + 1 == TARGET_ANCHOR_SIZE
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))

    assert set(anchors) == set(QUADS), "grid must contain one red anchor in each quadrant"

    tl = g[0:mid_r, 0:mid_c]
    patch_bb = None
    for pts in components_mask(tl != bg, connectivity=4):
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = tuple(map(int, bbox_coords(pts)))
            break
    assert patch_bb is not None, "top-left quadrant must contain a source component with red"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for key in TARGET_KEYS:
        qb, qr = key
        patch_t = patch
        if qr == 1:
            patch_t = np.fliplr(patch_t)
        if qb == 1:
            patch_t = np.flipud(patch_t)

        factor = legend_factors[key]
        patch_s = _scale_integer(patch_t, factor)
        if factor > 1:
            assert np.array_equal(patch_s[::factor, ::factor], patch_t), "integer legend scaling must be exact"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled source patch must still contain red anchor pixels"
        anchor_r0, anchor_c0, _anchor_r1, _anchor_c1 = anchors[key]
        place_r0 = anchor_r0 - int(rr_red.min())
        place_c0 = anchor_c0 - int(cc_red.min())

        rr_p, cc_p = np.where((patch_s != bg) & (patch_s != RED))
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if legend_mask[rr_t, cc_t]:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

    return out


def _assign_colors(non_red_cells: list[tuple[int, int]], h: int, w: int, red_cells: set[tuple[int, int]]) -> np.ndarray:
    """Color the selected filled cells while keeping the red anchor fixed."""
    patch = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        patch[r, c] = RED

    max_colors = min(MOTIF_COLORS_MAX, len(non_red_cells) + 1)
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = random.sample(COLOR_POOL, n_colors - 1)
    cells = non_red_cells[:]
    random.shuffle(cells)
    for i, (r, c) in enumerate(cells):
        patch[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return patch


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build one small asymmetric TL motif that still supports 4x scaling."""
    for _ in range(200):
        h, w = random.choice(SOURCE_DIMS)
        ar = random.randint(0, h - TARGET_ANCHOR_SIZE)
        ac = random.randint(0, w - TARGET_ANCHOR_SIZE)
        red_cells = {(ar + dr, ac + dc) for dr in range(TARGET_ANCHOR_SIZE) for dc in range(TARGET_ANCHOR_SIZE)}

        others = [(r, c) for r in range(h) for c in range(w) if (r, c) not in red_cells]
        n_extra = random.randint(2, len(others))
        filled = set(red_cells) | set(random.sample(others, n_extra))

        rows = {r for r, _ in filled}
        cols = {c for _, c in filled}
        if rows != set(range(h)) or cols != set(range(w)):
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in filled:
            mask[r, c] = True
        if len(components_mask(mask, connectivity=4)) != 1:
            continue

        patch = _assign_colors([p for p in filled if p not in red_cells], h, w, red_cells)
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        return patch, ar, ac
    return None


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, factor: int) -> np.ndarray:
    """Reflect by quadrant, then enlarge by a legend-coded power-of-two factor."""
    p = patch
    if qr == 1:
        p = np.fliplr(p)
    if qb == 1:
        p = np.flipud(p)
    return _scale_integer(p, factor)


def _choose_anchor(qb: int, qr: int, patch_s: np.ndarray) -> tuple[int, int] | None:
    """Choose a target anchor whose transformed patch fits inside the usable quadrant."""
    q_r0, q_r1, q_c0, q_c1 = USABLE_QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - TARGET_ANCHOR_SIZE + 1
    anch_c0, anch_c1 = q_c0, q_c1 - TARGET_ANCHOR_SIZE + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, legend_mask: np.ndarray) -> None:
    """Stamp non-red motif pixels while preserving divider, legend, and anchors."""
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where((patch_s != BG) & (patch_s != RED))
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < GRID_SIZE and 0 <= cc_t < GRID_SIZE):
            continue
        if rr_t == MID or cc_t == MID:
            continue
        if legend_mask[rr_t, cc_t]:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _sample_scale_map() -> dict[tuple[int, int], int]:
    """Pick a visibly varied scale triple from the power-of-two legend vocabulary."""
    patterns = [
        [1, 2, 4],
        [2, 2, 4],
        [2, 4, 4],
    ]
    vals = random.choice(patterns)
    random.shuffle(vals)
    return {key: vals[i] for i, key in enumerate(TARGET_KEYS)}


def _sample_legend_spans(scales: dict[tuple[int, int], int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Choose border positions for the three legend bars."""
    spans = {}

    tr_len = scales[(0, 1)]
    tr_c0 = random.randint(MID + 2, GRID_SIZE - tr_len - 1)
    spans[(0, 1)] = (0, tr_c0, 0, tr_c0 + tr_len - 1)

    bl_len = scales[(1, 0)]
    bl_r0 = random.randint(MID + 2, GRID_SIZE - bl_len - 1)
    spans[(1, 0)] = (bl_r0, 0, bl_r0 + bl_len - 1, 0)

    br_len = scales[(1, 1)]
    br_c0 = random.randint(MID + 2, GRID_SIZE - br_len - 1)
    spans[(1, 1)] = (GRID_SIZE - 1, br_c0, GRID_SIZE - 1, br_c0 + br_len - 1)

    return spans


def _draw_legends(grid: np.ndarray, spans: dict[tuple[int, int], tuple[int, int, int, int]]) -> np.ndarray:
    """Paint the legend bars and return their boolean mask."""
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r0, c0, r1, c1 in spans.values():
        grid[r0 : r1 + 1, c0 : c1 + 1] = LEGEND
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return legend_mask


def _gen_attempt() -> np.ndarray | None:
    """Single output-first generation attempt."""
    scales = _sample_scale_map()
    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {key: _scaled_patch(patch, key[0], key[1], scales[key]) for key in TARGET_KEYS}
    spans = _sample_legend_spans(scales)

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)

    for key, patch_s in scaled.items():
        pos = _choose_anchor(key[0], key[1], patch_s)
        if pos is None:
            return None
        anchors[key] = pos

    base = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    base[MID, :] = DIV
    base[:, MID] = DIV
    legend_mask = _draw_legends(base, spans)

    out = base.copy()
    paste(out, patch, tl_r0, tl_c0)
    for key in QUADS:
        ar, ac = anchors[key]
        paste(out, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], legend_mask)

    inp = base.copy()
    paste(inp, patch, tl_r0, tl_c0)
    for key in TARGET_KEYS:
        ar, ac = anchors[key]
        paste(inp, np.full((TARGET_ANCHOR_SIZE, TARGET_ANCHOR_SIZE), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
