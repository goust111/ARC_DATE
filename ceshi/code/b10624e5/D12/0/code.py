# Variant of b10624e5
# Dimension: D12 — 符号编码：图例或参数标记 (symbol_instructions)
# Change: 无符号编码——缩放因子和反射轴均从空间关系隐式推导... → 网格边缘的缩放因子图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for this D12 variant
# -----------------------------------
# a) Concrete behavior and trace
#    - The four-quadrant divider and the red anchors stay, so the task still reads as
#      "copy the TL motif to the other quadrants".
#    - The change is that the target anchors no longer encode scale through their size.
#      All non-TL anchors are fixed 2x2 red squares. Scale is instead written on the
#      outer border as three straight legend bars of color 8:
#        * top edge bar inside TR quadrant -> TR scale
#        * left edge bar inside BL quadrant -> BL scale
#        * bottom edge bar inside BR quadrant -> BR scale
#      The bar length is the integer scale factor (1, 2, or 3).
#    - Small trace:
#        original task: target anchor size 4 and source anchor size 2 => ratio 2, so
#        the solver enlarges the whole patch by 2.
#        variant: target anchor is still 2x2, but the border bar next to that quadrant
#        has length 3 => enlarge the reflected patch by 3 even though the anchor size
#        does not change.
#      The old "anchor size / source anchor size" step disappears; legend lookup replaces it.
#    - Because anchor size no longer tracks scale, the anchor now acts only as the
#      placement reference. The enlarged red region inside the scaled source patch is not
#      redrawn; only the non-red motif pixels are stamped around the fixed target anchor.
#      This adaptation is necessary to make D12 visible while keeping the rest of the
#      task structure intact.
# b) Complexity calibration
#    - Only three scale values are used: 1, 2, 3. This is enough to make the legend
#      meaningful without creating a broad numeric search space.
#    - At least one target quadrant is scaled by >1, and the three legend values are not
#      all identical. That prevents regression to the original size-only behavior.
#    - Source motifs are forced to be asymmetric under both LR and UD flips so that the
#      quadrant reflection rule is visually recoverable.
# c) Hidden assumptions from the original solver that must change
#    - Original anchor loop stored (r0, c0, size) and used size to compute the scale ratio.
#      In the variant, the solver must still locate all anchors, but it cannot derive the
#      scale from their size any more.
#    - Original paste alignment assumed that the scaled red region should overlap a target
#      red block of matching size. That premise no longer holds once target anchors are
#      fixed-size. The new solver only uses the scaled red bbox's top-left corner as the
#      placement reference and skips red pixels while stamping.
#    - Original generator sampled target anchor sizes 1..4. That would accidentally
#      reintroduce a size-based cue, so the variant generator keeps non-TL anchors fixed.
# d) Placement legality
#    - Legend bars live on reserved border strips: row 0 for TR, col 0 for BL, row 26 for BR.
#    - Non-TL motif placement is sampled so the entire transformed patch fits inside the
#      remaining interior of the target quadrant, avoiding border legends, the divider,
#      overlap, and out-of-bounds writes.
#    - Input contains only the TL motif, the four anchors, and the three legend bars, so
#      the solver sees exactly one red anchor per quadrant and exactly three legend bars.
# e) Visual observability
#    - Human-visible difference from the original task: all target anchors have the same
#      size, yet the copied motifs come out at different sizes. The only remaining scale
#      cue is the border legend length, so the D12 change is explicit in the pixels.
#    - Reflection is still tied to quadrant position, but asymmetry in the source motif
#      makes those flips observable.
# f) Visual cleanliness
#    - Legends are sparse, single-color, straight bars isolated on outer edges.
#    - Target motifs are kept away from the legend edges, so symbols do not merge with
#      copied content or with the red anchors.
# g) Color-change audit
#    - No rule depends on hidden color ordering or color-id arithmetic.
#    - Legend color 8 is dedicated to the instruction symbols and is excluded from motif
#      colors so the legend remains visually distinct.
#
# New solver assumptions guaranteed by generate_input()
# - Divider color is 1, anchor color is 2, legend color is 8, background is 4.
# - There are exactly three border-touching legend components of color 8, each a 1-cell
#   thick straight segment whose length is the integer scale factor for one target quadrant.
# - The TL patch is the only non-background connected component in TL that contains red.
# - Every non-TL anchor is a 2x2 red square, and the transformed motif fits inside the
#   target quadrant interior that excludes the legend-bearing edge.


BG = 4
DIV = 1
RED = 2
LEGEND = 8
GRID_SIZE = 27
MID = 13
TARGET_ANCHOR_SIZE = 2
MAX_SCALE = 3
TARGET_KEYS = ((0, 1), (1, 0), (1, 1))

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# Reserve the edge used by each legend so copied motifs stay visually separate.
USABLE_QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (1, 12, 14, 26),
    (1, 0): (14, 26, 1, 12),
    (1, 1): (14, 25, 14, 26),
}

FREEFORM_H_RANGE = (3, 4)
FREEFORM_W_RANGE = (3, 4)
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.85
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
MAX_RETRIES = 400


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
    assert factor >= 1
    if factor == 1:
        return img.copy()
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _read_legends(g: np.ndarray) -> tuple[dict[tuple[int, int], int], np.ndarray]:
    """Decode the three border legend bars into quadrant->scale."""
    h, w = g.shape
    legend_mask = np.zeros_like(g, dtype=bool)
    factors: dict[tuple[int, int], int] = {}
    for pts in components_mask(g == LEGEND, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        touches = [r0 == 0, c0 == 0, r1 == h - 1]
        assert sum(touches) == 1, "legend bars must touch exactly one designated outer edge"
        assert r0 == r1 or c0 == c1, "legend bars must be straight"
        factor = max(r1 - r0 + 1, c1 - c0 + 1)
        assert 1 <= factor <= MAX_SCALE, "legend length must encode a valid scale"
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        if r0 == 0:
            key = (0, 1)
        elif c0 == 0:
            key = (1, 0)
        else:
            key = (1, 1)
        assert key not in factors, "each target quadrant must have exactly one legend bar"
        factors[key] = factor
    assert set(factors) == set(TARGET_KEYS), "all target quadrants must have legend bars"
    return factors, legend_mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants using border legends for scale.

    Input structure:
    - A single-color full row and full column split the grid into four quadrants.
    - Each quadrant contains one red (2) square anchor; the TL quadrant also contains the
      source motif attached to that anchor.
    - Three straight legend bars of color 8 sit on the outer border: top edge for TR,
      left edge for BL, and bottom edge for BR.

    Rule:
    - Take the TL non-background component containing the red anchor as the source patch.
    - Reflect that patch horizontally for right quadrants and vertically for bottom
      quadrants, exactly as in the original task.
    - Read the target scale factor from the matching border legend length, enlarge the
      reflected patch by that integer factor, and place its non-red pixels relative to the
      target anchor.

    Key insight: Quadrant position still determines reflection, but motif size is no
    longer encoded by anchor size; it is written explicitly by the edge legend bars.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    legend_factors, legend_mask = _read_legends(g)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

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

    assert (0, 0) in anchors, "top-left quadrant must contain the source anchor"

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
        assert key in anchors, "each target quadrant must contain a red anchor"
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
        assert rr_red.size > 0, "scaled source patch must still contain the red anchor footprint"
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
    """Assign random colors to filled cells and keep the bbox fully used."""
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
    patch = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        patch[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        patch[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return patch


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build one asymmetric TL motif patch with a 2x2 red anchor."""
    for _ in range(80):
        h = random.randint(*FREEFORM_H_RANGE)
        w = random.randint(*FREEFORM_W_RANGE)
        ar = random.randint(0, h - TARGET_ANCHOR_SIZE)
        ac = random.randint(0, w - TARGET_ANCHOR_SIZE)
        red_cells = {(ar + dr, ac + dc) for dr in range(TARGET_ANCHOR_SIZE) for dc in range(TARGET_ANCHOR_SIZE)}
        min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
        max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
        if max_fill < min_fill:
            continue
        filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
        patch = _assign_colors(filled, red_cells, h, w)
        if patch is None:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        return patch, ar, ac
    return None


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, factor: int) -> np.ndarray:
    """Reflect by quadrant, then scale by the legend factor."""
    p = patch
    if qr == 1:
        p = np.fliplr(p)
    if qb == 1:
        p = np.flipud(p)
    return _scale_integer(p, factor)


def _choose_anchor(qb: int, qr: int, patch_s: np.ndarray) -> tuple[int, int] | None:
    """Choose a target anchor whose full transformed patch fits the usable quadrant interior."""
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
    """Stamp non-red motif pixels, preserving divider, legends, and anchors."""
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
    """Pick target scale factors with visible variation and at least one enlargement."""
    patterns = [
        [1, 1, 2],
        [1, 2, 2],
        [1, 3, 3],
        [2, 2, 3],
        [2, 3, 3],
        [1, 2, 3],
    ]
    vals = random.choice(patterns)
    random.shuffle(vals)
    return {key: vals[i] for i, key in enumerate(TARGET_KEYS)}


def _sample_legend_spans(scales: dict[tuple[int, int], int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Choose border bar positions for the three legend segments."""
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
    for key in TARGET_KEYS:
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
    raise RuntimeError("failed to generate a valid b10624e5 D12 input")
