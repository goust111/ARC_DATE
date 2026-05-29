# Variant of b10624e5
# Dimension: D8 — 象限结构与分区方式 (meta)
# Change: 网格被隐式分为四个象限（左上、右上、左下、右下），象限由红色锚块的位置暗示... → 三分区或六分区

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis
# a) Concrete behavior:
#    The grid is now split into six regions by one full divider row and two full divider
#    columns, making a 2x3 layout instead of 2x2 quadrants. The source motif still lives
#    in the top-left region and still contains a red square anchor. Every other region
#    contains only a red square in the input and receives a copied motif in the output.
#    The scale rule stays the same: target anchor size / source anchor size.
#    The spatial rule changes from quadrant-based reflection to divider-crossing parity:
#    crossing each vertical divider toggles a left-right flip, and crossing the single
#    horizontal divider toggles an up-down flip. Example: source at (row band 0, col band 0),
#    top-middle at (0,1) is mirrored LR once, top-right at (0,2) crosses two vertical
#    dividers so the LR flip cancels and its orientation matches the source horizontally.
#    This is the exact point where the variant diverges from the original 4-quadrant logic.
# b) Complexity calibration:
#    Six anchors make the layout visibly different from the original. To keep the rule
#    learnable, the transform family is still only {identity, LR, UD, both}; the added
#    complexity comes from having to infer it from strip crossings rather than quadrant
#    labels. The generator pairs the middle and right columns with equal anchor sizes so
#    the column-parity effect is visible without scale confounds.
# c) Old assumptions that break:
#    - `_find_dividers` can no longer search for exactly one full row and one full column.
#      It must find exactly one full row and exactly two full columns of the same color.
#    - Anchor classification can no longer use a binary left/right key; it must map into
#      three column bands.
#    - Reflection can no longer be keyed by `qr in {0,1}` alone; the far-right region needs
#      a second horizontal crossing that cancels the first.
#    - Generator clearing logic must clear five target regions rather than three quadrants.
# d) Placement legality:
#    Regions are 13x8, 13x8, 13x9 in each row. The source motif is kept narrow so its
#    largest 2x scale fits inside the narrowest region. Anchor placement prefers positions
#    where the fully transformed motif fits, so the output stays neat and avoids overlaps.
# e) Visual observability:
#    The source motif is forced to be asymmetric both horizontally and vertically; otherwise
#    the parity-based flips could become invisible. Equal-sized pairs in columns 1 and 2
#    make "one crossing flips / two crossings cancel" directly observable.
# f) Visual cleanliness:
#    Only the source region contains extra colors in the input. Other regions show a single
#    red block, and the output adds one motif per region. Full-fit placement avoids noisy
#    clipping in most cases.
# g) Color inference:
#    No color-id arithmetic is used. Colors are copied exactly; only geometry changes.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes there are exactly six red square anchors, one per region.
#      `generate_input()` constructs exactly that.
#    - `main()` assumes the source region is the top-left region and that its unique non-bg
#      component containing red is the source motif. The generator places only one such
#      component there.
#    - `main()` assumes the source patch is not invariant under LR or UD reflection.
#      The generator rejects symmetric source motifs.


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int]]:
    """Return the unique 2x3 divider layout: one full row and two full cols."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    colors = sorted({v for _, v in full_rows} & {v for _, v in full_cols})
    matches = []
    for color in colors:
        rows = sorted(r for r, v in full_rows if v == color)
        cols = sorted(c for c, v in full_cols if v == color)
        if len(rows) == 1 and len(cols) == 2:
            matches.append((int(rows[0]), (int(cols[0]), int(cols[1]))))
    assert len(matches) == 1, "grid must have exactly one matching full-row/two-full-col divider"
    return matches[0]


def _region_bounds(h: int, w: int, mid_r: int, mid_cs: tuple[int, int]) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    """Inclusive bounds for the six regions of a 2x3 layout."""
    c1, c2 = mid_cs
    row_ranges = [(0, mid_r - 1), (mid_r + 1, h - 1)]
    col_ranges = [(0, c1 - 1), (c1 + 1, c2 - 1), (c2 + 1, w - 1)]
    return {
        (rb, cb): (r0, r1, c0, c1_)
        for rb, (r0, r1) in enumerate(row_ranges)
        for cb, (c0, c1_) in enumerate(col_ranges)
    }


def _region_key(rr: int, cc: int, mid_r: int, mid_cs: tuple[int, int]) -> tuple[int, int]:
    """Map a point to its (row band, col band) region key."""
    rb = 0 if rr < mid_r else 1
    if cc < mid_cs[0]:
        cb = 0
    elif cc < mid_cs[1]:
        cb = 1
    else:
        cb = 2
    return rb, cb


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _transform_patch(patch: np.ndarray, key: tuple[int, int]) -> np.ndarray:
    """Apply divider-crossing parity transforms from the source region (0,0)."""
    rb, cb = key
    out = patch
    if cb % 2 == 1:
        out = np.fliplr(out)
    if rb % 2 == 1:
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate a motif across six regions with scaling and strip-crossing reflections keyed to red anchors.

    Input structure:
    - Grid divided into a 2x3 layout by one full-row divider and two full-column dividers of the same non-bg color.
    - Each region contains one red (2) square anchor block.
    - The top-left region also contains a colored motif attached to its red anchor.

    Rule:
    - Use the top-left region's motif as the source pattern.
    - For each other region, scale the motif by the ratio of that region's red block size to the source red block size.
    - Crossing each vertical divider toggles a horizontal reflection; crossing the horizontal divider toggles a vertical reflection.
    - Align the scaled/transformed motif so its red region overlaps the target anchor, then stamp non-bg pixels.

    Key insight: Red block size still encodes scale, but flip state now comes from divider-crossing parity in the 2x3 partition rather than from 4 quadrant identities.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_cs = _find_dividers(g, bg)
    regions = _region_bounds(h, w, mid_r, mid_cs)

    red = components_mask(g == 2, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        assert (r1 - r0) == (c1 - c0), "red anchors must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = _region_key(rr, cc, mid_r, mid_cs)
        size = r1 - r0 + 1
        assert key not in anchors, "each region must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == set(regions), "there must be exactly one red anchor in each of the six regions"
    source_key = (0, 0)
    sr0, sc0, sL = anchors[source_key]
    assert sL > 0, "top-left red anchor must have positive size"

    src_r0, src_r1, src_c0, src_c1 = regions[source_key]
    src_region = g[src_r0 : src_r1 + 1, src_c0 : src_c1 + 1]
    comps = components_mask(src_region != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(src_region[r, c] == 2 for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left region must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = src_region[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()

    for key, (tr0, tc0, tL) in anchors.items():
        if key == source_key:
            continue
        patch_t = _transform_patch(patch, key)
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == 2)
        assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
        place_r0 = tr0 - int(rr_red.min())
        place_c0 = tc0 - int(cc_red.min())

        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            val = int(patch_s[rr0, cc0])
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t in mid_cs:
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
DIV_ROW = 13
DIV_COLS = (8, 17)
REGIONS = _region_bounds(GRID_SIZE, GRID_SIZE, DIV_ROW, DIV_COLS)
SOURCE_KEY = (0, 0)
SOURCE_ANCHOR_SIZE = 2

BLOCK_DIMS = [(2, 2), (3, 2)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (4, 6)
FREEFORM_W_RANGE = (3, 4)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400


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


def _patch_is_observable(patch: np.ndarray) -> bool:
    """Require visible asymmetry so strip-parity reflections are inferable."""
    if np.array_equal(patch, np.fliplr(patch)):
        return False
    if np.array_equal(patch, np.flipud(patch)):
        return False
    return True


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """Build the source motif patch inside the narrow top-left region."""
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
        if not _patch_is_observable(patch):
            return None
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
    if patch is None or not _patch_is_observable(patch):
        return None
    return patch, ar, ac


def _scaled_patch(patch: np.ndarray, key: tuple[int, int], target_size: int) -> np.ndarray:
    """Transform by strip parity, then scale to the target anchor size."""
    p = _transform_patch(patch, key)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _choose_target_sizes() -> dict[tuple[int, int], int]:
    """Keep paired sizes across columns 1 and 2 so parity effects are directly visible."""
    while True:
        top_pair = random.randint(2, 4)
        bottom_pair = random.randint(2, 4)
        left_bottom = random.randint(*TARGET_SIZE_RANGE)
        sizes = {
            SOURCE_KEY: SOURCE_ANCHOR_SIZE,
            (0, 1): top_pair,
            (0, 2): top_pair,
            (1, 0): left_bottom,
            (1, 1): bottom_pair,
            (1, 2): bottom_pair,
        }
        if len(set(sizes.values())) >= 3 and top_pair != bottom_pair:
            return sizes


def _choose_anchor(key: tuple[int, int], size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick an anchor position for the red block within a target region."""
    q_r0, q_r1, q_c0, q_c1 = REGIONS[key]
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
    assert anch_r0 <= anch_r1 and anch_c0 <= anch_c1, "anchor must fit inside its region"
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    """Stamp non-bg pixels aligned by the red anchor while respecting the dividers."""
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
        & (tgt_rr != DIV_ROW)
        & (tgt_cc != DIV_COLS[0])
        & (tgt_cc != DIV_COLS[1])
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, derive input, oracle-check."""
    target_sizes = _choose_target_sizes()
    needs_block = any(s % SOURCE_ANCHOR_SIZE != 0 for key, s in target_sizes.items() if key != SOURCE_KEY)
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {key: _scaled_patch(patch, key, size) for key, size in target_sizes.items() if key != SOURCE_KEY}

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    src_r0, src_r1, src_c0, src_c1 = REGIONS[SOURCE_KEY]
    tl_r0 = random.randint(src_r0, src_r1 - patch.shape[0] + 1)
    tl_c0 = random.randint(src_c0, src_c1 - patch.shape[1] + 1)
    anchors[SOURCE_KEY] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key, target_sizes[key], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[DIV_ROW, :] = DIV
    out[:, DIV_COLS[0]] = DIV
    out[:, DIV_COLS[1]] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == SOURCE_KEY:
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key])

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == SOURCE_KEY:
            continue
        q_r0, q_r1, q_c0, q_c1 = REGIONS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D8 input")
