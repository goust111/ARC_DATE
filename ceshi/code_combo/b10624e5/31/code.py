# Combo variant of b10624e5 (index 31)
# combo_spec: D4/2+D11/1
# Name: 按锚块尺寸排序处理 + 缩小而非放大

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the fused variant
# a) Concrete behavior:
#    The original task copies the TL motif to the other quadrants by scaling each copy
#    according to that quadrant's red anchor size. Here the source anchor is enlarged to
#    4x4 and the targets are a permutation of 1x1, 2x2, 3x3, so every non-TL copy is a
#    strict shrink. Example: a logical 3x2 motif repeated into 4x4 blocks becomes a 12x8
#    TL patch; shrinking it to target sizes 1,2,3 yields 3x2, 6x4, 9x6 copies. The
#    smallest anchor therefore gets the coarsest copy, then the medium one, then the
#    largest one.
# b) Complexity calibration:
#    The three target sizes are fixed to {1,2,3} but assigned to quadrants randomly. This
#    keeps the rule visible without introducing too many cases: humans can see a clear
#    small/medium/large progression, yet there is no fixed quadrant->scale mapping.
# c) Broken implicit assumptions from the seed:
#    - The seed assumed the TL anchor was 2x2 and many targets enlarged the motif.
#      That no longer holds; the solver must accept a larger source anchor and ratios < 1.
#    - The seed relied on lossless roundtrip resize checks. Downscaling is not generally
#      invertible, so the generator instead constructs block-constant source motifs that
#      downsample cleanly by design.
#    - The seed processed targets independently. This variant defines an explicit
#      small-to-large processing order. Because quadrants are disjoint, order does not
#      create overwrite effects, so the generator makes the order visually observable by
#      randomizing which quadrant holds size 1, 2, or 3.
# d) Placement legality:
#    Every full source/target patch is kept inside its own quadrant. The divider is never
#    overwritten, anchors do not overlap, and each target patch is aligned so its red
#    sub-square lands exactly on the target anchor.
# e) Visual observability:
#    Across examples, the same source motif appears as three reflected shrunken copies of
#    different granularities. Since the quadrant carrying the smallest/medium/largest
#    anchor changes from sample to sample, the solver cannot use a fixed quadrant rule.
# f) Cleanliness:
#    Divider color is excluded from motif colors to avoid confusing motif pixels with the
#    quadrant split. Motifs are built from one connected component and fit comfortably
#    inside each quadrant, so outputs remain legible.
# g) Inferability:
#    No hidden numeric ordering such as color ids is used. The only ordering cue is the
#    visibly different anchor sizes, which directly support the "smallest first, then
#    larger" interpretation.


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

SOURCE_ANCHOR_SIZE = 4
TARGET_SIZES = (1, 2, 3)
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.55
FILL_RATIO_MAX = 0.85
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


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
    """Nearest-neighbor resize that supports shrinking and enlarging."""
    src_h, src_w = img.shape
    assert src_h > 0 and src_w > 0
    new_h = max(1, int(new_h))
    new_w = max(1, int(new_w))
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr == 1:
        out = np.fliplr(out)
    if qb == 1:
        out = np.flipud(out)
    return out


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, target_size: int, source_size: int) -> np.ndarray:
    """Reflect the source patch into the target quadrant, then resize by target/source."""
    p = _reflect_patch(patch, qb, qr)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, bg: int, mid_r: int, mid_c: int) -> None:
    """Stamp a patch aligned by the red block; preserve existing red anchor pixels."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must still contain red pixels"
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())

    rr_p, cc_p = np.where(patch_s != bg)
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + int(rr0)
        cc_t = place_c0 + int(cc0)
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _extract_source_patch(g: np.ndarray, mid_r: int, mid_c: int, bg: int) -> np.ndarray:
    tl = g[:mid_r, :mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    for pts in comps:
        if any(int(tl[r, c]) == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            return tl[r0 : r1 + 1, c0 : c1 + 1].copy()
    raise AssertionError("top-left quadrant must contain one non-bg component with the red anchor")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants as progressively less-shrunken reflected versions.

    Input structure:
    - A single full-row/full-column divider splits the board into four 13x13 quadrants.
    - Each quadrant contains one filled red square anchor; the top-left anchor is the largest source anchor.
    - Only the top-left quadrant contains the full colored motif attached to its anchor; other quadrants initially show just their red anchors.

    Rule:
    - Extract the top-left motif component as the source patch.
    - Sort the three target quadrants by red anchor size from smallest to largest.
    - For each target in that order, reflect the source patch according to quadrant position and shrink it by target_anchor_size / source_anchor_size.
    - Align the shrunken red region onto the target red square and stamp the non-background pixels, without painting over divider cells or existing red anchor pixels.

    Key insight:
    - Anchor size now carries two linked meanings: it orders the targets and it sets how much of the source detail survives the shrink.
      The smallest anchor gets the most reduced copy, and larger anchors receive progressively richer versions of the same reflected motif.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(size))

    assert set(anchors) == set(QUADS), "all four quadrants must contain a red anchor"
    source_r0, source_c0, source_size = anchors[(0, 0)]
    assert source_size > 0

    patch = _extract_source_patch(g, mid_r, mid_c, bg)

    targets = sorted(
        ((key, value) for key, value in anchors.items() if key != (0, 0)),
        key=lambda item: (int(item[1][2]), int(item[0][0]), int(item[0][1])),
    )

    for (qb, qr), (tr0, tc0, target_size) in targets:
        assert target_size <= source_size, "this variant expects every target anchor to be no larger than the source anchor"
        patch_s = _scaled_patch(patch, qb, qr, target_size, source_size)
        _stamp(out, patch_s, tr0, tc0, bg, mid_r, mid_c)

    return out


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    """Grow a 4-connected region from seeds until it reaches n_fill cells."""
    filled = set(seeds)
    frontier = set()
    for r, c in seeds:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    while len(filled) < n_fill and frontier:
        cell = random.choice(list(frontier))
        frontier.discard(cell)
        filled.add(cell)
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in filled:
                frontier.add((nr, nc))
    return filled


def _assign_logical_colors(filled: set[tuple[int, int]], red_cell: tuple[int, int], h: int, w: int) -> np.ndarray | None:
    """Color a logical motif while keeping it visually compact and row/col covering."""
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None
    non_red = [p for p in filled if p != red_cell]
    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None

    random.shuffle(non_red)
    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[: n_colors - 1]

    logical = np.full((h, w), BG, dtype=int)
    logical[red_cell] = RED
    for i, (r, c) in enumerate(non_red):
        logical[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return logical


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build a TL source patch whose 4x4 blocks downsample cleanly to 1x1/2x2/3x3 anchors."""
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    red_cell = (ar, ac)
    filled = _grow_4conn(lh, lw, {red_cell}, random.randint(min_fill, max_fill))
    logical = _assign_logical_colors(filled, red_cell, lh, lw)
    if logical is None:
        return None

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Choose an anchor position that keeps the full scaled motif inside the target quadrant."""
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
    if r_lo > r_hi or c_lo > c_hi:
        return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _gen_attempt() -> np.ndarray | None:
    """Construct one valid input by first building the fully solved output."""
    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    permuted_targets = list(TARGET_SIZES)
    random.shuffle(permuted_targets)
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): permuted_targets[0],
        (1, 0): permuted_targets[1],
        (1, 1): permuted_targets[2],
    }

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], size, SOURCE_ANCHOR_SIZE)
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

    ordered_targets = sorted(
        ((key, target_sizes[key]) for key in target_sizes if key != (0, 0)),
        key=lambda item: (int(item[1]), int(item[0][0]), int(item[0][1])),
    )
    for key, _size in ordered_targets:
        _stamp(out, scaled[key], anchors[key][0], anchors[key][1], BG, MID, MID)

    inp = out.copy()
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
