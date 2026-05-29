# Combo variant of b10624e5 (index 27)
# combo_spec: D6/2+D11/0
# Name: 源图案路径涂色轨迹 + 迭代传播
#
# Design analysis for this combo variant
# a) Concrete behavior:
#    - Base task: TL has the only full motif; TR / BR / BL have red anchors only.
#      The old solver copied TL directly into all three target quadrants.
#    - New combo: copying is no longer star-shaped. The propagation order is
#      TL -> TR, then TR -> BR, then BR -> BL. Each step uses the previous
#      step's produced patch as the next source, so later quadrants inherit any
#      coarsening caused by earlier resizes.
#    - At the same time, each propagation step leaves a thin visible trail in a
#      dedicated color from the current source anchor center to the current
#      target anchor center. The final output therefore shows three chained
#      connector paths rather than three independent TL-to-target spokes.
#    - Small example:
#      old flow: TL motif -> direct copies into TR / BR / BL.
#      combo flow:
#        1. TL motif -> reflect LR + scale to TR, draw TL->TR trail.
#        2. The produced TR patch -> reflect UD + scale to BR, draw TR->BR trail.
#        3. The produced BR patch -> reflect LR + scale to BL, draw BR->BL trail.
#      The visible difference is both geometric (later motifs depend on earlier
#      resized ones) and topological (the trails form a chain).
# b) Complexity calibration:
#    - The chain order is fixed to the quadrant ring TL, TR, BR, BL. That keeps
#      the extra reasoning dimension observable without exploding the search
#      space into arbitrary graph routing.
#    - A single dedicated trail color is reused for all three steps, so the new
#      signal is repeated several times inside one sample and is easy to notice.
#    - Generator rejects samples whose combo output matches a simple "TL copies
#      directly everywhere + TL spokes" reference, preventing silent collapse of
#      the D11 contribution.
# c) Hidden assumptions broken by the combo:
#    - The original loop assumed every target can be solved from the TL patch.
#      That is false now; BR and BL must read from patches created earlier in
#      the chain, so solver state must carry intermediate patches forward.
#    - The original roundtrip scaling assertion is intentionally removed. Chain
#      propagation is supposed to allow cumulative resizing artifacts; requiring
#      every step to be lossless would erase the D11 behavior.
#    - The old generator could derive input from `out.copy()` and clear the
#      non-TL quadrants only. Trails now cross the divider and may revisit the
#      source side, so the input must be rebuilt from scratch.
#    - Motif stamping should still skip the divider, but trail painting must be
#      allowed to pass through divider cells or the connector would break.
# d) Placement legality:
#    - Each transformed patch is required to fit fully inside its target
#      quadrant. That avoids clipping ambiguity when the copied result becomes
#      the next round's source.
#    - Trails only paint background, divider, or prior trail cells. They never
#      overwrite red anchors or motif pixels, so object identities remain clear.
# e) Visual observability:
#    - The trail color is reserved and excluded from motif generation. This
#      makes the propagation path easy to isolate by eye from only a few examples.
#    - The chained order is visible because the three paths are TL->TR, TR->BR,
#      and BR->BL, not three spokes from TL.
#    - Size changes between consecutive chain steps are encouraged so later
#      motifs visibly depend on earlier resizes instead of looking like direct
#      copies from TL.
# f) Visual cleanliness:
#    - Trails are one-cell DDA lines. They can cross each other or the divider,
#      but because they share one dedicated color and do not overwrite motifs,
#      the board stays readable.
#    - Only one motif lives in each quadrant, so the chain does not create local
#      pileups of multiple sprite copies inside the same area.
# g) Change inferability:
#    - No rule depends on hidden color ordering. Reflection depends only on the
#      source/target quadrant relation, scaling depends only on anchor sizes, and
#      the trail color has one fixed structural role.
# Additional main() assumptions and generator guarantees:
#    - main() assumes exactly one red square anchor per quadrant; generator
#      enforces that.
#    - main() assumes the TL non-background component containing red is the only
#      explicit source motif; generator constructs input that way.
#    - main() assumes intermediate target patches are fully visible and can be
#      reused as the next source; generator enforces full in-quadrant fit.
#    - main() assumes no trail pixels appear in the input; generator rebuilds
#      input from scratch so that only TL motif + red anchors are given.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


BG = 4
DIV = 1
RED = 2
TRAIL = 8
GRID_SIZE = 27
MID = 13
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}
CHAIN_EDGES = [((0, 0), (0, 1)), ((0, 1), (1, 1)), ((1, 1), (1, 0))]
STAR_EDGES = [((0, 0), (0, 1)), ((0, 0), (1, 1)), ((0, 0), (1, 0))]

SOURCE_ANCHOR_SIZE = 2
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]
TARGET_SIZE_RANGE = (1, 4)
MIN_TRAIL_CELLS = 10
MIN_CHAIN_DELTA = 8
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique full-row/full-col divider crossing."""
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
    """Nearest-neighbor resize supporting integer or fractional ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_center(r0: int, c0: int, size: int) -> tuple[int, int]:
    """Choose a stable integer endpoint inside a square anchor."""
    return int(r0 + min(size // 2, size - 1)), int(c0 + min(size // 2, size - 1))


def _line_cells(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    """Return a continuous 8-connected integer line between two cells."""
    dr = int(r1 - r0)
    dc = int(c1 - c0)
    steps = max(abs(dr), abs(dc))
    if steps == 0:
        return [(int(r0), int(c0))]
    pts: list[tuple[int, int]] = []
    last = None
    for i in range(steps + 1):
        rr = int(round(r0 + dr * i / steps))
        cc = int(round(c0 + dc * i / steps))
        cur = (rr, cc)
        if cur != last:
            pts.append(cur)
            last = cur
    return pts


def _paint_trail(
    out: np.ndarray,
    start: tuple[int, int],
    end: tuple[int, int],
    bg: int,
    divider: int,
    trail: int,
) -> None:
    """Paint a one-cell connector without overwriting motif or anchor pixels."""
    h, w = out.shape
    for rr, cc in _line_cells(start[0], start[1], end[0], end[1]):
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if int(out[rr, cc]) in (int(bg), int(divider), int(trail)):
            out[rr, cc] = int(trail)


def _parse_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Read one red square anchor per quadrant as (r0, c0, size)."""
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
    assert set(anchors) == set(QUADS), "expected one red anchor in every quadrant"
    return anchors


def _extract_tl_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    """Extract the top-left connected motif component that contains red."""
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = tuple(map(int, bbox_coords(pts)))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _scaled_patch(
    patch: np.ndarray,
    source_key: tuple[int, int],
    target_key: tuple[int, int],
    source_size: int,
    target_size: int,
) -> np.ndarray:
    """Reflect relative to source/target side changes, then resize by anchor-size ratio."""
    p = patch
    if source_key[1] != target_key[1]:
        p = np.fliplr(p)
    if source_key[0] != target_key[0]:
        p = np.flipud(p)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    out = _resize_nn(p, new_h, new_w)

    # Preserve the anchor block explicitly so every propagated patch can serve
    # as the next round's source even after aggressive downscaling.
    rr_red, cc_red = np.where(p == RED)
    assert rr_red.size > 0, "source patch must contain red anchor pixels"
    red_r0 = int(rr_red.min())
    red_c0 = int(cc_red.min())
    new_red_r0 = int(round(red_r0 * ratio))
    new_red_c0 = int(round(red_c0 * ratio))
    new_red_r0 = min(max(0, new_red_r0), max(0, new_h - target_size))
    new_red_c0 = min(max(0, new_red_c0), max(0, new_w - target_size))
    out[new_red_r0 : new_red_r0 + target_size, new_red_c0 : new_red_c0 + target_size] = RED
    return out


def _stamp(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    bg: int,
    mid_r: int,
    mid_c: int,
) -> None:
    """Stamp non-bg motif pixels, aligned so the transformed red subregion lands on the target anchor."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must contain red anchor pixels"
    place_r0 = int(anchor_r - int(rr_red.min()))
    place_c0 = int(anchor_c - int(cc_red.min()))

    src_rr, src_cc = np.where(patch_s != bg)
    tgt_rr = src_rr + place_r0
    tgt_cc = src_cc + place_c0
    valid = (
        (tgt_rr >= 0)
        & (tgt_rr < out.shape[0])
        & (tgt_cc >= 0)
        & (tgt_cc < out.shape[1])
        & (tgt_rr != mid_r)
        & (tgt_cc != mid_c)
    )
    tgt_rr = tgt_rr[valid]
    tgt_cc = tgt_cc[valid]
    src_rr = src_rr[valid]
    src_cc = src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _propagate(grid: np.ndarray, edges: list[tuple[tuple[int, int], tuple[int, int]]]) -> np.ndarray:
    """Run either chained or star propagation, depending on the provided edge order."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    divider = int(g[mid_r, mid_c])
    anchors = _parse_anchors(g, mid_r, mid_c)
    patches: dict[tuple[int, int], np.ndarray] = {(0, 0): _extract_tl_patch(g, bg, mid_r, mid_c)}

    for source_key, target_key in edges:
        assert source_key in patches, "edge order must make each source available"
        sr0, sc0, sL = anchors[source_key]
        tr0, tc0, tL = anchors[target_key]
        patch_s = _scaled_patch(patches[source_key], source_key, target_key, sL, tL)
        patches[target_key] = patch_s
        _stamp(out, patch_s, tr0, tc0, bg, mid_r, mid_c)
        _paint_trail(
            out,
            _anchor_center(sr0, sc0, sL),
            _anchor_center(tr0, tc0, tL),
            bg,
            divider,
            TRAIL,
        )
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Propagate the motif around the quadrants in a fixed chain and leave a trail for each step.

    Input structure:
    - A single monochrome divider row and divider column split the board into four 13x13 quadrants.
    - Every quadrant contains exactly one red square anchor, and only the top-left quadrant contains a full colored motif attached to its anchor.
    - The top-left motif is one connected non-background component; the other quadrants initially show only their red anchors.

    Rule:
    - First copy the top-left motif into the top-right quadrant, reflecting it according to the left-to-right move and scaling it by the TR-anchor/TL-anchor size ratio.
    - Then treat the produced top-right copy as the next source: copy it into the bottom-right quadrant using the current source-target relation, and then copy the produced bottom-right result into the bottom-left quadrant the same way.
    - After each propagation step, draw a thin trail in a dedicated color from that round's source anchor center to that round's target anchor center. Motif stamping skips the divider, but trails may pass through it.

    Key insight:
    - The output is not three independent copies from TL. Later quadrants inherit the already transformed patch from the previous quadrant, so both the motif appearance and the connector topology reveal the fixed chain TL -> TR -> BR -> BL.
    """
    return _propagate(grid, CHAIN_EDGES)


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
    """Assign random motif colors while keeping full bbox coverage."""
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


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif patch."""
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
        return patch, int(ar * s), int(ac * s)

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
    return patch, int(ar), int(ac)


def _choose_anchor(qkey: tuple[int, int], size: int, patch_s: np.ndarray) -> tuple[int, int] | None:
    """Choose an anchor position that keeps the full transformed patch inside its quadrant."""
    q_r0, q_r1, q_c0, q_c1 = QUADS[qkey]
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
    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return int(random.randint(r_lo, r_hi)), int(random.randint(c_lo, c_hi))


def _build_input(
    patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    anchors: dict[tuple[int, int], tuple[int, int]],
    sizes: dict[tuple[int, int], int],
) -> np.ndarray:
    """Rebuild the visible input from scratch."""
    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, size in sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])
    return inp


def _gen_attempt() -> np.ndarray | None:
    """Generate one valid input whose combo output is observably non-degenerate."""
    sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
    }
    if sizes[(0, 1)] == sizes[(1, 1)] == sizes[(1, 0)]:
        return None

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if patch.shape[0] > MID or patch.shape[1] > MID:
        return None

    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors: dict[tuple[int, int], tuple[int, int]] = {
        (0, 0): (int(tl_r0 + anchor_r_in_patch), int(tl_c0 + anchor_c_in_patch))
    }

    patches: dict[tuple[int, int], np.ndarray] = {(0, 0): patch}
    for source_key, target_key in CHAIN_EDGES:
        patch_s = _scaled_patch(patches[source_key], source_key, target_key, sizes[source_key], sizes[target_key])
        anchor = _choose_anchor(target_key, sizes[target_key], patch_s)
        if anchor is None:
            return None
        patches[target_key] = patch_s
        anchors[target_key] = anchor

    inp = _build_input(patch, tl_r0, tl_c0, anchors, sizes)
    combo_out = main(inp)
    star_out = _propagate(inp, STAR_EDGES)
    chain_delta = int(np.count_nonzero(combo_out != star_out))
    trail_cells = int(np.count_nonzero(combo_out == TRAIL))
    if trail_cells < MIN_TRAIL_CELLS or chain_delta < MIN_CHAIN_DELTA:
        return None
    if not np.array_equal(combo_out, _propagate(inp, CHAIN_EDGES)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input for b10624e5")
