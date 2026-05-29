# Variant of b10624e5
# Dimension: D6 — 轨迹：源图案保留策略 (trajectory)
# Change: 源图案在输出中保留不变（keep source）... → 源图案路径涂色轨迹
#
# Design analysis for this D6 variant
# a) Concrete behavior:
#    - The original task already copies the TL motif into TR / BL / BR after
#      reflection and scaling. That base behavior stays intact.
#    - The new observable change is a thin continuous trail of a dedicated color
#      (TRAIL=8) drawn from the source anchor center to each target anchor center.
#      The trail is global: it may cross quadrant boundaries and the divider.
#    - Small example:
#      old flow: TL motif -> scale/reflect -> stamp target motif only.
#      new flow: TL motif -> scale/reflect -> stamp target motif -> also draw a
#      continuous connector line from TL anchor to that target anchor.
#      The solver difference is therefore not in how motifs are transformed, but
#      in what additional pixels are painted after the transformed target exists.
# b) Complexity calibration:
#    - A single dedicated trail color is enough to make the new rule obvious in
#      2-3 ARC examples without adding a second hidden mapping problem.
#    - The trail is one-cell thick and continuous, so it is visible but does not
#      flood quadrants with many overlapping intermediate sprite placements.
# c) Hidden assumptions from the original code that break here:
#    - The original generator derived input via `inp = out.copy()` and then
#      cleared only non-TL quadrants. That was harmless before because all extra
#      output pixels lived only inside those quadrants.
#    - In this variant the trail crosses the divider and can re-enter the source
#      quadrant, so `inp = out.copy()` would leak solved pixels into the input.
#      Therefore the input must now be rebuilt from scratch.
#    - The original stamping logic skipped divider cells. That remains correct
#      for copied motifs, but not for the new trail because the trail must be
#      allowed to pass through the divider to be visually continuous.
# d) Placement legality:
#    - The copied motifs still use the original anchor-fitting placement rules.
#    - Trails only overwrite background, divider, or prior trail cells; they do
#      not overwrite red anchors or motif pixels, so object identities stay clear.
# e) Visual observability:
#    - The trail color is reserved and excluded from motif generation, so the new
#      connector is easy to isolate by eye.
#    - Every generated output has three target motifs and three connector paths,
#      so the new D6 behavior is repeatedly supported within each sample.
# f) Visual cleanliness:
#    - Using a one-pixel DDA/Bresenham-like line avoids the clutter that would
#      come from stamping whole moving sprites at every intermediate position.
#    - Crossings are acceptable because all trails share one dedicated color and
#      motifs themselves remain untouched.
# g) Color inference:
#    - No invisible color-id ordering is used.
#    - The trail color is fixed and structurally tied to the new connector role,
#      which is easy to infer from a few examples.
# Additional main() assumptions and generator guarantees:
#    - main() assumes exactly one red square anchor per quadrant; generator
#      enforces that.
#    - main() assumes the TL non-bg component containing red is the source motif;
#      generator keeps TL as a single connected component containing the anchor.
#    - main() assumes the trail color is not needed to identify the source motif;
#      generator never places trail pixels in the input.

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


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
    """Nearest-neighbor resize supporting fractional scale ratios."""
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _anchor_center(r0: int, c0: int, size: int) -> tuple[int, int]:
    """Pick a stable integer cell inside a square anchor to serve as the trail endpoint."""
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


def _paint_trail(out: np.ndarray, start: tuple[int, int], end: tuple[int, int], bg: int, divider: int, trail: int) -> None:
    """Paint a thin connector line, but do not overwrite motifs or red anchors."""
    h, w = out.shape
    for rr, cc in _line_cells(start[0], start[1], end[0], end[1]):
        if not (0 <= rr < h and 0 <= cc < w):
            continue
        if int(out[rr, cc]) in (int(bg), int(divider), int(trail)):
            out[rr, cc] = int(trail)


def _scaled_patch(patch: np.ndarray, qb: int, qr: int, source_size: int, target_size: int) -> np.ndarray:
    """Reflect and scale the source patch for a given quadrant and target anchor size."""
    p = patch
    if qr:
        p = np.fliplr(p)
    if qb:
        p = np.flipud(p)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, new_h, new_w)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, bg: int, mid_r: int, mid_c: int) -> None:
    """Stamp non-bg pixels from a scaled patch aligned by its red sub-bbox."""
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the top-left motif into the other quadrants and leave a continuous trail from the source anchor to each copy.

    Input structure:
    - A single full-row/full-column divider splits the board into four 13x13 quadrants.
    - Each quadrant contains one red square anchor block, and the top-left quadrant also contains the only full colored motif.
    - The top-left motif is one connected non-background component containing the source red anchor.

    Rule:
    - Use the top-left motif as the template; for each other quadrant, mirror it according to quadrant side and scale it by the target-anchor/source-anchor size ratio.
    - Align the transformed motif so its red part lands on the target red anchor, and copy all non-background motif pixels there.
    - After placing each target motif, draw a thin continuous trail from the source anchor center to the target anchor center using the dedicated trail color.

    Key insight:
    - The old scaling-and-reflection code still determines what each target motif looks like; the new reasoning step is recognizing that every replicated motif also implies a visible connector path back to the source anchor.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    divider = int(g[mid_r, mid_c])

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
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            patch_bb = tuple(map(int, bbox_coords(pts)))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"

    pr0, pc0, pr1, pc1 = patch_bb
    patch = tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
    source_center = _anchor_center(sr0, sc0, sL)

    target_centers: list[tuple[int, int]] = []
    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        patch_t = patch
        if qr:
            patch_t = np.fliplr(patch_t)
        if qb:
            patch_t = np.flipud(patch_t)
        patch_s = _scaled_patch(patch, qb, qr, sL, tL)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"
        _stamp(out, patch_s, tr0, tc0, bg, mid_r, mid_c)
        target_centers.append(_anchor_center(tr0, tc0, tL))

    for target_center in target_centers:
        _paint_trail(out, source_center, target_center, bg, divider, TRAIL)

    return out


# --------------- generator ---------------

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
MIN_TRAIL_CELLS = 8
MAX_RETRIES = 300


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


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
    """Assign random colors to filled cells; returns None if bbox coverage is incomplete."""
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


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    """Build the TL motif patch; block mode is forced when fractional target ratios appear."""
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


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
    """Pick a random anchor position, preferring full patch fit inside the quadrant."""
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


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build output, rebuild input, oracle-check."""
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    needs_block = any(size % SOURCE_ANCHOR_SIZE != 0 for key, size in target_sizes.items() if key != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    scaled = {
        key: _scaled_patch(patch, key[0], key[1], SOURCE_ANCHOR_SIZE, size)
        for key, size in target_sizes.items()
        if key != (0, 0)
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (int(tl_r0 + anchor_r_in_patch), int(tl_c0 + anchor_c_in_patch))
    for key, patch_s in scaled.items():
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], patch_s)

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(out, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], BG, MID, MID)

    source_center = _anchor_center(anchors[(0, 0)][0], anchors[(0, 0)][1], SOURCE_ANCHOR_SIZE)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        target_center = _anchor_center(anchors[key][0], anchors[key][1], size)
        _paint_trail(out, source_center, target_center, BG, DIV, TRAIL)

    if int(np.count_nonzero(out == TRAIL)) < MIN_TRAIL_CELLS:
        return None

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    paste(inp, patch, tl_r0, tl_c0)
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        paste(inp, np.full((size, size), RED, dtype=int), anchors[key][0], anchors[key][1])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D6 variant input for b10624e5")
