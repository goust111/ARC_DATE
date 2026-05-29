# Combo variant of b10624e5 (index 37)
# combo_spec: D9/3+D16/0
# Name: 分散多色块锚块 + 象限面积比缩放

import random
import sys

import numpy as np

if "/home/lixinyu/snap/new/solvers" not in sys.path:
    sys.path.append("/home/lixinyu/snap/new/solvers")

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The divider is off-center as in D16/0, so the four quadrant areas differ. We keep the same
#      near-center asymmetric split family as the single-dimension reference, which guarantees that
#      within each sample at least one target copy shrinks and at least one target copy enlarges.
#    - The red anchor is no longer a single square. Each quadrant now has one dispersed anchor group:
#      four isolated red pixels arranged inside a 5x5 anchor bbox. In TL those red pixels are embedded
#      inside one colored connected component; in the other quadrants the input shows only those red
#      pixels, so the solver must aggregate multiple red pieces into one logical anchor.
#    - Small trace: if the split is 11x15 for TL, then TR/BL/BR areas are 121/225/165 while TL is 165.
#      All non-TL inputs still show the same 5x5 dispersed red marker shape (up to reflection), so any
#      old "read scale from anchor size" rule predicts ratio 1 for every target. The combo rule instead
#      shrinks TR, enlarges BL, and keeps BR unchanged because the scale comes from quadrant area alone.
# b) Complexity calibration:
#    - The dispersed anchor pattern is fixed and asymmetric. That keeps grouping non-ambiguous while
#      making the anchor visibly different from the original filled square.
#    - The divider positions are restricted to a small family of off-center choices. This keeps the
#      area ratios moderate while still giving repeated distinct scales inside each sample.
#    - Source motifs stay compact (6-8 cells per side before scaling), so the output remains legible.
# c) Implicit-assumption audit from the original solver:
#    - Old code assumed one connected red component per quadrant. That fails here because each anchor is
#      split into four components. The new solver groups all red pixels by quadrant and uses their union
#      bbox as the anchor reference.
#    - Old code used red square side-length ratios. That assumption is intentionally broken by keeping
#      the non-TL anchors at the same dispersed bbox size even when the copied motif shrinks or grows.
#    - Old code required lossless roundtrip resizing. Area ratios like 121/165 are fractional, so that
#      assertion must be removed.
#    - Alignment now depends on the red union bbox top-left rather than a single red block's top-left.
# d) Placement legality:
#    - The generator first builds the output, then erases non-TL motif pixels to form the input.
#    - Non-TL anchor bboxes are placed so the full reflected/scaled motif preferably fits inside the
#      quadrant; if not, the anchor still fits and clipping is handled by the stamping logic.
#    - The source patch always fits fully inside TL, and only one TL non-background component contains red.
# e) Visual observability:
#    - Reflection remains tied to quadrant position, and the source motif is forced to be asymmetric under
#      both LR and UD flips so the mirrored relation stays visible.
#    - Because all non-TL inputs have the same dispersed anchor size but different quadrant areas, a human
#      can see that area, not anchor extent, controls the motif size.
# f) Visual cleanliness:
#    - Divider color is excluded from the motif palette so the off-center split stays easy to read.
#    - The anchor uses only four red pixels, which is enough to read "one grouped marker" without letting
#      unscaled input anchors visually dominate the copied motif.
# g) Color inference:
#    - Colors are copied directly from TL. No color remapping is introduced, so the only new concepts are
#      "group the red dots per quadrant" and "scale by quadrant area".
# h) New solver assumptions and generator guarantees:
#    - `main()` assumes every red pixel outside the divider belongs to the quadrant anchor for that quadrant.
#      The generator never introduces extra red pixels outside the motif-anchor system.
#    - `main()` assumes the TL source motif is the unique non-background component in TL containing red.
#    - `main()` assumes each scaled reflected target patch still contains at least one red pixel. The fixed
#      dispersed anchor, compact source patch sizes, and restricted divider ratios guarantee that.


BG = 4
DIV = 1
RED = 2
GRID_SIZE = 27

SMALL_DIVS = (11, 12)
LARGE_DIVS = (14, 15)

ANCHOR_H = 5
ANCHOR_W = 5
BASE_RED_POINTS = ((0, 0), (0, 3), (2, 1), (4, 4))

PATCH_H_RANGE = (6, 8)
PATCH_W_RANGE = (6, 8)
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.75
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 8, 9]

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


def _anchor_mask(qb: int, qr: int) -> np.ndarray:
    mask = np.full((ANCHOR_H, ANCHOR_W), BG, dtype=int)
    for r, c in BASE_RED_POINTS:
        mask[r, c] = RED
    if qr:
        mask = np.fliplr(mask)
    if qb:
        mask = np.flipud(mask)
    return mask


def _red_bbox_topleft(arr: np.ndarray) -> tuple[int, int]:
    rr, cc = np.where(arr == RED)
    assert rr.size > 0, "array must contain red anchor pixels"
    return int(rr.min()), int(cc.min())


def _collect_anchor_bboxes(
    g: np.ndarray, quads: dict[tuple[int, int], tuple[int, int, int, int]]
) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    red_coords = np.argwhere(g == RED)
    assert red_coords.size > 0, "grid must contain red (color 2) anchor pixels"
    mid_r = quads[(0, 0)][1] + 1
    mid_c = quads[(0, 0)][3] + 1

    anchors: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for r, c in red_coords.tolist():
        assert r != mid_r and c != mid_c, "red anchor pixels must not lie on the divider"
        qb = 0 if r <= quads[(0, 0)][1] else 1
        qr = 0 if c <= quads[(0, 0)][3] else 1
        key = (qb, qr)
        anchors.setdefault(key, []).append((int(r), int(c)))

    out: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for key, pts in anchors.items():
        arr = np.array(pts, dtype=int)
        out[key] = bbox_coords(arr)
    assert set(out) == set(quads), "each quadrant must contain red anchor pixels"
    return out


def _extract_source_patch(
    g: np.ndarray,
    quads: dict[tuple[int, int], tuple[int, int, int, int]],
    bg: int,
) -> np.ndarray:
    tl_r0, tl_r1, tl_c0, tl_c1 = quads[(0, 0)]
    tl = g[tl_r0 : tl_r1 + 1, tl_c0 : tl_c1 + 1]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with red anchor pixels"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr:
        out = np.fliplr(out)
    if qb:
        out = np.flipud(out)
    return out


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r0: int, anchor_c0: int, mid_r: int, mid_c: int) -> None:
    red_r0, red_c0 = _red_bbox_topleft(patch_s)
    place_r0 = anchor_r0 - red_r0
    place_c0 = anchor_c0 - red_c0

    src_rr, src_cc = np.where(patch_s != BG)
    for rr0, cc0 in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif into the other quadrants using dispersed red anchor groups and area-based scaling.

    Input structure:
    - One full row and one full column of the same non-background color split the grid into four visible
      quadrants, and the split is off-center so the quadrant areas differ.
    - Each quadrant contains one logical red anchor, but that anchor is represented by several separated
      red pixels rather than one connected square.
    - The top-left quadrant also contains one connected colored motif component that incorporates its red
      anchor pixels; the other quadrants contain only their dispersed red anchor pixels.

    Rule:
    - Extract the TL connected component containing red as the source motif.
    - For each target quadrant, reflect the source motif left-right for right quadrants and up-down for
      bottom quadrants, then resize it by `target_quadrant_area / source_quadrant_area`.
    - Group all red pixels in the target quadrant into one anchor, align the scaled motif's red union bbox
      to that target anchor bbox, and stamp non-background pixels without overwriting existing red cells.

    Key insight: the red anchor is now a grouped constellation of separated pieces, so its job is only to
    mark where the copy belongs; the actual scale must be inferred from the relative quadrant areas.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    quads = _quadrants(h, w, mid_r, mid_c)
    anchors = _collect_anchor_bboxes(g, quads)
    patch = _extract_source_patch(g, quads, bg)
    source_area = _quad_area(quads[(0, 0)])

    for key, (tr0, tc0, _tr1, _tc1) in anchors.items():
        if key == (0, 0):
            continue
        qb, qr = key
        patch_t = _reflect_patch(patch, qb, qr)
        target_area = _quad_area(quads[key])
        ratio = float(target_area) / float(source_area)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        assert np.any(patch_s == RED), "scaled source patch must contain red anchor pixels"
        _stamp(out, patch_s, tr0, tc0, mid_r, mid_c)

    return out


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


def _path(a: tuple[int, int], b: tuple[int, int]) -> set[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    cells = {(r0, c0), (r1, c1)}
    if random.random() < 0.5:
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r0, c))
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c1))
    else:
        step = 1 if r1 >= r0 else -1
        for r in range(r0, r1 + step, step):
            cells.add((r, c0))
        step = 1 if c1 >= c0 else -1
        for c in range(c0, c1 + step, step):
            cells.add((r1, c))
    return cells


def _assign_colors(filled: set[tuple[int, int]], red_cells: set[tuple[int, int]], h: int, w: int) -> np.ndarray | None:
    non_red = [p for p in filled if p not in red_cells]
    if len(non_red) < MOTIF_COLORS_MIN - 1:
        return None

    max_colors = min(MOTIF_COLORS_MAX, len(non_red) + 1)
    if max_colors < MOTIF_COLORS_MIN:
        return None

    n_colors = random.randint(MOTIF_COLORS_MIN, max_colors)
    extras = COLOR_POOL[:]
    random.shuffle(extras)
    extras = extras[: n_colors - 1]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED

    random.shuffle(non_red)
    for i, (r, c) in enumerate(non_red):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


def _motif_is_asymmetric(patch: np.ndarray) -> bool:
    return not np.array_equal(patch, np.fliplr(patch)) and not np.array_equal(patch, np.flipud(patch))


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    h = random.randint(*PATCH_H_RANGE)
    w = random.randint(*PATCH_W_RANGE)
    if h < ANCHOR_H or w < ANCHOR_W:
        return None

    anchor_r0 = random.randint(0, h - ANCHOR_H)
    anchor_c0 = random.randint(0, w - ANCHOR_W)
    red_cells = {(anchor_r0 + r, anchor_c0 + c) for r, c in BASE_RED_POINTS}

    anchor_pts = [(anchor_r0 + r, anchor_c0 + c) for r, c in BASE_RED_POINTS]
    seeds = set(red_cells)
    for a, b in zip(anchor_pts, anchor_pts[1:], strict=False):
        seeds |= _path(a, b)

    min_fill = max(len(seeds) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * h * w)))
    filled = _grow_4conn(h, w, seeds, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None

    coords = np.argwhere(patch != BG)
    r0, c0, r1, c1 = bbox_coords(coords)
    cropped = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
    red_r0, red_c0 = _red_bbox_topleft(cropped)
    if not _motif_is_asymmetric(cropped):
        return None
    return cropped, red_r0, red_c0


def _scaled_patch(
    patch: np.ndarray,
    qb: int,
    qr: int,
    source_area: int,
    target_area: int,
) -> np.ndarray:
    reflected = _reflect_patch(patch, qb, qr)
    ratio = float(target_area) / float(source_area)
    new_h = max(1, int(round(reflected.shape[0] * ratio)))
    new_w = max(1, int(round(reflected.shape[1] * ratio)))
    return _resize_nn(reflected, new_h, new_w)


def _place_mask_at_anchor(out: np.ndarray, mask: np.ndarray, anchor_r0: int, anchor_c0: int) -> None:
    red_r0, red_c0 = _red_bbox_topleft(mask)
    paste(out, mask, anchor_r0 - red_r0, anchor_c0 - red_c0)


def _choose_anchor_bbox(
    bounds: tuple[int, int, int, int],
    anchor_mask: np.ndarray,
    patch_s: np.ndarray,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = bounds
    patch_red_r0, patch_red_c0 = _red_bbox_topleft(patch_s)
    mask_red_r0, mask_red_c0 = _red_bbox_topleft(anchor_mask)

    fit_r0 = q_r0 + patch_red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + patch_red_r0
    fit_c0 = q_c0 + patch_red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + patch_red_c0

    anch_r0 = q_r0 + mask_red_r0
    anch_r1 = q_r1 - anchor_mask.shape[0] + 1 + mask_red_r0
    anch_c0 = q_c0 + mask_red_c0
    anch_c1 = q_c1 - anchor_mask.shape[1] + 1 + mask_red_c0
    if anch_r0 > anch_r1 or anch_c0 > anch_c1:
        return None

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _gen_attempt() -> np.ndarray | None:
    mid_r, mid_c = _choose_dividers()
    quads = _quadrants(GRID_SIZE, GRID_SIZE, mid_r, mid_c)

    source = _source_patch()
    if source is None:
        return None
    patch, source_anchor_r0, source_anchor_c0 = source

    tl_bounds = quads[(0, 0)]
    tl_h = tl_bounds[1] - tl_bounds[0] + 1
    tl_w = tl_bounds[3] - tl_bounds[2] + 1
    if patch.shape[0] > tl_h or patch.shape[1] > tl_w:
        return None

    source_area = _quad_area(tl_bounds)
    scaled: dict[tuple[int, int], np.ndarray] = {}
    masks: dict[tuple[int, int], np.ndarray] = {}
    for key in quads:
        if key == (0, 0):
            continue
        masks[key] = _anchor_mask(*key)
        patch_s = _scaled_patch(patch, key[0], key[1], source_area, _quad_area(quads[key]))
        if not np.any(patch_s == RED):
            return None
        scaled[key] = patch_s

    if scaled[(0, 1)].shape == _reflect_patch(patch, 0, 1).shape:
        return None
    if scaled[(1, 0)].shape == _reflect_patch(patch, 1, 0).shape:
        return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_patch_r0 = random.randint(tl_bounds[0], tl_bounds[1] - patch.shape[0] + 1)
    tl_patch_c0 = random.randint(tl_bounds[2], tl_bounds[3] - patch.shape[1] + 1)
    anchors[(0, 0)] = (tl_patch_r0 + source_anchor_r0, tl_patch_c0 + source_anchor_c0)

    for key, patch_s in scaled.items():
        pos = _choose_anchor_bbox(quads[key], masks[key], patch_s)
        if pos is None:
            return None
        anchors[key] = pos

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[mid_r, :] = DIV
    out[:, mid_c] = DIV
    paste(out, patch, tl_patch_r0, tl_patch_c0)
    for key, mask in masks.items():
        _place_mask_at_anchor(out, mask, *anchors[key])
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, *anchors[key], mid_r, mid_c)

    inp = out.copy()
    for key, mask in masks.items():
        q_r0, q_r1, q_c0, q_c1 = quads[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        _place_mask_at_anchor(inp, mask, *anchors[key])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
