# Combo variant of b10624e5 (index 45)
# combo_spec: D4/0+D6/3
# Name: 源图案缩放为最小目标大小 + 固定缩放因子
#
# Design analysis
# a) Concrete behavior and trace:
#    - Original task:
#      * the TL connected motif was kept unchanged in the output;
#      * each non-TL quadrant received a reflected copy whose scale came from
#        target_anchor_size / source_anchor_size.
#    - Fused task:
#      * every non-TL quadrant ignores its own anchor size and instead uses one
#        shared fixed 2x enlargement of the TL motif;
#      * the TL motif is no longer preserved as-is: in the output its ornament is
#        also rebuilt at that same canonical size, so the source participates in
#        the common target-sized rendering.
#    - Small trace:
#      * if the source patch is 5x4 with a 2x2 red anchor near its upper-left,
#        the old TR/BL/BR sizes could be 3x2, 5x4, 10x8 depending on anchor sizes;
#        now all three become the same reflected 10x8 patch.
#      * the old TL quadrant kept the original 5x4 ornament; now it is replaced by
#        its own 10x8 version aligned to the same anchor, so the source visibly changes too.
# b) Complexity calibration:
#    - The fusion remains readable because there is still one source motif, one
#      reflection rule, and one shared scale for all targets.
#    - We keep target anchors at several sizes so solvers must notice that anchor
#      size no longer drives target scaling, but we cap sizes to 2-4 to avoid
#      clutter from extreme mismatch against the implied 4x4 scaled anchor box.
# c) Hidden assumption review:
#    - Original code assumed scaled red pixels must coincide with target anchor size.
#      That assumption breaks under fixed target scaling because anchors now vary
#      independently. The new solver therefore treats red blocks as placement guides
#      only and stamps only the non-red ornament pixels.
#    - Original code assumed source output == source input. That no longer holds, so
#      the TL ornament must be cleared and rebuilt from the scaled patch.
#    - Reflection-by-quadrant and divider detection still hold unchanged.
# d) Placement legality:
#    - The generator chooses every anchor from the intersection of:
#      * positions where the whole fixed-scale patch fits inside its quadrant, and
#      * positions where the anchor square itself fits inside the quadrant.
#    - TL placement is chosen using the scaled patch as the constraint, which also
#      guarantees the smaller input patch fits.
# e) Visual observability:
#    - Every sample has at least one non-TL anchor whose size is not 4, so the
#      output cannot be explained by the original 2->4 dynamic ratio everywhere.
#    - The source patch is rejected if it is horizontally or vertically symmetric,
#      making the reflection rule and the source-rescaling rule visible by eye.
# f) Visual cleanliness:
#    - We keep the original 27x27 four-quadrant layout and require full in-quadrant
#      fit, so scaled motifs do not spill across dividers or over each other.
#    - Only one ornament exists per quadrant, and red anchors remain solid blocks,
#      which keeps the objects visually separable.
# g) Change inferability:
#    - No color remapping is introduced. The only new inference burden is geometric:
#      "all targets use the same 2x size" plus "the TL ornament is also rebuilt at that size".
# h) Generator-side new assumptions:
#    - There is exactly one red square anchor per quadrant and exactly one TL
#      connected non-background component containing that anchor.
#    - The TL motif has at least one non-red pixel outside the anchor and is not
#      mirror-symmetric, so rebuilding it at 2x always changes the visible output.

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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

SOURCE_ANCHOR_SIZE = 2
FIXED_SCALE = 2
TARGET_SIZE_CHOICES = (2, 3, 4)

BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 1, 3, 5, 6, 7, 8, 9]
MAX_RETRIES = 400


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches: list[tuple[int, int]] = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching full-row/full-col divider"
    return matches[0]


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect_patch(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    out = patch
    if qr:
        out = np.fliplr(out)
    if qb:
        out = np.flipud(out)
    return out


def _extract_source_patch(tl: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    patch_pts = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            patch_pts = pts
            break
    assert patch_bb is not None and patch_pts is not None, "top-left quadrant must contain a non-bg component with red"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy(), patch_pts, patch_bb


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int, int]]:
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"
    anchors: dict[tuple[int, int], tuple[int, int, int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        assert r1 - r0 == c1 - c0, "anchors must be square"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1), int(r1 - r0 + 1))
    assert len(anchors) == 4 and (0, 0) in anchors, "grid must contain one anchor per quadrant"
    return anchors


def _stamp_nonred(out: np.ndarray, patch_s: np.ndarray, anchor_r0: int, anchor_c0: int, mid_r: int, mid_c: int, bg: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must still contain red anchor pixels"
    place_r0 = anchor_r0 - int(rr_red.min())
    place_c0 = anchor_c0 - int(cc_red.min())

    rr_p, cc_p = np.where((patch_s != bg) & (patch_s != RED))
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < out.shape[0] and 0 <= cc_t < out.shape[1]):
            continue
        if rr_t == mid_r or cc_t == mid_c:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _scaled_patch_for_quadrant(patch: np.ndarray, qb: int, qr: int) -> np.ndarray:
    reflected = _reflect_patch(patch, qb, qr)
    return _resize_nn(reflected, reflected.shape[0] * FIXED_SCALE, reflected.shape[1] * FIXED_SCALE)


def _build_output_from_input(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    out = g.copy()

    anchors = _find_anchors(g, mid_r, mid_c)
    tl = g[0:mid_r, 0:mid_c]
    patch, patch_pts, _patch_bb = _extract_source_patch(tl, bg)

    # Source ornament is rebuilt instead of preserved.
    for r, c in patch_pts.tolist():
        if out[r, c] != RED:
            out[r, c] = bg

    for (qb, qr), (r0, c0, _r1, _c1, _size) in anchors.items():
        patch_s = _scaled_patch_for_quadrant(patch, qb, qr)
        _stamp_nonred(out, patch_s, r0, c0, mid_r, mid_c, bg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild one reflected motif per quadrant, but make every rendered motif use the same enlarged size.

    Input structure:
    - The grid is split into four 13x13 quadrants by one full divider row and one full divider column.
    - Each quadrant contains exactly one red square anchor block.
    - Only the top-left quadrant contains an extra connected colored motif attached to its anchor.

    Rule:
    - Extract the top-left connected motif that contains the red anchor.
    - Reflect that motif left-right for right quadrants and up-down for bottom quadrants.
    - Enlarge every reflected copy by one fixed 2x factor, regardless of the target anchor size.
    - Also rebuild the top-left motif at that same enlarged size.
    - Use each quadrant's red block only as the placement reference: align the enlarged motif's red
      sub-bbox to the anchor's top-left corner, keep the anchor itself unchanged, and stamp only the
      non-red motif pixels while respecting the divider.

    Key insight:
    - Quadrant position still determines reflection, but anchor size no longer determines scaling.
    - The shared enlarged size is visible because all non-TL motifs have equal extent, and the source
      ornament itself changes to match that common size instead of staying in its input form.
    """
    return _build_output_from_input(grid)


def _grow_4conn(h: int, w: int, seeds: set[tuple[int, int]], n_fill: int) -> set[tuple[int, int]]:
    filled = set(seeds)
    frontier: set[tuple[int, int]] = set()
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
        patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
        return patch, ar * SOURCE_ANCHOR_SIZE, ac * SOURCE_ANCHOR_SIZE

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar = random.randint(0, h - SOURCE_ANCHOR_SIZE)
    ac = random.randint(0, w - SOURCE_ANCHOR_SIZE)
    red_cells = {(ar + dr, ac + dc) for dr in range(SOURCE_ANCHOR_SIZE) for dc in range(SOURCE_ANCHOR_SIZE)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _patch_is_observable(patch: np.ndarray) -> bool:
    if np.array_equal(patch, np.fliplr(patch)):
        return False
    if np.array_equal(patch, np.flipud(patch)):
        return False
    rr, cc = np.where((patch != BG) & (patch != RED))
    return rr.size >= 2 and len(set(map(int, patch[patch != BG].flatten()))) >= 3


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int] | None:
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
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _original_style_output(inp: np.ndarray) -> np.ndarray:
    g = np.asarray(inp, dtype=int)
    out = g.copy()
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    anchors = _find_anchors(g, mid_r, mid_c)
    tl = g[0:mid_r, 0:mid_c]
    patch, _pts, _bb = _extract_source_patch(tl, bg)
    sL = anchors[(0, 0)][4]

    for (qb, qr), (r0, c0, _r1, _c1, tL) in anchors.items():
        if (qb, qr) == (0, 0):
            continue
        p = _reflect_patch(patch, qb, qr)
        new_h = max(1, int(round(p.shape[0] * tL / sL)))
        new_w = max(1, int(round(p.shape[1] * tL / sL)))
        patch_s = _resize_nn(p, new_h, new_w)
        rr_red, cc_red = np.where(patch_s == RED)
        place_r0 = r0 - int(rr_red.min())
        place_c0 = c0 - int(cc_red.min())
        rr_p, cc_p = np.where(patch_s != bg)
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < GRID_SIZE and 0 <= cc_t < GRID_SIZE):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])
    return out


def _gen_attempt() -> np.ndarray | None:
    patch_data = _source_patch()
    if patch_data is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = patch_data
    if not _patch_is_observable(patch):
        return None

    target_sizes = {
        (0, 1): random.choice(TARGET_SIZE_CHOICES),
        (1, 0): random.choice(TARGET_SIZE_CHOICES),
        (1, 1): random.choice(TARGET_SIZE_CHOICES),
    }
    if all(size == SOURCE_ANCHOR_SIZE * FIXED_SCALE for size in target_sizes.values()):
        return None

    scaled = {(qb, qr): _scaled_patch_for_quadrant(patch, qb, qr) for qb, qr in QUADS}

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_anchor = _choose_anchor(0, 0, SOURCE_ANCHOR_SIZE, scaled[(0, 0)])
    if tl_anchor is None:
        return None
    anchors[(0, 0)] = tl_anchor

    for key, size in target_sizes.items():
        pos = _choose_anchor(key[0], key[1], size, scaled[key])
        if pos is None:
            return None
        anchors[key] = pos

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV

    tl_anchor_r0, tl_anchor_c0 = anchors[(0, 0)]
    tl_patch_r0 = tl_anchor_r0 - anchor_r_in_patch
    tl_patch_c0 = tl_anchor_c0 - anchor_c_in_patch
    paste(inp, patch, tl_patch_r0, tl_patch_c0)

    paste(inp, np.full((SOURCE_ANCHOR_SIZE, SOURCE_ANCHOR_SIZE), RED, dtype=int), tl_anchor_r0, tl_anchor_c0)
    for key, size in target_sizes.items():
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)

    out = _build_output_from_input(inp)
    if not np.array_equal(main(inp), out):
        return None

    # Anti-degeneracy checks: the combo must differ from the original rule.
    if np.array_equal(out, _original_style_output(inp)):
        return None

    inp_tl = inp[0:MID, 0:MID]
    out_tl = out[0:MID, 0:MID]
    if np.array_equal(inp_tl, out_tl):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
