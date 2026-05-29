# Variant of b10624e5
# Dimension: D11 — 多步处理顺序 (multi_step)
# Change: 四步线性流程：识别锚块 -> 计算缩放因子 -> 确定反射轴 -> 复制缩放反射后的图案... → 双重源：两个象限各提供不同方面的信息

import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
_SOLVERS = os.path.join(_ROOT, "solvers")
if _SOLVERS not in sys.path:
    sys.path.append(_SOLVERS)

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the D11 variant
# a) Concrete behavior:
#    - The divider and red square anchors stay unchanged.
#    - TL contains a single connected source component whose non-red pixels all use one
#      "shape-only" color. It shows which cells belong to the motif and which cells are holes.
#    - TR contains another connected source component with the same bbox and the same red-anchor
#      placement, but it is a full color template and is shown in right-quadrant orientation
#      (so it is mirrored left-right relative to TL orientation).
#    - BL/BR contain only red anchors in the input. In the output, the solver first extracts TL
#      shape, then extracts TR colors, flips the TR patch back into TL orientation, masks the
#      colors by the TL silhouette, and only then performs the old scale+reflect copy step.
#    - Small example at the changed stage:
#      original logic: TL patch already contains both mask and colors -> directly scale/reflect.
#      new logic: TL says "keep cells {(0,0),(0,1),(1,0)}", TR says "these positions are
#      [red, green; blue, orange] after undoing the right-side mirror" -> merged canonical patch
#      becomes [red, green; blue, bg] before any target-quadrant reflection/scaling.
# b) Complexity calibration:
#    - Only two target quadrants are filled, but the rule now needs two coordinated sources,
#      source normalization, source fusion, then target transforms. That keeps the variant visibly
#      distinct without introducing many unrelated branches.
#    - The generated motifs use 2x-upsampled logical cells, so odd/even target anchor sizes still
#      produce stable nearest-neighbor scaling and preserve visual structure.
# c) Original assumptions that break under the variant:
#    - "TL alone contains all motif information" is no longer valid, so the solver cannot take one
#      TL component and replicate it. It must read two source components and align them by red bbox.
#    - "Every non-TL quadrant is a target" is no longer valid; TR is now a source and must not be
#      overwritten during synthesis.
#    - "Colors in the TL patch are semantically meaningful" is no longer valid; TL colors are now
#      intentionally dummy shape markers and must be ignored except for bg/non-bg structure.
# d) Placement legality:
#    - TL and TR source patches are fully placed inside their quadrants and never touch the divider.
#    - BL/BR anchors are chosen with preference for full-fit placement of the scaled merged motif.
#    - The merged patch uses the same 2x2 source anchor in both top quadrants, so source alignment
#      is unambiguous and the solver's bbox checks are guaranteed by construction.
# e) Visual observability:
#    - TL is a silhouette with holes; TR is a colored rectangle-like template; bottom outputs are
#      colored according to TR but clipped by TL. This is visible from a few examples without using
#      hidden metadata such as color ids for sorting or case selection.
#    - The generator rejects horizontally or vertically symmetric merged motifs so the reflections
#      in BL versus BR remain observable.
# f) Visual cleanliness:
#    - Each quadrant contains one main object around one red anchor. There are no extra legend
#      markers, no overlaps between independent objects, and no multi-layer clutter in the same area.
# g) Color inference:
#    - Color transfer is purely positional inside the shared source bbox. The number of colors is
#      kept small, and the top-right source directly exposes the palette layout that reappears in
#      the bottom outputs after masking and reflection.


BG = 4
DIV = 1
RED = 2
SHAPE_COLOR = 8
GRID_SIZE = 27
MID = 13
SOURCE_ANCHOR_SIZE = 2
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
COLOR_POOL = [0, 3, 5, 6, 7, 9]
QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}


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
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
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


def _red_signature(patch: np.ndarray) -> tuple[int, int, int, int]:
    rr, cc = np.where(patch == RED)
    assert rr.size > 0, "patch must contain a red anchor"
    return int(rr.min()), int(cc.min()), int(rr.max() - rr.min() + 1), int(cc.max() - cc.min() + 1)


def _extract_patch(quadrant: np.ndarray, bg: int) -> np.ndarray:
    comps = components_mask(quadrant != bg, connectivity=4)
    for pts in comps:
        if any(quadrant[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            return quadrant[r0 : r1 + 1, c0 : c1 + 1].copy()
    raise AssertionError("quadrant must contain a non-bg component with a red anchor")


def _merge_sources(shape_patch: np.ndarray, color_patch_right: np.ndarray, bg: int) -> np.ndarray:
    color_patch = np.fliplr(color_patch_right)
    assert shape_patch.shape == color_patch.shape, "shape and color sources must share a bbox"
    assert _red_signature(shape_patch) == _red_signature(color_patch), "source anchors must align"
    merged = color_patch.copy()
    merged[shape_patch == bg] = bg
    return merged


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, mid_r: int, mid_c: int, bg: int) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must contain red pixels"
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where(patch_s != bg)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine a shape source and a color source, then copy the merged motif to the bottom quadrants.

    Input structure:
    - A single-color full row and full column split the grid into four quadrants.
    - Every quadrant contains one red square anchor block.
    - The top-left quadrant contains a silhouette-like source component; the top-right quadrant
      contains a second source component carrying the color layout in mirrored right-side orientation.
    - The bottom quadrants contain only their red anchors in the input.

    Rule:
    - Read the top-left component as the canonical shape mask and ignore its non-red color identity.
    - Read the top-right component as the color template, flip it horizontally back into left-side
      orientation, and mask it by the top-left shape to build one merged canonical motif.
    - For each bottom quadrant, reflect the merged motif according to quadrant position, scale it by
      the ratio between that quadrant's red anchor size and the shared 2x2 source anchor size, align
      the scaled red region to the target anchor, and stamp non-background pixels.

    Key insight:
    - The task no longer has a single self-contained source. Shape comes from TL, colors come from TR,
      and only after those two streams are fused can the old scale/reflect transfer step be applied.
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
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size_h = r1 - r0 + 1
        size_w = c1 - c0 + 1
        assert size_h == size_w, "anchors must be square"
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size_h))

    assert (0, 0) in anchors and (0, 1) in anchors and (1, 0) in anchors and (1, 1) in anchors
    sL = anchors[(0, 0)][2]
    sR = anchors[(0, 1)][2]
    assert sL == sR and sL > 0, "top sources must use the same anchor size"

    tl = g[0:mid_r, 0:mid_c]
    tr = g[0:mid_r, mid_c + 1 :]
    shape_patch = _extract_patch(tl, bg)
    color_patch_right = _extract_patch(tr, bg)
    merged = _merge_sources(shape_patch, color_patch_right, bg)

    for (qb, qr), (tr0, tc0, tL) in anchors.items():
        if qb == 0:
            continue
        patch_t = _reflect_patch(merged, qb, qr)
        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve motif information"
        _stamp(out, patch_s, tr0, tc0, mid_r, mid_c, bg)

    return out


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


def _build_logical_shape() -> tuple[int, int, set[tuple[int, int]], tuple[int, int]] | None:
    lh, lw = random.choice(BLOCK_DIMS)
    n_cells = lh * lw
    min_fill = max(3, int(np.ceil(FILL_RATIO_MIN * n_cells)))
    max_fill = max(min_fill, int(np.floor(FILL_RATIO_MAX * n_cells)))
    ar, ac = random.randrange(lh), random.randrange(lw)
    filled = _grow_4conn(lh, lw, {(ar, ac)}, random.randint(min_fill, max_fill))
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(lh)) or cols != set(range(lw)):
        return None
    if len(filled) < 3:
        return None
    return lh, lw, filled, (ar, ac)


def _logical_to_shape_patch(lh: int, lw: int, filled: set[tuple[int, int]], anchor: tuple[int, int]) -> np.ndarray:
    patch = np.full((lh * SOURCE_ANCHOR_SIZE, lw * SOURCE_ANCHOR_SIZE), BG, dtype=int)
    for r, c in filled:
        val = RED if (r, c) == anchor else SHAPE_COLOR
        patch[
            r * SOURCE_ANCHOR_SIZE : (r + 1) * SOURCE_ANCHOR_SIZE,
            c * SOURCE_ANCHOR_SIZE : (c + 1) * SOURCE_ANCHOR_SIZE,
        ] = val
    return patch


def _build_color_template(lh: int, lw: int, anchor: tuple[int, int], filled: set[tuple[int, int]]) -> np.ndarray | None:
    logical = np.empty((lh, lw), dtype=int)
    logical.fill(-1)
    n_extra = random.randint(2, min(4, len(COLOR_POOL)))
    extras = random.sample(COLOR_POOL, n_extra)
    for r in range(lh):
        for c in range(lw):
            logical[r, c] = random.choice(extras)
    logical[anchor] = RED

    # The final merged patch must visibly depend on TR colors.
    support_colors = {int(logical[r, c]) for r, c in filled if (r, c) != anchor}
    if len(support_colors) < 2:
        return None

    patch = np.repeat(np.repeat(logical, SOURCE_ANCHOR_SIZE, axis=0), SOURCE_ANCHOR_SIZE, axis=1)
    return patch


def _choose_anchor(qb: int, qr: int, size: int, patch_s: np.ndarray) -> tuple[int, int]:
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


def _make_sources() -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    built = _build_logical_shape()
    if built is None:
        return None
    lh, lw, filled, anchor = built
    shape_patch = _logical_to_shape_patch(lh, lw, filled, anchor)
    color_patch = _build_color_template(lh, lw, anchor, filled)
    if color_patch is None:
        return None
    merged = color_patch.copy()
    merged[shape_patch == BG] = BG
    if np.array_equal(merged, np.fliplr(merged)):
        return None
    if np.array_equal(merged, np.flipud(merged)):
        return None
    return shape_patch, color_patch, merged


def _gen_attempt() -> np.ndarray | None:
    sources = _make_sources()
    if sources is None:
        return None
    shape_patch, color_patch_canon, merged = sources
    color_patch_right = np.fliplr(color_patch_canon)

    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): SOURCE_ANCHOR_SIZE,
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    scaled = {
        key: _resize_nn(_reflect_patch(merged, key[0], key[1]),
                        max(1, int(round(merged.shape[0] * target_sizes[key] / SOURCE_ANCHOR_SIZE))),
                        max(1, int(round(merged.shape[1] * target_sizes[key] / SOURCE_ANCHOR_SIZE))))
        for key in ((1, 0), (1, 1))
    }
    for key, patch_s in scaled.items():
        roundtrip = _resize_nn(patch_s, _reflect_patch(merged, key[0], key[1]).shape[0],
                               _reflect_patch(merged, key[0], key[1]).shape[1])
        if not np.array_equal(roundtrip, _reflect_patch(merged, key[0], key[1])):
            return None

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - shape_patch.shape[0])
    tl_c0 = random.randint(0, MID - shape_patch.shape[1])
    anchors[(0, 0)] = (_red_signature(shape_patch)[0] + tl_r0, _red_signature(shape_patch)[1] + tl_c0)

    tr_r0 = random.randint(0, MID - color_patch_right.shape[0])
    tr_c0 = random.randint(MID + 1, GRID_SIZE - color_patch_right.shape[1])
    anchors[(0, 1)] = (_red_signature(color_patch_right)[0] + tr_r0, _red_signature(color_patch_right)[1] + tr_c0)

    for key in ((1, 0), (1, 1)):
        anchors[key] = _choose_anchor(key[0], key[1], target_sizes[key], scaled[key])

    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, shape_patch, tl_r0, tl_c0)
    paste(out, color_patch_right, tr_r0, tr_c0)
    for key in ((1, 0), (1, 1)):
        ar, ac = anchors[key]
        paste(out, np.full((target_sizes[key], target_sizes[key]), RED, dtype=int), ar, ac)
        _stamp(out, scaled[key], ar, ac, MID, MID, BG)

    inp = out.copy()
    for key in ((1, 0), (1, 1)):
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((target_sizes[key], target_sizes[key]), RED, dtype=int), ar, ac)

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid D11 b10624e5 input")
