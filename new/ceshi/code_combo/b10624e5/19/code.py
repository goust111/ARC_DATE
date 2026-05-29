# Combo variant of b10624e5 (index 19)
# combo_spec: D5/1+D12/1
# Name: 箭头标记反射方向 + 反射轴由锚块特征决定

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_ROOT) not in sys.path:
    sys.path.append(str(_SOLVER_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization and behavior trace
# - Keep the original 27x27 four-quadrant board, divider, TL source motif, and
#   anchor-size scaling rule.
# - Add two visible control channels in every non-TL quadrant:
#   1. one green arrow glyph placed near the target anchor,
#   2. one asymmetric colored corner inside the target anchor itself.
# - The arrow gives the mandatory "primary" reflection:
#     right arrow  -> LR reflection is required,
#     down arrow   -> UD reflection is required.
# - The anchor corner then decides whether a second, orthogonal reflection is
#   also needed. Concretely, after reading the arrow, the solver compares the
#   observed target-anchor template against the TL anchor template under:
#     primary only  vs  primary + orthogonal flip.
# - Small trace:
#   Old TR behavior: always LR, because the quadrant sits on the right.
#   New TR behavior:
#     1. read the local arrow; if it is a down arrow, start from UD instead,
#     2. inspect the marked corner inside the target anchor,
#     3. if that corner matches "UD only", stop there,
#     4. if it matches "UD + LR", apply both flips to the whole motif.
#   Difference: quadrant position no longer selects the mirror; the arrow gives
#   one axis, and the anchor feature decides whether to add the other axis.
#
# b) Complexity calibration
# - The arrow vocabulary stays tiny: exactly two 3x3 glyphs.
# - The anchor feature carries only one extra bit: "primary only" or "both".
# - This keeps the search space small while still forcing fusion:
#   arrows alone are insufficient, and anchors alone are ambiguous without first
#   knowing which primary axis the arrow selected.
# - Target anchor sizes are restricted to {2, 4} so the 2x2 asymmetric anchor
#   template survives nearest-neighbor scaling without aliasing.
#
# c) Hidden-assumption audit of the original main()
# - Original `if qr == 1: fliplr` / `if qb == 1: flipud` assumed location was
#   the only reflection signal. That is invalid here.
# - Original non-TL input contained only plain red anchors. That would erase the
#   new anchor cue, so the generator must leave the decorated anchor visible.
# - Original `out = g.copy()` was safe because no extra symbols existed. Here we
#   must remove arrow markers before stamping or they would leak into the output.
# - Original source extraction from the TL connected component still works:
#   arrows exist only outside TL, and the TL motif remains one 4-connected
#   component containing the red anchor.
#
# d) Placement legality
# - The right arrow is drawn 1 cell above the anchor; the down arrow is drawn 1
#   cell left of the anchor. Anchor sampling reserves that space.
# - The transformed full motif is still used to choose anchor placements, so the
#   copied motif fits when possible and clips only under the same rules as the
#   base task.
# - The arrow is kept disconnected from the anchor, so component extraction for
#   the decorated anchor remains unambiguous.
#
# e) Visual observability
# - A human can read the arrow as the obvious first instruction, then notice
#   that two targets with the same arrow can still differ because of the anchor
#   corner orientation.
# - The generator enforces both arrow types, both anchor states, and at least
#   one mismatch from the old quadrant-implied mirror rule.
#
# f) Visual cleanliness
# - Green is reserved for arrows and never used inside the motif.
# - The anchor corner uses a separate dedicated color and stays embedded in the
#   anchor, while arrows sit outside with a one-cell gap. This avoids overlap and
#   keeps "symbol" and "object feature" visually distinct.
#
# g) Change inferability
# - No hidden color ordering is used. Red still marks anchors, green means arrow
#   instruction, and the anchor-corner color is purely a visible structural cue.
# - The generator rejects samples where the fused rule collapses to either
#   "arrows only" or "original quadrant rule" at the whole-output level.
#
# Solver-side assumptions introduced here and guaranteed by the generator
# - The TL anchor is a 2x2 asymmetric template with exactly one colored corner.
# - Every non-TL quadrant has exactly one arrow glyph and exactly one decorated
#   anchor component containing red.
# - Each target anchor matches exactly one of two templates once its arrow-chosen
#   primary axis is known: primary only, or primary + orthogonal flip.


BG = 4
DIV = 1
RED = 2
ARROW = 3
ANCHOR_MARK = 8
GRID_SIZE = 27
MID = 13

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

SOURCE_ANCHOR_SIZE = 2
TARGET_SIZES = (2, 4)
BLOCK_DIMS = [(2, 2), (2, 3), (3, 2), (3, 3)]
BLOCK_PROB = 0.5
FILL_RATIO_MIN = 0.5
FILL_RATIO_MAX = 0.8
FREEFORM_H_RANGE = (3, 6)
FREEFORM_W_RANGE = (3, 6)
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 5, 6, 7, 9]
MAX_RETRIES = 500
MIN_VARIANT_DIFF = 3

RIGHT_ARROW = frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)})
DOWN_ARROW = frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)})
TARGET_KEYS = [(0, 1), (1, 0), (1, 1)]
SPATIAL_DEFAULTS = {
    (0, 1): (False, True),
    (1, 0): (True, False),
    (1, 1): (True, True),
}
ARROW_TO_PRIMARY = {
    RIGHT_ARROW: (False, True),
    DOWN_ARROW: (True, False),
}
PRIMARY_TO_ARROW = {v: k for k, v in ARROW_TO_PRIMARY.items()}
BOTH_FLIPS = (True, True)


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


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    if new_h <= 0 or new_w <= 0:
        return img.copy()
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _reflect(img: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = img
    if flip_lr:
        out = np.fliplr(out)
    if flip_ud:
        out = np.flipud(out)
    return out


def _transform_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool, target_size: int) -> np.ndarray:
    p = _reflect(patch, flip_ud, flip_lr)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _normalize_component(pts: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(pts)
    return frozenset((int(r - r0), int(c - c0)) for r, c in pts)


def _component_patch(quadrant: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    comps = components_mask(quadrant != bg, connectivity=4)
    for pts in comps:
        if any(quadrant[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            return quadrant[r0 : r1 + 1, c0 : c1 + 1].copy(), (int(r0), int(c0), int(r1), int(c1))
    raise AssertionError("quadrant must contain a non-bg component with a red anchor")


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            return tl[r0 : r1 + 1, c0 : c1 + 1].copy()
    raise AssertionError("top-left quadrant must contain a non-bg component with a red pixel")


def _read_arrow_primaries(
    g: np.ndarray, mid_r: int, mid_c: int
) -> tuple[dict[tuple[int, int], tuple[bool, bool]], list[list[tuple[int, int]]]]:
    comps = components_mask(g == ARROW, connectivity=4)
    primaries: dict[tuple[int, int], tuple[bool, bool]] = {}
    marker_pts = []
    for pts_arr in comps:
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key != (0, 0), "arrows are only allowed in non-TL quadrants"
        assert key not in primaries, "each non-TL quadrant must contain exactly one arrow glyph"
        shape = _normalize_component(pts_arr)
        if shape not in ARROW_TO_PRIMARY:
            raise AssertionError("unrecognized arrow glyph")
        primaries[key] = ARROW_TO_PRIMARY[shape]
        marker_pts.append([(int(r), int(c)) for r, c in pts_arr])
    for key in TARGET_KEYS:
        assert key in primaries, "each non-TL quadrant must contain exactly one arrow glyph"
    return primaries, marker_pts


def _detect_transform(
    source_anchor: np.ndarray,
    target_anchor: np.ndarray,
    primary: tuple[bool, bool],
) -> tuple[bool, bool]:
    matches = []
    for flip_ud, flip_lr in (primary, BOTH_FLIPS):
        cand = _reflect(source_anchor, flip_ud, flip_lr)
        cand = _resize_nn(cand, target_anchor.shape[0], target_anchor.shape[1])
        if np.array_equal(cand, target_anchor):
            matches.append((flip_ud, flip_lr))
    assert len(matches) == 1, "target anchor must resolve primary-only vs primary-plus-orthogonal uniquely"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the TL motif across quadrants with size scaling, arrow-selected primary reflection, and anchor-selected extra reflection.

    Input structure:
    - A full row and full column of one non-background color split the grid into four 13x13 quadrants.
    - The TL quadrant contains the complete colored motif and a 2x2 asymmetric anchor whose colored corner is part of the motif.
    - Each non-TL quadrant contains one decorated red anchor plus exactly one nearby green arrow glyph; the anchor size still varies by quadrant.

    Rule:
    - Extract the TL connected component containing the red anchor as the source motif.
    - In each target quadrant, read the arrow first to get the mandatory reflection axis: right arrow means LR, down arrow means UD.
    - Compare the observed target-anchor template against the TL anchor under either that primary flip alone or both flips; apply the matching transform to the full source motif, scale by anchor-size ratio, align the red region to the target anchor, and stamp non-background pixels. Remove the green arrows from the final output.

    Key insight: The arrow does not fully specify the transform by itself. It only chooses the primary axis; the anchor's internal asymmetry tells you whether an additional orthogonal flip must be composed on top of it.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    primaries, marker_components = _read_arrow_primaries(g, mid_r, mid_c)
    for pts in marker_components:
        for r, c in pts:
            out[r, c] = bg

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchor blocks"

    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        size = max(r1 - r0 + 1, c1 - c0 + 1)
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor block"
        anchors[key] = (int(r0), int(c0), int(size))

    assert (0, 0) in anchors, "top-left quadrant must contain a red anchor block"
    sr0, sc0, sL = anchors[(0, 0)]
    assert sL == SOURCE_ANCHOR_SIZE, "top-left anchor must stay 2x2 in this combo"

    patch = _extract_source_patch(g, bg, mid_r, mid_c)
    tl = g[0:mid_r, 0:mid_c]
    source_anchor = tl[sr0 : sr0 + sL, sc0 : sc0 + sL].copy()
    assert source_anchor.shape == (SOURCE_ANCHOR_SIZE, SOURCE_ANCHOR_SIZE), "source anchor template must be 2x2"

    for key, (tr0, tc0, tL) in anchors.items():
        if key == (0, 0):
            continue

        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
        target_anchor, _ = _component_patch(quad, bg)
        flip_ud, flip_lr = _detect_transform(source_anchor, target_anchor, primaries[key])

        patch_s = _transform_patch(patch, flip_ud, flip_lr, tL)
        roundtrip = _resize_nn(patch_s, _reflect(patch, flip_ud, flip_lr).shape[0], _reflect(patch, flip_ud, flip_lr).shape[1])
        assert np.array_equal(roundtrip, _reflect(patch, flip_ud, flip_lr)), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == RED)
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
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = val

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


def _assign_colors(
    filled: set[tuple[int, int]],
    red_cells: set[tuple[int, int]],
    marker_cell: tuple[int, int],
    h: int,
    w: int,
) -> np.ndarray | None:
    rows = {r for r, _ in filled}
    cols = {c for _, c in filled}
    if rows != set(range(h)) or cols != set(range(w)):
        return None

    free_cells = [p for p in filled if p not in red_cells and p != marker_cell]
    max_extra = min(MOTIF_COLORS_MAX - 2, len(free_cells))
    if max_extra < 1:
        return None
    n_extra = random.randint(1, max_extra)

    pool = COLOR_POOL[:]
    random.shuffle(pool)
    extras = pool[:n_extra]

    grid = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        grid[r, c] = RED
    mr, mc = marker_cell
    grid[mr, mc] = ANCHOR_MARK

    random.shuffle(free_cells)
    for i, (r, c) in enumerate(free_cells):
        grid[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return grid


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
        logical = _assign_colors(filled, {(ar, ac)}, (ar, ac), lh, lw)
        if logical is None:
            return None
        patch = np.repeat(np.repeat(logical, s, axis=0), s, axis=1)
        patch[ar * s : ar * s + s, ac * s : ac * s + s] = RED
        patch[ar * s, ac * s] = ANCHOR_MARK
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - s), random.randint(0, w - s)
    anchor_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    marker_cell = (ar, ac)
    red_cells = anchor_cells - {marker_cell}
    min_fill = max(len(anchor_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, anchor_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, marker_cell, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _is_asymmetric(patch: np.ndarray) -> bool:
    return (
        not np.array_equal(patch, np.fliplr(patch))
        and not np.array_equal(patch, np.flipud(patch))
        and not np.array_equal(patch, np.flipud(np.fliplr(patch)))
    )


def _choose_controls() -> tuple[
    dict[tuple[int, int], tuple[bool, bool]],
    dict[tuple[int, int], bool],
    dict[tuple[int, int], tuple[bool, bool]],
] | None:
    primaries = {key: random.choice(list(PRIMARY_TO_ARROW)) for key in TARGET_KEYS}
    extras = {key: bool(random.getrandbits(1)) for key in TARGET_KEYS}
    full = {key: (BOTH_FLIPS if extras[key] else primaries[key]) for key in TARGET_KEYS}

    if len(set(primaries.values())) < 2:
        return None
    if not any(extras.values()):
        return None
    if all(extras.values()):
        return None
    if len(set(full.values())) < 2:
        return None
    if not any(full[key] != SPATIAL_DEFAULTS[key] for key in TARGET_KEYS):
        return None
    return primaries, extras, full


def _choose_anchor(
    qb: int,
    qr: int,
    size: int,
    patch_s: np.ndarray,
    need_up_marker: bool,
    need_left_marker: bool,
) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[(qb, qr)]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0 = q_r0 + (4 if need_up_marker else 0)
    anch_r1 = q_r1 - size + 1
    if need_left_marker:
        anch_r1 = min(anch_r1, q_r1 - 2)
    anch_c0 = q_c0 + (4 if need_left_marker else 0)
    anch_c1 = q_c1 - size + 1
    if need_up_marker:
        anch_c1 = min(anch_c1, q_c1 - 2)

    if anch_r0 > anch_r1 or anch_c0 > anch_c1:
        return None

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo <= r_hi and c_lo <= c_hi:
        return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)
    return random.randint(anch_r0, anch_r1), random.randint(anch_c0, anch_c1)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int) -> None:
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
        & (tgt_rr != MID)
        & (tgt_cc != MID)
    )
    tgt_rr, tgt_cc = tgt_rr[valid], tgt_cc[valid]
    src_rr, src_cc = src_rr[valid], src_cc[valid]
    not_red = out[tgt_rr, tgt_cc] != RED
    out[tgt_rr[not_red], tgt_cc[not_red]] = patch_s[src_rr[not_red], src_cc[not_red]]


def _paint_arrow(grid: np.ndarray, top: int, left: int, shape: frozenset[tuple[int, int]]) -> None:
    for dr, dc in shape:
        grid[top + dr, left + dc] = ARROW


def _place_arrow(grid: np.ndarray, anchor_r: int, anchor_c: int, primary: tuple[bool, bool]) -> None:
    shape = PRIMARY_TO_ARROW[primary]
    if shape == RIGHT_ARROW:
        _paint_arrow(grid, anchor_r - 4, anchor_c, shape)
    else:
        _paint_arrow(grid, anchor_r, anchor_c - 4, shape)


def _paste_anchor_template(out: np.ndarray, anchor_patch: np.ndarray, anchor_r: int, anchor_c: int) -> None:
    rr_red, cc_red = np.where(anchor_patch == RED)
    paste(out, anchor_patch, anchor_r - int(rr_red.min()), anchor_c - int(cc_red.min()))


def _build_output(
    patch: np.ndarray,
    anchor_patch: np.ndarray,
    tl_r0: int,
    tl_c0: int,
    target_sizes: dict[tuple[int, int], int],
    anchors: dict[tuple[int, int], tuple[int, int]],
    transforms: dict[tuple[int, int], tuple[bool, bool]],
) -> np.ndarray:
    out = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    out[MID, :] = DIV
    out[:, MID] = DIV
    paste(out, patch, tl_r0, tl_c0)
    for key in TARGET_KEYS:
        ar, ac = anchors[key]
        anchor_part = _transform_patch(anchor_patch, transforms[key][0], transforms[key][1], target_sizes[key])
        _paste_anchor_template(out, anchor_part, ar, ac)
    for key in TARGET_KEYS:
        patch_s = _transform_patch(patch, transforms[key][0], transforms[key][1], target_sizes[key])
        _stamp(out, patch_s, *anchors[key])
    return out


def _gen_attempt() -> np.ndarray | None:
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.choice(TARGET_SIZES),
        (1, 0): random.choice(TARGET_SIZES),
        (1, 1): random.choice(TARGET_SIZES),
    }
    if all(target_sizes[key] == SOURCE_ANCHOR_SIZE for key in TARGET_KEYS):
        target_sizes[random.choice(TARGET_KEYS)] = 4

    controls = _choose_controls()
    if controls is None:
        return None
    primaries, extras, full = controls

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if not _is_asymmetric(patch):
        return None

    anchor_patch = patch[
        anchor_r_in_patch : anchor_r_in_patch + SOURCE_ANCHOR_SIZE,
        anchor_c_in_patch : anchor_c_in_patch + SOURCE_ANCHOR_SIZE,
    ].copy()

    scaled_full = {
        key: _transform_patch(patch, full[key][0], full[key][1], target_sizes[key])
        for key in TARGET_KEYS
    }
    scaled_anchor = {
        key: _transform_patch(anchor_patch, full[key][0], full[key][1], target_sizes[key])
        for key in TARGET_KEYS
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key in TARGET_KEYS:
        primary = primaries[key]
        anchor = _choose_anchor(
            key[0],
            key[1],
            target_sizes[key],
            scaled_full[key],
            need_up_marker=bool(primary[1]),
            need_left_marker=bool(primary[0]),
        )
        if anchor is None:
            return None
        anchors[key] = anchor

    out = _build_output(patch, anchor_patch, tl_r0, tl_c0, target_sizes, anchors, full)
    arrow_only_out = _build_output(patch, anchor_patch, tl_r0, tl_c0, target_sizes, anchors, primaries)
    spatial_out = _build_output(patch, anchor_patch, tl_r0, tl_c0, target_sizes, anchors, SPATIAL_DEFAULTS)
    if int(np.count_nonzero(out != arrow_only_out)) < MIN_VARIANT_DIFF:
        return None
    if int(np.count_nonzero(out != spatial_out)) < MIN_VARIANT_DIFF:
        return None

    inp = out.copy()
    for key in TARGET_KEYS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        _paste_anchor_template(inp, scaled_anchor[key], ar, ac)
        _place_arrow(inp, ar, ac, primaries[key])

    # Keep the extras variable "used" in a semantically obvious way for readers:
    # it is the reason arrow_only_out differs from out.
    assert any(extras.values())
    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
