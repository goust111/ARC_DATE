# Combo variant of b10624e5 (index 35)
# combo_spec: D2/1+D6/0+D7/1
# Name: 控制像素指定源 + 擦除源 + 仅输出目标象限

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste

# ---------------------------------------------------------------------------
# Design analysis for combo D2/1 + D6/0 + D7/1
#
# a) Concrete behavior and trace
# - The base task has four 13x13 quadrants split by one full divider row and
#   one full divider column. Each quadrant still has one red square anchor.
# - D2/1 moves the source away from fixed TL: two border control pixels select
#   the source quadrant. The source anchor in that chosen quadrant is still the
#   2x2 reference anchor, and only that quadrant contains the colorful motif.
# - D6/0 changes copy semantics into move semantics: after the motif is used,
#   the source quadrant keeps only its red anchor and loses the surrounding
#   non-red motif pixels.
# - D7/1 removes the full-board output: instead of returning the whole 27x27
#   board, the solver returns only one target quadrant panel. To keep the crop
#   deterministic without adding a second symbol system, the kept target is the
#   quadrant diagonally opposite the controlled source. Example: controls mark
#   TR, so the source patch comes from TR, it is erased there, a scaled+flipped
#   copy is placed in BL, and the final output is only the 13x13 BL quadrant.
#
# b) Complexity calibration
# - Using exactly two control pixels keeps source decoding crisp: one edge cue
#   determines the row half, the other determines the column half.
# - Returning one cropped target panel would be too ambiguous if the three
#   non-source anchors looked interchangeable. The generator therefore gives the
#   three non-source anchors distinct sizes, so the visible output anchor size
#   identifies which input quadrant was kept.
# - The target is tied to the source by a simple geometric rule (diagonal
#   opposite), so the combo stays inferable from 2-4 examples rather than
#   needing an extra hidden convention.
#
# c) Hidden assumptions from the original solver that no longer hold
# - The source is no longer guaranteed to be TL.
# - Extracting the motif from the TL slice is invalid; the source quadrant must
#   be decoded from border controls first.
# - The original solver produced three copies and kept the source intact. Here
#   only one destination is materialized in the final output, and the source
#   motif is deleted after use.
# - The original `return out` assumption is invalid because the output is now a
#   cropped 13x13 panel, not a 27x27 board.
#
# d) Placement legality
# - The source motif is kept one cell away from the outer border whenever its
#   quadrant touches the border, so control pixels remain isolated.
# - Non-source anchors that are not shown in the output are also kept one cell
#   away from the outer border, preventing confusion with border controls.
# - The kept target anchor is placed where the full transformed motif fits
#   inside its quadrant, so the cropped output panel contains the entire result
#   and not a degenerate clipped fragment.
#
# e) Visual observability
# - The control color is reserved and never used inside motifs, so the source
#   selection cue is visually obvious.
# - The input always shows one multi-color motif plus three anchor-only
#   quadrants, while the output always shows a single quadrant panel with a
#   red anchor and the transformed motif. This makes the crop rule visible.
# - The generator rejects motifs that are symmetric under 180-degree rotation,
#   so the diagonal reflection is not visually erased by symmetry.
#
# f) Visual cleanliness
# - No extra legends are added beyond the two border controls.
# - The cropped output contains only one quadrant panel, which avoids the noisy
#   L-shaped "three quadrants kept" alternative that would be hard to parse.
#
# g) Color-rule audit
# - No rule depends on numeric color ordering. The control color is a fixed
#   reserved marker distinct from bg/divider/red and excluded from motif colors.
#
# h) Fusion-specific non-degeneracy checks used during validation
# - The meaningful D2 signal is the controlled source quadrant. We later measure
#   how often the combo output differs from a TL-assuming baseline on the same
#   generated inputs.
# - D6 is semantic rather than directly visible after cropping, so the solver is
#   implemented in two stages: build the full moved board with the source motif
#   erased, then crop the kept target quadrant. This prevents D6 from being
#   accidentally skipped in code.
#
# i) New assumptions introduced by the combo
# - There is exactly one control color on the border and exactly two control
#   pixels, one on a horizontal edge and one on a vertical edge.
# - The controlled source quadrant contains exactly one non-background component
#   that includes red: the source anchor plus its attached motif.
# - The diagonally opposite quadrant is the only one that needs a full-fit
#   transformed motif; the other two non-source quadrants are anchor-only
#   distractors.
# - The generator enforces all of these assumptions before accepting a sample.
# ---------------------------------------------------------------------------

BG = 4
DIV = 1
RED = 2
CTRL = 6
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
COLOR_POOL = [0, 1, 3, 5, 7, 8, 9]
MAX_RETRIES = 500

HALF_CENTER = {0: 6, 1: 20}


def _find_dividers(g: np.ndarray, bg: int) -> tuple[int, int]:
    """Return the unique matching full-row/full-column divider."""
    h, w = g.shape
    full_rows = [(r, int(g[r, 0])) for r in range(h) if np.all(g[r, :] == g[r, 0]) and int(g[r, 0]) != bg]
    full_cols = [(c, int(g[0, c])) for c in range(w) if np.all(g[:, c] == g[0, c]) and int(g[0, c]) != bg]
    matches = []
    for r, vr in full_rows:
        for c, vc in full_cols:
            if vr == vc:
                matches.append((int(r), int(c)))
    assert len(matches) == 1, "grid must have exactly one matching divider row/col"
    return matches[0]


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(h - 1, c) for c in range(w)]
    coords += [(r, 0) for r in range(1, h - 1)]
    coords += [(r, w - 1) for r in range(1, h - 1)]
    return coords


def _quad_bounds(h: int, w: int, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    return {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }


def _opposite_key(key: tuple[int, int]) -> tuple[int, int]:
    return 1 - key[0], 1 - key[1]


def _find_source_quadrant(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> tuple[tuple[int, int], int]:
    """Decode the source quadrant and control color from the border markers."""
    h, w = g.shape
    div = int(g[mid_r, mid_c])
    border_colors = {int(g[r, c]) for r, c in _border_coords(h, w)}
    border_colors.discard(bg)
    border_colors.discard(div)
    assert len(border_colors) == 1, "border must contain exactly one control color besides bg/divider"
    ctrl = int(next(iter(border_colors)))

    ctrl_pts = [(int(r), int(c)) for r, c in _border_coords(h, w) if int(g[r, c]) == ctrl]
    assert len(ctrl_pts) == 2, "must have exactly two border control pixels"

    keys = set()
    seen_row_edge = False
    seen_col_edge = False
    for r, c in ctrl_pts:
        if r == 0 or r == h - 1:
            seen_row_edge = True
            qb = 0 if r == 0 else 1
            qr = 0 if c < mid_c else 1
        else:
            assert c == 0 or c == w - 1, "control pixels must lie on the outer border"
            seen_col_edge = True
            qb = 0 if r < mid_r else 1
            qr = 0 if c == 0 else 1
        keys.add((qb, qr))

    assert seen_row_edge and seen_col_edge, "controls must include one horizontal-edge and one vertical-edge mark"
    assert len(keys) == 1, "the two control pixels must decode the same source quadrant"
    return next(iter(keys)), ctrl


def _extract_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int]]:
    """Return quadrant -> (anchor top row, anchor left col, anchor size)."""
    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red anchors"
    anchors: dict[tuple[int, int], tuple[int, int, int]] = {}
    for pts in red:
        r0, c0, r1, c1 = bbox_coords(pts)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        key = (0 if rr < mid_r else 1, 0 if cc < mid_c else 1)
        side = max(r1 - r0 + 1, c1 - c0 + 1)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(side))
    assert len(anchors) == 4, "there must be exactly four anchors"
    return anchors


def _source_component(src_quad: np.ndarray, bg: int) -> tuple[np.ndarray, np.ndarray]:
    """Return the source component crop and its local coordinates inside the quadrant."""
    comps = components_mask(src_quad != bg, connectivity=4)
    matches = [pts for pts in comps if any(int(src_quad[r, c]) == RED for r, c in pts)]
    assert len(matches) == 1, "source quadrant must contain exactly one non-bg component with red"
    pts = matches[0]
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = src_quad[r0 : r1 + 1, c0 : c1 + 1].copy()
    return patch, pts


def _resize_nn(img: np.ndarray, new_h: int, new_w: int) -> np.ndarray:
    src_h, src_w = img.shape
    assert new_h > 0 and new_w > 0, "resize target must be positive"
    rr = np.clip(np.floor(np.arange(new_h) * src_h / new_h).astype(int), 0, src_h - 1)
    cc = np.clip(np.floor(np.arange(new_w) * src_w / new_w).astype(int), 0, src_w - 1)
    return img[rr[:, None], cc[None, :]]


def _scaled_patch(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    source_size: int,
    target_size: int,
) -> np.ndarray:
    p = patch
    if dst_key[1] != src_key[1]:
        p = np.fliplr(p)
    if dst_key[0] != src_key[0]:
        p = np.flipud(p)
    ratio = float(target_size) / float(source_size)
    new_h = max(1, int(round(p.shape[0] * ratio)))
    new_w = max(1, int(round(p.shape[1] * ratio)))
    patch_s = _resize_nn(p, new_h, new_w)
    roundtrip = _resize_nn(patch_s, p.shape[0], p.shape[1])
    assert np.array_equal(roundtrip, p), "scaling must preserve all motif information"
    return patch_s


def _roundtrip_safe(
    patch: np.ndarray,
    src_key: tuple[int, int],
    dst_key: tuple[int, int],
    source_size: int,
    target_size: int,
) -> bool:
    try:
        _scaled_patch(patch, src_key, dst_key, source_size, target_size)
    except AssertionError:
        return False
    return True


def _place_patch_on_full_board(
    out: np.ndarray,
    patch_s: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    mid_r: int,
    mid_c: int,
    ctrl: int,
) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    assert rr_red.size > 0, "scaled patch must contain red anchor pixels"
    place_r0 = int(anchor_r) - int(rr_red.min())
    place_c0 = int(anchor_c) - int(cc_red.min())

    h, w = out.shape
    src_rr, src_cc = np.where(patch_s != BG)
    for sr, sc in zip(src_rr.tolist(), src_cc.tolist(), strict=False):
        tr = place_r0 + sr
        tc = place_c0 + sc
        if not (0 <= tr < h and 0 <= tc < w):
            continue
        if tr == mid_r or tc == mid_c:
            continue
        val = int(patch_s[sr, sc])
        if int(out[tr, tc]) == ctrl:
            continue
        if val != RED and int(out[tr, tc]) == RED:
            continue
        out[tr, tc] = val


def _solve_full(grid: np.ndarray) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], dict[tuple[int, int], tuple[int, int, int, int]]]:
    """
    Build the full 27x27 moved-board state before the final crop.
    This stage performs the D6 erase semantics explicitly.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    quad_bounds = _quad_bounds(h, w, mid_r, mid_c)
    src_key, ctrl = _find_source_quadrant(g, bg, mid_r, mid_c)
    dst_key = _opposite_key(src_key)
    anchors = _extract_anchors(g, mid_r, mid_c)
    assert src_key in anchors and dst_key in anchors, "source/target anchors must exist"

    q_r0, q_r1, q_c0, q_c1 = quad_bounds[src_key]
    src_quad = g[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1]
    patch, comp_pts = _source_component(src_quad, bg)

    out = g.copy()
    for rr, cc in comp_pts.tolist():
        gr = q_r0 + int(rr)
        gc = q_c0 + int(cc)
        if int(out[gr, gc]) != RED:
            out[gr, gc] = bg

    _sr0, _sc0, sL = anchors[src_key]
    tr0, tc0, tL = anchors[dst_key]
    patch_s = _scaled_patch(patch, src_key, dst_key, sL, tL)
    _place_patch_on_full_board(out, patch_s, tr0, tc0, mid_r, mid_c, ctrl)
    return out, src_key, dst_key, quad_bounds


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move one controlled quadrant motif into its opposite quadrant, then emit only that target panel.

    Input structure:
    - A 27x27 board is split into four 13x13 quadrants by one full divider row and one full divider column.
    - Each quadrant contains one red square anchor, but only one quadrant also contains a multi-color motif attached
      to its anchor.
    - Two reserved-color border control pixels identify which quadrant is the source.

    Rule:
    - Decode the source quadrant from the border controls, extract the source anchor+motif component, and treat the
      diagonally opposite quadrant as the single kept target.
    - Scale the source component by the target-anchor/source-anchor size ratio and reflect it across the row and/or
      column halves that differ between source and target.
    - Erase the source motif while keeping its red anchor, place the transformed copy onto the opposite anchor, then
      return only the 13x13 content of that target quadrant.

    Key insight: The border controls decide both where the motif comes from and which opposite quadrant survives the
    crop, while the red anchor sizes still encode the scaling factor.
    """
    out_full, _src_key, dst_key, quad_bounds = _solve_full(np.asarray(grid, dtype=int))
    q_r0, q_r1, q_c0, q_c1 = quad_bounds[dst_key]
    return out_full[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1].copy()


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
    h: int,
    w: int,
) -> np.ndarray | None:
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
        return patch, ar * s, ac * s

    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar = random.randint(0, h - s)
    ac = random.randint(0, w - s)
    red_cells = {(ar + dr, ac + dc) for dr in range(s) for dc in range(s)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _is_visibly_asymmetric(patch: np.ndarray) -> bool:
    return not np.array_equal(patch, np.flipud(np.fliplr(patch)))


def _control_points(src_key: tuple[int, int]) -> list[tuple[int, int]]:
    qb, qr = src_key
    return [
        (0 if qb == 0 else GRID_SIZE - 1, HALF_CENTER[qr]),
        (HALF_CENTER[qb], 0 if qr == 0 else GRID_SIZE - 1),
    ]


def _source_top_left_ranges(src_key: tuple[int, int], patch_shape: tuple[int, int]) -> tuple[range, range]:
    q_r0, q_r1, q_c0, q_c1 = QUADS[src_key]
    ph, pw = map(int, patch_shape)
    top_margin = 1 if q_r0 == 0 else 0
    bottom_margin = 1 if q_r1 == GRID_SIZE - 1 else 0
    left_margin = 1 if q_c0 == 0 else 0
    right_margin = 1 if q_c1 == GRID_SIZE - 1 else 0
    r_lo = q_r0 + top_margin
    r_hi = q_r1 - ph + 1 - bottom_margin
    c_lo = q_c0 + left_margin
    c_hi = q_c1 - pw + 1 - right_margin
    return range(r_lo, r_hi + 1), range(c_lo, c_hi + 1)


def _choose_anchor_with_margin(key: tuple[int, int], size: int) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    top_margin = 1 if q_r0 == 0 else 0
    bottom_margin = 1 if q_r1 == GRID_SIZE - 1 else 0
    left_margin = 1 if q_c0 == 0 else 0
    right_margin = 1 if q_c1 == GRID_SIZE - 1 else 0
    r_lo = q_r0 + top_margin
    r_hi = q_r1 - size + 1 - bottom_margin
    c_lo = q_c0 + left_margin
    c_hi = q_c1 - size + 1 - right_margin
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _choose_target_anchor(key: tuple[int, int], size: int, patch_s: np.ndarray) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = QUADS[key]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    top_margin = 1 if q_r0 == 0 else 0
    bottom_margin = 1 if q_r1 == GRID_SIZE - 1 else 0
    left_margin = 1 if q_c0 == 0 else 0
    right_margin = 1 if q_c1 == GRID_SIZE - 1 else 0

    anch_r0 = q_r0 + top_margin
    anch_r1 = q_r1 - size + 1 - bottom_margin
    anch_c0 = q_c0 + left_margin
    anch_c1 = q_c1 - size + 1 - right_margin

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _gen_attempt() -> np.ndarray | None:
    src_key = random.choice(list(QUADS))
    dst_key = _opposite_key(src_key)
    ctrl_pts = _control_points(src_key)

    other_keys = [key for key in QUADS if key != src_key]
    sampled_sizes = random.sample([1, 2, 3, 4], len(other_keys))
    target_sizes = {src_key: SOURCE_ANCHOR_SIZE}
    for key, size in zip(other_keys, sampled_sizes, strict=False):
        target_sizes[key] = int(size)

    needs_block = target_sizes[dst_key] % SOURCE_ANCHOR_SIZE != 0
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if not _is_visibly_asymmetric(patch):
        return None
    if not _roundtrip_safe(patch, src_key, dst_key, SOURCE_ANCHOR_SIZE, target_sizes[dst_key]):
        return None

    patch_s = _scaled_patch(patch, src_key, dst_key, SOURCE_ANCHOR_SIZE, target_sizes[dst_key])

    src_rows, src_cols = _source_top_left_ranges(src_key, patch.shape)
    if len(src_rows) == 0 or len(src_cols) == 0:
        return None
    src_patch_r0 = random.choice(list(src_rows))
    src_patch_c0 = random.choice(list(src_cols))

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    anchors[src_key] = (src_patch_r0 + anchor_r_in_patch, src_patch_c0 + anchor_c_in_patch)
    target_anchor = _choose_target_anchor(dst_key, target_sizes[dst_key], patch_s)
    if target_anchor is None:
        return None
    anchors[dst_key] = target_anchor

    for key in QUADS:
        if key in anchors:
            continue
        rc = _choose_anchor_with_margin(key, target_sizes[key])
        if rc is None:
            return None
        anchors[key] = rc

    inp = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    inp[MID, :] = DIV
    inp[:, MID] = DIV
    for r, c in ctrl_pts:
        inp[r, c] = CTRL

    paste(inp, patch, src_patch_r0, src_patch_c0)
    for key, size in target_sizes.items():
        if key == src_key:
            continue
        ar, ac = anchors[key]
        paste(inp, np.full((size, size), RED, dtype=int), ar, ac)
    for r, c in ctrl_pts:
        inp[r, c] = CTRL

    try:
        expected = main(inp)
    except AssertionError:
        return None
    if expected.shape != (13, 13):
        return None
    if not np.any((expected != BG) & (expected != RED)):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
