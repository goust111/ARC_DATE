# Variant of b10624e5
# Dimension: D12 — 符号编码：图例或参数标记 (symbol_instructions)
# Change: 无符号编码——缩放因子和反射轴均从空间关系隐式推导... → 箭头标记指示反射方向

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_ROOT = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_ROOT) not in sys.path:
    sys.path.append(str(_SOLVER_ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the D12 variant
#
# a) Concretization and behavior trace
# - The old task encoded the mirror rule implicitly: right quadrants meant LR flip,
#   bottom quadrants meant UD flip, and BR meant both.
# - This variant adds explicit green arrow glyphs inside each non-TL quadrant.
#   A right-pointing arrow means "apply LR reflection"; a down-pointing arrow
#   means "apply UD reflection". If a quadrant shows both arrows, both flips are
#   applied. The source TL quadrant still contains the full motif and its 2x2 red
#   anchor; target quadrants still contain only a red anchor in the output-less input.
# - Concrete trace: in the old logic, a TR target with size 3 always used only LR
#   reflection because it sat on the right. In the new logic, the same TR target can
#   instead show a down arrow and therefore use only UD reflection; the copied motif
#   differs even though the anchor size is unchanged.
#
# b) Complexity calibration
# - The symbol vocabulary is intentionally small: only two 3x3 arrow glyphs exist,
#   and they compose into three target cases {LR, UD, both}. This keeps the new
#   freedom visible without exploding the hypothesis space.
# - The generator enforces at least two distinct marker patterns across the three
#   targets and at least one mismatch from the original quadrant-implied mirrors.
#
# c) Hidden-assumption audit of the original main()
# - Original `if qr == 1: fliplr` and `if qb == 1: flipud` relied on quadrant
#   position being the only source of reflection instructions. That premise is no
#   longer valid, so reflection flags must be read from marker components instead.
# - Original `out = g.copy()` was harmless because non-TL quadrants only contained
#   anchors. With markers added, leaving `out` unchanged would leak arrows into the
#   output, so marker pixels must be removed before stamping motifs.
# - Original source-patch extraction from the TL connected component still holds,
#   because markers exist only outside TL and the TL motif remains a single 4-connected
#   non-bg component containing the red anchor.
#
# d) Placement legality
# - Marker arrows are placed relative to each target anchor with a one-cell gap:
#   the LR arrow sits above the anchor, the UD arrow sits left of the anchor.
# - Anchor sampling therefore reserves 4 rows above and/or 4 cols left when the
#   corresponding marker is present. The chooser also prefers placements where the
#   entire scaled motif fits in the quadrant, but may fall back to legal clipped
#   placements just like the original task.
#
# e) Visual observability
# - The generator rejects symmetric source motifs and rejects samples whose output
#   under the new marker-driven rule is too close to the old quadrant-driven rule.
#   This makes the D12 change visible in a few examples instead of hiding inside
#   symmetric or over-downscaled cases.
#
# f) Visual cleanliness
# - Markers use a dedicated green color that the source motif never uses.
# - Because markers are separated from anchors by a gap and removed in the output,
#   they stay readable in the input and never clutter the copied motif in the output.
#
# g) Color inferability
# - No new color remapping is introduced. Red still means anchor, green only means
#   instruction marker, divider keeps its own color, and motif colors are otherwise
#   unchanged aside from excluding green to avoid ambiguity.
#
# Solver-side assumptions introduced by this variant and guaranteed by the generator
# - Exactly one divider row/col pair exists.
# - Each non-TL quadrant has one or two green arrow components and at least one arrow.
# - Green appears only in marker glyphs, never inside the TL motif.
# - Each arrow glyph matches one of the two exact templates below.
# ---------------------------------------------------------------------------


BG = 4
DIV = 1
RED = 2
MARKER = 3
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
COLOR_POOL = [0, 5, 6, 7, 8, 9]
TARGET_SIZE_RANGE = (1, 4)
MAX_RETRIES = 400
MIN_VARIANT_DIFF = 3

RIGHT_ARROW = frozenset({(0, 2), (1, 0), (1, 1), (1, 2), (2, 2)})
DOWN_ARROW = frozenset({(0, 1), (1, 1), (2, 0), (2, 1), (2, 2)})
TARGET_KEYS = [(0, 1), (1, 0), (1, 1)]
TRANSFORM_OPTIONS = [(False, True), (True, False), (True, True)]  # (flip_ud, flip_lr)
SPATIAL_DEFAULTS = {
    (0, 1): (False, True),
    (1, 0): (True, False),
    (1, 1): (True, True),
}


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


def _transform_patch(patch: np.ndarray, flip_ud: bool, flip_lr: bool, target_size: int) -> np.ndarray:
    p = patch
    if flip_lr:
        p = np.fliplr(p)
    if flip_ud:
        p = np.flipud(p)
    ratio = target_size / SOURCE_ANCHOR_SIZE
    h = max(1, int(round(p.shape[0] * ratio)))
    w = max(1, int(round(p.shape[1] * ratio)))
    return _resize_nn(p, h, w)


def _normalize_component(pts: np.ndarray) -> frozenset[tuple[int, int]]:
    r0, c0, _r1, _c1 = bbox_coords(pts)
    return frozenset((int(r - r0), int(c - c0)) for r, c in pts)


def _extract_source_patch(g: np.ndarray, bg: int, mid_r: int, mid_c: int) -> np.ndarray:
    tl = g[0:mid_r, 0:mid_c]
    comps = components_mask(tl != bg, connectivity=4)
    patch_bb = None
    for pts in comps:
        if any(tl[r, c] == RED for r, c in pts):
            r0, c0, r1, c1 = bbox_coords(pts)
            patch_bb = (int(r0), int(c0), int(r1), int(c1))
            break
    assert patch_bb is not None, "top-left quadrant must contain a non-bg component with a red pixel"
    pr0, pc0, pr1, pc1 = patch_bb
    return tl[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def _read_marker_transforms(g: np.ndarray, mid_r: int, mid_c: int) -> tuple[dict[tuple[int, int], tuple[bool, bool]], list[list[tuple[int, int]]]]:
    comps = components_mask(g == MARKER, connectivity=4)
    transforms = {key: [False, False] for key in TARGET_KEYS}
    grouped: dict[tuple[int, int], int] = {key: 0 for key in TARGET_KEYS}
    marker_pts = []
    for pts_arr in comps:
        r0, c0, r1, c1 = bbox_coords(pts_arr)
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key != (0, 0), "markers are only allowed in non-TL quadrants"
        shape = _normalize_component(pts_arr)
        if shape == RIGHT_ARROW:
            transforms[key][1] = True
        elif shape == DOWN_ARROW:
            transforms[key][0] = True
        else:
            raise AssertionError("unrecognized marker glyph")
        grouped[key] += 1
        marker_pts.append([(int(r), int(c)) for r, c in pts_arr])
    for key in TARGET_KEYS:
        assert grouped[key] >= 1, "each non-TL quadrant must contain at least one arrow marker"
        assert any(transforms[key]), "marker set must encode at least one reflection"
    return {key: (bool(v[0]), bool(v[1])) for key, v in transforms.items()}, marker_pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy the TL motif into the other quadrants, scaling by anchor size and mirroring by arrow markers.

    Input structure:
    - A single full-row/full-column divider splits the board into four 13x13 quadrants.
    - Every quadrant contains one red square anchor; the TL quadrant also contains the full colored motif.
    - Each non-TL quadrant additionally contains one or two green arrow glyphs: a right arrow requests LR reflection, and a down arrow requests UD reflection.

    Rule:
    - Extract the TL connected motif component that contains the red anchor.
    - For each target quadrant, read its arrow glyphs to decide which reflections to apply, then scale the transformed motif by the ratio between target-anchor size and TL-anchor size.
    - Align the scaled red area to the target red anchor, stamp non-background pixels, and remove the green instruction markers from the final grid.

    Key insight: The red anchor still tells you the scale, but mirror behavior is no longer implied by quadrant position; it is written explicitly by the nearby arrow symbols.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)

    transforms, marker_components = _read_marker_transforms(g, mid_r, mid_c)
    for pts in marker_components:
        for r, c in pts:
            out[r, c] = bg

    red = components_mask(g == RED, connectivity=4)
    assert red, "grid must contain red (color 2) anchor blocks"

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
    _sr0, _sc0, sL = anchors[(0, 0)]
    assert sL > 0, "top-left red anchor must have positive size"

    patch = _extract_source_patch(g, bg, mid_r, mid_c)

    for key, (tr0, tc0, tL) in anchors.items():
        if key == (0, 0):
            continue
        flip_ud, flip_lr = transforms[key]
        patch_t = patch
        if flip_lr:
            patch_t = np.fliplr(patch_t)
        if flip_ud:
            patch_t = np.flipud(patch_t)

        ratio = float(tL) / float(sL)
        new_h = max(1, int(round(patch_t.shape[0] * ratio)))
        new_w = max(1, int(round(patch_t.shape[1] * ratio)))
        patch_s = _resize_nn(patch_t, new_h, new_w)
        roundtrip = _resize_nn(patch_s, patch_t.shape[0], patch_t.shape[1])
        assert np.array_equal(roundtrip, patch_t), "scaling must preserve all motif information"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled source patch must contain red anchor pixels"
        ar0, ac0 = int(rr_red.min()), int(cc_red.min())
        place_r0 = tr0 - ar0
        place_c0 = tc0 - ac0

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


def _source_patch(needs_block: bool) -> tuple[np.ndarray, int, int] | None:
    S = SOURCE_ANCHOR_SIZE
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
        patch = np.repeat(np.repeat(logical, S, axis=0), S, axis=1)
        return patch, ar * S, ac * S
    h = random.randint(*FREEFORM_H_RANGE)
    w = random.randint(*FREEFORM_W_RANGE)
    ar, ac = random.randint(0, h - S), random.randint(0, w - S)
    red_cells = {(ar + dr, ac + dc) for dr in range(S) for dc in range(S)}
    min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
    max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
    if max_fill < min_fill:
        return None
    filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
    patch = _assign_colors(filled, red_cells, h, w)
    if patch is None:
        return None
    return patch, ar, ac


def _is_asymmetric(patch: np.ndarray) -> bool:
    return (
        not np.array_equal(patch, np.fliplr(patch))
        and not np.array_equal(patch, np.flipud(patch))
        and not np.array_equal(patch, np.flipud(np.fliplr(patch)))
    )


def _choose_transforms() -> dict[tuple[int, int], tuple[bool, bool]] | None:
    transforms = {key: random.choice(TRANSFORM_OPTIONS) for key in TARGET_KEYS}
    if len(set(transforms.values())) < 2:
        return None
    if all(transforms[key] == SPATIAL_DEFAULTS[key] for key in TARGET_KEYS):
        return None
    return transforms


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
        grid[top + dr, left + dc] = MARKER


def _place_markers(grid: np.ndarray, anchor_r: int, anchor_c: int, flip_ud: bool, flip_lr: bool) -> None:
    if flip_lr:
        _paint_arrow(grid, anchor_r - 4, anchor_c, RIGHT_ARROW)
    if flip_ud:
        _paint_arrow(grid, anchor_r, anchor_c - 4, DOWN_ARROW)


def _build_output(
    patch: np.ndarray,
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
    for key, size in target_sizes.items():
        if key == (0, 0):
            continue
        ar, ac = anchors[key]
        paste(out, np.full((size, size), RED, dtype=int), ar, ac)
    for key in TARGET_KEYS:
        patch_s = _transform_patch(patch, transforms[key][0], transforms[key][1], target_sizes[key])
        _stamp(out, patch_s, *anchors[key])
    return out


def _gen_attempt() -> np.ndarray | None:
    target_sizes = {
        (0, 0): SOURCE_ANCHOR_SIZE,
        (0, 1): random.randint(*TARGET_SIZE_RANGE),
        (1, 0): random.randint(*TARGET_SIZE_RANGE),
        (1, 1): random.randint(*TARGET_SIZE_RANGE),
    }

    transforms = _choose_transforms()
    if transforms is None:
        return None

    needs_block = any(s % SOURCE_ANCHOR_SIZE != 0 for k, s in target_sizes.items() if k != (0, 0))
    source = _source_patch(needs_block)
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source
    if not _is_asymmetric(patch):
        return None

    scaled = {
        key: _transform_patch(patch, transforms[key][0], transforms[key][1], target_sizes[key])
        for key in TARGET_KEYS
    }

    anchors: dict[tuple[int, int], tuple[int, int]] = {}
    tl_r0 = random.randint(0, MID - patch.shape[0])
    tl_c0 = random.randint(0, MID - patch.shape[1])
    anchors[(0, 0)] = (tl_r0 + anchor_r_in_patch, tl_c0 + anchor_c_in_patch)
    for key in TARGET_KEYS:
        flip_ud, flip_lr = transforms[key]
        anchor = _choose_anchor(key[0], key[1], target_sizes[key], scaled[key], flip_lr, flip_ud)
        if anchor is None:
            return None
        anchors[key] = anchor

    out = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, transforms)
    old_out = _build_output(patch, tl_r0, tl_c0, target_sizes, anchors, SPATIAL_DEFAULTS)
    if int(np.count_nonzero(out != old_out)) < MIN_VARIANT_DIFF:
        return None

    inp = out.copy()
    for key in TARGET_KEYS:
        q_r0, q_r1, q_c0, q_c1 = QUADS[key]
        inp[q_r0 : q_r1 + 1, q_c0 : q_c1 + 1] = BG
        ar, ac = anchors[key]
        paste(inp, np.full((target_sizes[key], target_sizes[key]), RED, dtype=int), ar, ac)
        _place_markers(inp, ar, ac, transforms[key][0], transforms[key][1])

    return inp if np.array_equal(main(inp), out) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 D12 input")
