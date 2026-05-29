# Combo variant of b10624e5 (index 12)
# combo_spec: D2/1+D12/0
# Name: 控制像素指定源象限 + 网格边缘缩放因子图例

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste


# ---------------------------------------------------------------------------
# Design analysis for the fused D2/1 + D12/0 variant
#
# a) Concrete behavior and trace
# - The board is still a 27x27 grid split into four 13x13 quadrants by one
#   full divider row and one full divider column.
# - Every quadrant now has the same 2x2 red anchor size, so anchor size can no
#   longer encode scale.
# - Exactly one quadrant contains the full colored source motif attached to its
#   2x2 red anchor. The other three quadrants contain only their 2x2 anchors in
#   the input.
# - The outer border carries five instruction components of one reserved color:
#   two singleton control pixels and three straight legend bars.
#   * the two singleton pixels decode the source quadrant exactly as in D2/1:
#     one on a horizontal edge fixes top/bottom, one on a vertical edge fixes
#     left/right.
#   * the three bars encode target scale factors by length; each bar belongs to
#     one quadrant-specific border slot.
# - Small trace:
#   * if the source controls point to BR, the solver must extract the BR motif
#     rather than anything in TL.
#   * if the TL legend bar on the left edge has length 3, the BR source motif
#     reflected into TL must be enlarged by 3 even though the TL anchor is still
#     only 2x2.
#   * the old solver would fail twice: it would read the wrong source quadrant
#     and it would infer scale 1 from equal anchor sizes.
#
# b) Complexity calibration
# - All border instructions share one color so the solver must separate them by
#   geometry, not by color: size-1 components are source controls, longer
#   straight components are legends.
# - Legend factors are restricted to {2, 3}. This keeps the symbol vocabulary
#   small, avoids ambiguity with 1-pixel controls, and still makes scale visibly
#   different across targets.
# - The three target factors are sampled from patterns [2,2,3] or [2,3,3], so
#   every example contains both repeated evidence and visible variation.
#
# c) Hidden assumptions from the original solver that break here
# - "TL is always the source" no longer holds; the source quadrant is decoded
#   from border controls.
# - "Anchor size tells the scale ratio" no longer holds; all anchors are fixed
#   2x2 and scale comes only from legend-bar length.
# - Flips can no longer be keyed to absolute quadrant location relative to TL;
#   they must be keyed to whether the target is in a different row half and/or
#   column half from the decoded source quadrant.
# - Border scans can no longer treat all non-bg border pixels alike because the
#   same color now carries two semantics that must be split by component shape.
#
# d) Placement legality
# - The generator keeps all anchors and motifs inside an interior 12x12 usable
#   area for each quadrant, leaving the outer border free for instructions.
# - Legend bars occupy quadrant-specific edge slots:
#   TL -> left edge top half, TR -> top edge right half,
#   BL -> bottom edge left half, BR -> right edge bottom half.
# - This slot layout guarantees that control pixels and legend bars never touch
#   even though they share a color.
# - Target anchors are chosen only when the full reflected+scaled motif fits in
#   the usable interior, so no instruction overlap or out-of-bounds clipping is
#   needed in valid generated examples.
#
# e) Visual observability
# - The source quadrant varies across all four quadrants, so the control pixels
#   visibly matter.
# - All anchors have the same size while copied motifs still appear at two
#   different scales, so the legend bars visibly matter.
# - Source motifs are forced to be asymmetric under both LR and UD reflection,
#   making the relative-flip rule human-visible.
#
# f) Visual cleanliness
# - Border instructions are sparse and isolated from copied content by the
#   1-cell border margin inside each quadrant.
# - Using one reserved instruction color and simple shapes keeps the border
#   readable despite carrying two symbol systems.
#
# g) Color-rule audit
# - No logic depends on color ordering or numeric color ids.
# - The instruction color is excluded from motif generation, so the only way to
#   interpret the border marks is through visible position and geometry.
#
# New solver assumptions guaranteed by generate_input()
# - There are exactly five border instruction components of color 8:
#   two singleton controls and three straight contiguous bars of length 2 or 3.
# - Each quadrant contains exactly one 2x2 red anchor.
# - Exactly one quadrant contains a single non-background component with red:
#   the source motif attached to that quadrant's anchor.
# - Legend keys are exactly the three non-source quadrants, and their bars lie
#   in the dedicated edge slots listed above.
# ---------------------------------------------------------------------------


BG = 4
DIV = 1
RED = 2
INSTR = 8
GRID_SIZE = 27
MID = 13
ANCHOR_SIZE = 2
MAX_SCALE = 3
MAX_RETRIES = 400

QUADS = {
    (0, 0): (0, 12, 0, 12),
    (0, 1): (0, 12, 14, 26),
    (1, 0): (14, 26, 0, 12),
    (1, 1): (14, 26, 14, 26),
}

# Leave every outer-border cell free for instruction symbols.
USABLE_QUADS = {
    (0, 0): (1, 12, 1, 12),
    (0, 1): (1, 12, 14, 25),
    (1, 0): (14, 25, 1, 12),
    (1, 1): (14, 25, 14, 25),
}

HALF_CENTER = {0: 6, 1: 20}
SCALE_PATTERNS = ([2, 2, 3], [2, 3, 3])
FREEFORM_H_RANGE = (3, 4)
FREEFORM_W_RANGE = (3, 4)
FILL_RATIO_MIN = 0.6
FILL_RATIO_MAX = 0.85
MOTIF_COLORS_MIN = 3
MOTIF_COLORS_MAX = 5
COLOR_POOL = [0, 3, 5, 6, 7, 9]


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


def _border_coords(h: int, w: int) -> list[tuple[int, int]]:
    coords = [(0, c) for c in range(w)]
    coords += [(h - 1, c) for c in range(w)]
    coords += [(r, 0) for r in range(1, h - 1)]
    coords += [(r, w - 1) for r in range(1, h - 1)]
    return coords


def _scale_integer(img: np.ndarray, factor: int) -> np.ndarray:
    factor = int(factor)
    assert 2 <= factor <= MAX_SCALE
    return np.repeat(np.repeat(img, factor, axis=0), factor, axis=1)


def _quad_view(g: np.ndarray, key: tuple[int, int], mid_r: int, mid_c: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    h, w = g.shape
    bounds = {
        (0, 0): (0, mid_r - 1, 0, mid_c - 1),
        (0, 1): (0, mid_r - 1, mid_c + 1, w - 1),
        (1, 0): (mid_r + 1, h - 1, 0, mid_c - 1),
        (1, 1): (mid_r + 1, h - 1, mid_c + 1, w - 1),
    }[key]
    r0, r1, c0, c1 = bounds
    return g[r0 : r1 + 1, c0 : c1 + 1], bounds


def _decode_symbols(g: np.ndarray, mid_r: int, mid_c: int) -> tuple[tuple[int, int], dict[tuple[int, int], int], np.ndarray]:
    """Split same-color border symbols into singleton controls and bar legends."""
    h, w = g.shape
    instr_mask = g == INSTR
    controls: list[tuple[int, int]] = []
    factors: dict[tuple[int, int], int] = {}

    for pts in components_mask(instr_mask, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        size = len(pts)
        if size == 1:
            controls.append((int(pts[0][0]), int(pts[0][1])))
            continue

        if r0 == r1:
            assert len({int(r) for r, _c in pts}) == 1, "horizontal legend must stay on one row"
            assert size == c1 - c0 + 1, "horizontal legend must be contiguous"
        elif c0 == c1:
            assert len({int(c) for _r, c in pts}) == 1, "vertical legend must stay on one col"
            assert size == r1 - r0 + 1, "vertical legend must be contiguous"
        else:
            raise AssertionError("legend bars must be straight")

        factor = max(r1 - r0 + 1, c1 - c0 + 1)
        assert 2 <= factor <= MAX_SCALE, "legend length must encode a valid scale"

        if c0 == 0 and c1 == 0 and r1 < mid_r:
            key = (0, 0)
        elif r0 == 0 and r1 == 0 and c0 > mid_c:
            key = (0, 1)
        elif r0 == h - 1 and r1 == h - 1 and c1 < mid_c:
            key = (1, 0)
        elif c0 == w - 1 and c1 == w - 1 and r0 > mid_r:
            key = (1, 1)
        else:
            raise AssertionError("legend bar must occupy a dedicated quadrant edge slot")
        assert key not in factors, "each quadrant slot can contain at most one legend"
        factors[key] = factor

    assert len(controls) == 2, "must have exactly two singleton control pixels"

    src_keys = set()
    seen_h_edge = False
    seen_v_edge = False
    for r, c in controls:
        if r == 0 or r == h - 1:
            seen_h_edge = True
            qb = 0 if r == 0 else 1
            qr = 0 if c < mid_c else 1
        else:
            assert c == 0 or c == w - 1, "control pixels must lie on the outer border"
            seen_v_edge = True
            qb = 0 if r < mid_r else 1
            qr = 0 if c == 0 else 1
        src_keys.add((qb, qr))

    assert seen_h_edge and seen_v_edge, "controls must include one horizontal-edge and one vertical-edge mark"
    assert len(src_keys) == 1, "the two control pixels must decode the same source quadrant"
    src_key = next(iter(src_keys))
    assert set(factors) == (set(QUADS) - {src_key}), "legend bars must exist for exactly the non-source quadrants"
    return src_key, factors, instr_mask


def _find_anchors(g: np.ndarray, mid_r: int, mid_c: int) -> dict[tuple[int, int], tuple[int, int, int, int]]:
    anchors: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for pts in components_mask(g == RED, connectivity=4):
        r0, c0, r1, c1 = bbox_coords(pts)
        assert r1 - r0 + 1 == ANCHOR_SIZE and c1 - c0 + 1 == ANCHOR_SIZE, "anchors must be 2x2 red squares"
        rr = (r0 + r1) // 2
        cc = (c0 + c1) // 2
        qb = 0 if rr < mid_r else 1
        qr = 0 if cc < mid_c else 1
        key = (qb, qr)
        assert key not in anchors, "each quadrant must contain exactly one red anchor"
        anchors[key] = (int(r0), int(c0), int(r1), int(c1))
    assert len(anchors) == 4, "there must be exactly four anchors"
    return anchors


def _extract_source_patch(g: np.ndarray, src_key: tuple[int, int], mid_r: int, mid_c: int, bg: int) -> np.ndarray:
    src_quad, _bounds = _quad_view(g, src_key, mid_r, mid_c)
    patch_bb = None
    for pts in components_mask(src_quad != bg, connectivity=4):
        if any(int(src_quad[r, c]) == RED for r, c in pts):
            patch_bb = tuple(map(int, bbox_coords(pts)))
            break
    assert patch_bb is not None, "source quadrant must contain one non-bg component with red"
    pr0, pc0, pr1, pc1 = patch_bb
    return src_quad[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a source motif chosen by border controls, with border legend bars setting target scale.

    Input structure:
    - One full-row/full-column divider splits the board into four quadrants.
    - Each quadrant contains a 2x2 red anchor, but only one quadrant also contains the
      attached colorful motif component.
    - The outer border carries five instruction components of one color: two singleton
      control pixels that identify the source quadrant, plus three straight legend bars
      whose lengths encode the target scale factors.

    Rule:
    - Separate the border symbols by shape: singleton components are source controls,
      longer straight components are quadrant-specific scale legends.
    - Extract the source quadrant's non-background component containing red.
    - For every other quadrant, reflect the source patch relative to the source quadrant,
      enlarge it by the legend factor for that target quadrant, align the scaled red
      region's top-left corner to the target 2x2 anchor, and stamp only non-red pixels.

    Key insight:
    - The border now carries two different symbolic encodings at once. Geometry tells you
      which edge marks choose the source and which edge marks choose the scale, while
      quadrant relation still determines whether horizontal and/or vertical reflection is
      needed.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg = get_bg(g)
    mid_r, mid_c = _find_dividers(g, bg)
    src_key, scale_factors, instr_mask = _decode_symbols(g, mid_r, mid_c)
    anchors = _find_anchors(g, mid_r, mid_c)
    patch = _extract_source_patch(g, src_key, mid_r, mid_c, bg)

    for dst_key, (anchor_r0, anchor_c0, _anchor_r1, _anchor_c1) in anchors.items():
        if dst_key == src_key:
            continue

        patch_t = patch
        if dst_key[1] != src_key[1]:
            patch_t = np.fliplr(patch_t)
        if dst_key[0] != src_key[0]:
            patch_t = np.flipud(patch_t)

        factor = scale_factors[dst_key]
        patch_s = _scale_integer(patch_t, factor)
        assert np.array_equal(patch_s[::factor, ::factor], patch_t), "integer legend scaling must be exact"

        rr_red, cc_red = np.where(patch_s == RED)
        assert rr_red.size > 0, "scaled source patch must retain a red anchor footprint"
        place_r0 = anchor_r0 - int(rr_red.min())
        place_c0 = anchor_c0 - int(cc_red.min())

        rr_p, cc_p = np.where((patch_s != bg) & (patch_s != RED))
        for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
            rr_t = place_r0 + rr0
            cc_t = place_c0 + cc0
            if not (0 <= rr_t < h and 0 <= cc_t < w):
                continue
            if rr_t == mid_r or cc_t == mid_c:
                continue
            if instr_mask[rr_t, cc_t]:
                continue
            if out[rr_t, cc_t] == RED:
                continue
            out[rr_t, cc_t] = int(patch_s[rr0, cc0])

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
    extras = random.sample(COLOR_POOL, n_colors - 1)
    patch = np.full((h, w), BG, dtype=int)
    for r, c in red_cells:
        patch[r, c] = RED
    for i, (r, c) in enumerate(non_red):
        patch[r, c] = extras[i] if i < len(extras) else random.choice(extras)
    return patch


def _source_patch() -> tuple[np.ndarray, int, int] | None:
    for _ in range(80):
        h = random.randint(*FREEFORM_H_RANGE)
        w = random.randint(*FREEFORM_W_RANGE)
        ar = random.randint(0, h - ANCHOR_SIZE)
        ac = random.randint(0, w - ANCHOR_SIZE)
        red_cells = {(ar + dr, ac + dc) for dr in range(ANCHOR_SIZE) for dc in range(ANCHOR_SIZE)}
        min_fill = max(len(red_cells) + 2, int(np.ceil(FILL_RATIO_MIN * h * w)))
        max_fill = int(np.floor(FILL_RATIO_MAX * h * w))
        if max_fill < min_fill:
            continue
        filled = _grow_4conn(h, w, red_cells, random.randint(min_fill, max_fill))
        patch = _assign_colors(filled, red_cells, h, w)
        if patch is None:
            continue
        if np.array_equal(patch, np.fliplr(patch)):
            continue
        if np.array_equal(patch, np.flipud(patch)):
            continue
        return patch, ar, ac
    return None


def _control_points(src_key: tuple[int, int]) -> list[tuple[int, int]]:
    qb, qr = src_key
    return [
        (0 if qb == 0 else GRID_SIZE - 1, HALF_CENTER[qr]),
        (HALF_CENTER[qb], 0 if qr == 0 else GRID_SIZE - 1),
    ]


def _sample_scales(src_key: tuple[int, int]) -> dict[tuple[int, int], int]:
    values = list(random.choice(SCALE_PATTERNS))
    random.shuffle(values)
    keys = [key for key in QUADS if key != src_key]
    return {key: values[i] for i, key in enumerate(keys)}


def _legend_span(key: tuple[int, int], factor: int) -> tuple[int, int, int, int]:
    if key == (0, 0):
        r0 = random.randint(1, MID - factor)
        return (r0, 0, r0 + factor - 1, 0)
    if key == (0, 1):
        c0 = random.randint(MID + 2, GRID_SIZE - factor - 1)
        return (0, c0, 0, c0 + factor - 1)
    if key == (1, 0):
        c0 = random.randint(1, MID - factor)
        return (GRID_SIZE - 1, c0, GRID_SIZE - 1, c0 + factor - 1)
    assert key == (1, 1)
    r0 = random.randint(MID + 2, GRID_SIZE - factor - 1)
    return (r0, GRID_SIZE - 1, r0 + factor - 1, GRID_SIZE - 1)


def _draw_instructions(grid: np.ndarray, src_key: tuple[int, int], scales: dict[tuple[int, int], int]) -> np.ndarray:
    instr_mask = np.zeros_like(grid, dtype=bool)
    for r, c in _control_points(src_key):
        grid[r, c] = INSTR
        instr_mask[r, c] = True
    for key, factor in scales.items():
        r0, c0, r1, c1 = _legend_span(key, factor)
        grid[r0 : r1 + 1, c0 : c1 + 1] = INSTR
        instr_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return instr_mask


def _top_left_choices(key: tuple[int, int], patch_shape: tuple[int, int]) -> list[tuple[int, int]]:
    q_r0, q_r1, q_c0, q_c1 = USABLE_QUADS[key]
    ph, pw = map(int, patch_shape)
    if q_r0 > q_r1 - ph + 1 or q_c0 > q_c1 - pw + 1:
        return []
    return [
        (r0, c0)
        for r0 in range(q_r0, q_r1 - ph + 2)
        for c0 in range(q_c0, q_c1 - pw + 2)
    ]


def _scaled_patch(patch: np.ndarray, src_key: tuple[int, int], dst_key: tuple[int, int], factor: int) -> np.ndarray:
    p = patch
    if dst_key[1] != src_key[1]:
        p = np.fliplr(p)
    if dst_key[0] != src_key[0]:
        p = np.flipud(p)
    return _scale_integer(p, factor)


def _choose_anchor(key: tuple[int, int], patch_s: np.ndarray) -> tuple[int, int] | None:
    q_r0, q_r1, q_c0, q_c1 = USABLE_QUADS[key]
    rr, cc = np.where(patch_s == RED)
    red_r0, red_c0 = int(rr.min()), int(cc.min())

    fit_r0 = q_r0 + red_r0
    fit_r1 = q_r1 - patch_s.shape[0] + 1 + red_r0
    fit_c0 = q_c0 + red_c0
    fit_c1 = q_c1 - patch_s.shape[1] + 1 + red_c0

    anch_r0, anch_r1 = q_r0, q_r1 - ANCHOR_SIZE + 1
    anch_c0, anch_c1 = q_c0, q_c1 - ANCHOR_SIZE + 1

    r_lo, r_hi = max(fit_r0, anch_r0), min(fit_r1, anch_r1)
    c_lo, c_hi = max(fit_c0, anch_c0), min(fit_c1, anch_c1)
    if r_lo > r_hi or c_lo > c_hi:
        return None
    return random.randint(r_lo, r_hi), random.randint(c_lo, c_hi)


def _stamp(out: np.ndarray, patch_s: np.ndarray, anchor_r: int, anchor_c: int, instr_mask: np.ndarray) -> None:
    rr_red, cc_red = np.where(patch_s == RED)
    place_r0 = anchor_r - int(rr_red.min())
    place_c0 = anchor_c - int(cc_red.min())

    rr_p, cc_p = np.where((patch_s != BG) & (patch_s != RED))
    for rr0, cc0 in zip(rr_p.tolist(), cc_p.tolist(), strict=False):
        rr_t = place_r0 + rr0
        cc_t = place_c0 + cc0
        if not (0 <= rr_t < GRID_SIZE and 0 <= cc_t < GRID_SIZE):
            continue
        if rr_t == MID or cc_t == MID:
            continue
        if instr_mask[rr_t, cc_t]:
            continue
        if out[rr_t, cc_t] == RED:
            continue
        out[rr_t, cc_t] = int(patch_s[rr0, cc0])


def _gen_attempt() -> np.ndarray | None:
    src_key = random.choice(list(QUADS))
    scales = _sample_scales(src_key)

    source = _source_patch()
    if source is None:
        return None
    patch, anchor_r_in_patch, anchor_c_in_patch = source

    choices = _top_left_choices(src_key, patch.shape)
    if not choices:
        return None
    src_patch_r0, src_patch_c0 = random.choice(choices)
    anchors: dict[tuple[int, int], tuple[int, int]] = {
        src_key: (src_patch_r0 + anchor_r_in_patch, src_patch_c0 + anchor_c_in_patch)
    }

    scaled = {
        key: _scaled_patch(patch, src_key, key, factor)
        for key, factor in scales.items()
    }
    for key, patch_s in scaled.items():
        pos = _choose_anchor(key, patch_s)
        if pos is None:
            return None
        anchors[key] = pos

    base = np.full((GRID_SIZE, GRID_SIZE), BG, dtype=int)
    base[MID, :] = DIV
    base[:, MID] = DIV
    instr_mask = _draw_instructions(base, src_key, scales)

    out = base.copy()
    paste(out, patch, src_patch_r0, src_patch_c0)
    for key, (ar, ac) in anchors.items():
        if key == src_key:
            continue
        paste(out, np.full((ANCHOR_SIZE, ANCHOR_SIZE), RED, dtype=int), ar, ac)
    for key, patch_s in scaled.items():
        _stamp(out, patch_s, anchors[key][0], anchors[key][1], instr_mask)

    inp = base.copy()
    paste(inp, patch, src_patch_r0, src_patch_c0)
    for key, (ar, ac) in anchors.items():
        if key == src_key:
            continue
        paste(inp, np.full((ANCHOR_SIZE, ANCHOR_SIZE), RED, dtype=int), ar, ac)

    try:
        return inp if np.array_equal(main(inp), out) else None
    except AssertionError:
        return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b10624e5 combo input")
