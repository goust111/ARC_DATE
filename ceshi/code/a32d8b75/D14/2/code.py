# Variant of a32d8b75
# Dimension: D14 — 分隔符的结构 (meta)
# Change: 全高颜色6列作为图例与画布之间的分隔符... → 多列宽度分隔符

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite, scale_sprite
from scipy.ndimage import label as _scipy_label


# Design analysis for the D14 variant
# a) Concrete behavior:
#    The old puzzle used a single full-height color-6 column as a pure boundary.
#    Here that boundary becomes a run of 1-3 full-height color-6 columns. The run
#    width is the hidden integer s. The visible legend contents stay structurally
#    the same: unit sprite, frame mask, 5x5 anchor square, 3x3 rotation patch.
#    The changed behavior happens when assembling the composite: old logic tiled the
#    raw NxN unit into every active frame cell; new logic first scales that unit to
#    (N*s)x(N*s), then tiles the scaled unit. Example: if a 2x2 unit and a 3x4 frame
#    were rotated to shape 4x3, the old composite size was 8x6, while with a
#    separator width s=3 it becomes 24x18. The frame, anchor, and rotation markers
#    are read exactly as before; only the separator now contributes an extra inferred
#    parameter.
# b) Complexity calibration:
#    To keep the rule visible but controlled, s only has three values: 1, 2, 3.
#    All separator runs in one instance share the same width, so the user sees
#    repeated support for one hidden parameter rather than unrelated local widths.
#    The output difference is large enough to notice because composite dimensions
#    scale by s, not by a tiny color tweak.
# c) Old assumptions that break:
#    - The old generator hard-coded legend+1+canvas(+1+legend), so separator width
#      was ignored in layout. This must become legend+s+canvas(+s+legend).
#    - The old frame-fit checks used rh*N and rw*N. In the variant they must use
#      rh*(N*s) and rw*(N*s), otherwise generated composites would overflow.
#    - The old solver built the composite directly from unit_sprite.shape; that
#      silently assumes the separator carries no semantic load. The variant must
#      infer s from the separator run width and scale the unit before tiling.
#    - The old solver already used left_sep[1]+1 and right_sep[0], which is safely
#      width-agnostic. That part remains valid and is deliberately preserved.
# d) Placement legality:
#    The generator enumerates only legends whose rotated frame can fit after applying
#    the hidden scale. Anchors are chosen only from placements that stay in bounds,
#    and in the 2-legend case bounding boxes are forced not to overlap. Right-side
#    marker panels may borrow separator columns on their left, but the solver still
#    reads them from the same 5-column marker-aligned window, so parsing stays valid.
# e) Visual observability:
#    Across examples the user can compare otherwise similar legend instructions with
#    different separator widths and notice that thicker separators produce larger
#    composites. This uses a geometric cue, not any invisible ordering on color ids.
# f) Visual cleanliness:
#    The separator is still a uniform color-6 slab, so the grid remains easy to scan.
#    Only one extra idea is added: its width matters. The legend contents are not
#    overprinted with extra symbols, avoiding clutter.
# g) Color inferability:
#    No new color mapping is introduced. Colors keep their old roles, so the only new
#    inference burden is width→scale, which is purely spatial and directly visible.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    cols6.sort()
    runs: list[tuple[int, int]] = []
    s = e = cols6[0]
    for c in cols6[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
    h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _separator_scale(left_sep: tuple[int, int], right_sep: tuple[int, int] | None) -> int:
    """Infer the hidden scale from separator width and verify consistency."""
    scale = left_sep[1] - left_sep[0] + 1
    assert 1 <= scale <= 3, "separator width must encode scale 1..3"
    if right_sep is not None:
        right_scale = right_sep[1] - right_sep[0] + 1
        assert right_scale == scale, "all separator runs must share one width code"
    return scale


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[4, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, 4] == 6)
            ):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Unit sprite is the top-most 4-connected component of pixels not in {0,6}."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    st = [(rmin, cseed)]
    vis[rmin, cseed] = True
    pts: list[tuple[int, int]] = []
    while st:
        r, c = st.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                st.append((rr, cc))
    r0, c0, r1, c1 = bbox_coords(np.asarray(pts))
    spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
    cols = sorted({int(x) for x in np.unique(spr) if int(x) not in (0, 6)})
    if len(cols) == 2:
        a, b = cols
        spr2 = spr.copy()
        spr2[spr == a] = -1
        spr2[spr == b] = a
        spr2[spr2 == -1] = b
        spr = spr2
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels live between the unit sprite and the marker square."""
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    r0 = int(pts[:, 0].min())
    r1 = int(pts[:, 0].max())
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (r0, c0, r1, c1)


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the bottom 3x3 below the bordered square.
    Use the furthest non-{0,6} pixel from the center to read up/right/down/left.
    """
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        if dr < 0:
            return 0
        if dr > 0:
            return 2
    else:
        if dc > 0:
            return 1
        if dc < 0:
            return 3
    raise AssertionError("rotation marker pixel at center, direction ambiguous")


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """
    anchor_idx in {0,1,2} selects position on a length:
      0 -> start, 1 -> middle, 2 -> end
    Returns the top-left coordinate so that the corresponding point of the placed object
    aligns with that point on the canvas.
    """
    if anchor_idx <= 0:
        canvas_anchor = 0
        obj_anchor = 0
    elif anchor_idx >= 2:
        canvas_anchor = max(0, size - 1)
        obj_anchor = max(0, place_size - 1)
    else:
        canvas_anchor = (size - 1) // 2 if size > 1 else 0
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _composite_from_legend_panel(g: np.ndarray, marker_r0: int, marker_c0: int, sep_scale: int) -> np.ndarray | None:
    """Read one legend panel and build its rotated composite using the inferred scale."""
    h, w = g.shape
    if marker_c0 + 5 > w:
        return None
    panel = g[:, marker_c0 : marker_c0 + 5]
    unit = _extract_unit_sprite(panel)
    if unit is None:
        return None
    unit_sprite, unit_bb = unit
    frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
    if frame_bb is None:
        return None
    fr0, fc0, fr1, fc1 = frame_bb
    frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
    frame_mask0 = (frame != 0) & (frame != 6)
    _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
    assert n8 == 1, f"frame must be 8-connected, got {n8} components"

    dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
    frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
    scaled_unit = scale_sprite(unit_sprite, sep_scale)
    m_unit = (scaled_unit != 0) & (scaled_unit != 6)
    uh, uw = scaled_unit.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = scaled_unit[m_unit]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas using legend-panel instructions plus separator-width coding.

    Input structure:
    - Full-height color-6 separator runs split the grid into legend area(s) and the central canvas, and every separator run in one puzzle has the same width 1-3.
    - Each legend panel still contains a unit sprite, a frame template, a 5x5 bordered anchor square, and a 3x3 rotation clock-hand patch.

    Rule:
    - Infer the hidden scale from the separator width, scale the unit sprite by that factor, then tile the scaled unit into every active frame cell to build the composite.
    - Rotate the composite from the clock-hand marker and place it on the canvas using the anchor encoded by the marker pixel inside the bordered square.

    Key insight:
    - The separator is no longer just spacing: its thickness is the extra parameter that tells you how large each repeated unit should become before the frame is expanded.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    sep_scale = _separator_scale(left_sep, right_sep)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue

        comp = _composite_from_legend_panel(g, marker_r0, marker_c0, sep_scale)
        if comp is None:
            continue
        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


# --------------- generator (separator_width_scales_units) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
SEPARATOR_WIDTH_RANGE = (1, 3)
SEPARATOR_WIDTH_WEIGHTS = [4, 6, 1]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300


def _allocate_colors(num_legends):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

    bg_cands = [c for c in pool if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    bg_colors = (bg_cands[0], bg_cands[1])

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _legend_has_fit(N: int, sep_scale: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Check whether any frame/rotation combination can fit after scaling."""
    avail = marker_r0 - (N + 1)
    scaled_n = N * sep_scale
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * scaled_n <= grid_h and rw * scaled_n <= canvas_w:
                    return True
    return False


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w, sep_scale):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((N, N), uc1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = uc2

    legend_sprite = _color_swap(sprite, uc1, uc2)
    scaled_sprite = scale_sprite(sprite, sep_scale)

    avail = marker_r0 - (N + 1)
    combos = []
    scaled_n = N * sep_scale
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * scaled_n <= grid_h and rw * scaled_n <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_H,
        frame_W,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        scaled_sprite=scaled_sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        sep_scale=sep_scale,
    )
    return True


def _comp_size(leg):
    """Return (height, width) of the composite after rotation and hidden scaling."""
    rf = leg["rotated_frame"]
    sh, sw = leg["scaled_sprite"].shape
    return rf.shape[0] * sh, rf.shape[1] * sw


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap for 2-legend case. Returns False on failure."""
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        legends[0].update(
            pr=pr,
            pc=pc,
            comp_top=_anchor_to_top_left(pr, grid_h, ch),
            comp_left=_anchor_to_top_left(pc, canvas_w, cw),
        )
        return True

    a, b = legends[0], legends[1]
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_ch)
            al = _anchor_to_top_left(apc, canvas_w, a_cw)
            if at < 0 or al < 0 or at + a_ch > grid_h or al + a_cw > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    if bt < 0 or bl < 0 or bt + b_ch > grid_h or bl + b_cw > canvas_w:
                        continue
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
    """Build a canvas with alternating horizontal or vertical color bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2])
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas, legends):
    """Stamp each legend's tiled composite onto the canvas (non-{0,6} pixels overwrite)."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["scaled_sprite"], leg["rotated_frame"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape
        sh, sw = sp.shape

        comp = np.zeros((fh * sh, fw * sw), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * sh, j * sw
                    comp[r0 : r0 + sh, c0 : c0 + sw][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * sh, l : l + fw * sw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _candidate_n_sets(num_legends: int, sep_scale: int, grid_h: int, marker_r0: int) -> list[tuple[int, ...]]:
    """Enumerate unit-size choices that leave a feasible canvas and at least one frame fit."""
    out: list[tuple[int, ...]] = []
    for n1 in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
        left_lw = n1 + 2
        if num_legends == 1:
            canvas_w = GRID_W - left_lw - sep_scale
            if canvas_w < MIN_CANVAS_W:
                continue
            if _legend_has_fit(n1, sep_scale, marker_r0, grid_h, canvas_w):
                out.append((n1,))
            continue

        for n2 in range(UNIT_SIZE_RANGE[0], UNIT_SIZE_RANGE[1] + 1):
            right_lw = n2 + 2
            canvas_w = GRID_W - left_lw - sep_scale - sep_scale - right_lw
            if canvas_w < MIN_CANVAS_W:
                continue
            if not _legend_has_fit(n1, sep_scale, marker_r0, grid_h, canvas_w):
                continue
            if not _legend_has_fit(n2, sep_scale, marker_r0, grid_h, canvas_w):
                continue
            out.append((n1, n2))
    return out


def _place_legend(grid, leg, legend_start, sep_start, sep_scale, marker_r0, is_right):
    """Draw one legend panel with a variable-width separator."""
    N = leg["N"]
    grid[:, sep_start : sep_start + sep_scale] = 6

    if not is_right:
        mc0 = legend_start + max(0, N - 4)
    else:
        mc0 = legend_start + N - 4

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sprite_start = legend_start + 1
    grid[1 : 1 + N, sprite_start : sprite_start + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    sep_scale = random.choices(
        list(range(SEPARATOR_WIDTH_RANGE[0], SEPARATOR_WIDTH_RANGE[1] + 1)),
        weights=SEPARATOR_WIDTH_WEIGHTS,
    )[0]

    if sep_scale == 3:
        num_legends = 1
    else:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    n_sets = _candidate_n_sets(num_legends, sep_scale, grid_h, marker_r0)
    if not n_sets:
        return None
    chosen_ns = random.choice(n_sets)
    legends = [{"N": N, "legend_w": N + 2} for N in chosen_ns]

    left_lw = legends[0]["legend_w"]
    right_total = (sep_scale + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_lw - sep_scale - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            sep_scale,
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_lw + sep_scale
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(
        grid,
        legends[0],
        legend_start=0,
        sep_start=left_lw,
        sep_scale=sep_scale,
        marker_r0=marker_r0,
        is_right=False,
    )
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(
            grid,
            legends[1],
            legend_start=r_sep + sep_scale,
            sep_start=r_sep,
            sep_scale=sep_scale,
            marker_r0=marker_r0,
            is_right=True,
        )

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
