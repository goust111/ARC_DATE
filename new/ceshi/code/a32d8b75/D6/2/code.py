# Variant of a32d8b75
# Dimension: D6 — 符号编码：位置标记的编码方式 (symbol_instructions)
# Change: 5x5边框6方块内3x3区域中的单个颜色4像素，行列位置(0-2)编码锚点对齐（起始/中间/结束）... → 相对偏移编码：标记像素位置直接表示画布上的像素级偏移

"""
Design analysis for the D6 variant
==================================

a) Concrete behavior and trace
spec_like:
  unchanged_core:
    - Legends still encode a two-color unit sprite, an 8-connected frame mask, and a rotation mark.
    - The solver still color-swaps the unit sprite, tiles it into every active frame cell, rotates the frame layout, and overlays the result onto the canvas.
  D6_change:
    - The old position symbol was a 5x5 border-6 square with a 3x3 inner field.
    - The new symbol is a 7x7 border-6 square with a 5x5 inner field.
    - Exactly one color-4 marker pixel appears inside that 5x5 field.
    - Its row/col no longer mean start/middle/end anchors. They now mean a proportional top-left offset:
      top = round_half_up((marker_row / 4) * (canvas_h - comp_h))
      left = round_half_up((marker_col / 4) * (canvas_w - comp_w))
    - Example on one axis:
      old logic with comp_h=8 and canvas_h=20 gave only three legal top values: 0, 6, 12.
      new logic gives five values: 0, 3, 6, 9, 12.
      A marker at inner row 1 visibly means "one quarter down", not "middle".
  flow_impact:
    - solver impact: marker-square detection, marker decoding, and placement math.
    - generator impact: legend width must grow to fit the 7x7 marker square; placement assignment must choose 5x5-relative positions and reject cases that collapse back to old three-anchor behavior.

b) Complexity calibration
  - Using a fixed 5x5 inner field yields 25 visible symbol positions, which is richer than the old 9 but still simple enough for ARC.
  - The generator only changes D6-related structure: the square becomes larger and the placement grid becomes denser; sprite/frame/rotation logic stays the same.
  - To keep the new rule observable, generated samples force each legend to use at least one odd marker coordinate (1 or 3), so the marker is not visually restricted to the old coarse lattice.

c) Implicit assumption audit for the original main()
  - `_all_marker_panel_squares()` assumed a 5x5 bordered square and a 3x3 inner marker field. That no longer holds, so detection must use 7x7 / 5x5 and the lower rotation band must move down accordingly.
  - `_position_indices_from_border_square()` assumed the decoded values were in 0..2 and directly usable as anchor classes. That breaks under continuous offsets.
  - `_anchor_to_top_left()` depended on the semantic meaning "start/middle/end". The new variant needs a different mapping based on remaining travel distance, not anchor alignment.
  - The rest of the assembly pipeline does not depend on the old D6 assumption, so it can remain unchanged.

d) Placement legality
  - The new offset formula always places the composite fully inside the effective canvas by mapping into the available slack `(canvas_size - comp_size)`.
  - For two legends, the generator enumerates all relative-offset candidates and keeps only non-overlapping pairs.
  - The solver still asserts no pixel-wise overlap between composites.

e) Visual observability
  - The larger 7x7 symbol is visibly different from the source task.
  - The marker often lands on quarter positions rather than only corners/center, so the output placement differs in ways a human can see from a few examples.
  - The generator explicitly rejects placements that are simultaneously identical to an old anchor placement on both axes.

f) Visual cleanliness
  - The marker square occupies the full legend width, so there is no ambiguous partial overlap with sprite/frame content.
  - Rotation remains in its own 3-row band beneath the square.
  - Composite overlap is forbidden, keeping the output readable.

g) Color inference
  - No new color mapping is introduced.
  - Color 4 still marks position, color 7 still marks rotation, color 6 still marks separators/borders, so the new rule is geometric rather than color-dependent.

h) New assumptions introduced by this variant
  - main() now assumes each marker panel uses a 7x7 border square with exactly one inner marker pixel.
  - main() also assumes the rotation patch is centered under that wider square.
  - generate_input() guarantees both conditions by construction.
"""

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


MARKER_INNER_SIZE = 5
MARKER_BOX_SIZE = MARKER_INNER_SIZE + 2
ROTATION_PATCH_HEIGHT = 3
MARKER_PANEL_HEIGHT = MARKER_BOX_SIZE + ROTATION_PATCH_HEIGHT


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _h, w = g.shape
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
    _h, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    right_min_start = w - (MARKER_BOX_SIZE + 3)
    for s, e in reversed(runs[1:]):
        if s >= right_min_start and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 7x7 border-6 squares that look like the top of a marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - MARKER_PANEL_HEIGHT + 1):
        for c0 in range(w - MARKER_BOX_SIZE + 1):
            sub = g[r0 : r0 + MARKER_BOX_SIZE, c0 : c0 + MARKER_BOX_SIZE]
            if not (
                np.all(sub[0, :] == 6)
                and np.all(sub[-1, :] == 6)
                and np.all(sub[:, 0] == 6)
                and np.all(sub[:, -1] == 6)
            ):
                continue
            inner = sub[1:-1, 1:-1]
            pts = np.argwhere((inner != 0) & (inner != 6))
            if pts.shape[0] != 1:
                continue
            rot_band = g[r0 + MARKER_BOX_SIZE : r0 + MARKER_PANEL_HEIGHT, c0 : c0 + MARKER_BOX_SIZE]
            if rot_band.shape != (ROTATION_PATCH_HEIGHT, MARKER_BOX_SIZE):
                continue
            if not (np.all(rot_band[:, 0] == 6) and np.all(rot_band[:, -1] == 6)):
                continue
            rot = rot_band[:, 2:5]
            if rot.shape != (3, 3) or np.count_nonzero((rot != 0) & (rot != 6)) == 0:
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    # Unit sprite is the top-most 4-connected component of pixels not in {0,6}.
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
    # Frame pixels live between the unit sprite and the border-6 marker square.
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


def _marker_relative_indices(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = g[border6_r0 : border6_r0 + MARKER_BOX_SIZE, border6_c0 : border6_c0 + MARKER_BOX_SIZE]
    inner = sq[1:-1, 1:-1]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.shape[0] != 1:
        return None
    return int(pts[0, 0]), int(pts[0, 1])  # both in 0..4


def _rotation_from_clockhand(g: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the centered 3x3 patch directly below the 7x7 position square.
    Use the furthest non-{0,6} pixel from the patch center to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
    """
    h, w = g.shape
    r0 = border6_r0 + MARKER_BOX_SIZE
    c0 = border6_c0 + 2
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
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


def _relative_offset_to_top_left(marker_idx: int, size: int, place_size: int) -> int:
    """
    Convert a 0..4 marker coordinate into a proportional top-left offset.

    The marker value chooses a percentage of the remaining travel distance:
      0 -> 0%, 1 -> 25%, 2 -> 50%, 3 -> 75%, 4 -> 100%.
    """
    travel = size - place_size
    assert travel >= 0, "composite must fit the canvas"
    denom = MARKER_INNER_SIZE - 1
    return (marker_idx * travel + denom // 2) // denom


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """
    Legacy start/middle/end anchor mapping retained only for degeneration checks in the generator.
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legend panels and place them onto the canvas.

    Input structure:
    - Color-6 separator columns still split the full grid into legend panel(s) and the canvas.
    - Each legend still contains a unit sprite, a frame mask, a bordered position square, and a rotation cue.
    - The position square is now 7x7 overall, with a single marker pixel somewhere inside its 5x5 interior.

    Rule:
    - Read and color-swap the unit sprite, tile it into every active frame cell, and rotate the frame layout according to the clock-hand mark.
    - Decode the marker pixel as a relative offset instead of a three-way anchor: its row/col inside the 5x5 interior specify quarter-step percentages of the remaining vertical and horizontal travel room.
    - Place each rotated composite at that proportional top-left offset on the canvas.

    Key insight:
    - The new symbol no longer says "align to top / center / bottom". It directly encodes how far the composite should slide across the available canvas slack, using the marker's exact position in the larger inner grid.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas_region = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas_region.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas_region[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas_region[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas_region.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue

        if marker_c0 + MARKER_BOX_SIZE > w:
            continue
        panel = g[:, marker_c0 : marker_c0 + MARKER_BOX_SIZE]

        unit = _extract_unit_sprite(panel)
        if unit is None:
            continue
        unit_sprite, unit_bb = unit

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

        m_unit = (unit_sprite != 0) & (unit_sprite != 6)
        uh, uw = unit_sprite.shape
        fh, fw = frame_mask.shape
        comp = np.zeros((fh * uh, fw * uw), dtype=int)
        for i in range(fh):
            for j in range(fw):
                if not frame_mask[i, j]:
                    continue
                rr0 = i * uh
                cc0 = j * uw
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = unit_sprite[m_unit]

        pos = _marker_relative_indices(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _relative_offset_to_top_left(pr, eff_h, comp.shape[0])
        left = _relative_offset_to_top_left(pc, eff_w, comp.shape[1])

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


# --------------- generator (relative_offset_marker_square) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


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


def _color_swap(sprite, c1, c2):
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((N, N), uc1, dtype=int)
    num_c2 = random.randint(1, N * N - 1)
    for p in random.sample(range(N * N), num_c2):
        sprite[p // N, p % N] = uc2

    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (N + 1)
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            if fW > leg["legend_w"]:
                continue
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * N <= grid_h and rw * N <= canvas_w:
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
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
    )
    return True


def _comp_size(leg):
    """Return (height, width) of the composite after rotation."""
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _legacy_axis_positions(size: int, place_size: int) -> set[int]:
    return {_anchor_to_top_left(k, size, place_size) for k in range(3)}


def _relative_candidates(leg, grid_h, canvas_w):
    """
    Candidate (marker_row, marker_col, top, left) tuples.

    We reject cases that visually or behaviorally collapse to the original 3-anchor task:
    - marker must use at least one odd coordinate in the 5x5 inner grid
    - resulting placement cannot match a legacy anchor placement on both axes at once
    """
    ch, cw = _comp_size(leg)
    legacy_t = _legacy_axis_positions(grid_h, ch)
    legacy_l = _legacy_axis_positions(canvas_w, cw)
    candidates = []
    for mr in range(MARKER_INNER_SIZE):
        for mc in range(MARKER_INNER_SIZE):
            if mr % 2 == 0 and mc % 2 == 0:
                continue
            top = _relative_offset_to_top_left(mr, grid_h, ch)
            left = _relative_offset_to_top_left(mc, canvas_w, cw)
            if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
                continue
            if top in legacy_t and left in legacy_l:
                continue
            candidates.append((mr, mc, top, left))
    return candidates


def _assign_relative_offsets(legends, grid_h, canvas_w):
    """Assign 5x5-relative offsets to legends, avoiding overlap for 2-legend cases."""
    if len(legends) == 1:
        cands = _relative_candidates(legends[0], grid_h, canvas_w)
        if not cands:
            return False
        mr, mc, top, left = random.choice(cands)
        legends[0].update(marker_r=mr, marker_c=mc, comp_top=top, comp_left=left)
        return True

    cand_lists = [_relative_candidates(leg, grid_h, canvas_w) for leg in legends]
    if not cand_lists[0] or not cand_lists[1]:
        return False

    a, b = legends[0], legends[1]
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    valid_pairs = []
    for amr, amc, at, al in cand_lists[0]:
        for bmr, bmc, bt, bl in cand_lists[1]:
            if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                continue
            valid_pairs.append((amr, amc, at, al, bmr, bmc, bt, bl))
    if not valid_pairs:
        return False
    amr, amc, at, al, bmr, bmc, bt, bl = random.choice(valid_pairs)
    a.update(marker_r=amr, marker_c=amc, comp_top=at, comp_left=al)
    b.update(marker_r=bmr, marker_c=bmc, comp_top=bt, comp_left=bl)
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
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        N = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * N, fw * N), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * N, j * N
                    comp[r0 : r0 + N, c0 : c0 + N][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * N, l : l + fw * N]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid, leg, col_start, sep_col, marker_r0):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    N = leg["N"]
    legend_w = leg["legend_w"]
    grid[:, sep_col] = 6

    mc0 = col_start

    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0 : mc0 + MARKER_BOX_SIZE] = 0
    grid[marker_r0, mc0 : mc0 + MARKER_BOX_SIZE] = 6
    grid[marker_r0 + MARKER_BOX_SIZE - 1, mc0 : mc0 + MARKER_BOX_SIZE] = 6
    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0] = 6
    grid[marker_r0 : marker_r0 + MARKER_BOX_SIZE, mc0 + MARKER_BOX_SIZE - 1] = 6
    grid[marker_r0 + 1 + leg["marker_r"], mc0 + 1 + leg["marker_c"]] = 4

    rr0 = marker_r0 + MARKER_BOX_SIZE
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0 : mc0 + MARKER_BOX_SIZE] = 0
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0] = 6
    grid[rr0 : rr0 + ROTATION_PATCH_HEIGHT, mc0 + MARKER_BOX_SIZE - 1] = 6
    cx_r, cx_c = rr0 + 1, mc0 + MARKER_BOX_SIZE // 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_left = col_start + (legend_w - N) // 2
    grid[1 : 1 + N, sp_left : sp_left + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = col_start + random.randint(0, legend_w - fW)
    frame_region = grid[fr_start : fr_start + fH, f_col : f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        N = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": N, "legend_w": max(N + 2, MARKER_BOX_SIZE)})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    if marker_r0 <= 2:
        return None

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_relative_offsets(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0)

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
