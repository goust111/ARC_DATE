# Variant of a32d8b75
# Dimension: D12 — 复合精灵的放置约束 (meta)
# Change: 复合精灵必须完全在画布内（无裁剪），叠加时非{0,6}像素覆写画布，画布中颜色6的像素保留... → 精灵放置后背景色覆写

import numpy as np

from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label
import random


# Design analysis for the D12 variant
# a) Concrete behavior:
#    The original composite is already a full rectangle whose inactive frame cells are 0.
#    In the base task those 0 cells are ignored during overlay, so the canvas bands remain
#    visible between tiled sprite blocks. In this variant every non-6 cell inside that
#    rectangle is overwritten by the composite value, including 0, so inactive cells carve
#    black "clear zones" out of the canvas.
#    Small trace:
#      unit = [[1,2],[2,1]]
#      frame = [[1,0,1],
#               [0,1,0]]
#      comp  = [sprite, 0-block, sprite / 0-block, sprite, 0-block]
#      base overlay keeps the old canvas color where comp==0.
#      new overlay writes 0 there, so the bounding rectangle contains explicit black gaps.
# b) Complexity calibration:
#    The legend parsing, color swap, frame extraction, rotation and anchor logic stay
#    unchanged. To make the new behavior visible, generator rejects near-solid frames and
#    prefers masks with multiple perimeter gaps; otherwise the new overwrite rule could
#    degenerate into the old visual result.
# c) Solver assumption audit:
#    In the original main(), `if val != 0 and val != 6: out[...] = val` implicitly assumed
#    background writes are harmlessly skipped. That no longer holds. The new solver writes
#    every composite pixel except onto canvas color 6, and it tracks full drawable
#    rectangles to forbid overlap that would let one composite erase another.
# d) Placement legality:
#    The generator still keeps every composite fully inside the canvas. For two legends it
#    rejects any overlap between composite bounding rectangles, not just colored cells,
#    because zero-write rectangles now matter.
# e) Visual observability:
#    Outputs differ from the base task on the same input whenever a frame has inactive
#    cells. Generator enforces that by requiring a minimum number of inactive tiles and
#    gaps on multiple sides of the frame, so black clearance is repeated and easy to see.
# f) Visual cleanliness:
#    Only one overlay layer is active in any canvas cell. The cleared area lives exactly
#    inside each composite rectangle, so the blank structure is readable rather than noisy.
# g) Color inference:
#    No new color mapping is introduced. Color 0 becomes visible in output only because of
#    spatial placement, which is directly supported by the frame mask structure.
# h) New solver assumptions after modification:
#    - Marker panels still live outside the output canvas.
#    - Unit sprites remain fully filled, so all output zero regions come from inactive frame
#      cells rather than internal holes.
#    - The generator guarantees no drawable rectangle overlap, matching the solver assert.


def _find_border6_square_start(legend: np.ndarray) -> tuple[int, int] | None:
    h, w = legend.shape
    if h < 5 or w < 5:
        return None
    for r0 in range(h - 4):
        for c0 in range(w - 4):
            sub = legend[r0 : r0 + 5, c0 : c0 + 5]
            if np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6):
                return int(r0), int(c0)
    return None


def _find_marker_panel_square(g: np.ndarray, sep_c1: int) -> tuple[int, int] | None:
    """
    Find the 5x5 border-6 position-marker square that belongs to the 8x5 marker panel.

    In some instances the marker panel can appear inside the main canvas as well as in the
    left legend region. Prefer a candidate that lies in the main canvas (col > sep_c1) if any.
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    main_cands = [rc for rc in candidates if rc[1] > sep_c1]
    return min(main_cands) if main_cands else min(candidates)


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


def _find_marker_square_with_preferences(
    g: np.ndarray, left_sep: tuple[int, int], right_sep: tuple[int, int] | None
) -> tuple[int, int] | None:
    """
    Find the marker panel's 5x5 border-6 square.

    Preference order:
      1) left legend region (at/before left separator)
      2) right legend region (at/after right separator), if present
      3) anywhere (including inside the output canvas)
    """
    h, w = g.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    left_end = left_sep[1]
    left_cands = [rc for rc in candidates if rc[1] + 2 <= left_end]
    if left_cands:
        return min(left_cands)
    if right_sep is not None:
        right_start = right_sep[0]
        right_cands = [rc for rc in candidates if rc[1] + 2 >= right_start]
        if right_cands:
            return min(right_cands)
    return min(candidates)


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
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


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
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


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Rotation marker is in the bottom 3x3 at rows (border6_r0+5 .. border6_r0+7) and cols (1..3),
    drawn with non-{0,6} pixels (typically 7). Use the furthest pixel from the center (1,1)
    to determine direction:
      up -> 0, right -> 1, down -> 2, left -> 3.
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


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the dense unit sprite into each active frame cell, leaving 0 elsewhere."""
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
    return comp


def _frame_visibility_ok(frame_mask: np.ndarray) -> bool:
    """Reject frames whose clear region would be too small or too hidden."""
    total = int(frame_mask.size)
    active = int(frame_mask.sum())
    inactive = total - active
    if inactive < max(2, total // 5):
        return False
    gap_sides = 0
    if not np.all(frame_mask[0, :]):
        gap_sides += 1
    if not np.all(frame_mask[-1, :]):
        gap_sides += 1
    if not np.all(frame_mask[:, 0]):
        gap_sides += 1
    if not np.all(frame_mask[:, -1]):
        gap_sides += 1
    return gap_sides >= 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode one or two legend panels, assemble enlarged sprites, and stamp them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns split the grid into legend panel(s) and the output canvas.
    - Each legend panel contains, from top to bottom, a 2-color unit sprite, a sparse frame mask, a 5x5 bordered anchor marker, and a 3x3 rotation marker.

    Rule:
    - Swap the unit sprite colors back, tile that sprite into every active frame cell, and rotate the resulting composite according to the clock-hand marker.
    - Place the composite using the anchor marker, then overwrite every non-6 pixel in the composite rectangle onto the canvas, including 0-valued background cells so the canvas is explicitly cleared inside inactive frame blocks.

    Key insight:
    - The legend still determines shape, rotation, and anchor exactly as before; the variant only changes the final write semantics, because zeros inside the composite are now meaningful visible output rather than transparent gaps.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    left_sep_c1 = left_sep[1]
    canvas_c0 = left_sep_c1 + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main.shape
    full6_cols = [c for c in range(eff_w) if np.all(main[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue

        if marker_c0 + 5 > w:
            continue
        panel = g[:, marker_c0 : marker_c0 + 5]

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
        comp = _build_composite(unit_sprite, frame_mask)

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
                assert not composite_written[rr, cc], "composite rectangles must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = int(comp[r, c])
    return out


# --------------- generator (D12 clear-region overwrite) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
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
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * N <= grid_h and rw * N <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = None
    for _ in range(80):
        cand = random_sprite(
            frame_H, frame_W, density=FRAME_DENSITY, connectivity=8,
            color_palette=[1], symmetry="not_symmetric"
        ) != 0
        if _frame_visibility_ok(cand):
            frame_mask = cand
            break
    if frame_mask is None:
        return False

    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp = _build_composite(sprite, rot_frame)

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        comp=comp,
        clear_cells=int(np.count_nonzero(comp == 0)),
    )
    return True


def _comp_size(leg):
    """Return (height, width) of the composite after rotation."""
    comp = leg["comp"]
    return comp.shape


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap of composite rectangles."""
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
    """Stamp each composite rectangle, including its 0 background, except over color 6."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t:t + ch, l:l + cw]
        write_mask = region != 6
        region[write_mask] = comp[write_mask]
    return output


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        N = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": N, "legend_w": N + 2})

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

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    if not np.any(output == 0):
        return None
    if any(leg["clear_cells"] == 0 for leg in legends):
        return None

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0:canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    N = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, N - 4)) if not is_right else (sep_col + max(0, N - 3))

    grid[marker_r0:marker_r0 + 5, mc0:mc0 + 5] = 0
    grid[marker_r0, mc0:mc0 + 5] = 6
    grid[marker_r0 + 4, mc0:mc0 + 5] = 6
    grid[marker_r0:marker_r0 + 5, mc0] = 6
    grid[marker_r0:marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0:rr0 + 3, mc0:mc0 + 5] = 0
    grid[rr0:rr0 + 3, mc0] = 6
    grid[rr0:rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1:1 + N, mc0 + sp_col:mc0 + sp_col + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start:fr_start + fH, mc0 + f_col:mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
