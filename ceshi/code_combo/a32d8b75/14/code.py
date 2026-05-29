# Combo variant of a32d8b75 (index 14)
# combo_spec: D12/1+D13/0
# Name: 背景色覆写+紧凑裁剪

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    Base task:
#      - Build a rectangular composite by tiling the recovered unit sprite into active
#        frame cells; inactive frame cells remain 0 inside that rectangle.
#      - Overlay only non-zero composite pixels onto the hidden canvas.
#      - Return the whole canvas window.
#    Combo behavior:
#      - The same rectangular composite is built, but every non-6 cell inside that
#        rectangle now overwrites the canvas, including 0-valued inactive cells.
#      - After all composites are placed, output is tight-cropped to the union bbox of
#        those composite rectangles, not to the full canvas.
#    Small trace:
#      unit = [[1, 2], [2, 1]]
#      frame = [[1, 0, 1],
#               [0, 1, 0]]
#      comp  = a 4x6 rectangle containing three tiled blocks and three 0 blocks.
#      Old output keeps striped canvas colors in the 0 blocks and keeps the full canvas.
#      New output clears those blocks to 0 first, then discards every outer canvas row
#      and column outside the placed composite rectangle. If a frame side has inactive
#      cells, the cropped output shows black border segments that did not exist before.
# b) Complexity calibration:
#    We keep the original legend language: sprite, frame, anchor square, rotation marker.
#    The only new reasoning is that zeros inside the composite are now semantically active
#    and the final grid shape comes from the placed composite rectangles. Generator forces
#    sparse frames with edge gaps so the change is visible instead of collapsing back to
#    the base task.
# c) Hidden assumption audit:
#    - Original `if val != 0` overlay assumed 0 was transparent. That is invalid now.
#    - Original crop target was implicitly the fixed canvas. That is invalid now.
#    - Original overlap guard only needed to protect colored pixels. Now full composite
#      rectangles matter because one legend could erase another with zeros.
#    Therefore the solver tracks rectangle occupancy for both overwrite legality and crop.
# d) Placement legality:
#    Generator keeps every composite fully inside the hidden canvas and forbids overlap of
#    composite bounding rectangles, not merely overlap of colored pixels.
# e) Visual observability:
#    Samples are constrained so that:
#      - at least one cleared 0 cell is visible in the final output;
#      - at least one cleared 0 cell lies on the crop boundary, making the "clear then
#        crop" interaction visible;
#      - the crop is strictly smaller than the hidden canvas on both axes.
# f) Visual cleanliness:
#    Only one composite rectangle can occupy any canvas cell. Legends remain outside the
#    canvas, and the cropped answer contains one clean union window rather than layered
#    noise.
# g) Change inferability:
#    No new color remapping is introduced. Color 0 becomes meaningful only through
#    structure: inactive frame tiles create repeated blank blocks aligned to the unit size.
# h) New solver assumptions guaranteed by generate_input():
#    - Every legend outside the canvas yields exactly one 8-connected frame component.
#    - At least one legend produces visible clear cells after placement.
#    - The union crop mask is non-empty and observably smaller than the hidden canvas.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns filled with color 6."""
    _, w = g.shape
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
    """Locate the separator before the canvas and the optional separator after it."""
    _, w = g.shape
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List 5x5 border-6 squares whose 3x3 footer contains a rotation marker."""
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
    """Recover the top-most 4-connected non-{0,6} component and swap its two colors back."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    vis[rmin, cseed] = True
    stack = [(rmin, cseed)]
    pts: list[tuple[int, int]] = []
    while stack:
        r, c = stack.pop()
        pts.append((r, c))
        for rr, cc in neighbors4(r, c):
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not vis[rr, cc]:
                vis[rr, cc] = True
                stack.append((rr, cc))
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
    return spr, (int(r0), int(c0), int(r1), int(c1))


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Find the bbox of frame pixels between the unit sprite and the marker panel."""
    r_start = unit_bb[2] + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    pts = np.argwhere((sub != 0) & (sub != 6))
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
    """Decode up/right/down/left as rotation k in {0,1,2,3}."""
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
    """Map start/middle/end anchor index to the composite's top-left coordinate."""
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
    """Tile the unit sprite into active frame cells, leaving 0 blocks elsewhere."""
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


def _crop_to_mask(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Tight-crop `grid` to the bbox spanning all True cells of `mask`."""
    pts = np.argwhere(mask)
    assert pts.size > 0, "tight crop requires at least one covered cell"
    r0, c0, r1, c1 = bbox_coords(pts)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _boundary_zero_count(arr: np.ndarray) -> int:
    """Count 0 cells on the outer boundary of `arr`."""
    if arr.size == 0:
        return 0
    mask = np.zeros(arr.shape, dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return int(np.count_nonzero((arr == 0) & mask))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct legend-defined composites, clear the canvas with their full rectangles, then crop the result.

    Input structure:
    - Full-height color-6 separator columns split the grid into one or two legend panels and a hidden canvas.
    - Each legend panel still contains a 2-color unit sprite, a sparse frame mask, a 5x5 bordered anchor square, and a 3x3 clock-hand rotation marker.
    - The visible canvas in the input is only a staging area; the final answer is a crop taken after reconstruction.

    Rule:
    - Recover the unit sprite by swapping its two colors back, tile it into every active frame cell, and rotate the resulting composite from the clock-hand marker.
    - Place each composite on the canvas using the anchor marker and overwrite every non-6 canvas cell inside the composite rectangle, including 0-valued inactive-frame cells that explicitly clear the background.
    - After all placements, return the tight crop of the canvas covering the union of the placed composite rectangles.

    Key insight:
    - The legends still encode shape, rotation, and anchor exactly as before, but now inactive frame tiles are visible because they erase the canvas, and those cleared tiles still matter for the final crop window.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = canvas.copy()
    rectangle_written = np.zeros(out.shape, dtype=bool)

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
                assert not rectangle_written[rr, cc], "composite rectangles must not overlap"
                rectangle_written[rr, cc] = True
                if out[rr, cc] != 6:
                    out[rr, cc] = int(comp[r, c])

    return _crop_to_mask(out, rectangle_written)


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
PRIMARY_BRANCH_TRIES = 180
FRAME_GEN_TRIES = 80
MIN_CROP_REDUCTION = 1
MAX_CROP_AREA_RATIO = 0.95
MIN_VISIBLE_CLEAR_CELLS = 4
MIN_BOUNDARY_CLEAR_CELLS = 2


def _allocate_colors(num_legends: int):
    """Pick disjoint unit colors, canvas colors, and frame colors."""
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
        frame_colors.append(random.choice(list(all_set - set(uc))))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _frame_visibility_ok(frame_mask: np.ndarray) -> bool:
    """Require enough inactive tiles and at least two open sides."""
    total = int(frame_mask.size)
    inactive = total - int(frame_mask.sum())
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


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Sample a legend whose composite fits and exposes clear cells."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (f_h, f_w) if rot_k % 2 == 0 else (f_w, f_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((f_h, f_w, rot_k))
    if not combos:
        return False
    frame_h, frame_w, rot_k = random.choice(combos)

    frame_mask = None
    comp = None
    rot_frame = None
    for _ in range(FRAME_GEN_TRIES):
        cand = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        if not _frame_visibility_ok(cand):
            continue
        rot_cand = np.rot90(cand, (-rot_k) % 4)
        comp_cand = _build_composite(sprite, rot_cand)
        if np.count_nonzero(comp_cand == 0) < MIN_VISIBLE_CLEAR_CELLS:
            continue
        if _boundary_zero_count(comp_cand) < MIN_BOUNDARY_CLEAR_CELLS:
            continue
        frame_mask = cand
        rot_frame = rot_cand
        comp = comp_cand
        break
    if frame_mask is None or rot_frame is None or comp is None:
        return False

    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        comp=comp,
        clear_cells=int(np.count_nonzero(comp == 0)),
        boundary_clear_cells=_boundary_zero_count(comp),
    )
    return True


def _rectangles_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    at, al, ah, aw = a
    bt, bl, bh, bw = b
    return at < bt + bh and bt < at + ah and al < bl + bw and bl < al + aw


def _union_bbox_dims(placements: list[tuple[int, int, int, int]]) -> tuple[int, int]:
    top = min(t for t, _, _, _ in placements)
    left = min(l for _, l, _, _ in placements)
    bottom = max(t + h for t, _, h, _ in placements)
    right = max(l + w for _, l, _, w in placements)
    return int(bottom - top), int(right - left)


def _crop_is_salient(placements: list[tuple[int, int, int, int]], grid_h: int, canvas_w: int) -> bool:
    crop_h, crop_w = _union_bbox_dims(placements)
    if crop_h > grid_h - MIN_CROP_REDUCTION or crop_w > canvas_w - MIN_CROP_REDUCTION:
        return False
    return crop_h * crop_w <= int(MAX_CROP_AREA_RATIO * grid_h * canvas_w)


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchors so every composite fits, rectangles do not overlap, and crop is visible."""
    if len(legends) == 1:
        leg = legends[0]
        ch, cw = leg["comp"].shape
        valid = []
        for pr in range(3):
            for pc in range(3):
                top = _anchor_to_top_left(pr, grid_h, ch)
                left = _anchor_to_top_left(pc, canvas_w, cw)
                if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
                    continue
                placement = [(top, left, ch, cw)]
                if not _crop_is_salient(placement, grid_h, canvas_w):
                    continue
                valid.append((pr, pc, top, left))
        if not valid:
            return False
        pr, pc, top, left = random.choice(valid)
        leg.update(pr=pr, pc=pc, comp_top=top, comp_left=left)
        return True

    a, b = legends
    a_h, a_w = a["comp"].shape
    b_h, b_w = b["comp"].shape
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_h)
            al = _anchor_to_top_left(apc, canvas_w, a_w)
            if at < 0 or al < 0 or at + a_h > grid_h or al + a_w > canvas_w:
                continue
            pa = (at, al, a_h, a_w)
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_h)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_w)
                    if bt < 0 or bl < 0 or bt + b_h > grid_h or bl + b_w > canvas_w:
                        continue
                    pb = (bt, bl, b_h, b_w)
                    if _rectangles_overlap(pa, pb):
                        continue
                    placements = [pa, pb]
                    if not _crop_is_salient(placements, grid_h, canvas_w):
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    """Build a striped hidden canvas with two non-zero background colors."""
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


def _overlay_clear_rectangles(canvas: np.ndarray, legends: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Overlay composites with 0 overwrite and return the full rectangle crop mask."""
    output = canvas.copy()
    crop_mask = np.zeros(canvas.shape, dtype=bool)
    for leg in legends:
        comp = leg["comp"]
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region_mask = crop_mask[t : t + ch, l : l + cw]
        assert not np.any(region_mask), "composite rectangles must not overlap"
        region_mask[:] = True
        region = output[t : t + ch, l : l + cw]
        write_mask = region != 6
        region[write_mask] = comp[write_mask]
    return output, crop_mask


def _overlay_transparent(canvas: np.ndarray, legends: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Overlay only non-zero composite cells and return the painted mask."""
    output = canvas.copy()
    painted = np.zeros(canvas.shape, dtype=bool)
    for leg in legends:
        comp = leg["comp"]
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        assert not np.any(painted[t : t + ch, l : l + cw] & write_mask), "colored composites must not overlap"
        region[write_mask] = comp[write_mask]
        painted[t : t + ch, l : l + cw] |= write_mask
    return output, painted


def _fusion_metrics(canvas: np.ndarray, legends: list[dict]) -> dict[str, int]:
    """Quantify the visible effect of clear-then-crop against transparent overlay."""
    cleared_full, crop_mask = _overlay_clear_rectangles(canvas, legends)
    transparent_full, painted_mask = _overlay_transparent(canvas, legends)
    fused_crop = _crop_to_mask(cleared_full, crop_mask)

    clear_visible_cells = int(np.count_nonzero((cleared_full == 0) & (transparent_full != 0) & crop_mask))
    boundary_clear_cells = _boundary_zero_count(fused_crop)

    crop_pts = np.argwhere(crop_mask)
    crop_r0, crop_c0, crop_r1, crop_c1 = bbox_coords(crop_pts)
    crop_area = int((crop_r1 - crop_r0 + 1) * (crop_c1 - crop_c0 + 1))

    painted_pts = np.argwhere(painted_mask)
    assert painted_pts.size > 0, "transparent overlay must paint at least one cell"
    paint_r0, paint_c0, paint_r1, paint_c1 = bbox_coords(painted_pts)
    painted_area = int((paint_r1 - paint_r0 + 1) * (paint_c1 - paint_c0 + 1))

    return {
        "clear_visible_cells": clear_visible_cells,
        "boundary_clear_cells": boundary_clear_cells,
        "bbox_expansion": crop_area - painted_area,
        "crop_area": crop_area,
        "painted_area": painted_area,
    }


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool):
    """Draw one legend panel into the full input grid."""
    n = leg["N"]
    grid[:, sep_col] = 6

    # Keep the right separator column intact so solver-side canvas splitting remains observable.
    mc0 = (col_start + max(0, n - 4)) if not is_right else (grid.shape[1] - 5)

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

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - f_w))
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    if num_legends is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for idx in range(num_legends):
        n_hi = UNIT_SIZE_RANGE[1] - 1 if num_legends == 2 else UNIT_SIZE_RANGE[1]
        n_lo = UNIT_SIZE_RANGE[0]
        if num_legends == 2 and idx == 1:
            n_lo = max(n_lo, 3)
        n = random.randint(n_lo, n_hi)
        legends.append({"N": n, "legend_w": n + 2})

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
    for i in range(num_legends):
        if not _design_legend(legends[i], unit_color_sets[i], frame_colors[i], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output_full, crop_mask = _overlay_clear_rectangles(canvas, legends)
    output = _crop_to_mask(output_full, crop_mask)

    metrics = _fusion_metrics(canvas, legends)
    if metrics["clear_visible_cells"] < MIN_VISIBLE_CLEAR_CELLS:
        return None
    if metrics["boundary_clear_cells"] < MIN_BOUNDARY_CLEAR_CELLS:
        return None
    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    preferred = random.choice([1, 2])
    secondary = 1 if preferred == 2 else 2

    for _ in range(PRIMARY_BRANCH_TRIES):
        result = _gen_attempt(preferred)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES - PRIMARY_BRANCH_TRIES):
        result = _gen_attempt(secondary)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
