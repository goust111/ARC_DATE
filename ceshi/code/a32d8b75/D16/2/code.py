# Variant of a32d8b75
# Dimension: D16 — 锚点对齐语义 (meta)
# Change: 锚点索引0/1/2分别表示起始/中间/结束，独立应用于行和列方向... → 锚点由画布背景图案决定

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


"""
design_analysis:
  variant_goal:
    keep:
      - left/right legend layout, separator color 6, unit-sprite extraction, frame tiling, rotation clock-hand
      - output-first generator discipline and solver-side non-overlap / in-bounds guarantees
    change:
      - the 5x5 border-6 square no longer carries the anchor with a 3x3 inner marker
      - anchor information is moved into the canvas background as hidden color-4 marker pixels

  concrete_visualization:
    legend:
      - still ends with a 5x5 border-6 square plus the 3x3 rotation patch below it
      - the square interior is now empty (all 0), making the old position marker visibly absent
    canvas:
      - background remains banded and uses colors disjoint from 4
      - each legend owns exactly one hidden marker pixel of color 4 inside the canvas
      - that marker sits at the final location of the composite's top-left-most painted pixel
    output:
      - marker pixels disappear because the placed composite overwrites them

  behavior_trace:
    original:
      - read (pr, pc) from the 3x3 inner square
      - map each of pr, pc from {0,1,2} to start/middle/end against the canvas extent
      - compute top-left from anchor_to_top_left and place the rotated composite
    variant:
      - ignore the square interior entirely
      - read the hidden color-4 pixel(s) from the canvas
      - for each rotated composite, find its own top-left-most painted pixel offset (ar, ac)
      - place the composite so that (top + ar, left + ac) equals the hidden marker position
    difference:
      - placement is no longer quantized to three anchor buckets per axis
      - the user must search the canvas for marker pixels; the legend alone is insufficient

  complexity_calibration:
    - only 1 or 2 hidden pixels are used, matching the original 1 or 2 legends
    - markers are sorted left-to-right and legends are consumed left-to-right, so the pairing rule is stable
    - color 4 is reserved for markers, preventing ambiguity with sprites, frames, or background bands
    - the blank square preserves the original panel silhouette, so the variant is locally comparable to the seed task

  main_line_assumptions_review:
    - old _all_marker_panel_squares required a nonzero inner pixel; this is invalid now and must be relaxed
    - old _position_indices_from_border_square depended on exactly one inner marker; it must be removed
    - old _anchor_to_top_left is no longer part of placement and would misplace every generated sample
    - old code could tolerate color 4 appearing in sprites; the new marker search cannot, so generation must reserve color 4
    - marker-to-legend association now depends on left-to-right ordering, so generation must enforce distinct marker columns

  placement_legality:
    - every composite is generated first, then assigned a concrete top-left within the canvas bounds
    - hidden marker coordinates are derived from that concrete placement and the composite reference offset
    - for the 2-legend case, bounding boxes may not overlap and marker columns must be strictly ordered
    - solver writes only nonzero composite pixels and asserts no overlap, matching the generator contract

  observability_and_cleanliness:
    - the most obvious visual change is that the bordered square is empty while the canvas contains isolated 4-pixels
    - because the marker pixel is overwritten in the output, the input/output pair clearly suggests it as a placement cue
    - banded backgrounds stay simple, so the hidden marker remains easy to spot rather than buried in clutter

  new_solver_assumptions_backstopped_by_generator:
    - number of hidden canvas markers equals number of legend panels
    - no non-marker canvas pixel uses color 4
    - marker columns are distinct, so left-to-right pairing is deterministic
    - each composite has at least one painted pixel; frame generation plus fully filled unit sprites ensure this
"""


MARKER_COLOR = 4


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


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that act as the top of an 8x5 legend marker panel."""
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
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
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
    # Frame pixels live between the unit sprite and the border6 square.
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


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into each active frame cell."""
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


def _reference_offset(comp: np.ndarray) -> tuple[int, int]:
    """Use the top-left-most painted pixel as the marker-alignment reference."""
    pts = np.argwhere((comp != 0) & (comp != 6))
    assert pts.size > 0, "composite must contain painted pixels"
    r = int(pts[:, 0].min())
    c = int(pts[pts[:, 0] == r][:, 1].min())
    return r, c


def _hidden_markers(main: np.ndarray) -> list[tuple[int, int]]:
    """Return hidden canvas marker pixels in left-to-right order."""
    pts = np.argwhere(main == MARKER_COLOR)
    out = [(int(r), int(c)) for r, c in pts]
    out.sort(key=lambda rc: (rc[1], rc[0]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas using legend instructions plus hidden canvas markers.

    Input structure:
    - Color-6 separators still split the grid into legend column(s) and the main canvas.
    - Each legend keeps a unit sprite, a frame template, an empty 5x5 border-6 square, and a 3x3 rotation clock-hand below it.
    - The canvas background contains one hidden color-4 marker pixel per legend; those pixels replace the old square-internal anchor cue.

    Rule:
    - Tile the color-swapped unit sprite into every active frame cell to form the enlarged composite, then rotate it from the clock-hand.
    - Find the composite's top-left-most painted pixel and align that internal reference to the corresponding hidden color-4 canvas marker.
    - Draw the composite onto the canvas background, overwriting non-6 cells with nonzero composite pixels.

    Key insight: The border square no longer tells you where to place the object; the true placement cue is the isolated marker pixel hidden in the canvas itself.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no legend marker panel found"
    legend_squares = [rc for rc in marker_squares if not (canvas_c0 <= rc[1] < canvas_c1)]
    legend_squares.sort(key=lambda rc: rc[1])
    assert legend_squares, "no legend-side marker panel found"

    hidden = _hidden_markers(main)
    assert len(hidden) == len(legend_squares), "hidden canvas marker count must match legend count"

    out = main.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for (marker_r0, marker_c0), (mr, mc) in zip(legend_squares, hidden):
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
        ar, ac = _reference_offset(comp)
        top = mr - ar
        left = mc - ac

        ch, cw = comp.shape
        assert 0 <= top and top + ch <= out.shape[0], "sprite row out of canvas bounds"
        assert 0 <= left and left + cw <= out.shape[1], "sprite col out of canvas bounds"
        for r in range(ch):
            rr = top + r
            for c in range(cw):
                cc = left + c
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


# --------------- generator (hidden_canvas_marker_pixels) ---------------

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
USABLE_COLORS = [1, 2, 3, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


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
    """Fill in sprite, frame, rotation, and composite geometry for one legend dict."""
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
            for rk in range(4):
                rh, rw = (f_h, f_w) if rk % 2 == 0 else (f_w, f_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((f_h, f_w, rk))
    if not combos:
        return False
    frame_h, frame_w, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    comp = _build_composite(sprite, rot_frame)
    ref_r, ref_c = _reference_offset(comp)

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
        comp_h=comp.shape[0],
        comp_w=comp.shape[1],
        ref_r=ref_r,
        ref_c=ref_c,
    )
    return True


def _rects_overlap(a_top, a_left, a_h, a_w, b_top, b_left, b_h, b_w) -> bool:
    return a_top < b_top + b_h and b_top < a_top + a_h and a_left < b_left + b_w and b_left < a_left + a_w


def _assign_marker_positions(legends, grid_h, canvas_w):
    """Choose concrete placements, derive hidden marker pixels, and avoid overlap."""
    if len(legends) == 1:
        leg = legends[0]
        top = random.randint(0, grid_h - leg["comp_h"])
        left = random.randint(0, canvas_w - leg["comp_w"])
        leg.update(
            comp_top=top,
            comp_left=left,
            marker_r=top + leg["ref_r"],
            marker_c=left + leg["ref_c"],
        )
        return True

    for _ in range(PLACEMENT_RETRIES):
        for leg in legends:
            top = random.randint(0, grid_h - leg["comp_h"])
            left = random.randint(0, canvas_w - leg["comp_w"])
            leg["comp_top"] = top
            leg["comp_left"] = left
            leg["marker_r"] = top + leg["ref_r"]
            leg["marker_c"] = left + leg["ref_c"]

        a, b = legends
        if _rects_overlap(
            a["comp_top"], a["comp_left"], a["comp_h"], a["comp_w"],
            b["comp_top"], b["comp_left"], b["comp_h"], b["comp_w"],
        ):
            continue
        if not (a["marker_c"] + 1 < b["marker_c"]):
            continue
        return True
    return False


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
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + leg["comp_h"], l : l + leg["comp_w"]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
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

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_marker_positions(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    for leg in legends:
        grid[leg["marker_r"], canvas_c0 + leg["marker_c"]] = MARKER_COLOR

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
    """Draw one legend panel with an empty border square and a rotation patch."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6

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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
