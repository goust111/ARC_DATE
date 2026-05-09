# Combo variant of a32d8b75 (index 23)
# combo_spec: D9/1+D16/2
# Name: 同心环背景+背景标记定位

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


"""
design_analysis:
  fusion_goal:
    keep:
      - left/right legend panels, separator color 6, sprite color-swap on read
      - frame tiling into an enlarged composite, then rotation from the 3x3 clock-hand patch
      - output-first generation with explicit non-overlap / in-bounds guarantees
    change:
      - D16/2 removes the old 3x3 anchor cue from the bordered square; the square stays as a silhouette but its interior is blank
      - D9/1 replaces banded backgrounds with multi-color concentric rectangular rings
      - the hidden placement marker is now embedded inside that ring background as an isolated color-4 pixel

  concrete_visualization:
    legend:
      - each legend still ends with a 5x5 color-6 border square plus the rotation patch below it
      - unlike the seed task, the square interior has no anchor pixel at all
    canvas:
      - the bare input canvas is filled with nested rectangular rings using 3 alternating non-{0,4,6} colors
      - each legend contributes one hidden color-4 pixel placed on some interior ring cell
      - that pixel lies exactly on the composite's chosen internal reference point, so it disappears after overlay
    output:
      - the output is the ring canvas with composites stamped in, so the special marker pixels are overwritten and vanish

  behavior_trace:
    example:
      - suppose a legend yields a rotated composite whose first painted pixel is offset (1,2) inside a 12x15 bounding box
      - original task: read (pr, pc) from the square, map them to start/middle/end buckets, and derive one of only 9 possible top-left placements
      - fused task: ignore the square interior, find the hidden color-4 pixel inside the rings, and place the composite so that (top+1, left+2) lands on that marker
      - difference: placement is no longer bucketed by the legend; it is concretely encoded by the canvas itself, and the marker must be found amid the ring pattern

  complexity_calibration:
    - background uses exactly 3 ring colors, enough to make the canvas visibly richer than stripes without becoming unreadable
    - marker color 4 is globally reserved, so isolated marker pixels remain unambiguous even inside a colorful background
    - 1-2 markers match the 1-2 legend panels; legends are paired with markers left-to-right, which is human-visible and solver-stable
    - generator forbids old anchor-equivalent placements, so the hidden marker truly controls location instead of accidentally reproducing the seed behavior

  main_line_assumptions_review:
    - old marker-panel detection assumed a nonzero inner pixel in the 5x5 square; fused task invalidates that and the detector must accept empty interiors
    - old placement logic depended on _position_indices_from_border_square and _anchor_to_top_left; both become wrong once the square no longer encodes anchor buckets
    - hidden-marker search assumes color 4 appears only as a marker; generation therefore excludes 4 from sprites, frames, and ring colors
    - left-to-right pairing assumes marker columns are distinct; generator enforces strictly increasing marker columns for the 2-legend case
    - composite overlay assumes marker pixels are covered by the composite; generator derives each marker from an actual painted reference pixel, so this is guaranteed

  placement_legality:
    - each legend is designed first, including its rotated composite and internal reference offset
    - placement search samples only in-bounds top-left positions with reference markers away from the outer border ring
    - 2-legend samples reject overlapping bounding boxes and reject non-monotone marker-column order
    - placements that match any of the seed task's 9 anchor-bucket positions are rejected, preventing semantic collapse

  observability_and_cleanliness:
    - visually, the most obvious differences from the seed task are the blank border square and the concentric-ring canvas
    - the lone color-4 pixel stands out enough to be findable, but it is still embedded inside a structured background rather than floating on a flat field
    - composites never overlap, and marker pixels live in the canvas rather than on top of legend content, so regions stay readable

  inferability_check:
    - from a few I/O pairs, a human can notice that the blank square cannot explain placement anymore
    - the disappearing color-4 pixels on the ring canvas provide the placement clue, while the clock-hand still explains rotation
    - because generator excludes old-anchor placements, examples consistently teach the new rule instead of mixing two valid explanations
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
    _h, w = g.shape
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
    """Use the top-left-most painted pixel as the hidden-marker alignment reference."""
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
    Assemble and place enlarged sprites on a concentric-ring canvas using legend instructions plus hidden background markers.

    Input structure:
    - Color-6 separator columns still split the full grid into 1-2 legend panels and a central canvas.
    - Each legend contains a color-swapped unit sprite, a frame template, a blank 5x5 border-6 square, and a 3x3 rotation clock-hand beneath it.
    - The canvas itself is a multicolor set of concentric rectangular rings and contains one hidden color-4 marker pixel per legend.

    Rule:
    - Tile the recovered unit sprite into every active frame cell to build the enlarged composite, then rotate it according to the clock-hand.
    - Find the composite's top-left-most painted pixel and align that internal reference to the corresponding hidden color-4 marker in the ring background.
    - Overlay the composite onto the canvas, replacing ring colors with nonzero composite pixels while leaving color-6 separators untouched.

    Key insight: The bordered square no longer encodes placement at all; the real anchor is the lone marker pixel concealed inside the concentric background pattern.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
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


# --------------- generator (concentric_rings_plus_hidden_markers) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 2]
LEGEND_PANEL_W = 5
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400
MULTI_LEGEND_LAYOUT_RETRIES = 10
RING_THICKNESS_CHOICES = [1, 2]
RING_COLOR_COUNT = 3


def _allocate_colors(num_legends: int):
    """Pick unit colors, 3 ring colors, and 1 frame color per legend while reserving color 4."""
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
    if len(bg_cands) < RING_COLOR_COUNT:
        return None
    random.shuffle(bg_cands)
    bg_colors = tuple(bg_cands[:RING_COLOR_COUNT])

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


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Seed-task anchor mapping, kept only to reject degenerate placements."""
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


def _old_anchor_positions(grid_h: int, canvas_w: int, comp_h: int, comp_w: int) -> set[tuple[int, int]]:
    """All placements that the original 3x3 anchor semantics could express."""
    out: set[tuple[int, int]] = set()
    for pr in range(3):
        top = _anchor_to_top_left(pr, grid_h, comp_h)
        if top < 0 or top + comp_h > grid_h:
            continue
        for pc in range(3):
            left = _anchor_to_top_left(pc, canvas_w, comp_w)
            if left < 0 or left + comp_w > canvas_w:
                continue
            out.add((top, left))
    return out


def _ring_depth(r: int, c: int, h: int, w: int) -> int:
    return min(r, c, h - 1 - r, w - 1 - c)


def _design_legend(leg: dict, unit_colors, frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    """Fill in sprite, frame, rotation, and concrete composite geometry for one legend dict."""
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
    old_anchor_positions = _old_anchor_positions(grid_h, canvas_w, comp.shape[0], comp.shape[1])

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
        old_anchor_positions=old_anchor_positions,
    )
    return True


def _candidate_positions(leg: dict, grid_h: int, canvas_w: int) -> list[tuple[int, int, int, int]]:
    """List non-degenerate placements as (top, left, marker_r, marker_c)."""
    out: list[tuple[int, int, int, int]] = []
    for top in range(grid_h - leg["comp_h"] + 1):
        for left in range(canvas_w - leg["comp_w"] + 1):
            if (top, left) in leg["old_anchor_positions"]:
                continue
            marker_r = top + leg["ref_r"]
            marker_c = left + leg["ref_c"]
            if _ring_depth(marker_r, marker_c, grid_h, canvas_w) < 1:
                continue
            out.append((top, left, marker_r, marker_c))
    return out


def _rects_overlap(a_top, a_left, a_h, a_w, b_top, b_left, b_h, b_w) -> bool:
    return a_top < b_top + b_h and b_top < a_top + a_h and a_left < b_left + b_w and b_left < a_left + a_w


def _assign_marker_positions(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Choose concrete placements, derive hidden markers, and avoid overlap or seed-task collapse."""
    candidates = []
    for leg in legends:
        cands = _candidate_positions(leg, grid_h, canvas_w)
        if not cands:
            return False
        random.shuffle(cands)
        candidates.append(cands)

    if len(legends) == 1:
        top, left, marker_r, marker_c = random.choice(candidates[0])
        legends[0].update(comp_top=top, comp_left=left, marker_r=marker_r, marker_c=marker_c)
        return True

    for a_choice in candidates[0]:
        a_top, a_left, a_mr, a_mc = a_choice
        for b_choice in candidates[1]:
            b_top, b_left, b_mr, b_mc = b_choice
            if a_mc >= b_mc:
                continue
            if _rects_overlap(
                a_top,
                a_left,
                legends[0]["comp_h"],
                legends[0]["comp_w"],
                b_top,
                b_left,
                legends[1]["comp_h"],
                legends[1]["comp_w"],
            ):
                continue
            legends[0].update(comp_top=a_top, comp_left=a_left, marker_r=a_mr, marker_c=a_mc)
            legends[1].update(comp_top=b_top, comp_left=b_left, marker_r=b_mr, marker_c=b_mc)
            return True
    return False


def _build_ring_canvas(grid_h: int, canvas_w: int, bg_colors: tuple[int, ...]) -> np.ndarray:
    """Build a canvas with nested rectangular color rings."""
    rows = np.arange(grid_h)[:, None]
    cols = np.arange(canvas_w)[None, :]
    depth = np.minimum(
        np.minimum(rows, cols),
        np.minimum(grid_h - 1 - rows, canvas_w - 1 - cols),
    )
    thickness = random.choice(RING_THICKNESS_CHOICES)
    ring_idx = depth // thickness
    palette = np.array(bg_colors, dtype=int)
    return palette[ring_idx % len(palette)].astype(int)


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + leg["comp_h"], l : l + leg["comp_w"]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel with an empty border square and a rotation patch."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = col_start

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

    sp_col = random.randint(0, LEGEND_PANEL_W - n)
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    f_h, f_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - f_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, LEGEND_PANEL_W - f_w)
    frame_region = grid[fr_start : fr_start + f_h, mc0 + f_col : mc0 + f_col + f_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        n = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": n, "legend_w": LEGEND_PANEL_W})

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, bg_colors, frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    layout_tries = MULTI_LEGEND_LAYOUT_RETRIES if num_legends == 2 else 1
    placed = False
    for _ in range(layout_tries):
        design_ok = True
        for li in range(num_legends):
            if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
                design_ok = False
                break
        if not design_ok:
            continue
        if _assign_marker_positions(legends, grid_h, canvas_w):
            placed = True
            break
    if not placed:
        return None

    canvas = _build_ring_canvas(grid_h, canvas_w, bg_colors)
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
