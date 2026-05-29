# Combo variant of a32d8b75 (index 12)
# combo_spec: D9/0+D12/0
# Name: 棋盘格背景+允许裁剪

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The source task already decodes four legend ingredients: a 2-color unit
#    sprite, a frame mask, a 3x3 anchor marker inside a 5x5 color-6 border
#    square, and a 3x3 clock-hand for rotation. The output canvas used to be
#    a coarse striped background and every assembled composite had to fit fully
#    inside that canvas. This combo changes two stages at once:
#      - the bare canvas is now a 2-color checkerboard
#      - the anchored composite may extend off-canvas and is clipped
#    Example trace: suppose the canvas is 20x18, background colors are 8 and 9,
#    and a rotated composite is 24x12 with anchor code (0, 1). The source task
#    would reject the sample because rows 20..23 overflow. In this combo the
#    anchor is interpreted the same way, but only rows 0..19 are painted. The
#    visible fragment sits on top of alternating 8/9 cells, so the solver still
#    overlays a partial composite while a human sees both the crop and the
#    periodic background behind the missing continuation.
# b) Complexity calibration
#    The combo keeps the same legend syntax and the same tile-then-rotate logic,
#    so the task stays recognizable. The new burden is visual disentangling:
#    the checkerboard raises texture density, while clipping hides part of the
#    object. To keep the change observable instead of noisy, every generated
#    sample must satisfy all of:
#      - at least one legend is genuinely clipped
#      - the clipped legend still leaves a substantial visible fragment
#      - enough untouched checkerboard remains visible in the output
# c) Broken assumptions from the source solver
#    The source `main()` relied on two assumptions that no longer hold:
#      - anchor-derived top/left always gave an in-bounds bounding box
#      - per-pixel writes could assert 0 <= rr < H and 0 <= cc < W
#    Those are replaced with slice-based clipping. The striped background logic
#    also disappears from generation: the canvas constructor must now emit a
#    parity-regular checkerboard, but `main()` can still initialize `out` from
#    the copied input canvas because the input already contains the bare canvas.
# d) Placement legality
#    Visible foreground pixels from different legends still must not overlap, and
#    color 6 still blocks writing if it ever appears in the canvas region. The
#    generator therefore reasons about projected visible masks after clipping,
#    not only about unclipped bounding boxes. This is the key legality check.
# e) Visual observability
#    The checkerboard rule must be inferable from 2-3 examples, so the generator
#    keeps it global and deterministic: one parity gets color A, the other gets
#    color B, with only a global phase flip. Because unit colors are disjoint
#    from the two background colors, untouched background cells remain legible as
#    background rather than accidental sprite pixels.
# f) Visual cleanliness
#    A clipped legend is mandatory, but not all legends are forced to clip. When
#    there are two legends, only one must be cropped and visible masks must stay
#    disjoint. This avoids turning the whole canvas into several competing large
#    truncations on top of a dense background.
# g) Inference of the changed behavior
#    The fusion is only real if both control variables matter on the same sample:
#      - replacing clipping with the original no-crop rule would fail
#      - replacing checkerboard with stripes would visibly change the canvas
#    The generator enforces the first through minimum clipped nonzero counts, and
#    the second by checking that both checkerboard colors remain exposed in the
#    final output with enough surviving background area to reveal the pattern.
# h) New solver assumptions and matching generator guarantees
#    The updated solver assumes every legend panel still produces exactly one
#    meaningful composite projection and that any clipped projection which lands
#    on-canvas has nonempty visible content. `_candidate_placements()` and the
#    final exposure checks guarantee those assumptions before accepting a sample.


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
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
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= g.shape[1] - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


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


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
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
    Read the 3x3 clock-hand direction below the marker square:
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


def _clip_slices(top: int, left: int, comp_shape: tuple[int, int], canvas_shape: tuple[int, int]) -> tuple[slice, slice, slice, slice] | None:
    """Return aligned canvas/composite slices for the visible intersection."""
    ch, cw = comp_shape
    h, w = canvas_shape
    r0 = max(0, top)
    c0 = max(0, left)
    r1 = min(h, top + ch)
    c1 = min(w, left + cw)
    if r0 >= r1 or c0 >= c1:
        return None
    return slice(r0, r1), slice(c0, c1), slice(r0 - top, r1 - top), slice(c0 - left, c1 - left)


def _placement_metrics(comp: np.ndarray, top: int, left: int, canvas_shape: tuple[int, int]) -> dict:
    """Measure how a composite projects onto the canvas after clipping."""
    comp_mask = (comp != 0) & (comp != 6)
    total_nonzero = int(comp_mask.sum())
    occ = np.zeros(canvas_shape, dtype=bool)
    clipped = top < 0 or left < 0 or top + comp.shape[0] > canvas_shape[0] or left + comp.shape[1] > canvas_shape[1]
    clips = {
        "top": top < 0,
        "left": left < 0,
        "bottom": top + comp.shape[0] > canvas_shape[0],
        "right": left + comp.shape[1] > canvas_shape[1],
    }
    sl = _clip_slices(top, left, comp.shape, canvas_shape)
    if sl is not None:
        rs, cs, crs, ccs = sl
        occ[rs, cs] = comp_mask[crs, ccs]
    visible_nonzero = int(occ.sum())
    return {
        "mask": occ,
        "clipped": clipped,
        "clips": clips,
        "visible_nonzero": visible_nonzero,
        "clipped_nonzero": total_nonzero - visible_nonzero,
        "total_nonzero": total_nonzero,
        "top": top,
        "left": left,
    }


def _write_composite_clipped(
    out: np.ndarray,
    comp: np.ndarray,
    top: int,
    left: int,
    composite_written: np.ndarray | None = None,
) -> None:
    """Paint only the in-bounds part of comp onto out."""
    sl = _clip_slices(top, left, comp.shape, out.shape)
    if sl is None:
        return
    rs, cs, crs, ccs = sl
    comp_view = comp[crs, ccs]
    tgt = out[rs, cs]
    write_mask = (comp_view != 0) & (comp_view != 6) & (tgt != 6)
    if composite_written is not None:
        assert not np.any(composite_written[rs, cs] & write_mask), "composites must not overlap"
        composite_written[rs, cs] |= write_mask
    tgt[write_mask] = comp_view[write_mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each legend, assemble a tiled composite sprite, and project it onto the canvas.

    Input structure:
    - Full-height color-6 separator columns isolate the left legend panel and, optionally,
      a right legend panel from the central canvas.
    - The central canvas already shows the bare background, now as a 2-color checkerboard.
      Each legend panel still contains a top unit sprite, a frame mask, a 5x5 color-6
      border square with an anchor marker in its inner 3x3, and a 3x3 clock-hand patch
      directly underneath for rotation.

    Rule:
    - Swap the two unit-sprite colors back, tile that sprite into every active frame cell,
      and rotate the resulting composite according to the clock-hand.
    - Use the marker square to anchor the full composite to the checkerboard canvas, then
      draw only the in-bounds part of the composite; any off-canvas portion is cropped,
      while existing color-6 cells stay protected.

    Key insight:
    - The anchor marker still refers to the uncropped full composite, not just the visible
      fragment. The checkerboard is only a regular background texture; the real object is
      the same tiled-and-rotated composite, even when part of it has been cut off by the border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape[1] >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else g.shape[1]
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
        if marker_c0 + 5 > g.shape[1]:
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
        _write_composite_clipped(out, comp, top, left, composite_written)

    return out


# --------------- generator (checkerboard background + clipped placement) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
# Bias generation attempts toward two legends so accepted samples do not collapse
# back to the easier one-legend branch after rejection filtering.
LEGEND_COUNT_WEIGHTS = [1, 8]
UNIT_SIZE_RANGE = (2, 5)
CLIPPED_UNIT_SIZE_CHOICES = [4, 5]
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 600

MIN_VISIBLE_NONZERO = 10
MIN_CLIPPED_NONZERO = 6
MAX_VISIBLE_FILL_FRAC = 0.68
MIN_EXPOSED_BG_TOTAL = 36
MIN_EXPOSED_BG_PER_COLOR = 12


def _allocate_colors(num_legends: int):
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
        frame_colors.append(random.choice(list(all_set - set(uc))))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _sample_unit_size(must_clip: bool) -> int:
    if must_clip:
        return random.choice(CLIPPED_UNIT_SIZE_CHOICES)
    return random.randint(*UNIT_SIZE_RANGE)


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    must_clip: bool,
) -> bool:
    """Fill one legend dict and prebuild its rotated composite."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos: list[tuple[int, int, int]] = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (f_h, f_w) if rot_k % 2 == 0 else (f_w, f_h)
                comp_h = rh * n
                comp_w = rw * n
                oversized = comp_h > grid_h or comp_w > canvas_w
                if must_clip and not oversized:
                    continue
                if not must_clip and oversized:
                    continue
                combos.append((f_h, f_w, rot_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos[:40]:
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
            must_clip=must_clip,
        )
        return True
    return False


def _candidate_placements(leg: dict, grid_h: int, canvas_w: int) -> list[dict]:
    """Enumerate anchor placements that satisfy the visibility policy for this legend."""
    comp = leg["comp"]
    total = int(((comp != 0) & (comp != 6)).sum())
    min_visible = min(total, max(MIN_VISIBLE_NONZERO, leg["N"] * leg["N"]))
    min_clipped = min(total - 1, max(MIN_CLIPPED_NONZERO, leg["N"]))
    max_visible = int(grid_h * canvas_w * MAX_VISIBLE_FILL_FRAC)

    cands: list[dict] = []
    for pr in range(3):
        for pc in range(3):
            top = _anchor_to_top_left(pr, grid_h, comp.shape[0])
            left = _anchor_to_top_left(pc, canvas_w, comp.shape[1])
            metrics = _placement_metrics(comp, top, left, (grid_h, canvas_w))
            if metrics["visible_nonzero"] < min_visible:
                continue
            if metrics["visible_nonzero"] > max_visible:
                continue
            if leg["must_clip"]:
                if not metrics["clipped"]:
                    continue
                if metrics["clipped_nonzero"] < min_clipped:
                    continue
            else:
                if metrics["clipped"]:
                    continue
            cands.append({"pr": pr, "pc": pc, **metrics})
    random.shuffle(cands)
    return cands


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while avoiding overlap of visible composite pixels."""
    candidate_lists = [_candidate_placements(leg, grid_h, canvas_w) for leg in legends]
    if any(not cands for cands in candidate_lists):
        return False

    if len(legends) == 1:
        choice = random.choice(candidate_lists[0])
        legends[0].update(pr=choice["pr"], pc=choice["pc"], comp_top=choice["top"], comp_left=choice["left"], clip_sides=choice["clips"])
        return True

    valid_pairs = []
    for a in candidate_lists[0]:
        for b in candidate_lists[1]:
            if np.any(a["mask"] & b["mask"]):
                continue
            valid_pairs.append((a, b))
    if not valid_pairs:
        return False
    a, b = random.choice(valid_pairs)
    legends[0].update(pr=a["pr"], pc=a["pc"], comp_top=a["top"], comp_left=a["left"], clip_sides=a["clips"])
    legends[1].update(pr=b["pr"], pc=b["pc"], comp_top=b["top"], comp_left=b["left"], clip_sides=b["clips"])
    return True


def _build_checkerboard_canvas(grid_h: int, canvas_w: int, bg0: int, bg1: int, phase: int) -> np.ndarray:
    """Build a 2-color checkerboard canvas with a global phase flip."""
    rr, cc = np.indices((grid_h, canvas_w))
    parity = (rr + cc + phase) % 2
    colors = np.array([bg0, bg1], dtype=int)
    return colors[parity]


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's composite with clipping."""
    output = canvas.copy()
    composite_written = np.zeros(output.shape, dtype=bool)
    for leg in legends:
        _write_composite_clipped(output, leg["comp"], leg["comp_top"], leg["comp_left"], composite_written)
    return output


def _background_exposure_metrics(output: np.ndarray, canvas: np.ndarray, bg0: int, bg1: int) -> dict:
    """Count untouched checkerboard cells that remain visible in the final output."""
    exposed = output == canvas
    count0 = int(np.sum(exposed & (canvas == bg0)))
    count1 = int(np.sum(exposed & (canvas == bg1)))
    return {
        "total": int(exposed.sum()),
        "bg0": count0,
        "bg1": count1,
    }


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

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


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    clipped_idx = random.randrange(num_legends)

    legends = []
    for i in range(num_legends):
        must_clip = i == clipped_idx
        n = _sample_unit_size(must_clip)
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
    unit_color_sets, (bg0, bg1), frame_colors = colors
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            must_clip=(li == clipped_idx),
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    if not any(
        side
        for leg in legends
        if leg["must_clip"]
        for side in leg["clip_sides"].values()
    ):
        return None

    phase = random.randint(0, 1)
    canvas = _build_checkerboard_canvas(grid_h, canvas_w, bg0, bg1, phase)
    output = _overlay_composites(canvas, legends)

    exposure = _background_exposure_metrics(output, canvas, bg0, bg1)
    if exposure["total"] < MIN_EXPOSED_BG_TOTAL:
        return None
    if exposure["bg0"] < MIN_EXPOSED_BG_PER_COLOR or exposure["bg1"] < MIN_EXPOSED_BG_PER_COLOR:
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
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
