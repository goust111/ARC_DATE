# Combo variant of a32d8b75 (index 26)
# combo_spec: D12/0+D13/0+D16/1
# Name: 裁剪+紧凑输出+精灵锚点

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# =====================================
# a) Concrete behavior and trace
#    The input legends still encode four ingredients: a color-swapped unit
#    sprite, an 8-connected frame mask, a 3x3 anchor marker inside a 5x5
#    border-6 square, and a 3x3 rotation marker. The composite is still built
#    by tiling the recovered unit sprite into each active frame cell and then
#    rotating the whole result. The fused change happens at placement/output:
#    the marker no longer says "which canvas edge/corner the whole composite
#    goes to". Instead, each axis uses the same three canvas reference points
#    as before (start / middle / end), but the marker now selects which part of
#    the sprite is aligned there in reversed object order:
#      marker 0 -> sprite end, marker 1 -> sprite center, marker 2 -> sprite start
#    Small trace on one axis:
#      canvas length 18, composite length 8, marker index 0
#      original: start of sprite aligns to canvas start -> top-left = 0
#      combo:    end of sprite aligns to canvas start -> top-left = -(8-1) = -7
#    So the composite is intentionally pushed across the boundary, clipped by
#    the canvas, and only its visible prefix remains. After all legends are
#    rendered, the final output is tight-cropped to the bbox of written
#    composite pixels. This is why a bottom-right sprite corner aligned to the
#    canvas top-left leaves only the sprite's top-left fragment in the answer.
# b) Complexity calibration
#    The task still uses only the original three marker levels per axis and the
#    original legend structure, so the new inference burden stays local and
#    learnable. To make the fused rule observable, generation forces at least
#    one legend to be visibly clipped and rejects crops that stay too close to
#    the full hidden canvas. A second legend, when present, is kept fully
#    visible so the grid does not degenerate into several large truncated blobs.
# c) Hidden-assumption audit
#    The source solver assumed:
#    - anchor_to_top_left always returned an in-bounds top-left
#    - the visible output was the whole canvas strip
#    - anchor indices described the canvas side, not the sprite side
#    All three assumptions are false here. The write loop therefore clips
#    instead of asserting, cropping is delayed until after rendering, and the
#    anchor helper is replaced by a sprite-anchor mapping. The effective-canvas
#    parsing and legend decoding logic remain valid because separators, marker
#    squares, and frame extraction are unchanged.
# d) Placement legality
#    Only the visible in-bounds projection matters for overlap. The generator
#    computes the projected mask for every candidate anchor, keeps one clipped
#    candidate for the designated clipped legend, requires enough visible and
#    clipped pixels, and rejects any pair whose visible masks overlap. Tight
#    cropping uses the union of written pixels, not background colors.
# e) Visual observability
#    A human can compare input and output and notice two simultaneous effects:
#    the answer is smaller than the hidden canvas, and the surviving fragment is
#    not the whole reconstructed composite but the part left after a sprite-side
#    anchor pushed it beyond the border. The generator also rejects samples
#    where the combo output matches the old anchor semantics, so the D16 change
#    is not merely theoretical.
# f) Visual cleanliness
#    The canvas background is still simple banding, legends remain untouched,
#    and only one legend is forced to clip. This keeps the visible fragment and
#    the tight crop easy to read. Two-legend outputs are additionally filtered
#    by crop area ratio so they do not leave huge empty background gaps inside
#    the cropped rectangle.
# g) Change inferability
#    No new color logic is introduced. Unit colors are still swapped back on
#    read, frame color is just a mask carrier, and clipping/cropping depend only
#    on geometry. This avoids hidden color-condition rules that would be hard to
#    infer from 2-4 ARC examples.
# h) New solver assumptions and generator guarantees
#    The solver now assumes at least one legend outside the canvas paints at
#    least one pixel after clipping, and that visible painted regions do not
#    overlap. The generator guarantees those conditions by validating candidate
#    placements before assembling the input and by self-checking every sample
#    with main().


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
    Identify a left separator run and, optionally, a right separator run for a
    right-side legend panel.
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
    """Extract the top-most 4-connected non-{0,6} component and swap its two colors back."""
    mask = (legend != 0) & (legend != 6)
    pts_all = np.argwhere(mask)
    if pts_all.size == 0:
        return None
    rmin = int(pts_all[:, 0].min())
    cseed = int(np.argwhere(mask[rmin, :])[0, 0])
    h, w = legend.shape
    vis = np.zeros((h, w), dtype=bool)
    vis[rmin, cseed] = True
    st = [(rmin, cseed)]
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
    return spr, (int(r0), int(c0), int(r1), int(c1))


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels live between the unit sprite and the marker panel."""
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
    """Return the marker row/col in the 3x3 inner square."""
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """Read the 3x3 clock-hand direction below the marker square."""
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


def _axis_reference(idx: int, size: int) -> int:
    """Return start / middle / end reference on a one-dimensional canvas axis."""
    if idx <= 0:
        return 0
    if idx >= 2:
        return max(0, size - 1)
    return (size - 1) // 2 if size > 1 else 0


def _legacy_anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Original task's per-axis mapping, kept for generator-side anti-degeneracy checks."""
    canvas_anchor = _axis_reference(anchor_idx, size)
    obj_anchor = _axis_reference(anchor_idx, place_size)
    return canvas_anchor - obj_anchor


def _combo_anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """
    Combo mapping: keep the same canvas reference ladder, but interpret the
    marker as selecting the sprite's aligned part in reversed order.
    """
    canvas_anchor = _axis_reference(anchor_idx, size)
    if anchor_idx <= 0:
        obj_anchor = max(0, place_size - 1)
    elif anchor_idx >= 2:
        obj_anchor = 0
    else:
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _build_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into each active frame cell."""
    sprite_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][sprite_mask] = unit_sprite[sprite_mask]
    return comp


def _clip_slices(
    top: int, left: int, comp_shape: tuple[int, int], canvas_shape: tuple[int, int]
) -> tuple[slice, slice, slice, slice] | None:
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


def _crop_to_mask(grid: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return the tight crop spanning all True cells in mask."""
    pts = np.argwhere(mask)
    assert pts.size > 0, "tight crop requires at least one painted pixel"
    r0, c0, r1, c1 = bbox_coords(pts)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1].copy()


def _mask_bbox_dims(mask: np.ndarray) -> tuple[int, int]:
    """Return bbox height/width for a non-empty boolean mask."""
    pts = np.argwhere(mask)
    assert pts.size > 0, "bbox dims require a non-empty mask"
    r0, c0, r1, c1 = bbox_coords(pts)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild legend-defined composites on the hidden canvas, allow clipped placement, then tight-crop the visible result.

    Input structure:
    - Full-height color-6 separator columns still split the grid into one or two legend panels and a central canvas strip.
    - Each legend panel still contains a color-swapped unit sprite, a frame mask, a 5x5 border-6 marker square with one inner pixel, and a 3x3 rotation marker below it.
    - The canvas visible in the input is only an intermediate background surface; the final answer may be much smaller after cropping.

    Rule:
    - Recover the unit sprite, tile it into every active frame cell, and rotate the resulting composite from the clock-hand marker.
    - Read the marker indices, but interpret them as selecting which part of the sprite aligns to the corresponding canvas reference point on each axis; the composite may therefore extend beyond the canvas and is clipped to the in-bounds portion.
    - After all visible composite pixels are painted, discard outer rows and columns until only the minimal rectangle spanning written composite pixels remains.

    Key insight:
    - The marker now governs the sprite-side anchor rather than the canvas-side anchor, so the visible fragment after clipping can be a corner or side of a larger hidden composite, and the output shape is determined by that surviving fragment rather than by the full canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape[1] >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else g.shape[1]
    hidden_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = hidden_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(hidden_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(hidden_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = hidden_canvas.copy()
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
        top = _combo_anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _combo_anchor_to_top_left(pc, eff_w, comp.shape[1])
        _write_composite_clipped(out, comp, top, left, composite_written)

    return _crop_to_mask(out, composite_written)


# --------------- generator (combo clipped+tight-crop+sprite-anchor) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
CLIPPED_UNIT_SIZE_CHOICES = [4, 5]
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 600
MIN_VISIBLE_NONZERO = 10
MIN_CLIPPED_NONZERO = 6
MAX_VISIBLE_FILL_FRAC = 0.7
MIN_CROP_REDUCTION = 2
MAX_CROP_AREA_RATIO = 0.82


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


def _sample_unit_size(must_clip: bool, num_legends: int) -> int:
    """Bias clipped legends toward larger sprites and 2-legend layouts toward smaller non-clipped sprites."""
    if must_clip:
        return random.choice(CLIPPED_UNIT_SIZE_CHOICES)
    hi = UNIT_SIZE_RANGE[1] - 1 if num_legends == 2 else UNIT_SIZE_RANGE[1]
    return random.randint(UNIT_SIZE_RANGE[0], hi)


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
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
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                comp_h = rh * n
                comp_w = rw * n
                if comp_h > grid_h + n or comp_w > canvas_w + n:
                    continue
                combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos[:60]:
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
            comp_mask=(comp != 0) & (comp != 6),
        )
        return True
    return False


def _candidate_placements(leg: dict, grid_h: int, canvas_w: int, must_clip: bool) -> list[dict]:
    """Enumerate legal placements for one legend under the combo anchor semantics."""
    comp = leg["comp"]
    total = int(leg["comp_mask"].sum())
    min_visible = min(total, max(MIN_VISIBLE_NONZERO, leg["N"] * leg["N"]))
    min_clipped = min(total - 1, max(MIN_CLIPPED_NONZERO, leg["N"]))
    max_visible = int(grid_h * canvas_w * MAX_VISIBLE_FILL_FRAC)

    cands: list[dict] = []
    for pr in range(3):
        for pc in range(3):
            top = _combo_anchor_to_top_left(pr, grid_h, comp.shape[0])
            left = _combo_anchor_to_top_left(pc, canvas_w, comp.shape[1])
            metrics = _placement_metrics(comp, top, left, (grid_h, canvas_w))
            if metrics["visible_nonzero"] < min_visible:
                continue
            if metrics["visible_nonzero"] > max_visible:
                continue

            legacy_top = _legacy_anchor_to_top_left(pr, grid_h, comp.shape[0])
            legacy_left = _legacy_anchor_to_top_left(pc, canvas_w, comp.shape[1])
            legacy_metrics = _placement_metrics(comp, legacy_top, legacy_left, (grid_h, canvas_w))
            differs_from_legacy = (
                top != legacy_top
                or left != legacy_left
                or not np.array_equal(metrics["mask"], legacy_metrics["mask"])
            )

            if must_clip:
                if not metrics["clipped"]:
                    continue
                if metrics["clipped_nonzero"] < min_clipped:
                    continue
                if not differs_from_legacy:
                    continue
            else:
                if metrics["clipped"]:
                    continue

            cands.append(
                {
                    "pr": pr,
                    "pc": pc,
                    "differs_from_legacy": differs_from_legacy,
                    **metrics,
                }
            )
    random.shuffle(cands)
    return cands


def _crop_is_salient(mask: np.ndarray, grid_h: int, canvas_w: int) -> bool:
    """Require the final tight crop to be observably smaller than the hidden canvas."""
    crop_h, crop_w = _mask_bbox_dims(mask)
    if crop_h > grid_h - MIN_CROP_REDUCTION or crop_w > canvas_w - MIN_CROP_REDUCTION:
        return False
    return crop_h * crop_w <= int(MAX_CROP_AREA_RATIO * grid_h * canvas_w)


def _clip_axis_count(clips: dict) -> int:
    """Count how many axes are clipped, ignoring which side on that axis."""
    return int(clips["top"] or clips["bottom"]) + int(clips["left"] or clips["right"])


def _prefer_anchor_salience(cands: list[dict], must_clip: bool) -> list[dict]:
    """
    Bias clipped examples toward corner/edge anchors so the sprite-anchor
    inversion is easier to infer from a few training examples.
    """
    if not must_clip or len(cands) <= 1:
        return cands

    subset = list(cands)
    score_fns = [
        lambda c: int(c["pr"] != 1 and c["pc"] != 1),
        lambda c: int(c["pr"] != 1) + int(c["pc"] != 1),
        lambda c: _clip_axis_count(c["clips"]),
        lambda c: c["clipped_nonzero"],
    ]
    for score_fn in score_fns:
        best = max(score_fn(c) for c in subset)
        narrowed = [c for c in subset if score_fn(c) == best]
        if narrowed:
            subset = narrowed
        if len(subset) <= max(4, len(cands) // 2):
            break
    random.shuffle(subset)
    return subset


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign combo anchors while keeping visible painted pixels disjoint."""
    candidate_lists = [
        _prefer_anchor_salience(
            _candidate_placements(leg, grid_h, canvas_w, must_clip=leg["must_clip"]),
            must_clip=leg["must_clip"],
        )
        for leg in legends
    ]
    if any(not cands for cands in candidate_lists):
        return False

    if len(legends) == 1:
        good = [cand for cand in candidate_lists[0] if _crop_is_salient(cand["mask"], grid_h, canvas_w)]
        if not good:
            return False
        pick = random.choice(good)
        legends[0].update(
            pr=pick["pr"],
            pc=pick["pc"],
            comp_top=pick["top"],
            comp_left=pick["left"],
            clip_sides=pick["clips"],
            differs_from_legacy=pick["differs_from_legacy"],
            visible_mask=pick["mask"],
        )
        return True

    valid_pairs = []
    for a in candidate_lists[0]:
        for b in candidate_lists[1]:
            if np.any(a["mask"] & b["mask"]):
                continue
            if not (a["differs_from_legacy"] or b["differs_from_legacy"]):
                continue
            union_mask = a["mask"] | b["mask"]
            if not _crop_is_salient(union_mask, grid_h, canvas_w):
                continue
            valid_pairs.append((a, b, union_mask))
    if not valid_pairs:
        return False

    a, b, _ = random.choice(valid_pairs)
    legends[0].update(
        pr=a["pr"],
        pc=a["pc"],
        comp_top=a["top"],
        comp_left=a["left"],
        clip_sides=a["clips"],
        differs_from_legacy=a["differs_from_legacy"],
        visible_mask=a["mask"],
    )
    legends[1].update(
        pr=b["pr"],
        pc=b["pc"],
        comp_top=b["top"],
        comp_left=b["left"],
        clip_sides=b["clips"],
        differs_from_legacy=b["differs_from_legacy"],
        visible_mask=b["mask"],
    )
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """Stamp each legend's composite with clipping and record the painted mask."""
    output = canvas.copy()
    painted = np.zeros(output.shape, dtype=bool)
    for leg in legends:
        _write_composite_clipped(output, leg["comp"], leg["comp_top"], leg["comp_left"], painted)
    return output, painted


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
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
        n = _sample_unit_size(must_clip, num_legends)
        legends.append({"N": n, "legend_w": n + 2, "must_clip": must_clip})

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

    if not any(leg["must_clip"] and any(leg["clip_sides"].values()) for leg in legends):
        return None
    if not any(leg["differs_from_legacy"] for leg in legends):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output_full, painted_mask = _overlay_composites(canvas, legends)
    if not np.any(painted_mask):
        return None
    output = _crop_to_mask(output_full, painted_mask)

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
