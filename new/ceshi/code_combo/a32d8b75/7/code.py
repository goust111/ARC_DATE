# Combo variant of a32d8b75 (index 7)
# combo_spec: D4/1+D5/2
# Name: 位置旋转铺贴+双旋转
#
# Design analysis:
# a) Concrete behavior trace:
#    The original task reads a swapped unit sprite, a frame mask, one anchor marker,
#    and one rotation marker, then tiles one fixed sprite orientation into the frame
#    and rotates the whole composite once. The fused variant splits that rotation
#    information into two layers and adds a third positional term during tiling:
#      original:  tile(sprite, frame) -> rotate whole composite by k
#      fused   :  rotate(frame, kf) + for each active cell:
#                 rotate(sprite, ks + kpos(cell in rotated frame)) -> tile
#    Example: if the patch says frame=right and sprite-base=down, then the frame is
#    rotated 90 deg clockwise first. A cell in the rotated frame's upper sector gets
#    sprite rotation 180 deg, while a cell in the right sector gets 270 deg. The
#    difference from either single variant is that the repeated motif is controlled
#    both by a global baseline angle and by the cell's local direction field.
# b) Complexity calibration:
#    All rotations still live in four discrete quarter-turn buckets. The lower 3x3
#    patch uses color 7 on cardinals for frame rotation and color 8 on corners for
#    sprite-base rotation, so the code space stays small and visually repeated.
# c) Hidden assumptions audit:
#    The original composite loop assumed one shared sprite orientation for every
#    active frame cell; this is no longer valid and is replaced by per-cell rotation.
#    The original "one marker -> one whole-object rotation" assumption also breaks,
#    so `main()` must decode two markers separately. The bbox/anchor logic remains
#    valid because the unit sprite stays NxN under quarter-turn rotation.
# d) Placement legality:
#    Generation stays output-first: choose frame/sprite/rotations, compute composite
#    size from the rotated frame, assign anchors that keep every composite in bounds,
#    reject bounding-box overlap for the 2-legend case, then back-fill the input.
# e) Visual observability:
#    The generator forces the unit sprite to have four distinct quarter-turns and
#    forces the rotated frame to occupy at least three direction sectors. It also
#    requires both decoded rotations to be non-zero and different, so both controls
#    affect the visible result across examples.
# f) Visual cleanliness:
#    The anchor square is unchanged; the dual-rotation patch adds exactly two pixels
#    in the same 3x3 area. Reserved colors 7 and 8 are used only in that patch, so
#    the legends stay readable and do not blend into sprite/frame colors.
# g) Change inferability:
#    Color changes are not arbitrary. The only color manipulation remains the same
#    legend-side unit color swap as the base task; all new behavior is conveyed by
#    position in the marker patch and by the repeated orientation pattern on output.

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


FRAME_ROT_MARKER_COLOR = 7
SPRITE_ROT_MARKER_COLOR = 8
POSITION_MARKER_COLOR = 4
FRAME_ROT_POSITIONS = [(0, 1), (1, 2), (2, 1), (1, 0)]  # up/right/down/left
SPRITE_ROT_POSITIONS = [(0, 0), (0, 2), (2, 2), (2, 0)]  # tl/tr/br/bl


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
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
    """
    Identify a left separator run (between left legend and canvas), and optionally a
    right separator run (between canvas and a right-side legend panel).
    """
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


def _rotation_patch_looks_valid(g: np.ndarray, r0: int, c0: int) -> bool:
    sub = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
    if sub.shape != (3, 3):
        return False
    if np.count_nonzero(sub == FRAME_ROT_MARKER_COLOR) != 1:
        return False
    if np.count_nonzero(sub == SPRITE_ROT_MARKER_COLOR) != 1:
        return False
    allowed = {0, 6, FRAME_ROT_MARKER_COLOR, SPRITE_ROT_MARKER_COLOR}
    return all(int(x) in allowed for x in np.unique(sub))


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of a marker panel."""
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
            if np.count_nonzero((inner != 0) & (inner != 6)) != 1:
                continue
            if not _rotation_patch_looks_valid(g, r0, c0):
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


def _rot_index_from_position(pos: tuple[int, int], mapping: list[tuple[int, int]], label: str) -> int:
    if pos not in mapping:
        raise AssertionError(f"{label} marker at invalid position {pos}")
    return mapping.index(pos)


def _dual_rotations_from_patch(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    """
    Decode two independent rotations from the 3x3 patch below the marker square.

    The frame rotation uses color 7 on a cardinal cell:
      up/right/down/left -> 0/1/2/3.
    The sprite-base rotation uses color 8 on a corner cell:
      top-left/top-right/bottom-right/bottom-left -> 0/1/2/3.
    """
    h, w = g.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
    frame_pts = np.argwhere(sub == FRAME_ROT_MARKER_COLOR)
    sprite_pts = np.argwhere(sub == SPRITE_ROT_MARKER_COLOR)
    assert len(frame_pts) == 1, "frame rotation marker missing or duplicated"
    assert len(sprite_pts) == 1, "sprite rotation marker missing or duplicated"
    frame_pos = tuple(map(int, frame_pts[0]))
    sprite_pos = tuple(map(int, sprite_pts[0]))
    frame_k = _rot_index_from_position(frame_pos, FRAME_ROT_POSITIONS, "frame")
    sprite_k = _rot_index_from_position(sprite_pos, SPRITE_ROT_POSITIONS, "sprite")
    return frame_k, sprite_k


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


def _cell_rotation_from_position(r: int, c: int, h: int, w: int) -> int:
    """Map a frame-cell position to one of four compass-facing sprite rotations."""
    cr = (h - 1) / 2.0
    cc = (w - 1) / 2.0
    dr = float(r) - cr
    dc = float(c) - cc
    if dr == 0.0 and dc == 0.0:
        return 0
    if abs(dr) >= abs(dc):
        return 0 if dr < 0 else 2
    return 1 if dc > 0 else 3


def _direction_histogram(frame_mask: np.ndarray) -> dict[int, int]:
    hist = {0: 0, 1: 0, 2: 0, 3: 0}
    fh, fw = frame_mask.shape
    for i, j in np.argwhere(frame_mask):
        hist[_cell_rotation_from_position(int(i), int(j), fh, fw)] += 1
    return hist


def _rotated_frame(frame_mask0: np.ndarray, frame_k: int) -> np.ndarray:
    return np.rot90(frame_mask0, (-frame_k) % 4)


def _build_combo_composite(
    unit_sprite: np.ndarray, frame_mask0: np.ndarray, frame_k: int, sprite_base_k: int
) -> np.ndarray:
    """
    Rotate the frame by `frame_k`. For each active cell in the rotated frame, rotate
    the unit sprite by `sprite_base_k + local_direction(cell)` and tile it there.
    """
    frame_mask = _rotated_frame(frame_mask0, frame_k)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    rot_sprites = [np.rot90(unit_sprite, (-k) % 4) for k in range(4)]
    rot_masks = [(spr != 0) & (spr != 6) for spr in rot_sprites]
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            local_k = _cell_rotation_from_position(i, j, fh, fw)
            total_k = (sprite_base_k + local_k) % 4
            tile = rot_sprites[total_k]
            tile_mask = rot_masks[total_k]
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][tile_mask] = tile[tile_mask]
    return comp


def _build_uniform_composite(
    unit_sprite: np.ndarray, frame_mask0: np.ndarray, frame_k: int, sprite_base_k: int
) -> np.ndarray:
    """Control baseline: split rotation without position-dependent local rotation."""
    frame_mask = _rotated_frame(frame_mask0, frame_k)
    sprite = np.rot90(unit_sprite, (-sprite_base_k) % 4)
    sprite_mask = (sprite != 0) & (sprite != 6)
    uh, uw = sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][sprite_mask] = sprite[sprite_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode legend panels and stamp fused radial-split-rotation composites onto the canvas.

    Input structure:
    - Full-height color-6 separators split the grid into one or two legend panels and a canvas.
    - Each legend panel contains a two-color unit sprite, a frame template, a 5x5 anchor square,
      and a 3x3 dual-rotation patch under that square.
    - In the 3x3 patch, color 7 on a cardinal cell encodes the frame rotation, while color 8 on
      a corner cell encodes the unit sprite's baseline rotation.

    Rule:
    - Read the unit sprite with the original color-swap undo, read the frame mask, and rotate the
      frame by the color-7 marker.
    - For each active cell of that rotated frame, compute an extra quarter-turn from the cell's
      direction relative to the frame center, add it to the color-8 baseline rotation, and stamp
      the resulting sprite orientation into that cell.
    - Place the assembled composite on the canvas using the anchor encoded by the single pixel
      inside the 5x5 marker square.

    Key insight:
    - The lower patch no longer describes a single whole-object turn. One marker sets the large
      frame layout, another sets a global sprite baseline, and the final visible orientation at
      each tile comes from adding that baseline to a local position-dependent rotation field.
    """
    g = np.asarray(grid, dtype=int)
    _, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
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

        frame_k, sprite_base_k = _dual_rotations_from_patch(g, marker_r0, marker_c0)
        comp = _build_combo_composite(unit_sprite, frame_mask0, frame_k, sprite_base_k)

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


# --------------- generator (radial_split_rotation_fusion) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
# Two-legend cases are rejected more often by the no-overlap / fit constraints, so
# bias the proposal stage toward them to keep accepted samples branch-balanced.
LEGEND_COUNT_WEIGHTS = [1, 25]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
CORE_COLORS = [1, 2, 3, 4, 5, 9]
GEN_RETRIES = 500
SPRITE_RETRIES = 200
FRAME_RETRIES = 200


def _allocate_colors(num_legends: int):
    """Pick unit colors (2 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(CORE_COLORS)
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
    all_set = set(CORE_COLORS)
    for unit_colors in unit_color_sets:
        candidates = list(all_set - set(unit_colors))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _all_rotations_distinct(arr: np.ndarray) -> bool:
    rots = [np.rot90(arr, k) for k in range(4)]
    for i in range(4):
        for j in range(i + 1, 4):
            a, b = rots[i], rots[j]
            if a.shape == b.shape and np.array_equal(a, b):
                return False
    return True


def _make_asymmetric_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    for _ in range(SPRITE_RETRIES):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _all_rotations_distinct(sprite):
            return sprite
    return None


def _frame_has_visible_directionality(frame_mask: np.ndarray) -> bool:
    hist = _direction_histogram(frame_mask)
    used_dirs = sum(v > 0 for v in hist.values())
    has_vertical = hist[0] > 0 or hist[2] > 0
    has_horizontal = hist[1] > 0 or hist[3] > 0
    return used_dirs >= 3 and has_vertical and has_horizontal


def _make_frame_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(FRAME_RETRIES):
        frame_mask = random_sprite(
            h,
            w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3)))
        if n8 != 1:
            continue
        if not _all_rotations_distinct(frame_mask):
            continue
        if not _frame_has_visible_directionality(frame_mask):
            continue
        return frame_mask
    return None


def _fusion_is_visible(sprite: np.ndarray, frame_mask: np.ndarray, frame_k: int, sprite_base_k: int) -> bool:
    combo = _build_combo_composite(sprite, frame_mask, frame_k, sprite_base_k)
    radial_only = _build_combo_composite(sprite, frame_mask, frame_k, 0)
    dual_only = _build_uniform_composite(sprite, frame_mask, frame_k, sprite_base_k)
    if np.array_equal(combo, radial_only):
        return False
    if np.array_equal(combo, dual_only):
        return False
    return True


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill in sprite, frame, and fused rotations for one legend dict. Returns False on failure."""
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_asymmetric_unit_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for frame_k in (1, 2, 3):
                rh, rw = (frame_h, frame_w) if frame_k % 2 == 0 else (frame_w, frame_h)
                if rh * n > grid_h or rw * n > canvas_w:
                    continue
                for sprite_base_k in (1, 2, 3):
                    if sprite_base_k == frame_k:
                        continue
                    combos.append((frame_h, frame_w, frame_k, sprite_base_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, frame_k, sprite_base_k in combos:
        frame_mask = _make_frame_mask(frame_h, frame_w)
        if frame_mask is None:
            continue
        rotated_frame = _rotated_frame(frame_mask, frame_k)
        if not _frame_has_visible_directionality(rotated_frame):
            continue
        if not _fusion_is_visible(sprite, frame_mask, frame_k, sprite_base_k):
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            frame_rot_k=frame_k,
            sprite_base_k=sprite_base_k,
            rotated_frame=rotated_frame,
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the fused composite after frame rotation."""
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
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

    a, b = legends
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


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's fused composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_combo_composite(leg["sprite"], leg["frame_mask"], leg["frame_rot_k"], leg["sprite_base_k"])
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
    """Draw one legend panel (separator, marker square, fused-rotation patch, sprite, frame)."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = POSITION_MARKER_COLOR

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    fr, fc = FRAME_ROT_POSITIONS[leg["frame_rot_k"]]
    sr, sc = SPRITE_ROT_POSITIONS[leg["sprite_base_k"]]
    grid[rr0 + fr, mc0 + 1 + fc] = FRAME_ROT_MARKER_COLOR
    grid[rr0 + sr, mc0 + 1 + sc] = SPRITE_ROT_MARKER_COLOR

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


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

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

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
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
