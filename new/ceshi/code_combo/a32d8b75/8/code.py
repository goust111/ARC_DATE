# Combo variant of a32d8b75 (index 8)
# combo_spec: D5/1+D6/0
# Name: 旋转翻转+二进制位置编码

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for combo variant
# ---------------------------------
# a) Concrete fused behavior + trace:
#    - The overall panel layout is unchanged: unit sprite at top, frame mask in the middle,
#      then a 5x5 border-6 marker square, then a 3x3 rotation patch below it.
#    - The 5x5 square no longer means "choose one of 9 anchor cells". Instead its 3x3 inner
#      area is structured:
#        * center (1,1) is always color 4 as a sentinel,
#        * left column rows 0/1/2 encode row bits 1/2/4,
#        * right column rows 0/1/2 encode col bits 1/2/4.
#      So multiple color-4 pixels may appear at once.
#    - The 3x3 patch below now jointly encodes transform:
#        * endpoint direction still means rotation (up/right/down/left -> 0/90/180/270),
#        * center color means extra flip:
#            7 = no flip, 8 = horizontal flip, 9 = vertical flip.
#    - Small trace:
#        base rule: build composite -> rotate only -> place by coarse start/middle/end anchor.
#        fused rule: build composite -> rotate by endpoint -> flip by center color
#                    -> place at exact top-left (row_code, col_code).
#      Example: row bits at top and bottom left with right-column middle+bottom bits give
#      row_code=1+4=5, col_code=2+4=6. If the lower patch points right and its center is 8,
#      the object is rotated 90 degrees clockwise, then mirrored left-right, then placed with
#      top-left exactly at (5, 6). Under the original solver, the same marker would collapse to
#      the first nonzero inner cell, producing a coarse anchor instead of the exact offset.
#
# b) Complexity calibration:
#    - Rotation x flip has only 12 states, which is still local and learnable.
#    - Binary placement has 8 values per axis, but the encoding is factored into two visible
#      3-bit stacks, avoiding a large arbitrary codebook.
#    - The generator prefers samples where at least one legend uses a non-trivial flip and at
#      least one axis uses a multi-bit or >2 offset, so both dimensions visibly matter.
#
# c) Hidden-assumption audit against the original code path:
#    - Old position decoding assumed a unique marker pixel; invalid once multiple bits are on.
#    - Old anchor placement assumed semantic anchors {start, middle, end}; invalid because the
#      new marker stores exact top-left offsets.
#    - Old rotation decoding treated every non-{0,6} pixel in the lower patch as direction-only;
#      invalid because the center color now carries flip state independently.
#    - Rotating only the frame before tiling is no longer the cleanest abstraction because the
#      fused transform includes post-rotation flipping. The solver now builds the base composite
#      first, then applies the full transform to the composite itself.
#
# d) Placement legality:
#    - The generator computes transformed composite sizes before assigning binary offsets.
#    - Offsets are chosen only from 0..min(7, canvas_size - comp_size), so placement is exact
#      and unclipped.
#    - For 2 legends, bounding boxes are checked for overlap after the fused transform.
#
# e) Visual observability:
#    - The left marker column correlates with vertical displacement, and the right marker column
#      correlates with horizontal displacement; two-legend cases often share one axis and differ
#      on the other to make that relationship easier to notice.
#    - The lower patch is a two-channel cue: geometry gives rotation, color gives flip. The
#      generator rejects flip choices that would be visually inert on the chosen composite.
#
# f) Visual cleanliness:
#    - The position marker stays inside a rigid border-6 square with a permanent center sentinel,
#      so even when several bits are on, it still reads as one structured object rather than as
#      random noise.
#    - The rotation/flip patch remains a sparse 3x3 symbol directly under the marker square,
#      keeping instruction regions separated from the sprite and frame.
#
# g) Color-rule inference:
#    - The only new color-conditioned rule is the lower patch center color {7,8,9}, and it is
#      always seen in the same geometric role. Position encoding does not rely on color identity
#      beyond "present vs absent" inside the bordered square.
#
# h) New solver assumptions that the generator explicitly guarantees:
#    - The lower 3x3 patch contains exactly the center plus one endpoint.
#    - The center color of that patch is always one of {7,8,9}.
#    - The 5x5 marker square always contains the center sentinel at inner (1,1).
#    - At least one legend per sample has an effective non-zero flip.
#    - At least one legend per sample places differently from the legacy coarse-anchor decoder.
#    - Transformed composites fit and do not overlap.


MARKER_PANEL_HEIGHT = 8
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up/right/down/left
FLIP_NONE = 0
FLIP_H = 1
FLIP_V = 2
FLIP_COLOR_TO_MODE = {7: FLIP_NONE, 8: FLIP_H, 9: FLIP_V}
FLIP_MODE_TO_COLOR = {v: k for k, v in FLIP_COLOR_TO_MODE.items()}

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7]  # reserve 8/9 for flip-center states
GEN_RETRIES = 1200
LEGEND_DESIGN_RETRIES = 160


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
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
        swapped = spr.copy()
        swapped[spr == a] = -1
        swapped[spr == b] = a
        swapped[swapped == -1] = b
        spr = swapped
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
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


def _decode_binary_offsets_from_border_square(
    g: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int]:
    sq = g[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    assert int(inner[1, 1]) == 4, "binary marker square must contain center sentinel color 4"
    row_code = 0
    col_code = 0
    for bit in range(3):
        if int(inner[bit, 0]) not in (0, 6):
            row_code |= 1 << bit
        if int(inner[bit, 2]) not in (0, 6):
            col_code |= 1 << bit
    return row_code, col_code


def _rotation_flip_from_marker(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int]:
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    sub = g[r0 : r0 + 3, c0 : c0 + 3]
    assert sub.shape == (3, 3), "rotation marker patch out of bounds"
    center_color = int(sub[1, 1])
    assert center_color in FLIP_COLOR_TO_MODE, "invalid flip marker color"
    pts = np.argwhere((sub != 0) & (sub != 6))
    endpoint = [(int(r), int(c)) for r, c in pts if not (int(r) == 1 and int(c) == 1)]
    assert len(endpoint) == 1, "marker must contain center plus exactly one endpoint"
    rr, cc = endpoint[0]
    dr, dc = rr - 1, cc - 1
    if abs(dr) >= abs(dc):
        rot_k = 0 if dr < 0 else 2
    else:
        rot_k = 1 if dc > 0 else 3
    return rot_k, FLIP_COLOR_TO_MODE[center_color]


def _offset_from_binary_code(code: int, size: int, place_size: int) -> int:
    max_offset = size - place_size
    assert max_offset >= 0, "composite larger than canvas"
    assert 0 <= code <= min(7, max_offset), "binary offset outside legal range"
    return code


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
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


def _apply_transform(arr: np.ndarray, rot_k: int, flip_mode: int) -> np.ndarray:
    out = np.rot90(arr, (-rot_k) % 4)
    if flip_mode == FLIP_H:
        out = np.fliplr(out)
    elif flip_mode == FLIP_V:
        out = np.flipud(out)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged tiled sprites on the canvas using fused legend instructions for exact placement and mirrored rotation.

    Input structure:
    - Full-height color-6 separator columns still split the grid into legend panel(s) and the canvas.
    - Each legend contains a two-color unit sprite, a single-color frame mask, a 5x5 bordered position square, and a 3x3 transform patch below it.
    - Inside the 5x5 square, the left and right columns of the inner 3x3 area store binary row/col bits while the center is a fixed sentinel.
    - Inside the 3x3 lower patch, the endpoint direction gives the quarter-turn and the center color gives whether an extra horizontal or vertical flip is required.

    Rule:
    - Decode the unit sprite, tile it into every active frame cell, and treat that as the base composite.
    - Apply the fused transform in a fixed order: rotate by the endpoint direction, then flip according to the center color.
    - Decode the binary marker square into exact top-left row/column offsets and stamp the transformed composite at that location on the canvas.

    Key insight: The legend now hides two independent instruction channels at once: the lower patch combines rotation with mirroring, and the upper bordered square uses on/off bit patterns rather than a single anchor pixel to specify the final placement.
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
        frame_mask = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        rot_k, flip_mode = _rotation_flip_from_marker(g, marker_r0, marker_c0)
        base_comp = _build_composite(unit_sprite, frame_mask)
        comp = _apply_transform(base_comp, rot_k, flip_mode)

        row_code, col_code = _decode_binary_offsets_from_border_square(g, marker_r0, marker_c0)
        top = _offset_from_binary_code(row_code, eff_h, comp.shape[0])
        left = _offset_from_binary_code(col_code, eff_w, comp.shape[1])

        ch, cw = comp.shape
        assert top + ch <= eff_h and left + cw <= eff_w, "binary placement exceeds effective canvas"
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


def _allocate_colors(num_legends: int):
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

    all_set = set(USABLE_COLORS)
    frame_colors = []
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _make_asymmetric_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    for _ in range(240):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        if _same_shape(sprite, np.fliplr(sprite)):
            continue
        if _same_shape(sprite, np.flipud(sprite)):
            continue
        if _same_shape(sprite, np.rot90(sprite, 1)):
            continue
        if _same_shape(sprite, np.rot90(sprite, 2)):
            continue
        return sprite
    raise RuntimeError("failed to build asymmetric unit sprite")


def _make_frame_mask(frame_h: int, frame_w: int) -> np.ndarray:
    for _ in range(180):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
        active = int(mask.sum())
        if n8 != 1:
            continue
        if active < max(4, (frame_h * frame_w) // 3):
            continue
        if _same_shape(mask, np.fliplr(mask)):
            continue
        if _same_shape(mask, np.flipud(mask)):
            continue
        if frame_h == frame_w and _same_shape(mask, np.rot90(mask, 1)):
            continue
        return mask
    raise RuntimeError("failed to build asymmetric frame mask")


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    prefer_small: bool = False,
) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors
    avail = marker_r0 - (n + 1)

    for _ in range(LEGEND_DESIGN_RETRIES):
        sprite = _make_asymmetric_unit_sprite(n, uc1, uc2)
        legend_sprite = _color_swap(sprite, uc1, uc2)

        combos = []
        for frame_h in range(*FRAME_H_RANGE):
            if avail < frame_h + 1:
                continue
            for frame_w in range(*FRAME_W_RANGE):
                for rot_k in range(4):
                    rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                    if rh * n <= grid_h and rw * n <= canvas_w:
                        combos.append((frame_h, frame_w, rot_k, rh * rw))
        if not combos:
            return False

        if prefer_small:
            combos.sort(key=lambda x: x[3])
            combos = combos[: max(1, len(combos) // 3)]
        frame_h, frame_w, rot_k, _ = random.choice(combos)
        frame_mask = _make_frame_mask(frame_h, frame_w)
        base_comp = _build_composite(sprite, frame_mask)
        rotated_only_comp = _apply_transform(base_comp, rot_k, FLIP_NONE)

        valid_modes = [FLIP_NONE]
        if not _same_shape(rotated_only_comp, np.fliplr(rotated_only_comp)):
            valid_modes.append(FLIP_H)
        if not _same_shape(rotated_only_comp, np.flipud(rotated_only_comp)):
            valid_modes.append(FLIP_V)
        if len(valid_modes) == 1:
            continue

        flip_mode = random.choices(valid_modes, weights=[1] + [4] * (len(valid_modes) - 1))[0]
        transformed_comp = _apply_transform(base_comp, rot_k, flip_mode)
        if flip_mode != FLIP_NONE and _same_shape(rotated_only_comp, transformed_comp):
            continue

        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            rot_k=rot_k,
            flip_mode=flip_mode,
            base_comp=base_comp,
            rotated_only_comp=rotated_only_comp,
            transformed_comp=transformed_comp,
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    return leg["transformed_comp"].shape


def _code_options(max_offset: int) -> list[int]:
    return list(range(min(7, max_offset) + 1))


def _code_complexity(code: int) -> int:
    return int(bin(code).count("1"))


def _axis_code_weight(code: int) -> int:
    pop = _code_complexity(code)
    weight = 1
    if code > 0:
        weight += 2
    if code > 2:
        weight += 2
    if pop >= 2:
        weight += 2
    if code in (3, 5, 6):
        weight += 2
    return weight


def _single_legend_code_weight(row_code: int, col_code: int) -> int:
    return _axis_code_weight(row_code) + _axis_code_weight(col_code)


def _pair_code_weight(pair: tuple[int, int, int, int]) -> int:
    ar, ac, br, bc = pair
    base = sum(_axis_code_weight(code) for code in pair)
    same_one_axis = int((ar == br) ^ (ac == bc))
    distinct = len({ar, ac, br, bc})
    mixed = sum(int(code in (3, 5, 6)) for code in pair)
    return base + 3 * same_one_axis + 2 * distinct + mixed


def _legacy_anchor_from_binary_codes(row_code: int, col_code: int) -> tuple[int, int]:
    if row_code & 1:
        return 0, 0
    if col_code & 1:
        return 0, 2
    if row_code & 2:
        return 1, 0
    return 1, 1


def _legacy_top_left_for_codes(
    row_code: int, col_code: int, canvas_h: int, canvas_w: int, comp_h: int, comp_w: int
) -> tuple[int, int]:
    pr, pc = _legacy_anchor_from_binary_codes(row_code, col_code)
    return (
        _anchor_to_top_left(pr, canvas_h, comp_h),
        _anchor_to_top_left(pc, canvas_w, comp_w),
    )


def _assign_binary_offsets(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        leg = legends[0]
        ch, cw = _comp_size(leg)
        row_opts = _code_options(grid_h - ch)
        col_opts = _code_options(canvas_w - cw)
        if not row_opts or not col_opts:
            return False
        all_choices = [(r, c) for r in row_opts for c in col_opts if r > 0 or c > 0]
        if not all_choices:
            all_choices = [(r, c) for r in row_opts for c in col_opts]
        choices = [
            (r, c)
            for r, c in all_choices
            if _legacy_top_left_for_codes(r, c, grid_h, canvas_w, ch, cw) != (r, c)
        ]
        if not choices:
            choices = all_choices
        weights = [_single_legend_code_weight(r, c) for r, c in choices]
        row_code, col_code = random.choices(choices, weights=weights, k=1)[0]
        leg.update(row_code=row_code, col_code=col_code, comp_top=row_code, comp_left=col_code)
        return True

    a, b = legends
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    a_rows = _code_options(grid_h - a_ch)
    a_cols = _code_options(canvas_w - a_cw)
    b_rows = _code_options(grid_h - b_ch)
    b_cols = _code_options(canvas_w - b_cw)
    if not a_rows or not a_cols or not b_rows or not b_cols:
        return False

    valid_pairs = []
    for ar in a_rows:
        for ac in a_cols:
            for br in b_rows:
                for bc in b_cols:
                    if ar < br + b_ch and br < ar + a_ch and ac < bc + b_cw and bc < ac + a_cw:
                        continue
                    valid_pairs.append((ar, ac, br, bc))
    if not valid_pairs:
        return False

    effective_pairs = []
    for ar, ac, br, bc in valid_pairs:
        a_legacy = _legacy_top_left_for_codes(ar, ac, grid_h, canvas_w, a_ch, a_cw)
        b_legacy = _legacy_top_left_for_codes(br, bc, grid_h, canvas_w, b_ch, b_cw)
        if a_legacy != (ar, ac) or b_legacy != (br, bc):
            effective_pairs.append((ar, ac, br, bc))
    if effective_pairs:
        valid_pairs = effective_pairs

    weights = [_pair_code_weight(pair) for pair in valid_pairs]
    ar, ac, br, bc = random.choices(valid_pairs, weights=weights, k=1)[0]
    a.update(row_code=ar, col_code=ac, comp_top=ar, comp_left=ac)
    b.update(row_code=br, col_code=bc, comp_top=br, comp_left=bc)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
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
    output = canvas.copy()
    for leg in legends:
        comp = leg["transformed_comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool):
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 2, mc0 + 2] = 4
    for bit in range(3):
        if leg["row_code"] & (1 << bit):
            grid[marker_r0 + 1 + bit, mc0 + 1] = 4
        if leg["col_code"] & (1 << bit):
            grid[marker_r0 + 1 + bit, mc0 + 3] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = FLIP_MODE_TO_COLOR[leg["flip_mode"]]
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["legend_sprite"]

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _legend_has_effective_flip(leg: dict) -> bool:
    return int(leg["flip_mode"]) != FLIP_NONE and not _same_shape(leg["rotated_only_comp"], leg["transformed_comp"])


def _legend_has_effective_binary_position(leg: dict, grid_h: int, canvas_w: int) -> bool:
    ch, cw = _comp_size(leg)
    legacy_top, legacy_left = _legacy_top_left_for_codes(leg["row_code"], leg["col_code"], grid_h, canvas_w, ch, cw)
    return (legacy_top, legacy_left) != (leg["comp_top"], leg["comp_left"])


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        if num_legends == 2:
            n = random.choices([2, 3, 4, 5], weights=[6, 5, 2, 1])[0]
        else:
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
    for i in range(num_legends):
        if not _design_legend(
            legends[i],
            unit_color_sets[i],
            frame_colors[i],
            marker_r0,
            grid_h,
            canvas_w,
            prefer_small=(num_legends == 2),
        ):
            return None

    if not any(_legend_has_effective_flip(leg) for leg in legends):
        return None

    if not _assign_binary_offsets(legends, grid_h, canvas_w):
        return None

    if not any(_legend_has_effective_binary_position(leg, grid_h, canvas_w) for leg in legends):
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
