# Combo variant of a32d8b75 (index 29)
# combo_spec: D5/1+D9/3
# Name: 旋转翻转+对角条纹背景

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the fused variant
# a) Concrete behavior trace:
#    - Original task: read a 2-color unit sprite, tile it through the frame mask, rotate the whole
#      composite by a 4-way clock-hand cue, then place it on a striped canvas using the 3x3 anchor.
#    - New task: the same 3x3 clock-hand still gives quarter-turn direction, but the marker center
#      now adds a second instruction: 7=no flip, 8=horizontal flip, 9=vertical flip. The output
#      transform is therefore "rotate first, then flip if requested".
#    - Example trace: suppose the frame builds a 12x8 composite, the hand points right, and the
#      marker center is 8. Old rule => rotate to 8x12 and stop. New rule => rotate to 8x12, then
#      mirror left-right. Meanwhile the canvas beneath it is no longer horizontal/vertical bands;
#      it is a 2-color diagonal stripe field based on either r+c or r-c, so the same sprite lands
#      on slanted texture that can run parallel to or across its transformed silhouette.
# b) Complexity calibration:
#    - The transform cue has 12 states total (4 rotations x 3 flip modes), but it is factored into
#      two local signals that repeat in every legend: hand geometry and center color.
#    - The background only has 2 orientations (main-diagonal or anti-diagonal stripes) and 2 colors,
#      which keeps the new canvas visually obvious without creating a second hard inference problem.
# c) Broken old assumptions and fixes:
#    - The original solver assumed every non-{0,6} pixel in the marker patch belonged to the hand.
#      That fails once the center color itself carries meaning, so the solver now reads center and
#      endpoint separately and asserts that there is exactly one endpoint in addition to the center.
#    - The original generator allowed symmetric sprites/frames because pure rotation was enough.
#      Here a flip may otherwise become invisible, so the generator rejects composites whose chosen
#      non-zero flip matches the rotated-only result.
#    - The solver never needed to parse the background pattern explicitly before, and still does not;
#      diagonal stripes only change the generated canvas values, not the placement mechanics.
# d) Placement legality:
#    - Horizontal/vertical flips preserve the rotated composite bounding box, so the original anchor
#      fit logic still applies. The generator still computes transformed composite sizes up front and
#      rejects legend pairs whose placed bounding boxes would overlap or leave the canvas.
# e) Visual observability:
#    - Colors 8 and 9 are reserved for flip cues, so the new control channel is visually separated
#      from sprite/frame/background colors. The generator forces at least one flipped legend and
#      makes sure that flip changes the output, preventing accidental regression to rotation-only.
#    - The diagonal canvas uses clean 2-color slanted stripes on the raw input canvas, so a human
#      can directly see that the background rule changed even before comparing input/output.
# f) Visual cleanliness:
#    - The sprite still overwrites the background cleanly, with no extra overlay channels. The
#      background remains regular, the legend remains columnar, and transformed composites are kept
#      non-overlapping, so the fused variant stays readable instead of becoming cluttered.
# g) New solver assumptions and generator guarantees:
#    - Solver assumptions: marker center color is in {7,8,9}; marker patch contains center plus one
#      endpoint; frame is 8-connected; every requested composite fits inside the effective canvas.
#    - Generator guarantees: it writes exactly that marker format, reserves 8/9 from ordinary color
#      pools, samples connected frame masks, checks transformed composite fit/non-overlap, and uses
#      a true diagonal-stripe background model rather than axis-aligned banding.


MARKER_PANEL_HEIGHT = 8
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up/right/down/left
FLIP_NONE = 0
FLIP_H = 1
FLIP_V = 2
FLIP_COLOR_TO_MODE = {7: FLIP_NONE, 8: FLIP_H, 9: FLIP_V}
FLIP_MODE_TO_COLOR = {mode: color for color, mode in FLIP_COLOR_TO_MODE.items()}

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 4]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
DIAG_STRIPE_WIDTH_RANGE = (2, 5)
USABLE_COLORS = [1, 2, 3, 4, 5, 7]  # reserve 8/9 for flip cues in the marker center
GEN_RETRIES = 300
LEGEND_DESIGN_RETRIES = 120
GENERATION_PLANS = (
    {"num_legends": 1, "bg_mode": "sum", "required_flips": (FLIP_H,)},
    {"num_legends": 2, "bg_mode": "diff", "required_flips": (FLIP_V, FLIP_NONE)},
    {"num_legends": 1, "bg_mode": "diff", "required_flips": (FLIP_V,)},
    {"num_legends": 2, "bg_mode": "sum", "required_flips": (FLIP_H, FLIP_V)},
    {"num_legends": 1, "bg_mode": "sum", "required_flips": (FLIP_H,)},
    {"num_legends": 2, "bg_mode": "diff", "required_flips": (FLIP_NONE, FLIP_H)},
)
_generation_plan_index = 0


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


def _position_indices_from_border_square(g: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = g[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


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
    Assemble and place enlarged tiled sprites on a diagonally striped canvas using legend panels that encode both rotation and mirroring.

    Input structure:
    - Full-height color-6 separator columns still carve the grid into legend panel(s) and the main canvas.
    - Each legend contains a 2-color unit sprite, a single-color 8-connected frame mask, a 5x5 bordered anchor square, and a 3x3 marker patch below it.
    - In that 3x3 patch, the endpoint direction gives the quarter-turn, while the center color gives the extra flip state.

    Rule:
    - Decode the unit sprite by undoing the legend's color swap, tile it into every active frame cell, and build one base composite per legend.
    - Transform the composite in a fixed order: rotate by the clock-hand direction, then optionally flip horizontally or vertically according to the marker center color.
    - Place the transformed composite onto the preserved canvas background using the anchor cell inside the bordered square.

    Key insight: The marker patch now carries two independent instructions at once: geometry says how much to rotate, and center color says whether the rotated result must be mirrored before placement.
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
        comp = _apply_transform(_build_composite(unit_sprite, frame_mask), rot_k, flip_mode)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        for r in range(comp.shape[0]):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(comp.shape[1]):
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


def _same_shape(arr: np.ndarray, other: np.ndarray) -> bool:
    return arr.shape == other.shape and np.array_equal(arr, other)


def _make_asymmetric_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray:
    for _ in range(200):
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
    for _ in range(120):
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
    unit_colors,
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    required_flip_mode: int | None,
) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors

    for _ in range(LEGEND_DESIGN_RETRIES):
        sprite = _make_asymmetric_unit_sprite(n, uc1, uc2)
        legend_sprite = _color_swap(sprite, uc1, uc2)

        avail = marker_r0 - (n + 1)
        combos = []
        for frame_h in range(*FRAME_H_RANGE):
            if avail < frame_h + 1:
                continue
            for frame_w in range(*FRAME_W_RANGE):
                for rot_k in range(4):
                    rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                    if rh * n <= grid_h and rw * n <= canvas_w:
                        combos.append((frame_h, frame_w, rot_k))
        if not combos:
            return False

        frame_h, frame_w, rot_k = random.choice(combos)
        frame_mask = _make_frame_mask(frame_h, frame_w)
        base_comp = _build_composite(sprite, frame_mask)
        rotated_only = _apply_transform(base_comp, rot_k, FLIP_NONE)

        visible_modes = [FLIP_NONE]
        if not _same_shape(rotated_only, np.fliplr(rotated_only)):
            visible_modes.append(FLIP_H)
        if not _same_shape(rotated_only, np.flipud(rotated_only)):
            visible_modes.append(FLIP_V)
        if len(visible_modes) == 1 and required_flip_mode != FLIP_NONE:
            continue

        if required_flip_mode is not None:
            if required_flip_mode not in visible_modes:
                continue
            flip_mode = required_flip_mode
        else:
            flip_mode = random.choices(visible_modes, weights=[1] + [3] * (len(visible_modes) - 1))[0]
        transformed_comp = _apply_transform(base_comp, rot_k, flip_mode)
        if flip_mode != FLIP_NONE and _same_shape(rotated_only, transformed_comp):
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
            transformed_comp=transformed_comp,
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    return leg["transformed_comp"].shape


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
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


def _build_diagonal_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int, mode: str | None = None) -> np.ndarray:
    stripe_w = random.randint(*DIAG_STRIPE_WIDTH_RANGE)
    if mode is None:
        mode = random.choice(["sum", "diff"])
    rr = np.arange(grid_h, dtype=int)[:, None]
    cc = np.arange(canvas_w, dtype=int)[None, :]
    if mode == "sum":
        stripe_idx = (rr + cc) // stripe_w
    else:
        stripe_idx = (rr - cc + (grid_h + canvas_w)) // stripe_w
    colors = np.array([bg1, bg2], dtype=int)
    return colors[stripe_idx % 2]


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    output = canvas.copy()
    for leg in legends:
        comp = leg["transformed_comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        region[comp != 0] = comp[comp != 0]
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
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

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


def _gen_attempt(plan: dict | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    if plan is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
        bg_mode = None
        required_flips = [None] * num_legends
    else:
        num_legends = int(plan["num_legends"])
        bg_mode = str(plan["bg_mode"])
        required_flips = list(plan["required_flips"])
        assert len(required_flips) == num_legends, "required flip count must match legend count"

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
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
            required_flips[li],
        ):
            return None

    if not any(int(leg["flip_mode"]) != FLIP_NONE for leg in legends):
        return None
    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_diagonal_canvas(grid_h, canvas_w, bg1, bg2, mode=bg_mode)
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
    global _generation_plan_index
    plan = GENERATION_PLANS[_generation_plan_index % len(GENERATION_PLANS)]
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(plan=plan)
        if result is not None:
            _generation_plan_index += 1
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
