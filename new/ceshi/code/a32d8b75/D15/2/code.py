# Variant of a32d8b75
# Dimension: D15 — 处理流程的步骤顺序 (multi_step)
# Change: 五步流程：读取图例 -> 铺贴构建 -> 旋转 -> 确定位置 -> 放置... → 条件分支：根据框架特征选择不同的后续操作

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis
# a) Concretization and behavior trace
#    The visible cue stays in the same legend frame: the unit sprite sits above the
#    single-color frame mask, then the 5x5 anchor square, then the 3x3 rotation mark.
#    The new behavior is attached to the frame itself: count the frame's colored cells.
#    If that count is odd, the rotation marker acts on the coarse frame layout first,
#    then the unrotated unit sprite is tiled into the rotated layout.
#    If that count is even, first tile the unit sprite into the original frame layout,
#    then rotate the entire enlarged composite.
#    Small trace:
#      original implementation path: rotate frame mask -> tile unrotated unit.
#      new odd path: same as above.
#      new even path: tile unit into original frame -> rotate the full block.
#    The difference matters because rotating the full block also rotates the unit's
#    internal pattern, while rotating only the frame layout does not.
#
# b) Complexity calibration
#    The branch cue has exactly two values (odd/even), which is enough structure to
#    infer but not too much to memorize. To keep the effect visible, generation forces
#    90/270-degree rotations and rejects unit sprites whose 90-degree rotation looks
#    identical.
#
# c) Hidden assumption audit
#    The original main() effectively assumed that "rotate composite" could be
#    implemented as "rotate frame mask before tiling". That shortcut is only harmless
#    when the branch never asks for literal post-tiling rotation. In this variant that
#    assumption is false for even-cardinality frames, so assembly is split into two
#    explicit code paths.
#
# d) Placement legality
#    Both branches end with the same bounding-box dimensions for a given frame shape,
#    because rotating the frame or rotating the final composite swaps height/width in
#    the same way. The generator still computes the actual composite first, then picks
#    anchors and rejects overlap/out-of-bounds placements.
#
# e) Visual observability
#    Counting frame cells is a visible property. The result of the branch is also
#    visible because units are forced to be asymmetric under quarter-turns, and the
#    rotation marker is forced to request a quarter-turn instead of a no-op.
#
# f) Visual cleanliness
#    The legend structure, separator columns, banded canvas, and non-overlap rule are
#    preserved from the original task. No extra markers are added, so the only new
#    reasoning burden is deciding which object receives the rotation first.
#
# g) Color-change inferability
#    No new color mapping is introduced. The original color-swap-on-read behavior is
#    preserved, so the only new variable is operation order, not a hidden color rule.
#
# 9) New main() assumptions and generator guarantees
#    - main() now assumes frame parity is well-defined: generator always creates a
#      single 8-connected frame component.
#    - main() assumes the branch is visually non-degenerate: generator rejects legends
#      where the odd/even branch would accidentally yield the same composite.
#    - main() assumes the requested rotation is a quarter-turn for informative samples:
#      generator only emits right/left clock-hands (k in {1,3}).


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    runs: list[tuple[int, int]] = []
    start = end = cols6[0]
    for c in cols6[1:]:
        if c == end + 1:
            end = c
        else:
            runs.append((start, end))
            start = end = c
    runs.append((start, end))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    h, w = g.shape
    for s, e in reversed(runs[1:]):
        if s >= w - 10 and s > left[1] + 1:
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
        out = spr.copy()
        out[spr == a] = -1
        out[spr == b] = a
        out[out == -1] = b
        spr = out
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


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    inner = legend[border6_r0 + 1 : border6_r0 + 4, border6_c0 + 1 : border6_c0 + 4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
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


def _tile_unit_into_frame(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
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


def _rotate_frame_first(frame_mask0: np.ndarray) -> bool:
    return int(np.count_nonzero(frame_mask0)) % 2 == 1


def _build_composite(unit_sprite: np.ndarray, frame_mask0: np.ndarray, dir_k: int) -> np.ndarray:
    if _rotate_frame_first(frame_mask0):
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        return _tile_unit_into_frame(unit_sprite, frame_mask)
    base = _tile_unit_into_frame(unit_sprite, frame_mask0)
    return np.rot90(base, (-dir_k) % 4)


def _legend_records_from_grid(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    records: list[dict] = []
    for marker_r0, marker_c0 in _all_marker_panel_squares(g):
        if canvas_c0 <= marker_c0 < canvas_c1:
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
        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_composite(unit_sprite, frame_mask0, dir_k)
        alt = (
            np.rot90(_tile_unit_into_frame(unit_sprite, frame_mask0), (-dir_k) % 4)
            if _rotate_frame_first(frame_mask0)
            else _tile_unit_into_frame(unit_sprite, np.rot90(frame_mask0, (-dir_k) % 4))
        )
        records.append(
            {
                "marker": (marker_r0, marker_c0),
                "active_count": int(np.count_nonzero(frame_mask0)),
                "rotate_frame_first": _rotate_frame_first(frame_mask0),
                "dir_k": dir_k,
                "visible_branch_effect": not np.array_equal(comp, alt),
            }
        )
    return records


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged sprites from legend panels and place them onto the canvas, with the rotation stage chosen by a frame-dependent branch.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend region, the output canvas, and sometimes a right legend region.
    - Each legend panel still contains four ingredients from top to bottom: a two-color unit sprite, a single-color frame mask, a 5x5 anchor square, and a 3x3 clock-hand rotation mark.
    - The frame mask's active-cell count is the branch cue: its parity decides which object gets rotated first.

    Rule:
    - Read the unit sprite, frame, anchor marker, and rotation direction from each legend panel outside the canvas.
    - If the frame has an odd number of active cells, rotate the frame layout first and tile the unrotated unit sprite into the rotated frame.
    - If the frame has an even number of active cells, tile the unit sprite into the original frame first and rotate the whole enlarged composite afterward.
    - Use the 3x3 anchor marker to align the final composite on the canvas and paint its non-background pixels.

    Key insight:
    - The clock-hand alone is not enough: you must also decide whether that turn applies to the coarse frame scaffold or to the already-expanded composite, and the deciding signal is the frame's odd/even cell count.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
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
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_composite(unit_sprite, frame_mask0, dir_k)

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


GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 12]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
FRAME_ACTIVE_RANGE = (7, 17)
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
ROTATION_CHOICES = [1, 3]
GEN_RETRIES = 500


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

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        frame_colors.append(random.choice(list(all_set - set(uc))))
    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _make_unit_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    for _ in range(120):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, n * n - 1)
        for p in random.sample(range(n * n), num_c2):
            sprite[p // n, p % n] = c2
        rot = np.rot90(sprite, 1)
        if np.array_equal(sprite, rot):
            continue
        if int(np.count_nonzero(sprite != rot)) < max(3, n):
            continue
        return sprite
    return None


def _make_frame_mask(frame_h: int, frame_w: int, desired_parity: int) -> np.ndarray | None:
    lo, hi = FRAME_ACTIVE_RANGE
    for _ in range(160):
        mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        count = int(np.count_nonzero(mask))
        if count < lo or count > hi:
            continue
        if count % 2 != desired_parity:
            continue
        _, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
        if n8 != 1:
            continue
        return mask
    return None


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    desired_parity: int,
) -> bool:
    n = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_unit_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    avail = marker_r0 - (n + 1)
    combos: list[tuple[int, int, int]] = []
    for f_h in range(*FRAME_H_RANGE):
        if avail < f_h + 1:
            continue
        for f_w in range(*FRAME_W_RANGE):
            for rot_k in ROTATION_CHOICES:
                rh, rw = (f_h, f_w) if rot_k % 2 == 0 else (f_w, f_h)
                if rh * n <= grid_h and rw * n <= canvas_w:
                    combos.append((f_h, f_w, rot_k))
    random.shuffle(combos)

    for frame_h, frame_w, rot_k in combos:
        frame_mask = _make_frame_mask(frame_h, frame_w, desired_parity)
        if frame_mask is None:
            continue
        comp = _build_composite(sprite, frame_mask, rot_k)
        alt = (
            np.rot90(_tile_unit_into_frame(sprite, frame_mask), (-rot_k) % 4)
            if desired_parity == 1
            else _tile_unit_into_frame(sprite, np.rot90(frame_mask, (-rot_k) % 4))
        )
        if np.array_equal(comp, alt):
            continue
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame_mask=frame_mask,
            frame_H=frame_h,
            frame_W=frame_w,
            frame_color=frame_color,
            rot_k=rot_k,
            desired_parity=desired_parity,
            active_count=int(np.count_nonzero(frame_mask)),
            comp=comp,
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    return int(leg["comp"].shape[0]), int(leg["comp"].shape[1])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        top = _anchor_to_top_left(pr, grid_h, ch)
        left = _anchor_to_top_left(pc, canvas_w, cw)
        if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
            return False
        legends[0].update(pr=pr, pc=pc, comp_top=top, comp_left=left)
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
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(
    grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool
) -> None:
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

    desired_parities = [random.randint(0, 1)] if num_legends == 1 else [0, 1]
    random.shuffle(desired_parities)

    for i in range(num_legends):
        ok = _design_legend(
            legends[i],
            unit_color_sets[i],
            frame_colors[i],
            marker_r0,
            grid_h,
            canvas_w,
            desired_parities[i],
        )
        if not ok:
            return None

    if num_legends == 2 and {leg["desired_parity"] for leg in legends} != {0, 1}:
        return None
    if any(not _legend_has_visible_branch(leg) for leg in legends):
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
        return None
    return None


def _legend_has_visible_branch(leg: dict) -> bool:
    sprite = leg["sprite"]
    frame_mask = leg["frame_mask"]
    rot_k = leg["rot_k"]
    chosen = leg["comp"]
    other = (
        np.rot90(_tile_unit_into_frame(sprite, frame_mask), (-rot_k) % 4)
        if leg["desired_parity"] == 1
        else _tile_unit_into_frame(sprite, np.rot90(frame_mask, (-rot_k) % 4))
    )
    return not np.array_equal(chosen, other)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
