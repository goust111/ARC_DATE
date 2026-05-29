# Variant of a32d8b75
# Dimension: D4 — 变换：分形铺贴构建 (transform)
# Change: 将颜色交换后的单元精灵铺贴到框架每个活跃像素位置，构建(fh*uh)x(fw*uw)的复合精灵... → 铺贴时精灵根据框架位置旋转
#
# Design analysis:
# spec:
#   variant_core:
#     summary: >
#       保留原题的 legend/canvas/anchor/global-rotation 协议，只改“框架活跃像素如何铺贴单元精灵”。
#       旧规则是在每个活跃框架格放入同一个朝向的颜色交换后单元精灵；新规则是在每个活跃框架格，
#       根据该格相对框架中心的方向，选取单元精灵的一个 90 度档位旋转后再铺贴。
#     local_direction_rule:
#       center_reference: 几何中心 ((fh-1)/2, (fw-1)/2)，允许偶数尺寸产生半格中心
#       mapping:
#         up:    dr < 0 and abs(dr) >= abs(dc) -> 0 度
#         right: dc > 0 and abs(dc) >  abs(dr) -> 90 度顺时针
#         down:  dr > 0 and abs(dr) >= abs(dc) -> 180 度
#         left:  dc < 0 and abs(dc) >  abs(dr) -> 270 度顺时针
#         exact_center: 退化为 0 度；生成器不依赖该特例来体现规则
#   concrete_behavior_trace:
#     original:
#       - 从 legend 读出 unit sprite，并做颜色对换。
#       - 按 clock-hand 先把 frame mask 整体旋转。
#       - 对 frame 的每个活跃格，复制同一个朝向的 sprite。
#       - 再按 marker square 的 3x3 锚点把 composite 放到 canvas。
#     variant:
#       - 前两步保持不变。
#       - 但在 frame 的每个活跃格，不再直接复制同一 sprite，而是先看该格位于 frame 中心的上/右/下/左哪一侧。
#       - 例如一个位于上半部的框架格放置 0 度 sprite，右半部放置顺时针 90 度 sprite。
#       - 因为 frame 仍然先被 clock-hand 整体旋转，局部朝向场也会随之转动，所以最后形成“辐射状”的 composite。
#   complexity_calibration:
#     - 方向只使用 4 个离散档位，避免引入难以从少量样例归纳的连续角度。
#     - 方向判定只依赖相对几何位置，不依赖颜色 id、遍历顺序或不可见元信息。
#     - 为了让变体肉眼可见，生成器强制 unit sprite 的四个 90 度旋转互不相同，并要求 frame 至少激活 3 个方向扇区。
#   implicit_assumption_audit:
#     - 原 main 中 `frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)` 依赖“全局时钟标记只影响整体布局”。
#       该前提在变体中仍成立，因此保留。
#     - 原 main 中构造 composite 的双循环默认“所有活跃格使用同一 sprite 朝向”。
#       该前提已失效，必须改为每格独立求方向并选择相应旋转。
#     - 原 main 的放置部分只依赖 composite 的外接矩形尺寸。
#       因为 unit sprite 仍是 NxN 正方形，局部旋转不会改变 tile 尺寸，因此锚点和防越界逻辑仍成立。
#     - 原 generator 只需保证 composite bbox 不重叠；变体下局部旋转不会改变 bbox，所以该合法性检查继续有效。
#   placement_legality:
#     - 仍采用 output-first：先设计 legends，计算 composite 尺寸，选择锚点，确保不越界且多图例不重叠，再回填 input。
#     - generator 和 solver 共用同一 `_build_radial_composite`，避免“生成时一种规则、求解时另一种规则”的漂移。
#   observability_and_cleanliness:
#     - unit sprite 强制非 90 度/180 度对称，避免旋转不可见。
#     - frame 必须在上下/左右方向上都出现活跃格，并至少覆盖 3 个方向类别，避免只出现一条竖带导致近似原题。
#     - 不新增任何新标记；视觉负担仍与原题相同，只是输出中的 tiled object 从整齐同向改为放射状。

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


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


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
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


def _build_radial_composite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """
    Tile the unit sprite into each active frame cell, rotating the sprite according to
    the cell's direction relative to the frame center.
    """
    m_unit = (unit_sprite != 0) & (unit_sprite != 6)
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
            rk = _cell_rotation_from_position(i, j, fh, fw)
            tile = rot_sprites[rk]
            tile_mask = rot_masks[rk]
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][tile_mask] = tile[tile_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas using legend-panel instructions.

    Input structure:
    - Color-6 separator columns split the grid into a legend region and a canvas region, with an optional second legend on the right.
    - Each legend panel stores, from top to bottom: a two-color unit sprite, a frame mask, a bordered 5x5 anchor square, and a 3x3 clock-hand rotation mark.

    Rule:
    - Read the unit sprite, undo its legend-side color swap, and rotate the frame layout according to the clock-hand direction.
    - For every active frame cell, stamp a copy of the unit sprite whose orientation depends on that cell's direction inside the frame (up/right/down/left relative to the frame center).
    - Place the resulting composite on the canvas so its start/middle/end anchor matches the marker pixel inside the 5x5 square.

    Key insight: The 5x5 square still chooses where the composite is anchored, but the internal appearance now comes from a second coordinate system: each frame cell's position around the frame center selects the local sprite rotation.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    left_sep_c1 = left_sep[1]
    canvas_c0 = left_sep_c1 + 1
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

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_radial_composite(unit_sprite, frame_mask)

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


# --------------- generator (radial_tiling_from_frame_direction) ---------------

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
SPRITE_RETRIES = 200
FRAME_RETRIES = 200


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


def _all_rotations_distinct(sprite: np.ndarray) -> bool:
    sigs = {np.rot90(sprite, k).tobytes() for k in range(4)}
    return len(sigs) == 4


def _make_asymmetric_sprite(n: int, c1: int, c2: int) -> np.ndarray | None:
    """Generate a fully filled NxN 2-color sprite with four distinct quarter-turn rotations."""
    area = n * n
    for _ in range(SPRITE_RETRIES):
        sprite = np.full((n, n), c1, dtype=int)
        num_c2 = random.randint(1, area - 1)
        for p in random.sample(range(area), num_c2):
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


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = _make_asymmetric_sprite(n, uc1, uc2)
    if sprite is None:
        return False
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
    random.shuffle(combos)

    for frame_H, frame_W, rot_k in combos:
        for _ in range(FRAME_RETRIES):
            frame_mask = random_sprite(
                frame_H,
                frame_W,
                density=FRAME_DENSITY,
                connectivity=8,
                color_palette=[1],
                symmetry="not_symmetric",
            ) != 0
            rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
            if not _frame_has_visible_directionality(rot_frame):
                continue
            leg.update(
                sprite=sprite,
                legend_sprite=legend_sprite,
                frame_mask=frame_mask,
                frame_H=frame_H,
                frame_W=frame_W,
                frame_color=frame_color,
                rot_k=rot_k,
                rotated_frame=rot_frame,
            )
            return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the composite after frame rotation and local sprite rotations."""
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
    """Stamp each legend's rotated-by-position composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        t, l = leg["comp_top"], leg["comp_left"]
        comp = _build_radial_composite(sp, rf)

        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
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
        if not _design_legend(
            legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w
        ):
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
        _place_legend(
            grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True
        )

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


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

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
