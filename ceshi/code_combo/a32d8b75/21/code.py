# Combo variant of a32d8b75 (index 21)
# combo_spec: D3/0+D4/1
# Name: 嵌套框架+位置旋转

from __future__ import annotations

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 原题：读 unit sprite 并换色 -> 读单层 frame -> 按底部时钟把整层 frame 全局旋转 ->
#      在每个活跃格贴同朝向 sprite -> 按 5x5 方框中的 3x3 单像素锚到画布。
#    - D3/0 单独变化后：frame 不是一层，而是上下分离的 outer frame + inner frame。
#      outer 活跃格不直接放 unit，而是先放一个完整 inner-frame composite。
#    - D4/1 单独变化后：某层 frame 的活跃格不再贴同朝向 sprite，而是按该格相对这层 frame
#      中心的方位选择 0/90/180/270 度。
#    - 本融合把两者真正耦合起来：
#      1. 先按时钟把 outer frame 全局旋转。
#      2. 对 rotated outer frame 的每个活跃格，按它相对 outer 中心的方位，决定该格里
#         放置的“子框架”朝向。
#      3. 这个子框架不是静态图块，而是把 inner frame 先旋到该外层方位，再在它的每个活跃格
#         里按相对 inner 中心的方位旋转 unit sprite。
#      4. 因此叶子 sprite 的朝向不只看自己在 inner 里的位置，还要看所属 inner frame 是被
#         哪个 outer 位置召唤出来的。解题者必须递归地跟踪“哪一层的方向在起作用”。
#    - 小例子：若 outer 右侧活跃格对应 outer-direction=right，则该格里放一个右转后的 inner frame。
#      这时 inner 中“上方”的格在全局图像里会出现在右侧，于是其 unit sprite 也随 rotated inner
#      的坐标系一起改变；这与“outer 只决定摆放位置、inner 永远原样贴图”的简单叠加不同。
#
# b) 复杂度校准
#    - 方向只保留 4 档：up/right/down/left。这样人类能从少量样例归纳，不会引入连续角度歧义。
#    - legend 明确画出两层 frame，而不是把它们混成一个纹理，保证递归层次可见。
#    - 为了让“外层决定子框架朝向”可实现且不需要不规则排版，inner frame 固定为 3x3 正方模板；
#      它旋转 90 度后外接框尺寸不变，outer 的每个格都能占用同样大小的子块。
#    - 同时保留 1-2 个 legend、原题的 anchor square、原题的全局时钟，所以变化集中在“嵌套+方向”
#      这一条主线，不会引入过多独立自由度。
#
# c) 隐式假设审查
#    - 原 `_extract_frame_bbox()` 假设 sprite 与 marker 之间只有一个 8 连通 frame。这里失效，
#      因为现在有 outer/inner 两个分离模板；改为 `_extract_nested_frames()` 读取两个 8 连通分量。
#    - 原 composite builder 假设 `frame cell -> unit sprite` 一次展开。这里失效，改成
#      `outer cell -> rotated inner-frame composite -> rotated unit sprites` 两级展开。
#    - 原 D4/1 中“局部方向只在一层 frame 上求一次”的前提也不够了；这里需要先求 outer 方位，
#      再在对应朝向的 inner frame 上重新求 inner 方位，属于递归方向场。
#    - 原放置逻辑只依赖最终 composite 外接矩形，这一点仍然成立；因为 generator 保证最终 bbox
#      完整落入 canvas，双 legend 时也保证 bbox 不重叠。
#
# d) 放置合法性
#    - 继续采用 output-first：先设计 sprite/outer/inner/rot_k，直接构造最终 fused composite，
#      再枚举合法 anchor 位置，最后回填输入 legend。
#    - 双 legend 时按最终 composite bbox 检查互不重叠。
#    - 生成器和 solver 共用 `_build_fused_composite()`，避免“生成时一种递归方向规则、求解时另一种”的漂移。
#
# e) 视觉可观察性
#    - unit sprite 强制四个 90 度旋转都不同，否则局部旋转不可见。
#    - outer 和 inner 两层 frame 都要求覆盖多个方向类别，并同时包含横向与纵向类别，
#      否则“递归方向场”会退化成单向条带。
#    - 生成器拒绝 fused composite 与“外层不旋转子框架”的消融版本完全相同的样本；
#      这保证 outer-level direction 不是摆设。
#
# f) 可视化整洁度
#    - legend 仍保持紧凑纵向布局：sprite -> outer frame -> inner frame -> marker square -> clock-hand。
#    - outer/inner 共用同一种 frame color，但它们被空行分隔，视觉上仍易于辨认。
#    - canvas 继续使用简单条带背景，不引入额外叠层，也不允许 composite 裁剪或互相覆盖。
#
# g) 变化的可推断性
#    - 颜色规则没有额外复杂化，仍只有原题的 unit 两色互换。
#    - 所有新变化都由可见几何结构驱动：outer 中心方位、inner 中心方位、底部全局时钟、5x5 方框锚点。
#      没有依赖颜色编号大小、遍历顺序或其它人类不可见信号。
#
# Post-change solver assumptions audited against the generator
# ------------------------------------------------------------
# - 恰有一个 top-most 4 连通 unit sprite 位于两层 frame 之上。
# - unit 与两个 frame 之间的非零结构只有这三个对象；sprite 下方到 marker 上方恰有两个 8 连通 frame 分量。
# - inner frame 为 3x3 正方模板，因此 outer 不同方向上的子块尺寸一致。
# - outer/inner 模板都足够非对称，fused composite 不会退化成 D3-only 或 flat D4-only 的同像素输出。
# - `generate_input()` 显式构造并验证这些条件；`main()` 因此可以依赖它们。


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
    """Unit sprite is the top-most 4-connected component of pixels not in {0,6}."""
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


def _extract_nested_frames(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], marker_r0: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Read two vertically ordered 8-connected frame components: outer first, inner second."""
    r_start = unit_bb[2] + 1
    r_end = marker_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    labeled, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    if n8 != 2:
        return None

    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, n8 + 1):
        pts = np.argwhere(labeled == idx)
        if pts.size == 0:
            continue
        pts[:, 0] += r_start
        r0 = int(pts[:, 0].min())
        r1 = int(pts[:, 0].max())
        c0 = int(pts[:, 1].min())
        c1 = int(pts[:, 1].max())
        comps.append((r0, c0, r1, c1))
    comps.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))
    outer_bb, inner_bb = comps

    def _crop_mask(bb: tuple[int, int, int, int]) -> np.ndarray:
        r0, c0, r1, c1 = bb
        piece = legend[r0 : r1 + 1, c0 : c1 + 1]
        out = (piece != 0) & (piece != 6)
        _, piece_n8 = _scipy_label(out, structure=np.ones((3, 3), dtype=int))
        assert piece_n8 == 1, "each frame component must be 8-connected"
        return out

    return _crop_mask(outer_bb), _crop_mask(inner_bb)


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
    Read the 3x3 clock-hand marker below the 5x5 anchor square.

    Use the furthest non-{0,6} pixel from the center to determine direction:
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
    """Map a cell position to one of four compass-facing rotations."""
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


def _build_plain_inner_composite(unit_sprite: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
    """Tile the same sprite orientation into each active inner-frame cell."""
    unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    ih, iw = inner_mask.shape
    comp = np.zeros((ih * uh, iw * uw), dtype=int)
    for i in range(ih):
        for j in range(iw):
            if not inner_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][unit_mask] = unit_sprite[unit_mask]
    return comp


def _build_directional_inner_composite(unit_sprite: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
    """
    Tile the unit sprite into each active inner-frame cell, rotating the sprite according
    to that inner cell's direction relative to the current inner-frame center.
    """
    uh, uw = unit_sprite.shape
    ih, iw = inner_mask.shape
    comp = np.zeros((ih * uh, iw * uw), dtype=int)
    rot_sprites = [np.rot90(unit_sprite, (-k) % 4) for k in range(4)]
    rot_masks = [(spr != 0) & (spr != 6) for spr in rot_sprites]
    for i in range(ih):
        for j in range(iw):
            if not inner_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            local_k = _cell_rotation_from_position(i, j, ih, iw)
            tile = rot_sprites[local_k]
            tile_mask = rot_masks[local_k]
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][tile_mask] = tile[tile_mask]
    return comp


def _build_outer_neutral_composite(
    unit_sprite: np.ndarray, outer_mask: np.ndarray, inner_mask: np.ndarray, dir_k: int
) -> np.ndarray:
    """
    Ablation helper: outer direction does not rotate child frames.

    This is used only by the generator to reject degenerate samples where the fusion's
    outer-level direction field would have no visible effect.
    """
    rot_outer = np.rot90(outer_mask, (-dir_k) % 4)
    inner_comp = _build_directional_inner_composite(unit_sprite, inner_mask)
    block_mask = inner_comp != 0
    bh, bw = inner_comp.shape
    oh, ow = rot_outer.shape
    comp = np.zeros((oh * bh, ow * bw), dtype=int)
    for i in range(oh):
        for j in range(ow):
            if not rot_outer[i, j]:
                continue
            rr0 = i * bh
            cc0 = j * bw
            comp[rr0 : rr0 + bh, cc0 : cc0 + bw][block_mask] = inner_comp[block_mask]
    return comp


def _build_nested_plain_composite(
    unit_sprite: np.ndarray, outer_mask: np.ndarray, inner_mask: np.ndarray, dir_k: int
) -> np.ndarray:
    """Ablation helper matching a D3-only nested construction with no position-dependent rotations."""
    rot_outer = np.rot90(outer_mask, (-dir_k) % 4)
    inner_comp = _build_plain_inner_composite(unit_sprite, inner_mask)
    block_mask = inner_comp != 0
    bh, bw = inner_comp.shape
    oh, ow = rot_outer.shape
    comp = np.zeros((oh * bh, ow * bw), dtype=int)
    for i in range(oh):
        for j in range(ow):
            if not rot_outer[i, j]:
                continue
            rr0 = i * bh
            cc0 = j * bw
            comp[rr0 : rr0 + bh, cc0 : cc0 + bw][block_mask] = inner_comp[block_mask]
    return comp


def _build_fused_composite(
    unit_sprite: np.ndarray, outer_mask: np.ndarray, inner_mask: np.ndarray, dir_k: int
) -> np.ndarray:
    """
    Build the fused recursive composite.

    Steps:
    - Rotate the outer frame by the legend clock-hand.
    - Each active outer-frame cell chooses a direction around the outer center.
    - That direction rotates the inner frame before the inner level is interpreted.
    - Inside that rotated inner frame, each active inner cell rotates the unit sprite by
      its own direction around the rotated inner center.
    """
    rot_outer = np.rot90(outer_mask, (-dir_k) % 4)
    outer_h, outer_w = rot_outer.shape

    block_cache = {
        k: _build_directional_inner_composite(unit_sprite, np.rot90(inner_mask, (-k) % 4))
        for k in range(4)
    }
    block_h, block_w = next(iter(block_cache.values())).shape
    comp = np.zeros((outer_h * block_h, outer_w * block_w), dtype=int)

    for i in range(outer_h):
        for j in range(outer_w):
            if not rot_outer[i, j]:
                continue
            outer_k = _cell_rotation_from_position(i, j, outer_h, outer_w)
            block = block_cache[outer_k]
            block_mask = block != 0
            rr0 = i * block_h
            cc0 = j * block_w
            comp[rr0 : rr0 + block_h, cc0 : cc0 + block_w][block_mask] = block[block_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode recursively directed nested frame legends and place the resulting composites onto the canvas.

    Input structure:
    - Full-height color-6 separator columns isolate a central canvas from one or two 5-column legend panels.
    - Each legend panel contains, from top to bottom, a swapped two-color unit sprite, an outer frame mask, an inner frame mask, a 5x5 anchor square, and a 3x3 clock-hand rotation mark.
    - The two frame masks are separate 8-connected components stacked vertically, so the legend explicitly names the outer template and the reusable inner sub-template.

    Rule:
    - Read the unit sprite and undo its legend-side color swap. Rotate the outer frame layout by the clock-hand direction, then inspect each active outer-frame cell's direction around the outer-frame center.
    - For each active outer cell, place a copy of the inner frame that is rotated to that outer direction; inside that rotated inner frame, rotate each unit sprite again according to the inner cell's direction around the inner-frame center.
    - Place the finished recursive composite on the canvas using the start/middle/end anchor encoded by the marker square, and overwrite the canvas background with the composite pixels.

    Key insight:
    - Direction is no longer a one-layer property. The outer frame first chooses how each subframe is oriented, and only inside that chosen subframe do inner positions decide the final unit-sprite rotations, so the solver must follow nested coordinate systems rather than a single flat radial field.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
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

        frames = _extract_nested_frames(panel, unit_bb, marker_r0)
        if frames is None:
            continue
        outer_mask0, inner_mask0 = frames

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        comp = _build_fused_composite(unit_sprite, outer_mask0, inner_mask0, dir_k)

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


# --------------- generator (nested_frame_recursive_direction) ---------------

GRID_H_RANGE = (22, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [3, 2]  # weights for [1-legend, 2-legend]
LEGEND_PANEL_W = 5
UNIT_SIZE_RANGE = (2, 3)
OUTER_FRAME_H_RANGE = (2, 4)
OUTER_FRAME_W_RANGE = (2, 3)
INNER_FRAME_SIZE = 3
MARKER_PANEL_HEIGHT = 8
BAND_WIDTH_MIN = 4
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 600
SPRITE_RETRIES = 200
FRAME_RETRIES = 300


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

    all_set = set(USABLE_COLORS)
    frame_colors = []
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


def _all_rotations_distinct(arr: np.ndarray) -> bool:
    return len({np.rot90(arr, k).tobytes() for k in range(4)}) == 4


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


def _generate_directional_mask(h: int, w: int, min_active: int, max_active: int) -> np.ndarray:
    """Generate an 8-connected non-full mask whose active cells span several direction classes."""
    max_active = min(max_active, h * w - 1)
    for _ in range(FRAME_RETRIES):
        mask = random_sprite(
            h,
            w,
            density=0.6,
            symmetry="not_symmetric",
            color_palette=[1],
            connectivity=8,
        ) != 0
        active = int(mask.sum())
        if active < min_active or active > max_active:
            continue
        if active == h * w:
            continue
        if not _frame_has_visible_directionality(mask):
            continue
        if h == w and not _all_rotations_distinct(mask.astype(np.uint8)):
            continue
        return mask
    raise RuntimeError("failed to generate a suitable directional frame mask")


def _comp_size(leg: dict) -> tuple[int, int]:
    return int(leg["comp"].shape[0]), int(leg["comp"].shape[1])


def _design_legend(
    leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int
) -> bool:
    """Fill one legend with a sprite plus two stacked frame masks under the fused rule."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = _make_asymmetric_sprite(n, uc1, uc2)
    if sprite is None:
        return False
    legend_sprite = _color_swap(sprite, uc1, uc2)

    inner_h = inner_w = INNER_FRAME_SIZE
    combos: list[tuple[int, int, int]] = []
    for outer_h in range(OUTER_FRAME_H_RANGE[0], OUTER_FRAME_H_RANGE[1] + 1):
        for outer_w in range(OUTER_FRAME_W_RANGE[0], OUTER_FRAME_W_RANGE[1] + 1):
            min_rows_used = 1 + n + 1 + outer_h + 1 + inner_h
            if min_rows_used > marker_r0:
                continue
            for rot_k in range(4):
                rot_outer_h, rot_outer_w = (outer_h, outer_w) if rot_k % 2 == 0 else (outer_w, outer_h)
                comp_h = rot_outer_h * inner_h * n
                comp_w = rot_outer_w * inner_w * n
                if comp_h <= grid_h and comp_w <= canvas_w:
                    combos.append((outer_h, outer_w, rot_k))
    random.shuffle(combos)
    if not combos:
        return False

    for outer_h, outer_w, rot_k in combos:
        try:
            outer_max_active = min(6, outer_h * outer_w - 1)
            outer_mask = _generate_directional_mask(outer_h, outer_w, min_active=3, max_active=outer_max_active)
            inner_mask = _generate_directional_mask(inner_h, inner_w, min_active=4, max_active=6)
        except RuntimeError:
            continue

        comp = _build_fused_composite(sprite, outer_mask, inner_mask, rot_k)
        if comp.shape[0] > grid_h or comp.shape[1] > canvas_w:
            continue

        if np.array_equal(comp, _build_outer_neutral_composite(sprite, outer_mask, inner_mask, rot_k)):
            continue
        if np.array_equal(comp, _build_nested_plain_composite(sprite, outer_mask, inner_mask, rot_k)):
            continue

        spare = marker_r0 - (1 + n + 1 + outer_h + 1 + inner_h)
        extra1 = random.randint(0, min(2, spare))
        spare -= extra1
        extra2 = random.randint(0, min(2, spare))
        gap1 = 1 + extra1
        gap2 = 1 + extra2

        outer_r0 = 1 + n + gap1
        inner_r0 = outer_r0 + outer_h + gap2
        sprite_c0 = random.randint(0, LEGEND_PANEL_W - n)
        outer_c0 = random.randint(0, LEGEND_PANEL_W - outer_w)
        inner_c0 = random.randint(0, LEGEND_PANEL_W - inner_w)

        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            sprite_c0=sprite_c0,
            frame_color=frame_color,
            outer_mask=outer_mask,
            inner_mask=inner_mask,
            outer_r0=outer_r0,
            outer_c0=outer_c0,
            inner_r0=inner_r0,
            inner_c0=inner_c0,
            rot_k=rot_k,
            comp=comp,
        )
        return True
    return False


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions, avoiding overlap for the 2-legend case."""
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
    """Stamp each legend's fused recursive composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = leg["comp"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int) -> None:
    """Draw one 5-column legend panel plus its separator column."""
    n = leg["N"]
    grid[:, sep_col] = 6

    panel = grid[:, col_start : col_start + LEGEND_PANEL_W]
    panel[:] = 0

    panel[marker_r0 : marker_r0 + 5, :] = 0
    panel[marker_r0, :] = 6
    panel[marker_r0 + 4, :] = 6
    panel[marker_r0 : marker_r0 + 5, 0] = 6
    panel[marker_r0 : marker_r0 + 5, 4] = 6
    panel[marker_r0 + 1 + leg["pr"], 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    panel[rr0 : rr0 + 3, :] = 0
    panel[rr0 : rr0 + 3, 0] = 6
    panel[rr0 : rr0 + 3, 4] = 6
    cx_r, cx_c = rr0 + 1, 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    panel[cx_r, cx_c] = 7
    panel[cx_r + dr, cx_c + dc] = 7

    panel[1 : 1 + n, leg["sprite_c0"] : leg["sprite_c0"] + n] = leg["legend_sprite"]

    outer_region = panel[
        leg["outer_r0"] : leg["outer_r0"] + leg["outer_mask"].shape[0],
        leg["outer_c0"] : leg["outer_c0"] + leg["outer_mask"].shape[1],
    ]
    outer_region[leg["outer_mask"]] = leg["frame_color"]

    inner_region = panel[
        leg["inner_r0"] : leg["inner_r0"] + leg["inner_mask"].shape[0],
        leg["inner_c0"] : leg["inner_c0"] + leg["inner_mask"].shape[1],
    ]
    inner_region[leg["inner_mask"]] = leg["frame_color"]


def _choose_grid_h(num_legends: int) -> int:
    lo, hi = GRID_H_RANGE
    if num_legends == 2:
        lo = max(lo, hi - 4)
    return random.randint(lo, hi)


def _choose_legend_sizes(num_legends: int) -> list[int]:
    if num_legends == 2:
        return [UNIT_SIZE_RANGE[0] for _ in range(num_legends)]
    return [random.randint(*UNIT_SIZE_RANGE) for _ in range(num_legends)]


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    if num_legends is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    grid_h = _choose_grid_h(num_legends)
    legends = [{"N": n} for n in _choose_legend_sizes(num_legends)]

    left_total = LEGEND_PANEL_W + 1
    right_total = (1 + LEGEND_PANEL_W) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w <= 0:
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

    _place_legend(grid, legends[0], col_start=0, sep_col=LEGEND_PANEL_W, marker_r0=marker_r0)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    branch_retry_budget = max(1, GEN_RETRIES // 3)
    target_num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    for _ in range(branch_retry_budget):
        result = _gen_attempt(target_num_legends)
        if result is not None:
            return result

    for _ in range(GEN_RETRIES - branch_retry_budget):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
