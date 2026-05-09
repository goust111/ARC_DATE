# Combo variant of a32d8b75 (index 10)
# combo_spec: D6/3+D18/0
# Name: 双标记定位+动态面板位置

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite


# ---------------------------------------------------------------------------
# Design analysis for combo D6/3 + D18/0
#
# a) 具体化与行为追踪:
#    - 保留原题的大框架: full-height 6 分隔列把整张图切成 legend 区和 canvas 区；
#      legend 里仍然依次包含 unit sprite、frame、位置标记、旋转标记。
#    - D18/0 改动先落地: marker panel 不再固定在底部 8 行，而是改成
#      “marker_square 的顶行 = frame_bottom + 1”。所以 solver 不再能先扫固定底部，
#      必须先识别顶端 sprite，再识别其下方的 frame，最后去 frame 正下方找 panel。
#    - D6/3 改动再叠加: 5x5 方框内部不再只有 1 个像素，而是一个十字骨架上的双 1D 标记。
#      纵向臂(top/both/bottom) 编码 row anchor，横向臂(left/both/right) 编码 col anchor。
#    - 例子: 若目标锚点是 (row=2, col=0)，原题会在 3x3 内部的 (2,0) 放单个 4；
#      融合后改为 vertical=center+bottom, horizontal=left+center，形成一个左下 L 形。
#      同时这整个 marker panel 还不在底部，而是贴着 frame 下缘。
#    - 因此原逻辑在两个点都失效:
#      1) “全局扫描固定底部 marker” 失效；
#      2) “读取第一个非零内部像素就是二维锚点” 也失效。
#
# b) 复杂度校准:
#    - 位置编码仍然只落在 3x3 的九种 anchor 组合上，没有扩大决策空间；
#      但视觉上从单像素变成 3-5 像素的 cross marker，差异足够明显。
#    - marker panel 仍然紧贴 frame 下方，而不是漂浮到任意高度，避免布局杂乱。
#    - frame 活跃像素数强制 >= 6，大于 cross marker 的最大 5 像素，保证 solver 可把
#      “sprite 下方最大 8 连通块”稳定识别为 frame，不会和 marker 混淆。
#
# c) 隐式假设审查:
#    - 原 `_all_marker_panel_squares(g)` 依赖 marker 可被全局独立扫描；融合后删除该假设。
#    - 原 `_extract_frame_bbox(..., border6_r0)` 依赖先知道 marker 行；融合后改为仅凭
#      “sprite 下方的最大 8 连通块”提取 frame。
#    - 原 `_position_indices_from_border_square` 依赖首个非零像素就是 (pr,pc)；融合后改为
#      分别读取 vertical / horizontal 两个 1D cue，再合成为 2D 坐标。
#    - 新 solver 的额外假设是:
#      1) frame 位于 sprite 下方；
#      2) marker panel 恰好位于 frame 下方一行；
#      3) cross marker 只使用十字骨架像素且中心必亮。
#      generator 会显式构造并验证这些条件。
#
# d) 放置合法性:
#    - 继续采用 output-first: 先选 anchor 与旋转后的 composite 尺寸，确保完整落在 canvas 内。
#    - 双 legend 时仍显式排除 composite bbox 重叠。
#    - 纵向新增约束: marker_r0 = frame_top + frame_h，且 marker_r0 + 8 <= grid_h - 1，
#      这样 panel 完整落入网格并且不退化回原题的贴底位置。
#
# e) 视觉可观察性:
#    - 观察 2-4 组样例时，人能直接看到两件事:
#      1) marker panel 跟着 frame 走，不贴底；
#      2) 锚点不是单点，而是横条 + 竖条共同决定。
#    - 规则只依赖可见几何关系，不依赖颜色编号大小或不可见排序。
#
# f) 可视化整洁度:
#    - sprite、frame、marker panel 仍然沿同一 legend 竖向排布；
#      panel 虽然动态，但始终贴 frame，下方留至少 1 行空白，避免拥挤到底边。
#    - cross marker 被限制在 3x3 的十字骨架中，信息密度增加但不会遮挡方框边界。
#
# g) 变化的可推断性:
#    - 颜色逻辑保持原题: unit 读取时交换双色，frame 为单色，位置标记用 4，旋转标记用 7。
#    - 没有新增颜色映射；变化由形状与相对位置承载，因此少量样例即可归纳。
# ---------------------------------------------------------------------------


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    h, w = g.shape
    cols6 = [c for c in range(w) if np.all(g[:, c] == 6)]
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


def _extract_unit_sprite(
    legend: np.ndarray,
) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray] | None:
    """Return the top-most 4-connected non-{0,6} component, plus bbox and mask."""
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
    return spr, (r0, c0, r1, c1), vis


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], unit_mask: np.ndarray
) -> tuple[int, int, int, int] | None:
    """
    Find the frame as the largest 8-connected component below the unit sprite.

    This supports the dynamic marker placement: the marker panel can only be located
    after the frame has been identified.
    """
    mask = (legend != 0) & (legend != 6)
    mask[unit_mask] = False
    mask[: unit_bb[2] + 1, :] = False
    lbl, n8 = _scipy_label(mask, structure=np.ones((3, 3), dtype=int))
    if n8 == 0:
        return None
    candidates: list[tuple[int, int, int, int, int]] = []
    for idx in range(1, n8 + 1):
        pts = np.argwhere(lbl == idx)
        if pts.size == 0:
            continue
        size = int(len(pts))
        if size < 6:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        candidates.append((size, r0, c0, r1, c1))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-t[0], t[1], t[2]))
    _, r0, c0, r1, c1 = candidates[0]
    return r0, c0, r1, c1


def _find_marker_panel_below_frame(
    legend: np.ndarray, frame_bb: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    """Find the 5x5 marker square whose top edge starts exactly one row below the frame."""
    h, w = legend.shape
    r0 = frame_bb[2] + 1
    if r0 + 7 >= h:
        return None
    for c0 in range(w - 4):
        sq = legend[r0 : r0 + 5, c0 : c0 + 5]
        if not (
            np.all(sq[0, :] == 6)
            and np.all(sq[4, :] == 6)
            and np.all(sq[:, 0] == 6)
            and np.all(sq[:, 4] == 6)
        ):
            continue
        inner = sq[1:4, 1:4]
        if not np.any((inner != 0) & (inner != 6)):
            continue
        rot_block = legend[r0 + 5 : r0 + 8, c0 : c0 + 5]
        if rot_block.shape != (3, 5):
            continue
        if not (np.all(rot_block[:, 0] == 6) and np.all(rot_block[:, 4] == 6)):
            continue
        rot = rot_block[:, 1:4]
        if not np.any((rot != 0) & (rot != 6)):
            continue
        return int(r0), int(c0)
    return None


def _decode_axis_from_cross(neg_side: bool, pos_side: bool, axis_name: str) -> int:
    if neg_side and pos_side:
        return 1
    if neg_side:
        return 0
    if pos_side:
        return 2
    raise AssertionError(f"{axis_name} arm missing from crossed marker")


def _position_indices_from_border_square(
    legend: np.ndarray, border6_r0: int, border6_c0: int
) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    mask = (inner != 0) & (inner != 6)
    if not mask[1, 1]:
        return None
    allowed = np.zeros((3, 3), dtype=bool)
    allowed[0, 1] = True
    allowed[1, 0] = True
    allowed[1, 1] = True
    allowed[1, 2] = True
    allowed[2, 1] = True
    if np.any(mask & ~allowed):
        raise AssertionError("marker contains pixels outside the cross skeleton")
    pr = _decode_axis_from_cross(bool(mask[0, 1]), bool(mask[2, 1]), "row")
    pc = _decode_axis_from_cross(bool(mask[1, 0]), bool(mask[1, 2]), "col")
    return pr, pc


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """
    Read the 3x3 clock-hand under the marker square:
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
    sprite_mask = (unit_sprite != 0) & (unit_sprite != 6)
    uh, uw = unit_sprite.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            r0 = i * uh
            c0 = j * uw
            comp[r0 : r0 + uh, c0 : c0 + uw][sprite_mask] = unit_sprite[sprite_mask]
    return comp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged sprites from legends with floating marker panels and split anchor codes.

    Input structure:
    - Full-height color-6 separator columns still divide the grid into one or two legend regions plus a central canvas.
    - Within each legend, the unit sprite stays near the top and the frame stays below it, but the marker panel is not fixed to the bottom; its 5x5 square begins immediately below that legend's frame.
    - The marker square's 3x3 interior uses a crossed code: the vertical arm encodes the row anchor and the horizontal arm encodes the column anchor, while the 3x3 clock-hand underneath still encodes rotation.

    Rule:
    - Extract the top sprite, swap its two colors, and identify the frame as the main 8-connected component below it.
    - Step to the row directly beneath the frame to find the marker panel, decode the vertical and horizontal 1D markers separately, rotate the tiled composite by the clock-hand direction, and place it using the combined 2D anchor.

    Key insight:
    - Solving now requires two linked inferences: the marker panel's row is determined by the frame rather than a fixed bottom slot, and the anchor coordinate itself is recovered by combining two independent one-dimensional cues inside the marker square.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    legend_strips = [g[:, : left_sep[1] + 1]]
    if right_sep is not None:
        legend_strips.append(g[:, right_sep[0] :])

    for legend in legend_strips:
        unit = _extract_unit_sprite(legend)
        if unit is None:
            continue
        unit_sprite, unit_bb, unit_mask = unit

        frame_bb = _extract_frame_bbox(legend, unit_bb, unit_mask)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = legend[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3), dtype=int))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        marker = _find_marker_panel_below_frame(legend, frame_bb)
        assert marker is not None, "marker panel not found below frame"
        marker_r0, marker_c0 = marker

        dir_k = _rotation_from_clockhand(legend, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _build_composite(unit_sprite, frame_mask)

        pos = _position_indices_from_border_square(legend, marker_r0, marker_c0)
        assert pos is not None, "position marker missing"
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


# --------------- generator (cross_marker_dynamic_panel) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 34
# Bias toward 2 legends so accepted samples cover both legend-count branches
# more evenly after geometric feasibility filtering.
LEGEND_COUNT_WEIGHTS = [1, 10]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
FRAME_GAP_RANGE = (1, 6)
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
BOTTOM_PAD_MIN = 1
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 400


def _allocate_colors(num_legends):
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
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _design_legend(leg, unit_colors, frame_color, grid_h, canvas_w):
    """Fill in sprite, frame, marker row, and rotation for one legend dict."""
    n = leg["N"]
    uc1, uc2 = unit_colors

    sprite = np.full((n, n), uc1, dtype=int)
    num_c2 = random.randint(1, n * n - 1)
    for p in random.sample(range(n * n), num_c2):
        sprite[p // n, p % n] = uc2
    legend_sprite = _color_swap(sprite, uc1, uc2)

    combos = []
    for frame_gap in range(FRAME_GAP_RANGE[0], FRAME_GAP_RANGE[1] + 1):
        frame_top = n + 1 + frame_gap
        for frame_h in range(*FRAME_H_RANGE):
            marker_r0 = frame_top + frame_h
            tail_pad = grid_h - (marker_r0 + MARKER_PANEL_HEIGHT)
            if tail_pad < BOTTOM_PAD_MIN:
                continue
            for frame_w in range(*FRAME_W_RANGE):
                for rot_k in range(4):
                    rh, rw = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                    if rh * n <= grid_h and rw * n <= canvas_w:
                        combos.append((frame_gap, frame_top, frame_h, frame_w, rot_k, marker_r0))
    if not combos:
        return False

    frame_gap, frame_top, frame_h, frame_w, rot_k, marker_r0 = random.choice(combos)

    frame_mask = None
    for _ in range(50):
        candidate = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        _, n8 = _scipy_label(candidate, structure=np.ones((3, 3), dtype=int))
        if int(candidate.sum()) >= 6 and n8 == 1:
            frame_mask = candidate
            break
    if frame_mask is None:
        return False

    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)
    leg.update(
        sprite=sprite,
        legend_sprite=legend_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        frame_gap=frame_gap,
        frame_top=frame_top,
        marker_r0=marker_r0,
    )
    return True


def _comp_size(leg):
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends, grid_h, canvas_w):
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


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
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
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg["sprite"], leg["rotated_frame"])
        t, l = leg["comp_top"], leg["comp_left"]
        ch, cw = comp.shape
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _encode_cross_marker(pr: int, pc: int) -> np.ndarray:
    inner = np.zeros((3, 3), dtype=int)
    inner[1, 1] = 4
    if pr <= 0:
        inner[0, 1] = 4
    elif pr >= 2:
        inner[2, 1] = 4
    else:
        inner[0, 1] = 4
        inner[2, 1] = 4

    if pc <= 0:
        inner[1, 0] = 4
    elif pc >= 2:
        inner[1, 2] = 4
    else:
        inner[1, 0] = 4
        inner[1, 2] = 4
    return inner


def _place_legend(grid, leg, col_start, sep_col, is_right):
    """Draw one legend panel with a cross-marker block immediately below the frame."""
    n = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, n - 4)) if not is_right else (sep_col + max(0, n - 3))
    marker_r0 = leg["marker_r0"]

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 : marker_r0 + 4, mc0 + 1 : mc0 + 4] = _encode_cross_marker(
        leg["pr"], leg["pc"]
    )

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

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    frame_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[
        leg["frame_top"] : leg["frame_top"] + frame_h,
        mc0 + frame_col : mc0 + frame_col + frame_w,
    ]
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

    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, is_right=True)

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
