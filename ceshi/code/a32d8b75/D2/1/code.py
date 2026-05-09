# Variant of a32d8b75
# Dimension: D2 — 源对象：单元精灵的结构 (src)
# Change: NxN（N=2-5）全填充精灵，恰好2种非{0,6}颜色，读取时进行颜色交换（a<->b）... → 单元精灵使用3种颜色

"""
design_analysis:
  source_task: none
  concretization:
    - 输入仍然由左图例、中央画布、可选右图例组成，图例内部仍然是"单元精灵 -> 框架 -> 5x5锚点方框 -> 3x3旋转标记"四段。
    - 关键变化只在单元精灵：它现在是 NxN 全填充且恰好 3 种非 {0,6} 颜色。solver 读取这个精灵后，不再做双色互换，而是做三色循环置换。
    - 三色循环的顺序不能靠颜色 id 排序，否则属于不可见规则；这里改成按精灵里行优先首次出现的顺序定义 a,b,c，再执行 a->b, b->c, c->a。
    - 为了让这个顺序肉眼可见，generator 固定让三个颜色的首次出现位置分别落在 (0,0)、(0,1)、(1,0)。这样解题者能直接从图例左上角看出三色顺序。
  behavior_trace:
    old_logic:
      - 例：图例单元是 [[2,9],[9,2]]，solver 按两色互换读成 [[9,2],[2,9]]，再把这个读出的单元铺到框架每个激活格。
    new_logic:
      - 例：图例单元是 [[2,8],[5,2]]。按行优先首次出现顺序得到 (2,8,5)，循环后读成 [[8,5],[2,8]]。
      - 后续流程不变：先按框架放大铺贴，再按旋转标记旋转，再按 3x3 位置标记确定锚点并放到画布。
    difference:
      - 差异集中在"读单元精灵"这一步；原任务只有一对颜色可交换，现在必须识别三个颜色的有向循环关系。
  complexity_calibration:
    - 仍保留 1-2 个 legend、相同的框架与锚点机制，避免引入额外维度。
    - 每个单元恰好 3 色，颜色数没有膨胀到难以归纳，但又明显区别于原题的双色交换。
    - 三色顺序由固定的可见首次出现位置给出，减少歧义；同时框架中会重复铺贴这个单元，给出足够多的对应像素支撑颜色映射。
  implicit_assumptions_audit_main:
    - _extract_unit_sprite 原先默认 `len(cols)==2` 时排序后交换；这依赖了"颜色集合大小固定且排序可用"。变体中必须改成读取 3 色，并按可见出现顺序循环。
    - main 中后续 `m_unit = (unit_sprite != 0) & (unit_sprite != 6)`、铺贴、旋转、锚点放置都只依赖"单元非空且全填充"，与颜色数无关，可保持不变。
    - `_extract_frame_bbox` 依赖 frame 与 unit 之间至少有一行空白，否则会把它们混在同一个纵向区段里。generator 继续保证这点。
    - `_all_marker_panel_squares` 与 `_rotation_from_clockhand` 依赖 marker square 下方紧跟 3x3 旋转区块；generator 保持原布局。
  placement_legality:
    - generator 仍然先决定输出端的复合图尺寸，再筛选能完整放入画布的 frame 尺寸、旋转和锚点。
    - 双 legend 情况继续枚举 3x3 锚点组合并排除包围盒重叠，所以不会出现输出复合图交叠。
    - 输出覆盖时依然跳过画布中的 6 色隔离线，保证 solver 的覆盖断言成立。
  visual_observability:
    - 单元精灵的三色首次出现位置固定在左上角三格，可直接看出 a,b,c 的顺序。
    - 输出中的每个放大 tile 都保留同样的内部结构，只是颜色整体前移一位，因此从 2-3 组样例可肉眼归纳出循环关系。
    - 不使用颜色 id 大小、排序或任何隐藏元数据决定映射。
  visual_cleanliness:
    - 变体没有增加额外标记层；只是单元内部多一种颜色，整体布局仍然清晰。
    - frame 只出现在 legend，输出只显示画布与复合图，避免多层叠加难以分辨。
  color_change_inferability:
    - 恰好 3 色而不是更多色，映射复杂度可控。
    - 每个 legend 内同一三色循环会在多个 frame 激活格里重复出现，提供重复证据。
    - 由于首次出现位置固定，可稳定区分循环方向，而不是仅知道"三色被重排了"。
  new_main_assumptions_to_guarantee:
    - generator 必须确保单元精灵恰好 3 色，并且 (0,0)、(0,1)、(1,0) 这三个位置互不相同，从而 `_ordered_sprite_colors` 一定得到长度 3 的可见顺序。
    - generator 仍需确保 frame 8 连通、marker panel 完整、复合图不越界不重叠。
"""

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


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


def _ordered_sprite_colors(sprite: np.ndarray) -> list[int]:
    """Visible color order = first appearance order in row-major scan."""
    colors: list[int] = []
    seen: set[int] = set()
    for val in sprite.flat:
        iv = int(val)
        if iv in (0, 6) or iv in seen:
            continue
        seen.add(iv)
        colors.append(iv)
    return colors


def _cycle_sprite_colors(sprite: np.ndarray) -> np.ndarray:
    """Cycle a->b->c->a using the sprite's visible first-appearance order."""
    colors = _ordered_sprite_colors(sprite)
    assert len(colors) == 3, f"unit sprite must use exactly 3 colors, got {colors}"
    out = sprite.copy()
    for i, color in enumerate(colors):
        out[sprite == color] = colors[(i + 1) % 3]
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
    spr = _cycle_sprite_colors(spr)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged sprites from legend panels and place them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend, a central canvas, and optionally a right legend.
    - Each legend contains, from top to bottom, a fully filled NxN unit sprite, a single-color frame mask, a 5x5 bordered anchor square, and a 3x3 rotation clock-hand.
    - The unit sprite now uses exactly three non-{0,6} colors; its color order is read from the first time each color appears in row-major order.

    Rule:
    - Read the unit sprite and apply a visible three-color cycle a->b->c->a, where (a,b,c) are the sprite colors in row-major first-appearance order.
    - Tile that cycled unit into each active cell of the frame, rotate the resulting composite according to the clock-hand, then place it on the canvas according to the anchor marker.

    Key insight:
    - The legend does not show the final tile colors directly: the solver must recover the output tile by cycling the three visible sprite colors in their appearance order, then the rest of the task proceeds exactly like the original frame/rotation/anchor assembly.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    left_sep_c1 = left_sep[1]
    canvas_c0 = left_sep_c1 + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main.shape
    full6_cols = [c for c in range(eff_w) if np.all(main[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main.copy()
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


# --------------- generator (refine_8conn_fill_overlap) ---------------

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


def _allocate_colors(num_legends: int):
    """Pick unit colors (3 per legend), 2 background colors, and 1 frame color per legend."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets: list[tuple[int, int, int]] = []
    used_unit: set[int] = set()
    for _ in range(num_legends):
        if len(pool) < 3:
            return None
        cols = [pool.pop(), pool.pop(), pool.pop()]
        random.shuffle(cols)
        unit_color_sets.append(tuple(cols))
        used_unit.update(cols)

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


def _make_legend_sprite(N: int, unit_colors: tuple[int, int, int]) -> np.ndarray:
    """
    Build a 3-color fully filled sprite.

    The cycle order is made visible by forcing the first appearances to be:
      (0,0) -> a, (0,1) -> b, (1,0) -> c.
    """
    order = list(unit_colors)
    random.shuffle(order)
    sprite = np.array(random.choices(order, k=N * N), dtype=int).reshape(N, N)
    seed_positions = [(0, 0), (0, 1), (1, 0)]
    for (r, c), color in zip(seed_positions, order):
        sprite[r, c] = color

    if N * N >= 6:
        counts = {color: int(np.sum(sprite == color)) for color in order}
        for color in order:
            if counts[color] > 1:
                continue
            candidates = [
                (r, c)
                for r in range(N)
                for c in range(N)
                if (r, c) not in seed_positions and sprite[r, c] != color
            ]
            if candidates:
                rr, cc = random.choice(candidates)
                sprite[rr, cc] = color
                counts[color] += 1

    assert len(_ordered_sprite_colors(sprite)) == 3
    return sprite


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]

    legend_sprite = _make_legend_sprite(N, unit_colors)
    sprite = _cycle_sprite_colors(legend_sprite)

    avail = marker_r0 - (N + 1)
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * N <= grid_h and rw * N <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False
    frame_H, frame_W, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_H,
        frame_W,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

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


def _comp_size(leg):
    """Return (height, width) of the composite after rotation."""
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _assign_anchors(legends, grid_h, canvas_w):
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


def _build_banded_canvas(grid_h, canvas_w, bg1, bg2):
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


def _overlay_composites(canvas, legends):
    """Stamp each legend's tiled composite onto the canvas (non-{0,6} pixels overwrite)."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["sprite"], leg["rotated_frame"]
        N = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * N, fw * N), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * N, j * N
                    comp[r0 : r0 + N, c0 : c0 + N][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * N, l : l + fw * N]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, frame)."""
    N = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, N - 4)) if not is_right else (sep_col + max(0, N - 3))

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
    grid[1 : 1 + N, mc0 + sp_col : mc0 + sp_col + N] = leg["legend_sprite"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (N + 1)
    gap = random.randint(1, max(1, avail - fH))
    fr_start = N + 1 + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]

    legends = []
    for _ in range(num_legends):
        N = random.randint(*UNIT_SIZE_RANGE)
        legends.append({"N": N, "legend_w": N + 2})

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
