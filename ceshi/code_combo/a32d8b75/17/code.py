# Combo variant of a32d8b75 (index 17)
# combo_spec: D2/1+D11/0+D17/0
# Name: 空洞精灵+无交换+颜色重叠

"""
design_analysis:
  conflict_resolution:
    - 组合名称写成“空洞精灵”，但 combo_spec 中 D2/1 的明确“改为”是三色精灵；按审查要求，以 combo_spec 的改为行为为准。
    - 因为 D11/0 又把“颜色交换”取消，所以融合后的真实语义是：图例中的单元精灵使用 3 种颜色，且直接按图例原样铺贴，不再做任何交换或循环置换。
    - D17/0 继续保留：允许精灵中的一种颜色与画布背景重叠，使部分精灵像素在输出里视觉伪装。
  concretization_and_behavior_trace:
    - 输入整体版式不变：左 legend / 中央画布 / 可选右 legend，legend 内部仍是“unit sprite -> frame -> 5x5 anchor square -> 3x3 rotation hand”。
    - 原题的 unit sprite 是双色块，solver 读取后会做颜色交换；这里改成全填充三色 sprite，solver 直接按图例原样使用。
    - 小例子：
      old_logic:
        - legend sprite = [[1,2],[2,1]]，solver 实际铺贴 [[2,1],[1,2]]。
        - 若背景条纹是 {3,5}，所有精灵像素都会显式改写背景。
      fused_logic:
        - legend sprite = [[1,2,3],[2,3,1],[3,1,2]]，solver 原样读取，不做任何颜色交换。
        - 若颜色 2 同时也是背景条纹色，则某些 2 像素虽然逻辑上属于精灵，但视觉上会“消失”进背景；而 1 和 3 仍把完整三色结构暴露出来。
      difference:
        - 锚点和旋转规则没变，变化集中在“三色直读”与“部分颜色和背景重合后的可见性”。
  complexity_calibration:
    - 只保留 1-2 个 legends、原有 frame/rotation/anchor 语法，避免同时引入太多新自由度。
    - sprite 固定只用 3 种非 {0,6} 颜色，复杂度集中在“读图例时不交换”和“部分颜色被背景伪装”。
    - generator 强制每个样本同时具有：三色 sprite、至少一些与背景同色而隐藏的精灵像素、以及与“三色循环置换版”不同的可见输出。
  implicit_assumptions_audit_main:
    - `_extract_unit_sprite()` 仍然从顶部 4 连通对象裁出 bbox；融合后唯一关键变化是颜色直接使用，不再交换。
    - `m_unit = (unit_sprite != 0) & (unit_sprite != 6)` 仍然只复制前景色，兼容偶发 0，但 generator 会稳定使用全填充三色 sprite。
    - `_extract_frame_bbox()` 仍依赖 unit 和 frame 之间有空隙；generator 继续保证。
    - `composite_written` 的含义仍是“逻辑上被某个 composite 占据的非零像素不能与别的 composite 重叠”；即使该像素与背景同色而不可见，也仍算占据。
    - 位置与旋转的求解代码不依赖颜色不相交假设，所以 D17 只影响覆盖结果，不影响 panel 解析。
  placement_legality:
    - 仍采用 output-first：先设计每个 legend 的 sprite/frame/rotation，再筛掉放不进 canvas 的组合。
    - 双 legend 时继续枚举锚点组合并禁止 composite bbox 重叠，避免隐藏像素把真正重叠伪装过去。
    - 背景条纹单独重采样，直到每个 legend 都出现一定数量的“sprite 像素 == 底色”隐藏现象。
  visual_observability:
    - 人类可以直接在 legend 里看到 sprite 有 3 种颜色；输出中反复重复同一个三色纹理，因此三色直读是可归纳的。
    - 同时，某一类 sprite 颜色会在部分条纹上“凭空消失”，但在另一类条纹上仍可见，因此能从少量样例归纳出“不是没画，而是与背景同色”。
    - 不依赖颜色 id 排序、随机 tie-break 等不可见规则。
  visual_cleanliness:
    - 画面元素仍只有 legend、分隔线、条纹背景和 composite，没有额外提示层。
    - frame 非激活区域仍由原始 frame 决定；背景伪装只作用于精灵的单一颜色，不引入额外提示层或噪声。
  color_change_inferability:
    - 颜色规则只有一个：不交换，直接用 legend sprite。
    - generator 额外拒绝“直接用”和“三色循环置换后用”产生相同输出的样本，因此 D11 分量不会退化为原题式颜色变化。
    - 背景只用 2 种颜色，并且每个 legend 只让其中 1 种与它的 sprite 色重合，方便解题者从重复出现的消失模式反推出伪装来源。
  new_main_assumptions_to_guarantee:
    - 每个 unit sprite 的非零像素必须 4 连通，且 bbox 四边都被前景触及，这样 `_extract_unit_sprite()` 才能恢复完整三色 bbox。
    - 每个 legend 至少有 3 种 sprite 颜色、至少一些隐藏的同色覆盖像素；并且“原样使用”与“三色循环置换后使用”必须得到不同可见结果。
"""

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    _h, w = g.shape
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
    _h, w = g.shape
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
    """
    Extract the top-most 4-connected non-{0,6} component as the unit sprite bbox.

    Any internal 0 cells are preserved if present, but the key fused behavior is that
    the legend colors are used directly with no swapping.
    """
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
    return spr, (r0, c0, r1, c1)


def _extract_frame_bbox(
    legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int
) -> tuple[int, int, int, int] | None:
    """Frame pixels live between the unit sprite and the border-6 marker square."""
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
    Read the 3x3 clock-hand below the marker square.

    The furthest non-{0,6} pixel from the center determines direction:
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


def _compose_tiled_sprite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into every active frame cell using legend colors literally."""
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


def _effective_canvas_shape(canvas: np.ndarray) -> tuple[int, int]:
    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)
    return eff_h, eff_w


def _scene_from_grid(
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    base_canvas = g[:, canvas_c0:canvas_c1].copy()
    eff_h, eff_w = _effective_canvas_shape(base_canvas)

    legends: list[dict] = []
    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

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
        _, n8 = _scipy_label(frame_mask0, structure=STRUCT8)
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        rotated_frame = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _compose_tiled_sprite(unit_sprite, rotated_frame)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        legends.append(
            {
                "unit_sprite": unit_sprite,
                "rotated_frame": rotated_frame,
                "composite": comp,
                "top": top,
                "left": left,
                "marker_r0": marker_r0,
                "marker_c0": marker_c0,
                "anchor": (pr, pc),
            }
        )

    return g, base_canvas, eff_h, eff_w, legends


def _render_output_from_scene(base_canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    out = base_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)
    for leg in legends:
        comp = leg["composite"]
        top, left = int(leg["top"]), int(leg["left"])
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
                if val == 0 or val == 6:
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                out[rr, cc] = val
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legend panels and paint them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns split the grid into a left legend, a central canvas, and sometimes a right legend.
    - Each legend still stacks four cues: a small unit sprite at the top, an 8-connected frame mask, a 5x5 anchor square, and a 3x3 rotation hand.
    - Unlike the source task, the unit sprite is read literally with all of its legend colors intact; no color exchange is applied after extraction.

    Rule:
    - Read the unit sprite exactly as shown in the legend, with no color exchange at all.
    - Tile that three-color sprite into every active frame cell, rotate the enlarged composite according to the clock-hand, and place it using the anchor marker.
    - When a placed sprite color equals the pre-existing canvas color, it still counts as a sprite pixel logically, but it becomes visually camouflaged against the background.

    Key insight:
    - Some apparent gaps are only camouflage: the legend still tells you the full three-color sprite, but one of those colors may blend into the striped background on placement.
    """
    _g, base_canvas, _eff_h, _eff_w, legends = _scene_from_grid(grid)
    return _render_output_from_scene(base_canvas, legends)


# --------------- generator (tricolor_literal_camouflage) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNTS = (1, 2)
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
CANVAS_RETRIES = 120
MIN_HIDDEN_PIXELS = 6


def _allocate_colors(
    num_legends: int,
) -> tuple[list[tuple[int, int, int]], tuple[int, int], list[int], list[int]] | None:
    """
    Pick sprite colors, background colors, frame colors, and one hidden color per legend.

    At least one canvas background color is taken from each legend's sprite palette so the
    camouflage behavior is guaranteed to be available.
    """
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets: list[tuple[int, int, int]] = []
    for _ in range(num_legends):
        if len(pool) < 3:
            return None
        c1, c2, c3 = pool.pop(), pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2, c3))

    hidden_colors = [random.choice(cols) for cols in unit_color_sets]
    if num_legends == 1:
        bg_hidden = hidden_colors[0]
        bg_other_choices = [c for c in USABLE_COLORS if c not in set(unit_color_sets[0])]
        if not bg_other_choices:
            return None
        bg_colors = (bg_hidden, random.choice(bg_other_choices))
    else:
        bg_colors = (hidden_colors[0], hidden_colors[1])

    if random.random() < 0.5:
        bg_colors = (bg_colors[1], bg_colors[0])

    frame_colors: list[int] = []
    all_set = set(USABLE_COLORS)
    for cols in unit_color_sets:
        candidates = list(all_set - set(cols))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors, hidden_colors


def _make_tricolor_unit_sprite(
    N: int, unit_colors: tuple[int, int, int], hidden_color: int
) -> np.ndarray | None:
    """Create a bbox-tight, 4-connected, fully filled tricolor sprite."""
    min_hidden = 1 if N == 2 else max(2, N - 1)
    for _ in range(200):
        rr, cc = np.indices((N, N))
        mode = random.choice(["diag", "diag_flip", "rows_plus_cols", "rows_plus_diag"])
        if mode == "diag":
            idx = (rr + cc) % 3
        elif mode == "diag_flip":
            idx = (2 * rr + cc) % 3
        elif mode == "rows_plus_cols":
            idx = (rr + 2 * cc) % 3
        else:
            idx = (2 * rr + 2 * cc + rr // max(1, N - 1)) % 3
        colors = list(unit_colors)
        random.shuffle(colors)
        sprite = np.take(np.array(colors, dtype=int), idx)
        hidden_pts = np.argwhere(sprite == hidden_color)
        if hidden_pts.shape[0] < min_hidden:
            continue
        if N >= 3 and len(np.unique(hidden_pts[:, 0])) < 2:
            continue
        if N >= 3 and len(np.unique(hidden_pts[:, 1])) < 2:
            continue
        if len({int(x) for x in np.unique(sprite)}) != 3:
            continue
        return sprite
    return None


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int, int],
    frame_color: int,
    hidden_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    sprite = _make_tricolor_unit_sprite(N, unit_colors, hidden_color)
    if sprite is None:
        return False

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
        legend_sprite=sprite.copy(),
        frame_mask=frame_mask,
        frame_H=frame_H,
        frame_W=frame_W,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
        hidden_color=hidden_color,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _build_composite(leg: dict, sprite: np.ndarray | None = None) -> np.ndarray:
    use_sprite = leg["sprite"] if sprite is None else sprite
    return _compose_tiled_sprite(use_sprite, leg["rotated_frame"])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions while keeping all composites in bounds and disjoint."""
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


def _build_banded_canvas(
    grid_h: int, canvas_w: int, bg1: int, bg2: int, band_dir: str, band_width: int, offset: int
) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical bands."""
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    dim = grid_h if band_dir == "h" else canvas_w
    band_indices = ((np.arange(dim) + offset) // band_width) % 2
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _hidden_pixel_counts(canvas: np.ndarray, legends: list[dict]) -> list[int]:
    """Count per-legend pixels whose sprite color matches the underlying canvas color."""
    counts: list[int] = []
    for leg in legends:
        comp = _build_composite(leg)
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = canvas[t : t + ch, l : l + cw]
        hidden = (comp != 0) & (comp != 6) & (comp == region)
        counts.append(int(hidden.sum()))
    return counts


def _build_camouflage_canvas(
    grid_h: int, canvas_w: int, bg1: int, bg2: int, legends: list[dict]
) -> tuple[np.ndarray, list[int]] | None:
    best: tuple[np.ndarray, list[int]] | None = None
    best_score = (-1, -1)
    for _ in range(CANVAS_RETRIES):
        band_dir = random.choice(["h", "v"])
        dim = grid_h if band_dir == "h" else canvas_w
        band_width = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
        offset = random.randint(0, 2 * band_width - 1)
        canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2, band_dir, band_width, offset)
        counts = _hidden_pixel_counts(canvas, legends)
        score = (min(counts), sum(counts))
        if score > best_score:
            best = (canvas, counts)
            best_score = score
        if min(counts) > 0 and sum(counts) >= MIN_HIDDEN_PIXELS:
            return canvas, counts
    if best is not None and best_score[0] > 0 and best_score[1] >= MIN_HIDDEN_PIXELS:
        return best
    return None


def _overlay_composites(canvas: np.ndarray, legends: list[dict], sprite_key: str = "sprite") -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg, leg[sprite_key])
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (comp != 6) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _cycle_three_colors(sprite: np.ndarray) -> np.ndarray:
    """Verification helper: apply a 3-cycle to the sprite's non-zero colors."""
    colors = sorted(int(x) for x in np.unique(sprite) if int(x) not in (0, 6))
    assert len(colors) == 3, f"expected exactly three sprite colors, got {colors}"
    out = sprite.copy()
    for src, dst in zip(colors, colors[1:] + colors[:1]):
        out[sprite == src] = dst
    return out


def _gen_attempt(target_legend_count: int | None = None) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    num_legends = target_legend_count if target_legend_count is not None else random.choice(LEGEND_COUNTS)

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
    unit_color_sets, (bg1, bg2), frame_colors, hidden_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            hidden_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
        ):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    camo = _build_camouflage_canvas(grid_h, canvas_w, bg1, bg2, legends)
    if camo is None:
        return None
    canvas, hidden_counts = camo

    if min(len({int(x) for x in np.unique(leg["sprite"]) if int(x) not in (0, 6)}) for leg in legends) != 3:
        return None

    output = _overlay_composites(canvas, legends)

    cycled_legends = [dict(leg, sprite=_cycle_three_colors(leg["sprite"])) for leg in legends]
    cycled_output = _overlay_composites(canvas, cycled_legends)
    if np.array_equal(output, cycled_output):
        return None

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(
            grid,
            legends[1],
            col_start=r_sep + 1,
            sep_col=r_sep,
            marker_r0=marker_r0,
            is_right=True,
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


def fusion_metrics_from_input(grid: np.ndarray) -> dict:
    """
    Quantify the fused behavior on a generated sample.

    Metrics:
      - unit_color_counts: per-legend number of distinct sprite colors in the legend.
      - hidden_pixels: per-legend count of same-color camouflage writes.
      - cycle_diff_pixels: visible pixels that differ from the 3-cycle ablation.
    """
    _g, base_canvas, _eff_h, _eff_w, legends = _scene_from_grid(grid)
    actual = _render_output_from_scene(base_canvas, legends)

    hidden_counts: list[int] = []
    unit_color_counts: list[int] = []
    for leg in legends:
        comp = leg["composite"]
        ch, cw = comp.shape
        region = base_canvas[leg["top"] : leg["top"] + ch, leg["left"] : leg["left"] + cw]
        hidden = (comp != 0) & (comp != 6) & (comp == region)
        hidden_counts.append(int(hidden.sum()))
        unit_color_counts.append(len({int(c) for c in np.unique(leg["unit_sprite"]) if int(c) not in (0, 6)}))

    cycled_legends = [dict(leg, composite=_build_composite({"sprite": _cycle_three_colors(leg["unit_sprite"]), "rotated_frame": leg["rotated_frame"]})) for leg in legends]
    for src, dst in zip(legends, cycled_legends):
        dst["top"] = src["top"]
        dst["left"] = src["left"]
    cycled = _render_output_from_scene(base_canvas, cycled_legends)

    return {
        "legend_count": len(legends),
        "unit_color_counts": unit_color_counts,
        "hidden_pixels": hidden_counts,
        "cycle_diff_pixels": int(np.count_nonzero(actual != cycled)),
    }


def generate_input() -> np.ndarray:
    target_legend_count = random.choice(LEGEND_COUNTS)
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(target_legend_count=target_legend_count)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
