# Variant of a32d8b75
# Dimension: D11 — 颜色交换规则 (transform)
# Change: 读取单元精灵时将两种颜色交换（a<->b），即图例中显示的是交换前的版本，实际铺贴使用交换后的版本... → 颜色映射由框架颜色决定

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite

# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的整体布局: 左/右图例、6色分隔列、画布、5x5 位置标记、3x3 旋转标记都不变。
#    - 变体只改“读取单元精灵后的颜色解释”:
#      图例里的单位精灵仍然是 2 色填满的 NxN 小块，但不再执行固定 a<->b 交换。
#      取单位精灵左上角那个颜色作为颜色A，另一种颜色作为颜色B。
#      实际铺贴时，把颜色A全部改成“该图例框架的颜色”，颜色B保持不变。
#    - 小例子:
#      图例精灵:
#        1 2
#        1 2
#      框架颜色 = 8
#      原题输出会变成:
#        2 1
#        2 1
#      本变体输出会变成:
#        8 2
#        8 2
#      差异发生在“读出单位精灵并准备铺贴”的那一步，而不是旋转、锚点或画布放置。
# b) 复杂度校准
#    - 若只说“按两种颜色中的某一个替换”但不给可见线索，会退化成猜谜。
#      因此颜色A固定定义为“单位精灵左上角的颜色”，这是肉眼可见的结构线索，不依赖颜色编号大小。
#    - 每个图例的框架颜色不同，故不同图例会得到不同映射结果；但仍只保留一个自由度
#      （A -> frame_color, B -> B），不会把规则扩散成过多分支。
# c) 隐式假设审查
#    - 原版 `_extract_unit_sprite()` 在读出精灵后按颜色编号排序并交换，这隐含了:
#      1. 颜色身份只由 id 决定；
#      2. 框架颜色与精灵着色无关。
#      这两个前提在 D11 中都不成立，所以该函数现在只负责提取原始精灵，不做变换。
#    - 原版 `main()` 读取 frame 后立刻转成布尔 mask，实际丢弃了 frame 的颜色。
#      D11 需要 frame_color 参与着色，因此必须先提取 frame 的唯一有效颜色，再构造 mask。
#    - 原版 generator 可允许 frame_color 与画布背景重合；在 D11 中这会让替换结果被背景吞没，
#      导致颜色迁移不可观察，所以生成器必须保证 unit colors / frame colors / canvas bg colors 两两分离。
# d) 放置合法性
#    - 复合图案尺寸、旋转、锚点、双图例不重叠、不得越界，继续沿用原题约束。
#    - 新增保证: frame_color 与单位精灵原始两色都不同，避免 A->frame_color 退化成不变。
# e) 视觉可观察性
#    - 人类可以在样例里看到: 图例的小精灵中，左上角那一色在最终大图中消失，
#      被同一面板框架的颜色接管；另一色则保持不变。
#    - 多图例样例中，不同框架颜色会导致不同的替换结果，形成重复支撑。
# f) 可视化整洁度
#    - 仍使用单色框架、双彩条背景、互不重叠的大复合体；不新增额外标记层，因此视觉负担与原题接近。
# g) 颜色变化可推断性
#    - 每个样例的颜色种类仍受控: 1-2 个图例，总共 2 个背景色、每图例 2 个单位精灵色、1 个框架色。
#      颜色变化只依赖一个显式对象属性（框架颜色），不是依赖位置排序或隐藏编号。
#
# Solver<->generator 契约
# - 单位精灵始终是 2 色、完整填满、左上角颜色视为颜色A。
# - 每个 frame 是单色且 8 连通。
# - frame_color 不与任何单位精灵色或背景色重复。
# - 双图例时两个 frame_color 也不同，这样“不同面板不同映射”在输入里有明确支撑。


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


def _extract_unit_sprite(legend: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Read the top-most 4-connected non-{0,6} component as the raw legend sprite."""
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
    Rotation marker is in the bottom 3x3 at rows (border6_r0+5 .. border6_r0+7) and cols (1..3),
    drawn with non-{0,6} pixels. The furthest pixel from the center determines direction:
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


def _first_sprite_color(sprite: np.ndarray) -> int:
    """Use the upper-left visible sprite color as the transferable color A."""
    pts = np.argwhere((sprite != 0) & (sprite != 6))
    assert pts.size > 0, "sprite has no colored cells"
    r0, c0 = pts[0]
    return int(sprite[r0, c0])


def _map_sprite_with_frame_color(sprite: np.ndarray, frame_color: int) -> np.ndarray:
    """Map color A (upper-left visible sprite color) to frame_color; keep all other colors."""
    out = sprite.copy()
    color_a = _first_sprite_color(sprite)
    out[sprite == color_a] = frame_color
    return out


def _single_frame_color(frame: np.ndarray) -> int:
    cols = {int(x) for x in np.unique(frame) if int(x) not in (0, 6)}
    assert cols, "frame has no active color"
    assert len(cols) == 1, f"frame must use exactly one color, got {cols}"
    return next(iter(cols))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct enlarged sprites from legend panels and recolor them using each frame's own hue.

    Input structure:
    - Color-6 separator columns split the grid into a left legend area, the output canvas, and
      optionally a right legend area.
    - Each legend panel still contains, from top to bottom: a 2-color unit sprite, a single-color
      frame template, a 5x5 anchor square, and a 3x3 rotation clock-hand.

    Rule:
    - Read the raw unit sprite exactly as drawn in the legend.
    - Before tiling it into the frame, replace every occurrence of the sprite's upper-left color
      with that legend frame's color; keep the sprite's other color unchanged.
    - Tile the recolored unit into the active frame cells, rotate by the clock-hand direction,
      then place the composite on the canvas using the anchor encoded by the marker square.

    Key insight: the frame is not just a shape mask anymore; its single color tells you how to
    recolor one of the sprite colors before the enlarged composite is assembled.
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
        frame_color = _single_frame_color(frame)
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        mapped_sprite = _map_sprite_with_frame_color(unit_sprite, frame_color)

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)

        m_unit = (mapped_sprite != 0) & (mapped_sprite != 6)
        uh, uw = mapped_sprite.shape
        fh, fw = frame_mask.shape
        comp = np.zeros((fh * uh, fw * uw), dtype=int)
        for i in range(fh):
            for j in range(fw):
                if not frame_mask[i, j]:
                    continue
                rr0 = i * uh
                cc0 = j * uw
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][m_unit] = mapped_sprite[m_unit]

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


# --------------- generator (frame-driven color transfer) ---------------

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
    """
    Allocate pairwise-distinct colors so the D11 recoloring stays visible:
    unit colors, background colors, and frame colors are all disjoint.
    """
    needed = (2 * num_legends) + 2 + num_legends
    if needed > len(USABLE_COLORS):
        return None
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    idx = 0
    for _ in range(num_legends):
        unit_color_sets.append((pool[idx], pool[idx + 1]))
        idx += 2

    bg_colors = (pool[idx], pool[idx + 1])
    idx += 2

    frame_colors = []
    for _ in range(num_legends):
        frame_colors.append(pool[idx])
        idx += 1

    return unit_color_sets, bg_colors, frame_colors


def _design_unit_sprite(n: int, color_a: int, color_b: int) -> np.ndarray:
    """
    Build a fully-filled 2-color sprite with:
    - top-left cell fixed to color_a, which defines the transferable color A
    - both colors appearing at least twice when possible, so the remapping is visible
    """
    total = n * n
    sprite = np.full((n, n), color_a, dtype=int)
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]
    num_b = random.randint(2, total - 2)
    for r, c in random.sample(candidates, num_b):
        sprite[r, c] = color_b
    return sprite


def _design_legend(leg, unit_colors, frame_color, marker_r0, grid_h, canvas_w):
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    n = leg["N"]
    color_a, color_b = unit_colors
    sprite = _design_unit_sprite(n, color_a, color_b)
    output_sprite = _map_sprite_with_frame_color(sprite, frame_color)

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
    frame_h, frame_w, rot_k = random.choice(combos)

    frame_mask = random_sprite(
        frame_h,
        frame_w,
        density=FRAME_DENSITY,
        connectivity=8,
        color_palette=[1],
        symmetry="not_symmetric",
    ) != 0
    rot_frame = np.rot90(frame_mask, (-rot_k) % 4)

    leg.update(
        sprite=sprite,
        output_sprite=output_sprite,
        frame_mask=frame_mask,
        frame_H=frame_h,
        frame_W=frame_w,
        frame_color=frame_color,
        rot_k=rot_k,
        rotated_frame=rot_frame,
    )
    return True


def _comp_size(leg):
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions while keeping all composites in-bounds and disjoint."""
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
    """Stamp each legend's recolored tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        sp, rf = leg["output_sprite"], leg["rotated_frame"]
        n = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * n, fw * n), dtype=int)
        sprite_mask = (sp != 0) & (sp != 6)
        for i in range(fh):
            for j in range(fw):
                if rf[i, j]:
                    r0, c0 = i * n, j * n
                    comp[r0 : r0 + n, c0 : c0 + n][sprite_mask] = sp[sprite_mask]

        region = output[t : t + fh * n, l : l + fw * n]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid, leg, col_start, sep_col, marker_r0, is_right):
    """Draw one legend panel (separator, marker square, rotation patch, raw sprite, frame)."""
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
    grid[1 : 1 + n, mc0 + sp_col : mc0 + sp_col + n] = leg["sprite"]

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

    for li in range(num_legends):
        if not _design_legend(
            legends[li],
            unit_color_sets[li],
            frame_colors[li],
            marker_r0,
            grid_h,
            canvas_w,
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


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
