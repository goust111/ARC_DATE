# Combo variant of a32d8b75 (index 1)
# combo_spec: D1/1+D11/2
# Name: 颜色匹配铺贴+框架色决定映射

import random

import numpy as np
from scipy.ndimage import label as _scipy_label

from arc_utils import bbox_coords, neighbors4, random_sprite

# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的总布局: 左/右 legend、6 色分隔列、中央 canvas、5x5 锚点框、3x3 旋转标记都不变。
#    - 融合后每个 frame active cell 不再只表达“这里要铺一个 tile”，而是同时表达两件事:
#      1. 这个 tile 该使用 raw sprite 还是 color-swapped sprite；
#      2. 这个 tile 内哪一类像素要被改成 frame cell 自己的颜色。
#    - 具体小例子:
#      raw sprite =
#        1 1
#        1 2
#      swapped sprite =
#        2 2
#        2 1
#      若 raw sprite 的左上角色是颜色 A，另一色是颜色 B，
#      某个 frame cell 颜色为 8，且它属于“raw 变体”那一类，则 tile 变成
#        8 8
#        8 bg
#      若另一个 frame cell 颜色为 3，且它属于“swapped 变体”那一类，则颜色角色仍按 raw sprite 定义，
#      因而 tile 会把原本属于颜色 B 的那部分改成 3，变成
#        bg bg
#        bg 3
#      这里 bg 不是固定色，而是该 tile 下方 canvas 条带的背景色。原题会把两处都铺成同一个固定配色；
#      融合后，同一复合体里既会因为 frame 颜色不同而切换 raw/swapped，又会因为底下背景带不同而改变次色。
# b) 复杂度校准
#    - 每个 legend 只使用两种 frame 颜色，对应两个 sprite 变体；自由度控制在 2 档，避免无约束多映射。
#    - canvas 只用两种背景色做水平或垂直条带，且条带宽度按 tile 大小对齐，解题者能稳定观察“次色 = 当地背景色”。
#    - generator 强制两个 frame 颜色都出现，且复合体跨越至少两条背景带，避免退化为单变体或固定背景色映射。
# c) 隐式假设审查
#    - 原版 `_extract_unit_sprite()` 读出后立即固定交换颜色；融合版不能这样做，因为 raw/swapped 两个版本都要保留。
#    - 原版把 frame 立刻压成布尔 mask；融合版必须保留 frame 的颜色，并且颜色顺序要可解释。
#      这里采用“frame 中按行扫描最先出现的颜色 = raw 变体，其余颜色 = swapped 变体”。
#      这个顺序来自可见结构，不依赖颜色编号。
#    - 新规则还依赖“tile 下方的背景色”。因此 solver 不能先整体做 comp 再盲写到 canvas，
#      必须在知道放置坐标后，逐 tile 从原始 canvas 读取局部背景色。
#    - 因为次色会映射回背景色，部分 tile 像素会与底图同色；这些像素在写回时应视作不改动，
#      否则 overlap 检查会把“透明效果”误判为冲突。
# d) 放置合法性
#    - 复合体外接尺寸仍是 rotated_frame.shape * unit_size。
#    - 双 legend 情况继续用放置框不重叠约束，避免不同复合体互相覆盖。
#    - 为了让“局部背景色”定义稳定，generator 让背景条带宽度是 unit_size 的整数倍，
#      并让复合体在条带变化的那一轴上按 unit_size 对齐。
# e) 视觉可观察性
#    - 输入中 frame 明显是双色分区，输出中每个 tile 只出现“frame 色 + 周围背景色”两色，
#      人类能把 frame 色与 tile 里被点亮的那部分形状对应起来。
#    - 同一复合体会跨越两种背景条带，因此解题者能看到相同 frame 色在不同带上共享相同主形状，
#      但次色会跟着当地背景切换。
#    - 不使用颜色编号排序、黑色特殊意义等不可视信息；所有判别都来自可见位置、区域和背景。
# f) 可视化整洁度
#    - sprite/frame/background 只使用 1,2,3,5,8,9，避免与锚点色 4 和旋转色 7 混淆。
#    - frame 两色做成较大片区，而不是噪声散点；输出里每个 tile 只额外引入一个 frame 色，层次保持清楚。
# g) 变化的可推断性
#    - 每个样例内每个 legend 只有 2 个 frame 色、2 个 sprite 色、2 个背景色，颜色种类受控。
#    - frame 色决定 raw/swapped 的差异，是“哪一部分变成 frame 色”；背景色决定剩余部分保留成哪种条带色。
#      两层信号都能在 2-3 个样例里重复观察到。
# h) Solver<->generator 契约
#    - unit sprite 始终是完整填满的双色方阵，左上角颜色稳定定义 raw 版本的颜色 A，另一色为颜色 B。
#    - frame 始终 8 连通、恰有两种 active 颜色，且按行扫描最早出现的 active cell 使用第一种 frame 色。
#    - frame 色与该 legend 的 sprite 色、全局背景色都不重合。
#    - 背景条带宽度是 unit_size 的整数倍，且复合体会跨越至少两条背景带，使“次色 = 局部背景色”真正生效。


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


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    cols = [int(x) for x in np.unique(sprite) if int(x) not in (0, 6)]
    assert len(cols) == 2, f"unit sprite must use exactly two colors, got {cols}"
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
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


def _first_active_color(sprite: np.ndarray) -> int:
    vals = [int(x) for x in sprite.flat if int(x) not in (0, 6)]
    assert vals, "sprite must contain active pixels"
    return vals[0]


def _other_active_color(sprite: np.ndarray, first_color: int) -> int:
    cols = [int(x) for x in np.unique(sprite) if int(x) not in (0, 6)]
    assert len(cols) == 2, f"sprite must use exactly two colors, got {cols}"
    if cols[0] == first_color:
        return cols[1]
    if cols[1] == first_color:
        return cols[0]
    raise AssertionError("first color missing from sprite palette")


def _ordered_frame_colors(frame: np.ndarray) -> list[int]:
    seen: list[int] = []
    for x in frame.flat:
        val = int(x)
        if val in (0, 6):
            continue
        if val not in seen:
            seen.append(val)
    assert len(seen) == 2, f"frame must use exactly two active colors, got {seen}"
    return seen


def _map_variant_to_colors(
    raw_sprite: np.ndarray, variant: np.ndarray, frame_color: int, bg_color: int
) -> np.ndarray:
    """
    Recolor one sprite variant for a specific tile:
      raw sprite color A -> frame color
      raw sprite color B -> local background color

    The A/B roles stay tied to the raw legend sprite even after variant swapping. This is what
    makes the raw/swapped choice materially change which positions get the frame color.
    """
    primary = _first_active_color(raw_sprite)
    secondary = _other_active_color(raw_sprite, primary)
    out = np.zeros_like(variant)
    out[variant == primary] = frame_color
    out[variant == secondary] = bg_color
    return out


def _tile_background_color(canvas: np.ndarray, top: int, left: int, h: int, w: int) -> int:
    """
    Read the dominant non-6 color under a tile region.
    Generator aligns tiles to background bands, so this is normally unique; using the dominant
    color keeps the solver tolerant if a region barely touches a band boundary.
    """
    region = canvas[top : top + h, left : left + w]
    vals = np.asarray([int(x) for x in region.flat if int(x) != 6], dtype=int)
    assert vals.size > 0, "tile background region is empty"
    colors, counts = np.unique(vals, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from legend panels whose frame colors jointly choose both tiling variant and recoloring.

    Input structure:
    - Color-6 separator columns still isolate one or two full-height legend panels from the central canvas.
    - Each legend panel still contains a 2-color unit sprite, a 2-color frame template, a 5x5 anchor square,
      and a 3x3 rotation clock-hand.
    - The canvas keeps its striped background, and the frame now uses two different colors to mark different tile classes.

    Rule:
    - Read the raw legend sprite and derive two variants: the raw coloring and its 2-color swap.
    - The frame color that appears first when scanning the legend frame selects the raw variant; the other frame color
      selects the swapped variant. For each active frame cell, recolor the chosen variant using the raw sprite's two color roles:
      raw color A becomes that frame cell's color, while raw color B becomes the local canvas background under that tile.
    - Rotate the colored frame layout by the clock-hand direction, tile the per-cell recolored sprite blocks, and place the
      resulting composite using the anchor encoded by the bordered square.

    Key insight: A frame cell is now a coupled instruction. Its color tells you both which sprite variant to use and which
    raw sprite role should be painted with that color; the other raw role is not fixed at all and instead copies the background
    stripe already sitting underneath that tile.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    base_canvas = canvas.copy()
    out = canvas.copy()
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
        raw_sprite, unit_bb = unit
        swapped_sprite = _swap_two_colors(raw_sprite)
        unit_h, unit_w = raw_sprite.shape

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"
        ordered_frame_colors = _ordered_frame_colors(frame)

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_rot = np.rot90(frame, (-dir_k) % 4)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, frame_rot.shape[0] * unit_h)
        left = _anchor_to_top_left(pc, eff_w, frame_rot.shape[1] * unit_w)

        fh, fw = frame_rot.shape
        for i in range(fh):
            for j in range(fw):
                frame_color = int(frame_rot[i, j])
                if frame_color in (0, 6):
                    continue
                variant = raw_sprite if frame_color == ordered_frame_colors[0] else swapped_sprite
                tile_top = top + i * unit_h
                tile_left = left + j * unit_w
                bg_color = _tile_background_color(base_canvas, tile_top, tile_left, unit_h, unit_w)
                tile = _map_variant_to_colors(raw_sprite, variant, frame_color, bg_color)

                for r in range(unit_h):
                    rr = tile_top + r
                    assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
                    for c in range(unit_w):
                        cc = tile_left + c
                        assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                        if out[rr, cc] == 6:
                            continue
                        val = int(tile[r, c])
                        if val in (0, 6):
                            continue
                        if val == int(base_canvas[rr, cc]):
                            continue
                        assert not composite_written[rr, cc], "composites must not overlap on changed pixels"
                        composite_written[rr, cc] = True
                        out[rr, cc] = val
    return out


# --------------- generator (frame-color chooses variant + local background transfer) ---------------

GRID_H_RANGE = (20, 30)
GRID_W = 30
LEGEND_COUNT_WEIGHTS = [1, 1]
UNIT_SIZE_RANGE = (2, 5)
MIN_CANVAS_W = 8
FRAME_H_RANGE = (4, 9)
FRAME_W_RANGE = (3, 6)
FRAME_DENSITY = 0.5
FRAME_ACTIVE_RANGE = (8, 18)
MARKER_PANEL_HEIGHT = 8
PALETTE = [1, 2, 3, 5, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MIN_COLOR_GROUP = 2
GEN_RETRIES = 600


def _choose_bg_colors() -> tuple[int, int]:
    bg = random.sample(PALETTE, 2)
    return int(bg[0]), int(bg[1])


def _allocate_legend_colors(num_legends: int, bg_colors: tuple[int, int]):
    """
    Per legend:
    - choose 2 sprite colors from the non-background palette
    - use the remaining 2 colors as frame colors
    Colors may repeat across legends, but within one legend sprite/frame/background stay disjoint.
    """
    non_bg = [c for c in PALETTE if c not in bg_colors]
    legends = []
    for _ in range(num_legends):
        sprite_colors = random.sample(non_bg, 2)
        frame_colors = [c for c in non_bg if c not in sprite_colors]
        assert len(frame_colors) == 2
        if random.random() < 0.5:
            frame_colors = frame_colors[::-1]
        legends.append(((int(sprite_colors[0]), int(sprite_colors[1])), (int(frame_colors[0]), int(frame_colors[1]))))
    return legends


def _make_unit_sprite(n: int, color_a: int, color_b: int) -> np.ndarray:
    """Build a fully-filled 2-color sprite with top-left fixed to color_a."""
    total = n * n
    sprite = np.full((n, n), color_a, dtype=int)
    candidates = [(r, c) for r in range(n) for c in range(n) if (r, c) != (0, 0)]
    if total <= 4:
        num_b = 1
    else:
        num_b = random.randint(2, total - 2)
    for r, c in random.sample(candidates, num_b):
        sprite[r, c] = color_b
    return sprite


def _generate_frame_mask(frame_h: int, frame_w: int) -> np.ndarray | None:
    for _ in range(80):
        frame_mask = random_sprite(
            frame_h,
            frame_w,
            density=FRAME_DENSITY,
            connectivity=8,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        active = int(frame_mask.sum())
        if not (FRAME_ACTIVE_RANGE[0] <= active <= FRAME_ACTIVE_RANGE[1]):
            continue
        _, n8 = _scipy_label(frame_mask, structure=np.ones((3, 3)))
        if n8 != 1:
            continue
        return frame_mask
    return None


def _colorize_frame(frame_mask: np.ndarray, frame_colors: tuple[int, int]) -> np.ndarray | None:
    pts = np.argwhere(frame_mask)
    if len(pts) < 2 * MIN_COLOR_GROUP:
        return None

    coord_modes = [
        pts[:, 0],
        pts[:, 1],
        pts[:, 0] + pts[:, 1],
        pts[:, 0] - pts[:, 1],
    ]
    random.shuffle(coord_modes)

    split_mask = None
    for vals in coord_modes:
        threshold = np.median(vals)
        cand = vals <= threshold
        count_a = int(cand.sum())
        count_b = int((~cand).sum())
        if count_a >= MIN_COLOR_GROUP and count_b >= MIN_COLOR_GROUP:
            split_mask = cand
            break
    if split_mask is None:
        order = np.argsort(pts[:, 0] * 100 + pts[:, 1])
        split_mask = np.zeros(len(pts), dtype=bool)
        split_mask[order[: len(pts) // 2]] = True
        count_a = int(split_mask.sum())
        count_b = int((~split_mask).sum())
        if count_a < MIN_COLOR_GROUP or count_b < MIN_COLOR_GROUP:
            return None

    first_pt = np.lexsort((pts[:, 1], pts[:, 0]))[0]
    if not split_mask[first_pt]:
        split_mask = ~split_mask

    frame = np.zeros(frame_mask.shape, dtype=int)
    frame[tuple(pts[split_mask].T)] = frame_colors[0]
    frame[tuple(pts[~split_mask].T)] = frame_colors[1]
    assert _ordered_frame_colors(frame) == [frame_colors[0], frame_colors[1]]
    return frame


def _design_legend(
    leg,
    unit_colors: tuple[int, int],
    frame_colors: tuple[int, int],
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
    band_dir: str,
    band_cells: int,
) -> bool:
    """
    Fill in sprite, 2-color frame, and rotation for one legend dict.
    Enforce that the rotated frame crosses at least two background bands.
    """
    n = leg["N"]
    sprite = _make_unit_sprite(n, unit_colors[0], unit_colors[1])

    avail = marker_r0 - (n + 1)
    combos = []
    for frame_h in range(*FRAME_H_RANGE):
        if avail < frame_h + 1:
            continue
        for frame_w in range(*FRAME_W_RANGE):
            for rot_k in range(4):
                rot_h, rot_w = (frame_h, frame_w) if rot_k % 2 == 0 else (frame_w, frame_h)
                if rot_h * n > grid_h or rot_w * n > canvas_w:
                    continue
                span_tiles = rot_h if band_dir == "h" else rot_w
                if span_tiles <= band_cells:
                    continue
                combos.append((frame_h, frame_w, rot_k))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_h, frame_w, rot_k in combos:
        frame_mask = _generate_frame_mask(frame_h, frame_w)
        if frame_mask is None:
            continue
        frame = _colorize_frame(frame_mask, frame_colors)
        if frame is None:
            continue
        rotated_frame = np.rot90(frame, (-rot_k) % 4)
        leg.update(
            sprite=sprite,
            frame=frame,
            frame_colors=frame_colors,
            frame_H=frame_h,
            frame_W=frame_w,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
        )
        return True
    return False


def _comp_size(leg) -> tuple[int, int]:
    rf = leg["rotated_frame"]
    n = leg["N"]
    return rf.shape[0] * n, rf.shape[1] * n


def _is_band_aligned(top: int, left: int, n: int, band_dir: str) -> bool:
    return (top % n == 0) if band_dir == "h" else (left % n == 0)


def _assign_anchors(legends, grid_h: int, canvas_w: int, band_dir: str) -> bool:
    """Assign anchor positions while keeping composites in-bounds, band-aligned, and disjoint."""
    if len(legends) == 1:
        leg = legends[0]
        ch, cw = _comp_size(leg)
        choices = []
        for pr in range(3):
            for pc in range(3):
                top = _anchor_to_top_left(pr, grid_h, ch)
                left = _anchor_to_top_left(pc, canvas_w, cw)
                if top < 0 or left < 0 or top + ch > grid_h or left + cw > canvas_w:
                    continue
                if not _is_band_aligned(top, left, leg["N"], band_dir):
                    continue
                choices.append((pr, pc, top, left))
        if not choices:
            return False
        pr, pc, top, left = random.choice(choices)
        leg.update(pr=pr, pc=pc, comp_top=top, comp_left=left)
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
            if not _is_band_aligned(at, al, a["N"], band_dir):
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    if bt < 0 or bl < 0 or bt + b_ch > grid_h or bl + b_cw > canvas_w:
                        continue
                    if not _is_band_aligned(bt, bl, b["N"], band_dir):
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
    grid_h: int, canvas_w: int, bg1: int, bg2: int, band_dir: str, band_width: int
) -> np.ndarray:
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    colors = np.array([bg1, bg2], dtype=int)
    if band_dir == "h":
        band_indices = np.arange(grid_h) // band_width % 2
        canvas[:] = colors[band_indices][:, None]
    else:
        band_indices = np.arange(canvas_w) // band_width % 2
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends) -> np.ndarray:
    """Stamp each legend's fused composite onto the canvas."""
    output = canvas.copy()
    written = np.zeros(canvas.shape, dtype=bool)
    for leg in legends:
        raw_sprite = leg["sprite"]
        swapped_sprite = _swap_two_colors(raw_sprite)
        frame = leg["rotated_frame"]
        n = leg["N"]
        top = leg["comp_top"]
        left = leg["comp_left"]
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                frame_color = int(frame[i, j])
                if frame_color == 0:
                    continue
                variant = raw_sprite if frame_color == leg["frame_colors"][0] else swapped_sprite
                tile_top = top + i * n
                tile_left = left + j * n
                bg_color = _tile_background_color(canvas, tile_top, tile_left, n, n)
                tile = _map_variant_to_colors(raw_sprite, variant, frame_color, bg_color)
                base_region = canvas[tile_top : tile_top + n, tile_left : tile_left + n]
                write_mask = (tile != 0) & (tile != 6) & (tile != base_region)
                assert not np.any(written[tile_top : tile_top + n, tile_left : tile_left + n] & write_mask)
                written[tile_top : tile_top + n, tile_left : tile_left + n][write_mask] = True
                output[tile_top : tile_top + n, tile_left : tile_left + n][write_mask] = tile[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg, col_start: int, sep_col: int, marker_r0: int, is_right: bool):
    """Draw one legend panel (separator, marker square, rotation patch, raw sprite, colored frame)."""
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

    frame_h, frame_w = leg["frame_H"], leg["frame_W"]
    avail = marker_r0 - (n + 1)
    gap = random.randint(1, max(1, avail - frame_h))
    fr_start = n + 1 + gap
    f_col = random.randint(0, max(0, 4 - frame_w))
    frame_region = grid[fr_start : fr_start + frame_h, mc0 + f_col : mc0 + f_col + frame_w]
    mask = leg["frame"] != 0
    frame_region[mask] = leg["frame"][mask]


def _gen_attempt(num_legends: int | None = None) -> np.ndarray | None:
    if num_legends is None:
        num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    n = random.randint(*UNIT_SIZE_RANGE)
    legends = [{"N": n, "legend_w": n + 2} for _ in range(num_legends)]

    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    grid_h = random.randint(*GRID_H_RANGE)
    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    band_dir = random.choice(["h", "v"])
    band_cells = random.choice([1, 2])
    band_width = band_cells * n

    bg1, bg2 = _choose_bg_colors()
    legend_colors = _allocate_legend_colors(num_legends, (bg1, bg2))

    for i, leg in enumerate(legends):
        unit_colors, frame_colors = legend_colors[i]
        if not _design_legend(leg, unit_colors, frame_colors, marker_r0, grid_h, canvas_w, band_dir, band_cells):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w, band_dir):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2, band_dir, band_width)
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
    target_num_legends = random.choices([1, 2], weights=LEGEND_COUNT_WEIGHTS)[0]
    primary_retries = max(1, GEN_RETRIES * 3 // 4)
    for _ in range(primary_retries):
        result = _gen_attempt(target_num_legends)
        if result is not None:
            return result
    alt_num_legends = 1 if target_num_legends == 2 else 2
    for _ in range(GEN_RETRIES - primary_retries):
        result = _gen_attempt(alt_num_legends)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
