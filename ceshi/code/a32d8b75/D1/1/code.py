# Variant of a32d8b75
# Dimension: D1 — 匹配条件：精灵组装方式 (match_cond)
# Change: s1（单元精灵）替换 d1（框架模板）中每个活跃像素，形成分形式的放大复合精灵... → 颜色匹配铺贴：框架中不同颜色的活跃像素对应不同的单元精灵变体

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label

# Design analysis
# a) 具体化与行为追踪
#    - 保留原题的四段式 legend: 顶部是 NxN 单元精灵，中段是框架，底部是 5x5 锚点框和 3x3 旋转标记。
#    - 关键改动只发生在“框架 -> 复合精灵”的组装环节。原题把 frame 当成布尔 mask，
#      只要某个 frame cell 非零，就铺同一个放大单元块。现在 frame 的活跃像素必须带两种颜色，
#      这两种颜色都来自单位精灵本身；每种颜色对应单位精灵的一个双色版本。
#    - 具体小例子：若单位精灵（solver 读出后）左上角颜色是 1，交换色后的版本左上角是 8，
#      那么 frame 中颜色 1 的活跃格铺“原版本”，颜色 8 的活跃格铺“交换版本”。
#      原逻辑在这一步会把两类格都铺成同一个版本；变体逻辑会在同一复合体内混用两个版本。
# b) 复杂度校准
#    - 只使用两种 frame 活跃颜色，对应单位精灵的两个颜色交换版本，避免自由度过多。
#    - 生成器强制两种 frame 颜色都出现，而且每种出现不止一次，避免退化为原题的单色框架。
#    - 单位精灵仍是完整填充的 2 色方阵，保证“交换颜色”在视觉上稳定可见。
# c) 隐式假设审查（与 main 直接相关）
#    - `_extract_unit_sprite` 仍依赖“最上方非 {0,6} 连通块就是单位精灵”；生成器保留 sprite/frame/marker 的纵向间隔。
#    - 旧代码把 `frame != 0 and frame != 6` 视为布尔 mask；新代码不能再丢掉 frame 颜色，必须旋转整个彩色 frame。
#    - 新增 `_variant_lookup` 依赖“单位精灵恰有两种非 {0,6} 颜色，且首个有效像素颜色可区分两个版本”；生成器保证 NxN 全填充且双色。
#    - 旧代码对每个活跃 frame cell 都写同一个 `unit_sprite`；新代码必须按 cell 颜色取不同 variant，否则会悄悄退化。
# d) 放置合法性
#    - 复合体尺寸与原题一致，仍按旋转后的 frame 外接矩形乘以单位精灵尺寸计算。
#    - 双 legend 情况继续用 bbox 不重叠约束，保证两个复合体不越界、不互相覆盖。
#    - generator 先构造输出，再反推输入，并用 `main(input) == output` 验证。
# e) 视觉可观察性
#    - frame 两色做成成片区域，而不是随机散点。解题者能从输入里的双色框架和输出里的双色 tile 区块看出对应关系。
#    - 不使用颜色编号排序、黑色参与排序等不可见规则；颜色选择完全来自局部可见对象。
# f) 可视化整洁度
#    - 单位精灵颜色不使用 4 和 7，避免与锚点色、旋转标记色混淆。
#    - 背景色仍与单位精灵颜色集合分离，输出里复合精灵边界清晰。
# g) 颜色变化可推断性
#    - 每个 legend 只涉及两种单位颜色与两种 frame 颜色，且二者是同一组颜色。
#    - 这使得“frame 颜色选择哪种单位精灵配色”在 2-3 个样例里可重复观察，而不是无提示的大量换色。
# h) main() 的新假设与 generator 对应保证
#    - `frame` 的所有活跃颜色都必须能在 `_variant_lookup` 中找到；generator 仅使用单位精灵的两种颜色给 frame 着色。
#    - `_variant_lookup` 默认用单位精灵首个有效像素的颜色作为 variant key；generator 生成的单位精灵无 0 空洞，因此该 key 稳定可见。


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


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    assert len(cols) == 2, "unit sprite must use exactly two colors"
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
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
    # Keep the original task's read-time swap so the variant only changes D1.
    return _swap_two_colors(spr), (r0, c0, r1, c1)


def _extract_frame_bbox(legend: np.ndarray, unit_bb: tuple[int, int, int, int], border6_r0: int) -> tuple[int, int, int, int] | None:
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


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
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


def _first_active_color(sprite: np.ndarray) -> int:
    vals = [int(x) for x in sprite.flat if int(x) not in (0, 6)]
    assert vals, "sprite must contain active pixels"
    return vals[0]


def _variant_lookup(unit_sprite: np.ndarray) -> dict[int, np.ndarray]:
    variant_a = unit_sprite
    variant_b = _swap_two_colors(unit_sprite)
    lookup = {
        _first_active_color(variant_a): variant_a,
        _first_active_color(variant_b): variant_b,
    }
    assert len(lookup) == 2, "unit sprite variants must be distinguishable by color"
    return lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged sprites on a canvas using color-coded legend panels.

    Input structure:
    - Color-6 separator columns still split the left/right legend panels from the central canvas.
    - Each legend panel still contains a unit sprite, a frame template, a 5x5 bordered anchor square, and a 3x3 rotation clock-hand.
    - Unlike the original task, the frame is multi-colored: its active cells use the unit sprite's two colors to mark which sprite coloring belongs in each block.

    Rule:
    - Read the unit sprite, derive its two colorings (the extracted sprite and its color-swapped counterpart), and rotate the full colored frame by the clock-hand direction.
    - For each active frame cell, choose the sprite coloring whose reference color matches that frame-cell color, then tile that variant into the corresponding enlarged block.
    - Place the finished composite on the canvas using the anchor encoded by the position marker in the bordered square.

    Key insight: The frame is no longer just an occupancy mask; its colors are local instructions that decide which color-swapped version of the same unit sprite gets stamped at each frame position.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
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
        variants = _variant_lookup(unit_sprite)
        unit_mask = (unit_sprite != 0) & (unit_sprite != 6)
        uh, uw = unit_sprite.shape

        frame_bb = _extract_frame_bbox(panel, unit_bb, marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=np.ones((3, 3)))
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_rot = np.rot90(frame, (-dir_k) % 4)
        active_colors = {int(x) for x in np.unique(frame_rot) if int(x) not in (0, 6)}
        assert active_colors <= set(variants), "frame color must select one of the sprite variants"

        fh, fw = frame_rot.shape
        comp = np.zeros((fh * uh, fw * uw), dtype=int)
        for i in range(fh):
            for j in range(fw):
                color = int(frame_rot[i, j])
                if color in (0, 6):
                    continue
                tile = variants[color]
                rr0 = i * uh
                cc0 = j * uw
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][unit_mask] = tile[unit_mask]

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


# --------------- generator (color_match_fill_variants) ---------------

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
UNIT_COLOR_POOL = [1, 2, 3, 5, 8, 9]
BG_COLOR_POOL = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
FRAME_ACTIVE_RANGE = (8, 18)
MIN_FRAME_COLOR_COUNT = 2
GEN_RETRIES = 400


def _allocate_colors(num_legends: int):
    """Pick 2 unit colors per legend and 2 canvas background colors."""
    pool = list(UNIT_COLOR_POOL)
    random.shuffle(pool)
    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

    bg_cands = [c for c in BG_COLOR_POOL if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    return unit_color_sets, (bg_cands[0], bg_cands[1])


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _make_unit_sprite(N: int, c1: int, c2: int) -> np.ndarray:
    sprite = np.full((N, N), c1, dtype=int)
    num_c2 = random.randint(max(1, N), N * N - 1)
    picks = random.sample(range(N * N), num_c2)
    for p in picks:
        sprite[p // N, p % N] = c2
    if np.all(sprite == c1) or np.all(sprite == c2):
        sprite[0, 0] = c1
        sprite[-1, -1] = c2
    return sprite


def _generate_frame_mask(frame_H: int, frame_W: int) -> np.ndarray | None:
    for _ in range(80):
        frame_mask = random_sprite(
            frame_H,
            frame_W,
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


def _colorize_frame(frame_mask: np.ndarray, sprite: np.ndarray) -> np.ndarray | None:
    key_a = _first_active_color(sprite)
    key_b = _first_active_color(_swap_two_colors(sprite))
    colors = [key_a, key_b]
    pts = np.argwhere(frame_mask)
    if len(pts) < 2 * MIN_FRAME_COLOR_COUNT:
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
        c0 = int(cand.sum())
        c1 = int((~cand).sum())
        if c0 >= MIN_FRAME_COLOR_COUNT and c1 >= MIN_FRAME_COLOR_COUNT:
            split_mask = cand
            break
    if split_mask is None:
        order = np.argsort(pts[:, 0] * 100 + pts[:, 1])
        split_mask = np.zeros(len(pts), dtype=bool)
        split_mask[order[: len(pts) // 2]] = True
        c0 = int(split_mask.sum())
        c1 = int((~split_mask).sum())
        if c0 < MIN_FRAME_COLOR_COUNT or c1 < MIN_FRAME_COLOR_COUNT:
            return None

    frame = np.zeros(frame_mask.shape, dtype=int)
    first_color, second_color = colors if random.random() < 0.5 else colors[::-1]
    frame[tuple(pts[split_mask].T)] = first_color
    frame[tuple(pts[~split_mask].T)] = second_color
    return frame


def _design_legend(leg, unit_colors, marker_r0, grid_h, canvas_w):
    """Fill in sprite, colored frame, anchor, and rotation for one legend dict."""
    N = leg["N"]
    uc1, uc2 = unit_colors
    sprite = _make_unit_sprite(N, uc1, uc2)
    legend_sprite = _color_swap(sprite, uc1, uc2)

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

    random.shuffle(combos)
    for frame_H, frame_W, rot_k in combos:
        frame_mask = _generate_frame_mask(frame_H, frame_W)
        if frame_mask is None:
            continue
        frame = _colorize_frame(frame_mask, sprite)
        if frame is None:
            continue
        rot_frame = np.rot90(frame, (-rot_k) % 4)
        leg.update(
            sprite=sprite,
            legend_sprite=legend_sprite,
            frame=frame,
            frame_H=frame_H,
            frame_W=frame_W,
            rot_k=rot_k,
            rotated_frame=rot_frame,
        )
        return True
    return False


def _comp_size(leg):
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _assign_anchors(legends, grid_h, canvas_w):
    """Assign anchor positions to legends, avoiding overlap for 2-legend case."""
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


def _overlay_composites(canvas: np.ndarray, legends) -> np.ndarray:
    """Stamp each legend's color-matched composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        rf = leg["rotated_frame"]
        variants = _variant_lookup(leg["sprite"])
        N = leg["N"]
        t, l = leg["comp_top"], leg["comp_left"]
        fh, fw = rf.shape

        comp = np.zeros((fh * N, fw * N), dtype=int)
        unit_mask = (leg["sprite"] != 0) & (leg["sprite"] != 6)
        for i in range(fh):
            for j in range(fw):
                color = int(rf[i, j])
                if color == 0:
                    continue
                tile = variants[color]
                r0, c0 = i * N, j * N
                comp[r0 : r0 + N, c0 : c0 + N][unit_mask] = tile[unit_mask]

        region = output[t : t + fh * N, l : l + fw * N]
        write_mask = (comp != 0) & (region != 6)
        region[write_mask] = comp[write_mask]
    return output


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
    unit_color_sets, (bg1, bg2) = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], marker_r0, grid_h, canvas_w):
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


def _place_legend(grid: np.ndarray, leg, col_start: int, sep_col: int, marker_r0: int, is_right: bool):
    """Draw one legend panel (separator, marker square, rotation patch, sprite, colored frame)."""
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
    mask = leg["frame"] != 0
    frame_region[mask] = leg["frame"][mask]


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
