# Variant of a32d8b75
# Dimension: D17 — 颜色分配约束 (meta)
# Change: 单元精灵颜色与画布背景颜色必须不相交，框架颜色与单元精灵颜色不相交... → 允许精灵颜色与背景颜色重叠
#
# Design analysis:
# spec:
#   concretization:
#     - 仍然是“图例读精灵 + 读框架 + 读锚点 + 读旋转，再把复合精灵放到画布上”的任务。
#     - D17 的唯一核心变化放在颜色关系：输出用到的单元精灵颜色里，至少有一种会被画布背景复用。
#     - 因此某些复合精灵像素虽然逻辑上被放置了，但因为颜色与底下背景一致，视觉上不会改变该格。
#     - 输入图例中的单元精灵仍然完整可见；被“吞掉”的只是输出覆盖后的可见性，不是对象结构本身。
#   behavior_trace:
#     - 原逻辑例子：若单元颜色为 {2, 8}，背景条纹为 {3, 5}，那么复合精灵覆盖后，每个非 0/6 像素都会显式改写背景。
#     - 变体例子：单元颜色为 {2, 8}，背景条纹为 {8, 5}。当复合精灵中的 8 落到 8 条纹上时，该像素仍属于精灵，但输出看起来不变；2 落到 8 或 5 上时仍然可见。
#     - 差异发生在“overlay”阶段，不发生在图例解析、框架旋转或锚点计算阶段。
#   complexity_calibration:
#     - 只放宽“精灵色可与背景色重合”，不改 marker、rotation、frame、anchor 的语法，避免把 D17 混成别的维度。
#     - 为了让隐藏效果肉眼可见，生成器强制至少一个图例实际出现若干个“精灵色 == 背景色”的覆盖像素，而不是仅仅允许这种情况。
#     - 保留每个 legend 的 frame_color 与其 unit colors 不相交，避免 frame 和 unit 在图例读取区混淆；这不是本变体的目标，不应顺手放宽。
#   implicit_assumptions_review:
#     - `_extract_unit_sprite` 只在图例列里读顶部 4 连通块，不依赖背景配色；D17 不影响。
#     - `_extract_frame_bbox` 依赖“unit 与 frame 之间有空隙，frame 位于 marker 之上”；生成器继续保证该布局。
#     - `main()` 原本直接把复合精灵像素写到画布；在 D17 中这仍然正确，因为“同色覆盖后视觉不变”与“赋值为同一个值”等价。
#     - `composite_written` 不能因为像素看不见就放过重叠检查；隐藏像素仍然属于该复合精灵，所以生成器仍需保证 legend 之间不重叠。
#   placement_legality:
#     - 继续沿用 output-first：先确定 legend 参数、旋转后尺寸和 anchor，再挑选背景条纹，使复合精灵完全落在画布内且不同 legend 的 bbox 不相交。
#     - 生成器在构造背景后统计每个 legend 真正发生了多少“同色隐藏像素”；若任何 legend 在本变体要求下完全没触发隐藏，则重采样背景或整题。
#   visual_observability:
#     - 输出中仍能从重复铺贴的精灵局部和剩余可见颜色看出完整复合结构；隐藏部分来自背景同色，不靠颜色 id 排序或不可见元信息。
#     - 通过让背景条纹使用 2 色、且至少一色来自精灵颜色集，可以在 2-4 组样例里稳定看到“有些块缺口状消失，但图例精灵并未缺失”的现象。
#   visual_cleanliness:
#     - 不增加额外标记层；画面仍然只有图例、分隔线、条纹画布和最终复合精灵。
#     - 通过要求隐藏像素分散到多行多列，避免只在一个角落消失，保证现象显著但不致于整块糊掉。
#   color_inference:
#     - 背景始终只用两种颜色；单个 legend 只让其中一种背景色与该 legend 的精灵色重合，使“哪一类像素会被背景吞掉”可从结构上归纳。
#     - 两 legend 情况下，两种背景色分别对应两个 legend 的隐藏色，因此仍然是少量、重复支撑的类别，而不是稀疏的一次性映射。

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


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
    drawn with non-{0,6} pixels. Use the furthest pixel from the center (1,1)
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


def _compose_tiled_sprite(unit_sprite: np.ndarray, frame_mask: np.ndarray) -> np.ndarray:
    """Tile the unit sprite into every active frame cell."""
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild enlarged sprites from the legend panels and paint them onto the canvas.

    Input structure:
    - Full-height color-6 separator columns split the input into one or two legend panels plus a central canvas.
    - Each legend panel stacks four cues from top to bottom: a small two-color unit sprite, a frame mask, a 5x5 bordered anchor marker, and a 3x3 rotation marker.

    Rule:
    - Read the unit sprite, color-swap it back, tile it into every active frame cell, rotate the resulting composite, and place it using the anchor marker.
    - A placed sprite pixel still counts even when its color equals the pre-existing canvas color underneath, so some parts can visually blend into the background instead of appearing as a visible overwrite.

    Key insight:
    - The legend fully determines the hidden structure: background-color overlaps only affect which already-determined sprite pixels remain visible after placement, not where the sprite exists.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    base_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = base_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(base_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(base_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = base_canvas.copy()
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
        comp = _compose_tiled_sprite(unit_sprite, frame_mask)

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
                if val == 0 or val == 6:
                    continue
                assert not composite_written[rr, cc], "composites must not overlap"
                composite_written[rr, cc] = True
                # Same-color writes are intentional in this variant: the sprite still occupies
                # the cell even when the visible canvas value does not change.
                out[rr, cc] = val
    return out


# --------------- generator (D17 camouflage overlap) ---------------

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
CANVAS_RETRIES = 120
MIN_HIDDEN_PIXELS = 6


def _allocate_colors(
    num_legends: int,
) -> tuple[list[tuple[int, int]], tuple[int, int], list[int], list[int]] | None:
    """
    Pick unit colors, two canvas colors, and one frame color per legend.

    The D17 change is enforced here: at least one background color is taken from the
    output sprite palette, so hidden sprite pixels are possible.
    """
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets: list[tuple[int, int]] = []
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))

    hidden_colors = [random.choice(uc) for uc in unit_color_sets]
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
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors, hidden_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    """Return a copy of sprite with the two colors swapped."""
    sa, sb = sorted([c1, c2])
    out = sprite.copy()
    out[sprite == sa] = sb
    out[sprite == sb] = sa
    return out


def _make_unit_sprite(N: int, c1: int, c2: int, hidden_color: int) -> np.ndarray:
    """
    Generate a full NxN two-color sprite.

    Hidden-color pixels must be spread out so the camouflage effect is visible in the output.
    """
    min_hidden = max(2, N)
    for _ in range(120):
        sprite = np.full((N, N), c1, dtype=int)
        num_c2 = random.randint(1, N * N - 1)
        for p in random.sample(range(N * N), num_c2):
            sprite[p // N, p % N] = c2
        if np.count_nonzero(sprite == c1) < 2 or np.count_nonzero(sprite == c2) < 2:
            continue
        pts = np.argwhere(sprite == hidden_color)
        if pts.shape[0] < min_hidden:
            continue
        if len(np.unique(pts[:, 0])) < min(2, N):
            continue
        if len(np.unique(pts[:, 1])) < min(2, N):
            continue
        return sprite

    other = c2 if hidden_color == c1 else c1
    rr, cc = np.indices((N, N))
    sprite = np.where((rr + cc) % 2 == 0, hidden_color, other).astype(int)
    if np.all(sprite == hidden_color) or np.all(sprite == other):
        sprite[0, 0] = hidden_color
        sprite[-1, -1] = other
    return sprite


def _design_legend(
    leg: dict,
    unit_colors: tuple[int, int],
    frame_color: int,
    hidden_color: int,
    marker_r0: int,
    grid_h: int,
    canvas_w: int,
) -> bool:
    """Fill in sprite, frame, and rotation for one legend dict. Returns False on failure."""
    N = leg["N"]
    uc1, uc2 = unit_colors

    sprite = _make_unit_sprite(N, uc1, uc2, hidden_color)
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
        hidden_color=hidden_color,
    )
    return True


def _comp_size(leg: dict) -> tuple[int, int]:
    """Return (height, width) of the composite after rotation."""
    rf = leg["rotated_frame"]
    N = leg["N"]
    return rf.shape[0] * N, rf.shape[1] * N


def _build_composite(leg: dict) -> np.ndarray:
    comp = leg.get("composite")
    if comp is None:
        comp = _compose_tiled_sprite(leg["sprite"], leg["rotated_frame"])
        leg["composite"] = comp
    return comp


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


def _build_banded_canvas(
    grid_h: int, canvas_w: int, bg1: int, bg2: int, band_dir: str, band_width: int, offset: int
) -> np.ndarray:
    """Build a canvas with alternating horizontal or vertical color bands."""
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
    return best if best is not None and best_score[0] > 0 and best_score[1] >= MIN_HIDDEN_PIXELS else None


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    """Stamp each legend's tiled composite onto the canvas."""
    output = canvas.copy()
    for leg in legends:
        comp = _build_composite(leg)
        ch, cw = comp.shape
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + ch, l : l + cw]
        write_mask = (comp != 0) & (comp != 6) & (region != 6)
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
    if min(hidden_counts) <= 0 or sum(hidden_counts) < MIN_HIDDEN_PIXELS:
        return None

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


def _camouflage_counts_from_input(grid: np.ndarray) -> list[int]:
    """
    Test helper: count, per legend, how many placed composite pixels are hidden because
    they equal the pre-overlay canvas color.
    """
    g = np.asarray(grid, dtype=int)
    _h, w = g.shape
    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    canvas = g[:, canvas_c0:canvas_c1].copy()

    eff_h, eff_w = canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    counts: list[int] = []
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
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp = _compose_tiled_sprite(unit_sprite, frame_mask)
        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])
        ch, cw = comp.shape
        region = canvas[top : top + ch, left : left + cw]
        hidden = (comp != 0) & (comp != 6) & (comp == region)
        counts.append(int(hidden.sum()))
    return counts


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
