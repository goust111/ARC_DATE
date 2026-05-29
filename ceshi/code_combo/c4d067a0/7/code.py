# Combo variant of c4d067a0 (index 7)
# combo_spec: D3/0+D4/1
# Name: 框架填充+空心方框放大

from __future__ import annotations

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


# Design analysis
# a) 具体化与行为追踪:
#    原题把 legend 单像素直接放大成右侧的实心 NxN 色块；D3 单变体把“落点”改成已有外框的洞；
#    本融合版再叠加 D4，把“洞里的填充形状”从实心块改成空心方框。
#    一个具体追踪:
#    - 原题: legend[(1,2)] = 红 -> 右侧对应位置整块 3x3 都涂红。
#    - D3: 同一格点先对应一个中性 5x5 外框，solver 把其 3x3 洞整块涂红。
#    - 本融合: 仍先定位到那个 5x5 外框，但洞内只画一个 3x3 红色空心方框，
#      即洞边缘 8 个格子变红，中心 1 格保持背景色。
#    输入中所有外框都已存在，只有两个相邻锚点外框的洞里已经出现彩色空心方框，其余洞为空。
#    输出补齐所有 legend 对应位置的彩色空心方框，外层中性框始终不变。
# b) 复杂度校准:
#    如果让洞尺寸太小，空心与实心的区别不可见；如果洞尺寸太大，23-27 的画布上不易容纳 3-4 行阶梯阵列。
#    因此这里固定洞尺寸为 3、外框尺寸为 5、外框间隔为 1。这样每个彩色目标都稳定呈现为“8 格边框 + 1 格中心背景”，
#    与原题、D3 单变体都有清晰视觉差异，同时仍保留原题的“通过两个锚点反推整体偏移”。
# c) 隐式假设审查:
#    原 solver 假设主区域对象是“单色实心正方形”，融合后失效，必须改成先检测主区域的中性外框家族。
#    原 D3 solver 假设洞若非空就应是“整块同色填满”，这里也失效；新的洞状态只能是:
#    1) 全背景，表示未填；2) 单一颜色的 3x3 空心方框，表示已填锚点。
#    原代码最终可直接覆盖整块区域；这里若整块写色会退化回 D3，所以必须显式保留洞中心背景。
#    另外，step 不能再由已着色对象推导，因为输入只有两个彩色内框；必须继续依赖所有中性外框的位置求 gcd。
# d) 放置合法性:
#    generator 保证 legend 仍在左上 `(1,1)` 起、步长 2 的晶格上；主区域外框整体右移，`base_hole_c >= 8`，
#    因而与 legend 视觉分栏。
#    外框只放在 staircase 真实占位处，不在空缺格点上造框；外框尺寸固定 5、间隔固定 1，互不重叠且完整落在画布内。
# e) 视觉可观察性:
#    解题者肉眼能看到两层结构: 左上稀疏 legend；右侧是一组相同中性外框，其中只有两个洞里已有彩色“内层空心方框”。
#    输出时其余洞也出现同样的内层空心方框，中心继续留背景，因此“不是实心填满”这一规律可直接观察。
# f) 可视化整洁度:
#    整个主区域只含一种外框色，图例颜色绝不与外框色冲突；彩色内容只出现在外框洞内。
#    没有重叠对象、噪声像素或额外标记，区域功能分明。
# g) 变化的可推断性:
#    颜色没有发生映射，仍是 legend 原色；变化只在目标位置和几何形状上。
#    人类可从样例中同时读出两条结构线索:
#    1) 哪些外框位置对应 legend 占位，由两个相邻锚点对齐；
#    2) 每个占位生成的不是实心块，而是洞内同色空心方框。
# h) 新 solver 假设与生成保证:
#    - 所有任务相关外框共享同一外框色和外尺寸，且数量明显多于已预填的彩色内框；generator 固定如此构造。
#    - 每个非空洞都恰好是一个单色 3x3 空心方框；generator 不会产生实心洞、混色洞或半缺边。
#    - 两个预填锚点位于同一行的相邻 legend 列，颜色不同；`_assign_legend_colors()` 保证这对颜色顺序在 legend 中唯一可定位。
#    - 每个 legend 点都对应一个真实外框洞；因此 solver 末尾按投影坐标写入时必能命中现有 frame。


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: framed target area ===
HOLE_SIZE = 3
OUTER_SIZE = HOLE_SIZE + 2
FRAME_GAP = 1
FRAME_STEP = OUTER_SIZE + FRAME_GAP
BASE_HOLE_C_MIN = 8
BASE_HOLE_R_MIN = 2

# === CFG: generation ===
MAX_RETRIES = 400


def _positive_gcd(values: list[int]) -> int:
    vals = [int(v) for v in values if int(v) > 0]
    assert vals, "need at least one positive difference"
    return reduce(gcd, vals)


def _is_square_frame(sub: np.ndarray, color: int) -> bool:
    """Return True iff `sub` is a monochrome hollow square frame of `color`."""
    hh, ww = sub.shape
    if hh != ww or hh < 3:
        return False
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    return not np.any(sub[1:-1, 1:-1] == color)


def _is_colored_hollow(sub: np.ndarray, color: int, bg: int) -> bool:
    """Return True iff `sub` is a colored hollow square whose interior is pure bg."""
    if not _is_square_frame(sub, color):
        return False
    return bool(np.all(sub[1:-1, 1:-1] == bg))


def _draw_outer_frame(canvas: np.ndarray, fr0: int, fc0: int, color: int) -> None:
    canvas[fr0, fc0 : fc0 + OUTER_SIZE] = color
    canvas[fr0 + OUTER_SIZE - 1, fc0 : fc0 + OUTER_SIZE] = color
    canvas[fr0 : fr0 + OUTER_SIZE, fc0] = color
    canvas[fr0 : fr0 + OUTER_SIZE, fc0 + OUTER_SIZE - 1] = color


def _draw_inner_hollow(
    canvas: np.ndarray, hr0: int, hc0: int, color: int, bg: int
) -> None:
    canvas[hr0 : hr0 + HOLE_SIZE, hc0 : hc0 + HOLE_SIZE] = bg
    canvas[hr0, hc0 : hc0 + HOLE_SIZE] = color
    canvas[hr0 + HOLE_SIZE - 1, hc0 : hc0 + HOLE_SIZE] = color
    canvas[hr0 : hr0 + HOLE_SIZE, hc0] = color
    canvas[hr0 : hr0 + HOLE_SIZE, hc0 + HOLE_SIZE - 1] = color


def _enumerate_staircase_shapes() -> list[tuple[int, ...]]:
    shapes: list[tuple[int, ...]] = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix: list[int], remaining: int, r: int) -> None:
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase() -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(
    pts: list[tuple[int, int]], anchor_pair: tuple[int, int], palette: list[int]
) -> dict[tuple[int, int], int] | None:
    """Assign colors while keeping the ordered adjacent anchor pair uniquely identifiable."""
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
    colors: dict[tuple[int, int], int] = {first: a, second: b}
    counts[a] -= 1
    counts[b] -= 1
    if counts[a] < 0 or counts[b] < 0:
        return None

    anchor_colors = (a, b)
    cells = [p for p in pts if p not in colors]

    def allowed(pos: tuple[int, int], col: int) -> bool:
        r, c = pos
        left = (r, c - 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
        right = (r, c + 1)
        if right in pos_set and right in colors:
            if (col, colors[right]) == anchor_colors and (pos, right) != (first, second):
                return False
        return True

    def search() -> bool:
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a lattice of neutral outer frames using a legend of singleton colors in the top-left corner.

    Input structure:
    - The background is the most common color.
    - Top-left: isolated legend pixels on a regular lattice, still encoding the intended color pattern.
    - Main area: same-size hollow outer frames already occupy the target lattice; only two adjacent frames currently contain a colored 3x3 hollow square inside their holes.

    Rule:
    - Use the legend pixels as lattice coordinates and use the two pre-colored inner hollows to infer how those coordinates align with the frame lattice.
    - Every occupied legend position maps to one existing outer frame hole.
    - For each mapped frame, draw a same-color 3x3 hollow square inside the hole, leaving the hole center as background and leaving the outer frame unchanged.

    Key insight: The old scale-and-offset reasoning still drives the alignment, but the rendered object is now nested: a colored hollow square must be embedded inside a pre-existing neutral frame rather than painting a solid block.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))

    assert singles, "grid must contain legend pixels"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    stepL = _positive_gcd(
        [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
        + [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    )
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    frame_candidates: list[tuple[int, int, int, int]] = []
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == int(col), connectivity=4):
            r0, c0, r1, c1 = bbox_coords(pts)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if hh == ww and _is_square_frame(sub, int(col)):
                frame_candidates.append((int(r0), int(c0), int(hh), int(col)))

    assert frame_candidates, "grid must contain hollow square structures"

    size_vals, size_cnts = np.unique(
        np.array([size for _, _, size, _ in frame_candidates], dtype=int),
        return_counts=True,
    )
    outer_size = int(size_vals[size_cnts.argmax()])
    frame_candidates = [f for f in frame_candidates if f[2] == outer_size]

    col_vals, col_cnts = np.unique(
        np.array([col for _, _, _, col in frame_candidates], dtype=int),
        return_counts=True,
    )
    frame_color = int(col_vals[col_cnts.argmax()])
    frames = [f for f in frame_candidates if f[3] == frame_color]
    assert frames, "must detect a dominant family of neutral outer frames"

    hole_size = outer_size - 2
    assert hole_size == HOLE_SIZE, "this combo expects 3x3 frame holes"

    frame_by_hole: dict[tuple[int, int], tuple[int, int]] = {}
    filled_holes: list[tuple[int, int, int]] = []
    for fr0, fc0, _size, _col in frames:
        hr0, hc0 = fr0 + 1, fc0 + 1
        hole = g[hr0 : hr0 + hole_size, hc0 : hc0 + hole_size]
        frame_by_hole[(int(hr0), int(hc0))] = (int(fr0), int(fc0))
        if np.all(hole == bg):
            continue
        vals = sorted(set(map(int, np.unique(hole))) - {int(bg)})
        assert len(vals) == 1, "non-empty hole must use exactly one color"
        fill_col = int(vals[0])
        assert fill_col != frame_color, "inner hollow color must differ from outer frame color"
        assert _is_colored_hollow(hole, fill_col, int(bg)), "non-empty hole must contain a colored hollow square"
        filled_holes.append((int(hr0), int(hc0), fill_col))

    assert len(filled_holes) >= 2, "need at least two pre-colored inner hollows"

    hole_rows = sorted({r0 for r0, _ in frame_by_hole})
    hole_cols = sorted({c0 for _, c0 in frame_by_hole})
    step = _positive_gcd(
        [hole_rows[i + 1] - hole_rows[i] for i in range(len(hole_rows) - 1)]
        + [hole_cols[i + 1] - hole_cols[i] for i in range(len(hole_cols) - 1)]
    )

    by_color: dict[int, list[tuple[int, int]]] = {}
    for hr0, hc0, col in filled_holes:
        by_color.setdefault(int(col), []).append((int(hr0), int(hc0)))

    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for hr0, hc0 in by_color[col]:
            base_r = hr0 - rr * step
            base_c = hc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1

    assert bases, "pre-colored inner hollows must determine the legend-to-frame offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-frame offset must be unique"
    base_r, base_c = best_bases[0]

    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - hole_size and 0 <= tc0 <= w - hole_size, \
            "projected frame hole must fit in the canvas"
        assert (tr0, tc0) in frame_by_hole, "each legend point must map to an existing outer frame"
        _draw_inner_hollow(out, tr0, tc0, int(col), int(bg))

    return out


def _gen_attempt() -> np.ndarray | None:
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    frame_color = random.choice([c for c in range(NUM_COLORS) if c != bg])
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    palette_pool = [c for c in range(NUM_COLORS) if c not in (bg, frame_color)]
    if len(palette_pool) < k_min:
        return None
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts), len(palette_pool)))
    palette = random.sample(palette_pool, k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - HOLE_SIZE - 1 - max_r * FRAME_STEP
    c_hi = size - HOLE_SIZE - 1 - max_c * FRAME_STEP
    if r_hi < BASE_HOLE_R_MIN or c_hi < BASE_HOLE_C_MIN:
        return None
    base_r = random.randint(BASE_HOLE_R_MIN, r_hi)
    base_c = random.randint(BASE_HOLE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    for rr, cc in pts:
        hr, hc = base_r + rr * FRAME_STEP, base_c + cc * FRAME_STEP
        _draw_outer_frame(out, hr - 1, hc - 1, frame_color)
        _draw_inner_hollow(out, hr, hc, legend_colors[(rr, cc)], bg)

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        hr, hc = base_r + rr * FRAME_STEP, base_c + cc * FRAME_STEP
        inp[hr : hr + HOLE_SIZE, hc : hc + HOLE_SIZE] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
