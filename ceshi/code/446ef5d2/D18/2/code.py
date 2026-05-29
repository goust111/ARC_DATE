# Variant of 446ef5d2
# Dimension: D18 — 碎片切割方式 (meta)
# Change: 矩形被切割成网格状（行列划分后可合并相邻格），所有碎片都是矩形且恰好拼成完整矩形... → 仅水平或仅垂直切割

from __future__ import annotations

from collections import Counter
import math
import random

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D18 variant
# ---------------------------------------------------------------------------
#
# spec_variant:
#   preserved_core:
#     - bg 仍是全局最多色
#     - 3 像素 L marker 仍然指示输出大矩形的一个角
#     - rect_col 仍是所有碎片中的主色，并继续充当边缘匹配时的通配色
#     - 碎片仍然只做平移，不旋转
#   d18_change:
#     from: 先做行列网格切分，再可合并成任意矩形块，因此需要二维拼装
#     to: 只沿一个方向切割成条带状碎片；每个样例要么全是水平条带，要么全是垂直条带
#   concrete_pixel_behavior:
#     - 水平条带时：所有碎片宽度相同，等于最终矩形宽度；每块左右边都落在最终外框上，因此左右边全是 rect_col
#     - 垂直条带时：所有碎片高度相同，等于最终矩形高度；每块上下边都落在最终外框上，因此上下边全是 rect_col
#     - 排序信息只存在于相邻条带接触的那一对边上：
#       水平条带匹配 bottom(row) -> top(row)，垂直条带匹配 right(col) -> left(col)
#   miniature_trace:
#     original:
#       - 4 块矩形可能是 2x3、3x2、2x2、3x3，solver 需要猜 HxW，再做二维回溯填满空白格
#     variant:
#       - 4 块若是水平条带，宽度全为 9，高度分别 2/3/2/4
#       - solver 无需再枚举因子对和二维放置，只需验证“公共宽度=9，H=11”，然后求一条 top->bottom 顺序链
#     key_difference:
#       - 原题的难点是二维定位；变体把难点收缩成一维排序，但每块边界更长、匹配证据更丰富
#   complexity_calibration:
#     - 若所有条带厚度都相同，则可能弱化“单向切割”这个视觉特征，也可能让另一方向产生伪候选
#     - 因此生成器强制至少两种不同厚度，避免“所有块完全同形”
#     - 若内部图案与切割方向平行，边界匹配会过弱；因此改为生成与切割方向垂直的连续色带，使每条内部切缝都有可见匹配
#   original_main_assumptions_audited:
#     - factor-pair 枚举 total_area -> (H,W): 依赖“最终可由任意矩形块平铺”，条带变体中不再需要
#     - next_empty + can_place 的二维回溯: 依赖“每块可能占据任意位置”，条带变体中应收缩为序列排序
#     - neighbor_ok 四方向检查: 原题必要；变体只需检查一个序列方向上的相邻边
#     - anchor piece 预放置逻辑: 原代码里几乎是无效分支，因为散点放置时 marker 周围会留空；本变体直接不依赖它
#   placement_legality:
#     - 条带数 4-8，厚度 2-4，公共长边 8-18，使“长条”外观明显且能在 10-26 网格内散开
#     - marker 位置仍按四角约束选取，保证 assembled rectangle 完整落在网格中
#     - 散落时继续要求至少 1 格 bg 间距，避免碎片粘连成更大块
#   visual_observability:
#     - 同一样例内全部碎片都呈同向长条，这是肉眼最先看到的结构差异
#     - 水平切割时内部图案是纵向色带；垂直切割时内部图案是横向色带，因此“匹配信息沿切缝传播”的机制也可见
#   cleanliness_and_color_inference:
#     - 颜色仍只承担“图案连续/不连续”的局部结构信息，不参与任何按数值排序
#     - interior colors 控制在 1-3 种；不同色带之间保留 rect_col 间隔，避免视觉糊在一起
#   new_solver_assumptions_to_match_generator:
#     - horizontal candidate 要求所有 piece 宽度相同，且每块左右边全是 rect_col
#     - vertical candidate 要求所有 piece 高度相同，且每块上下边全是 rect_col
#     - 序列必须存在唯一可行排序；生成器通过不同厚度 + 每道切缝都有非 rect_col 匹配来支持这一点


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = miss[0]
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(missing[0]), int(missing[1])), corner
    raise AssertionError("marker not found")


def _extract_rects(g: np.ndarray, bg: int, mark: int) -> list[np.ndarray]:
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    GH, GW = g.shape
    rects: list[np.ndarray] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append(g[rr : rr + h, cc : cc + w].copy())
    return rects


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def _order_strips(pieces: list[np.ndarray], rect_col: int, orientation: str) -> list[np.ndarray] | None:
    n = len(pieces)
    if n == 0:
        return None

    if orientation == "H":
        common = {int(p.shape[1]) for p in pieces}
        if len(common) != 1:
            return None
        for p in pieces:
            if not (np.all(p[:, 0] == int(rect_col)) and np.all(p[:, -1] == int(rect_col))):
                return None

        start_ok = [bool(np.all(p[0, :] == int(rect_col))) for p in pieces]
        end_ok = [bool(np.all(p[-1, :] == int(rect_col))) for p in pieces]
        nxt_ok = [[False] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i != j:
                    nxt_ok[i][j] = _compat(pieces[i][-1, :], pieces[j][0, :], rect_col)

        order: list[int] = []
        used = [False] * n
        solved = False

        def bt(last: int | None) -> None:
            nonlocal solved
            if solved:
                return
            if len(order) == n:
                if end_ok[order[-1]]:
                    solved = True
                return
            for i in range(n):
                if used[i]:
                    continue
                if last is None:
                    if not start_ok[i]:
                        continue
                else:
                    if not nxt_ok[last][i]:
                        continue
                used[i] = True
                order.append(i)
                bt(i)
                if solved:
                    return
                order.pop()
                used[i] = False

        bt(None)
        if not solved:
            return None
        return [pieces[i] for i in order]

    common = {int(p.shape[0]) for p in pieces}
    if len(common) != 1:
        return None
    for p in pieces:
        if not (np.all(p[0, :] == int(rect_col)) and np.all(p[-1, :] == int(rect_col))):
            return None

    start_ok = [bool(np.all(p[:, 0] == int(rect_col))) for p in pieces]
    end_ok = [bool(np.all(p[:, -1] == int(rect_col))) for p in pieces]
    nxt_ok = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                nxt_ok[i][j] = _compat(pieces[i][:, -1], pieces[j][:, 0], rect_col)

    order = []
    used = [False] * n
    solved = False

    def bt(last: int | None) -> None:
        nonlocal solved
        if solved:
            return
        if len(order) == n:
            if end_ok[order[-1]]:
                solved = True
            return
        for i in range(n):
            if used[i]:
                continue
            if last is None:
                if not start_ok[i]:
                    continue
            else:
                if not nxt_ok[last][i]:
                    continue
            used[i] = True
            order.append(i)
            bt(i)
            if solved:
                return
            order.pop()
            used[i] = False

    bt(None)
    if not solved:
        return None
    return [pieces[i] for i in order]


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, (mr, mc), corner = _find_marker(g, bg)
    pieces = _extract_rects(g, bg, mark)
    assert pieces

    allpix: list[int] = []
    for p in pieces:
        allpix.extend(map(int, p.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    candidates: list[tuple[str, np.ndarray]] = []

    ordered = _order_strips(pieces, rect_col, "H")
    if ordered is not None:
        W = int(ordered[0].shape[1])
        H = int(sum(int(p.shape[0]) for p in ordered))
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        if 0 <= tr and 0 <= tc and tr + H <= GH and tc + W <= GW:
            ass = np.vstack(ordered)
            candidates.append(("H", ass))

    ordered = _order_strips(pieces, rect_col, "V")
    if ordered is not None:
        H = int(ordered[0].shape[0])
        W = int(sum(int(p.shape[1]) for p in ordered))
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        if 0 <= tr and 0 <= tc and tr + H <= GH and tc + W <= GW:
            ass = np.hstack(ordered)
            candidates.append(("V", ass))

    assert candidates

    # Generator is designed to make the orientation unique, but if both survive,
    # keep the one with fewer ordering degrees of freedom by preferring the first
    # valid orientation after deterministic H-first then V-first checks.
    _orientation, ass = candidates[0]
    H, W = map(int, ass.shape)
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)

    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def _partition_total(total: int, n: int, lo: int, hi: int) -> list[int] | None:
    vals = []
    rem = total
    for i in range(n):
        left = n - i - 1
        cur_lo = max(lo, rem - left * hi)
        cur_hi = min(hi, rem - left * lo)
        if cur_lo > cur_hi:
            return None
        vals.append(random.randint(cur_lo, cur_hi))
        rem -= vals[-1]
    if rem != 0:
        return None
    return vals


def _make_band_fill(iH: int, iW: int, orientation: str, rect_col: int, interior_cols: list[int]) -> np.ndarray | None:
    interior = np.full((iH, iW), int(rect_col), dtype=int)
    axis = iW if orientation == "H" else iH
    n_cols = len(interior_cols)
    if axis < 3:
        return None

    min_cov = max(2 * n_cols, math.ceil(0.35 * axis))
    max_cov = min(axis - (n_cols - 1), math.floor(0.52 * axis))
    if min_cov > max_cov:
        return None
    total_cov = random.randint(min_cov, max_cov)
    widths = _partition_total(total_cov, n_cols, 2, min(5, axis))
    if widths is None:
        return None

    gap_total = axis - total_cov
    if gap_total < (n_cols - 1):
        return None
    inner_gap_base = [1] * max(0, n_cols - 1)
    rem = gap_total - sum(inner_gap_base)
    outer = [0] * (n_cols + 1)
    for _ in range(rem):
        outer[random.randint(0, n_cols)] += 1

    starts = []
    pos = outer[0]
    for i, w in enumerate(widths):
        starts.append(pos)
        pos += w
        if i < n_cols - 1:
            pos += 1 + outer[i + 1]

    if orientation == "H":
        for color, start, width in zip(interior_cols, starts, widths):
            for r in range(iH):
                shrink_l = 1 if width >= 3 and random.random() < 0.25 else 0
                shrink_r = 1 if width - shrink_l >= 3 and random.random() < 0.25 else 0
                c0 = start + shrink_l
                c1 = start + width - shrink_r
                interior[r, c0:c1] = int(color)
    else:
        for color, start, width in zip(interior_cols, starts, widths):
            for c in range(iW):
                shrink_t = 1 if width >= 3 and random.random() < 0.25 else 0
                shrink_b = 1 if width - shrink_t >= 3 and random.random() < 0.25 else 0
                r0 = start + shrink_t
                r1 = start + width - shrink_b
                interior[r0:r1, c] = int(color)

    return interior


def _marker_cells(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(12, 26)
    GW = random.randint(12, 26)

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]

    orientation = random.choice(["H", "V"])
    corner = random.choice(["TL", "TR", "BL", "BR"])
    n_pieces = random.randint(4, 7)

    segs = [random.randint(2, 4) for _ in range(n_pieces)]
    if len(set(segs)) == 1:
        k = random.randrange(n_pieces)
        segs[k] = 2 if segs[k] != 2 else 4

    if orientation == "H":
        RH = sum(segs)
        if RH > GH - 2:
            return None
        min_long = max(8, max(segs) + 4)
        max_long = min(18, GW - 2)
        if min_long > max_long:
            return None
        RW = random.randint(min_long, max_long)
    else:
        RW = sum(segs)
        if RW > GW - 2:
            return None
        min_long = max(8, max(segs) + 4)
        max_long = min(18, GH - 2)
        if min_long > max_long:
            return None
        RH = random.randint(min_long, max_long)

    n_interior = random.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    if orientation == "H":
        if n_interior > RW - 3:
            return None
    else:
        if n_interior > RH - 3:
            return None

    if corner == "TL":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = 2, GW - RW
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = RW - 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = 2, GW - RW
    else:
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = RW - 1, GW - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    rect = np.full((RH, RW), int(rect_col), dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    interior = _make_band_fill(iH, iW, orientation, rect_col, interior_cols)
    if interior is None:
        return None
    rect[1:-1, 1:-1] = interior

    pieces: list[np.ndarray] = []
    if orientation == "H":
        cuts = [0]
        s = 0
        for h in segs:
            s += h
            cuts.append(s)
        for r0, r1 in zip(cuts[:-1], cuts[1:]):
            pieces.append(rect[r0:r1, :].copy())
        for s in cuts[1:-1]:
            top = rect[s - 1, :]
            bot = rect[s, :]
            if not np.any((top != rect_col) & (bot != rect_col)):
                return None
        for p in pieces:
            if not (np.all(p[:, 0] == rect_col) and np.all(p[:, -1] == rect_col)):
                return None
    else:
        cuts = [0]
        s = 0
        for w in segs:
            s += w
            cuts.append(s)
        for c0, c1 in zip(cuts[:-1], cuts[1:]):
            pieces.append(rect[:, c0:c1].copy())
        for s in cuts[1:-1]:
            left = rect[:, s - 1]
            right = rect[:, s]
            if not np.any((left != rect_col) & (right != rect_col)):
                return None
        for p in pieces:
            if not (np.all(p[0, :] == rect_col) and np.all(p[-1, :] == rect_col)):
                return None

    for p in pieces:
        ph, pw = map(int, p.shape)
        fill = float(np.mean(p != rect_col))
        if fill <= 0.10 or fill >= 0.62:
            return None
        if not np.any(p != rect_col):
            return None
        has_non_rc_edge = any(
            np.any(edge != rect_col) for edge in [p[0, :], p[-1, :], p[:, 0], p[:, -1]]
        )
        if not has_non_rc_edge:
            return None

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for r, c in _marker_cells(mr, mc, corner):
        if r < 0 or r >= GH or c < 0 or c >= GW:
            return None
        canvas[r, c] = int(marker_col)

    order = list(range(len(pieces)))
    random.shuffle(order)
    for idx in order:
        pa = pieces[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    expected = np.full((GH, GW), int(bg), dtype=int)
    expected[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        x = _gen_attempt()
        if x is not None:
            return x
    raise RuntimeError("generate_input: failed after 300 attempts")
