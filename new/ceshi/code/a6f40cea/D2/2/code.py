# Variant of a6f40cea
# Dimension: D2 — 轨迹着色模式 (trajectory)
# Change: 单色组件用纯色填充路径；双色组件用两色交替填充，相位由端点颜色锚定... → 间歇跳跃轨迹

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for the D2 intermittent_skip variant
# ----------------------------------------------------
# a) 具体化与行为追踪
#    - 保持原题的对象语法：一个封闭主框，2-3 个贴着主框 ring 的开放矩形组件，外加可选干扰点。
#    - 变化只发生在“内部轨迹如何着色”。原题把推断出的曼哈顿骨架整段涂满；这里仍先恢复同一条骨架，
#      但只在索引 0, N, 2N, ... 的位置落笔，于是输出出现虚线。
#    - N 取自外部开放矩形离主框的外扩深度。这个深度在输入里肉眼可见：组件的最外侧角比 ring 多伸出多少格。
#      为了让人类更容易归纳，整张输入里的所有 ring-touching 组件共用同一个深度 N。
#    - 小例子：若某个 L 组件映射得到内部路径长度为 11，且外部开口框比主框外扩 3 格，
#      原题会涂满 11 格；变体只涂第 0/3/6/9/10 格，形成明显断续。
#
# b) 复杂度校准
#    - N 只允许少量离散值 {3, 5}，避免自由度过大，又足以和原题连续填充区分。
#    - 所有组件共享同一个 N，单个样例内会重复出现同一节奏，便于从 2-4 个样例归纳。
#    - 双色组件保留双色交替，只是只在抽样点上可见，从而不引入新的颜色映射维度。
#
# c) 隐式假设审查（与原 main 逐段对应）
#    - `_find_framed_rect_uniform_fill`、ring 提取、端点映射、曼哈顿骨架选择都依赖几何而非连续涂色，
#      因此仍然成立。
#    - 原来所有 `for rr, cc in chosen: out[...] = color` 都假设“整条骨架都应着色”；变体下必须改成
#      “先取骨架，再按 N 抽样”，否则会直接退化回原题。
#    - 原 deferred 仅以颜色为 key 聚合单色 U-pair；若未来存在同色不同 N，会错误合并。
#      这里改成 `(color, N)` 聚合，即使当前生成器让同一输入共享 N，也不把正确性押在这个巧合上。
#    - 原双色相位逻辑比较路径两端颜色。这里仍沿“完整骨架”的索引奇偶来定义颜色，
#      然后只在抽样点上落笔。这样端点颜色约束继续有效；生成器再额外保证 N 为奇数，
#      使被画出的点通常仍表现为显式交替。
#
# d) 放置合法性
#    - 生成器继续先构造期望输出，再反推外部开口框，并用 solver 回验。
#    - 所有组件共享固定外扩深度 N，因此开放矩形的最外角位置可直接由 frame 坐标 ±N 得出。
#    - 为了让虚线稳定可见，内部尺寸只取满足 `(H-1) % N == 0` 与 `(W-1) % N == 0` 的档位；
#      L 组件还要求其路径长度减一至少为 `2N`，避免只剩 1-2 个点。
#
# e) 视觉可观察性
#    - N 完全由可见几何深度决定，不依赖颜色 id、排序或隐藏计数。
#    - 因为同一输入内所有组件都共享 N，解题者能看到：外部开口框都离主框更远/更近，
#      而输出里所有内部轨迹也同步更稀疏/更密集。
#
# f) 可视化整洁度
#    - 路径骨架本身不变，只减少着色密度，因此不会新增重叠层次。
#    - 继续禁止内部骨架互相相交，保持原题的清晰布局。
#
# g) 颜色变化可推断性
#    - 颜色集合与原题同级：单色组件仍单色，双色组件仍只用两色。
#    - 颜色变化没有新增条件分支；变化点仅是“哪些骨架位置被画出来”。
#
# 9) 新增假设及其生成保证
#    - solver 现在默认 ring-touching 组件的外扩深度可从组件 bbox 相对主框 bbox 直接读出；
#      生成器把所有开放矩形都做成规则的矩形折线，保证这个测量稳定。
#    - solver 的双色抽样继续依赖完整骨架端点颜色；生成器把 N 限定为奇数并挑选满足长度约束
#      的内部路径，保证端点锚定与可见交替都成立。


def _find_framed_rect_uniform_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with uniform interior fill.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest such rectangle, else None.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None
    for bc in sorted(set(map(int, g.flatten())) - {int(bg)}):
        lab, n = label(g == int(bc), structure=structure)
        for cid in range(1, int(n) + 1):
            pts = np.argwhere(lab == int(cid))
            if int(len(pts)) < 8:
                continue

            r0, c0 = pts.min(axis=0)
            r1, c1 = pts.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h < 3 or w < 3:
                continue

            per = int(2 * (h + w) - 4)
            if int(len(pts)) != per:
                continue
            if not (g[r0, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r1, c0 : c1 + 1] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c0] == int(bc)).all():
                continue
            if not (g[r0 : r1 + 1, c1] == int(bc)).all():
                continue

            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size == 0:
                continue
            fill = int(Counter(inner.flatten()).most_common(1)[0][0])
            if fill == int(bc):
                continue
            if not (inner == int(fill)).all():
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _sparse_indices(length: int, skip: int) -> list[int]:
    assert int(skip) >= 1
    if int(length) <= 0:
        return []
    idxs = list(range(0, int(length), int(skip)))
    if idxs[-1] != int(length) - 1:
        idxs.append(int(length) - 1)
    return idxs


def _paint_sparse_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    skip: int,
) -> None:
    if not path:
        return
    idxs = _sparse_indices(len(path), int(skip))
    if len(colors) == 1:
        col = int(colors[0])
        for idx in idxs:
            rr, cc = path[int(idx)]
            grid[int(rr), int(cc)] = int(col)
        return

    a, b = map(int, colors)
    for idx in idxs:
        rr, cc = path[int(idx)]
        grid[int(rr), int(cc)] = int(a if int(idx) % 2 == 0 else b)


def _paint_outer_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = int(colors[0])
        return

    a, b = map(int, colors)
    vals = [int(a if i % 2 == 0 else b) for i in range(len(path))]
    rs, cs = zip(*path)
    grid[rs, cs] = vals


def _component_skip_from_bbox(
    bbox: tuple[int, int, int, int],
    frame: tuple[int, int, int, int],
) -> int:
    br0, bc0, br1, bc1 = map(int, bbox)
    r0, c0, r1, c1 = map(int, frame)
    depths: list[int] = []
    if int(br0) < int(r0):
        depths.append(int(r0 - br0))
    if int(br1) > int(r1):
        depths.append(int(br1 - r1))
    if int(bc0) < int(c0):
        depths.append(int(c0 - bc0))
    if int(bc1) > int(c1):
        depths.append(int(bc1 - c1))
    assert depths
    return int(max(depths))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct sparse Manhattan traces inside a framed rectangle from surrounding open frames.

    Input structure:
    - A background field containing one 1px closed rectangle with a uniform fill.
    - Colored open-rectangle components sit outside the frame and touch its ring on one or two sides.

    Rule:
    - Output the frame interior on the fill color.
    - Each ring-touching component still determines the same Manhattan stripe or L-shaped skeleton as in the base task.
    - Do not fill that skeleton continuously: paint only every Nth position along it, where N is the component's outward depth beyond the closed frame.
    - Monochrome components keep one color; two-color components keep their endpoint-anchored parity while only the sampled positions are drawn.

    Key insight:
    - The visible gap/depth of the external open frame encodes the dash period, so geometry tells you how sparse the internal trace should be.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = np.full((outH, outW), int(fill), dtype=int)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb2) = p1, p2
        if ra == rb:
            step = 1 if cb2 >= ca else -1
            return [(int(ra), int(c)) for c in range(int(ca), int(cb2) + step, step)]
        if ca == cb2:
            step = 1 if rb >= ra else -1
            return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        if cornerA:
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb2 >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb2) + step, step)]
            return path
        step = 1 if cb2 >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb2) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb2)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb2) = p1, p2
        if ra == rb or ca == cb2:
            return True
        corner = (rb, ca) if cornerA else (ra, cb2)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    comp_mask = (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < H:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < W:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((mr, mc, col))

    comp_cols: dict[int, list[int]] = {}
    comp_skip: dict[int, int] = {}
    for cid in range(1, int(cn) + 1):
        mask = clab == int(cid)
        if not mask.any():
            continue
        comp_cols[cid] = sorted(set(map(int, g[mask].flatten())))
        pts = np.argwhere(mask)
        bbox = (
            int(pts[:, 0].min()),
            int(pts[:, 1].min()),
            int(pts[:, 0].max()),
            int(pts[:, 1].max()),
        )
        comp_skip[cid] = _component_skip_from_bbox(bbox, (r0, c0, r1, c1))

    deferred: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue
        skip = int(comp_skip[cid])

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[(int(colv), int(skip))].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                    if best is None or d > best[0]:
                        best = (d, uniq[i], uniq[j])
            if best is None:
                continue
            _d, p1, p2 = best
            pathA = build_manhattan(p1, p2, True)
            pathB = build_manhattan(p1, p2, False)
            chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
            _paint_sparse_path(out, chosen, (int(colv),), int(skip))
            continue

        if len(cols) == 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, a, b = best
            ca, cb2 = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (ca, cb2):
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb2 if start_color == ca else ca)
                    if int(start_color) != int(ha):
                        continue
                    if int(end_color) != int(hb):
                        continue
                    chosen = (path_pts, int(start_color))
                    break
                if chosen is not None:
                    break

            assert chosen is not None
            path_pts, start_color = chosen
            other = int(cb2 if int(start_color) == int(ca) else ca)
            _paint_sparse_path(out, path_pts, (int(start_color), int(other)), int(skip))
            continue

    for (col, skip), pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        if any(p[0] == 0 for p in pts) and any(p[0] == outH - 1 for p in pts):
            for cc in sorted({c for _r, c in pts}):
                path = [(int(rr), int(cc)) for rr in range(0, int(outH))]
                _paint_sparse_path(out, path, (int(col),), int(skip))
            continue
        if any(p[1] == 0 for p in pts) and any(p[1] == outW - 1 for p in pts):
            for rr in sorted({r for r, _c in pts}):
                path = [(int(rr), int(cc)) for cc in range(0, int(outW))]
                _paint_sparse_path(out, path, (int(col),), int(skip))
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if best is None or d > best[0]:
                    best = (d, pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        _paint_sparse_path(out, chosen, (int(col),), int(skip))

    return out


# --------------- generator (a6 intermittent_skip) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_SKIP_VALUES = (3, 5)
_INTERIOR_H_OPTIONS = {3: [4, 7, 10, 13], 5: [6, 11]}
_INTERIOR_W_OPTIONS = {3: [7, 10, 13], 5: [11]}
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 6
_FRAME_MARGIN_FAR = 8
_N_COMPONENTS_RANGE = (2, 3)
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_outer_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(
    kind: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    skip: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    row_lo, row_hi = ((1, max(1, H // 2)) if is_top else (max(1, H // 2), H - 2))
    col_lo, col_hi = ((1, max(1, W // 2)) if is_left else (max(1, W // 2), W - 2))
    if row_hi < row_lo or col_hi < col_lo:
        return None

    candidates: list[tuple[int, int]] = []
    for r in range(int(row_lo), int(row_hi) + 1):
        for c in range(int(col_lo), int(col_hi) + 1):
            drow = int(r if is_top else (H - 1 - r))
            dcol = int(c if is_left else (W - 1 - c))
            span = int(drow + dcol)
            if span < int(2 * skip):
                continue
            if span % int(skip) != 0:
                continue
            candidates.append((int(r), int(c)))
    if not candidates:
        return None

    r, c = random.choice(candidates)
    R = (r0 - int(skip)) if is_top else (r1 + int(skip))
    C = (c0 - int(skip)) if is_left else (c1 + int(skip))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _u_pair_specs(
    orientation: str,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    skip: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]], list[tuple[int, int]]] | None:
    if orientation == "vertical":
        if (H - 1) < int(2 * skip) or (H - 1) % int(skip) != 0:
            return None
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - int(skip)
        bot_r = r1 + int(skip)
        a_top, b_top = (r0 - 1, c0 + 1 + c_a), (r0 - 1, c0 + 1 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 1 + c_a), (r1 + 1, c0 + 1 + c_b)
        outer_a = _join(
            _vpath(a_top[0], top_r, a_top[1]),
            _hpath(top_r, a_top[1], b_top[1]),
            _vpath(top_r, b_top[0], b_top[1]),
        )
        outer_b = _join(
            _vpath(a_bot[0], bot_r, a_bot[1]),
            _hpath(bot_r, a_bot[1], b_bot[1]),
            _vpath(bot_r, b_bot[0], b_bot[1]),
        )
        inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
        return inner, outer_a, outer_b

    if (W - 1) < int(2 * skip) or (W - 1) % int(skip) != 0:
        return None
    r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
    if r_b - r_a < 2:
        r_b = min(H - 2, r_a + 2)
    left_c = c0 - int(skip)
    right_c = c1 + int(skip)
    a_left, b_left = (r0 + 1 + r_a, c0 - 1), (r0 + 1 + r_b, c0 - 1)
    a_right, b_right = (r0 + 1 + r_a, c1 + 1), (r0 + 1 + r_b, c1 + 1)
    outer_a = _join(
        _hpath(a_left[0], a_left[1], left_c),
        _vpath(a_left[0], b_left[0], left_c),
        _hpath(b_left[0], left_c, b_left[1]),
    )
    outer_b = _join(
        _hpath(a_right[0], a_right[1], right_c),
        _vpath(a_right[0], b_right[0], right_c),
        _hpath(b_right[0], right_c, b_right[1]),
    )
    inner = [[(r_a, c) for c in range(W)], [(r_b, c) for c in range(W)]]
    return inner, outer_a, outer_b


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (nr, nc) not in used
            ]
            if not opts:
                break
            p = random.choice(opts)
            used.add(p)
            coords.append(p)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    skip = int(random.choice(_SKIP_VALUES))
    H = int(random.choice(_INTERIOR_H_OPTIONS[skip]))
    W = int(random.choice(_INTERIOR_W_OPTIONS[skip]))
    min_rows = max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD, H + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    min_cols = max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD, W + _FRAME_MARGIN + _FRAME_MARGIN_FAR)
    rows = random.randint(int(min_rows), _GRID_SIZE_RANGE[1])
    cols = random.randint(int(min_cols), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation_choices = []
        if (H - 1) >= 2 * skip and (H - 1) % skip == 0:
            orientation_choices.append("vertical")
        if (W - 1) >= 2 * skip and (W - 1) % skip == 0:
            orientation_choices.append("horizontal")
        if orientation_choices:
            orientation = random.choice(orientation_choices)
            color = int(palette[color_i])
            color_i += 1
            packed = _u_pair_specs(orientation, H, W, r0, c0, r1, c1, skip)
            if packed is None:
                return None
            inner_lines, outer_a, outer_b = packed
            all_inner = [p for line in inner_lines for p in line]
            irs, ics = zip(*all_inner)
            if used_inner[irs, ics].any():
                return None
            if not _place_component(grid, comp_mask, outer_a, (int(color),), bg):
                return None
            if not _place_component(grid, comp_mask, outer_b, (int(color),), bg):
                return None
            for line in inner_lines:
                _paint_sparse_path(expected, line, (int(color),), skip)
            used_inner[irs, ics] = True
            remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        packed = _l_specs(kind, H, W, r0, c0, r1, c1, skip)
        if packed is None:
            return None
        inner, outer = packed
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_sparse_path(expected, inner, colors, skip)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), int(palette[color_i]))
        color_i += 1

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid intermittent-skip a6f40cea input")
