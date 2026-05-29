# Combo variant of a6f40cea (index 39)
# combo_spec: D1/1+D4/2+D12/0+D14/2
# Name: 竞争障碍绕行
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 保留原题骨架：输入里仍然只有一个闭合矩形框，框外仍是接触框环带的开放矩形组件，
#      输出仍然只裁剪框内区域。
#    - 融合后的变化全部发生在“框内怎么补路径”：
#      1. 框内出现 1-2 个小障碍块，颜色与主填充色不同，并且它们会保留到输出里。
#      2. 每个外部 L 组件先被还原成一个“想走的”框内最短避障路径。
#      3. 所有这些候选路径不是一次性画完，而是按边界可见顺序逐轮同步前进一步。
#      4. 若某条路径的下一格已被障碍或更早画出的路径占据，则该路径永久停止。
#    - 一个具体小例子：
#      原题里 tl 组件会把 top 触点与 left 触点直接连成一个 L；br 组件也会直接连成一个 L。
#      现在先在 tl 路径的竖段上放一个 2-3 格小障碍，tl 的最短路会绕开障碍，多出一次横向闪避；
#      与此同时 br 的最短路仍朝交叉区域前进。逐轮扩展时，tl 因为从 top 边更早进入，
#      会先占住共享格，br 到达该格前一轮后便被截断。于是输出里同时出现：
#      “看得见的绕障弯折” + “看得见的竞争停止前缀”。
#
# b) 复杂度校准
#    - 没有把规则简化成单条射线或单一障碍。每题固定 2 个主要 L 组件，且至少有 1 个组件真的绕障，
#      至少有 1 个组件真的被另一条路径截断。这样变化不是装饰性的。
#    - 障碍只允许 1-2 个小 4 连通 blob，尺寸 1-4 格，延续 D12 的“小干扰块”视觉词汇，
#      避免像 D1 单变体那样引入大面积墙体，导致题面语言偏离原题过远。
#    - 颜色规则保持克制：路径仍是单色或双色交替；障碍统一使用一种专门颜色，避免颜色语义膨胀。
#
# c) 隐式假设审查
#    - 原 helper 假设框内完全同色，这在障碍进入内部后失效，所以检测逻辑改为
#      “边框完整 + 主填充色占多数 + 少量内部非填充像素不碰内边界”。
#    - 原 solver 假设每个组件对应一条固定 L 路径；现在必须先做最短路搜索，再做逐轮竞争模拟。
#    - 原 solver 一次性整条路径上色；现在每条路径只在每轮前进 1 格，并可能中途永久停止。
#    - 原双色相位只由端点颜色与整条路径长度决定；在融合后仍先按完整候选最短路确定相位，
#      然后把这个相位用于被截断后的前缀。生成器显式保证外部 alternating 轨迹和完整候选路的奇偶兼容。
#    - 原 generator 用 `used_inner` 避免任何路径重叠；融合后反而必须允许“静态规划路径彼此相交”，
#      否则竞争停止永远不可观察。
#
# d) 放置合法性
#    - 外部组件仍使用原题的开放矩形语言，并通过组件掩码保证彼此不重叠、也不与闭合框 4 邻接冲突。
#    - 内部障碍只允许出现在严格内层，不覆盖端点，不越界，不互相重叠。
#    - 生成器先选两条具有原始交叉趋势的 L 组件，再搜索小障碍 blob；
#      只有当“最短避障路存在 + 至少一条真的绕障 + 同步模拟里至少一条真的被截断”时才接受。
#
# e) 视觉可观察性
#    - 人类在 2-4 个样例里能同时看到两类证据：
#      1. 输入内已有小色块，输出路径会绕开它们。
#      2. 两条路径并不是都补完，一条会在靠近另一条时停止。
#    - 顺序只依赖可见的边界进入位置：top 从左到右、right 从上到下、bottom 从右到左、left 从下到上。
#      不使用颜色 id、组件编号等人类不可见因素排序。
#
# f) 可视化整洁度
#    - 只生成 2 条主路径，避免多路径竞争把框内挤成杂乱网格。
#    - 障碍保持小而稀疏，且通常放在原始 L 路的内部段上；这能让“绕障”很明显，但不会遮蔽整体结构。
#    - 外部仍保留原题规整的开放矩形痕迹，因此输入到输出的对应关系仍然好读。
#
# g) 变化的可推断性
#    - 路径颜色规则基本不变：单色为实线，双色按格交替。
#    - 新增的“为什么停下”并不靠颜色暗示，而靠几何关系：路径停止的位置总是紧贴另一条已占住的路径，
#      这比引入额外颜色编码更容易从少量样例归纳。
#
# Main() 新隐式假设与 generator 保障
#    - 假设每个生成样本都只有相邻两边触点的 L 组件；generator 只生成这种外部组件。
#    - 假设框内非填充像素都是障碍而不是预绘路径；generator 不在框内预绘任何路径，只放障碍。
#    - 假设双色组件的外部 alternating 轨迹与完整候选最短路奇偶一致；generator 在放置前做兼容性检查。
#    - 假设至少有一条候选路径的最短路与原始一拐角 L 不同，且同步模拟确有阻塞；generator 用指标过滤。

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_with_sparse_obstacles(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px frame whose interior is mostly one fill color plus a few
    small obstacle blobs fully inside the crop.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, fill_count, r0, c0, r1, c1, bc, fill)
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
            fill, fill_count = Counter(map(int, inner.flatten())).most_common(1)[0]
            fill = int(fill)
            fill_count = int(fill_count)
            if fill == int(bc):
                continue
            if fill_count * 2 <= int(inner.size):
                continue

            diff = inner != int(fill)
            if diff.any():
                if diff[0, :].any() or diff[-1, :].any() or diff[:, 0].any() or diff[:, -1].any():
                    continue
                dlab, dn = label(diff, structure=structure)
                if int(dn) > 2:
                    continue
                ok = True
                for did in range(1, int(dn) + 1):
                    mask = dlab == int(did)
                    if int(mask.sum()) > 4:
                        ok = False
                        break
                    cols = set(map(int, inner[mask].flatten()))
                    if len(cols) != 1 or int(fill) in cols or int(bc) in cols:
                        ok = False
                        break
                if not ok:
                    continue

            area = int(h * w)
            cand = (area, fill_count, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    _area, _fill_count, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _line(a: int, b: int) -> range:
    step = 1 if int(b) >= int(a) else -1
    return range(int(a), int(b) + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for r in _line(int(r0), int(r1))]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(int(c0), int(c1))]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            q = (int(p[0]), int(p[1]))
            if not out or out[-1] != q:
                out.append(q)
    return out


def _build_manhattan(
    p1: tuple[int, int], p2: tuple[int, int], prefer_vertical_first: bool
) -> list[tuple[int, int]]:
    ra, ca = int(p1[0]), int(p1[1])
    rb, cb = int(p2[0]), int(p2[1])
    if ra == rb:
        return [(int(ra), int(c)) for c in _line(int(ca), int(cb))]
    if ca == cb:
        return [(int(r), int(ca)) for r in _line(int(ra), int(rb))]
    if prefer_vertical_first:
        return _join(_vpath(int(ra), int(rb), int(ca)), _hpath(int(rb), int(ca), int(cb)))
    return _join(_hpath(int(ra), int(ca), int(cb)), _vpath(int(ra), int(rb), int(cb)))


def _inside_corner(
    p1: tuple[int, int], p2: tuple[int, int], prefer_vertical_first: bool, out_h: int, out_w: int
) -> bool:
    ra, ca = map(int, p1)
    rb, cb = map(int, p2)
    if ra == rb or ca == cb:
        return True
    rr, cc = (int(rb), int(ca)) if prefer_vertical_first else (int(ra), int(cb))
    return 0 < int(rr) < int(out_h) - 1 and 0 < int(cc) < int(out_w) - 1


def _preferred_vertical_first(
    p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int
) -> bool:
    if _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w):
        return False
    return True


def _ordered_dirs(
    pos: tuple[int, int], end: tuple[int, int], prefer_vertical_first: bool
) -> list[tuple[int, int]]:
    r, c = map(int, pos)
    er, ec = map(int, end)
    dr = 1 if er > r else -1 if er < r else 0
    dc = 1 if ec > c else -1 if ec < c else 0
    ordered: list[tuple[int, int]] = []
    if prefer_vertical_first:
        if dr != 0:
            ordered.append((int(dr), 0))
        if dc != 0:
            ordered.append((0, int(dc)))
    else:
        if dc != 0:
            ordered.append((0, int(dc)))
        if dr != 0:
            ordered.append((int(dr), 0))
    for step in [(-1, 0), (0, -1), (0, 1), (1, 0)]:
        if step not in ordered:
            ordered.append(step)
    return ordered


def _shortest_path(
    start: tuple[int, int],
    end: tuple[int, int],
    blocked: np.ndarray,
    *,
    prefer_vertical_first: bool,
) -> list[tuple[int, int]] | None:
    h, w = blocked.shape
    start = (int(start[0]), int(start[1]))
    end = (int(end[0]), int(end[1]))
    if blocked[start] or blocked[end]:
        return None

    q: deque[tuple[int, int]] = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while q:
        cur = q.popleft()
        if cur == end:
            break
        for dr, dc in _ordered_dirs(cur, end, prefer_vertical_first):
            nr = int(cur[0] + dr)
            nc = int(cur[1] + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            nxt = (int(nr), int(nc))
            if blocked[nxt] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if end not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = end
    while cur is not None:
        path.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    path.reverse()
    return path


def _count_bends(path: list[tuple[int, int]]) -> int:
    bends = 0
    for i in range(1, len(path) - 1):
        dr1 = int(path[i][0] - path[i - 1][0])
        dc1 = int(path[i][1] - path[i - 1][1])
        dr2 = int(path[i + 1][0] - path[i][0])
        dc2 = int(path[i + 1][1] - path[i][1])
        if (dr1, dc1) != (dr2, dc2):
            bends += 1
    return int(bends)


def _boundary_key(p: tuple[int, int], out_h: int, out_w: int) -> int:
    r, c = map(int, p)
    if r == 0:
        return int(c)
    if c == out_w - 1:
        return int((out_w - 1) + r)
    if r == out_h - 1:
        return int((out_w - 1) + (out_h - 1) + (out_w - 1 - c))
    assert c == 0
    return int(2 * (out_w - 1) + (out_h - 1) + (out_h - 1 - r))


def _orient_path_by_boundary(
    path: list[tuple[int, int]], out_h: int, out_w: int
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int], int]:
    a = (int(path[0][0]), int(path[0][1]))
    b = (int(path[-1][0]), int(path[-1][1]))
    ka = _boundary_key(a, out_h, out_w)
    kb = _boundary_key(b, out_h, out_w)
    if kb < ka:
        rev = list(reversed(path))
        return rev, rev[0], rev[-1], int(kb)
    return path, a, b, int(ka)


def _color_at(colors: tuple[int, ...], idx: int) -> int:
    if len(colors) == 1:
        return int(colors[0])
    a, b = map(int, colors)
    return int(a if idx % 2 == 0 else b)


class PaintCandidate:
    def __init__(
        self,
        path: list[tuple[int, int]],
        colors: tuple[int, ...],
        start_key: int,
        tie_key: int,
        tag: str,
    ) -> None:
        self.path = [(int(r), int(c)) for r, c in path]
        self.colors = tuple(map(int, colors))
        self.start_key = int(start_key)
        self.tie_key = int(tie_key)
        self.tag = str(tag)


def _simulate_growth(
    out_h: int,
    out_w: int,
    fill: int,
    obstacle_mask: np.ndarray,
    candidates: list[PaintCandidate],
) -> tuple[np.ndarray, dict[str, object]]:
    out = np.full((out_h, out_w), int(fill), dtype=int)
    out[obstacle_mask] = -1
    order = sorted(range(len(candidates)), key=lambda i: (candidates[i].start_key, candidates[i].tie_key))
    idxs = [0] * len(candidates)
    stopped = [False] * len(candidates)
    stop_reason: list[str | None] = [None] * len(candidates)
    rounds = 0
    painted = 0

    while True:
        advanced = False
        rounds += 1
        for i in order:
            cand = candidates[i]
            if stopped[i] or idxs[i] >= len(cand.path):
                continue
            rr, cc = cand.path[idxs[i]]
            if int(out[int(rr), int(cc)]) != int(fill):
                stopped[i] = True
                stop_reason[i] = "obstacle" if obstacle_mask[int(rr), int(cc)] else "path"
                continue
            out[int(rr), int(cc)] = _color_at(cand.colors, idxs[i])
            idxs[i] += 1
            painted += 1
            advanced = True
        if not advanced:
            break

    finished = sum(1 for i, cand in enumerate(candidates) if idxs[i] == len(cand.path))
    blocked = sum(1 for i, cand in enumerate(candidates) if idxs[i] < len(cand.path))
    return out, {
        "order": order,
        "idxs": idxs,
        "stop_reason": stop_reason,
        "rounds": int(rounds - 1),
        "painted": int(painted),
        "planned": int(sum(len(c.path) for c in candidates)),
        "finished": int(finished),
        "blocked": int(blocked),
        "blocked_on_path": int(sum(1 for reason in stop_reason if reason == "path")),
    }


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start_color: int | None = None,
) -> None:
    if not path:
        return
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    cur = int(a if start_color is None else start_color)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = int(b) if int(cur) == int(a) else int(a)
    rs, cs = zip(*path)
    grid[rs, cs] = vals


def _l_endpoints(kind: str, elbow_r: int, elbow_c: int, out_h: int, out_w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, int(elbow_c)), (int(elbow_r), 0)
    if kind == "tr":
        return (0, int(elbow_c)), (int(elbow_r), int(out_w - 1))
    if kind == "bl":
        return (int(out_h - 1), int(elbow_c)), (int(elbow_r), 0)
    return (int(out_h - 1), int(elbow_c)), (int(elbow_r), int(out_w - 1))


def _canonical_l_path(kind: str, elbow_r: int, elbow_c: int, out_h: int, out_w: int) -> list[tuple[int, int]]:
    if kind == "tl":
        return _join(_vpath(0, int(elbow_r), int(elbow_c)), _hpath(int(elbow_r), int(elbow_c), 0))
    if kind == "tr":
        return _join(_vpath(0, int(elbow_r), int(elbow_c)), _hpath(int(elbow_r), int(elbow_c), int(out_w - 1)))
    if kind == "bl":
        return _join(_vpath(int(out_h - 1), int(elbow_r), int(elbow_c)), _hpath(int(elbow_r), int(elbow_c), 0))
    return _join(
        _vpath(int(out_h - 1), int(elbow_r), int(elbow_c)),
        _hpath(int(elbow_r), int(elbow_c), int(out_w - 1)),
    )


def _outer_l_path(
    kind: str,
    elbow_r: int,
    elbow_c: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    offset_range: tuple[int, int],
) -> list[tuple[int, int]]:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    lo, hi = offset_range
    outer_r = int(r0 - random.randint(lo, hi)) if is_top else int(r1 + random.randint(lo, hi))
    outer_c = int(c0 - random.randint(lo, hi)) if is_left else int(c1 + random.randint(lo, hi))

    ring_a = (int(r0 - 1 if is_top else r1 + 1), int(c0 + 1 + elbow_c))
    ring_b = (int(r0 + 1 + elbow_r), int(c0 - 1 if is_left else c1 + 1))
    return _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )


def _choose_outer_coloring(
    outer: list[tuple[int, int]],
    colors: tuple[int, int],
    need_a: int,
    need_b: int,
) -> tuple[list[tuple[int, int]], int] | None:
    ring_a = (int(outer[0][0]), int(outer[0][1]))
    ring_b = (int(outer[-1][0]), int(outer[-1][1]))
    for cand_path in (outer, list(reversed(outer))):
        for start_color in colors:
            cur = int(start_color)
            first_col = None
            last_col = None
            for idx, p in enumerate(cand_path):
                if p == ring_a:
                    first_col = int(cur)
                if p == ring_b:
                    last_col = int(cur)
                if idx != len(cand_path) - 1:
                    cur = int(colors[1]) if int(cur) == int(colors[0]) else int(colors[0])
            if int(first_col) == int(need_a) and int(last_col) == int(need_b):
                return cand_path, int(start_color)
    return None


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    h, w = grid.shape
    s = {(int(r), int(c)) for r, c in coords}
    for rr, cc in s:
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if int(grid[int(rr), int(cc)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(rr), int(cc)):
            if (int(nr), int(nc)) not in s and 0 <= int(nr) < h and 0 <= int(nc) < w and comp_mask[int(nr), int(nc)]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    *,
    start_color: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, int(bg), comp_mask):
        return False
    _paint_path(grid, path, colors, start_color=start_color)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _map_to_out(
    rr: int, cc: int, r_int0: int, c_int0: int, out_h: int, out_w: int
) -> tuple[int, int]:
    mr = max(0, min(int(out_h - 1), int(rr - r_int0)))
    mc = max(0, min(int(out_w - 1), int(cc - c_int0)))
    return int(mr), int(mc)


def _extract_candidates_from_grid(
    g: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    bc: int,
    fill: int,
) -> tuple[np.ndarray, list[PaintCandidate]]:
    bg = int(get_bg(g))
    h, w = g.shape
    inner = np.array(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    out_h, out_w = inner.shape
    obstacle_mask = inner != int(fill)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    outside_mask = np.ones_like(g, dtype=bool)
    outside_mask[r0 + 1 : r1, c0 + 1 : c1] = False
    comp_mask = outside_mask & (g != int(bg)) & (g != int(fill)) & (g != int(bc))
    clab, cn = label(comp_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = _map_to_out(int(rr), int(cc), int(r_int0), int(c_int0), int(out_h), int(out_w))
        comp_pts[int(cid)].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols = {
        int(cid): sorted(set(map(int, g[clab == int(cid)].flatten())))
        for cid in range(1, int(cn) + 1)
    }

    candidates: list[PaintCandidate] = []
    tie_key = 0
    for cid, pts in sorted(comp_pts.items()):
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
        coords = list(hist.keys())

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(int(coords[i][0]) - int(coords[j][0])) + abs(int(coords[i][1]) - int(coords[j][1]))
                if best is None or d > best[0]:
                    best = (int(d), coords[i], coords[j])
        if best is None:
            continue
        _d, p1, p2 = best
        prefer_vertical_first = _preferred_vertical_first(p1, p2, int(out_h), int(out_w))
        path = _shortest_path(p1, p2, obstacle_mask, prefer_vertical_first=prefer_vertical_first)
        assert path is not None
        path, start_pt, end_pt, start_key = _orient_path_by_boundary(path, int(out_h), int(out_w))

        if len(cols) == 1:
            candidates.append(
                PaintCandidate(
                    path=path,
                    colors=(int(cols[0]),),
                    start_key=int(start_key),
                    tie_key=int(tie_key),
                    tag=f"mono-{cid}",
                )
            )
            tie_key += 1
            continue

        if len(cols) == 2:
            ca, cb = map(int, cols)
            hs = int(hist[start_pt].most_common(1)[0][0])
            he = int(hist[end_pt].most_common(1)[0][0])
            chosen_colors = None
            for start_color in (int(ca), int(cb)):
                other = int(cb) if int(start_color) == int(ca) else int(ca)
                end_color = int(start_color) if (len(path) - 1) % 2 == 0 else int(other)
                if int(start_color) == int(hs) and int(end_color) == int(he):
                    chosen_colors = (int(start_color), int(other))
                    break
            assert chosen_colors is not None
            candidates.append(
                PaintCandidate(
                    path=path,
                    colors=tuple(chosen_colors),
                    start_key=int(start_key),
                    tie_key=int(tie_key),
                    tag=f"alt-{cid}",
                )
            )
            tie_key += 1
            continue

    candidates.sort(key=lambda cand: (cand.start_key, cand.tie_key))
    return obstacle_mask, candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover obstacle-avoiding interior routes, then let them compete cell by cell inside the frame.

    Input structure:
    - One 1px closed rectangular frame encloses an interior whose dominant color is the base fill.
    - Inside that frame there may be 1-2 small colored obstacle blobs that are not on the inner border.
    - Outside the frame sit colored open-rectangle components touching the ring around the border; they still encode endpoint positions and mono/alternating color patterns.

    Rule:
    - Output only the cropped frame interior, preserving every internal obstacle cell.
    - For each outside component, project its touch points onto the frame interior boundary and reconstruct the shortest 4-connected path between them that avoids the obstacle cells.
    - Order those candidate paths by the visible clockwise position of their earlier boundary touch point.
    - Starting from the obstacle-preserving interior, advance all candidates synchronously one cell per round; if a candidate's next cell is already occupied by an obstacle or an earlier-painted path, that candidate stops permanently.
    - Monochrome candidates stay solid; two-color candidates alternate along the full intended path, with the same phase used on the surviving prefix.

    Key insight:
    - The outside traces still tell you which border points belong together, but the final painted extent is no longer just geometry: first compute each path with static obstacle avoidance, then simulate competitive growth so earlier entrants claim shared cells and truncate later paths.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    found = _find_framed_rect_with_sparse_obstacles(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    inner = np.array(g[r0 + 1 : r1, c0 + 1 : c1], dtype=int)
    out_h, out_w = inner.shape

    obstacle_mask, candidates = _extract_candidates_from_grid(g, int(r0), int(c0), int(r1), int(c1), int(bc), int(fill))
    painted, _metrics = _simulate_growth(int(out_h), int(out_w), int(fill), obstacle_mask, candidates)
    out = inner.copy()
    write_mask = (painted != int(fill)) & (~obstacle_mask)
    out[write_mask] = painted[write_mask]
    return out


# --------------- generator ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (7, 13)
_INTERIOR_W_RANGE = (8, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_GEN_RETRIES = 500
_OBSTACLE_COUNT_OPTIONS = [1, 1, 2, 2]
_OBSTACLE_PLAN_TRIES = 120
_OUTER_PLACE_TRIES = 40
_OBSTACLE_SHAPES = [
    [(0, 0)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (0, -1)],
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (-1, 0), (0, 1)],
    [(0, 0), (1, 0), (0, -1)],
    [(0, 0), (-1, 0), (0, -1)],
    [(0, 0), (1, 0), (-1, 0)],
    [(0, 0), (0, 1), (0, -1)],
    [(0, 0), (1, 0), (0, 1), (1, 1)],
]


def _core_pair_elbows(out_h: int, out_w: int) -> list[tuple[str, int, int]] | None:
    for _ in range(80):
        elbow_tl_r = random.randint(max(2, out_h // 2), out_h - 2)
        elbow_tl_c = random.randint(2, max(2, out_w - 4))
        elbow_br_r = random.randint(1, min(out_h - 2, elbow_tl_r - 1))
        elbow_br_c = random.randint(1, min(out_w - 2, elbow_tl_c - 1))
        if int(elbow_br_r + 1) > int(out_w - elbow_tl_c):
            continue
        return [("tl", int(elbow_tl_r), int(elbow_tl_c)), ("br", int(elbow_br_r), int(elbow_br_c))]
    return None


def _path_strict_inner_cells(path: list[tuple[int, int]], out_h: int, out_w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for rr, cc in path[1:-1]:
        if 0 < int(rr) < int(out_h - 1) and 0 < int(cc) < int(out_w - 1):
            out.append((int(rr), int(cc)))
    return out


def _blob_coords_from_shape(
    anchor: tuple[int, int], shape: list[tuple[int, int]], pivot: tuple[int, int]
) -> list[tuple[int, int]]:
    ar, ac = map(int, anchor)
    pr, pc = map(int, pivot)
    return [(int(ar + dr - pr), int(ac + dc - pc)) for dr, dc in shape]


def _task_from_desc(
    kind: str,
    elbow_r: int,
    elbow_c: int,
    out_h: int,
    out_w: int,
) -> dict[str, object]:
    p1, p2 = _l_endpoints(kind, int(elbow_r), int(elbow_c), int(out_h), int(out_w))
    canonical = _canonical_l_path(kind, int(elbow_r), int(elbow_c), int(out_h), int(out_w))
    canonical, start, end, start_key = _orient_path_by_boundary(canonical, int(out_h), int(out_w))
    return {
        "kind": str(kind),
        "elbow_r": int(elbow_r),
        "elbow_c": int(elbow_c),
        "p1": (int(p1[0]), int(p1[1])),
        "p2": (int(p2[0]), int(p2[1])),
        "canonical": canonical,
        "prefer_vertical_first": bool(_preferred_vertical_first(p1, p2, int(out_h), int(out_w))),
        "start_key": int(start_key),
        "canonical_start": start,
        "canonical_end": end,
    }


def _solve_tasks_with_obstacles(
    tasks: list[dict[str, object]],
    obstacle_mask: np.ndarray,
) -> list[dict[str, object]] | None:
    out_h, out_w = obstacle_mask.shape
    solved: list[dict[str, object]] = []
    for idx, task in enumerate(tasks):
        path = _shortest_path(
            task["p1"],
            task["p2"],
            obstacle_mask,
            prefer_vertical_first=bool(task["prefer_vertical_first"]),
        )
        if path is None:
            return None
        path, start, end, start_key = _orient_path_by_boundary(path, int(out_h), int(out_w))
        solved.append(
            {
                **task,
                "path": path,
                "path_start": start,
                "path_end": end,
                "start_key": int(start_key),
                "tie_key": int(idx),
            }
        )
    return solved


def _plan_obstacles_and_paths(
    tasks: list[dict[str, object]],
    out_h: int,
    out_w: int,
) -> tuple[np.ndarray, list[list[tuple[int, int]]], list[dict[str, object]], dict[str, object]] | None:
    endpoints = {tuple(map(int, task["p1"])) for task in tasks} | {tuple(map(int, task["p2"])) for task in tasks}
    want_count = random.choice(_OBSTACLE_COUNT_OPTIONS)

    base_candidate_cells = _path_strict_inner_cells(list(tasks[0]["canonical"]), int(out_h), int(out_w))
    extra_cells = _path_strict_inner_cells(list(tasks[1]["canonical"]), int(out_h), int(out_w))
    anchor_pool = list(dict.fromkeys(base_candidate_cells + extra_cells))
    if not anchor_pool:
        return None

    for _ in range(_OBSTACLE_PLAN_TRIES):
        blocked = np.zeros((out_h, out_w), dtype=bool)
        blobs: list[list[tuple[int, int]]] = []
        ok = True
        for _blob_idx in range(int(want_count)):
            placed = False
            cells = anchor_pool[:]
            random.shuffle(cells)
            shapes = list(_OBSTACLE_SHAPES)
            random.shuffle(shapes)
            for anchor in cells:
                for shape in shapes:
                    pivots = list(shape)
                    random.shuffle(pivots)
                    for pivot in pivots:
                        coords = _blob_coords_from_shape(anchor, shape, pivot)
                        unique = {(int(r), int(c)) for r, c in coords}
                        if len(unique) != len(coords):
                            continue
                        if any(not (0 < int(r) < int(out_h - 1) and 0 < int(c) < int(out_w - 1)) for r, c in coords):
                            continue
                        if any((int(r), int(c)) in endpoints for r, c in coords):
                            continue
                        if any(blocked[int(r), int(c)] for r, c in coords):
                            continue
                        test_blocked = blocked.copy()
                        for rr, cc in coords:
                            test_blocked[int(rr), int(cc)] = True
                        solved = _solve_tasks_with_obstacles(tasks, test_blocked)
                        if solved is None:
                            continue
                        blocked = test_blocked
                        blobs.append([(int(r), int(c)) for r, c in coords])
                        placed = True
                        break
                    if placed:
                        break
                if placed:
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        solved = _solve_tasks_with_obstacles(tasks, blocked)
        if solved is None:
            continue
        detoured = sum(1 for item in solved if list(item["path"]) != list(item["canonical"]))
        if detoured < 1:
            continue

        dummy_candidates = [
            PaintCandidate(
                path=list(item["path"]),
                colors=(idx + 1,),
                start_key=int(item["start_key"]),
                tie_key=int(item["tie_key"]),
                tag=str(item["kind"]),
            )
            for idx, item in enumerate(solved)
        ]
        _dummy, sim = _simulate_growth(int(out_h), int(out_w), 0, blocked, dummy_candidates)
        if int(sim["blocked_on_path"]) < 1:
            continue
        if int(sim["blocked"]) < 1 or int(sim["finished"]) < 1:
            continue
        if int(sum(blocked.flatten())) < 1:
            continue
        info = {
            "detoured_candidates": int(detoured),
            "blocked_candidates": int(sim["blocked"]),
            "blocked_on_path": int(sim["blocked_on_path"]),
            "obstacle_cells": int(sum(blocked.flatten())),
            "obstacle_blobs": int(len(blobs)),
            "planned_cells": int(sim["planned"]),
            "painted_cells": int(sim["painted"]),
            "rounds": int(sim["rounds"]),
            "path_lengths": [int(len(item["path"])) for item in solved],
            "bends": [int(_count_bends(list(item["path"]))) for item in solved],
        }
        return blocked, blobs, solved, info
    return None


def _gen_attempt() -> tuple[np.ndarray, dict[str, object]] | None:
    bg = int(random.choice(_BG_CHOICES))
    bc = int(_BORDER_COLOR)
    fill = int(bg) if random.random() < _FILL_EQ_BG_PROB else int(random.choice([x for x in range(10) if x not in (bg, bc)]))
    out_h = int(random.randint(*_INTERIOR_H_RANGE))
    out_w = int(random.randint(*_INTERIOR_W_RANGE))
    rows = int(random.randint(max(_GRID_SIZE_RANGE[0], out_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1]))
    cols = int(random.randint(max(_GRID_SIZE_RANGE[0], out_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1]))
    r0 = int(random.randint(_FRAME_MARGIN, rows - out_h - _FRAME_MARGIN_FAR))
    c0 = int(random.randint(_FRAME_MARGIN, cols - out_w - _FRAME_MARGIN_FAR))
    r1 = int(r0 + out_h + 1)
    c1 = int(c0 + out_w + 1)

    core = _core_pair_elbows(int(out_h), int(out_w))
    if core is None:
        return None
    tasks = [_task_from_desc(kind, rr, cc, int(out_h), int(out_w)) for kind, rr, cc in core]
    planned = _plan_obstacles_and_paths(tasks, int(out_h), int(out_w))
    if planned is None:
        return None
    obstacle_mask, obstacle_blobs, solved, behavior = planned

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 4:
        return None
    obstacle_color = int(palette[0])
    color_i = 1

    expected = np.full((out_h, out_w), int(fill), dtype=int)
    for blob in obstacle_blobs:
        for rr, cc in blob:
            grid[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(obstacle_color)
            expected[int(rr), int(cc)] = int(obstacle_color)

    comp_mask = np.zeros_like(grid, dtype=bool)
    candidates: list[PaintCandidate] = []

    for solved_item in solved:
        if color_i >= len(palette):
            return None
        outer_placed = False
        for _ in range(_OUTER_PLACE_TRIES):
            outer = _outer_l_path(
                str(solved_item["kind"]),
                int(solved_item["elbow_r"]),
                int(solved_item["elbow_c"]),
                int(r0),
                int(c0),
                int(r1),
                int(c1),
                _OUTER_OFFSET_RANGE,
            )
            if random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette):
                raw_colors = (int(palette[color_i]), int(palette[color_i + 1]))
                start_pt = tuple(map(int, solved_item["path_start"]))
                end_pt = tuple(map(int, solved_item["path_end"]))
                chosen = None
                for inner_start in raw_colors:
                    other = int(raw_colors[1]) if int(inner_start) == int(raw_colors[0]) else int(raw_colors[0])
                    inner_end = int(inner_start) if (len(solved_item["path"]) - 1) % 2 == 0 else int(other)
                    outer_choice = _choose_outer_coloring(outer, raw_colors, int(inner_start), int(inner_end))
                    if outer_choice is not None:
                        chosen = (tuple((int(inner_start), int(other))), outer_choice[0], int(outer_choice[1]))
                        break
                if chosen is None:
                    continue
                inner_colors, outer_path, outer_start = chosen
                if not _can_place(grid, outer_path, int(bg), comp_mask):
                    continue
                if not _place_component(grid, comp_mask, outer_path, raw_colors, int(bg), start_color=int(outer_start)):
                    return None
                candidates.append(
                    PaintCandidate(
                        path=list(solved_item["path"]),
                        colors=tuple(inner_colors),
                        start_key=int(solved_item["start_key"]),
                        tie_key=int(solved_item["tie_key"]),
                        tag=str(solved_item["kind"]),
                    )
                )
                color_i += 2
                outer_placed = True
                break
            else:
                col = int(palette[color_i])
                if not _can_place(grid, outer, int(bg), comp_mask):
                    continue
                if not _place_component(grid, comp_mask, outer, (int(col),), int(bg)):
                    return None
                candidates.append(
                    PaintCandidate(
                        path=list(solved_item["path"]),
                        colors=(int(col),),
                        start_key=int(solved_item["start_key"]),
                        tie_key=int(solved_item["tie_key"]),
                        tag=str(solved_item["kind"]),
                    )
                )
                color_i += 1
                outer_placed = True
                break
        if not outer_placed:
            return None

    grown, sim = _simulate_growth(int(out_h), int(out_w), int(fill), obstacle_mask, candidates)
    path_mask = (grown != int(fill)) & (~obstacle_mask)
    expected[path_mask] = grown[path_mask]
    if int(sim["blocked_on_path"]) < 1:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    info = {
        **behavior,
        "candidate_start_keys": [int(c.start_key) for c in candidates],
        "candidate_tags": [str(c.tag) for c in candidates],
        "stop_reason": list(sim["stop_reason"]),
        "progress": list(sim["idxs"]),
        "full_lengths": [int(len(c.path)) for c in candidates],
    }
    return grid, info


def generate_case() -> tuple[np.ndarray, dict[str, object]]:
    for _ in range(_GEN_RETRIES):
        built = _gen_attempt()
        if built is not None:
            return built
    raise RuntimeError("failed to generate a valid a6f40cea combo input")


def generate_input() -> np.ndarray:
    return generate_case()[0]
