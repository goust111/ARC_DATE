# Variant of a6f40cea
# Dimension: D1 — 路径形态 (path)
# Change: 两端点间的曼哈顿L形路径（一个拐角，由水平段+垂直段组成），拐角选择使其严格位于内部而非边缘... → 最短路径绕障

from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis:
# a) 具体化与行为追踪
#    仍然保留“外部开放矩形框痕迹 + 内部闭合框 + 只输出框内裁剪”的骨架；变化只发生在框内。
#    输入中的框内区域不再纯填充，而是在主填充色上加入少量障碍色块。外部连通块仍只提供两个触点和
#    单色/双色交替的颜色信息。输出保留这些障碍块，并在剩余可走单元上连出避障最短路。
#    小例子：原题某个 tl 组件把顶部触点 (0,5) 与左侧触点 (4,0) 连成经过 (4,5) 的 L。
#    现在若在 rows 1..3, cols 0..1 与 cols 3..5 放两块障碍，只留下 col=2 的竖直走廊，
#    那么路径会变成 top->col2->down->left，共 2 个拐点；旧逻辑在这里会直接撞上障碍。
# b) 复杂度校准
#    只增加一种新对象：框内障碍块，且同一题内统一为一种颜色，避免颜色规则膨胀。
#    每个 L 型组件都仍只有两个端点，变化集中在“走法”而不是“识别谁和谁相连”。
#    生成器强制至少一个路径拥有 >=2 个拐点，避免样本视觉上退化为原题的一拐角 L。
# c) 隐式假设审查
#    原 helper 要求框内完全同色；这在变体中失效，所以改成“边框完整 + 内部主色占多数”。
#    原 main 直接把输出初始化为 fill；现在必须先复制输入框内切片，保留障碍。
#    原 main 的 build_manhattan/inside_corner 假设一条固定 L 足够；现在改为 BFS 最短路。
#    原双色逻辑只依赖端点提示色与路径奇偶，这一点仍成立，但要额外保证外部开放框的奇偶与内部路长匹配。
# d) 放置合法性
#    生成器先为每个 L 组件选端点，再在对应角落制造“只留一条竖直走廊”的障碍布局；
#    用与 solver 相同的 BFS 验证路径存在、唯一、且不与已有路径/障碍重叠。
#    外部开放框仍通过原来的 ring-touch 方式放在框外，使用组件掩码避免互相接触。
# e) 视觉可观察性
#    解题者在 2-4 组样例里能稳定看到：外部端点位置相似，但内部会绕开明显的色块，
#    且路径拐点数增加。规则依赖空间阻挡，不依赖颜色 id 排序等不可见信息。
# f) 可视化整洁度
#    障碍只用一种颜色，且每个组件的障碍局限在它对应的角落扇区，不把中心区域堆满。
#    路径之间禁止相交；外部仍保留原题整洁的开放框语言，输入/输出层次清晰。
# g) 颜色变化可推断性
#    路径颜色规则保持原题：单色为实线，双色按格交替，端点相位由 ring 触点颜色锚定。
#    障碍颜色与组件颜色分离，避免“这是预先画好的路径还是障碍”这种歧义。
#
# Main() 新假设审查：
# - 框内主填充色在内部占多数：generator 让障碍面积显著小于内部总面积。
# - 每个 L 组件的最短路在障碍约束下可达且唯一：generator 显式检查 shortest-path count == 1。
# - 单色跨整边的 U 型条纹仍可直连且不穿障碍：solver 保留原逻辑，generator 不生成该分支，
#   因而不会触发新的覆盖冲突。


def _find_framed_rect_majority_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """Detect the main 1px rectangular frame and infer its dominant interior fill."""
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, fill_count, r0, c0, r1, c1, border_color, fill_color)
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

            area = int(h * w)
            cand = (area, fill_count, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    _area, _fill_count, r0, c0, r1, c1, bc, fill = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(fill)


def _ordered_neighbors(shape: tuple[int, int], p: tuple[int, int], goal: tuple[int, int]) -> list[tuple[int, int]]:
    """Deterministic 4-neighbors, preferring moves that reduce distance to the goal."""
    H, W = map(int, shape)
    r, c = map(int, p)
    gr, gc = map(int, goal)
    cand: list[tuple[int, int]] = []
    for nr, nc in neighbors4(int(r), int(c)):
        if 0 <= nr < H and 0 <= nc < W:
            cand.append((int(nr), int(nc)))
    cand.sort(key=lambda q: (abs(int(q[0]) - int(gr)) + abs(int(q[1]) - int(gc)), int(q[0]), int(q[1])))
    return cand


def _shortest_path(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
) -> list[tuple[int, int]] | None:
    """BFS shortest path on 4-neighborhood, avoiding blocked cells except at endpoints."""
    H, W = map(int, shape)
    sr, sc = map(int, start)
    gr, gc = map(int, goal)
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return None
    if (blocked[int(sr), int(sc)] and (sr, sc) != (gr, gc)) or blocked[int(gr), int(gc)]:
        return None

    q = deque([(int(sr), int(sc))])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {(int(sr), int(sc)): None}
    while q:
        cur = q.popleft()
        if cur == (int(gr), int(gc)):
            break
        for nxt in _ordered_neighbors((int(H), int(W)), cur, (int(gr), int(gc))):
            if blocked[int(nxt[0]), int(nxt[1])] and nxt != (int(gr), int(gc)):
                continue
            if nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    goal_pt = (int(gr), int(gc))
    if goal_pt not in prev:
        return None

    out: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal_pt
    while cur is not None:
        out.append((int(cur[0]), int(cur[1])))
        cur = prev[cur]
    out.reverse()
    return out


def _count_shortest_paths(
    shape: tuple[int, int],
    start: tuple[int, int],
    goal: tuple[int, int],
    blocked: np.ndarray,
    limit: int = 2,
) -> int:
    """Count shortest paths up to `limit` for ambiguity rejection in the generator."""
    H, W = map(int, shape)
    sr, sc = map(int, start)
    gr, gc = map(int, goal)
    if not (0 <= sr < H and 0 <= sc < W and 0 <= gr < H and 0 <= gc < W):
        return 0
    if (blocked[int(sr), int(sc)] and (sr, sc) != (gr, gc)) or blocked[int(gr), int(gc)]:
        return 0

    dist = np.full((H, W), -1, dtype=int)
    q = deque([(int(sr), int(sc))])
    dist[int(sr), int(sc)] = 0
    while q:
        r, c = q.popleft()
        for nr, nc in _ordered_neighbors((int(H), int(W)), (int(r), int(c)), (int(gr), int(gc))):
            if blocked[int(nr), int(nc)] and (nr, nc) != (int(gr), int(gc)):
                continue
            if dist[int(nr), int(nc)] != -1:
                continue
            dist[int(nr), int(nc)] = int(dist[int(r), int(c)] + 1)
            q.append((int(nr), int(nc)))

    if dist[int(gr), int(gc)] < 0:
        return 0

    cells = [(int(r), int(c)) for r in range(H) for c in range(W) if dist[int(r), int(c)] >= 0]
    cells.sort(key=lambda p: int(dist[int(p[0]), int(p[1])]))
    ways = np.zeros((H, W), dtype=int)
    ways[int(sr), int(sc)] = 1
    for r, c in cells:
        cur = int(ways[int(r), int(c)])
        if cur == 0:
            continue
        for nr, nc in _ordered_neighbors((int(H), int(W)), (int(r), int(c)), (int(gr), int(gc))):
            if dist[int(nr), int(nc)] != int(dist[int(r), int(c)] + 1):
                continue
            ways[int(nr), int(nc)] = min(int(limit), int(ways[int(nr), int(nc)]) + int(cur))
    return int(ways[int(gr), int(gc)])


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct obstacle-avoiding paths inside a framed rectangle from external touch traces.

    Input structure:
    - A single 1px closed rectangular frame surrounds an interior whose dominant color is the base fill.
    - Some cells inside that frame already contain obstacle blocks.
    - Colored components outside the frame touch its ring and encode endpoint positions plus mono/alternating colors.

    Rule:
    - Output only the frame interior.
    - Keep the interior obstacle cells exactly where they are.
    - For each ring-touching component, map its touch points to interior border coordinates and connect the chosen endpoints by a shortest 4-connected path that avoids obstacles.
    - Monochrome components paint the path solidly; two-color components alternate along the recovered shortest path.

    Key insight:
    - The outside traces still specify which border points must be connected; the new work is path planning on the interior grid, treating every pre-drawn non-fill cell as blocked terrain.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_majority_fill(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, bc, fill = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    out = inner.copy()
    blocked = inner != int(fill)
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

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
        cid = int(clab[int(rr), int(cc)])
        if cid == 0:
            continue
        mr, mc = map_to_out(int(rr), int(cc))
        comp_pts[int(cid)].append((int(mr), int(mc), int(g[int(rr), int(cc)])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[int(cid)] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(int(cid), [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1
        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            uniq = coords
            rows = {int(r) for r, _c in uniq}
            cols2 = {int(c) for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    d = abs(int(uniq[i][0]) - int(uniq[j][0])) + abs(int(uniq[i][1]) - int(uniq[j][1]))
                    if best is None or d > best[0]:
                        best = (int(d), uniq[i], uniq[j])
            if best is None:
                continue
            _d, p1, p2 = best
            path = _shortest_path((int(outH), int(outW)), p1, p2, blocked)
            assert not (path is None)
            for rr, cc in path:
                out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(int(coords[i][0]) - int(coords[j][0])) + abs(int(coords[i][1]) - int(coords[j][1]))
                    if best is None or d > best[0]:
                        best = (int(d), coords[i], coords[j])
            if best is None:
                continue
            _d, a, b = best
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            path = _shortest_path((int(outH), int(outW)), a, b, blocked)
            assert not (path is None)

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (int(ca), int(cb)):
                    end_color = int(start_color) if ((len(path_pts) - 1) % 2 == 0) else (int(cb) if int(start_color) == int(ca) else int(ca))
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
            cur = int(start_color)
            for rr, cc in path_pts:
                out[int(rr), int(cc)] = int(cur)
                cur = int(cb) if int(cur) == int(ca) else int(ca)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if int(p[0]) == 0]
        bots = [p for p in pts if int(p[0]) == outH - 1]
        if tops and bots:
            for cc in sorted({int(c) for _r, c in pts}):
                rs = [int(r) for r, c in pts if int(c) == int(cc)]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if int(p[1]) == 0]
        rights = [p for p in pts if int(p[1]) == outW - 1]
        if lefts and rights:
            for rr in sorted({int(r) for r, _c in pts}):
                cs = [int(c) for r, c in pts if int(r) == int(rr)]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        if len(pts) == 1:
            rr, cc = pts[0]
            out[int(rr), int(cc)] = int(col)
            continue
        best = None
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                d = abs(int(pts[i][0]) - int(pts[j][0])) + abs(int(pts[i][1]) - int(pts[j][1]))
                if best is None or d > best[0]:
                    best = (int(d), pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        path = _shortest_path((int(outH), int(outW)), p1, p2, blocked)
        assert not (path is None)
        for rr, cc in path:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 D1 shortest-path variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_OUTER_OFFSET_RANGE = (2, 3)
_N_COMPONENTS_RANGE = (2, 3)
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_COMPONENT_TRIES = 80
_GEN_RETRIES = 400


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
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _paint_component_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    start_color: int | None = None,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
        return
    a, b = map(int, colors)
    cur = int(a if start_color is None else start_color)
    vals = []
    for _ in path:
        vals.append(int(cur))
        cur = int(b) if int(cur) == int(a) else int(a)
    grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = {(int(r), int(c)) for r, c in coords}
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False
        for nr, nc in neighbors4(int(r), int(c)):
            if (int(nr), int(nc)) not in s and 0 <= nr < H and 0 <= nc < W and comp_mask[int(nr), int(nc)]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    start_color: int | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_component_path(grid, path, colors, start_color=start_color)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _rect_cells(rect: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, rect)
    if r0 > r1 or c0 > c1:
        return []
    return [(int(r), int(c)) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)]


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


def _old_l_path(kind: str, H: int, W: int, end_r: int, end_c: int) -> list[tuple[int, int]]:
    if kind == "tl":
        return _join(_vpath(0, end_r, end_c), _hpath(end_r, end_c, 0))
    if kind == "tr":
        return _join(_vpath(0, end_r, end_c), _hpath(end_r, end_c, W - 1))
    if kind == "bl":
        return _join(_vpath(H - 1, end_r, end_c), _hpath(end_r, end_c, 0))
    return _join(_vpath(H - 1, end_r, end_c), _hpath(end_r, end_c, W - 1))


def _l_detour_spec(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    if kind == "tl":
        end_r = random.randint(2, H - 2)
        end_c = random.randint(2, W - 2)
        gate_c = random.randint(1, end_c - 1)
        a = (0, end_c)
        b = (end_r, 0)
        rects = [(1, 0, end_r - 1, gate_c - 1), (1, gate_c + 1, end_r - 1, end_c)]
        desired = _join(_hpath(0, end_c, gate_c), _vpath(0, end_r, gate_c), _hpath(end_r, gate_c, 0))
    elif kind == "tr":
        end_r = random.randint(2, H - 2)
        end_c = random.randint(1, W - 3)
        gate_c = random.randint(end_c + 1, W - 2)
        a = (0, end_c)
        b = (end_r, W - 1)
        rects = [(1, end_c, end_r - 1, gate_c - 1), (1, gate_c + 1, end_r - 1, W - 1)]
        desired = _join(_hpath(0, end_c, gate_c), _vpath(0, end_r, gate_c), _hpath(end_r, gate_c, W - 1))
    elif kind == "bl":
        end_r = random.randint(1, H - 3)
        end_c = random.randint(2, W - 2)
        gate_c = random.randint(1, end_c - 1)
        a = (H - 1, end_c)
        b = (end_r, 0)
        rects = [(end_r + 1, 0, H - 2, gate_c - 1), (end_r + 1, gate_c + 1, H - 2, end_c)]
        desired = _join(_hpath(H - 1, end_c, gate_c), _vpath(H - 1, end_r, gate_c), _hpath(end_r, gate_c, 0))
    else:
        end_r = random.randint(1, H - 3)
        end_c = random.randint(1, W - 3)
        gate_c = random.randint(end_c + 1, W - 2)
        a = (H - 1, end_c)
        b = (end_r, W - 1)
        rects = [(end_r + 1, end_c, H - 2, gate_c - 1), (end_r + 1, gate_c + 1, H - 2, W - 1)]
        desired = _join(_hpath(H - 1, end_c, gate_c), _vpath(H - 1, end_r, gate_c), _hpath(end_r, gate_c, W - 1))

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))
    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + end_c)
    ring_b = (r0 + 1 + end_r, c0 - 1 if is_left else c1 + 1)
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )

    return {
        "kind": kind,
        "end_r": int(end_r),
        "end_c": int(end_c),
        "a": (int(a[0]), int(a[1])),
        "b": (int(b[0]), int(b[1])),
        "blocks": [rect for rect in rects if rect[0] <= rect[2] and rect[1] <= rect[3]],
        "desired": [(int(r), int(c)) for r, c in desired],
        "outer": [(int(r), int(c)) for r, c in outer],
        "old_l": _old_l_path(kind, H, W, int(end_r), int(end_c)),
    }


def _add_distractor(grid: np.ndarray, comp_mask: np.ndarray, bg: int,
                    frame: tuple[int, int, int, int], color: int) -> None:
    H, W = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(int(r), int(c))]
        used = {(int(r), int(c))}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [(int(nr), int(nc)) for nr, nc in neighbors4(*base)
                    if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (int(nr), int(nc)) not in used]
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
    H = random.randint(*_INTERIOR_H_RANGE)
    W = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], H + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], W + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - H - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - W - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + H + 1, c0 + W + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc

    input_inner = np.full((H, W), fill, dtype=int)
    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)

    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    if len(palette) < 4:
        return None
    obstacle_color = int(palette[0])
    color_i = 1

    blocked = np.zeros((H, W), dtype=bool)
    used_path = np.zeros((H, W), dtype=bool)
    stats_bends: list[int] = []
    n_components = random.randint(*_N_COMPONENTS_RANGE)
    corners = random.sample(["tl", "tr", "bl", "br"], k=int(n_components))

    for kind in corners:
        placed = False
        for _ in range(_COMPONENT_TRIES):
            spec = _l_detour_spec(kind, H, W, r0, c0, r1, c1)
            block_cells = [p for rect in spec["blocks"] for p in _rect_cells(rect)]
            if not block_cells:
                continue
            if any(blocked[int(r), int(c)] or used_path[int(r), int(c)] for r, c in block_cells):
                continue

            blocked_try = blocked.copy()
            for rr, cc in block_cells:
                blocked_try[int(rr), int(cc)] = True

            path_blocked = blocked_try | used_path
            path = _shortest_path((H, W), spec["a"], spec["b"], path_blocked)
            if path is None:
                continue
            if _count_shortest_paths((H, W), spec["a"], spec["b"], path_blocked, limit=2) != 1:
                continue
            bends = _count_bends(path)
            if bends < 2:
                continue
            if path == spec["old_l"]:
                continue
            if any(used_path[int(r), int(c)] for r, c in path):
                continue

            want_dual = random.random() < _DUAL_COLOR_PROB and color_i + 1 < len(palette)
            colors: tuple[int, ...]
            outer_path = spec["outer"]
            outer_start_color = None
            exp_start_color = None
            if want_dual and ((len(path) - 1) % 2 == (len(outer_path) - 1) % 2):
                colors2 = (int(palette[color_i]), int(palette[color_i + 1]))
                candidates: list[tuple[int, list[tuple[int, int]], int, int]] = []
                for exp_sc in colors2:
                    end_sc = int(exp_sc) if ((len(path) - 1) % 2 == 0) else (int(colors2[1]) if int(exp_sc) == int(colors2[0]) else int(colors2[0]))
                    chosen_outer = _choose_outer_coloring(outer_path, colors2, int(exp_sc), int(end_sc))
                    if chosen_outer is not None:
                        candidates.append((int(exp_sc), chosen_outer[0], int(chosen_outer[1]), bends))
                if candidates:
                    exp_start_color, outer_path, outer_start_color, _ = random.choice(candidates)
                    colors = colors2
                else:
                    colors = (int(palette[color_i]),)
            else:
                colors = (int(palette[color_i]),)

            if len(colors) == 1:
                if not _can_place(grid, outer_path, bg, comp_mask):
                    continue
                for rr, cc in block_cells:
                    input_inner[int(rr), int(cc)] = int(obstacle_color)
                    expected[int(rr), int(cc)] = int(obstacle_color)
                if not _place_component(grid, comp_mask, outer_path, colors, bg):
                    return None
                _paint_component_path(expected, path, colors)
                color_i += 1
            else:
                if color_i + 1 >= len(palette):
                    continue
                if outer_start_color is None or exp_start_color is None:
                    continue
                if not _can_place(grid, outer_path, bg, comp_mask):
                    continue
                for rr, cc in block_cells:
                    input_inner[int(rr), int(cc)] = int(obstacle_color)
                    expected[int(rr), int(cc)] = int(obstacle_color)
                if not _place_component(grid, comp_mask, outer_path, colors, bg, start_color=int(outer_start_color)):
                    return None
                _paint_component_path(expected, path, colors, start_color=int(exp_start_color))
                color_i += 2

            for rr, cc in block_cells:
                blocked[int(rr), int(cc)] = True
            for rr, cc in path:
                used_path[int(rr), int(cc)] = True
            stats_bends.append(int(bends))
            placed = True
            break
        if not placed:
            return None

    if not stats_bends or max(stats_bends) < 2:
        return None

    grid[r0 + 1 : r1, c0 + 1 : c1] = input_inner

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
    raise RuntimeError("failed to generate a valid a6f40cea D1 shortest-path input")
