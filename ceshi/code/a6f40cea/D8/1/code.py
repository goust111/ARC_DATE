# Variant of a6f40cea
# Dimension: D8 — 拐角选择规则 (path)
# Change: L形路径拐角选择使其严格位于内部（不在边缘上），优先选择cornerA方案... → 拐角由外部框角位置决定
#
# Design analysis
# spec:
#   source_task: none
#   concrete_behavior:
#     - 输入仍然只有一个封闭矩形框，以及若干贴着框外ring的开放矩形组件。
#     - 变体只改 L 组件：原题把两端点投到输出后，在两个曼哈顿拐角候选里选“严格在内部”的那个。
#     - 这里改成读取该开放矩形真正的外部角；把那个外角用和 ring 点相同的裁剪投影映到裁剪后的输出坐标系。
#     - 由于生成器中的 L 组件总是连接相邻两条框边，这个投影会落到输出四个边角之一，因此路径会贴着对应输出角走，而不是穿过内部拐点。
#   behavior_trace:
#     - 例：组件触 top 和 left，ring 端点映到 (0, 5) 与 (3, 0)。
#     - 原题：候选拐角是 (3, 5) 与 (0, 0)，因为 (3, 5) 严格在内部，所以走内部 L。
#     - 变体：先看外部开放框的外角位于 frame 的左上方；把这个外角投到输出，得到 (0, 0)。
#     - 于是输出改为沿顶边到左上角，再沿左边下行。
#   complexity_calibration:
#     - 如果把“外角投影”实现成仍然得到 (3, 5) 的内部点，那么同一输入上新旧 solver 会完全同解，变体会退化。
#     - 因此这里必须让投影选择到旧逻辑的另一个候选拐角，也就是输出边角，这样 D8 变化在像素层面稳定可见。
#     - 为了让样例持续触发 D8，关闭 U-pair 分支，只生成 2-3 个 L 组件；这样每张图都有多个重复支撑。
#   implicit_assumptions:
#     - 原 main() 对 L 组件的关键假设是：只需看两个 ring 端点，内部候选拐角就唯一确定。
#     - 变体后，这个假设失效；solver 必须额外恢复“组件 bbox 在外部的哪个角”。
#     - 旧代码中所有调用 inside_corner() 选择 pathA/pathB 的位置，都依赖“内部拐角优先”这一旧属性，必须替换。
#     - 两色交替的相位仍然只依赖 ring 端点颜色提示，这一点保持不变。
#   placement_legality:
#     - 生成器继续先构造输出再反推输入。
#     - L 路径改贴输出边角后，更容易在同一条边上重叠；因此保留原来的“按象限取上/下半区、左/右半区”的 elbow 采样，并继续用 used_inner 拒绝重叠。
#     - 外部开放矩形仍用原放置逻辑与 comp_mask 校验，因此不会越界、重叠或与框相接触错误。
#   visual_observability:
#     - 人类从少量样例就能看出：外部开口框在哪个外角张开，内部路径就贴着对应输出角走。
#     - 这不依赖颜色 id 排序，也不依赖隐藏元数据；只依赖肉眼可见的几何对应。
#   cleanliness_and_color:
#     - 输出仍是少量单色/双色 L 路径叠在统一填充底上，视觉层次清楚。
#     - 颜色机制不新增额外映射，只保留原题的单色实线与双色交替，避免把推理难点转移到颜色规则上。

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


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

    best = None  # (area, r0,c0,r1,c1, bc, fill)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the cropped frame interior by extending outside open-rectangle traces inward.

    Input structure:
    - A uniform background contains one 1px closed rectangular frame with a uniform fill.
    - Colored 4-connected open-rectangle components lie outside the frame and touch the ring just beyond its border.

    Rule:
    - Output only the frame interior, initialized with the fill color.
    - Straight same-side components still produce full-span monochrome stripes after endpoint aggregation.
    - For an L-component, detect which vertical side and horizontal side it touches, recover its exterior bbox corner, and project that outer corner into the cropped interior coordinates.
    - Use that projected outer corner as the Manhattan elbow; paint it solid for 1-color components or alternately for 2-color components.

    Key insight: The bend is no longer chosen by asking which endpoint-derived elbow stays strictly inside; the outside corner of the open rectangle directly selects the output corner that the path must pass through.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
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

    def segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb) = a, b
        if ra == rb:
            step = 1 if cb >= ca else -1
            return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        if ca == cb:
            step = 1 if rb >= ra else -1
            return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        raise ValueError("segment expects row- or col-aligned endpoints")

    def build_corner_path(
        p_tb: tuple[int, int],
        p_lr: tuple[int, int],
        corner: tuple[int, int],
    ) -> list[tuple[int, int]]:
        head = segment(p_tb, corner)
        tail = segment(corner, p_lr)
        return head + tail[1:]

    def build_manhattan(
        p1: tuple[int, int],
        p2: tuple[int, int],
        cornerA: bool,
    ) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return segment(p1, p2)
        if cornerA:
            return segment(p1, (int(rb), int(ca))) + segment((int(rb), int(ca)), p2)[1:]
        return segment(p1, (int(ra), int(cb))) + segment((int(ra), int(cb)), p2)[1:]

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    def dominant_coord(entries: list[tuple[int, int, int]]) -> tuple[int, int]:
        cnt = Counter((int(r), int(c)) for r, c, _col in entries)
        r, c = cnt.most_common(1)[0][0]
        return int(r), int(c)

    def dominant_color(entries: list[tuple[int, int, int]]) -> int:
        return int(Counter(int(col) for _r, _c, col in entries).most_common(1)[0][0])

    def paint_alt(path_pts: list[tuple[int, int]], cols2: tuple[int, int], start_color: int) -> None:
        a, b = map(int, cols2)
        cur = int(start_color)
        for rr, cc in path_pts:
            out[int(rr), int(cc)] = int(cur)
            cur = int(b) if cur == int(a) else int(a)

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

    comp_pts: dict[int, list[tuple[int, int, int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        col = int(g[rr, cc])
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(rr), int(cc), int(mr), int(mc), int(col)))

    comp_cols: dict[int, list[int]] = {}
    comp_boxes: dict[int, tuple[int, int, int, int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))
        pts = np.argwhere(clab == int(cid))
        if pts.size == 0:
            continue
        mins = pts.min(axis=0)
        maxs = pts.max(axis=0)
        comp_boxes[cid] = (int(mins[0]), int(mins[1]), int(maxs[0]), int(maxs[1]))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        side_hits: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
        for rr, cc, mr, mc, col in pts:
            hist[(int(mr), int(mc))][int(col)] += 1
            if rr == r0 - 1:
                side_hits["top"].append((int(mr), int(mc), int(col)))
            if rr == r1 + 1:
                side_hits["bottom"].append((int(mr), int(mc), int(col)))
            if cc == c0 - 1:
                side_hits["left"].append((int(mr), int(mc), int(col)))
            if cc == c1 + 1:
                side_hits["right"].append((int(mr), int(mc), int(col)))

        coords = list(hist.keys())
        tb_sides = [side for side in ("top", "bottom") if side_hits[side]]
        lr_sides = [side for side in ("left", "right") if side_hits[side]]
        is_l = len(tb_sides) == 1 and len(lr_sides) == 1 and cid in comp_boxes

        if is_l:
            tb_side = tb_sides[0]
            lr_side = lr_sides[0]
            p_tb = dominant_coord(side_hits[tb_side])
            p_lr = dominant_coord(side_hits[lr_side])
            minr, minc, maxr, maxc = comp_boxes[cid]
            outer_r = minr if tb_side == "top" else maxr
            outer_c = minc if lr_side == "left" else maxc
            corner = map_to_out(outer_r, outer_c)
            path = build_corner_path(p_tb, p_lr, corner)

            if len(cols) == 1:
                for rr, cc in path:
                    out[int(rr), int(cc)] = int(cols[0])
                continue

            if len(cols) == 2:
                ca, cb = map(int, cols)
                hint_a = dominant_color(side_hits[tb_side])
                hint_b = dominant_color(side_hits[lr_side])
                chosen = None
                for path_pts, start_hint, end_hint in (
                    (path, hint_a, hint_b),
                    (list(reversed(path)), hint_b, hint_a),
                ):
                    for start_color in (ca, cb):
                        end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
                        if int(start_color) != int(start_hint):
                            continue
                        if int(end_color) != int(end_hint):
                            continue
                        chosen = (path_pts, int(start_color))
                        break
                    if chosen is not None:
                        break
                assert chosen is not None
                path_pts, start_color = chosen
                paint_alt(path_pts, (ca, cb), start_color)
                continue

        if len(cols) == 1:
            colv = int(cols[0])
            uniq = coords
            rows2 = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows2) == 1 or len(cols2) == 1:
                for p in uniq:
                    deferred[colv].add((int(p[0]), int(p[1])))
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
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(colv)
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
            ca, cb = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (ca, cb):
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (cb if start_color == ca else ca)
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
            paint_alt(path_pts, (ca, cb), start_color)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
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
                d = abs(pts[i][0] - pts[j][0]) + abs(pts[i][1] - pts[j][1])
                if best is None or d > best[0]:
                    best = (d, pts[i], pts[j])
        if best is None:
            continue
        _d, p1, p2 = best
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


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
_U_PAIR_PROB = 0.0
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300


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


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if grid[r, c] != bg:
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
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build one L component whose output path is routed through the projected outer corner."""
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
    c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else H - 1
    edge_c = 0 if is_left else W - 1
    inner = _join(_hpath(edge_r, c, edge_c), _vpath(edge_r, r, edge_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer


def _u_pair_specs(orientation: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
        if c_b - c_a < 2:
            c_b = min(W - 2, c_a + 2)
        top_r = r0 - random.randint(lo, hi)
        bot_r = r1 + random.randint(lo, hi)
        a_top, b_top = (r0 - 1, c0 + 1 + c_a), (r0 - 1, c0 + 1 + c_b)
        a_bot, b_bot = (r1 + 1, c0 + 1 + c_a), (r1 + 1, c0 + 1 + c_b)
        outer_a = _join(_vpath(a_top[0], top_r, a_top[1]), _hpath(top_r, a_top[1], b_top[1]), _vpath(top_r, b_top[0], b_top[1]))
        outer_b = _join(_vpath(a_bot[0], bot_r, a_bot[1]), _hpath(bot_r, a_bot[1], b_bot[1]), _vpath(bot_r, b_bot[0], b_bot[1]))
        inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
    else:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = c0 - random.randint(lo, hi)
        right_c = c1 + random.randint(lo, hi)
        a_left, b_left = (r0 + 1 + r_a, c0 - 1), (r0 + 1 + r_b, c0 - 1)
        a_right, b_right = (r0 + 1 + r_a, c1 + 1), (r0 + 1 + r_b, c1 + 1)
        outer_a = _join(_hpath(a_left[0], a_left[1], left_c), _vpath(a_left[0], b_left[0], left_c), _hpath(b_left[0], left_c, b_left[1]))
        outer_b = _join(_hpath(a_right[0], a_right[1], right_c), _vpath(a_right[0], b_right[0], right_c), _hpath(b_right[0], right_c, b_right[1]))
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
        if _place_component(grid, comp_mask, coords, (color,), bg):
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
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        color = palette[color_i]
        color_i += 1
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        expected[irs, ics] = color
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB:
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True

    for _ in range(random.randint(*_DISTRACTOR_COUNT_RANGE)):
        if color_i >= len(palette):
            break
        _add_distractor(grid, comp_mask, bg, (r0, c0, r1, c1), palette[color_i])
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
    raise RuntimeError("failed to generate a valid a6f40cea D8 input")
