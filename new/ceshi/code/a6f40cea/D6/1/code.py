# Variant of a6f40cea
# Dimension: D6 — 路径上的变换效果 (transform)
# Change: 路径用开放框架的颜色（或交替颜色）直接涂色到填充色背景上，无额外变换... → 路径填充封闭区域

from __future__ import annotations

# Design analysis for the D6 variant
# - concretization:
#   - 原题把开放矩形映射成内部的 L 形线或成对直线；本变体保留相同输入几何，只改输出阶段。
#   - 单色 L 形不再只画折线，而是把“折线 + 裁剪边界”围出的角落矩形整块填色。
#   - U-pair 原本只得到两条全跨度线；现在改为填满两条线之间的整条带状区域。
#   - 双色 L 形若直接改成纯色块，会让第二种颜色失去作用；因此用相对角点的曼哈顿奇偶扩展成交替棋盘块，
#     这样既是 flood-fill 式区域扩张，又保持端点颜色锚定仍然可见。
#   - 具体例子：top-left L 若原先输出 path={(0,c)..(r,c)}∪{(r,c-1)..(r,0)}，现在输出改为
#     rows 0..r, cols 0..c 的整块。单色时整块同色；双色时该矩形按到 (0,0) 的曼哈顿距离奇偶交替。
# - complexity calibration:
#   - 输入对象种类、外部开放矩形、干扰物、框尺寸都保持原约束，避免把题目简化成“识别一个矩形然后上色”。
#   - 只允许原题已有的两类内部结构：角落 L 和对向 U-pair。这样 flood-fill 的形状只有角块或整带，视觉稳定。
#   - 保留双色 L，但不把双色扩展到 U-pair，避免引入额外二维传播规则，控制推断复杂度。
# - implicit assumptions review:
#   - 原 solver 在 L 分支中对 path 逐点赋值，默认不同组件即使相交也只有线宽 1；本变体若仍逐点赋值会漏掉块区域，
#     所以必须改成先识别所触边界，再恢复角点/带宽并整体填充。
#   - 原 generator 的 used_inner 只占路径像素，这在块填充下不再安全；需要升级成整块占用检查，否则不同区域会覆盖。
#   - 原双色逻辑依赖端点颜色确定 path 相位；本变体仍保留这个前提，但相位扩展到整个矩形块，而不是只扩展到折线。
# - placement legality:
#   - 外部组件的放置规则保持不变，仍要求不重叠、不越界、不与已有组件 4 邻接。
#   - 内部期望输出改为矩形/带状区后，generator 用区域掩码阻止不同块重叠，从而保证 solver 的覆盖顺序不重要。
# - visual observability:
#   - 与原题相比，新输出会把 1px 线扩成明显色块；只看 2-3 对样例也能看到“沿路径围出的那块区域被灌满”。
#   - 双色 L 的棋盘块与外部交替色 open-rect 直接对应，不依赖颜色 id 排序等不可见信息。
# - visual cleanliness:
#   - 区域只允许矩形角块或整带，避免任意折线填充造成噪声。
#   - 颜色仍然一组件一调色板；U-pair 只共用自己的一种颜色，不与其他块复用，避免块之间的归属歧义。
# - new solver assumptions and how generation satisfies them:
#   - L 组件仍然只接触相邻两条边，且其肘点严格在内部；_l_specs 延续原约束保证这一点。
#   - U-pair 仍然由同色的两条同向全跨度边界线组成；_u_pair_specs 继续成对构造并共享颜色。
#   - 同一输出区域不会被两个组件共同占用；generator 在 expected 端用 used_inner 矩形占用检查保证这一点。

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


def _rect_bounds(a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), max(a[0], b[0]), min(a[1], b[1]), max(a[1], b[1])


def _other_color(colors: tuple[int, int], color: int) -> int:
    return int(colors[1]) if int(color) == int(colors[0]) else int(colors[0])


def _paint_checker_rect(
    out: np.ndarray,
    bounds: tuple[int, int, int, int],
    colors: tuple[int, int],
    corner: tuple[int, int],
    corner_color: int,
) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    other = _other_color(colors, int(corner_color))
    rs = np.arange(r_lo, r_hi + 1)[:, None]
    cs = np.arange(c_lo, c_hi + 1)[None, :]
    parity = (np.abs(rs - int(corner[0])) + np.abs(cs - int(corner[1]))) % 2
    block = np.where(parity == 0, int(corner_color), int(other))
    out[r_lo : r_hi + 1, c_lo : c_hi + 1] = block


def _edge_corner_and_elbow(
    p1: tuple[int, int],
    p2: tuple[int, int],
    outH: int,
    outW: int,
    cornerA: bool,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if p1[0] == p2[0] or p1[1] == p2[1]:
        return None, None
    row_edge = None
    if p1[0] == 0 or p2[0] == 0:
        row_edge = 0
    elif p1[0] == outH - 1 or p2[0] == outH - 1:
        row_edge = outH - 1
    col_edge = None
    if p1[1] == 0 or p2[1] == 0:
        col_edge = 0
    elif p1[1] == outW - 1 or p2[1] == outW - 1:
        col_edge = outW - 1
    if row_edge is None or col_edge is None:
        return None, None
    elbow = (int(p2[0]), int(p1[1])) if cornerA else (int(p1[0]), int(p2[1]))
    return (int(row_edge), int(col_edge)), elbow


def _checker_color_at(
    pt: tuple[int, int],
    colors: tuple[int, int],
    corner: tuple[int, int],
    corner_color: int,
) -> int:
    d = abs(int(pt[0]) - int(corner[0])) + abs(int(pt[1]) - int(corner[1]))
    if d % 2 == 0:
        return int(corner_color)
    return _other_color(colors, int(corner_color))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project the outside open-rectangle traces inward and flood the regions they fence off.

    Input structure:
    - A background grid contains one 1px closed rectangular frame with a uniform interior color.
    - Separate colored components lie outside the frame and touch the ring just beyond its border.

    Rule:
    - Crop the output to the frame interior and start from the interior fill color.
    - Map each ring-touch point into interior coordinates exactly as in the original task.
    - L-shaped traces no longer stay as 1px paths: they flood-fill the corner block enclosed by the trace and the crop boundary.
    - Matched opposite-side monochrome traces become solid bands, while dual-color L traces expand with the same endpoint-anchored alternation across the filled block.

    Key insight: Once the ring-touch points are mapped inward, the touched sides determine which corner or full-span band is enclosed, so the fill region can be reconstructed from the elbow and endpoint colors alone.
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

    def build_manhattan(
        p1: tuple[int, int],
        p2: tuple[int, int],
        cornerA: bool,
    ) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb:
            step = 1 if cb >= ca else -1
            return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        if ca == cb:
            step = 1 if rb >= ra else -1
            return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        if cornerA:
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
            return path
        step = 1 if cb >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
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
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        if len(pts) < 2:
            continue
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        for r, c, col in pts:
            hist[(int(r), int(c))][int(col)] += 1

        coords = list(hist.keys())

        if len(cols) == 1:
            colv = int(cols[0])
            rows = {r for r, _c in coords}
            cols2 = {c for _r, c in coords}
            if len(rows) == 1 or len(cols2) == 1:
                for p in coords:
                    deferred[colv].add((int(p[0]), int(p[1])))
                continue

            best = None
            for i in range(len(coords)):
                for j in range(i + 1, len(coords)):
                    d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                    if best is None or d > best[0]:
                        best = (d, coords[i], coords[j])
            if best is None:
                continue
            _d, p1, p2 = best
            use_cornerA = not (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True))
            corner, elbow = _edge_corner_and_elbow(p1, p2, outH, outW, use_cornerA)
            if corner is None or elbow is None:
                chosen = build_manhattan(p1, p2, use_cornerA)
                for rr, cc in chosen:
                    out[int(rr), int(cc)] = int(colv)
                continue
            r_lo, r_hi, c_lo, c_hi = _rect_bounds(corner, elbow)
            out[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(colv)
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
            colors = (ca, cb)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            use_cornerA = not (inside_corner(a, b, False) and not inside_corner(a, b, True))
            corner, elbow = _edge_corner_and_elbow(a, b, outH, outW, use_cornerA)
            if corner is None or elbow is None:
                path = build_manhattan(a, b, use_cornerA)

                chosen = None
                for path_pts in (path, list(reversed(path))):
                    for start_color in colors:
                        end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else _other_color(colors, start_color)
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
                    cur = _other_color(colors, cur)
                continue

            corner_color = None
            for seed in colors:
                if _checker_color_at(a, colors, corner, seed) != int(ha):
                    continue
                if _checker_color_at(b, colors, corner, seed) != int(hb):
                    continue
                corner_color = int(seed)
                break
            assert corner_color is not None
            bounds = _rect_bounds(corner, elbow)
            _paint_checker_rect(out, bounds, colors, corner, corner_color)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == outH - 1]
        if tops and bots:
            c_lo = min(c for _r, c in pts)
            c_hi = max(c for _r, c in pts)
            out[:, c_lo : c_hi + 1] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == outW - 1]
        if lefts and rights:
            r_lo = min(r for r, _c in pts)
            r_hi = max(r for r, _c in pts)
            out[r_lo : r_hi + 1, :] = int(col)
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
        use_cornerA = not (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True))
        corner, elbow = _edge_corner_and_elbow(p1, p2, outH, outW, use_cornerA)
        if corner is None or elbow is None:
            chosen = build_manhattan(p1, p2, use_cornerA)
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(col)
            continue
        r_lo, r_hi, c_lo, c_hi = _rect_bounds(corner, elbow)
        out[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(col)

    return out


# --------------- generator (a6f40cea D6) ---------------

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


def _l_region_meta(
    kind: str,
    H: int,
    W: int,
    r: int,
    c: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, r, 0, c), (0, 0), (0, c)
    if kind == "tr":
        return (0, r, c, W - 1), (0, W - 1), (0, c)
    if kind == "bl":
        return (r, H - 1, 0, c), (H - 1, 0), (H - 1, c)
    return (r, H - 1, c, W - 1), (H - 1, W - 1), (H - 1, c)


def _paint_l_region(
    expected: np.ndarray,
    kind: str,
    H: int,
    W: int,
    r: int,
    c: int,
    colors: tuple[int, ...],
) -> tuple[int, int, int, int]:
    bounds, corner, start = _l_region_meta(kind, H, W, r, c)
    r_lo, r_hi, c_lo, c_hi = bounds
    if len(colors) == 1:
        expected[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(colors[0])
    else:
        a, b = map(int, colors)
        start_color = int(colors[0])
        d0 = abs(int(start[0]) - int(corner[0])) + abs(int(start[1]) - int(corner[1]))
        corner_color = int(start_color) if d0 % 2 == 0 else int(b if start_color == a else a)
        _paint_checker_rect(expected, bounds, (a, b), corner, corner_color)
    return bounds


def _paint_band(expected: np.ndarray, bounds: tuple[int, int, int, int], color: int) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    expected[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(color)


def _bbox_is_free(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> bool:
    r_lo, r_hi, c_lo, c_hi = bounds
    return not mask[r_lo : r_hi + 1, c_lo : c_hi + 1].any()


def _mark_bbox(mask: np.ndarray, bounds: tuple[int, int, int, int]) -> None:
    r_lo, r_hi, c_lo, c_hi = bounds
    mask[r_lo : r_hi + 1, c_lo : c_hi + 1] = True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
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
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer, r, c


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
        band = (0, H - 1, c_a, c_b)
    else:
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            r_b = min(H - 2, r_a + 2)
        left_c = c0 - random.randint(lo, hi)
        right_c = c1 + random.randint(lo, hi)
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
        band = (r_a, r_b, 0, W - 1)
    return inner, outer_a, outer_b, band


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
        inner_lines, outer_a, outer_b, band = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        if not _bbox_is_free(used_inner, band):
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        _paint_band(expected, band, color)
        _mark_bbox(used_inner, band)
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        inner, outer, r, c = _l_specs(kind, H, W, r0, c0, r1, c1)
        bounds, _corner, _start = _l_region_meta(kind, H, W, r, c)
        if not _bbox_is_free(used_inner, bounds):
            return None
        if random.random() < _DUAL_COLOR_PROB:
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
        else:
            colors = (palette[color_i],)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_l_region(expected, kind, H, W, r, c, colors)
        _mark_bbox(used_inner, bounds)

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
    raise RuntimeError("failed to generate a valid a6f40cea D6 input")
