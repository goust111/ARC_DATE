# Variant of a6f40cea
# Dimension: D13 — U形配对机制 (path)
# Change: 可选的U形对（两个U形开放框在框架对侧，产生贯穿内部的全跨直线条纹）... → H形连接
#
# Design analysis
# ---------------
# variant_spec:
#   concrete_behavior:
#     - 保留原任务的主框、均匀填充、角落L形开放框、双色交替路径、干扰块。
#     - 唯一核心改动是：原来“同色U形开放框在相对边成对出现 -> 两条全跨直线条纹”，
#       现在改成“同色U形开放框分别在一条水平边和一条垂直边上 -> 框内生成单色H形”。
#     - 水平边U提供H的两列(c0,c1)，垂直边U提供H的两行(r0,r1)，输出为：
#       两条竖线 (r0..r1,c0)/(r0..r1,c1) + 中点横线 ((r0+r1)//2, c0..c1)。
#     - 为了让变化肉眼可见，H只在 H>=5 时生成，并且其两行/两列都放在 interior 内部
#       (索引1..H-2 / 1..W-2)，避免退化成原题那种贴边全跨条纹。
#   behavior_trace:
#     - 原逻辑：top-U 与 bottom-U 同色时，各自只给出同列触点；solver 先延后，再按颜色聚合，
#       最终画出贯穿整块 interior 的两条竖条。
#     - 新逻辑：top-U 与 left-U 同色时，若仍沿用旧聚合，只会得到一条顶边线段和一条左边线段，
#       完全看不出“配对”的意义；因此 solver 必须把“同色 + 同侧U”先分类，再跨组件配对。
#     - 新执行：top-U 给两列，left-U 给两行，solver 配对后二者共同决定一个内部包围盒，
#       再画成 H，而不是各画各的。
#   complexity_calibration:
#     - H-pair 仍只占 2 个组件槽位，最多一个 pair，避免颜色/配对歧义。
#     - H-pair 使用单色；双色仍只用于原有L形路径，避免在分叉图形上引入额外相位推理。
#     - 为保证训练样本中经常能看到该维度，pair 概率提高到 0.75；否则 3 个样例里完全看不到
#       H 的概率过高，不利于 ARC 观察推断。
#   implicit_assumptions_review:
#     - 原 solver 里“同一行/列的单色 ring 触点 -> deferred -> 颜色聚合 -> 全跨条纹”依赖
#       的前提是：这些点只会来自相对边 U-pair。变体下该前提失效，必须改为“先按 side 分类，再做
#       horizontal-U / vertical-U 配对”。
#     - 原来按颜色直接把 deferred 点并集处理是无害的，因为一组颜色只表达一个条纹族；
#       现在若仍这么做，会把两个相邻边的 U 错画成彼此无关的边线。
#     - 新 solver 引入的假设是：同色的 same-side 单色组件至多组成一个 horizontal-U 和一个
#       vertical-U。generator 用唯一色板保证这一点。
#   placement_legality:
#     - 外部 U 仍是单个不自交开放矩形框，且只在各自侧面的 ring 上有两个接触点。
#     - H 的两行/两列相隔至少 2，确保中横线落在两端之间，图形不退化成门字或矩形边。
#     - inner 仍用 used_inner 做占位检查，任何与 L 形路径重叠的候选都会整次放弃重试。
#   visual_observability:
#     - H 位于 interior 内部，不贴边；与原题的全跨条纹在视觉上明显不同。
#     - 四种 side 组合(top/left, top/right, bottom/left, bottom/right)都服从同一个“横边给列，
#       竖边给行”的规则，不引入多套难以从少量样例总结的变换。
#   cleanliness_and_color:
#     - 每个配对颜色唯一，避免 solver 或人类把 H-pair 和普通 L 混淆。
#     - 干扰块仍远离主框，不参与 ring，不会污染配对推理。
#   post_change_main_audit:
#     - same-side 单色组件分支假设其投影在输出边界上是水平或竖直的；generator 的 U 结构保证。
#     - H 配对分支假设 paired rows/cols 都非退化区间；generator 通过 min gap=2 保证。
#     - 未配对 leftover 的 same-side 组件被保守地还原为边界线段，保证 main 对异常输入也不崩。

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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct interior traces from colored open frames around a uniform filled rectangle.

    Input structure:
    - A background grid contains one closed 1px frame with a solid fill.
    - Colored 4-connected components sit outside the frame and touch its surrounding ring.

    Rule:
    - Return only the interior crop of the closed frame, initialized with the fill color.
    - Ordinary corner components still project to solid or alternating Manhattan L-paths.
    - Same-color U components on one horizontal side and one vertical side now pair up to define one solid H:
      the horizontal-side U supplies the two columns, the vertical-side U supplies the two rows.

    Key insight:
    - Monochrome same-side touches cannot be solved independently anymore; they must first be recognized as
      U-side signatures and then paired across adjacent frame sides to recover the coupled H geometry.
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

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
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

    def classify_same_side(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]] | None:
        rows = sorted({r for r, _c in coords})
        cols = sorted({c for _r, c in coords})
        if len(rows) == 1 and len(cols) >= 2:
            rr = int(rows[0])
            if rr == 0 or rr == outH - 1:
                side = "top" if rr == 0 else "bottom"
                return side, (int(cols[0]), int(cols[-1]))
        if len(cols) == 1 and len(rows) >= 2:
            cc = int(cols[0])
            if cc == 0 or cc == outW - 1:
                side = "left" if cc == 0 else "right"
                return side, (int(rows[0]), int(rows[-1]))
        return None

    def paint_h(color: int, hsig: tuple[str, tuple[int, int]], vsig: tuple[str, tuple[int, int]]) -> None:
        _hside, (c_a, c_b) = hsig
        _vside, (r_a, r_b) = vsig
        mid_r = (int(r_a) + int(r_b)) // 2
        for rr in range(int(r_a), int(r_b) + 1):
            out[int(rr), int(c_a)] = int(color)
            out[int(rr), int(c_b)] = int(color)
        for cc in range(int(c_a), int(c_b) + 1):
            out[int(mid_r), int(cc)] = int(color)

    def paint_same_side_segment(color: int, sig: tuple[str, tuple[int, int]]) -> None:
        side, span = sig
        a, b = map(int, span)
        if side == "top":
            out[0, a : b + 1] = int(color)
        elif side == "bottom":
            out[outH - 1, a : b + 1] = int(color)
        elif side == "left":
            out[a : b + 1, 0] = int(color)
        else:
            out[a : b + 1, outW - 1] = int(color)

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

    deferred_u: dict[int, list[tuple[str, tuple[int, int]]]] = defaultdict(list)

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
            same_side = classify_same_side(coords)
            if same_side is not None:
                deferred_u[colv].append(same_side)
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

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(cb) if cur == int(ca) else int(ca)

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
            paint_alt(path_pts, start_color)
            continue

    for color, sigs in deferred_u.items():
        horiz = [sig for sig in sigs if sig[0] in ("top", "bottom")]
        vert = [sig for sig in sigs if sig[0] in ("left", "right")]

        while horiz and vert:
            best = None
            for i, (_hside, (c_a, c_b)) in enumerate(horiz):
                for j, (_vside, (r_a, r_b)) in enumerate(vert):
                    score = (int(c_b) - int(c_a)) * (int(r_b) - int(r_a))
                    if best is None or score > best[0]:
                        best = (score, i, j)
            assert best is not None
            _score, hi, vi = best
            hsig = horiz.pop(int(hi))
            vsig = vert.pop(int(vi))
            paint_h(int(color), hsig, vsig)

        for sig in horiz + vert:
            paint_same_side_segment(int(color), sig)

    return out


# --------------- generator (a6 / D13 variant) ---------------

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
_H_PAIR_PROB = 0.75
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


def _merge_unique(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for part in parts:
        for p in part:
            pp = (int(p[0]), int(p[1]))
            if pp not in seen:
                seen.add(pp)
                out.append(pp)
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


def _paint_cells(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    if not coords:
        return
    rs, cs = zip(*coords)
    grid[rs, cs] = int(color)


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


def _pick_spaced_pair(lo: int, hi: int, min_gap: int) -> tuple[int, int] | None:
    vals = list(range(int(lo), int(hi) + 1))
    cand = [(a, b) for i, a in enumerate(vals) for b in vals[i + 1 :] if b - a >= int(min_gap)]
    if not cand:
        return None
    return random.choice(cand)


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
    return inner, outer


def _u_side_specs(side: str, span: tuple[int, int], r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    a, b = map(int, span)
    if side == "top":
        outer_r = r0 - random.randint(lo, hi)
        ring_a = (r0 - 1, c0 + 1 + a)
        ring_b = (r0 - 1, c0 + 1 + b)
        return _join(
            _vpath(ring_a[0], outer_r, ring_a[1]),
            _hpath(outer_r, ring_a[1], ring_b[1]),
            _vpath(outer_r, ring_b[0], ring_b[1]),
        )
    if side == "bottom":
        outer_r = r1 + random.randint(lo, hi)
        ring_a = (r1 + 1, c0 + 1 + a)
        ring_b = (r1 + 1, c0 + 1 + b)
        return _join(
            _vpath(ring_a[0], outer_r, ring_a[1]),
            _hpath(outer_r, ring_a[1], ring_b[1]),
            _vpath(outer_r, ring_b[0], ring_b[1]),
        )
    if side == "left":
        outer_c = c0 - random.randint(lo, hi)
        ring_a = (r0 + 1 + a, c0 - 1)
        ring_b = (r0 + 1 + b, c0 - 1)
        return _join(
            _hpath(ring_a[0], ring_a[1], outer_c),
            _vpath(ring_a[0], ring_b[0], outer_c),
            _hpath(ring_b[0], outer_c, ring_b[1]),
        )
    assert side == "right"
    outer_c = c1 + random.randint(lo, hi)
    ring_a = (r0 + 1 + a, c1 + 1)
    ring_b = (r0 + 1 + b, c1 + 1)
    return _join(
        _hpath(ring_a[0], ring_a[1], outer_c),
        _vpath(ring_a[0], ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )


def _h_pair_specs(h_side: str, v_side: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    cols = _pick_spaced_pair(1, W - 2, 2)
    rows = _pick_spaced_pair(1, H - 2, 2)
    if cols is None or rows is None:
        return None
    c_a, c_b = cols
    r_a, r_b = rows
    mid_r = (int(r_a) + int(r_b)) // 2
    inner = _merge_unique(
        _vpath(r_a, r_b, c_a),
        _vpath(r_a, r_b, c_b),
        _hpath(mid_r, c_a, c_b),
    )
    outer_h = _u_side_specs(h_side, (c_a, c_b), r0, c0, r1, c1)
    outer_v = _u_side_specs(v_side, (r_a, r_b), r0, c0, r1, c1)
    return inner, outer_h, outer_v


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

    if remaining >= 2 and H >= 5 and W >= 5 and random.random() < _H_PAIR_PROB:
        h_side = random.choice(["top", "bottom"])
        v_side = random.choice(["left", "right"])
        if color_i >= len(palette):
            return None
        color = palette[color_i]
        color_i += 1
        specs = _h_pair_specs(h_side, v_side, H, W, r0, c0, r1, c1)
        if specs is None:
            return None
        inner_h, outer_h, outer_v = specs
        irs, ics = zip(*inner_h)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_h, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_v, (color,), bg):
            return None
        _paint_cells(expected, inner_h, color)
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
    raise RuntimeError("failed to generate a valid a6f40cea D13 input")
