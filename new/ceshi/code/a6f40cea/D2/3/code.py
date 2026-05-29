# Variant of a6f40cea
# Dimension: D2 — 轨迹着色模式 (trajectory)
# Change: 单色组件用纯色填充路径；双色组件用两色交替填充，相位由端点颜色锚定
#   → 反射变色：路径在结构拐角处切换颜色，拐角前一色，拐角及其后一色
#
# Design analysis
# ----------------
# variant_analysis:
#   concretization_and_trace:
#     - 输入仍然是一个 1px 封闭矩形框，外侧有 2-3 个接触框外环的开放矩形组件，输出仍然裁剪为框内部。
#     - 单色组件完全不变：映射为框内的纯色直线或 L 形曼哈顿路径。
#     - 双色 L 组件改为“分段双色”，而不是逐格交替：
#       输入外部组件从第一个 ring 端点走到其外侧主拐角之前保持颜色 A，从该主拐角开始到第二个 ring 端点改为颜色 B。
#       输出内部路径从第一个映射端点走到内部肘点之前保持颜色 A，从内部肘点开始到第二个端点改为颜色 B。
#     - 具体小例子：若映射端点为 (0,4) 和 (3,0)，原题双色输出会沿整条 L 路径 A/B/A/B... 交替。
#       本变体中，若 top 端点颜色为 8、left 端点颜色为 4，则输出会是：
#       从 (0,4) 向下到拐角前都填 8，拐角格本身以及之后向左到 (3,0) 都填 4。
#       差异不再依赖路径长度奇偶，而只依赖“是否已经转弯”。
#   complexity_calibration:
#     - 不引入第三种颜色，也不让不同组件使用不同切色规则；双色组件始终只在一个结构性拐角处切色。
#     - 保留原题的 L 型与 U 型两类几何，避免形状复杂度漂移；U 型仍保持单色，不承担新规则。
#     - 为保证训练样例肉眼可见该维度变化，生成器强制每个样本至少包含一个双色 L 组件。
#   implicit_assumptions_review:
#     - 原 solver 的双色分支依赖“整条路径逐格交替，且相位由两个端点颜色决定”，对应旧代码里的 `paint_alt`
#       与 `end_color` 奇偶推断。这一前提在变体中不成立，必须改成“按拐角分段上色”。
#     - 原 `_paint_path` 对双色外部组件也做逐格交替；若保留会让输入提示和输出规则脱节，因此生成器同步改成按指定拐角切色。
#     - 原始 `build_manhattan` 只返回路径。变体需要显式拿到内部肘点，否则无法定义颜色边界；因此这里让它同时返回 corner。
#     - 原单色分支、U-pair 汇总、框检测、ring 映射都不依赖交替奇偶，保持不变。
#   placement_legality:
#     - 外部组件仍通过 `_place_component` 做不重叠、4 邻接隔离、边界检查。
#     - 双色 L 的外部颜色切换点固定为开放矩形的远侧主角 `(R,C)`，该点一定在 `outer` 路径上。
#     - 内部双色颜色切换点固定为曼哈顿 L 的唯一内部肘点；`_l_specs` 继续保证肘点不在边界角上，便于可视识别。
#     - 若放置失败或 solver 复验不一致，生成器整次重试，维持 output-first 正确性。
#   visual_observability:
#     - 变体的关键视觉信号是“双色不再棋盘式闪烁，而是拐角前后整段换色”。
#     - 每个样本至少 1 个双色 L，可从 2-4 个样例中稳定看到：输入外部主角切色，输出内部肘点切色。
#     - 禁止基于颜色 id 排序或做隐藏映射；颜色只作为可见标签，通过 ring 端点和转弯位置传递。
#   visual_cleanliness:
#     - 组件数量、尺寸、边距沿用原题，避免过密叠加。
#     - U 型条纹仍是单色整条，双色仅出现在 L 组件，观察时不会与原有整列/整行条纹混淆。
#   color_change_inference:
#     - 双色组件最多两色，且切换总发生在几何拐角处，不依赖步数奇偶或颜色数值。
#     - 输出中的切色方向由输入 ring 两端显式给出：哪一端是什么颜色，对应内部路径转弯前/后的两段颜色。
#   post_change_solver_assumptions:
#     - 双色组件必须是非退化 L：两个映射端点位于相邻边，不在同一行/列；生成器保证这一点。
#     - 双色组件的两个 ring 端点主色必须不同；生成器按主角切色，所以自然满足。
#     - 选中的曼哈顿角必须严格在内部；`inside_corner(...)` 与原题一致，生成器的 `_l_specs` 也按此构造。

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
    Reconstruct paths inside a framed rectangle from external open-rect traces.

    Input structure:
    - A background grid contains one 1px closed rectangular frame with uniform fill.
    - Outside the frame sit colored 4-connected components touching the frame ring.

    Rule:
    - Output the cropped interior of the frame, initialized with the frame fill color.
    - Map each component's ring-touch endpoints onto the interior boundary.
    - Monochrome components still create solid Manhattan stripes or L-paths.
    - Two-color L components no longer alternate cell-by-cell: they keep the first endpoint's color until the elbow, then switch to the second endpoint's color at the elbow and after it.

    Key insight: For双色 components, the ring-touch colors identify the two leg colors, while the unique interior elbow determines where the color transition happens.
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
        p1: tuple[int, int], p2: tuple[int, int], cornerA: bool
    ) -> tuple[list[tuple[int, int]], tuple[int, int] | None]:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb:
            step = 1 if cb >= ca else -1
            return ([(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)], None)
        if ca == cb:
            step = 1 if rb >= ra else -1
            return ([(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)], None)
        if cornerA:
            corner = (int(rb), int(ca))
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb) + step, step)]
            return path, corner
        corner = (int(ra), int(cb))
        step = 1 if cb >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path, corner

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb) = p1, p2
        if ra == rb or ca == cb:
            return True
        corner = (rb, ca) if cornerA else (ra, cb)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    def paint_corner_split(
        path_pts: list[tuple[int, int]],
        corner: tuple[int, int],
        first_color: int,
        second_color: int,
    ) -> None:
        assert corner in path_pts
        switch_idx = int(path_pts.index(corner))
        for idx, (rr, cc) in enumerate(path_pts):
            out[int(rr), int(cc)] = int(first_color if idx < switch_idx else second_color)

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
            uniq = coords
            rows = {r for r, _c in uniq}
            cols2 = {c for _r, c in uniq}
            if len(rows) == 1 or len(cols2) == 1:
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
            pathA, _corner_a = build_manhattan(p1, p2, True)
            pathB, _corner_b = build_manhattan(p1, p2, False)
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
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])
            if ha == hb:
                ha, hb = map(int, cols)

            pathA, corner_a = build_manhattan(a, b, True)
            pathB, corner_b = build_manhattan(a, b, False)
            if inside_corner(a, b, False) and not inside_corner(a, b, True):
                path, corner = pathB, corner_b
            else:
                path, corner = pathA, corner_a

            assert corner is not None
            paint_corner_split(path, corner, ha, hb)
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
                if not cs:
                    continue
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
        pathA, _corner_a = build_manhattan(p1, p2, True)
        pathB, _corner_b = build_manhattan(p1, p2, False)
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
_U_PAIR_PROB = 0.25
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


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    switch_at: tuple[int, int] | None = None,
) -> None:
    """Paint a solid path or a two-color path that switches at a designated corner."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
        return
    assert switch_at is not None and switch_at in path
    switch_idx = int(path.index(switch_at))
    vals = [colors[0] if i < switch_idx else colors[1] for i in range(len(path))]
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
    switch_at: tuple[int, int] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors, switch_at=switch_at)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner/outer paths plus the designated color-switch corners for one L-component."""
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
    inner_corner = (int(r), int(c))
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))

    outer_corner = (int(R), int(C))
    outer = _join(
        _vpath(ring_a[0], R, ring_a[1]),
        _hpath(R, ring_a[1], C),
        _vpath(R, ring_b[0], C),
        _hpath(ring_b[0], C, ring_b[1]),
    )
    return inner, outer, inner_corner, outer_corner


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
    """Single attempt to generate a valid input grid. Returns None on failure."""
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice(
        [x for x in range(10) if x not in (bg, bc)]
    )
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
    dual_left_required = 1

    if remaining >= 3 and random.random() < _U_PAIR_PROB:
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
    dual_slots = set(random.sample(range(remaining), k=min(dual_left_required, remaining)))
    for idx, kind in enumerate(corners):
        need_dual = idx in dual_slots
        want_dual = need_dual or (remaining > dual_left_required and random.random() < _DUAL_COLOR_PROB)
        if want_dual:
            if color_i >= len(palette) - 1:
                return None
            colors = (palette[color_i], palette[color_i + 1])
            color_i += 2
            dual_left_required = max(0, dual_left_required - 1)
        else:
            if color_i >= len(palette):
                return None
            colors = (palette[color_i],)
            color_i += 1

        inner, outer, inner_corner, outer_corner = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer, colors, bg, switch_at=outer_corner if len(colors) == 2 else None):
            return None
        _paint_path(expected, inner, colors, switch_at=inner_corner if len(colors) == 2 else None)
        used_inner[irs, ics] = True

    if dual_left_required != 0:
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea input")
