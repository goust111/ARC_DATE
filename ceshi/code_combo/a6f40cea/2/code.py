# Combo variant of a6f40cea (index 2)
# combo_spec: D1/4+D2/3
# Name: Z形变色路径
#
# Design analysis
# ----------------
# variant_analysis:
#   concretization_and_trace:
#     - 原题的主组件把两个框边触点映射成框内单拐角 L 路径；双色时颜色要么逐格交替，要么在单拐角处分段。
#     - 本融合版把主组件统一改成“对边触点 -> 三段双拐角路径”：
#       例如框内高 H=7、宽 W=10，若一个组件触到顶部第 2 列和底部第 7 列，则输出路径为
#       (0,2) 向下到中轴行，再水平走到第 7 列，再向下到 (6,7)，形成旋转后的 Z/S 形三段折线。
#       若组件改为触到左边第 1 行和右边第 5 行，则规则对称转置：先水平到中轴列，再竖直换行，再水平到另一端。
#     - 若该组件是双色，外部组件的两端 ring 触点都呈颜色 A，中间外脊呈颜色 B；输出对应为 A-B-A：
#       第一段在第一个内部拐角前保持 A，第一个拐角起切到 B，第二个拐角起切回 A。
#     - 与原题相比，差异同时发生在几何和着色两层：
#       路径不再是单肘 L，而是双肘 Z/S；双色也不再看步数奇偶，而只看是否跨过两个结构性拐角。
#   complexity_calibration:
#     - 为了让 2-4 个样例足够稳定可读，单个网格内所有主组件共用同一种“对边 -> 中轴折返”的方向
#       （要么全是上下对边，要么全是左右对边），这样解题者能在同一图里重复看到相同的几何规律。
#     - 保留单色/双色两类主组件，但强制每个样例至少 1 个双色组件，避免退化成“只是换了路径形状”。
#     - 不再生成原题的 U-pair 条纹；否则旧条纹规则会削弱新变体最关键的双拐角视觉信号。
#   implicit_assumptions_review:
#     - 原 solver 默认双色组件的两个 ring 端点颜色能锚定起止相位；本题不成立，因为 A-B-A 的两个端点同色。
#       新 solver 改为：从 ring 端点读出“端段颜色 A”，从组件剩余颜色读出“中段颜色 B”。
#     - 原 solver 的 `build_manhattan` 只需要一个拐角；本题必须显式构造两个拐角并在两个索引处分段上色。
#     - 原 generator 的双色外部路径按逐格交替或单拐角切色绘制；若沿用会让输入提示和输出规则错位。
#       因此新 generator 让外部组件也呈现“端段 A / 中脊 B / 端段 A”的分区。
#   placement_legality:
#     - 框、背景、干扰项的放置约束沿用原题。
#     - 主组件改为一对相对边触点；若走上下对边，则内部使用互不相交的列区间，若走左右对边，则改用互不相交的行区间，
#       以保证输出三段路径之间不重叠。
#     - 外部组件根据方向选择左右或上下的不同绕行槽位，仍通过 `_place_component` 做越界、重叠和 4 邻接检查。
#   visual_observability:
#     - 可见线索有两层：一是输出主路径都出现明显双拐角；二是双色组件的两端同色、中段异色。
#     - 输入里外部组件也把中脊染成第二种颜色，因此人类能直接把“中段变色”与“中轴折返”联系起来。
#   visual_cleanliness:
#     - 主组件全部走同一中轴，且投影区间彼此不交叉，避免输出中心区域出现难以分辨的覆盖。
#     - 外部组件只占框外单一方向的一组槽位，干扰项仍远离框，避免把 ring 触点信息埋掉。
#   color_change_inference:
#     - 双色组件始终只有两种颜色，且变化规则固定为 `A-B-A`，不依赖颜色编号、长度奇偶或组件顺序。
#     - 两端 ring 触点的同色性提供“首尾同色”证据，外部中脊提供“中段换色”证据，足以从少量样例归纳。
#   post_change_solver_assumptions:
#     - solver 假设真正的主组件恰有两个 ring 映射点，且这两个点位于一对相对的框边；generator 明确保证。
#     - solver 假设双色主组件的 ring 端段颜色是组件主色，另一种颜色只承担中段；generator 统一按该模式着色。
#     - solver 假设中轴行/列严格在框内部；generator 只在对应方向跨度足够时才启用该方向，因此中轴总在内部。

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

            cand = (int(h * w), r0, c0, r1, c1, int(bc), int(fill))
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
    return [(int(r), int(c)) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    return [(int(r), int(c)) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append((int(p[0]), int(p[1])))
    return out


def _classify_opposite_sides(
    p1: tuple[int, int], p2: tuple[int, int], H: int, W: int
) -> str | None:
    a, b = (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))
    if a[0] == 0 and b[0] == H - 1:
        return "tb"
    if a[0] == H - 1 and b[0] == 0:
        return "bt"
    if a[1] == 0 and b[1] == W - 1:
        return "lr"
    if a[1] == W - 1 and b[1] == 0:
        return "rl"
    return None


def _build_z_path(
    p1: tuple[int, int], p2: tuple[int, int], H: int, W: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    kind = _classify_opposite_sides(p1, p2, H, W)
    if kind is None:
        return None

    if kind in {"tb", "bt"}:
        top = (p1 if p1[0] == 0 else p2)
        bot = (p2 if top == p1 else p1)
        if top[1] == bot[1]:
            return None
        mid_r = int(H // 2)
        if not (0 < mid_r < H - 1):
            return None
        path = _join(
            _vpath(top[0], mid_r, top[1]),
            _hpath(mid_r, top[1], bot[1]),
            _vpath(mid_r, bot[0], bot[1]),
        )
        return path, [(int(mid_r), int(top[1])), (int(mid_r), int(bot[1]))]

    left = (p1 if p1[1] == 0 else p2)
    right = (p2 if left == p1 else p1)
    if left[0] == right[0]:
        return None
    mid_c = int(W // 2)
    if not (0 < mid_c < W - 1):
        return None
    path = _join(
        _hpath(left[0], left[1], mid_c),
        _vpath(left[0], right[0], mid_c),
        _hpath(right[0], mid_c, right[1]),
    )
    return path, [(int(left[0]), int(mid_c)), (int(right[0]), int(mid_c))]


def _paint_aba(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    switches: list[tuple[int, int]] | None = None,
) -> None:
    if not path:
        return
    if len(colors) == 1:
        rs, cs = zip(*path)
        grid[rs, cs] = int(colors[0])
        return

    assert switches is not None and len(switches) == 2
    end_color, mid_color = map(int, colors)
    i1 = int(path.index((int(switches[0][0]), int(switches[0][1]))))
    i2 = int(path.index((int(switches[1][0]), int(switches[1][1]))))
    assert i1 < i2
    vals = [
        end_color if idx < i1 else mid_color if idx < i2 else end_color
        for idx in range(len(path))
    ]
    rs, cs = zip(*path)
    grid[rs, cs] = vals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover cropped interior zigzags from colored traces around a framed rectangle.

    Input structure:
    - A background grid contains one 1px closed rectangular frame with uniform fill.
    - Outside the frame sit colored connected components, each touching the frame ring at two projected endpoints on opposite sides.

    Rule:
    - Output the cropped frame interior, initialized to the frame fill color.
    - Map each ring-touch pair onto the interior boundary, then connect the pair by a three-segment Manhattan Z/S path that bends at the interior mid-axis.
    - Monochrome components paint that full path in one color.
    - Two-color components use the ring-end color on the first and third segments and the other component color on the middle segment, switching exactly at the two internal corners.

    Key insight: The opposite-side touch pair fixes the Z/S geometry, while the component's color layout reveals an `A-B-A` segmentation rather than old alternation or single-elbow behavior.
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
        mr = max(0, min(outH - 1, int(rr - r_int0)))
        mc = max(0, min(outW - 1, int(cc - c_int0)))
        return int(mr), int(mc)

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
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(mr), int(mc), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {
        int(cid): sorted(set(map(int, g[clab == int(cid)].flatten())))
        for cid in range(1, int(cn) + 1)
    }

    for cid, pts in comp_pts.items():
        cols = comp_cols.get(int(cid), [])
        if len(cols) not in (1, 2):
            continue

        hist: dict[tuple[int, int], Counter] = defaultdict(Counter)
        ring_color_counter: Counter = Counter()
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
            ring_color_counter[int(col)] += 1

        coords = list(hist.keys())
        if len(coords) < 2:
            continue

        best = None
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                d = abs(coords[i][0] - coords[j][0]) + abs(coords[i][1] - coords[j][1])
                if best is None or d > best[0]:
                    best = (d, coords[i], coords[j])
        if best is None:
            continue
        _d, a, b = best

        built = _build_z_path(a, b, outH, outW)
        if built is None:
            continue
        path, corners = built

        if len(cols) == 1:
            _paint_aba(out, path, (int(cols[0]),))
            continue

        end_color = int(ring_color_counter.most_common(1)[0][0])
        middle_choices = [int(col) for col in cols if int(col) != int(end_color)]
        if len(middle_choices) != 1:
            continue
        _paint_aba(out, path, (int(end_color), int(middle_choices[0])), corners)

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
_DUAL_COLOR_PROB = 0.55
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 400


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    H, W = grid.shape
    own = set(coords)
    for r, c in own:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in own and 0 <= nr < H and 0 <= nc < W and comp_mask[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
    switches: list[tuple[int, int]] | None = None,
) -> bool:
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_aba(grid, path, colors, switches=switches)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _choose_disjoint_intervals(width: int, count: int) -> list[tuple[int, int]] | None:
    intervals = [(lo, hi) for lo in range(width) for hi in range(lo + 2, width)]
    random.shuffle(intervals)

    def rec(start: int, chosen: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
        if len(chosen) == count:
            return list(chosen)
        for idx in range(start, len(intervals)):
            lo, hi = intervals[idx]
            if any(not (hi < clo or chi < lo) for clo, chi in chosen):
                continue
            chosen.append((int(lo), int(hi)))
            got = rec(idx + 1, chosen)
            if got is not None:
                return got
            chosen.pop()
        return None

    return rec(0, [])


def _build_inner_tb(
    H: int, W: int, top_col: int, bottom_col: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    return _build_z_path((0, int(top_col)), (int(H - 1), int(bottom_col)), int(H), int(W))


def _build_inner_lr(
    H: int, W: int, left_row: int, right_row: int
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    return _build_z_path((int(left_row), 0), (int(right_row), int(W - 1)), int(H), int(W))


def _tb_outer_specs(
    top_col: int,
    bottom_col: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    side: str,
    offset: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    top_ring = (int(r0 - 1), int(c0 + 1 + top_col))
    bot_ring = (int(r1 + 1), int(c0 + 1 + bottom_col))
    top_row = int(r0 - offset)
    bot_row = int(r1 + offset)
    side_col = int(c0 - offset) if side == "left" else int(c1 + offset)

    outer = _join(
        _vpath(top_ring[0], top_row, top_ring[1]),
        _hpath(top_row, top_ring[1], side_col),
        _vpath(top_row, bot_row, side_col),
        _hpath(bot_row, side_col, bot_ring[1]),
        _vpath(bot_row, bot_ring[0], bot_ring[1]),
    )
    switches = [(int(top_row), int(side_col)), (int(bot_row), int(side_col))]
    return outer, switches


def _lr_outer_specs(
    left_row: int,
    right_row: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    *,
    side: str,
    offset: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    left_ring = (int(r0 + 1 + left_row), int(c0 - 1))
    right_ring = (int(r0 + 1 + right_row), int(c1 + 1))
    left_col = int(c0 - offset)
    right_col = int(c1 + offset)
    side_row = int(r0 - offset) if side == "top" else int(r1 + offset)

    outer = _join(
        _hpath(left_ring[0], left_ring[1], left_col),
        _vpath(left_ring[0], side_row, left_col),
        _hpath(side_row, left_col, right_col),
        _vpath(side_row, right_ring[0], right_col),
        _hpath(right_ring[0], right_col, right_ring[1]),
    )
    switches = [(int(side_row), int(left_col)), (int(side_row), int(right_col))]
    return outer, switches


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
        sr, sc = random.randint(1, H - 2), random.randint(1, W - 2)
        coords = [(int(sr), int(sc))]
        used = {(int(sr), int(sc))}
        for _ in range(size - 1):
            base = random.choice(coords)
            options = [
                (int(nr), int(nc))
                for nr, nc in neighbors4(*base)
                if 1 <= nr < H - 1 and 1 <= nc < W - 1 and (int(nr), int(nc)) not in used
            ]
            if not options:
                break
            pick = random.choice(options)
            used.add(pick)
            coords.append(pick)
        if any(r0 - gap <= rr <= r1 + gap and c0 - gap <= cc <= c1 + gap for rr, cc in coords):
            continue
        if _place_component(grid, comp_mask, coords, (int(color),), bg):
            return


def _gen_attempt() -> np.ndarray | None:
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
    r1, c1 = int(r0 + H + 1), int(c0 + W + 1)

    grid = np.full((rows, cols), int(bg), dtype=int)
    grid[r0, c0 : c1 + 1] = int(bc)
    grid[r1, c0 : c1 + 1] = int(bc)
    grid[r0 : r1 + 1, c0] = int(bc)
    grid[r0 : r1 + 1, c1] = int(bc)
    grid[r0 + 1 : r1, c0 + 1 : c1] = int(fill)

    expected = np.full((H, W), int(fill), dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((H, W), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    color_i = 0

    feasible_orients: list[str] = ["tb"]
    if H >= 6:
        feasible_orients.append("lr")
    orient = random.choice(feasible_orients)

    span = W if orient == "tb" else H
    max_components = 3 if span >= 9 else 2
    n_components = random.randint(2, max_components)
    intervals = _choose_disjoint_intervals(span, n_components)
    if intervals is None:
        return None

    if orient == "tb":
        route_slots = [("left", 2), ("left", 3), ("right", 2), ("right", 3)]
    else:
        route_slots = [("top", 2), ("top", 3), ("bottom", 2), ("bottom", 3)]
    random.shuffle(route_slots)
    if len(route_slots) < n_components:
        return None
    route_slots = route_slots[:n_components]

    dual_slots = {random.randrange(n_components)}
    for idx in range(n_components):
        if idx not in dual_slots and random.random() < _DUAL_COLOR_PROB:
            dual_slots.add(idx)

    comps: list[tuple[int, int, tuple[str, int], tuple[int, ...]]] = []
    for idx, (lo, hi) in enumerate(intervals):
        if random.random() < 0.5:
            a, b = int(lo), int(hi)
        else:
            a, b = int(hi), int(lo)

        want_dual = idx in dual_slots
        if want_dual:
            if color_i >= len(palette) - 1:
                return None
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            if color_i >= len(palette):
                return None
            colors = (int(palette[color_i]),)
            color_i += 1
        comps.append((int(a), int(b), route_slots[idx], colors))

    random.shuffle(comps)
    for a, b, slot, colors in comps:
        if orient == "tb":
            inner_built = _build_inner_tb(H, W, a, b)
        else:
            inner_built = _build_inner_lr(H, W, a, b)
        if inner_built is None:
            return None
        inner, inner_switches = inner_built
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        side, offset = slot
        if orient == "tb":
            outer, outer_switches = _tb_outer_specs(a, b, r0, c0, r1, c1, side=side, offset=offset)
        else:
            outer, outer_switches = _lr_outer_specs(a, b, r0, c0, r1, c1, side=side, offset=offset)
        if not _place_component(
            grid,
            comp_mask,
            outer,
            colors,
            bg,
            switches=outer_switches if len(colors) == 2 else None,
        ):
            return None

        _paint_aba(expected, inner, colors, switches=inner_switches if len(colors) == 2 else None)
        used_inner[irs, ics] = True

    if not any(len(colors) == 2 for *_rest, colors in comps):
        return None

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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
