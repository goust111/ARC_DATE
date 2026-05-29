# Combo variant of faa9f03d (index 17)
# combo_spec: D4/0+D10/1
# Name: 分配给最远可见线色 + 按距离从近到远逐步修复
#
# Design analysis
# ----------------
# 1. 具体化与行为追踪
#    - 原题里所有 marker 先一次性分配给最近可见线色, 然后每种线色只做一轮 gap fill.
#      这意味着某个长缺口如果一开始没有两端轴向支撑, 即使别的较近缺口补完后本可解锁它,
#      原题也不会回头再补.
#    - 组合后有两个相反的距离偏好:
#      a) D4/0: marker 的归属看“最远可见线色”, 不是最近.
#      b) D10/1: 真正执行修复时, gap 要按“距 marker 从近到远”逐个补, 每补一步都重算支撑.
#    - 一个最小水平例子:
#        颜色 1 的目标线是 row=5 上的 col 1..11.
#        输入保留 col={1,2,6,10,11}, 在 col=8 放红 marker, 删除 col={3,4,5,7,9}.
#        同时在 (4,8) 放一条别色竖线, 让 marker 最近看到它(dist=1), 但左右看到本色(dist=2).
#        原题/最近分配: marker 会错分给近处别色.
#        本组合的分配: marker 归给本色 1.
#        之后修复顺序不是一次性补全:
#          step1 补离 marker 最近的 gap: 6-8 间的 col7, 以及 8-10 间的 col9;
#          step2 重算后, col6 获得轴向支撑, 才能补更远的长缺口 col3..5.
#        所以“最远分配”负责把 marker 交给正确颜色, “最近先补”负责把同一条线按链式顺序修回去.
#
# 2. 复杂度校准
#    - 只保留红 marker, 不强行叠加 yellow. 否则“最远分配”和“逐步修复”同时与翻臂交织,
#      容易让局部区域过于拥挤, 肉眼难以判断是哪条规则在起作用.
#    - 每个样例至少放 2 个真正触发 D4 的红 marker: 最近可见色 != 最远可见色.
#    - 每个红 marker 都制造“近 gap + 远 gap”的两级链:
#      近 gap 长度固定为 1, 远 gap 长度至少 2. 这样 D10/1 的时序肉眼可见, 且不依赖隐藏 tie-break.
#    - 仍保留原题的 2-4 条折线、交叉、碎片度裁决与 crossing 重着色, 任务没有退化成单一路径修补.
#
# 3. 隐式假设审查
#    - 原 solver 假设“一轮 gap fill 足够”. 组合后不成立, 因为新样例明确需要:
#        先补近 gap -> 某个孤立锚点获得轴向支撑 -> 再补远 gap.
#    - 原 solver 用 candidates.sort()[0] 默认最近优先. 这里必须改成按 (-dist, support, ...)
#      选最远, 且黄色若出现也必须沿这个被选方向找 pivot.
#    - should_fill() 本身不需要改规则; 变化在于它现在会在多轮重算里被重复调用.
#      generator 必须保证远 gap 初始时确实不满足 should_fill, 但补完近 gap 后会满足.
#    - crossing 消歧依赖 comp_count 是“把 marker 当成所属线色后, 但在填补前”的碎片数.
#      generator 继续按这个口径构造 target.
#
# 4. 放置合法性
#    - 红 marker 只放在非 crossing 的直线内部像素上.
#    - marker 两侧都保留可见本色锚点, 且某个正交方向有距离 1 的异色干扰, 才能同时触发 D4.
#    - 远 gap 与近 gap 的被删单元都不能碰 crossing、别的 marker、别的 marker 邻域或别色显示像素.
#    - 多个 marker 的整套删格区域彼此隔离, 避免一枚 marker 的近 gap 成为另一枚 marker 的远 gap 支撑.
#
# 5. 视觉可观察性与整洁度
#    - 人类能在样例里重复看到同一种局部模板:
#        “marker 正上/下方紧贴异色, 左右或上下更远处才是本色” + “输出中本线会分两轮向外长回来”.
#    - 近 gap 总是比远 gap 更靠 marker, 远 gap 至少有两个空格, 所以 input/output 对比能直接看到
#      不是一次性补平, 而是由 marker 向外逐级修复.
#    - 去掉 yellow 可显著减少局部叠层; 交叉仍保留在别处, 保持原题视觉家族相似度.
#
# 6. 新 main() 的额外输入假设
#    - 若一个样例想体现 D10/1, 至少要存在某个 gap 在初始状态不可填, 但在较近 gap 填完后可填.
#      generator 用链式红 marker 强制这一点.
#    - 若一个样例想体现 D4/0, 至少要存在某个 marker 的 nearest/farthest 可见颜色不同.
#      generator 对每个被选 marker 都显式检查这个条件.
#    - 逐步修复的距离 tie 最好少见. generator 让真正起关键作用的远 gap 距离严格大于近 gap,
#      即便同一步里有两个 dist=1 的局部补格, 也不会影响“先近后远”的主逻辑.

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _pts(mask: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in np.argwhere(mask)}


def _axis_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r, c - 1) in mask or (r, c + 1) in mask
    return (r - 1, c) in mask or (r + 1, c) in mask


def _perp_support(mask: set[tuple[int, int]], r: int, c: int, axis: int) -> bool:
    if axis == 0:
        return (r - 1, c) in mask or (r + 1, c) in mask
    return (r, c - 1) in mask or (r, c + 1) in mask


def _component_count(shape: tuple[int, int], pts: set[tuple[int, int]]) -> int:
    mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _visible_candidates(
    grid: np.ndarray,
    pt: tuple[int, int],
) -> list[tuple[int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    colors = set(map(int, g.ravel())) - {bg}
    marker_colors = {2, 4} & colors
    line_colors = colors - marker_colors
    original = {color: _pts(g == color) for color in line_colors}
    h, w = g.shape
    r0, c0 = pt
    candidates: list[tuple[int, int, int, int, int]] = []
    for dr, dc in DIRS4:
        r, c = r0 + dr, c0 + dc
        dist = 1
        while 0 <= r < h and 0 <= c < w:
            value = int(g[r, c])
            if value in line_colors:
                axis = 0 if dr == 0 else 1
                supported = _axis_support(original[value], r, c, axis)
                candidates.append((dist, 0 if supported else 1, value, dr, dc))
                break
            if value == bg or value in marker_colors:
                r += dr
                c += dc
                dist += 1
                continue
            break
    return candidates


def _pick_visible(
    candidates: list[tuple[int, int, int, int, int]],
    *,
    farthest: bool,
) -> tuple[int, int, int, int, int] | None:
    if not candidates:
        return None
    if farthest:
        return min(candidates, key=lambda x: (-x[0], x[1], x[2], x[3], x[4]))
    return min(candidates, key=lambda x: (x[0], x[1], x[2], x[3], x[4]))


def _gap_cells(a: tuple[int, int], b: tuple[int, int], axis: int) -> list[tuple[int, int]]:
    r1, c1 = a
    r2, c2 = b
    if axis == 0:
        return [(r1, c) for c in range(c1 + 1, c2)]
    return [(r, c1) for r in range(r1 + 1, r2)]


def _repair_distance(
    marker_pts: set[tuple[int, int]],
    gap_cells: list[tuple[int, int]],
) -> int:
    if not gap_cells:
        return 10**9
    if not marker_pts:
        return 10**9
    return min(abs(r - mr) + abs(c - mc) for r, c in gap_cells for mr, mc in marker_pts)


def _solve(
    grid: np.ndarray,
    *,
    iterative: bool,
    return_meta: bool = False,
    validate: bool = True,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"

    marker_colors: set[int] = set()
    for color in colors:
        comps = components_mask(g == color, connectivity=4)
        assert comps, "foreground color must have at least one component"
        if max(len(comp) for comp in comps) == 1:
            marker_colors.add(int(color))

    line_colors = colors - marker_colors
    assert line_colors, "expected at least one non-marker line color"
    assert marker_colors, "expected singleton marker colors"

    original = {color: _pts(g == color) for color in line_colors}
    assignment: dict[tuple[int, int], int] = {}
    assign_dir: dict[tuple[int, int], tuple[int, int]] = {}

    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < h and 0 <= c < w:
                value = int(g[r, c])
                if value in line_colors:
                    axis = 0 if dr == 0 else 1
                    supported = _axis_support(original[value], r, c, axis)
                    candidates.append((dist, 0 if supported else 1, value, dr, dc))
                    break
                if value == bg or value in marker_colors:
                    r += dr
                    c += dc
                    dist += 1
                    continue
                break

        assert candidates, f"marker at {(r0, c0)} cannot see a line color"
        dist, unsupported, color, dr, dc = min(
            candidates,
            key=lambda x: (-x[0], x[1], x[2], x[3], x[4]),
        )
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    comp_count = {color: _component_count(g.shape, masks[color]) for color in line_colors}

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        r1, c1 = a
        r2, c2 = b
        gap = c2 - c1 - 1 if axis == 0 else r2 - r1 - 1
        if gap <= 0:
            return False

        a_axis = has_axis_support(color, r1, c1, axis)
        b_axis = has_axis_support(color, r2, c2, axis)
        if gap == 1:
            if a_axis or b_axis:
                return True
            a_perp = _perp_support(masks[color], r1, c1, axis)
            b_perp = _perp_support(masks[color], r2, c2, axis)
            return not (a_perp and b_perp)

        if (r1, c1) in marker_pts[color] and (r2, c2) in marker_pts[color]:
            return False
        return a_axis and b_axis

    fill_steps: list[dict[str, object]] = []

    if iterative:
        while True:
            candidates: list[tuple[int, int, int, int, int, int, int, tuple[int, int], tuple[int, int], list[tuple[int, int]]]] = []
            for color in sorted(line_colors):
                for r in range(h):
                    cols = sorted(c for rr, c in masks[color] if rr == r)
                    for c1, c2 in zip(cols, cols[1:]):
                        a = (r, c1)
                        b = (r, c2)
                        if not should_fill(color, a, b, axis=0):
                            continue
                        gap = _gap_cells(a, b, axis=0)
                        candidates.append(
                            (
                                _repair_distance(marker_pts[color], gap),
                                len(gap),
                                r,
                                c1,
                                0,
                                c2,
                                color,
                                a,
                                b,
                                gap,
                            )
                        )
                for c in range(w):
                    rows = sorted(r for r, cc in masks[color] if cc == c)
                    for r1, r2 in zip(rows, rows[1:]):
                        a = (r1, c)
                        b = (r2, c)
                        if not should_fill(color, a, b, axis=1):
                            continue
                        gap = _gap_cells(a, b, axis=1)
                        candidates.append(
                            (
                                _repair_distance(marker_pts[color], gap),
                                len(gap),
                                c,
                                r1,
                                1,
                                r2,
                                color,
                                a,
                                b,
                                gap,
                            )
                        )
            if not candidates:
                break
            dist, gap_len, anchor_major, anchor_minor, axis, other_coord, color, a, b, gap = min(candidates)
            masks[color].update(gap)
            fill_steps.append(
                {
                    "color": int(color),
                    "a": a,
                    "b": b,
                    "axis": int(axis),
                    "gap_len": int(gap_len),
                    "distance": int(dist),
                    "anchor_major": int(anchor_major),
                    "anchor_minor": int(anchor_minor),
                    "other_coord": int(other_coord),
                }
            )
    else:
        for color in sorted(line_colors):
            additions: set[tuple[int, int]] = set()
            for r in range(h):
                cols = sorted(c for rr, c in masks[color] if rr == r)
                for c1, c2 in zip(cols, cols[1:]):
                    a = (r, c1)
                    b = (r, c2)
                    if should_fill(color, a, b, axis=0):
                        gap = _gap_cells(a, b, axis=0)
                        additions.update(gap)
                        fill_steps.append(
                            {
                                "color": int(color),
                                "a": a,
                                "b": b,
                                "axis": 0,
                                "gap_len": len(gap),
                                "distance": _repair_distance(marker_pts[color], gap),
                            }
                        )

            for c in range(w):
                rows = sorted(r for r, cc in masks[color] if cc == c)
                for r1, r2 in zip(rows, rows[1:]):
                    a = (r1, c)
                    b = (r2, c)
                    if should_fill(color, a, b, axis=1):
                        gap = _gap_cells(a, b, axis=1)
                        additions.update(gap)
                        fill_steps.append(
                            {
                                "color": int(color),
                                "a": a,
                                "b": b,
                                "axis": 1,
                                "gap_len": len(gap),
                                "distance": _repair_distance(marker_pts[color], gap),
                            }
                        )
            masks[color].update(additions)

    control_ray = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in _pts(g == 4):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < h and 0 <= c < w:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned endpoint"

            old_start = (pivot[0] + dr, pivot[1] + dc)
            if old_start in masks[color]:
                stack = [old_start]
                old_arm: set[tuple[int, int]] = set()
                while stack:
                    q = stack.pop()
                    if q == pivot or q in old_arm or q not in masks[color]:
                        continue
                    old_arm.add(q)
                    qr, qc = q
                    for ndr, ndc in DIRS4:
                        stack.append((qr + ndr, qc + ndc))

                assert old_arm, f"yellow control at {control} found no old arm to erase"
                for q in old_arm:
                    masks[color].discard(q)
                    marker_pts[color].discard(q)
                    control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < h and 0 <= c < w:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)

    def marker_brackets(color: int, pt: tuple[int, int]) -> bool:
        r, c = pt
        same_row = [mc for mr, mc in marker_pts[color] if mr == r]
        if any(mc < c for mc in same_row) and any(mc > c for mc in same_row):
            return True
        same_col = [mr for mr, mc in marker_pts[color] if mc == c]
        return any(mr < r for mr in same_col) and any(mr > r for mr in same_col)

    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue

        original_crossing = [color for color in colors_here if int(g[pt]) == color]
        if original_crossing:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
        else:
            bracketed = [color for color in colors_here if marker_brackets(color, pt)]
            if bracketed:
                out[pt] = min(bracketed, key=lambda color: (-comp_count[color], color))
            else:
                out[pt] = min(colors_here)

    if validate:
        for color in line_colors:
            for comp in components_mask(out == color, connectivity=4):
                pts_set = set(map(tuple, comp))
                for r, c in pts_set:
                    nbrs = [
                        (nr, nc)
                        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if (nr, nc) in pts_set
                    ]
                    if len(nbrs) != 1:
                        continue
                    on_boundary = r == 0 or r == h - 1 or c == 0 or c == w - 1
                    if on_boundary:
                        nr, nc = nbrs[0]
                        dr, dc = nr - r, nc - c
                        if not ((r in (0, h - 1)) and (c in (0, w - 1))):
                            perp = (
                                (r == 0 and dr == 1)
                                or (r == h - 1 and dr == -1)
                                or (c == 0 and dc == 1)
                                or (c == w - 1 and dc == -1)
                            )
                            assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                        continue
                    adj_other = any(
                        0 <= r + dr < h
                        and 0 <= c + dc < w
                        and int(out[r + dr, c + dc]) not in (bg, color)
                        for dr, dc in DIRS4
                    )
                    assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

        color_cells = {c: set() for c in line_colors}
        for r in range(h):
            for c in range(w):
                value = int(out[r, c])
                if value in line_colors:
                    color_cells[value].add((r, c))
        crossing_colors = set()
        for r in range(h):
            for c in range(w):
                a = int(out[r, c])
                if a not in line_colors:
                    continue
                for b in line_colors - {a}:
                    if (
                        ((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b])
                        or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])
                    ):
                        crossing_colors.add(a)
                        crossing_colors.add(b)
        for color in line_colors:
            assert color in crossing_colors, f"color {color} has no crossing with any other color"

    if return_meta:
        return out, {
            "assignment": assignment,
            "assign_dir": assign_dir,
            "fill_steps": fill_steps,
            "marker_pts": marker_pts,
            "line_colors": sorted(line_colors),
            "marker_colors": sorted(marker_colors),
        }
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by assigning singleton markers to the farthest visible line color, then rebuilding the network one distance layer at a time.

    Input structure:
    - A rectangular grid with one dominant background color.
    - Non-marker colors form one-cell-thick horizontal and vertical polyline fragments that may cross.
    - Marker colors are exactly the foreground colors whose 4-connected components are all singleton pixels.
    - Red singletons sit inside broken line corridors; yellow singletons, when present, still sit beside endpoints that must flip to the marked side.

    Rule:
    - For every marker, scan orthogonally and assign it to the farthest visible line color, still preferring an endpoint that already has support along that scan axis when distances tie.
    - After those assignments, do not fill every eligible gap in one batch. Instead, repeatedly choose the repairable gap whose missing cells are nearest to one of that color's markers, fill only that gap, and then recompute which farther gaps have become supported.
    - Once all marker-driven gap repairs stabilize, execute any yellow-control arm flip using the assigned direction: erase the old straight arm beyond the pivot and extend the new arm from the pivot through the control cell to the boundary.
    - At overlaps, yellow-control rays override everything else; otherwise crossings are resolved by the pre-fill fragmentation counts, with lower color id breaking residual ties.

    Key insight:
    The puzzle mixes two opposite distance cues. A marker belongs to the line color that is farther away in line-of-sight, but the actual reconstruction radiates outward from that marker by repairing the nearest missing span first. The intended output only appears when those two distance preferences are coordinated rather than conflated.
    """
    return _solve(grid, iterative=True, return_meta=False, validate=True)  # type: ignore[return-value]


# --------------- generator (o5x, combo D4/0 + D10/1) ---------------

H = W = 12
BG = 0
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

CROSSING_COUNT_WEIGHTS = [3, 4]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12

RED_COUNT = 2
CHAIN_SUPPORT_IDX_RANGE = (4, 6)

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 1000


def _neighbors(pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(r + dr, c + dc) for dr, dc in DIRS4 if 0 <= r + dr < H and 0 <= c + dc < W]


def _segment(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    step = 1 if r1 >= r0 else -1
    return [(r, c0) for r in range(r0, r1 + step, step)]


def _path(points: list[tuple[int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for a, b in zip(points, points[1:]):
        seg = _segment(a, b)
        cells.extend(seg[1:] if cells else seg)
    return cells


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _orientation(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
        return None
    return "v" if dirs[0][0] else "h"


def _gen_component_count(pts: set[tuple[int, int]]) -> int:
    if not pts:
        return 0
    mask = np.zeros((H, W), dtype=bool)
    for r, c in pts:
        mask[r, c] = True
    return len(components_mask(mask, connectivity=4))


def _on_boundary(r: int, c: int) -> bool:
    return r == 0 or r == H - 1 or c == 0 or c == W - 1


def _valid_path(cells: list[tuple[int, int]]) -> bool:
    pts = set(cells)
    if len(pts) != len(cells) or len(pts) < 2:
        return False
    for r, c in pts:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if len(_dirs_in(pts, (r, c))) > 2:
            return False
    if _gen_component_count(pts) != 1:
        return False
    for r, c in pts:
        if not _on_boundary(r, c):
            continue
        for dr, dc in _dirs_in(pts, (r, c)):
            nr, nc = r + dr, c + dc
            if (r in (0, H - 1) and nr == r) or (c in (0, W - 1) and nc == c):
                return False
        nbrs = _dirs_in(pts, (r, c))
        if len(nbrs) != 1:
            continue
        if (r in (0, H - 1)) and (c in (0, W - 1)):
            continue
        dr, dc = nbrs[0]
        if not (
            (r == 0 and dr == 1)
            or (r == H - 1 and dr == -1)
            or (c == 0 and dc == 1)
            or (c == W - 1 and dc == -1)
        ):
            return False
    for r, c in pts:
        if len(_dirs_in(pts, (r, c))) == 1 and not _on_boundary(r, c):
            return False
    return True


def _rc(ax: int, major: int, minor: int) -> tuple[int, int]:
    return (major, minor) if ax == 0 else (minor, major)


def _dim(ax: int) -> int:
    return H if ax == 0 else W


def _random_span(limit: int) -> tuple[int, int]:
    if random.random() < 0.5:
        return 0, random.randint(MIN_SPAN_LEN, limit - 1)
    return random.randint(0, limit - MIN_SPAN_LEN - 1), limit - 1


def _build_polyline_waypoints(ax: int) -> list[tuple[int, int]]:
    pieces = random.choices(
        range(1, len(POLYLINE_PIECE_WEIGHTS) + 1),
        weights=POLYLINE_PIECE_WEIGHTS,
    )[0]
    a0, a1 = _random_span(_dim(ax))
    pieces = min(pieces, max(1, a1 - a0))
    minors = random.sample(range(1, _dim(1 - ax) - 1), pieces)
    bends = sorted(random.sample(range(a0 + 1, a1), pieces - 1)) if pieces > 1 else []
    pts = [_rc(ax, a0, minors[0])]
    for i, bend in enumerate(bends):
        pts.append(_rc(ax, bend, minors[i]))
        pts.append(_rc(ax, bend, minors[i + 1]))
    pts.append(_rc(ax, a1, minors[-1]))
    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, random.choice([0, W - 1])))
        else:
            pts.insert(0, (random.choice([0, H - 1]), c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, random.choice([0, W - 1])))
        else:
            pts.append((random.choice([0, H - 1]), cn))
    return pts


def _build_arm(
    ax: int,
    main_coord: int,
    cross_lo: int,
    cross_hi: int,
    is_start: bool,
) -> list[tuple[int, int]]:
    dim_minor = _dim(1 - ax)
    arm_choices = [v for v in range(1, dim_minor - 1) if abs(v - main_coord) >= ARM_MIN_OFFSET]
    if not arm_choices:
        return []
    minor = random.choice(arm_choices)
    if is_start:
        if cross_lo <= 1:
            return []
        arm_major = random.randint(1, cross_lo - 1)
        return [_rc(ax, 0, minor), _rc(ax, arm_major, minor), _rc(ax, arm_major, main_coord)]
    if cross_hi >= _dim(ax) - 2:
        return []
    arm_major = random.randint(cross_hi + 1, _dim(ax) - 2)
    return [
        _rc(ax, arm_major, main_coord),
        _rc(ax, arm_major, minor),
        _rc(ax, _dim(ax) - 1, minor),
    ]


def _path_through(cell: tuple[int, int], orient: str) -> list[tuple[int, int]] | None:
    r, c = cell
    ax = 0 if orient == "v" else 1
    major = r if ax == 0 else c
    minor = c if ax == 0 else r
    if major == 0 or major == _dim(ax) - 1:
        return None
    lo = 0 if random.random() < FULL_SPAN_PROB else random.randint(0, major - 1)
    hi = _dim(ax) - 1 if random.random() < FULL_SPAN_PROB else random.randint(major + 1, _dim(ax) - 1)
    pts = [_rc(ax, lo, minor), _rc(ax, hi, minor)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms") and lo > 0:
        arm = _build_arm(ax, minor, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms") and hi < _dim(ax) - 1:
        arm = _build_arm(ax, minor, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    if not _on_boundary(*pts[0]):
        pts[0] = _rc(ax, 0, minor)
    if not _on_boundary(*pts[-1]):
        pts[-1] = _rc(ax, _dim(ax) - 1, minor)
    cells = _path(pts)
    return cells if _valid_path(cells) and _orientation(set(cells), cell) == orient else None


def _spaced_values(limit: int, count: int) -> list[int] | None:
    choices = list(range(1, limit - 1))
    for _ in range(SPACED_VALUES_RETRIES):
        vals = sorted(random.sample(choices, count))
        if all(b - a >= 2 for a, b in zip(vals, vals[1:])):
            return vals
    return None


def _serpentine_path(count: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    ax = random.randint(0, 1)
    lanes = _spaced_values(_dim(ax), count)
    if lanes is None:
        return None
    if random.random() < 0.5:
        m0, m1 = 1, random.randint(SERPENTINE_MINOR_LO, _dim(1 - ax) - 2)
    else:
        m0, m1 = random.randint(1, SERPENTINE_MINOR_HI), _dim(1 - ax) - 2
    cross_minor = random.randint(m0 + CROSS_MINOR_MARGIN, m1 - CROSS_MINOR_MARGIN)

    start_at_m0 = random.random() < 0.5
    pts = [_rc(ax, lanes[0], m0 if start_at_m0 else m1)]
    fwd = start_at_m0
    for i, lane in enumerate(lanes):
        end_m = m1 if fwd else m0
        if pts[-1] != _rc(ax, lane, end_m):
            pts.append(_rc(ax, lane, end_m))
        if i + 1 < len(lanes):
            pts.append(_rc(ax, lanes[i + 1], end_m))
            fwd = not fwd

    if not _on_boundary(*pts[0]):
        r0, c0 = pts[0]
        if ax == 0:
            pts.insert(0, (r0, 0 if c0 == m0 else W - 1))
        else:
            pts.insert(0, (0 if r0 == m0 else H - 1, c0))
    if not _on_boundary(*pts[-1]):
        rn, cn = pts[-1]
        if ax == 0:
            pts.append((rn, 0 if cn == m0 else W - 1))
        else:
            pts.append((0 if rn == m0 else H - 1, cn))

    cells = _path(pts)
    crossings = [_rc(ax, lane, cross_minor) for lane in lanes]
    return (cells, crossings) if _valid_path(cells) else None


def _path_crossing_all(crossings: list[tuple[int, int]]) -> list[tuple[int, int]] | None:
    rows = {r for r, _ in crossings}
    cols = {c for _, c in crossings}
    if len(cols) == 1:
        ax, main_coord, cross_vals = 0, next(iter(cols)), rows
    elif len(rows) == 1:
        ax, main_coord, cross_vals = 1, next(iter(rows)), cols
    else:
        return None

    lo, hi = min(cross_vals), max(cross_vals)
    if lo <= 1 or hi >= _dim(ax) - 2:
        return None

    pts = [_rc(ax, 0, main_coord), _rc(ax, _dim(ax) - 1, main_coord)]
    flavor = random.choice(("straight", "start_arm", "end_arm", "both_arms"))
    if flavor in ("start_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=True)
        if arm:
            pts = arm + pts[1:]
    if flavor in ("end_arm", "both_arms"):
        arm = _build_arm(ax, main_coord, lo, hi, is_start=False)
        if arm:
            pts = pts[:-1] + arm
    cells = _path(pts)
    return cells if _valid_path(cells) else None


def _can_add(logical: dict[int, set[tuple[int, int]]], color: int, cells: list[tuple[int, int]]) -> bool:
    if not _valid_path(cells):
        return False
    pts = set(cells)
    for pt in pts:
        if pt in logical[color]:
            return False
        if any(nb in logical[color] for nb in _neighbors(pt)):
            return False
        if sum(pt in pset for pset in logical.values()) >= 2:
            return False
    return True


def _crossing_cells(logical: dict[int, set[tuple[int, int]]]) -> list[tuple[int, int]]:
    cells = set().union(*logical.values())
    return [pt for pt in cells if sum(pt in logical[color] for color in logical) > 1]


def _valid_crossings(logical: dict[int, set[tuple[int, int]]], dominant: int) -> bool:
    crossings = _crossing_cells(logical)
    if not 1 <= len(crossings) <= MAX_CROSSINGS:
        return False
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if len(colors) != 2 or dominant not in colors:
            return False
        orientations = [_orientation(logical[color], pt) for color in colors]
        if None in orientations or sorted(orientations) != ["h", "v"]:
            return False
    return True


def _find_crossing_point(logical: dict[int, set[tuple[int, int]]]) -> tuple[tuple[int, int], str] | None:
    candidates = []
    for pts in logical.values():
        for r, c in pts:
            orient = _orientation(pts, (r, c))
            if orient is None:
                continue
            perp = "v" if orient == "h" else "h"
            major = r if perp == "v" else c
            if major <= 1 or major >= (H if perp == "v" else W) - 2:
                continue
            candidates.append(((r, c), perp))
    random.shuffle(candidates)
    return candidates[0] if candidates else None


def _build_logical() -> tuple[dict[int, set[tuple[int, int]]], dict[int, list[set[tuple[int, int]]]], int] | None:
    desired_crossings = random.choices(
        range(1, len(CROSSING_COUNT_WEIGHTS) + 1),
        weights=CROSSING_COUNT_WEIGHTS,
    )[0]
    if min(MAX_COLORS, TOTAL_COMPONENTS[1] - desired_crossings) < 4:
        return None
    colors = random.sample(LINE_PALETTE, 4)
    dominant = random.choice(colors)
    logical: dict[int, set[tuple[int, int]]] = {c: set() for c in colors}
    paths: dict[int, list[set[tuple[int, int]]]] = {c: [] for c in colors}

    built = _serpentine_path(desired_crossings)
    if built is None:
        return None
    dom_cells, forced_crossings = built
    if not _can_add(logical, dominant, dom_cells):
        return None
    logical[dominant].update(dom_cells)
    paths[dominant].append(set(dom_cells))

    crossing_color = random.choice([c for c in colors if c != dominant])
    cross_cells = _path_crossing_all(forced_crossings)
    if cross_cells is None or not _can_add(logical, crossing_color, cross_cells):
        return None
    logical[crossing_color].update(cross_cells)
    paths[crossing_color].append(set(cross_cells))

    for color in [c for c in colors if c not in (dominant, crossing_color)]:
        cells = None
        for _ in range(EXTRA_POLYLINE_RETRIES):
            cp = _find_crossing_point(logical)
            if cp is None:
                break
            trial = _path_through(cp[0], cp[1])
            if trial is not None and _can_add(logical, color, trial):
                cells = trial
                break
        if cells is None:
            return None
        logical[color].update(cells)
        paths[color].append(set(cells))

    if not _valid_crossings(logical, dominant):
        return None
    return logical, paths, dominant


def _render_target(
    logical: dict[int, set[tuple[int, int]]],
    comp_count: dict[int, int],
) -> np.ndarray:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
        else:
            out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _crossing_display(
    logical: dict[int, set[tuple[int, int]]],
    dominant: int,
    crossings: set[tuple[int, int]],
) -> dict[tuple[int, int], int] | None:
    display: dict[tuple[int, int], int] = {}
    for pt in crossings:
        colors = [color for color in logical if pt in logical[color]]
        if dominant not in colors or len(colors) != 2:
            return None
        display[pt] = colors[0] if colors[1] == dominant else colors[1]
    return display


def _render_base_grid(
    logical: dict[int, set[tuple[int, int]]],
    crossings: set[tuple[int, int]],
    crossing_display: dict[tuple[int, int], int],
) -> np.ndarray | None:
    grid = np.full((H, W), BG, dtype=int)
    for color, pts in logical.items():
        for pt in pts:
            if pt in crossings and crossing_display[pt] != color:
                continue
            if grid[pt] not in (BG, color):
                return None
            grid[pt] = color
    return grid


def _ray_along(
    pts: set[tuple[int, int]],
    pt: tuple[int, int],
    direction: tuple[int, int],
) -> list[tuple[int, int]]:
    ray: list[tuple[int, int]] = []
    r, c = pt
    dr, dc = direction
    while (r + dr, c + dc) in pts:
        r += dr
        c += dc
        ray.append((r, c))
    return ray


def _batch_farthest(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, iterative=False, return_meta=False, validate=False)  # type: ignore[return-value]


def _d4_effect_count(inp: np.ndarray, out: np.ndarray | None = None) -> int:
    out = main(inp) if out is None else out
    count = 0
    for pt in map(tuple, np.argwhere(inp == 2)):
        target_color = int(out[pt])
        nearest = _pick_visible(_visible_candidates(inp, pt), farthest=False)
        farthest = _pick_visible(_visible_candidates(inp, pt), farthest=True)
        if nearest is None or farthest is None:
            continue
        if farthest[2] == target_color and nearest[2] != target_color:
            count += 1
    return count


def _multi_step_effect(inp: np.ndarray) -> tuple[int, int]:
    combo_out, meta = _solve(inp, iterative=True, return_meta=True)  # type: ignore[misc]
    batch_out = _batch_farthest(inp)
    diff = int(np.count_nonzero(combo_out != batch_out))
    long_steps = sum(int(step["distance"]) >= 2 for step in meta["fill_steps"])
    return diff, long_steps


def _list_chain_red_candidates(
    logical: dict[int, set[tuple[int, int]]],
    base: np.ndarray,
    crossings: set[tuple[int, int]],
) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for color, pts in logical.items():
        for pt in pts:
            if pt in crossings or int(base[pt]) != color:
                continue
            dirs = _dirs_in(pts, pt)
            if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
                continue
            for chain_dir, support_dir in ((dirs[0], dirs[1]), (dirs[1], dirs[0])):
                chain_ray = _ray_along(pts, pt, chain_dir)
                support_ray = _ray_along(pts, pt, support_dir)
                if len(chain_ray) < CHAIN_SUPPORT_IDX_RANGE[0] + 2 or len(support_ray) < 3:
                    continue
                support_keep = support_ray[1:3]
                if len(support_keep) < 2:
                    continue
                if any(q in crossings or int(base[q]) != color for q in support_keep):
                    continue
                if support_ray[0] in crossings or int(base[support_ray[0]]) != color:
                    continue
                max_idx = min(len(chain_ray) - 2, CHAIN_SUPPORT_IDX_RANGE[1])
                for support_idx in range(CHAIN_SUPPORT_IDX_RANGE[0], max_idx + 1):
                    anchor = chain_ray[1]
                    far_support = chain_ray[support_idx : support_idx + 2]
                    if len(far_support) < 2:
                        continue
                    removed = {support_ray[0], chain_ray[0], *chain_ray[2:support_idx]}
                    if len(removed) < 4:
                        continue
                    if any(q in crossings or int(base[q]) != color for q in removed):
                        continue
                    if anchor in crossings or int(base[anchor]) != color:
                        continue
                    if any(q in crossings or int(base[q]) != color for q in far_support):
                        continue

                    trial = base.copy()
                    for q in removed:
                        trial[q] = BG
                    trial[pt] = 2

                    if not _marker_isolation_ok(trial):
                        continue
                    nearest = _pick_visible(_visible_candidates(trial, pt), farthest=False)
                    farthest = _pick_visible(_visible_candidates(trial, pt), farthest=True)
                    if nearest is None or farthest is None:
                        continue
                    if farthest[2] != color or nearest[2] == color:
                        continue

                    combo_out = main(trial)
                    batch_out = _batch_farthest(trial)
                    diff = int(np.count_nonzero(combo_out != batch_out))
                    if diff <= 0:
                        continue

                    d4_margin = int(farthest[0] - nearest[0])
                    far_gap_len = support_idx - 2
                    candidates.append(
                        {
                            "color": int(color),
                            "pt": pt,
                            "removed": removed,
                            "score": (diff, far_gap_len, d4_margin),
                            "support_keep": set(support_keep) | {anchor} | set(far_support),
                        }
                    )
    random.shuffle(candidates)
    candidates.sort(key=lambda item: item["score"], reverse=True)
    return candidates


def _transform_pt(
    pt: tuple[int, int],
    *,
    flip_h: bool,
    flip_v: bool,
) -> tuple[int, int]:
    r, c = pt
    if flip_v:
        r = H - 1 - r
    if flip_h:
        c = W - 1 - c
    return int(r), int(c)


def _build_template_case() -> tuple[
    dict[int, set[tuple[int, int]]],
    list[dict[str, object]],
]:
    colors = sorted(random.sample(LINE_PALETTE, 3))
    left_color, right_color = random.sample(colors[:2], 2)
    top_color = colors[2]
    logical_raw = {
        top_color: set(_path([(5, 0), (5, 11)])),
        left_color: set(_path([(0, 3), (11, 3)])),
        right_color: set(_path([(0, 8), (11, 8)])),
    }
    marker_specs_raw = [
        {
            "color": left_color,
            "pt": (6, 3),
            "removed": {(2, 3), (3, 3), (7, 3)},
        },
        {
            "color": right_color,
            "pt": (6, 8),
            "removed": {(2, 8), (3, 8), (7, 8)},
        },
    ]

    flip_h = random.random() < 0.5
    flip_v = False

    logical: dict[int, set[tuple[int, int]]] = {}
    for color, pts in logical_raw.items():
        mapped = {_transform_pt(pt, flip_h=flip_h, flip_v=flip_v) for pt in pts}
        logical[color] = mapped
        assert _valid_path(sorted(mapped)), "template path must stay valid after mirroring"

    marker_specs: list[dict[str, object]] = []
    for spec in marker_specs_raw:
        marker_specs.append(
            {
                "color": int(spec["color"]),
                "pt": _transform_pt(spec["pt"], flip_h=flip_h, flip_v=flip_v),
                "removed": {
                    _transform_pt(pt, flip_h=flip_h, flip_v=flip_v)
                    for pt in set(spec["removed"])
                },
            }
        )
    return logical, marker_specs


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    marker_specs: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    if len(crossings) != 2:
        return None

    color_list = sorted(logical)
    top_out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in color_list if pt in logical[color]]
        if len(colors_here) == 1:
            top_out[pt] = colors_here[0]
        else:
            top_out[pt] = min(colors_here)

    inp = np.full((H, W), BG, dtype=int)
    assigned_markers: list[tuple[int, tuple[int, int]]] = []
    removed_all = set()
    for spec in marker_specs:
        removed_all.update(set(spec["removed"]))
        assigned_markers.append((int(spec["color"]), tuple(spec["pt"])))
    marker_by_pt = {pt: color for color, pt in assigned_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_all:
                continue
            if pt in marker_by_pt:
                continue
            inp[pt] = color
    for _color, pt in assigned_markers:
        inp[pt] = 2

    if not _marker_isolation_ok(inp):
        return None

    comp_count_masks = {c: _pts(inp == c) for c in logical}
    for color, pt in assigned_markers:
        comp_count_masks[color].add(pt)
    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    target = _render_target(logical, comp_count)

    for pt in crossings:
        colors_here = [color for color in color_list if pt in logical[color]]
        winner = int(target[pt])
        loser = next(color for color in colors_here if color != winner)
        if pt in removed_all or pt in marker_by_pt:
            return None
        inp[pt] = loser
        if inp[pt] == target[pt]:
            return None

    solved = main(inp)
    batch = _batch_farthest(inp)
    if not np.array_equal(solved, target):
        return None
    if np.array_equal(solved, batch):
        return None

    if _d4_effect_count(inp, solved) < RED_COUNT:
        return None
    multi_diff, long_steps = _multi_step_effect(inp)
    if multi_diff <= 0 or long_steps < RED_COUNT:
        return None

    counts = {c: len(components_mask(target == c, connectivity=4)) for c in color_list}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
        return None
    return inp, target


def _gen_attempt() -> np.ndarray | None:
    logical, marker_specs = _build_template_case()
    derived = _derive_input(logical, marker_specs)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid faa9f03d combo D4/0+D10/1 input")
