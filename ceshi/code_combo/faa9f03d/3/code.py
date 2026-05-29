# Combo variant of faa9f03d (index 3)
# combo_spec: D1/2+D5/1
# Name: 反射旧臂 + 交叉处选线段更长颜色
#
# 融合设计分析
# -------------
# a) 具体化与行为追踪
#    - 红标记仍然表示“这格本来属于某条线”，先按原题的视线分配恢复颜色，再参与普通 gap fill。
#    - 黄标记不再触发“删旧臂+射线到边界”，而是落在某个拐角 pivot 的目标臂第一格；pivot 的另一条垂直臂仍可见。
#    - solver 先阻止普通 gap fill 把黄色定向缺口直接补回，再从 pivot 量出可见垂直源臂长度，把同样长度复制到黄色一侧。
#    - 这一复制出来的前缀允许穿过别色已有直线，于是会制造“输入里不存在、输出里重新出现”的交叉。
#    - 这些交叉不再由“黄色优先”或“碎片度更高者”决定，而是比较两种颜色在该格穿过的局部直线段长度；更长者显示。
#    - 例：蓝线在 pivot 处向上可见 3 格，向右原本有 5 格；输入把右侧前 3 格去掉，并把第 1 格改成黄。若这 3 格中第 2 格与一条长度 2 的绿竖线相交，则输出恢复蓝色并显示蓝；若相交的是长度 6 的绿竖线，则交叉格保持绿，但蓝线在其下方被恢复。
# b) 复杂度校准
#    - 仅允许 1 个黄色反射控制，避免多重反射叠在一起难以读图。
#    - 生成器强制黄色缺口前缀里至少包含一个真实交叉，并优先选择输出中该交叉会切换可见颜色的案例，让融合效果肉眼可见。
#    - 仍维持 2-4 种线色、稀疏直角折线、二元正交交叉，避免新增对象类型。
# c) 隐式假设审查
#    - 原 solver 默认黄色后的对向缺口可以被普通 fill 或射线规则解释；这里不成立，所以增加 `yellow_blocks_fill`，只屏蔽沿黄色“目标臂方向”的补线。
#    - 原 D1 单变体默认反射 span 在冲突处必胜；这里不成立，所以删除强制覆盖集合，所有多色 claim 统一走长度判胜。
#    - 长度判胜依赖“交叉是二元、且两色在该格都有可测的直线通过”；生成器继续用输出先构造的正交交叉保证这一点，并拒绝等长 tie。
#    - 反射长度测量依赖 pivot 恰是非交叉拐角，且只有一条可见垂直源臂；生成器仅从这类 turn 中选黄色候选。
# d) 放置合法性
#    - 黄色候选要求：源臂长度 >= 2，目标臂长度严格大于源臂长度，marker 本格不是交叉，且缺口后面仍保留远端碎片。
#    - 允许缺口内部出现交叉，但禁止源臂测量段本身经过交叉，避免“源长度”被别色遮挡而不可读。
#    - 红标记与黄标记都避开交叉、pivot 及其邻域，保持 marker singleton 条件。
# e) 视觉可观察性
#    - 解题者会反复看到同一局部 motif：一个拐角旁的黄点，以及与其等长的垂直可见臂；输出把黄点一侧补回同长度。
#    - 同时，补回前缀中的某些交叉显示结果取决于哪条直线更长，而不是颜色编号，也不是“黄控制一定赢”。
# f) 可视化整洁度
#    - 反射只发生在单个拐角附近，且交叉仍是单像素二元正交交叉；没有三色堆叠或边界贴线。
#    - 生成器拒绝等长交叉与 marker 相邻碰撞，减少杂乱和歧义。
# g) 变化可推断性
#    - 没有隐藏颜色排序。非单色格的显示仅依赖可见几何：黄色缺口用垂直臂长度复制，交叉用局部直线段长度比较。
# h) 新 solver 假设与 generator 保证
#    - 假设：黄色 marker 的 assign_dir 指向 pivot。保证：黄色只放在目标臂首格，且视线分配规则与原题一致。
#    - 假设：被反射恢复的交叉至少有唯一长度赢家。保证：target 渲染阶段拒绝 equal-length crossing。
#    - 假设：若样本声明是融合样本，则至少有一个“黄色恢复前缀里的交叉”在输出中改色，且该格按旧碎片规则不会得到同样结果。保证：derive_input 显式检查这一量化条件。

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


def _perp_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return ((0, -1), (0, 1))
    return ((-1, 0), (1, 0))


def _dirs_in(pts: set[tuple[int, int]], pt: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = pt
    return [(dr, dc) for dr, dc in DIRS4 if (r + dr, c + dc) in pts]


def _run_length(pts: set[tuple[int, int]], start: tuple[int, int], direction: tuple[int, int]) -> int:
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    length = 0
    while (r, c) in pts:
        length += 1
        r += dr
        c += dc
    return length


def _straight_run_len(pts: set[tuple[int, int]], pt: tuple[int, int], axis: int) -> int:
    r, c = pt
    length = 1
    if axis == 0:
        cc = c - 1
        while (r, cc) in pts:
            length += 1
            cc -= 1
        cc = c + 1
        while (r, cc) in pts:
            length += 1
            cc += 1
        return length

    rr = r - 1
    while (rr, c) in pts:
        length += 1
        rr -= 1
    rr = r + 1
    while (rr, c) in pts:
        length += 1
        rr += 1
    return length


def _local_span_profile(pts: set[tuple[int, int]], pt: tuple[int, int]) -> tuple[int, int]:
    r, c = pt
    lengths: list[int] = []
    if (r, c - 1) in pts or (r, c + 1) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=0))
    if (r - 1, c) in pts or (r + 1, c) in pts:
        lengths.append(_straight_run_len(pts, pt, axis=1))
    if not lengths:
        return (1, 0)
    lengths.sort(reverse=True)
    if len(lengths) == 1:
        lengths.append(0)
    return (lengths[0], lengths[1])


def _longer_segment_winner(
    masks: dict[int, set[tuple[int, int]]],
    colors_here: list[int],
    pt: tuple[int, int],
    visible_color: int | None = None,
) -> int:
    profiles = {color: _local_span_profile(masks[color], pt) for color in colors_here}
    best_profile = max(profiles.values())
    best = [color for color, profile in profiles.items() if profile == best_profile]
    if len(best) == 1:
        return best[0]
    if visible_color in best:
        return int(visible_color)
    raise AssertionError(f"ambiguous equal-length crossing at {pt}: {profiles}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick colored line network by restoring singleton markers, filling ordinary orthogonal gaps, reflecting yellow-controlled arms by 90 degrees, and resolving overlaps by local segment length.

    Input structure:
    - A rectangular grid with a dominant background.
    - Non-marker colors form one-cell-thick axis-aligned stroke fragments that may cross.
    - Marker colors are exactly the foreground colors whose 4-connected components are all singleton pixels.
    - Red singleton markers replace missing line cells; yellow singleton markers replace the first cell of a missing arm that starts at a visible corner.

    Rule:
    - Assign every marker to the nearest visible line color in row/column line-of-sight, preferring a sightline that hits a cell already supported along that same axis.
    - For each line color, fill ordinary row/column gaps between consecutive facing endpoints, except for the directed target-side gaps that begin at yellow markers.
    - For a yellow control, find its pivot on the assigned color, keep the visible perpendicular source arm, measure that straight run, and paint the same number of cells from the pivot through the marker on the missing arm's side.
    - When multiple colors claim one cell after repair, display the color whose repaired straight segment through that cell is longer.

    Key insight: The markers still tell you which sparse fragments belong together, but yellow now means “copy the visible orthogonal arm length onto the marked side,” and any crossing created by that copied span must be settled geometrically by comparing the local through-segment lengths.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

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

    yellow_controls = _pts(g == 4) if 4 in marker_colors else set()
    marker_mask = np.isin(g, list(marker_colors))
    for r0, c0 in _pts(marker_mask):
        candidates: list[tuple[int, int, int, int, int]] = []
        for dr, dc in DIRS4:
            r, c = r0 + dr, c0 + dc
            dist = 1
            while 0 <= r < H and 0 <= c < W:
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
        candidates.sort()
        _dist, _unsupported, color, dr, dc = candidates[0]
        assignment[(r0, c0)] = int(color)
        assign_dir[(r0, c0)] = (int(dr), int(dc))

    def yellow_blocks_fill(a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        for marker, other in ((a, b), (b, a)):
            if marker not in yellow_controls:
                continue
            dr, dc = assign_dir[marker]
            away = (-dr, -dc)
            if axis == 0 and away[0] == 0:
                delta = other[1] - marker[1]
                if delta * away[1] > 0:
                    return True
            if axis == 1 and away[1] == 0:
                delta = other[0] - marker[0]
                if delta * away[0] > 0:
                    return True
        return False

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)

    def has_axis_support(color: int, r: int, c: int, axis: int) -> bool:
        return (r, c) in marker_pts[color] or _axis_support(masks[color], r, c, axis)

    def should_fill(color: int, a: tuple[int, int], b: tuple[int, int], axis: int) -> bool:
        if yellow_blocks_fill(a, b, axis):
            return False

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

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        for r in range(H):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    additions.update((r, c) for c in range(c1 + 1, c2))

        for c in range(W):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    additions.update((r, c) for r in range(r1 + 1, r2))

        masks[color].update(additions)

    if 4 in marker_colors:
        for control in yellow_controls:
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H and 0 <= c < W:
                if (r, c) in masks[color]:
                    pivot = (r, c)
                    break
                r += dr
                c += dc
            assert pivot is not None, f"yellow control at {control} has no assigned pivot"

            target_dir = (-dr, -dc)
            source_dirs = []
            for sdr, sdc in _perp_dirs(target_dir):
                if (pivot[0] + sdr, pivot[1] + sdc) in masks[color]:
                    source_dirs.append((sdr, sdc))
            assert len(source_dirs) == 1, f"yellow control at {control} needs exactly one perpendicular source arm"

            source_len = _run_length(masks[color], pivot, source_dirs[0])
            assert source_len > 0, f"yellow control at {control} found no source arm length"

            r, c = pivot
            for _ in range(source_len + 1):
                assert 0 <= r < H and 0 <= c < W, f"yellow reflection at {control} leaves grid"
                masks[color].add((r, c))
                r += target_dir[0]
                c += target_dir[1]

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue
        visible_color = int(g[pt]) if int(g[pt]) in colors_here else None
        out[pt] = _longer_segment_winner(masks, colors_here, pt, visible_color)

    for color in line_colors:
        for comp in components_mask(out == color, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)) if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = r == 0 or r == H - 1 or c == 0 or c == W - 1
                if on_boundary:
                    nr, nc = nbrs[0]
                    dr, dc = nr - r, nc - c
                    if not ((r in (0, H - 1)) and (c in (0, W - 1))):
                        perp = (
                            (r == 0 and dr == 1)
                            or (r == H - 1 and dr == -1)
                            or (c == 0 and dc == 1)
                            or (c == W - 1 and dc == -1)
                        )
                        assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"
                    continue
                adj_other = any(
                    0 <= r + dr < H
                    and 0 <= c + dc < W
                    and int(out[r + dr, c + dc]) not in (bg, color)
                    for dr, dc in DIRS4
                )
                assert adj_other, f"dangling endpoint ({r},{c}) color {color}: not on boundary or crossing"

    color_cells = {c: set() for c in line_colors}
    for r in range(H):
        for c in range(W):
            v = int(out[r, c])
            if v in line_colors:
                color_cells[v].add((r, c))
    crossing_colors = set()
    for r in range(H):
        for c in range(W):
            a = int(out[r, c])
            if a not in line_colors:
                continue
            for b in line_colors - {a}:
                if (((r, c - 1) in color_cells[b] and (r, c + 1) in color_cells[b]) or ((r - 1, c) in color_cells[b] and (r + 1, c) in color_cells[b])):
                    crossing_colors.add(a)
                    crossing_colors.add(b)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


# --------------- generator (combo o5x) ---------------

H = W = 12
BG = 0
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
LINE_PALETTE = [1, 3, 5, 6, 7, 8, 9]

POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12

MARKER_MODES = ["both"]
MARKER_MODE_WEIGHTS = [1]
RED_COUNT_RANGE = (3, 6)
MIN_CELLS_FOR_RED = 2
MIN_REFLECT_SOURCE = 2

COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (4, 6)
FILL_RANGE = (33, 55)

SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
RED_PICK_RETRIES = 5
GEN_RETRIES = 1600


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


def _orientation(pts: set[tuple[int, int]], pt: tuple[int, int]) -> str | None:
    dirs = _dirs_in(pts, pt)
    if len(dirs) != 2 or (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0):
        return None
    return "v" if dirs[0][0] else "h"


def _is_turn_or_endpoint(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    if len(dirs) == 1:
        return True
    return len(dirs) == 2 and (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0)


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
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1) or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
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
    pieces = random.choices(range(1, len(POLYLINE_PIECE_WEIGHTS) + 1), weights=POLYLINE_PIECE_WEIGHTS)[0]
    a0, a1 = _random_span(_dim(ax))
    pieces = min(pieces, max(1, a1 - a0))
    minors = random.sample(range(1, _dim(1 - ax) - 1), pieces)
    bends = sorted(random.sample(range(a0 + 1, a1), pieces - 1)) if pieces > 1 else []
    pts = [_rc(ax, a0, minors[0])]
    for i, b in enumerate(bends):
        pts.append(_rc(ax, b, minors[i]))
        pts.append(_rc(ax, b, minors[i + 1]))
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


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int, is_start: bool) -> list[tuple[int, int]]:
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
    return [_rc(ax, arm_major, main_coord), _rc(ax, arm_major, minor), _rc(ax, _dim(ax) - 1, minor)]


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


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [(color, pt) for color, comps in paths.items() for comp in comps for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
    random.shuffle(candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    per_color_left = {color: len(logical[color]) for color in logical}
    for color, pt in candidates:
        if len(chosen) == count:
            break
        if per_color_left[color] <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        per_color_left[color] -= 1
    return chosen if len(chosen) == count else []


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
    desired_crossings = random.choices(range(1, len(CROSSING_COUNT_WEIGHTS) + 1), weights=CROSSING_COUNT_WEIGHTS)[0]
    n_colors = random.randint(2, min(MAX_COLORS, TOTAL_COMPONENTS[1] - desired_crossings))
    colors = random.sample(LINE_PALETTE, n_colors)
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


def _render_target(logical: dict[int, set[tuple[int, int]]]) -> np.ndarray | None:
    out = np.full((H, W), BG, dtype=int)
    for pt in set().union(*logical.values()):
        colors_here = [color for color in logical if pt in logical[color]]
        if len(colors_here) == 1:
            out[pt] = colors_here[0]
            continue
        try:
            out[pt] = _longer_segment_winner(logical, colors_here, pt)
        except AssertionError:
            return None
    return out


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _boundary_steps_from(pt: tuple[int, int], direction: tuple[int, int]) -> int:
    r, c = pt
    dr, dc = direction
    steps = 0
    while 0 <= r + dr < H and 0 <= c + dc < W:
        r += dr
        c += dc
        steps += 1
    return steps


def _straight_run_cells(comp: set[tuple[int, int]], pivot: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    r, c = pivot[0] + dr, pivot[1] + dc
    out: list[tuple[int, int]] = []
    while (r, c) in comp:
        out.append((r, c))
        r += dr
        c += dc
    return out


def _old_fragment_winner(comp_count: dict[int, int], colors_here: list[int]) -> int:
    return min(colors_here, key=lambda color: (-comp_count[color], color))


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]]]]:
    candidates = []
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings:
                    continue
                dirs = _dirs_in(comp, pivot)
                if len(dirs) != 2:
                    continue
                if (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) == (0, 0):
                    continue
                for source_dir, target_dir in ((dirs[0], dirs[1]), (dirs[1], dirs[0])):
                    source_cells = _straight_run_cells(comp, pivot, source_dir)
                    target_cells = _straight_run_cells(comp, pivot, target_dir)
                    source_len = len(source_cells)
                    if source_len < MIN_REFLECT_SOURCE or len(target_cells) < source_len:
                        continue
                    marker = target_cells[0]
                    if marker in crossings:
                        continue
                    if any(pt in crossings for pt in source_cells[:source_len]):
                        continue
                    boundary_steps = _boundary_steps_from(pivot, target_dir)
                    if boundary_steps <= source_len:
                        continue
                    removed = target_cells[:source_len]
                    created = [pt for pt in removed[1:] if pt in crossings]
                    if not created:
                        continue
                    visible_created = []
                    for pt in created:
                        colors_here = [c for c in logical if pt in logical[c]]
                        try:
                            winner = _longer_segment_winner(logical, colors_here, pt)
                        except AssertionError:
                            winner = None
                        if winner == color:
                            visible_created.append(pt)
                    if not visible_created:
                        continue
                    candidates.append((color, pivot, source_dir, target_dir, source_cells[:source_len], removed, visible_created))
    random.shuffle(candidates)
    candidates.sort(key=lambda item: (-len(item[6]), -len(item[5])))
    return candidates


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    del dominant
    crossings = set(_crossing_cells(logical))
    target = _render_target(logical)
    if target is None:
        return None

    mode = random.choices(MARKER_MODES, weights=MARKER_MODE_WEIGHTS)[0]
    red_count = random.randint(*RED_COUNT_RANGE) if mode == "both" else 0

    crossing_colors = {pt: [color for color in logical if pt in logical[color]] for pt in crossings}
    crossing_loser: dict[tuple[int, int], int] = {}
    for pt, colors_here in crossing_colors.items():
        winner = int(target[pt])
        losers = [color for color in colors_here if color != winner]
        if len(losers) != 1:
            return None
        crossing_loser[pt] = losers[0]

    yellow_infos = _yellow_candidates(logical, paths, crossings)
    if not yellow_infos:
        return None

    for yellow_info in yellow_infos:
        color, pivot, _source_dir, _target_dir, source_cells, removed, visible_created = yellow_info
        marker = removed[0]
        footprint = set(source_cells) | set(removed) | {pivot}
        removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
        removed_for_yellow[color].update(removed)
        yellow_by_pt = {marker: color}
        interaction_pts = [pt for pt in visible_created if pt in removed_for_yellow[color]]
        if not interaction_pts:
            continue

        safe_crossings = [pt for pt in crossings if pt not in removed_for_yellow[color]]
        divergence_options = [None] + random.sample(safe_crossings, len(safe_crossings))

        for _ in range(max(1, RED_PICK_RETRIES)):
            blocked = set(crossings) | footprint | set(_neighbors(pivot))
            red_markers = _choose_red_markers(logical, paths, blocked, red_count) if red_count else []
            if red_count and not red_markers:
                continue
            red_by_pt = {pt: color for color, pt in red_markers}

            for divergence_pt in divergence_options:
                crossing_display = {pt: crossing_loser[pt] for pt in crossings}
                if divergence_pt is not None:
                    crossing_display[divergence_pt] = int(target[divergence_pt])

                inp = np.full((H, W), BG, dtype=int)
                comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

                failed = False
                for line_color, pts in logical.items():
                    for pt in pts:
                        if pt in removed_for_yellow[line_color]:
                            continue
                        if pt in crossings and crossing_display[pt] != line_color:
                            continue
                        if pt in red_by_pt or pt in yellow_by_pt:
                            continue
                        if inp[pt] != BG and inp[pt] != line_color:
                            failed = True
                            break
                        inp[pt] = line_color
                        comp_count_masks[line_color].add(pt)
                    if failed:
                        break
                if failed:
                    continue

                for pt, marker_color in red_by_pt.items():
                    inp[pt] = 2
                    comp_count_masks[marker_color].add(pt)
                for pt, marker_color in yellow_by_pt.items():
                    inp[pt] = 4
                    comp_count_masks[marker_color].add(pt)

                if not _marker_isolation_ok(inp):
                    continue
                if any(not any(inp[nb] == marker_color for nb in _neighbors(pt)) for pt, marker_color in yellow_by_pt.items()):
                    continue

                comp_count = {line_color: _gen_component_count(pts) for line_color, pts in comp_count_masks.items()}
                visible_interactions = [pt for pt in interaction_pts if inp[pt] != target[pt]]
                if not visible_interactions:
                    continue
                if not any(_old_fragment_winner(comp_count, crossing_colors[pt]) != int(target[pt]) for pt in crossings):
                    continue

                counts = {c: len(components_mask(target == c, connectivity=4)) for c in sorted(logical)}
                if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
                    continue
                if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
                    continue
                if not FILL_RANGE[0] <= int(np.count_nonzero(target)) <= FILL_RANGE[1]:
                    continue

                try:
                    solved = main(inp)
                except Exception:
                    continue
                if np.array_equal(solved, target):
                    return inp, target
    return None


def _gen_attempt() -> np.ndarray | None:
    built = _build_logical()
    if built is None:
        return None
    logical, paths, dominant = built
    derived = _derive_input(logical, paths, dominant)
    return derived[0] if derived is not None else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid combo faa9f03d input")
