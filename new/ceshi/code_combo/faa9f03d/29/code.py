# Combo variant of faa9f03d (index 29)
# combo_spec: D1/3+D7/0+D14/0
# Name: 特殊色填充 + 裁剪输出 + 交叉处背景色
#
# Design analysis
# ---------------
# spec:
#   variant_core:
#     unchanged:
#       - 输入仍是 12x12、背景为 0、线仍是单像素宽的正交折线网络。
#       - 红标记仍表示“这里属于被修复的缺口链”，黄标记仍表示“端点改向并向边界延伸”。
#       - 标记到线色的分配、黄射线的擦除/延伸、交叉处按可见碎片度判胜，这三条原始主逻辑仍保留。
#     changed:
#       - D14/0: 真正交叉格在输入中不再显示任一线色，而是直接留成背景色空洞。
#       - D1/3: 红标记触发的修复痕迹在输出中不显示为线色，而统一显示为固定特殊色 5。
#         这里的“修复痕迹”包括红标记像素本身，以及由该红标记参与桥接出来的同轴补段。
#       - D7/0: 最终输出不是原地 12x12，而是把修复后的非背景区域裁成最小外接矩形。
#   concrete_behavior:
#     - 若蓝线和橙线在 (5,6) 真实交叉，则输入把 (5,6) 留空；solver 需分别补回水平/竖直两条逻辑路径，
#       再按碎片度决定输出中心显示哪一种线色。
#     - 若某个红标记位于绿线缺口上，则该位置在输出中不恢复成绿色，而恢复成特殊色 5；
#       如果它还带动同轴空段一起被桥接，那整段桥接痕迹都显示为 5。
#     - 因而输出里同时存在两类“新恢复”的格子：
#       1) 隐藏交叉恢复后显示胜者线色；
#       2) 红缺口恢复后显示特殊色 5。
#   local_trace:
#     original:
#       - 输入交叉中心已有“失败者”线色，红标记恢复后也直接变回所属线色，输出尺寸不变。
#     combo:
#       - 同一交叉中心在输入里变成背景洞，需要从两条正交 1 格缺口共同推回。
#       - 红标记对应的修复格在输出里变成灰色 5，而不是所属线色。
#       - 整体修复完成后，再把整张 12x12 输出裁到所有非背景像素的紧凑包围盒。
#     difference:
#       - 解题者不能再把“背景洞”统一解释成普通缺口，因为交叉洞和红缺口洞的输出语义不同。
#       - 解题者也不能只看颜色修复，因为最终尺寸还取决于修复后整网的外接矩形。
#   complexity_controls:
#     - 特殊色固定为 5，且生成器禁止把 5 当作线色使用，避免“它到底是普通线色还是修复痕迹”。
#     - 交叉仍限制为每格两色、且两线正交；黄射线仍禁止经过真实交叉，避免“灰色修复痕迹”和“黄射线强制胜出”叠在同一洞上。
#     - 生成器强制至少放 2 个红标记，并给跨越全部交叉的主线额外制造碎片，避免隐藏交叉只靠颜色 id 平局决胜。
#     - 生成器强制裁剪后尺寸小于 12x12，保证 D7/0 在样例里肉眼可见。
#   implicit_assumption_audit:
#     - `main()` 仍依赖“标记能在四向视线中看到某条可见线色”；隐藏交叉只把交叉格挖空，不破坏这点。
#     - 原版把最终可见网格当作线本体做端点验证；本组合不成立，因为特殊色会把线可视化地切开。
#       所以新的 solver 改为对内部逻辑线掩码 `masks[color]` 做几何合法性检查，而不是对显示色网格做检查。
#     - 原版的 `g[pt] == bg` 多色分支里带有“背景多色重叠可能只是普通填充冲突”的启发式；
#       本组合里，生成器保证所有无黄射线的多色重叠都是真实隐藏交叉，因此 solver 可直接按碎片度判胜。
#     - 新增假设是“特殊色 5 只在输出出现，不会在输入里作为普通线色出现”；生成器通过调色板约束保证。
#   placement_legality:
#     - 交叉格始终加入 `blocked`，红/黄标记都不能占用或贴邻真实交叉。
#     - 黄候选若其目标射线穿过真实交叉则直接弃用，避免背景交叉同时承载改向语义。
#     - 红标记至少 2 个，且优先放在主导色的转角上，既制造特殊色痕迹，也制造可靠的碎片度差异。
#     - 生成器最后用同文件的 `_solve_full()` 和 `main()` 双重自检：既验证完整修复逻辑，也验证裁剪后的最终输出。
#   visual_observability:
#     - 输入中可重复观察到“十字型背景洞”；输出中这些洞会变成某条线色，而红标记位置会稳定地变成灰色 5。
#     - 输出尺寸会明显缩小；灰色痕迹和裁剪共同让这不是“仅仅把线补好”的原题。
#   tidiness:
#     - 特殊色只承担一种语义：红缺口修复痕迹。交叉恢复仍显示线色，避免一个格子同时表达两种规则。
#     - 黄射线、灰色修复痕迹、隐藏交叉三种机制被分配到不同位置约束中，减少视觉叠层。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
H = W = 12
BG = 0
SPECIAL = 5
LINE_PALETTE = [1, 3, 6, 7, 8, 9]


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


def _tight_crop(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0, "expected non-empty output"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _solve_full(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    H0, W0 = g.shape

    colors = set(map(int, g.ravel())) - {bg}
    assert colors, "expected foreground line and marker colors"
    assert SPECIAL not in colors, "special repair color must not appear in the input palette"

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
            while 0 <= r < H0 and 0 <= c < W0:
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

    masks = {color: set(pts) for color, pts in original.items()}
    marker_pts = {color: set() for color in line_colors}
    red_marker_pts = {color: set() for color in line_colors}
    repair_fill = {color: set() for color in line_colors}
    for pt, color in assignment.items():
        masks[color].add(pt)
        marker_pts[color].add(pt)
        if int(g[pt]) == 2:
            red_marker_pts[color].add(pt)
            repair_fill[color].add(pt)

    comp_count = {color: _component_count(g.shape, original[color]) for color in line_colors}

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

    for color in sorted(line_colors):
        additions: set[tuple[int, int]] = set()
        repair_additions: set[tuple[int, int]] = set()
        for r in range(H0):
            cols = sorted(c for rr, c in masks[color] if rr == r)
            for c1, c2 in zip(cols, cols[1:]):
                if should_fill(color, (r, c1), (r, c2), axis=0):
                    new_cells = {(r, c) for c in range(c1 + 1, c2)}
                    additions.update(new_cells)
                    if (r, c1) in red_marker_pts[color] or (r, c2) in red_marker_pts[color]:
                        repair_additions.update(new_cells)

        for c in range(W0):
            rows = sorted(r for r, cc in masks[color] if cc == c)
            for r1, r2 in zip(rows, rows[1:]):
                if should_fill(color, (r1, c), (r2, c), axis=1):
                    new_cells = {(r, c) for r in range(r1 + 1, r2)}
                    additions.update(new_cells)
                    if (r1, c) in red_marker_pts[color] or (r2, c) in red_marker_pts[color]:
                        repair_additions.update(new_cells)

        masks[color].update(additions)
        repair_fill[color].update(repair_additions)

    control_ray = {color: set() for color in line_colors}
    if 4 in marker_colors:
        for control in _pts(g == 4):
            color = assignment[control]
            dr, dc = assign_dir[control]

            r, c = control[0] + dr, control[1] + dc
            pivot: tuple[int, int] | None = None
            while 0 <= r < H0 and 0 <= c < W0:
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
                    red_marker_pts[color].discard(q)
                    repair_fill[color].discard(q)
                    control_ray[color].discard(q)

            pr, pc = -dr, -dc
            r, c = pivot
            while 0 <= r < H0 and 0 <= c < W0:
                masks[color].add((r, c))
                control_ray[color].add((r, c))
                r += pr
                c += pc

    claims: dict[tuple[int, int], list[int]] = {}
    for color, pts in masks.items():
        for pt in pts:
            claims.setdefault(pt, []).append(int(color))

    out = np.full_like(g, bg)
    for pt, colors_here in claims.items():
        forced = {color for color in colors_here if pt in control_ray[color]}
        if forced:
            assert len(forced) == 1, f"ambiguous yellow-control rays at {pt}: {forced}"
            out[pt] = next(iter(forced))
            continue

        if len(colors_here) == 1:
            color = colors_here[0]
            out[pt] = SPECIAL if pt in repair_fill[color] else color
            continue

        out[pt] = min(colors_here, key=lambda color: (-comp_count[color], color))

    # Validate the hidden logical network, not the visible display grid:
    # the visible grid may contain SPECIAL cells that intentionally break a line visually.
    for color in line_colors:
        logical_mask = np.zeros(g.shape, dtype=bool)
        for r, c in masks[color]:
            logical_mask[r, c] = True
        for comp in components_mask(logical_mask, connectivity=4):
            pts_set = set(map(tuple, comp))
            for r, c in pts_set:
                nbrs = [(nr, nc) for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
                        if (nr, nc) in pts_set]
                if len(nbrs) != 1:
                    continue
                on_boundary = (r == 0 or r == H0 - 1 or c == 0 or c == W0 - 1)
                assert on_boundary, f"logical endpoint ({r},{c}) color {color} not on boundary"
                nr, nc = nbrs[0]
                dr, dc = nr - r, nc - c
                if not ((r in (0, H0 - 1)) and (c in (0, W0 - 1))):
                    perp = ((r == 0 and dr == 1) or (r == H0 - 1 and dr == -1)
                            or (c == 0 and dc == 1) or (c == W0 - 1 and dc == -1))
                    assert perp, f"boundary endpoint ({r},{c}) color {color} not perpendicular"

    crossing_colors = set()
    for pt, colors_here in claims.items():
        if len(colors_here) < 2:
            continue
        crossing_colors.update(colors_here)
    for color in line_colors:
        assert color in crossing_colors, f"color {color} has no crossing with any other color"

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair a one-cell-thick rectilinear line network by restoring hidden crossings, marking red-gap repairs with a dedicated trace color, and then tightly cropping the repaired region.

    Input structure:
    - A rectangular grid with a dominant background.
    - Line colors form one-cell-thick horizontal/vertical stroke fragments.
    - Marker colors are exactly those whose 4-connected components are all singleton pixels.
    - Red singleton markers indicate cells belonging to a repaired gap, yellow singleton markers indicate endpoint redirection, and true crossings are hidden as background holes.

    Rule:
    - Assign every marker to the nearest visible line color in row/column line-of-sight, preferring an endpoint supported along that same axis.
    - Reconstruct each line color by bridging facing row/column endpoints; a background hole jointly claimed by two orthogonal colors is a hidden crossing and is displayed in the winner's line color, while a bridge created by a red-gap repair is displayed in the fixed special color 5.
    - A yellow control erases the old straight arm on the far side of its endpoint and paints the assigned color from that endpoint through the control pixel to the grid edge.
    - After all repairs, crop the output to the minimal bounding box containing every non-background cell.

    Key insight: Background holes and red-marker repairs are not the same thing. A hidden crossing is inferred because two colors independently need the same empty center, so it resolves back to a line color by fragmentation; a red repair is inferred from marker assignment and remains visibly tagged by the special color, after which the whole repaired network is tightly cropped.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    return _tight_crop(_solve_full(g), bg)


# --------------- generator (combo) ---------------

# --- CFG: polyline shape ---
POLYLINE_PIECE_WEIGHTS = [5, 4, 2, 1]
MIN_SPAN_LEN = 4
FULL_SPAN_PROB = 0.65
ARM_MIN_OFFSET = 2

# --- CFG: serpentine & crossings ---
CROSSING_COUNT_WEIGHTS = [2, 3, 3, 2]
SERPENTINE_MINOR_LO = 6
SERPENTINE_MINOR_HI = 5
CROSS_MINOR_MARGIN = 2
MAX_COLORS = 4
MAX_CROSSINGS = 12

# --- CFG: markers ---
RED_COUNT_RANGE = (2, 6)
MIN_CELLS_FOR_RED = 2
HIDDEN_CROSSING_DOMINANT_REDS = 2

# --- CFG: output validation ---
COMPONENTS_PER_COLOR = (1, 4)
TOTAL_COMPONENTS = (2, 6)
FILL_RANGE = (33, 55)

# --- CFG: retry budgets ---
SPACED_VALUES_RETRIES = 60
EXTRA_POLYLINE_RETRIES = 80
GEN_RETRIES = 400


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


def _is_turn_or_endpoint(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
    if len(dirs) == 1:
        return True
    return len(dirs) == 2 and (dirs[0][0] + dirs[1][0], dirs[0][1] + dirs[1][1]) != (0, 0)


def _is_turn(pts: set[tuple[int, int]], pt: tuple[int, int]) -> bool:
    dirs = _dirs_in(pts, pt)
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
        if not ((r == 0 and dr == 1) or (r == H - 1 and dr == -1)
                or (c == 0 and dc == 1) or (c == W - 1 and dc == -1)):
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


def _build_arm(ax: int, main_coord: int, cross_lo: int, cross_hi: int,
               is_start: bool) -> list[tuple[int, int]]:
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


def _ray_from(pivot: tuple[int, int], direction: tuple[int, int]) -> set[tuple[int, int]]:
    out = set()
    r, c = pivot
    dr, dc = direction
    while 0 <= r < H and 0 <= c < W:
        out.add((r, c))
        r += dr
        c += dc
    return out


def _yellow_candidates(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    crossings: set[tuple[int, int]],
) -> list[tuple[int, tuple[int, int], tuple[int, int], set[tuple[int, int]], set[tuple[int, int]]]]:
    candidates = []
    occupied = set().union(*logical.values())
    for color, comps in paths.items():
        for comp in comps:
            for pivot in comp:
                if pivot in crossings or not _is_turn_or_endpoint(comp, pivot):
                    continue
                for out_dir in DIRS4:
                    mr, mc = pivot[0] + out_dir[0], pivot[1] + out_dir[1]
                    if not (0 <= mr < H and 0 <= mc < W) or (mr, mc) not in comp:
                        continue
                    target_ray = _ray_from(pivot, out_dir)
                    if not target_ray.issubset(comp):
                        continue
                    old_dir = (-out_dir[0], -out_dir[1])
                    old_cells: set[tuple[int, int]] = set()
                    r, c = pivot[0] + old_dir[0], pivot[1] + old_dir[1]
                    while 0 <= r < H and 0 <= c < W and (r, c) not in occupied:
                        old_cells.add((r, c))
                        r += old_dir[0]
                        c += old_dir[1]
                    if not old_cells:
                        continue
                    old_len = random.randint(1, len(old_cells))
                    old_arm = set(list(_ray_from((pivot[0] + old_dir[0], pivot[1] + old_dir[1]), old_dir))[:old_len])
                    if old_arm and old_arm.issubset(old_cells):
                        candidates.append((color, pivot, out_dir, target_ray, old_arm))
    random.shuffle(candidates)
    return candidates


def _choose_red_markers(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [(color, pt) for color, comps in paths.items() for comp in comps
                  for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
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


def _choose_red_markers_for_color(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    blocked: set[tuple[int, int]],
    color: int,
    count: int,
) -> list[tuple[int, tuple[int, int]]]:
    candidates = [pt for comp in paths[color] for pt in comp if pt not in blocked and _is_turn_or_endpoint(comp, pt)]
    random.shuffle(candidates)
    chosen: list[tuple[int, tuple[int, int]]] = []
    used: set[tuple[int, int]] = set(blocked)
    cells_left = len(logical[color])
    for pt in candidates:
        if len(chosen) == count:
            break
        if cells_left <= MIN_CELLS_FOR_RED:
            continue
        if any(nb in used for nb in [pt] + _neighbors(pt)):
            continue
        chosen.append((color, pt))
        used.add(pt)
        cells_left -= 1
    return chosen if len(chosen) == count else []


def _find_crossing_point(logical: dict[int, set[tuple[int, int]]]) -> tuple[tuple[int, int], str] | None:
    candidates = []
    for color, pts in logical.items():
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


def _marker_isolation_ok(grid: np.ndarray) -> bool:
    for color in (2, 4):
        for pt in map(tuple, np.argwhere(grid == color)):
            if any(grid[nb] == color for nb in _neighbors(pt)):
                return False
    return True


def _derive_input(
    logical: dict[int, set[tuple[int, int]]],
    paths: dict[int, list[set[tuple[int, int]]]],
    dominant: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    crossings = set(_crossing_cells(logical))
    red_count = max(random.randint(*RED_COUNT_RANGE), HIDDEN_CROSSING_DOMINANT_REDS)

    yellow_infos = []
    blocked = set(crossings)

    forced_reds = _choose_red_markers_for_color(logical, paths, blocked, dominant, HIDDEN_CROSSING_DOMINANT_REDS)
    if len(forced_reds) != HIDDEN_CROSSING_DOMINANT_REDS:
        return None
    blocked_with_forced = blocked | {pt for _, pt in forced_reds}
    extra_need = red_count - HIDDEN_CROSSING_DOMINANT_REDS
    extra_reds = _choose_red_markers(logical, paths, blocked_with_forced, extra_need) if extra_need else []
    if extra_need and not extra_reds:
        return None
    red_markers = forced_reds + extra_reds

    inp = np.full((H, W), BG, dtype=int)
    comp_count_masks: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}

    removed_for_yellow: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    old_arms: dict[int, set[tuple[int, int]]] = {c: set() for c in logical}
    yellow_markers: list[tuple[int, tuple[int, int]]] = []
    for color, pivot, out_dir, target_ray, old_arm in yellow_infos:
        marker = (pivot[0] + out_dir[0], pivot[1] + out_dir[1])
        removed_for_yellow[color].update(target_ray - {pivot})
        old_arms[color].update(old_arm)
        yellow_markers.append((color, marker))

    red_by_pt = {pt: color for color, pt in red_markers}
    yellow_by_pt = {pt: color for color, pt in yellow_markers}

    for color, pts in logical.items():
        for pt in pts:
            if pt in removed_for_yellow[color]:
                continue
            if pt in crossings:
                continue
            if pt in red_by_pt or pt in yellow_by_pt:
                continue
            inp[pt] = color
            comp_count_masks[color].add(pt)
    for color, pts in old_arms.items():
        for pt in pts:
            if inp[pt] != BG:
                return None
            inp[pt] = color
            comp_count_masks[color].add(pt)

    for pt in red_by_pt:
        inp[pt] = 2
    for pt in yellow_by_pt:
        inp[pt] = 4

    if not _marker_isolation_ok(inp):
        return None
    for pt, color in yellow_by_pt.items():
        if not any(inp[nb] == color for nb in _neighbors(pt)):
            return None

    comp_count = {color: _gen_component_count(pts) for color, pts in comp_count_masks.items()}
    for pt in crossings:
        colors_here = [color for color in logical if pt in logical[color]]
        best = max(comp_count[color] for color in colors_here)
        if sum(comp_count[color] == best for color in colors_here) != 1:
            return None

    try:
        full_target = _solve_full(inp)
        cropped_target = _tight_crop(full_target, BG)
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, cropped_target):
        return None

    counts = {c: len(paths[c]) for c in logical}
    if not all(COMPONENTS_PER_COLOR[0] <= v <= COMPONENTS_PER_COLOR[1] for v in counts.values()):
        return None
    if not TOTAL_COMPONENTS[0] <= sum(counts.values()) <= TOTAL_COMPONENTS[1]:
        return None
    if not FILL_RANGE[0] <= int(np.count_nonzero(full_target)) <= FILL_RANGE[1]:
        return None

    if not all(inp[pt] == BG for pt in crossings):
        return None
    if int(np.count_nonzero(full_target == SPECIAL)) < len(red_markers):
        return None
    if cropped_target.shape == inp.shape:
        return None

    return inp, cropped_target


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
    raise RuntimeError("failed to generate a valid faa9f03d combo input")
