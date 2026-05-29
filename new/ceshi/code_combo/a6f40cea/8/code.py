# Combo variant of a6f40cea (index 8)
# combo_spec: D3/1+D16/0
# Name: 边缘像素镜像投影
"""
Design analysis for the combo variant
=====================================

combo_spec:
  preserved_core:
    - 仍然存在一个最大的 1px 闭合矩形框，框内为统一填充色。
    - 输出仍然裁剪为框内区域，并在填充底色上画单色路径、双色交替 L 路径，或整列/整行条纹。
    - L 路径仍优先选取“拐点严格落在内部”的那条曼哈顿折线。
  fused_change:
    - D3/1: 删除框外开放矩形，改为整张网格最外圈的控制像素。
    - D16/0: 控制像素不再投影到同侧内部边缘，而是镜像到对侧内部边缘。
    - 于是：顶边像素映射到底部内部边缘；底边映射到顶部；左边映射到右侧；右边映射到左侧。

behavior_trace:
  concrete_example:
    - 设框内尺寸为 6x8，某条单色 L 输出路径的真实端点在内部 top=(0,2) 与 left=(4,0)。
    - 原题外部开放框会在框外上侧和左侧留下接触信息；D3 单变体则会直接在整图顶边第 2 列、左边第 4 行放同色控制像素。
    - 本融合版中，为了得到同一条内部路径，控制像素必须改放在“镜像对侧”：整图底边第 2 列、右边第 4 行。
    - solver 读取到底边/右边像素后，若做直投影会得到内部 bottom/right 端点；只有做镜像投影才会还原到 top/left 端点。
  local_difference:
    - D3 只改变“源对象长相”；D16 进一步改变“边缘位置如何解释”。
    - 二者叠加后，控制像素的颜色负责配对/交替，相对边的位置负责镜像后的端点边。

complexity_calibration:
  - 最多只允许 1 个双色 L 组件，避免少量样例下出现多个 singleton 颜色导致配对歧义。
  - 保留原题的 U/双条纹分支，但只在样本里至少还有 1 个 L 组件时才允许放入，避免整题退化成“镜像与否都一样”的直条纹例子。
  - 单色 L 和 U 仍共享原题规模范围，使视觉上仍是原任务家族，而不是新造的无关小游戏。

implicit_assumption_audit:
  - 原 solver 假设“有框外 4 连通组件可做连通域分析”；D3/1 后失效，必须直接扫描整图外边界。
  - D3 单变体还默认“边在哪，就投到哪一侧内部边缘”；D16/0 使这个假设失效，必须把 side->endpoint 的映射全部翻到对边。
  - 原 deferred 条纹逻辑仍可保留，因为条纹本质只依赖成组控制点映射后的坐标集合，与外部几何已无关。
  - 双色相位仍依赖端点颜色，但 generator 必须保证双色 L 的路径长度为偶数，这样两端颜色才不同且可由两个 singleton 控制像素锚定。

placement_legality:
  - 控制像素只能放在整图最外圈，且同一格不能复用。
  - L 组件的两个控制点必须放在与真实内部端点相对的外边界。
  - U 组件虽在镜像规则下仍使用 top/bottom 或 left/right 成对控制，但要求至少再有一个 L 组件承担“镜像可观察性”。
  - 内部路径继续 output-first 构造，用 used_inner 防止重叠；干扰块不放在边界，避免被误读为控制像素。

visual_observability:
  - 解题者能直接看到：输入只剩边缘单像素和中部闭合框，框附近没有原题那种开放框架。
  - 真正的难点是“底边像素对应顶部端点、右边像素对应左侧端点”这一反向关系，它会在至少一个 L 组件上反复出现。
  - 不使用颜色编号排序、黑色特殊义等不可视规则；颜色只承担组件身份与双色交替相位提示。

visual_cleanliness:
  - 控制像素与主框分层明确：控制信息在整图外圈，输出对象在框内。
  - 通过限制组件数、避免内部路径重叠、避免 U-only 样本，保证样例既干净又能体现融合差异。

inference_guardrail:
  - 若一个样本只有 U 条纹，那么镜像投影与直投影会得到同样输出，这是融合退化。
  - 因此 generator 明确要求每个样本至少含 1 个 L 组件，并在验收脚本中统计“镜像 solver 与直投影 solver 输出不同”的比例。
"""

from __future__ import annotations

from collections import Counter
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


def _build_manhattan(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
) -> list[tuple[int, int]]:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb:
        step = 1 if cb >= ca else -1
        return [(int(ra), int(c)) for c in range(int(ca), int(cb) + step, step)]
    if ca == cb:
        step = 1 if rb >= ra else -1
        return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
    if corner_a:
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


def _inside_corner(
    p1: tuple[int, int],
    p2: tuple[int, int],
    corner_a: bool,
    out_h: int,
    out_w: int,
) -> bool:
    (ra, ca), (rb, cb) = p1, p2
    if ra == rb or ca == cb:
        return True
    corner = (rb, ca) if corner_a else (ra, cb)
    rr, cc = corner
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _choose_path(
    p1: tuple[int, int],
    p2: tuple[int, int],
    out_h: int,
    out_w: int,
) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    use_b = _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w)
    return path_b if use_b else path_a


def _paint_alt(
    out: np.ndarray,
    path_pts: list[tuple[int, int]],
    colors: tuple[int, int],
    start_color: int,
) -> None:
    ca, cb = map(int, colors)
    cur = int(start_color)
    for rr, cc in path_pts:
        out[int(rr), int(cc)] = int(cur)
        cur = int(cb) if cur == int(ca) else int(ca)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover clipped interior paths from edge control pixels using mirrored projection across the frame.

    Input structure:
    - A uniform-background grid contains one largest 1px closed rectangular frame with uniform fill.
    - Colored single-pixel controls appear only on the outer border of the whole grid; distractor blobs may appear away from that border.

    Rule:
    - Output the frame interior, initialized to the fill color.
    - Each control pixel projects to the opposite interior edge while preserving its row or column: top->bottom, bottom->top, left->right, right->left.
    - Same-color controls form solid L paths or full-span stripes; one optional pair of singleton colors forms one alternating-color L path whose endpoint colors fix the phase.

    Key insight:
    - The source is minimal and misleading on purpose: the informative part is not just the control color, but that every border side maps to its mirrored side inside the frame rather than the nearest side.
    """
    g = np.array(grid, dtype=int)
    grid_h, grid_w = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_uniform_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)

    def clamp_row(rr: int) -> int:
        return max(0, min(out_h - 1, int(rr - (r0 + 1))))

    def clamp_col(cc: int) -> int:
        return max(0, min(out_w - 1, int(cc - (c0 + 1))))

    controls: list[tuple[str, tuple[int, int], int]] = []
    for cc in range(grid_w):
        top_val = int(g[0, cc])
        if top_val not in (int(bg), int(bc), int(fill)):
            controls.append(("top", (out_h - 1, clamp_col(cc)), int(top_val)))
        bottom_val = int(g[grid_h - 1, cc])
        if bottom_val not in (int(bg), int(bc), int(fill)):
            controls.append(("bottom", (0, clamp_col(cc)), int(bottom_val)))
    for rr in range(1, grid_h - 1):
        left_val = int(g[rr, 0])
        if left_val not in (int(bg), int(bc), int(fill)):
            controls.append(("left", (clamp_row(rr), out_w - 1), int(left_val)))
        right_val = int(g[rr, grid_w - 1])
        if right_val not in (int(bg), int(bc), int(fill)):
            controls.append(("right", (clamp_row(rr), 0), int(right_val)))

    by_color: dict[int, list[tuple[str, tuple[int, int]]]] = {}
    for side, pt, col in controls:
        by_color.setdefault(int(col), []).append((side, pt))

    singleton_colors = [int(col) for col, recs in by_color.items() if len(recs) == 1]
    if singleton_colors:
        assert len(singleton_colors) == 2
        ca, cb2 = map(int, singleton_colors)
        rec_a = by_color[int(ca)][0]
        rec_b = by_color[int(cb2)][0]
        path = _choose_path(rec_a[1], rec_b[1], out_h, out_w)

        chosen = None
        for path_pts, start_color, end_color in (
            (path, int(ca), int(cb2)),
            (list(reversed(path)), int(cb2), int(ca)),
        ):
            actual_end = int(start_color) if ((len(path_pts) - 1) % 2 == 0) else (int(cb2) if int(start_color) == int(ca) else int(ca))
            if int(actual_end) == int(end_color):
                chosen = (path_pts, int(start_color))
                break
        assert chosen is not None
        path_pts, start_color = chosen
        _paint_alt(out, path_pts, (int(ca), int(cb2)), int(start_color))

    for col, recs in by_color.items():
        if len(recs) < 2:
            continue
        sides = {side for side, _pt in recs}
        uniq = sorted({(int(pt[0]), int(pt[1])) for _side, pt in recs})

        if sides <= {"top", "bottom"} and ("top" in sides) and ("bottom" in sides):
            for cc in sorted({c for _r, c in uniq}):
                rs = [r for r, c in uniq if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue

        if sides <= {"left", "right"} and ("left" in sides) and ("right" in sides):
            for rr in sorted({r for r, _c in uniq}):
                cs = [c for r, c in uniq if r == rr]
                for cc in range(min(cs), max(cs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue

        if len(uniq) == 1:
            rr, cc = uniq[0]
            out[int(rr), int(cc)] = int(col)
            continue

        best = None
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                d = abs(uniq[i][0] - uniq[j][0]) + abs(uniq[i][1] - uniq[j][1])
                if best is None or d > best[0]:
                    best = (d, uniq[i], uniq[j])
        assert best is not None
        _dist, p1, p2 = best
        for rr, cc in _choose_path(p1, p2, out_h, out_w):
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 combo variant) ---------------

_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_FILL_EQ_BG_PROB = 0.6
_INTERIOR_H_RANGE = (4, 13)
_INTERIOR_W_RANGE = (7, 14)
_GRID_SIZE_RANGE = (20, 30)
_GRID_INTERIOR_PAD = 10
_FRAME_MARGIN = 4
_FRAME_MARGIN_FAR = 6
_N_COMPONENTS_RANGE = (2, 3)
_U_BRANCH_PROB = 0.7
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300
_L_SPEC_TRIES = 80
_U_SPEC_TRIES = 80
_L_PLACEMENT_TRIES = 40


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
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        grid[rs, cs] = [a if i % 2 == 0 else b for i in range(len(path))]


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    grid_h, grid_w = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if int(grid[r, c]) != int(bg):
            return False
        for nr, nc in neighbors4(r, c):
            if (nr, nc) not in s and 0 <= nr < grid_h and 0 <= nc < grid_w and comp_mask[nr, nc]:
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


def _place_controls(
    grid: np.ndarray,
    edge_mask: np.ndarray,
    controls: list[tuple[tuple[int, int], int]],
    bg: int,
) -> bool:
    pts = [pt for pt, _col in controls]
    if len(set(pts)) != len(pts):
        return False
    for (r, c), _col in controls:
        if int(grid[r, c]) != int(bg):
            return False
        if bool(edge_mask[r, c]):
            return False
    for (r, c), col in controls:
        grid[r, c] = int(col)
        edge_mask[r, c] = True
    return True


def _controls_free(
    grid: np.ndarray,
    edge_mask: np.ndarray,
    controls: list[tuple[tuple[int, int], int]],
    bg: int,
) -> bool:
    pts = [pt for pt, _col in controls]
    if len(set(pts)) != len(pts):
        return False
    for (r, c), _col in controls:
        if int(grid[r, c]) != int(bg):
            return False
        if bool(edge_mask[r, c]):
            return False
    return True


def _sample_l_specs(
    kind: str,
    inner_h: int,
    inner_w: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    require_even_len: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    for _ in range(_L_SPEC_TRIES):
        r = random.randint(1, max(1, inner_h // 2)) if is_top else random.randint(max(1, inner_h // 2), inner_h - 2)
        c = random.randint(1, max(1, inner_w // 2)) if is_left else random.randint(max(1, inner_w // 2), inner_w - 2)
        edge_r = 0 if is_top else inner_h - 1
        edge_c = 0 if is_left else inner_w - 1
        inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
        if require_even_len and (len(inner) % 2 != 0):
            continue
        ctrl_a = (rows - 1 if is_top else 0, c0 + 1 + c)
        ctrl_b = (r0 + 1 + r, cols - 1 if is_left else 0)
        return inner, ctrl_a, ctrl_b
    return None


def _sample_u_specs(
    orientation: str,
    inner_h: int,
    inner_w: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    for _ in range(_U_SPEC_TRIES):
        if orientation == "vertical":
            c_a, c_b = sorted(random.sample(range(1, inner_w - 1), 2))
            if c_b - c_a < 2:
                continue
            inner = [[(r, c_a) for r in range(inner_h)], [(r, c_b) for r in range(inner_h)]]
            controls = [
                (0, c0 + 1 + c_a),
                (rows - 1, c0 + 1 + c_a),
                (0, c0 + 1 + c_b),
                (rows - 1, c0 + 1 + c_b),
            ]
            return inner, controls
        r_a, r_b = sorted(random.sample(range(1, inner_h - 1), 2))
        if r_b - r_a < 2:
            continue
        inner = [[(r_a, c) for c in range(inner_w)], [(r_b, c) for c in range(inner_w)]]
        controls = [
            (r0 + 1 + r_a, 0),
            (r0 + 1 + r_a, cols - 1),
            (r0 + 1 + r_b, 0),
            (r0 + 1 + r_b, cols - 1),
        ]
        return inner, controls
    return None


def _add_distractor(
    grid: np.ndarray,
    comp_mask: np.ndarray,
    bg: int,
    frame: tuple[int, int, int, int],
    color: int,
) -> None:
    grid_h, grid_w = grid.shape
    r0, c0, r1, c1 = frame
    gap = _DISTRACTOR_CLEARANCE
    sz_lo, sz_hi = _DISTRACTOR_SIZE_RANGE
    for _ in range(_DISTRACTOR_TRIES):
        size = random.randint(sz_lo, sz_hi)
        r, c = random.randint(1, grid_h - 2), random.randint(1, grid_w - 2)
        coords = [(r, c)]
        used = {(r, c)}
        for _ in range(size - 1):
            base = random.choice(coords)
            opts = [
                (nr, nc)
                for nr, nc in neighbors4(*base)
                if 1 <= nr < grid_h - 1 and 1 <= nc < grid_w - 1 and (nr, nc) not in used
            ]
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
    inner_h = random.randint(*_INTERIOR_H_RANGE)
    inner_w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], inner_h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], inner_w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - inner_h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - inner_w - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + inner_h + 1, c0 + inner_w + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((inner_h, inner_w), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    edge_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((inner_h, inner_w), dtype=bool)

    use_u_branch = inner_h >= 5 and random.random() < _U_BRANCH_PROB
    remaining = 1 if use_u_branch else random.randint(*_N_COMPONENTS_RANGE)
    color_i = 0
    dual_used = False
    l_count = 0

    # U remains valid under mirrored projection, but we forbid U-only samples so the mirror rule stays observable.
    if use_u_branch:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and inner_h < 5:
            orientation = "vertical"
        specs = _sample_u_specs(orientation, inner_h, inner_w, rows, cols, r0, c0)
        if specs is None:
            return None
        inner_lines, ctrl_pts = specs
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        color = int(palette[color_i])
        color_i += 1
        u_controls = [(pt, color) for pt in ctrl_pts]
        if not _controls_free(grid, edge_mask, u_controls, bg):
            return None
        if not _place_controls(grid, edge_mask, u_controls, bg):
            return None
        expected[irs, ics] = color
        used_inner[irs, ics] = True

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        want_dual = (not dual_used) and (color_i + 1 < len(palette)) and (random.random() < _DUAL_COLOR_PROB)
        placed = False
        for _ in range(_L_PLACEMENT_TRIES):
            specs = _sample_l_specs(kind, inner_h, inner_w, rows, cols, r0, c0, require_even_len=want_dual)
            if specs is None:
                continue
            inner, ctrl_a, ctrl_b = specs
            irs, ics = zip(*inner)
            if used_inner[irs, ics].any():
                continue

            if want_dual:
                colors = (int(palette[color_i]), int(palette[color_i + 1]))
                controls = [(ctrl_a, int(colors[0])), (ctrl_b, int(colors[1]))]
            else:
                colors = (int(palette[color_i]),)
                controls = [(ctrl_a, int(colors[0])), (ctrl_b, int(colors[0]))]

            if not _controls_free(grid, edge_mask, controls, bg):
                continue
            if not _place_controls(grid, edge_mask, controls, bg):
                continue

            _paint_path(expected, inner, colors)
            used_inner[irs, ics] = True
            if want_dual:
                color_i += 2
                dual_used = True
            else:
                color_i += 1
            l_count += 1
            placed = True
            break
        if not placed:
            return None

    if l_count == 0:
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
