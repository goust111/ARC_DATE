# Variant of a6f40cea
# Dimension: D3 — 源对象类型 (src)
# Change: 外部开放矩形框架（L形：2条完整边+2条截断stub；或U形：3条完整边），通过触碰闭合框边来标记端点位置... → 边缘控制像素对
"""
Design analysis for the D3 variant
==================================

variant_spec:
  preserved_core:
    - 输入仍然有一个最大的 1px 闭合矩形框，框内是统一填充色。
    - 输出仍然只裁剪框内区域，并在填充底色上画单色或双色交替的曼哈顿路径。
    - L 型路径仍然优先选择“拐点严格落在内部”的那条曼哈顿折线。
    - 原题可选的 U 情况仍然保留为“整列/整行条纹”，只是源对象的编码方式变了。
  new_source_encoding:
    - 删除所有靠近框外侧的开放矩形框。
    - 改为在整张网格的外边界放置控制像素。
    - 顶边/底边控制像素的列坐标直接投影到框内部的列端点。
    - 左边/右边控制像素的行坐标直接投影到框内部的行端点。
    - 同色两像素表示一个单色组件。
    - 恰好一组双色 L 组件时，用两个只出现一次的颜色分别标记两个端点；颜色同时决定交替相位。
    - U 型双条纹保留为“两对同色控制像素”：例如顶边两点 + 底边两点，同色，投影成两条整列条纹。

behavior_trace:
  original_local_step:
    - solver 先找框，再找框外接触 ring 的彩色连通块。
    - 连通块真正被用到的，是它在 ring 上提供的端点投影坐标，以及双色时端点颜色提示。
    - 外部开放框的大段几何本身并不进入输出。
  variant_local_step:
    - 直接把“端点投影坐标 + 端点颜色提示”压缩成边界单像素。
    - 例如：顶边某列一个红点、左边某行一个蓝点，投影到框内后就是一条红蓝交替 L 路径的两个端点。
    - 例如：顶边两列和底边同两列各放一个黄点，投影后就是两条黄色整列条纹。
  difference:
    - 原题的外部源对象是长折线/开放框；变体里源对象退化成极简控制标记。
    - 识别难点从“理解外部框架的缺口和触点”转成“理解边界像素的坐标投影规则”。

complexity_calibration:
  - 不把所有组件都做成双色，否则只靠 2-4 个样例很难归纳配对关系。
  - 最多允许 1 个双色 L 组件；其余组件保持单色，同色两点天然成对。
  - 保留原题的 U 分支，但仍然只允许单色，且一次 U 分支消耗 2 个组件槽位，维持原题复杂度量级。
  - 控制像素只出现在整张网格最外圈，框附近不再出现长外部结构，视觉差异足够明显。

implicit_assumption_audit:
  - 原 solver 的 comp_mask / 连通域标号依赖“源对象是 4 连通外部组件”；变体中这个前提失效，必须改为直接读网格外边界像素。
  - 原 solver 的 ring 投影依赖“对象至少在 ring 上留下两个接触点”；变体中接触点不再存在，改为边缘像素按行/列直接投影。
  - 原 solver 的 deferred 逻辑依赖“同色 U 对在 ring 上给出多点”；变体仍保留这一思想，但多点来自边界同色像素集合而非外部连通块。
  - 原 solver 对双色组件使用端点颜色锚定相位；这个假设仍然成立，但现在端点颜色来自两个单独控制像素，因此 generator 必须保证双色路径长度奇偶性与端点颜色兼容。

placement_legality:
  - 控制像素只能放在整张网格的最外边界，且其列/行必须落在框内部投影范围内。
  - 不允许两个控制像素占同一格。
  - U 分支要求两列或两行相隔至少 2，避免视觉上粘成一段。
  - 内部路径仍然用 output-first 构造，并通过 used_inner 保证互不重叠。
  - distractor 仍然远离框，且不放在边界，避免和控制像素混淆。

visual_observability:
  - 看样例时，人类会明显看到：外部不再有大型开放框，只有边缘散落的 4-6 个控制点。
  - 控制点与输出路径之间的关系由“同列/同行对齐到框内边界”给出，是像素级可见关系，不依赖颜色 id 排序。
  - 双色情况只出现单个 L，避免需要从少量样例里推断多组 singleton 颜色如何配对。

visual_cleanliness:
  - 控制像素和闭合框完全分层：一个在整图外圈，一个在中部。
  - 内部路径仍然不重叠，distractor 远离主结构，减少杂讯。

color_inference:
  - 单色路径: 同色控制像素 => 同色输出路径/条纹。
  - 双色路径: 两端点像素颜色直接就是交替色集合；端点本身也给出相位。
  - 不使用颜色大小、颜色排序、黑色特殊性等不可见规则。
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


def _choose_path(p1: tuple[int, int], p2: tuple[int, int], out_h: int, out_w: int) -> list[tuple[int, int]]:
    path_a = _build_manhattan(p1, p2, True)
    path_b = _build_manhattan(p1, p2, False)
    use_b = _inside_corner(p1, p2, False, out_h, out_w) and not _inside_corner(p1, p2, True, out_h, out_w)
    return path_b if use_b else path_a


def _paint_alt(out: np.ndarray, path_pts: list[tuple[int, int]], colors: tuple[int, int], start_color: int) -> None:
    ca, cb = map(int, colors)
    cur = int(start_color)
    for rr, cc in path_pts:
        out[int(rr), int(cc)] = int(cur)
        cur = int(cb) if cur == int(ca) else int(ca)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct clipped interior paths from edge control pixels around a framed rectangle.

    Input structure:
    - bg, a 1px rectangular frame (border color) with uniform interior fill.
    - Colored single pixels on the outer grid border; their rows/cols align with path endpoints inside the frame.

    Rule:
    - Output = the frame's interior, initialized with the fill color.
    - Project each border control pixel inward along its row or column onto the corresponding interior edge.
    - Same-color control sets produce solid Manhattan paths or full-span stripes; one optional two-color singleton pair produces an alternating-color L path.

    Key insight: The external geometry is gone; the only data source is border pixel position plus endpoint color, so solving means pairing controls and projecting their coordinates into the frame.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
    for cc in range(W):
        for rr, side in ((0, "top"), (H - 1, "bottom")):
            val = int(g[rr, cc])
            if val in (int(bg), int(bc), int(fill)):
                continue
            pt = (0, clamp_col(cc)) if side == "top" else (out_h - 1, clamp_col(cc))
            controls.append((side, pt, val))
    for rr in range(1, H - 1):
        for cc, side in ((0, "left"), (W - 1, "right")):
            val = int(g[rr, cc])
            if val in (int(bg), int(bc), int(fill)):
                continue
            pt = (clamp_row(rr), 0) if side == "left" else (clamp_row(rr), out_w - 1)
            controls.append((side, pt, val))

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


# --------------- generator (a6 D3 variant) ---------------

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
_U_PAIR_PROB = 0.25
_DUAL_COLOR_PROB = 0.45
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 300
_L_SPEC_TRIES = 80
_U_SPEC_TRIES = 80


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
    H, W = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != int(bg):
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


def _sample_l_specs(
    kind: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
    require_even_len: bool,
) -> tuple[list[tuple[int, int]], tuple[int, int], tuple[int, int]] | None:
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    for _ in range(_L_SPEC_TRIES):
        r = random.randint(1, max(1, H // 2)) if is_top else random.randint(max(1, H // 2), H - 2)
        c = random.randint(1, max(1, W // 2)) if is_left else random.randint(max(1, W // 2), W - 2)
        edge_r = 0 if is_top else H - 1
        edge_c = 0 if is_left else W - 1
        inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
        if require_even_len and (len(inner) % 2 != 0):
            continue
        ctrl_a = (0 if is_top else rows - 1, c0 + 1 + c)
        ctrl_b = (r0 + 1 + r, 0 if is_left else cols - 1)
        return inner, ctrl_a, ctrl_b
    return None


def _sample_u_specs(
    orientation: str,
    H: int,
    W: int,
    rows: int,
    cols: int,
    r0: int,
    c0: int,
) -> tuple[list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    for _ in range(_U_SPEC_TRIES):
        if orientation == "vertical":
            c_a, c_b = sorted(random.sample(range(1, W - 1), 2))
            if c_b - c_a < 2:
                continue
            inner = [[(r, c_a) for r in range(H)], [(r, c_b) for r in range(H)]]
            controls = [
                (0, c0 + 1 + c_a),
                (rows - 1, c0 + 1 + c_a),
                (0, c0 + 1 + c_b),
                (rows - 1, c0 + 1 + c_b),
            ]
            return inner, controls
        r_a, r_b = sorted(random.sample(range(1, H - 1), 2))
        if r_b - r_a < 2:
            continue
        inner = [[(r_a, c) for c in range(W)], [(r_b, c) for c in range(W)]]
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
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    edge_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0
    dual_used = False

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        specs = _sample_u_specs(orientation, H, W, rows, cols, r0, c0)
        if specs is None:
            return None
        inner_lines, ctrl_pts = specs
        all_inner = [p for line in inner_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        color = int(palette[color_i])
        color_i += 1
        if not _place_controls(grid, edge_mask, [(pt, color) for pt in ctrl_pts], bg):
            return None
        expected[irs, ics] = color
        used_inner[irs, ics] = True
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        want_dual = (not dual_used) and (color_i + 1 < len(palette)) and (random.random() < _DUAL_COLOR_PROB)
        specs = _sample_l_specs(kind, H, W, rows, cols, r0, c0, require_even_len=want_dual)
        if specs is None:
            return None
        inner, ctrl_a, ctrl_b = specs
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None

        if want_dual:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
            controls = [(ctrl_a, int(colors[0])), (ctrl_b, int(colors[1]))]
            dual_used = True
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
            controls = [(ctrl_a, int(colors[0])), (ctrl_b, int(colors[0]))]

        if not _place_controls(grid, edge_mask, controls, bg):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True

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
    raise RuntimeError("failed to generate a valid a6f40cea D3 input")
