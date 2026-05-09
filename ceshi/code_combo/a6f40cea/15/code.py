# Combo variant of a6f40cea (index 15)
# combo_spec: D6/1+D15/1
# Name: 部分路径补全填充

from __future__ import annotations

# Design analysis for the combo variant
# - concretization and behavior trace:
#   - 原题里，框内完全空白；solver 只需把框外 open-rect 的 ring 触点映射成内部 L 线或成对直线。
#   - D15 变化要求框内先出现一小段真实路径骨架 clue；D6 变化要求路径补全后，不停在线上，而是把该路径与裁剪边界围成的区域整块填满。
#   - 因此本融合版采用“两阶段”推理：先用 ring 触点枚举候选骨架 path，再用框内 clue 选择正确 path/双色相位，最后把已确定 path 转成角落矩形块、双色棋盘块或整条带。
#   - 具体例子：若外部组件编码 top+left 的 L，且框内只给出该 L 中段 3 个彩色格，则 solver 先判断两种肘点候选里哪一种覆盖 clue，
#     补全整条 L，再把该 L 与左上裁剪边界围出的矩形区域 flood fill；若是双色 L，则整块按相对角点的曼哈顿奇偶扩展成双色块。
# - complexity calibration:
#   - 保留原题的 closed frame、外部 open-rect、可选 U-pair 和 distractor，不引入新对象类别。
#   - clue 长度固定在 2-4 格，既足够可见，又不会直接把完整 path 给出。
#   - flood-fill 只允许两类区域：角落矩形块和全跨度带状区，视觉差异明显但自由度仍受控。
# - implicit assumptions audit:
#   - 原 `_find_framed_rect_uniform_fill()` 假设内部全单色；现在 clue 会破坏它，所以改成“完整边框 + 内部多数色”为 fill 的检测。
#   - 原外部组件掩码会把框内 clue 一并算进去；现在必须显式排除整个框内区域，只在框外做连通域。
#   - 原 solver 选择 path 后直接逐点上色；现在 path 只用于匹配 clue 和双色相位，最终渲染改成按区域填充，否则会退化回 D15。
#   - 原 generator 的 `used_inner` 只占骨架 path；现在输出是面状区域，必须按整块/整带做占用检查，避免不同组件在输出里互相覆盖。
# - placement legality:
#   - generator 先构造最终 `expected` flood-fill 输出，再从每个组件的骨架 path 上截取 clue 写回输入，因此 clue 必然合法且属于目标结构。
#   - L 组件用角块 bbox 占用检查，U-pair 用整条带 bbox 占用检查；外部组件仍保持不重叠、不越界、不与已有组件 4 邻接。
# - visual observability and cleanliness:
#   - 输入里可以直接看见“框外几何关系 + 框内残缺 path”，输出里则变成同一结构对应的整块填充，人类可从少量样例观察到“先补线，再灌区域”。
#   - 每个结构使用独立单色或独立双色对；双色只出现在 L 角块上，不把带状区也做成二维双色，避免画面过乱。
# - new solver assumptions and how generation enforces them:
#   - 同一单色只对应一个目标结构；若是 U-pair，则两条边界线共享同色并共同定义一条带。generator 通过调色板唯一分配保证这一点。
#   - clue 永远是某条真实骨架 path 的连续子段，不会漂浮在区域内部。generator 只从骨架本身切段。
#   - 双色 checker 块的相位仍由 ring 端点颜色约束，clue 只负责在多候选 path 之间打破平局；generator 延续原始交替上色顺序来满足这个前提。

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


def _find_framed_rect_majority_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int] | None:
    """
    Detect the main 1px rectangular frame even when sparse clue pixels appear inside.

    Returns (r0, c0, r1, c1, border_color, fill_color) for the largest valid frame.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, fill_count, r0, c0, r1, c1, bc, fill)
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
            fill, fill_count = Counter(map(int, inner.flatten())).most_common(1)[0]
            fill = int(fill)
            fill_count = int(fill_count)
            if fill == int(bc):
                continue
            if fill_count * 2 <= int(inner.size):
                continue

            area = int(h * w)
            cand = (area, fill_count, r0, c0, r1, c1, int(bc), int(fill))
            if best is None or cand[:2] > best[:2]:
                best = cand

    if best is None:
        return None
    _area, _fill_count, r0, c0, r1, c1, bc, fill = best
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
    rr, cc = (rb, ca) if corner_a else (ra, cb)
    return (0 < rr < out_h - 1) and (0 < cc < out_w - 1)


def _path_color_values(length: int, colors: tuple[int, ...], start_index: int = 0) -> list[int]:
    if len(colors) == 1:
        return [int(colors[0])] * int(length)
    a, b = map(int, colors)
    return [a if (start_index + i) % 2 == 0 else b for i in range(int(length))]


def _paint_path(
    grid: np.ndarray,
    path: list[tuple[int, int]],
    colors: tuple[int, ...],
    *,
    start_index: int = 0,
) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    grid[rs, cs] = _path_color_values(len(path), colors, start_index=start_index)


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
    out_h: int,
    out_w: int,
    corner_a: bool,
) -> tuple[tuple[int, int] | None, tuple[int, int] | None]:
    if p1[0] == p2[0] or p1[1] == p2[1]:
        return None, None
    row_edge = None
    if p1[0] == 0 or p2[0] == 0:
        row_edge = 0
    elif p1[0] == out_h - 1 or p2[0] == out_h - 1:
        row_edge = out_h - 1
    col_edge = None
    if p1[1] == 0 or p2[1] == 0:
        col_edge = 0
    elif p1[1] == out_w - 1 or p2[1] == out_w - 1:
        col_edge = out_w - 1
    if row_edge is None or col_edge is None:
        return None, None
    elbow = (int(p2[0]), int(p1[1])) if corner_a else (int(p1[0]), int(p2[1]))
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


def _score_mono_candidate(
    path: list[tuple[int, int]],
    color: int,
    clue_map: dict[tuple[int, int], int],
    *,
    prefer_inside: bool,
) -> tuple[int, int, int]:
    path_set = set(path)
    relevant = [pos for pos, col in clue_map.items() if int(col) == int(color)]
    covered = sum(1 for pos in relevant if pos in path_set)
    return (1 if covered == len(relevant) else 0, covered, 1 if prefer_inside else 0)


def _score_alt_candidate(
    path: list[tuple[int, int]],
    colors: tuple[int, int],
    start_color: int,
    clue_map: dict[tuple[int, int], int],
    *,
    prefer_inside: bool,
) -> tuple[int, int, int, int]:
    values = _path_color_values(len(path), colors)
    if int(start_color) != int(colors[0]):
        values = _path_color_values(len(path), (int(colors[1]), int(colors[0])))
    assigned = {pos: val for pos, val in zip(path, values)}
    palette = {int(colors[0]), int(colors[1])}
    relevant = [(pos, col) for pos, col in clue_map.items() if int(col) in palette]
    match = sum(1 for pos, col in relevant if assigned.get(pos) == int(col))
    miss = sum(1 for pos, _col in relevant if pos not in assigned)
    wrong = sum(1 for pos, col in relevant if pos in assigned and assigned[pos] != int(col))
    return (1 if wrong == 0 and miss == 0 else 0, match, -wrong, 1 if prefer_inside else 0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover partially shown internal path skeletons, then flood the regions those paths would enclose.

    Input structure:
    - One 1px closed rectangular frame sits on a uniform background; its interior is mostly a single fill color but now includes sparse path-colored clue pixels.
    - Colored open-rectangle components remain outside the frame and touch the ring around it, encoding the same endpoint geometry as the original task.

    Rule:
    - Crop the output to the frame interior and start from the majority interior fill color.
    - Use each outside component to reconstruct the candidate Manhattan path or full-span stripe it implies, and use the internal clue pixels to choose the consistent completion and alternating phase.
    - After the missing path is inferred, do not stop at the line: flood the enclosed corner block or band region with the path color pattern.

    Key insight: The outer traces still determine where each path must meet the cropped border, while the sparse internal clues reveal which candidate skeleton is already present; once that skeleton is fixed, the filled region is geometrically forced.
    """
    g = np.array(grid, dtype=int)
    grid_h, grid_w = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_majority_fill(g, bg=int(bg))
    assert found is not None
    r0, c0, r1, c1, bc, fill = found
    out_h = int(r1 - r0 - 1)
    out_w = int(c1 - c0 - 1)
    out = np.full((out_h, out_w), int(fill), dtype=int)
    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    clue_map = {
        (int(rr), int(cc)): int(inner[rr, cc])
        for rr in range(out_h)
        for cc in range(out_w)
        if int(inner[rr, cc]) != int(fill)
    }
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        return (
            max(0, min(out_h - 1, int(rr - r_int0))),
            max(0, min(out_w - 1, int(cc - c_int0))),
        )

    outer_mask = (g != int(bg)) & (g != int(bc))
    outer_mask[r0 : r1 + 1, c0 : c1 + 1] = False
    clab, cn = label(outer_mask, structure=STRUCT4)

    ring = set()
    for cc in range(c0, c1 + 1):
        for rr in (r0 - 1, r1 + 1):
            if 0 <= rr < grid_h:
                ring.add((int(rr), int(cc)))
    for rr in range(r0, r1 + 1):
        for cc in (c0 - 1, c1 + 1):
            if 0 <= cc < grid_w:
                ring.add((int(rr), int(cc)))

    comp_pts: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        mr, mc = map_to_out(rr, cc)
        comp_pts[cid].append((int(mr), int(mc), int(g[rr, cc])))

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
        for rr, cc, col in pts:
            hist[(int(rr), int(cc))][int(col)] += 1
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

            candidates = []
            for corner_a in (True, False):
                path = _build_manhattan(p1, p2, corner_a)
                score = _score_mono_candidate(
                    path,
                    colv,
                    clue_map,
                    prefer_inside=_inside_corner(p1, p2, corner_a, out_h, out_w),
                )
                candidates.append((score, corner_a))
            _score, chosen_corner_a = max(candidates, key=lambda item: item[0])
            corner, elbow = _edge_corner_and_elbow(p1, p2, out_h, out_w, chosen_corner_a)
            if corner is None or elbow is None:
                chosen = _build_manhattan(p1, p2, chosen_corner_a)
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

            candidates = []
            for corner_a in (True, False):
                base_path = _build_manhattan(a, b, corner_a)
                prefer_inside = _inside_corner(a, b, corner_a, out_h, out_w)
                for path_pts, start_hint, end_hint in (
                    (base_path, ha, hb),
                    (list(reversed(base_path)), hb, ha),
                ):
                    for start_color in colors:
                        vals = _path_color_values(
                            len(path_pts),
                            colors if int(start_color) == int(ca) else (int(cb), int(ca)),
                        )
                        if int(vals[0]) != int(start_hint) or int(vals[-1]) != int(end_hint):
                            continue
                        score = _score_alt_candidate(
                            path_pts,
                            colors,
                            int(start_color),
                            clue_map,
                            prefer_inside=prefer_inside,
                        )
                        candidates.append((score, corner_a))

            assert candidates
            _score, chosen_corner_a = max(candidates, key=lambda item: item[0])
            corner, elbow = _edge_corner_and_elbow(a, b, out_h, out_w, chosen_corner_a)
            if corner is None or elbow is None:
                path = _build_manhattan(a, b, chosen_corner_a)
                chosen = None
                for path_pts in (path, list(reversed(path))):
                    for start_color in colors:
                        vals = _path_color_values(
                            len(path_pts),
                            colors if int(start_color) == int(ca) else (int(cb), int(ca)),
                        )
                        if int(vals[0]) != int(ha) or int(vals[-1]) != int(hb):
                            continue
                        chosen = (path_pts, int(start_color))
                        break
                    if chosen is not None:
                        break
                assert chosen is not None
                path_pts, start_color = chosen
                _paint_path(
                    out,
                    path_pts,
                    colors if int(start_color) == int(ca) else (int(cb), int(ca)),
                )
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
        bots = [p for p in pts if p[0] == out_h - 1]
        if tops and bots:
            c_lo = min(c for _r, c in pts)
            c_hi = max(c for _r, c in pts)
            out[:, c_lo : c_hi + 1] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == out_w - 1]
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
        candidates = []
        for corner_a in (True, False):
            path = _build_manhattan(p1, p2, corner_a)
            score = _score_mono_candidate(
                path,
                int(col),
                clue_map,
                prefer_inside=_inside_corner(p1, p2, corner_a, out_h, out_w),
            )
            candidates.append((score, corner_a))
        _score, chosen_corner_a = max(candidates, key=lambda item: item[0])
        corner, elbow = _edge_corner_and_elbow(p1, p2, out_h, out_w, chosen_corner_a)
        if corner is None or elbow is None:
            chosen = _build_manhattan(p1, p2, chosen_corner_a)
            for rr, cc in chosen:
                out[int(rr), int(cc)] = int(col)
            continue
        r_lo, r_hi, c_lo, c_hi = _rect_bounds(corner, elbow)
        out[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(col)

    return out


# --------------- generator (a6f40cea combo D6/1 + D15/1) ---------------
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
_CLUE_SEG_LEN_RANGE = (2, 4)
_GEN_RETRIES = 300


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    grid_h, grid_w = grid.shape
    s = set(coords)
    for r, c in s:
        if not (0 <= r < grid_h and 0 <= c < grid_w):
            return False
        if grid[r, c] != bg:
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


def _l_region_meta(
    kind: str,
    h: int,
    w: int,
    r: int,
    c: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    if kind == "tl":
        return (0, r, 0, c), (0, 0), (0, c)
    if kind == "tr":
        return (0, r, c, w - 1), (0, w - 1), (0, c)
    if kind == "bl":
        return (r, h - 1, 0, c), (h - 1, 0), (h - 1, c)
    return (r, h - 1, c, w - 1), (h - 1, w - 1), (h - 1, c)


def _paint_l_region(
    expected: np.ndarray,
    kind: str,
    h: int,
    w: int,
    r: int,
    c: int,
    colors: tuple[int, ...],
) -> tuple[int, int, int, int]:
    bounds, corner, start = _l_region_meta(kind, h, w, r, c)
    r_lo, r_hi, c_lo, c_hi = bounds
    if len(colors) == 1:
        expected[r_lo : r_hi + 1, c_lo : c_hi + 1] = int(colors[0])
    else:
        a, b = map(int, colors)
        d0 = abs(int(start[0]) - int(corner[0])) + abs(int(start[1]) - int(corner[1]))
        corner_color = int(a) if d0 % 2 == 0 else int(b)
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


def _l_specs(kind: str, h: int, w: int, r0: int, c0: int, r1: int, c1: int):
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"

    r = random.randint(1, max(1, h // 2)) if is_top else random.randint(max(1, h // 2), h - 2)
    c = random.randint(1, max(1, w // 2)) if is_left else random.randint(max(1, w // 2), w - 2)

    lo, hi = _OUTER_OFFSET_RANGE
    outer_r = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    outer_c = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    ring_a = (r0 - 1 if is_top else r1 + 1, c0 + 1 + c)
    ring_b = (r0 + 1 + r, c0 - 1 if is_left else c1 + 1)

    edge_r = 0 if is_top else h - 1
    edge_c = 0 if is_left else w - 1
    inner = _join(_vpath(edge_r, r, c), _hpath(r, c, edge_c))
    outer = _join(
        _vpath(ring_a[0], outer_r, ring_a[1]),
        _hpath(outer_r, ring_a[1], outer_c),
        _vpath(outer_r, ring_b[0], outer_c),
        _hpath(ring_b[0], outer_c, ring_b[1]),
    )
    return inner, outer, r, c


def _u_pair_specs(orientation: str, h: int, w: int, r0: int, c0: int, r1: int, c1: int):
    lo, hi = _OUTER_OFFSET_RANGE
    if orientation == "vertical":
        col_pairs = [(a, b) for a in range(1, w - 1) for b in range(a + 2, w - 1)]
        c_a, c_b = random.choice(col_pairs)
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
        inner = [[(r, c_a) for r in range(h)], [(r, c_b) for r in range(h)]]
        band = (0, h - 1, c_a, c_b)
    else:
        row_pairs = [(a, b) for a in range(1, h - 1) for b in range(a + 2, h - 1)]
        r_a, r_b = random.choice(row_pairs)
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
        inner = [[(r_a, c) for c in range(w)], [(r_b, c) for c in range(w)]]
        band = (r_a, r_b, 0, w - 1)
    return inner, outer_a, outer_b, band


def _choose_clue_segment(path: list[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
    n = len(path)
    max_len = max(1, min(_CLUE_SEG_LEN_RANGE[1], n - 1))
    min_len = min(_CLUE_SEG_LEN_RANGE[0], max_len)
    seg_len = random.randint(min_len, max_len)
    if n - seg_len >= 2:
        start_lo, start_hi = 1, n - seg_len - 1
    else:
        start_lo, start_hi = 0, n - seg_len
    start = random.randint(start_lo, start_hi)
    return start, path[start : start + seg_len]


def _paint_clue_segment(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    start, seg = _choose_clue_segment(path)
    _paint_path(grid, seg, colors, start_index=start)


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
        if _place_component(grid, comp_mask, coords, (color,), bg):
            return


def _gen_attempt() -> np.ndarray | None:
    bg = random.choice(_BG_CHOICES)
    bc = _BORDER_COLOR
    fill = bg if random.random() < _FILL_EQ_BG_PROB else random.choice([x for x in range(10) if x not in (bg, bc)])
    h = random.randint(*_INTERIOR_H_RANGE)
    w = random.randint(*_INTERIOR_W_RANGE)
    rows = random.randint(max(_GRID_SIZE_RANGE[0], h + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    cols = random.randint(max(_GRID_SIZE_RANGE[0], w + _GRID_INTERIOR_PAD), _GRID_SIZE_RANGE[1])
    r0 = random.randint(_FRAME_MARGIN, rows - h - _FRAME_MARGIN_FAR)
    c0 = random.randint(_FRAME_MARGIN, cols - w - _FRAME_MARGIN_FAR)
    r1, c1 = r0 + h + 1, c0 + w + 1

    grid = np.full((rows, cols), bg, dtype=int)
    grid[r0, c0 : c1 + 1] = bc
    grid[r1, c0 : c1 + 1] = bc
    grid[r0 : r1 + 1, c0] = bc
    grid[r0 : r1 + 1, c1] = bc
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill

    expected = np.full((h, w), fill, dtype=int)
    clues = np.full((h, w), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    used_inner = np.zeros((h, w), dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0
    clue_count = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and h < 5:
            orientation = "vertical"
        color = palette[color_i]
        color_i += 1
        inner_lines, outer_a, outer_b, band = _u_pair_specs(orientation, h, w, r0, c0, r1, c1)
        if not _bbox_is_free(used_inner, band):
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        _paint_band(expected, band, color)
        _mark_bbox(used_inner, band)
        for line in inner_lines:
            _paint_clue_segment(clues, line, (color,))
            clue_count += 1
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        inner_path, outer, r, c = _l_specs(kind, h, w, r0, c0, r1, c1)
        bounds, _corner, _start = _l_region_meta(kind, h, w, r, c)
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
        _paint_l_region(expected, kind, h, w, r, c, colors)
        _mark_bbox(used_inner, bounds)
        _paint_clue_segment(clues, inner_path, colors)
        clue_count += 1

    if clue_count == 0:
        return None

    grid[r0 + 1 : r1, c0 + 1 : c1] = clues

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
    raise RuntimeError("failed to generate a valid a6f40cea combo input")
