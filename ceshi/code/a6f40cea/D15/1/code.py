# Variant of a6f40cea
# Dimension: D15 — 内部填充色基底 (meta)
# Change: 输出基底为封闭框架的内部填充色（均匀单色），可能等于背景色... → 内部有预设部分路径

# Design analysis:
# spec:
#   variant_core:
#     closed_frame:
#       border: 与原题相同，仍是 1px 完整矩形框
#       interior: 不再要求全为单色；内部仍以 fill 为多数底色，但会预先出现少量路径像素
#     visible_change:
#       input: 每条目标路径在框内先给出 1 段连续彩色子段，颜色与最终路径一致
#       output: 仍输出整个框内区域，但要把这些已知子段补全成完整 L 路径或整条条纹
#   concretization_and_trace:
#     - 原题里，一个 top+left 的外部 L 组件只给出框外开口矩形；框内完全空白，solver 直接从两个映射端点画完整 L。
#     - 现在同一个组件会在框内先出现 2-4 个连续彩色格，通常位于最终路径中段；solver 先识别这些格属于哪条候选路径，再补齐缺失腿段。
#     - 差异不在输出形状，而在输入构造和求解过程：从“全空白绘制”改成“部分已知的补全”。
#   complexity_calibration:
#     - 不新增新的对象类别，仍只使用原题的 closed frame、outer open-rect、distractor。
#     - 每条内部路径只暴露一小段连续线索，长度 2-4，既肉眼可见，又不会让整条路径几乎直接给出。
#     - 颜色仍沿用路径颜色，不引入额外颜色映射；组件颜色集保持互异，避免线索归属歧义。
#   implicit_assumptions_audit:
#     - 原 `_find_framed_rect_uniform_fill()` 假设内部完全单色；变体下失效，因此改为“完整边框 + 内部多数色为 fill”的检测。
#     - 原 `comp_mask = (g != bg) & (g != fill) & (g != bc)` 在原题无害，因为框内没有非 fill 像素；现在会把内部线索也纳入组件标记。
#       新 solver 显式把框内区域从 outer-component 标记中排除，只分析框外 ring-touching 组件。
#     - 原路径选择主要靠 `inside_corner()`；现在若继续无条件使用它，可能忽略框内已知子段，所以改为“先匹配线索，再用 inside 作为并列 tie-breaker”。
#     - 原 `out` 先整块填 fill 再覆盖是安全的；现在仍可这样做，但必须保证后续候选路径与输入线索一致，否则会覆盖掉本应保留的已知像素。
#   placement_legality:
#     - generator 仍先构造完整 expected output，再从每条路径切出严格短于全长的 clue 子段写回输入。
#     - clue 只取自各自最终路径，因此不会越界，也不会与其他路径重叠；原 `used_inner` 继续保证完整目标路径彼此不重叠。
#     - 外部组件放置逻辑保持原题约束，distractor 仍不得接触 ring。
#   observability_and_cleanliness:
#     - 人类在样例中能直接看到：外部开口框仍提示连线关系，但框内已出现彩色残段，输出只是把这些残段延长补全。
#     - 线索只出现在框内，不与外部组件重叠；数量少、位置规整，避免画面拥挤。
#   color_inference:
#     - 不按颜色 id 做任何排序或变换，只把颜色当作可见标签。
#     - 两色交替路径的 clue 保留原始奇偶相位，solver 用 ring 端点颜色和 clue 共同约束相位，规则可由样例直接观察。

from __future__ import annotations

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
    Detect the main 1px rectangular frame even when the interior contains sparse clue pixels.

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


def _inside_corner(p1: tuple[int, int], p2: tuple[int, int], corner_a: bool, out_h: int, out_w: int) -> bool:
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
    values = _path_color_values(len(path), colors if isinstance(colors, tuple) else tuple(colors))
    # `start_color` selects the phase; reuse the helper by swapping order when needed.
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
    Complete partially pre-drawn paths inside a framed rectangle by extending them from external color traces.

    Input structure:
    - Background plus one 1px rectangular frame whose interior is mostly one fill color, but now contains a few path-colored clue pixels.
    - Colored components outside the frame still touch the ring around the frame and encode the target paths.

    Rule:
    - Output is the cropped frame interior.
    - Start from the majority fill color, infer each path from the external component as in the original task, and complete the missing cells.
    - Internal clue pixels are treated as already-correct path fragments and used to choose the consistent completion.

    Key insight: The outer ring still determines the endpoints, while the sparse internal fragments tell you which candidate completion and alternating phase are already present.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
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
            candidates = []
            for corner_a in (True, False):
                path = _build_manhattan(p1, p2, corner_a)
                score = _score_mono_candidate(
                    path,
                    colv,
                    clue_map,
                    prefer_inside=_inside_corner(p1, p2, corner_a, out_h, out_w),
                )
                candidates.append((score, path))
            _score, chosen = max(candidates, key=lambda item: item[0])
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

            candidates = []
            for corner_a in (True, False):
                base_path = _build_manhattan(a, b, corner_a)
                prefer_inside = _inside_corner(a, b, corner_a, out_h, out_w)
                for path_pts, start_hint, end_hint in (
                    (base_path, ha, hb),
                    (list(reversed(base_path)), hb, ha),
                ):
                    for start_color in (ca, cb):
                        vals = _path_color_values(
                            len(path_pts),
                            (ca, cb) if int(start_color) == int(ca) else (cb, ca),
                        )
                        if int(vals[0]) != int(start_hint) or int(vals[-1]) != int(end_hint):
                            continue
                        score = _score_alt_candidate(
                            path_pts,
                            (ca, cb),
                            start_color,
                            clue_map,
                            prefer_inside=prefer_inside,
                        )
                        candidates.append((score, path_pts, start_color))

            assert candidates
            _score, path_pts, start_color = max(candidates, key=lambda item: item[0])
            _paint_path(
                out,
                path_pts,
                (ca, cb) if int(start_color) == int(ca) else (cb, ca),
            )
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        tops = [p for p in pts if p[0] == 0]
        bots = [p for p in pts if p[0] == out_h - 1]
        if tops and bots:
            for cc in sorted({c for _r, c in pts}):
                rs = [r for r, c in pts if c == cc]
                for rr in range(min(rs), max(rs) + 1):
                    out[int(rr), int(cc)] = int(col)
            continue
        lefts = [p for p in pts if p[1] == 0]
        rights = [p for p in pts if p[1] == out_w - 1]
        if lefts and rights:
            for rr in sorted({r for r, _c in pts}):
                cs = [c for r, c in pts if r == rr]
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
        candidates = []
        for corner_a in (True, False):
            path = _build_manhattan(p1, p2, corner_a)
            score = _score_mono_candidate(
                path,
                int(col),
                clue_map,
                prefer_inside=_inside_corner(p1, p2, corner_a, out_h, out_w),
            )
            candidates.append((score, path))
        _score, chosen = max(candidates, key=lambda item: item[0])
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6, D15 variant) ---------------
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

    expected = np.full((H, W), fill, dtype=int)
    clues = np.full((H, W), fill, dtype=int)
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0
    clue_count = 0

    if remaining >= 2 and random.random() < _U_PAIR_PROB:
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
        for line in inner_lines:
            _paint_clue_segment(clues, line, (color,))
            clue_count += 1
        remaining -= 2

    corners = random.sample(["tl", "tr", "bl", "br"], k=remaining)
    for kind in corners:
        if color_i >= len(palette) - 1:
            return None
        inner_path, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        irs, ics = zip(*inner_path)
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
        _paint_path(expected, inner_path, colors)
        used_inner[irs, ics] = True
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
    raise RuntimeError("failed to generate a valid a6f40cea D15 input")
