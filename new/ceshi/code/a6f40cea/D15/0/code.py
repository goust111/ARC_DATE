# Variant of a6f40cea
# Dimension: D15 — 内部填充色基底 (meta)
# Change: 输出基底为封闭框架的内部填充色（均匀单色），可能等于背景色... → 填充色为棋盘格图案

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4

# Design analysis
# spec_delta:
#   preserved_core:
#     - 仍然只有一个 1px 封闭矩形框，框外仍是 2-3 个触碰 frame ring 的开放矩形组件。
#     - 这些外部组件仍然编码要在框内恢复的曼哈顿 L 路径 / 直条纹。
#     - 单色组件恢复为纯色路径，双色组件恢复为双色交替路径，双色相位仍由 ring 端点颜色锁定。
#   D15_change:
#     - 原题里输出底板是单一 fill 色；这里改为 2 色 1x1 棋盘格。
#     - 输入中的封闭框内部同样直接展示该棋盘格，输出先复制该棋盘格，再把路径覆盖到其上。
#     - 两个棋盘格颜色中允许有一个等于全局背景色，保留“底板可能含背景色”的原题气质，但不再是整块纯色。
# behavior_trace:
#   original:
#     - 框检测依赖 `inner == fill` 全成立；输出先 `np.full(..., fill)`，之后再覆盖路径。
#   variant:
#     - 例如 5x8 内部若是 `a,b` 交替，则检测应验证 `(r+c)%2` 奇偶上的两色一致性。
#     - 输出初始化不再是 `full(fill)`，而是复原同相位棋盘格；随后路径覆盖掉落在路径上的格子。
#   difference:
#     - 旧逻辑会把另一半棋盘格颜色误判为“非填充噪声”，甚至无法找到 frame。
# complexity_calibration:
#   - 只引入一种新底板结构：严格 1x1 两色棋盘格，不引入多种纹理类别，避免学习目标发散。
#   - 路径颜色始终避开两种底板色与边框色，保证路径仍可见，但视觉上会被棋盘格干扰。
# implicit_assumption_audit:
#   - `_find_framed_rect_uniform_fill` 的“内部全同色”假设失效，必须换成“内部恰为两色棋盘格”。
#   - `comp_mask = (g != bg) & (g != fill) & (g != bc)` 依赖单个 fill；现在必须排除两种底板色。
#   - `out = np.full(..., fill)` 会直接破坏 D15；现在必须先精确复原输入中的棋盘格相位。
# placement_legality:
#   - generator 仍然 output-first：先构造棋盘格 expected，再在 expected 上画路径，输入只放 frame+外部组件。
#   - 外部组件与原题同样要求不重叠、不接触；路径颜色不允许等于 bg / bc / 任一棋盘格色。
# visual_observability:
#   - 所有样例都强制内部非单色，并且严格呈现整块棋盘格；这是肉眼直接可见的全局规律。
#   - 通过随机化“哪一个棋盘格色等于 bg”与 top-left 奇偶色，避免退化成固定配色模板。
# cleanliness:
#   - 输入内部只有棋盘格，没有再叠加别的标记；复杂性来自底板纹理本身，而非额外遮挡层。
# color_inference:
#   - 不依赖颜色 id 做语义排序；棋盘格相位直接读取 interior 左上角与其相邻格的可见关系。


def _checkerboard_colors(inner: np.ndarray) -> tuple[int, int] | None:
    """Return the two checkerboard colors as (top_left_parity, odd_parity), else None."""
    h, w = map(int, inner.shape)
    if h == 0 or w == 0:
        return None
    if h == 1 and w == 1:
        return None

    even = int(inner[0, 0])
    if w >= 2:
        odd = int(inner[0, 1])
    else:
        odd = int(inner[1, 0])
    if even == odd:
        return None

    if len(set(map(int, inner.flatten()))) != 2:
        return None

    for r in range(h):
        for c in range(w):
            expected = even if (r + c) % 2 == 0 else odd
            if int(inner[r, c]) != expected:
                return None
    return int(even), int(odd)


def _make_checkerboard(h: int, w: int, even: int, odd: int) -> np.ndarray:
    """Construct a checkerboard whose (0,0) cell is `even`."""
    rr, cc = np.indices((int(h), int(w)))
    return np.where((rr + cc) % 2 == 0, int(even), int(odd)).astype(int)


def _find_framed_rect_checkerboard_fill(
    g: np.ndarray,
    *,
    bg: int | None = None,
    structure=STRUCT4,
) -> tuple[int, int, int, int, int, int, int] | None:
    """
    Detect the main 1px-thick rectangular frame with a 2-color checkerboard interior.

    Returns (r0, c0, r1, c1, border_color, even_color, odd_color) for the largest such rectangle.
    """
    if bg is None:
        bg = int(get_bg(g))
    else:
        bg = int(bg)

    best = None  # (area, r0,c0,r1,c1, bc, even, odd)
    for bc in dict.fromkeys(map(int, g.flatten())):
        if int(bc) == int(bg):
            continue
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
            fills = _checkerboard_colors(inner)
            if fills is None:
                continue
            even, odd = fills
            if int(even) == int(bc) or int(odd) == int(bc):
                continue

            area = int(h * w)
            cand = (area, r0, c0, r1, c1, int(bc), int(even), int(odd))
            if best is None or cand[0] > best[0]:
                best = cand

    if best is None:
        return None
    _area, r0, c0, r1, c1, bc, even, odd = best
    return int(r0), int(c0), int(r1), int(c1), int(bc), int(even), int(odd)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden interior paths of a framed rectangle on top of a checkerboard floor.

    Input structure:
    - One 1px rectangular frame encloses a two-color checkerboard interior; one checker color may match the global background.
    - Colored 4-connected traces sit outside the frame and touch the ring immediately around its border.

    Rule:
    - Output is the cropped frame interior, keeping the same checkerboard colors and parity as the input.
    - Each outside component contributes projected touch points inside the frame and becomes a straight stripe or an L-shaped Manhattan path.
    - Monochrome components paint solid paths; two-color components paint alternating paths whose endpoint colors determine the phase.

    Key insight:
    - The frame is no longer identified by a uniform fill; it is the largest complete rectangle whose interior is a strict checkerboard, and that visible parity pattern defines the output base before paths are overlaid.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    found = _find_framed_rect_checkerboard_fill(g, bg=int(bg))
    assert not (found is None)
    r0, c0, r1, c1, bc, base_even, base_odd = found
    outH = int(r1 - r0 - 1)
    outW = int(c1 - c0 - 1)
    out = _make_checkerboard(outH, outW, int(base_even), int(base_odd))
    r_int0 = int(r0 + 1)
    c_int0 = int(c0 + 1)

    def map_to_out(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def build_manhattan(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> list[tuple[int, int]]:
        (ra, ca), (rb, cb2) = p1, p2
        if ra == rb:
            step = 1 if cb2 >= ca else -1
            return [(int(ra), int(c)) for c in range(int(ca), int(cb2) + step, step)]
        if ca == cb2:
            step = 1 if rb >= ra else -1
            return [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
        if cornerA:
            step = 1 if rb >= ra else -1
            path = [(int(r), int(ca)) for r in range(int(ra), int(rb) + step, step)]
            step = 1 if cb2 >= ca else -1
            path += [(int(rb), int(c)) for c in range(int(ca) + step, int(cb2) + step, step)]
            return path
        step = 1 if cb2 >= ca else -1
        path = [(int(ra), int(c)) for c in range(int(ca), int(cb2) + step, step)]
        step = 1 if rb >= ra else -1
        path += [(int(r), int(cb2)) for r in range(int(ra) + step, int(rb) + step, step)]
        return path

    def inside_corner(p1: tuple[int, int], p2: tuple[int, int], cornerA: bool) -> bool:
        (ra, ca), (rb, cb2) = p1, p2
        if ra == rb or ca == cb2:
            return True
        corner = (rb, ca) if cornerA else (ra, cb2)
        rr, cc = corner
        return (0 < rr < outH - 1) and (0 < cc < outW - 1)

    base_colors = {int(base_even), int(base_odd)}
    excluded = {int(bg), int(bc), *base_colors}
    comp_mask = np.ones_like(g, dtype=bool)
    for col in excluded:
        comp_mask &= g != int(col)
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
        comp_cols[cid] = list(dict.fromkeys(map(int, g[clab == int(cid)].flatten())))

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
            col_a, col_b = map(int, cols)
            ha = int(hist[a].most_common(1)[0][0])
            hb = int(hist[b].most_common(1)[0][0])

            pathA = build_manhattan(a, b, True)
            pathB = build_manhattan(a, b, False)
            path = pathB if (inside_corner(a, b, False) and not inside_corner(a, b, True)) else pathA

            def paint_alt(path_pts: list[tuple[int, int]], start_color: int) -> None:
                cur = int(start_color)
                for rr, cc in path_pts:
                    out[int(rr), int(cc)] = int(cur)
                    cur = int(col_b) if cur == int(col_a) else int(col_a)

            chosen = None
            for path_pts in (path, list(reversed(path))):
                for start_color in (col_a, col_b):
                    end_color = start_color if ((len(path_pts) - 1) % 2 == 0) else (col_b if start_color == col_a else col_a)
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
        pathA = build_manhattan(p1, p2, True)
        pathB = build_manhattan(p1, p2, False)
        chosen = pathB if (inside_corner(p1, p2, False) and not inside_corner(p1, p2, True)) else pathA
        for rr, cc in chosen:
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (a6 / D15) ---------------

# --- CFG: tunable generation parameters ---
_BORDER_COLOR = 3
_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
_CHECKER_EQ_BG_PROB = 0.6
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
    """Inclusive range from a to b."""
    step = 1 if b >= a else -1
    return range(a, b + step, step)


def _vpath(r0: int, r1: int, c: int) -> list[tuple[int, int]]:
    """Vertical segment from (r0, c) to (r1, c) inclusive."""
    return [(r, c) for r in _line(r0, r1)]


def _hpath(r: int, c0: int, c1: int) -> list[tuple[int, int]]:
    """Horizontal segment from (r, c0) to (r, c1) inclusive."""
    return [(r, c) for c in _line(c0, c1)]


def _join(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Concatenate path segments, deduplicating shared endpoints."""
    out: list[tuple[int, int]] = []
    for part in parts:
        for p in part:
            if not out or out[-1] != p:
                out.append(p)
    return out


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    """Paint a path with solid or alternating colors."""
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = colors[0]
    else:
        a, b = colors
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


def _can_place(grid: np.ndarray, coords: list[tuple[int, int]], bg: int, comp_mask: np.ndarray) -> bool:
    """Check that all coords are in-bounds, on bg, and not 4-adjacent to existing components."""
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
    """Validate, paint, and register a component. Returns True on success."""
    if len(set(path)) != len(path):
        return False
    if not _can_place(grid, path, bg, comp_mask):
        return False
    _paint_path(grid, path, colors)
    rs, cs = zip(*path)
    comp_mask[rs, cs] = True
    return True


def _pick_checker_colors(bg: int, bc: int) -> tuple[int, int]:
    """Choose the two checkerboard colors and keep the top-left phase visible."""
    usable = [x for x in range(10) if x not in (bc,)]
    non_bg_non_bc = [x for x in usable if x != bg]
    if random.random() < _CHECKER_EQ_BG_PROB:
        other = random.choice(non_bg_non_bc)
        if random.random() < 0.5:
            return int(bg), int(other)
        return int(other), int(bg)
    a, b = random.sample(non_bg_non_bc, 2)
    return int(a), int(b)


def _l_specs(kind: str, H: int, W: int, r0: int, c0: int, r1: int, c1: int):
    """Build inner (interior L-path) and outer (open-rect) coords for one L-component."""
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
    """Build inner (two full-span stripes) and two outer U-shaped open-rects."""
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
    """Try to place a small random blob away from the frame."""
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
    base_even, base_odd = _pick_checker_colors(int(bg), int(bc))
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
    grid[r0 + 1 : r1, c0 + 1 : c1] = _make_checkerboard(H, W, int(base_even), int(base_odd))

    expected = _make_checkerboard(H, W, int(base_even), int(base_odd))
    comp_mask = np.zeros_like(grid, dtype=bool)
    palette = [x for x in range(10) if x not in {int(bg), int(bc), int(base_even), int(base_odd)}]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

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
    raise RuntimeError("failed to generate a valid a6f40cea D15 input")
