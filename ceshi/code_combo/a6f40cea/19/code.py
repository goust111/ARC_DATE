# Combo variant of a6f40cea (index 19)
# combo_spec: D11/1+D16/0
# Name: 框边配对镜像投影

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, get_bg, neighbors4


# Design analysis for combo D11/1 + D16/0
# a) Concrete behavior:
#    原题把外部触点直接投影到对应内边，再从同一连通块里选一对端点画一条 L。
#    这里先做 D11：触点按“相邻框边”分组配对；再做 D16：配好的直接投影坐标整体
#    绕框中心做 180 度镜像。例：左上角角块若在上边触到列 1/3、在左边触到行 2/4，
#    原 D11 会在左上角内部画两条嵌套 L；本融合版先得到这些左上角 L，再把整条路径
#    镜像到右下角，于是端点落到底边与右边，且列/行顺序也反转，形成右下角嵌套 L。
# b) Complexity calibration:
#    多触点仍只允许发生在共享一个外角的两条相邻边上，且限制为 2-3 对，避免三边混合。
#    镜像统一采用“中心镜像”这一种模式，不混入多种反射/旋转，保证 2-4 个样例可归纳。
# c) Implicit assumptions audit:
#    原 solver 假设“相邻双边组件只需选最远一对”，本变体改为逐边排序配对。
#    原 solver 还假设“直接投影后的端点就是最终端点”，本变体要求在画路径前再做一次
#    中心镜像。为避免新冲突，生成器强制至少有一个多配对角块，并让其触点落在靠源角
#    的半边候选区内，使配对顺序和镜像后的嵌套结构都稳定可见。
# d) Placement legality:
#    生成器继续 output-first：先在内部构造不重叠的源角 L，再把内部路径整体镜像得到
#    真实输出；外部仍反推为开口框/梳子状连通块，并用 _place_component 校验不越界、
#    不覆盖、不与其他组件 4 邻接。
# e) Visual observability:
#    人类能同时看到两个线索：同色块在相邻两边有多处触碰，且输出路径却出现在对侧角。
#    这种“按边配对 + 对侧镜像”只依赖几何方位，不依赖颜色数值等不可视因素。
# f) Visual cleanliness:
#    触点候选仍局限在各角的半区，镜像只是把整组路径搬到对角，因此不同组件的内部区域
#    仍保持分散；多配对组件固定单色，避免外部树形连通块再叠加难读的双色相位。
# g) Color inferability:
#    双色规则仅保留在单对角块上，端点颜色仍由对应 ring 触点锚定；镜像不改路径顺序，
#    所以解题者能从“哪一侧触点是什么颜色”直接对应到镜像后路径两端的颜色。


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


def _uniq_coords(*parts: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts: set[tuple[int, int]] = set()
    for part in parts:
        for p in part:
            pts.add((int(p[0]), int(p[1])))
    return sorted(pts)


_CORNER_TO_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
_SIDES_TO_CORNER = {frozenset(v): k for k, v in _CORNER_TO_SIDES.items()}


def _ring_side(rr: int, cc: int, r0: int, c0: int, r1: int, c1: int) -> str | None:
    if rr == r0 - 1 and c0 <= cc <= c1:
        return "top"
    if cc == c1 + 1 and r0 <= rr <= r1:
        return "right"
    if rr == r1 + 1 and c0 <= cc <= c1:
        return "bottom"
    if cc == c0 - 1 and r0 <= rr <= r1:
        return "left"
    return None


def _corner_sort_key(kind: str, side: str):
    if kind == "tl":
        return (lambda p: int(p[1])) if side == "top" else (lambda p: int(p[0]))
    if kind == "tr":
        return (lambda p: -int(p[1])) if side == "top" else (lambda p: int(p[0]))
    if kind == "bl":
        return (lambda p: int(p[1])) if side == "bottom" else (lambda p: -int(p[0]))
    return (lambda p: -int(p[1])) if side == "bottom" else (lambda p: -int(p[0]))


def _build_corner_path(
    kind: str,
    p_a: tuple[int, int],
    p_b: tuple[int, int],
    outH: int,
    outW: int,
) -> list[tuple[int, int]]:
    if kind == "tl":
        _rt, ct = p_a
        rl, _cl = p_b
        return _join(_vpath(0, rl, ct), _hpath(rl, ct, 0))
    if kind == "tr":
        _rt, ct = p_a
        rr, _cr = p_b
        return _join(_vpath(0, rr, ct), _hpath(rr, ct, outW - 1))
    if kind == "bl":
        _rb, cb = p_a
        rl, _cl = p_b
        return _join(_vpath(outH - 1, rl, cb), _hpath(rl, cb, 0))
    _rb, cb = p_a
    rr, _cr = p_b
    return _join(_vpath(outH - 1, rr, cb), _hpath(rr, cb, outW - 1))


def _mirror_coord(p: tuple[int, int], outH: int, outW: int) -> tuple[int, int]:
    return int(outH - 1 - p[0]), int(outW - 1 - p[1])


def _mirror_path(path_pts: list[tuple[int, int]], outH: int, outW: int) -> list[tuple[int, int]]:
    return [_mirror_coord(p, outH, outW) for p in path_pts]


def _paint_alternating(
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


def _solve(grid: np.ndarray, *, mirrored: bool) -> np.ndarray:
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

    def map_direct(rr: int, cc: int) -> tuple[int, int]:
        r = max(0, min(outH - 1, int(rr - r_int0)))
        c = max(0, min(outW - 1, int(cc - c_int0)))
        return int(r), int(c)

    def maybe_mirror_coord(p: tuple[int, int]) -> tuple[int, int]:
        if not mirrored:
            return int(p[0]), int(p[1])
        return _mirror_coord(p, outH, outW)

    def maybe_mirror_path(path_pts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        if not mirrored:
            return path_pts
        return _mirror_path(path_pts, outH, outW)

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

    comp_pts: dict[int, list[tuple[str, tuple[int, int], int]]] = defaultdict(list)
    for rr, cc in ring:
        cid = int(clab[rr, cc])
        if cid == 0:
            continue
        side = _ring_side(int(rr), int(cc), int(r0), int(c0), int(r1), int(c1))
        if side is None:
            continue
        comp_pts[cid].append((side, map_direct(int(rr), int(cc)), int(g[rr, cc])))

    comp_cols: dict[int, list[int]] = {}
    for cid in range(1, int(cn) + 1):
        comp_cols[cid] = sorted(set(map(int, g[clab == int(cid)].flatten())))

    deferred: dict[int, set[tuple[int, int]]] = defaultdict(set)

    for cid, pts in comp_pts.items():
        cols = comp_cols.get(cid, [])
        if not cols:
            continue

        side_hist: dict[str, dict[tuple[int, int], Counter]] = defaultdict(lambda: defaultdict(Counter))
        for side, coord, col in pts:
            side_hist[side][(int(coord[0]), int(coord[1]))][int(col)] += 1

        active_sides = [side for side, hist in side_hist.items() if hist]
        if not active_sides:
            continue

        if len(active_sides) == 1:
            if len(cols) != 1:
                continue
            colv = int(cols[0])
            for coord in side_hist[active_sides[0]].keys():
                deferred[colv].add(maybe_mirror_coord(coord))
            continue

        if len(active_sides) != 2:
            continue

        kind = _SIDES_TO_CORNER.get(frozenset(active_sides))
        if kind is None:
            continue
        side_a, side_b = _CORNER_TO_SIDES[kind]
        if side_a not in side_hist or side_b not in side_hist:
            continue

        pts_a = sorted(side_hist[side_a].keys(), key=_corner_sort_key(kind, side_a))
        pts_b = sorted(side_hist[side_b].keys(), key=_corner_sort_key(kind, side_b))
        pair_count = min(len(pts_a), len(pts_b))
        if pair_count == 0:
            continue

        if len(cols) == 1:
            colv = int(cols[0])
            for i in range(pair_count):
                path = maybe_mirror_path(_build_corner_path(kind, pts_a[i], pts_b[i], outH, outW))
                for rr, cc in path:
                    out[int(rr), int(cc)] = int(colv)
            continue

        if len(cols) == 2:
            palette = (int(cols[0]), int(cols[1]))
            for i in range(pair_count):
                a = pts_a[i]
                b = pts_b[i]
                ha = int(side_hist[side_a][a].most_common(1)[0][0])
                hb = int(side_hist[side_b][b].most_common(1)[0][0])
                path = maybe_mirror_path(_build_corner_path(kind, a, b, outH, outW))
                chosen_start = None
                for start_color in palette:
                    end_color = int(start_color) if ((len(path) - 1) % 2 == 0) else (
                        palette[1] if int(start_color) == palette[0] else palette[0]
                    )
                    if int(start_color) == int(ha) and int(end_color) == int(hb):
                        chosen_start = int(start_color)
                        break
                assert chosen_start is not None
                _paint_alternating(out, path, palette, chosen_start)
            continue

    for col, pts in deferred.items():
        pts = list(pts)
        if not pts:
            continue
        if all(p[0] in (0, outH - 1) for p in pts):
            for cc in sorted({c for _r, c in pts}):
                out[:, int(cc)] = int(col)
            continue
        if all(p[1] in (0, outW - 1) for p in pts):
            for rr in sorted({r for r, _c in pts}):
                out[int(rr), :] = int(col)
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

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover interior paths from exterior frame contacts, then mirror the projection across the frame center.

    Input structure:
    - A uniform background contains one closed 1px rectangular frame with one solid fill color inside.
    - Colored components lie outside the frame and touch the ring just beyond one side or two adjacent sides.

    Rule:
    - Output only the frame interior, starting from the fill color.
    - Same-side monochrome contacts still combine into full stripes.
    - For components touching two adjacent sides, first group contacts by side, infer their shared corner,
      sort each side by distance from that corner, and pair the i-th contact on one side with the i-th on the other.
    - Build the corresponding Manhattan L path in direct-projection space, then center-mirror the whole path
      into its final interior location. Monochrome paths stay solid; two-color paths still alternate with
      endpoint colors anchored by the ring-touch hints.

    Key insight:
    The important geometry is two-stage: pairing is decided by the source frame sides and their shared corner,
    but the painted endpoints are the center-mirrored images of those direct projections, so each exterior corner
    controls a path family that appears in the opposite interior corner.
    """
    return _solve(grid, mirrored=True)


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
_U_PAIR_PROB = 1.0
_DUAL_COLOR_PROB = 0.45
_MULTI_PAIR_RANGE = (2, 3)
_DISTRACTOR_COUNT_RANGE = (0, 2)
_DISTRACTOR_SIZE_RANGE = (1, 4)
_DISTRACTOR_TRIES = 40
_DISTRACTOR_CLEARANCE = 2
_GEN_RETRIES = 450


def _paint_path(grid: np.ndarray, path: list[tuple[int, int]], colors: tuple[int, ...]) -> None:
    if not path:
        return
    rs, cs = zip(*path)
    if len(colors) == 1:
        grid[rs, cs] = int(colors[0])
    else:
        a, b = map(int, colors)
        vals = [a if i % 2 == 0 else b for i in range(len(path))]
        grid[rs, cs] = vals


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
    coords: list[tuple[int, int]],
    colors: tuple[int, ...],
    bg: int,
) -> bool:
    if len(set(coords)) != len(coords):
        return False
    if not _can_place(grid, coords, bg, comp_mask):
        return False
    _paint_path(grid, coords, colors)
    rs, cs = zip(*coords)
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


def _corner_candidates(
    kind: str,
    H: int,
    W: int,
    *,
    min_col: int | None = None,
    max_col: int | None = None,
    min_row: int | None = None,
    max_row: int | None = None,
) -> tuple[list[int], list[int]]:
    left_cols = list(range(1, max(2, W // 2 + 1)))
    right_cols = list(range(max(1, W // 2), W - 1))
    top_rows = list(range(1, max(2, H // 2 + 1)))
    bottom_rows = list(range(max(1, H // 2), H - 1))
    if kind == "tl":
        cols, rows = left_cols, top_rows
    elif kind == "tr":
        cols, rows = right_cols, top_rows
    elif kind == "bl":
        cols, rows = left_cols, bottom_rows
    else:
        cols, rows = right_cols, bottom_rows

    if min_col is not None:
        cols = [c for c in cols if c >= int(min_col)]
    if max_col is not None:
        cols = [c for c in cols if c <= int(max_col)]
    if min_row is not None:
        rows = [r for r in rows if r >= int(min_row)]
    if max_row is not None:
        rows = [r for r in rows if r <= int(max_row)]
    return cols, rows


def _corner_max_pairs(kind: str, H: int, W: int, **bounds: int | None) -> int:
    cols, rows = _corner_candidates(kind, H, W, **bounds)
    return min(len(cols), len(rows))


def _corner_multi_specs(
    kind: str,
    n_pairs: int,
    H: int,
    W: int,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    **bounds: int | None,
):
    cols_pool, rows_pool = _corner_candidates(kind, H, W, **bounds)
    if min(len(cols_pool), len(rows_pool)) < n_pairs:
        return None

    cols = random.sample(cols_pool, n_pairs)
    rows = random.sample(rows_pool, n_pairs)
    if kind == "tl":
        cols.sort()
        rows.sort()
        inner_paths = [_join(_vpath(0, r, c), _hpath(r, c, 0)) for c, r in zip(cols, rows)]
    elif kind == "tr":
        cols.sort(reverse=True)
        rows.sort()
        inner_paths = [_join(_vpath(0, r, c), _hpath(r, c, W - 1)) for c, r in zip(cols, rows)]
    elif kind == "bl":
        cols.sort()
        rows.sort(reverse=True)
        inner_paths = [_join(_vpath(H - 1, r, c), _hpath(r, c, 0)) for c, r in zip(cols, rows)]
    else:
        cols.sort(reverse=True)
        rows.sort(reverse=True)
        inner_paths = [_join(_vpath(H - 1, r, c), _hpath(r, c, W - 1)) for c, r in zip(cols, rows)]

    all_inner = [p for path in inner_paths for p in path]
    if len(set(all_inner)) != len(all_inner):
        return None

    lo, hi = _OUTER_OFFSET_RANGE
    is_top = kind[0] == "t"
    is_left = kind[1] == "l"
    R = (r0 - random.randint(lo, hi)) if is_top else (r1 + random.randint(lo, hi))
    C = (c0 - random.randint(lo, hi)) if is_left else (c1 + random.randint(lo, hi))

    if kind == "tl":
        top_pts = [(r0 - 1, c0 + 1 + c) for c in cols]
        left_pts = [(r0 + 1 + r, c0 - 1) for r in rows]
        parts = [_hpath(R, C, max(c for _r, c in top_pts)), _vpath(R, max(r for r, _c in left_pts), C)]
        parts.extend(_vpath(rr, R, cc) for rr, cc in top_pts)
        parts.extend(_hpath(rr, cc, C) for rr, cc in left_pts)
    elif kind == "tr":
        top_pts = [(r0 - 1, c0 + 1 + c) for c in cols]
        right_pts = [(r0 + 1 + r, c1 + 1) for r in rows]
        parts = [_hpath(R, min(c for _r, c in top_pts), C), _vpath(R, max(r for r, _c in right_pts), C)]
        parts.extend(_vpath(rr, R, cc) for rr, cc in top_pts)
        parts.extend(_hpath(rr, cc, C) for rr, cc in right_pts)
    elif kind == "bl":
        bottom_pts = [(r1 + 1, c0 + 1 + c) for c in cols]
        left_pts = [(r0 + 1 + r, c0 - 1) for r in rows]
        parts = [_hpath(R, C, max(c for _r, c in bottom_pts)), _vpath(min(r for r, _c in left_pts), R, C)]
        parts.extend(_vpath(rr, R, cc) for rr, cc in bottom_pts)
        parts.extend(_hpath(rr, cc, C) for rr, cc in left_pts)
    else:
        bottom_pts = [(r1 + 1, c0 + 1 + c) for c in cols]
        right_pts = [(r0 + 1 + r, c1 + 1) for r in rows]
        parts = [_hpath(R, min(c for _r, c in bottom_pts), C), _vpath(min(r for r, _c in right_pts), R, C)]
        parts.extend(_vpath(rr, R, cc) for rr, cc in bottom_pts)
        parts.extend(_hpath(rr, cc, C) for rr, cc in right_pts)

    outer = _uniq_coords(*parts)
    return inner_paths, outer


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
    palette = [x for x in range(10) if x not in (bg, bc, fill)]
    random.shuffle(palette)
    used_inner = np.zeros((H, W), dtype=bool)

    n_components = random.randint(*_N_COMPONENTS_RANGE)
    remaining = n_components
    color_i = 0

    use_u_pair = remaining >= 3 and random.random() < _U_PAIR_PROB
    corner_bounds: dict[str, dict[str, int | None]] = {kind: {} for kind in ["tl", "tr", "bl", "br"]}
    if use_u_pair:
        orientation = random.choice(["vertical", "horizontal"])
        if orientation == "horizontal" and H < 5:
            orientation = "vertical"
        color = int(palette[color_i])
        color_i += 1
        inner_lines, outer_a, outer_b = _u_pair_specs(orientation, H, W, r0, c0, r1, c1)
        mirrored_lines = [_mirror_path(line, H, W) for line in inner_lines]
        all_inner = [p for line in mirrored_lines for p in line]
        irs, ics = zip(*all_inner)
        if used_inner[irs, ics].any():
            return None
        if not _place_component(grid, comp_mask, outer_a, (color,), bg):
            return None
        if not _place_component(grid, comp_mask, outer_b, (color,), bg):
            return None
        expected[irs, ics] = int(color)
        used_inner[irs, ics] = True
        remaining -= 2

        if orientation == "vertical":
            stripe_cols = sorted({c for line in inner_lines for _r, c in line})
            if len(stripe_cols) != 2:
                return None
            left_cap = int(stripe_cols[0] - 1)
            right_floor = int(stripe_cols[-1] + 1)
            for kind in ("tl", "bl"):
                corner_bounds[kind]["max_col"] = left_cap
            for kind in ("tr", "br"):
                corner_bounds[kind]["min_col"] = right_floor
        else:
            stripe_rows = sorted({r for line in inner_lines for r, _c in line})
            if len(stripe_rows) != 2:
                return None
            top_cap = int(stripe_rows[0] - 1)
            bottom_floor = int(stripe_rows[-1] + 1)
            for kind in ("tl", "tr"):
                corner_bounds[kind]["max_row"] = top_cap
            for kind in ("bl", "br"):
                corner_bounds[kind]["min_row"] = bottom_floor

    all_corners = ["tl", "tr", "bl", "br"]
    eligible_multi = [
        kind
        for kind in all_corners
        if _corner_max_pairs(kind, H, W, **corner_bounds[kind]) >= _MULTI_PAIR_RANGE[0]
    ]
    if not eligible_multi or remaining <= 0:
        return None
    multi_corner = random.choice(eligible_multi)
    other_pool = [kind for kind in all_corners if kind != multi_corner]
    if remaining - 1 > len(other_pool):
        return None
    corners = [multi_corner] + random.sample(other_pool, k=remaining - 1)
    random.shuffle(corners)

    multi_done = False
    for kind in corners:
        if kind == multi_corner and not multi_done:
            if color_i >= len(palette):
                return None
            max_pairs = min(_MULTI_PAIR_RANGE[1], _corner_max_pairs(kind, H, W, **corner_bounds[kind]))
            if max_pairs < _MULTI_PAIR_RANGE[0]:
                return None
            n_pairs = random.randint(_MULTI_PAIR_RANGE[0], max_pairs)
            spec = _corner_multi_specs(kind, n_pairs, H, W, r0, c0, r1, c1, **corner_bounds[kind])
            if spec is None:
                return None
            inner_paths, outer = spec
            mirrored_paths = [_mirror_path(path, H, W) for path in inner_paths]
            all_inner = [p for path in mirrored_paths for p in path]
            irs, ics = zip(*all_inner)
            if used_inner[irs, ics].any():
                return None
            color = int(palette[color_i])
            color_i += 1
            if not _place_component(grid, comp_mask, outer, (color,), bg):
                return None
            for path in mirrored_paths:
                _paint_path(expected, path, (color,))
            used_inner[irs, ics] = True
            multi_done = True
            continue

        if color_i >= len(palette) - 1:
            return None
        inner, outer = _l_specs(kind, H, W, r0, c0, r1, c1)
        inner = _mirror_path(inner, H, W)
        irs, ics = zip(*inner)
        if used_inner[irs, ics].any():
            return None
        if random.random() < _DUAL_COLOR_PROB:
            colors = (int(palette[color_i]), int(palette[color_i + 1]))
            color_i += 2
        else:
            colors = (int(palette[color_i]),)
            color_i += 1
        if not _place_component(grid, comp_mask, outer, colors, bg):
            return None
        _paint_path(expected, inner, colors)
        used_inner[irs, ics] = True

    if not multi_done:
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
