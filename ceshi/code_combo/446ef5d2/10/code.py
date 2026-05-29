from __future__ import annotations

# Combo variant of 446ef5d2 (index 10)
# combo_spec: D1/4+D12/1
# Name: 颜色互补匹配配合独立分配内部色
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题里，背景上散落若干矩形碎片和一个 3 像素 L-marker；marker 的缺失格给出最终大矩形的一个角。
#   solver 只做平移拼接，不旋转。原题共享边规则是：若两侧同位置都不是 rect_col，则颜色必须相同。
# - D1/4 把“相同颜色”改成“固定互补颜色”。这里固定互补表
#   {0<->9, 1<->2, 3<->4, 5<->6, 7<->8}。例如左块右边 [7,1,7] 对右块左边 [8,2,8]，
#   原题不合法，融合版合法，因为每个非 rect_col 位置都互补。
# - D12/1 把“整块矩形共享全局 interior_cols”改成“每块碎片独立分配内部色组合”。
#   例如四块可分别主要使用 {0,3}、{9,4}、{0,4}、{9,3}，只在真实接缝上暴露互补关系；
#   因而解题者不能再靠跨碎片延续的大色斑猜配对，只能靠边缘互补证据。
# - 融合后的观察链是：先从 train 输出中看到“能拼上的接缝总是固定颜色对，而非同色”，再发现每块内部主色组合
#   彼此不同，所以全局颜色连续性不再可靠，最终必须依赖互补边匹配来完成拼图。
#
# b) 复杂度校准
# - 互补对若过多，人类难以从少量样例归纳；若只有 1 对，又会太弱。这里每个样例固定启用 2 个互补对，共 4 个活动色，
#   并在多条接缝上重复出现，保证规则可见。
# - 若切分方式过于自由，接缝太短、角点冲突太多、回代成功率也差。这里采用 2x2 到 4x3/3x4 的规则网格分块，
#   每块仍是 4-5 像素的矩形，总数 4-12，既满足原题物理约束，又让每条接缝都有 2-3 个可见证据位。
# - 独立配色若完全随机，画面会很乱。这里每块只使用自己的 2 色局部调色板，并通过网格图案让相邻块的调色板不同，
#   使 D12/1 肉眼可见但不至于失控。
#
# c) 隐式假设审查
# - 原 solver 直接把碎片像素众数当 rect_col。D12/1 下各块可以独立高填充，该前提不再稳健；融合版改为把所有碎片颜色
#   都当作候选 rect_col，按“边缘频次优先、总频次次之”尝试装配。
# - 原 solver 的 `compat()` 假设非 rect_col 边像素必须相等。D1/4 使此前提失效，这里改为 `comp(left)==right`。
# - `boundary_ok()` 仍成立：完整矩形四条外边仍必须全是 rect_col。
# - `neighbor_ok()` 的其余结构仍成立，但新增前提是：真实接缝上要有足够多的非 rect_col 证据位，否则错误候选也可能过关。
#   generator 对每条内部接缝都显式写入 2-3 个互补位置，并保留端点为 rect_col 以避免横纵双重约束冲突。
# - anchor 预放逻辑仍成立，因为 marker 语义、碎片形状、无旋转设定都没有变。
#
# d) 放置合法性
# - 生成器先构造目标矩形，再切成规则网格碎片，给每条内部接缝写入互补色，随后在每块内部独立扩展局部色块。
# - marker 先放，碎片用 `random_free_location(..., padding=1)` 散开，保证互不重叠、与 marker/其他碎片至少隔 1 格背景。
# - 外边框始终保持 rect_col；所有接缝端点也保持 rect_col，因此不会出现越界或角点颜色冲突。
#
# e) 视觉可观察性
# - 从 train 样例看，输出中的真实接缝不是“同色延续”，而是稳定互补对反复出现；这能支持人类归纳颜色映射。
# - 同时，相邻碎片经常拥有不同的局部颜色签名，所以“这两块都有一大片绿色”之类的全局线索被抹掉，
#   迫使解题者改看边界证据。
# - 颜色映射不是基于数值大小规律，而是固定任务级配对规则；每个样例只用其中 2 对，并让每对在多个接缝上重复。
#
# f) 可视化整洁度
# - 背景、marker、rect_col 外边框、互补接缝、局部内部色块这几类视觉角色分层明确，没有叠在同一位置。
# - 每块只保留 2 色局部调色板，且接缝颜色会向内轻微扩展，所以既能看出边界证据，也不会把整块染成噪声。
#
# g) 变化的可推断性
# - 为防止融合退化，generator 额外约束两个量化指标：
#   1) `complement_seam_cells`：真实邻接边上满足“双方都非 rect_col 且互补且不相等”的位置数，要求足够多；
#   2) `diff_signature_adjacencies`：相邻碎片的非 rect_col 颜色集合不同的邻接条数，要求覆盖大多数邻接。
# - 这两个指标分别对应 D1/4 与 D12/1 的核心行为。若它们不足，样本直接丢弃重生，避免退化回原题或弱化成普通拼图。
#
# 新增 solver 假设与 generator 保证
# - 假设 1：除 bg/marker 外，所有碎片仍是轴对齐矩形且无需旋转。生成器维持原题物理结构。
# - 假设 2：真实 rect_col 是某个实际出现过的碎片颜色。生成器保证外边框统一为 rect_col，因此该候选一定存在。
# - 假设 3：若按正确 rect_col 和互补规则装配，能得到唯一稳定解。生成器通过密集互补接缝和不同局部签名来强化唯一性。
# - 假设 4：错误的“同色匹配”不应与正确规则等价。生成器要求大量 `a != b 且 comp(a)==b` 的接缝位置来避免退化。

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


COMPLEMENT_MAP = {
    0: 9,
    9: 0,
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    5: 6,
    6: 5,
    7: 8,
    8: 7,
}
PAIR_UNITS = [(0, 9), (1, 2), (3, 4), (5, 6), (7, 8)]
GRID_SHAPES = [(2, 2), (2, 3), (3, 2), (3, 3), (2, 4), (4, 2), (3, 4), (4, 3)]


def _comp(color: int) -> int:
    return int(COMPLEMENT_MAP[int(color)])


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
    cnt = Counter(map(int, g.flatten()))
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c0)) for r, c0 in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c0 for _, c0 in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = miss[0]
        if missing == (r0, c0):
            corner = "BR"
        elif missing == (r0, c1):
            corner = "BL"
        elif missing == (r1, c0):
            corner = "TR"
        else:
            corner = "TL"
        return int(col), (int(missing[0]), int(missing[1])), corner
    raise AssertionError("marker not found")


def _extract_rects(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < GW and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < GH and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))
    return rects


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int, mode: str) -> bool:
    m = (v1 != int(rect_col)) & (v2 != int(rect_col))
    if not bool(m.any()):
        return True
    left = np.array(v1[m], dtype=int)
    right = np.array(v2[m], dtype=int)
    if mode == "equal":
        return bool(np.all(left == right))
    return bool(np.all(np.array([_comp(x) for x in left], dtype=int) == right))


def _solve(grid: np.ndarray, mode: str = "complement") -> np.ndarray:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, corner = _find_marker(g, bg)
    rects = _extract_rects(g, bg, mark)
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    mr, mc = missing

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        dims.append((hh, total_area // hh))
        if hh * hh != total_area:
            dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    total_counts: Counter[int] = Counter()
    edge_counts: Counter[int] = Counter()
    for a in pieces:
        total_counts.update(map(int, a.flatten()))
        perim = list(map(int, a[0, :])) + list(map(int, a[-1, :]))
        if a.shape[0] > 2:
            perim += list(map(int, a[1:-1, 0])) + list(map(int, a[1:-1, -1]))
        edge_counts.update(perim)
    candidate_rect_cols = sorted(
        total_counts.keys(),
        key=lambda c: (-edge_counts[int(c)], -total_counts[int(c)], int(c)),
    )

    def assemble(H: int, W: int, rect_col: int) -> np.ndarray | None:
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)

        informative = []
        for a in pieces:
            perim = list(map(int, a[0, :])) + list(map(int, a[-1, :]))
            if a.shape[0] > 2:
                perim += list(map(int, a[1:-1, 0])) + list(map(int, a[1:-1, -1]))
            informative.append(sum(1 for v in perim if int(v) != int(rect_col)))

        order = sorted(
            range(len(pieces)),
            key=lambda i: (-informative[i], -(ph[i] * pw[i]), -ph[i], -pw[i], i),
        )

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if (r + h) == H and not np.all(a[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if (c + w) == W and not np.all(a[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col, mode)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col, mode)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col, mode)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col, mode)):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > H or (c + w) > W:
                return False
            if (occ2[r : r + h, c : c + w] != -1).any():
                return False
            if not boundary_ok(a, r, c):
                return False
            if not neighbor_ok(a, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = a
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False

        if anchor_idx is not None:
            rr, cc, h0, w0, _ = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + w0 - 1): "TR",
                (rr + h0 - 1, cc): "BL",
                (rr + h0 - 1, cc + w0 - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                ah, aw = ph[anchor_idx], pw[anchor_idx]
                if corner == "TL":
                    ar0, ac0 = 0, 0
                elif corner == "TR":
                    ar0, ac0 = 0, W - aw
                elif corner == "BL":
                    ar0, ac0 = H - ah, 0
                else:
                    ar0, ac0 = H - ah, W - aw
                if can_place(anchor_idx, ar0, ac0):
                    place(anchor_idx, ar0, ac0)

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return int(idx[0][0]), int(idx[0][1])

        solved = False

        def bt() -> None:
            nonlocal solved
            nxt = next_empty()
            if nxt is None:
                solved = True
                return
            r, c = nxt
            for i in order:
                if usedp[i]:
                    continue
                if can_place(i, r, c):
                    place(i, r, c)
                    bt()
                    if solved:
                        return
                    unplace(i, r, c)

        bt()
        return occ2 if solved else None

    ass = None
    used_dim = None
    for rect_col in candidate_rect_cols:
        for H, W in dims:
            ass = assemble(H, W, int(rect_col))
            if ass is not None:
                used_dim = (H, W)
                break
        if ass is not None:
            break

    if ass is None or used_dim is None:
        raise AssertionError("no assembly")

    H, W = used_dim
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="complement")


def _sample_segments(n_parts: int, rng: random.Random) -> list[int]:
    return [rng.randint(4, 5) for _ in range(n_parts)]


def _build_piece_specs(
    row_segs: list[int],
    col_segs: list[int],
) -> tuple[list[dict[str, int]], list[list[int]]]:
    row_starts = np.cumsum([0] + row_segs[:-1]).tolist()
    col_starts = np.cumsum([0] + col_segs[:-1]).tolist()
    nr = len(row_segs)
    nc = len(col_segs)
    specs: list[dict[str, int]] = []
    idx_grid = [[-1 for _ in range(nc)] for _ in range(nr)]
    for i in range(nr):
        for j in range(nc):
            idx = len(specs)
            idx_grid[i][j] = idx
            specs.append(
                {
                    "pr": int(row_starts[i]),
                    "pc": int(col_starts[j]),
                    "ph": int(row_segs[i]),
                    "pw": int(col_segs[j]),
                    "gi": int(i),
                    "gj": int(j),
                }
            )
    return specs, idx_grid


def _build_adjacencies(
    specs: list[dict[str, int]],
    idx_grid: list[list[int]],
    row_segs: list[int],
    col_segs: list[int],
) -> list[dict[str, object]]:
    nr = len(row_segs)
    nc = len(col_segs)
    adj = []
    for i in range(nr):
        for j in range(nc - 1):
            a = idx_grid[i][j]
            b = idx_grid[i][j + 1]
            cells = [(r, specs[a]["pw"] - 1, r, 0) for r in range(row_segs[i])]
            adj.append({"a": a, "b": b, "cells": cells})
    for i in range(nr - 1):
        for j in range(nc):
            a = idx_grid[i][j]
            b = idx_grid[i + 1][j]
            cells = [(specs[a]["ph"] - 1, c, 0, c) for c in range(col_segs[j])]
            adj.append({"a": a, "b": b, "cells": cells})
    return adj


def _grow_piece(
    arr: np.ndarray,
    palette: list[int],
    protected: np.ndarray,
    rect_col: int,
    rng: random.Random,
) -> np.ndarray | None:
    h, w = arr.shape
    free_cells = [
        (r, c)
        for r in range(h)
        for c in range(w)
        if int(arr[r, c]) == int(rect_col) and not bool(protected[r, c])
    ]

    for color in palette:
        if np.any(arr == int(color)):
            continue
        if not free_cells:
            return None
        rr, cc = rng.choice(free_cells)
        free_cells.remove((rr, cc))
        arr[rr, cc] = int(color)

    counts = {int(color): int(np.sum(arr == int(color))) for color in palette}
    piece_area = h * w
    fill_target = rng.uniform(0.18, 0.42)
    target_total = max(sum(counts.values()), int(round(fill_target * piece_area)))

    weights = [rng.randint(1, 3) for _ in palette]
    extra_budget = target_total - sum(counts.values())
    targets = dict(counts)
    if extra_budget > 0:
        wsum = sum(weights)
        added = 0
        for idx, color in enumerate(palette):
            extra = extra_budget * weights[idx] // wsum
            targets[int(color)] += extra
            added += extra
        for _ in range(extra_budget - added):
            color = int(rng.choice(palette))
            targets[color] += 1

    frontiers = {
        int(color): [(int(r), int(c)) for r, c in np.argwhere(arr == int(color))]
        for color in palette
    }
    stalls = 0
    while sum(counts.values()) < target_total and stalls < 120:
        progressed = False
        order = list(palette)
        rng.shuffle(order)
        for color in order:
            color = int(color)
            if counts[color] >= targets[color]:
                continue
            frontier = frontiers[color]
            rng.shuffle(frontier)
            for fr, fc in frontier[:]:
                nbrs = [(fr - 1, fc), (fr + 1, fc), (fr, fc - 1), (fr, fc + 1)]
                rng.shuffle(nbrs)
                for nr, nc in nbrs:
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if bool(protected[nr, nc]) or int(arr[nr, nc]) != int(rect_col):
                        continue
                    arr[nr, nc] = color
                    frontier.append((nr, nc))
                    counts[color] += 1
                    progressed = True
                    break
                if progressed:
                    break
            if progressed:
                break
        if progressed:
            stalls = 0
        else:
            stalls += 1

    fill = float(np.mean(arr != int(rect_col)))
    if fill < 0.12 or fill > 0.48:
        return None
    return arr


def _layout_metrics(
    piece_arrays: list[np.ndarray],
    adj: list[dict[str, object]],
    rect_col: int,
) -> dict[str, int]:
    complement_seam_cells = 0
    diff_signature_adjacencies = 0
    signatures = []
    for arr in piece_arrays:
        sig = tuple(sorted(set(map(int, arr[arr != int(rect_col)].flatten()))))
        signatures.append(sig)
    for item in adj:
        a = int(item["a"])
        b = int(item["b"])
        if signatures[a] != signatures[b]:
            diff_signature_adjacencies += 1
        for ra, ca, rb, cb in item["cells"]:
            va = int(piece_arrays[a][ra, ca])
            vb = int(piece_arrays[b][rb, cb])
            if va != rect_col and vb != rect_col and va != vb and _comp(va) == vb:
                complement_seam_cells += 1
    return {
        "complement_seam_cells": int(complement_seam_cells),
        "diff_signature_adjacencies": int(diff_signature_adjacencies),
        "distinct_signatures": int(len(set(signatures))),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    nr, nc = rng.choice(GRID_SHAPES)
    row_segs = _sample_segments(nr, rng)
    col_segs = _sample_segments(nc, rng)
    RH, RW = sum(row_segs), sum(col_segs)

    if RH > 22 or RW > 22:
        return None

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])

    avail_pairs = [pair for pair in PAIR_UNITS if pair[0] not in {bg, rect_col, marker_col} and pair[1] not in {bg, rect_col, marker_col}]
    if len(avail_pairs) < 2:
        return None
    pair_a, pair_b = rng.sample(avail_pairs, 2)
    active_cols = [int(pair_a[0]), int(pair_a[1]), int(pair_b[0]), int(pair_b[1])]
    templates = [
        [active_cols[0], active_cols[2]],
        [active_cols[1], active_cols[3]],
        [active_cols[0], active_cols[3]],
        [active_cols[1], active_cols[2]],
    ]

    specs, idx_grid = _build_piece_specs(row_segs, col_segs)
    adj = _build_adjacencies(specs, idx_grid, row_segs, col_segs)
    piece_count = len(specs)
    if piece_count < 4 or piece_count > 12:
        return None

    shift = rng.randrange(4)
    piece_palettes: list[list[int]] = []
    for spec in specs:
        template_idx = (int(spec["gi"]) + 2 * int(spec["gj"]) + shift) % 4
        piece_palettes.append(list(templates[template_idx]))

    piece_arrays = [np.full((int(spec["ph"]), int(spec["pw"])), rect_col, dtype=int) for spec in specs]
    touched = [0] * piece_count
    active_counts: Counter[int] = Counter()

    rng.shuffle(adj)
    for item in adj:
        a = int(item["a"])
        b = int(item["b"])
        cells = item["cells"]
        allowed = [int(c) for c in piece_palettes[a] if _comp(int(c)) in set(piece_palettes[b])]
        if not allowed:
            return None
        usable = list(range(1, len(cells) - 1))
        if not usable:
            return None
        offset = rng.randrange(len(allowed))
        for k, pos in enumerate(usable):
            ra, ca, rb, cb = cells[pos]
            left_color = int(allowed[(k + offset) % len(allowed)])
            right_color = _comp(left_color)
            piece_arrays[a][ra, ca] = left_color
            piece_arrays[b][rb, cb] = right_color
            active_counts[left_color] += 1
            active_counts[right_color] += 1
        touched[a] += 1
        touched[b] += 1

    if min(touched) == 0:
        return None

    colored_pieces = []
    for idx, spec in enumerate(specs):
        pr, pc, ph, pw = (int(spec["pr"]), int(spec["pc"]), int(spec["ph"]), int(spec["pw"]))
        arr = piece_arrays[idx].copy()
        protected = np.zeros_like(arr, dtype=bool)
        if pr == 0:
            protected[0, :] = True
        if pr + ph == RH:
            protected[-1, :] = True
        if pc == 0:
            protected[:, 0] = True
        if pc + pw == RW:
            protected[:, -1] = True
        protected[arr != int(rect_col)] = True
        grown = _grow_piece(arr, piece_palettes[idx], protected, rect_col, rng)
        if grown is None:
            return None
        colored_pieces.append(grown)

    rect = np.full((RH, RW), rect_col, dtype=int)
    for spec, arr in zip(specs, colored_pieces):
        pr, pc, ph, pw = (int(spec["pr"]), int(spec["pc"]), int(spec["ph"]), int(spec["pw"]))
        rect[pr : pr + ph, pc : pc + pw] = arr

    if not np.all(rect[0, :] == rect_col):
        return None
    if not np.all(rect[-1, :] == rect_col):
        return None
    if not np.all(rect[:, 0] == rect_col):
        return None
    if not np.all(rect[:, -1] == rect_col):
        return None

    metrics = _layout_metrics(colored_pieces, adj, rect_col)
    if metrics["complement_seam_cells"] < max(8, len(adj) * 2):
        return None
    if metrics["diff_signature_adjacencies"] < len(adj):
        return None
    if metrics["distinct_signatures"] < min(4, piece_count):
        return None
    if any(active_counts[int(c)] < 4 for c in active_cols):
        return None

    total_counts = Counter(map(int, rect.flatten()))
    if total_counts[rect_col] <= max(v for k, v in total_counts.items() if k != rect_col):
        return None

    corner = rng.choice(["TL", "TR", "BL", "BR"])
    GH_lo = max(10, RH + 4)
    GW_lo = max(10, RW + 4)
    GH_hi = min(26, RH + 9)
    GW_hi = min(26, RW + 9)
    if GH_lo > GH_hi or GW_lo > GW_hi:
        return None
    GH = rng.randint(GH_lo, GH_hi)
    GW = rng.randint(GW_lo, GW_hi)
    if GH * GW < RH * RW * 2:
        return None

    if corner == "TL":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = 2, GW - RW
    elif corner == "TR":
        mr_min, mr_max = 2, GH - RH
        mc_min, mc_max = RW - 1, GW - 3
    elif corner == "BL":
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = 2, GW - RW
    else:
        mr_min, mr_max = RH - 1, GH - 3
        mc_min, mc_max = RW - 1, GW - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None
    mr = rng.randint(mr_min, mr_max)
    mc = rng.randint(mc_min, mc_max)

    if corner == "TL":
        marker_cells = [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_ok = False
    for spec in specs:
        pr, pc, ph, pw = (int(spec["pr"]), int(spec["pc"]), int(spec["ph"]), int(spec["pw"]))
        if not (pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw):
            continue
        if corner == "TL" and pr == 0 and pc == 0:
            anchor_ok = True
        elif corner == "TR" and pr == 0 and pc + pw == RW:
            anchor_ok = True
        elif corner == "BL" and pr + ph == RH and pc == 0:
            anchor_ok = True
        elif corner == "BR" and pr + ph == RH and pc + pw == RW:
            anchor_ok = True
        break
    if not anchor_ok:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    order = list(range(piece_count))
    rng.shuffle(order)
    for idx in order:
        piece = colored_pieces[idx]
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
            paste(canvas, piece, r0, c0)
        except ValueError:
            return None

    tr, tc = _top_left_from_corner(mr, mc, RH, RW, corner)
    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
