# Variant of 446ef5d2
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻碎片的接触边上，非 rect_col 像素位置和颜色必须一致；rect_col 充当通配符可匹配任何颜色
#   -> 严格全像素匹配（取消通配符）
from __future__ import annotations
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    原题里，两块碎片接触时，若一侧是 rect_col，另一侧是什么都不重要；因此
#    `[rect_col, 2, rect_col, 3]` 和 `[4, 2, rect_col, 3]` 会被旧逻辑视为兼容。
#    本变体要求整条边逐像素完全一致，上面这对边现在不兼容。输入/输出外观仍是：
#    背景上散落若干矩形碎片，旁边有一个 3 像素 L 标记，输出是在标记缺口处拼回大矩形。
#    差异体现在“缝”的颜色条纹：每条真实拼接缝的两侧必须复制成完全相同的像素序列。
#    小例子：若真实缝是 `[1, 2, 1, 3]` 对 `[1, 2, 1, 3]`，严格规则接受；
#    旧规则还会额外接受 `[4, 2, 1, 3]` 之类的假边，因为第 1 位被 rect_col 通配掉。
#
# b) 复杂度校准
#    仅把 solver 的 compat 改成相等判断还不够：若 generator 继续产生“随便切开”的矩形，
#    真实相邻边往往根本不相等，样例会失效。为使新规则既可见又可行，生成器先做矩形分块，
#    再在所有潜在切缝上强制“成对复制”的行/列。这样每条真实缝都满足严格匹配，同时仍保留
#    多色、不规则内部图案。为了让新规则有辨识度，生成器还要求存在一些“旧规则会误判兼容、
#    但严格规则能区分”的候选边对。
#
# c) 隐式假设审查
#    - `rect_col = pieces` 中最常见颜色：仍成立，因为外边框和大量空隙继续由 rect_col 构成。
#    - `boundary_ok()` 依赖“外矩形四边全是 rect_col”：仍保持不变。
#    - 原 `compat()` 依赖“只比较双方都非 rect_col 的位置”；本变体不成立，必须改为整边相等。
#    - `neighbor_ok()` 用已放好的相邻像素直接做边比较。旧题中这一步允许部分错位被通配掩盖；
#      现在这一步必须精确比较整条重叠边。
#    - 生成器原先先画任意纹理再切块，这隐含依赖旧通配能容忍切缝两侧不同；新变体下不成立，
#      所以改为在切缝对应的行/列上构造共享像素来源。
#
# d) 放置合法性
#    仍沿用 output-first：先构造目标矩形，再切成矩形碎片，再随机散落到背景上。
#    `random_free_location(..., padding=1)` 保证碎片和标记互不重叠且至少隔 1 格背景。
#    角标记的位置仍先于碎片放置，并且目标矩形必定能以其缺口为角落落在网格内。
#
# e) 视觉可观察性
#    解题者能直接看到：碎片边缘上有成段的 rect_col / 非 rect_col 条纹，真正能拼接的缝需要
#    整条条纹一模一样，而不是“只看彩色像素”。样例里会保留多条混合边（同一条边上同时含
#    rect_col 与内部色），这样严格匹配的变化是肉眼可见的。
#
# f) 可视化整洁度
#    元素类型仍然只有背景、L 标记和矩形碎片三类；碎片保持轴对齐矩形且彼此留空隙，避免重叠。
#    内部图案继续使用少量颜色的连通块，不引入额外符号层。
#
# g) 颜色变化可推断性
#    本变体不引入颜色映射，只改变“边如何匹配”。颜色数量仍控制在 4-6 种，且内部颜色以连通
#    块形式出现，解题者无需推断隐藏的颜色数值语义。
#
# h) 量化指标
#    生成器会检查两项指标：
#    - mixed_internal_edges: 内部边中同时含 rect_col 与非 rect_col 的边数量，确保严格匹配可见。
#    - strict_only_pairs: 候选边对里，旧规则判兼容但严格规则判不兼容的数量，确保样例确实利用了
#      “取消通配”这个变化。
#    另外在调试测试中会统计旧 solver 与新 solver 在同一输入上的输出是否分歧。
#
# i) main() 新假设审查
#    新 solver 假设：真实相邻边必须逐像素完全相同；因此 generator 必须保证每条切缝两侧来自同
#    一组复制后的行/列。代码中的每个相邻检查都只在长度相同、已占据的位置上比较整条边；
#    generator 通过全局切缝复制和后验校验来保证这一点。

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


def _edge_compat(v1: np.ndarray, v2: np.ndarray, rect_col: int, *, strict: bool) -> bool:
    v1 = np.array(v1, dtype=int)
    v2 = np.array(v2, dtype=int)
    if strict:
        return bool(np.array_equal(v1, v2))
    mask = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[mask] == v2[mask]).all())


def _solve(grid: np.ndarray, *, strict: bool, return_meta: bool = False):
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
    mb = None
    for col in sorted([c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3]):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) == 1 and (c1 - c0) == 1:
            all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
            miss = list(all4 - set(pts))
            if len(miss) == 1:
                mark = int(col)
                missing = miss[0]
                mb = (r0, c0, r1, c1)
                break

    assert not (mark is None or missing is None or mb is None)

    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        corner = "BR"
    elif missing == (r0, c1):
        corner = "BL"
    elif missing == (r1, c0):
        corner = "TR"
    else:
        corner = "TL"

    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
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

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    mr, mc = map(int, missing)

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - W + 1
        elif corner == "BL":
            tr, tc = mr - H + 1, mc
        else:
            tr, tc = mr - H + 1, mc - W + 1
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def assemble(H: int, W: int):
        occ2 = np.full((H, W), -1, dtype=int)
        usedp = [False] * len(pieces)
        placements: list[tuple[int, int] | None] = [None] * len(pieces)

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
                if m.any() and (not _edge_compat(left[m], a[:, 0][m], rect_col, strict=strict)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _edge_compat(right[m], a[:, -1][m], rect_col, strict=strict)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _edge_compat(up[m], a[0, :][m], rect_col, strict=strict)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _edge_compat(down[m], a[-1, :][m], rect_col, strict=strict)):
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
            placements[i] = (r, c)

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            h, w = a.shape
            occ2[r : r + h, c : c + w] = -1
            usedp[i] = False
            placements[i] = None

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

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ2 == -1)
            if idx.size == 0:
                return None
            return tuple(map(int, idx[0]))

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
        return (occ2.copy(), placements.copy()) if solved else None

    ass = None
    solved_placements = None
    used_dim = None
    for H, W in dims:
        assembled = assemble(H, W)
        if assembled is not None:
            ass, solved_placements = assembled
            used_dim = (H, W)
            break

    assert not (ass is None or used_dim is None or solved_placements is None)

    H, W = used_dim
    if corner == "TL":
        top_left = (mr, mc)
    elif corner == "TR":
        top_left = (mr, mc - W + 1)
    elif corner == "BL":
        top_left = (mr - H + 1, mc)
    else:
        top_left = (mr - H + 1, mc - W + 1)

    out = np.full((GH, GW), int(bg), dtype=int)
    tr, tc = map(int, top_left)
    out[tr : tr + H, tc : tc + W] = ass
    if not return_meta:
        return out
    return out, {
        "bg": int(bg),
        "mark": int(mark),
        "rect_col": int(rect_col),
        "pieces": [p.copy() for p in pieces],
        "piece_positions": list(solved_placements),
        "assembled_shape": (int(H), int(W)),
        "top_left": (int(tr), int(tc)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble the scattered rectangular pieces into the marked rectangle.

    Variant rule:
    touching edges must match exactly, including rect_col pixels.
    """
    return _solve(grid, strict=True)


def _rand_splits(total: int, *, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
    segs: list[int] = []
    rem = total
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(rem)
            rem = 0
        else:
            hi = min(eff_max, rem - min_s)
            if min_s > hi:
                return None
            s = rng.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _build_groups(size: int, cuts: list[int]) -> list[list[int]]:
    cutset = set(int(c) for c in cuts)
    groups: list[list[int]] = []
    i = 0
    while i < size:
        if (i + 1) in cutset:
            groups.append([i, i + 1])
            i += 2
        else:
            groups.append([i])
            i += 1
    return groups


def _grow_connected_blob(
    interior: np.ndarray,
    occupied: np.ndarray,
    color: int,
    budget: int,
    *,
    rng: random.Random,
) -> None:
    H, W = interior.shape
    attempts = 0
    seed = None
    while attempts < 200:
        sr, sc = rng.randint(0, H - 1), rng.randint(0, W - 1)
        if occupied[sr, sc]:
            attempts += 1
            continue
        ok = True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = sr + dr, sc + dc
                if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            seed = (sr, sc)
            break
        attempts += 1
    if seed is None:
        return

    blob = {seed}
    frontier = [seed]
    while len(blob) < budget and frontier:
        rng.shuffle(frontier)
        nxt_frontier: list[tuple[int, int]] = []
        for fr, fc in frontier:
            dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            rng.shuffle(dirs)
            for dr, dc in dirs:
                nr, nc = fr + dr, fc + dc
                if (nr, nc) in blob:
                    continue
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
                    continue
                if occupied[nr, nc]:
                    continue
                conflict = False
                for dr2 in (-1, 0, 1):
                    for dc2 in (-1, 0, 1):
                        if dr2 == 0 and dc2 == 0:
                            continue
                        ar, ac = nr + dr2, nc + dc2
                        if 0 <= ar < H and 0 <= ac < W and occupied[ar, ac] and (ar, ac) not in blob:
                            conflict = True
                            break
                    if conflict:
                        break
                if conflict:
                    continue
                blob.add((nr, nc))
                nxt_frontier.append((nr, nc))
                if len(blob) >= budget:
                    break
            if len(blob) >= budget:
                break
        frontier = nxt_frontier

    for br, bc in blob:
        interior[br, bc] = int(color)
        occupied[br, bc] = True


def _build_rect(
    RH: int,
    RW: int,
    row_segs: list[int],
    col_segs: list[int],
    rect_col: int,
    interior_cols: list[int],
    *,
    rng: random.Random,
) -> np.ndarray | None:
    row_cuts = list(np.cumsum(row_segs[:-1], dtype=int))
    col_cuts = list(np.cumsum(col_segs[:-1], dtype=int))
    row_groups = _build_groups(RH, row_cuts)
    col_groups = _build_groups(RW, col_cuts)

    LH = len(row_groups)
    LW = len(col_groups)
    if LH < 4 or LW < 4:
        return None

    latent = np.full((LH, LW), int(rect_col), dtype=int)
    iH, iW = LH - 2, LW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), int(rect_col), dtype=int)
    target_fill = rng.uniform(0.38, 0.82)
    target_pixels = max(1, int(round(target_fill * iH * iW)))
    occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        budget = max(1, budget)
        _grow_connected_blob(interior, occupied, ic, budget, rng=rng)

    non_rc_count = int(np.sum(interior != int(rect_col)))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.28 or fill_ratio > 0.90:
        return None

    latent[1:-1, 1:-1] = interior

    rect = np.full((RH, RW), int(rect_col), dtype=int)
    for gr, rows in enumerate(row_groups):
        for gc, cols in enumerate(col_groups):
            rect[np.ix_(rows, cols)] = latent[gr, gc]
    return rect


def _extract_pieces(
    row_segs: list[int],
    col_segs: list[int],
    *,
    rng: random.Random,
) -> list[tuple[int, int, int, int]] | None:
    row_starts: list[int] = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts: list[int] = []
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr = len(row_segs)
    nc_grid = len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ris = [c[0] for c in cells]
        cis = [c[1] for c in cells]
        return min(ris), min(cis), max(ris), max(cis)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = rng.randint(4, min(12, n_cells))
    merges: list[tuple[str, int, int]] = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    rng.shuffle(merges)

    def count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    for direction, mi, mj in merges:
        if count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = get_piece_bbox(pid1)
        bb2 = get_piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] != pid1 and cell_id[ii][jj] != pid2:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc_grid):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces: list[tuple[int, int, int, int]] = []
    seen: set[int] = set()
    for i in range(nr):
        for j in range(nc_grid):
            pid = cell_id[i][j]
            if pid in seen:
                continue
            seen.add(pid)
            bb = get_piece_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 12:
        return None
    return pieces


def _edge_metrics(
    rect: np.ndarray,
    pieces: list[tuple[int, int, int, int]],
    rect_col: int,
) -> tuple[int, int]:
    RH, RW = rect.shape
    edges_v_left: list[np.ndarray] = []
    edges_v_right: list[np.ndarray] = []
    edges_h_top: list[np.ndarray] = []
    edges_h_bottom: list[np.ndarray] = []
    mixed_internal_edges = 0

    for pr, pc, ph, pw in pieces:
        if pc > 0:
            edge = rect[pr : pr + ph, pc]
            edges_v_left.append(edge.copy())
            if np.any(edge == int(rect_col)) and np.any(edge != int(rect_col)):
                mixed_internal_edges += 1
        if pc + pw < RW:
            edge = rect[pr : pr + ph, pc + pw - 1]
            edges_v_right.append(edge.copy())
            if np.any(edge == int(rect_col)) and np.any(edge != int(rect_col)):
                mixed_internal_edges += 1
        if pr > 0:
            edge = rect[pr, pc : pc + pw]
            edges_h_top.append(edge.copy())
            if np.any(edge == int(rect_col)) and np.any(edge != int(rect_col)):
                mixed_internal_edges += 1
        if pr + ph < RH:
            edge = rect[pr + ph - 1, pc : pc + pw]
            edges_h_bottom.append(edge.copy())
            if np.any(edge == int(rect_col)) and np.any(edge != int(rect_col)):
                mixed_internal_edges += 1

    strict_only_pairs = 0
    for e1 in edges_v_right:
        for e2 in edges_v_left:
            if len(e1) != len(e2):
                continue
            if _edge_compat(e1, e2, rect_col, strict=False) and not _edge_compat(e1, e2, rect_col, strict=True):
                strict_only_pairs += 1
    for e1 in edges_h_bottom:
        for e2 in edges_h_top:
            if len(e1) != len(e2):
                continue
            if _edge_compat(e1, e2, rect_col, strict=False) and not _edge_compat(e1, e2, rect_col, strict=True):
                strict_only_pairs += 1
    return mixed_internal_edges, strict_only_pairs


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = rng.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = rng.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = rng.randint(min_side, max_side)
    RW = rng.randint(min_side, max_side)
    if rng.random() < 0.5 and max_side >= 8:
        RH = rng.randint(max(6, min_side), max_side)
        RW = rng.randint(max(6, min_side), max_side)

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

    row_segs = _rand_splits(RH, rng=rng)
    col_segs = _rand_splits(RW, rng=rng)
    if row_segs is None or col_segs is None:
        return None

    pieces = _extract_pieces(row_segs, col_segs, rng=rng)
    if pieces is None:
        return None

    rect = _build_rect(RH, RW, row_segs, col_segs, rect_col, interior_cols, rng=rng)
    if rect is None:
        return None

    non_rc_total = int(np.sum(rect[1:-1, 1:-1] != int(rect_col)))
    total_interior = max(1, (RH - 2) * (RW - 2))
    fill_ratio = non_rc_total / total_interior
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        nrc = int(np.sum(piece != int(rect_col)))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = False
        for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
            if np.any(edge != int(rect_col)):
                has_non_rc_edge = True
                break
        if not has_non_rc_edge:
            return None
        if pr == 0 and not np.all(piece[0, :] == int(rect_col)):
            return None
        if pr + ph == RH and not np.all(piece[-1, :] == int(rect_col)):
            return None
        if pc == 0 and not np.all(piece[:, 0] == int(rect_col)):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == int(rect_col)):
            return None

    mixed_internal_edges, strict_only_pairs = _edge_metrics(rect, pieces, rect_col)
    if mixed_internal_edges < max(4, len(pieces)):
        return None
    if strict_only_pairs < 2:
        return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, RW - 1
    elif corner == "BL":
        anchor_r, anchor_c = RH - 1, 0
    else:
        anchor_r, anchor_c = RH - 1, RW - 1

    anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_r < pr + ph and pc <= anchor_c < pc + pw:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    apr, apc, aph, apw = pieces[anchor_idx]
    if corner == "TL" and not (apr == 0 and apc == 0):
        return None
    if corner == "TR" and not (apr == 0 and apc + apw == RW):
        return None
    if corner == "BL" and not (apr + aph == RH and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == RH and apc + apw == RW):
        return None

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

    canvas = np.full((GH, GW), int(bg), dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = int(marker_col)

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    order = list(range(len(piece_arrays)))
    rng.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
        except ValueError:
            return None

    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        tr, tc = mr - RH + 1, mc
    else:
        tr, tc = mr - RH + 1, mc - RW + 1

    expected_out = np.full((GH, GW), int(bg), dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    # Non-degeneracy check for the D1 variant:
    # reject samples where the original wildcard matcher still reconstructs
    # the same output, because such inputs do not force the stricter rule.
    try:
        wildcard_out = _solve(canvas, strict=False)
    except Exception:
        wildcard_out = None
    if wildcard_out is not None and np.array_equal(wildcard_out, expected_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 800 attempts")
