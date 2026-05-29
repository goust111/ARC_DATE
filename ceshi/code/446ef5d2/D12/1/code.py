# Variant of 446ef5d2
# Dimension: D12 — interior_cols 的数量与分布 (meta)
# Change: 1-3 种 interior_cols，在矩形内部形成不规则连通色块，不同颜色之间 8 连通分离...
#   → interior_cols 按碎片独立分配；每块碎片都有自己的局部颜色组合，不要求整块矩形共享同一组内部颜色。
#
# Design analysis
# - Concrete behavior:
#   输入仍然是若干矩形碎片和一个 3 像素 L 标记；输出仍然是把碎片平移回标记指示的矩形角。
#   变化只在碎片内部配色：原题先生成整块矩形内部的 1-3 个全局 blob，再切成碎片，所以同一种 interior_col
#   往往跨多个碎片延续。这里改成先切碎，再给每块碎片独立生成 1-3 个局部 blob。共享边界上只保留少量同步的
#   “缝线”颜色用于匹配，因此人眼不再能依赖“这两块都含有同一大块绿色”之类的全局线索，只能看边缘是否拼得上。
#   小例子：原题里左块右边界和右块左边界之所以能对上，常常同时也属于同一个跨块颜色区域；现在左块可能主色是红，
#   右块主色是黄，但它们在共享边界的 2 个位置都放蓝色缝线，于是 solver 必须按边界向量匹配，而不是按整块内部颜色猜。
# - Complexity calibration:
#   仍限制每块最多 1-3 个局部颜色、总颜色池 3-5 个，避免颜色过多导致杂乱；同时要求多数相邻碎片的颜色集合不同，
#   让 D12 变化肉眼可见。共享边界上每段至少放 1 个非 rect_col 像素，避免边界信息过弱导致装配歧义。
# - Implicit assumptions review for main():
#   原代码 `rect_col = most_common(piece pixels)` 依赖“全局 interior_col 数少且被整块 blob 分摊”，因此 rect_col
#   通常是单色最高频。D12 下各碎片可独立高填充、独立组合颜色，这个假设不再稳健。新 solver 不再硬编码最高频色，
#   而是把所有出现过的碎片颜色当作候选 wildcard/border 色逐个尝试装配，按边缘频次排序以保持效率。
#   其余装配逻辑仍成立：碎片仍是轴对齐矩形、不能旋转、外边框仍全是 rect_col、共享边只要求非 rect_col 位置相等。
# - Placement legality:
#   生成器先分割出矩形拼图，再在 assembled 坐标系里给每块上色，最后散落到背景上。标记先放置，碎片用
#   `random_free_location(..., padding=1)` 散开，保证不重叠、不错位、彼此间至少隔 1 个背景格。
# - Visual observability and cleanliness:
#   每块碎片都保留局部 blob，且相邻碎片常有不同颜色组合；共享边界的缝线很短，所以输出看起来是“边能对上、内部花纹
#   各自为政”的拼图，而不是原题那种跨块连续大色斑。颜色池控制在 3-5 个，避免网格过花。
# - New assumptions introduced by main():
#   新 solver 假设真实 rect_col 至少是某个实际出现过的碎片颜色，并且用它做 wildcard 时存在唯一可装配矩形。
#   生成器通过保证外边框全为 rect_col、内边界缝线足够多、每块至少一条信息性边来满足这一点。

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import paste, random_free_location


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], str]:
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
    return int(mark), (int(missing[0]), int(missing[1])), corner


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


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    m = (v1 != int(rect_col)) & (v2 != int(rect_col))
    return bool((v1[m] == v2[m]).all())


def main(grid: np.ndarray) -> np.ndarray:
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
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
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

    assert not (ass is None or used_dim is None)

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
    return out


def _rand_splits(total: int, rng: random.Random, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
    segs = []
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


def _partition_rectangle(RH: int, RW: int, rng: random.Random) -> list[tuple[int, int, int, int]] | None:
    row_segs = _rand_splits(RH, rng)
    col_segs = _rand_splits(RW, rng)
    if row_segs is None or col_segs is None:
        return None

    row_starts = []
    s = 0
    for rs in row_segs:
        row_starts.append(s)
        s += rs
    col_starts = []
    s = 0
    for cs in col_segs:
        col_starts.append(s)
        s += cs

    nr, nc_grid = len(row_segs), len(col_segs)
    cell_id = [[i * nc_grid + j for j in range(nc_grid)] for i in range(nr)]
    n_cells = nr * nc_grid
    if n_cells < 4:
        return None

    def get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = rng.randint(4, min(12, n_cells))
    merges = []
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

    pieces = []
    seen = set()
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
    for _pr, _pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
    return pieces


def _build_adjacencies(
    pieces: list[tuple[int, int, int, int]]
) -> list[dict[str, object]]:
    adj = []
    for i, (pr1, pc1, ph1, pw1) in enumerate(pieces):
        for j in range(i + 1, len(pieces)):
            pr2, pc2, ph2, pw2 = pieces[j]
            if pc1 + pw1 == pc2 or pc2 + pw2 == pc1:
                r0 = max(pr1, pr2)
                r1 = min(pr1 + ph1, pr2 + ph2)
                if r0 < r1:
                    if pc1 + pw1 == pc2:
                        left, right = i, j
                    else:
                        left, right = j, i
                    lpr, lpc, lph, lpw = pieces[left]
                    rpr, rpc, rph, rpw = pieces[right]
                    cells = []
                    for gr in range(r0, r1):
                        cells.append((gr - lpr, lpw - 1, gr - rpr, 0))
                    adj.append({"a": left, "b": right, "cells": cells})
            if pr1 + ph1 == pr2 or pr2 + ph2 == pr1:
                c0 = max(pc1, pc2)
                c1 = min(pc1 + pw1, pc2 + pw2)
                if c0 < c1:
                    if pr1 + ph1 == pr2:
                        top, bottom = i, j
                    else:
                        top, bottom = j, i
                    tpr, tpc, tph, tpw = pieces[top]
                    bpr, bpc, bph, bpw = pieces[bottom]
                    cells = []
                    for gc in range(c0, c1):
                        cells.append((tph - 1, gc - tpc, 0, gc - bpc))
                    adj.append({"a": top, "b": bottom, "cells": cells})
    return adj


def _neighbors4_local(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _neighbors8_local(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                out.append((nr, nc))
    return out


def _connect_same_color(
    arr: np.ndarray,
    protected: np.ndarray,
    color: int,
    rect_col: int,
) -> bool:
    h, w = arr.shape
    seeds = [(int(r), int(c)) for r, c in np.argwhere(arr == int(color))]
    if not seeds:
        return True

    comp_seen = set()
    components = []
    for seed in seeds:
        if seed in comp_seen:
            continue
        q = deque([seed])
        comp_seen.add(seed)
        comp = [seed]
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4_local(r, c, h, w):
                if (nr, nc) in comp_seen or int(arr[nr, nc]) != int(color):
                    continue
                comp_seen.add((nr, nc))
                q.append((nr, nc))
                comp.append((nr, nc))
        components.append(comp)
    if len(components) <= 1:
        return True

    main_comp = set(components[0])
    for comp in components[1:]:
        start = next(iter(main_comp))
        goal = comp[0]
        q = deque([start])
        parent = {start: None}
        found = False
        while q and not found:
            r, c = q.popleft()
            for nr, nc in _neighbors4_local(r, c, h, w):
                if (nr, nc) in parent:
                    continue
                cell = int(arr[nr, nc])
                if cell != int(rect_col) and cell != int(color):
                    continue
                if bool(protected[nr, nc]) and cell != int(color):
                    continue
                parent[(nr, nc)] = (r, c)
                if (nr, nc) == goal:
                    found = True
                    break
                q.append((nr, nc))
        if not found:
            return False
        cur = goal
        while cur is not None:
            rr, cc = cur
            arr[rr, cc] = int(color)
            main_comp.add(cur)
            cur = parent[cur]
    return True


def _grow_piece_blobs(
    arr: np.ndarray,
    protected: np.ndarray,
    colors: list[int],
    rect_col: int,
    rng: random.Random,
) -> bool:
    h, w = arr.shape
    piece_area = h * w
    existing = int(np.sum(arr != int(rect_col)))
    min_target = max(existing, max(1, int(np.ceil(0.10 * piece_area))))
    max_target = max(min_target, int(np.floor(0.85 * piece_area)))
    if min_target > max_target:
        return False
    target_total = rng.randint(min_target, max_target)

    counts = {int(c): int(np.sum(arr == int(c))) for c in colors}
    remaining = target_total - sum(counts.values())
    if remaining < 0:
        return False

    weights = [rng.randint(1, 4) for _ in colors]
    targets = dict(counts)
    if remaining > 0:
        ws = sum(weights)
        added = 0
        for idx, color in enumerate(colors):
            extra = remaining * weights[idx] // ws
            targets[int(color)] += extra
            added += extra
        for _ in range(remaining - added):
            color = int(rng.choice(colors))
            targets[color] += 1

    frontiers = {int(c): [(int(r), int(c0)) for r, c0 in np.argwhere(arr == int(c))] for c in colors}
    stall_rounds = 0
    while sum(counts.values()) < target_total and stall_rounds < len(colors) * 4 + 8:
        progressed = False
        color_order = list(colors)
        rng.shuffle(color_order)
        for color in color_order:
            color = int(color)
            if counts[color] >= targets[color]:
                continue
            frontier = frontiers[color]
            rng.shuffle(frontier)
            grown = False
            for fr, fc in frontier[:]:
                nbrs = _neighbors4_local(fr, fc, h, w)
                rng.shuffle(nbrs)
                for nr, nc in nbrs:
                    if int(arr[nr, nc]) != int(rect_col):
                        continue
                    if bool(protected[nr, nc]):
                        continue
                    arr[nr, nc] = color
                    frontier.append((nr, nc))
                    counts[color] += 1
                    progressed = True
                    grown = True
                    break
                if grown:
                    break
        if progressed:
            stall_rounds = 0
        else:
            stall_rounds += 1

    fill = int(np.sum(arr != int(rect_col))) / piece_area
    return 0.10 <= fill <= 0.85


def _color_piece(
    base: np.ndarray,
    piece_box: tuple[int, int, int, int],
    rect_dims: tuple[int, int],
    pool: list[int],
    rect_col: int,
    rng: random.Random,
) -> np.ndarray | None:
    pr, pc, ph, pw = piece_box
    RH, RW = rect_dims
    arr = base.copy()
    protected = np.zeros_like(arr, dtype=bool)
    if pr == 0:
        protected[0, :] = True
    if pr + ph == RH:
        protected[-1, :] = True
    if pc == 0:
        protected[:, 0] = True
    if pc + pw == RW:
        protected[:, -1] = True

    seed_colors = sorted(set(map(int, arr[arr != int(rect_col)].flatten())))
    if not seed_colors:
        return None

    extras = [int(c) for c in pool if int(c) not in seed_colors]
    rng.shuffle(extras)
    while len(seed_colors) < min(3, len(pool)) and extras and rng.random() < 0.65:
        seed_colors.append(extras.pop())
    seed_colors = seed_colors[:3]

    free_cells = [
        (r, c)
        for r in range(ph)
        for c in range(pw)
        if int(arr[r, c]) == int(rect_col) and not bool(protected[r, c])
    ]
    if len(free_cells) < max(1, len(seed_colors) - sum(np.any(arr == int(color)) for color in seed_colors)):
        return None

    for color in seed_colors:
        if np.any(arr == int(color)):
            continue
        rr, cc = rng.choice(free_cells)
        free_cells.remove((rr, cc))
        arr[rr, cc] = int(color)

    counts = {int(c): int(np.sum(arr == int(c))) for c in seed_colors}
    piece_area = ph * pw
    target_total = rng.randint(max(sum(counts.values()), max(1, piece_area // 8)), max(sum(counts.values()), int(piece_area * 0.55)))
    frontiers = {
        int(c): [(int(r), int(c0)) for r, c0 in np.argwhere(arr == int(c))]
        for c in seed_colors
    }
    stalls = 0
    while sum(counts.values()) < target_total and stalls < 80:
        progressed = False
        color_order = list(seed_colors)
        rng.shuffle(color_order)
        for color in color_order:
            frontier = frontiers[int(color)]
            rng.shuffle(frontier)
            for fr, fc in frontier[:]:
                nbrs = _neighbors4_local(fr, fc, ph, pw)
                rng.shuffle(nbrs)
                for nr, nc in nbrs:
                    if bool(protected[nr, nc]) or int(arr[nr, nc]) != int(rect_col):
                        continue
                    arr[nr, nc] = int(color)
                    frontier.append((nr, nc))
                    counts[int(color)] += 1
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
    if fill < 0.08 or fill > 0.70:
        return None

    if pr == 0 and not np.all(arr[0, :] == int(rect_col)):
        return None
    if pr + ph == RH and not np.all(arr[-1, :] == int(rect_col)):
        return None
    if pc == 0 and not np.all(arr[:, 0] == int(rect_col)):
        return None
    if pc + pw == RW and not np.all(arr[:, -1] == int(rect_col)):
        return None
    has_non_rc_edge = False
    for edge in [arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]]:
        if np.any(edge != int(rect_col)):
            has_non_rc_edge = True
            break
    if not has_non_rc_edge:
        return None
    return arr


def _is_outer_border_cell(
    piece_box: tuple[int, int, int, int],
    rect_dims: tuple[int, int],
    r: int,
    c: int,
) -> bool:
    pr, pc, ph, pw = piece_box
    RH, RW = rect_dims
    if pr == 0 and r == 0:
        return True
    if pr + ph == RH and r == ph - 1:
        return True
    if pc == 0 and c == 0:
        return True
    if pc + pw == RW and c == pw - 1:
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()

    GH = rng.randint(10, 26)
    GW = rng.randint(10, 26)

    palette = list(range(10))
    rng.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    pool_size = rng.randint(3, 5)
    interior_pool = [int(c) for c in palette[3 : 3 + pool_size]]

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

    pieces = _partition_rectangle(RH, RW, rng)
    if pieces is None:
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

    adj = _build_adjacencies(pieces)
    piece_arrays = [np.full((ph, pw), rect_col, dtype=int) for _pr, _pc, ph, pw in pieces]
    piece_palettes = []
    for _ in pieces:
        k = rng.randint(1, min(2, len(interior_pool)))
        piece_palettes.append(set(rng.sample(interior_pool, k)))

    if len({tuple(sorted(s)) for s in piece_palettes}) < 2:
        return None

    rng.shuffle(adj)
    touched = [0] * len(pieces)
    for item in adj:
        a = int(item["a"])
        b = int(item["b"])
        cells = item["cells"]
        shared = list(piece_palettes[a] | piece_palettes[b])
        if not shared:
            return None
        color = int(rng.choice(shared))
        piece_palettes[a].add(color)
        piece_palettes[b].add(color)
        available = []
        for pos, (ra, ca, rb, cb) in enumerate(cells):
            if _is_outer_border_cell(pieces[a], (RH, RW), ra, ca):
                continue
            if _is_outer_border_cell(pieces[b], (RH, RW), rb, cb):
                continue
            available.append(pos)
        seg_len = len(available)
        if seg_len == 0:
            return None
        min_marks = 1 if seg_len <= 2 else 2
        max_marks = max(min_marks, min(seg_len, int(np.ceil(0.7 * seg_len))))
        mark_count = rng.randint(min_marks, max_marks)
        idxs = sorted(rng.sample(available, mark_count))
        for pos in idxs:
            ra, ca, rb, cb = cells[pos]
            piece_arrays[a][ra, ca] = color
            piece_arrays[b][rb, cb] = color
        touched[a] += 1
        touched[b] += 1
    if min(touched) == 0:
        return None

    colored_pieces = []
    for i, box in enumerate(pieces):
        arr = _color_piece(
            piece_arrays[i],
            box,
            (RH, RW),
            sorted(piece_palettes[i]),
            rect_col,
            rng,
        )
        if arr is None:
            return None
        colored_pieces.append(arr)

    rect = np.full((RH, RW), rect_col, dtype=int)
    for (pr, pc, ph, pw), arr in zip(pieces, colored_pieces):
        rect[pr : pr + ph, pc : pc + pw] = arr

    for item in adj:
        a = int(item["a"])
        b = int(item["b"])
        cells = item["cells"]
        for ra, ca, rb, cb in cells:
            va = int(colored_pieces[a][ra, ca])
            vb = int(colored_pieces[b][rb, cb])
            if va != rect_col and vb != rect_col and va != vb:
                return None

    palette_signatures = [tuple(sorted(set(map(int, arr[arr != int(rect_col)].flatten())))) for arr in colored_pieces]
    if len(set(palette_signatures)) < 2:
        return None
    diff_adj = 0
    for item in adj:
        sa = set(palette_signatures[int(item["a"])])
        sb = set(palette_signatures[int(item["b"])])
        if sa != sb:
            diff_adj += 1
    if diff_adj < max(1, len(adj) // 2):
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

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    order = list(range(len(colored_pieces)))
    rng.shuffle(order)
    for idx in order:
        pa = colored_pieces[idx]
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
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
