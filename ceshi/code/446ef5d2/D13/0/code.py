# Variant of 446ef5d2
# Dimension: D13 — 标记角类型的确定方式 (meta)
# Change: L 标记的缺失角直接编码了矩形的角类型（TL/TR/BL/BR），解题者通过观察 L 的朝向即可确定... → 角类型由碎片属性推断

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D13 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     old:
#       - marker 的 2x2 缺口位置直接决定 assembled rectangle 的 corner 类型
#       - solver 读出 missing 在 marker bbox 的哪一个角，就立刻知道是 TL/TR/BL/BR
#     new:
#       - marker 仍然只给出一个“最终拼接角落要放在哪个网格坐标”的位置 `(mr, mc)`
#       - marker 的 L 朝向与真实 corner 脱钩，4 种朝向独立随机
#       - 为了让“该看哪块碎片”在视觉上明确，generator 固定把锚块散放在 marker 右侧近邻
#       - 真正的 corner 类型不再来自 marker，而来自这块锚块的边属性：
#         哪两条相邻边是全 `rect_col`，就说明这块碎片在完整矩形中占据哪个角
#         top+left -> TL, top+right -> TR, bottom+left -> BL, bottom+right -> BR
#   concrete_pixel_behavior:
#     - 输入仍然包含 3 像素 L marker 和若干散落矩形碎片
#     - 但现在 L 的朝向不能再当作 corner 线索使用
#     - marker 右边固定贴着一块“锚块”，它是完整矩形四个角块之一
#     - 解题者需要先观察这块锚块，再看它哪两条邻边是整条 `rect_col`
#   small_trace:
#     old_flow:
#       - missing == marker bbox 的左上角 => 直接判成 BR
#       - 之后所有 candidate dimensions / top_left 计算都沿用这个 corner
#     new_flow:
#       - missing 只提供最终输出的锚定坐标
#       - 先根据 marker 右侧固定近邻位置找到锚块
#       - 若锚块的 top 和 left 两条边全是 `rect_col`，则 corner = TL
#       - 再按 TL 去筛候选尺寸、预放置锚块、拼装整矩形
#     difference:
#       - 原逻辑在读到 marker 后立即知道 corner；新逻辑必须额外读取碎片属性
#   complexity_calibration:
#     - 若只是“忽略 marker 朝向并暴力试四种 corner”，视觉上看不出 D13 变化，而且会退化成隐藏规则
#     - 因此需要一个可见的锚块定位机制：marker 右侧近邻固定有一块角块
#     - 同时约束该锚块“真实外边界对应的两条边全 `rect_col`，另外两条内部拼接边都含非 `rect_col`”
#       这样锚块自身的角型是单值、肉眼也可读
#   implicit_assumption_review:
#     - 原代码 `if missing == ...: corner = ...` 依赖“marker 朝向编码 corner”，在本变体中不成立，必须删除
#     - 原代码 `fits(H, W)`、`top_left` 计算本身仍然正确，但它们依赖一个已知 corner；
#       现在这个 corner 必须先从锚块推断出来
#     - 原代码 `anchor_idx` 通过“哪块碎片覆盖 missing”寻找锚块；原 generator 实际并不保证这件事，
#       且本变体也不再采用这个约定，因此需要改成新的固定近邻定位规则
#     - 原代码预放置 anchor piece 的条件是 `(mr, mc)` 正好是该 piece 当前 bbox 的某个角；
#       这在散放状态下没有意义。现在改成：只要找到锚块，就按推断出的 corner 把它预放到 assembled rectangle 对应角
#   placement_legality:
#     - marker 的 missing 坐标需要同时满足：
#       1. assembled rectangle 按真实 corner 放得下
#       2. 散放锚块按固定近邻位置放得下
#       3. marker 的任意随机 L 朝向不越界、不与锚块重叠
#     - 其余碎片继续用 `random_free_location(..., padding=1)` 散放，保证不重叠且彼此隔开
#   visual_observability:
#     - 人类从样例中能看到：marker 的朝向会变，但输出角型并不跟着 marker 朝向变
#     - 同时 marker 旁边总有一块近邻锚块，且这块锚块的两条整色边稳定地决定输出展开方向
#   cleanliness:
#     - marker 与锚块只做近邻，不做覆盖
#     - 其余碎片保持 1-cell 间隔，避免 marker / 锚块 / 散片混成一团
#   color_inference:
#     - 不引入新的颜色语义
#     - 仍然只依赖 `rect_col` 的“整条边”结构线索，不依赖颜色编号大小或排序


ANCHOR_COL_GAP = 2


def _marker_cells_from_gap(mr: int, mc: int, gap_corner: str) -> list[tuple[int, int]]:
    if gap_corner == "TL":
        return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
    if gap_corner == "TR":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    if gap_corner == "BL":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]


def _top_left_from_corner(mr: int, mc: int, H: int, W: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - W + 1
    if corner == "BL":
        return mr - H + 1, mc
    return mr - H + 1, mc - W + 1


def _infer_corner_from_anchor(piece: np.ndarray, rect_col: int) -> str:
    top_full = bool(np.all(piece[0, :] == int(rect_col)))
    bottom_full = bool(np.all(piece[-1, :] == int(rect_col)))
    left_full = bool(np.all(piece[:, 0] == int(rect_col)))
    right_full = bool(np.all(piece[:, -1] == int(rect_col)))

    candidates = []
    if top_full and left_full:
        candidates.append("TL")
    if top_full and right_full:
        candidates.append("TR")
    if bottom_full and left_full:
        candidates.append("BL")
    if bottom_full and right_full:
        candidates.append("BR")
    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at a marker-indicated position.

    D13 variant:
    - The L-marker no longer tells which corner of the assembled rectangle is anchored.
    - Instead, the marker points to a location, and a fixed nearby anchor piece tells the corner
      type via its two adjacent all-rect_col edges.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    # Find marker: a color appearing exactly 3 times forming an L in a 2x2 bbox.
    cnt = Counter(map(int, g.flatten()))
    mark = None
    missing = None
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
                break

    assert not (mark is None or missing is None)
    mr, mc = map(int, missing)

    # Decompose occupied mask (non-bg and non-marker) into rectangles.
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects = []  # (r, c, h, w, arr)
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
    assert total_area > 0

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    # Fixed near-marker placement rule for the anchor piece:
    # the anchor piece is the unique rectangle whose left edge starts two columns right
    # of the marker gap and whose vertical midpoint aligns with the marker gap row.
    anchor_candidates = []
    for i, (rr, cc, h, _w, _a) in enumerate(rects):
        if cc == mc + ANCHOR_COL_GAP and rr + (h // 2) == mr:
            anchor_candidates.append(i)
    assert len(anchor_candidates) == 1
    anchor_idx = anchor_candidates[0]
    corner = _infer_corner_from_anchor(pieces[anchor_idx], rect_col)

    # Candidate rectangle dimensions are factor pairs of total_area that fit in the grid
    # when anchored at (mr, mc) as the inferred corner.
    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def fits(H: int, W: int) -> bool:
        tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
        return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW

    dims = [(int(H), int(W)) for H, W in dims if fits(int(H), int(W))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1, v2) -> bool:
        a1 = np.array(v1, dtype=int)
        a2 = np.array(v2, dtype=int)
        mask = (a1 != int(rect_col)) & (a2 != int(rect_col))
        return bool((a1[mask] == a2[mask]).all())

    def assemble(H: int, W: int) -> np.ndarray | None:
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
                if m.any() and (not compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < W:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < H:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not compat(down[m], a[-1, :][m])):
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
        else:
            return None

        order = sorted(range(len(pieces)), key=lambda i: (-ph[i] * pw[i], -ph[i], -pw[i], i))

        def next_empty():
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
        return occ2 if solved else None

    ass = None
    used_dim = None
    for H, W in dims:
        ass = assemble(H, W)
        if ass is not None:
            used_dim = (H, W)
            break

    assert not (ass is None or used_dim is None)

    H, W = used_dim
    tr, tc = _top_left_from_corner(mr, mc, H, W, corner)
    out = np.full((GH, GW), int(bg), dtype=int)
    out[tr : tr + H, tc : tc + W] = ass
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(10, 26)
    GW = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = _r.randint(min_side, max_side)
    RW = _r.randint(min_side, max_side)
    if _r.random() < 0.5 and max_side >= 8:
        RH = _r.randint(max(6, min_side), max_side)
        RW = _r.randint(max(6, min_side), max_side)

    rect = np.full((RH, RW), rect_col, dtype=int)

    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = _r.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 1

        attempts = 0
        seed = None
        while attempts < 200:
            sr, sc = _r.randint(0, iH - 1), _r.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < iH and 0 <= nc < iW and blob_occupied[nr, nc]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                seed = (sr, sc)
                break
            attempts += 1

        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            _r.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob or nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW:
                                if blob_occupied[ar, ac] and (ar, ac) not in blob:
                                    conflict = True
                                    break
                        if conflict:
                            break
                    if conflict:
                        continue
                    blob.add((nr, nc))
                    nf.append((nr, nc))
                    if len(blob) >= budget:
                        break
                if len(blob) >= budget:
                    break
            frontier = nf

        for br, bc in blob:
            interior[br, bc] = ic
            blob_occupied[br, bc] = True

    non_rc_count = int(np.sum(interior != rect_col))
    fill_ratio = non_rc_count / (iH * iW)
    if fill_ratio < 0.30 or fill_ratio > 0.90:
        return None

    rect[1:-1, 1:-1] = interior

    def _rand_splits(total: int, min_s: int = 2, max_s: int = 7):
        eff_max = min(max_s, max(min_s, _r.randint(min_s, max_s)))
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
                s = _r.randint(min_s, hi)
                segs.append(s)
                rem -= s
        return segs

    row_segs = _rand_splits(RH)
    col_segs = _rand_splits(RW)
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

    def _get_piece_bbox(pid: int):
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int):
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = _r.randint(4, min(12, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    _r.shuffle(merges)

    def _count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc_grid)))

    for direction, mi, mj in merges:
        if _count_pieces() <= target_pieces:
            break
        if direction == "h":
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi][mj + 1]
        else:
            pid1 = cell_id[mi][mj]
            pid2 = cell_id[mi + 1][mj]
        if pid1 == pid2:
            continue
        bb1 = _get_piece_bbox(pid1)
        bb2 = _get_piece_bbox(pid2)
        if bb1 is None or bb2 is None:
            continue
        mr0 = min(bb1[0], bb2[0])
        mc0 = min(bb1[1], bb2[1])
        mr1 = max(bb1[2], bb2[2])
        mc1 = max(bb1[3], bb2[3])
        ph, pw = _pixel_size(mr0, mc0, mr1, mc1)
        if ph > 7 or pw > 7:
            continue
        ok = True
        for ii in range(mr0, mr1 + 1):
            for jj in range(mc0, mc1 + 1):
                if cell_id[ii][jj] not in (pid1, pid2):
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
            bb = _get_piece_bbox(pid)
            assert bb is not None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < 4 or len(pieces) > 12:
        return None

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = False
        for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]:
            if np.any(edge != rect_col):
                has_non_rc_edge = True
                break
        if not has_non_rc_edge:
            return None
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
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
    if aph < 3 or apw < 3:
        return None
    if corner == "TL" and not (apr == 0 and apc == 0):
        return None
    if corner == "TR" and not (apr == 0 and apc + apw == RW):
        return None
    if corner == "BL" and not (apr + aph == RH and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == RH and apc + apw == RW):
        return None

    anchor_piece = rect[apr : apr + aph, apc : apc + apw]
    top_full = bool(np.all(anchor_piece[0, :] == rect_col))
    bottom_full = bool(np.all(anchor_piece[-1, :] == rect_col))
    left_full = bool(np.all(anchor_piece[:, 0] == rect_col))
    right_full = bool(np.all(anchor_piece[:, -1] == rect_col))
    if corner == "TL":
        if not (top_full and left_full and (not bottom_full) and (not right_full)):
            return None
    elif corner == "TR":
        if not (top_full and right_full and (not bottom_full) and (not left_full)):
            return None
    elif corner == "BL":
        if not (bottom_full and left_full and (not top_full) and (not right_full)):
            return None
    else:
        if not (bottom_full and right_full and (not top_full) and (not left_full)):
            return None

    def _marker_ranges():
        row_lo = max(1, aph // 2)
        row_hi = min(GH - 2, GH - (aph - aph // 2))
        col_lo = 1
        col_hi = min(GW - 2, GW - apw - ANCHOR_COL_GAP)

        if corner == "TL":
            row_lo = max(row_lo, 0)
            row_hi = min(row_hi, GH - RH)
            col_lo = max(col_lo, 0)
            col_hi = min(col_hi, GW - RW)
        elif corner == "TR":
            row_lo = max(row_lo, 0)
            row_hi = min(row_hi, GH - RH)
            col_lo = max(col_lo, RW - 1)
            col_hi = min(col_hi, GW - 1)
        elif corner == "BL":
            row_lo = max(row_lo, RH - 1)
            row_hi = min(row_hi, GH - 1)
            col_lo = max(col_lo, 0)
            col_hi = min(col_hi, GW - RW)
        else:
            row_lo = max(row_lo, RH - 1)
            row_hi = min(row_hi, GH - 1)
            col_lo = max(col_lo, RW - 1)
            col_hi = min(col_hi, GW - 1)
        return row_lo, row_hi, col_lo, col_hi

    mr_lo, mr_hi, mc_lo, mc_hi = _marker_ranges()
    if mr_lo > mr_hi or mc_lo > mc_hi:
        return None
    mr = _r.randint(mr_lo, mr_hi)
    mc = _r.randint(mc_lo, mc_hi)

    marker_gap = _r.choice(["TL", "TR", "BL", "BR"])
    marker_cells = _marker_cells_from_gap(mr, mc, marker_gap)
    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    anchor_input_r = mr - (aph // 2)
    anchor_input_c = mc + ANCHOR_COL_GAP
    if anchor_input_r < 0 or anchor_input_c < 0:
        return None
    if anchor_input_r + aph > GH or anchor_input_c + apw > GW:
        return None
    paste(canvas, piece_arrays[anchor_idx], anchor_input_r, anchor_input_c)

    order = [i for i in range(len(piece_arrays)) if i != anchor_idx]
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1)
            paste(canvas, pa, r0, c0)
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
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
