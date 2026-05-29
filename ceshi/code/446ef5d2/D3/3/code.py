# Variant of 446ef5d2
# Dimension: D3 — 目标结构的形式 (dst)
# Change: 以 L 形标记的缺失角为起点，碎片拼接成一个完整矩形，矩形四条外边全为 rect_col... → 无标记角——从碎片边缘约束推断矩形位置

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the D3 variant
# ---------------------------------------------------------------------------
#
# variant_spec:
#   core_change:
#     old:
#       - 输入里有一个 3 像素 L 标记。
#       - L 标记缺失的那一格就是目标矩形的角，solver 先得到绝对角位置，再只做拼接。
#     new:
#       - 完全去掉 L 标记；输入只剩散落的矩形碎片。
#       - solver 必须只靠碎片边缘约束判断哪些边能成为矩形外边界：
#         某块若落在矩形外边，对应整条边必须全是 rect_col；
#         碎片相邻时，只有非 rect_col 像素必须逐格一致，rect_col 仍为通配色。
#       - 因为去掉 marker 会让“输出放在网格哪里”变成未定义，所以增加一个最小适配：
#         输出矩形的左上角固定为“输入所有碎片占据区域的左上角”。
#         这个锚点只解决绝对平移歧义，不参与拼接方向判断；真正的难点仍是边界推断。
#   concrete_pixel_behavior:
#     input:
#       - 背景色 bg 最常见。
#       - 4-9 个矩形碎片随机散落，彼此至少隔 1 格 bg。
#       - 没有任何单独标记物，也没有 3 像素 L 形颜色。
#       - 每块都是从某个更大矩形切出来的子矩形；最终大矩形四条外边都全是 rect_col。
#     output:
#       - 在与输入同尺寸的网格上，把所有碎片重新平移拼成完整矩形。
#       - 该矩形左上角 = 输入碎片并集 bbox 的左上角。
#   behavior_trace:
#     small_example:
#       - 原题：若 marker 缺口在 (5,7) 且表示 TL 角，则 solver 直接把最终矩形左上角定在 (5,7)。
#       - 变体：没有 marker。假设输入碎片最左上非 bg 像素在 (3,4)，则输出矩形左上角固定为 (3,4)。
#         接着 solver 枚举面积因子 (H,W)，并尝试把碎片放入 HxW 画布：
#         只有落在外边界的边才允许整条为 rect_col；内部接缝则检查非 rect_col 是否匹配。
#       - 差异：原题先知道角与朝向，再拼；变体必须从“哪些边能当外边界”反推每块在矩形中的角色。
#   complexity_calibration:
#     - 若仍允许 12 块且大矩形过大，去掉 anchor 后回溯会明显变慢，因此把生成器收紧到 4-9 块，
#       但保留原题的矩形分块、wildcard 匹配和不规则内部纹理，保证视觉上仍是同一题族。
#     - 为避免“去掉 marker 但样例肉眼差异不明显”，额外禁止任何颜色恰好以 3 像素 L 形出现。
#     - 为避免纯对称导致多解，生成器继续使用不规则 blob 内纹，并要求既定尺寸下只有 1 个拼装解。
#   implicit_assumption_review:
#     old_main_dependencies_removed:
#       - `cnt == 3` 且形成 2x2-L：新题不成立，整段 marker 识别与 corner 映射必须删除。
#       - `missing cell` 给出 anchor piece：新题不成立，不能再预放置角块。
#       - `fits(H,W)` 以 marker corner 为基准：改成以输入碎片 bbox 左上角为基准。
#     new_main_assumptions:
#       - 所有非 bg 连通块都是轴对齐矩形；generator 用矩形切块并留 1 格间距保证。
#       - rect_col 是所有碎片像素的众数；generator 在矩形边框和内部留白上保证这一点。
#       - 目标矩形左上角等于输入非 bg bbox 左上角；generator 通过把所有碎片散落在该点右下方并
#         固定一块从该点开始放置来保证。
#       - 固定尺寸 (H,W) 下拼装解唯一；generator 用计数回溯做拒绝采样保证。
#   placement_legality:
#     - 散落时所有碎片 top-left 都满足 r>=target_r, c>=target_c，因此输入 bbox 左上角稳定。
#     - 一块碎片固定放在 (target_r,target_c)，其余碎片在右下子画布内用 `random_free_location`
#       加 padding=1 放置，避免重叠与粘连。
#     - 输出仍是整块矩形写回原网格，不越界。
#   observability_and_cleanliness:
#     - 样例中不会再看到显眼的 L 标记，视觉差异直接可见。
#     - 输入上只有背景和多个矩形块，没有重叠层；输出只有一个完整矩形，结构整洁。
#     - 颜色规则没有引入基于颜色编号的排序或隐藏映射，只使用人可见的边界是否全 rect_col。
#


def _extract_rectangles(g: np.ndarray, bg: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    """Extract axis-aligned rectangular pieces from non-bg cells."""
    occ = g != int(bg)
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    GH, GW = g.shape
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


def _candidate_dims(total_area: int, max_h: int, max_w: int) -> list[tuple[int, int]]:
    dims: list[tuple[int, int]] = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh != 0:
            continue
        ww = total_area // hh
        for H, W in ((hh, ww), (ww, hh)):
            if H <= max_h and W <= max_w and H >= 5 and W >= 5:
                dims.append((int(H), int(W)))
    dims = sorted(set(dims), key=lambda t: (-min(t), -max(t), -t[0], -t[1]))
    return dims


def _assemble_rectangle(
    pieces: list[np.ndarray],
    rect_col: int,
    H: int,
    W: int,
    *,
    solution_limit: int = 1,
) -> tuple[np.ndarray | None, int]:
    """Backtracking packer. Returns (first_solution, number_of_solutions_up_to_limit)."""
    if not pieces:
        return None, 0

    shapes = [(int(p.shape[0]), int(p.shape[1])) for p in pieces]
    if any(h > H or w > W for h, w in shapes):
        return None, 0

    top_ok = [bool(np.all(p[0, :] == int(rect_col))) for p in pieces]
    bottom_ok = [bool(np.all(p[-1, :] == int(rect_col))) for p in pieces]
    left_ok = [bool(np.all(p[:, 0] == int(rect_col))) for p in pieces]
    right_ok = [bool(np.all(p[:, -1] == int(rect_col))) for p in pieces]

    order = sorted(
        range(len(pieces)),
        key=lambda i: (-shapes[i][0] * shapes[i][1], -shapes[i][0], -shapes[i][1], i),
    )

    board = np.full((H, W), -1, dtype=int)
    used = [False] * len(pieces)
    first_solution: np.ndarray | None = None
    solution_count = 0

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

    def can_place(i: int, r: int, c: int) -> bool:
        a = pieces[i]
        h, w = shapes[i]
        if r + h > H or c + w > W:
            return False
        if (board[r : r + h, c : c + w] != -1).any():
            return False
        if r == 0 and not top_ok[i]:
            return False
        if r + h == H and not bottom_ok[i]:
            return False
        if c == 0 and not left_ok[i]:
            return False
        if c + w == W and not right_ok[i]:
            return False

        if c > 0:
            left = board[r : r + h, c - 1]
            m = left != -1
            if m.any() and (not compat(left[m], a[:, 0][m])):
                return False
        if c + w < W:
            right = board[r : r + h, c + w]
            m = right != -1
            if m.any() and (not compat(right[m], a[:, -1][m])):
                return False
        if r > 0:
            up = board[r - 1, c : c + w]
            m = up != -1
            if m.any() and (not compat(up[m], a[0, :][m])):
                return False
        if r + h < H:
            down = board[r + h, c : c + w]
            m = down != -1
            if m.any() and (not compat(down[m], a[-1, :][m])):
                return False
        return True

    def place(i: int, r: int, c: int) -> None:
        a = pieces[i]
        h, w = shapes[i]
        board[r : r + h, c : c + w] = a
        used[i] = True

    def unplace(i: int, r: int, c: int) -> None:
        h, w = shapes[i]
        board[r : r + h, c : c + w] = -1
        used[i] = False

    def next_empty() -> tuple[int, int] | None:
        idx = np.argwhere(board == -1)
        if idx.size == 0:
            return None
        return int(idx[0, 0]), int(idx[0, 1])

    def bt() -> bool:
        nonlocal first_solution, solution_count
        nxt = next_empty()
        if nxt is None:
            solution_count += 1
            if first_solution is None:
                first_solution = board.copy()
            return solution_count >= solution_limit

        r, c = nxt
        for i in order:
            if used[i]:
                continue
            if not can_place(i, r, c):
                continue
            place(i, r, c)
            if bt():
                return True
            unplace(i, r, c)
        return False

    bt()
    return first_solution, solution_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a single rectangle without any marker.

    New rule:
    - All non-bg components are rectangular pieces cut from one larger rectangle.
    - rect_col is still the most common color across piece pixels and all four outer sides
      of the assembled rectangle must be entirely rect_col.
    - The assembled rectangle is written back with its top-left corner aligned to the
      top-left corner of the input pieces' occupied bbox.
    """
    g = np.array(grid, dtype=int)
    GH, GW = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    rects = _extract_rectangles(g, bg)
    assert rects

    pieces = [a for _r, _c, _h, _w, a in rects]
    total_area = int(sum(int(a.shape[0]) * int(a.shape[1]) for a in pieces))

    allpix: list[int] = []
    for a in pieces:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    top_r = min(int(r) for r, _c, _h, _w, _a in rects)
    top_c = min(int(c) for _r, c, _h, _w, _a in rects)
    dims = _candidate_dims(total_area, GH - top_r, GW - top_c)

    assembled = None
    used_dim = None
    for H, W in dims:
        candidate, count = _assemble_rectangle(pieces, rect_col, H, W, solution_limit=1)
        if count >= 1 and candidate is not None:
            assembled = candidate
            used_dim = (H, W)
            break

    assert assembled is not None and used_dim is not None

    out = np.full((GH, GW), int(bg), dtype=int)
    H, W = used_dim
    out[top_r : top_r + H, top_c : top_c + W] = assembled
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _has_marker_like_triplet(grid: np.ndarray, bg: int) -> bool:
    """Reject accidental leftovers that visually mimic the removed L marker."""
    cnt = Counter(map(int, grid.flatten()))
    for col, n in cnt.items():
        if int(col) == int(bg) or int(n) != 3:
            continue
        pts = [(int(r), int(c)) for r, c in np.argwhere(grid == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        if max(rs) - min(rs) == 1 and max(cs) - min(cs) == 1:
            all4 = {
                (min(rs), min(cs)),
                (min(rs), max(cs)),
                (max(rs), min(cs)),
                (max(rs), max(cs)),
            }
            if len(all4 - set(pts)) == 1:
                return True
    return False


def _rand_splits(total: int, rng, min_s: int = 2, max_s: int = 6) -> list[int] | None:
    """Split total into segments each in [min_s, max_s]."""
    eff_max = min(max_s, max(min_s, rng.randint(min_s, max_s)))
    segs: list[int] = []
    rem = int(total)
    while rem > 0:
        if rem <= eff_max:
            if rem < min_s:
                return None
            segs.append(rem)
            break
        hi = min(eff_max, rem - min_s)
        if min_s > hi:
            return None
        s = rng.randint(min_s, hi)
        segs.append(s)
        rem -= s
    return segs


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    GH = _r.randint(12, 26)
    GW = _r.randint(12, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[2 : 2 + n_interior]

    max_side = min(16, GH - 2, GW - 2)
    if max_side < 5:
        return None
    RH = _r.randint(5, max_side)
    RW = _r.randint(5, max_side)
    if RH * RW > int(0.42 * GH * GW):
        return None

    tr = _r.randint(0, GH - RH)
    tc = _r.randint(0, GW - RW)
    if (GH - tr) * (GW - tc) < int(1.30 * RH * RW):
        return None

    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None

    interior = np.full((iH, iW), rect_col, dtype=int)
    target_fill = _r.uniform(0.40, 0.82)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(2, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 2

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
                    if nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if (nr, nc) in blob or blob_occupied[nr, nc]:
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

    row_segs = _rand_splits(RH, _r)
    col_segs = _rand_splits(RW, _r)
    if row_segs is None or col_segs is None:
        return None

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
    nc = len(col_segs)
    cell_id = [[i * nc + j for j in range(nc)] for i in range(nr)]
    n_cells = nr * nc
    if n_cells < 4:
        return None

    target_pieces = _r.randint(4, min(9, n_cells))

    def _get_piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc) if cell_id[i][j] == pid]
        if not cells:
            return None
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def _pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    merges = []
    for i in range(nr):
        for j in range(nc):
            if j + 1 < nc:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    _r.shuffle(merges)

    def _count_pieces() -> int:
        return len(set(cell_id[i][j] for i in range(nr) for j in range(nc)))

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
                if cell_id[ii][jj] != pid1 and cell_id[ii][jj] != pid2:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        for ii in range(nr):
            for jj in range(nc):
                if cell_id[ii][jj] == pid2:
                    cell_id[ii][jj] = pid1

    pieces_meta: list[tuple[int, int, int, int]] = []
    seen = set()
    for i in range(nr):
        for j in range(nc):
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
            pieces_meta.append((pr, pc, ph, pw))

    if len(pieces_meta) < 4 or len(pieces_meta) > 9:
        return None

    top_left_piece_idx = None
    for i, (pr, pc, _ph, _pw) in enumerate(pieces_meta):
        if pr == 0 and pc == 0:
            top_left_piece_idx = i
            break
    if top_left_piece_idx is None:
        return None

    piece_arrays: list[np.ndarray] = []
    for pr, pc, ph, pw in pieces_meta:
        piece = rect[pr : pr + ph, pc : pc + pw].copy()
        piece_arrays.append(piece)
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        has_non_rc_edge = any(
            np.any(edge != rect_col) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]
        )
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

    all_piece_pixels = [int(x) for a in piece_arrays for x in a.flatten()]
    if int(Counter(all_piece_pixels).most_common(1)[0][0]) != int(rect_col):
        return None

    _, unique_count = _assemble_rectangle(piece_arrays, rect_col, RH, RW, solution_limit=2)
    if unique_count != 1:
        return None

    canvas = np.full((GH, GW), bg, dtype=int)
    pinned_idx = _r.randrange(len(piece_arrays))
    paste(canvas, piece_arrays[pinned_idx], tr, tc)

    order = [i for i in range(len(piece_arrays)) if i != pinned_idx]
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        sub = canvas[tr:, tc:]
        try:
            rr, cc = random_free_location(sub, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        paste(canvas, pa, tr + rr, tc + cc)

    if int(Counter(map(int, canvas.flatten())).most_common(1)[0][0]) != int(bg):
        return None
    if _has_marker_like_triplet(canvas, bg):
        return None

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
