# Variant of 446ef5d2
# Dimension: D8 — 标记系统的形式 (symbol_instructions)
# Change: 3 像素 L 形标记（2x2 缺一角），使用独特颜色（仅出现 3 次），缺失角指示矩形的一个顶点... → 单像素标记+方向编码
#
# Design analysis
# ---------------
# spec_like:
#   unchanged_core:
#     - 输入仍由若干散落的矩形 piece 组成；piece 本身不旋转，只平移。
#     - 所有 piece 恰好可拼成一个大矩形；大矩形四条外边都为 rect_col。
#     - 相邻 piece 的共享边按“非 rect_col 必须逐像素相等；rect_col 充当通配色”来匹配。
#     - 输出仍是同尺寸网格，只有拼好的矩形，其余为 bg。
#   D8_change:
#     - 旧题的 marker 是 3 像素 L 形，缺失格直接告诉 solver “哪个角是矩形角点”。
#     - 本变体改成 1 个独立像素 marker；它不再通过局部几何缺角编码方向，而是通过
#       自己位于整张网格中心的哪个象限来编码方向：
#         * marker 在左上象限 -> assembled rectangle 以该像素为 TL 角，向右下展开
#         * marker 在右上象限 -> 以该像素为 TR 角，向左下展开
#         * marker 在左下象限 -> 以该像素为 BL 角，向右上展开
#         * marker 在右下象限 -> 以该像素为 BR 角，向左上展开
#     - 为了避免“中心线归属不明”，generator 不会把 marker 放到中心行/中心列上。
#   concrete_pixel_behavior:
#     - 输入里 marker 是一个孤立的单像素非背景连通分量；所有 piece 至少 2x2，因此不会与
#       marker 的 1x1 形态混淆。
#     - 输出里 marker 会消失，因为最终矩形会覆盖它所在的那个角点。
#   behavior_trace:
#     - 原逻辑：先找一个出现 3 次且构成 2x2 缺一角的颜色 -> 得到缺失格 missing -> 用
#       missing 推出 corner。
#     - 新逻辑：先找唯一的 1x1 非背景连通分量 -> 得到 marker 坐标 (mr, mc) -> 与
#       网格中心比较，推出 corner。
#     - 差异：方向信息从“marker 的内部形状”转移到“marker 的全局位置”。
#   complexity_calibration:
#     - 若只用“单像素 marker”而不额外编码方向，题目会退化为四种输出都可能的歧义任务。
#       这里用 4 个象限做 4 档方向编码，保持自由度很小且每档都能在少量样例中重复出现。
#     - 仍保留原题的 piece 边缘匹配复杂度，因此变体只改变 D8，不把题目简化成纯定位题。
#   implicit_assumption_audit:
#     - 原 solver 中“找出现 3 次的颜色并检查其 2x2 缺角”这段代码直接依赖旧 marker 形态；
#       变体下不成立，必须改成“找面积为 1 的非背景连通分量”。
#     - 原 solver 中 `missing` 同时还是未来矩形角点；新题没有 `missing`，marker 自身就是角点。
#     - 原 solver 的 anchor 假设：散落输入中有一个 piece 恰好覆盖 `missing`，且该格是 piece 的
#       对应角。这在旧题无害，因为 missing 为空；新题里 marker 自己占据角点，该假设失效，
#       不能再依赖预放置 anchor piece。
#     - 其余矩形分解、候选尺寸枚举、边界颜色检查、piece 边匹配逻辑都不依赖 marker 形态，
#       可以保持。
#   placement_legality:
#     - generator 先选 corner，再把 marker 放到对应象限内一个合法坐标，保证：
#         1. marker 所在位置作为该角时，矩形能完全放进网格；
#         2. marker 不在中心行/列；
#         3. 散落 piece 使用 `random_free_location(..., padding=1)`，不会与 marker 重叠。
#   visual_observability:
#     - 人类可见的变化是：输入不再有显眼的 L 形标记，只剩一个孤立像素；但这个像素出现在
#       不同象限时，输出矩形会从该点向不同方向展开。对比 2-4 组样例可以观察出“象限->方向”。
#   cleanliness:
#     - marker 独立、piece 之间至少隔 1 个 bg 像素，避免与碎片混成一团。
#     - 不使用颜色编号大小关系，方向仅由可见位置决定。

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.ndimage import label

from arc_utils import STRUCT4, paste, random_free_location


def _corner_from_quadrant(mr: int, mc: int, gh: int, gw: int) -> list[str]:
    """Infer candidate rectangle corners from the marker quadrant.

    Generator never places the marker on a center line, so this normally returns
    exactly one corner. The fallback to 2 or 4 candidates only exists so the
    solver fails gracefully on ambiguous hand-made inputs.
    """

    if gh % 2 == 1 and mr == gh // 2:
        row_tags = ["T", "B"]
    else:
        row_tags = ["T"] if mr < (gh / 2.0) else ["B"]

    if gw % 2 == 1 and mc == gw // 2:
        col_tags = ["L", "R"]
    else:
        col_tags = ["L"] if mc < (gw / 2.0) else ["R"]

    out = []
    for rt in row_tags:
        for ct in col_tags:
            out.append(rt + ct)
    return out


def _find_singleton_marker(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    """Return (marker_color, row, col) for the unique 1-cell non-bg component."""

    non_bg = g != int(bg)
    lab, nlab = label(non_bg.astype(int), structure=STRUCT4)
    marker = None
    for k in range(1, int(nlab) + 1):
        pts = np.argwhere(lab == k)
        if pts.shape[0] == 1:
            rr, cc = map(int, pts[0])
            if marker is not None:
                raise AssertionError("expected exactly one singleton marker component")
            marker = (int(g[rr, cc]), rr, cc)
    if marker is None:
        raise AssertionError("singleton marker not found")
    return marker


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at a singleton marker.

    Input structure:
    - bg = most common color.
    - One isolated 1x1 marker pixel of a distinct color.
    - Several scattered non-bg rectangular pieces.

    Rule:
    - The marker pixel itself is the rectangle corner cell.
    - Which corner it denotes is inferred from the marker's quadrant relative to
      the grid center: TL/TR/BL/BR respectively.
    - rect_col is the most common color across all piece pixels and acts as a
      wildcard when matching shared boundaries.
    """

    g = np.array(grid, dtype=int)
    gh, gw = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, mr, mc = _find_singleton_marker(g, bg)
    corner_candidates = _corner_from_quadrant(mr, mc, gh, gw)

    # Decompose occupied mask (non-bg and non-marker) into rectangles.
    occ = g != int(bg)
    occ[mr, mc] = False
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            break
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < gw and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < gh and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))

    assert rects
    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))

    allpix: list[int] = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))

    def top_left_for(corner: str, h: int, w: int) -> tuple[int, int]:
        if corner == "TL":
            return mr, mc
        if corner == "TR":
            return mr, mc - w + 1
        if corner == "BL":
            return mr - h + 1, mc
        return mr - h + 1, mc - w + 1

    def fits(corner: str, h: int, w: int) -> bool:
        tr, tc = top_left_for(corner, h, w)
        return 0 <= tr and 0 <= tc and (tr + h) <= gh and (tc + w) <= gw

    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        vv1 = np.array(v1, dtype=int)
        vv2 = np.array(v2, dtype=int)
        m = (vv1 != int(rect_col)) & (vv2 != int(rect_col))
        return bool((vv1[m] == vv2[m]).all())

    def assemble(h_big: int, w_big: int) -> np.ndarray | None:
        occ2 = np.full((h_big, w_big), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if r == 0 and not np.all(a[0, :] == int(rect_col)):
                return False
            if (r + h) == h_big and not np.all(a[-1, :] == int(rect_col)):
                return False
            if c == 0 and not np.all(a[:, 0] == int(rect_col)):
                return False
            if (c + w) == w_big and not np.all(a[:, -1] == int(rect_col)):
                return False
            return True

        def neighbor_ok(a: np.ndarray, r: int, c: int) -> bool:
            h, w = a.shape
            if c > 0:
                left = occ2[r : r + h, c - 1]
                m = left != -1
                if m.any() and (not compat(left[m], a[:, 0][m])):
                    return False
            if (c + w) < w_big:
                right = occ2[r : r + h, c + w]
                m = right != -1
                if m.any() and (not compat(right[m], a[:, -1][m])):
                    return False
            if r > 0:
                up = occ2[r - 1, c : c + w]
                m = up != -1
                if m.any() and (not compat(up[m], a[0, :][m])):
                    return False
            if (r + h) < h_big:
                down = occ2[r + h, c : c + w]
                m = down != -1
                if m.any() and (not compat(down[m], a[-1, :][m])):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            h, w = a.shape
            if (r + h) > h_big or (c + w) > w_big:
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
        return occ2 if solved else None

    ass = None
    used_dim = None
    used_corner = None
    for corner in corner_candidates:
        valid_dims = [(int(hh), int(ww)) for hh, ww in dims if fits(corner, int(hh), int(ww))]
        for h_big, w_big in valid_dims:
            ass = assemble(h_big, w_big)
            if ass is not None:
                used_dim = (h_big, w_big)
                used_corner = corner
                break
        if ass is not None:
            break

    assert not (ass is None or used_dim is None or used_corner is None)

    h_big, w_big = used_dim
    tr, tc = top_left_for(used_corner, h_big, w_big)

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + h_big, tc : tc + w_big] = ass
    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _build_interior(interior: np.ndarray, rect_col: int, interior_cols: list[int], rng) -> bool:
    i_h, i_w = interior.shape
    target_fill = rng.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * i_h * i_w))
    blob_occupied = np.zeros((i_h, i_w), dtype=bool)

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
            sr, sc = rng.randint(0, i_h - 1), rng.randint(0, i_w - 1)
            if blob_occupied[sr, sc]:
                attempts += 1
                continue
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = sr + dr, sc + dc
                    if 0 <= nr < i_h and 0 <= nc < i_w and blob_occupied[nr, nc]:
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
            rng.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob:
                        continue
                    if nr < 0 or nr >= i_h or nc < 0 or nc >= i_w:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < i_h and 0 <= ac < i_w:
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

    fill_ratio = float(np.mean(interior != int(rect_col))) if interior.size else 0.0
    return 0.30 <= fill_ratio <= 0.90


def _rand_splits(total: int, rng, min_s: int = 2, max_s: int = 7) -> list[int] | None:
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


def _pick_marker_position(corner: str, gh: int, gw: int, rh: int, rw: int, rng) -> tuple[int, int] | None:
    top_max = gh // 2 - 1
    bottom_min = gh // 2 + (gh % 2)
    left_max = gw // 2 - 1
    right_min = gw // 2 + (gw % 2)

    if corner == "TL":
        mr_min, mr_max = 0, min(gh - rh, top_max)
        mc_min, mc_max = 0, min(gw - rw, left_max)
    elif corner == "TR":
        mr_min, mr_max = 0, min(gh - rh, top_max)
        mc_min, mc_max = max(rw - 1, right_min), gw - 1
    elif corner == "BL":
        mr_min, mr_max = max(rh - 1, bottom_min), gh - 1
        mc_min, mc_max = 0, min(gw - rw, left_max)
    else:
        mr_min, mr_max = max(rh - 1, bottom_min), gh - 1
        mc_min, mc_max = max(rw - 1, right_min), gw - 1

    if mr_min > mr_max or mc_min > mc_max:
        return None
    return rng.randint(mr_min, mr_max), rng.randint(mc_min, mc_max)


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    gh = _r.randint(10, 26)
    gw = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_col = palette[2]
    n_interior = _r.randint(1, 3)
    interior_cols = palette[3 : 3 + n_interior]

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    min_side = 5
    max_side = min(22, min(gh, gw) - 1)
    if max_side < min_side:
        return None
    rh = _r.randint(min_side, max_side)
    rw = _r.randint(min_side, max_side)
    if _r.random() < 0.5 and max_side >= 8:
        rh = _r.randint(max(6, min_side), max_side)
        rw = _r.randint(max(6, min_side), max_side)

    marker_pos = _pick_marker_position(corner, gh, gw, rh, rw, _r)
    if marker_pos is None:
        return None
    mr, mc = marker_pos

    rect = np.full((rh, rw), rect_col, dtype=int)
    i_h, i_w = rh - 2, rw - 2
    if i_h < 1 or i_w < 1:
        return None

    interior = np.full((i_h, i_w), rect_col, dtype=int)
    if not _build_interior(interior, rect_col, interior_cols, _r):
        return None
    rect[1:-1, 1:-1] = interior

    row_segs = _rand_splits(rh, _r)
    col_segs = _rand_splits(rw, _r)
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
        ri = [c[0] for c in cells]
        ci = [c[1] for c in cells]
        return min(ri), min(ci), max(ri), max(ci)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
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

    for pr, pc, ph, pw in pieces:
        if ph < 2 or pw < 2 or ph > 7 or pw > 7:
            return None
        piece = rect[pr : pr + ph, pc : pc + pw]
        nrc = int(np.sum(piece != rect_col))
        fill = nrc / (ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        if not any(np.any(edge != rect_col) for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]):
            return None
        if pr == 0 and not np.all(piece[0, :] == rect_col):
            return None
        if pr + ph == rh and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == rw and not np.all(piece[:, -1] == rect_col):
            return None

    canvas = np.full((gh, gw), bg, dtype=int)
    canvas[mr, mc] = marker_col

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]
    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            loc = random_free_location(canvas, pa, transparent=bg, padding=1)
        except ValueError:
            return None
        if loc is None:
            return None
        paste(canvas, pa, loc[0], loc[1])

    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - rw + 1
    elif corner == "BL":
        tr, tc = mr - rh + 1, mc
    else:
        tr, tc = mr - rh + 1, mc - rw + 1

    expected_out = np.full((gh, gw), bg, dtype=int)
    expected_out[tr : tr + rh, tc : tc + rw] = rect

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
