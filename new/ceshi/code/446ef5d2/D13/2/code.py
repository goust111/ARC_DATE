# Variant of 446ef5d2
# Dimension: D13 — 标记角类型的确定方式 (meta)
# Change: L 标记的缺失角直接编码了矩形的角类型（TL/TR/BL/BR），解题者通过观察 L 的朝向即可确定... → 多个候选标记，只有一个有效

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import paste, random_free_location


"""
Design analysis for the D13 variant
===================================

a) 具体化与行为追踪
   - 原题输入里只有 1 个 3 像素 L 标记。它的缺失格既告诉你“矩形哪个角被钉住”，也唯一决定输出位置。
   - 这个变体里输入会出现 2-3 个不同颜色的 L 标记。每个 L 都贴着某个散块的一个角，看起来都像候选锚点。
   - 只有 1 个 L 贴住的那块，真的是最终大矩形在该角上的角块；其他 L 贴住的是普通散块。若把普通散块强行当角块固定到对应角位，
     就会违反“外框边必须全是 rect_col”的约束，装配失败。
   - 小例子：假设真角为 TL。原逻辑读取唯一 L 的朝向后，直接按 TL 去拼。现在若额外放一个 BR 朝向的假 L，并让它贴着某个中间块的 BR 角，
     则旧逻辑若只按“哪个 L 能提供朝向”来处理，会把这个中间块也当成 BR 角块尝试；但它的下边或右边并非全 rect_col，因此不可能出现在
     最终矩形的 BR 角。新逻辑必须逐个候选 L 尝试，并验证“被该 L 钉住的散块真的能作为该角块固定”。

b) 复杂度校准
   - 候选标记数控制在 2-3 个，既能形成消歧，又不会把画面塞满。
   - 每个候选 L 都贴着某个块角，避免“只有一个 L 看起来像真的”而退化成局部几何判断。
   - 假标记不靠颜色编号、排序等不可见因素区分；唯一差异来自它所锚定的那块是否真的满足矩形角块约束。

c) 隐式假设审查
   - 原 solver 的关键旧前提：只要拿到一个 marker，就能在对应位置输出装配结果。原因是旧代码里 anchor piece 只是“可选预放置”；
     如果 marker 附近根本没有锚定散块，回溯仍可能在局部坐标里把矩形拼出来，因此“假 marker”也会成功。
   - 变体下这会直接出错，因为多个 marker 里很多都能仅凭几何 fit 成功。
   - 因此新 solver 必须把“marker 的缺失格被某个散块占据，且该格正好是那块在对应朝向上的角”升级成强制条件；并且这块必须成功预放置到
     最终矩形的对应角位，否则该 marker 无效。

d) 放置合法性
   - 真标记：仍按原题规则决定最终矩形位置；只是把真实角块直接放在该 marker 旁边，形成显式锚定。
   - 假标记：各自和一个非对应角块做成同样的“L + 块角”局部结构，但这些块在该朝向上不具备双外框边，因此无法成为真角块。
   - 所有散块/标记都保持不重叠；不同对象之间保留至少 1 个背景单元的间隔，只有 marker 与它自己的锚定块允许贴边。

e) 视觉可观察性
   - 人类能直接看到“有多个 L 候选，而且每个都像在指一个块角”。这和原题单 L 的视觉差异明显。
   - 真规律可从少量样例中观察到：只有某一个候选 L 对应的散块，能与其他块一起拼成满足边缘颜色约束的完整矩形。

f) 可视化整洁度
   - 候选 L 使用不同颜色，且每个 L 只占 3 个像素。
   - 假标记也贴着块角，但不叠到块内部；因此局部结构清楚，不会出现颜色覆盖混乱。

g) 颜色变化可推断性
   - 标记颜色只承担“候选身份”的作用，不参与排序或映射。
   - 真/假 marker 的区别完全来自结构可装配性，不来自颜色值大小。

h) 新 solver 的输入假设
   - 至少存在 1 个候选 marker 颜色，其全局出现次数恰为 3，且形成 2x2 缺 1 的 L。
   - 所有 marker 颜色都不用于散块像素。
   - 真 marker 的缺失格位于某个散块角上；假 marker 也通常如此，但对应的块不是那个朝向的真角块。
   - generator 会显式保证这些条件，并验证恰好只有 1 个 marker 能成功求解。
"""


CORNER_NAMES = ("TL", "TR", "BL", "BR")


def _marker_cells(mr: int, mc: int, corner: str) -> list[tuple[int, int]]:
    if corner == "TL":
        return [(mr - 1, mc - 1), (mr - 1, mc), (mr, mc - 1)]
    if corner == "TR":
        return [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
    if corner == "BL":
        return [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
    return [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]


def _piece_top_left_for_corner(mr: int, mc: int, h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "TL":
        return mr, mc
    if corner == "TR":
        return mr, mc - w + 1
    if corner == "BL":
        return mr - h + 1, mc
    return mr - h + 1, mc - w + 1


def _rect_top_left(mr: int, mc: int, h: int, w: int, corner: str) -> tuple[int, int]:
    return _piece_top_left_for_corner(mr, mc, h, w, corner)


def _corner_from_missing(missing: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _find_candidate_markers(g: np.ndarray, bg: int) -> list[dict]:
    cnt = Counter(map(int, g.flatten()))
    out = []
    for col in sorted(c for c, n in cnt.items() if int(c) != int(bg) and int(n) == 3):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(col))]
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if (r1 - r0) != 1 or (c1 - c0) != 1:
            continue
        all4 = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
        miss = list(all4 - set(pts))
        if len(miss) != 1:
            continue
        missing = tuple(map(int, miss[0]))
        out.append(
            {
                "color": int(col),
                "missing": missing,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "corner": _corner_from_missing(missing, (int(r0), int(c0), int(r1), int(c1))),
            }
        )
    return out


def _decompose_rectangles(g: np.ndarray, bg: int, marker_colors: set[int]) -> list[tuple[int, int, int, int, np.ndarray]]:
    GH, GW = g.shape
    occ = (g != int(bg))
    for col in marker_colors:
        occ &= g != int(col)
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


def _fits_anchor(H: int, W: int, mr: int, mc: int, corner: str, GH: int, GW: int) -> bool:
    tr, tc = _rect_top_left(mr, mc, H, W, corner)
    return 0 <= tr and 0 <= tc and (tr + H) <= GH and (tc + W) <= GW


def _attempt_candidate(
    GH: int,
    GW: int,
    rects: list[tuple[int, int, int, int, np.ndarray]],
    rect_col: int,
    total_area: int,
    cand: dict,
) -> dict | None:
    mr, mc = cand["missing"]
    corner = cand["corner"]

    anchor_idx = None
    for i, (rr, cc, h, w, _a) in enumerate(rects):
        if rr <= mr < rr + h and cc <= mc < cc + w:
            anchor_idx = i
            break
    if anchor_idx is None:
        return None

    rr, cc, h0, w0, _ = rects[anchor_idx]
    corners = {
        (rr, cc): "TL",
        (rr, cc + w0 - 1): "TR",
        (rr + h0 - 1, cc): "BL",
        (rr + h0 - 1, cc + w0 - 1): "BR",
    }
    if corners.get((mr, mc)) != corner:
        return None

    pieces = [a for _r, _c, _h, _w, a in rects]
    ph = [int(p.shape[0]) for p in pieces]
    pw = [int(p.shape[1]) for p in pieces]

    dims = []
    for hh in range(1, int(total_area**0.5) + 1):
        if total_area % hh == 0:
            dims.append((hh, total_area // hh))
            if hh * hh != total_area:
                dims.append((total_area // hh, hh))
    dims = [(int(H), int(W)) for H, W in dims if _fits_anchor(int(H), int(W), mr, mc, corner, GH, GW)]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    def compat(v1, v2):
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        return bool((v1[m] == v2[m]).all())

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
        ar0, ac0 = _piece_top_left_for_corner(0 if corner in ("TL", "TR") else H - 1, 0 if corner in ("TL", "BL") else W - 1, ah, aw, corner)
        if not can_place(anchor_idx, ar0, ac0):
            return None
        place(anchor_idx, ar0, ac0)

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

    for H, W in dims:
        ass = assemble(H, W)
        if ass is None:
            continue
        tr, tc = _rect_top_left(mr, mc, H, W, corner)
        return {
            "marker_color": int(cand["color"]),
            "missing": (int(mr), int(mc)),
            "corner": corner,
            "dims": (int(H), int(W)),
            "top_left": (int(tr), int(tc)),
            "assembly": ass,
        }
    return None


def _solve_all(grid: np.ndarray) -> list[dict]:
    g = np.array(grid, dtype=int)
    GH, GW = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])

    candidates = _find_candidate_markers(g, bg)
    assert candidates

    marker_colors = {int(c["color"]) for c in candidates}
    rects = _decompose_rectangles(g, bg, marker_colors)
    assert rects

    total_area = int(sum(int(h) * int(w) for _r, _c, h, w, _a in rects))
    allpix = []
    for _r, _c, _h, _w, a in rects:
        allpix.extend(map(int, a.flatten()))
    rect_col = int(Counter(allpix).most_common(1)[0][0])

    sols = []
    for cand in candidates:
        sol = _attempt_candidate(GH, GW, rects, rect_col, total_area, cand)
        if sol is None:
            continue
        H, W = sol["dims"]
        tr, tc = sol["top_left"]
        out = np.full((GH, GW), int(bg), dtype=int)
        out[tr : tr + H, tc : tc + W] = sol["assembly"]
        sol["output"] = out
        sols.append(sol)
    return sols


def main(grid: np.ndarray) -> np.ndarray:
    sols = _solve_all(grid)
    assert sols

    # Generator guarantees uniqueness. If multiple candidates somehow lead to
    # the same reconstruction, returning that reconstruction is harmless.
    sigs = {tuple(map(int, s["output"].flatten())) for s in sols}
    assert len(sigs) == 1 or len(sols) == 1
    return sols[0]["output"]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


def _rand_splits(total: int, min_s: int = 2, max_s: int = 7) -> list[int] | None:
    eff_max = min(max_s, max(min_s, random.randint(min_s, max_s)))
    segs = []
    rem = int(total)
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
            s = random.randint(min_s, hi)
            segs.append(s)
            rem -= s
    return segs


def _piece_supports_corner(piece_info: tuple[int, int, int, int], corner: str, RH: int, RW: int) -> bool:
    pr, pc, ph, pw = piece_info
    if corner == "TL":
        return pr == 0 and pc == 0
    if corner == "TR":
        return pr == 0 and (pc + pw) == RW
    if corner == "BL":
        return (pr + ph) == RH and pc == 0
    return (pr + ph) == RH and (pc + pw) == RW


def _make_piece_marker_patch(piece: np.ndarray, corner: str, marker_col: int, bg: int) -> np.ndarray:
    ph, pw = piece.shape
    marker_coords = {
        "TL": [(-1, -1), (-1, 0), (0, -1)],
        "TR": [(-1, pw - 1), (-1, pw), (0, pw)],
        "BL": [(ph - 1, -1), (ph, -1), (ph, 0)],
        "BR": [(ph - 1, pw), (ph, pw - 1), (ph, pw)],
    }[corner]
    coords = [(0, 0), (ph - 1, pw - 1)] + marker_coords
    min_r = min(r for r, _ in coords)
    max_r = max(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    max_c = max(c for _, c in coords)
    patch = np.full((max_r - min_r + 1, max_c - min_c + 1), int(bg), dtype=int)
    pr0 = -min_r
    pc0 = -min_c
    patch[pr0 : pr0 + ph, pc0 : pc0 + pw] = piece
    for rr, cc in marker_coords:
        patch[rr - min_r, cc - min_c] = int(marker_col)
    return patch


def _gen_attempt() -> np.ndarray | None:
    GH = random.randint(10, 26)
    GW = random.randint(10, 26)

    n_markers = random.randint(2, 3)
    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    rect_col = palette[1]
    marker_cols = palette[2 : 2 + n_markers]
    n_interior = random.randint(1, min(3, 10 - (2 + n_markers)))
    interior_cols = palette[2 + n_markers : 2 + n_markers + n_interior]

    corner = random.choice(list(CORNER_NAMES))

    min_side, max_side = 5, min(22, min(GH, GW) - 3)
    if max_side < min_side:
        return None
    RH = random.randint(min_side, max_side)
    RW = random.randint(min_side, max_side)
    if random.random() < 0.5 and max_side >= 8:
        RH = random.randint(max(6, min_side), max_side)
        RW = random.randint(max(6, min_side), max_side)

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
    mr = random.randint(mr_min, mr_max)
    mc = random.randint(mc_min, mc_max)

    rect = np.full((RH, RW), rect_col, dtype=int)
    iH, iW = RH - 2, RW - 2
    if iH < 1 or iW < 1:
        return None
    interior = np.full((iH, iW), rect_col, dtype=int)

    target_fill = random.uniform(0.35, 0.85)
    target_pixels = max(1, int(target_fill * iH * iW))
    blob_occupied = np.zeros((iH, iW), dtype=bool)

    for ci, ic in enumerate(interior_cols):
        if ci == len(interior_cols) - 1:
            budget = target_pixels - int(blob_occupied.sum())
        else:
            budget = max(1, target_pixels // len(interior_cols))
        if budget <= 0:
            budget = 1

        seed = None
        for _ in range(200):
            sr, sc = random.randint(0, iH - 1), random.randint(0, iW - 1)
            if blob_occupied[sr, sc]:
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
        if seed is None:
            continue

        blob = {seed}
        frontier = [seed]
        while len(blob) < budget and frontier:
            random.shuffle(frontier)
            nf = []
            for fr, fc in frontier:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = fr + dr, fc + dc
                    if (nr, nc) in blob:
                        continue
                    if nr < 0 or nr >= iH or nc < 0 or nc >= iW:
                        continue
                    if blob_occupied[nr, nc]:
                        continue
                    conflict = False
                    for dr2 in (-1, 0, 1):
                        for dc2 in (-1, 0, 1):
                            if dr2 == 0 and dc2 == 0:
                                continue
                            ar, ac = nr + dr2, nc + dc2
                            if 0 <= ar < iH and 0 <= ac < iW and blob_occupied[ar, ac] and (ar, ac) not in blob:
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

    def piece_bbox(pid: int) -> tuple[int, int, int, int] | None:
        cells = [(i, j) for i in range(nr) for j in range(nc_grid) if cell_id[i][j] == pid]
        if not cells:
            return None
        ris = [r for r, _ in cells]
        cis = [c for _, c in cells]
        return min(ris), min(cis), max(ris), max(cis)

    def pixel_size(r0i: int, c0i: int, r1i: int, c1i: int) -> tuple[int, int]:
        h = sum(row_segs[i] for i in range(r0i, r1i + 1))
        w = sum(col_segs[j] for j in range(c0i, c1i + 1))
        return h, w

    target_pieces = random.randint(max(n_markers + 1, 4), min(12, n_cells))
    merges = []
    for i in range(nr):
        for j in range(nc_grid):
            if j + 1 < nc_grid:
                merges.append(("h", i, j))
            if i + 1 < nr:
                merges.append(("v", i, j))
    random.shuffle(merges)

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
        bb1 = piece_bbox(pid1)
        bb2 = piece_bbox(pid2)
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
            bb = piece_bbox(pid)
            if bb is None:
                return None
            r0i, c0i, r1i, c1i = bb
            pr = row_starts[r0i]
            pc = col_starts[c0i]
            ph = sum(row_segs[ii] for ii in range(r0i, r1i + 1))
            pw = sum(col_segs[jj] for jj in range(c0i, c1i + 1))
            pieces.append((pr, pc, ph, pw))

    if len(pieces) < max(4, n_markers) or len(pieces) > 12:
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
        if pr + ph == RH and not np.all(piece[-1, :] == rect_col):
            return None
        if pc == 0 and not np.all(piece[:, 0] == rect_col):
            return None
        if pc + pw == RW and not np.all(piece[:, -1] == rect_col):
            return None

    anchor_cell = {
        "TL": (0, 0),
        "TR": (0, RW - 1),
        "BL": (RH - 1, 0),
        "BR": (RH - 1, RW - 1),
    }[corner]
    true_anchor_idx = None
    for i, (pr, pc, ph, pw) in enumerate(pieces):
        if pr <= anchor_cell[0] < pr + ph and pc <= anchor_cell[1] < pc + pw:
            true_anchor_idx = i
            break
    if true_anchor_idx is None:
        return None
    if not _piece_supports_corner(pieces[true_anchor_idx], corner, RH, RW):
        return None

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    canvas = np.full((GH, GW), bg, dtype=int)

    actual_marker_cells = _marker_cells(mr, mc, corner)
    for cr, cc in actual_marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None
        canvas[cr, cc] = marker_cols[0]

    tr, tc = _rect_top_left(mr, mc, RH, RW, corner)
    apr, apc, aph, apw = pieces[true_anchor_idx]
    anchor_abs_r = tr + apr
    anchor_abs_c = tc + apc
    if anchor_abs_r < 0 or anchor_abs_c < 0 or anchor_abs_r + aph > GH or anchor_abs_c + apw > GW:
        return None
    paste(canvas, piece_arrays[true_anchor_idx], anchor_abs_r, anchor_abs_c)

    placed_piece_indices = {true_anchor_idx}

    for decoy_col in marker_cols[1:]:
        candidate_pairs = []
        for idx, piece_info in enumerate(pieces):
            if idx in placed_piece_indices:
                continue
            for decoy_corner in CORNER_NAMES:
                if not _piece_supports_corner(piece_info, decoy_corner, RH, RW):
                    candidate_pairs.append((idx, decoy_corner))
        random.shuffle(candidate_pairs)
        placed = False
        for idx, decoy_corner in candidate_pairs:
            patch = _make_piece_marker_patch(piece_arrays[idx], decoy_corner, decoy_col, bg)
            try:
                r0, c0 = random_free_location(canvas, patch, transparent=bg, padding=1, border=1)
            except ValueError:
                continue
            paste(canvas, patch, r0, c0, transparent=bg)
            placed_piece_indices.add(idx)
            placed = True
            break
        if not placed:
            return None

    order = [i for i in range(len(piece_arrays)) if i not in placed_piece_indices]
    random.shuffle(order)
    for idx in order:
        pa = piece_arrays[idx]
        try:
            r0, c0 = random_free_location(canvas, pa, transparent=bg, padding=1, border=1)
        except ValueError:
            return None
        paste(canvas, pa, r0, c0)

    counts = Counter(map(int, canvas.flatten()))
    for col in set(interior_cols + [rect_col]):
        if counts[int(col)] == 3:
            return None

    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        sols = _solve_all(canvas)
    except Exception:
        return None
    if len(sols) != 1:
        return None
    if not np.array_equal(sols[0]["output"], expected_out):
        return None
    if len(_find_candidate_markers(canvas, bg)) != n_markers:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
