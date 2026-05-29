# Variant of 446ef5d2
# Dimension: D10 — 矩形外边缘约束 (meta)
# Change: 组装矩形的四条外边必须全部由 rect_col 组成（纯色边框）... → 外边缘允许非 rect_col 像素

from __future__ import annotations

from collections import Counter

import numpy as np

from arc_utils import paste, random_free_location

"""
design_analysis:
  variant_core:
    original_rule: assembled rectangle 的四条外边必须全部为 rect_col，因此 solver 可把“全 rect_col 的边”直接当作外边线索。
    new_rule: 外边允许出现非 rect_col 像素；因此“纯 rect_col 边 = 朝外”这一判断失效。
  concretization:
    visible_change:
      - 输出中矩形外轮廓不再是整圈纯 rect_col。
      - 外边会保留稀疏彩色像素，这些彩色像素和内部拼缝上的彩色签名一起成为新的定位线索。
    support_change_needed:
      - 原生成器允许一般矩形拼法，单条边可能接多块碎片；在失去纯色外框后，这会让“哪条边应该朝外”过于难判。
      - 为了让 D10 可观察且逻辑稳定，分块改为规则网格切分（2-4 行 × 2-4 列，无合并），这样每条内边只对应唯一一条对边。
      - 这不是额外任务规则，而是保证 D10 可实现的最小结构性适配。
  behavior_trace:
    example:
      - 原逻辑: 某碎片若放在最上边，则要求 its top edge 全为 rect_col；否则立即剪枝。
      - 新逻辑: 某碎片若放在最上边，则要求 its top edge 在所有其它碎片的 bottom edges 中没有任何兼容对象；反之若它有兼容对边，就不能放到最上边。
      - 差异: “外边判定”从颜色纯度约束，变成了边签名的匹配存在性约束。
  complexity_calibration:
    - 仍然只有一个大矩形、一个 L 标记、4-12 个矩形碎片。
    - 边签名使用 3 个 accent colors，在每条边的两个固定位置着色；每个样例中同长度的行高/列宽互不相同，避免签名空间被同长度大量复用。
    - 这样人类可以直接看出：有些彩色边能和别的边对上，有些彩色边没有匹配对象，因此会成为外框。
  implicit_assumption_review:
    original_main_dependency:
      - boundary_ok 的四个 `np.all(edge == rect_col)` 直接依赖旧 D10；在新变体中不成立，必须删除。
      - rect_col 仍需是 piece pixels 中出现次数最多的颜色；生成器通过稀疏签名和大面积留白保持这一点。
      - anchor 预放置依赖“marker 缺口正好对应某块碎片的真实角”；该前提保留。
    new_main_assumptions:
      - 若一条边在所有对向边中没有兼容对象，则它必须朝外。
      - 若一条边有兼容对象，则它不能朝外。
      - 每条内边至多对应一个真实邻居；生成器用规则网格切分和唯一边签名保证。
  placement_legality:
    - 碎片仍以 padding=1 散落，互不重叠、不越界。
    - marker 仍是 2x2 缺一格的 L，且缺口就是 assembled rectangle 的角。
    - assembled rectangle 仍完全位于原网格内。
  visual_observability:
    - 输入里每块碎片边缘都带有少量彩色“插针”，输出里这些插针不仅出现在内部拼缝，也会出现在最终外轮廓。
    - 这与原题“外框整圈单色”的视觉效果明显不同，2-3 组样例即可观察到。
  visual_cleanliness:
    - 颜色只落在边的两个固定非角点位置，避免角点多重覆盖。
    - 额外内部装饰只放在 piece 内部，不干扰边签名。
"""


def _find_marker(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int], tuple[int, int, int, int]]:
    cnt = Counter(map(int, g.flatten()))
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
                return int(col), miss[0], (r0, c0, r1, c1)
    raise AssertionError("marker not found")


def _corner_from_missing(missing: tuple[int, int], mb: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = mb
    if missing == (r0, c0):
        return "BR"
    if missing == (r0, c1):
        return "BL"
    if missing == (r1, c0):
        return "TR"
    return "TL"


def _decompose_rects(g: np.ndarray, bg: int, mark: int) -> list[tuple[int, int, int, int, np.ndarray]]:
    occ = (g != int(bg)) & (g != int(mark))
    used = np.zeros_like(occ, dtype=bool)
    rects: list[tuple[int, int, int, int, np.ndarray]] = []
    gh, gw = g.shape
    while True:
        idx = np.argwhere(occ & (~used))
        if idx.size == 0:
            return rects
        rr, cc = map(int, idx[0])
        w = 0
        while cc + w < gw and bool(occ[rr, cc + w]) and (not bool(used[rr, cc + w])):
            w += 1
        h = 0
        while rr + h < gh and bool((occ[rr + h, cc : cc + w] & (~used[rr + h, cc : cc + w])).all()):
            h += 1
        used[rr : rr + h, cc : cc + w] = True
        rects.append((rr, cc, h, w, g[rr : rr + h, cc : cc + w].copy()))


def _compat(v1: np.ndarray, v2: np.ndarray, rect_col: int) -> bool:
    a = np.array(v1, dtype=int)
    b = np.array(v2, dtype=int)
    m = (a != int(rect_col)) & (b != int(rect_col))
    return bool((a[m] == b[m]).all())


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    D10 variant:
    - The assembled rectangle's outer border may contain non-rect_col pixels.
    - Therefore outer-border detection cannot rely on `edge is all rect_col`.
    - Instead, an edge is treated as forced-outer iff it has no compatible opposite edge
      among all other pieces. The generator guarantees this property.
    """

    g = np.array(grid, dtype=int)
    gh, gw = g.shape

    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    mark, missing, mb = _find_marker(g, bg)
    corner = _corner_from_missing(missing, mb)

    rects = _decompose_rects(g, bg, mark)
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

    def fits(h: int, w: int) -> bool:
        if corner == "TL":
            tr, tc = mr, mc
        elif corner == "TR":
            tr, tc = mr, mc - w + 1
        elif corner == "BL":
            tr, tc = mr - h + 1, mc
        else:
            tr, tc = mr - h + 1, mc - w + 1
        return 0 <= tr and 0 <= tc and (tr + h) <= gh and (tc + w) <= gw

    dims = [(int(h), int(w)) for h, w in dims if fits(int(h), int(w))]
    dims = sorted(dims, key=lambda t: (-t[0] * t[1], -t[0], -t[1]))

    top_matches: list[list[int]] = [[] for _ in pieces]
    bottom_matches: list[list[int]] = [[] for _ in pieces]
    left_matches: list[list[int]] = [[] for _ in pieces]
    right_matches: list[list[int]] = [[] for _ in pieces]
    for i, a in enumerate(pieces):
        for j, b in enumerate(pieces):
            if i == j:
                continue
            if pw[i] == pw[j]:
                if _compat(a[0, :], b[-1, :], rect_col):
                    top_matches[i].append(j)
                if _compat(a[-1, :], b[0, :], rect_col):
                    bottom_matches[i].append(j)
            if ph[i] == ph[j]:
                if _compat(a[:, 0], b[:, -1], rect_col):
                    left_matches[i].append(j)
                if _compat(a[:, -1], b[:, 0], rect_col):
                    right_matches[i].append(j)

    must_top_outer = [len(v) == 0 for v in top_matches]
    must_bottom_outer = [len(v) == 0 for v in bottom_matches]
    must_left_outer = [len(v) == 0 for v in left_matches]
    must_right_outer = [len(v) == 0 for v in right_matches]
    outer_score = [
        int(must_top_outer[i]) + int(must_bottom_outer[i]) + int(must_left_outer[i]) + int(must_right_outer[i])
        for i in range(len(pieces))
    ]

    def assemble(h: int, w: int) -> np.ndarray | None:
        occ = np.full((h, w), -1, dtype=int)
        usedp = [False] * len(pieces)

        def boundary_ok(i: int, r: int, c: int) -> bool:
            ah, aw = ph[i], pw[i]
            if (r == 0) != must_top_outer[i]:
                return False
            if ((r + ah) == h) != must_bottom_outer[i]:
                return False
            if (c == 0) != must_left_outer[i]:
                return False
            if ((c + aw) == w) != must_right_outer[i]:
                return False
            return True

        def neighbor_ok(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            ah, aw = a.shape
            if c > 0:
                left = occ[r : r + ah, c - 1]
                m = left != -1
                if m.any() and (not _compat(left[m], a[:, 0][m], rect_col)):
                    return False
            if (c + aw) < w:
                right = occ[r : r + ah, c + aw]
                m = right != -1
                if m.any() and (not _compat(right[m], a[:, -1][m], rect_col)):
                    return False
            if r > 0:
                up = occ[r - 1, c : c + aw]
                m = up != -1
                if m.any() and (not _compat(up[m], a[0, :][m], rect_col)):
                    return False
            if (r + ah) < h:
                down = occ[r + ah, c : c + aw]
                m = down != -1
                if m.any() and (not _compat(down[m], a[-1, :][m], rect_col)):
                    return False
            return True

        def can_place(i: int, r: int, c: int) -> bool:
            a = pieces[i]
            ah, aw = a.shape
            if (r + ah) > h or (c + aw) > w:
                return False
            if (occ[r : r + ah, c : c + aw] != -1).any():
                return False
            if not boundary_ok(i, r, c):
                return False
            if not neighbor_ok(i, r, c):
                return False
            return True

        def place(i: int, r: int, c: int) -> None:
            a = pieces[i]
            ah, aw = a.shape
            occ[r : r + ah, c : c + aw] = a
            usedp[i] = True

        def unplace(i: int, r: int, c: int) -> None:
            a = pieces[i]
            ah, aw = a.shape
            occ[r : r + ah, c : c + aw] = -1
            usedp[i] = False

        if anchor_idx is not None:
            rr, cc, ah, aw, _ = rects[anchor_idx]
            corners = {
                (rr, cc): "TL",
                (rr, cc + aw - 1): "TR",
                (rr + ah - 1, cc): "BL",
                (rr + ah - 1, cc + aw - 1): "BR",
            }
            if (mr, mc) in corners and corners[(mr, mc)] == corner:
                if corner == "TL":
                    ar0, ac0 = 0, 0
                elif corner == "TR":
                    ar0, ac0 = 0, w - aw
                elif corner == "BL":
                    ar0, ac0 = h - ah, 0
                else:
                    ar0, ac0 = h - ah, w - aw
                if can_place(anchor_idx, ar0, ac0):
                    place(anchor_idx, ar0, ac0)

        order = sorted(
            range(len(pieces)),
            key=lambda i: (-outer_score[i], -ph[i] * pw[i], -ph[i], -pw[i], i),
        )

        def next_empty() -> tuple[int, int] | None:
            idx = np.argwhere(occ == -1)
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
        return occ if solved else None

    ass = None
    used_dim = None
    for h, w in dims:
        ass = assemble(h, w)
        if ass is not None:
            used_dim = (h, w)
            break

    assert ass is not None and used_dim is not None

    h, w = used_dim
    if corner == "TL":
        tr, tc = mr, mc
    elif corner == "TR":
        tr, tc = mr, mc - w + 1
    elif corner == "BL":
        tr, tc = mr - h + 1, mc
    else:
        tr, tc = mr - h + 1, mc - w + 1

    out = np.full((gh, gw), int(bg), dtype=int)
    out[tr : tr + h, tc : tc + w] = ass
    return out


def _rand_partition(total_max: int, count: int, lo: int, hi: int, rng) -> list[int] | None:
    vals = list(range(lo, hi + 1))
    if count > len(vals):
        return None
    for _ in range(60):
        segs = rng.sample(vals, count)
        rng.shuffle(segs)
        if sum(segs) <= total_max:
            return segs
    return None


def _sig_positions(length: int) -> tuple[int, int]:
    return 1, length - 2


def _all_codes(colors: list[int]) -> list[tuple[int, int]]:
    return [(a, b) for a in colors for b in colors]


def _paint_h_edge(arr: np.ndarray, row: int, c0: int, width: int, code: tuple[int, int]) -> None:
    p0, p1 = _sig_positions(width)
    arr[row, c0 + p0] = int(code[0])
    arr[row, c0 + p1] = int(code[1])


def _paint_v_edge(arr: np.ndarray, r0: int, col: int, height: int, code: tuple[int, int]) -> None:
    p0, p1 = _sig_positions(height)
    arr[r0 + p0, col] = int(code[0])
    arr[r0 + p1, col] = int(code[1])


def _place_piece_interior_marks(rect: np.ndarray, piece: tuple[int, int, int, int], accent_colors: list[int], rng) -> None:
    pr, pc, ph, pw = piece
    if ph < 4 or pw < 4:
        return
    if rng.random() < 0.45:
        return
    color = int(rng.choice(accent_colors))
    rr = pr + ph // 2
    cc = pc + pw // 2
    rect[rr, cc] = color
    if ph >= 6 and rng.random() < 0.5:
        rect[rr - 1, cc] = color
    if pw >= 6 and rng.random() < 0.5:
        rect[rr, cc - 1] = color


def _gen_attempt() -> np.ndarray | None:
    import random as _r

    gh = _r.randint(10, 26)
    gw = _r.randint(10, 26)

    palette = list(range(10))
    _r.shuffle(palette)
    bg = int(palette[0])
    rect_col = int(palette[1])
    marker_col = int(palette[2])
    accent_colors = [int(x) for x in palette[3:6]]

    valid_grids = [(r, c) for r in range(2, 5) for c in range(2, 5) if 4 <= r * c <= 12]
    n_rows, n_cols = _r.choice(valid_grids)

    row_segs = _rand_partition(gh - 2, n_rows, 4, 7, _r)
    col_segs = _rand_partition(gw - 2, n_cols, 4, 7, _r)
    if row_segs is None or col_segs is None:
        return None

    rh = sum(row_segs)
    rw = sum(col_segs)
    if rh > gh - 2 or rw > gw - 2:
        return None

    corner = _r.choice(["TL", "TR", "BL", "BR"])

    if corner == "TL":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = 2, gw - rw
    elif corner == "TR":
        mr_min, mr_max = 2, gh - rh
        mc_min, mc_max = rw - 1, gw - 3
    elif corner == "BL":
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = 2, gw - rw
    else:
        mr_min, mr_max = rh - 1, gh - 3
        mc_min, mc_max = rw - 1, gw - 3
    if mr_min > mr_max or mc_min > mc_max:
        return None

    mr = _r.randint(mr_min, mr_max)
    mc = _r.randint(mc_min, mc_max)

    rect = np.full((rh, rw), rect_col, dtype=int)

    row_starts: list[int] = []
    s = 0
    for seg in row_segs:
        row_starts.append(s)
        s += seg
    col_starts: list[int] = []
    s = 0
    for seg in col_segs:
        col_starts.append(s)
        s += seg

    pieces: list[tuple[int, int, int, int]] = []
    for i, ph in enumerate(row_segs):
        for j, pw in enumerate(col_segs):
            pieces.append((row_starts[i], col_starts[j], ph, pw))

    codes = _all_codes(accent_colors)

    for i, ph in enumerate(row_segs):
        if ph < 4:
            return None
        need = n_cols + 1
        if need > len(codes):
            return None
        chosen = _r.sample(codes, need)
        left_ext = chosen[0]
        right_ext = chosen[-1]
        seam_codes = chosen[1:-1]
        r0 = row_starts[i]
        for j, pw in enumerate(col_segs):
            c0 = col_starts[j]
            if j == 0:
                _paint_v_edge(rect, r0, c0, ph, left_ext)
            else:
                _paint_v_edge(rect, r0, c0, ph, seam_codes[j - 1])
            if j == n_cols - 1:
                _paint_v_edge(rect, r0, c0 + pw - 1, ph, right_ext)
            else:
                _paint_v_edge(rect, r0, c0 + pw - 1, ph, seam_codes[j])

    for j, pw in enumerate(col_segs):
        if pw < 4:
            return None
        need = n_rows + 1
        if need > len(codes):
            return None
        chosen = _r.sample(codes, need)
        top_ext = chosen[0]
        bottom_ext = chosen[-1]
        seam_codes = chosen[1:-1]
        c0 = col_starts[j]
        for i, ph in enumerate(row_segs):
            r0 = row_starts[i]
            if i == 0:
                _paint_h_edge(rect, r0, c0, pw, top_ext)
            else:
                _paint_h_edge(rect, r0, c0, pw, seam_codes[i - 1])
            if i == n_rows - 1:
                _paint_h_edge(rect, r0 + ph - 1, c0, pw, bottom_ext)
            else:
                _paint_h_edge(rect, r0 + ph - 1, c0, pw, seam_codes[i])

    for piece in pieces:
        _place_piece_interior_marks(rect, piece, accent_colors, _r)

    piece_arrays = [rect[pr : pr + ph, pc : pc + pw].copy() for pr, pc, ph, pw in pieces]

    allpix: list[int] = []
    for pa in piece_arrays:
        allpix.extend(map(int, pa.flatten()))
    if Counter(allpix).most_common(1)[0][0] != rect_col:
        return None

    for pa in piece_arrays:
        ph, pw = pa.shape
        fill = float(np.sum(pa != rect_col)) / float(ph * pw)
        if fill < 0.10 or fill > 0.60:
            return None
        if not (
            np.any(pa[0, :] != rect_col)
            or np.any(pa[-1, :] != rect_col)
            or np.any(pa[:, 0] != rect_col)
            or np.any(pa[:, -1] != rect_col)
        ):
            return None

    # Validate the intended D10 behavior quantitatively before scattering:
    # every assembled outer side should contain accent pixels.
    if not any(rect[0, :] != rect_col):
        return None
    if not any(rect[-1, :] != rect_col):
        return None
    if not any(rect[:, 0] != rect_col):
        return None
    if not any(rect[:, -1] != rect_col):
        return None

    if corner == "TL":
        anchor_r, anchor_c = 0, 0
    elif corner == "TR":
        anchor_r, anchor_c = 0, rw - 1
    elif corner == "BL":
        anchor_r, anchor_c = rh - 1, 0
    else:
        anchor_r, anchor_c = rh - 1, rw - 1

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
    if corner == "TR" and not (apr == 0 and apc + apw == rw):
        return None
    if corner == "BL" and not (apr + aph == rh and apc == 0):
        return None
    if corner == "BR" and not (apr + aph == rh and apc + apw == rw):
        return None

    # Check the solver's new assumption: internal edges have a mate, outer edges do not.
    def side_counts() -> tuple[list[int], list[int], list[int], list[int]]:
        tops = [[] for _ in piece_arrays]
        bottoms = [[] for _ in piece_arrays]
        lefts = [[] for _ in piece_arrays]
        rights = [[] for _ in piece_arrays]
        for i, a in enumerate(piece_arrays):
            for j, b in enumerate(piece_arrays):
                if i == j:
                    continue
                if a.shape[1] == b.shape[1]:
                    if _compat(a[0, :], b[-1, :], rect_col):
                        tops[i].append(j)
                    if _compat(a[-1, :], b[0, :], rect_col):
                        bottoms[i].append(j)
                if a.shape[0] == b.shape[0]:
                    if _compat(a[:, 0], b[:, -1], rect_col):
                        lefts[i].append(j)
                    if _compat(a[:, -1], b[:, 0], rect_col):
                        rights[i].append(j)
        return (
            [len(x) for x in tops],
            [len(x) for x in bottoms],
            [len(x) for x in lefts],
            [len(x) for x in rights],
        )

    top_n, bottom_n, left_n, right_n = side_counts()
    for idx, (pr, pc, ph, pw) in enumerate(pieces):
        if (pr == 0) != (top_n[idx] == 0):
            return None
        if (pr + ph == rh) != (bottom_n[idx] == 0):
            return None
        if (pc == 0) != (left_n[idx] == 0):
            return None
        if (pc + pw == rw) != (right_n[idx] == 0):
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
        if cr < 0 or cr >= gh or cc < 0 or cc >= gw:
            return None

    canvas = np.full((gh, gw), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    order = list(range(len(piece_arrays)))
    _r.shuffle(order)
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
        tr, tc = mr, mc - rw + 1
    elif corner == "BL":
        tr, tc = mr - rh + 1, mc
    else:
        tr, tc = mr - rh + 1, mc - rw + 1

    expected = np.full((gh, gw), bg, dtype=int)
    expected[tr : tr + rh, tc : tc + rw] = rect

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
