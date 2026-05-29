from __future__ import annotations

# Variant of 446ef5d2
# Dimension: D1 — 边缘匹配条件 (match_cond)
# Change: 相邻碎片的接触边上，非 rect_col 像素位置和颜色必须一致；rect_col 充当通配符
# ... → 颜色互补匹配。固定互补表为
# {0<->9, 1<->2, 3<->4, 5<->6, 7<->8}，因此 solver 需要按互补色而不是同色来拼边。
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 输入仍是：背景上散放若干矩形碎片，以及一个 3 像素 L 形 marker；marker 缺失格仍然指示最终大矩形的一个角。
# - 输出仍是：把所有碎片平移回 marker 指向的位置，组成一个完整矩形。
# - 核心变化在“共享边如何判断可拼”：
#   原题：若两块接触边某位置都不是 rect_col，则两侧颜色必须相同。
#   变体：若两块接触边某位置都不是 rect_col，则两侧颜色必须互为固定互补色。
# - 小例子：左块右边列是 [7,1,7]，右块左边列是 [8,2,8]。原题会判错（7!=8, 1!=2），变体会判对
#   （7<->8, 1<->2）。因此最终拼好的矩形内部会出现“红挨蓝、绿挨黄”这类接缝，而不是同色延续。
#
# b) 复杂度校准
# - 若颜色太多，人类无法从少量样例归纳映射；若只用 1 个颜色对，又容易弱化变体。这里固定使用 2 组互补对，
#   每条内部接缝大多有多个非 rect_col 位置，既能重复提供证据，又不会把颜色空间扩大得太乱。
# - 为了稳定构造互补接缝，生成器把矩形分成规则的行列网格（2x2 到 4x3 / 3x4），每块仍是矩形、数量仍在 4-12，
#   只是去掉了随机 merge，这样每条共享边都只面对一个邻居，互补约束更清晰。
#
# c) 隐式假设审查（main 中与 D1 相关的旧前提）
# - `compat(v1, v2)` 原先假设“非 rect_col 位置应相等”。变体下该前提失效，必须改成
#   `complement(v1[pos]) == v2[pos]`。
# - `neighbor_ok()` 其余逻辑仍成立，因为它只依赖 `compat()` 返回值，不依赖颜色必须相等。
# - `boundary_ok()` 仍成立：大矩形外边框仍然必须全是 rect_col，这一维没有变化。
# - `rect_col = most common piece color` 仍成立，但生成器需要保证非 rect_col 填充比例不过高，避免其它颜色反超。
#
# d) 放置合法性
# - 生成器先确定最终矩形与规则分块，再给每条内部共享边成对着色，最后把各块散放回背景。
# - 所有碎片仍为轴对齐矩形、互不重叠、与 marker 保持至少 1 格背景间隔，并通过 `main(input)` 回代验证。
# - 为避免边角单元同时受横向和纵向接缝约束而冲突，所有内部接缝的两个端点统一保留为 rect_col。
#
# e) 视觉可观察性
# - 每个样例中都会出现多条互补接缝，而且这些接缝在输出里可直接看到：接缝两侧是稳定的颜色对，而不是同色延续。
# - 互补表固定为任务级规则。代码内置该表，符合 ARC 设定：人类从 train 对中归纳映射，solver 则直接执行。
#
# f) 可视化整洁度
# - 只在内部接缝和少量向内延伸的短条/小块上使用互补色，矩形外边框继续保持单一 rect_col，marker 也保持原 L 形。
# - 这样能把“背景 / marker / 可拼边 / 大矩形边框”四类视觉角色分开，不会混成一团。
#
# g) 颜色变化可推断性
# - 颜色映射不是数值规则，而是固定配对规则；每个样例只启用 2 对颜色，并在多条接缝上重复出现。
# - 接缝两侧的结构完全对齐，只改颜色关系，因此人类能把“哪些边能拼”归因到颜色互补，而不是位置巧合。
#
# 新增的 solver 假设与 generator 保证
# - 假设 1：碎片中除 rect_col 外出现的颜色都来自固定互补表。生成器只从完整的互补对里取颜色。
# - 假设 2：每条真正的内部共享边至少有 1 个非 rect_col 互补位置。生成器显式保证。
# - 假设 3：所有边角共享点若参与多条接缝会引起多重约束，因此它们保持 rect_col。生成器显式保证。
# - 假设 4：rect_col 仍是碎片像素的众数。生成器控制整体非 rect_col 填充比例并在失败时重试。

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


def _comp(color: int) -> int:
    return int(COMPLEMENT_MAP[int(color)])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered rectangular pieces into a rectangle at an L-marker corner.

    Variant rule:
    - `rect_col` remains the wildcard border/background color for the assembled rectangle.
    - For adjacent pieces, non-rect_col pixels along a shared edge must be complementary
      according to COMPLEMENT_MAP, not identical.
    """
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

    def compat(v1: np.ndarray, v2: np.ndarray) -> bool:
        v1 = np.array(v1, dtype=int)
        v2 = np.array(v2, dtype=int)
        m = (v1 != int(rect_col)) & (v2 != int(rect_col))
        if not bool(m.any()):
            return True
        left = v1[m]
        right = v2[m]
        return bool(np.all(np.vectorize(_comp)(left) == right))

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


def _sample_segments(n_parts: int, rng: random.Random) -> list[int]:
    return [rng.randint(3, 6) for _ in range(n_parts)]


def _inner_positions(length: int, rng: random.Random) -> list[int]:
    if length <= 2:
        return []
    inner = list(range(1, length - 1))
    density = rng.uniform(0.55, 0.9)
    pos = [p for p in inner if rng.random() < density]
    if not pos:
        pos = [rng.choice(inner)]
    return pos


def _augment_piece(
    piece: np.ndarray,
    rng: random.Random,
    active_cols: list[int],
) -> None:
    h, w = piece.shape
    edge_colors = [int(x) for x in np.unique(np.concatenate([piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]))]
    edge_colors = [c for c in edge_colors if c in active_cols]

    # Extend seam colors inward so the complement rule remains visually obvious inside each tile.
    for c in range(w):
        if piece[0, c] in active_cols and h > 1:
            piece[1, c] = piece[0, c]
            if h > 3 and rng.random() < 0.5:
                piece[2, c] = piece[0, c]
        if piece[h - 1, c] in active_cols and h > 1:
            piece[h - 2, c] = piece[h - 1, c]
            if h > 3 and rng.random() < 0.5:
                piece[h - 3, c] = piece[h - 1, c]
    for r in range(h):
        if piece[r, 0] in active_cols and w > 1:
            piece[r, 1] = piece[r, 0]
            if w > 3 and rng.random() < 0.5:
                piece[r, 2] = piece[r, 0]
        if piece[r, w - 1] in active_cols and w > 1:
            piece[r, w - 2] = piece[r, w - 1]
            if w > 3 and rng.random() < 0.5:
                piece[r, w - 3] = piece[r, w - 1]

    if h < 3 or w < 3:
        return

    target_fill = rng.uniform(0.18, 0.42)
    strict = piece[1:-1, 1:-1]
    max_steps = max(1, (h * w) // 5)
    while float(np.mean(piece != piece[0, 0])) < target_fill:
        free = np.argwhere(strict == piece[0, 0])
        if free.size == 0:
            break
        sr, sc = map(int, free[rng.randrange(len(free))])
        color = rng.choice(edge_colors if edge_colors else active_cols)
        frontier = [(sr, sc)]
        seen = {(sr, sc)}
        strict[sr, sc] = color
        goal = rng.randint(1, max_steps)
        while frontier and len(seen) < goal:
            cr, cc = frontier.pop(rng.randrange(len(frontier)))
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = cr + dr, cc + dc
                if not (0 <= nr < strict.shape[0] and 0 <= nc < strict.shape[1]):
                    continue
                if (nr, nc) in seen:
                    continue
                if strict[nr, nc] != piece[0, 0]:
                    continue
                if rng.random() < 0.7:
                    strict[nr, nc] = color
                    frontier.append((nr, nc))
                    seen.add((nr, nc))
                    if len(seen) >= goal:
                        break


def _build_rect_and_pieces(rng: random.Random):
    # Prefer 6-12 pieces so the complement evidence repeats across multiple joins.
    shapes = [(2, 3), (3, 2), (2, 4), (4, 2), (3, 3), (3, 4), (4, 3), (2, 2)]
    nr, nc = rng.choice(shapes)
    row_segs = _sample_segments(nr, rng)
    col_segs = _sample_segments(nc, rng)
    RH, RW = sum(row_segs), sum(col_segs)

    specials = rng.sample(list(range(10)), 3)
    bg, rect_col, marker_col = map(int, specials)
    full_pairs = [pair for pair in PAIR_UNITS if pair[0] not in specials and pair[1] not in specials]
    if len(full_pairs) < 2:
        return None
    pair_units = rng.sample(full_pairs, 2)
    active_cols = [c for pair in pair_units for c in pair]

    pieces: list[list[np.ndarray]] = [
        [np.full((row_segs[i], col_segs[j]), rect_col, dtype=int) for j in range(nc)]
        for i in range(nr)
    ]
    seam_active = 0

    for i in range(nr):
        for j in range(nc - 1):
            left = pieces[i][j]
            right = pieces[i][j + 1]
            pair = rng.choice(pair_units)
            cols = pair if rng.random() < 0.5 else (pair[1], pair[0])
            pos = _inner_positions(left.shape[0], rng)
            for rr in pos:
                left[rr, -1] = cols[0]
                right[rr, 0] = cols[1]
            seam_active += len(pos)

    for i in range(nr - 1):
        for j in range(nc):
            up = pieces[i][j]
            down = pieces[i + 1][j]
            pair = rng.choice(pair_units)
            cols = pair if rng.random() < 0.5 else (pair[1], pair[0])
            pos = _inner_positions(up.shape[1], rng)
            for cc in pos:
                up[-1, cc] = cols[0]
                down[0, cc] = cols[1]
            seam_active += len(pos)

    if seam_active < 8:
        return None

    piece_specs = []
    r_starts = np.cumsum([0] + row_segs[:-1]).tolist()
    c_starts = np.cumsum([0] + col_segs[:-1]).tolist()
    for i in range(nr):
        for j in range(nc):
            piece = pieces[i][j]
            _augment_piece(piece, rng, active_cols)
            non_rc_edges = sum(
                int(np.any(edge != rect_col))
                for edge in [piece[0, :], piece[-1, :], piece[:, 0], piece[:, -1]]
            )
            if non_rc_edges == 0:
                return None
            fill = float(np.mean(piece != rect_col))
            if fill < 0.12 or fill > 0.58:
                return None
            piece_specs.append((r_starts[i], c_starts[j], piece.shape[0], piece.shape[1], piece.copy()))

    rect = np.full((RH, RW), rect_col, dtype=int)
    for pr, pc, ph, pw, arr in piece_specs:
        rect[pr : pr + ph, pc : pc + pw] = arr

    if not np.all(rect[0, :] == rect_col):
        return None
    if not np.all(rect[-1, :] == rect_col):
        return None
    if not np.all(rect[:, 0] == rect_col):
        return None
    if not np.all(rect[:, -1] == rect_col):
        return None

    rect_fill = float(np.mean(rect != rect_col))
    if rect_fill < 0.22 or rect_fill > 0.55:
        return None

    counts = Counter(map(int, rect.flatten()))
    if counts[rect_col] != max(counts.values()):
        return None
    if any(counts[c] < 4 for c in active_cols):
        return None

    return {
        "rect": rect,
        "pieces": piece_specs,
        "RH": RH,
        "RW": RW,
        "bg": bg,
        "rect_col": rect_col,
        "marker_col": marker_col,
        "active_cols": active_cols,
        "seam_active": seam_active,
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    built = _build_rect_and_pieces(rng)
    if built is None:
        return None

    rect = built["rect"]
    piece_specs = built["pieces"]
    RH = int(built["RH"])
    RW = int(built["RW"])
    bg = int(built["bg"])
    marker_col = int(built["marker_col"])

    area = RH * RW
    if RH > 22 or RW > 22:
        return None

    GH_lo = max(10, RH + 4)
    GW_lo = max(10, RW + 4)
    GH_hi = min(26, RH + 10)
    GW_hi = min(26, RW + 10)
    if GH_lo > GH_hi or GW_lo > GW_hi:
        return None

    GH = rng.randint(GH_lo, GH_hi)
    GW = rng.randint(GW_lo, GW_hi)
    if GH * GW < int(area * 2.2) + 12:
        return None

    corner = rng.choice(["TL", "TR", "BL", "BR"])
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
        tr, tc = mr, mc
    elif corner == "TR":
        marker_cells = [(mr - 1, mc), (mr - 1, mc + 1), (mr, mc + 1)]
        tr, tc = mr, mc - RW + 1
    elif corner == "BL":
        marker_cells = [(mr, mc - 1), (mr + 1, mc - 1), (mr + 1, mc)]
        tr, tc = mr - RH + 1, mc
    else:
        marker_cells = [(mr, mc + 1), (mr + 1, mc), (mr + 1, mc + 1)]
        tr, tc = mr - RH + 1, mc - RW + 1

    for cr, cc in marker_cells:
        if cr < 0 or cr >= GH or cc < 0 or cc >= GW:
            return None

    canvas = np.full((GH, GW), bg, dtype=int)
    for cr, cc in marker_cells:
        canvas[cr, cc] = marker_col

    order = list(range(len(piece_specs)))
    rng.shuffle(order)
    for idx in order:
        _pr, _pc, _ph, _pw, piece = piece_specs[idx]
        try:
            r0, c0 = random_free_location(canvas, piece, transparent=bg, padding=1)
            paste(canvas, piece, r0, c0)
        except ValueError:
            return None

    expected_out = np.full((GH, GW), bg, dtype=int)
    expected_out[tr : tr + RH, tc : tc + RW] = rect

    try:
        actual_out = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual_out, expected_out):
        return None

    # Guard against accidental regression to the old rule:
    # there must be many shared-edge positions where complement matches but equality fails.
    old_rule_fail = 0
    for r in range(tr, tr + RH):
        for c in range(tc, tc + RW - 1):
            a = int(expected_out[r, c])
            b = int(expected_out[r, c + 1])
            if a != built["rect_col"] and b != built["rect_col"] and _comp(a) == b and a != b:
                old_rule_fail += 1
    for r in range(tr, tr + RH - 1):
        for c in range(tc, tc + RW):
            a = int(expected_out[r, c])
            b = int(expected_out[r + 1, c])
            if a != built["rect_col"] and b != built["rect_col"] and _comp(a) == b and a != b:
                old_rule_fail += 1
    if old_rule_fail < 8:
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
