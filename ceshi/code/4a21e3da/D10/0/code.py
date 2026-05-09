# Variant of 4a21e3da
# Dimension: D10 — 片段目标位置 (meta)
# Change: 片段外接矩形紧贴对应角落放置（flush with corner）... → 片段对齐到边缘中点而非角落
#
# 设计分析
# - 参考任务: 已阅读 `/home/lixinyu/snap/new_code/solvers/arc2/58f5dbd5.py`。其中与本变体
#   相关的机制不是对象识别，而是“按 bbox 尺寸做整数居中偏移”：
#   `off_r = (5 - mh) // 2`, `off_c = (5 - mw) // 2`。本题借用的是这个
#   “目标位置靠居中公式而不是靠角对齐公式” 的思路。
# - 具体化:
#   输入仍是一个近中心的单连通 fg 精灵，外加 1-2 个边缘 ctrl 像素。ctrl 沿自身所在行/列向内
#   发射间歇射线，切割精灵；分割线上的 fg 保持原位；与 ctrl 所在边接壤的区域保留，其余消失。
#   原题里保留区域会把 bbox 推到对应角落；本变体中，保留区域仍然“靠某条有效控制边”，但不再
#   靠角，而是在该边对应的可用边段里做整数居中。
# - 小例子追踪:
#   若只有一个 top 控制点和一条竖切线，原题左右两块都会移到左上/右上角；现在仍然都贴着 top，
#   但左块在列区间 `[0, c_split-1]` 内水平居中，右块在 `[c_split+1, W-1]` 内水平居中。
#   因而输出中两块仍在顶边，却不再顶到左右角。
# - 双控制歧义处理:
#   当某块同时接触一条横边和一条竖边（例如 `top+left` 的 TL 象限）时，若同时让它“靠 top 中点”
#   和“靠 left 中点”会产生二义性。这里固定采用“横边优先”：
#   若该块接触了有效的 top/bottom 控制边，就贴那条横边并在该边段内水平居中；
#   否则才贴 left/right 并在对应边段内垂直居中。这样双控制时三块保留片段仍会分散到两条控制边上，
#   不会压到同一位置。
# - 复杂度校准:
#   只改目标位置，不改颜色、射线、消失规则、输入连通性、尺寸密度等。为了让 D10 在肉眼上稳定可见，
#   generator 额外拒绝那些“新输出与原始落角输出完全相同”的样本，避免退化。
# - main() 的旧隐式假设审查:
#   1. 原 Stage 2 用 `dr = -rmin / H-1-rmax`, `dc = -cmin / W-1-cmax`，隐含目标总是角点；
#      在本变体里该前提失效，必须改成“先选目标边段，再按 bbox 尺寸做居中偏移”。
#   2. 原代码把一个矩形区域里的所有 fg 当作同一片段平移；这仍成立，因为 generator 继续保证每个
#      保留/消失区域各自只有一块实际片段。
#   3. 原代码默认平移后不会越界或穿过 split line；本变体新增了“边段长度需容纳片段 bbox”的假设，
#      generator 会显式检查这一点。
# - 放置合法性:
#   top/bottom 目标使用按 split 划分后的左/右边段；left/right 目标使用按 split 划分后的上/下边段。
#   因此不同片段的目标段天然互斥，且不会覆盖 split line。generator 在 output-first 构造时还会检查
#   centered 版本与 old-corner 版本是否不同、以及 `main(inp) == out`。
# - 视觉可观察性:
#   样例中保留片段依然贴边，因此与原题属于同一视觉范式；但它们相对角点会留出明显背景间隙，
#   2-3 组样例即可看出“贴边但居中”而不是“挤到角上”。
# - 新 main() 的隐式假设:
#   `main()` 现在假设：若某区域被保留，则其 bbox 能放入所选目标边段。generator 通过控制 split 位于
#   40-60%、片段像素数较小、并在生成后以 centered 目标段实际验证，来保证这个前提成立。

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


def _parse_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = c
        elif r == H - 1:
            E.add("bottom")
            c_split = c
        elif c == 0:
            E.add("left")
            r_split = r
        elif c == W - 1:
            E.add("right")
            r_split = r

    return g, H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split


def _touches_edges(hs: str, vs: str) -> set[str]:
    te: set[str] = set()
    if hs in ("T", "A"):
        te.add("top")
    if hs in ("B", "A"):
        te.add("bottom")
    if vs in ("L", "A"):
        te.add("left")
    if vs in ("R", "A"):
        te.add("right")
    return te


def _pick_corner_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
    if r_split is not None:
        dh = hs
    elif ("top" in E) and ("bottom" not in E):
        dh = "T"
    elif ("bottom" in E) and ("top" not in E):
        dh = "B"
    else:
        dh = "T"

    if c_split is not None:
        dv = vs
    elif ("left" in E) and ("right" not in E):
        dv = "L"
    elif ("right" in E) and ("left" not in E):
        dv = "R"
    else:
        dv = "L"
    return dh, dv


def _pick_center_edge(hs: str, vs: str, E: set[str]) -> str:
    touched = _touches_edges(hs, vs) & E
    assert touched
    if "top" in touched:
        return "top"
    if "bottom" in touched:
        return "bottom"
    if "left" in touched:
        return "left"
    return "right"


def _segment_for_edge(
    edge: str,
    hs: str,
    vs: str,
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int]:
    if edge in ("top", "bottom"):
        if c_split is None:
            return 0, W - 1
        if vs == "L":
            return 0, c_split - 1
        if vs == "R":
            return c_split + 1, W - 1
        return 0, W - 1

    if r_split is None:
        return 0, H - 1
    if hs == "T":
        return 0, r_split - 1
    if hs == "B":
        return r_split + 1, H - 1
    return 0, H - 1


def _move_piece_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = -rmin if dh == "T" else (H - 1 - rmax)
    dc = -cmin if dv == "L" else (W - 1 - cmax)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _move_piece_to_centered_edge(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[frozenset[tuple[int, int]], str, tuple[int, int]]:
    edge = _pick_center_edge(hs, vs, E)
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    ph = int(rmax - rmin + 1)
    pw = int(cmax - cmin + 1)

    seg0, seg1 = _segment_for_edge(edge, hs, vs, H, W, r_split, c_split)
    seg_len = int(seg1 - seg0 + 1)
    assert seg_len > 0

    if edge == "top":
        target_r0 = 0
        target_c0 = int(seg0 + (seg_len - pw) // 2)
        margins = (int(target_c0 - seg0), int(seg1 - (target_c0 + pw - 1)))
    elif edge == "bottom":
        target_r0 = int(H - ph)
        target_c0 = int(seg0 + (seg_len - pw) // 2)
        margins = (int(target_c0 - seg0), int(seg1 - (target_c0 + pw - 1)))
    elif edge == "left":
        target_r0 = int(seg0 + (seg_len - ph) // 2)
        target_c0 = 0
        margins = (int(target_r0 - seg0), int(seg1 - (target_r0 + ph - 1)))
    else:
        target_r0 = int(seg0 + (seg_len - ph) // 2)
        target_c0 = int(W - pw)
        margins = (int(target_r0 - seg0), int(seg1 - (target_r0 + ph - 1)))

    dr = int(target_r0 - rmin)
    dc = int(target_c0 - cmin)
    moved = frozenset((int(r + dr), int(c + dc)) for r, c in pts)
    return moved, edge, margins


def _solve(grid: np.ndarray, *, centered: bool) -> np.ndarray:
    g, H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _parse_grid(grid)
    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            out[rr, cc] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if rr < r_split else "B"
        if c_split is not None:
            vs = "L" if cc < c_split else "R"
        regions.setdefault((hs, vs), []).append((rr, cc))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue

        if centered:
            moved, _edge, _margins = _move_piece_to_centered_edge(pts, hs, vs, E, H, W, r_split, c_split)
        else:
            dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
            moved = _move_piece_to_corner(pts, dh, dv, H, W)

        for rr, cc in moved:
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg
    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, centered=True)


def _corner_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, centered=False)


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(target_px - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= max_h
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= max_w
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts: list[tuple[int, int]] = []
    E = set(ctrl_edges)
    has_v = has_h = False

    for edge in ctrl_edges:
        if edge in ("top", "bottom"):
            c_split = random.randint(int(W * 0.4), int(W * 0.6))
            ctrl_pts.append((0 if edge == "top" else H - 1, c_split))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((r_split, 0 if edge == "left" else W - 1))
            has_h = True

    def touches_E(hs: str, vs: str) -> bool:
        return bool(_touches_edges(hs, vs) & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if touches_E(*reg)]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = c_split - int(frac * sprite_w)
        sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
        if sprite_c0 > W - 5 - sprite_w + 1:
            return None
        if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
            return None
    else:
        hi = W - 5 - sprite_w + 1
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = r_split - int(frac * sprite_h)
        sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
        if sprite_r0 > H - 5 - sprite_h + 1:
            return None
        if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
            return None
    else:
        hi = H - 5 - sprite_h + 1
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split + 1)
        r1 = (int(r_split - 1) if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split + 1)
        c1 = (int(c_split - 1) if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((r, int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), c))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = target_total - n_split
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= vanish_budget
    else:
        vanish_budget = 0

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3))
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    centered_gaps: list[int] = []

    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        if has_h:
            inp_r0 = (r1 - ph + 1) if hs == "T" else r0
        else:
            split_rows = sorted(r for r, c in split_pixels if c == c_split)
            if split_rows:
                mid = (split_rows[0] + split_rows[-1]) // 2
                inp_r0 = max(r0, min(mid - ph // 2, r1 - ph + 1))
            else:
                inp_r0 = r0

        if has_v:
            inp_c0 = (c1 - pw + 1) if vs == "L" else c0
        else:
            split_cols = sorted(c for r, c in split_pixels if r == r_split)
            if split_cols:
                mid = (split_cols[0] + split_cols[-1]) // 2
                inp_c0 = max(c0, min(mid - pw // 2, c1 - pw + 1))
            else:
                inp_c0 = c0

        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        input_pieces[reg] = inp_pts

        centered_pts, edge, margins = _move_piece_to_centered_edge(inp_pts, hs, vs, E, H, W, r_split, c_split)
        if min(margins) < 0:
            return None
        centered_gaps.append(max(margins))

        for r, c in centered_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
            if edge in ("top", "bottom") and not (0 <= c < W):
                return None
            if edge in ("left", "right") and not (0 <= r < H):
                return None
        output_pieces[reg] = centered_pts

    if not centered_gaps or max(centered_gaps) <= 0:
        return None

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= split_pixels
    if not all_inp_fg:
        return None

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = max(r for r, _ in piece) + 1
            ppw = max(c for _, c in piece) + 1
            vr_off = (vr1 - pph + 1) if vhs == "T" else vr0
            vc_off = (vc1 - ppw + 1) if vvs == "L" else vc0
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg = vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[r, c] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg

    for cr, cc in ctrl_pts:
        if cr == 0:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(max(fg_rows) + 1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            fg_rows = [r for r in range(H) if out[r, cc] == fg]
            if fg_rows:
                for r in range(H - 1, min(fg_rows) - 1, -1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(max(fg_cols) + 1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            fg_cols = [c for c in range(W) if out[cr, c] == fg]
            if fg_cols:
                for c in range(W - 1, min(fg_cols) - 1, -1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_corner_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
