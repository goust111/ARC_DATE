# Variant of 4a21e3da
# Dimension: D9 — 分割线上前景像素的处理 (meta)
# Change: 分割线上的前景像素保留在原位不移动... → 分割线像素变色标记

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
design_analysis:
  variant_goal:
    dimension: D9
    old_behavior: 前景像素如果正好落在分割线(控制像素定义的行/列)上, 输出里仍保持 fg 颜色并停在原位。
    new_behavior: 这些像素不再保留 fg, 而是统一改涂为 ctrl 颜色, 作为可见的切割标记线。
  concrete_behavior:
    input:
      - 仍然只有 3 种颜色: bg / fg / ctrl。
      - 仍然有 1 或 2 个边界控制像素, 它们定义一条竖切线和/或一条横切线。
      - sprite 仍是单个连通 fg 组件, 且每条切线上都有 2-4 个 fg 像素。
    output:
      - 切线所在的 bg 单元依旧被 ctrl 射线涂色。
      - 新变体里, 切线上的那些 fg 也被改涂成 ctrl, 因而切线会变成连续、显眼的 ctrl 线。
      - 其他区域的切块仍按原规则滑到角落或消失。
  trace_example:
    setup:
      - 顶边控制像素位于列 c=8, sprite 在该列上有 fg 于 r=6,8,10。
    old:
      - 射线先把 (0..10,8) 中的 bg 涂成 ctrl, 但 r=6,8,10 仍保留 fg。
      - 输出线看起来是 ctrl-ctrl-...-fg-...-fg, 属于“间歇射线”。
    new:
      - 同样先把 bg 涂成 ctrl。
      - 再把 r=6,8,10 这些切线 fg 也改成 ctrl。
      - 输出线变成从边界到最远切点的连续 ctrl 分隔线。
    difference:
      - 改动只发生在“原本留在原地的 split fg 像素”的着色阶段, 不改变区域划分与位移逻辑。
  complexity_calibration:
    - 保持颜色数仍为 3, 避免引入新的颜色映射链条。
    - 保持每条切线 2-4 个 split 像素, 让“fg 变 ctrl”在 2-4 个样例里稳定可见。
    - 不增加额外 case 分叉; 仍只分 1 控制像素 / 2 控制像素两档。
  implicit_assumption_review:
    - 原 solver 的 Stage 1 依赖: 射线终点由输入中 split 行/列里的 fg 决定。这在变体中仍成立, 因为输入不变。
    - 原 solver 的 “split fg remain in place” 代码依赖旧属性, 现在必须改成 recolor 为 ctrl。
    - Stage 2 依赖: 分割线像素不参与切块平移。该前提仍成立, 因为它们依旧被 continue 排除, 只是输出颜色变化。
    - 原 generator 用终态 out 中的 fg 位置来回填射线长度。旧版 split 像素是 fg, 所以无害; 新版若先把 split 像素改成 ctrl, 这段写法会错误地把射线缩短甚至不画。必须改成用 split_pixels 自身确定 stop。
  placement_legality:
    - 输入生成仍要求 kept/vanish pieces 与 split pixels 不重叠、不过线、不越界。
    - 输出中 piece 仍严格留在各自半区/象限, 因而不会覆盖 split 线上的 ctrl 标记。
  visual_observability:
    - 旧任务输出中的切线是“ctrl 背景格 + fg 孔洞”; 新任务变成连续 ctrl 线。
    - 由于每条切线稳定含 2-4 个 split fg, 差异不会退化成偶发单点。
  cleanliness:
    - 仍只在一条行/列上叠加标记, 视觉比原题更整洁而不是更乱。
  color_inference:
    - 只有一种颜色变化: “位于切线上的 fg -> ctrl”。
    - 线的位置由边界控制像素直接提示, 人类可从位置结构推断颜色变化条件。
  new_main_assumptions_to_support:
    - main() 假设 split 线由 1 条竖线和/或 1 条横线定义, generator 继续保证这一点。
    - main() 假设所有应 recolor 的像素都能通过 `(c==c_split) or (r==r_split)` 检出, generator 保证 split_pixels 只放在这些位置。
    - main() 假设平移后的 kept pieces 不会落回 split 行/列上覆盖 ctrl 标记, generator 继续通过半区约束保证这一点。
"""


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide pieces to grid corners.

    Variant D9:
    - fg on the split line does not stay fg in the output.
    - Instead, those split-line fg pixels are recolored to ctrl, turning the cut
      into a continuous visible separator.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})

    # Control color is the only non-bg color present on the border.
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

    # Split lines (at most one vertical and one horizontal in these tasks).
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if int(r) == 0:
            E.add("top")
            c_split = int(c)
        elif int(r) == int(H - 1):
            E.add("bottom")
            c_split = int(c)
        elif int(c) == 0:
            E.add("left")
            r_split = int(r)
        elif int(c) == int(W - 1):
            E.add("right")
            r_split = int(r)

    def on_split(r: int, c: int) -> bool:
        return (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split))

    out = np.full((H, W), int(bg), dtype=int)

    # --- Stage 1: draw rays on background cells only (but keep the control pixel itself). ---
    for r, c in ctrl_pts:
        if int(r) == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, int(stop) + 1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(r) == int(H - 1):
            rows = [int(rr) for rr in np.argwhere(g[:, int(c)] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(int(H - 1), int(stop) - 1, -1):
                    if int(g[int(rr), int(c)]) == int(bg):
                        out[int(rr), int(c)] = int(ctrl)
        elif int(c) == 0:
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, int(stop) + 1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)
        elif int(c) == int(W - 1):
            cols = [int(cc) for cc in np.argwhere(g[int(r), :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(int(W - 1), int(stop) - 1, -1):
                    if int(g[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl)

    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    # Variant D9: split-line fg becomes ctrl, acting as a visible divider.
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if on_split(r, c):
            out[int(r), int(c)] = int(ctrl)

    # --- Stage 2: move cut parts from each rectangular region to its target corner. ---
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if on_split(r, c):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    def touches_edges(hs: str, vs: str) -> set[str]:
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

    def pick_dest(hs: str, vs: str) -> tuple[str, str]:
        if r_split is not None:
            dh = hs
        else:
            if ("top" in E) and ("bottom" not in E):
                dh = "T"
            elif ("bottom" in E) and ("top" not in E):
                dh = "B"
            else:
                dh = "T"
        if c_split is not None:
            dv = vs
        else:
            if ("left" in E) and ("right" not in E):
                dv = "L"
            elif ("right" in E) and ("left" not in E):
                dv = "R"
            else:
                dv = "L"
        return str(dh), str(dv)

    for (hs, vs), pts in regions.items():
        if not (touches_edges(hs, vs) & E):
            continue

        dh, dv = pick_dest(hs, vs)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
        dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h, max_w, target_px):
    """Grow a connected piece up to max_h x max_w targeting target_px pixels."""
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
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _gen_attempt():
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    # Weight toward 2-ctrl to compensate for higher rejection rate.
    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = r_split = None
    ctrl_pts = []
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

    def pick_dest(hs, vs):
        if r_split is not None:
            dh = hs
        elif "top" in E and "bottom" not in E:
            dh = "T"
        elif "bottom" in E and "top" not in E:
            dh = "B"
        else:
            dh = "T"
        if c_split is not None:
            dv = vs
        elif "left" in E and "right" not in E:
            dv = "L"
        elif "right" in E and "left" not in E:
            dv = "R"
        else:
            dv = "L"
        return dh, dv

    def touches_E(hs, vs):
        te = set()
        if hs in ("T", "A"):
            te.add("top")
        if hs in ("B", "A"):
            te.add("bottom")
        if vs in ("L", "A"):
            te.add("left")
        if vs in ("R", "A"):
            te.add("right")
        return bool(te & E)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [r for r in all_regions if touches_E(*r)]
    vanish = [r for r in all_regions if r not in kept]

    # Sprite bbox (spec: 5-8 x 8-10).
    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    # Place sprite with split at 40-60% through bbox.
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

    def sub_bbox(hs, vs):
        r0 = sprite_r0 if hs in ("T", "A") else r_split + 1
        r1 = (r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else c_split + 1
        c1 = (c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    # Split-line fg pixels (2-4 per split). These will recolor to ctrl in the output.
    split_pixels = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((r, c_split))
    if has_h:
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((r_split, c))

    n_split = len(split_pixels)

    # Target total fg pixels for density 0.39-0.60.
    area = sprite_h * sprite_w
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    # Budget for pieces (excluding split pixels and vanish).
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

    # Distribute budget among kept pieces.
    piece_budgets = []
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

    input_pieces = {}
    output_pieces = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = pick_dest(hs, vs)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        target_px = piece_budgets[idx]

        piece = _grow_piece(sub_h, sub_w, target_px)
        if len(piece) < 3:
            return None

        ph = max(r for r, _ in piece) + 1
        pw = max(c for _, c in piece) + 1

        # Place flush toward split for connectivity.
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

        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        input_pieces[reg] = inp_pts

        inp_rs = [r for r, _ in inp_pts]
        inp_cs = [c for _, c in inp_pts]
        dr = -min(inp_rs) if dh == "T" else (H - 1 - max(inp_rs))
        dc = -min(inp_cs) if dv == "L" else (W - 1 - max(inp_cs))
        out_pts = frozenset((r + dr, c + dc) for r, c in inp_pts)

        for r, c in out_pts:
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
        output_pieces[reg] = out_pts

    # Sprite bbox check.
    all_inp_fg = set()
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

    # Vanishing region.
    vanish_fg = set()
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
            vpts = {(r + vr_off, c + vc_off) for r, c in piece}
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

    # Draw rays using the input split pixels as stop anchors; using output fg would
    # be wrong in this variant because split pixels are no longer fg in the output.
    for cr, cc in ctrl_pts:
        if cr == 0:
            rows = [r for r, c in split_pixels if c == cc]
            if rows:
                for r in range(max(rows) + 1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cr == H - 1:
            rows = [r for r, c in split_pixels if c == cc]
            if rows:
                for r in range(H - 1, min(rows) - 1, -1):
                    if out[r, cc] == bg:
                        out[r, cc] = ctrl_color
        elif cc == 0:
            cols = [c for r, c in split_pixels if r == cr]
            if cols:
                for c in range(max(cols) + 1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color
        elif cc == W - 1:
            cols = [c for r, c in split_pixels if r == cr]
            if cols:
                for c in range(W - 1, min(cols) - 1, -1):
                    if out[cr, c] == bg:
                        out[cr, c] = ctrl_color

    for r, c in split_pixels:
        out[r, c] = ctrl_color
    for cr, cc in ctrl_pts:
        out[cr, cc] = ctrl_color

    # Density check (spec: 0.39-0.60) - recompute bbox over combined fg in the input.
    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / (final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
