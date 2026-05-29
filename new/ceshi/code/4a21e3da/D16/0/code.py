# Variant of 4a21e3da
# Dimension: D16 — 消失区域的存在与功能 (meta)
# Change: 2个控制像素时存在消失区域（不与任何控制边接壤的象限），其前景像素在输出中删除... → 消失区域变为干扰项

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


"""
design_analysis:
  concrete_behavior:
    input:
      - 与原题相同：中心附近有一个单连通 fg 精灵，1 或 2 个 ctrl 像素在边界上，分别定义竖切线/横切线。
      - 当有 2 个 ctrl 时，精灵被分成 4 个象限区域；其中恰有 1 个象限不接壤任何控制边。
    output:
      - 接壤控制边的区域仍按原规则滑到对应角落。
      - 分割线上的 fg 仍留在原位，ctrl 射线仍只画在背景格上。
      - 关键变体：原本会“消失”的那个象限现在保留在原坐标，既不移动也不删除，成为中心干扰残留。
  behavior_trace:
    original_two_ctrl_example:
      - top+left 控制时，TL/TR/BL 三块会移动，BR 象限因为不接壤 top/left 而被跳过，所以输出里没有 BR 块。
    variant_two_ctrl_example:
      - 同样的输入下，BR 象限仍不参与滑移，但其 fg 像素直接拷贝回输出原位置。
    delta:
      - 唯一语义差异是“非接壤象限”的后果从 delete 改为 keep-in-place。
  complexity_calibration:
    - 为了让 D16 可观察，generator 固定生成双控制样本，并要求静止残留块至少 3 个像素。
    - 其他结构保持原题：仍只有 3 种颜色、1 个主体精灵、1 条竖切与/或 1 条横切。
  implicit_assumption_audit:
    main_stage2_original_continue:
      - 原代码 `if not (touches_edges(hs, vs) & E): continue` 隐含“非接壤区域必定消失”。
      - 该前提在 D16 下失效，必须改为把这些 pts 原位写回输出。
    generator_output_builder:
      - 原 generator 只把 kept pieces 和 split pixels 画进输出，默认 vanish piece 不可见。
      - D16 下要把该区域也画进输出，否则生成样本会退化回原题。
    ray_logic:
      - 射线依赖 split line 上的 fg，且只在 bg 上涂色；静止残留不在 split line 上，因此这一前提仍成立。
    overlap_safety:
      - 原题里“删除”意味着不会和移动后的角落块冲突；D16 下需要生成器保证静止残留仍位于中心象限，不会与角落块重叠。
  placement_legality:
    - 静止残留块仍只放在唯一的非接壤象限，严格避开 split row / split col。
    - kept pieces 的输出仍贴角放置；静止残留维持输入坐标，因此中心与角落天然分离。
    - 输入仍要求全部 fg 单连通；输出允许多连通分量。
  visual_observability:
    - 每个生成样本都是双控制样本，因此解题者总能同时看到“有些块滑到角落”和“有一块同色 fg 留在中心”两种状态。
    - 该中心残留不能太小，因此生成时为它预留 3-7 像素预算。
  neatness:
    - 仍维持原题 4 格以上边距、5-8 x 8-10 的主体 bbox 和 0.39-0.60 密度范围，避免中心残留与射线、角落块挤在一起。
  color_inference:
    - 无新增颜色映射；颜色语义保持原题不变，因此不存在不可推断的颜色规则。
  new_main_assumptions_and_generator_guarantees:
    - main 假设至多一条竖切和一条横切；generator 仍按这一结构造样本。
    - main 把每个非 split-line fg 按相对 split 的象限归类；generator 保证静止残留严格待在唯一非接壤象限内。
    - main 默认保留下来的静止残留不会落在 ctrl 射线上；generator 通过不在 split row/col 放置该块来保证这一点。
"""


def _detect_structure(
    g: np.ndarray,
) -> tuple[int, int, int, int, int, list[tuple[int, int]], set[str], int | None, int | None]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = (
        set(map(int, g[0, :]))
        | set(map(int, g[-1, :]))
        | set(map(int, g[:, 0]))
        | set(map(int, g[:, -1]))
    )
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

    return H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split


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


def _pick_dest(hs: str, vs: str, E: set[str], r_split: int | None, c_split: int | None) -> tuple[str, str]:
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

    return str(dh), str(dv)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide pieces to grid corners.

    D16 variant:
    - regions touching a control edge still slide to their corner;
    - the single non-touching region in the 2-control case now stays in place
      instead of disappearing.
    """
    g = np.array(grid, dtype=int)
    H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _detect_structure(g)

    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            out[r, c] = fg

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == c_split) or (r_split is not None and r == r_split):
            continue

        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if r < r_split else "B"
        if c_split is not None:
            vs = "L" if c < c_split else "R"
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            for r, c in pts:
                out[r, c] = fg
            continue

        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        rs = [r for r, _ in pts]
        cs = [c for _, c in pts]
        rmin, rmax = int(min(rs)), int(max(rs))
        cmin, cmax = int(min(cs)), int(max(cs))
        dr = -rmin if dh == "T" else (H - 1) - rmax
        dc = -cmin if dv == "L" else (W - 1) - cmax

        for r, c in pts:
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg

    return out


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
    return frozenset((r - min(rs), c - min(cs)) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    # D16 only changes the 2-control case, so generation is fixed to that regime.
    num_ctrl = 2
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

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    static_regs = [reg for reg in all_regions if reg not in kept]

    def sub_bbox(hs: str, vs: str, sprite_r0: int, sprite_r1: int, sprite_c0: int, sprite_c1: int) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
        r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
        c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

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

    for reg in kept + static_regs:
        r0, r1, c0, c1 = sub_bbox(*reg, sprite_r0, sprite_r1, sprite_c0, sprite_c1)
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

    budget_pieces = target_total - n_split
    if static_regs:
        static_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= static_budget
    else:
        static_budget = 0

    if budget_pieces < len(kept) * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i, reg in enumerate(kept):
        r0, r1, c0, c1 = sub_bbox(*reg, sprite_r0, sprite_r1, sprite_c0, sprite_c1)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if i == len(kept) - 1:
            px = remaining
        else:
            avg = remaining // (len(kept) - i)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (len(kept) - i - 1) * 3))
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs, sprite_r0, sprite_r1, sprite_c0, sprite_c1)
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

    all_inp_fg = set().union(*input_pieces.values()) if input_pieces else set()
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

    static_fg: set[tuple[int, int]] = set()
    if static_regs:
        reg = static_regs[0]
        vhs, vvs = reg
        vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs, sprite_r0, sprite_r1, sprite_c0, sprite_c1)
        vh, vw = vr1 - vr0 + 1, vc1 - vc0 + 1
        if vh < 1 or vw < 1:
            return None
        max_px = min(static_budget, vh * vw)
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
        static_fg = vpts
        if len(static_fg) < 3:
            return None

    combined = all_inp_fg | static_fg
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
    for r, c in static_fg:
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

    if static_regs and not static_fg:
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
