# Variant of 4a21e3da
# Dimension: D16 — 消失区域的存在与功能 (meta)
# Change: 2个控制像素时存在消失区域（不与任何控制边接壤的象限），其前景像素在输出中删除... → 消失区域移到网格中心

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
design_analysis:
  concretization:
    input:
      - 与原题一致: 一个居中的单连通 fg 精灵, 1 或 2 个边缘 ctrl 像素, ctrl 定义一条纵向和/或横向切分线。
      - 当有 2 个 ctrl 时, 精灵被划成 4 个象限; 其中恰有 1 个象限不接壤任一控制边。
    output:
      - 接壤控制边的片段仍然滑去对应外角, 分割线上的 fg 仍原地保留, ctrl 射线画法不变。
      - 唯一变化: 原先“消失”的象限不再删除, 而是滑向分割线交叉点附近的内角:
        TL->贴到(r_split-1,c_split-1), TR->贴到(r_split-1,c_split+1),
        BL->贴到(r_split+1,c_split-1), BR->贴到(r_split+1,c_split+1)。
    trace_example: |
      例如 ctrl 在 top 和 left, 非接壤象限是 BR。
      原逻辑: Stage 2 遍历到 (B,R) 时, `touches_edges(B,R) & E == empty`, 直接 continue, 所以该片段在输出消失。
      变体逻辑: 同一处不再 continue, 而是把 BR 片段整体平移, 使其 bbox 左上角贴到
      `(r_split+1, c_split+1)` 的内角位置。它仍避开分割线, 但会在输出中心附近保留。
  complexity_calibration:
    - 只增加一种新目标位置类别: “中心内角”; 不引入旋转、翻转或颜色映射。
    - 两控制像素样例继续高频出现, 这样人眼能从少量样例中看到“原本会消失的那块现在保留在中心”。
    - 中心目标固定为相对 split 的 4 种离散内角之一, 自由度低, 不会造成匹配歧义。
  implicit_assumptions_audit:
    - 原 main 的 Stage 1 射线只依赖输入中的 split-line fg, 与 D16 无关, 可保持。
    - 原 main 的 split-line 保留循环假设“所有不在 split line 的片段都会被另行处理”; 仍成立。
    - 原 main 的关键旧假设是 `if not (touches_edges(...) & E): continue`, 即非接壤象限应删除。
      该假设在变体中失效, 必须改成“送往中心内角”。
    - 原 generator 的 `vanish_budget` / `vanish_fg` 假设该象限只在输入出现、输出删除; 该路径必须移除。
    - 原 generator 的输入放置方式已经把每个象限都朝 split 贴齐以保证连通性。
      这对新中心目标仍然安全, 因为中心输出同样要求片段留在各自象限并贴近 split。
  placement_legality:
    - 中心片段仍留在自己的象限内, 且与 split line 至少相隔 1 格, 因而不会覆盖 ctrl 射线或 split-line fg。
    - generator 继续限制 sprite bbox 为 5-8 x 8-10, margin >= 4, 并检查每个子区域面积足以容纳最少 3 像素片段。
    - 所有片段输入时组成单连通精灵; 输出时各片段落点互不重叠, 因为它们仍分属不同象限/角落。
  visual_observability:
    - 变体的可见差异集中在双控制样例: 输出中心附近出现原题不会保留的一块 fg。
    - 该块保留在 split 交叉点旁边, 与“去角落”的其他片段形成稳定对比, 2-3 个样例即可观察到。
  cleanliness:
    - 中心片段不落在 split line 上, 避免与静止 fg 或 ctrl 射线叠层。
    - 颜色仍只有 bg/fg/ctrl 三色, 视觉角色清晰。
  color_inference:
    - 无新增颜色规则; 颜色只承担原题角色, 不依赖颜色值做排序。
  main_new_assumptions:
    - solver 只在同时存在 `r_split` 和 `c_split` 时才会把某区域送往中心; generator 保证非接壤区域只会出现在此情形。
    - generator 保证每个区域至少有 3 个像素且输出中心目标不会越 split line, 因此 solver 的平移不会造成越界或意外覆盖。
"""


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
    return str(dh), str(dv)


def _translate_to_outer_corner(pts: list[tuple[int, int]] | set[tuple[int, int]], dh: str, dv: str, H: int, W: int) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int(0 - rmin) if dh == "T" else int((H - 1) - rmax)
    dc = int(0 - cmin) if dv == "L" else int((W - 1) - cmax)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _translate_to_center_inner_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    hs: str,
    vs: str,
    r_split: int,
    c_split: int,
) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int((r_split - 1) - rmax) if hs == "T" else int((r_split + 1) - rmin)
    dc = int((c_split - 1) - cmax) if vs == "L" else int((c_split + 1) - cmin)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _target_points_for_region(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> set[tuple[int, int]]:
    if _touches_edges(hs, vs) & E:
        dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
        return _translate_to_outer_corner(pts, dh, dv, H, W)

    assert r_split is not None and c_split is not None
    return _translate_to_center_inner_corner(pts, hs, vs, r_split, c_split)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut a sprite along border-control rays and slide pieces to corners or, for the
    unique non-touching quadrant under two controls, to the split intersection.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(int(bg))
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]

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

    out = np.full((H, W), int(bg), dtype=int)

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

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            out[int(r), int(c)] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and int(c) == int(c_split)) or (r_split is not None and int(r) == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if int(r) < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if int(c) < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(r), int(c)))

    for (hs, vs), pts in regions.items():
        for rr, cc in _target_points_for_region(pts, hs, vs, E, H, W, r_split, c_split):
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

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

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

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
        r0 = sprite_r0 if hs in ("T", "A") else int(r_split) + 1
        r1 = (int(r_split) - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1
        c0 = sprite_c0 if vs in ("L", "A") else int(c_split) + 1
        c1 = (int(c_split) - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1
        return r0, r1, c0, c1

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        if (r1 - r0 + 1) * (c1 - c0 + 1) < 3:
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

    if budget_pieces < len(all_regions) * 3:
        return None

    piece_budgets: list[int] = []
    remaining = budget_pieces
    for i, reg in enumerate(all_regions):
        r0, r1, c0, c1 = sub_bbox(*reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        remaining_slots = len(all_regions) - i - 1
        min_rest = remaining_slots * 3
        hi_px = min(max_area, remaining - min_rest)
        if hi_px < 3:
            return None
        if i == len(all_regions) - 1:
            px = remaining
        else:
            avg = remaining // (len(all_regions) - i)
            lo_px = max(3, avg - 2)
            hi_band = min(hi_px, avg + 2)
            if lo_px > hi_band:
                lo_px = 3
                hi_band = hi_px
            if lo_px > hi_band:
                return None
            px = random.randint(lo_px, hi_band)
        if px > max_area:
            return None
        piece_budgets.append(px)
        remaining -= px

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}

    for idx, reg in enumerate(all_regions):
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

        inp_pts = frozenset((r + inp_r0, c + inp_c0) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None

        input_pieces[reg] = inp_pts

        out_pts = _target_points_for_region(list(inp_pts), hs, vs, E, H, W, r_split, c_split)
        for r, c in out_pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                return None
            if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                return None
            if (r, c) in split_pixels:
                return None
        output_pieces[reg] = out_pts

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

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_inp_fg:
        inp[r, c] = fg
    for cr, cc in ctrl_pts:
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    occupied_out: set[tuple[int, int]] = set()
    for pts in output_pieces.values():
        if occupied_out & pts:
            return None
        occupied_out |= pts
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

    fg_all = np.array(list(all_inp_fg))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(all_inp_fg) / (final_h * final_w)
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
