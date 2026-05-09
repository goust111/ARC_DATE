# Combo variant of 4a21e3da (index 33)
# combo_spec: D5/1+D10/0+D16/1
# Name: 镜像翻转+边缘中点对齐+消失区域到中心

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
design_analysis:
  concretization_and_trace:
    - 输入家族保持原题结构: 一个近中心、单连通的 fg 精灵, 两个边缘 ctrl 像素
      （本生成器固定高频使用 2 ctrl, 以稳定暴露“中心保留片段”这一融合特征）, ctrl 定义
      一条竖切线和一条横切线, 分割线上的少量 fg 像素在输出中原地保留。
    - 原题 Stage 2:
      1. 按 split 把非 split-line fg 分到 4 个矩形区域。
      2. 与 ctrl 边接壤的 3 块平移到外角。
      3. 唯一不接壤 ctrl 边的 1 块删除。
    - 融合后 Stage 2:
      1. 仍按同样的区域划分。
      2. 接壤 ctrl 边的片段不再去角落, 而是先按“滑动方向”做单轴镜像:
         去 top/bottom 的片段翻转行序, 去 left/right 的片段翻转列序。
      3. 镜像后的片段贴到对应边段, 但在该边段内做整数居中, 不再贴角。
      4. 唯一不接壤 ctrl 边的片段不消失; 它保持原形, 直接移到网格几何中心。
    - 具体小例子:
      若 ctrl 在 top 和 left, 则 TL/TR/BL 三块保留, BR 是原题中的“消失块”。
      原题:
        TL -> 左上角, TR -> 右上角, BL -> 左下角, BR -> 删除。
      本变体:
        TL -> top 边左半段的中点位置, 且做上下镜像;
        TR -> top 边右半段的中点位置, 且做上下镜像;
        BL -> left 边下半段的中点位置, 且做左右镜像;
        BR -> 保持形状, 放到整张网格中心。
      差异非常稳定: 角落被腾空, 边缘中部出现翻转块, 中心出现原题不会保留的第四块。
  complexity_calibration:
    - 三个改动彼此协同但自由度有限:
      * 变换只有两类: edge-piece 单轴镜像, center-piece 不变形。
      * 目标位只有两类: 对应边段中点 / 全局中心。
      * 颜色逻辑完全不变, 避免多维度同时扩散。
    - 为避免退化:
      * generator 拒绝“镜像后与未镜像完全相同”的保留片段样本。
      * generator 拒绝“边段中点放置与角落放置没有差异”的样本。
      * generator 拒绝“中心块恰好退化成旧 D16 内角贴齐”的样本。
  implicit_assumptions_audit:
    - 原题 `dr/dc` 直接平移到角落, 隐含“片段内部相对坐标保持不变”。
      融合后此假设对保留片段失效, 必须先归一化到 bbox 内再镜像再放置。
    - 原题默认所有保留块的目标都是 corner, 这在 D10 融合后失效。
      现在需要先选“哪条边”, 再在该边可用段里按 bbox 尺寸居中。
    - 原题对非接壤区域 `continue` 删除, 这在 D16 融合后失效。
      现在必须显式把该区域放到中心。
    - 新冲突点:
      * 中心块若与 split-line 静止 fg 过度重叠, 会削弱可观察性。
        生成器因此显式控制 split-line fg 的选点, 优先避开中心输出 bbox。
      * 双接壤区域（如 TL 同时接 top/left）若既可去 top 又可去 left 会产生二义性。
        这里沿用 D10 单变体的“横边优先”: 先看 top/bottom, 否则看 left/right。
  placement_legality:
    - 生成器仍用原题的 centered sprite 约束, 保证输入 bbox 距边界至少 4。
    - 输入四块都朝 split 贴齐, 再由 split-line fg 把它们连成单连通精灵。
    - 输出阶段:
      * 边缘块只会落在彼此互斥的边段中, 不会跨 split line。
      * 中心块放在网格几何中心, 与边缘块天然远离。
      * split-line fg 允许与中心块局部同色叠合, 但生成器优先规避这种情况。
  visual_observability:
    - 从 2-3 组样例中, 人类可直接观察到三条信号:
      1. 角落空了, 片段改为贴边中部。
      2. 边缘片段的轮廓相对输入发生了单轴镜像。
      3. 原本应消失的对角块现在稳定保留在中心。
    - 这些信号都依赖形状和位置, 不依赖颜色数值。
  cleanliness:
    - 仍保持三色: bg / fg / ctrl。
    - ctrl 射线只在 split 行列上出现, 边缘块分散到外圈, 中心块集中在中区, 层次清晰。
  color_inference:
    - 无新增颜色映射; 颜色角色与原题完全一致。
  new_main_assumptions:
    - solver 现在假设边缘保留片段的 bbox 能放进对应边段。
      generator 通过 split 位于 40-60%、片段尺寸小、并在生成后实算目标坐标来保证。
    - solver 现在假设中心块可以安全放到网格中心。
      generator 通过限制中心块预算、检查越界与重叠来保证。
    - solver 仍假设 split-line fg 是输入中的唯一静止 fg; generator 保持该构造不变。
"""


def _parse_grid(grid: np.ndarray):
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != int(ctrl)))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == int(ctrl))]
    E: set[str] = set()
    c_split: int | None = None
    r_split: int | None = None
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            c_split = int(c)
        elif r == H - 1:
            E.add("bottom")
            c_split = int(c)
        elif c == 0:
            E.add("left")
            r_split = int(r)
        elif c == W - 1:
            E.add("right")
            r_split = int(r)

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
    return str(dh), str(dv)


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
            return 0, int(W - 1)
        if vs == "L":
            return 0, int(c_split - 1)
        if vs == "R":
            return int(c_split + 1), int(W - 1)
        return 0, int(W - 1)

    if r_split is None:
        return 0, int(H - 1)
    if hs == "T":
        return 0, int(r_split - 1)
    if hs == "B":
        return int(r_split + 1), int(H - 1)
    return 0, int(H - 1)


def _normalize_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin = int(min(rs))
    cmin = int(min(cs))
    shape = frozenset((int(r - rmin), int(c - cmin)) for r, c in pts)
    h = int(max(r for r, _ in shape) + 1)
    w = int(max(c for _, c in shape) + 1)
    return shape, h, w


def _flip_shape(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    *,
    flip_v: bool,
    flip_h: bool,
) -> frozenset[tuple[int, int]]:
    out = set()
    for r, c in shape:
        rr = int(h - 1 - r) if flip_v else int(r)
        cc = int(w - 1 - c) if flip_h else int(c)
        out.add((int(rr), int(cc)))
    return frozenset(out)


def _place_shape(shape: frozenset[tuple[int, int]], r0: int, c0: int) -> frozenset[tuple[int, int]]:
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _move_piece_to_centered_edge(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
    *,
    mirror: bool,
) -> tuple[frozenset[tuple[int, int]], str, tuple[int, int]]:
    edge = _pick_center_edge(hs, vs, E)
    shape, h, w = _normalize_points(pts)
    if mirror:
        shape = _flip_shape(shape, h, w, flip_v=(edge in ("top", "bottom")), flip_h=(edge in ("left", "right")))

    seg0, seg1 = _segment_for_edge(edge, hs, vs, H, W, r_split, c_split)
    seg_len = int(seg1 - seg0 + 1)

    if edge == "top":
        target_r0 = 0
        target_c0 = int(seg0 + (seg_len - w) // 2)
        margins = (int(target_c0 - seg0), int(seg1 - (target_c0 + w - 1)))
    elif edge == "bottom":
        target_r0 = int(H - h)
        target_c0 = int(seg0 + (seg_len - w) // 2)
        margins = (int(target_c0 - seg0), int(seg1 - (target_c0 + w - 1)))
    elif edge == "left":
        target_r0 = int(seg0 + (seg_len - h) // 2)
        target_c0 = 0
        margins = (int(target_r0 - seg0), int(seg1 - (target_r0 + h - 1)))
    else:
        target_r0 = int(seg0 + (seg_len - h) // 2)
        target_c0 = int(W - w)
        margins = (int(target_r0 - seg0), int(seg1 - (target_r0 + h - 1)))

    return _place_shape(shape, int(target_r0), int(target_c0)), edge, margins


def _move_piece_to_grid_center(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]],
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    shape, h, w = _normalize_points(pts)
    target_r0 = int((H - h) // 2)
    target_c0 = int((W - w) // 2)
    return _place_shape(shape, target_r0, target_c0)


def _move_piece_to_inner_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]],
    hs: str,
    vs: str,
    r_split: int,
    c_split: int,
) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin = int(min(rs))
    rmax = int(max(rs))
    cmin = int(min(cs))
    cmax = int(max(cs))
    dr = int((r_split - 1) - rmax) if hs == "T" else int((r_split + 1) - rmin)
    dc = int((c_split - 1) - cmax) if vs == "L" else int((c_split + 1) - cmin)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _move_piece_to_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
    *,
    mirror: bool,
) -> frozenset[tuple[int, int]]:
    dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
    shape, h, w = _normalize_points(pts)
    if mirror:
        shape = _flip_shape(shape, h, w, flip_v=(dh in ("T", "B")), flip_h=(dv in ("L", "R")))
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return _place_shape(shape, int(r0), int(c0))


def _classify_regions(g: np.ndarray, fg: int, r_split: int | None, c_split: int | None) -> dict[tuple[str, str], list[tuple[int, int]]]:
    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == int(c_split)) or (r_split is not None and rr == int(r_split)):
            continue
        hs = "A"
        vs = "A"
        if r_split is not None:
            hs = "T" if rr < int(r_split) else "B"
        if c_split is not None:
            vs = "L" if cc < int(c_split) else "R"
        regions.setdefault((hs, vs), []).append((int(rr), int(cc)))
    return regions


def _draw_rays(out: np.ndarray, g: np.ndarray, ctrl_pts: list[tuple[int, int]], fg: int, ctrl: int) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == int(get_bg(g)):
                        out[rr, c] = int(ctrl)
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == int(fg)).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == int(get_bg(g)):
                        out[rr, c] = int(ctrl)
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == int(get_bg(g)):
                        out[r, cc] = int(ctrl)
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == int(fg)).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == int(get_bg(g)):
                        out[r, cc] = int(ctrl)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the centered sprite with border-control rays.

    Combo rule:
    - split-line fg stays in place;
    - edge-touching regions survive, mirror along the movement axis, then align to
      the midpoint of the corresponding edge segment;
    - the unique non-touching region under two controls keeps its orientation and
      moves to the grid center.
    """
    g, H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _parse_grid(grid)
    out = np.full((H, W), int(bg), dtype=int)

    _draw_rays(out, g, ctrl_pts, fg, ctrl)
    for r, c in ctrl_pts:
        out[int(r), int(c)] = int(ctrl)

    for r, c in np.argwhere(g == int(fg)):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == int(c_split)) or (r_split is not None and rr == int(r_split)):
            out[rr, cc] = int(fg)

    regions = _classify_regions(g, fg, r_split, c_split)
    for (hs, vs), pts in regions.items():
        if _touches_edges(hs, vs) & E:
            moved, _edge, _margins = _move_piece_to_centered_edge(
                pts, hs, vs, E, H, W, r_split, c_split, mirror=True
            )
        else:
            moved = _move_piece_to_grid_center(pts, H, W)
        for rr, cc in moved:
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = int(r + dr)
                nc = int(c + dc)
                if (int(nr), int(nc)) not in cells:
                    neighbors.add((int(nr), int(nc)))
        valid = [
            (int(nr), int(nc))
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(int(nr), int(nc))}) - min(r for r, _ in cells | {(int(nr), int(nc))}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(int(nr), int(nc))}) - min(c for _, c in cells | {(int(nr), int(nc))}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _bbox_of_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    return int(min(rs)), int(min(cs)), int(max(rs)), int(max(cs))


def _pick_preferred(values: list[int], avoid: set[int]) -> int | None:
    preferred = [int(v) for v in values if int(v) not in avoid]
    if preferred:
        return int(random.choice(preferred))
    if values:
        return int(random.choice(values))
    return None


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = int(S)
    bg, fg, ctrl_color = 1, 7, 2

    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
    E = set(ctrl_edges)
    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if "top" in E else int(H - 1), int(c_split)),
        (int(r_split), 0 if "left" in E else int(W - 1)),
    ]

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    touching = [reg for reg in all_regions if _touches_edges(*reg) & E]
    center_reg = next(reg for reg in all_regions if reg not in touching)

    sprite_h = random.randint(6, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = int(c_split - int(frac_c * sprite_w))
    sprite_c0 = max(4, min(int(sprite_c0), int(W - 5 - sprite_w + 1)))
    if sprite_c0 > int(W - 5 - sprite_w + 1):
        return None
    if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = int(r_split - int(frac_r * sprite_h))
    sprite_r0 = max(4, min(int(sprite_r0), int(H - 5 - sprite_h + 1)))
    if sprite_r0 > int(H - 5 - sprite_h + 1):
        return None
    if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
        return None

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs == "T" else r_split + 1)
        r1 = int(r_split - 1 if hs == "T" else sprite_r1)
        c0 = int(sprite_c0 if vs == "L" else c_split + 1)
        c1 = int(c_split - 1 if vs == "L" else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    areas: dict[tuple[str, str], int] = {}
    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area < 3:
            return None
        areas[reg] = int(area)

    n_v = random.randint(2, 3)
    n_h = random.randint(2, 3)
    n_split = int(n_v + n_h)
    area = int(sprite_h * sprite_w)
    target_total = int(random.uniform(0.43, 0.57) * area)

    center_budget = random.randint(3, min(5, areas[center_reg]))
    remaining = int(target_total - n_split - center_budget)
    if remaining < int(len(touching) * 4):
        return None

    budgets: dict[tuple[str, str], int] = {center_reg: int(center_budget)}
    regs_left = list(touching)
    remaining_touch = int(remaining)
    for idx, reg in enumerate(regs_left):
        max_area = int(areas[reg])
        min_rest = int((len(regs_left) - idx - 1) * 4)
        hi = int(min(max_area, remaining_touch - min_rest))
        if hi < 4:
            return None
        if idx == len(regs_left) - 1:
            px = int(remaining_touch)
        else:
            avg = int(remaining_touch // (len(regs_left) - idx))
            lo = int(max(4, avg - 2))
            hi_band = int(min(hi, avg + 2))
            if lo > hi_band:
                lo = 4
                hi_band = hi
            if lo > hi_band:
                return None
            px = int(random.randint(lo, hi_band))
        if px > max_area:
            return None
        budgets[reg] = int(px)
        remaining_touch -= int(px)

    if remaining_touch != 0:
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    mirror_delta = 0
    corner_delta = 0
    mirrored_piece_count = 0
    centered_piece_count = 0

    for reg in all_regions:
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        piece = _grow_piece(sub_h, sub_w, budgets[reg])
        if len(piece) < 3:
            return None

        ph = int(max(r for r, _ in piece) + 1)
        pw = int(max(c for _, c in piece) + 1)
        inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                return None
            if (hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split)):
                return None
            if (vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split)):
                return None
        input_pieces[reg] = inp_pts

        if reg in touching:
            combo_pts, _edge, margins = _move_piece_to_centered_edge(
                inp_pts, hs, vs, E, H, W, r_split, c_split, mirror=True
            )
            plain_centered_pts, _edge2, _margins2 = _move_piece_to_centered_edge(
                inp_pts, hs, vs, E, H, W, r_split, c_split, mirror=False
            )
            mirror_corner_pts = _move_piece_to_corner(
                inp_pts, hs, vs, E, H, W, r_split, c_split, mirror=True
            )
            if min(margins) < 0:
                return None
            md = int(len(combo_pts ^ plain_centered_pts))
            cd = int(len(combo_pts ^ mirror_corner_pts))
            mirror_delta += int(md)
            corner_delta += int(cd)
            mirrored_piece_count += int(md > 0)
            centered_piece_count += int(cd > 0)
        else:
            combo_pts = _move_piece_to_grid_center(inp_pts, H, W)
            inner_corner_pts = _move_piece_to_inner_corner(inp_pts, hs, vs, r_split, c_split)
            if combo_pts == inner_corner_pts:
                return None

        for r, c in combo_pts:
            if not (0 <= int(r) < H and 0 <= int(c) < W):
                return None
        output_pieces[reg] = combo_pts

    occupied_output: set[tuple[int, int]] = set()
    for pts in output_pieces.values():
        if occupied_output & set(pts):
            return None
        occupied_output |= set(pts)

    if mirror_delta < 4 or mirrored_piece_count < 2:
        return None
    if corner_delta < 6 or centered_piece_count < 2:
        return None

    center_out = output_pieces[center_reg]
    center_r0, center_c0, center_r1, center_c1 = _bbox_of_points(center_out)
    avoid_rows = set(range(int(center_r0), int(center_r1 + 1)))
    avoid_cols = set(range(int(center_c0), int(center_c1 + 1)))

    tl = input_pieces[("T", "L")]
    tr = input_pieces[("T", "R")]
    bl = input_pieces[("B", "L")]
    br = input_pieces[("B", "R")]

    top_rows = sorted(set(r for r, _ in tl) & set(r for r, _ in tr))
    bottom_rows = sorted(set(r for r, _ in bl) & set(r for r, _ in br))
    left_cols = sorted(set(c for _, c in tl) & set(c for _, c in bl))
    right_cols = sorted(set(c for _, c in tr) & set(c for _, c in br))
    if not top_rows or not bottom_rows or not left_cols or not right_cols:
        return None

    v_rows: list[int] = []
    h_cols: list[int] = []
    top_pick = _pick_preferred(top_rows, avoid_rows)
    bottom_pick = _pick_preferred(bottom_rows, avoid_rows)
    left_pick = _pick_preferred(left_cols, avoid_cols)
    right_pick = _pick_preferred(right_cols, avoid_cols)
    if None in {top_pick, bottom_pick, left_pick, right_pick}:
        return None
    v_rows.extend([int(top_pick), int(bottom_pick)])
    h_cols.extend([int(left_pick), int(right_pick)])

    extra_v_candidates = sorted(
        set(range(int(sprite_r0), int(sprite_r1 + 1))) - {int(r_split)} - set(v_rows)
    )
    extra_h_candidates = sorted(
        set(range(int(sprite_c0), int(sprite_c1 + 1))) - {int(c_split)} - set(h_cols)
    )
    if len(v_rows) < n_v:
        pick = _pick_preferred(extra_v_candidates, avoid_rows)
        if pick is None:
            return None
        v_rows.append(int(pick))
    if len(h_cols) < n_h:
        pick = _pick_preferred(extra_h_candidates, avoid_cols)
        if pick is None:
            return None
        h_cols.append(int(pick))

    split_pixels = {(int(r), int(c_split)) for r in v_rows} | {(int(r_split), int(c)) for c in h_cols}
    if len(split_pixels) != len(v_rows) + len(h_cols):
        return None

    all_inp_fg: set[tuple[int, int]] = set()
    for pts in input_pieces.values():
        all_inp_fg |= set(pts)
    all_inp_fg |= split_pixels

    fg_arr = np.array(list(all_inp_fg))
    ah = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    aw = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if ah < 5 or ah > 8 or aw < 8 or aw > 10:
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), int(bg), dtype=int)
    for r, c in all_inp_fg:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc in ctrl_pts:
        if cr == 0:
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(0, int(max(fg_rows)) + 1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cr == H - 1:
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(H - 1), int(min(fg_rows)) - 1, -1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif cc == 0:
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(0, int(max(fg_cols)) + 1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif cc == W - 1:
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(W - 1), int(min(fg_cols)) - 1, -1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    final_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    final_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    actual_density = len(all_inp_fg) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(800):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 800 attempts")
