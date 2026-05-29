# Combo variant of 4a21e3da (index 30)
# combo_spec: D8/0+D16/1
# Name: 三控制像素+消失区域移到中心

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
design_analysis:
  concretization:
    input:
      - 固定 3 个 ctrl 像素。
      - 其中 2 个来自同一类型边缘, 因而形成两条同向切割线; 第 3 个来自另一类型边缘, 形成交叉切割。
      - 因此精灵会被切成 6 个矩形子区: 要么是 2 行 x 3 列, 要么是 3 行 x 2 列。
      - 分割线上的 fg 仍静止, ctrl 射线仍只在 bg 上间歇绘制。
    output:
      - 四个真正的“角区”仍按原题去四个外角。
      - 两个位于双切割线之间的“中条区”不再消失, 而是被平移到分割交汇附近的中心条带。
      - 若退化到 1 横 1 竖的普通十字切割, 则沿用 D16/1: 唯一不接壤象限去中心内角。
    trace_example: |
      例: ctrl 在 top/bottom/left, 产生两条竖切列 c1<c2 和一条横切行 r。
      原题代码只支持 1 条竖线 + 1 条横线, 所以会把“最后读到的竖线”当成唯一 c_split, 其余分区信息丢失。
      融合后应把区域显式写成:
        TL | TM | TR
        ---+----+---
        BL | BM | BR
      其中 TL/TR/BL/BR 去四角; TM/BM 去中心条带(位于 c1,c2 之间且紧贴 r 上下)。
      差异点不只是“多一条线”, 更是“中间条带片段有第二目标: 中心, 而非角落/消失”。
  complexity_calibration:
    - 中心目标只新增为两种稳定条带形态:
      - 竖双切时: top-middle / bottom-middle
      - 横双切时: middle-left / middle-right
    - 不引入旋转、翻转、颜色变换; 仍然只是平移, 观察负担可控。
    - 生成器强制 6 个区域都有 fg, 确保样例不会退化成原题的 2 区/4 区情形。
  implicit_assumptions_audit:
    - 原 main 假设“至多一条竖线和一条横线”; 融合后必须改成 `v_splits` / `h_splits` 列表。
    - 原 main 用 `(c == c_split) or (r == r_split)` 判定静止 fg; 融合后必须扩展为“位于任一切割线”。
    - 原 Stage 2 只用 `(T/B) x (L/R)` 四象限键; 融合后需要支持含 `M` 的 6 区域键。
    - 原 `if not touches_edges(...): continue` 隐含“非接壤区域应消失”。
      D16/1 已经打破这点; 在本融合里进一步要求中条区始终去中心。
    - 新 solver 对输入新增的关键假设是:
      - 3 个 ctrl 中恰有一维有 2 条切割线, 另一维有 1 条切割线;
      - 中条区的宽/高足以容纳至少 3 像素片段;
      - split-line fg 的摆放足以把 6 个区连接成单连通输入。
      generator 会逐项保证这些条件。
  placement_legality:
    - 生成器先固定 sprite bbox 与各子区尺寸, 再在每个子区内生成连通片段, 因而不会越区。
    - 中心目标只放在双切割线形成的内部条带, 且始终与切割线保持至少 1 格距离, 不覆盖 ctrl 射线或静止 fg。
    - 所有输出片段在写入前都会做越界/重叠检查。
  visual_observability:
    - 每个样本都固定出现 3 个 ctrl 和 6 个有像素的区域, 人眼能直接看出“切得更多”。
    - 输出始终同时出现:
      - 四角中的多个角落片段
      - 中心附近的 2 个片段
      这样 2-3 组样例就能归纳出“角区去角落, 中条区去中心”的双目标规则。
  cleanliness:
    - 只有 bg/fg/ctrl 三色。
    - split-line fg 与中心片段分层明确: split-line fg 永远在切割线上, 中心片段永远离切割线 1 格。
    - 中心区域最多容纳 2 个片段, 并按上下或左右分离, 不会堆成无法辨识的一团。
  color_inference:
    - 无颜色映射; 颜色只表达角色。
  main_new_assumptions:
    - 若某区域标签含 `M`, solver 直接把它解释为“中心条带目标”。
    - 若是一横一竖的普通十字分割且某象限不接壤控制边, solver 将其送往中心内角。
    - generator 保证不会出现 `M x M` 双中间区, 因为总共只有 3 条切割线。
"""


def _parse_controls(
    g: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[int], list[int], set[str]]:
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})
    border_cols = set(map(int, g[0, :])) | set(map(int, g[-1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, -1]))
    border_cols.discard(bg)
    assert border_cols
    ctrl = int(sorted(border_cols)[0])
    fg = int(next(c for c in nonbg if int(c) != ctrl))

    ctrl_pts = [(int(r), int(c)) for r, c in np.argwhere(g == ctrl)]
    E: set[str] = set()
    v_splits: list[int] = []
    h_splits: list[int] = []
    for r, c in ctrl_pts:
        if r == 0:
            E.add("top")
            v_splits.append(c)
        elif r == g.shape[0] - 1:
            E.add("bottom")
            v_splits.append(c)
        elif c == 0:
            E.add("left")
            h_splits.append(r)
        elif c == g.shape[1] - 1:
            E.add("right")
            h_splits.append(r)

    return ctrl, fg, ctrl_pts, sorted(set(v_splits)), sorted(set(h_splits)), E


def _touches_edges(row_band: str, col_band: str) -> set[str]:
    te: set[str] = set()
    if row_band in ("T", "A"):
        te.add("top")
    if row_band in ("B", "A"):
        te.add("bottom")
    if col_band in ("L", "A"):
        te.add("left")
    if col_band in ("R", "A"):
        te.add("right")
    return te


def _band(index: int, splits: list[int], low: str, high: str) -> str:
    if not splits:
        return "A"
    if len(splits) == 1:
        return low if index < splits[0] else high
    s0, s1 = splits
    if index < s0:
        return low
    if index > s1:
        return high
    return "M"


def _pick_row_dest(row_band: str, E: set[str], h_splits: list[int]) -> str:
    if row_band in ("T", "B"):
        return row_band
    if h_splits:
        return "T" if "top" in E else "B"
    if "top" in E and "bottom" not in E:
        return "T"
    if "bottom" in E and "top" not in E:
        return "B"
    return "T"


def _pick_col_dest(col_band: str, E: set[str], v_splits: list[int]) -> str:
    if col_band in ("L", "R"):
        return col_band
    if v_splits:
        return "L" if "left" in E else "R"
    if "left" in E and "right" not in E:
        return "L"
    if "right" in E and "left" not in E:
        return "R"
    return "L"


def _translate_to_outer_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    row_dest: str,
    col_dest: str,
    H: int,
    W: int,
) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = -rmin if row_dest == "T" else (H - 1 - rmax)
    dc = -cmin if col_dest == "L" else (W - 1 - cmax)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _translate_to_center_inner_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    row_band: str,
    col_band: str,
    h_splits: list[int],
    v_splits: list[int],
) -> set[tuple[int, int]]:
    assert len(h_splits) == 1 and len(v_splits) == 1
    r_split = h_splits[0]
    c_split = v_splits[0]
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    dr = (r_split - 1 - rmax) if row_band == "T" else (r_split + 1 - rmin)
    dc = (c_split - 1 - cmax) if col_band == "L" else (c_split + 1 - cmin)
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _translate_to_center_strip(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    row_band: str,
    col_band: str,
    h_splits: list[int],
    v_splits: list[int],
) -> set[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    ph = rmax - rmin + 1
    pw = cmax - cmin + 1

    if col_band == "M":
        assert len(v_splits) == 2 and len(h_splits) == 1
        inner_c0 = v_splits[0] + 1
        inner_c1 = v_splits[1] - 1
        avail_w = inner_c1 - inner_c0 + 1
        dc = inner_c0 + (avail_w - pw) // 2 - cmin
        r_split = h_splits[0]
        if row_band == "T":
            dr = (r_split - 1) - rmax
        else:
            dr = (r_split + 1) - rmin
        return {(int(r + dr), int(c + dc)) for r, c in pts}

    assert row_band == "M"
    assert len(h_splits) == 2 and len(v_splits) == 1
    inner_r0 = h_splits[0] + 1
    inner_r1 = h_splits[1] - 1
    avail_h = inner_r1 - inner_r0 + 1
    dr = inner_r0 + (avail_h - ph) // 2 - rmin
    c_split = v_splits[0]
    if col_band == "L":
        dc = (c_split - 1) - cmax
    else:
        dc = (c_split + 1) - cmin
    return {(int(r + dr), int(c + dc)) for r, c in pts}


def _target_points_for_region(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    row_band: str,
    col_band: str,
    E: set[str],
    H: int,
    W: int,
    h_splits: list[int],
    v_splits: list[int],
    *,
    move_middle_to_center: bool,
) -> set[tuple[int, int]]:
    if move_middle_to_center and ("M" in (row_band, col_band)):
        return _translate_to_center_strip(pts, row_band, col_band, h_splits, v_splits)

    if _touches_edges(row_band, col_band) & E:
        row_dest = _pick_row_dest(row_band, E, h_splits)
        col_dest = _pick_col_dest(col_band, E, v_splits)
        return _translate_to_outer_corner(pts, row_dest, col_dest, H, W)

    if move_middle_to_center and len(h_splits) == 1 and len(v_splits) == 1:
        return _translate_to_center_inner_corner(pts, row_band, col_band, h_splits, v_splits)

    return set()


def _main_core(grid: np.ndarray, *, move_middle_to_center: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    ctrl, fg, ctrl_pts, v_splits, h_splits, E = _parse_controls(g)

    out = np.full((H, W), bg, dtype=int)

    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = max(rows)
                for rr in range(0, stop + 1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = min(rows)
                for rr in range(H - 1, stop - 1, -1):
                    if g[rr, c] == bg:
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = max(cols)
                for cc in range(0, stop + 1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = min(cols)
                for cc in range(W - 1, stop - 1, -1):
                    if g[r, cc] == bg:
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == fg):
        r = int(r)
        c = int(c)
        if c in v_splits or r in h_splits:
            out[r, c] = fg
            continue
        row_band = _band(r, h_splits, "T", "B")
        col_band = _band(c, v_splits, "L", "R")
        regions.setdefault((row_band, col_band), []).append((r, c))

    for (row_band, col_band), pts in regions.items():
        for rr, cc in _target_points_for_region(
            pts,
            row_band,
            col_band,
            E,
            H,
            W,
            h_splits,
            v_splits,
            move_middle_to_center=move_middle_to_center,
        ):
            if 0 <= rr < H and 0 <= cc < W:
                out[rr, cc] = fg

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Three-control fusion of 4a21e3da:
    - draw all border rays;
    - keep fg on any split line in place;
    - outer corner regions slide to corners;
    - middle-strip / otherwise non-touching regions slide to the center zone.
    """
    return _main_core(grid, move_middle_to_center=True)


def _corner_only_baseline(grid: np.ndarray) -> np.ndarray:
    """Ablation used in validation: identical solver but without center relocation."""
    return _main_core(grid, move_middle_to_center=False)


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


def _choose_layout(H: int, W: int) -> dict[str, object]:
    mode = random.choice(["vpair", "hpair"])

    if mode == "vpair":
        top_h = bottom_h = random.choice([3, 3, 2])
        if top_h == 2 and bottom_h == 2:
            top_h = 3
        widths = random.choice(
            [
                (2, 2, 2),
                (2, 2, 3),
                (2, 3, 2),
                (3, 2, 2),
                (2, 3, 3),
                (3, 2, 3),
                (3, 3, 2),
            ]
        )
        left_w, mid_w, right_w = widths
        sprite_h = top_h + 1 + bottom_h
        sprite_w = left_w + 1 + mid_w + 1 + right_w
        sprite_r0 = random.randint(4, H - 5 - sprite_h)
        sprite_c0 = random.randint(4, W - 5 - sprite_w)
        sprite_r1 = sprite_r0 + sprite_h - 1
        sprite_c1 = sprite_c0 + sprite_w - 1
        r_split = sprite_r0 + top_h
        c_left = sprite_c0 + left_w
        c_right = c_left + 1 + mid_w
        third_edge = random.choice(["left", "right"])
        ctrl_pts = [(0, c_left), (H - 1, c_right), (r_split, 0 if third_edge == "left" else W - 1)]
        E = {"top", "bottom", third_edge}
        h_splits = [r_split]
        v_splits = [c_left, c_right]
        all_regions = [("T", "L"), ("T", "M"), ("T", "R"), ("B", "L"), ("B", "M"), ("B", "R")]
    else:
        left_w, right_w = random.choice([(3, 4), (4, 3), (4, 4)])
        top_h = mid_h = bottom_h = 2
        sprite_h = top_h + 1 + mid_h + 1 + bottom_h
        sprite_w = left_w + 1 + right_w
        sprite_r0 = random.randint(4, H - 5 - sprite_h)
        sprite_c0 = random.randint(4, W - 5 - sprite_w)
        sprite_r1 = sprite_r0 + sprite_h - 1
        sprite_c1 = sprite_c0 + sprite_w - 1
        r_top = sprite_r0 + top_h
        r_bottom = r_top + 1 + mid_h
        c_split = sprite_c0 + left_w
        third_edge = random.choice(["top", "bottom"])
        ctrl_pts = [(r_top, 0), (r_bottom, W - 1), (0 if third_edge == "top" else H - 1, c_split)]
        E = {"left", "right", third_edge}
        h_splits = [r_top, r_bottom]
        v_splits = [c_split]
        all_regions = [("T", "L"), ("T", "R"), ("M", "L"), ("M", "R"), ("B", "L"), ("B", "R")]

    assert sprite_r0 >= 4 and sprite_c0 >= 4
    assert sprite_r1 <= H - 5 and sprite_c1 <= W - 5

    return {
        "mode": mode,
        "sprite_r0": sprite_r0,
        "sprite_r1": sprite_r1,
        "sprite_c0": sprite_c0,
        "sprite_c1": sprite_c1,
        "ctrl_pts": ctrl_pts,
        "E": E,
        "h_splits": h_splits,
        "v_splits": v_splits,
        "all_regions": all_regions,
    }


def _sub_bbox(layout: dict[str, object], row_band: str, col_band: str) -> tuple[int, int, int, int]:
    sprite_r0 = int(layout["sprite_r0"])
    sprite_r1 = int(layout["sprite_r1"])
    sprite_c0 = int(layout["sprite_c0"])
    sprite_c1 = int(layout["sprite_c1"])
    h_splits = list(layout["h_splits"])
    v_splits = list(layout["v_splits"])

    if row_band == "T":
        r0, r1 = sprite_r0, h_splits[0] - 1
    elif row_band == "M":
        r0, r1 = h_splits[0] + 1, h_splits[1] - 1
    elif row_band == "B":
        r0, r1 = h_splits[-1] + 1, sprite_r1
    else:
        r0, r1 = sprite_r0, sprite_r1

    if col_band == "L":
        c0, c1 = sprite_c0, v_splits[0] - 1
    elif col_band == "M":
        c0, c1 = v_splits[0] + 1, v_splits[1] - 1
    elif col_band == "R":
        c0, c1 = v_splits[-1] + 1, sprite_c1
    else:
        c0, c1 = sprite_c0, sprite_c1

    return r0, r1, c0, c1


def _build_split_pixels(layout: dict[str, object]) -> set[tuple[int, int]]:
    h_splits = list(layout["h_splits"])
    v_splits = list(layout["v_splits"])
    split_pixels: set[tuple[int, int]] = set()

    if len(v_splits) == 2:
        r = h_splits[0]
        c0, c1 = v_splits
        split_pixels |= {
            (r - 1, c0),
            (r + 1, c0),
            (r - 1, c1),
            (r + 1, c1),
            (r, c0 + 1),
            (r, c1 - 1),
            (r, c0 + 2) if c1 - c0 >= 4 else (r, c0 + 1),
            (r, c1 - 2) if c1 - c0 >= 4 else (r, c1 - 1),
        }
    else:
        c = v_splits[0]
        r0, r1 = h_splits
        split_pixels |= {
            (r0, c - 1),
            (r0, c + 1),
            (r1, c - 1),
            (r1, c + 1),
            (r0 + 1, c),
            (r1 - 1, c),
            (r0 + 2, c) if r1 - r0 >= 4 else (r0 + 1, c),
            (r1 - 2, c) if r1 - r0 >= 4 else (r1 - 1, c),
        }

    return set(split_pixels)


def _place_piece_in_region(
    piece: frozenset[tuple[int, int]],
    layout: dict[str, object],
    row_band: str,
    col_band: str,
) -> frozenset[tuple[int, int]]:
    r0, r1, c0, c1 = _sub_bbox(layout, row_band, col_band)
    sub_h = r1 - r0 + 1
    sub_w = c1 - c0 + 1
    ph = max(r for r, _ in piece) + 1
    pw = max(c for _, c in piece) + 1
    mode = str(layout["mode"])

    if row_band == "T":
        inp_r0 = r1 - ph + 1
    elif row_band == "B":
        inp_r0 = r0
    elif row_band == "M" and mode == "hpair":
        inp_r0 = r0 if col_band == "L" else r1 - ph + 1
    else:
        inp_r0 = r0 + (sub_h - ph) // 2

    if col_band == "L":
        inp_c0 = c1 - pw + 1
    elif col_band == "R":
        inp_c0 = c0
    elif col_band == "M" and mode == "vpair":
        inp_c0 = c0 if row_band == "T" else c1 - pw + 1
    else:
        inp_c0 = c0 + (sub_w - pw) // 2

    return frozenset((r + inp_r0, c + inp_c0) for r, c in piece)


def _allocate_piece_sizes(
    layout: dict[str, object],
    split_pixels: set[tuple[int, int]],
) -> dict[tuple[str, str], int] | None:
    all_regions = list(layout["all_regions"])
    area = (int(layout["sprite_r1"]) - int(layout["sprite_r0"]) + 1) * (int(layout["sprite_c1"]) - int(layout["sprite_c0"]) + 1)
    min_piece_budget = len(all_regions) * 3
    lo_total = max(len(split_pixels) + min_piece_budget, int(0.42 * area))
    hi_total = min(int(0.58 * area), area - 4)
    if lo_total > hi_total:
        return None
    target_total = random.randint(lo_total, hi_total)
    piece_budget = target_total - len(split_pixels)

    max_by_region = {}
    max_total = 0
    for reg in all_regions:
        r0, r1, c0, c1 = _sub_bbox(layout, *reg)
        max_area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if max_area < 3:
            return None
        max_by_region[reg] = max_area
        max_total += max_area
    if piece_budget > max_total:
        return None

    sizes: dict[tuple[str, str], int] = {}
    remaining = piece_budget
    for i, reg in enumerate(all_regions):
        remaining_slots = len(all_regions) - i - 1
        min_rest = remaining_slots * 3
        max_here = min(max_by_region[reg], remaining - min_rest)
        if max_here < 3:
            return None
        if i == len(all_regions) - 1:
            px = remaining
        else:
            avg = remaining // (len(all_regions) - i)
            lo = max(3, avg - 1)
            hi = min(max_here, avg + 1)
            if lo > hi:
                lo, hi = 3, max_here
            px = random.randint(lo, hi)
        sizes[reg] = px
        remaining -= px

    if remaining != 0:
        return None
    return sizes


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(19, 20)
    bg, fg, ctrl_color = 1, 7, 2

    layout = _choose_layout(H, W)
    split_pixels = _build_split_pixels(layout)
    size_plan = _allocate_piece_sizes(layout, split_pixels)
    if size_plan is None:
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], set[tuple[int, int]]] = {}

    for reg in list(layout["all_regions"]):
        row_band, col_band = reg
        r0, r1, c0, c1 = _sub_bbox(layout, row_band, col_band)
        sub_h = r1 - r0 + 1
        sub_w = c1 - c0 + 1
        piece = _grow_piece(sub_h, sub_w, size_plan[reg])
        if len(piece) < 3:
            return None
        inp_pts = _place_piece_in_region(piece, layout, row_band, col_band)

        for r, c in inp_pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if (r, c) in split_pixels:
                return None
        input_pieces[reg] = inp_pts

        out_pts = _target_points_for_region(
            list(inp_pts),
            row_band,
            col_band,
            set(layout["E"]),
            H,
            W,
            list(layout["h_splits"]),
            list(layout["v_splits"]),
            move_middle_to_center=True,
        )
        if not out_pts:
            return None
        output_pieces[reg] = out_pts

    all_inp_fg: set[tuple[int, int]] = set(split_pixels)
    for pts in input_pieces.values():
        if all_inp_fg & pts:
            return None
        all_inp_fg |= set(pts)

    fg_arr = np.array(list(all_inp_fg))
    bbox_h = int(fg_arr[:, 0].max() - fg_arr[:, 0].min() + 1)
    bbox_w = int(fg_arr[:, 1].max() - fg_arr[:, 1].min() + 1)
    if not (5 <= bbox_h <= 8 and 8 <= bbox_w <= 10):
        return None

    mask = np.zeros((H, W), dtype=int)
    for r, c in all_inp_fg:
        mask[r, c] = 1
    if not is_contiguous(mask):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in all_inp_fg:
        inp[r, c] = fg
    for cr, cc in list(layout["ctrl_pts"]):
        inp[cr, cc] = ctrl_color

    out = np.full((H, W), bg, dtype=int)
    occupied_out: set[tuple[int, int]] = set()
    for pts in output_pieces.values():
        if occupied_out & pts:
            return None
        if pts & split_pixels:
            return None
        occupied_out |= pts
        for r, c in pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            out[r, c] = fg
    for r, c in split_pixels:
        out[r, c] = fg

    for cr, cc in list(layout["ctrl_pts"]):
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
    for cr, cc in list(layout["ctrl_pts"]):
        out[cr, cc] = ctrl_color

    actual_density = len(all_inp_fg) / (bbox_h * bbox_w)
    if not (0.39 <= actual_density <= 0.60):
        return None

    if not np.array_equal(main(inp), out):
        return None

    if np.array_equal(main(inp), _corner_only_baseline(inp)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
