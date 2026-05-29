# Combo variant of 4a21e3da (index 16)
# combo_spec: D10/0+D16/1
# Name: 边缘中点对齐+消失区域移到中心

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous

"""
设计分析

a) 具体化与行为追踪
- 原题保持不变的骨架:
  输入仍是一个近中心、单连通的 fg 精灵, 以及 2 个边缘 ctrl 像素; ctrl 定义一条纵切线和一条横切线。
  射线仍只在背景上着色, 分割线上的 fg 仍保持原位。
- 融合后的唯一实质变化发生在 Stage 2 的片段落点:
  1. 与控制边接壤的区域不再去外角, 而是去“对应控制边的中段”。
     这里采用 D10/0 中已经验证过的离散化方案:
     - 若该区域接触 top/bottom 控制边, 则贴该横边, 并在该边对应的可用列段内做整数居中。
     - 否则贴 left/right 控制边, 并在该边对应的可用行段内做整数居中。
     这样可以避免 TL 之类同时接壤两条控制边时的二义性; 规则固定为“横边优先, 否则竖边”。
  2. 唯一不接壤控制边的象限不再消失, 而是移到 split 交叉点附近的中心区域。
     为了让“中心”可离散实现且不覆盖 split-line fg / ctrl 射线, 这里把中心区域具体化为:
     该片段的 bbox 贴到属于自己象限的中心内角上, 即紧邻 `(r_split, c_split)` 但与两条 split
     都保留 1 格间隔。例:
     - 非接壤象限为 BR 时, 片段贴到 `(r_split+1, c_split+1)` 一侧。
- 小例子:
  若 ctrl 在 top 与 left, 四象限分别是 TL/TR/BL/BR, 其中 BR 不接壤控制边。
  原题:
  - TL -> 左上外角
  - TR -> 右上外角
  - BL -> 左下外角
  - BR -> 消失
  融合后:
  - TL -> top 边的左半段中部
  - TR -> top 边的右半段中部
  - BL -> left 边的下半段中部
  - BR -> split 交点右下侧的中心内角
  差异非常明确: 输出不再是“三角落+空中心”, 而是“两条边的中部片段 + 一个中心片段”。

b) 复杂度校准
- 为了让两种新规则都在少量样例里稳定出现, generator 固定生成 2 个 ctrl 像素。
  这样每个样例都会同时出现:
  - 至少 3 个“接壤控制边”的保留片段, 用于观察边缘中点对齐;
  - 1 个“原本会消失”的中心片段, 用于观察 D16/1 的新去向。
- 目标位置自由度仍然很低:
  - 保留片段只有 4 种候选控制边, 且由接壤关系唯一决定;
  - 中心片段只有 4 种中心内角, 也由其象限唯一决定。
  没有引入旋转、翻转、颜色映射或连续参数拟合, 人类可从 2-3 组样例归纳。

c) 隐式假设审查
- 原 main 中 Stage 1 的前提是“ctrl 射线只取决于输入里 split-line fg 的最远端位置”; 该前提未变。
- 原 split-line 保留循环假设“分割线上的 fg 不参与平移”; 该前提未变。
- 原 Stage 2 的关键旧假设有两条:
  1. `touches_edges(...) & E` 为空时直接 continue, 即非接壤象限应删除。
     融合后不成立, 必须改成“送往中心内角”。
  2. 保留片段的目标总是外角, 所以可用 `dr/dc` 直接把 bbox 推到 0 或 H-1/W-1。
     融合后也不成立, 必须先选目标边段, 再按 bbox 尺寸做整数居中。
- 叠加后的新前提:
  1. 若片段走边缘中段, 则其 bbox 必须能放进对应边段。
  2. 若片段走中心内角, 则其 bbox 不能跨过 split line, 且不能覆盖 split-line fg。
  generator 会显式验证这两个前提。

d) 放置合法性
- 边缘中段目标:
  - top/bottom 目标只使用 split 左/右分出的列段;
  - left/right 目标只使用 split 上/下分出的行段。
  这些段彼此天然错开, 不会越 split。
- 中心目标:
  - 片段仍留在自己的象限内;
  - bbox 与 split 行、split 列各保留 1 格间隔, 不覆盖静止 fg 或 ctrl 射线。
- input 仍需是单连通精灵:
  generator 继续把各区域片段朝 split 方向贴齐, 再用 split-line fg 把四块连成一体。
- output 还需避免彼此重叠:
  generator 对所有目标点做占位检查。

e) 视觉可观察性
- 每个样例都含有中心保留片段, 所以“原消失区域去中心”是稳定可见的。
- 同时 generator 还拒绝“所有保留片段都恰好与原题角落位置重合”的退化样本。
  因而解题者能看到保留片段虽然仍贴边, 但显著离开角落, 改为落在边缘中部。

f) 可视化整洁度
- 颜色仍只用 bg/fg/ctrl 三色, 角色单纯。
- 中心片段与 split-line fg 不重叠, 与边缘片段也分层分区:
  边缘片段在外围, 中心片段在 split 交点附近, 画面结构清楚。

g) 变化的可推断性
- 本融合不引入颜色变化, 所有推断都来自几何位置关系:
  “接壤控制边 -> 去对应边缘中部”
  “不接壤控制边 -> 去中心”
  这两条规则都可从少量样例中直接读出。

main() 的新隐式假设总结
- 输入中 ctrl 仍是边界上唯一的非 bg 颜色, fg 是另一种非 bg 颜色。
- 每个 split 后区域里至多只有一块真正需要整体平移的 fg 片段。
- 当存在非接壤区域时, 必然同时存在 `r_split` 与 `c_split`; generator 固定生成双控制样例保证这一点。
- 边缘中段/中心内角目标都能容纳对应 bbox; generator 通过构造和验证保证。
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


def _move_piece_to_center_inner_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    hs: str,
    vs: str,
    r_split: int,
    c_split: int,
) -> frozenset[tuple[int, int]]:
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    rmin, rmax = int(min(rs)), int(max(rs))
    cmin, cmax = int(min(cs)), int(max(cs))
    dr = int((r_split - 1) - rmax) if hs == "T" else int((r_split + 1) - rmin)
    dc = int((c_split - 1) - cmax) if vs == "L" else int((c_split + 1) - cmin)
    return frozenset((int(r + dr), int(c + dc)) for r, c in pts)


def _target_points_for_region(
    pts: list[tuple[int, int]] | set[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> frozenset[tuple[int, int]]:
    if _touches_edges(hs, vs) & E:
        moved, _edge, _margins = _move_piece_to_centered_edge(pts, hs, vs, E, H, W, r_split, c_split)
        return moved
    assert r_split is not None and c_split is not None
    return _move_piece_to_center_inner_corner(pts, hs, vs, r_split, c_split)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cut the centered sprite with border-control rays.

    Pieces touching a control edge move to the midpoint-aligned segment of that
    edge; the unique non-touching quadrant under two controls moves to the
    center inner corner near the split intersection.
    """
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
        for rr, cc in _target_points_for_region(pts, hs, vs, E, H, W, r_split, c_split):
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
    return frozenset((int(r - min(rs)), int(c - min(cs))) for r, c in cells)


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    # Force 2 controls so every generated sample exhibits both fused rules.
    ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]

    c_split = random.randint(int(W * 0.4), int(W * 0.6))
    r_split = random.randint(int(H * 0.4), int(H * 0.6))
    ctrl_pts = [
        (0 if ctrl_edges[0] == "top" else H - 1, c_split),
        (r_split, 0 if ctrl_edges[1] == "left" else W - 1),
    ]
    E = set(ctrl_edges)

    all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    center_region = next(reg for reg in all_regions if reg not in kept)

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    frac_c = random.uniform(0.4, 0.6)
    sprite_c0 = c_split - int(frac_c * sprite_w)
    sprite_c0 = max(4, min(sprite_c0, W - 5 - sprite_w + 1))
    if sprite_c0 > W - 5 - sprite_w + 1:
        return None
    if c_split <= sprite_c0 or c_split >= sprite_c0 + sprite_w - 1:
        return None

    frac_r = random.uniform(0.4, 0.6)
    sprite_r0 = r_split - int(frac_r * sprite_h)
    sprite_r0 = max(4, min(sprite_r0, H - 5 - sprite_h + 1))
    if sprite_r0 > H - 5 - sprite_h + 1:
        return None
    if r_split <= sprite_r0 or r_split >= sprite_r0 + sprite_h - 1:
        return None

    sprite_r1 = sprite_r0 + sprite_h - 1
    sprite_c1 = sprite_c0 + sprite_w - 1

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = sprite_r0 if hs == "T" else int(r_split) + 1
        r1 = int(r_split) - 1 if hs == "T" else sprite_r1
        c0 = sprite_c0 if vs == "L" else int(c_split) + 1
        c1 = int(c_split) - 1 if vs == "L" else sprite_c1
        return r0, r1, c0, c1

    for reg in all_regions:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None
        if (r1 - r0 + 1) * (c1 - c0 + 1) < 3:
            return None

    split_pixels: set[tuple[int, int]] = set()
    n_v = random.randint(2, 4)
    rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
    if len(rows) < n_v:
        return None
    for r in random.sample(rows, n_v):
        split_pixels.add((r, int(c_split)))

    n_h = random.randint(2, 4)
    cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
    if len(cols) < n_h:
        return None
    for c in random.sample(cols, n_h):
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

    if remaining != 0 or any(px < 3 for px in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    centered_diff_count = 0

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
        inp_r0 = int(r1 - ph + 1) if hs == "T" else int(r0)
        inp_c0 = int(c1 - pw + 1) if vs == "L" else int(c0)
        inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)

        for r, c in inp_pts:
            if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
        input_pieces[reg] = inp_pts

        if reg in kept:
            centered_pts, _edge, margins = _move_piece_to_centered_edge(inp_pts, hs, vs, E, H, W, r_split, c_split)
            if min(margins) < 0:
                return None
            dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
            corner_pts = _move_piece_to_corner(inp_pts, dh, dv, H, W)
            if centered_pts != corner_pts:
                centered_diff_count += 1
            out_pts = centered_pts
        else:
            assert reg == center_region
            out_pts = _move_piece_to_center_inner_corner(inp_pts, hs, vs, r_split, c_split)

        for r, c in out_pts:
            if not (0 <= r < H and 0 <= c < W):
                return None
            if (vs == "L" and c >= c_split) or (vs == "R" and c <= c_split):
                return None
            if (hs == "T" and r >= r_split) or (hs == "B" and r <= r_split):
                return None
            if (r, c) in split_pixels:
                return None
        output_pieces[reg] = out_pts

    if centered_diff_count < 2:
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

    actual_density = len(all_inp_fg) / (ah * aw)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if center_region not in output_pieces:
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
