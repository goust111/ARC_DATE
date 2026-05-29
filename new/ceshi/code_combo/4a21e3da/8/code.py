# Combo variant of 4a21e3da (index 8)
# combo_spec: D5/1+D10/0
# Name: 镜像翻转+边缘中点对齐
#
# 设计分析
# a) 具体化与行为追踪
#    - 输入家族保持原题不变: 一个近中心的单连通 fg 精灵, 以及 1-2 个位于边界的 ctrl 像素。
#      ctrl 只在输出里沿自身行/列向内发射间歇射线; 分割线上的 fg 保持原位。
#    - 原题 Stage 2: 与控制边接壤的片段保留, 直接平移到对应角落; 不接壤片段消失。
#    - 本融合版 Stage 2:
#      1. 先决定片段最终依附的“目标边”。规则沿用 D10/0: 若片段同时接触横边和竖边, 固定横边优先;
#         否则取唯一接触到的有效控制边。
#      2. 再根据目标边决定镜像轴。若目标边是 top/bottom, 片段只做上下翻转; 若目标边是 left/right,
#         片段只做左右翻转。因为现在运动方向只有一个主轴, 不再像 D5/1 落角时那样双轴同时翻转。
#      3. 最后不落角, 而是把翻转后的片段 bbox 贴到目标边, 并在该边被 split 划出的可用边段内做整数居中。
#    - 小例子: 单个 top 控制点给出一条竖切线。原题左、右两块分别移到左上角和右上角; D10/0 会改成
#      两块都贴顶边, 但在左右半段内水平居中; 本融合版还要先把两块在各自 bbox 内做上下翻转, 再贴到
#      顶边半段中点。于是观察者必须同时看出“不是原样平移”与“不是落角”这两件事。
# b) 复杂度校准
#    - 仍只有 1-2 个控制点, 至多 4 个切分区域, 没有颜色分支或对象匹配分支, 复杂度与原题同量级。
#    - 自由度控制在少量离散选择: 目标边只有 top/bottom/left/right 四档, 镜像也只有单轴翻转四档。
#      generator 额外拒绝退化样本, 保证人眼能在 2-4 个例子里稳定看出“翻转 + 中点贴边”。
# c) 对原 main() 的隐式假设审查
#    - 原代码默认保留片段只需平移, 局部偏移不变; 融合后失效, 因为必须先在 bbox 归一化坐标内翻转。
#    - 原代码默认目标总是角点; 融合后失效, 需要先选目标边, 再根据 bbox 尺寸算居中偏移。
#    - 原代码把每个 cut region 里的 fg 当作同一片段处理; 这仍成立, generator 继续保证各区域至多一块有效片段。
#    - 原射线逻辑只依赖 split 线上的 fg 极值, 与片段翻转/贴边方式无关, 因而保持不变。
# d) 放置合法性
#    - top/bottom 目标使用按 c_split 划出的左/右边段; left/right 目标使用按 r_split 划出的上/下边段。
#      这些边段互斥, 且不会穿过 split line。
#    - generator 对每个候选片段都实际调用 solver helper 计算融合后位置, 检查不越界、不跨 split、
#      不与已有输出片段重叠。
# e) 视觉可观察性
#    - 仅“贴边居中”还不够, 因为对某些对称片段翻转不可见; 仅“镜像”也不够, 因为居中差异可能退化。
#      因此 generator 同时量化两类差异:
#      `mirror_delta = combo vs centered-only`, `centering_delta = combo vs mirror-corner`。
#      两者都必须明显非零, 才接受样本。
# f) 可视化整洁度
#    - 颜色仍只有 bg/fg/ctrl 三类。split 线 fg 留在中心, 保留片段靠边, 射线只覆盖背景格, 图层关系清楚。
#    - 双控制时沿用“横边优先”后, 三块保留片段会分散到不同边段, 避免多块压在同一角附近。
# g) 变化的可推断性
#    - 无新增颜色映射。需要推断的只有几何规则: 先切, 保留接壤片段, 再按目标边做单轴翻转并居中贴边。
# h) main() 的新隐式假设
#    - 新 solver 假设保留片段的 bbox 能装进其目标边段。generator 通过 split 位于 40-60%、片段较小、
#      并对每个候选实际验证 margins>=0 来保证这一点。
#    - 新 solver 还假设“横边优先”能消除双控制二义性; generator 的 kept-region 结构与该约定一致。

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


def _normalize_points(pts: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[frozenset[tuple[int, int]], int, int]:
    rs = [int(r) for r, _ in pts]
    cs = [int(c) for _, c in pts]
    rmin, cmin = int(min(rs)), int(min(cs))
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


def _place_shape_in_corner(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    dh: str,
    dv: str,
    H: int,
    W: int,
) -> frozenset[tuple[int, int]]:
    r0 = 0 if dh == "T" else int(H - h)
    c0 = 0 if dv == "L" else int(W - w)
    return frozenset((int(r0 + r), int(c0 + c)) for r, c in shape)


def _place_shape_on_centered_edge(
    shape: frozenset[tuple[int, int]],
    h: int,
    w: int,
    edge: str,
    hs: str,
    vs: str,
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[frozenset[tuple[int, int]], tuple[int, int]]:
    seg0, seg1 = _segment_for_edge(edge, hs, vs, H, W, r_split, c_split)
    seg_len = int(seg1 - seg0 + 1)
    assert seg_len > 0

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

    placed = frozenset((int(target_r0 + r), int(target_c0 + c)) for r, c in shape)
    return placed, margins


def _transform_piece_combo(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[frozenset[tuple[int, int]], str, tuple[int, int]]:
    edge = _pick_center_edge(hs, vs, E)
    shape, h, w = _normalize_points(pts)
    mirrored = _flip_shape(shape, h, w, flip_v=(edge in ("top", "bottom")), flip_h=(edge in ("left", "right")))
    moved, margins = _place_shape_on_centered_edge(mirrored, h, w, edge, hs, vs, H, W, r_split, c_split)
    return moved, edge, margins


def _transform_piece_center_only(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> tuple[frozenset[tuple[int, int]], str, tuple[int, int]]:
    edge = _pick_center_edge(hs, vs, E)
    shape, h, w = _normalize_points(pts)
    moved, margins = _place_shape_on_centered_edge(shape, h, w, edge, hs, vs, H, W, r_split, c_split)
    return moved, edge, margins


def _transform_piece_mirror_corner(
    pts: set[tuple[int, int]] | frozenset[tuple[int, int]],
    hs: str,
    vs: str,
    E: set[str],
    H: int,
    W: int,
    r_split: int | None,
    c_split: int | None,
) -> frozenset[tuple[int, int]]:
    dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
    shape, h, w = _normalize_points(pts)
    mirrored = _flip_shape(shape, h, w, flip_v=(dh in ("T", "B")), flip_h=(dv in ("L", "R")))
    return _place_shape_in_corner(mirrored, h, w, dh, dv, H, W)


def _draw_rays(out: np.ndarray, g: np.ndarray, fg: int, ctrl: int, ctrl_pts: list[tuple[int, int]]) -> None:
    H, W = g.shape
    for r, c in ctrl_pts:
        if r == 0:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(max(rows))
                for rr in range(0, stop + 1):
                    if int(g[rr, c]) == int(get_bg(g)):
                        out[rr, c] = ctrl
        elif r == H - 1:
            rows = [int(rr) for rr in np.argwhere(g[:, c] == fg).flatten()]
            if rows:
                stop = int(min(rows))
                for rr in range(H - 1, stop - 1, -1):
                    if int(g[rr, c]) == int(get_bg(g)):
                        out[rr, c] = ctrl
        elif c == 0:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(max(cols))
                for cc in range(0, stop + 1):
                    if int(g[r, cc]) == int(get_bg(g)):
                        out[r, cc] = ctrl
        elif c == W - 1:
            cols = [int(cc) for cc in np.argwhere(g[r, :] == fg).flatten()]
            if cols:
                stop = int(min(cols))
                for cc in range(W - 1, stop - 1, -1):
                    if int(g[r, cc]) == int(get_bg(g)):
                        out[r, cc] = ctrl

    for r, c in ctrl_pts:
        out[r, c] = ctrl


def _collect_regions(g: np.ndarray, fg: int, r_split: int | None, c_split: int | None) -> dict[tuple[str, str], list[tuple[int, int]]]:
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
    return regions


def _solve(grid: np.ndarray, mode: str) -> np.ndarray:
    g, H, W, bg, fg, ctrl, ctrl_pts, E, r_split, c_split = _parse_grid(grid)
    out = np.full((H, W), bg, dtype=int)

    _draw_rays(out, g, fg, ctrl, ctrl_pts)

    for r, c in np.argwhere(g == fg):
        rr = int(r)
        cc = int(c)
        if (c_split is not None and cc == c_split) or (r_split is not None and rr == r_split):
            out[rr, cc] = fg

    regions = _collect_regions(g, fg, r_split, c_split)
    for (hs, vs), pts in regions.items():
        if not (_touches_edges(hs, vs) & E):
            continue

        if mode == "combo":
            moved, _edge, _margins = _transform_piece_combo(pts, hs, vs, E, H, W, r_split, c_split)
        elif mode == "center_only":
            moved, _edge, _margins = _transform_piece_center_only(pts, hs, vs, E, H, W, r_split, c_split)
        elif mode == "mirror_corner":
            moved = _transform_piece_mirror_corner(pts, hs, vs, E, H, W, r_split, c_split)
        elif mode == "corner_only":
            dh, dv = _pick_corner_dest(hs, vs, E, r_split, c_split)
            shape, h, w = _normalize_points(pts)
            moved = _place_shape_in_corner(shape, h, w, dh, dv, H, W)
        else:
            raise ValueError(mode)

        for rr, cc in moved:
            if 0 <= int(rr) < H and 0 <= int(cc) < W:
                out[int(rr), int(cc)] = fg

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="combo")


def _center_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="center_only")


def _mirror_corner_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="mirror_corner")


def _corner_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, mode="corner_only")


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    combo = main(grid)
    center_only = _center_only_main(grid)
    mirror_corner = _mirror_corner_main(grid)
    return {
        "mirror_delta": int(np.count_nonzero(combo != center_only)),
        "centering_delta": int(np.count_nonzero(combo != mirror_corner)),
        "base_delta": int(np.count_nonzero(combo != _corner_only_main(grid))),
    }


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = int(r + dr), int(c + dc)
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


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg, ctrl_color = 1, 7, 2

    num_ctrl = random.choices([1, 2], weights=[1, 10])[0]
    if int(num_ctrl) == 1:
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
            ctrl_pts.append((0 if edge == "top" else int(H - 1), int(c_split)))
            has_v = True
        else:
            r_split = random.randint(int(H * 0.4), int(H * 0.6))
            ctrl_pts.append((int(r_split), 0 if edge == "left" else int(W - 1)))
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
        sprite_c0 = int(c_split - int(frac * sprite_w))
        sprite_c0 = max(4, min(int(sprite_c0), int(W - 5 - sprite_w + 1)))
        if sprite_c0 > int(W - 5 - sprite_w + 1):
            return None
        if int(c_split) <= int(sprite_c0) or int(c_split) >= int(sprite_c0 + sprite_w - 1):
            return None
    else:
        hi = int(W - 5 - sprite_w + 1)
        if hi < 4:
            return None
        sprite_c0 = random.randint(4, hi)

    if has_h:
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
        sprite_r0 = max(4, min(int(sprite_r0), int(H - 5 - sprite_h + 1)))
        if sprite_r0 > int(H - 5 - sprite_h + 1):
            return None
        if int(r_split) <= int(sprite_r0) or int(r_split) >= int(sprite_r0 + sprite_h - 1):
            return None
    else:
        hi = int(H - 5 - sprite_h + 1)
        if hi < 4:
            return None
        sprite_r0 = random.randint(4, hi)

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else r_split + 1)
        r1 = int((r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else c_split + 1)
        c1 = int((c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return int(r0), int(r1), int(c0), int(c1)

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        n = random.randint(2, 4)
        rows = [int(r) for r in range(int(sprite_r0), int(sprite_r1) + 1) if r_split is None or int(r) != int(r_split)]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        n = random.randint(2, 4)
        cols = [int(c) for c in range(int(sprite_c0), int(sprite_c1) + 1) if c_split is None or int(c) != int(c_split)]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = int(sprite_h * sprite_w)
    target_density = random.uniform(0.39, 0.60)
    target_total = int(target_density * area)

    n_kept = len(kept)
    n_vanish = len(vanish)
    budget_pieces = int(target_total - n_split)
    if n_vanish > 0:
        vanish_budget = random.randint(3, min(7, max(3, budget_pieces // 3)))
        budget_pieces -= int(vanish_budget)
    else:
        vanish_budget = 0

    if budget_pieces < int(n_kept * 3):
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for i in range(n_kept):
        r0, r1, c0, c1 = sub_bbox(*kept[i])
        max_area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if i == int(n_kept - 1):
            px = int(remaining)
        else:
            avg = int(remaining // (n_kept - i))
            lo_px = int(max(3, avg - 2))
            hi_px = int(min(max_area, avg + 2))
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = int(max(3, min(px, max_area, remaining - (n_kept - i - 1) * 3)))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(int(b) < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_combo: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_center: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_mirror_corner: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    centered_gaps: list[int] = []
    occupied_combo: set[tuple[int, int]] = set()

    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        target_edge = _pick_center_edge(hs, vs, E)

        chosen = None
        for _ in range(40):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) < 3:
                continue

            ph = int(max(r for r, _ in piece) + 1)
            pw = int(max(c for _, c in piece) + 1)
            if target_edge in ("top", "bottom") and ph < 2:
                continue
            if target_edge in ("left", "right") and pw < 2:
                continue

            if has_h:
                inp_r0 = int((r1 - ph + 1) if hs == "T" else r0)
            else:
                split_rows = sorted(int(r) for r, c in split_pixels if int(c) == int(c_split))
                if split_rows:
                    mid = int((split_rows[0] + split_rows[-1]) // 2)
                    inp_r0 = int(max(r0, min(mid - ph // 2, r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int((c1 - pw + 1) if vs == "L" else c0)
            else:
                split_cols = sorted(int(c) for r, c in split_pixels if int(r) == int(r_split))
                if split_cols:
                    mid = int((split_cols[0] + split_cols[-1]) // 2)
                    inp_c0 = int(max(c0, min(mid - pw // 2, c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
            bad = False
            for r, c in inp_pts:
                if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                    bad = True
                    break
                if has_v and ((vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split))):
                    bad = True
                    break
                if has_h and ((hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split))):
                    bad = True
                    break
            if bad:
                continue

            combo_pts, _edge, margins = _transform_piece_combo(inp_pts, hs, vs, E, H, W, r_split, c_split)
            center_pts, _edge2, _margins2 = _transform_piece_center_only(inp_pts, hs, vs, E, H, W, r_split, c_split)
            mirror_corner_pts = _transform_piece_mirror_corner(inp_pts, hs, vs, E, H, W, r_split, c_split)

            if min(margins) < 0:
                continue
            if combo_pts & occupied_combo:
                continue

            valid_combo = True
            for r, c in combo_pts:
                if has_v and ((vs == "L" and int(c) >= int(c_split)) or (vs == "R" and int(c) <= int(c_split))):
                    valid_combo = False
                    break
                if has_h and ((hs == "T" and int(r) >= int(r_split)) or (hs == "B" and int(r) <= int(r_split))):
                    valid_combo = False
                    break
            if not valid_combo:
                continue

            chosen = (inp_pts, combo_pts, center_pts, mirror_corner_pts, margins)
            break

        if chosen is None:
            return None

        inp_pts, combo_pts, center_pts, mirror_corner_pts, margins = chosen
        input_pieces[reg] = inp_pts
        output_combo[reg] = combo_pts
        output_center[reg] = center_pts
        output_mirror_corner[reg] = mirror_corner_pts
        centered_gaps.append(max(margins))
        occupied_combo |= set(combo_pts)

    mirror_delta = sum(len(output_combo[reg] ^ output_center[reg]) for reg in kept)
    centering_delta = sum(len(output_combo[reg] ^ output_mirror_corner[reg]) for reg in kept)
    if mirror_delta < 4 or centering_delta < 4:
        return None
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
        mask[int(r), int(c)] = 1
    if not is_contiguous(mask):
        return None

    vanish_fg: set[tuple[int, int]] = set()
    if vanish:
        for vreg in vanish:
            vhs, vvs = vreg
            vr0, vr1, vc0, vc1 = sub_bbox(vhs, vvs)
            vh, vw = int(vr1 - vr0 + 1), int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = int(min(vanish_budget, vh * vw))
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph = int(max(r for r, _ in piece) + 1)
            ppw = int(max(c for _, c in piece) + 1)
            vr_off = int((vr1 - pph + 1) if vhs == "T" else vr0)
            vc_off = int((vc1 - ppw + 1) if vvs == "L" else vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if int(r) < 4 or int(r) > int(H - 5) or int(c) < 4 or int(c) > int(W - 5):
                    return None
                if (int(r), int(c)) in all_inp_fg:
                    return None
            vanish_fg = vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for cr, cc in ctrl_pts:
        inp[int(cr), int(cc)] = int(ctrl_color)

    out = np.full((H, W), bg, dtype=int)
    for pts in output_combo.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)

    for cr, cc in ctrl_pts:
        if int(cr) == 0:
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(max(fg_rows)) + 1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cr) == int(H - 1):
            fg_rows = [int(r) for r in range(H) if int(out[int(r), int(cc)]) == int(fg)]
            if fg_rows:
                for r in range(int(H - 1), int(min(fg_rows)) - 1, -1):
                    if int(out[int(r), int(cc)]) == int(bg):
                        out[int(r), int(cc)] = int(ctrl_color)
        elif int(cc) == 0:
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(max(fg_cols)) + 1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
        elif int(cc) == int(W - 1):
            fg_cols = [int(c) for c in range(W) if int(out[int(cr), int(c)]) == int(fg)]
            if fg_cols:
                for c in range(int(W - 1), int(min(fg_cols)) - 1, -1):
                    if int(out[int(cr), int(c)]) == int(bg):
                        out[int(cr), int(c)] = int(ctrl_color)
    for cr, cc in ctrl_pts:
        out[int(cr), int(cc)] = int(ctrl_color)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_center_only_main(inp), out):
        return None
    if np.array_equal(_mirror_corner_main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(700):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 700 attempts")
