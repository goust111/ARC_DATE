# Combo variant of 4a21e3da (index 6)
# combo_spec: D3/3+D17/0
# Name: 箭头指示方向+颜色编码指令

"""
融合设计分析
===========

a) 具体化与行为追踪
- 原题的控制源是边缘单像素；这里改成 3 像素、4 连通、单色的 L 形箭头。箭头朝向即发射方向：
  - top:    {(0,s), (0,s+1), (1,s)}，向下发射，split 列为 s
  - bottom: {(H-1,s-1), (H-1,s), (H-2,s)}，向上发射，split 列为 s
  - left:   {(s-1,0), (s,0), (s,1)}，向右发射，split 行为 s
  - right:  {(s,W-2), (s,W-1), (s+1,W-1)}，向左发射，split 行为 s
- 同时再引入两种控制色：红色(2) 表示对应保留区域直接滑入角落；青色(8) 表示对应保留区域先顺时针旋转 90 度，再滑入角落。
- 具体例子：若顶部是红箭头、左侧是青箭头，则 vertical split 由顶部箭头给出，horizontal split 由左箭头给出。
  - 右上区域只接触 top，因此直接滑到右上角。
  - 左下区域只接触 left，因此先旋转再滑到左下角。
  - 左上区域同时接触 top 与 left；这里规定“旋转指令优先”，所以它也先旋转再滑到左上角。
  - 不接触任何控制边的区域仍然消失。
- 与原题相比，解题链从“看边缘单点”变成“先识别 L 形箭头朝向，再读箭头颜色的动作语义”。

b) 复杂度校准
- 方向只有四种箭头朝向，动作只有两种颜色指令，离散档位很少，避免自由度过多。
- 生成器继续大量采样 2 控制、且偏向红青混合，从而在单个样例内同时出现“直接滑动”和“旋转后滑动”，让颜色语义可见。
- 旋转色作用到的 piece 强制选择成“旋转后外观会变”的形状，避免颜色变化存在但输出与平移版完全一样。

c) 隐式假设审查
- 原 solver 假设 `border_cols` 里只有一种控制色；融合后不成立，所以必须先从“边缘非背景色集合”里分离控制色，再把唯一的非边缘非背景色识别成 fg。
- 原 solver 假设 `np.argwhere(g == ctrl)` 得到的每个点都是独立发射源；融合后不成立，必须先对所有控制色像素做 4 连通分量分组，再把每个 3 像素分量解码成一个箭头。
- 原 solver 假设“触边区域一定只做平移”；融合后不成立，区域动作要由其接触到的控制边颜色决定，并处理双边接触时的优先级。
- 原 generator 假设输出 ray 只有一种控制色、控制符号只有单像素；融合后必须输出整支箭头，并用各自颜色重画 ray。

d) 放置合法性
- 箭头 split 坐标限制在 40%-60% 且远离角落，并额外约束在 `[1, W-2]` 或 `[1, H-2]`，保证 2x2 箭头框完全落在网格内且只接触一条边。
- sprite 仍保持四周至少 4 格 margin，因此箭头与 sprite、split-line fg、角落输出 piece 均不重叠。
- 对每个保留区域，先检查输入子框能放下原 piece，再检查输出角落一侧能放下旋转后或未旋转的 piece，避免越 split、越界或角落冲突。

e) 视觉可观察性
- 输入里箭头形状本身直接暴露“方向是从形状读出来，而不是只看所在边”。
- 输出里 ray 使用与箭头相同的颜色，颜色指令在输入和输出中都重复出现。
- 生成器偏向混色双箭头，并强制旋转受影响 piece 非 90 度对称，因此观察者能从少量样例看出“某色箭头附近的片段会转向后进角落”。

f) 可视化整洁度
- 控制信息只占边框附近 2x2 小块；sprite 继续位于中央；输出 piece 继续位于角落，层次与原题一致。
- ray 仍只画在 split 行/列的背景格上，不会把 fg 本体染色，因此不会制造难以分层的覆盖。

g) 变化的可推断性
- 颜色语义只使用 2 和 8 两种控制色，不依赖数值大小。
- 颜色变化总是伴随固定结构线索：同色箭头 + 同色 ray + 与该控制边接壤的区域动作一致。
- 为防退化，生成器在末尾显式比较“融合规则输出”与“忽略颜色、只按箭头平移”的基线；只要存在旋转受影响区域，就要求两者输出必须不同。

h) 融合行为量化指标
- `rotating_kept_regions`: 当前样例中受青色箭头影响、应发生旋转的保留区域数量。
- `changed_pixels_vs_move_only`: 融合 solver 输出与“同样解箭头、但忽略颜色只平移”的基线输出的像素差数量。
- 合法样例要求：若 `rotating_kept_regions > 0`，则 `changed_pixels_vs_move_only > 0`。

i) 新 solver 假设与 generator 保证
- 所有控制像素恰好组成 1 或 2 个 3 像素、单色、4 连通的 L 形箭头；generator 保证这一点。
- 最多一个 vertical split、最多一个 horizontal split；generator 只在互相垂直的边上放置箭头。
- fg 是唯一一种“出现于内部但不出现在边框上”的非背景色；generator 保持 sprite 距离边缘至少 4 格。
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import colors_present, get_bg, is_contiguous


MOVE_CTRL = 2
ROTATE_CTRL = 8

TOP_ARROW = frozenset({(0, 0), (0, 1), (1, 0)})
RIGHT_ARROW = frozenset({(0, 0), (0, 1), (1, 1)})
LEFT_ARROW = frozenset({(0, 0), (1, 0), (1, 1)})
BOTTOM_ARROW = frozenset({(0, 1), (1, 0), (1, 1)})


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    comps: list[list[tuple[int, int]]] = []
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(comp)
    return comps


def _normalize_piece(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]] | list[tuple[int, int]]
) -> frozenset[tuple[int, int]]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    r0 = min(rs)
    c0 = min(cs)
    return frozenset((int(r - r0), int(c - c0)) for r, c in cells)


def _piece_hw(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> tuple[int, int]:
    rs = [int(r) for r, _ in cells]
    cs = [int(c) for _, c in cells]
    return int(max(rs) + 1), int(max(cs) + 1)


def _rotate_piece_cw(
    cells: set[tuple[int, int]] | frozenset[tuple[int, int]], turns: int = 1
) -> frozenset[tuple[int, int]]:
    piece = _normalize_piece(cells)
    for _ in range(int(turns) % 4):
        height, _width = _piece_hw(piece)
        piece = frozenset((int(c), int(height - 1 - r)) for r, c in piece)
        piece = _normalize_piece(piece)
    return piece


def _rotation_is_visible(cells: set[tuple[int, int]] | frozenset[tuple[int, int]]) -> bool:
    piece = _normalize_piece(cells)
    return _rotate_piece_cw(piece) != piece


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


def _region_tags_for_pixel(r: int, c: int, r_split: int | None, c_split: int | None) -> tuple[str, str]:
    hs = "A"
    vs = "A"
    if r_split is not None:
        hs = "T" if int(r) < int(r_split) else "B"
    if c_split is not None:
        vs = "L" if int(c) < int(c_split) else "R"
    return hs, vs


def _place_piece_to_corner(
    pts: list[tuple[int, int]] | set[tuple[int, int]] | frozenset[tuple[int, int]],
    H: int,
    W: int,
    dh: str,
    dv: str,
    rotate: bool,
) -> frozenset[tuple[int, int]]:
    local = _normalize_piece(pts)
    if rotate:
        local = _rotate_piece_cw(local)
    ph, pw = _piece_hw(local)
    base_r = 0 if dh == "T" else int(H - ph)
    base_c = 0 if dv == "L" else int(W - pw)
    return frozenset((int(base_r + r), int(base_c + c)) for r, c in local)


def _decode_arrow(comp: list[tuple[int, int]], H: int, W: int) -> dict[str, object]:
    rs = [int(r) for r, _ in comp]
    cs = [int(c) for _, c in comp]
    rmin, rmax = min(rs), max(rs)
    cmin, cmax = min(cs), max(cs)
    norm = frozenset((int(r - rmin), int(c - cmin)) for r, c in comp)

    touches: list[str] = []
    if rmin == 0:
        touches.append("top")
    if rmax == H - 1:
        touches.append("bottom")
    if cmin == 0:
        touches.append("left")
    if cmax == W - 1:
        touches.append("right")

    assert len(comp) == 3
    assert (rmax - rmin) == 1 and (cmax - cmin) == 1
    assert len(touches) == 1

    edge = touches[0]
    if norm == TOP_ARROW:
        assert edge == "top"
        launch = (0, cmin)
    elif norm == BOTTOM_ARROW:
        assert edge == "bottom"
        launch = (H - 1, cmax)
    elif norm == LEFT_ARROW:
        assert edge == "left"
        launch = (rmax, 0)
    elif norm == RIGHT_ARROW:
        assert edge == "right"
        launch = (rmin, W - 1)
    else:
        raise AssertionError("Unexpected arrow shape")

    return {"edge": edge, "launch": launch, "cells": tuple(sorted(comp))}


def _extract_scene(g: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    H, W = g.shape
    bg = int(get_bg(g))
    nonbg = colors_present(g, exclude={bg})

    border_nonbg = set(map(int, g[0, :])) | set(map(int, g[H - 1, :])) | set(map(int, g[:, 0])) | set(map(int, g[:, W - 1]))
    border_nonbg.discard(int(bg))
    assert border_nonbg

    fg_candidates = [int(c) for c in nonbg if int(c) not in border_nonbg]
    assert len(fg_candidates) == 1
    fg = int(fg_candidates[0])

    ctrl_mask = (g != int(bg)) & (g != int(fg))
    controls: list[dict[str, object]] = []
    for comp in _components4(ctrl_mask):
        color = int(g[comp[0][0], comp[0][1]])
        assert all(int(g[r, c]) == int(color) for r, c in comp)
        arrow = _decode_arrow(comp, H, W)
        arrow["color"] = int(color)
        controls.append(arrow)

    controls.sort(key=lambda a: (str(a["edge"]), int(a["color"]), tuple(a["launch"])))
    assert len(controls) in (1, 2)
    return bg, fg, controls


def _split_info(controls: list[dict[str, object]]) -> tuple[set[str], dict[str, int], int | None, int | None]:
    E: set[str] = set()
    edge_to_color: dict[str, int] = {}
    c_split: int | None = None
    r_split: int | None = None
    for ctrl in controls:
        edge = str(ctrl["edge"])
        launch_r, launch_c = tuple(ctrl["launch"])
        E.add(edge)
        edge_to_color[edge] = int(ctrl["color"])
        if edge in ("top", "bottom"):
            assert c_split is None or c_split == int(launch_c)
            c_split = int(launch_c)
        else:
            assert r_split is None or r_split == int(launch_r)
            r_split = int(launch_r)
    return E, edge_to_color, r_split, c_split


def _paint_ray(scan_grid: np.ndarray, out: np.ndarray, edge: str, launch: tuple[int, int], fg: int, color: int, bg: int) -> None:
    H, W = scan_grid.shape
    r0, c0 = (int(launch[0]), int(launch[1]))
    if edge == "top":
        rows = [int(rr) for rr in np.flatnonzero(scan_grid[:, c0] == int(fg))]
        if rows:
            stop = max(rows)
            for rr in range(0, stop + 1):
                if int(scan_grid[rr, c0]) == int(bg):
                    out[rr, c0] = int(color)
    elif edge == "bottom":
        rows = [int(rr) for rr in np.flatnonzero(scan_grid[:, c0] == int(fg))]
        if rows:
            stop = min(rows)
            for rr in range(H - 1, stop - 1, -1):
                if int(scan_grid[rr, c0]) == int(bg):
                    out[rr, c0] = int(color)
    elif edge == "left":
        cols = [int(cc) for cc in np.flatnonzero(scan_grid[r0, :] == int(fg))]
        if cols:
            stop = max(cols)
            for cc in range(0, stop + 1):
                if int(scan_grid[r0, cc]) == int(bg):
                    out[r0, cc] = int(color)
    elif edge == "right":
        cols = [int(cc) for cc in np.flatnonzero(scan_grid[r0, :] == int(fg))]
        if cols:
            stop = min(cols)
            for cc in range(W - 1, stop - 1, -1):
                if int(scan_grid[r0, cc]) == int(bg):
                    out[r0, cc] = int(color)
    else:
        raise ValueError(edge)


def _simulate(grid: np.ndarray, honor_color: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, fg, controls = _extract_scene(g)
    E, edge_to_color, r_split, c_split = _split_info(controls)

    out = np.full((H, W), int(bg), dtype=int)

    for ctrl in controls:
        _paint_ray(g, out, str(ctrl["edge"]), tuple(ctrl["launch"]), fg, int(ctrl["color"]), bg)

    for ctrl in controls:
        for r, c in ctrl["cells"]:
            out[int(r), int(c)] = int(ctrl["color"])

    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            out[r, c] = int(fg)

    regions: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        hs, vs = _region_tags_for_pixel(r, c, r_split, c_split)
        regions.setdefault((hs, vs), []).append((r, c))

    for (hs, vs), pts in regions.items():
        touched = _touches_edges(hs, vs) & E
        if not touched:
            continue
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        rotate = bool(honor_color and any(int(edge_to_color[edge]) == int(ROTATE_CTRL) for edge in touched))
        out_pts = _place_piece_to_corner(pts, H, W, dh, dv, rotate)
        for rr, cc in out_pts:
            if 0 <= rr < H and 0 <= cc < W:
                out[int(rr), int(cc)] = int(fg)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    融合规则：
    - 先把边缘控制对象识别成 3 像素 L 形箭头，并由形状恢复 inward ray 的方向与 split 行/列。
    - ray 的几何作用与原题一致：沿 split 行/列向内画间歇控制线，只涂背景格，fg 切线保留。
    - 再由箭头颜色决定各保留区域的动作：
      红(2) = 直接滑入对应角落；青(8) = 先顺时针旋转 90 度，再滑入。
    - 若某区域同时接触多条控制边，只要其中任一边是青色，就执行旋转。
    """
    return _simulate(grid, honor_color=True)


def behavior_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg, fg, controls = _extract_scene(g)
    E, edge_to_color, r_split, c_split = _split_info(controls)

    rotating_kept_regions = 0
    regions_seen: set[tuple[str, str]] = set()
    for r, c in np.argwhere(g == int(fg)):
        r = int(r)
        c = int(c)
        if (c_split is not None and c == int(c_split)) or (r_split is not None and r == int(r_split)):
            continue
        regions_seen.add(_region_tags_for_pixel(r, c, r_split, c_split))

    for hs, vs in regions_seen:
        touched = _touches_edges(hs, vs) & E
        if touched and any(int(edge_to_color[edge]) == int(ROTATE_CTRL) for edge in touched):
            rotating_kept_regions += 1

    full = _simulate(g, honor_color=True)
    move_only = _simulate(g, honor_color=False)
    return {
        "num_controls": int(len(controls)),
        "mixed_control_colors": int(len({int(ctrl["color"]) for ctrl in controls}) > 1),
        "rotating_kept_regions": int(rotating_kept_regions),
        "changed_pixels_vs_move_only": int(np.sum(full != move_only)),
        "control_component_count": int(len(controls)),
    }


def _grow_piece(max_h: int, max_w: int, target_px: int) -> frozenset[tuple[int, int]]:
    cells = {(0, 0)}
    for _ in range(int(target_px) - 1):
        neighbors = set()
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = int(r + dr)
                nc = int(c + dc)
                if (nr, nc) not in cells:
                    neighbors.add((nr, nc))
        valid = [
            (nr, nc)
            for nr, nc in neighbors
            if max(r for r, _ in cells | {(nr, nc)}) - min(r for r, _ in cells | {(nr, nc)}) + 1 <= int(max_h)
            and max(c for _, c in cells | {(nr, nc)}) - min(c for _, c in cells | {(nr, nc)}) + 1 <= int(max_w)
        ]
        if not valid:
            break
        cells.add(random.choice(valid))
    return _normalize_piece(cells)


def _output_caps(
    hs: str,
    vs: str,
    H: int,
    W: int,
    has_h: bool,
    has_v: bool,
    r_split: int | None,
    c_split: int | None,
) -> tuple[int, int]:
    if has_h:
        assert r_split is not None
        cap_h = int(r_split) if hs == "T" else int(H - 1 - r_split)
    else:
        cap_h = int(H)

    if has_v:
        assert c_split is not None
        cap_w = int(c_split) if vs == "L" else int(W - 1 - c_split)
    else:
        cap_w = int(W)
    return cap_h, cap_w


def _make_arrow(edge: str, split: int, H: int, W: int) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    if edge == "top":
        return {(0, split), (0, split + 1), (1, split)}, (0, split)
    if edge == "bottom":
        return {(H - 1, split - 1), (H - 1, split), (H - 2, split)}, (H - 1, split)
    if edge == "left":
        return {(split - 1, 0), (split, 0), (split, 1)}, (split, 0)
    if edge == "right":
        return {(split, W - 2), (split, W - 1), (split + 1, W - 1)}, (split, W - 1)
    raise ValueError(edge)


def _build_output(
    H: int,
    W: int,
    bg: int,
    fg: int,
    controls: list[dict[str, object]],
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]],
    split_pixels: set[tuple[int, int]],
) -> np.ndarray:
    out = np.full((H, W), int(bg), dtype=int)
    for pts in output_pieces.values():
        for r, c in pts:
            out[int(r), int(c)] = int(fg)
    for r, c in split_pixels:
        out[int(r), int(c)] = int(fg)
    for ctrl in controls:
        _paint_ray(out, out, str(ctrl["edge"]), tuple(ctrl["launch"]), fg, int(ctrl["color"]), bg)
    for ctrl in controls:
        for r, c in ctrl["cells"]:
            out[int(r), int(c)] = int(ctrl["color"])
    return out


def _gen_attempt() -> np.ndarray | None:
    S = random.randint(18, 20)
    H = W = S
    bg, fg = 1, 7

    num_ctrl = random.choices([1, 2], weights=[1, 4])[0]
    if num_ctrl == 1:
        ctrl_edges = [random.choice(["top", "bottom", "left", "right"])]
        ctrl_colors = [random.choice([MOVE_CTRL, ROTATE_CTRL])]
    else:
        ctrl_edges = [random.choice(["top", "bottom"]), random.choice(["left", "right"])]
        ctrl_colors = random.choices(
            [
                [MOVE_CTRL, MOVE_CTRL],
                [ROTATE_CTRL, ROTATE_CTRL],
                [MOVE_CTRL, ROTATE_CTRL],
                [ROTATE_CTRL, MOVE_CTRL],
            ],
            weights=[1, 1, 3, 3],
        )[0]

    split_lo_w = max(1, int(W * 0.4))
    split_hi_w = min(W - 2, int(W * 0.6))
    split_lo_h = max(1, int(H * 0.4))
    split_hi_h = min(H - 2, int(H * 0.6))

    c_split = None
    r_split = None
    controls: list[dict[str, object]] = []
    edge_to_color: dict[str, int] = {}
    E = set(ctrl_edges)
    has_v = False
    has_h = False

    for edge, color in zip(ctrl_edges, ctrl_colors):
        if edge in ("top", "bottom"):
            c_split = random.randint(split_lo_w, split_hi_w)
            cells, launch = _make_arrow(edge, c_split, H, W)
            has_v = True
        else:
            r_split = random.randint(split_lo_h, split_hi_h)
            cells, launch = _make_arrow(edge, r_split, H, W)
            has_h = True
        controls.append({"edge": edge, "color": int(color), "cells": tuple(sorted(cells)), "launch": launch})
        edge_to_color[edge] = int(color)

    def region_rotates(hs: str, vs: str) -> bool:
        touched = _touches_edges(hs, vs) & E
        return any(int(edge_to_color[edge]) == int(ROTATE_CTRL) for edge in touched)

    if has_h and has_v:
        all_regions = [("T", "L"), ("T", "R"), ("B", "L"), ("B", "R")]
    elif has_v:
        all_regions = [("A", "L"), ("A", "R")]
    else:
        all_regions = [("T", "A"), ("B", "A")]

    kept = [reg for reg in all_regions if _touches_edges(*reg) & E]
    vanish = [reg for reg in all_regions if reg not in kept]

    sprite_h = random.randint(5, 8)
    sprite_w = random.randint(8, 10)

    if has_v:
        assert c_split is not None
        frac = random.uniform(0.4, 0.6)
        sprite_c0 = int(c_split - int(frac * sprite_w))
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
        assert r_split is not None
        frac = random.uniform(0.4, 0.6)
        sprite_r0 = int(r_split - int(frac * sprite_h))
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

    sprite_r1 = int(sprite_r0 + sprite_h - 1)
    sprite_c1 = int(sprite_c0 + sprite_w - 1)

    def sub_bbox(hs: str, vs: str) -> tuple[int, int, int, int]:
        r0 = int(sprite_r0 if hs in ("T", "A") else r_split + 1)
        r1 = int((r_split - 1 if hs == "T" else sprite_r1) if has_h else sprite_r1)
        c0 = int(sprite_c0 if vs in ("L", "A") else c_split + 1)
        c1 = int((c_split - 1 if vs == "L" else sprite_c1) if has_v else sprite_c1)
        return r0, r1, c0, c1

    for reg in kept + vanish:
        r0, r1, c0, c1 = sub_bbox(*reg)
        if r1 < r0 or c1 < c0:
            return None

    split_pixels: set[tuple[int, int]] = set()
    if has_v:
        assert c_split is not None
        n = random.randint(2, 4)
        rows = [r for r in range(sprite_r0, sprite_r1 + 1) if r != r_split]
        if len(rows) < n:
            return None
        for r in random.sample(rows, n):
            split_pixels.add((int(r), int(c_split)))
    if has_h:
        assert r_split is not None
        n = random.randint(2, 4)
        cols = [c for c in range(sprite_c0, sprite_c1 + 1) if c != c_split]
        if len(cols) < n:
            return None
        for c in random.sample(cols, n):
            split_pixels.add((int(r_split), int(c)))

    n_split = len(split_pixels)
    area = sprite_h * sprite_w
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

    if budget_pieces < n_kept * 3:
        return None

    piece_budgets: list[int] = []
    remaining = int(budget_pieces)
    for idx, reg in enumerate(kept):
        hs, vs = reg
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        cap_h, cap_w = _output_caps(hs, vs, H, W, has_h, has_v, r_split, c_split)
        max_area = int(min(sub_h * sub_w, cap_h * cap_w))
        if idx == n_kept - 1:
            px = remaining
        else:
            avg = remaining // (n_kept - idx)
            lo_px = max(3, avg - 2)
            hi_px = min(max_area, avg + 2)
            if lo_px > hi_px:
                return None
            px = random.randint(lo_px, hi_px)
        px = max(3, min(px, max_area, remaining - (n_kept - idx - 1) * 3))
        piece_budgets.append(int(px))
        remaining -= int(px)

    if remaining != 0 or any(b < 3 for b in piece_budgets):
        return None

    input_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}
    output_pieces: dict[tuple[str, str], frozenset[tuple[int, int]]] = {}

    for idx, reg in enumerate(kept):
        hs, vs = reg
        dh, dv = _pick_dest(hs, vs, E, r_split, c_split)
        r0, r1, c0, c1 = sub_bbox(hs, vs)
        sub_h = int(r1 - r0 + 1)
        sub_w = int(c1 - c0 + 1)
        target_px = int(piece_budgets[idx])
        rotate = region_rotates(hs, vs)
        cap_h, cap_w = _output_caps(hs, vs, H, W, has_h, has_v, r_split, c_split)

        placed = False
        for _ in range(40):
            piece = _grow_piece(sub_h, sub_w, target_px)
            if len(piece) != target_px:
                continue
            if rotate and not _rotation_is_visible(piece):
                continue

            ph, pw = _piece_hw(piece)
            out_shape = _rotate_piece_cw(piece) if rotate else piece
            oh, ow = _piece_hw(out_shape)
            if ph > sub_h or pw > sub_w or oh > cap_h or ow > cap_w:
                continue

            if has_h:
                inp_r0 = int((r1 - ph + 1) if hs == "T" else r0)
            else:
                split_rows = sorted(r for r, c in split_pixels if c == c_split)
                if split_rows:
                    mid = (split_rows[0] + split_rows[-1]) // 2
                    inp_r0 = int(max(r0, min(mid - ph // 2, r1 - ph + 1)))
                else:
                    inp_r0 = int(r0)

            if has_v:
                inp_c0 = int((c1 - pw + 1) if vs == "L" else c0)
            else:
                split_cols = sorted(c for r, c in split_pixels if r == r_split)
                if split_cols:
                    mid = (split_cols[0] + split_cols[-1]) // 2
                    inp_c0 = int(max(c0, min(mid - pw // 2, c1 - pw + 1)))
                else:
                    inp_c0 = int(c0)

            inp_pts = frozenset((int(r + inp_r0), int(c + inp_c0)) for r, c in piece)
            ok = True
            for r, c in inp_pts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    ok = False
                    break
                if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                    ok = False
                    break
            if not ok:
                continue

            out_pts = _place_piece_to_corner(inp_pts, H, W, dh, dv, rotate)
            for r, c in out_pts:
                if has_v and ((vs == "L" and c >= c_split) or (vs == "R" and c <= c_split)):
                    ok = False
                    break
                if has_h and ((hs == "T" and r >= r_split) or (hs == "B" and r <= r_split)):
                    ok = False
                    break
            if not ok:
                continue

            input_pieces[reg] = inp_pts
            output_pieces[reg] = out_pts
            placed = True
            break

        if not placed:
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
            vh = int(vr1 - vr0 + 1)
            vw = int(vc1 - vc0 + 1)
            if vh < 1 or vw < 1:
                return None
            max_px = min(vanish_budget, vh * vw)
            if max_px < 3:
                return None
            piece = _grow_piece(vh, vw, max_px)
            if len(piece) < 3:
                return None
            pph, ppw = _piece_hw(piece)
            vr_off = int((vr1 - pph + 1) if vhs == "T" else vr0)
            vc_off = int((vc1 - ppw + 1) if vvs == "L" else vc0)
            vpts = {(int(r + vr_off), int(c + vc_off)) for r, c in piece}
            for r, c in vpts:
                if r < 4 or r > H - 5 or c < 4 or c > W - 5:
                    return None
                if (r, c) in all_inp_fg:
                    return None
            vanish_fg |= vpts

    combined = all_inp_fg | vanish_fg
    mask2 = np.zeros((H, W), dtype=int)
    for r, c in combined:
        mask2[int(r), int(c)] = 1
    if not is_contiguous(mask2):
        return None

    inp = np.full((H, W), bg, dtype=int)
    for r, c in combined:
        inp[int(r), int(c)] = int(fg)
    for ctrl in controls:
        for r, c in ctrl["cells"]:
            inp[int(r), int(c)] = int(ctrl["color"])

    out = _build_output(H, W, bg, fg, controls, output_pieces, split_pixels)

    fg_all = np.array(list(combined))
    final_h = int(fg_all[:, 0].max() - fg_all[:, 0].min() + 1)
    final_w = int(fg_all[:, 1].max() - fg_all[:, 1].min() + 1)
    actual_density = len(combined) / float(final_h * final_w)
    if actual_density < 0.38 or actual_density > 0.62:
        return None

    if not np.array_equal(main(inp), out):
        return None

    metrics = behavior_metrics(inp)
    if metrics["rotating_kept_regions"] > 0 and metrics["changed_pixels_vs_move_only"] == 0:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 300 attempts")
