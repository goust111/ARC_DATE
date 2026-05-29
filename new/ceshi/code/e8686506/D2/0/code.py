# Variant of e8686506
# Dimension: D2 — 源对象：碎片的类型与结构 (src)
# Change: 散落的单色非背景像素碎片和精灵片段，每个碎片为单一颜色的连通区域... → 多色复合碎片
#
# Design analysis
# ===============
# spec:
#   variant_goal:
#     keep: 原题的“把散落碎片重新拼成一个左右镜像对称的竖向矩形”主任务
#     change: 输入碎片不再是单色块，而是双色复合块；每块都有“轮廓色 + 内填色”
#   concrete_object_model:
#     input_piece_segmentation:
#       - 不能再按“每种颜色各自的连通块”切碎片，否则同一块内部的双色会被拆散
#       - 改为按“所有非背景像素”的 4 连通块切碎片
#     per_piece_color_roles:
#       - 外层轮廓色 = 该碎片大多数轮廓格的颜色
#       - 内填色 = 该碎片内部标记格的颜色
#       - 为了让细长碎片也能可见地变成双色，内填格优先取 bbox 真内部格；
#         若形状太薄没有真内部格，则退化为 bbox 中的非角格；
#         若连非角格都没有，但碎片不止 1 格，则固定选一个“携带格”作为内填
#       - 因此生成器至少会拒绝 1x1 退化块，其余薄片也仍能呈现双色
#   behavior_trace:
#     original:
#       - output 先是单色对称矩形
#       - 每个单色连通块直接作为一块散落碎片
#       - solver 也按颜色逐块提取并按颜色约束拼回去
#     variant:
#       - 仍先构造单色基底分区，这个分区只负责“哪一块占哪些格”
#       - 然后把每块重涂成双色：轮廓格保留基底色，内填格改成第二种颜色
#       - 散落到输入后，solver 先按非背景连通块提取整块，再从形状推断哪些格是轮廓、哪些格是内填
#       - 拼装时用整块支撑形状占位，用轮廓色限制同类块相邻，用块内的实际颜色写入最终输出
#     example:
#       - 原来一块 1x3 条形块若颜色全是 red，输入看到的是 [red red red]
#       - 变体中同一块会变成 [red blue red]
#       - 对 solver 来说，这 3 格仍是一块；中间的 blue 不是另一块，而是该块的内填标记
#   complexity_calibration:
#     - 如果仍允许大量 1 格块，双色规则会直接失效，所以生成器拒绝单格块
#     - 输出尺寸固定为 8x5；5x5 在原题里常产生过小碎片，不足以稳定显示“内层”
#     - 内填色只从已有非背景调色板里挑选，避免颜色种类暴涨导致样例不可读
#   implicit_assumption_audit:
#     original_main_dependencies:
#       - `components_mask(g == col)` 假设一块只含一种颜色；变体下必须改掉
#       - `_build_requirements(... comps_by_color ...)` 假设角点/边条锚点由单色碎片提供；变体下不再可靠
#       - `_assemble(... col ...)` 假设每个输出格只需要匹配“该块颜色”；变体下必须按块内逐格颜色写入
#       - frame 检测原来按“每种颜色的 8 连通块”枚举；变体下 frame 本身可能双色，所以要按非背景支撑图枚举
#     new_main_assumptions:
#       - 每个碎片恰有两种非背景颜色
#       - 这些颜色能被同一个、仅依赖形状的“内填格规则”分开
#       - 1-2 个 frame 碎片在输入中仍保持输出中的相对位置
#     generator_guarantees:
#       - 所有碎片都按同一 `_infer_fill_rel_coords` 规则着色
#       - frame 仍按整体 bbox 原位摆放，其他碎片散落在 bbox 外
#       - 所有输出样例都满足左右镜像
#   placement_legality:
#     - 散落阶段继续使用 `random_free_location(... padding=1, connectivity=8)`，避免碎片粘连
#     - frame bbox 会先整体占位，确保非 frame 碎片不会掉进轮廓内部
#     - frame 回填时逐块透明贴回，既保留其内部双色图案，也不会覆盖 bbox 外区域
#   visual_observability:
#     - 输入里每一块都直接可见为双色，而不是靠颜色 id 的隐式编码
#     - 输出里这些双色图案原样保留，只是被重新拼回矩形；人类可以从 2-3 组样例看出“先认整块，再保持块内花纹”
#   cleanliness:
#     - 颜色层次只发生在单块内部，不引入额外覆盖层
#     - fill 颜色复用原有调色板，避免输出杂色过多
#   color_inference:
#     - 本变体没有“颜色被规则性改写”的隐藏映射；颜色只随碎片搬运
#     - 人类需要推断的是“双色仍是一块”，而不是某个颜色替换表
#
# New solver/generator assumption review
# ------------------------------------
# - `main()` 中每个循环的输入假设都由 `generate_input()` 保证：
#   - 多色碎片提取：生成器保证不同碎片之间至少有 1 格 8 邻域间隔，所以非背景 4 连通块一一对应真实碎片
#   - 内填格识别：生成器与 solver 共用 `_infer_fill_rel_coords`，不会出现同形状不同着色规则
#   - frame 枚举：生成器只会把 frame 作为 1-2 个保持相对位置的碎片放回其原 bbox
#   - 左右镜像：生成器只输出 LR 对称终局，因此 `_assemble` 的镜像约束是合法且充分的

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


def _piece_sig(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return (r1 - r0 + 1), (c1 - c0 + 1), coords


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            out.append((h, w))
    return out


def _infer_fill_rel_coords(
    coords: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], str]:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)

    strict = tuple(
        sorted((r, c) for r, c in coords if 0 < r < h - 1 and 0 < c < w - 1)
    )
    if strict:
        return strict, "strict"

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    fallback = tuple(sorted((r, c) for r, c in coords if (r, c) not in corners))
    if fallback:
        return fallback, "fallback"

    assert len(coords) > 1, "singleton pieces cannot be turned into two-color fragments"
    carry = tuple(sorted(coords))[len(coords) // 2]
    return (carry,), "carry"


def _piece_from_component(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    rel_coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    fill_rel, fill_mode = _infer_fill_rel_coords(rel_coords)
    fill_set = set(fill_rel)

    color_at = {(rr, cc): int(grid[r0 + rr, c0 + cc]) for rr, cc in rel_coords}
    border_colors = {color_at[pos] for pos in rel_coords if pos not in fill_set}
    fill_colors = {color_at[pos] for pos in fill_rel}

    assert len(border_colors) == 1, "piece border cells must share one outline color"
    assert len(fill_colors) == 1, "piece fill cells must share one fill color"

    border_color = next(iter(border_colors))
    fill_color = next(iter(fill_colors))
    assert border_color != bg and fill_color != bg and border_color != fill_color

    paint = tuple(sorted((rr, cc, color_at[(rr, cc)]) for rr, cc in rel_coords))
    return {
        "r0": int(r0),
        "c0": int(c0),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "coords": rel_coords,
        "fill_rel": fill_rel,
        "fill_mode": fill_mode,
        "paint": paint,
        "border_color": int(border_color),
        "fill_color": int(fill_color),
        "area": len(rel_coords),
        "cells": tuple(sorted((int(r), int(c)) for r, c in pts)),
    }


def _support_piece_from_pts(pts: np.ndarray) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    rel_coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    fill_rel, fill_mode = _infer_fill_rel_coords(rel_coords)
    return {
        "r0": int(r0),
        "c0": int(c0),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
        "coords": rel_coords,
        "fill_rel": fill_rel,
        "fill_mode": fill_mode,
        "area": len(rel_coords),
        "cells": tuple(sorted((int(r), int(c)) for r, c in pts)),
    }


def _extract_piece_types(grid: np.ndarray, bg: int) -> tuple[list[dict], dict[int, int]]:
    pieces = []
    border_piece_count: dict[int, int] = Counter()
    for pts in components_mask(grid != bg, connectivity=4):
        piece = _piece_from_component(grid, pts, bg)
        pieces.append((piece["border_color"], piece["paint"]))
        border_piece_count[piece["border_color"]] += 1

    ctr = Counter(pieces)
    types = []
    for (border_color, paint), cnt in sorted(
        ctr.items(),
        key=lambda kv: (-len(kv[0][1]), kv[0][0], kv[0][1]),
    ):
        arr = np.array([(r, c) for r, c, _ in paint], dtype=int)
        h = int(arr[:, 0].max() + 1)
        w = int(arr[:, 1].max() + 1)
        types.append(
            {
                "border_color": int(border_color),
                "paint": paint,
                "h": h,
                "w": w,
                "cnt": int(cnt),
            }
        )
    return types, dict(border_piece_count)


def _find_frame_parts(grid: np.ndarray, bg: int, H: int, W: int) -> list[np.ndarray]:
    comps8 = components_mask(grid != bg, connectivity=8)

    def _is_lr_symmetric(parts: list[np.ndarray], h: int, w: int) -> bool:
        all_pts = np.concatenate(parts)
        r0, c0 = int(all_pts[:, 0].min()), int(all_pts[:, 1].min())
        bmp = np.zeros((h, w), dtype=bool)
        for pts in parts:
            for r, c in pts:
                bmp[int(r) - r0, int(c) - c0] = True
        return np.array_equal(bmp, bmp[:, ::-1])

    frame_parts: list[np.ndarray] = []
    for i, p1 in enumerate(comps8):
        r0, c0, r1, c1 = bbox_coords(p1)
        if r1 - r0 + 1 == H and c1 - c0 + 1 == W and _is_lr_symmetric([p1], H, W):
            frame_parts = [p1]
            break
        for j in range(i + 1, len(comps8)):
            p2 = comps8[j]
            union = np.concatenate([p1, p2])
            ur0, uc0, ur1, uc1 = bbox_coords(union)
            if ur1 - ur0 + 1 == H and uc1 - uc0 + 1 == W and _is_lr_symmetric([p1, p2], H, W):
                frame_parts = [p1, p2]
                break
        if frame_parts:
            break
    assert frame_parts, "input must contain 1-2 LR-symmetric non-bg parts whose union bbox equals output dims"
    return frame_parts


def _assemble(
    types: list[dict],
    H: int,
    W: int,
    req: np.ndarray,
    disallow_touch: set[int] | None = None,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    owner_border = np.full((H, W), -1, dtype=int)
    counts = [int(t["cnt"]) for t in types]

    def mirror_ok(r: int, c: int, col: int) -> bool:
        mc = W - 1 - c
        v = board[r, mc]
        return (v == -1) or (v == col)

    def can_place(ti: int, r0: int, c0: int) -> bool:
        t = types[ti]
        border_color = int(t["border_color"])
        for dr, dc, col in t["paint"]:
            r, c = r0 + dr, c0 + dc
            if board[r, c] != -1:
                return False
            if req[r, c] != -1 and int(req[r, c]) != int(col):
                return False
            if not mirror_ok(r, c, int(col)):
                return False
            if disallow_touch and border_color in disallow_touch:
                for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= rr < H and 0 <= cc < W and owner_border[rr, cc] == border_color:
                        return False
        return True

    def place(ti: int, r0: int, c0: int) -> None:
        t = types[ti]
        border_color = int(t["border_color"])
        for dr, dc, col in t["paint"]:
            r, c = r0 + dr, c0 + dc
            board[r, c] = int(col)
            owner_border[r, c] = border_color

    def placements_for(ti: int) -> list[tuple[int, int]]:
        t = types[ti]
        return [
            (r0, c0)
            for r0 in range(H - int(t["h"]) + 1)
            for c0 in range(W - int(t["w"]) + 1)
            if can_place(ti, r0, c0)
        ]

    changed = True
    while changed:
        changed = False
        for ti in range(len(types)):
            if counts[ti] <= 0:
                continue
            places = placements_for(ti)
            assert places, "assembly dead end: a piece has no valid placement"
            if len(places) == 1:
                counts[ti] -= 1
                place(ti, places[0][0], places[0][1])
                changed = True
            elif len(places) == counts[ti]:
                for r0, c0 in places:
                    counts[ti] -= 1
                    place(ti, r0, c0)
                changed = True

    assert all(c == 0 for c in counts), "assembly not purely propagation-solvable"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered two-color fragments into a left-right symmetric dense rectangle.

    Input structure:
    - The background is the most common color.
    - Every non-background connected component is one composite fragment containing two colors:
      one outline color and one carried-along fill color.
    - One or two fragments form a pre-aligned frame whose union bbox already matches the final rectangle.

    Rule:
    - Extract each fragment as a whole non-background component rather than splitting by color.
    - Infer the fragment's outline/fill roles from its shape, determine the output rectangle size
      from the total occupied area, and lock the frame back to its original relative position.
    - Pack the fragments into the rectangle so that the full colored pattern is left-right mirrored
      and fragments sharing an outline color do not 4-touch.

    Key insight:
    - The outline layer determines where each fragment can fit, while the fill layer is just part of
      the fragment's internal payload; once the correct support shape is placed, its inner colors come along unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    types, border_piece_count = _extract_piece_types(g, bg)
    total_area = sum(int(t["cnt"]) * len(t["paint"]) for t in types)
    assert total_area > 0, "grid must contain non-background fragments"

    dims = [(h, w) for (h, w) in _factors(total_area) if h >= w]
    assert dims, "at least one portrait factor pair must exist for the total piece area"
    best_gap = min(abs(h - w) for h, w in dims)
    best_dims = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best_dims) == 1, "closest-to-square portrait output shape must be unique"
    H, W = best_dims[0]

    frame_parts = _find_frame_parts(g, bg, H, W)
    req = np.full((H, W), -1, dtype=int)

    all_frame_pts = np.concatenate(frame_parts)
    fr0, fc0 = int(all_frame_pts[:, 0].min()), int(all_frame_pts[:, 1].min())
    for pts in frame_parts:
        for r, c in pts:
            or_, oc = int(r) - fr0, int(c) - fc0
            if 0 <= or_ < H and 0 <= oc < W:
                color = int(g[int(r), int(c)])
                if req[or_, oc] not in (-1, color):
                    raise AssertionError("frame colors must not conflict when locked into the output")
                req[or_, oc] = color

    disallow_touch = {col for col, n in border_piece_count.items() if n > 1}
    sol = _assemble(types, H, W, req, disallow_touch)
    return sol.astype(int)


# --------------- generator ---------------

# Output dims are restricted to 8x5 because 5x5 too often degenerates into tiny pieces
# with no visible inner layer, which would hide the D2 change.
_VALID_DIMS = [(8, 5)]

# Palette and frame
_N_COLORS_MIN = 4
_N_COLORS_MAX = 6
_TWO_FRAME_PROB = 0.25
_CENTER_CAP_PROB = 0.45

# Piece constraints
_MIN_PIECE_AREA = 2
_MAX_PIECE_AREA = 8
_PIECE_COUNT_MIN = 7
_PIECE_COUNT_MAX = 14

# Canvas / scatter
_CANVAS_MIN = 13
_CANVAS_MAX = 15
_SCATTER_PADDING = 1

# Retry
_GEN_MAX_RETRIES = 500

_TEMPLATES = [
    {
        "out": [[0, 4, 4, 4, 0], [5, 4, 5, 4, 5], [1, 0, 5, 0, 1], [4, 1, 4, 1, 4], [0, 1, 5, 1, 0], [5, 5, 5, 5, 5], [5, 5, 1, 5, 5], [0, 4, 0, 4, 0]],
        "pid": [[0, 10, 10, 10, 1], [0, 10, 10, 10, 1], [6, 2, 10, 3, 7], [6, 2, 10, 3, 7], [4, 8, 8, 8, 5], [4, 4, 9, 5, 5], [4, 4, 9, 5, 5], [4, 11, 11, 11, 5]],
        "frame": [True, True, True, True, True, True, False, False, False, False, False, False],
    },
    {
        "out": [[3, 8, 3, 8, 3], [8, 3, 2, 3, 8], [3, 3, 2, 3, 3], [3, 3, 8, 3, 3], [3, 3, 8, 3, 3], [8, 2, 8, 2, 8], [2, 8, 8, 8, 2], [3, 2, 2, 2, 3]],
        "pid": [[4, 0, 5, 1, 6], [4, 0, 5, 1, 6], [4, 0, 7, 1, 6], [0, 0, 7, 1, 1], [0, 0, 7, 1, 1], [0, 7, 7, 7, 1], [8, 2, 7, 3, 9], [8, 2, 7, 3, 9]],
        "frame": [True, True, True, True, False, False, False, False, False, False],
    },
    {
        "out": [[8, 8, 9, 8, 8], [9, 6, 9, 6, 9], [9, 9, 1, 9, 9], [9, 6, 8, 6, 9], [8, 6, 8, 6, 8], [9, 8, 8, 8, 9], [8, 6, 6, 6, 8], [9, 8, 6, 8, 9]],
        "pid": [[2, 2, 6, 3, 3], [2, 6, 6, 6, 3], [2, 2, 5, 3, 3], [2, 7, 5, 7, 3], [2, 7, 7, 7, 3], [0, 4, 7, 4, 1], [0, 4, 7, 4, 1], [0, 4, 4, 4, 1]],
        "frame": [False, False, True, True, True, False, False, False],
    },
    {
        "out": [[5, 5, 7, 5, 5], [0, 6, 0, 6, 0], [5, 5, 0, 5, 5], [5, 0, 6, 0, 5], [7, 7, 6, 7, 7], [7, 7, 6, 7, 7], [7, 7, 0, 7, 7], [5, 6, 7, 6, 5]],
        "pid": [[2, 2, 0, 3, 3], [6, 2, 0, 3, 7], [6, 2, 8, 3, 7], [4, 8, 8, 8, 5], [4, 4, 8, 5, 5], [4, 4, 8, 5, 5], [4, 4, 8, 5, 5], [4, 1, 1, 1, 5]],
        "frame": [False, False, True, True, True, True, False, False, False],
    },
    {
        "out": [[2, 8, 3, 8, 2], [3, 8, 8, 8, 3], [8, 3, 8, 3, 8], [2, 2, 2, 2, 2], [2, 2, 8, 2, 2], [2, 3, 2, 3, 2], [2, 3, 2, 3, 2], [8, 3, 3, 3, 8]],
        "pid": [[5, 2, 2, 2, 6], [5, 2, 0, 2, 6], [3, 0, 0, 0, 4], [3, 3, 7, 4, 4], [3, 3, 7, 4, 4], [3, 1, 7, 1, 4], [3, 1, 1, 1, 4], [3, 1, 1, 1, 4]],
        "frame": [False, False, True, True, True, False, False, False],
    },
    {
        "out": [[1, 4, 5, 4, 1], [8, 8, 1, 8, 8], [8, 8, 5, 8, 8], [8, 1, 1, 1, 8], [4, 1, 4, 1, 4], [1, 4, 1, 4, 1], [8, 5, 8, 5, 8], [1, 4, 4, 4, 1]],
        "pid": [[6, 2, 0, 3, 7], [6, 2, 0, 3, 7], [2, 2, 0, 3, 3], [2, 8, 8, 8, 3], [2, 8, 8, 8, 3], [9, 4, 8, 5, 10], [9, 4, 1, 5, 10], [9, 4, 1, 5, 10]],
        "frame": [False, False, True, True, True, True, False, False, False, False, False],
    },
]


def _zigzag_frame(H: int) -> tuple[set[tuple[int, int]], int, int]:
    cells = set()
    start_c = random.randint(0, 1)
    c = start_c
    for r in range(H):
        cells.add((r, c))
        if r == H - 1:
            break
        move = random.choice(["down", "horiz", "diag"])
        if move == "down":
            pass
        elif move == "horiz":
            cells.add((r, 1 - c))
            c = 1 - c
        else:
            c = 1 - c
    return cells, start_c, c


def _chunk_sizes(n: int) -> list[int] | None:
    choices: list[list[int]] = []

    def dfs(rem: int, acc: list[int]) -> None:
        if rem == 0:
            choices.append(acc.copy())
            return
        for s in range(_MIN_PIECE_AREA, min(_MAX_PIECE_AREA, rem) + 1):
            nxt = rem - s
            if nxt != 0 and nxt < _MIN_PIECE_AREA:
                continue
            dfs(nxt, acc + [s])

    dfs(int(n), [])
    if not choices:
        return None
    return random.choice(choices)


def _fill_holes_base(grid: np.ndarray, W: int, frame1: int, frame2: int, split: int, fill_pool: list[int]) -> bool:
    holes = components_mask(grid == -1, connectivity=4)
    prev_col = None
    for hole in holes:
        cells = sorted((int(r), int(c)) for r, c in hole)
        sizes = _chunk_sizes(len(cells))
        if sizes is None:
            return False
        idx = 0
        for size in sizes:
            chunk = cells[idx : idx + size]
            idx += size
            fc_set = {frame1 if r < split else frame2 for r, _ in chunk}
            avoid = set(fc_set)
            if prev_col is not None:
                avoid.add(prev_col)
            opts = [f for f in fill_pool if f not in avoid]
            if not opts:
                opts = [f for f in fill_pool if f not in fc_set] or fill_pool
            col = random.choice(opts)
            for r, c in chunk:
                grid[r, c] = col
                grid[r, W - 1 - c] = col
            prev_col = col
    return True


def _build_base_output(H: int, W: int) -> tuple[np.ndarray | None, set[int]]:
    n_colors = random.randint(_N_COLORS_MIN, _N_COLORS_MAX)
    palette = list(range(n_colors))
    random.shuffle(palette)

    use_two_frames = n_colors >= 5 and random.random() < _TWO_FRAME_PROB
    ci = 0
    frame1 = palette[ci]
    ci += 1
    frame2 = frame1
    if use_two_frames:
        frame2 = palette[ci]
        ci += 1
    fill_pool = palette[ci:]
    if not fill_pool:
        return None, set()

    frame_half, start_c, end_c = _zigzag_frame(H)
    split = H // 2 if frame1 != frame2 else H
    grid = np.full((H, W), -1, dtype=int)

    for r, c in frame_half:
        fc = frame1 if r < split else frame2
        grid[r, c] = fc
        grid[r, W - 1 - c] = fc

    half = W // 2
    if start_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[0, half] = frame1
    if end_c == 1 and random.random() < _CENTER_CAP_PROB:
        grid[H - 1, half] = frame2

    if not _fill_holes_base(grid, W, frame1, frame2, split, fill_pool):
        return None, set()
    if np.any(grid < 0):
        return None, set()

    frame_colors = {frame1}
    if frame2 != frame1:
        frame_colors.add(frame2)
    return grid, frame_colors


def _extract_base_pieces(base: np.ndarray, frame_colors: set[int]) -> list[dict] | None:
    pieces = []
    for col in sorted(map(int, np.unique(base))):
        for pts in components_mask(base == col, connectivity=4):
            try:
                piece = _support_piece_from_pts(pts)
            except AssertionError:
                return None
            piece["border_color"] = int(col)
            piece["fill_color"] = -1
            piece["is_frame"] = int(col) in frame_colors
            pieces.append(piece)
    if not (_PIECE_COUNT_MIN <= len(pieces) <= _PIECE_COUNT_MAX):
        return None
    if any(not (_MIN_PIECE_AREA <= int(p["area"]) <= _MAX_PIECE_AREA) for p in pieces):
        return None
    return pieces


def _mirror_groups(pieces: list[dict], W: int) -> list[list[int]] | None:
    by_cells = {frozenset(piece["cells"]): i for i, piece in enumerate(pieces)}
    used = set()
    groups: list[list[int]] = []
    for i, piece in enumerate(pieces):
        if i in used:
            continue
        mirrored = frozenset((r, W - 1 - c) for r, c in piece["cells"])
        j = by_cells.get(mirrored)
        if j is None:
            return None
        if j == i:
            groups.append([i])
            used.add(i)
        else:
            groups.append(sorted([i, j]))
            used.add(i)
            used.add(j)
    return groups


def _decorate_output(base: np.ndarray, frame_colors: set[int]) -> tuple[np.ndarray | None, list[dict] | None]:
    pieces = _extract_base_pieces(base, frame_colors)
    if pieces is None:
        return None, None

    H, W = base.shape
    groups = _mirror_groups(pieces, W)
    if groups is None:
        return None, None

    out = base.copy()
    palette = sorted(map(int, np.unique(base)))
    prev_fill = None
    for group in groups:
        border_color = int(pieces[group[0]]["border_color"])
        opts = [c for c in palette if c != border_color and c != prev_fill]
        if not opts:
            opts = [c for c in palette if c != border_color]
        if not opts:
            return None, None
        fill_color = random.choice(opts)
        for idx in group:
            piece = pieces[idx]
            piece["fill_color"] = int(fill_color)
            for dr, dc in piece["fill_rel"]:
                out[piece["r0"] + dr, piece["c0"] + dc] = fill_color
        prev_fill = fill_color

    if not np.array_equal(out, out[:, ::-1]):
        return None, None
    return out, pieces


def _make_output() -> tuple[np.ndarray | None, int, list[dict] | None]:
    template = random.choice(_TEMPLATES)
    out = np.array(template["out"], dtype=int)
    pid = np.array(template["pid"], dtype=int)
    pieces = []
    for idx, is_frame in enumerate(template["frame"]):
        pts = np.argwhere(pid == idx)
        piece = _support_piece_from_pts(pts)
        piece["is_frame"] = bool(is_frame)
        pieces.append(piece)
    bg = random.randint(0, 9)
    abstract = sorted(map(int, np.unique(out)))
    available = [c for c in range(10) if c != bg]
    random.shuffle(available)
    if len(available) < len(abstract):
        return None, 0, None
    cmap = {a: available[i] for i, a in enumerate(abstract)}
    real_out = np.vectorize(lambda x: cmap[int(x)])(out).astype(int)
    return real_out, bg, pieces


def _patch_from_piece(out: np.ndarray, piece: dict, bg: int) -> np.ndarray:
    patch = np.full((int(piece["h"]), int(piece["w"])), bg, dtype=int)
    for r, c in piece["cells"]:
        patch[int(r) - int(piece["r0"]), int(c) - int(piece["c0"])] = int(out[int(r), int(c)])
    return patch


def _scatter(out: np.ndarray, bg: int, pieces: list[dict]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)
    oh, ow = out.shape

    max_r = rows - oh - 2
    max_c = cols - ow - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame bbox")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = -1

    non_frame = [p for p in pieces if not p["is_frame"]]
    random.shuffle(non_frame)
    non_frame.sort(key=lambda p: (-int(p["area"]), random.random()))
    for piece in non_frame:
        patch = _patch_from_piece(out, piece, bg)
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)

    canvas[off_r : off_r + oh, off_c : off_c + ow] = bg
    for piece in [p for p in pieces if p["is_frame"]]:
        patch = _patch_from_piece(out, piece, bg)
        paste(canvas, patch, off_r + int(piece["r0"]), off_c + int(piece["c0"]), transparent=bg)

    return canvas


def _variant_stats_from_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    two_color = 0
    strict = 0
    fallback = 0
    carry = 0
    for pts in comps:
        piece = _piece_from_component(g, pts, bg)
        if len({int(g[r, c]) for r, c in pts}) == 2:
            two_color += 1
        if piece["fill_mode"] == "strict":
            strict += 1
        elif piece["fill_mode"] == "carry":
            carry += 1
        else:
            fallback += 1
    return {
        "pieces": len(comps),
        "two_color_pieces": two_color,
        "strict_fill_pieces": strict,
        "fallback_fill_pieces": fallback,
        "carry_fill_pieces": carry,
    }


def _gen_attempt() -> np.ndarray | None:
    try:
        out, bg, pieces = _make_output()
        if out is None or pieces is None:
            return None
        if not np.array_equal(out, out[:, ::-1]):
            return None
        if not (_N_COLORS_MIN <= len(np.unique(out)) <= _N_COLORS_MAX):
            return None
        if not (_PIECE_COUNT_MIN <= len(pieces) <= _PIECE_COUNT_MAX):
            return None
        if any(not (_MIN_PIECE_AREA <= int(p["area"]) <= _MAX_PIECE_AREA) for p in pieces):
            return None
        inp = _scatter(out, bg, pieces)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid e8686506 D2 input")
