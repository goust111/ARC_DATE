# Combo variant of 5961cc34 (index 10)
# combo_spec: D2/3+D5/0+D10/2
# Name: 条件激活+多端口色块+长链级联

from __future__ import annotations

from collections import Counter
import importlib.util
from itertools import combinations
from pathlib import Path
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


_ORIGINAL_MAIN = None
_ORIGINAL_SOLVER = Path("/home/lixinyu/snap/new/arc2_solver_select/5961cc34.py")


def _original_main():
    global _ORIGINAL_MAIN
    if _ORIGINAL_MAIN is None:
        spec = importlib.util.spec_from_file_location("orig_5961cc34_combo10", _ORIGINAL_SOLVER)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _ORIGINAL_MAIN = module.main
    return _ORIGINAL_MAIN


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) 具体化与行为追踪
#    - 保留原题的可视元素: 底部黄帽+红杆起点、蓝色菱形/八边形 blob、绿色端口条、输出中的红色射线网络。
#    - 融合后每个 blob 有 2-3 个端口, 但不会“命中即激活”:
#      * 主干上的激活 blob 统一做成 3 端口, 因而它恰好剩 1 条没有绿色的边。
#        射线只有从这条“唯一无绿边”射入时才会激活它; 激活后 3 个端口全部发射。
#      * 某条支路上会专门放一个多端口 blocker。它会被红线实际打中, 但命中方向不是其允许入口,
#        所以只会出现“红线穿过蓝块”的现象, 不会整块转红。
#    - 一个典型小例子:
#      * 起始上射线打到根 blob, 根 blob 的下边是唯一无绿边, 因此被激活。
#      * 根 blob 同时向上、左、右三侧发射。左侧分支可能直接到边界, 右侧分支继续接到第二个激活 blob,
#        上侧分支则穿过一个 2 端口 blocker 后出界。
#      * 第二个 blob 再重复该过程, 于是得到一条 4-6 个激活块的长主干, 外加多条侧枝。
#    - 相比原题, 差异不再只是“单链继续”, 而是“树状扩散 + 选择性激活 + 可见的错误方向穿透”同时出现。
#
# b) 复杂度校准
#    - 颜色仍只有 5 种, 规律完全由几何关系提供, 不靠颜色 id。
#    - 激活主干固定为 4-6 个 blob, 保证推理步数显著增加。
#    - 主干 blob 固定 3 端口, 让“唯一无绿边 = 正确入口”在少量样例里可重复出现、可归纳。
#    - blocker 优先使用 2 端口, 从而在不把核心激活规则做得过于隐晦的前提下, 让 2-3 端口两种形态都出现。
#
# c) 对原 solver 隐式假设的审查
#    - 原版按蓝色连通分量分 blob。多端口时若只看蓝色, 某些小 shape 会被绿色切断, 所以这里必须改成
#      蓝+绿联合连通分量。
#    - 原版 `hit blue -> activate -> break` 不成立。现在命中后需要先判断入射方向, 错误方向要继续穿透。
#    - 原版默认每个 blob 只有一个端口; 现在一个激活块会把多个分支都压入 BFS 传播。
#    - 原版末尾对“未激活块数量”的约束不成立。这里故意保留 1 个被命中但未激活的 blocker。
#
# d) 放置合法性
#    - 生成器仍然 output-first, 但把每个激活 blob 拆成三类射线:
#      * 1 条主干分支, 继续放下一个可激活 3 端口 blob
#      * 若干终止分支, 直接射到边界
#      * 恰好 1 条 blocker 分支, 先穿过一个错误方向命中的多端口 blob, 再射到边界
#    - 所有 blob 都要求完整在界内、彼此保持至少 1 像素 8 邻域间隔, 且新射线不能与既有 blob / 射线交叉。
#    - blocker 还要求当前厚度能在其入边和出边上对齐, 使“直穿”在像素层面真正成立。
#
# e) 视觉可观察性
#    - 解题者可以直接看到三件事:
#      * 主干上的 blob 都有多条绿色端口, 输出中确实会长出分支
#      * 这些 blob 只有从无绿的一侧被打中时才会整块转红
#      * 至少有一个多端口 blob 被红线穿过却没有整块变红, 迫使人注意“方向条件”
#    - 这三件事在 2-4 组样例中会重复出现, 因而不会退化成偶然布局。
#
# f) 可视化整洁度
#    - 采用单主干 + 侧枝终止的树形生成, 避免多层交叉。
#    - 一旦某条侧枝被提交到输出, 后续放置会把它当作不可穿越障碍, 保持图面清晰。
#    - 网格尺寸整体放大到 23-34 左右, 给 4-6 个激活块和支路留出足够空间。
#
# g) 变化的可推断性
#    - 仍只有蓝/绿/红/黄/bg 五色, 没有新增颜色映射问题。
#    - 绿色只承担“端口布局”这一种语义; 红色只表示保留下来的网络。
#    - 新信息来自端口集合与入射边的关系: 射线是整块激活, 还是仅沿几条 lane 穿过, 都是肉眼可见的。
#
# 新 main() 的额外假设与 generator 的配合
#    1. 所有真正参与长链激活的 blob 都是 3 端口, 因而可激活入口在视觉上是唯一无绿边。
#    2. 2 端口 blob 只作为 blocker 使用, generator 保证它不会恰好从自己的允许方向被再次命中。
#    3. 每条主干只有一个继续扩展的端口, 其余端口都在当前步骤被终止到边界或 blocker, 因此 solver 的 BFS
#       会得到稳定树形而不会出现歧义交叉。
#    4. 不会有两条不同分支同时命中同一未激活 blob; generator 通过 ray_mask / occupied 检查保证这一点。
# ---------------------------------------------------------------------------


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}
CLOCKWISE = (UP, RIGHT, DOWN, LEFT)

# --- Diamond/octagon shapes (h, w) ---
SHAPES = ((3, 4), (4, 4), (5, 4), (5, 6), (6, 6), (8, 8))
_SHAPE_WIDTHS = {
    (3, 4): (2, 4, 2),
    (4, 4): (2, 4, 4, 2),
    (5, 4): (2, 4, 4, 4, 2),
    (5, 6): (2, 4, 6, 4, 2),
    (6, 6): (2, 4, 6, 6, 4, 2),
    (8, 8): (2, 4, 6, 8, 8, 6, 4, 2),
}

# --- Canvas bounds ---
H_MIN, H_MAX = 24, 38
W_MIN, W_MAX = 23, 34
START_COL_PAD = 3

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / blocker ---
ACTIVATED_COUNT_CHOICES = (4, 5, 6)
GEN_RETRIES = 500
MIN_GAP_FIRST = 5
MIN_GAP_CHAIN = 4
ACTIVE_CHAIN_SIZES = {(3, 4), (4, 4)}


def _required_incoming(port_dirs: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    """Recover the unique allowed ray direction from the port layout.

    For 3-port blobs this is simply the edge with no green markers.
    For 2-port blobs the generator never relies on activation, but the solver still
    needs a deterministic rule, so it takes the first green-free edge in clockwise
    compass order.
    """
    missing = [d for d in CLOCKWISE if d not in port_dirs]
    assert missing
    entry_side = missing[0]
    return OPP[entry_side]


def _sort_dirs(dirs: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    return tuple(d for d in CLOCKWISE if d in dirs)


def _shape_mask(h: int, w: int) -> np.ndarray:
    widths = _SHAPE_WIDTHS[(h, w)]
    mask = np.zeros((h, w), dtype=bool)
    for r, ww in enumerate(widths):
        c0 = (w - ww) // 2
        mask[r, c0 : c0 + ww] = True
    return mask


def _perp_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[1] if d[0] else rc[0]


def _along_value(rc: tuple[int, int], d: tuple[int, int]) -> int:
    return rc[0] if d[0] else rc[1]


def _sort_on_ray(coords: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    return sorted(coords, key=lambda p: _perp_value(p, d))


def _edge_offsets(mask: np.ndarray, d: tuple[int, int]) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    axis = 0 if d in (UP, DOWN) else 1
    extreme = pts[:, axis].min() if d in (UP, LEFT) else pts[:, axis].max()
    keep = pts[:, axis] == extreme
    return _sort_on_ray([(int(r), int(c)) for r, c in pts[keep]], d)


def _center_port_window(edge: list[tuple[int, int]], d: tuple[int, int]) -> list[tuple[int, int]]:
    width = min(2, len(edge))
    start = (len(edge) - width) // 2
    return _sort_on_ray(edge[start : start + width], d)


def _blob_spec(size: tuple[int, int], port_dirs: tuple[tuple[int, int], ...]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)

    ports_local: dict[tuple[int, int], list[tuple[int, int]]] = {}
    green_set: set[tuple[int, int]] = set()
    for d in port_dirs:
        edge = _edge_offsets(mask, d)
        green = _center_port_window(edge, d)
        if not (1 <= len(green) <= 2):
            return None
        if green_set & set(green):
            return None
        port_pixels: list[tuple[int, int]] = []
        for gr, gc in green:
            br = gr - d[0]
            bc = gc - d[1]
            if not (0 <= br < h and 0 <= bc < w):
                return None
            if not mask[br, bc] or (br, bc) in green_set:
                return None
            port_pixels.append((int(br), int(bc)))
        if len(set(port_pixels)) != len(port_pixels):
            return None
        green_set |= set(green)
        ports_local[d] = _sort_on_ray(port_pixels, d)

    blue_mask = mask.copy()
    for r, c in green_set:
        blue_mask[r, c] = False
    if len(components_mask(blue_mask, connectivity=8)) != 1:
        return None

    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green_set:
        patch[r, c] = GREEN

    port_dirs = _sort_dirs(port_dirs)
    blue = [(int(r), int(c)) for r, c in np.argwhere(blue_mask)]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dirs": port_dirs,
        "ports_by_dir": ports_local,
        "required_in": _required_incoming(port_dirs),
    }


def _build_all_specs() -> list[dict]:
    specs: list[dict] = []
    for size in SHAPES:
        for k in (2, 3):
            for dirs in combinations(DIRS, k):
                spec = _blob_spec(size, tuple(dirs))
                if spec is not None:
                    specs.append(spec)
    return specs


ALL_SPECS = _build_all_specs()
ACTIVE_SPECS = [spec for spec in ALL_SPECS if len(spec["port_dirs"]) == 3]
BLOCKER2_SPECS = [spec for spec in ALL_SPECS if len(spec["port_dirs"]) == 2]


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _path_to_entries(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    entries: list[tuple[int, int]],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]] | None:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for (r, c), (er, ec) in zip(starts, entries):
        if include_origin:
            cells.append((r, c))
        guard = H + W + 10
        while (r, c) != (er, ec):
            r += dr
            c += dc
            guard -= 1
            if guard < 0 or not (0 <= r < H and 0 <= c < W):
                return None
            cells.append((r, c))
    return cells


def _path_to_boundary(
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    H: int,
    W: int,
) -> list[tuple[int, int]]:
    dr, dc = d
    cells: list[tuple[int, int]] = []
    for r, c in starts:
        if include_origin:
            cells.append((r, c))
        while True:
            r += dr
            c += dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _path_is_clear(
    path: list[tuple[int, int]],
    occupied: np.ndarray,
    ray_mask: np.ndarray,
    allowed: set[tuple[int, int]],
) -> bool:
    for r, c in path:
        if (r, c) in allowed:
            continue
        if occupied[r, c] or ray_mask[r, c]:
            return False
    return True


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _clone_state(state: dict) -> dict:
    return {
        "inp": state["inp"].copy(),
        "out": state["out"].copy(),
        "occupied": state["occupied"].copy(),
        "blue_mask": state["blue_mask"].copy(),
        "ray_mask": state["ray_mask"].copy(),
    }


def _extract_blobs(grid: np.ndarray) -> tuple[int, int, int, int, int, list[tuple[int, int]], list[dict]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    start_candidates: list[tuple[int, int, int, list[tuple[int, int]]]] = []
    for col in colors_present(g, exclude={bg}):
        for coords in components_mask(g == int(col), connectivity=8):
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
            if c0 != c1 or (r1 - r0 + 1) < 2:
                continue
            if r1 != H - 1:
                continue
            if any(int(g[rr, c0]) != int(col) for rr in range(r0, r1 + 1)):
                continue
            if r0 - 1 < 0:
                continue
            cap = int(g[r0 - 1, c0])
            if cap == bg or cap == int(col):
                continue
            start_candidates.append(
                (int(col), int(cap), int(c0), [(int(rr), int(c0)) for rr in range(r0, r1 + 1)])
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    object_cols = colors_present(g, exclude={bg, red_col, yellow_col})
    assert len(object_cols) == 2

    green_candidates: list[int] = []
    for col in object_cols:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        other = next(other_col for other_col in object_cols if int(other_col) != int(col))
        if all(
            any(
                0 <= int(r + dr) < H
                and 0 <= int(c + dc) < W
                and int(g[int(r + dr), int(c + dc)]) == int(other)
                for dr, dc in DIRS
            )
            for r, c in pts
        ):
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])
    blue_col = next(int(col) for col in object_cols if int(col) != int(green_col))

    blobs: list[dict] = []
    combo_mask = (g == blue_col) | (g == green_col)
    for coords in components_mask(combo_mask, connectivity=8):
        full = [(int(r), int(c)) for r, c in coords]
        blue_pts = [(r, c) for r, c in full if int(g[r, c]) == blue_col]
        green_pts = [(r, c) for r, c in full if int(g[r, c]) == green_col]
        assert blue_pts and green_pts
        min_r = min(r for r, _c in full)
        max_r = max(r for r, _c in full)
        min_c = min(c for _r, c in full)
        max_c = max(c for _r, c in full)
        edge_greens = {
            UP: [(r, c) for r, c in green_pts if r == min_r],
            DOWN: [(r, c) for r, c in green_pts if r == max_r],
            LEFT: [(r, c) for r, c in green_pts if c == min_c],
            RIGHT: [(r, c) for r, c in green_pts if c == max_c],
        }
        ports = []
        blue_set = set(blue_pts)
        port_dirs: list[tuple[int, int]] = []
        for d in CLOCKWISE:
            greens = edge_greens[d]
            if not greens:
                continue
            port_pixels = []
            for gr, gc in greens:
                br = gr - d[0]
                bc = gc - d[1]
                assert (br, bc) in blue_set
                port_pixels.append((int(br), int(bc)))
            port_dirs.append(d)
            ports.append({"dir": d, "pixels": _sort_on_ray(port_pixels, d)})
        assert 2 <= len(ports) <= 3
        port_dir_tuple = _sort_dirs(port_dirs)
        blobs.append(
            {
                "full": full,
                "blue": blue_pts,
                "green": green_pts,
                "ports": ports,
                "port_dirs": port_dir_tuple,
                "required_in": _required_incoming(port_dir_tuple),
            }
        )

    return bg, red_col, yellow_col, blue_col, start_col, start_seg, blobs


def _solve(grid: np.ndarray, *, collect_stats: bool = False) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, red_col, yellow_col, blue_col, start_col, start_seg, blobs = _extract_blobs(g)

    out = np.full_like(g, bg)
    blob_id = np.full((H, W), -1, dtype=int)
    for bi, blob in enumerate(blobs):
        for r, c in blob["blue"]:
            blob_id[r, c] = int(bi)

    stats = {
        "activations": 0,
        "wrong_dir_hits": 0,
        "pass_through_cells": 0,
        "activated_blob_port_counts": [],
        "wrong_hit_blob_port_counts": [],
        "two_port_blobs": sum(1 for blob in blobs if len(blob["ports"]) == 2),
        "three_port_blobs": sum(1 for blob in blobs if len(blob["ports"]) == 3),
        "total_blobs": len(blobs),
    }

    activated: set[int] = set()
    queue: list[int] = []
    counted_wrong_hits: set[tuple[int, tuple[int, int]]] = set()

    def activate_blob_at(r: int, c: int) -> None:
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(bi)
        queue.append(bi)
        stats["activations"] += 1
        stats["activated_blob_port_counts"].append(len(blobs[bi]["ports"]))
        paint_coords(out, blobs[bi]["blue"], red_col)

    def extend_ray(origins: list[tuple[int, int]], d: tuple[int, int]) -> None:
        dr, dc = d
        for pr, pc in origins:
            r, c = int(pr), int(pc)
            while True:
                r += dr
                c += dc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[r, c] = red_col
                if int(g[r, c]) != blue_col:
                    continue
                bi = int(blob_id[r, c])
                assert bi >= 0
                if bi in activated:
                    break
                if tuple(d) == tuple(blobs[bi]["required_in"]):
                    activate_blob_at(r, c)
                    break
                key = (bi, tuple(d))
                if key not in counted_wrong_hits:
                    counted_wrong_hits.add(key)
                    stats["wrong_dir_hits"] += 1
                    stats["wrong_hit_blob_port_counts"].append(len(blobs[bi]["ports"]))
                stats["pass_through_cells"] += 1

    for r, c in start_seg:
        out[r, c] = red_col
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, start_col]) == yellow_col
    out[top_r - 1, start_col] = red_col
    extend_ray([(top_r - 1, start_col)], UP)

    while queue:
        bi = int(queue.pop(0))
        for port in blobs[bi]["ports"]:
            extend_ray(port["pixels"], port["dir"])

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a red branching cascade where only the correct incoming direction can wake a multi-port blob.

    Input structure:
    - The background is the most common color; every blue diamond/octagon blob has 2-3 green edge bars, each bar marking one outgoing port.
    - A yellow cap above a short red stem at the bottom still marks the initial upward ray.

    Rule:
    - Launch the upward start ray, then let every activated blob emit through all of its green ports.
    - A blob activates only if the incoming ray arrives from its designated entry direction; in the generated long-chain cases this is the unique edge without green markers on a 3-port blob.
    - Rays that hit a blob from any other side do not activate it; they keep drilling straight through and may continue on to the boundary or later objects.

    Key insight:
    Solving now requires combining branching and direction filtering: the green bars tell you every possible exit, but only one entrance side is valid, so the final red network is a selective BFS tree rather than an unconditional chain.
    """
    return _solve(grid, collect_stats=False)[0]


def analyze_grid(grid: np.ndarray) -> dict:
    _out, stats = _solve(grid, collect_stats=True)
    return stats


def _try_place_along_axis(
    spec: dict,
    window: list[tuple[int, int]],
    d: tuple[int, int],
    lane_vals: list[int],
    starts: list[tuple[int, int]],
    include_origin: bool,
    min_gap: int,
    occ_pad: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    H: int,
    W: int,
) -> list[dict]:
    h, w = spec["mask"].shape
    win_vals = [_perp_value(p, d) for p in window]
    edge_line = _along_value(window[0], d)
    start_line = _along_value(starts[0], d)
    is_vertical = bool(d[0])

    fixed = lane_vals[0] - win_vals[0]
    extent = H - h if is_vertical else W - w
    low = edge_line
    high = extent + edge_line
    if d in (UP, LEFT):
        high = min(high, start_line - min_gap)
    else:
        low = max(low, start_line + min_gap)

    entry_lines = list(range(low, high + 1))
    random.shuffle(entry_lines)
    candidates: list[dict] = []

    for entry_line in entry_lines:
        origin_along = entry_line - edge_line
        r0 = origin_along if is_vertical else fixed
        c0 = fixed if is_vertical else origin_along
        if is_vertical:
            entry_cells = [(entry_line, c0 + c) for _, c in window]
        else:
            entry_cells = [(r0 + r, entry_line) for r, _ in window]

        full = _offset(spec["full"], r0, c0)
        if not _in_bounds(full, H, W):
            continue
        if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
            continue

        path = _path_to_entries(starts, d, entry_cells, include_origin, H, W)
        if path is None:
            continue
        entry_set = set(entry_cells)
        if any(blue_mask[r, c] for r, c in path if (r, c) not in entry_set):
            continue

        candidates.append(
            {
                "spec": spec,
                "r0": r0,
                "c0": c0,
                "entry": entry_cells,
                "path": path,
                "blue": _offset(spec["blue"], r0, c0),
                "full": full,
                "ports_by_dir": {
                    dd: _sort_on_ray(_offset(local, r0, c0), dd)
                    for dd, local in spec["ports_by_dir"].items()
                },
            }
        )

    return candidates


def _candidate_activated_blob_placements(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> list[dict]:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    specs = ACTIVE_SPECS.copy()
    random.shuffle(specs)
    candidates: list[dict] = []

    for spec in specs:
        if tuple(spec["required_in"]) != tuple(d):
            continue
        edge = _edge_offsets(spec["mask"], entry_side)
        if len(edge) < thickness:
            continue
        for k in range(len(edge) - thickness + 1):
            window = edge[k : k + thickness]
            win_delta = [_perp_value(p, d) - _perp_value(window[0], d) for p in window]
            if win_delta != lane_delta:
                continue
            candidates.extend(
                _try_place_along_axis(
                    spec,
                    window,
                    d,
                    lane_vals,
                    starts,
                    include_origin,
                    min_gap,
                    occ_pad,
                    blue_mask,
                    ray_mask,
                    H,
                    W,
                )
            )

    random.shuffle(candidates)
    return candidates


def _candidate_pass_blob_placements(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    *,
    preferred_two_port: bool,
) -> list[dict]:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    exit_side = d
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN

    if preferred_two_port:
        specs = BLOCKER2_SPECS.copy()
    else:
        specs = ALL_SPECS.copy()
    random.shuffle(specs)

    candidates: list[dict] = []
    for spec in specs:
        if tuple(spec["required_in"]) == tuple(d):
            continue
        entry_edge = _edge_offsets(spec["mask"], entry_side)
        exit_edge = _edge_offsets(spec["mask"], exit_side)
        if len(entry_edge) < thickness or len(exit_edge) < thickness:
            continue
        max_k = min(len(entry_edge), len(exit_edge)) - thickness
        for k in range(max_k + 1):
            entry_window = entry_edge[k : k + thickness]
            exit_window = exit_edge[k : k + thickness]
            entry_delta = [_perp_value(p, d) - _perp_value(entry_window[0], d) for p in entry_window]
            exit_delta = [_perp_value(p, d) - _perp_value(exit_window[0], d) for p in exit_window]
            if entry_delta != lane_delta or exit_delta != lane_delta:
                continue

            h, w = spec["mask"].shape
            win_vals = [_perp_value(p, d) for p in entry_window]
            edge_line = _along_value(entry_window[0], d)
            start_line = _along_value(starts[0], d)
            is_vertical = bool(d[0])

            fixed = lane_vals[0] - win_vals[0]
            extent = H - h if is_vertical else W - w
            low = edge_line
            high = extent + edge_line
            if d in (UP, LEFT):
                high = min(high, start_line - min_gap)
            else:
                low = max(low, start_line + min_gap)

            entry_lines = list(range(low, high + 1))
            random.shuffle(entry_lines)
            for entry_line in entry_lines:
                origin_along = entry_line - edge_line
                r0 = origin_along if is_vertical else fixed
                c0 = fixed if is_vertical else origin_along
                if is_vertical:
                    entry_cells = [(entry_line, c0 + c) for _, c in entry_window]
                    exit_cells = [(r0 + r, c0 + c) for r, c in exit_window]
                else:
                    entry_cells = [(r0 + r, entry_line) for r, _ in entry_window]
                    exit_cells = [(r0 + r, c0 + c) for r, c in exit_window]
                full = _offset(spec["full"], r0, c0)
                if not _in_bounds(full, H, W):
                    continue
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                path = _path_to_entries(starts, d, exit_cells, include_origin, H, W)
                if path is None:
                    continue
                full_set = set(full)
                if any(blue_mask[r, c] for r, c in path if (r, c) not in full_set):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "entry": entry_cells,
                        "exit": _sort_on_ray(exit_cells, d),
                        "path": path,
                        "blue": _offset(spec["blue"], r0, c0),
                        "full": full,
                        "ports_by_dir": {
                            dd: _sort_on_ray(_offset(local, r0, c0), dd)
                            for dd, local in spec["ports_by_dir"].items()
                        },
                    }
                )

    random.shuffle(candidates)
    return candidates


def _place_terminal_branch(
    H: int,
    W: int,
    state: dict,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    source_full: set[tuple[int, int]],
    *,
    want_blocker: bool,
) -> dict | None:
    if not want_blocker:
        path = _path_to_boundary(starts, d, False, H, W)
        if not _path_is_clear(path, state["occupied"], state["ray_mask"], source_full):
            return None
        trial = _clone_state(state)
        _mark_ray(trial["out"], trial["ray_mask"], path)
        return trial

    candidates = _candidate_pass_blob_placements(
        H,
        W,
        starts,
        d,
        False,
        state["occupied"],
        state["blue_mask"],
        state["ray_mask"],
        preferred_two_port=True,
    )
    for blocker in candidates:
        if not _path_is_clear(blocker["path"], state["occupied"], state["ray_mask"], source_full):
            continue
        trial = _clone_state(state)
        _mark_ray(trial["out"], trial["ray_mask"], blocker["path"])
        _place_blob(trial["inp"], trial["occupied"], trial["blue_mask"], blocker)
        final_path = _path_to_boundary(blocker["exit"], d, False, H, W)
        if not _path_is_clear(final_path, trial["occupied"], trial["ray_mask"], set(blocker["full"])):
            continue
        _mark_ray(trial["out"], trial["ray_mask"], final_path)
        return trial

    return None


def _build_chain(
    H: int,
    W: int,
    state: dict,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    remaining_activations: int,
    blocker_remaining: bool,
    allowed_overlap: set[tuple[int, int]],
) -> dict | None:
    assert remaining_activations >= 1

    candidates = _candidate_activated_blob_placements(
        H,
        W,
        starts,
        d,
        include_origin,
        state["occupied"],
        state["blue_mask"],
        state["ray_mask"],
    )

    for blob in candidates[:80]:
        if not _path_is_clear(blob["path"], state["occupied"], state["ray_mask"], allowed_overlap):
            continue

        base = _clone_state(state)
        _mark_ray(base["out"], base["ray_mask"], blob["path"])
        _place_blob(base["inp"], base["occupied"], base["blue_mask"], blob)
        paint_coords(base["out"], blob["blue"], RED)

        source_full = set(blob["full"])
        port_dirs = list(blob["ports_by_dir"].keys())
        random.shuffle(port_dirs)
        if remaining_activations > 1:
            child_choices = port_dirs
        else:
            child_choices = [None]

        for child_dir in child_choices:
            branch_dirs = [dd for dd in port_dirs if dd != child_dir]
            blocker_choices: list[tuple[int, int] | None]
            if blocker_remaining:
                blocker_choices = branch_dirs.copy()
                if remaining_activations > 1:
                    blocker_choices.append(None)
                random.shuffle(blocker_choices)
            else:
                blocker_choices = [None]

            for blocker_dir in blocker_choices:
                if blocker_remaining and remaining_activations == 1 and blocker_dir is None:
                    continue
                branch_state = _clone_state(base)
                ok = True
                used_blocker = False
                for branch_dir in branch_dirs:
                    want_blocker = blocker_remaining and branch_dir == blocker_dir
                    placed = _place_terminal_branch(
                        H,
                        W,
                        branch_state,
                        blob["ports_by_dir"][branch_dir],
                        branch_dir,
                        source_full,
                        want_blocker=want_blocker,
                    )
                    if placed is None:
                        ok = False
                        break
                    branch_state = placed
                    used_blocker = used_blocker or want_blocker
                if not ok:
                    continue

                next_blocker_remaining = blocker_remaining and not used_blocker
                if child_dir is None:
                    if next_blocker_remaining:
                        continue
                    return branch_state

                child_state = _build_chain(
                    H,
                    W,
                    branch_state,
                    blob["ports_by_dir"][child_dir],
                    child_dir,
                    False,
                    remaining_activations - 1,
                    next_blocker_remaining,
                    source_full,
                )
                if child_state is not None:
                    return child_state

    return None


def _build_vertical_tree(
    H: int,
    W: int,
    state: dict,
    cap_r: int,
    start_col: int,
    activated_count: int,
) -> dict | None:
    """Construct a long upward trunk; every blob also emits left/right side branches."""
    blocker_step = random.randrange(activated_count)
    blocker_side = random.choice([LEFT, RIGHT])

    starts = [(cap_r, start_col)]
    include_origin = True
    allowed_overlap = {(cap_r, start_col)}
    built = _clone_state(state)

    for step in range(activated_count):
        candidates = _candidate_activated_blob_placements(
            H,
            W,
            starts,
            UP,
            include_origin,
            built["occupied"],
            built["blue_mask"],
            built["ray_mask"],
        )
        min_r0_needed = 5 * (activated_count - step - 1)
        candidates = [
            blob
            for blob in candidates
            if blob["spec"]["size"] in ACTIVE_CHAIN_SIZES and blob["r0"] >= min_r0_needed
        ]
        candidates.sort(key=lambda blob: (blob["r0"], -blob["c0"]), reverse=True)

        committed = None
        for blob in candidates[:40]:
            if UP not in blob["ports_by_dir"]:
                continue
            if not _path_is_clear(blob["path"], built["occupied"], built["ray_mask"], allowed_overlap):
                continue

            trial = _clone_state(built)
            _mark_ray(trial["out"], trial["ray_mask"], blob["path"])
            _place_blob(trial["inp"], trial["occupied"], trial["blue_mask"], blob)
            paint_coords(trial["out"], blob["blue"], RED)

            source_full = set(blob["full"])
            branch_dirs = [LEFT, RIGHT]
            if step == activated_count - 1:
                branch_dirs.append(UP)

            ok = True
            for branch_dir in branch_dirs:
                want_blocker = (
                    step == blocker_step
                    and branch_dir == blocker_side
                    and branch_dir in (LEFT, RIGHT)
                )
                placed = _place_terminal_branch(
                    H,
                    W,
                    trial,
                    blob["ports_by_dir"][branch_dir],
                    branch_dir,
                    source_full,
                    want_blocker=want_blocker,
                )
                if placed is None:
                    ok = False
                    break
                trial = placed
            if not ok:
                continue

            committed = trial
            starts = blob["ports_by_dir"][UP]
            include_origin = False
            allowed_overlap = source_full
            break

        if committed is None:
            return None
        built = committed

    return built


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    activated_count = random.choice(ACTIVATED_COUNT_CHOICES)
    H_low = max(H_MIN, 7 + 5 * activated_count)
    H = random.randint(H_low, H_MAX)
    W = random.randint(W_MIN, W_MAX)

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    mid_lo = max(START_COL_PAD, W // 3)
    mid_hi = min(W - START_COL_PAD - 1, (2 * W) // 3)
    start_col = random.randint(mid_lo, mid_hi)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    occupied[cap_r, start_col] = True
    ray_mask[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    init_state = {
        "inp": inp,
        "out": out,
        "occupied": occupied,
        "blue_mask": blue_mask,
        "ray_mask": ray_mask,
    }
    built = _build_vertical_tree(H, W, init_state, cap_r, start_col, activated_count)
    if built is None:
        return None

    solved, stats = _solve(built["inp"], collect_stats=True)
    if not np.array_equal(solved, built["out"]):
        return None
    if stats["activations"] != activated_count:
        return None
    if stats["wrong_dir_hits"] != 1:
        return None
    if stats["two_port_blobs"] != 1:
        return None
    if stats["wrong_hit_blob_port_counts"] != [2]:
        return None
    if stats["total_blobs"] != activated_count + 1:
        return None
    if stats["three_port_blobs"] < activated_count:
        return None
    try:
        if np.array_equal(_original_main()(built["inp"]), built["out"]):
            return None
    except Exception:
        pass
    return built["inp"], built["out"], stats


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        made = _gen_attempt()
        if made is not None:
            return made[0]
    raise RuntimeError("failed to generate a valid 5961cc34 combo input")
