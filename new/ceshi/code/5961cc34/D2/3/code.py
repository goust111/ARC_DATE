# Variant of 5961cc34
# Dimension: D2 — 级联激活机制 (path)
# Change: BFS 级联：命中色块后立即激活并发射新射线，所有被间接命中的色块也被激活，直到无新命中... → 条件激活

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# ---------------------------------------------------------------------------
# Design analysis for the D2 variant
#
# a) 具体化与行为追踪
#    - 保留原题的主要视觉元素: 蓝色菱形/八边形色块、单侧绿色端口、底部黄帽+红杆起点。
#    - 核心改动放在“命中后的后续行为”:
#      * 原题: 任何方向打到蓝块, 该蓝块都会整块激活为红色, 然后沿绿色端口继续发射。
#      * 变体: 只有当射线按特定方向入射时才激活。这里采用固定的可见关系:
#        `射线必须从绿色端口的逆时针相邻侧进入`, 激活后向绿色端口方向发射。
#        等价地说: 绿色端口是“顺时针转向后的出口”。
#      * 例子:
#        - 若绿色端口在右侧, 则只有“自下向上”的射线能激活它, 激活后向右转。
#        - 若自左向右的射线撞上这个同样的色块, 射线会直穿过去, 色块不激活。
#    - 输出上的可见差异:
#      * 被正确方向命中的色块: 整块变红, 并在端口方向生出新射线。
#      * 被错误方向命中的色块: 不整块变红, 只在被射线穿过的那一条/两条轨道上出现红色直线。
#
# b) 复杂度校准
#    - 不增加新颜色, 仍只使用原题的 5 种颜色, 避免把规律藏到颜色 id 或额外图例中。
#    - 不允许“任意方向都可能是正确入口”; 只保留一个固定转向关系, 让人能从少量样例中归纳。
#    - 为了让条件激活真正可观察, 生成器强制每个样例至少出现 1 个“错误方向穿透”的 blocker。
#
# c) 对原 solver 旧假设的审查
#    - 旧版 `if hit blue: activate; break` 依赖“命中即激活”。新规则下这会误把 blocker 激活。
#    - 旧版 start ray 在首次碰到蓝块后立即停止; 新规则下若方向不对, 必须继续向前追踪。
#    - 旧版 thick ray 每条 lane 命中蓝块后立即停止; 新规则下 lane 可能要穿过整块再继续。
#    - 旧版最后 `len(blobs) - len(activated) <= 1` 假设未激活块至多是 distractor。新变体里
#      未激活块还可能是被射线穿透过的 blocker, 所以这个断言必须去掉。
#
# d) 放置合法性
#    - 生成器仍使用 path-first 思路, 但把“下一步遇到的对象”分成两类:
#      * turn blob: 当前方向满足其激活入口条件, 会激活并顺时针转向。
#      * pass blob: 当前方向不满足入口条件, 射线直穿而过并保持方向。
#    - 所有 blob 仍需:
#      * 完整落在网格内
#      * 与已有 blob 保持至少 1 像素的 8 邻域间隔
#      * 不与已画好的旧射线重叠, 除了当前正在构造的那一条目标射线本身
#    - pass blob 还必须保证: 当前射线以同样厚度从其 opposite edge 穿出, 之后仍可继续放置对象。
#
# e) 视觉可观察性
#    - 只靠“是否激活”来区分不够, 所以 blocker 专门放在实际红射线路径上。人眼能直接看到:
#      同样被红线打到, 有的蓝块整块变红并转向, 有的只出现一道直穿红线。
#    - 激活块与 blocker 会混合出现, 且端口朝向覆盖多个方向, 使“方向条件”而不是“位置巧合”
#      成为唯一可解释规律。
#
# f) 可视化整洁度
#    - 仍限制总 blob 数不大于 4, 避免满屏交叉。
#    - 顺时针转向链和穿透 blocker 都沿单一路径构造, 减少杂乱的多重交叉。
#    - distractor 最多 1 个, 且永不放在任何射线上。
#
# g) 颜色变化可推断性
#    - 没有新增颜色映射。绿色仍然只承担端口提示作用, 红色仍然表示“被保留下来的网络”。
#    - 唯一新信息来自空间关系: 射线从哪一侧进入, 以及是否因此转向。
#
# 新 main() 的隐式假设与 generator 的配合:
#    - 每个蓝块只带 1 个绿色端口边; generator 始终如此生成。
#    - 需要能从蓝块邻接的绿色像素唯一恢复端口方向; generator 保证绿色只贴在一个边上。
#    - 对于刻意放在射线路径上的 blocker, 当前 ray thickness 必须与其入边/出边窗口对齐;
#      generator 用边窗口匹配来保证这一点。
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
ROT_CW = {UP: RIGHT, RIGHT: DOWN, DOWN: LEFT, LEFT: UP}
ROT_CCW = {UP: LEFT, LEFT: DOWN, DOWN: RIGHT, RIGHT: UP}


def _required_incoming(port_dir: tuple[int, int]) -> tuple[int, int]:
    """A blob activates only if the incoming ray turns clockwise into its green side."""
    return ROT_CCW[tuple(port_dir)]


def _solve(grid: np.ndarray, *, collect_stats: bool = False) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    stats = {
        "activations": 0,
        "wrong_dir_hits": 0,
        "pass_through_cells": 0,
    }

    # --- STEP 1) Detect the bottom start marker (red stem with a differently-colored cap). ---
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
                (
                    int(col),
                    int(cap),
                    int(c0),
                    [(int(rr), int(c0)) for rr in range(r0, r1 + 1)],
                )
            )

    assert len(start_candidates) == 1
    red_col, yellow_col, start_col, start_seg = start_candidates[0]

    # --- STEP 2) Identify the blob body color and the port color. ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, yellow_col)
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, yellow_col, blue_col}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        ok = False
        for r, c in pts:
            r = int(r)
            c = int(c)
            for dr, dc in DIRS:
                rr = r + dr
                cc = c + dc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == blue_col:
                    ok = True
                    break
            if ok:
                break
        if ok:
            green_candidates.append(int(col))

    assert len(green_candidates) == 1
    green_col = int(green_candidates[0])

    out = np.full_like(g, bg)

    # --- STEP 3) Extract blue components and recover each blob's unique green-side direction. ---
    blob_id = np.full((H, W), -1, dtype=int)
    blobs: list[dict] = []
    for coords in components_mask(g == blue_col, connectivity=8):
        pts = [(int(r), int(c)) for r, c in coords]
        bi = len(blobs)
        for r, c in pts:
            blob_id[r, c] = int(bi)
        blobs.append({"pts": pts, "ports": []})

    for blob in blobs:
        seen_dirs: set[tuple[int, int]] = set()
        for r, c in blob["pts"]:
            for ddr, ddc in DIRS:
                rr = r + ddr
                cc = c + ddc
                if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == green_col:
                    seen_dirs.add((int(ddr), int(ddc)))
        for ddr, ddc in seen_dirs:
            port_pixels = [
                (int(r), int(c))
                for (r, c) in blob["pts"]
                if 0 <= r + ddr < H
                and 0 <= c + ddc < W
                and int(g[r + ddr, c + ddc]) == green_col
            ]
            assert port_pixels
            blob["ports"].append({"dir": (int(ddr), int(ddc)), "pixels": port_pixels})
        assert len(blob["ports"]) == 1
        blob["port_dir"] = blob["ports"][0]["dir"]
        blob["port_pixels"] = blob["ports"][0]["pixels"]
        blob["required_in"] = _required_incoming(blob["port_dir"])

    activated: set[int] = set()
    queue: list[int] = []
    counted_wrong_hits: set[tuple[int, tuple[int, int]]] = set()

    def activate_blob_at(r: int, c: int) -> None:
        """Activate the whole blob, recolor its blue body to red, then enqueue its emission."""
        bi = int(blob_id[int(r), int(c)])
        assert bi >= 0
        if bi in activated:
            return
        activated.add(int(bi))
        queue.append(int(bi))
        stats["activations"] += 1
        for pr, pc in blobs[bi]["pts"]:
            out[int(pr), int(pc)] = int(red_col)

    def extend_ray(origins: list[tuple[int, int]], d: tuple[int, int]) -> None:
        """Trace a ray lane-by-lane; wrong-way hits drill through, correct-way hits activate."""
        ddr, ddc = map(int, d)
        for pr, pc in origins:
            r, c = int(pr), int(pc)
            while True:
                r += ddr
                c += ddc
                if not (0 <= r < H and 0 <= c < W):
                    break
                out[int(r), int(c)] = int(red_col)
                if int(g[int(r), int(c)]) != int(blue_col):
                    continue
                bi = int(blob_id[int(r), int(c)])
                assert bi >= 0
                if bi in activated:
                    break
                if tuple(d) == tuple(blobs[bi]["required_in"]):
                    activate_blob_at(int(r), int(c))
                    break
                key = (int(bi), tuple(d))
                if key not in counted_wrong_hits:
                    counted_wrong_hits.add(key)
                    stats["wrong_dir_hits"] += 1
                stats["pass_through_cells"] += 1

    # --- STEP 4) Keep the start marker red and launch the initial upward ray. ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(yellow_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    extend_ray([(int(top_r) - 1, int(start_col))], UP)

    # --- STEP 5) BFS over activated blobs: each one emits only after a valid directional hit. ---
    while queue:
        bi = int(queue.pop(0))
        extend_ray(blobs[bi]["port_pixels"], blobs[bi]["port_dir"])

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network where blue blobs react only to correctly-oriented hits.

    Input structure:
    - The background is the most frequent color; blue diamond/octagon blobs each carry one green edge port.
    - A yellow cap sits above a short red stem at the bottom, defining the initial upward ray.

    Rule:
    - Red rays travel straight until they leave the grid or meet a blue blob.
    - A blob activates only when the incoming ray arrives from the side that makes a clockwise turn into its green port; then the blue body turns red and a new ray is emitted through that port.
    - If the ray hits the blob from any other side, it simply drills straight through the blob without activating it.

    Key insight:
    The green edge no longer means "any hit will fire here"; it marks an exit that works only for one entry direction, so solving the task requires tracking ray direction at each contact.
    """
    return _solve(grid, collect_stats=False)[0]


def analyze_grid(grid: np.ndarray) -> dict:
    """Expose variant-specific interaction counts for testing and anti-degeneration checks."""
    _out, stats = _solve(grid, collect_stats=True)
    return stats


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
H_MIN, H_MAX = 13, 27
W_MIN, W_MAX = 15, 27
START_COL_PAD = 2

# --- Start segment ---
START_H_MIN, START_H_MAX = 2, 3

# --- Chain / blocker / distractor ---
CHAIN_LEN_WEIGHTS = [1, 3, 3, 3]
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 400


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


def _blob_spec(size: tuple[int, int], port_dir: tuple[int, int]) -> dict | None:
    h, w = size
    mask = _shape_mask(h, w)
    green = _edge_offsets(mask, port_dir)
    if len(green) not in (1, 2):
        return None

    green_set = set(green)
    patch = np.full((h, w), BG, dtype=int)
    patch[mask] = BLUE
    for r, c in green:
        patch[r, c] = GREEN

    port_pixels: list[tuple[int, int]] = []
    for gr, gc in green:
        br, bc = gr - port_dir[0], gc - port_dir[1]
        if not (0 <= br < h and 0 <= bc < w):
            return None
        if not mask[br, bc] or (br, bc) in green_set:
            return None
        port_pixels.append((br, bc))

    blue = [(int(r), int(c)) for r, c in np.argwhere(mask) if (int(r), int(c)) not in green_set]
    full = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    return {
        "size": (h, w),
        "mask": mask,
        "patch": patch,
        "blue": blue,
        "full": full,
        "port_dir": port_dir,
        "port_pixels": _sort_on_ray(port_pixels, port_dir),
        "required_in": _required_incoming(port_dir),
    }


def _all_specs(port_dir: tuple[int, int] | None = None) -> list[dict]:
    dirs = [port_dir] if port_dir is not None else list(DIRS)
    out = [s for size in SHAPES for d in dirs if (s := _blob_spec(size, d)) is not None]
    random.shuffle(out)
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    H, W = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            r0, r1 = max(0, dr), H + min(0, dr)
            c0, c1 = max(0, dc), W + min(0, dc)
            out[r0:r1, c0:c1] |= mask[r0 - dr : r1 - dr, c0 - dc : c1 - dc]
    return out


def _offset(offsets: list[tuple[int, int]], r0: int, c0: int) -> list[tuple[int, int]]:
    return [(r0 + r, c0 + c) for r, c in offsets]


def _in_bounds(coords: list[tuple[int, int]], H: int, W: int) -> bool:
    return all(0 <= r < H and 0 <= c < W for r, c in coords)


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
            r, c = r + dr, c + dc
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
            r, c = r + dr, c + dc
            if not (0 <= r < H and 0 <= c < W):
                break
            cells.append((r, c))
    return cells


def _mark_ray(out: np.ndarray, ray_mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    paint_coords(out, cells, RED)
    paint_coords(ray_mask, cells, True)


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
                "ports": _sort_on_ray(_offset(spec["port_pixels"], r0, c0), spec["port_dir"]),
            }
        )

    return candidates


def _candidate_turn_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    wanted_port = ROT_CW[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(port_dir=wanted_port):
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

    return random.choice(candidates) if candidates else None


def _candidate_pass_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    exit_side = d
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs():
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
                    }
                )

    return random.choice(candidates) if candidates else None


def _place_blob(inp: np.ndarray, occupied: np.ndarray, blue_mask: np.ndarray, blob: dict) -> None:
    paste(inp, blob["spec"]["patch"], blob["r0"], blob["c0"], transparent=BG)
    for r, c in blob["full"]:
        occupied[r, c] = True
    for r, c in blob["blue"]:
        blue_mask[r, c] = True


def _place_distractor(
    inp: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
) -> bool:
    H, W = inp.shape
    occ_pad = _dilate8(occupied)
    candidates: list[dict] = []
    for spec in _all_specs():
        h, w = spec["mask"].shape
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                full = _offset(spec["full"], r0, c0)
                if any(occ_pad[r, c] or ray_mask[r, c] for r, c in full):
                    continue
                candidates.append(
                    {
                        "spec": spec,
                        "r0": r0,
                        "c0": c0,
                        "full": full,
                        "blue": _offset(spec["blue"], r0, c0),
                    }
                )
    if not candidates:
        return False
    _place_blob(inp, occupied, blue_mask, random.choice(candidates))
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (yellow cap + red column at the bottom).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = YELLOW
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    chain_len = random.choices(range(len(CHAIN_LEN_WEIGHTS)), weights=CHAIN_LEN_WEIGHTS)[0]
    segment_count = chain_len + 1
    max_blockers = min(segment_count, MAX_BLOB_COUNT - chain_len)
    if max_blockers <= 0:
        return None
    blocker_count = random.randint(1, max_blockers)
    blocker_slots = set(random.sample(range(segment_count), blocker_count))
    remaining = MAX_BLOB_COUNT - chain_len - blocker_count
    num_distractors = random.randint(0, min(MAX_DISTRACTORS, remaining))

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True

    for seg in range(segment_count):
        if seg in blocker_slots:
            blocker = _candidate_pass_blob(
                H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask
            )
            if blocker is None:
                return None
            _mark_ray(out, ray_mask, blocker["path"])
            _place_blob(inp, occupied, blue_mask, blocker)
            starts = blocker["exit"]
            include_origin = False

        if seg < chain_len:
            blob = _candidate_turn_blob(
                H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask
            )
            if blob is None:
                return None
            _mark_ray(out, ray_mask, blob["path"])
            _place_blob(inp, occupied, blue_mask, blob)
            paint_coords(out, blob["blue"], RED)
            starts = blob["ports"]
            direction = blob["spec"]["port_dir"]
            include_origin = False

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    solved, stats = _solve(inp, collect_stats=True)
    if not np.array_equal(solved, out):
        return None
    if stats["activations"] != chain_len:
        return None
    if stats["wrong_dir_hits"] != blocker_count:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 5961cc34 D2 conditional-activation input")
