# Combo variant of 5961cc34 (index 22)
# combo_spec: D2/3+D20/2
# Name: 条件激活+起始帽颜色编码激活模式

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import colors_present, components_mask, get_bg, paint_coords, paste


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) 具体化与行为追踪
#    - 保留原题的主体几何: 底部 1 格宽红杆 + 顶帽、蓝色菱形/八边形色块、单侧绿色端口、输出中的红色射线。
#    - 融合后的关键变化是“帽颜色先选激活谓词，再用该谓词解释所有命中”:
#      * 黄帽: 任意方向命中蓝块都会激活, 相当于原题 BFS.
#      * 橙帽: 只有射线朝着绿色端口那一侧前进时才激活, 也就是“从端口对面进入”。
#        例如端口在右边, 只有自左向右的射线能激活它; 激活后射线继续向右穿出。
#      * 粉帽: 只有水平来袭(左->右或右->左)才能激活; 上下来的射线只能穿透, 不会点亮色块。
#    - 一个具体小例子:
#      * 原题/黄帽: 起始竖直红线自下向上碰到一个右端口 blob, 立即整块变红, 再向右发射粗红线。
#      * 橙帽: 同样的右端口 blob 若是被竖直上行射线撞到, 不会激活; 红线会在 blob 内直穿并继续上行。
#        只有当某段水平右行红线撞到它时, 它才会整块变红并继续向右。
#      * 粉帽: 所有起始竖直命中都不激活, 所以若没有先前的水平段, 整条网络就只会留下直穿红柱。
#    - 这种组合不是“颜色变了但逻辑没变”, 而是把 D2 的方向条件本身交给 D20 的帽颜色来参数化。
#
# b) 复杂度校准
#    - 只增加 2 种帽颜色: 黄=4, 粉=6, 橙=7. 其余角色色保持原题固定, 不把规则藏到过多颜色里。
#    - 激活模式只有 3 档, 都是肉眼可表述的方向条件:
#      * any
#      * toward-port
#      * horizontal-only
#    - generator 不让样例退化成“换了帽色但输出恰好一样”:
#      * 黄帽样例强制首个激活来自竖直命中且端口朝左右, 从而明显区别于橙帽/粉帽。
#      * 橙帽样例强制至少出现 1 个竖直穿透 blocker + 1 个竖直激活 blob, 暴露“同为竖直命中,
#        但是否激活取决于端口是否在前方”。
#      * 粉帽样例强制起始竖线真的穿过至少 1 个 blob 且零激活, 使“只认左右来袭”可见。
#
# c) 隐式假设审查（main 相关）
#    - 原版 `if hit blue: activate; break` 假设“命中即激活”; 融合后必须改成“先查帽色模式再决定”。
#    - 原版 start ray 首次碰到 blue 就停止; 粉帽/橙帽下这会误停在 blocker 上, 所以必须允许继续钻穿。
#    - 原版 thick ray 对任意 blue 也立即停止; 新版只有激活时才停止, 否则当前 lane 要持续前进。
#    - 原版把顶帽颜色只当起点装饰; 新版 solver 必须把它读成全局模式开关。
#    - 原版最后默认未激活 blob 至多 1 个 distractor; 现在未激活 blob 还可能是路径上的 blocker,
#      这个假设必须移除。
#    - 新增假设: 一个 blob 仍然只有 1 个绿色端口边。generator 继续保证这一点, solver 才能唯一恢复 port_dir.
#
# d) 放置合法性
#    - 仍使用 output-first / path-first: 先在输出里构造真实红射线路径, 再反推输入里的蓝块与绿端口。
#    - 激活 blob 的放置要求:
#      * 入边窗口宽度必须匹配当前 ray thickness
#      * 完整落在网格内
#      * 与既有 blob 保持至少 1 像素 8 邻域间隔
#      * 当前段路径到该 blob 入口之间不能穿过其他 blue
#    - blocker blob 额外要求:
#      * 当前 ray 必须能从 entry edge 穿到 opposite edge, 厚度不变
#      * 该 blob 的端口方向在当前帽模式下不能满足激活条件
#    - 这些条件由 `_candidate_activate_blob()` / `_candidate_pass_blob()` 显式检查。
#
# e) 视觉可观察性
#    - 解题者需要从少量样例看出“帽颜色 -> 激活条件”而不是记住单个对象。
#    - 因此每种模式都让差异直接发生在真实红线路径上:
#      * 黄帽: 竖直命中就能转弯点亮
#      * 橙帽: 竖直命中有的只穿过, 有的整块变红, 差别来自端口是否位于前进方向
#      * 粉帽: 同样的竖直命中全部只穿不亮
#    - 颜色语义有重复支撑: 帽色总是单独放在起点, 端口色总是绿色, 激活结果总是红色。
#
# f) 可视化整洁度
#    - 总 blob 数仍限制在 4 以内, 避免让“直穿”和“激活”混杂成难读的红块。
#    - 粉帽样例刻意不生成横向激活链, 让它保持“竖线穿透若干 blob”这一简洁外观。
#    - 橙帽样例只做直行链, 不引入多次拐弯, 这样“端口在前方才点亮”的模式更干净。
#
# g) 变化的可推断性
#    - 这次颜色变化只发生在起始帽, 而且每种帽色都绑定一种固定的全局激活法则, 不在样例间置换。
#    - 人类不需要读颜色数值大小, 只需要观察:
#      * 黄帽时, 竖直撞到侧端口也会点亮
#      * 橙帽时, 只有“朝端口方向前进”的命中会点亮
#      * 粉帽时, 只有左右来袭才会点亮; 起始竖线本身不会点亮任何 blob
#
# 9) 对新 main() 假设的逐项复查
#    - 起点检测仍假设“底部同色细竖杆 + 上方异色帽”唯一存在; generator 只放一个这样的结构。
#    - 蓝色仍通过“除背景/红杆/帽色之外的最大像素数颜色”恢复; generator 保持 blob body 是唯一大块前景色。
#    - 绿色仍通过“与 blue 4 邻接的唯一其他颜色”恢复; generator 不引入其他贴着 blue 的颜色。
#    - `_activates(mode, incoming, port_dir)` 假设 mode 是全局统一规则; generator 每张图只放一种帽色。
#    - `_candidate_pass_blob()` 假设路径上的 blocker 在当前模式下确实不会激活; generator 用同一谓词筛掉
#      会被误激活的 spec, 并用反事实差异检查再次验证。
# ---------------------------------------------------------------------------


# --- Colors ---
BG = 8
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
ORANGE = 7

# --- Cap-coded activation modes ---
MODE_ANY = "any"
MODE_TOWARD_PORT = "toward_port"
MODE_HORIZONTAL = "horizontal_only"
MODE_TO_CAP = {
    MODE_ANY: YELLOW,
    MODE_HORIZONTAL: PINK,
    MODE_TOWARD_PORT: ORANGE,
}
CAP_TO_MODE = {v: k for k, v in MODE_TO_CAP.items()}
CAP_COLORS = set(CAP_TO_MODE)
MODES = tuple(MODE_TO_CAP)

# --- Directions ---
UP = (-1, 0)
DOWN = (1, 0)
LEFT = (0, -1)
RIGHT = (0, 1)
DIRS = (UP, DOWN, LEFT, RIGHT)
OPP = {UP: DOWN, DOWN: UP, LEFT: RIGHT, RIGHT: LEFT}

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

# --- Object budgets ---
MAX_BLOB_COUNT = 4
MAX_DISTRACTORS = 1

# --- Placement ---
MIN_GAP_FIRST = 4
MIN_GAP_CHAIN = 3

# --- Retry ---
GEN_RETRIES = 500
_GENERATE_CALL_COUNT = 0


def _mode_from_cap(cap_color: int) -> str:
    assert cap_color in CAP_TO_MODE
    return CAP_TO_MODE[int(cap_color)]


def _activates(mode: str, incoming_dir: tuple[int, int], port_dir: tuple[int, int]) -> bool:
    if mode == MODE_ANY:
        return True
    if mode == MODE_TOWARD_PORT:
        return tuple(incoming_dir) == tuple(port_dir)
    if mode == MODE_HORIZONTAL:
        return tuple(incoming_dir) in (LEFT, RIGHT)
    raise AssertionError(f"unknown mode {mode}")


def _solve(
    grid: np.ndarray,
    *,
    collect_stats: bool = False,
    mode_override: str | None = None,
) -> tuple[np.ndarray, dict]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    stats = {
        "mode": None,
        "cap_color": None,
        "activations": 0,
        "wrong_dir_hits": 0,
        "pass_through_cells": 0,
        "blob_count": 0,
    }

    # --- STEP 1) Detect the bottom start marker and decode the cap-controlled mode. ---
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
    red_col, cap_col, start_col, start_seg = start_candidates[0]
    mode = _mode_from_cap(cap_col) if mode_override is None else mode_override
    stats["mode"] = mode
    stats["cap_color"] = int(cap_col)

    # --- STEP 2) Identify the blob body color and the port color. ---
    counts = Counter(map(int, g.flatten()))
    other_cols = [
        (int(cnt), int(col))
        for col, cnt in counts.items()
        if int(col) not in (bg, red_col, cap_col)
    ]
    assert other_cols
    max_count = max(cnt for cnt, _col in other_cols)
    blue_candidates = [int(col) for cnt, col in other_cols if cnt == max_count]
    assert len(blue_candidates) == 1
    blue_col = int(blue_candidates[0])

    green_candidates: list[int] = []
    for col in colors_present(g, exclude={bg, red_col, cap_col, blue_col}):
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
    stats["blob_count"] = len(blobs)

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

    activated: set[int] = set()
    queue: list[int] = []
    counted_wrong_hits: set[tuple[int, tuple[int, int]]] = set()

    def activate_blob_at(r: int, c: int) -> None:
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
                if _activates(mode, d, blobs[bi]["port_dir"]):
                    activate_blob_at(int(r), int(c))
                    break
                key = (int(bi), tuple(d))
                if key not in counted_wrong_hits:
                    counted_wrong_hits.add(key)
                    stats["wrong_dir_hits"] += 1
                stats["pass_through_cells"] += 1

    # --- STEP 4) Keep the start marker red and launch the cap-coded initial upward ray. ---
    for r, c in start_seg:
        out[int(r), int(c)] = int(red_col)
    top_r = min(int(r) for r, _c in start_seg)
    assert top_r > 0 and int(g[top_r - 1, int(start_col)]) == int(cap_col)
    out[int(top_r) - 1, int(start_col)] = int(red_col)
    extend_ray([(int(top_r) - 1, int(start_col))], UP)

    # --- STEP 5) BFS only over blobs that satisfied the current cap-selected activation rule. ---
    while queue:
        bi = int(queue.pop(0))
        extend_ray(blobs[bi]["port_pixels"], blobs[bi]["port_dir"])

    return out, stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Build a red ray network whose activation law is encoded by the start cap color.

    Input structure:
    - Blue diamond/octagon blobs each have one green edge port, and a short red stem rises from the grid bottom.
    - The single cap pixel above that stem can be yellow, orange, or pink; that color selects the activation mode.

    Rule:
    - The start ray goes upward and every activated blob emits a thick ray through its green port, as in the original cascade task.
    - Yellow cap: any incoming direction activates a hit blob.
    - Orange cap: a blob activates only when the ray is moving toward its green side, i.e. entering from the opposite side.
    - Pink cap: a blob activates only when hit by a horizontal ray; vertical hits simply pass straight through.

    Key insight:
    The green side still marks where an activated blob will emit, but the top cap color now tells you which contacts count as activations in the first place.
    """
    return _solve(grid, collect_stats=False)[0]


def analyze_grid(grid: np.ndarray) -> dict:
    out, stats = _solve(grid, collect_stats=True)
    gaps: dict[str, int] = {}
    for mode in MODES:
        alt_out, _ = _solve(grid, collect_stats=False, mode_override=mode)
        gaps[mode] = int(np.count_nonzero(out != alt_out))
    stats["counterfactual_gaps"] = gaps
    stats["min_other_gap"] = min(g for mode, g in gaps.items() if mode != stats["mode"])
    return stats


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
    }


def _all_specs(port_dirs: tuple[tuple[int, int], ...] | list[tuple[int, int]] | None = None) -> list[dict]:
    dirs = list(port_dirs) if port_dirs is not None else list(DIRS)
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


def _candidate_activate_blob(
    H: int,
    W: int,
    starts: list[tuple[int, int]],
    d: tuple[int, int],
    include_origin: bool,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    allowed_port_dirs: list[tuple[int, int]] | tuple[tuple[int, int], ...],
) -> dict | None:
    starts = _sort_on_ray(starts, d)
    thickness = len(starts)
    lane_vals = [_perp_value(p, d) for p in starts]
    lane_delta = [v - lane_vals[0] for v in lane_vals]
    entry_side = OPP[d]
    occ_pad = _dilate8(occupied)
    min_gap = MIN_GAP_FIRST if include_origin else MIN_GAP_CHAIN
    candidates: list[dict] = []

    for spec in _all_specs(tuple(allowed_port_dirs)):
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
    mode: str,
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
        if _activates(mode, d, spec["port_dir"]):
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


def _add_activator(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    blob: dict,
) -> tuple[list[tuple[int, int]], tuple[int, int]]:
    _mark_ray(out, ray_mask, blob["path"])
    _place_blob(inp, occupied, blue_mask, blob)
    paint_coords(out, blob["blue"], RED)
    return blob["ports"], blob["spec"]["port_dir"]


def _add_blocker(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    blue_mask: np.ndarray,
    ray_mask: np.ndarray,
    blob: dict,
) -> list[tuple[int, int]]:
    _mark_ray(out, ray_mask, blob["path"])
    _place_blob(inp, occupied, blue_mask, blob)
    return blob["exit"]


def _gen_attempt(mode: str | None = None) -> np.ndarray | None:
    H = random.randint(H_MIN, H_MAX)
    W = random.randint(W_MIN, W_MAX)
    mode = random.choice(MODES) if mode is None else mode
    cap_color = MODE_TO_CAP[mode]

    inp = np.full((H, W), BG, dtype=int)
    out = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    blue_mask = np.zeros((H, W), dtype=bool)
    ray_mask = np.zeros((H, W), dtype=bool)

    # Place start segment (cap + red column at the bottom).
    start_h = random.randint(START_H_MIN, START_H_MAX)
    cap_r = H - start_h - 1
    start_col = random.randint(START_COL_PAD, W - START_COL_PAD - 1)
    inp[cap_r, start_col] = cap_color
    out[cap_r, start_col] = RED
    ray_mask[cap_r, start_col] = True
    occupied[cap_r, start_col] = True
    for r in range(cap_r + 1, H):
        inp[r, start_col] = RED
        out[r, start_col] = RED
        occupied[r, start_col] = True

    starts = [(cap_r, start_col)]
    direction = UP
    include_origin = True
    used_blobs = 0

    if mode == MODE_ANY:
        chain_len = random.choice((1, 2, 3))
        first = _candidate_activate_blob(
            H,
            W,
            starts,
            direction,
            include_origin,
            occupied,
            blue_mask,
            ray_mask,
            [LEFT, RIGHT],
        )
        if first is None:
            return None
        starts, direction = _add_activator(inp, out, occupied, blue_mask, ray_mask, first)
        include_origin = False
        used_blobs += 1

        for _ in range(chain_len - 1):
            allowed = [d for d in DIRS if d != OPP[direction]]
            blob = _candidate_activate_blob(
                H,
                W,
                starts,
                direction,
                include_origin,
                occupied,
                blue_mask,
                ray_mask,
                allowed,
            )
            if blob is None:
                return None
            starts, direction = _add_activator(inp, out, occupied, blue_mask, ray_mask, blob)
            used_blobs += 1

    elif mode == MODE_TOWARD_PORT:
        activation_count = random.randint(1, 2)
        blocker_budget = random.randint(1, MAX_BLOB_COUNT - activation_count)
        blockers_left = blocker_budget
        for seg in range(activation_count):
            if blockers_left > 0 and (seg == 0 or random.random() < 0.7):
                blocker = _candidate_pass_blob(
                    H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask, mode
                )
                if blocker is None:
                    return None
                starts = _add_blocker(inp, out, occupied, blue_mask, ray_mask, blocker)
                include_origin = False
                used_blobs += 1
                blockers_left -= 1

            blob = _candidate_activate_blob(
                H,
                W,
                starts,
                direction,
                include_origin,
                occupied,
                blue_mask,
                ray_mask,
                [direction],
            )
            if blob is None:
                return None
            starts, direction = _add_activator(inp, out, occupied, blue_mask, ray_mask, blob)
            include_origin = False
            used_blobs += 1

        while blockers_left > 0:
            blocker = _candidate_pass_blob(
                H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask, mode
            )
            if blocker is None:
                return None
            starts = _add_blocker(inp, out, occupied, blue_mask, ray_mask, blocker)
            used_blobs += 1
            blockers_left -= 1

    else:
        blocker_count = random.randint(1, min(3, MAX_BLOB_COUNT))
        for _ in range(blocker_count):
            blocker = _candidate_pass_blob(
                H, W, starts, direction, include_origin, occupied, blue_mask, ray_mask, mode
            )
            if blocker is None:
                return None
            starts = _add_blocker(inp, out, occupied, blue_mask, ray_mask, blocker)
            include_origin = False
            used_blobs += 1

    final_path = _path_to_boundary(starts, direction, include_origin, H, W)
    if any(blue_mask[r, c] for r, c in final_path):
        return None
    _mark_ray(out, ray_mask, final_path)

    remaining = MAX_BLOB_COUNT - used_blobs
    num_distractors = random.randint(0, min(MAX_DISTRACTORS, remaining))
    for _ in range(num_distractors):
        if not _place_distractor(inp, occupied, blue_mask, ray_mask):
            return None

    solved, stats = _solve(inp, collect_stats=True)
    if not np.array_equal(solved, out):
        return None

    info = analyze_grid(inp)
    if info["mode"] != mode or info["cap_color"] != cap_color:
        return None
    if info["min_other_gap"] <= 0:
        return None

    if mode == MODE_ANY:
        if info["activations"] < 1 or info["wrong_dir_hits"] != 0:
            return None
        if info["counterfactual_gaps"][MODE_TOWARD_PORT] <= 0:
            return None
        if info["counterfactual_gaps"][MODE_HORIZONTAL] <= 0:
            return None
    elif mode == MODE_TOWARD_PORT:
        if info["activations"] < 1 or info["wrong_dir_hits"] < 1:
            return None
    else:
        if info["activations"] != 0 or info["wrong_dir_hits"] < 1:
            return None

    return inp


def generate_input() -> np.ndarray:
    global _GENERATE_CALL_COUNT

    # Cycle the preferred cap mode so a 2-4 example training set exposes the
    # three activation laws instead of relying on lucky random coverage.
    preferred_index = _GENERATE_CALL_COUNT % len(MODES)
    preferred_mode = MODES[preferred_index]
    _GENERATE_CALL_COUNT += 1

    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(preferred_mode)
        if grid is not None:
            return grid
    for offset in range(1, len(MODES)):
        mode = MODES[(preferred_index + offset) % len(MODES)]
        for _ in range(max(50, GEN_RETRIES // 4)):
            grid = _gen_attempt(mode)
            if grid is not None:
                return grid
    raise RuntimeError("failed to generate a valid 5961cc34 combo D2/3+D20/2 input")
