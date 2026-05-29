# Combo variant of 3dc255db (index 32)
# combo_spec: D8/0+D15/0
# Name: 距离排序处理+共享边框色
#
# Design analysis
# ===============
# a) 具体化与行为追踪
# - 原题按“每个边框颜色对应一个杯”处理，多个杯互不影响；内容从杯口抹掉后，直接沿尾刺方向
#   画出等长直线。
# - 本融合版把两处前提同时拿掉：
#   1. 所有杯共用一个边框色，所以不能再靠颜色区分杯，必须先在共享边框色里做 8 连通分割。
#   2. 多个杯不是并行独立处理，而是按 bbox 中心到网格中心的距离从近到远依次处理；先处理
#      的杯会把弹射线留在画布上，后处理的杯若撞到已有非背景像素就停止。
# - 具体例子：两个同色边框杯位于同一行，左边杯更靠近中心、尾巴朝右，右边杯更远、尾巴朝左。
#   solver 先在共享边框色里拆出两个连通块，再识别它们各自的尾刺/杯口。近中心左杯先发射，
#   在线路上留下若干内容色；轮到右杯时，它向左发射，走到左杯留下的线时停止，因此右杯输出
#   比原题更短。原题差异点在“按颜色逐杯且无遮挡”，本题差异点在“连通分杯 + 排序阻挡”。
# b) 复杂度校准
# - 为了让 D8 和 D15 同时可见，generator 固定生成 2-3 个杯，并强制其中一对形成同轴相向的
#   交互。这样观察者既会看到所有边框同色，也会看到远杯被近杯阻挡。
# - 第三个杯若存在，只放在清空通道里，不参与冲突，避免每个样例都过于拥挤。
# - 内容色保持每杯唯一且不同于共享边框色，减少“同色边框”带来的不必要歧义。
# c) 隐式假设审查
# - 原 `for border_col in colors_present(...)` 依赖“每杯边框色唯一”。融合后不成立；现在仍可枚举
#   颜色来寻找候选边框色，但真正区分杯实体必须依赖 `components_mask(g == border_col, 8)`。
# - 原 `cups.sort(... border_color)` 的 tie-break 不再有信息量，因为所有杯边框色相同；改为只用
#   可见几何量 `(dist2, r0, c0, r1, c1)`，并由 generator 避免距离平局。
# - 原无条件写出弹射线默认路径静态为空。D8 让这个前提失效，所以发射时必须在动态 `out` 上遇到
#   非背景即停；但杯识别仍必须只看静态输入 `g`，否则先前画出的线会污染后续杯解析。
# d) 放置合法性
# - 共享边框色并不允许边框相连；generator 仍要求 cup bbox 之间至少留 1 格，并检查 8 邻域间距，
#   保证同色边框在输入里会被稳定拆成多个连通块。
# - 交互只允许“线撞线”，不允许线穿过任何杯 bbox。pair 构造阶段显式约束两杯相向、间隔 >= 2，
#   且近杯的线只写到远杯发射通道中，不进入远杯框体。
# e) 视觉可观察性
# - 排序键只依赖空间位置，与颜色编号无关，解题者可以肉眼比较哪个杯更靠近画布中心。
# - 所有边框同色这一变化在输入里直接可见；同时样例中至少有一对同色杯输出不同长度的发射线，
#   迫使解题者推断“不是按颜色区分，而是按位置顺序处理”。
# f) 可视化整洁度
# - 关闭 stray pixel，避免共享边框色环境里再引入额外同色噪声。
# - 只保留一组主交互 pair，其余杯走独立清空通道，输出中不会出现线穿框、多重覆盖或难以归因的
#   障碍来源。
# g) 变化的可推断性
# - 颜色变化规则很简单：边框全同色，内容色各自保持不变并沿尾刺弹出。没有额外颜色映射。
# - 结构线索足够强：同色的多个杯、不同的内容色、以及明显受“谁更靠中心”影响的截断线段，共同
#   支撑解题者从少量样例中归纳融合规则。
# 新 main() 的隐式假设审查
# - 假设 1：有效杯仍对应某个单色 8 连通边框分量，且 bbox 内恰有一个能从对边 flood 到内容的尾刺
#   方向。generator 继续保证这一点。
# - 假设 2：每个杯 mouth 内只有一种“本杯内容色”；generator 为每杯分配唯一内容色并只在 mouth 内
#   散点填充。
# - 假设 3：排序距离最好无平局；solver 用几何 tie-break 保持确定性，generator 额外过滤掉距离平局。
# - 假设 4：动态阻挡只来自更早发射的线，而不是静态杂质或边框穿越；generator 通过禁用 stray 且检查
#   full ray 不碰其他 bbox 来保证这一点.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood within bbox from one edge, blocked by border cells."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _cup_order_key(bbox: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int, int, int]:
    """Sort by bbox-center distance to grid center, then by visible geometry."""
    r0, c0, r1, c1 = bbox
    dr2 = (r0 + r1) - (H - 1)
    dc2 = (c0 + c1) - (W - 1)
    return dr2 * dr2 + dc2 * dc2, r0, c0, r1, c1


def _parse_cup_component(
    g: np.ndarray,
    border_col: int,
    border_coords: np.ndarray,
    bg: int,
) -> dict | None:
    """Return cup metadata if one monochrome component matches the cup grammar."""
    if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
        return None

    r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
    bbox = (r0, c0, r1, c1)
    border_set = {(int(r), int(c)) for r, c in border_coords}
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

    edges = (
        ("U", [p for p in border_set if p[0] == r0]),
        ("D", [p for p in border_set if p[0] == r1]),
        ("L", [p for p in border_set if p[1] == c0]),
        ("R", [p for p in border_set if p[1] == c1]),
    )
    candidates = [
        (d, cells[0]) for d, cells in edges
        if len(cells) == 1 and cells[0] not in corners
    ]
    if not candidates:
        return None

    passing = []
    for tail_dir, spike in candidates:
        mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
        cnt = Counter()
        for r, c in mouth:
            v = int(g[r, c])
            if v != bg and v != border_col:
                cnt[v] += 1
        if not cnt:
            continue
        content_col = int(cnt.most_common(1)[0][0])
        content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
        passing.append((tail_dir, spike, content_col, content_cells, sorted(mouth)))

    if len(passing) != 1:
        return None

    tail_dir, spike, content_col, content_cells, mouth_cells = passing[0]
    return {
        "border_color": int(border_col),
        "bbox": bbox,
        "tail_dir": tail_dir,
        "spike": tuple(spike),
        "content_color": int(content_col),
        "content_cells": list(content_cells),
        "mouth_cells": mouth_cells,
    }


def _detect_cups(grid: np.ndarray) -> list[dict]:
    """
    Parse all cups from the static input grid.

    Important for D15: cup identity comes from 8-connected border components, not from a unique color.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            cup = _parse_cup_component(g, int(border_col), border_coords, bg)
            if cup is not None:
                cups.append(cup)

    return cups


def _ordered_ejection_cells(
    out: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    content_count: int,
    bg: int,
) -> list[tuple[int, int]]:
    """Emit until count is exhausted, edge is reached, or a non-bg obstacle is encountered."""
    H, W = out.shape
    dr, dc = DIR4[tail_dir]
    rr, cc = spike[0] + dr, spike[1] + dc
    cells: list[tuple[int, int]] = []
    while len(cells) < content_count and 0 <= rr < H and 0 <= cc < W:
        if int(out[rr, cc]) != bg:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each cup's content from near-center to far-center.

    Shared-color cups are first separated by 8-connectivity. Then the cup closer to the grid
    center is processed first; its drawn ejection line can block later cups.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _detect_cups(g)
    cups.sort(key=lambda cup: _cup_order_key(cup["bbox"], H, W))

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg
        for r, c in _ordered_ejection_cells(out, cup["spike"], cup["tail_dir"], len(cup["content_cells"]), bg):
            out[r, c] = int(cup["content_color"])

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.0
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the patch mouth interior from the opposite edge."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]
    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical (tail=U) patch into the target tail direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Sample a cup shape that fits the orientation-specific bbox limits."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _build_cup_spec(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    direction: str,
) -> dict | None:
    """Sample one cup shape with a fixed tail direction."""
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": direction,
        "patch": patch,
        "spike_rel": spike,
        "mouth_rel": mouth,
        "mouth_size": len(mouth),
        "ph": int(patch.shape[0]),
        "pw": int(patch.shape[1]),
    }


def _ray_from_spike(spike_abs: tuple[int, int], direction: str, H: int, W: int) -> list[tuple[int, int]]:
    """Cells from the spike outward to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    cells = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _finalize_cup(spec: dict, r0: int, c0: int, content_count: int, H: int, W: int) -> dict:
    """Attach absolute placement data to a sampled cup."""
    spike_abs = (r0 + int(spec["spike_rel"][0]), c0 + int(spec["spike_rel"][1]))
    patch = spec["patch"]
    ph, pw = patch.shape
    return {
        "border_color": int(spec["border_color"]),
        "content_color": int(spec["content_color"]),
        "direction": str(spec["direction"]),
        "patch": patch.copy(),
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": spike_abs,
        "spike_rel": tuple(spec["spike_rel"]),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in spec["mouth_rel"]],
        "content_count": int(content_count),
        "full_ray": _ray_from_spike(spike_abs, str(spec["direction"]), H, W),
    }


def _distance_value(bbox: tuple[int, int, int, int], H: int, W: int) -> int:
    return _cup_order_key(bbox, H, W)[0]


def _find_interacting_pair_placement(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec_a: dict,
    spec_b: dict,
    layout: str,
) -> tuple[dict, dict] | None:
    """
    Place a near-center cup A and a farther cup B so they face each other on one axis.
    A's line reaches into B's path; B keeps 1-3 cells then stops on A's already-written line.
    """
    ph_a, pw_a = int(spec_a["ph"]), int(spec_a["pw"])
    ph_b, pw_b = int(spec_b["ph"]), int(spec_b["pw"])
    sr_a, sc_a = spec_a["spike_rel"]
    sr_b, sc_b = spec_b["spike_rel"]
    lim_a = min(CONTENT_COUNT_MAX, int(spec_a["mouth_size"]))
    lim_b = min(CONTENT_COUNT_MAX, int(spec_b["mouth_size"]))
    candidates = []

    if layout in ("right", "left"):
        row_lo = max(int(sr_a), int(sr_b))
        row_hi = min(H - ph_a + int(sr_a), H - ph_b + int(sr_b))
        spike_rows = list(range(row_lo, row_hi + 1))
        rng.shuffle(spike_rows)
        for spike_row in spike_rows:
            r_a = spike_row - int(sr_a)
            r_b = spike_row - int(sr_b)
            a_cols = list(range(W - pw_a + 1))
            b_cols = list(range(W - pw_b + 1))
            rng.shuffle(a_cols)
            rng.shuffle(b_cols)
            for c_a in a_cols:
                bbox_a = (r_a, c_a, r_a + ph_a - 1, c_a + pw_a - 1)
                sa = c_a + int(sc_a)
                for c_b in b_cols:
                    bbox_b = (r_b, c_b, r_b + ph_b - 1, c_b + pw_b - 1)
                    if layout == "right":
                        if c_b < c_a + pw_a + 1:
                            continue
                        sb = c_b + int(sc_b)
                        gap = sb - sa - 1
                    else:
                        if c_a < c_b + pw_b + 1:
                            continue
                        sb = c_b + int(sc_b)
                        gap = sa - sb - 1
                    if gap < 2:
                        continue
                    dist_a = _distance_value(bbox_a, H, W)
                    dist_b = _distance_value(bbox_b, H, W)
                    if dist_a >= dist_b:
                        continue
                    free_max = min(3, gap - 1, lim_b - 1)
                    if free_max < 1:
                        continue
                    for free_before_block in range(1, free_max + 1):
                        count_a = gap - free_before_block
                        if not (1 <= count_a <= lim_a):
                            continue
                        count_b_min = free_before_block + 1
                        if count_b_min > lim_b:
                            continue
                        candidates.append((r_a, c_a, r_b, c_b, count_a, count_b_min, lim_b))
    else:
        col_lo = max(int(sc_a), int(sc_b))
        col_hi = min(W - pw_a + int(sc_a), W - pw_b + int(sc_b))
        spike_cols = list(range(col_lo, col_hi + 1))
        rng.shuffle(spike_cols)
        for spike_col in spike_cols:
            c_a = spike_col - int(sc_a)
            c_b = spike_col - int(sc_b)
            a_rows = list(range(H - ph_a + 1))
            b_rows = list(range(H - ph_b + 1))
            rng.shuffle(a_rows)
            rng.shuffle(b_rows)
            for r_a in a_rows:
                bbox_a = (r_a, c_a, r_a + ph_a - 1, c_a + pw_a - 1)
                sa = r_a + int(sr_a)
                for r_b in b_rows:
                    bbox_b = (r_b, c_b, r_b + ph_b - 1, c_b + pw_b - 1)
                    if layout == "down":
                        if r_b < r_a + ph_a + 1:
                            continue
                        sb = r_b + int(sr_b)
                        gap = sb - sa - 1
                    else:
                        if r_a < r_b + ph_b + 1:
                            continue
                        sb = r_b + int(sr_b)
                        gap = sa - sb - 1
                    if gap < 2:
                        continue
                    dist_a = _distance_value(bbox_a, H, W)
                    dist_b = _distance_value(bbox_b, H, W)
                    if dist_a >= dist_b:
                        continue
                    free_max = min(3, gap - 1, lim_b - 1)
                    if free_max < 1:
                        continue
                    for free_before_block in range(1, free_max + 1):
                        count_a = gap - free_before_block
                        if not (1 <= count_a <= lim_a):
                            continue
                        count_b_min = free_before_block + 1
                        if count_b_min > lim_b:
                            continue
                        candidates.append((r_a, c_a, r_b, c_b, count_a, count_b_min, lim_b))

    if not candidates:
        return None

    r_a, c_a, r_b, c_b, count_a, count_b_min, lim_b = candidates[int(rng.integers(0, len(candidates)))]
    count_b = int(rng.integers(count_b_min, min(lim_b, count_b_min + 2) + 1))
    cup_a = _finalize_cup(spec_a, r_a, c_a, count_a, H, W)
    cup_b = _finalize_cup(spec_b, r_b, c_b, count_b, H, W)
    return cup_a, cup_b


def _try_place_clear_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec: dict,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Place an additional cup whose full ray is entirely clear, to avoid extra clutter."""
    ph, pw = int(spec["ph"]), int(spec["pw"])
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)

    for r0, c0 in coords:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue
        trial = _finalize_cup(spec, r0, c0, 1, H, W)
        full_ray = trial["full_ray"]
        if not full_ray:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for (r, c) in full_ray):
            continue
        limit = min(CONTENT_COUNT_MAX, int(spec["mouth_size"]))
        trial["content_count"] = int(rng.integers(1, limit + 1))
        return trial
    return None


def _draw_borders(canvas: np.ndarray, cups_meta: list[dict]) -> None:
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        patch = meta["patch"]
        canvas[r0:r1 + 1, c0:c1 + 1][patch != 0] = int(meta["border_color"])


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's chosen number of content pixels inside its mouth."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = int(meta["content_count"])
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = int(meta["content_color"])
    return True


def _simulate_output(canvas_in: np.ndarray, cups_meta: list[dict], ordered: bool) -> tuple[np.ndarray, int]:
    """Simulate either the new ordered solver or the old order-insensitive one."""
    out = np.array(canvas_in, dtype=int)
    H, W = out.shape
    bg = int(get_bg(out))
    cups = list(cups_meta)
    cups.sort(key=lambda cup: _cup_order_key(cup["bbox"], H, W))
    blocked = 0
    for meta in cups:
        for r, c in meta["mouth_abs"]:
            if int(out[r, c]) == int(meta["content_color"]):
                out[r, c] = bg
        limit = min(int(meta["content_count"]), len(meta["full_ray"]))
        if ordered:
            cells = _ordered_ejection_cells(out, meta["spike_abs"], meta["direction"], int(meta["content_count"]), bg)
            if len(cells) < limit:
                blocked += 1
        else:
            cells = meta["full_ray"][:limit]
        for r, c in cells:
            out[r, c] = int(meta["content_color"])
    return out, blocked


def _distance_keys_are_unique(cups_meta: list[dict], H: int, W: int) -> bool:
    keys = [_cup_order_key(meta["bbox"], H, W)[0] for meta in cups_meta]
    return len(keys) == len(set(keys))


def _all_borders_share_one_color(cups_meta: list[dict]) -> bool:
    return len({int(meta["border_color"]) for meta in cups_meta}) == 1


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Build one instance with shared border color and at least one genuine order-dependent block."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    if num_cups < 2:
        return None

    palette = list(range(1, 10))
    rng.shuffle(palette)
    shared_border = int(palette.pop())
    if len(palette) < num_cups:
        return None
    content_colors = [int(palette.pop()) for _ in range(num_cups)]

    layout_to_dirs = {
        "right": ("R", "L"),
        "left": ("L", "R"),
        "down": ("D", "U"),
        "up": ("U", "D"),
    }
    layout = str(rng.choice(list(layout_to_dirs)))
    dir_a, dir_b = layout_to_dirs[layout]

    spec_a = _build_cup_spec(rng, shared_border, content_colors[0], dir_a)
    spec_b = _build_cup_spec(rng, shared_border, content_colors[1], dir_b)
    if spec_a is None or spec_b is None:
        return None

    placed_pair = _find_interacting_pair_placement(rng, H, W, spec_a, spec_b, layout)
    if placed_pair is None:
        return None
    cups_meta = [placed_pair[0], placed_pair[1]]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        occ_cup[r0:r1 + 1, c0:c1 + 1] = True
        for r, c in meta["full_ray"]:
            occ_path[r, c] = True

    for idx in range(2, num_cups):
        for _ in range(40):
            direction = str(rng.choice(["U", "D", "L", "R"]))
            spec = _build_cup_spec(rng, shared_border, content_colors[idx], direction)
            if spec is None:
                continue
            extra = _try_place_clear_cup(rng, H, W, spec, occ_cup, occ_path)
            if extra is None:
                continue
            cups_meta.append(extra)
            r0, c0, r1, c1 = extra["bbox"]
            occ_cup[r0:r0 + 1 + (r1 - r0), c0:c0 + 1 + (c1 - c0)] = True
            for r, c in extra["full_ray"]:
                occ_path[r, c] = True
            break
        else:
            return None

    if len(cups_meta) != num_cups:
        return None
    if not _all_borders_share_one_color(cups_meta):
        return None
    if not _distance_keys_are_unique(cups_meta, H, W):
        return None

    canvas_in = np.zeros((H, W), dtype=int)
    _draw_borders(canvas_in, cups_meta)
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    ordered_out, blocked_count = _simulate_output(canvas_in, cups_meta, ordered=True)
    parallel_out, _ = _simulate_output(canvas_in, cups_meta, ordered=False)
    if blocked_count < 1:
        return None
    if np.array_equal(ordered_out, parallel_out):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, ordered_out):
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
