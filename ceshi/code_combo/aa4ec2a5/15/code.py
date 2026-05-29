# Combo variant of aa4ec2a5 (index 15)
# combo_spec: D1/4+D7/1
# Name: 方向性射线轮廓+迭代轮廓绘制
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入外观保持原题骨架: 黄色背景(4)上有 3-4 个蓝色(1) 4 连通对象, 其中 1-2 个带完全封闭的背景洞.
#    - 先做原题的洞检测与重着色: 带洞对象主体改青色(8), 洞改品红色(6).
#    - 真正的融合发生在“外部标记”步骤: 不再一次性画整圈 8 邻域轮廓, 也不再一次性把整条射线全涂成同色.
#      对每个对象的每条正交暴露边, 先按照 D1/4 追踪出一条完整直线射线, 直到下一步会撞到别的对象或越界.
#      然后把这些完整射线按回合同步显现: 第 1 回合画每条射线离对象最近的第 1 格, 第 2 回合继续在各射线末端前进 1 格,
#      第 3 回合再前进 1 格, 直到该射线走廊耗尽. 颜色按回合交替为红(2)、绿(3)、红(2)、绿(3)...
#    - 小例子: 两个对象水平对齐, 中间隔 5 格. 原题只会各自长一圈贴边红环; D1 单变体会把整条 5 格走廊全变红;
#      本融合版会让这条走廊呈现红-绿-红-绿-红的条纹, 视觉上既能看出“这是直射到对面对象的走廊”, 又能看出“它是逐轮生长的”.
# b) 复杂度校准
#    - 终止规则只保留两种可见情形: 撞对象 / 撞边界, 不引入旋转、分支或对象排序之类难以从少量样例归纳的隐藏条件.
#    - 多轮颜色只在两个值之间交替, 避免颜色自由度过高; 人类只需注意“沿射线距离奇偶不同而换色”.
#    - 迭代不是退化成局部两层小环, 因为生成器强制出现长度 >= 4 的长射线, 让第二轮乃至更远轮次都能直接看到.
# c) 隐式假设审查
#    - 原 `_outline_mask` 默认“红色都在对象 Chebyshev 距离 1 内”; 融合后失效, 必须先追踪整条 4 向射线.
#    - 原按对象顺序直接写 `out` 是安全的, 因为单层轮廓局部且同色; 融合后若仍按对象顺序写多轮颜色,
#      两个对象射线穿过同一格时会产生顺序依赖. 这里改成先汇总全局“最早到达回合”再统一上色.
#    - 原洞对象需要先把洞并入 `solid` 才算外轮廓; 这个前提在融合后更重要, 否则对象会向内部洞口发射伪射线.
#    - 射线阻挡只能看静态输入占用, 不能把已画出的射线当作墙, 否则早先对象会错误阻塞后续对象.
# d) 放置合法性
#    - 生成器仍保证对象本体互不重叠且保持最小 Chebyshev 间距.
#    - 额外强制放置一对水平或垂直对齐的大对象, 并给出 4-9 格走廊, 从而稳定触发“撞对象终止”的长射线.
#    - 边界留白加大到至少 4 格, 让“撞边界终止”的多轮射线也能显著出现, 而不是只剩 1 格短刺.
# e) 视觉可观察性
#    - 观察 2-3 个样例时, 人类能稳定看到两件事: 红绿条纹沿正交方向远离对象生长; 有些条纹撞到另一对象停止, 有些撞边界停止.
#    - 带洞对象仍然被改成青/品红, 所以“内部结构决定对象颜色”与“外部射线按距离交替着色”是两个可分离的规律.
# f) 可视化整洁度
#    - 只使用 1/2/3/4/6/8 六种颜色, 且新增的绿色只出现在红射线的偶数层.
#    - 通过扩大边距、对象间距并限制射线重叠格数量, 避免整张图被多方向条纹搅乱.
# g) 变化的可推断性
#    - 颜色模式不是任意映射, 而是沿每条直线射线的距离奇偶交替: 第 1、3、5... 格红, 第 2、4、6... 格绿.
#    - 该模式在每个样例里都会重复出现多次, 并且同时覆盖“射到对象”和“射到边界”两种终止情形.
# h) 量化退化检查
#    - 生成后统计 `far_ray_cells`, `green_cells`, `max_step`, `long_object_hit_rays`, `long_boundary_hit_rays`.
#    - 若没有远离对象的一段条纹, 或没有绿色偶数层, 或只有一种终止机制, 样本视为退化并丢弃.
# i) 新 solver 的输入假设
#    - 至少存在一些正交暴露边可向背景发射射线; 原题的多对象蓝块天然满足这一点.
#    - 为了让“迭代”可见, 生成器显式保证至少有一条长度 >= 4 的撞对象射线和一条长度 >= 4 的撞边界射线.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
LAYER_COLORS = (2, 3)


def _coords_mask(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _interior_holes(mask: np.ndarray, coords: np.ndarray) -> list[np.ndarray]:
    r0, c0, r1, c1 = bbox_coords(coords)
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]

    holes: list[np.ndarray] = []
    for hole in components_mask(~sub, connectivity=4):
        touches_edge = np.any(
            (hole[:, 0] == 0)
            | (hole[:, 1] == 0)
            | (hole[:, 0] == sub.shape[0] - 1)
            | (hole[:, 1] == sub.shape[1] - 1)
        )
        if not bool(touches_edge):
            shifted = hole.copy()
            shifted[:, 0] += int(r0)
            shifted[:, 1] += int(c0)
            holes.append(shifted)
    return holes


def _component_records(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        records.append({"coords": coords, "mask": mask, "holes": holes, "solid": solid})
    return records


def _trace_rays(solid: np.ndarray, occupied: np.ndarray) -> tuple[list[np.ndarray], list[int], list[str]]:
    h, w = solid.shape
    rays: list[np.ndarray] = []
    lengths: list[int] = []
    stops: list[str] = []

    rows, cols = np.where(solid)
    for r, c in zip(rows, cols):
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and solid[nr, nc]:
                continue

            rr, cc = nr, nc
            cells: list[tuple[int, int]] = []
            while 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]:
                cells.append((rr, cc))
                rr += dr
                cc += dc

            if cells:
                rays.append(np.array(cells, dtype=int))
                lengths.append(len(cells))
                stops.append("object" if 0 <= rr < h and 0 <= cc < w else "boundary")

    return rays, lengths, stops


def _outline_zone(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    zone = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            zone[rr[ok], cc[ok]] = True
    zone &= ~mask
    return zone


def _arrival_map_and_stats(
    records: list[dict[str, object]], occupied: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    arrival = np.zeros_like(occupied, dtype=int)
    reach_counts = np.zeros_like(occupied, dtype=int)
    total_rays = 0
    object_hit_rays = 0
    boundary_hit_rays = 0
    long_object_hit_rays = 0
    long_boundary_hit_rays = 0
    max_object_hit_len = 0
    max_boundary_hit_len = 0

    for record in records:
        rays, lengths, stops = _trace_rays(record["solid"], occupied)
        total_rays += len(rays)
        for ray, length, stop in zip(rays, lengths, stops):
            if stop == "object":
                object_hit_rays += 1
                max_object_hit_len = max(max_object_hit_len, length)
                if length >= 4:
                    long_object_hit_rays += 1
            else:
                boundary_hit_rays += 1
                max_boundary_hit_len = max(max_boundary_hit_len, length)
                if length >= 4:
                    long_boundary_hit_rays += 1

            for step, (r, c) in enumerate(ray, start=1):
                rr, cc = int(r), int(c)
                reach_counts[rr, cc] += 1
                prev = arrival[rr, cc]
                if prev == 0 or step < prev:
                    arrival[rr, cc] = step

    stats = {
        "total_rays": total_rays,
        "object_hit_rays": object_hit_rays,
        "boundary_hit_rays": boundary_hit_rays,
        "long_object_hit_rays": long_object_hit_rays,
        "long_boundary_hit_rays": long_boundary_hit_rays,
        "max_object_hit_len": max_object_hit_len,
        "max_boundary_hit_len": max_boundary_hit_len,
    }
    return arrival, reach_counts, stats


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    occupied = g != bg
    records = _component_records(g, bg)
    arrival, reach_counts, stats = _arrival_map_and_stats(records, occupied)

    solid_union = np.zeros_like(occupied, dtype=bool)
    hollow_count = 0
    for record in records:
        solid_union |= record["solid"]
        if record["holes"]:
            hollow_count += 1

    far_ray = (arrival > 0) & ~_outline_zone(solid_union)
    green = (arrival > 0) & ((arrival % 2) == 0)
    green_far = green & far_ray

    return {
        "components": len(records),
        "hollow_components": hollow_count,
        "total_ray_cells": int(np.count_nonzero(arrival)),
        "far_ray_cells": int(np.count_nonzero(far_ray)),
        "green_cells": int(np.count_nonzero(green)),
        "green_far_cells": int(np.count_nonzero(green_far)),
        "max_step": int(arrival.max(initial=0)),
        "contested_cells": int(np.count_nonzero(reach_counts > 1)),
        **stats,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects, then grow orthogonal ray outlines one step per round.

    Input structure:
    - A mostly uniform background contains several blue 4-connected foreground objects.
    - Some objects enclose 4-connected background holes fully inside their bounding boxes.

    Rule:
    - First detect enclosed holes as in the base task: hollow objects become teal (8) and their
      holes become magenta (6).
    - For each object's hole-filled silhouette, send rays outward from every orthogonally exposed
      side across background cells until the next step would hit another object or leave the grid.
    - Do not paint each full ray in one shot. Instead, color all first ray cells red (2), all
      second ray cells green (3), then keep alternating red/green for later steps.

    Key insight:
    - The geometry is still defined by straight D1-style blocker-aware rays, but the visible
      coloring depends on synchronous iteration distance from the source object. Use the static
      input occupancy to trace rays, fill holes before tracing, and assign colors by earliest
      arrival round rather than by object processing order.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    occupied = g != bg
    records = _component_records(g, bg)
    arrival, _reach_counts, _stats = _arrival_map_and_stats(records, occupied)

    out = g.copy()
    for record in records:
        mask = record["mask"]
        holes = record["holes"]
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6

    max_step = int(arrival.max(initial=0))
    for step in range(1, max_step + 1):
        out[arrival == step] = LAYER_COLORS[(step - 1) % len(LAYER_COLORS)]

    return out


# --------------- generator (c) ---------------

GRID_H_RANGE = (20, 34)
GRID_W_RANGE = (20, 33)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 4
BORDER_MIN = 4
PLACEMENT_RETRIES = 450
PAIR_PLACEMENT_RETRIES = 220
PAIR_GAP_RANGE = (4, 9)
PAIR_OVERLAP_MIN = 3

GEN_OUTER_RETRIES = 450
MIN_TOTAL_RAY_CELLS = 18
MIN_FAR_RAY_CELLS = 8
MIN_GREEN_CELLS = 6
MIN_GREEN_FAR_CELLS = 4
MIN_MAX_STEP = 4
MAX_CONTESTED_CELLS = 320


def _crop_mask(mask: np.ndarray) -> np.ndarray | None:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for bg_reg in components_mask(~mask, connectivity=4):
        touches_edge = np.any(
            (bg_reg[:, 0] == 0)
            | (bg_reg[:, 1] == 0)
            | (bg_reg[:, 0] == h - 1)
            | (bg_reg[:, 1] == w - 1)
        )
        if not touches_edge:
            return True
    return False


def _rect_dims(area: int) -> tuple[int, int] | None:
    factors = [
        (i, area // i)
        for i in range(1, int(area**0.5) + 1)
        if area % i == 0 and i <= HOLE_DIM_MAX and area // i <= HOLE_DIM_MAX
    ]
    if not factors:
        return None
    h, w = random.choice(factors)
    if random.random() < 0.5:
        h, w = w, h
    return h, w


def _chebyshev_conflict(
    rows_a: np.ndarray,
    cols_a: np.ndarray,
    pr: int,
    pc: int,
    pm: np.ndarray,
    min_sep: int,
) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < min_sep:
            return True
    return False


def _mask_world_coords(mask: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows + r, cols + c


def _placement_conflict(
    mask: np.ndarray,
    r: int,
    c: int,
    placed: list[tuple[int, int, np.ndarray]],
    min_sep: int = CHEBYSHEV_MIN,
) -> bool:
    rows_a, cols_a = _mask_world_coords(mask, r, c)
    for pr, pc, pm in placed:
        row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mask.shape[0] - 1)))
        col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mask.shape[1] - 1)))
        if max(row_gap, col_gap) >= min_sep:
            continue
        if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm, min_sep):
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(rh for rh, _ in rects) + POLY_CANVAS_PAD
    canvas_w = sum(rw for _, rw in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for rh, rw in rects[1:]:
        for _ in range(POLY_PLACEMENT_RETRIES):
            nr = random.randint(max(0, r0 - rh + 1), r0 + rects[0][0] + 2)
            nc = random.randint(max(0, c0 - rw + 1), c0 + rects[0][1] + 2)
            if nr + rh > canvas_h or nc + rw > canvas_w or nr < 0 or nc < 0:
                continue
            if mask[nr : nr + rh, nc : nc + rw].any():
                mask[nr : nr + rh, nc : nc + rw] = True
                break

    mask = _crop_mask(mask)
    if mask is None or not is_contiguous(mask.astype(int), background=0, connectivity=4):
        return None
    return mask


def _extend_mask(mask: np.ndarray, base_h: int, base_w: int) -> np.ndarray:
    if random.random() >= HOLLOW_EXTENSION_PROB:
        return mask

    side = random.choice(["top", "bottom", "left", "right"])
    ext_h = random.randint(*HOLLOW_EXT_H_RANGE)

    if side in ("top", "bottom"):
        ext_w = min(random.randint(2, max(2, base_w)), base_w)
        offset = random.randint(0, base_w - ext_w)
        new_mask = np.zeros((base_h + ext_h, base_w), dtype=bool)
        if side == "top":
            new_mask[ext_h:, :] = mask
            new_mask[0:ext_h, offset : offset + ext_w] = True
        else:
            new_mask[:base_h, :] = mask
            new_mask[base_h : base_h + ext_h, offset : offset + ext_w] = True
    else:
        ext_h_actual = min(ext_h, base_h)
        ext_w_actual = random.randint(*HOLLOW_EXT_SIDE_W_RANGE)
        offset = random.randint(0, base_h - ext_h_actual)
        new_mask = np.zeros((base_h, base_w + ext_w_actual), dtype=bool)
        if side == "left":
            new_mask[:, ext_w_actual:] = mask
            new_mask[offset : offset + ext_h_actual, 0:ext_w_actual] = True
        else:
            new_mask[:, :base_w] = mask
            new_mask[offset : offset + ext_h_actual, base_w : base_w + ext_w_actual] = True

    return new_mask


def _carve_hole(mask: np.ndarray, hh: int, hw: int) -> np.ndarray | None:
    mh, mw = mask.shape
    for _ in range(HOLLOW_HOLE_SEARCH_RETRIES):
        hr = random.randint(1, mh - hh - 1)
        hc = random.randint(1, mw - hw - 1)
        if not mask[hr : hr + hh, hc : hc + hw].all():
            continue
        if (
            not mask[hr - 1, hc : hc + hw].any()
            or not mask[hr + hh, hc : hc + hw].any()
            or not mask[hr : hr + hh, hc - 1].any()
            or not mask[hr : hr + hh, hc + hw].any()
        ):
            continue

        carved = mask.copy()
        carved[hr : hr + hh, hc : hc + hw] = False
        if not is_contiguous(carved.astype(int), background=0, connectivity=4):
            continue
        cropped = _crop_mask(carved)
        if cropped is not None and _has_interior_hole(cropped):
            return cropped
    return None


def _make_hollow_component(target_hole_area: int) -> np.ndarray | None:
    dims = _rect_dims(target_hole_area)
    if dims is None:
        return None
    hh, hw = dims

    pad_top = random.randint(*HOLLOW_PAD_RANGE)
    pad_bot = random.randint(*HOLLOW_PAD_RANGE)
    pad_left = random.randint(*HOLLOW_PAD_RANGE)
    pad_right = random.randint(*HOLLOW_PAD_RANGE)

    base_h = hh + pad_top + pad_bot
    base_w = hw + pad_left + pad_right
    mask = np.ones((base_h, base_w), dtype=bool)

    num_notches = random.randint(0, HOLLOW_MAX_NOTCHES)
    for _ in range(num_notches):
        corner = random.choice(["tl", "tr", "bl", "br"])
        nh = random.randint(
            1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1)
        )
        nw = random.randint(
            1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1)
        )
        if corner == "tl":
            mask[0:nh, 0:nw] = False
        elif corner == "tr":
            mask[0:nh, base_w - nw : base_w] = False
        elif corner == "bl":
            mask[base_h - nh : base_h, 0:nw] = False
        else:
            mask[base_h - nh : base_h, base_w - nw : base_w] = False

    mask = _extend_mask(mask, base_h, base_w)
    return _carve_hole(mask, hh, hw)


def _make_small_shape() -> np.ndarray:
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return np.ones((1, 2), dtype=bool) if random.random() < 0.5 else np.ones((2, 1), dtype=bool)
    if size == 3:
        return random.choice(
            [
                np.array([[1, 1], [1, 0]], dtype=bool),
                np.array([[1, 0], [1, 1]], dtype=bool),
                np.array([[1, 1, 1]], dtype=bool),
                np.array([[1], [1], [1]], dtype=bool),
            ]
        )
    return np.ones((2, 2), dtype=bool)


def _place_aligned_pair(
    components: list[np.ndarray], H: int, W: int, pair_indices: tuple[int, int]
) -> list[tuple[int, int, int, np.ndarray]] | None:
    for _ in range(PAIR_PLACEMENT_RETRIES):
        idx_a, idx_b = pair_indices
        if random.random() < 0.5:
            idx_a, idx_b = idx_b, idx_a

        mask_a = components[idx_a]
        mask_b = components[idx_b]
        mha, mwa = mask_a.shape
        mhb, mwb = mask_b.shape
        orientation = random.choice(["horizontal", "vertical"])

        if orientation == "horizontal":
            gap = random.randint(*PAIR_GAP_RANGE)
            total_w = mwa + gap + mwb
            max_c_a = W - BORDER_MIN - total_w
            max_r_a = H - BORDER_MIN - mha
            max_r_b = H - BORDER_MIN - mhb
            if max_c_a < BORDER_MIN or max_r_a < BORDER_MIN or max_r_b < BORDER_MIN:
                continue

            c_a = random.randint(BORDER_MIN, max_c_a)
            c_b = c_a + mwa + gap
            r_a = random.randint(BORDER_MIN, max_r_a)
            req = min(PAIR_OVERLAP_MIN, mha, mhb)
            low_b = max(BORDER_MIN, r_a - mhb + req)
            high_b = min(max_r_b, r_a + mha - req)
            if low_b > high_b:
                continue
            r_b = random.randint(low_b, high_b)
        else:
            gap = random.randint(*PAIR_GAP_RANGE)
            total_h = mha + gap + mhb
            max_r_a = H - BORDER_MIN - total_h
            max_c_a = W - BORDER_MIN - mwa
            max_c_b = W - BORDER_MIN - mwb
            if max_r_a < BORDER_MIN or max_c_a < BORDER_MIN or max_c_b < BORDER_MIN:
                continue

            r_a = random.randint(BORDER_MIN, max_r_a)
            r_b = r_a + mha + gap
            c_a = random.randint(BORDER_MIN, max_c_a)
            req = min(PAIR_OVERLAP_MIN, mwa, mwb)
            low_b = max(BORDER_MIN, c_a - mwb + req)
            high_b = min(max_c_b, c_a + mwa - req)
            if low_b > high_b:
                continue
            c_b = random.randint(low_b, high_b)

        placed: list[tuple[int, int, np.ndarray]] = []
        if _placement_conflict(mask_a, r_a, c_a, placed):
            continue
        placed.append((r_a, c_a, mask_a))
        if _placement_conflict(mask_b, r_b, c_b, placed):
            continue
        return [(idx_a, r_a, c_a, mask_a), (idx_b, r_b, c_b, mask_b)]

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components: list[tuple[np.ndarray, bool]] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and mask.sum() >= HOLLOW_MIN_AREA:
                    components.append((mask, True))
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append((_make_small_shape(), False))
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append((mask, False))
                    break
            else:
                return None

    if len(components) != num_components:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []

    order = sorted(range(num_components), key=lambda i: -components[i][0].sum())
    pair_indices = (order[0], order[1])
    paired = _place_aligned_pair([mask for mask, _ in components], H, W, pair_indices)
    if paired is None:
        return None

    already_placed: set[int] = set()
    for idx, r, c, mask in paired:
        placed.append((r, c, mask))
        already_placed.add(idx)

    for idx in order:
        if idx in already_placed:
            continue

        mask, _is_hollow = components[idx]
        mh, mw = mask.shape
        max_r = H - BORDER_MIN - mh
        max_c = W - BORDER_MIN - mw
        if max_r < BORDER_MIN or max_c < BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)
            if _placement_conflict(mask, r, c, placed):
                continue
            placed.append((r, c, mask))
            placed_ok = True
            break

        if not placed_ok:
            return None

    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        _ = main(grid)
    except Exception:
        return None

    metrics = variant_metrics(grid)
    if metrics["total_ray_cells"] < MIN_TOTAL_RAY_CELLS:
        return None
    if metrics["far_ray_cells"] < MIN_FAR_RAY_CELLS:
        return None
    if metrics["green_cells"] < MIN_GREEN_CELLS:
        return None
    if metrics["green_far_cells"] < MIN_GREEN_FAR_CELLS:
        return None
    if metrics["max_step"] < MIN_MAX_STEP:
        return None
    if metrics["long_object_hit_rays"] < 1 or metrics["long_boundary_hit_rays"] < 1:
        return None
    if metrics["contested_cells"] > MAX_CONTESTED_CELLS:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
