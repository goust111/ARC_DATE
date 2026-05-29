# Variant of aa4ec2a5
# Dimension: D1 — 轮廓绘制方式 (transform)
# Change: 在每个蓝色对象周围绘制1像素宽的8邻域（Chebyshev距离=1）红色轮廓... → 方向性射线轮廓
#
# Design analysis
# a) 具体化与行为追踪
#    - 输入仍然是黄色背景上的 3-4 个蓝色 4 连通对象，其中 1-2 个带有内部背景洞。
#    - 输出仍然先对带洞对象做内部重着色：对象本体改为青色 8，洞改为品红色 6。
#    - 真正变化的是红色标记：不再在整圈 8 邻域画固定厚度的环，而是从对象“正交暴露边”向外发射红色射线。
#      某个对象像素若其上/下/左/右某一侧紧邻背景，则从该侧外面的第一个背景格开始，沿该方向一直涂红，
#      直到下一步将撞上任意对象像素，或离开网格为止。
#    - 小例子：若左侧对象的右边界与右侧对象的左边界在若干行上对齐，中间留 4 格空隙：
#      原题只会各自画一圈近邻红环；本变体会在这些重叠行上把整条走廊染成红色，直到右侧对象挡住射线。
# b) 复杂度校准
#    - 只保留两种终止机制：撞到对象 / 撞到网格边界。规则单一，但长度动态变化，视觉上明显不同于原题。
#    - 生成器强制至少出现一组“对象挡住射线”的走廊，同时保留一些“射线打到边界”的情况，避免样例全都退化成同一种终止。
#    - 颜色体系不扩张，仍然只使用 1/2/4/6/8，解题负担集中在几何推理而不是颜色记忆。
# c) 隐式假设审查
#    - 原 `_outline_mask` 依赖“所有红色都在 Chebyshev 距离 1 内”，这一前提在变体中失效，必须改成沿 4 个方向逐格推进。
#    - 原实现把 `out` 当作可逐步更新的画布是安全的，因为环状轮廓是局部的；新变体若用已涂出的红色作为阻挡物，
#      会错误地让先处理的对象阻塞后处理的对象。因此射线阻挡必须基于静态输入占用图，而不是基于部分输出。
#    - 原题对带洞对象先填洞再算轮廓；这在变体中依然必须保留，否则对象会向内部洞口错误发射射线，改变了 D1 之外的维度。
# d) 放置合法性
#    - 对象仍然不能重叠，且保持最小 Chebyshev 间距，避免输入本身不合法。
#    - 另外需要至少一对对象在同一行段或同一列段上形成清晰走廊，这样“射线撞对象”才稳定可见。
#    - 放置完成后使用新 `main()` 和额外行为指标做验收，只接受真正触发长射线的样本。
# e) 视觉可观察性
#    - 人类从 2-4 个样例中应能看出：红色不再只贴着对象一圈，而会沿行/列远距离延伸。
#    - 生成器要求出现超出原题 8 邻域环带的红格，并且至少有一个对象阻挡终止事件，保证新规律不是偶然噪声。
# f) 可视化整洁度
#    - 射线只走正交方向，不引入对角线碎片。
#    - 对象之间仍保留空隙，且颜色层次固定为对象 / 洞 / 红射线三层，避免局部难以辨认。
# g) 颜色变化可推断性
#    - 颜色映射与原题一致：普通对象保留蓝色，带洞对象改青色，洞改品红色，新增标记统一为红色。
#    - 因而解题者只需额外推断“红色从哪里射出、在哪里停止”，不用学习新的多色条件分支。
# h) 量化退化检查
#    - 使用 `far_red_cells` 衡量是否出现原题不可能出现的远距离红格。
#    - 使用 `object_hit_rays` / `boundary_hit_rays` 分别统计两种终止分支是否都被覆盖。
# i) 新 solver 的输入假设
#    - 射线阻挡依据是输入里的非背景对象占用；生成器据此保证存在至少一条清晰走廊。
#    - 带洞对象的洞必须完全封闭；生成器延续原题的洞构造与连通性检查来保证这一点。

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


def _component_records(grid: np.ndarray, bg: int) -> list[dict]:
    records: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        records.append({"coords": coords, "mask": mask, "holes": holes, "solid": solid})
    return records


def _ray_mask_and_stats(solid: np.ndarray, occupied: np.ndarray) -> tuple[np.ndarray, list[int], list[str]]:
    h, w = solid.shape
    rays = np.zeros_like(solid, dtype=bool)
    lengths: list[int] = []
    stops: list[str] = []

    rows, cols = np.where(solid)
    for r, c in zip(rows, cols):
        for dr, dc in DIRS4:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and solid[nr, nc]:
                continue

            rr, cc = nr, nc
            length = 0
            while 0 <= rr < h and 0 <= cc < w and not occupied[rr, cc]:
                rays[rr, cc] = True
                length += 1
                rr += dr
                cc += dc

            if length > 0:
                lengths.append(length)
                if 0 <= rr < h and 0 <= cc < w:
                    stops.append("object")
                else:
                    stops.append("boundary")

    return rays, lengths, stops


def _ring_zone(mask: np.ndarray) -> np.ndarray:
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


def variant_metrics(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    occupied = g != bg
    records = _component_records(g, bg)
    solid_union = np.zeros_like(occupied, dtype=bool)
    red_union = np.zeros_like(occupied, dtype=bool)
    all_lengths: list[int] = []
    stop_counts = {"object": 0, "boundary": 0}
    hollow_count = 0

    for record in records:
        solid = record["solid"]
        if record["holes"]:
            hollow_count += 1
        solid_union |= solid
        rays, lengths, stops = _ray_mask_and_stats(solid, occupied)
        red_union |= rays
        all_lengths.extend(lengths)
        for stop in stops:
            stop_counts[stop] += 1

    far_red = red_union & ~_ring_zone(solid_union)
    return {
        "components": len(records),
        "hollow_components": hollow_count,
        "total_red_cells": int(red_union.sum()),
        "far_red_cells": int(far_red.sum()),
        "max_ray_len": max(all_lengths, default=0),
        "object_hit_rays": int(stop_counts["object"]),
        "boundary_hit_rays": int(stop_counts["boundary"]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects, then emit red orthogonal rays from their exposed sides.

    Input structure:
    - A mostly uniform background contains one or more blue 4-connected foreground objects.
    - Some objects enclose 4-connected background holes inside their bounding boxes.

    Rule:
    - For every foreground component, emit red rays from each orthogonally exposed side.
    - Each ray starts on the adjacent background cell and keeps extending until the next step
      would hit another object cell or leave the grid.
    - If a component encloses holes, recolor the component to teal (8) and those holes to
      magenta (6) before drawing the rays.

    Key insight:
    - Rays are computed from the hole-filled silhouette of each object, so hollow interiors do
      not generate inward rays; the red pattern depends on outward directions and blockers, not
      on a fixed one-cell ring.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    occupied = g != bg
    out = g.copy()

    for record in _component_records(g, bg):
        mask = record["mask"]
        holes = record["holes"]
        solid = record["solid"]

        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6

        rays, _lengths, _stops = _ray_mask_and_stats(solid, occupied)
        out[rays & (g == bg)] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300
PAIR_PLACEMENT_RETRIES = 160
PAIR_GAP_RANGE = (3, 8)
PAIR_OVERLAP_MIN = 2

GEN_OUTER_RETRIES = 300
MIN_FAR_RED_CELLS = 3
MIN_MAX_RAY_LEN = 3


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


def _chebyshev_conflict(rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray) -> bool:
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _mask_world_coords(mask: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows + r, cols + c


def _placement_conflict(mask: np.ndarray, r: int, c: int, placed: list[tuple[int, int, np.ndarray]]) -> bool:
    rows_a, cols_a = _mask_world_coords(mask, r, c)
    for pr, pc, pm in placed:
        row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mask.shape[0] - 1)))
        col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mask.shape[1] - 1)))
        if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
            continue
        if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
            return True
    return False


def _make_polyomino() -> np.ndarray | None:
    num_rects = random.randint(*POLY_NUM_RECTS_RANGE)
    rects = [
        (random.randint(*POLY_RECT_DIM_RANGE), random.randint(*POLY_RECT_DIM_RANGE))
        for _ in range(num_rects)
    ]

    canvas_h = sum(r[0] for r in rects) + POLY_CANVAS_PAD
    canvas_w = sum(r[1] for r in rects) + POLY_CANVAS_PAD
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    r0, c0 = 2, 2
    mask[r0 : r0 + rects[0][0], c0 : c0 + rects[0][1]] = True

    for i in range(1, num_rects):
        rh, rw = rects[i]
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
        nh = random.randint(1, max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1))
        nw = random.randint(1, max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1))
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

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i in hollow_indices:
            found = False
            for _ in range(HOLLOW_GEN_RETRIES):
                mask = _make_hollow_component(random.choice(HOLE_VALID_AREAS))
                if mask is not None and mask.sum() >= HOLLOW_MIN_AREA:
                    components.append(mask)
                    found = True
                    break
            if not found:
                return None
        elif random.random() < SMALL_SHAPE_PROB:
            components.append(_make_small_shape())
        else:
            for _ in range(POLY_GEN_RETRIES):
                mask = _make_polyomino()
                if mask is not None:
                    components.append(mask)
                    break
            else:
                return None

    if len(components) != num_components:
        return None

    order = sorted(range(num_components), key=lambda i: -components[i].sum())
    pair_indices = (order[0], order[1])
    paired = _place_aligned_pair(components, H, W, pair_indices)
    if paired is None:
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
    already_placed = set()

    for idx, r, c, mask in paired:
        placed.append((r, c, mask))
        already_placed.add(idx)

    for idx in order:
        if idx in already_placed:
            continue

        mask = components[idx]
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
    if metrics["far_red_cells"] < MIN_FAR_RED_CELLS:
        return None
    if metrics["max_ray_len"] < MIN_MAX_RAY_LEN:
        return None
    if metrics["object_hit_rays"] < 1 or metrics["boundary_hit_rays"] < 1:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
