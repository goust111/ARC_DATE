# Variant of aa4ec2a5
# Dimension: D15 — 轮廓的视觉样式 (transform)
# Change: 连续的1像素宽红色实线轮廓，覆盖对象8邻域的所有背景像素... → 间断点轮廓（虚线效果）

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---------------
# variant_spec:
#   core_change:
#     original: 对每个对象的完整8邻域背景像素全部上红色，形成连续实线轮廓。
#     variant: 先照旧求出完整8邻域轮廓，再只保留固定棋盘格奇偶类中的一半像素；
#       这里固定保留 (row + col) % 2 == 0 的轮廓格，因此视觉上变成“隔一个点一个红点”的稀疏虚线。
#   concrete_behavior_trace:
#     example:
#       input_object: 2x2 实心蓝块
#       original_flow: 完整外圈8个方向的所有背景格都会变红，得到封闭实框。
#       variant_flow: 仍先求出同一圈候选背景格，但只在其中的偶奇校验格着红，
#         于是外圈只剩交替分布的红点；对象本身和洞的重着色规则不变。
#       difference: 差异只发生在“outline candidate -> final red cells”的筛选环节。
#   complexity_calibration:
#     - 轮廓筛选只使用一个全局可见的棋盘格模式，没有按对象分类，也没有依赖隐藏排序。
#     - 相比原题，红色数量显著减少，但仍保留轮廓位置线索，2-4 个样例足以观察到“隔一个像素上色”。
#   implicit_assumptions_review:
#     - 原 _outline_mask 假设“所有候选轮廓格都应上色”；变体下此假设失效，所以 main() 不能直接把完整轮廓写成红色。
#     - 原 out[outline & (out == bg)] = 2 依赖轮廓只落在背景上；该前提在变体中仍成立，因为虚线只是完整轮廓的子集。
#     - 原 holes 在描边前并入 solid mask；该前提仍需保留，否则 hollow 对象的外轮廓会错误地绕洞内部一圈。
#   placement_legality:
#     - 原生成器要求组件间 Chebyshev >= 3，足以保证“完整实线轮廓”互不重叠。
#     - 变体轮廓更稀疏，因此沿用同一放置约束仍然安全，不会越界，也不会让对象间轮廓相互覆盖。
#   visual_observability:
#     - 若直接随机删点会像噪声；固定全局棋盘格则能在多个对象上重复出现同一节奏，便于人眼归纳。
#     - 生成器额外过滤掉总红点过少的样本，避免出现虽然合法但虚线效果过弱的训练对。
#   cleanliness:
#     - 只减少红色轮廓密度，不增加新颜色和新层级，整体比原题更疏朗，不会引入区域遮挡。
#   color_inference:
#     - 颜色语义完全延续原题：红=轮廓，青=带洞对象，品红=洞。
#     - 变体不新增颜色映射，只改变红色轮廓的空间采样模式。
#   generator_guarantees_for_new_main:
#     - 至少仍有 1-2 个 hollow 组件，确保青/品红分支持续出现。
#     - 通过完整轮廓数与虚线轮廓数的差值阈值，保证样本真正体现 D15，而不是退化成近似原题。


OUTLINE_KEEP_PARITY = 0


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


def _outline_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    outline = np.zeros_like(mask, dtype=bool)
    rows, cols = np.where(mask)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            outline[rr[ok], cc[ok]] = True
    outline &= ~mask
    return outline


def _dashed_outline_mask(mask: np.ndarray) -> np.ndarray:
    outline = _outline_mask(mask)
    parity = (np.indices(mask.shape).sum(axis=0) % 2) == OUTLINE_KEEP_PARITY
    return outline & parity


def _component_solid_mask(shape: tuple[int, int], coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    mask = _coords_mask(shape, coords)
    holes = _interior_holes(mask, coords)
    solid = mask.copy()
    for hole in holes:
        solid[hole[:, 0], hole[:, 1]] = True
    return solid, holes


def _outline_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    full_outline_cells = 0
    dashed_outline_cells = 0
    hollow_components = 0
    solid_components = 0

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        solid, holes = _component_solid_mask(g.shape, comp["coords"])
        if holes:
            hollow_components += 1
        else:
            solid_components += 1
        full_outline_cells += int(_outline_mask(solid).sum())
        dashed_outline_cells += int(_dashed_outline_mask(solid).sum())

    return {
        "full_outline_cells": full_outline_cells,
        "dashed_outline_cells": dashed_outline_cells,
        "hollow_components": hollow_components,
        "solid_components": solid_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add sparse red dotted outlines around objects, and mark enclosed holes inside hollow objects.

    Input structure:
    - A mostly uniform background contains one or more 4-connected blue foreground objects.
    - Some objects enclose 4-connected background holes fully inside their bounding boxes.

    Rule:
    - For every foreground component, first determine its usual one-cell 8-neighborhood outline.
    - If a component encloses holes, recolor the component to teal (8) and those holes to magenta (6).
    - Paint only every other candidate outline cell in red (2), using a fixed checkerboard parity on background cells.

    Key insight:
    - Hole handling is unchanged from the original task: holes count as part of the solid mass for outlining.
    - The new behavior is entirely in the final outline sampling step, which turns a full border into an alternating dotted one.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _dashed_outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)
BG_COLOR = 4
FG_COLOR = 1

# Component counts
COMP_COUNT_RANGE = (3, 4)
HOLLOW_COUNT_RANGE = (1, 2)

# Polyomino construction (solid shapes)
POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 7)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 30

# Small solid shapes
SMALL_SHAPE_PROB = 0.2
SMALL_SHAPE_SIZE_RANGE = (1, 4)

# Hollow component construction
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

# Placement
CHEBYSHEV_MIN = 3
BORDER_MIN = 1
PLACEMENT_RETRIES = 300

# Top-level generation
GEN_OUTER_RETRIES = 300

# Variant visibility checks
MIN_DASHED_OUTLINE_CELLS = 12
MIN_OUTLINE_CELL_DROP = 6


def _crop_mask(mask):
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask):
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


def _rect_dims(area):
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


def _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _make_polyomino():
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


def _extend_mask(mask, base_h, base_w):
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


def _carve_hole(mask, hh, hw):
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


def _make_hollow_component(target_hole_area):
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
            1,
            max(1, pad_top - 1) if corner[0] == "t" else max(1, pad_bot - 1),
        )
        nw = random.randint(
            1,
            max(1, pad_left - 1) if corner[1] == "l" else max(1, pad_right - 1),
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


def _make_small_shape():
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        if random.random() < 0.5:
            return np.ones((1, 2), dtype=bool)
        return np.ones((2, 1), dtype=bool)
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


def _gen_attempt():
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])

    components = []
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

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed = []
    order = sorted(range(num_components), key=lambda i: -components[i].sum())

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, H - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, W - mw - BORDER_MIN))
            if r + mh > H - BORDER_MIN or c + mw > W - BORDER_MIN:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= CHEBYSHEV_MIN:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        out = main(grid)
    except Exception:
        return None

    metrics = _outline_metrics(grid)
    if metrics["dashed_outline_cells"] < MIN_DASHED_OUTLINE_CELLS:
        return None
    if metrics["full_outline_cells"] - metrics["dashed_outline_cells"] < MIN_OUTLINE_CELL_DROP:
        return None
    if int(np.count_nonzero(out == 2)) != metrics["dashed_outline_cells"]:
        return None

    return grid


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
