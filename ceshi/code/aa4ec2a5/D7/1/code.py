# Variant of aa4ec2a5
# Dimension: D7 — 多步处理序列 (multi_step)
# Change: 两步序列：第一步检测洞并条件性重着色（对象变青色、洞变品红）；第二步对所有对象统一绘制红色轮廓... → 迭代轮廓绘制

# Design analysis:
# spec:
#   a_concretization_and_trace:
#     input:
#       - 保持原题输入外观: 黄色背景(4)上有 3-4 个蓝色(1) 4 连通对象.
#       - 其中 1-2 个对象含有完全包围的背景洞; 其余对象为实心.
#     output:
#       - 先做原题的洞检测: 有洞对象主体改成青色(8), 洞填成品红(6).
#       - 再做多轮 8 邻域外扩轮廓.
#       - 实心对象总共画 2 层: 第 1 层红(2), 第 2 层绿(3).
#       - 空心对象总共画 3 层: 第 1 层红(2), 第 2 层绿(3), 第 3 层红(2).
#     trace_example:
#       - 原题里, 一个实心 2x2 蓝块只会得到一圈紧贴的红边.
#       - 现在同一个 2x2 蓝块会先出现紧贴的红圈, 再在红圈外再长一圈绿圈.
#       - 原题里, 一个带洞对象会被改成青色/品红后只画一圈红边.
#       - 现在同一个带洞对象会先重着色, 再连续长出红圈、绿圈、再一圈红圈.
#       - 差异集中在“第二步”: 旧逻辑一次性求单层 outline, 新逻辑按层迭代生长并在不同步填不同颜色.
#   b_complexity_calibration:
#     - 只保留两类层数, 2 层和 3 层, 避免把规则做成依赖面积/长宽等难以从少量样例归纳的细粒度函数.
#     - 层数由“是否有洞”决定, 这是输入里可见且在每个样例里都有重复支撑的结构特征.
#     - 颜色模式固定为红->绿->红, 使用很少的颜色并跨对象复用, 人眼容易比较不同对象何时停止.
#   c_hidden_assumptions_review:
#     - 原 `_outline_mask(solid)` 默认只调用一次, 隐含“所有轮廓同色且只有一层”; 变体里不成立, 所以改成对 growing mask 迭代调用.
#     - 原 `out[outline & (out == bg)] = 2` 假设无后续层需要写入其他颜色; 变体里仍保留 `out == bg` 保护, 但每层单独写入对应颜色.
#     - 原 `solid` 在有洞时把洞并入 mask, 这个前提在变体里仍成立; 否则多层轮廓会错误地围绕内部洞继续扩张.
#     - 原放置约束只够 1 层轮廓安全; 变体需要依据对象层数预留更大边界和对象间距.
#   d_placement_legality:
#     - 若对象需要 L 层轮廓, 它离网格边缘至少要有 L 格, 否则外层会被截断.
#     - 两个对象的任意原始像素 Chebyshev 距离至少要满足 L1 + L2 + 1, 才能保证各自的 L1/L2 层轮廓不重叠.
#     - 生成器据此在 placement 阶段使用按对象类型动态变化的最小间距.
#   e_visual_observability:
#     - 样例中同时存在实心与空心对象, 所以会同时出现“停在第 2 层”和“继续到第 3 层”两种情况.
#     - 第二层统一为绿色, 第三层只出现在空心对象外侧, 因而规则可以从 2-3 组例子中稳定归纳.
#     - 不使用颜色 id 排序、面积排名等不可见条件.
#   f_visual_cleanliness:
#     - 颜色只增加 1 种新轮廓色(绿), 叠加层次最多三层.
#     - 扩大对象间距和边界余量, 避免不同对象外圈互相覆盖成杂乱色块.
#   g_color_inference:
#     - 颜色映射不是任意替换, 而是固定的层级序列: 第 1 层红, 第 2 层绿, 第 3 层红.
#     - 洞相关的青/品红规则保持不变, 让“对象内部重着色”和“外部多层轮廓”各自承担单一语义.

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
if str(ROOT / "solvers") not in sys.path:
    sys.path.append(str(ROOT / "solvers"))

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


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


LAYER_COLORS = (2, 3)
SOLID_LAYER_COUNT = 2
HOLLOW_LAYER_COUNT = 3
MAX_LAYER_COUNT = HOLLOW_LAYER_COUNT


def _iterative_outlines(mask: np.ndarray, layer_count: int) -> list[np.ndarray]:
    grown = mask.copy()
    layers: list[np.ndarray] = []
    for _ in range(layer_count):
        ring = _outline_mask(grown)
        layers.append(ring)
        grown |= ring
    return layers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects, then grow multi-ring outlines around every object.

    Input structure:
    - A mostly uniform background contains several blue 4-connected components.
    - Some components enclose one or more 4-connected background holes inside their bbox.

    Rule:
    - First detect enclosed holes exactly as in the base task: hollow components become teal (8)
      and their holes become magenta (6).
    - Then draw iterative 8-neighborhood outline layers around each component's exterior.
    - Solid components receive two layers colored red (2) then green (3); hollow components
      receive three layers colored red (2), green (3), red (2).

    Key insight:
    - The recoloring step still depends only on whether a component contains enclosed holes, but
      the outline step is no longer single-shot: treat the object (with holes filled for boundary
      purposes) as a seed that expands outward one pixel at a time, coloring each new ring by its
      layer index.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        layer_count = SOLID_LAYER_COUNT
        if holes:
            layer_count = HOLLOW_LAYER_COUNT
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        for layer_idx, ring in enumerate(_iterative_outlines(solid, layer_count)):
            out[ring & (out == bg)] = LAYER_COLORS[layer_idx % len(LAYER_COLORS)]

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
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

BORDER_MIN = SOLID_LAYER_COUNT
PLACEMENT_RETRIES = 400
GEN_OUTER_RETRIES = 400


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


def _component_layers(is_hollow: bool) -> int:
    return HOLLOW_LAYER_COUNT if is_hollow else SOLID_LAYER_COUNT


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
    placed: list[tuple[int, int, np.ndarray, int]] = []

    order = sorted(range(num_components), key=lambda i: -(components[i][0].sum() + 8 * _component_layers(components[i][1])))

    for idx in order:
        mask, is_hollow = components[idx]
        layers = _component_layers(is_hollow)
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(layers, max(layers, H - mh - layers))
            c = random.randint(layers, max(layers, W - mw - layers))
            if r + mh > H - layers or c + mw > W - layers:
                continue

            rows_a, cols_a = np.where(mask)
            rows_a, cols_a = rows_a + r, cols_a + c

            conflict = False
            for pr, pc, pm, prev_layers in placed:
                row_gap = max(0, max(r - (pr + pm.shape[0] - 1), pr - (r + mh - 1)))
                col_gap = max(0, max(c - (pc + pm.shape[1] - 1), pc - (c + mw - 1)))
                if max(row_gap, col_gap) >= layers + prev_layers + 1:
                    continue
                if _chebyshev_conflict(rows_a, cols_a, pr, pc, pm, layers + prev_layers + 1):
                    conflict = True
                    break

            if not conflict:
                placed.append((r, c, mask, layers))
                placed_ok = True
                break

        if not placed_ok:
            return None

    for r, c, mask, _layers in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        _ = main(grid)
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
