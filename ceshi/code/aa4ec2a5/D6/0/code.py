# Variant of aa4ec2a5
# Dimension: D6 — 网格尺寸与输出形状 (grid_shape)
# Change: 输出网格与输入尺寸相同（same），轮廓绘制在原始网格上... → 输出扩展以容纳外轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# ---------------------------------------------------------------------------
# Design analysis for the D6 variant
# ---------------------------------------------------------------------------
# variant_analysis:
#   source_task_baseline:
#     - 参考任务 20270e3b 的关键机制不是“怎么描边”，而是“输出尺寸本身由变换决定”。
#     - 本题沿用原题的对象级逻辑，只把 D6 改成可见且稳定的尺寸变化：
#       轮廓宽度固定为 1，所以输出在四周各扩展恰好 1 格，尺寸从 (H, W) 变成 (H+2, W+2)。
#   a_specific_behavior_and_trace:
#     - 输入仍是黄色背景上的 3-4 个蓝色 4 连通对象，其中 1-2 个对象带内部背景洞。
#     - 变体后的输入允许对象贴边；输出先在四周补一圈黄色背景，再把整张输入平移到中心。
#     - 之后完全沿用原题的对象处理：实心对象保持蓝色并画红色 8 邻域轮廓；
#       空心对象先改成青色、洞改成品红色，再把“对象+洞”当成实心区域来画红色轮廓。
#     - 小例子追踪：
#       输入若有一个单蓝格在原图左上角，原题会只在其右、下、右下画红色，左上方向的轮廓被裁掉；
#       变体会先把该蓝格放到扩展画布的 (1,1)，于是它左、上、左上方向的红色也能出现在新增外框里。
#   b_complexity_calibration:
#     - 不改变颜色规则、洞规则、组件数量、组件间距规则，避免把 D6 变化和别的维度混在一起。
#     - 为了让 D6 在 2-4 个样例里肉眼可见，生成器强制至少一个对象贴到输入边界，
#       这样新增外框里通常会真的出现红色“被救回”的轮廓，而不是只有一圈空背景。
#   c_hidden_assumptions_in_original_main:
#     - `out = g.copy()` 依赖“输出尺寸 == 输入尺寸”；变体下不成立，必须改成构造更大的画布。
#     - `find_components(g, ...)` 默认组件坐标直接就是输出坐标；变体下要先整体平移 +1,+1。
#     - `_outline_mask(mask)` 在原图尺寸内做邻域扩张时，越界像素会被静默裁掉；原题这是设定的一部分，
#       但变体要求保留这些像素，所以必须在扩张前先加外框。
#     - `out[outline & (out == bg)] = 2` 仍然安全：新增外框全是背景，且对象之间 Chebyshev 距离仍至少 3，
#       因此不同对象的轮廓不会互相覆盖。
#   d_placement_legality:
#     - 生成器把原来的 `BORDER_MIN=1` 改为允许 `0`，并强制最大的一个组件贴某一条边放置。
#     - 组件之间仍保持 Chebyshev 距离 >= 3，所以即使每个对象外扩 1 格轮廓，轮廓也不会相撞。
#     - 贴边只发生在输入里；solver 会先加外框再描边，因此不会越界。
#   e_visual_observability:
#     - 解题者会同时看到两个稳定信号：
#       1. 输出总比输入高 2、宽 2；
#       2. 靠边对象的红色外轮廓会出现在新增边框上，说明新尺寸不是任意填充而是为描边服务。
#   f_visual_cleanliness:
#     - 不增加新颜色，也不叠加多层标记；唯一新视觉元素就是外围一圈背景，以及边缘处补全的红色轮廓。
#     - 原有大间距保留，避免边缘轮廓与别的对象混淆。
#   g_color_inference:
#     - 颜色语义完全继承原题：蓝=普通对象，青=带洞对象主体，品红=洞，红=外轮廓，黄=背景。
#     - 因为颜色映射没有新增分支，人类无需推断新的颜色条件，只需推断“先扩框，再做原规则”。
#   new_main_assumptions_to_check:
#     - 输入背景仍应为主色；generator 继续用统一背景 4 保证这一点。
#     - 输入中的前景组件仍按 4 连通、单色分量解析；generator 保持所有对象初始为蓝色 1。
#     - 至少一个组件贴边不是 solver 的必要条件，只是 generator 为了让 D6 可观察而主动保证的性质。


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


def _expand_canvas(grid: np.ndarray, bg: int, pad: int = 1) -> np.ndarray:
    h, w = grid.shape
    out = np.full((h + 2 * pad, w + 2 * pad), int(bg), dtype=int)
    out[pad : pad + h, pad : pad + w] = grid
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the canvas before outlining so border-adjacent objects keep their full red halo.

    Input structure:
    - A mostly uniform background contains several blue 4-connected objects.
    - Some objects may enclose 4-connected background holes, and objects may sit against the grid edge.

    Rule:
    - Create an output canvas that is one cell larger on every side and place the input grid centered inside it.
    - On that expanded canvas, draw a one-cell 8-neighborhood red outline around each foreground component.
    - If a component encloses holes, recolor the component to teal and the enclosed holes to magenta before outlining.

    Key insight: The object-wise recoloring and outlining rule is unchanged; the only new step is
    to add a background margin first, because the outline width is one cell and should no longer be clipped by the old border.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    expanded = _expand_canvas(g, bg, pad=1)
    out = expanded.copy()

    for comp in find_components(expanded, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(expanded.shape, coords)
        holes = _interior_holes(mask, coords)

        solid = mask.copy()
        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
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
BORDER_MIN = 0
PLACEMENT_RETRIES = 300

# Top-level generation
GEN_OUTER_RETRIES = 300


def _crop_mask(mask):
    """Crop a boolean mask to its tight bounding box. Returns None if empty."""
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _has_interior_hole(mask):
    """Return True if the boolean mask contains at least one 4-connected interior bg region."""
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
    """Return (h, w) for a rectangle with given area, randomized among valid factorizations."""
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
    """Return True if any cell in (rows_a, cols_a) is within CHEBYSHEV_MIN of any cell in pm."""
    rows_b, cols_b = np.where(pm)
    rows_b, cols_b = rows_b + pr, cols_b + pc
    for ra, ca in zip(rows_a, cols_a):
        if np.min(np.maximum(np.abs(rows_b - ra), np.abs(cols_b - ca))) < CHEBYSHEV_MIN:
            return True
    return False


def _touches_grid_edge(r: int, c: int, mask: np.ndarray, H: int, W: int) -> bool:
    rows, cols = np.where(mask)
    return bool(
        np.any(rows + r == 0)
        or np.any(rows + r == H - 1)
        or np.any(cols + c == 0)
        or np.any(cols + c == W - 1)
    )


def _make_polyomino():
    """Generate an irregular polyomino from overlapping axis-aligned rectangles."""
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
    """Optionally extend mask with a rectangle on one side to create L/T shapes."""
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
    """Try to carve a hh x hw rectangular hole in mask's interior."""
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
    """Build a hollow component with a rectangular hole of the given area."""
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


def _make_small_shape():
    """Return a small (1-4 cell) solid polyomino mask."""
    size = random.randint(*SMALL_SHAPE_SIZE_RANGE)
    if size == 1:
        return np.ones((1, 1), dtype=bool)
    if size == 2:
        return (
            np.ones((1, 2), dtype=bool)
            if random.random() < 0.5
            else np.ones((2, 1), dtype=bool)
        )
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


def _sample_edge_position(H: int, W: int, mh: int, mw: int) -> tuple[int, int]:
    side = random.choice(["top", "bottom", "left", "right"])
    if side == "top":
        return 0, random.randint(0, W - mw)
    if side == "bottom":
        return H - mh, random.randint(0, W - mw)
    if side == "left":
        return random.randint(0, H - mh), 0
    return random.randint(0, H - mh), W - mw


def _gen_attempt():
    """Single attempt to generate a valid input grid."""
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(
        HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1)
    )

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
    edge_idx = order[0]

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        if mh > H or mw > W:
            return None

        if idx == edge_idx:
            r, c = _sample_edge_position(H, W, mh, mw)
            placed.append((r, c, mask))
            continue

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

    touches_edge = False
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR
        touches_edge = touches_edge or _touches_grid_edge(r, c, mask, H, W)

    if not touches_edge:
        return None

    out = main(grid)
    if out.shape != (H + 2, W + 2):
        return None
    outer_frame = np.concatenate([out[0, :], out[-1, :], out[1:-1, 0], out[1:-1, -1]])
    if not np.any(outer_frame == 2):
        return None

    return grid


def generate_input():
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
