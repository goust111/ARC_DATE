# Variant of aa4ec2a5
# Dimension: D7 — 多步处理序列 (multi_step)
# Change: 两步序列：第一步检测洞并条件性重着色（对象变青色、洞变品红）；第二步对所有对象统一绘制红色轮廓... → 三步序列：检测洞、重着色、按类别分别绘制不同样式轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# ---
# variant_spec:
#   preserved_structure:
#     - 输入仍是黄色背景上的 3-4 个蓝色 4 连通对象。
#     - 仍然允许 1-2 个对象带有完全包围的矩形洞，其余对象无洞。
#     - 有洞对象仍先重着色为青色，洞填成品红色。
#   new_d7_change:
#     - 原题只有两步链条：检测洞/重着色 -> 给所有对象画同一种红色轮廓。
#     - 这里改成三步链条：检测洞 -> 条件重着色 -> 按“是否有洞”选择不同轮廓模板。
#     - 有洞对象继续使用完整 8 邻域实线轮廓。
#     - 无洞对象改用同一个 8 邻域轮廓的交替子集，形成稳定的点状/间断轮廓。
#   concrete_trace:
#     - 例：一个 3x3 实心方块，原题会在周围 16 个背景格全部画红。
#     - 变体中它没有洞，所以仍保持蓝色，但红色只画在这 16 个格中的交替一半。
#     - 例：一个包围 2x2 洞的环形对象，原题会把环涂青、洞涂品红，并在外围画完整红环。
#     - 变体中它仍做同样重着色，但“有洞”这一判断还会继续驱动最后一步，保留完整实线轮廓。
#   complexity_calibration:
#     - 只引入两类轮廓样式，避免样式档位过多导致样例不足以归纳。
#     - 不增加新颜色，仍只使用原题中的 1/2/4/6/8，避免把难度转移成颜色映射题。
#     - 生成器保证同一题里同时出现 hollow 和 solid，且至少 2 个 solid，让“点状轮廓”有重复证据。
#   implicit_assumptions_review:
#     - 原题 `outline = _outline_mask(solid)` 默认所有对象共享同一描边规则；该前提已失效，必须按洞检测结果分支。
#     - 原题 `out[outline & (out == bg)] = 2` 依赖 outline 之间互不相交；本变体保留 1 像素外扩，所以仍需维持原放置间距。
#     - 原题把洞并入 `solid` 后再描边；这个前提对 hollow 分支仍成立，否则会错误地把洞内侧也当作外轮廓。
#     - 新增点状轮廓后，样式锚点必须可见且稳定，因此用对象 bbox 左上角作局部棋盘锚点，而不是依赖不可见元数据。
#   placement_legality:
#     - 仍保持组件间 Chebyshev 距离 >= 3，保证完整轮廓与点状轮廓都不会互相覆盖。
#     - 仍保持边界留白 >= 1，允许外扩后的轮廓接触画布边缘但不越界。
#   visual_observability:
#     - 人类能直接看到：有洞对象输出里内部先变青/品红，外围红圈完整；无洞对象保持蓝色，外围红圈呈交替断点。
#     - 这个差异在 2-3 组样例中即可重复出现，且由“有无洞”这一明显几何特征触发。
#   cleanliness:
#     - 点状轮廓仍只画在背景格上，不会压住对象本体或品红洞。
#     - 不引入第二层额外标记，避免轮廓、重着色、洞填充互相打架。


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


def _dotted_outline_mask(mask: np.ndarray, coords: np.ndarray) -> np.ndarray:
    outline = _outline_mask(mask)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    rr, cc = np.indices(mask.shape)
    dotted = outline & (((rr - int(r0) + cc - int(c0)) % 2) == 0)

    if outline.any() and not dotted.any():
        first_r, first_c = np.argwhere(outline)[0]
        dotted[first_r, first_c] = True
    return dotted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Process each object in three linked stages: classify holes, recolor interiors, then choose an outline style.

    Input structure:
    - A uniform background contains several blue 4-connected foreground objects.
    - Some objects enclose 4-connected background holes inside their bounding boxes, while others are solid.

    Rule:
    - First detect whether each object contains enclosed holes.
    - If holes are present, recolor the object to teal (8), recolor the holes to magenta (6), and draw a full red (2) 8-neighborhood outline around the object-plus-holes mask.
    - If no holes are present, keep the object blue (1) and draw only an alternating dotted subset of that same red 8-neighborhood outline.

    Key insight:
    - The hole test now controls two later actions: hollow objects get interior recoloring and a solid outline, while solid objects skip recoloring and receive a dotted outline chosen from the same outer ring.
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
            outline = _outline_mask(solid)
        else:
            outline = _dotted_outline_mask(solid, coords)

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
MIN_SOLID_COUNT = 2

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


# ── Helpers ──────────────────────────────────────────────────────────────────

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


def _count_hollow_components(grid: np.ndarray) -> int:
    bg = get_bg(grid)
    count = 0
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        mask = _coords_mask(grid.shape, comp["coords"])
        if _interior_holes(mask, comp["coords"]):
            count += 1
    return count


def _solid_outline_delta(grid: np.ndarray) -> int:
    bg = get_bg(grid)
    total = 0
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(grid.shape, coords)
        holes = _interior_holes(mask, coords)
        if holes:
            continue
        full_outline = _outline_mask(mask)
        dotted_outline = _dotted_outline_mask(mask, coords)
        total += int(full_outline.sum() - dotted_outline.sum())
    return total


# ── Shape generators ─────────────────────────────────────────────────────────

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


# ── Orchestration ────────────────────────────────────────────────────────────

def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    max_hollow = min(HOLLOW_COUNT_RANGE[1], num_components - MIN_SOLID_COUNT)
    if max_hollow < HOLLOW_COUNT_RANGE[0]:
        return None
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], max_hollow)

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

    grid = np.full((H, W), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []
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
        _ = main(grid)
    except Exception:
        return None

    hollow_count = _count_hollow_components(grid)
    solid_count = num_components - hollow_count
    if hollow_count < 1 or solid_count < MIN_SOLID_COUNT:
        return None
    if _solid_outline_delta(grid) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
