# Variant of aa4ec2a5
# Dimension: D6 — 网格尺寸与输出形状 (grid_shape)
# Change: 输出网格与输入尺寸相同（same），轮廓绘制在原始网格上... → 紧凑裁剪到包含所有轮廓的最小矩形

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, is_contiguous

# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 输入仍是黄色背景上的 3-4 个蓝色 4 连通对象，其中 1-2 个对象含有完全包围的背景洞。
#    - 原任务输出是在原画布上直接改色并描红；本变体先执行同样的改色/描边，再把结果裁到“所有非背景像素”
#      的最小外接矩形。被保留的非背景像素包括：蓝/青对象、品红洞、红色轮廓。
#    - 例子：若某个 3x3 实心蓝块放在输入网格中央，原任务输出仍是整张大网格；新任务会先得到一个 5x5
#      的红框包住蓝块，再把外侧纯黄背景全部删掉，最终只保留这块 5x5。
#    - 因此流程变化只发生在“所有组件处理完成之后”的最后一步：原逻辑 return out；变体逻辑 return tight_crop(out != bg)。
# b) 复杂度校准
#    - 颜色逻辑、洞检测、8 邻域轮廓都保持不变，解题者仍需识别“有洞对象会改成青/品红”的分支。
#    - 新增的唯一自由度是裁剪边界，但它由最终所有着色像素共同决定，结构性很强，不会引入不可见的隐式排序。
#    - 为了让 D6 可观察，生成器会强制对象离输入边界至少 2 格：对象离边 2 格，轮廓离边 1 格，裁剪后四边都会缩小。
# c) 对原 main() 的隐式假设审查
#    - g = np.asarray(...), bg = get_bg(g)：依赖输入仍有单一主背景色；变体不改变这一点。
#    - out = g.copy()：原先默认输出尺寸等于输入尺寸；现在它只作为“完整装饰画布”，最终不能直接返回。
#    - find_components(... monochromatic=True)：依赖输入前景仍是单色蓝组件；生成器继续保证。
#    - _interior_holes(mask, coords)：依赖洞定义仍基于对象 bbox 内的 4 连通背景区；变体未改洞语义。
#    - out[outline & (out == bg)] = 2：依赖轮廓只覆盖背景格，避免覆盖对象/洞颜色；裁剪不影响这一前提。
#    - 真正失效的旧前提只有“完整画布就是答案”，所以需要把 return out 改成 return tight_crop(out, bg)。
# d) 放置合法性
#    - 组件仍需彼此不重叠，且任意两组件单元的切比雪夫距离 >= 3，这样 1 像素红轮廓不会互相接触或混成一团。
#    - 额外把对象本体的边界余量从 1 提到 2，确保轮廓离输入边缘至少 1，从而裁剪后一定严格小于输入尺寸。
#    - generate_input() 仍调用 solver 相关逻辑进行自检，并额外检查 full-output 的内容 bbox 四侧确有背景余量。
# e) 视觉可观察性
#    - 人类在 2-3 个样例里会直接看到：输出不再保留原图外围大片黄背景，最外圈通常是红轮廓或其紧邻区域。
#    - 规则没有依赖颜色 id 排序或坐标奇偶等不可见因素，裁剪边界完全由可见的着色图形决定。
# f) 可视化整洁度
#    - 红轮廓仍只画在背景上，洞对象内部仍是青/品红，层级没有增加；裁剪反而去掉无信息背景，使图像更紧凑。
#    - 保留原有较大组件间距，避免裁剪后的多对象场景因为轮廓粘连而难分辨。
# g) 颜色变化可推断性
#    - 颜色集合与原题一致：蓝实心对象保持蓝；有洞对象变青，洞变品红，所有外轮廓变红。
#    - 新维度只改变输出范围，不增加新的颜色映射条件。
# h) 新 main() 的输入假设复查
#    - tight crop 假设最终至少存在一个非背景像素；生成器保证始终有前景组件。
#    - 裁剪基于完整装饰后的非背景 bbox，而不是仅基于输入前景 bbox；这样红轮廓不会被意外裁掉。
#    - 生成器通过 BORDER_MIN=2 和内容 bbox 余量检查保证这一裁剪目标总是存在且可见。


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


def _decorate_full(grid: np.ndarray) -> tuple[np.ndarray, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
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
        out[outline & (out == bg)] = 2

    return out, bg


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(grid, bbox_coords(coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Add red outlines around objects, recolor hollow ones, then crop away empty border area.

    Input structure:
    - A mostly uniform background contains one or more blue 4-connected foreground objects.
    - Some objects enclose 4-connected background holes inside their own bounding boxes.

    Rule:
    - For every foreground component, draw a one-cell red outline on its 8-neighborhood background cells.
    - If a component encloses holes, recolor that component to teal (8) and recolor each enclosed hole to magenta (6).
    - After all recoloring and outlining is complete, crop the result to the smallest rectangle containing every non-background result pixel.

    Key insight: The geometric transform is unchanged from the original task until the very end:
    compute outlines using the hole-filled solid mask, then treat the finished decorated scene as content
    and tightly trim away the surrounding background.
    """
    decorated, bg = _decorate_full(grid)
    return _tight_crop_non_bg(decorated, bg)


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
OUTLINE_RADIUS = 1
CROP_MARGIN_MIN = 1
BORDER_MIN = OUTLINE_RADIUS + CROP_MARGIN_MIN
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


def _content_bbox(grid: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return None
    return bbox_coords(coords)


def _content_margins(grid: np.ndarray, bg: int) -> tuple[int, int, int, int] | None:
    bbox = _content_bbox(grid, bg)
    if bbox is None:
        return None
    r0, c0, r1, c1 = bbox
    h, w = grid.shape
    return r0, c0, h - 1 - r1, w - 1 - c1


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
                if mask is not None and int(mask.sum()) >= HOLLOW_MIN_AREA:
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
    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))

    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape
        max_r = H - mh - BORDER_MIN
        max_c = W - mw - BORDER_MIN
        if max_r < BORDER_MIN or max_c < BORDER_MIN:
            return None

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max_r)
            c = random.randint(BORDER_MIN, max_c)

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

    decorated, bg = _decorate_full(grid)
    cropped = _tight_crop_non_bg(decorated, bg)
    margins = _content_margins(decorated, bg)
    if margins is None:
        return None
    if min(margins) < CROP_MARGIN_MIN:
        return None
    if cropped.shape == grid.shape:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
