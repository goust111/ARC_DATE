# Variant of aa4ec2a5
# Dimension: D13 — 符号编码与图例系统 (symbol_instructions)
# Change: 无显式图例或符号编码，所有规则通过输入输出对隐式传达 → 图例定义轮廓样式

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis:
#   concrete_behavior:
#     - 输入保留原题的 3-4 个主蓝色对象和黄色背景，但额外在某个角落放一个微型图例。
#     - 图例固定为一个 2x2 蓝色小方块，外侧已经有 1 像素宽红色轮廓；因为方块是 2x2，
#       人类可以直接看出轮廓包含对角角点，所以模板明确编码了“8 邻域、宽度 1”。
#     - 输出中图例原样保留；真正需要变换的是其余蓝色对象。实心对象仍只加红轮廓，
#       有洞对象仍先改成青色并把洞填成品红，再按同一轮廓样式描边。
#   behavior_trace:
#     - 原题在输入中没有红色，solver 可以把所有非背景连通块都当作目标对象。
#     - 现在若沿用原写法，角落红图例会被误识别为一个对象，而且其围住的蓝色中心会被
#       当作“洞”，从而被错误重着色；这正是 D13 变体引入的新失败模式。
#     - 因此新 solver 必须先识别图例：用角落蓝块与现有红像素做匹配，反推出轮廓参数，
#       再把该蓝块从主对象集合中排除。
#   complexity_calibration:
#     - 只增加一个图例，不引入多图例、多符号表或颜色对应表，避免任务从“读模板”膨胀成
#       “做类比推理”。模板只负责显式说明轮廓样式，洞的重着色逻辑保持原题复杂度。
#     - 图例形状固定、位置只在四个角中变化，这样视觉上足够显著，但不会制造新的歧义轴。
#   assumption_review:
#     - 原 main 的 `find_components(..., monochromatic=True)` 默认遍历所有非背景颜色；在
#       变体里这个前提失效，因为输入合法地出现了红色图例。所以新代码只枚举蓝色组件。
#     - 原 `_outline_mask` 固定 8 邻域宽 1；变体要求从图例读出该参数，因此改为支持
#       4/8 邻域与少量宽度候选，再由图例选择其一。
#     - 原 `out[outline & (out == bg)] = 2` 仍然安全，因为图例已占据非背景色，真正对象
#       只会在背景上绘制轮廓，不会覆盖图例。
#   placement_legality:
#     - 生成器先把图例作为一个已占用块放进某个角落，再按原有 Chebyshev >= 3 规则放置主对象，
#       保证后续新轮廓不会碰到图例，也不会越界。
#   visual_observability:
#     - 输入中唯一的红色就是角落模板，视觉角色非常清晰；输出中主对象的红轮廓与图例红轮廓
#       同构，人类能从 2-3 个样例中归纳“照着角落示例去描边”。
#   color_inference:
#     - 只新增一种输入颜色红色，而且它总以“包围蓝块的模板轮廓”身份出现，不依赖任何
#       不可见排序或颜色 id 规则；青色/品红色仍只由“对象是否有洞”这个可见结构决定。


BG_COLOR = 4
FG_COLOR = 1
OUTLINE_COLOR = 2
HOLE_FILL_COLOR = 6
HOLLOW_OBJECT_COLOR = 8


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


def _expand_once(mask: np.ndarray, connectivity: int) -> np.ndarray:
    h, w = mask.shape
    rows, cols = np.where(mask)
    expanded = mask.copy()

    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ]

    for dr, dc in deltas:
        rr = rows + dr
        cc = cols + dc
        ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
        expanded[rr[ok], cc[ok]] = True
    return expanded


def _outline_mask(mask: np.ndarray, connectivity: int = 8, width: int = 1) -> np.ndarray:
    if width < 1:
        raise ValueError("outline width must be >= 1")
    grown = mask.copy()
    for _ in range(width):
        grown = _expand_once(grown, connectivity)
    return grown & ~mask


def _bbox_is_in_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int], slack: int = 3) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 <= slack and c0 <= slack)
        or (r0 <= slack and c1 >= w - 1 - slack)
        or (r1 >= h - 1 - slack and c0 <= slack)
        or (r1 >= h - 1 - slack and c1 >= w - 1 - slack)
    )


def _infer_legend_rule(grid: np.ndarray, bg: int) -> tuple[int, int, np.ndarray]:
    red_mask = grid == OUTLINE_COLOR
    empty = np.zeros_like(red_mask, dtype=bool)
    if not red_mask.any():
        return 8, 1, empty

    candidates = [(4, 1), (8, 1), (4, 2), (8, 2)]
    blue_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[FG_COLOR],
    )
    for comp in blue_components:
        if not _bbox_is_in_corner(comp["bbox"], grid.shape):
            continue
        mask = _coords_mask(grid.shape, comp["coords"])
        for connectivity, width in candidates:
            if np.array_equal(_outline_mask(mask, connectivity=connectivity, width=width), red_mask):
                return connectivity, width, mask
    return 8, 1, empty


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the corner legend to determine the contour style, then apply it to the real objects.

    Input structure:
    - A yellow background contains several main blue 4-connected objects plus one tiny corner legend.
    - The legend is a small blue sample shape that already has a red outline, and some main objects may enclose background holes.

    Rule:
    - Read the legend to infer how outlines are drawn, and leave that legend unchanged.
    - For every other blue component, draw the same red outline on background cells only.
    - If a main component encloses holes, recolor the component teal and those holes magenta before outlining, treating the holes as filled for the outline calculation.

    Key insight: The pre-outlined corner sample is instructional rather than transformable; it explicitly encodes the border neighborhood/width, while the hollow-object recoloring rule stays the same as in the original task.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    outline_connectivity, outline_width, legend_mask = _infer_legend_rule(g, bg)
    out = g.copy()

    for comp in find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[FG_COLOR],
    ):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        if legend_mask.any() and np.array_equal(mask, legend_mask):
            continue

        holes = _interior_holes(mask, coords)
        solid = mask.copy()

        if holes:
            out[mask] = HOLLOW_OBJECT_COLOR
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = HOLE_FILL_COLOR
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid, connectivity=outline_connectivity, width=outline_width)
        out[outline & (out == bg)] = OUTLINE_COLOR

    return out


# --------------- generator (c) ---------------

# ── Configuration ────────────────────────────────────────────────────────────
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Main task components only; the legend is an extra instructional object.
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

LEGEND_PATCH_SIZE = 4
LEGEND_INSET = 1
LEGEND_CORNERS = ("tl", "tr", "bl", "br")

GEN_OUTER_RETRIES = 300


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
        if not bool(touches_edge):
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


def _legend_patch() -> tuple[np.ndarray, np.ndarray]:
    patch = np.full((LEGEND_PATCH_SIZE, LEGEND_PATCH_SIZE), BG_COLOR, dtype=int)
    blue_mask = np.zeros((LEGEND_PATCH_SIZE, LEGEND_PATCH_SIZE), dtype=bool)
    blue_mask[1:3, 1:3] = True
    patch[blue_mask] = FG_COLOR
    patch[_outline_mask(blue_mask, connectivity=8, width=1)] = OUTLINE_COLOR
    return patch, patch != BG_COLOR


def _legend_anchor(h: int, w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return LEGEND_INSET, LEGEND_INSET
    if corner == "tr":
        return LEGEND_INSET, w - LEGEND_PATCH_SIZE - LEGEND_INSET
    if corner == "bl":
        return h - LEGEND_PATCH_SIZE - LEGEND_INSET, LEGEND_INSET
    return h - LEGEND_PATCH_SIZE - LEGEND_INSET, w - LEGEND_PATCH_SIZE - LEGEND_INSET


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)

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

    grid = np.full((h, w), BG_COLOR, dtype=int)
    placed: list[tuple[int, int, np.ndarray]] = []

    legend_corner = random.choice(LEGEND_CORNERS)
    legend_patch, legend_occ = _legend_patch()
    legend_r, legend_c = _legend_anchor(h, w, legend_corner)
    placed.append((legend_r, legend_c, legend_occ))

    order = sorted(range(num_components), key=lambda i: -int(components[i].sum()))
    for idx in order:
        mask = components[idx]
        mh, mw = mask.shape

        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            r = random.randint(BORDER_MIN, max(BORDER_MIN, h - mh - BORDER_MIN))
            c = random.randint(BORDER_MIN, max(BORDER_MIN, w - mw - BORDER_MIN))
            if r + mh > h - BORDER_MIN or c + mw > w - BORDER_MIN:
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

    grid[legend_r : legend_r + LEGEND_PATCH_SIZE, legend_c : legend_c + LEGEND_PATCH_SIZE] = legend_patch

    for r, c, mask in placed[1:]:
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
