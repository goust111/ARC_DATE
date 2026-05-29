# Combo variant of aa4ec2a5 (index 29)
# combo_spec: D1/0+D13/2
# Name: 图例定义轮廓样式+4邻域轮廓

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪:
#    - 原题输入只有黄底+蓝色主对象，输出对每个对象画 8 邻域红轮廓；若对象有洞，则对象先变青、
#      洞变品红，再按“把洞视作已填满”的实体去描边。
#    - 本融合版在输入角落额外放一个 4x4 微型图例：中心是 2x2 蓝方块，外围只在上下左右四边
#      画红色，不画四个对角角点。这个模板把“宽度 1、4 邻域轮廓”显式展示出来。
#    - 具体例子: 若主区域里有一个 2x2 蓝块，原题输出会把其外接 4x4 框的 12 个外圈格都变红；
#      融合版则只把边中间的 8 个格变红，四角保持黄。若另一个对象有洞，它仍先变青/品红，但
#      随后外轮廓也只沿正交方向扩张。差异同时发生在“先读图例”与“改成 4 邻域描边”两步。
# b) 复杂度校准:
#    - 图例只编码一个小自由度: 轮廓邻域类型；不引入多种模板、颜色对照表或宽度档位，避免规则
#      爆炸。
#    - 主对象仍保留原题的 3-4 个不规则组件、1-2 个中空组件、大小混合与洞重着色，因此任务并未
#      被简化成“只临摹一个角落贴纸”。
#    - 生成器额外拒绝“主对象在 4 邻域与 8 邻域输出完全相同”的退化样本，确保组合中的 D1 改动
#      在主任务区域也可见，而不只是角落图例可见。
# c) 隐式假设审查:
#    - 原 main 默认“所有非背景连通块都是待变换对象”，在本变体中失效，因为输入合法地含有红色
#      图例；因此 solver 只枚举蓝色连通块。
#    - 原 main 默认轮廓样式固定为 8 邻域；本变体中必须先从角落蓝块与红色模板的关系中推断轮廓
#      规则，再应用到主对象。
#    - 原洞检测依赖“对象 mask 的 bbox 内部的背景连通域”，这一点不受图例影响，因为图例与主对
#      象分离，且洞定义与轮廓邻域无关。
#    - 原 `out[outline & (out == bg)] = 2` 的安全前提仍成立，但现在它还隐含了“图例区域在输入里
#      已经是非背景，因此不会被主对象描边覆盖”，生成器需保证这一点。
# d) 放置合法性:
#    - 先把图例占用块当作一个已放置组件，再用原任务的 Chebyshev >= 3 规则放置主对象。这个约束
#      强于 4 邻域描边所需间距，因此红轮廓不会彼此接触，也不会碰到图例。
#    - 图例固定放在四角之一且内缩 1 格，主对象仍至少离全局边界 1 格，因此图例和主对象的轮廓都
#      在网格内可完整呈现。
# e) 视觉可观察性:
#    - 输入里唯一的红色就是角落图例，解题者会自然把它读成说明而不是普通对象。
#    - 图例使用 2x2 蓝方块而不是 1x1 点，因为 2x2 能明确区分“有无四个对角红角点”，几组样例后
#      人类可以直接归纳“主对象也按这种正交轮廓画”。
# f) 可视化整洁度:
#    - 图例只占 4x4，且与主对象保持大间距；主区域只保留原题已有的蓝/红/青/品红四层语义，不再
#      添加额外标记，所以不会产生密集叠画。
# g) 变化的可推断性:
#    - 新增输入颜色只有红色，且始终作为“围绕蓝模板的预画轮廓”出现；洞触发的青/品红变化仍由可
#      见结构“对象是否包围背景洞”决定。颜色语义总数仍然很低，足以从少量样例归纳。
# 新 main 的额外假设与生成器保证:
#    - 假设存在且仅存在一个角落蓝组件能与全部红色像素精确匹配某个候选轮廓规则；生成器固定使用
#      单个角落图例，并确保输入中无其他红色像素。
#    - 假设主对象都是蓝色 4 连通块；生成器沿用原构造方式满足这一点。
#    - 假设主对象与图例分离，避免跳过图例时误伤主对象；生成器把图例占用区纳入同一碰撞检测。


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


def _outline_mask(mask: np.ndarray, connectivity: int = 4, width: int = 1) -> np.ndarray:
    if width < 1:
        raise ValueError("outline width must be >= 1")
    grown = mask.copy()
    for _ in range(width):
        grown = _expand_once(grown, connectivity)
    return grown & ~mask


def _bbox_is_in_corner(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], slack: int = 3
) -> bool:
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
        return 4, 1, empty

    candidates = [(4, 1), (8, 1)]
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
    return 4, 1, empty


def _transform_with_rule(
    grid: np.ndarray, outline_connectivity: int, outline_width: int, legend_mask: np.ndarray
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
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


def _main_object_delta_count(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    _, _, legend_mask = _infer_legend_rule(g, bg)
    orth = _transform_with_rule(g, outline_connectivity=4, outline_width=1, legend_mask=legend_mask)
    diag = _transform_with_rule(g, outline_connectivity=8, outline_width=1, legend_mask=legend_mask)
    return int(np.count_nonzero(orth != diag))


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    connectivity, width, legend_mask = _infer_legend_rule(g, bg)
    red_mask = g == OUTLINE_COLOR
    return {
        "legend_detected": int(legend_mask.any()),
        "legend_connectivity": int(connectivity),
        "legend_width": int(width),
        "legend_red_cells": int(red_mask.sum()),
        "main_delta": int(_main_object_delta_count(g)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a corner legend that defines the contour style, then redraw all main objects with that style.

    Input structure:
    - A yellow background contains several blue 4-connected main objects plus one tiny corner legend.
    - The legend is a small blue sample shape already surrounded by red pixels, and some main objects enclose background holes.

    Rule:
    - Treat the corner sample as an instruction: infer its outline neighborhood/width and leave the legend unchanged.
    - For every other blue component, draw the same one-pixel red outline on background cells only.
    - If a main component contains enclosed holes, recolor the component teal and those holes magenta before outlining, with holes counted as filled when computing the outline.

    Key insight: The red pixels in the input are not another object class; they are a miniature example showing that corners are excluded, so the real task is to generalize that 4-neighborhood contour rule to all non-legend components.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    outline_connectivity, outline_width, legend_mask = _infer_legend_rule(g, bg)
    return _transform_with_rule(
        g,
        outline_connectivity=outline_connectivity,
        outline_width=outline_width,
        legend_mask=legend_mask,
    )


# --------------- generator (c) ---------------

# Canvas / grid bounds
GRID_H_RANGE = (15, 28)
GRID_W_RANGE = (15, 27)

# Main task components only; the legend is extra.
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

# Legend
LEGEND_PATCH_SIZE = 4
LEGEND_INSET = 1
LEGEND_CORNERS = ("tl", "tr", "bl", "br")

# Top-level generation
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


def _chebyshev_conflict(
    rows_a: np.ndarray, cols_a: np.ndarray, pr: int, pc: int, pm: np.ndarray
) -> bool:
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
    patch[_outline_mask(blue_mask, connectivity=4, width=1)] = OUTLINE_COLOR
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
    num_hollow = random.randint(
        HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1)
    )

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

    metrics = _fusion_metrics(grid)
    if metrics["legend_detected"] != 1:
        return None
    if metrics["legend_connectivity"] != 4 or metrics["legend_width"] != 1:
        return None
    if metrics["main_delta"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
