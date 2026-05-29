# Combo variant of aa4ec2a5 (index 8)
# combo_spec: D6/1+D9/0
# Name: 紧凑裁剪+轮廓覆盖一切

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, find_components, get_bg, is_contiguous

# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 输入仍是黄底上的 3-4 个蓝色 4 连通对象，且 1-2 个对象带有完全包围的矩形洞。
#    - 原题流程是：逐个对象处理；若有洞则先把对象改成青色、洞改成品红；然后只在背景格上画红色 8 邻域轮廓；
#      最后保持原画布尺寸不变。
#    - 本融合变体改成两处联动：
#      1) 红轮廓不再受“只能落在背景上”的限制，而是对所有非本对象实体格都能覆写，因此相邻对象的边缘格
#         也可能被染成红色。
#      2) 所有着色完成后，再把整张结果图裁到包含全部非背景像素的最小矩形。
#    - 具体小例子：若一个 3x3 大块与一个 2x2 小块在输入里只隔一个对角（4 连通下仍是两个对象），原题输出中
#      两者仍保持蓝/青主体，只在外侧黄背景上添红；本变体里，大块的轮廓会把小块的一个角改成红，小块的轮廓
#      也会反过来改掉大块的一个角。最终裁剪时保留的是“被红色覆写后的整片内容”的最小外接框。
#    - 因此 solver 不能再边遍历边写红，否则后处理对象会把先前的红色盖回去，违背“红色最高优先级”。
# b) 复杂度校准
#    - 裁剪维度只增加一个全局 bbox 推断；覆盖维度只增加一种稳定且可重复观察的局部现象：对角相邻对象会互相
#      吃掉角点。两者都来自可见几何关系，没有额外隐式排序。
#    - 为避免退化成原题，生成器强制至少放置一对“仅对角接触”的组件，使 outline-overwrite 像素数始终 > 0。
#    - 同时只制造一组近邻，其余组件仍保持较大间距，这样样例中既能看出新行为，又不会整图变成大面积红色混战。
# c) 对原 main() 的隐式假设审查
#    - `bg = get_bg(g)` 仍要求输入有单一主背景色；生成器继续保证黄底占多数。
#    - `find_components(... connectivity=4, monochromatic=True)` 仍依赖输入前景都是蓝色且 4 连通分割；对角接触不会合并，
#      所以近邻方案与该前提兼容。
#    - `_interior_holes(mask, coords)` 仍按组件 bbox 内的 4 连通背景区找洞；融合规则没有改变洞定义。
#    - 原题的 `out[outline & (out == bg)] = 2` 依赖“红色只画背景”。这个前提在 D9/0 下失效，必须改成：
#      先累积所有对象的轮廓掩码，再统一 `out[outline_union] = 2`。
#    - 原题默认 `return out`，等价于输出尺寸固定等于输入；D6/1 下这个前提也失效，必须改成对最终非背景内容 tight crop。
# d) 放置合法性
#    - 输入组件本体仍必须互不重叠，且除刻意制造的那一对近邻外，其余组件之间保持切比雪夫距离 >= 3。
#    - 近邻对采用“至少一处对角接触、绝不出现正交接触”的约束，这样它们在输入里依旧是两个 4 连通对象，
#      但输出里一定会发生红色覆写。
#    - 所有对象本体离输入边界至少 2 格，使最终轮廓离边界至少 1 格，保证裁剪后输出严格小于输入。
# e) 视觉可观察性
#    - 从 2-3 个样例里，人类能直接看到两件事：外围黄底被裁掉；一对本来分开的蓝/青对象在输出里有角点被红色侵蚀。
#    - 规则只依赖肉眼可见的相对位置关系，尤其是“对角接近会被红轮廓盖掉”，没有用颜色编号大小或坐标奇偶等
#      不可见信息。
# f) 可视化整洁度
#    - 红、蓝、青、品红四种角色仍各司其职：蓝/青是对象主体，品红是洞，红是最高层轮廓。为了避免画面脏乱，
#      生成器只保留一组近邻对，其余组件拉开距离。
#    - 小配对组件限定为 3-4 格，避免像 1x1 那样被红色完全吞没，导致样例看起来像随机红点。
# g) 变化的可推断性
#    - 颜色集合仍很克制：实心对象蓝色、有洞对象青色、洞品红、轮廓红色。新增行为不是“多了一种颜色规则”，
#      而是“红色何时能压过对象颜色”的层级变化，结构线索很明确。
# h) 新 main() 的输入假设复查
#    - 新 solver 假设至少存在一个对象，且 tight crop 的内容 bbox 非空；生成器保证 3-4 个组件。
#    - 新 solver 还隐含假设：若要观察 D9/0，输入里最好存在至少一个轮廓覆写他物的像素。生成器用
#      `MIN_OUTLINE_OVERWRITE` 指标强制这一点，避免样本退化为纯 D6/1。
#    - 由于红色最终统一落笔，处理顺序不再影响结果；生成器无需依赖组件枚举顺序来制造特定图案。


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


def _decorate_full(grid: np.ndarray) -> tuple[np.ndarray, int, int]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    outline_union = np.zeros(g.shape, dtype=bool)

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

        outline_union |= _outline_mask(solid)

    pre_outline = out.copy()
    overwrite_count = int(np.count_nonzero(outline_union & (pre_outline != bg)))
    out[outline_union] = 2
    return out, bg, overwrite_count


def _tight_crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != int(bg))
    if coords.size == 0:
        return np.full((0, 0), int(bg), dtype=int)
    return crop_bbox(grid, bbox_coords(coords))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects, let red outlines overwrite anything they touch, then trim the canvas tightly.

    Input structure:
    - A mostly uniform yellow background contains 3-4 blue 4-connected foreground components.
    - Some components contain enclosed 4-connected background holes inside their own bounding boxes.

    Rule:
    - For each component, treat enclosed holes as interior: recolor hole-bearing components to teal (8) and their holes to magenta (6).
    - Compute a one-cell 8-neighborhood outline around each hole-filled component shape.
    - Apply the union of all those outlines in red (2) as the final top layer, even when the red pixels land on neighboring objects.
    - After all recoloring and red overwriting is finished, crop the output to the smallest rectangle containing every non-background pixel.

    Key insight: The outline is no longer a background-only decoration. It is a global overwrite layer that must be applied
    after every component's base colors are settled; only then can the final non-background bounding box be cropped correctly.
    """
    decorated, bg, _overwrite_count = _decorate_full(grid)
    return _tight_crop_non_bg(decorated, bg)


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

SMALL_SHAPE_PROB = 0.15
SMALL_SHAPE_SIZE_RANGE = (1, 4)
PAIR_SMALL_CHOICES = [
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
]

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

OUTLINE_RADIUS = 1
CROP_MARGIN_MIN = 1
BORDER_MIN = OUTLINE_RADIUS + CROP_MARGIN_MIN
FAR_CHEBYSHEV_MIN = 3
ANCHOR_PLACEMENT_RETRIES = 120
PLACEMENT_RETRIES = 300
MIN_OUTLINE_OVERWRITE = 2

GEN_OUTER_RETRIES = 400


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


def _placed_rows_cols(mask: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows + int(r), cols + int(c)


def _min_chebyshev(rows_a: np.ndarray, cols_a: np.ndarray, rows_b: np.ndarray, cols_b: np.ndarray) -> int:
    return int(
        np.min(
            np.maximum(
                np.abs(rows_a[:, None] - rows_b[None, :]),
                np.abs(cols_a[:, None] - cols_b[None, :]),
            )
        )
    )


def _has_orthogonal_touch(rows_a: np.ndarray, cols_a: np.ndarray, rows_b: np.ndarray, cols_b: np.ndarray) -> bool:
    return bool(np.any((np.abs(rows_a[:, None] - rows_b[None, :]) + np.abs(cols_a[:, None] - cols_b[None, :])) == 1))


def _has_diagonal_touch(rows_a: np.ndarray, cols_a: np.ndarray, rows_b: np.ndarray, cols_b: np.ndarray) -> bool:
    return bool(
        np.any((np.abs(rows_a[:, None] - rows_b[None, :]) == 1) & (np.abs(cols_a[:, None] - cols_b[None, :]) == 1))
    )


def _can_place_far(rows: np.ndarray, cols: np.ndarray, placed: list[dict]) -> bool:
    for item in placed:
        if _min_chebyshev(rows, cols, item["rows"], item["cols"]) < FAR_CHEBYSHEV_MIN:
            return False
    return True


def _in_bounds_with_margin(H: int, W: int, mask: np.ndarray, r: int, c: int) -> bool:
    mh, mw = mask.shape
    return BORDER_MIN <= r and BORDER_MIN <= c and r + mh <= H - BORDER_MIN and c + mw <= W - BORDER_MIN


def _choose_pair_small_shape() -> np.ndarray:
    return random.choice(PAIR_SMALL_CHOICES).copy()


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


# ── Placement helpers ────────────────────────────────────────────────────────

def _random_far_placement(H: int, W: int, mask: np.ndarray, placed: list[dict]) -> tuple[int, int] | None:
    mh, mw = mask.shape
    max_r = H - mh - BORDER_MIN
    max_c = W - mw - BORDER_MIN
    if max_r < BORDER_MIN or max_c < BORDER_MIN:
        return None

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(BORDER_MIN, max_r)
        c = random.randint(BORDER_MIN, max_c)
        rows, cols = _placed_rows_cols(mask, r, c)
        if _can_place_far(rows, cols, placed):
            return r, c
    return None


def _diagonal_touch_placement(
    H: int,
    W: int,
    anchor_mask: np.ndarray,
    anchor_r: int,
    anchor_c: int,
    pair_mask: np.ndarray,
) -> tuple[int, int] | None:
    anchor_rows, anchor_cols = _placed_rows_cols(anchor_mask, anchor_r, anchor_c)
    pair_local_rows, pair_local_cols = np.where(pair_mask)
    diagonals = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    anchor_order = list(range(len(anchor_rows)))
    pair_order = list(range(len(pair_local_rows)))
    random.shuffle(anchor_order)
    random.shuffle(pair_order)
    random.shuffle(diagonals)

    seen: set[tuple[int, int]] = set()
    for ai in anchor_order:
        ar = int(anchor_rows[ai])
        ac = int(anchor_cols[ai])
        for bi in pair_order:
            br = int(pair_local_rows[bi])
            bc = int(pair_local_cols[bi])
            for dr, dc in diagonals:
                r = ar + dr - br
                c = ac + dc - bc
                if (r, c) in seen:
                    continue
                seen.add((r, c))
                if not _in_bounds_with_margin(H, W, pair_mask, r, c):
                    continue
                rows, cols = _placed_rows_cols(pair_mask, r, c)
                if _min_chebyshev(rows, cols, anchor_rows, anchor_cols) != 1:
                    continue
                if _has_orthogonal_touch(rows, cols, anchor_rows, anchor_cols):
                    continue
                if not _has_diagonal_touch(rows, cols, anchor_rows, anchor_cols):
                    continue
                return r, c
    return None


def _place_anchor_and_pair(
    H: int,
    W: int,
    anchor_mask: np.ndarray,
    pair_mask: np.ndarray,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    mh, mw = anchor_mask.shape
    max_r = H - mh - BORDER_MIN
    max_c = W - mw - BORDER_MIN
    if max_r < BORDER_MIN or max_c < BORDER_MIN:
        return None

    for _ in range(ANCHOR_PLACEMENT_RETRIES):
        anchor_r = random.randint(BORDER_MIN, max_r)
        anchor_c = random.randint(BORDER_MIN, max_c)
        pair_pos = _diagonal_touch_placement(H, W, anchor_mask, anchor_r, anchor_c, pair_mask)
        if pair_pos is not None:
            return (anchor_r, anchor_c), pair_pos
    return None


# ── Orchestration ────────────────────────────────────────────────────────────

def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1))

    indices = list(range(num_components))
    random.shuffle(indices)
    hollow_indices = set(indices[:num_hollow])
    solid_indices = [i for i in range(num_components) if i not in hollow_indices]
    pair_small_idx = random.choice(solid_indices)

    components: list[np.ndarray] = []
    for i in range(num_components):
        if i == pair_small_idx:
            components.append(_choose_pair_small_shape())
            continue
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

    anchor_candidates = [i for i in range(num_components) if i != pair_small_idx]
    hollow_anchor_candidates = [i for i in anchor_candidates if i in hollow_indices]
    if hollow_anchor_candidates:
        pair_anchor_idx = max(hollow_anchor_candidates, key=lambda i: int(components[i].sum()))
    else:
        pair_anchor_idx = max(anchor_candidates, key=lambda i: int(components[i].sum()))

    placed: list[dict] = []
    anchor_and_pair = _place_anchor_and_pair(H, W, components[pair_anchor_idx], components[pair_small_idx])
    if anchor_and_pair is None:
        return None
    (anchor_r, anchor_c), (pair_r, pair_c) = anchor_and_pair

    for idx, (r, c) in (
        (pair_anchor_idx, (anchor_r, anchor_c)),
        (pair_small_idx, (pair_r, pair_c)),
    ):
        rows, cols = _placed_rows_cols(components[idx], r, c)
        placed.append({"idx": idx, "r": r, "c": c, "mask": components[idx], "rows": rows, "cols": cols})

    remaining = [i for i in range(num_components) if i not in {pair_anchor_idx, pair_small_idx}]
    remaining.sort(key=lambda i: -int(components[i].sum()))
    for idx in remaining:
        pos = _random_far_placement(H, W, components[idx], placed)
        if pos is None:
            return None
        r, c = pos
        rows, cols = _placed_rows_cols(components[idx], r, c)
        placed.append({"idx": idx, "r": r, "c": c, "mask": components[idx], "rows": rows, "cols": cols})

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for item in placed:
        grid[item["rows"], item["cols"]] = FG_COLOR

    decorated, bg, overwrite_count = _decorate_full(grid)
    cropped = _tight_crop_non_bg(decorated, bg)
    margins = _content_margins(decorated, bg)

    if margins is None or min(margins) < CROP_MARGIN_MIN:
        return None
    if cropped.shape == grid.shape:
        return None
    if overwrite_count < MIN_OUTLINE_OVERWRITE:
        return None

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
