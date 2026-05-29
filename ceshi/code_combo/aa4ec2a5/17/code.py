# Combo variant of aa4ec2a5 (index 17)
# combo_spec: D1/3+D9/2
# Name: 轮廓仅绘制在背景侧+轮廓避让洞的区域

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# ---------------------------------------------------------------------------
# 设计分析（融合版：D1/3 + D9/2）
# ---------------------------------------------------------------------------
# a) 具体化与行为追踪
# - 原题：每个蓝色 4 连通对象都得到一圈 8 邻域红色轮廓；若对象有洞，则先把对象改成
#   青色、洞改成品红，再把洞视作“填实”后的一部分来求外轮廓。
# - D1/3 融合后，红轮廓只允许落在输入原本就是背景色的格子上。如果另一个对象刚好占了
#   这个候选格，红色就不会画进去，所以相邻对象之间会出现“少一格轮廓”的缺口。
# - D9/2 融合后，轮廓还要避开洞的 8 邻域。这里必须保留“先填洞再取外轮廓”的原题骨架，
#   但 generator 需要专门构造一种空心对象：洞旁边的外壳角点被挖掉一个格子。这个角缺口
#   会成为外轮廓候选，同时又与洞对角相邻，因此会被“洞 8 邻域”规则抑制。
# - 小例子：若局部形状是
#       0 1 1
#       1 0 1
#       1 1 1
#   其中 0 是背景，中心 0 是洞，左上 0 是外壳缺角。把洞填实后，左上角仍是外轮廓候选；
#   但它又处在洞的 8 邻域里，所以最终不会变红。若对象另一侧再与一个小实心对象形成对角
#   接触，则该侧也会因为“不是背景”而不画红色。这样同一个空心对象同时出现两类缺口。
#
# b) 复杂度校准
# - 若继续使用原题那种“对象间 Chebyshev 距离 >= 3”的放置，D1/3 会退化成原题。
#   因此本生成器强制至少一对对象形成纯对角接触，但仍保持 4 不连通，避免视觉混乱。
# - 若空心对象只是普通矩形环，洞 8 邻域根本碰不到外轮廓，D9/2 会退化成原题。因此至少
#   一个空心对象必须使用“洞角旁缺一角”的特制形状，并且在样本验收时显式检查确实存在被
#   洞邻域抑制的外轮廓候选。
# - 除这一特制对象外，其他对象仍沿用原题的多矩形 polyomino / 常规空心对象，避免把题目
#   简化成单一模板，同时又不引入过多形态自由度。
#
# c) 原 main() 的隐式假设审查
# - 原 `out[outline & (out == bg)] = 2` 把“能否画轮廓”绑定在已修改的输出缓冲区上。对象
#   一旦允许近邻，这个判断就不够精确，因为它混合了输入占用和前面对象已经画出的红色。
#   融合后必须改成显式使用 `bg_mask = (g == bg)`，含义是“只在输入原本为背景的位置画”。
# - 原 solver 默认对象彼此隔得很开，所以“轮廓候选落到别的对象上”这种情况不会发生；现在
#   生成器必须主动制造这种情况，并且 solver 不能把这些占位格误画成红色。
# - 原 solver 对空心对象的做法是先把洞填入 `solid` 再求外轮廓。这个前提在融合后仍保留，
#   否则会把洞内部当成可画轮廓区域，反而破坏“洞抑制外轮廓缺口”的目标。
# - 新增前提是：若某个对象存在洞，则“洞 8 邻域”需要单独算出并从外轮廓候选中扣掉。这个
#   缓冲区必须基于原洞像素，而不是基于已经填实后的 `solid`。
#
# d) 放置合法性
# - 所有组件仍要求不重叠、不越界。
# - 生成器固定先放一个特制空心对象，再放一个与它纯对角接触的小实心对象；其余对象必须与
#   已放对象在 8 邻域意义下完全分离，避免出现难以分辨的多重接触。
# - 特制空心对象的“洞角缺口”不能被近邻对象直接占掉，否则 D9/2 的缺口会被 D1/3 的占位
#   缺口遮蔽。放置时会显式避开这一个关键格。
#
# e) 视觉可观察性
# - 每个样本都至少包含一个“双重缺口”空心对象：同一对象既有因近邻对象而少掉的轮廓位置，
#   也有因洞 8 邻域而少掉的轮廓位置。解题者能在 2-3 个样例里反复看到这两类缺口共存。
# - 颜色种类仍只有黄底、蓝实心、青空心、品红洞、红轮廓五种，结构线索稳定，没有额外的
#   隐式编码。
#
# f) 可视化整洁度
# - 只有一对对象被允许近邻，其他对象仍保持明显间隔，避免整张图到处都是红轮廓缺口。
# - 特制空心对象只缺一个角点，形状规则、易读，不会让“洞缺口”和“对象本体”混在一起。
#
# g) main() 的新隐式假设与 generator 的对应保证
# - 假设 1：至少一个空心对象存在真实洞。generator 强制构造。
# - 假设 2：至少一个空心对象的外轮廓候选与洞 8 邻域相交。generator 用特制缺角环并在验收
#   指标里检查 `hole_suppressed > 0`。
# - 假设 3：至少一个对象对之间存在纯对角接触。generator 强制 special hollow 与 companion
#   构成 close pair，并检查 `blocked_by_other > 0`。
# - 假设 4：这两类缺口不会互相覆盖成同一个格。generator 禁止 companion 占用特制缺角格。


ORTH_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


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


def _neighbor_mask(shape: tuple[int, int], coords: np.ndarray, *, include_center: bool) -> np.ndarray:
    h, w = shape
    out = np.zeros(shape, dtype=bool)
    rows = coords[:, 0]
    cols = coords[:, 1]
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if not include_center and dr == 0 and dc == 0:
                continue
            rr = rows + dr
            cc = cols + dc
            ok = (0 <= rr) & (rr < h) & (0 <= cc) & (cc < w)
            out[rr[ok], cc[ok]] = True
    return out


def _outline_candidates(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    outline = _neighbor_mask(mask.shape, coords, include_center=False)
    outline &= ~mask
    return outline


def _hole_buffer_mask(shape: tuple[int, int], holes: list[np.ndarray]) -> np.ndarray:
    if not holes:
        return np.zeros(shape, dtype=bool)
    merged = np.concatenate(holes, axis=0)
    return _neighbor_mask(shape, merged, include_center=True)


def _filled_component_and_holes(mask: np.ndarray, coords: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
    holes = _interior_holes(mask, coords)
    solid = mask.copy()
    for hole in holes:
        solid[hole[:, 0], hole[:, 1]] = True
    return solid, holes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and draw only the background-side outline, with hole-neighborhood gaps.

    Input structure:
    - A yellow background contains 3-4 blue 4-connected components.
    - One or two components are hollow: they enclose 4-connected background holes.
    - Components may now sit in pure diagonal contact, so a would-be outline pixel can already
      be occupied by another object instead of by background.

    Rule:
    - Process each foreground component independently.
    - If a component has enclosed holes, recolor its body to teal (8) and those holes to magenta (6).
    - Compute the one-pixel 8-neighborhood outer outline of the hole-filled component, but paint it
      red (2) only on cells that were background in the input.
    - For hollow components, remove any such outline candidate that lies inside the 8-neighborhood
      of the hole pixels; those positions stay non-red and create visible gaps.

    Key insight:
    - Two separate suppression rules interact: occupied cells block the outline from the outside,
      while hole neighborhoods carve missing segments out of the same outer outline. Hollow objects
      therefore need both local hole geometry and surrounding object layout to be understood.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bg_mask = g == bg
    out = g.copy()

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        solid, holes = _filled_component_and_holes(mask, coords)

        if holes:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6

        outline = _outline_candidates(solid) & bg_mask
        if holes:
            outline &= ~_hole_buffer_mask(g.shape, holes)

        out[outline] = 2

    return out


# --------------- generator (c) ---------------

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

SMALL_SHAPE_SIZE_RANGE = (1, 4)
SMALL_SHAPE_PROB = 0.2

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12, 16]
SPECIAL_HOLE_AREAS = [area for area in HOLE_VALID_AREAS if area != 2]
HOLE_DIM_MAX = 4
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 200
HOLLOW_GEN_RETRIES = 50
HOLLOW_MIN_AREA = 10

BORDER_MIN = 1
PLACEMENT_RETRIES = 300
PAIR_PLACEMENT_RETRIES = 180

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


def _make_special_hollow_component() -> np.ndarray:
    dims = _rect_dims(random.choice(SPECIAL_HOLE_AREAS))
    assert dims is not None
    hh, hw = dims

    mask = np.ones((hh + 2, hw + 2), dtype=bool)
    mask[1 : 1 + hh, 1 : 1 + hw] = False

    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        mask[0, 0] = False
    elif corner == "tr":
        mask[0, -1] = False
    elif corner == "bl":
        mask[-1, 0] = False
    else:
        mask[-1, -1] = False

    assert is_contiguous(mask.astype(int), background=0, connectivity=4)
    assert _has_interior_hole(mask)
    assert int(mask.sum()) >= HOLLOW_MIN_AREA
    return mask


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


def _absolute_cells(mask: np.ndarray, r: int, c: int) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.where(mask)
    return rows + r, cols + c


def _cell_set(rows: np.ndarray, cols: np.ndarray) -> set[tuple[int, int]]:
    return set(zip(rows.tolist(), cols.tolist()))


def _contact_profile(
    rows_a: np.ndarray, cols_a: np.ndarray, rows_b: np.ndarray, cols_b: np.ndarray
) -> tuple[bool, bool, bool]:
    bset = _cell_set(rows_b, cols_b)
    overlap = False
    orth = False
    diag = False
    for ra, ca in zip(rows_a.tolist(), cols_a.tolist()):
        if (ra, ca) in bset:
            overlap = True
            break
        for dr, dc in ORTH_DIRS:
            if (ra + dr, ca + dc) in bset:
                orth = True
        for dr, dc in DIAG_DIRS:
            if (ra + dr, ca + dc) in bset:
                diag = True
    return overlap, orth, diag


def _is_separated_from_placed(
    rows: np.ndarray, cols: np.ndarray, placed: list[dict[str, object]]
) -> bool:
    for item in placed:
        overlap, orth, diag = _contact_profile(
            rows, cols, item["rows"], item["cols"]  # type: ignore[arg-type]
        )
        if overlap or orth or diag:
            return False
    return True


def _can_place_close_pair(
    rows: np.ndarray,
    cols: np.ndarray,
    anchor: dict[str, object],
    forbidden: set[tuple[int, int]],
) -> bool:
    overlap, orth, diag = _contact_profile(
        rows, cols, anchor["rows"], anchor["cols"]  # type: ignore[arg-type]
    )
    if overlap or orth or not diag:
        return False
    return _cell_set(rows, cols).isdisjoint(forbidden)


def _random_origin(H: int, W: int, mh: int, mw: int) -> tuple[int, int] | None:
    rmax = H - mh - BORDER_MIN
    cmax = W - mw - BORDER_MIN
    if rmax < BORDER_MIN or cmax < BORDER_MIN:
        return None
    return random.randint(BORDER_MIN, rmax), random.randint(BORDER_MIN, cmax)


def _append_placement(
    placed: list[dict[str, object]], idx: int, mask: np.ndarray, r: int, c: int
) -> None:
    rows, cols = _absolute_cells(mask, r, c)
    placed.append({"idx": idx, "mask": mask, "r": r, "c": c, "rows": rows, "cols": cols})


def _local_hole_gap_cells(mask: np.ndarray) -> set[tuple[int, int]]:
    coords = np.argwhere(mask)
    holes = _interior_holes(mask, coords)
    solid = mask.copy()
    for hole in holes:
        solid[hole[:, 0], hole[:, 1]] = True
    outline = _outline_candidates(solid) & (~mask)
    gap = outline & _hole_buffer_mask(mask.shape, holes)
    return {(int(r), int(c)) for r, c in np.argwhere(gap)}


def _expand_points(points: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in points:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((int(r + dr), int(c + dc)))
    return expanded


def _variant_signal_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    bg_mask = g == bg
    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True)

    blocked_outline_cells = 0
    hole_suppressed_cells = 0
    dual_gap_hollows = 0
    close_pairs = 0

    comp_masks: list[np.ndarray] = []
    comp_rows_cols: list[tuple[np.ndarray, np.ndarray]] = []

    for comp in comps:
        mask = _coords_mask(g.shape, comp["coords"])
        solid, holes = _filled_component_and_holes(mask, comp["coords"])
        candidate = _outline_candidates(solid)

        blocked = candidate & (~bg_mask)
        blocked_outline_cells += int(np.sum(blocked))

        suppressed = np.zeros_like(candidate, dtype=bool)
        if holes:
            suppressed = candidate & bg_mask & _hole_buffer_mask(g.shape, holes)
            hole_suppressed_cells += int(np.sum(suppressed))
            if int(np.sum(blocked)) > 0 and int(np.sum(suppressed)) > 0:
                dual_gap_hollows += 1

        comp_masks.append(mask)
        comp_rows_cols.append(np.where(mask))

    for i, (rows_i, cols_i) in enumerate(comp_rows_cols):
        for rows_j, cols_j in comp_rows_cols[i + 1 :]:
            overlap, orth, diag = _contact_profile(rows_i, cols_i, rows_j, cols_j)
            if not overlap and not orth and diag:
                close_pairs += 1

    return {
        "blocked_outline_cells": int(blocked_outline_cells),
        "hole_suppressed_cells": int(hole_suppressed_cells),
        "dual_gap_hollows": int(dual_gap_hollows),
        "close_pairs": int(close_pairs),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    num_components = random.randint(*COMP_COUNT_RANGE)
    num_hollow = random.randint(
        HOLLOW_COUNT_RANGE[0], min(HOLLOW_COUNT_RANGE[1], num_components - 1)
    )

    components: list[np.ndarray] = []
    components.append(_make_special_hollow_component())
    components.append(_make_small_shape())

    remaining_hollows = num_hollow - 1
    remaining_total = num_components - 2

    for i in range(remaining_total):
        if i < remaining_hollows:
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
    placed: list[dict[str, object]] = []

    special_mask = components[0]
    companion_mask = components[1]
    special_gap_cells = _local_hole_gap_cells(special_mask)
    if not special_gap_cells:
        return None

    pair_ok = False
    reserved_outline_guard: set[tuple[int, int]] = set()
    for _ in range(PAIR_PLACEMENT_RETRIES):
        first_origin = _random_origin(H, W, special_mask.shape[0], special_mask.shape[1])
        if first_origin is None:
            return None
        r1, c1 = first_origin
        placed.clear()
        _append_placement(placed, 0, special_mask, r1, c1)

        forbidden_abs = {(r1 + r, c1 + c) for r, c in special_gap_cells}
        forbidden_buffer = _expand_points(forbidden_abs)
        for _ in range(PLACEMENT_RETRIES):
            second_origin = _random_origin(H, W, companion_mask.shape[0], companion_mask.shape[1])
            if second_origin is None:
                return None
            r2, c2 = second_origin
            rows2, cols2 = _absolute_cells(companion_mask, r2, c2)
            if _can_place_close_pair(rows2, cols2, placed[0], forbidden_buffer):
                _append_placement(placed, 1, companion_mask, r2, c2)
                reserved_outline_guard = forbidden_buffer
                pair_ok = True
                break
        if pair_ok:
            break

    if not pair_ok:
        return None

    remaining = sorted(range(2, num_components), key=lambda i: -int(components[i].sum()))
    for idx in remaining:
        mask = components[idx]
        placed_ok = False
        for _ in range(PLACEMENT_RETRIES):
            origin = _random_origin(H, W, mask.shape[0], mask.shape[1])
            if origin is None:
                return None
            r, c = origin
            rows, cols = _absolute_cells(mask, r, c)
            if _is_separated_from_placed(rows, cols, placed) and _cell_set(
                rows, cols
            ).isdisjoint(reserved_outline_guard):
                _append_placement(placed, idx, mask, r, c)
                placed_ok = True
                break
        if not placed_ok:
            return None

    for item in placed:
        rows = item["rows"]  # type: ignore[assignment]
        cols = item["cols"]  # type: ignore[assignment]
        grid[rows, cols] = FG_COLOR

    metrics = _variant_signal_metrics(grid)
    if (
        metrics["close_pairs"] < 1
        or metrics["blocked_outline_cells"] < 1
        or metrics["hole_suppressed_cells"] < 1
        or metrics["dual_gap_hollows"] < 1
    ):
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
