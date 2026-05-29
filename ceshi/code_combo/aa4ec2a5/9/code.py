# Combo variant of aa4ec2a5 (index 9)
# combo_spec: D7/0+D9/0
# Name: 按面积排序处理+轮廓覆盖一切

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

# Design analysis for combo D7/0 + D9/0
# --------------------------------------
# a) 具体化与行为追踪:
#   - 原题: 先统一检测所有对象的洞并做条件重着色, 然后只在背景像素上补红色外轮廓.
#   - 本融合版: 先按对象面积从大到小排序, 然后逐个对象执行
#       1. 恢复/重着色当前对象本体
#       2. 若有洞, 把对象染成青色、洞染成品红, 并把洞并入用于算轮廓的 solid mask
#       3. 直接把该对象的 8 邻域外轮廓写成红色, 不再要求目标格必须是背景
#   - 小例子: 两个蓝对象只在对角上相邻. 原题下双方本体都保持蓝/青, 只在外围黄底上见红.
#     本题下, 大对象先画轮廓时会暂时把小对象角点刷红; 小对象轮到时先恢复自己,
#     再把大对象的对应角点刷红. 最终红色落在大对象边缘上, 这同时体现了面积顺序和覆盖优先级.
# b) 复杂度校准:
#   - 仍保留 3-4 个对象、1-2 个中空对象, 颜色规则不增加新类别, 只改变处理顺序和覆盖关系.
#   - 为了让人眼能看出顺序, 生成器强制面积互异, 并让前两大对象形成一次对角接触.
#   - 为了避免过乱, 其余对象与任何已放对象都保持 Chebyshev 距离至少 2.
# c) 隐式假设审查:
#   - 原 `out[outline & (out == bg)] = 2` 依赖“轮廓永不覆盖已有对象”; 融合后已失效, 必须改为无条件覆写.
#   - 原代码默认实体对象不会被别的轮廓破坏, 所以不用重画本体; 融合后每个对象轮到时都必须先恢复自己.
#   - 组件检测仍必须从原始输入 `g` 提取, 不能从正在被覆写的 `out` 提取, 否则次序会污染识别结果.
#   - 轮廓定义本身不变: 还是“填洞后的 solid mask 的外一圈”, 不是把对象自身也纳入 outline.
# d) 放置合法性:
#   - 所有输入对象仍然不能重叠、不能 4 邻接, 否则输入的蓝对象组件数会变.
#   - 生成器只允许指定的一对前两大对象出现对角接触; 其他对象保持 min Chebyshev >= 2.
#   - 所有对象仍离边界至少 1 格, 因而外轮廓可安全贴到边缘但不会越界.
# e) 视觉可观察性:
#   - 前两大对象的面积差被强制拉开, 人类可直接从大小判断处理顺序.
#   - 生成器拒绝与“面积降序但只画背景轮廓”相同的样本, 保证 D9 的覆盖效应真的出现.
#   - 生成器也拒绝与“面积升序处理”相同的样本, 保证 D7 的顺序效应真的出现.
# f) 可视化整洁度:
#   - 只制造一处关键对角接触, 其余对象远离, 从而让红色侵蚀局部可见但不会把全图刷乱.
#   - 洞对象仍只用青/品红这一组固定颜色, 不再引入额外编码负担.
# g) 变化的可推断性:
#   - 颜色映射仍然高度结构化: 中空对象 -> 青, 洞 -> 品红, 所有轮廓 -> 红.
#   - 新信息只体现在“哪个对象后处理”和“红色可以压到对象像素上”, 可通过 2-3 个样例归纳.
# h) main() 的新隐式假设与生成器保证:
#   - 假设存在可见的面积排序依据: 生成器强制面积互异.
#   - 假设至少有一处跨对象 outline 覆写: 生成器强制前两大对象对角接触并做差异验证.
#   - 假设输入组件仍可按 4 连通独立识别: 生成器禁止重叠和正交接触.


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


def _component_records(grid: np.ndarray) -> tuple[np.ndarray, int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps: list[dict[str, object]] = []

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)
        solid = mask.copy()
        for hole in holes:
            solid[hole[:, 0], hole[:, 1]] = True
        comps.append(
            {
                "coords": coords,
                "mask": mask,
                "solid": solid,
                "holes": holes,
                "area": int(coords.shape[0]),
                "bbox": bbox_coords(coords),
                "color": int(comp["color"]),
            }
        )

    return g, int(bg), comps


def _solve_from_records(
    g: np.ndarray,
    bg: int,
    comps: list[dict[str, object]],
    *,
    descending: bool,
    cover_all: bool,
) -> np.ndarray:
    out = g.copy()
    ordered = list(comps)

    if descending:
        ordered.sort(key=lambda comp: (-int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))
    else:
        ordered.sort(key=lambda comp: (int(comp["area"]), comp["bbox"][0], comp["bbox"][1]))

    for comp in ordered:
        mask = comp["mask"]
        solid = comp["solid"]
        holes = comp["holes"]

        out[mask] = 8 if holes else int(comp["color"])
        for hole in holes:
            out[hole[:, 0], hole[:, 1]] = 6

        outline = _outline_mask(solid)
        if cover_all:
            out[outline] = 2
        else:
            out[outline & (out == bg)] = 2

    return out


def _solve_variant(
    grid: np.ndarray,
    *,
    descending: bool,
    cover_all: bool,
) -> np.ndarray:
    g, bg, comps = _component_records(grid)
    return _solve_from_records(g, bg, comps, descending=descending, cover_all=cover_all)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor hollow objects and draw overwriting outlines in descending area order.

    Input structure:
    - A uniform yellow background contains 3-4 blue 4-connected components.
    - Some components are hollow and enclose 4-connected background holes fully inside
      their own bounding boxes.

    Rule:
    - Sort the blue components from largest area to smallest area.
    - Process one component at a time: first restore its body, then recolor hollow ones
      to teal with magenta holes, and finally paint its one-pixel 8-neighborhood outline.
    - The outline write is unconditional: red can overwrite background, object pixels,
      and hole pixels that already exist on the current canvas.

    Key insight: The geometry used to detect holes and outlines is still taken from the
    untouched input components, but the output is built sequentially, so later smaller
    components can visibly cut red notches into earlier larger components.
    """
    return _solve_variant(grid, descending=True, cover_all=True)


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
SMALL_SHAPE_SIZE_RANGE = (2, 4)

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

BORDER_MIN = 1
PLACEMENT_RETRIES = 350
GEN_OUTER_RETRIES = 400
MIN_AREA_GAP_TOP2 = 3


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


def _placed_coords(mask: np.ndarray, r: int, c: int) -> np.ndarray:
    coords = np.argwhere(mask)
    placed = coords.copy()
    placed[:, 0] += int(r)
    placed[:, 1] += int(c)
    return placed


def _pair_stats(coords_a: np.ndarray, coords_b: np.ndarray) -> tuple[bool, bool, bool, int]:
    has_overlap = False
    has_orth = False
    has_diag = False
    min_cheb = 10**9

    for ra, ca in coords_a:
        for rb, cb in coords_b:
            dr = abs(int(ra) - int(rb))
            dc = abs(int(ca) - int(cb))
            cheb = max(dr, dc)
            min_cheb = min(min_cheb, cheb)
            if dr == 0 and dc == 0:
                has_overlap = True
            elif dr + dc == 1:
                has_orth = True
            elif dr == 1 and dc == 1:
                has_diag = True

    return has_overlap, has_orth, has_diag, int(min_cheb)


def _can_place_regular(
    coords: np.ndarray,
    existing: list[tuple[int, int, np.ndarray]],
) -> bool:
    for _, _, other_coords in existing:
        has_overlap, has_orth, _, min_cheb = _pair_stats(coords, other_coords)
        if has_overlap or has_orth or min_cheb < 2:
            return False
    return True


def _can_place_designated(
    coords: np.ndarray,
    existing: list[tuple[int, int, np.ndarray]],
    contact_with_idx: int,
) -> bool:
    saw_required_diag = False

    for idx, (_, _, other_coords) in enumerate(existing):
        has_overlap, has_orth, has_diag, min_cheb = _pair_stats(coords, other_coords)
        if has_overlap or has_orth:
            return False
        if idx == contact_with_idx:
            if not has_diag or min_cheb != 1:
                return False
            saw_required_diag = True
        elif min_cheb < 2:
            return False

    return saw_required_diag


def _place_components(H: int, W: int, components: list[np.ndarray]) -> list[tuple[int, int, np.ndarray]] | None:
    order = sorted(range(len(components)), key=lambda i: -int(components[i].sum()))
    second_idx = order[1]

    placed: list[tuple[int, int, np.ndarray]] = []

    for slot, idx in enumerate(order):
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
            coords = _placed_coords(mask, r, c)
            if slot == 0:
                placed.append((r, c, coords))
                placed_ok = True
                break
            if idx == second_idx:
                if not _can_place_designated(coords, placed, contact_with_idx=0):
                    continue
            else:
                if not _can_place_regular(coords, placed):
                    continue
            placed.append((r, c, coords))
            placed_ok = True
            break

        if not placed_ok:
            return None

    return [(r, c, components[idx]) for idx, (r, c, _coords) in zip(order, placed)]


def _diag_contact_exists(placed: list[tuple[int, int, np.ndarray]]) -> bool:
    coords_list = [_placed_coords(mask, r, c) for r, c, mask in placed]
    for i in range(len(coords_list)):
        for j in range(i + 1, len(coords_list)):
            _, has_orth, has_diag, _ = _pair_stats(coords_list[i], coords_list[j])
            if has_diag and not has_orth:
                return True
    return False


def _fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    g, bg, comps = _component_records(grid)
    out = _solve_from_records(g, bg, comps, descending=True, cover_all=True)
    bg_only = _solve_from_records(g, bg, comps, descending=True, cover_all=False)
    asc_cover_all = _solve_from_records(g, bg, comps, descending=False, cover_all=True)
    return {
        "red_on_input_fg": int(np.count_nonzero((out == 2) & (g != bg))),
        "diff_vs_bg_only": int(np.count_nonzero(out != bg_only)),
        "diff_vs_asc": int(np.count_nonzero(out != asc_cover_all)),
    }


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

    areas = sorted((int(mask.sum()) for mask in components), reverse=True)
    if len(set(areas)) != len(areas):
        return None
    if areas[0] - areas[1] < MIN_AREA_GAP_TOP2:
        return None

    placed = _place_components(H, W, components)
    if placed is None or not _diag_contact_exists(placed):
        return None

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for r, c, mask in placed:
        rows, cols = np.where(mask)
        grid[rows + r, cols + c] = FG_COLOR

    try:
        metrics = _fusion_metrics(grid)
    except Exception:
        return None

    if metrics["red_on_input_fg"] <= 0:
        return None
    if metrics["diff_vs_bg_only"] <= 0:
        return None
    if metrics["diff_vs_asc"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
