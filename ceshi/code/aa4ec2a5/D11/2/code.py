# Variant of aa4ec2a5
# Dimension: D11 — 对象间的空间关系与分组 (meta)
# Change: 各蓝色对象独立处理，对象间无特殊空间关系约束
#   → 对象的处理结果取决于其与特定参考对象的关系

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous

"""
design_analysis:
  reference_rule:
    reference_object: 唯一面积最大的蓝色 4 连通对象
    grouping: 其余对象按与参考对象的最小 Chebyshev 距离分成近组/远组
    threshold: 距离 <= 4 为近组; 距离 >= 7 为远组
  concrete_behavior:
    input:
      - 黄色背景上放置 4 个蓝色对象
      - 参考对象固定为空心大对象
      - 另外至少有一个近处实心对象、一个近处空心对象、一个远处空心对象
    output:
      - 所有对象都保留红色 8 邻域外轮廓
      - 参考对象和近组对象整体改为青色
      - 若近组对象有洞, 洞填为品红色, 并把洞并入轮廓计算
      - 远组对象保持蓝色; 若其有洞, 洞保持背景黄, 因而会出现红色内轮廓
  behavior_trace:
    original_local_logic:
      - 对每个对象独立判断是否有洞
      - 有洞就一律青色+品红洞, 无洞就保持蓝色
    new_meta_logic:
      - 先找唯一最大对象作为参考
      - 再按“是否靠近参考对象”决定对象是否进入青色组
      - 洞是否被填充, 也由这个分组决定, 不再只看对象自身是否空心
    example:
      - 一个离参考 3 格的实心小块: 原题保持蓝色; 变体会改成青色
      - 一个离参考 8 格的空心小环: 原题会变成青色+品红洞; 变体保持蓝色且保留黄色洞
  complexity_calibration:
    - 分组只保留近/远两档, 避免规则自由度过多
    - 每个样例都同时包含近处实心对象和远处空心对象
    - 不使用颜色 id 或隐藏元数据排序, 只用可见的面积和距离
  implicit_assumption_review:
    - 原版 `for comp in find_components(...)` 仍可复用, 但不能再在循环内独立完成全部决策
    - 原版 `if holes:` 作为唯一着色条件已失效, 必须改成 `if linked_to_reference`
    - 原版对空心对象总是 `solid[hole]=True` 在本变体里会出错:
      远组空心对象若也这样做, 会错误地抹掉应保留的黄色洞和红色内轮廓
    - 原版 `out[outline & (out == bg)] = 2` 仍安全, 因为生成器继续保证对象间隔足够大, 轮廓不互相覆盖
  placement_legality:
    - 生成器保证参考对象面积唯一最大
    - 生成器强制近组距离在 3..4, 远组距离在 7..12, 中间 5..6 的模糊带不出现
    - 任意两对象之间的最小 Chebyshev 距离仍 >= 3, 且离边框 >= 1
  visual_observability:
    - 近处对象无论是否有洞都会变青色, 说明规则不是“有洞才变色”
    - 远处空心对象保留黄色洞并出现红色内轮廓, 使“距离控制洞处理方式”在肉眼上直接可见
    - 参考对象始终最大, 人类可在 2-3 个样例里稳定识别它
  visual_cleanliness:
    - 输出只使用蓝/红/黄/品红/青五色
    - 对象间距大, 不会出现轮廓叠压造成的脏图
  solver_assumptions_after_change:
    - `main()` 假设存在唯一最大对象; 生成器通过面积约束保证
    - `main()` 假设样例里确实存在近/远两组; 生成器通过距离区间保证
    - `main()` 假设背景仍是众数颜色; 生成器继续用统一黄色背景保证
"""


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


def _min_chebyshev_distance(coords_a: np.ndarray, coords_b: np.ndarray) -> int:
    ra = coords_a[:, 0][:, None]
    ca = coords_a[:, 1][:, None]
    rb = coords_b[:, 0][None, :]
    cb = coords_b[:, 1][None, :]
    dist = np.maximum(np.abs(ra - rb), np.abs(ca - cb))
    return int(dist.min())


def _component_key(comp: dict) -> tuple[int, int, int]:
    r0, c0, _r1, _c1 = comp["bbox"]
    return (-int(comp["area"]), int(r0), int(c0))


LINK_DISTANCE = 4


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the largest object as a reference, then recolor nearby objects before outlining.

    Input structure:
    - A uniform background contains several blue 4-connected components.
    - The largest component acts as a visual reference; some components may enclose holes.

    Rule:
    - Draw a one-cell red 8-neighborhood outline around every component.
    - Find the largest component and recolor it, plus every component within Chebyshev distance 4 of it, to teal (8).
    - If one of those near/reference components has enclosed holes, fill the holes with magenta (6) before outlining; farther hollow components keep their background-colored holes.

    Key insight: The task is no longer solved component-by-component. You must first identify
    the reference object, group the remaining objects by proximity to it, and only then decide
    whether a hole is absorbed into the outlined mass or left visible as a background void.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    components = list(find_components(g, bg=bg, connectivity=4, monochromatic=True))
    if not components:
        return out

    reference = sorted(components, key=_component_key)[0]
    ref_coords = reference["coords"]

    for comp in components:
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        holes = _interior_holes(mask, coords)

        linked = comp is reference or _min_chebyshev_distance(coords, ref_coords) <= LINK_DISTANCE
        solid = mask.copy()

        if linked:
            out[mask] = 8
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (22, 27)
BG_COLOR = 4
FG_COLOR = 1

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
NEAR_DISTANCE_RANGE = (3, 4)
FAR_DISTANCE_RANGE = (7, 12)
REFERENCE_EDGE_PAD = 4

GEN_OUTER_RETRIES = 400
PLACEMENT_RETRIES = 800

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 60

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12]
REF_HOLE_AREAS = [4, 6, 8, 9, 12]
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 240
HOLLOW_GEN_RETRIES = 80
HOLE_DIM_MAX = 4

REFERENCE_MIN_AREA = 22
REFERENCE_MAX_AREA = 54
REFERENCE_MAX_SPAN = 11
SECONDARY_HOLLOW_MIN_AREA = 10
SECONDARY_HOLLOW_MAX_AREA = 26
SECONDARY_HOLLOW_MAX_SPAN = 9
SOLID_MIN_AREA = 6
SOLID_MAX_AREA = 18
SOLID_MAX_SPAN = 8
UNIQUE_REFERENCE_GAP = 5


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


def _random_transform(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, k=random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.asarray(out, dtype=bool)


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
    return _random_transform(mask)


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
    if mh <= hh + 1 or mw <= hw + 1:
        return None

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
            return _random_transform(cropped)
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


def _fits_area_and_span(mask: np.ndarray, min_area: int, max_area: int, max_span: int) -> bool:
    area = int(mask.sum())
    return min_area <= area <= max_area and max(mask.shape) <= max_span


def _sample_hollow(min_area: int, max_area: int, max_span: int, hole_areas: list[int]) -> np.ndarray | None:
    for _ in range(HOLLOW_GEN_RETRIES):
        mask = _make_hollow_component(random.choice(hole_areas))
        if mask is not None and _fits_area_and_span(mask, min_area, max_area, max_span):
            return mask
    return None


def _sample_solid(min_area: int, max_area: int, max_span: int) -> np.ndarray | None:
    for _ in range(POLY_GEN_RETRIES):
        mask = _make_polyomino()
        if mask is not None and _fits_area_and_span(mask, min_area, max_area, max_span):
            return mask
    return None


def _translate(coords: np.ndarray, r: int, c: int) -> np.ndarray:
    return coords + np.array([r, c], dtype=int)


def _random_pos(limit: int, size: int, edge_pad: int = 0) -> tuple[int, int]:
    lo = BORDER_MIN
    hi = limit - size - BORDER_MIN
    if hi < lo:
        raise ValueError("object does not fit in the grid")

    padded_lo = min(max(lo, BORDER_MIN + edge_pad), hi)
    padded_hi = max(min(hi, limit - size - BORDER_MIN - edge_pad), padded_lo)
    return padded_lo, padded_hi


def _place_mask(
    H: int,
    W: int,
    mask: np.ndarray,
    placed: list[dict],
    ref_coords: np.ndarray | None = None,
    dist_range: tuple[int, int] | None = None,
    edge_pad: int = 0,
) -> dict | None:
    mh, mw = mask.shape
    local_coords = np.argwhere(mask)
    r_lo, r_hi = _random_pos(H, mh, edge_pad=edge_pad)
    c_lo, c_hi = _random_pos(W, mw, edge_pad=edge_pad)

    for _ in range(PLACEMENT_RETRIES):
        r = random.randint(r_lo, r_hi)
        c = random.randint(c_lo, c_hi)
        world_coords = _translate(local_coords, r, c)

        if dist_range is not None:
            d_ref = _min_chebyshev_distance(world_coords, ref_coords)
            if not (dist_range[0] <= d_ref <= dist_range[1]):
                continue

        conflict = False
        for other in placed:
            if _min_chebyshev_distance(world_coords, other["coords"]) < CHEBYSHEV_MIN:
                conflict = True
                break
        if conflict:
            continue

        return {"r": r, "c": c, "mask": mask, "coords": world_coords}
    return None


def _make_components() -> list[np.ndarray] | None:
    reference = _sample_hollow(REFERENCE_MIN_AREA, REFERENCE_MAX_AREA, REFERENCE_MAX_SPAN, REF_HOLE_AREAS)
    if reference is None:
        return None
    ref_area = int(reference.sum())

    near_hollow = _sample_hollow(
        SECONDARY_HOLLOW_MIN_AREA,
        min(SECONDARY_HOLLOW_MAX_AREA, ref_area - UNIQUE_REFERENCE_GAP),
        SECONDARY_HOLLOW_MAX_SPAN,
        HOLE_VALID_AREAS,
    )
    if near_hollow is None:
        return None

    far_hollow = _sample_hollow(
        SECONDARY_HOLLOW_MIN_AREA,
        min(SECONDARY_HOLLOW_MAX_AREA, ref_area - UNIQUE_REFERENCE_GAP),
        SECONDARY_HOLLOW_MAX_SPAN,
        HOLE_VALID_AREAS,
    )
    if far_hollow is None:
        return None

    near_solid = _sample_solid(
        SOLID_MIN_AREA,
        min(SOLID_MAX_AREA, ref_area - UNIQUE_REFERENCE_GAP),
        SOLID_MAX_SPAN,
    )
    if near_solid is None:
        return None

    others = [near_hollow, far_hollow, near_solid]
    if any(int(mask.sum()) >= ref_area for mask in others):
        return None
    return [reference, near_hollow, near_solid, far_hollow]


def _variant_signature(grid: np.ndarray) -> dict[str, int]:
    comps = list(find_components(grid, bg=BG_COLOR, connectivity=4, monochromatic=True))
    reference = sorted(comps, key=_component_key)[0]
    ref_coords = reference["coords"]

    near = 0
    far = 0
    hollow = 0
    for comp in comps:
        mask = _coords_mask(grid.shape, comp["coords"])
        holes = _interior_holes(mask, comp["coords"])
        if holes:
            hollow += 1
        if comp is reference:
            continue
        d = _min_chebyshev_distance(comp["coords"], ref_coords)
        if d <= LINK_DISTANCE:
            near += 1
        else:
            far += 1
    return {"near": near, "far": far, "hollow": hollow}


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    components = _make_components()
    if components is None:
        return None
    reference_mask, near_hollow_mask, near_solid_mask, far_hollow_mask = components

    placed: list[dict] = []

    ref = _place_mask(H, W, reference_mask, placed, edge_pad=REFERENCE_EDGE_PAD)
    if ref is None:
        return None
    placed.append(ref)

    near_hollow = _place_mask(
        H,
        W,
        near_hollow_mask,
        placed,
        ref_coords=ref["coords"],
        dist_range=NEAR_DISTANCE_RANGE,
    )
    if near_hollow is None:
        return None
    placed.append(near_hollow)

    near_solid = _place_mask(
        H,
        W,
        near_solid_mask,
        placed,
        ref_coords=ref["coords"],
        dist_range=NEAR_DISTANCE_RANGE,
    )
    if near_solid is None:
        return None
    placed.append(near_solid)

    far_hollow = _place_mask(
        H,
        W,
        far_hollow_mask,
        placed,
        ref_coords=ref["coords"],
        dist_range=FAR_DISTANCE_RANGE,
    )
    if far_hollow is None:
        return None
    placed.append(far_hollow)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for item in placed:
        coords = item["coords"]
        grid[coords[:, 0], coords[:, 1]] = FG_COLOR

    signature = _variant_signature(grid)
    if signature["near"] != 2 or signature["far"] != 1 or signature["hollow"] < 3:
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
