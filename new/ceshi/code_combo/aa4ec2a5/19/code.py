# Combo variant of aa4ec2a5 (index 19)
# combo_spec: D8/1+D11/2
# Name: 着色由位置决定+处理取决于参考对象

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, find_components, get_bg, is_contiguous


# Design analysis
# a) Concrete behavior and trace:
#    原题里每个对象都能独立处理: 只要对象有洞, 就必定变成青色并把洞填成品红。
#    融合后必须先做全局判断。先找唯一最大的蓝色对象作为参考对象, 再看其他对象到它的最小
#    Chebyshev 距离。只有“参考对象本身 + 距参考不超过 4 的空心对象”才会进入重着色分支。
#    进入分支后, 还不能直接固定成青色, 而是要继续看该对象质心位于网格左半还是右半:
#    左侧变青色(8), 右侧变绿色(3), 洞统一变品红(6)。远处空心对象则保持蓝体黄洞, 因为
#    洞未被吸收进实体, 所以其内边界也会出现红色轮廓。例子: 原题里“近处右侧空心环”和
#    “远处右侧空心环”都会变成青色+品红洞; 本变体里前者会变成绿色+品红洞, 后者则仍是
#    蓝色且保留黄色洞。两者同在右半边却输出不同, 说明位置和参考距离同时参与了决策。
# b) Complexity calibration:
#    规则只保留少量离散档位, 避免自由度过多:
#    - 参考对象唯一, 由“最大面积”可见特征确定
#    - 距离只分成激活/未激活两档
#    - 激活后的颜色只分左青/右绿两档
#    生成器固定 4 个对象: 参考空心、近处空心、近处实心、远处空心。这样每个样例都同时
#    暴露两种激活颜色和一种未激活空心, 视觉证据足够稳定。
# c) Implicit-assumption review:
#    - 原版在 `for comp in find_components(...)` 循环里就能独立完成全部决策; 融合后不行,
#      因为是否进入重着色分支取决于“与参考对象的关系”。因此 main() 先整体收集组件。
#    - 原版 `if holes:` 即可决定重着色; 融合后必须改成 `if holes and linked_to_reference:`.
#    - 原版对所有空心对象都把洞并入 `solid` 用于描边; 融合后只能对激活空心对象这样做,
#      否则会错误抹掉远处空心对象应保留的黄色洞与红色内轮廓。
#    - 原版 `out[outline & (out == bg)] = 2` 仍安全, 前提是生成器继续保证对象间距充足。
# d) Placement legality:
#    生成器继续保证: 对象 4 连通、互不重叠、与边界至少留 1 格、任意两对象最小 Chebyshev
#    距离至少为 3。额外约束是参考对象唯一最大, 且两个激活空心对象分居左右两半。
# e) Visual observability:
#    人类解题者在 2-4 个样例中应能看到:
#    - 最大空心对象总是关键参考对象
#    - 参考附近的两个空心对象分别在左/右半边, 输出中分别变青/绿
#    - 远处空心对象即使也在某一半边, 仍保持蓝体黄洞
#    这样同侧对象可因距离不同而分化, 不会误读成“只看左右位置”。
# f) Visual cleanliness:
#    输出只使用黄/蓝/红/品红/青/绿六色, 且对象间有间隔, 红轮廓不会彼此覆盖。每个区域只承担
#    一种语义角色, 不引入额外标记。
# g) Color-rule inferability:
#    颜色映射只在激活空心对象上使用, 且仅有“左青右绿”两档, 洞始终品红。远处空心对象保留
#    蓝色, 从而让解题者看出“是否先被参考对象激活”是位置着色之前的前置条件。
# h) New solver assumptions and generator guarantees:
#    main() 新增的关键假设是: 存在唯一最大对象, 且样例里同时出现左右两侧的激活空心对象与
#    至少一个未激活空心对象。generate_input() 通过面积间隔、近远距离区间、左右侧约束和
#    末端签名校验来保证这些假设成立。


LINK_DISTANCE = 4


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
    return int(np.maximum(np.abs(ra - rb), np.abs(ca - cb)).min())


def _component_key(record: dict[str, object]) -> tuple[int, int, int]:
    coords = np.asarray(record["coords"], dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    return (-int(coords.shape[0]), int(r0), int(c0))


def _component_records(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    records: list[dict[str, object]] = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        coords = comp["coords"]
        mask = _coords_mask(g.shape, coords)
        records.append(
            {
                "coords": coords,
                "mask": mask,
                "holes": _interior_holes(mask, coords),
                "centroid_col": float(coords[:, 1].mean()),
            }
        )
    return bg, records


def _reference_index(records: list[dict[str, object]]) -> int:
    return min(range(len(records)), key=lambda i: _component_key(records[i]))


def _side_color(centroid_col: float, grid_w: int) -> int:
    return 8 if centroid_col < (grid_w / 2.0) else 3


def main(grid: np.ndarray) -> np.ndarray:
    """
    Outline every object, but only reference-linked hollow ones use the side-based recolor rule.

    Input structure:
    - A uniform background contains several blue 4-connected components.
    - The unique largest component is the visual reference, and multiple components may enclose holes.

    Rule:
    - Draw a one-cell red 8-neighborhood outline around every component.
    - Identify the largest component as the reference object.
    - A hollow component is activated only if it is the reference itself or lies within Chebyshev distance 4 of the reference.
    - Activated hollow components fill their holes with magenta, and their body color is chosen by centroid position:
      left half of the grid becomes teal, right half becomes green.
    - Solid components and hollow components farther from the reference stay blue; unactivated holes remain background.

    Key insight:
    - The recolor decision is now two-stage: first find which hollow objects are linked to the reference object,
      then use each linked hollow object's left/right position to choose teal versus green.
    """
    g = np.asarray(grid, dtype=int)
    bg, records = _component_records(g)
    out = g.copy()
    if not records:
        return out

    ref_idx = _reference_index(records)
    ref_coords = np.asarray(records[ref_idx]["coords"], dtype=int)

    for idx, record in enumerate(records):
        coords = np.asarray(record["coords"], dtype=int)
        mask = np.asarray(record["mask"], dtype=bool)
        holes = list(record["holes"])
        linked = idx == ref_idx or _min_chebyshev_distance(coords, ref_coords) <= LINK_DISTANCE

        solid = mask.copy()
        if linked and holes:
            out[mask] = _side_color(float(record["centroid_col"]), g.shape[1])
            for hole in holes:
                out[hole[:, 0], hole[:, 1]] = 6
                solid[hole[:, 0], hole[:, 1]] = True

        outline = _outline_mask(solid)
        out[outline & (out == bg)] = 2

    return out


# --------------- generator ---------------

GRID_H_RANGE = (22, 28)
GRID_W_RANGE = (23, 27)
BG_COLOR = 4
FG_COLOR = 1

CHEBYSHEV_MIN = 3
BORDER_MIN = 1
NEAR_DISTANCE_RANGE = (3, LINK_DISTANCE)
FAR_DISTANCE_RANGE = (7, 12)
SIDE_MARGIN = 1.0
MIDLINE_WINDOW = 4.0

GEN_OUTER_RETRIES = 500
PLACEMENT_RETRIES = 1200

POLY_NUM_RECTS_RANGE = (2, 3)
POLY_RECT_DIM_RANGE = (2, 6)
POLY_CANVAS_PAD = 4
POLY_PLACEMENT_RETRIES = 50
POLY_GEN_RETRIES = 70

HOLE_VALID_AREAS = [2, 3, 4, 6, 8, 9, 12]
REF_HOLE_AREAS = [4, 6, 8, 9, 12]
HOLLOW_PAD_RANGE = (1, 3)
HOLLOW_MAX_NOTCHES = 2
HOLLOW_EXTENSION_PROB = 0.5
HOLLOW_EXT_H_RANGE = (1, 4)
HOLLOW_EXT_SIDE_W_RANGE = (1, 3)
HOLLOW_HOLE_SEARCH_RETRIES = 240
HOLLOW_GEN_RETRIES = 90
HOLE_DIM_MAX = 4

REFERENCE_MIN_AREA = 20
REFERENCE_MAX_AREA = 44
REFERENCE_MAX_SPAN = 10
SECONDARY_HOLLOW_MIN_AREA = 10
SECONDARY_HOLLOW_MAX_AREA = 22
SECONDARY_HOLLOW_MAX_SPAN = 8
SOLID_MIN_AREA = 4
SOLID_MAX_AREA = 16
SOLID_MAX_SPAN = 7
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


def _local_centroid_col(mask: np.ndarray) -> float:
    return float(np.where(mask)[1].mean())


def _world_centroid_col(mask: np.ndarray, c: int) -> float:
    return _local_centroid_col(mask) + float(c)


def _matches_side(centroid_col: float, grid_w: int, side: str | None) -> bool:
    if side is None:
        return True
    midpoint = grid_w / 2.0
    if side == "left":
        return centroid_col <= midpoint - SIDE_MARGIN
    return centroid_col >= midpoint + SIDE_MARGIN


def _matches_midline_window(centroid_col: float, grid_w: int, max_offset: float | None) -> bool:
    if max_offset is None:
        return True
    midpoint = grid_w / 2.0
    return abs(centroid_col - midpoint) <= max_offset


def _candidate_positions(
    H: int,
    W: int,
    mask: np.ndarray,
    side: str | None = None,
    midline_window: float | None = None,
) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    max_r = H - mh - BORDER_MIN
    max_c = W - mw - BORDER_MIN
    if max_r < BORDER_MIN or max_c < BORDER_MIN:
        return []

    candidates: list[tuple[int, int]] = []
    for r in range(BORDER_MIN, max_r + 1):
        for c in range(BORDER_MIN, max_c + 1):
            centroid_col = _world_centroid_col(mask, c)
            if not _matches_side(centroid_col, W, side):
                continue
            if not _matches_midline_window(centroid_col, W, midline_window):
                continue
            candidates.append((r, c))
    random.shuffle(candidates)
    return candidates


def _translate(coords: np.ndarray, r: int, c: int) -> np.ndarray:
    return coords + np.array([r, c], dtype=int)


def _place_mask(
    H: int,
    W: int,
    mask: np.ndarray,
    placed: list[dict[str, object]],
    ref_coords: np.ndarray | None = None,
    dist_range: tuple[int, int] | None = None,
    side: str | None = None,
    midline_window: float | None = None,
) -> dict[str, object] | None:
    local_coords = np.argwhere(mask)
    candidates = _candidate_positions(H, W, mask, side=side, midline_window=midline_window)
    if not candidates:
        return None

    for r, c in candidates[:PLACEMENT_RETRIES]:
        world_coords = _translate(local_coords, r, c)

        if dist_range is not None:
            if ref_coords is None:
                raise ValueError("dist_range requires ref_coords")
            d_ref = _min_chebyshev_distance(world_coords, ref_coords)
            if not (dist_range[0] <= d_ref <= dist_range[1]):
                continue

        if any(_min_chebyshev_distance(world_coords, np.asarray(other["coords"], dtype=int)) < CHEBYSHEV_MIN for other in placed):
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
    return [reference, near_hollow, far_hollow, near_solid]


def _variant_signature(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    _bg, records = _component_records(g)
    if len(records) != 4:
        return {"components": len(records)}

    ref_idx = _reference_index(records)
    ref_coords = np.asarray(records[ref_idx]["coords"], dtype=int)
    areas = sorted((int(np.asarray(record["coords"]).shape[0]) for record in records), reverse=True)

    stats = {
        "components": len(records),
        "active_left": 0,
        "active_right": 0,
        "inactive_hollow": 0,
        "linked_solid": 0,
        "near_nonref": 0,
        "far_nonref": 0,
        "unique_reference": int(len(areas) == 1 or areas[0] >= areas[1] + UNIQUE_REFERENCE_GAP),
    }

    midpoint = g.shape[1] / 2.0
    for idx, record in enumerate(records):
        coords = np.asarray(record["coords"], dtype=int)
        linked = idx == ref_idx or _min_chebyshev_distance(coords, ref_coords) <= LINK_DISTANCE
        holes = list(record["holes"])
        centroid_col = float(record["centroid_col"])

        if idx != ref_idx:
            d_ref = _min_chebyshev_distance(coords, ref_coords)
            if d_ref <= LINK_DISTANCE:
                stats["near_nonref"] += 1
            elif d_ref >= FAR_DISTANCE_RANGE[0]:
                stats["far_nonref"] += 1

        if holes:
            if linked:
                if centroid_col < midpoint:
                    stats["active_left"] += 1
                else:
                    stats["active_right"] += 1
            else:
                stats["inactive_hollow"] += 1
        elif linked:
            stats["linked_solid"] += 1

    return stats


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)

    components = _make_components()
    if components is None:
        return None
    reference_mask, near_hollow_mask, far_hollow_mask, near_solid_mask = components

    reference_side = random.choice(["left", "right"])
    near_hollow_side = "right" if reference_side == "left" else "left"
    far_hollow_side = random.choice([reference_side, near_hollow_side])

    placed: list[dict[str, object]] = []

    reference = _place_mask(
        H,
        W,
        reference_mask,
        placed,
        side=reference_side,
        midline_window=MIDLINE_WINDOW,
    )
    if reference is None:
        return None
    placed.append(reference)

    near_hollow = _place_mask(
        H,
        W,
        near_hollow_mask,
        placed,
        ref_coords=np.asarray(reference["coords"], dtype=int),
        dist_range=NEAR_DISTANCE_RANGE,
        side=near_hollow_side,
        midline_window=MIDLINE_WINDOW,
    )
    if near_hollow is None:
        return None
    placed.append(near_hollow)

    far_hollow = _place_mask(
        H,
        W,
        far_hollow_mask,
        placed,
        ref_coords=np.asarray(reference["coords"], dtype=int),
        dist_range=FAR_DISTANCE_RANGE,
        side=far_hollow_side,
    )
    if far_hollow is None:
        return None
    placed.append(far_hollow)

    near_solid = _place_mask(
        H,
        W,
        near_solid_mask,
        placed,
        ref_coords=np.asarray(reference["coords"], dtype=int),
        dist_range=NEAR_DISTANCE_RANGE,
    )
    if near_solid is None:
        return None
    placed.append(near_solid)

    grid = np.full((H, W), BG_COLOR, dtype=int)
    for item in placed:
        coords = np.asarray(item["coords"], dtype=int)
        grid[coords[:, 0], coords[:, 1]] = FG_COLOR

    signature = _variant_signature(grid)
    if (
        signature.get("components") != 4
        or signature.get("unique_reference") != 1
        or signature.get("active_left") != 1
        or signature.get("active_right") != 1
        or signature.get("inactive_hollow") != 1
        or signature.get("linked_solid") != 1
        or signature.get("near_nonref") != 2
        or signature.get("far_nonref") != 1
    ):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if not np.any(out == 8) or not np.any(out == 3) or not np.any(out == 6):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_OUTER_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_OUTER_RETRIES} attempts")
