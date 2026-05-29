# Combo variant of 2c181942 (index 34)
# combo_spec: D3/0+D4/0+D7/2
# Name: 八变换+多色+非对称
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
# - 原题里，每个精灵是单色且沿接触轴对称。solver 对某个 hub 边颜色 col 直接做
#   `m = (g == col) & (~hub_mask)`，于是“同色像素集合”就等于“整只精灵”。
#   然后只试 4 个旋转，并用整体镜像对称性来消歧。
# - 融合后，精灵变成“外壳色 + 内芯色”的双色复合对象，且轮廓完全非对称。匹配时
#   只有外壳色负责和 hub 边配对；内芯色只是装饰，贴附后必须保留。
# - 具体小例子：若 hub 左边是红色，输入里散落的是一只“红壳 + 黄芯”的非对称组件。
#   原逻辑会只抓到红色部分、把黄色芯丢掉，也无法处理镜像。新逻辑必须先把 hub 外的
#   整个前景做 8 连通分块，拿到整只红黄组件；再枚举 8 个 D8 变换（4 个旋转 + 4 个
#   翻转态），检查哪一个变换后：
#   1. 目标接触边正好露出长度为 N-2 的连续边界段；
#   2. 该边界段上的像素都等于 hub 边颜色；
#   3. 内芯锚点位于“接触边起点向内一格”的定向位置；
#   4. 在 8 个变换中只有这一个满足约束。
#   找到后，把整只双色对象贴到 hub 左边，黄色内芯原样保留。
# - 差异的本质是：原题的“组件提取”与“朝向判定”都依赖单色+对称；融合后这两点都失效，
#   必须改成“整组件提取 + 8 变换唯一合法性搜索”。
#
# b) 复杂度校准
# - 若允许任意双色非对称自由形状，很容易出现两个问题：
#   1. 接触边在多个变换下都合法，导致歧义；
#   2. 内芯颜色太自由，视觉上像噪声，看不出“外层匹配、内层保留”。
# - 因此生成器仍然控制自由度：
#   1. hub 固定为 4x4 hollow square，接触长度固定为 2；
#   2. 精灵像素数限制在 5-12，bbox 维持小尺寸；
#   3. 轮廓通过构造式随机生长，但强制对全部 7 个非恒等 D8 变换都不自同构；
#   4. 每只精灵恰好双色，内芯色从非 hub 边色的小调色板重复取值，形成重复支撑。
# - 同时在散放阶段强制至少一只匹配精灵以“反射态”出现，避免样本退化成只需 4 旋转。
#
# c) 隐式假设审查
# - 原代码 `m = (g == col) & (~hub_mask)` 假设“目标颜色像素 = 整只精灵”。融合后失效，
#   因为内芯像素不是 outer_color。新 solver 改为先抽取 hub 外所有前景组件。
# - 原代码 `label_mask(m, connectivity=8)` 后取最大同色块，假设不存在同色碎片和多色组件。
#   融合后不再以同色块为组件，而是以整个前景组件为单位评估。
# - 原代码对每个旋转立即用整体对称性筛选；融合后非对称，所以这一行前提失效。
#   新 solver 用局部接触边约束替代整体对称性约束，并要求 8 变换里唯一合法。
# - 原代码默认 paste 的数组是单色；融合后冲突检测和 paste 都必须对整只多色数组进行。
# - 新 solver 的新增假设：
#   1. 每个可匹配精灵在输入中是一个独立 8 连通前景组件；
#   2. hub 边色不会被拿去当别的精灵的内芯色；
#   3. 对于某条 hub 边，恰有一个组件含有该 outer_color 且在 8 变换下唯一合法。
#   这些都由生成器显式保证。
#
# d) 放置合法性
# - 继续沿用 output-first：先生成最终 attached 结果，验证不越界、不重叠，再把精灵
#   施加随机 D8 变换后散放回输入。
# - hub 周围保留 `max_extent + 2` 边距；散放与干扰物放置都走
#   `random_free_location(..., padding=2, padding_connectivity=8)`，确保各组件分离。
#
# e) 视觉可观察性
# - 人类从样例能直接看到三件事：
#   1. 贴上 hub 的不是单色块，而是双色整组件；
#   2. 只有外壳色和 hub 边色对应，内芯色在不同边上可重复，因此内芯不负责匹配；
#   3. 同一类精灵在输入里会出现镜像态，输出后恢复成唯一贴附朝向，因此必须考虑翻转。
# - 这些线索都来自可见的像素结构，不依赖颜色编号大小等不可视因素。尤其“内芯锚点”
#   总是贴着接触边某一端，翻转后会跑到另一端，解题者能肉眼看到朝向差异。
#
# f) 可视化整洁度
# - 所有对象尺寸小、间距固定、无遮挡。hub 保持中心清晰，精灵散放远离 hub；
#   干扰物若存在也用同一家族的小双色非对称形状，但外壳色不在 hub 边色里。
#
# g) 变化的可推断性
# - 颜色变化规则被压缩成“外壳色决定归属，内芯色只是装饰”。内芯色只从 2-3 个非边色里
#   复用，不会出现大量无提示的颜色置换。
# - 变换规则也被压缩成“在 8 种显式可见的朝向里找唯一合法接触边”，不混入额外排序规则。
#
# 9) 对 main() 新假设的逐段审查
# - `_extract_components()` 假设 hub 外所有非背景像素都可以按 8 连通拆成独立对象；
#   生成器通过 padding>=2 保证。
# - `vals = {arr[mask]}` 之后检查 `outer_color in vals and len(vals) >= 2`，假设匹配精灵是双色；
#   生成器强制每个可匹配精灵恰好双色。
# - `_unique_attachment_transform(...)` 假设目标组件在 8 变换下只有一个接触边合法朝向；
#   `_make_canonical_mask()` 和最终的 `_count_valid_transforms(..., with_reflections=True)==1`
#   会筛掉不满足该条件的样本。
# - `used_ids` 假设一个组件最多对应一个 hub 边；生成器保证 inner_color 不取 hub 边色，
#   所以不会一只组件同时命中多条边。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.zeros(mask.shape, dtype=int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        edge_r = int(pts[:, 0].min())
        cols = sorted(int(c) for c in pts[pts[:, 0] == edge_r][:, 1])
        return ("h", edge_r, cols)
    if side == "bot":
        edge_r = int(pts[:, 0].max())
        cols = sorted(int(c) for c in pts[pts[:, 0] == edge_r][:, 1])
        return ("h", edge_r, cols)
    if side == "left":
        edge_c = int(pts[:, 1].min())
        rows = sorted(int(r) for r in pts[pts[:, 1] == edge_c][:, 0])
        return ("v", edge_c, rows)
    if side == "right":
        edge_c = int(pts[:, 1].max())
        rows = sorted(int(r) for r in pts[pts[:, 1] == edge_c][:, 0])
        return ("v", edge_c, rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _apply_d8(arr: np.ndarray, transform_id: int) -> np.ndarray:
    transform_id = int(transform_id)
    if transform_id < 4:
        return np.rot90(arr, transform_id)
    return np.rot90(np.fliplr(arr), transform_id - 4)


def _all_other_d8_masks(mask: np.ndarray) -> list[np.ndarray]:
    return [_apply_d8(mask, tid) for tid in range(1, 8)]


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, other) for other in _all_other_d8_masks(mask))


def _segment_match(mask: np.ndarray, need_edge: str, target_len: int):
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target_len) or not _is_contig(rows):
            return None
        return ("v", int(edge_c), rows)
    _typ, edge_r, cols = seg
    if len(cols) != int(target_len) or not _is_contig(cols):
        return None
    return ("h", int(edge_r), cols)


def _anchor_cell(need_edge: str, seg) -> tuple[int, int] | None:
    if need_edge == "right":
        _typ, edge_c, rows = seg
        return int(rows[0]), int(edge_c - 1)
    if need_edge == "left":
        _typ, edge_c, rows = seg
        return int(rows[0]), int(edge_c + 1)
    if need_edge == "top":
        _typ, edge_r, cols = seg
        return int(edge_r + 1), int(cols[0])
    if need_edge == "bot":
        _typ, edge_r, cols = seg
        return int(edge_r - 1), int(cols[0])
    return None


def _attachment_info(arr: np.ndarray, mask: np.ndarray, need_edge: str, target_len: int, outer_color: int):
    seg = _segment_match(mask, need_edge, target_len)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if any(int(arr[int(r), int(edge_c)]) != int(outer_color) for r in rows):
            return None
    else:
        _typ, edge_r, cols = seg
        if any(int(arr[int(edge_r), int(c)]) != int(outer_color) for c in cols):
            return None
    anchor = _anchor_cell(need_edge, seg)
    if anchor is None:
        return None
    anchor_r, anchor_c = anchor
    if not (0 <= int(anchor_r) < mask.shape[0] and 0 <= int(anchor_c) < mask.shape[1]):
        return None
    if not bool(mask[int(anchor_r), int(anchor_c)]):
        return None
    if int(arr[int(anchor_r), int(anchor_c)]) == int(outer_color):
        return None
    colors = {int(v) for v in arr[mask]}
    if len(colors) < 2:
        return None
    return seg


def _count_valid_mask_transforms(mask: np.ndarray, need_edge: str, target_len: int, *, with_reflections: bool) -> int:
    total = 0
    upper = 8 if with_reflections else 4
    for transform_id in range(upper):
        transformed_mask = _crop_mask(_apply_d8(mask, transform_id))
        if _segment_match(transformed_mask, need_edge, target_len) is not None:
            total += 1
    return int(total)


def _count_valid_transforms(
    arr: np.ndarray,
    mask: np.ndarray,
    need_edge: str,
    target_len: int,
    outer_color: int,
    *,
    with_reflections: bool,
) -> int:
    total = 0
    upper = 8 if with_reflections else 4
    for transform_id in range(upper):
        transformed_arr = _apply_d8(arr, transform_id)
        transformed_mask = _apply_d8(mask, transform_id)
        transformed_arr, transformed_mask = _crop_to_mask(transformed_arr, transformed_mask)
        if _attachment_info(transformed_arr, transformed_mask, need_edge, target_len, outer_color) is not None:
            total += 1
    return int(total)


def _unique_attachment_transform(
    arr: np.ndarray,
    mask: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
    target_len: int,
    outer_color: int,
):
    candidates = []
    for transform_id in range(8):
        transformed_arr = _apply_d8(arr, transform_id)
        transformed_mask = _apply_d8(mask, transform_id)
        transformed_arr, transformed_mask = _crop_to_mask(transformed_arr, transformed_mask)
        seg = _attachment_info(transformed_arr, transformed_mask, need_edge, target_len, outer_color)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])
        candidates.append((int(transform_id), transformed_arr, transformed_mask, int(dr), int(dc)))
    if len(candidates) != 1:
        return None
    return candidates[0]


def _can_place(arr: np.ndarray, mask: np.ndarray, dr: int, dc: int, out: np.ndarray, bg: int) -> bool:
    h, w = out.shape
    placed = {(int(r + dr), int(c + dc)) for r, c in np.argwhere(mask)}
    if any(not (0 <= int(r) < h and 0 <= int(c) < w) for r, c in placed):
        return False
    for r, c in placed:
        val = int(arr[int(r - dr), int(c - dc)])
        if val == int(bg):
            continue
        if int(out[int(r), int(c)]) != int(bg) and int(out[int(r), int(c)]) != val:
            return False
    return True


def _find_hub(g: np.ndarray, bg: int):
    h, w = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, h - n + 1):
            for c in range(0, w - n + 1):
                if (
                    int(g[int(r), int(c)]) != int(bg)
                    or int(g[int(r), int(c + n - 1)]) != int(bg)
                    or int(g[int(r + n - 1), int(c)]) != int(bg)
                    or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue
                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue
                top = g[int(r), int(c + 1) : int(c + n - 1)]
                bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = g[int(r + 1) : int(r + n - 1), int(c)]
                right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue
                center_r = float(r + (n - 1) / 2)
                center_c = float(c + (n - 1) / 2)
                score = abs(float(center_r - (h - 1) / 2)) + abs(float(center_c - (w - 1) / 2))
                candidate = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(candidate[0]) < float(best[0]):
                    best = candidate
    return best


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int):
    fg = (g != int(bg)) & (~hub_mask)
    labels, ncomp = label_mask(fg, connectivity=8)
    components = []
    for cid in range(1, int(ncomp) + 1):
        comp_mask = labels == int(cid)
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        components.append((int(cid), arr, comp_mask))
    return components


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


def _attachment_position(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        cols = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(cols[0]))
    if side == "bot":
        cols = np.where(mask[0])[0]
        return int(hub_r + n), int(hub_c + 1 - int(cols[0]))
    if side == "left":
        rows = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(rows[0])), int(hub_c - sw)
    rows = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(rows[0])), int(hub_c + n)


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(800):
        h = rng.choice([4, 5])
        w = rng.choice([3, 4, 5])
        run_start = rng.randint(0, int(h - edge_len))
        target_pixels = rng.randint(5, min(12, h * w - max(0, h - edge_len - 1)))

        mask = np.zeros((h, w), dtype=bool)
        mask[int(run_start) : int(run_start + edge_len), int(w - 1)] = True
        mask[int(run_start), int(w - 2)] = True

        while int(mask.sum()) < int(target_pixels):
            candidates = set()
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if bool(mask[int(nr), int(nc)]):
                            continue
                        if nc == int(w - 1) and not (int(run_start) <= nr < int(run_start + edge_len)):
                            continue
                        candidates.add((int(nr), int(nc)))
            if not candidates:
                break

            cand_list = list(candidates)
            weights = []
            for nr, nc in cand_list:
                weight = 1 + (int(w - 1) - int(nc))
                if int(nc) <= int(w - 3):
                    weight += 2
                if not (int(run_start) <= int(nr) < int(run_start + edge_len)):
                    weight += 1
                weights.append(float(weight))
            nr, nc = rng.choices(cand_list, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) > 12:
            continue
        _labels, ncomp = label_mask(mask, connectivity=8)
        if int(ncomp) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_mask_transforms(mask, "right", edge_len, with_reflections=False) != 1:
            continue
        return mask
    return None


def _choose_inner_cells(mask: np.ndarray, edge_len: int, rng: random.Random) -> tuple[tuple[int, int], ...] | None:
    seg = _segment_match(mask, "right", edge_len)
    if seg is None:
        return None
    anchor = _anchor_cell("right", seg)
    if anchor is None:
        return None
    _typ, edge_c, rows = seg
    forbidden = {(int(r), int(edge_c)) for r in rows}
    if anchor not in {(int(r), int(c)) for r, c in np.argwhere(mask)}:
        return None
    candidates = []
    weights = []
    for rr, cc in np.argwhere(mask):
        coord = (int(rr), int(cc))
        if coord in forbidden:
            continue
        if coord == anchor:
            continue
        weight = 1
        if 0 < int(rr) < mask.shape[0] - 1:
            weight += 1
        if 0 < int(cc) < mask.shape[1] - 1:
            weight += 2
        if int(cc) <= mask.shape[1] - 2:
            weight += 1
        candidates.append(coord)
        weights.append(float(weight))
    if not candidates:
        return (anchor,)
    extra_count = rng.choice([0, 1])
    picked = rng.choices(candidates, weights=weights, k=max(1, extra_count * 2))
    out = [anchor]
    seen = {anchor}
    for coord in picked:
        if coord in seen:
            continue
        seen.add(coord)
        out.append(coord)
        if len(out) == 1 + int(extra_count):
            break
    return tuple(out)


def _make_composite_sprite(side: str, outer_color: int, inner_color: int, edge_len: int, rng: random.Random):
    for _ in range(300):
        left_mask = _make_canonical_mask(edge_len, rng)
        if left_mask is None:
            continue
        inner_cells = _choose_inner_cells(left_mask, edge_len, rng)
        if inner_cells is None:
            continue

        left_arr = np.full(left_mask.shape, int(outer_color), dtype=int)
        for rr, cc in inner_cells:
            left_arr[int(rr), int(cc)] = int(inner_color)

        oriented_arr = _orient_from_left(left_arr, side)
        oriented_mask = _orient_from_left(left_mask, side)
        oriented_arr, oriented_mask = _crop_to_mask(oriented_arr, oriented_mask)

        need_edge = _contact_side_for_hub_side(side)
        if _count_valid_transforms(
            oriented_arr,
            oriented_mask,
            need_edge,
            edge_len,
            outer_color,
            with_reflections=True,
        ) != 1:
            continue
        if not _is_fully_asymmetric(oriented_mask):
            continue
        return oriented_arr, oriented_mask
    return None


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (hub_r, hub_c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((h, w), dtype=bool)
    hub_mask[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = g[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)]

    components = _extract_components(g, hub_mask, bg)
    target_len = int(n - 2)
    segs = {
        "top": (int(top_col), ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1))))),
        "bot": (int(bot_col), ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1))))),
        "left": (int(left_col), ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1))))),
        "right": (int(right_col), ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1))))),
    }

    used_ids: set[int] = set()
    for _side, (outer_color, (need_edge, fixed, span)) in segs.items():
        best = None
        best_score = None
        for cid, arr, mask in components:
            if int(cid) in used_ids:
                continue
            values = {int(v) for v in arr[mask]}
            if int(outer_color) not in values or len(values) < 2:
                continue

            placed = _unique_attachment_transform(arr, mask, need_edge, fixed, span, target_len, outer_color)
            if placed is None:
                continue

            _transform_id, placed_arr, placed_mask, dr, dc = placed
            if not _can_place(placed_arr, placed_mask, dr, dc, out, bg):
                continue

            score = (int((arr[mask] == int(outer_color)).sum()), int(mask.sum()))
            if best is None or score > best_score:
                best = (int(cid), placed_arr, placed_mask, int(dr), int(dc))
                best_score = score

        if best is None:
            continue

        cid, placed_arr, placed_mask, dr, dc = best
        used_ids.add(int(cid))
        paste(out, placed_arr, int(dr), int(dc), mask=placed_mask, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    h = rng.randint(18, 26)
    w = rng.randint(20, 26)

    all_colors = [color for color in range(10) if color != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    edge_colors = [int(top_col), int(bot_col), int(left_col), int(right_col)]
    side_colors = {"top": int(top_col), "bot": int(bot_col), "left": int(left_col), "right": int(right_col)}
    remaining = [int(c) for c in all_colors if int(c) not in edge_colors]
    if len(remaining) < 3:
        return None

    accent_palette = remaining[: min(3, len(remaining))]
    sides = ["top", "bot", "left", "right"]
    k = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, k)

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}
    for side in sprite_sides:
        outer_color = int(side_colors[side])
        inner_color = int(rng.choice(accent_palette))
        if inner_color == outer_color:
            return None
        composite = _make_composite_sprite(side, outer_color, inner_color, edge_len, rng)
        if composite is None:
            return None
        arr, mask = composite
        sprite_defs[side] = (arr, mask, inner_color)

    max_sprite_h = max(arr.shape[0] for arr, _mask, _inner in sprite_defs.values())
    max_sprite_w = max(arr.shape[1] for arr, _mask, _inner in sprite_defs.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = h - n - margin
    hub_c_min = margin
    hub_c_max = w - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((h, w), int(bg), dtype=int)
    for ci in range(1, n - 1):
        out[int(hub_r), int(hub_c + ci)] = int(top_col)
        out[int(hub_r + n - 1), int(hub_c + ci)] = int(bot_col)
    for ri in range(1, n - 1):
        out[int(hub_r + ri), int(hub_c)] = int(left_col)
        out[int(hub_r + ri), int(hub_c + n - 1)] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        arr, mask, _inner = sprite_defs[side]
        sr, sc = _attachment_position(mask, side, hub_r, hub_c, n)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > h or sc + sw > w:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None
        patch = np.where(mask, arr, bg).astype(int)
        paste(out, patch, int(sr), int(sc), transparent=bg)
        attached_sprites[side] = (patch, mask)

    inp = np.full((h, w), int(bg), dtype=int)
    inp[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = out[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)]
    canvas = inp.copy()

    forced_reflect_side = rng.choice(sprite_sides)
    for side in sprite_sides:
        patch, mask = attached_sprites[side]
        if side == forced_reflect_side:
            transform_id = rng.choice([4, 5, 6, 7])
        else:
            transform_id = rng.choice([1, 2, 3, 4, 5, 6, 7])
        scattered_patch = _apply_d8(patch, transform_id)
        scattered_mask = _apply_d8(mask, transform_id)
        scattered_patch, scattered_mask = _crop_to_mask(scattered_patch, scattered_mask)
        scatter = np.where(scattered_mask, scattered_patch, bg).astype(int)
        try:
            pr, pc = random_free_location(
                canvas,
                scatter,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, scatter, pr, pc, transparent=bg)
        paste(inp, scatter, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        distractor_outer_choices = [c for c in remaining if c not in accent_palette]
        if distractor_outer_choices:
            d_outer = int(rng.choice(distractor_outer_choices))
            d_inner = int(rng.choice(accent_palette))
            d_side = rng.choice(sides)
            distractor = _make_composite_sprite(d_side, d_outer, d_inner, edge_len, rng)
            if distractor is not None:
                dist_arr, dist_mask = distractor
                dist_transform = rng.choice(list(range(8)))
                dist_arr = _apply_d8(dist_arr, dist_transform)
                dist_mask = _apply_d8(dist_mask, dist_transform)
                dist_arr, dist_mask = _crop_to_mask(dist_arr, dist_mask)
                dist_patch = np.where(dist_mask, dist_arr, bg).astype(int)
                try:
                    pr, pc = random_free_location(
                        canvas,
                        dist_patch,
                        transparent=bg,
                        padding=2,
                        padding_connectivity=8,
                    )
                    paste(canvas, dist_patch, pr, pc, transparent=bg)
                    paste(inp, dist_patch, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
