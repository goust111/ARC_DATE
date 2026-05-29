# Variant of 53fb4810
# Dimension: D19 — 输入-输出差异构造 (multi_step)
# Change: output-first生成：先构造完整输出（所有射线展开），然后擦除部分射线得到输入，保留一条最长射线作为线索... → 需要擦除干扰射线
#
# Design analysis
# a) 具体化与行为追踪
# - 输出仍保持原题核心：每个 plus 只拥有一个真正的 marker seed，seed 与某一条臂尖 4 邻接，
#   且 seed 远离 plus 的方向决定正确射线；输出要把这个 seed 周期性铺到边界。
# - 新输入不再“缺少大部分射线”，而是“所有 plus 都给出 seed，其中一些 plus 的长射线方向是错的”。
#   具体地：若 seed 在 plus 右侧，则正确射线应继续向右；变体输入会把这条右向延伸擦掉，
#   再从同一个 seed 出发画一条向上或向下的长干扰射线。另一些 plus 会保留正确长射线，
#   让解题者在同一幅图里同时看到“对”和“错”两种情况。
# - 小例子：旧逻辑里，右侧 1x2 seed 可能只保留 seed 自身，或保留整条向右 clue。新逻辑里，
#   这个 plus 仍在右侧保留同样的 1x2 seed，但长条可能被换成向上堆叠的 1x2 列。输出需要删掉
#   这条向上支线，并恢复向右的周期铺展。
#
# b) 复杂度校准
# - 每张图强制同时含有至少 1 条正确长射线和至少 1 条错误长射线，避免任务退化成
#   “全部都错”或“几乎还是原题”。
# - 干扰方向只允许取与正确方向垂直的两个方向，而不允许朝回 plus 的反向。这样视觉上明显，
#   也避免错误射线穿过 plus。
# - 为了让 solver 在“输入中有完整正确长射线”时仍能稳定恢复 seed，seed 的沿轴厚度固定为 2。
#   若仍允许 1/2 混合，则一条完整正确射线会把 seed 厚度信息吞掉，solver 无法只凭局部条带可靠
#   地知道 seed 是 1 格还是 2 格，这是为 D19 可解性做的必要适配。
#
# c) 隐式假设审查
# - 原 solver 把所有非 bg/非 plus 的连通块都当成紧凑 seed，并直接对连通块 bbox 取 tile。
#   这个前提在变体里失效，因为正确长射线和错误长射线都会与 seed 连成一个大组件。
# - 原 solver 通过“组件中紧贴 plus 的像素集合”推方向，这部分仍然成立；但后续的 bbox 对齐断言
#   依赖“组件就是 seed 本体”，在变体里会误把整条长射线当 seed，因此必须改为直接从 plus 边缘
#   读取固定 2 格深的规范 seed 条带。
# - 原 generator 的输入构造是“从输出擦除大多数正确射线”。变体需要改成“从输出擦除一部分正确射线，
#   并补画垂直干扰射线”，否则不会出现 D19 的核心输入-输出差异。
#
# d) 放置合法性
# - 输出阶段仍先放所有正确射线，并要求它们彼此不碰撞、不覆盖 plus。
# - 输入阶段再为被污染的 plus 选择垂直干扰方向；干扰射线必须避开其他 plus、所有保留的正确射线、
#   以及所有 seed，才能保证输入整洁且 solver 的“唯一相邻边”假设成立。
# - 若找不到满足约束的干扰分配，则整张样例重生成。
#
# e) 视觉可观察性
# - 人类能直接看到：seed 总贴在某一条臂上，因此正确方向一定是“继续远离 plus”；凡是从 seed
#   折去垂直方向的长条就是干扰。至少一条正确长射线会在同图中提供正例。
# - 规则不依赖颜色 id 的排序或隐藏映射；颜色只用来构成周期 tile，本质线索是几何方向关系。
#
# f) 可视化整洁度
# - 射线之间禁止碰撞；plus 周围保留一圈占位缓冲，避免别的对象贴到非真实边上。
# - 错误射线只从 seed 出发，不额外引入新标记，因此画面仍然是“plus + 一个 seed/射线组件”的整洁结构。
#
# g) 颜色变化可推断性
# - tile 使用 2-4 色中的至少 2 色，避免整条纯色条纹让周期结构过弱。
# - 颜色本身不决定方向；不同样例只需从 seed 的位置和重复图案推导“删错射线、补正射线”。
#
# main() 的新假设与 generator 保证
# - 每个 plus 只有一侧紧邻一个真实 seed；generator 用 plus 周边缓冲保证这一点。
# - seed 的沿轴厚度恒为 2；generator 与 solver 共用这个约束。
# - 输入中错误射线若存在，只会与 seed 连通且方向垂直于正确方向；generator 在构造时显式保证。

from __future__ import annotations

from itertools import combinations

import numpy as np

from arc_utils import bbox_coords, crop_bbox, get_bg, label_mask


def detect_dominant_big_component_color(
    grid: np.ndarray,
    *,
    bg: int | None = None,
    min_h: int = 3,
    min_w: int = 3,
    connectivity: int = 4,
) -> int | None:
    """
    Local helper (only used by this solver): choose a color that has the most "big" components.

    Big component = connected component whose bbox is at least (min_h x min_w).
    Tie-break: total area of big components (larger wins).
    """
    if bg is None:
        bg = get_bg(grid)
    bg = int(bg)

    best_key: tuple[int, int] | None = None
    best_col: int | None = None

    for col in sorted(set(map(int, grid.flatten())) - {int(bg)}):
        lab, n = label_mask(grid == int(col), connectivity=int(connectivity))
        big_areas: list[int] = []
        for k in range(1, int(n) + 1):
            coords = np.argwhere(lab == int(k))
            if coords.size == 0:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            h = int(r1 - r0 + 1)
            w = int(c1 - c0 + 1)
            if h >= int(min_h) and w >= int(min_w):
                big_areas.append(int(coords.shape[0]))
        if not big_areas:
            continue
        key = (int(len(big_areas)), int(sum(big_areas)))
        if best_key is None or key > best_key:
            best_key = key
            best_col = int(col)

    return int(best_col) if best_col is not None else None


def extend_repeating_tile(
    out: np.ndarray,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
    *,
    bg: int,
    protect: set[int] | None = None,
) -> None:
    """Local helper (only used by this solver): extend a tile pattern from (r0,c0) until boundary."""
    protect = set() if protect is None else set(map(int, protect))
    H, W = out.shape
    h, w = tile.shape
    dr, dc = map(int, direction)
    dr = 0 if dr == 0 else int(dr / abs(dr))
    dc = 0 if dc == 0 else int(dc / abs(dc))
    bg = int(bg)
    r0 = int(r0)
    c0 = int(c0)

    def put(rr: int, cc: int, v: int) -> None:
        if v == bg:
            return
        if int(out[rr, cc]) in protect:
            return
        out[rr, cc] = int(v)

    if (dr, dc) == (0, 1):
        start_c = c0 + w
        for k, tc in enumerate(range(start_c, W)):
            col_idx = k % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (0, -1):
        for k, tc in enumerate(range(c0 - 1, -1, -1), start=1):
            col_idx = (-k) % w
            for i in range(h):
                put(r0 + i, tc, int(tile[i, col_idx]))
    elif (dr, dc) == (1, 0):
        start_r = r0 + h
        for k, tr in enumerate(range(start_r, H)):
            row_idx = k % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    elif (dr, dc) == (-1, 0):
        for k, tr in enumerate(range(r0 - 1, -1, -1), start=1):
            row_idx = (-k) % h
            for j in range(w):
                put(tr, c0 + j, int(tile[row_idx, j]))
    else:
        raise ValueError("extend_repeating_tile: direction must be one of (±1,0),(0,±1)")


def _paint_tile(out: np.ndarray, tile: np.ndarray, r0: int, c0: int, *, bg: int) -> None:
    """Stamp the seed tile itself before extending from it."""
    h, w = tile.shape
    for i in range(h):
        for j in range(w):
            v = int(tile[i, j])
            if v != int(bg):
                out[int(r0) + i, int(c0) + j] = v


def _nozzle_span(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    """Return (start, end) inclusive of nozzle cells along the cross-axis."""
    dr, dc = direction
    if dc == 0:
        row = plus_mask.shape[0] - 1 if dr == 1 else 0
        cols = np.where(plus_mask[row])[0]
        return int(cols[0]), int(cols[-1])
    col = plus_mask.shape[1] - 1 if dc == 1 else 0
    rows = np.where(plus_mask[:, col])[0]
    return int(rows[0]), int(rows[-1])


def _seed_shape_for_direction(plus_mask: np.ndarray, direction: tuple[int, int]) -> tuple[int, int]:
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    dr, dc = direction
    return (2, cross) if dc == 0 else (cross, 2)


def _seed_anchor_from_plus(
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile_shape: tuple[int, int],
) -> tuple[int, int]:
    pr0, pc0, pr1, pc1 = plus_bbox
    dr, dc = direction
    th, tw = tile_shape
    s, _ = _nozzle_span(plus_mask, direction)
    if dc == 0:
        c0 = pc0 + s
        r0 = (pr1 + 1) if dr == 1 else (pr0 - th)
    else:
        r0 = pr0 + s
        c0 = (pc1 + 1) if dc == 1 else (pc0 - tw)
    return int(r0), int(c0)


def _project_from_seed(
    H: int,
    W: int,
    tile: np.ndarray,
    r0: int,
    c0: int,
    direction: tuple[int, int],
) -> tuple[list[tuple[int, int, int]], int]:
    """Return all cells in the repeated ray, including the seed tile itself."""
    th, tw = tile.shape
    dr, dc = direction
    cells: list[tuple[int, int, int]] = []
    units = 0

    if dc == 0:
        r = int(r0)
        step = th if dr == 1 else -th
        while (r < H) if dr == 1 else (r + th > 0):
            units += 1
            for i in range(th):
                rr = r + i
                if 0 <= rr < H:
                    for j in range(tw):
                        cells.append((rr, int(c0) + j, int(tile[i, j])))
            r += step
    else:
        c = int(c0)
        step = tw if dc == 1 else -tw
        while (c < W) if dc == 1 else (c + tw > 0):
            units += 1
            for i in range(th):
                for j in range(tw):
                    cc = c + j
                    if 0 <= cc < W:
                        cells.append((int(r0) + i, cc, int(tile[i, j])))
            c += step

    return cells, int(units)


def _ray_cells_from_plus(
    H: int,
    W: int,
    plus_bbox: tuple[int, int, int, int],
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    tile: np.ndarray,
) -> tuple[list[tuple[int, int, int]], int, int, int]:
    """Return (cells, seed_r0, seed_c0, units) for the correct ray of one plus."""
    seed_r0, seed_c0 = _seed_anchor_from_plus(plus_bbox, plus_mask, direction, tile.shape)
    cells, units = _project_from_seed(H, W, tile, seed_r0, seed_c0, direction)
    return cells, int(seed_r0), int(seed_c0), int(units)


def _perpendicular_directions(direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    if dc == 0:
        return [(0, -1), (0, 1)]
    return [(-1, 0), (1, 0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the outward periodic rays from the seed strips attached to '+' shapes.

    Input structure:
    - bg is the majority color.
    - One dominant non-bg color forms several identical plus-like anchors.
    - Each plus has one true seed strip touching exactly one arm tip; some long rays in the input
      already continue in the correct away-from-plus direction, while others are distracting branches
      that leave the same seed in a perpendicular direction.

    Rule:
    - For each plus, identify the unique side whose adjacent strip is the seed.
    - Keep only that seed strip and extend its tile periodically away from the plus until the boundary.
    - Any pre-drawn branch in another direction is ignored; plus pixels are protected from overwrite.

    Key insight: The attachment side, not the currently drawn long branch, determines the true ray direction;
    the canonical seed is the fixed two-cell-deep strip immediately outside the matching arm tip.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    plus_col = detect_dominant_big_component_color(g, bg=bg, min_h=3, min_w=3, connectivity=4)
    if plus_col is None:
        return g.copy()

    out = np.full_like(g, bg)
    plus_mask_all = g == int(plus_col)
    out[plus_mask_all] = int(plus_col)
    plus_lab, pn = label_mask(plus_mask_all, connectivity=4)

    for pk in range(1, int(pn) + 1):
        plus_mask = plus_lab == int(pk)
        plus_coords = np.argwhere(plus_mask)
        if plus_coords.size == 0:
            continue

        pr0, pc0, pr1, pc1 = bbox_coords(plus_coords)
        local_plus = crop_bbox(plus_mask.astype(int), (int(pr0), int(pc0), int(pr1), int(pc1))).astype(bool)

        candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for direction in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            th, tw = _seed_shape_for_direction(local_plus, direction)
            sr0, sc0 = _seed_anchor_from_plus((pr0, pc0, pr1, pc1), local_plus, direction, (th, tw))
            if sr0 < 0 or sc0 < 0 or sr0 + th > H or sc0 + tw > W:
                continue
            seed = g[sr0 : sr0 + th, sc0 : sc0 + tw]
            if np.all(seed != bg) and np.all(seed != plus_col):
                candidates.append((direction, (sr0, sc0)))

        if len(candidates) != 1:
            continue

        direction, (seed_r0, seed_c0) = candidates[0]
        th, tw = _seed_shape_for_direction(local_plus, direction)
        tile = g[seed_r0 : seed_r0 + th, seed_c0 : seed_c0 + tw].copy()
        _paint_tile(out, tile, seed_r0, seed_c0, bg=bg)
        extend_repeating_tile(
            out,
            tile,
            seed_r0,
            seed_c0,
            direction,
            bg=bg,
            protect={int(plus_col)},
        )

    return out


# --------------- generator (D19 distractor-ray variant) ---------------

_H_RANGE = (13, 29)
_W_RANGE = (13, 23)
_N_PLUS_RANGE = (2, 4)
_SEED_ALONG = 2
_SEED_MAX_COLORS = 4
_SEED_MAX_CROSS = 2
_MIN_RAY_UNITS = 2
_PLACEMENT_TRIES = 120
_GEN_RETRIES = 400

_PLUS_SHAPES = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[0, 1, 1, 0], [1, 1, 1, 1], [0, 1, 1, 0]], dtype=int),
    np.array(
        [
            [0, 1, 1, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=int,
    ),
]

_DIRECTIONS = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _build_seed_tile(
    plus_mask: np.ndarray,
    direction: tuple[int, int],
    palette: list[int],
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Build a dense seed tile whose cross-axis matches the nozzle span and whose along-axis depth is fixed."""
    s, e = _nozzle_span(plus_mask, direction)
    cross = int(e - s + 1)
    if cross > _SEED_MAX_CROSS:
        return None

    dr, dc = direction
    th, tw = (_SEED_ALONG, cross) if dc == 0 else (cross, _SEED_ALONG)
    n_cells = th * tw
    max_k = min(_SEED_MAX_COLORS, n_cells, len(palette))
    if max_k < 2:
        return None

    n_colors = int(rng.integers(2, max_k + 1))
    chosen = list(rng.choice(palette, size=n_colors, replace=False))
    vals = list(chosen)
    vals.extend(int(rng.choice(chosen)) for _ in range(n_cells - n_colors))
    rng.shuffle(vals)
    return np.array(vals, dtype=int).reshape(th, tw)


def _place_plus_and_ray(
    rng: np.random.Generator,
    plus_shape: np.ndarray,
    palette: list[int],
    H: int,
    W: int,
    occ: np.ndarray,
    plus_occ: np.ndarray,
    out: np.ndarray,
    plus_color: int,
) -> dict | None:
    """Place one plus with its correct output ray."""
    ph, pw = plus_shape.shape

    for _ in range(_PLACEMENT_TRIES):
        r0 = int(rng.integers(1, H - ph))
        c0 = int(rng.integers(1, W - pw))
        plus_cells = [
            (r0 + dr, c0 + dc)
            for dr in range(ph)
            for dc in range(pw)
            if plus_shape[dr, dc]
        ]
        if any(occ[r, c] != 0 for r, c in plus_cells):
            continue
        if any(plus_occ[r, c] == 1 for r, c in plus_cells):
            continue

        dirs = list(_DIRECTIONS)
        rng.shuffle(dirs)
        for direction in dirs:
            tile = _build_seed_tile(plus_shape, direction, palette, rng)
            if tile is None:
                continue

            cells, sr0, sc0, units = _ray_cells_from_plus(
                H,
                W,
                (r0, c0, r0 + ph - 1, c0 + pw - 1),
                plus_shape.astype(bool),
                direction,
                tile,
            )
            if units < _MIN_RAY_UNITS:
                continue
            if any(occ[r, c] != 0 for r, c, _ in cells if 0 <= r < H and 0 <= c < W):
                continue

            for r, c in plus_cells:
                occ[r, c] = 1
                out[r, c] = plus_color
                for dr2, dc2 in _DIRECTIONS:
                    rr, cc = r + dr2, c + dc2
                    if 0 <= rr < H and 0 <= cc < W:
                        plus_occ[rr, cc] = 1
            for r, c, v in cells:
                occ[r, c] = 2
                out[r, c] = v

            seed_h, seed_w = tile.shape
            seed_set = {
                (sr0 + i, sc0 + j)
                for i in range(seed_h)
                for j in range(seed_w)
            }
            return {
                "plus_bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
                "plus_mask": plus_shape.astype(bool).copy(),
                "plus_cells": plus_cells,
                "direction": direction,
                "tile": tile.copy(),
                "cells": cells,
                "seed_r0": sr0,
                "seed_c0": sc0,
                "seed_set": seed_set,
                "correct_extra": [(r, c, v) for r, c, v in cells if (r, c) not in seed_set],
                "units": units,
            }

    return None


def _compute_wrong_candidates(
    info: dict,
    H: int,
    W: int,
) -> list[dict]:
    out: list[dict] = []
    tile = np.array(info["tile"], dtype=int)
    sr0 = int(info["seed_r0"])
    sc0 = int(info["seed_c0"])
    seed_set = set(info["seed_set"])

    for wrong_dir in _perpendicular_directions(tuple(info["direction"])):
        cells, units = _project_from_seed(H, W, tile, sr0, sc0, wrong_dir)
        if units < _MIN_RAY_UNITS:
            continue
        extra = [(r, c, v) for r, c, v in cells if (r, c) not in seed_set]
        if not extra:
            continue
        out.append(
            {
                "direction": wrong_dir,
                "cells": cells,
                "extra": extra,
                "extra_set": {(r, c) for r, c, _ in extra},
                "units": units,
            }
        )
    return out


def _choose_wrong_assignments(
    pluses: list[dict],
    H: int,
    W: int,
    rng: np.random.Generator,
) -> dict[int, dict] | None:
    n = len(pluses)
    all_ids = list(range(n))

    base_plus_cells = {cell for info in pluses for cell in info["plus_cells"]}
    base_seed_cells = {cell for info in pluses for cell in info["seed_set"]}
    wrong_candidates = [_compute_wrong_candidates(info, H, W) for info in pluses]

    subset_order: list[tuple[int, ...]] = []
    for k in range(1, n):
        subset_order.extend(tuple(s) for s in combinations(all_ids, k))
    rng.shuffle(subset_order)

    for wrong_ids in subset_order:
        wrong_set = set(wrong_ids)
        if any(not wrong_candidates[i] for i in wrong_ids):
            continue

        occupied = set(base_plus_cells) | set(base_seed_cells)
        for i in all_ids:
            if i in wrong_set:
                continue
            occupied.update((r, c) for r, c, _ in pluses[i]["correct_extra"])

        order = list(wrong_ids)
        rng.shuffle(order)
        chosen: dict[int, dict] = {}

        def backtrack(pos: int, occ_cells: set[tuple[int, int]]) -> bool:
            if pos == len(order):
                return True
            idx = order[pos]
            cand_list = list(wrong_candidates[idx])
            rng.shuffle(cand_list)
            for cand in cand_list:
                if cand["extra_set"] & occ_cells:
                    continue
                chosen[idx] = cand
                if backtrack(pos + 1, occ_cells | cand["extra_set"]):
                    return True
                chosen.pop(idx, None)
            return False

        if backtrack(0, occupied):
            return chosen

    return None


def _build_variant_pair(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict] | None:
    H = int(rng.integers(*_H_RANGE))
    W = int(rng.integers(*_W_RANGE))

    bg = int(rng.integers(0, 10))
    others = [c for c in range(10) if c != bg]
    plus_color = int(rng.choice(others))
    palette = [c for c in others if c != plus_color]

    plus_shape = _PLUS_SHAPES[int(rng.integers(0, len(_PLUS_SHAPES)))]
    N = int(rng.integers(*_N_PLUS_RANGE))

    out = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=int)
    plus_occ = np.zeros((H, W), dtype=int)

    pluses: list[dict] = []
    for _ in range(N):
        info = _place_plus_and_ray(
            rng,
            plus_shape,
            palette,
            H,
            W,
            occ,
            plus_occ,
            out,
            plus_color,
        )
        if info is None:
            return None
        pluses.append(info)

    assignments = _choose_wrong_assignments(pluses, H, W, rng)
    if assignments is None:
        return None

    inp = out.copy()
    for idx, cand in assignments.items():
        for r, c, _ in pluses[idx]["correct_extra"]:
            inp[r, c] = bg
        for r, c, v in cand["extra"]:
            inp[r, c] = int(v)

    meta = {
        "wrong_count": len(assignments),
        "correct_count": len(pluses) - len(assignments),
        "wrong_dirs": [tuple(assignments[i]["direction"]) for i in sorted(assignments)],
        "true_dirs": [tuple(info["direction"]) for info in pluses],
        "plus_count": len(pluses),
    }
    return inp, out, meta


def _gen_attempt(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, dict] | None:
    built = _build_variant_pair(rng)
    if built is None:
        return None
    inp, out, meta = built

    try:
        produced = main(inp)
    except Exception:
        return None
    if not np.array_equal(produced, out):
        return None
    return inp, out, meta


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        attempt = _gen_attempt(rng)
        if attempt is not None:
            return attempt[0]
    raise RuntimeError("generate_input: failed after %d attempts" % _GEN_RETRIES)
