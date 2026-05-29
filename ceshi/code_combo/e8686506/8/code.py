# Combo variant of e8686506 (index 8)
# combo_spec: D2/0+D11/0
# Name: 多色复合碎片+两阶段拼合
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保留原题的终局轮廓：输出仍是一个紧凑裁剪的 5 列竖向矩形，并保持严格左右镜像。
#    - 融合后的输入对象不再是“单色碎片”也不再是“颜色已经代表最终块”，而是：
#      每个散落碎片自身就是一个 4 连通双色复合块，颜色分成外框色和内填色；
#      同外框色的 2-3 个碎片先要在第一阶段拼成一个更大的子组件；
#      再由这些子组件在第二阶段拼成最终矩形。
#    - 小例子：
#      原题里，顶部长条可能直接是一整块单色 3x2 区域。
#      D11 部分要求它先被切成两段同色碎片，例如上排 `[r r r]` 和下排 `[r r r]`。
#      D2 部分再把每段变成双色，例如 `[r b r]` 和一排里中间列保留为 `b`。
#      因此 solver 不能按颜色逐块切，也不能把 `b` 当成另一块；它必须先把每段双色整块识别出来，
#      提取其支撑形状和外框色，然后在同外框色组内枚举拼接候选，恢复 3x2 子组件，
#      最后再利用框架空腔把该子组件放回终局。
#    - 差异的核心不是“多一个装饰色”或“多一个分组色”，而是两者联动：
#      第一阶段的分组键来自碎片的外框色，而内填色只是随着该碎片一起搬运到第二阶段。
#
# b) 复杂度校准
#    - 终局结构沿用 D11 的“框架 + 三个空腔子组件”，这样第二阶段有明确的层次，不会退化回原题的
#      直接全局拼图。
#    - 每个最终子组件都切成 2-3 个碎片，且所有碎片面积都至少为 2，避免 D2 规则在 1x1 退化块上失效。
#    - 输出宽度固定为 5，高度固定为 13。这既保证总面积仍会被“最近平方因子”规则唯一地解析成 13x5，
#      又给中间 1 列空腔和上下 3 列空腔留出足够高度去拆成 2-3 段可见双色碎片。
#    - 外框色固定为 4 组，内填色只从额外 2-3 个颜色里复用，控制总色数，避免视觉噪声。
#
# c) 隐式假设审查
#    - 原题 `components_mask(g == col)` 默认“一个碎片只有一种颜色”；融合后改为
#      `components_mask(g != bg)` 先提整块，再从形状推断哪些格属于内填。
#    - 原 D11 假设“同色=同组”；融合后必须改成“同外框色=同组”，因为同组碎片内部还带着第二种颜色。
#    - 原 D2 可以依赖预对齐 frame；原 D11 则不能。融合后采用 D11 的更强设定：所有碎片都散落，
#      frame 也必须先在第一阶段组装出来。
#    - 第二阶段不能只按子组件支撑形状输出单色块；必须把第一阶段每个碎片携带的逐格 paint 一起带回终局，
#      否则就退化成“先分彩色再丢掉颜色”的假规则。
#    - 新增假设：
#      * 每个非背景 4 连通块恰有两种颜色，且它们能被同一个形状规则 `_infer_fill_rel_coords` 稳定分离；
#      * 同外框色组恰有 2-3 个碎片；
#      * 其中恰有一组能拼成完整 LR 对称框架，其余各组分别恰好匹配该框架的一个空腔。
#      这些都由生成器构造保证。
#
# d) 放置合法性
#    - 生成器先在终局矩形内构造 4 个最终子组件，再把每个子组件按连续行带切分成 2-3 个 4 连通碎片。
#      行带切分天然不会重叠、不会越界，且每段都留在原组件内部。
#    - 所有散落碎片最终用 `random_free_location(..., padding=1, connectivity=8)` 投放到 13-15 的背景画布上，
#      确保碎片之间不重叠且至少有 1 格切比雪夫间隔，因此 solver 的非背景 4 连通提取与真实碎片一一对应。
#
# e) 视觉可观察性
#    - 输入里可以直接看到两件事同时发生：
#      * 同外框色的碎片重复出现，但它们还没组成最终大块；
#      * 每个碎片内部都存在“外框色 + 内填色”的双色结构。
#    - 输出里这些双色图样被原样带到终局矩形里，并且相同外框色的多个碎片会相邻组成一个更大的子组件。
#      因此人类可从少量样例看出：先认“哪几块属于同一组”，再把组内花纹整体搬回去。
#    - 规则不依赖具体颜色编号大小；颜色只通过可见的“在同一碎片的外层/内层位置”参与推理。
#
# f) 可视化整洁度
#    - 终局只保留 4 个大区域：一个框架、两个 3 列腔体、一个中轴竖腔。区域边界清楚，不会出现多层覆盖。
#    - 双色只发生在碎片内部，不再额外叠加独立标记层。行带切分使碎片纹理有秩序，不会产生密集锯齿噪声。
#
# g) 变化可推断性
#    - 没有隐藏颜色映射。内填色不是“被转换后才有意义”，而是碎片自身的一部分 payload。
#    - 第一阶段的分组信号是外框色，第二阶段的定位信号是子组件支撑形状与框架空腔匹配。
#      颜色种类被限制在 6-7 种总非背景色以内，且填充色与外框色分池，避免解题者把内填色误读成分组键。
#
# New solver/generator assumption review
# ------------------------------------
# - `main()` 的输入前提全部由 `generate_input()` 保证：
#   - 非背景 4 连通块就是单个双色碎片，不会被别的碎片粘连；
#   - 同一形状的内填格推断规则在 solver 和 generator 中共享，不会出现“同形状不同着色语法”；
#   - 每个外框色组都恰有 2-3 块，且其所有连接候选中恰有一个能在第二阶段参与唯一终局；
#   - 框架组的候选空腔形状互不相同，因而第二阶段的空腔匹配不需要依赖颜色编号顺序。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_WIDTH = 5
_HEIGHT_OPTIONS = [13]
_CANVAS_MIN = 16
_CANVAS_MAX = 18
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 600
_CANDIDATE_LIMIT = 1200


def _norm_coords_from_iter(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _normalize_paint(paint) -> tuple[tuple[int, int, int], ...]:
    pts = [(int(r), int(c), int(col)) for r, c, col in paint]
    r0 = min(r for r, _, _ in pts)
    c0 = min(c for _, c, _ in pts)
    return tuple(sorted((r - r0, c - c0, col) for r, c, col in pts))


def _shape_dims(coords: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return 1 + max(r for r, _ in coords), 1 + max(c for _, c in coords)


def _factors(n: int) -> list[tuple[int, int]]:
    out = []
    for w in range(1, n + 1):
        if n % w == 0:
            h = n // w
            if h >= w:
                out.append((h, w))
    return out


def _coords_to_mask(
    coords: tuple[tuple[int, int], ...],
    h: int | None = None,
    w: int | None = None,
) -> np.ndarray:
    if h is None or w is None:
        hh, ww = _shape_dims(coords)
        if h is None:
            h = hh
        if w is None:
            w = ww
    mask = np.zeros((int(h), int(w)), dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _is_lr_symmetric(
    coords: tuple[tuple[int, int], ...],
    h: int | None = None,
    w: int | None = None,
) -> bool:
    mask = _coords_to_mask(coords, h, w)
    return np.array_equal(mask, mask[:, ::-1])


def _is_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    arr = np.array(coords, dtype=int)
    r0, c0, r1, c1 = bbox_coords(arr)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[int(r) - r0, int(c) - c0] = True
    return len(components_mask(mask, connectivity=4)) == 1


def _infer_fill_rel_coords(
    coords: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], str]:
    arr = np.array(coords, dtype=int)
    h = int(arr[:, 0].max() + 1)
    w = int(arr[:, 1].max() + 1)

    strict = tuple(
        sorted((r, c) for r, c in coords if 0 < r < h - 1 and 0 < c < w - 1)
    )
    if strict:
        return strict, "strict"

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    fallback = tuple(sorted((r, c) for r, c in coords if (r, c) not in corners))
    if fallback:
        return fallback, "fallback"

    assert len(coords) > 1, "singleton fragments cannot encode a two-color structure"
    carry = tuple(sorted(coords))[len(coords) // 2]
    return (carry,), "carry"


def _piece_from_component(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    r0, c0, r1, c1 = bbox_coords(pts)
    rel_coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    fill_rel, fill_mode = _infer_fill_rel_coords(rel_coords)
    fill_set = set(fill_rel)

    color_at = {(rr, cc): int(grid[r0 + rr, c0 + cc]) for rr, cc in rel_coords}
    border_colors = {color_at[pos] for pos in rel_coords if pos not in fill_set}
    fill_colors = {color_at[pos] for pos in fill_rel}

    assert len(border_colors) == 1, "fragment border cells must share one outline color"
    assert len(fill_colors) == 1, "fragment fill cells must share one inner color"

    border_color = next(iter(border_colors))
    fill_color = next(iter(fill_colors))
    assert border_color != bg and fill_color != bg and border_color != fill_color

    paint = tuple(sorted((rr, cc, color_at[(rr, cc)]) for rr, cc in rel_coords))
    return {
        "coords": rel_coords,
        "paint": paint,
        "fill_rel": fill_rel,
        "fill_mode": fill_mode,
        "border_color": int(border_color),
        "fill_color": int(fill_color),
        "area": len(rel_coords),
        "h": int(r1 - r0 + 1),
        "w": int(c1 - c0 + 1),
    }


def _candidate_assemblies(fragments: list[dict]) -> list[dict]:
    ordered = list(fragments)
    ordered.sort(key=lambda frag: (-len(frag["coords"]), frag["coords"], frag["paint"]))
    base = ordered[0]
    results: dict[tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int, int], ...]], dict] = {}
    dirs4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def rec(
        union_cells: set[tuple[int, int]],
        union_paint: tuple[tuple[int, int, int], ...],
        remaining: tuple[int, ...],
    ) -> None:
        if len(results) > _CANDIDATE_LIMIT:
            return
        if not remaining:
            coords = _norm_coords_from_iter(union_cells)
            paint = _normalize_paint(union_paint)
            key = (coords, paint)
            if key not in results:
                h, w = _shape_dims(coords)
                results[key] = {"coords": coords, "paint": paint, "h": h, "w": w}
            return

        for idx in sorted(remaining, key=lambda j: (-len(ordered[j]["coords"]), ordered[j]["coords"])):
            frag = ordered[idx]
            translations = set()
            for fr, fc in frag["coords"]:
                for ur, uc in union_cells:
                    for dr, dc in dirs4:
                        translations.add((ur + dr - fr, uc + dc - fc))
            for tr, tc in translations:
                placed = {(r + tr, c + tc) for r, c in frag["coords"]}
                if placed & union_cells:
                    continue
                if not any((r + dr, c + dc) in union_cells for r, c in placed for dr, dc in dirs4):
                    continue
                merged = union_cells | placed
                norm = _norm_coords_from_iter(merged)
                h, w = _shape_dims(norm)
                if h > max(_HEIGHT_OPTIONS) or w > _WIDTH:
                    continue
                shifted_paint = tuple((r + tr, c + tc, col) for r, c, col in frag["paint"])
                rec(merged, union_paint + shifted_paint, tuple(j for j in remaining if j != idx))
                if len(results) > _CANDIDATE_LIMIT:
                    return

    rec(set(base["coords"]), tuple(base["paint"]), tuple(range(1, len(ordered))))
    assert results, "each outline-color group must have at least one connected assembly candidate"
    return sorted(results.values(), key=lambda cand: (len(cand["coords"]), cand["h"], cand["w"], cand["paint"]))


def _hole_specs(frame_coords: tuple[tuple[int, int], ...], h: int, w: int) -> list[dict]:
    mask = _coords_to_mask(frame_coords, h, w)
    holes = components_mask(~mask, connectivity=4)
    specs = []
    for pts in holes:
        abs_coords = tuple(sorted((int(r), int(c)) for r, c in pts))
        specs.append({"shape": _norm_coords_from_iter(abs_coords), "coords": abs_coords})
    return specs


def _build_board(
    frame_cand: dict,
    assignment: dict[int, tuple[dict, dict]],
    H: int,
    W: int,
) -> np.ndarray:
    board = np.full((H, W), -1, dtype=int)
    for r, c, col in frame_cand["paint"]:
        board[int(r), int(c)] = int(col)

    for hole_spec, cand in assignment.values():
        hr0 = min(r for r, _ in hole_spec["coords"])
        hc0 = min(c for _, c in hole_spec["coords"])
        for r, c, col in cand["paint"]:
            rr = hr0 + int(r)
            cc = hc0 + int(c)
            assert board[rr, cc] == -1, "subcomponent paint must only fill frame holes"
            board[rr, cc] = int(col)

    assert np.all(board >= 0), "all output cells must be painted"
    assert np.array_equal(board, board[:, ::-1]), "assembled board must be LR-symmetric"
    return board


def _row_runs(row: np.ndarray) -> int:
    vals = [int(x) for x in row.tolist()]
    return 1 + sum(vals[i] != vals[i - 1] for i in range(1, len(vals)))


def _board_rank(board: np.ndarray) -> tuple[int, int, int, int]:
    return (
        _row_runs(board[0]),
        _row_runs(board[-1]),
        _row_runs(board[1]),
        _row_runs(board[-2]),
    )


def _solve_from_candidates(
    candidates_by_border: dict[int, list[dict]],
    H: int,
    W: int,
) -> np.ndarray:
    solutions: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}

    for frame_border, cand_list in sorted(candidates_by_border.items()):
        other_borders = [b for b in sorted(candidates_by_border) if b != frame_border]
        for frame_cand in cand_list:
            if (frame_cand["h"], frame_cand["w"]) != (H, W):
                continue
            if not _is_lr_symmetric(frame_cand["coords"], H, W):
                continue

            hole_specs = _hole_specs(frame_cand["coords"], H, W)
            if len(hole_specs) != len(other_borders):
                continue
            if len({spec["shape"] for spec in hole_specs}) != len(hole_specs):
                continue
            ordered_holes = sorted(hole_specs, key=lambda spec: min(r for r, _ in spec["coords"]))
            if tuple(len(spec["coords"]) for spec in ordered_holes) != (9, 4, 6):
                continue

            options: dict[int, list[tuple[dict, dict]]] = {}
            valid = True
            for border in other_borders:
                matches = []
                for spec in ordered_holes:
                    for cand in candidates_by_border[border]:
                        if cand["coords"] == spec["shape"]:
                            matches.append((spec, cand))
                if not matches:
                    valid = False
                    break
                options[border] = matches
            if not valid:
                continue

            order = sorted(other_borders, key=lambda border: (len(options[border]), border))
            assignment: dict[int, tuple[dict, dict]] = {}
            used_holes: set[tuple[tuple[int, int], ...]] = set()

            def rec(i: int) -> None:
                if i == len(order):
                    board = _build_board(frame_cand, assignment, H, W)
                    solutions[tuple(map(tuple, board.tolist()))] = board
                    return
                border = order[i]
                for spec, cand in options[border]:
                    hole_key = spec["coords"]
                    if hole_key in used_holes:
                        continue
                    assignment[border] = (spec, cand)
                    used_holes.add(hole_key)
                    rec(i + 1)
                    used_holes.remove(hole_key)
                    del assignment[border]

            rec(0)

    assert solutions, "fused variant must yield at least one final rectangle"
    ranked = sorted(solutions.values(), key=_board_rank)
    return ranked[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered two-color scraps by first merging outline-color groups into subcomponents, then fitting those subcomponents into a symmetric rectangle.

    Input structure:
    - The majority color is background.
    - Every non-background 4-connected component is one composite fragment containing exactly two colors: an outline color and an inner carried-along color.
    - Fragments sharing the same outline color belong to one intermediate subcomponent, and each outline-color group contains 2-3 fragments.

    Rule:
    - Extract each fragment as a whole non-background component, infer its outline/fill split from the fragment shape, and group fragments by outline color.
    - Inside each outline-color group, enumerate connected assemblies of the fragment support shapes; these are the stage-1 subcomponent candidates.
    - Infer the output rectangle size from total foreground area, find the stage-1 candidate that forms the full left-right symmetric frame, then place the remaining assembled subcomponents into that frame's holes while preserving every fragment's internal colors.

    Key insight:
    - The outline layer tells you which scraps must be merged first, while the fill layer is not a grouping key at all: it is payload that rides along once the correct support assembly has been found.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    groups: dict[int, list[dict]] = {}
    total_area = 0
    for pts in components_mask(g != bg, connectivity=4):
        frag = _piece_from_component(g, pts, bg)
        groups.setdefault(int(frag["border_color"]), []).append(frag)
        total_area += int(frag["area"])

    assert total_area > 0, "grid must contain composite foreground fragments"
    for border, frags in groups.items():
        assert 2 <= len(frags) <= 3, f"outline-color group {border} must contain 2-3 fragments"

    dims = _factors(total_area)
    best_gap = min(abs(h - w) for h, w in dims)
    best = [(h, w) for h, w in dims if abs(h - w) == best_gap]
    assert len(best) == 1, "closest-to-square portrait factorization must be unique"
    H, W = best[0]
    assert W == _WIDTH and H in _HEIGHT_OPTIONS, "variant outputs use the fixed 5-column portrait family"

    candidates_by_border = {}
    for border, frags in groups.items():
        cands = _candidate_assemblies(frags)
        cands = [cand for cand in cands if cand["h"] <= H and cand["w"] <= W]
        assert cands, "an outline-color group has no assembly candidate that fits the output box"
        candidates_by_border[int(border)] = cands

    sol = _solve_from_candidates(candidates_by_border, H, W)
    return sol.astype(int)


# --------------- generator ---------------


def _abstract_support(h: int) -> np.ndarray:
    assert h == 13, "fused variant uses the smallest uniquely factorized height that can still split every final subcomponent into 2+ visible two-color fragments"
    top_h, mid_h, bot_h = 3, 4, 2

    rows = []
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, 1, 1, 1, 0] for _ in range(top_h)])
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, 0, 2, 0, 0] for _ in range(mid_h)])
    rows.append([0, 0, 0, 0, 0])
    rows.extend([[0, 3, 3, 3, 0] for _ in range(bot_h)])
    rows.append([0, 0, 0, 0, 0])
    out = np.array(rows, dtype=int)

    for comp_id in range(4):
        comps = components_mask(out == comp_id, connectivity=4)
        assert len(comps) == 1, "each abstract component must stay connected"
    return out


def _compositions(total: int, k: int, min_part: int = 1) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(rem: int, left: int, acc: list[int]) -> None:
        if left == 1:
            if rem >= min_part:
                out.append(tuple(acc + [rem]))
            return
        max_first = rem - min_part * (left - 1)
        for first in range(min_part, max_first + 1):
            rec(rem - first, left - 1, acc + [first])

    rec(int(total), int(k), [])
    return out


def _valid_band_partitions(coords_abs: tuple[tuple[int, int], ...], k: int) -> list[list[tuple[tuple[int, int], ...]]]:
    rows = sorted({r for r, _ in coords_abs})
    r0, r1 = min(rows), max(rows)
    assert rows == list(range(r0, r1 + 1)), "component rows must be contiguous for row-band splitting"
    total_rows = r1 - r0 + 1
    choices: list[list[tuple[tuple[int, int], ...]]] = []
    for lengths in _compositions(total_rows, k, min_part=1):
        start = r0
        parts = []
        ok = True
        for length in lengths:
            end = start + length - 1
            part = tuple(sorted((r, c) for r, c in coords_abs if start <= r <= end))
            if len(part) < 2 or not _is_connected(part):
                ok = False
                break
            parts.append(part)
            start = end + 1
        if ok:
            choices.append(parts)
    return choices


def _choose_partitions(pid_grid: np.ndarray) -> dict[int, list[tuple[tuple[int, int], ...]]] | None:
    coords_by_id = {
        comp_id: tuple(sorted((int(r), int(c)) for r, c in np.argwhere(pid_grid == comp_id)))
        for comp_id in range(4)
    }
    valid_by_id: dict[int, dict[int, list[list[tuple[tuple[int, int], ...]]]]] = {}
    for comp_id, coords_abs in coords_by_id.items():
        valid_by_id[comp_id] = {}
        for k in (2, 3):
            valid_by_id[comp_id][k] = _valid_band_partitions(coords_abs, k)
        if not valid_by_id[comp_id][2]:
            return None

    can_three = [comp_id for comp_id in range(4) if valid_by_id[comp_id][3]]
    promote_three = random.random() < 0.7 and bool(can_three)
    three_id = random.choice(can_three) if promote_three else None

    chosen = {}
    for comp_id in range(4):
        k = 3 if comp_id == three_id else 2
        chosen[comp_id] = random.choice(valid_by_id[comp_id][k])
    return chosen


def _paint_piece(
    out: np.ndarray,
    cells_abs: tuple[tuple[int, int], ...],
    border_color: int,
    fill_color: int,
) -> dict:
    norm = _norm_coords_from_iter(cells_abs)
    fill_rel, fill_mode = _infer_fill_rel_coords(norm)
    fill_set = set(fill_rel)
    r0 = min(r for r, _ in cells_abs)
    c0 = min(c for _, c in cells_abs)

    paint_abs = []
    for rr, cc in norm:
        color = fill_color if (rr, cc) in fill_set else border_color
        out[r0 + rr, c0 + cc] = int(color)
        paint_abs.append((r0 + rr, c0 + cc, int(color)))

    return {
        "coords": norm,
        "paint": _normalize_paint(paint_abs),
        "cells_abs": cells_abs,
        "r0": int(r0),
        "c0": int(c0),
        "h": _shape_dims(norm)[0],
        "w": _shape_dims(norm)[1],
        "area": len(norm),
        "fill_mode": fill_mode,
        "border_color": int(border_color),
        "fill_color": int(fill_color),
    }


def _make_output() -> tuple[np.ndarray | None, int, list[dict] | None]:
    h = random.choice(_HEIGHT_OPTIONS)
    pid_grid = _abstract_support(h)
    parts_by_id = _choose_partitions(pid_grid)
    if parts_by_id is None:
        return None, 0, None

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)
    border_colors = palette[:4]
    fill_pool = palette[4:7]
    if len(fill_pool) < 2:
        return None, 0, None

    fill_colors = []
    prev = None
    for _ in range(4):
        opts = [c for c in fill_pool if c != prev]
        col = random.choice(opts or fill_pool)
        fill_colors.append(col)
        prev = col

    out = np.full(pid_grid.shape, -1, dtype=int)
    pieces = []
    for comp_id in range(4):
        border_color = int(border_colors[comp_id])
        fill_color = int(fill_colors[comp_id])
        for cells_abs in parts_by_id[comp_id]:
            piece = _paint_piece(out, cells_abs, border_color, fill_color)
            piece["is_frame"] = comp_id == 0
            pieces.append(piece)

    if np.any(out < 0):
        return None, 0, None
    if not np.array_equal(out, out[:, ::-1]):
        return None, 0, None
    return out, bg, pieces


def _patch_from_piece(out: np.ndarray, piece: dict, bg: int) -> np.ndarray:
    patch = np.full((int(piece["h"]), int(piece["w"])), bg, dtype=int)
    for r, c in piece["cells_abs"]:
        patch[int(r) - int(piece["r0"]), int(c) - int(piece["c0"])] = int(out[int(r), int(c)])
    return patch


def _scatter_pieces(out: np.ndarray, bg: int, pieces: list[dict]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    order = list(pieces)
    random.shuffle(order)
    order.sort(key=lambda piece: (-int(piece["area"]), random.random()))
    for piece in order:
        patch = _patch_from_piece(out, piece, bg)
        r, c = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r, c, transparent=bg)
    return canvas


def _variant_stats_from_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    comps = components_mask(g != bg, connectivity=4)
    groups = Counter()
    two_color = 0
    fill_modes = Counter()
    fill_colors = set()
    for pts in comps:
        piece = _piece_from_component(g, pts, bg)
        groups[int(piece["border_color"])] += 1
        if len({int(g[r, c]) for r, c in pts}) == 2:
            two_color += 1
        fill_modes[piece["fill_mode"]] += 1
        fill_colors.add(int(piece["fill_color"]))
    return {
        "pieces": len(comps),
        "two_color_pieces": two_color,
        "outline_groups": len(groups),
        "group_sizes": tuple(sorted(groups.values())),
        "hierarchy_gap": len(comps) - len(groups),
        "fill_modes": dict(fill_modes),
        "distinct_fill_colors": len(fill_colors),
    }


def _gen_attempt() -> np.ndarray | None:
    out, bg, pieces = _make_output()
    if out is None or pieces is None:
        return None

    groups = Counter(int(piece["border_color"]) for piece in pieces)
    if sorted(groups.values()) not in ([2, 2, 2, 2], [2, 2, 2, 3]):
        return None

    try:
        inp = _scatter_pieces(out, bg, pieces)
        got = main(inp)
    except Exception:
        return None

    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fused e8686506 input")
