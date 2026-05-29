# Combo variant of 21897d95 (index 1)
# combo_spec: D1/0+D10/0
# Name: 顺序交换链
#
# design_analysis:
#   concretization:
#     - 画面外观继续沿用原题: 不规则单色区域拼成马赛克, 区域边界上放 T, 可选地在某个区域里嵌一个 9 色 L 旋转标记。
#     - 融合后的变化集中在 T 的语义: 每个 T 仍然有“源侧”和“目标侧”, 但它不再把颜色单向写入目标区, 而是交换两侧区域的当前颜色。
#     - 同时, 这些交换不是并行发生, 而是按 T 的中心坐标做从左上到右下的扫描顺序依次执行。
#     - 例子: 三个区域 A/B/C 的输入色是 2/3/4, 两个 T 依次表示 A↔B、B↔C。
#       原题: B 变成 2, C 变成 3, A 不变, 输出是 2/2/3。
#       只做交换但不强调链时: 仍要问“先换哪一个”; 融合版明确规定先 A↔B 得到 3/2/4, 再 B↔C 得到 3/4/2。
#       关键差异是第二个 T 看到的 B 已经不是原色, 而是第一步交换后的当前颜色。
#   complexity_calibration:
#     - 如果每个区域只在一个 T 里出现一次, 题面会退化成若干互不相关的二元换色, 顺序信息几乎不可见。
#     - 因此生成器强制选择一条长度 4-6 的相邻区域交换游走, 让至少一个区域参与多次交换, 形成真正的链式传播。
#     - 同时不再生成 colored-center T; 否则同一题里会混入“字面颜色提示”和“区域交换”两套规则, 反而削弱观察重点。
#   implicit_assumptions_review:
#     - 原 solver 的 `replacement = root_color[source]` 假设源色永远读取初始颜色; 在顺序交换下失效, 必须读取可变的 `current_color[source]`。
#     - 原 solver 的 `final_color[dest] = replacement` 假设只有目标区变化; 在交换规则下也失效, 因为源区和目标区都要同步更新。
#     - 原 solver 对同一目标区的多次控制带有“无冲突覆盖”假设; 现在不再需要这种断言, 因为顺序本身就是语义的一部分。
#     - 标记几何仍必须先整体解析, 不能边交换边重新识别 T, 否则前一步把标记擦成区域色后会破坏后续匹配。
#     - 新 solver 额外假设“每个 T 的源区和目标区是两个不同的基础区域”; 生成器通过边界放置约束保证这一点。
#   placement_legality:
#     - 每个 T 仍满足原题几何: 4 格 T, 不越界, 不重叠, 周围保留缓冲, 茎方向前方第一个基础区域属于目标区, 茎后对角属于源区。
#     - 交换链只使用相邻区域对, 所以每一步都有合法的边界位置可放 T。
#     - 基础底色池显式排除 1 和 9: 1 专供 T, 9 专供 L 标记, 避免控制元素和普通区域混色。
#   visual_observability:
#     - 顺序键完全来自可见几何位置, 不依赖颜色 id、区域编号等人类不可见因素。
#     - 所有基础区域使用互异底色, 所以后续样例中“哪种颜色沿链移动到哪里”可以直接肉眼跟踪。
#     - 生成器会拒绝那些“顺序交换输出 == 顺序单向重着色输出”的输入, 确保 D1 的交换性是可见的。
#     - 生成器也会拒绝那些没有任何后续 T 读取到已变化源色的输入, 确保 D10 的顺序链条确实发生。
#   cleanliness:
#     - 仍只使用大色块区域、T 控制和可选 L9, 不额外叠加纹理或第二套符号系统。
#     - T 周围保留空缓冲, 避免多个标记挤成团, 让解题者能清楚看出每个 T 对应哪一侧是源、哪一侧是目标。
#   color_inference:
#     - 颜色变化完全由区域邻接关系和 T 的空间顺序决定, 不是“某个颜色值触发某种行为”。
#     - 因为使用交换而非复制, 源区自己也会失去原色; 这为观察者提供了比原题更强的结构线索。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(int(n)))

    def find(self, x: int) -> int:
        x = int(x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return int(x)

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _opp(d: tuple[int, int]) -> tuple[int, int]:
    return (-int(d[0]), -int(d[1]))


def _stem_from_branches(branches: list[tuple[int, int]]) -> tuple[int, int]:
    assert len(branches) == 3
    branch_set = set(branches)
    for d in branches:
        if _opp(d) not in branch_set:
            return d
    raise AssertionError("T marker has no single stem")


def _parse_tetromino(
    cells: set[tuple[int, int]],
) -> tuple[tuple[int, int], tuple[int, int], list[tuple[int, int]]] | None:
    if len(cells) != 4:
        return None
    for r, c in cells:
        branches = [(dr, dc) for dr, dc in DIRS if (r + dr, c + dc) in cells]
        if len(branches) != 3:
            continue
        stem = _stem_from_branches(branches)
        expected = {(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)}
        expected.remove((r - stem[0], c - stem[1]))
        if cells == expected:
            bar = [d for d in branches if d != stem]
            assert len(bar) == 2 and bar[0] == _opp(bar[1])
            return (int(r), int(c)), stem, bar
    return None


def _label_base_components(
    g: np.ndarray, control: np.ndarray
) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(col))
            comp_areas.append(int(len(coords)))
            cid += 1
    return comp_id, comp_colors, comp_areas


def _parse_scene(g: np.ndarray) -> dict:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {k: len(cells) for k, cells in one_components.items()}

    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)

    for cells in one_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        markers.append(
            {
                "kind": "normal",
                "center": center,
                "stem": stem,
                "bar": bar,
                "cells": cells,
                "order_key": center,
            }
        )
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            branches: list[tuple[int, int]] = []
            cells: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = r + d[0], c + d[1]
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != 1:
                    continue
                label = int(labels_1[rr, cc])
                if one_sizes[label] == 1 and (rr, cc) not in used_ones:
                    branches.append(d)
                    cells.append((rr, cc))
                else:
                    ok = False
            if ok and len(branches) == 3:
                stem = _stem_from_branches(branches)
                bar = [d for d in branches if d != stem]
                assert len(bar) == 2 and bar[0] == _opp(bar[1])
                marker_cells = set(cells) | {(int(r), int(c))}
                markers.append(
                    {
                        "kind": "colored",
                        "center": (int(r), int(c)),
                        "stem": stem,
                        "bar": bar,
                        "cells": marker_cells,
                        "center_color": int(g[r, c]),
                        "order_key": (int(r), int(c)),
                    }
                )
                used_ones |= set(cells)
                for rr, cc in marker_cells:
                    control[rr, cc] = True

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a T control marker"

    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == col) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == k)
            if len(coords) > 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if len(coords) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
                continue
            cells = {tuple(map(int, xy)) for xy in coords}
            markers.append({"kind": "small", "cells": cells})
            for rr, cc in cells:
                control[rr, cc] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def first_component_from(
        center: tuple[int, int], direction: tuple[int, int], start: int
    ) -> int | None:
        r, c = center
        step = int(start)
        while True:
            rr = r + direction[0] * step
            cc = c + direction[1] * step
            if not _in_bounds(g.shape, rr, cc):
                return None
            if comp_id[rr, cc] >= 0:
                return int(comp_id[rr, cc])
            step += 1

    def adjacent_component_counts(cells: set[tuple[int, int]]) -> Counter:
        counts: Counter = Counter()
        for r, c in cells:
            for dr, dc in DIRS:
                rr, cc = r + dr, c + dc
                if _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0:
                    counts[int(comp_id[rr, cc])] += 1
        return counts

    for marker in markers:
        if marker["kind"] in {"normal", "colored"}:
            center = marker["center"]
            stem = marker["stem"]
            dest = first_component_from(center, stem, start=2)
            assert dest is not None, "T marker must point at a destination region"

            source_parts: list[int] = []
            for bar_dir in marker["bar"]:
                rr = center[0] + stem[0] + bar_dir[0]
                cc = center[1] + stem[1] + bar_dir[1]
                assert _in_bounds(g.shape, rr, cc) and comp_id[rr, cc] >= 0
                source_parts.append(int(comp_id[rr, cc]))
            source = source_parts[0]
            for part in source_parts[1:]:
                assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
                dsu.union(source, part)

            for comp in adjacent_component_counts(marker["cells"]):
                if comp_colors[comp] == comp_colors[source]:
                    dsu.union(source, comp)
            marker["dest_comp"] = dest
            marker["source_comp"] = source
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch its owner region"
            owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[owner]:
                    dsu.union(owner, comp)
            marker["source_comp"] = owner

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    for marker in markers:
        marker["source_root"] = dsu.find(marker["source_comp"])
        if marker["kind"] in {"normal", "colored"}:
            marker["dest_root"] = dsu.find(marker["dest_comp"])

    return {
        "shape": g.shape,
        "markers": markers,
        "comp_id": comp_id,
        "dsu": dsu,
        "root_color": root_color,
    }


def _ordered_t_markers(scene: dict) -> list[dict]:
    return sorted(
        [m for m in scene["markers"] if m["kind"] in {"normal", "colored"}],
        key=lambda m: m["order_key"],
    )


def _apply_sequential_swaps(scene: dict) -> dict[int, int]:
    current_color = dict(scene["root_color"])
    for marker in _ordered_t_markers(scene):
        source = int(marker["source_root"])
        dest = int(marker["dest_root"])
        assert source != dest, "T marker must separate two distinct regions"
        current_color[source], current_color[dest] = current_color[dest], current_color[source]
    return current_color


def _apply_copy_mode(scene: dict, *, sequential: bool) -> dict[int, int]:
    current_color = dict(scene["root_color"])
    ordered = _ordered_t_markers(scene)
    for marker in ordered:
        source = int(marker["source_root"])
        dest = int(marker["dest_root"])
        if marker["kind"] == "colored":
            replacement = int(marker["center_color"])
        elif sequential:
            replacement = int(current_color[source])
        else:
            replacement = int(scene["root_color"][source])
        current_color[dest] = replacement
    return current_color


def _render_scene(scene: dict, final_color: dict[int, int]) -> np.ndarray:
    comp_id = scene["comp_id"]
    dsu = scene["dsu"]
    out = np.full(scene["shape"], -1, dtype=int)
    for r in range(scene["shape"][0]):
        for c in range(scene["shape"][1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in scene["markers"]:
        color = int(final_color[marker["source_root"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
    scene = _parse_scene(g)
    final_color = _apply_sequential_swaps(scene)
    return _render_scene(scene, final_color)


def _find_l9_orientation(g: np.ndarray) -> str | None:
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == k)}
        if len(coords) != 5:
            continue
        rows = [r for r, _c in coords]
        cols = [c for _r, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in [(r0, c0), (r0, c1), (r1, c0), (r1, c1)]:
            h_nb = (cr, cc + 1 if cc == c0 else cc - 1)
            v_nb = (cr + 1 if cr == r0 else cr - 1, cc)
            if h_nb in coords and v_nb in coords:
                if cr == r0 and cc == c1:
                    return "top_right"
                if cr == r0 and cc == c0:
                    return "top_left"
                assert False, f"unobserved L9 corner at ({cr},{cc})"
    return None


def _solve_with_rule(
    grid: np.ndarray,
    *,
    mode: str,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    if mode == "sequential_swap":
        final_color = _apply_sequential_swaps(scene)
    elif mode == "sequential_copy":
        final_color = _apply_copy_mode(scene, sequential=True)
    elif mode == "simultaneous_copy":
        final_color = _apply_copy_mode(scene, sequential=False)
    else:
        raise ValueError(mode)
    out = _render_scene(scene, final_color)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange region colors across T-shaped controls in scan order, erase the controls, and optionally rotate the finished image with a color-9 L marker.

    Input structure:
    - The grid is partitioned into irregular monochrome regions.
    - Color 1 forms T-shaped markers on boundaries between regions.
    - A small non-rectangular scrap may appear and is absorbed into its neighboring region, matching the base task's cleanup behavior.
    - An optional 5-cell L-shaped marker of color 9 sits inside one region and encodes a 90-degree rotation of the final output.

    Rule:
    - For each T, the single stem points toward the target-side region, while the two diagonal cells behind the stem identify the source-side region that owns the marker.
    - Read all T markers by their centers from top-left to bottom-right.
    - Each T swaps the current colors of its source and target regions; later T markers therefore act on colors that may already have moved earlier in the chain.
    - After all swaps, every marker cell is filled with the final color of its source-side region.
    - The color-9 L marker rotates the whole output: a top-right corner means CCW 90 degrees, and a top-left corner means CW 90 degrees.

    Key insight:
    - A T no longer performs one-way repainting. It is an ordered swap instruction, so the visible position of the T markers determines a chain of color exchanges across multiple regions.
    """
    return _solve_with_rule(grid, mode="sequential_swap")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    ordered = _ordered_t_markers(scene)
    initial_color = dict(scene["root_color"])
    current_color = dict(initial_color)
    participation = Counter()
    dependent_swaps = 0
    for marker in ordered:
        src = int(marker["source_root"])
        dst = int(marker["dest_root"])
        participation[src] += 1
        participation[dst] += 1
        if current_color[src] != initial_color[src]:
            dependent_swaps += 1
        current_color[src], current_color[dst] = current_color[dst], current_color[src]

    combo_out = _solve_with_rule(g, mode="sequential_swap")
    seq_copy_out = _solve_with_rule(g, mode="sequential_copy")
    sim_copy_out = _solve_with_rule(g, mode="simultaneous_copy")

    changed_regions = sum(1 for root, col in initial_color.items() if current_color[root] != col)
    return {
        "t_markers": len(ordered),
        "dependent_swaps": int(dependent_swaps),
        "max_region_participation": max(participation.values()) if participation else 0,
        "changed_regions": int(changed_regions),
        "diff_vs_sequential_copy": int(np.sum(combo_out != seq_copy_out)),
        "diff_vs_simultaneous_copy": int(np.sum(combo_out != sim_copy_out)),
    }


# --------------- generator (sequential-swap-chain + l9rot) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 6

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 500

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _t_cells(center, stem)
    neighborhood = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    seeds = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier = []
    for i, (r, c) in enumerate(list(seeds)):
        region[r, c] = i
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, i))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if region[r, c] != -1:
            continue
        region[r, c] = rid
        for dr, dc in _DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and region[nr, nc] == -1:
                frontier.append((nr, nc, rid))

    for rid in range(n_regions):
        mask = region == rid
        if int(mask.sum()) < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if int(n) != 1:
            return None
    return region


def _region_adjacency(region: np.ndarray) -> dict[int, set[int]]:
    H, W = region.shape
    adjacency: dict[int, set[int]] = {int(rid): set() for rid in np.unique(region)}
    for r in range(H):
        for c in range(W):
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    a = int(region[r, c])
                    b = int(region[nr, nc])
                    if a != b:
                        adjacency[a].add(b)
    return adjacency


def _choose_marker_count() -> int:
    return random.randint(MARKER_MIN, MARKER_MAX)


def _choose_swap_walk(adjacency: dict[int, set[int]], n_markers: int) -> list[tuple[int, int]] | None:
    nodes = [rid for rid, nbs in adjacency.items() if nbs]
    if len(nodes) < 2:
        return None

    for _ in range(250):
        start = random.choice(nodes)
        path = [start]
        for _step in range(n_markers):
            cur = path[-1]
            options = list(adjacency[cur])
            if not options:
                break
            prev = path[-2] if len(path) >= 2 else None
            non_backtrack = [rid for rid in options if rid != prev]
            if non_backtrack:
                options = non_backtrack
            random.shuffle(options)
            options.sort(key=lambda rid: (path.count(rid), rid))
            path.append(int(options[0]))
        if len(path) != n_markers + 1:
            continue
        pairs = [(int(a), int(b)) for a, b in zip(path, path[1:])]
        participation = Counter()
        for src, dst in pairs:
            participation[src] += 1
            participation[dst] += 1
        if max(participation.values()) < 2:
            continue
        return pairs
    return None


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    H: int,
    W: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
                if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                    continue
                if cells & occupied_cells:
                    continue
                if _t_neighborhood((r, c), stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if not all(
                    0 <= r + stem[0] + bd[0] < H
                    and 0 <= c + stem[1] + bd[1] < W
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == int(src)
                    for bd in bar
                ):
                    continue
                step = 2
                found = False
                while True:
                    nr, nc = r + stem[0] * step, c + stem[1] * step
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if (nr, nc) in occupied_cells or (nr, nc) in cells:
                        step += 1
                        continue
                    found = int(region[nr, nc]) == int(dst)
                    break
                if found:
                    candidates.append(((int(r), int(c)), stem))
    return candidates


def _register_marker(
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> None:
    for _kind, center, stem, _src, _dst, _tc in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _ordered_placed_markers(
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]]:
    return sorted(placed_markers, key=lambda m: (int(m[1][0]), int(m[1][1])))


def _simulate_swap_colors(
    input_colors: list[int],
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> list[int]:
    out_colors = list(map(int, input_colors))
    for _kind, _center, _stem, src, dst, _tc in _ordered_placed_markers(placed_markers):
        out_colors[src], out_colors[dst] = out_colors[dst], out_colors[src]
    return out_colors


def _simulate_copy_colors(
    input_colors: list[int],
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> list[int]:
    out_colors = list(map(int, input_colors))
    for _kind, _center, _stem, src, dst, _tc in _ordered_placed_markers(placed_markers):
        out_colors[dst] = out_colors[src]
    return out_colors


def _stamp_markers_output(
    grid: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    out_colors: list[int],
) -> None:
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = int(out_colors[src])
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    occupied_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells = set()
    for _kind, center, stem, *_rest in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = [(r + dr, c + dc) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in all_marker_cells for cr, cc in cells):
                continue
            if any((cr, cc) in occupied_neighbors for cr, cc in cells):
                continue
            candidates.append((int(r), int(c), rid))

    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = int(out_colors[rid])
    return orient


def _swap_behavior_metrics(
    input_colors: list[int],
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
) -> dict[str, int]:
    ordered = _ordered_placed_markers(placed_markers)
    current = list(map(int, input_colors))
    initial = list(map(int, input_colors))
    participation = Counter()
    dependent_swaps = 0
    for _kind, _center, _stem, src, dst, _tc in ordered:
        participation[src] += 1
        participation[dst] += 1
        if current[src] != initial[src]:
            dependent_swaps += 1
        current[src], current[dst] = current[dst], current[src]
    changed_regions = sum(1 for before, after in zip(initial, current) if before != after)
    seq_copy = _simulate_copy_colors(input_colors, placed_markers)
    diff_vs_seq_copy = sum(1 for a, b in zip(current, seq_copy) if a != b)
    return {
        "dependent_swaps": int(dependent_swaps),
        "max_region_participation": max(participation.values()) if participation else 0,
        "changed_regions": int(changed_regions),
        "region_diff_vs_sequential_copy": int(diff_vs_seq_copy),
    }


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(10, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    adjacency = _region_adjacency(region)
    n_markers = _choose_marker_count()
    marker_pairs = _choose_swap_walk(adjacency, n_markers)
    if marker_pairs is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] = []

    placement_defs = list(marker_pairs)
    random.shuffle(placement_defs)
    for src, dst in placement_defs:
        candidates = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
        if not candidates:
            return None
        center, stem = random.choice(candidates)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append(("normal", center, stem, int(src), int(dst), None))

    metrics = _swap_behavior_metrics(input_colors, placed_markers)
    if metrics["dependent_swaps"] < 1:
        return None
    if metrics["max_region_participation"] < 2:
        return None
    if metrics["changed_regions"] < 3:
        return None
    if metrics["region_diff_vs_sequential_copy"] < 2:
        return None

    out_colors = _simulate_swap_colors(input_colors, placed_markers)
    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(
        region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors
    )
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        result = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(result, output_grid):
        return None

    scene_metrics = variant_metrics(input_grid)
    if scene_metrics["dependent_swaps"] < 1:
        return None
    if scene_metrics["diff_vs_sequential_copy"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
