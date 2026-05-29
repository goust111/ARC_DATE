# Variant of 21897d95
# Dimension: D10 — 重着色的同时性与顺序性 (multi_step)
# Change: 所有T标记的重着色同时执行（simultaneous），不存在处理顺序... → 按T标记在网格中的位置顺序依次重着色

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


# Design analysis
# a) 具体化与行为追踪
#    - 视觉对象保持不变: 输入仍是单色区域马赛克, 区域边界上放置 T 标记, 可选 9 色 L 旋转标记。
#    - 变化只发生在“何时读取源色”: 原题里所有普通 T 都读取源区域的原始颜色; 现在改成按 raster-scan
#      顺序逐个处理, 后面的普通 T 读取的是“当前已经被前面步骤更新过”的区域颜色。
#    - 小例子: 若先有 A->B, 再有 B->C, 且两者都是普通 T:
#      原题: B 变成 A 色, C 仍变成 B 的原始色。
#      变体: B 先变成 A 色, 再处理 B->C 时, C 读取到的是 B 的新色, 因而也变成 A 色。
# b) 复杂度校准
#    - 不引入新的标记形状或颜色编码, 仍只用原题已有 T/L 语法, 复杂度集中在“顺序”。
#    - 生成器强制至少存在 1 条长度 >= 2 的链, 否则样例可能退化为原题的同时赋值。
#    - 为了避免视觉噪声, 彩色中心 T 仍保留但数量收紧; 关键链使用普通 T, 这样链式传色最明显。
# c) 隐式假设审查
#    - 原 solver 的 `replacement = root_color[source]` 假设源色永远取初始色; 变体下不成立, 必须改成
#      读取 `current_color[source]`。
#    - 原 solver 对目标区域的赋值允许“无序同时覆盖”; 变体下若出现重复目标, 合法语义应是后出现者覆盖前者,
#      因而不能再用“冲突 assert”限制顺序写入。
#    - 标记几何识别仍应只基于输入一次性完成, 不能边处理边重新找 T, 否则前一步把标记擦掉会破坏后续解析。
# d) 放置合法性
#    - 每个 T 仍需满足: 三叉形状合法、不越界、不重叠、与其他标记保留缓冲、源对角落在 source 区域、stem 前方
#      的首个非标记基础区域属于 destination。
#    - 生成器额外约束链式两枚普通 T 的扫描顺序: 第一枚的首个像素必须早于第二枚。
# e) 视觉可观察性
#    - 排序不依赖颜色 id, 只依赖像素扫描顺序, 人眼可直接从空间位置看出“先左上后右下”。
#    - 生成器拒绝 `sequential == simultaneous` 的退化样本, 确保训练样例里能看出多步效应。
# f) 可视化整洁度
#    - 区域仍是大块单色, 标记数量控制在 4-6, 并保留缓冲带避免 T 紧贴成团。
#    - 实际上将 9 从区域底色池中移除, 让 9 色 L 标记始终是孤立的可见对象, 不与底色融在一起。
# g) 颜色变化可推断性
#    - 基础区域输入颜色保持互异, 所以“某区域后来变成了谁的颜色”是可观察的。
#    - 顺序链用普通 T 实现, 其结果完全由位置顺序与区域关系决定, 不需要记忆隐式颜色表。


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
UP = (-1, 0)
DOWN = (1, 0)
RIGHT = (0, 1)


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
    cells: set[tuple[int, int]]
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


def _scan_key(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return min(cells)


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
                "order_key": _scan_key(cells),
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
                        "target_color": int(g[r, c]),
                        "order_key": _scan_key(marker_cells),
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


def _apply_recoloring(scene: dict, sequential: bool) -> dict[int, int]:
    current_color = dict(scene["root_color"])
    t_markers = [m for m in scene["markers"] if m["kind"] in {"normal", "colored"}]
    if sequential:
        t_markers = sorted(t_markers, key=lambda m: m["order_key"])
    for marker in t_markers:
        source = marker["source_root"]
        dest = marker["dest_root"]
        if marker["kind"] == "colored":
            replacement = int(marker["target_color"])
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


def _solve_unrotated(g: np.ndarray, *, sequential: bool) -> np.ndarray:
    scene = _parse_scene(g)
    final_color = _apply_recoloring(scene, sequential=sequential)
    return _render_scene(scene, final_color)


def _find_l9_orientation(g: np.ndarray) -> str | None:
    """Find a 5-cell L-shaped color-9 marker and return its corner position."""
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


def _solve_with_mode(grid: np.ndarray, *, sequential: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g, sequential=sequential)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint regions from T-shaped controls one after another, then erase the controls and optionally rotate via a color-9 L-marker.

    Input structure:
    - The grid is partitioned into monochrome regions with irregular borders.
    - Color 1 marks T-shaped controls on those borders.
    - Some controls use a colored center plus three singleton color-1 branches; that center is a literal repaint color.
    - An optional 5-cell color-9 L-shape is embedded inside one region and encodes a 90-degree output rotation.

    Rule:
    - For each T, the stem points toward the destination region, while the two diagonal cells behind the stem identify the source-side region.
    - T markers are processed in top-left to bottom-right scan order, using the first pixel belonging to each T as its position key.
    - A normal T repaints its destination with the source region's current color at that moment; a colored-center T repaints with its center color.
    - After all steps, every marker cell is filled with the final color of its source-side region, and the L-marker rotates the whole output: ┐ → CCW 90°, ┌ → CW 90°.

    Key insight:
    - The geometry of every T is read once from the input, but the color copied by a later T may already have been changed by an earlier T.
    """
    return _solve_with_mode(grid, sequential=True)


def _main_simultaneous(grid: np.ndarray) -> np.ndarray:
    return _solve_with_mode(grid, sequential=False)


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    t_markers = sorted(
        [m for m in scene["markers"] if m["kind"] in {"normal", "colored"}],
        key=lambda m: m["order_key"],
    )
    depth_by_region: dict[int, int] = {}
    dependent_normals = 0
    max_chain_depth = 0
    for marker in t_markers:
        src = marker["source_root"]
        dst = marker["dest_root"]
        prior_depth = depth_by_region.get(src, 0)
        if marker["kind"] == "normal" and prior_depth > 0:
            dependent_normals += 1
        depth_by_region[dst] = max(depth_by_region.get(dst, 0), prior_depth + 1)
        max_chain_depth = max(max_chain_depth, depth_by_region[dst])

    seq = _solve_with_mode(g, sequential=True)
    sim = _solve_with_mode(g, sequential=False)
    return {
        "t_markers": len(t_markers),
        "dependent_normals": int(dependent_normals),
        "max_chain_depth": int(max_chain_depth),
        "diff_cells_vs_simultaneous": int(np.sum(seq != sim)),
    }


# --------------- generator (multi-step l9rot) ---------------

# Canvas / region sizing
GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [5, 6, 7]
REGION_WEIGHTS = [3, 3, 1]

# Marker counts
MARKER_MIN = 4
MARKER_MAX = 6
COLORED_MAX = 1

# Color pool: reserve 1 for T branches and reserve 9 for the L marker.
COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

# L9 rotation marker
L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

# Retry budget
SEED_ATTEMPTS = 200
GEN_RETRIES = 400

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


def _t_branch_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
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


def _placement_key(center: tuple[int, int], stem: tuple[int, int]) -> tuple[int, int]:
    return _scan_key(_t_cells(center, stem))


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


def _find_adjacent_pairs(region: np.ndarray) -> list[tuple[int, int]]:
    H, W = region.shape
    undirected = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in _DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    a = int(region[r, c])
                    b = int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_regions - 1)
    if max_possible < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_possible)
    n_colored = random.randint(0, min(COLORED_MAX, n_markers - 2))
    n_normal = n_markers - n_colored
    return n_markers, n_normal, n_colored


def _choose_chain(directed_pairs: list[tuple[int, int]]) -> tuple[int, int, int] | None:
    by_src: dict[int, set[int]] = {}
    for src, dst in directed_pairs:
        by_src.setdefault(int(src), set()).add(int(dst))
    triples: list[tuple[int, int, int]] = []
    for a, bs in by_src.items():
        for b in bs:
            for c in by_src.get(b, set()):
                if len({a, b, c}) == 3:
                    triples.append((a, b, c))
    if not triples:
        return None
    random.shuffle(triples)
    return triples[0]


def _pick_extra_pairs(
    directed_pairs: list[tuple[int, int]],
    *,
    chain_src: int,
    chain_mid: int,
    chain_dst: int,
    n_needed: int,
) -> list[tuple[int, int]] | None:
    dest_used = {chain_mid, chain_dst}
    random_pairs = list(directed_pairs)
    random.shuffle(random_pairs)
    picked: list[tuple[int, int]] = []
    for src, dst in random_pairs:
        if dst in dest_used or dst == chain_src:
            continue
        if src in {chain_mid, chain_dst}:
            continue
        picked.append((int(src), int(dst)))
        dest_used.add(int(dst))
        if len(picked) == n_needed:
            return picked
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
                    candidates.append(((r, c), stem))
    return candidates


def _register_marker(
    kind: str,
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _t_cells(center, stem)
    if kind == "normal":
        occupied_cells |= cells
        occupied_neighbors |= (_t_neighborhood(center, stem) - cells)
    else:
        branch = _t_branch_cells(center, stem)
        occupied_cells |= branch
        occupied_cells.add(center)
        for br, bc in branch:
            for dr, dc in _DIRS:
                nb = (br + dr, bc + dc)
                if nb != center and nb not in branch:
                    occupied_neighbors.add(nb)
        for dr, dc in _DIRS:
            nb = (center[0] + dr, center[1] + dc)
            if nb not in branch:
                occupied_neighbors.add(nb)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    for marker in placed_markers:
        center = marker["center"]
        stem = marker["stem"]
        if marker["kind"] == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[cr, cc] = 1
        else:
            grid[center[0], center[1]] = int(marker["tc"])
            for cr, cc in _t_branch_cells(center, stem):
                grid[cr, cc] = 1


def _stamp_markers_output(
    grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]
) -> None:
    for marker in placed_markers:
        color = int(out_colors[marker["src"]])
        for cr, cc in _t_cells(marker["center"], marker["stem"]):
            grid[cr, cc] = color


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[dict],
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
    for marker in placed_markers:
        all_marker_cells |= _t_cells(marker["center"], marker["stem"])

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
            candidates.append((r, c, rid))

    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = int(out_colors[rid])
    return orient


def _simulate_sequential_region_colors(
    input_colors: list[int], placed_markers: list[dict]
) -> list[int]:
    current = list(map(int, input_colors))
    for marker in sorted(placed_markers, key=lambda m: m["order_key"]):
        if marker["kind"] == "normal":
            current[marker["dst"]] = int(current[marker["src"]])
        else:
            current[marker["dst"]] = int(marker["tc"])
    return current


def _place_markers(
    region: np.ndarray,
    chain_defs: list[dict],
    extra_defs: list[dict],
    H: int,
    W: int,
) -> list[dict] | None:
    chain_first = list(_find_valid_placements(region, chain_defs[0]["src"], chain_defs[0]["dst"], set(), set(), H, W))
    random.shuffle(chain_first)
    for center1, stem1 in chain_first:
        occupied_cells: set[tuple[int, int]] = set()
        occupied_neighbors: set[tuple[int, int]] = set()
        placed: list[dict] = []

        first = dict(chain_defs[0])
        first["center"] = center1
        first["stem"] = stem1
        first["order_key"] = _placement_key(center1, stem1)
        _register_marker(first["kind"], center1, stem1, occupied_cells, occupied_neighbors)
        placed.append(first)

        second_candidates = _find_valid_placements(
            region,
            chain_defs[1]["src"],
            chain_defs[1]["dst"],
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        second_candidates = [
            (center2, stem2)
            for center2, stem2 in second_candidates
            if _placement_key(center2, stem2) > first["order_key"]
        ]
        random.shuffle(second_candidates)
        if not second_candidates:
            continue

        second = dict(chain_defs[1])
        center2, stem2 = second_candidates[0]
        second["center"] = center2
        second["stem"] = stem2
        second["order_key"] = _placement_key(center2, stem2)
        _register_marker(second["kind"], center2, stem2, occupied_cells, occupied_neighbors)
        placed.append(second)

        ok = True
        shuffled_extras = list(extra_defs)
        random.shuffle(shuffled_extras)
        for base in shuffled_extras:
            cands = _find_valid_placements(
                region, base["src"], base["dst"], occupied_cells, occupied_neighbors, H, W
            )
            random.shuffle(cands)
            if not cands:
                ok = False
                break
            center, stem = cands[0]
            marker = dict(base)
            marker["center"] = center
            marker["stem"] = stem
            marker["order_key"] = _placement_key(center, stem)
            _register_marker(marker["kind"], center, stem, occupied_cells, occupied_neighbors)
            placed.append(marker)
        if ok:
            return placed
    return None


def _marker_buffers(placed_markers: list[dict]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    for marker in placed_markers:
        _register_marker(
            marker["kind"], marker["center"], marker["stem"], occupied_cells, occupied_neighbors
        )
    return occupied_cells, occupied_neighbors


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(10, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    directed_pairs = _find_adjacent_pairs(region)
    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, _n_normal, n_colored = counts

    chain = _choose_chain(directed_pairs)
    if chain is None:
        return None
    chain_src, chain_mid, chain_dst = chain

    extra_pairs = _pick_extra_pairs(
        directed_pairs,
        chain_src=chain_src,
        chain_mid=chain_mid,
        chain_dst=chain_dst,
        n_needed=n_markers - 2,
    )
    if extra_pairs is None:
        return None

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    chain_defs = [
        {"kind": "normal", "src": chain_src, "dst": chain_mid, "tc": None},
        {"kind": "normal", "src": chain_mid, "dst": chain_dst, "tc": None},
    ]

    colored_indices = set(random.sample(range(len(extra_pairs)), k=n_colored)) if n_colored else set()
    extra_defs: list[dict] = []
    for idx, (src, dst) in enumerate(extra_pairs):
        if idx in colored_indices:
            cands = [c for c in COLOR_POOL if c != input_colors[dst]]
            if not cands:
                return None
            extra_defs.append({"kind": "colored", "src": src, "dst": dst, "tc": int(random.choice(cands))})
        else:
            extra_defs.append({"kind": "normal", "src": src, "dst": dst, "tc": None})

    placed_markers = _place_markers(region, chain_defs, extra_defs, H, W)
    if placed_markers is None:
        return None

    input_grid = _build_color_map(region, input_colors)
    seq_colors = _simulate_sequential_region_colors(input_colors, placed_markers)
    output_grid = _build_color_map(region, seq_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, seq_colors)

    _occupied_cells, occupied_neighbors = _marker_buffers(placed_markers)
    l9_orient = _try_place_l9(
        region, placed_markers, occupied_neighbors, input_grid, output_grid, seq_colors
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

    metrics = variant_metrics(input_grid)
    if metrics["dependent_normals"] < 1 or metrics["max_chain_depth"] < 2:
        return None
    if metrics["diff_cells_vs_simultaneous"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
