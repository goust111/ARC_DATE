# Variant of 21897d95
# Dimension: D1 — 核心重着色变换规则 (transform)
# Change: T标记的茎指向目标区域，目标区域被重着色为源区域颜色（normal T）或中心颜色（colored T）；标记单元被填充... → 颜色交换而非单向重着色
#
# design_analysis:
#   concretization:
#     - 保留原题的“单色区域拼图 + 边界 T 标记 + 可选 L9 旋转”外观；D1 只改 T 的颜色传播。
#     - 这里把 T 仍解释为“源侧 + 目标侧”的有向边界标记：茎指向目标区，两条对角线锁定源侧区域，标记像素仍归属源侧。
#     - 具体例子：三个区域 A/B/C 的输入色分别为 2/3/4，阅读顺序中的两个 T 为 A→B、B→C。
#       原题：B 变成 2，C 变成 3，A 保持 2，因此输出色是 2/2/3。
#       本变体：先交换 A/B 得到 3/2/4，再交换 B/C 得到 3/4/2；A 和 B 都会被后续连锁影响。
#   complexity_calibration:
#     - 如果每个区域只参与一次交换，题面会退化成若干独立的二元换色，复杂度不够，所以生成器强制放置一条多步交换链。
#     - 同时不引入额外不可见排序键；唯一顺序规则是 T 中心的行优先顺序，这在像素网格中可直接观察。
#   implicit_assumptions_review:
#     - 原 solver 的 `replacement = root_color[source]` 假设“源色恒为初始色”，这在交换规则下失效。
#     - 原 solver 的 `final_color[dest] = replacement` 假设“只有目标会变”，这也失效，因为源区必须同步拿到目标的当前颜色。
#     - 因此新 solver 需要维护可变的 `current_color` 映射，并按确定顺序执行 `swap(source, dest)`。
#   placement_legality:
#     - 生成器继续使用原题的 T 几何约束：T 不重叠、不越界、带缓冲区、对角线落在源区、茎方向首次命中的区域是目标区。
#     - 为了保证 L9 始终可见，基础区域颜色池排除 9，把 9 专门留给旋转标记。
#   visual_observability:
#     - 各基础区域仍使用互异颜色，且至少一个区域会出现在多于一个 T 中，从而让“源区自身也会变色”在 2-3 组样例里可见。
#     - 不再生成 colored-center T；否则同一题面会混入“字面颜色替换”和“区域交换”两套 D1 规则，削弱可归纳性。
#   cleanliness:
#     - 仍只叠加少量控制元素：T 标记与可选 L9；不额外引入新符号或纹理。
#     - 标记缓冲与单色大区保留，因此交换后的整体画面依然是清晰的大块区域重排。
#   color_inference:
#     - 颜色变化完全由“哪个区域与哪个区域交换、交换顺序如何”决定，不依赖颜色 id 的大小或排序。
#     - 生成器拒绝那些虽然通过 solver、但所有源区最终仍保持原色的退化样本，确保变体行为真实发生。

from __future__ import annotations

from collections import Counter
import numpy as np

from arc_utils import label_mask
import random


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


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
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


def _solve_unrotated(g: np.ndarray) -> np.ndarray:
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
        markers.append({"kind": "normal", "center": center, "stem": stem, "bar": bar, "cells": cells})
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

    def first_component_from(center: tuple[int, int], direction: tuple[int, int], start: int) -> int | None:
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

    final_color = dict(root_color)
    t_markers = []
    for marker in markers:
        if marker["kind"] not in {"normal", "colored"}:
            marker["source"] = dsu.find(marker["source_comp"])
            continue
        source = dsu.find(marker["source_comp"])
        dest = dsu.find(marker["dest_comp"])
        assert source != dest, "T marker must separate two distinct regions"
        marker["source"] = source
        marker["dest"] = dest
        t_markers.append(marker)

    for marker in sorted(t_markers, key=lambda m: tuple(m["center"])):
        source = int(marker["source"])
        dest = int(marker["dest"])
        final_color[source], final_color[dest] = final_color[dest], final_color[source]

    out = np.full(g.shape, -1, dtype=int)
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in markers:
        color = int(final_color[marker["source"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap region colors across T-shaped controls, erase the controls, and optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is a mosaic of monochrome regions.
    - Color 1 forms T-shaped controls on region boundaries.
    - A small non-rectangular scrap may appear and is absorbed into its neighboring region, as in the base task.
    - An optional 5-cell L-shaped color-9 marker can request a 90-degree rotation of the finished image.

    Rule:
    - For each T, the single stem points at the target region.
    - The two diagonal cells behind that stem identify the source-side region that owns the marker.
    - Process all T markers in row-major order of their centers; each one swaps the current colors of its source and target regions.
    - After every swap is complete, fill each marker cell with the final color of its source-side region.
    - A 5-cell L-shaped color-9 marker controls rotation: ┐ (corner top-right) → CCW 90°, ┌ (corner top-left) → CW 90°.

    Key insight:
    - A T no longer performs one-way copying. It exchanges two region colors, so later T markers may act on colors that earlier T markers already moved.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (swap-chain + l9rot) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [4, 5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 3, 1, 1]

MARKER_MIN = 4
MARKER_MAX = 8

BASE_COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 400

_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _get_bar(stem):
    if stem[0] == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center, stem):
    r, c = center
    bar = _get_bar(stem)
    return {
        (r, c),
        (r + stem[0], c + stem[1]),
        (r + bar[0][0], c + bar[0][1]),
        (r + bar[1][0], c + bar[1][1]),
    }


def _t_neighborhood(center, stem):
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
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
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

    for _ in range(200):
        start = random.choice(nodes)
        path = [start]
        for _step in range(n_markers):
            cur = path[-1]
            options = list(adjacency[cur])
            if not options:
                break
            prev = path[-2] if len(path) >= 2 else None
            non_backtrack = [x for x in options if x != prev]
            if non_backtrack:
                options = non_backtrack
            random.shuffle(options)
            options.sort(key=lambda rid: (path.count(rid), rid))
            nxt = options[0]
            path.append(int(nxt))
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


def _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W):
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
                    and region[r + stem[0] + bd[0], c + stem[1] + bd[1]] == src
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
                    found = int(region[nr, nc]) == dst
                    break
                if found:
                    candidates.append(((r, c), stem))
    return candidates


def _register_marker(center, stem, occupied_cells, occupied_neighbors):
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= (_t_neighborhood(center, stem) - cells)


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid, placed_markers):
    for _kind, center, stem, _src, _dst, _tc in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _marker_order(placed_markers) -> list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]]:
    return sorted(placed_markers, key=lambda m: (int(m[1][0]), int(m[1][1])))


def _simulate_swap_colors(input_colors: list[int], placed_markers) -> list[int]:
    out_colors = list(input_colors)
    for _kind, _center, _stem, src, dst, _tc in _marker_order(placed_markers):
        out_colors[src], out_colors[dst] = out_colors[dst], out_colors[src]
    return out_colors


def _swap_behavior_metrics(input_colors: list[int], placed_markers, out_colors: list[int]) -> dict[str, int]:
    ordered = _marker_order(placed_markers)
    participation = Counter()
    source_regions = set()
    for _kind, _center, _stem, src, dst, _tc in ordered:
        participation[src] += 1
        participation[dst] += 1
        source_regions.add(src)
    changed_sources = sum(1 for src in source_regions if out_colors[src] != input_colors[src])
    return {
        "marker_count": len(ordered),
        "max_region_participation": max(participation.values()) if participation else 0,
        "changed_source_regions": changed_sources,
    }


def _stamp_markers_output(grid, placed_markers, out_colors):
    for _kind, center, stem, src, _dst, _tc in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors):
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
            candidates.append((r, c, rid))

    if not candidates:
        return None
    lr, lc, rid = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[lr + dr, lc + dc] = 9
        output_grid[lr + dr, lc + dc] = out_colors[rid]
    return orient


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
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

    pool = list(BASE_COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers = []
    for src, dst in marker_pairs:
        cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors, H, W)
        if not cands:
            return None
        center, stem = random.choice(cands)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append(("normal", center, stem, src, dst, None))

    out_colors = _simulate_swap_colors(input_colors, placed_markers)
    metrics = _swap_behavior_metrics(input_colors, placed_markers, out_colors)
    if metrics["max_region_participation"] < 2 or metrics["changed_source_regions"] == 0:
        return None

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, out_colors)

    l9_orient = _try_place_l9(region, placed_markers, occupied_neighbors, input_grid, output_grid, out_colors)
    if l9_orient == "top_right":
        output_grid = np.rot90(output_grid, 1)
    elif l9_orient == "top_left":
        output_grid = np.rot90(output_grid, 3)

    try:
        result = main(input_grid)
    except Exception:
        return None
    return input_grid if np.array_equal(result, output_grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
