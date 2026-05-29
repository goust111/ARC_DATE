# Combo variant of 21897d95 (index 9)
# combo_spec: D1/0+D7/0+D11/0
# Name: 二值交换纯化
#
# design_analysis:
#   concretization_and_trace:
#     - 画面仍然是由 4 连通单色区域组成的拼图，但所有大区域只使用两种重复颜色；
#       颜色 1 只用于普通的 4 格 T 标记，不再出现彩色中心 T，也不再出现 L9。
#     - 为了让“只有两色却仍要区分多个区域”可观察，生成器不再用任意 Voronoi，
#       而是构造宽而起伏的条带链，相邻条带必为异色、非相邻条带可复用同色。
#     - 具体追踪：若条带 A/B/C/D 的输入色是 2/6/2/6，四个 T 依次表示 A→B、B→C、C→D。
#       原题单向复制会得到 2/2/6/2；本融合版按中心的行优先顺序做交换后得到 6/2/6/2，
#       变化不再只落在“被指向者”，而会沿链条把颜色继续搬运。
#   complexity_calibration:
#     - 若每个 T 都是互不相干的二元交换，样例会过于简单；因此生成器固定放置一条
#       至少 4 步的相邻条带交换链，让内部区域至少参与两次。
#     - 若只保留两色但允许同色相邻，solver 会把多个语义区域误并成一个可见块；
#       因此生成器强制相邻区域交替着色，并让条带足够宽，维持人眼可分的区域拓扑。
#   implicit_assumptions_review:
#     - 原 solver 依赖“源区颜色是静态的、只有目标区改色”，对应的
#       `replacement = root_color[source]` 与 `final_color[dest] = ...` 在交换规则下失效，
#       必须改成可变 `current_color` 并按可见顺序执行 `swap(source, dest)`。
#     - 原 solver 还支持 colored-center T 与 L9 旋转；本融合版去掉这两类对象，
#       因而输入中的每个颜色 1 连通块都必须是一个普通 T 四连块，输出尺寸与方向保持不变。
#     - 由于只剩两种区域色，DSU 不能依赖“颜色唯一”来辨认区域；安全性改由生成器保证：
#       任意相邻的大区域一定异色，所以同色碎片只会来自同一条带被 T 暂时切开。
#   placement_legality:
#     - 每个 T 仍满足原题几何：不越界、不重叠、带缓冲区、茎方向首次碰到的是目标区，
#       茎后两个对角单元都落在源区。
#     - 交换链按预定顺序放置，并强制 T 中心坐标单调递增，使 row-major 顺序就是执行顺序。
#   visual_observability:
#     - 解题者能直接看到：只有两种底色在多个不相邻区域重复出现，而 T 的朝向决定哪两个区域互换。
#     - 多步链让“源区自身也会翻色”在 2-3 个样例里持续可见，因此不会被误读成原题的单向复制。
#   cleanliness:
#     - 去掉彩色中心 T、L9 和小碎片后，题面只剩大块区域与少量白色 T，视觉层次比原题更纯净。
#     - 条带式区域比任意随机分块更整洁，也能稳定提供足够的合法 T 放置位置。
#   change_inferability:
#     - 颜色变化只依赖可见的区域连通性、T 朝向与 T 中心的阅读顺序，不依赖颜色编号大小。
#     - 生成器会拒绝那些“交换输出与原题单向复制输出一致”的退化样本，确保融合行为真实发生。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


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


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
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
            cid += 1
    return comp_id, comp_colors


def _extract_markers(g: np.ndarray) -> tuple[list[dict], np.ndarray]:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)
    for k in range(1, int(n_1) + 1):
        cells = {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        parsed = _parse_tetromino(cells)
        assert parsed is not None, "every color-1 component must be a normal T marker"
        center, stem, bar = parsed
        markers.append({"center": center, "stem": stem, "bar": bar, "cells": cells})
        used_ones |= cells
        for r, c in cells:
            control[r, c] = True
    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to T markers"
    assert markers, "at least one T marker is required"
    return markers, control


def _build_region_model(g: np.ndarray, markers: list[dict], control: np.ndarray) -> tuple[np.ndarray, _DSU, dict[int, int]]:
    comp_id, comp_colors = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    assert len(set(comp_colors)) == 2, "binary variant expects exactly two region colors"
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
        marker["source_comp"] = source
        marker["dest_comp"] = dest

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region pieces must keep one color"

    for marker in markers:
        marker["source"] = dsu.find(marker["source_comp"])
        marker["dest"] = dsu.find(marker["dest_comp"])
        assert marker["source"] != marker["dest"], "a T marker must separate two distinct regions"

    return comp_id, dsu, root_color


def _ordered_markers(markers: list[dict]) -> list[dict]:
    return sorted(markers, key=lambda marker: tuple(marker["center"]))


def _swap_result_on_roots(root_color: dict[int, int], ordered_markers: list[dict]) -> dict[int, int]:
    current_color = dict(root_color)
    for marker in ordered_markers:
        source = int(marker["source"])
        dest = int(marker["dest"])
        current_color[source], current_color[dest] = current_color[dest], current_color[source]
    return current_color


def _copy_result_on_roots(root_color: dict[int, int], ordered_markers: list[dict]) -> dict[int, int]:
    copied = dict(root_color)
    seen_dest: set[int] = set()
    for marker in ordered_markers:
        source = int(marker["source"])
        dest = int(marker["dest"])
        assert dest not in seen_dest, "generator must keep destinations unique for copy comparison"
        seen_dest.add(dest)
        copied[dest] = int(root_color[source])
    return copied


def _solve_binary_swap(g: np.ndarray) -> np.ndarray:
    markers, control = _extract_markers(g)
    comp_id, dsu, root_color = _build_region_model(g, markers, control)
    ordered = _ordered_markers(markers)
    final_color = _swap_result_on_roots(root_color, ordered)

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


def _behavior_metrics_from_grid(g: np.ndarray) -> dict[str, int]:
    markers, control = _extract_markers(g)
    _comp_id, _dsu, root_color = _build_region_model(g, markers, control)
    ordered = _ordered_markers(markers)
    swap_colors = _swap_result_on_roots(root_color, ordered)
    copy_colors = _copy_result_on_roots(root_color, ordered)
    participation = Counter()
    for marker in ordered:
        participation[int(marker["source"])] += 1
        participation[int(marker["dest"])] += 1
    source_roots = {int(marker["source"]) for marker in ordered}
    return {
        "marker_count": len(ordered),
        "max_region_participation": max(participation.values()) if participation else 0,
        "changed_source_regions": sum(
            1 for root in source_roots if swap_colors[root] != root_color[root]
        ),
        "swap_vs_copy_region_diff": sum(
            1 for root in root_color if swap_colors[root] != copy_colors[root]
        ),
        "flipped_regions": sum(1 for root in root_color if swap_colors[root] != root_color[root]),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange colors across boundary T markers on a binary region mosaic, then erase the markers.

    Input structure:
    - The grid is partitioned into monochrome regions, but all large regions use only two repeating colors.
    - Color 1 is reserved for ordinary 4-cell T markers placed on boundaries between regions.
    - There is no colored-center T and no color-9 L marker in this fusion variant.

    Rule:
    - For each T, the single stem points toward the target region.
    - The two diagonal cells just behind that stem identify the source-side region that owns the marker.
    - Process the T markers in row-major order of their centers; each one swaps the current colors of its source and target regions.
    - After all swaps, fill every T cell with the final color of the source-side region.
    - The output keeps the same size and orientation as the input.

    Key insight:
    - With only two region colors, each swap looks like both touched regions flipping, but repeated colors appear in multiple disconnected places, so the solver must follow region connectivity and T geometry instead of treating color alone as region identity.
    """
    return _solve_binary_swap(np.asarray(grid, dtype=int))


# --------------- generator (binary strip swap chain) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_COUNTS = [5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 2, 1]
MIN_STRIP_WIDTH = 3
REGION_SIZE_FACTOR = 3.0

MARKER_MIN = 4
MARKER_MAX = 7
GEN_RETRIES = 400

REGION_COLOR_POOL = [2, 3, 4, 5, 6, 7, 8]


def _split_total(total: int, parts: int, min_each: int) -> list[int] | None:
    extra = total - parts * min_each
    if extra < 0:
        return None
    widths = [min_each] * parts
    for _ in range(extra):
        widths[random.randrange(parts)] += 1
    random.shuffle(widths)
    return widths


def _build_strip_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    vertical = random.choice([False, True])
    major = W if vertical else H
    minor = H if vertical else W
    widths = _split_total(major, n_regions, MIN_STRIP_WIDTH)
    if widths is None or minor < 9:
        return None

    n_bounds = n_regions - 1
    current = []
    pos = widths[0] - 1
    for i in range(n_bounds):
        current.append(pos)
        pos += widths[i + 1]

    boundary_rows = [current.copy()]
    for _ in range(1, minor):
        nxt = current.copy()
        for i in range(n_bounds):
            lo = MIN_STRIP_WIDTH - 1 if i == 0 else nxt[i - 1] + MIN_STRIP_WIDTH
            hi = (major - MIN_STRIP_WIDTH - 1) if i == n_bounds - 1 else current[i + 1] - MIN_STRIP_WIDTH
            step = random.choice([-1, 0, 1])
            nxt[i] = min(max(current[i] + step, lo), hi)
        current = nxt
        boundary_rows.append(current.copy())

    region = np.full((H, W), -1, dtype=int)
    if vertical:
        for r in range(H):
            prev = -1
            for rid, boundary in enumerate(boundary_rows[r]):
                region[r, prev + 1 : boundary + 1] = rid
                prev = boundary
            region[r, prev + 1 :] = n_regions - 1
    else:
        for c in range(W):
            prev = -1
            for rid, boundary in enumerate(boundary_rows[c]):
                region[prev + 1 : boundary + 1, c] = rid
                prev = boundary
            region[prev + 1 :, c] = n_regions - 1

    for rid in range(n_regions):
        mask = region == rid
        if mask.sum() < MIN_REGION_CELLS:
            return None
        _, n = label_mask(mask, connectivity=4)
        if n != 1:
            return None
    # Reverse abstract region ids on about half the samples so the same visible strip layout
    # can realize opposite abstract chain directions without changing the human-visible topology.
    if random.choice([False, True]):
        region = (n_regions - 1) - region
    return region


def _choose_chain_pairs(n_regions: int) -> list[tuple[int, int]] | None:
    max_markers = min(MARKER_MAX, n_regions - 1)
    if max_markers < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_markers)
    direction = random.choice([-1, 1])
    if direction == 1:
        start = random.randint(0, n_regions - n_markers - 1)
        path = [start + i for i in range(n_markers + 1)]
    else:
        start = random.randint(n_markers, n_regions - 1)
        path = [start - i for i in range(n_markers + 1)]
    return [(int(a), int(b)) for a, b in zip(path, path[1:])]


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
    min_center: tuple[int, int] | None,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    H, W = region.shape
    candidates = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            center = (r, c)
            if min_center is not None and center <= min_center:
                continue
            for stem in _DIRS:
                bar = _get_bar(stem)
                cells = _t_cells(center, stem)
                if cells & occupied_cells:
                    continue
                if _t_neighborhood(center, stem) & occupied_cells:
                    continue
                if cells & occupied_neighbors:
                    continue
                if not all(
                    0 <= r + stem[0] + bd[0] < H
                    and 0 <= c + stem[1] + bd[1] < W
                    and int(region[r + stem[0] + bd[0], c + stem[1] + bd[1]]) == src
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
                    candidates.append((center, stem))
    return candidates


def _register_marker(
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> None:
    cells = _t_cells(center, stem)
    occupied_cells |= cells
    occupied_neighbors |= _t_neighborhood(center, stem) - cells


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    return np.asarray(colors, dtype=int)[region]


def _stamp_markers_input(
    grid: np.ndarray,
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> None:
    for center, stem, _src, _dst in placed_markers:
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = 1


def _marker_order(
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]]
) -> list[tuple[tuple[int, int], tuple[int, int], int, int]]:
    return sorted(placed_markers, key=lambda marker: marker[0])


def _simulate_swap_colors(
    input_colors: list[int],
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> list[int]:
    out_colors = list(input_colors)
    for _center, _stem, src, dst in _marker_order(placed_markers):
        out_colors[src], out_colors[dst] = out_colors[dst], out_colors[src]
    return out_colors


def _simulate_copy_colors(
    input_colors: list[int],
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
) -> list[int]:
    out_colors = list(input_colors)
    seen_dest: set[int] = set()
    for _center, _stem, src, dst in _marker_order(placed_markers):
        assert dst not in seen_dest
        seen_dest.add(dst)
        out_colors[dst] = input_colors[src]
    return out_colors


def _swap_behavior_metrics(
    input_colors: list[int],
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
    swap_colors: list[int],
    copy_colors: list[int],
) -> dict[str, int]:
    ordered = _marker_order(placed_markers)
    participation = Counter()
    source_regions = set()
    for _center, _stem, src, dst in ordered:
        participation[src] += 1
        participation[dst] += 1
        source_regions.add(src)
    return {
        "marker_count": len(ordered),
        "max_region_participation": max(participation.values()) if participation else 0,
        "changed_source_regions": sum(
            1 for src in source_regions if swap_colors[src] != input_colors[src]
        ),
        "swap_vs_copy_region_diff": sum(
            1 for before, after in zip(swap_colors, copy_colors) if before != after
        ),
        "flipped_regions": sum(
            1 for before, after in zip(input_colors, swap_colors) if before != after
        ),
    }


def _stamp_markers_output(
    grid: np.ndarray,
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]],
    out_colors: list[int],
) -> None:
    for center, stem, src, _dst in placed_markers:
        color = out_colors[src]
        for cr, cc in _t_cells(center, stem):
            grid[cr, cc] = color


def _apply_symmetry(grid: np.ndarray, flip_ud: bool, flip_lr: bool) -> np.ndarray:
    out = np.array(grid, copy=True)
    if flip_ud:
        out = np.flipud(out)
    if flip_lr:
        out = np.fliplr(out)
    return np.array(out, copy=True)


def _meets_behavior_thresholds(metrics: dict[str, int]) -> bool:
    return (
        metrics["max_region_participation"] >= 2
        and metrics["changed_source_regions"] >= 2
        and metrics["swap_vs_copy_region_diff"] >= 2
        and metrics["flipped_regions"] >= 3
    )


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(12, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _build_strip_regions(H, W, n_regions)
    if region is None:
        return None

    marker_pairs = _choose_chain_pairs(n_regions)
    if marker_pairs is None:
        return None

    color_a, color_b = random.sample(REGION_COLOR_POOL, 2)
    input_colors = [color_a if rid % 2 == 0 else color_b for rid in range(n_regions)]

    occupied_cells: set[tuple[int, int]] = set()
    occupied_neighbors: set[tuple[int, int]] = set()
    placed_markers: list[tuple[tuple[int, int], tuple[int, int], int, int]] = []
    last_center: tuple[int, int] | None = None
    for src, dst in marker_pairs:
        candidates = _find_valid_placements(
            region,
            src,
            dst,
            occupied_cells,
            occupied_neighbors,
            last_center,
        )
        if not candidates:
            return None
        center, stem = random.choice(candidates)
        _register_marker(center, stem, occupied_cells, occupied_neighbors)
        placed_markers.append((center, stem, src, dst))
        last_center = center

    swap_colors = _simulate_swap_colors(input_colors, placed_markers)
    copy_colors = _simulate_copy_colors(input_colors, placed_markers)
    metrics = _swap_behavior_metrics(input_colors, placed_markers, swap_colors, copy_colors)
    if not _meets_behavior_thresholds(metrics):
        return None

    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, swap_colors)

    _stamp_markers_input(input_grid, placed_markers)
    _stamp_markers_output(output_grid, placed_markers, swap_colors)

    try:
        result = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(result, output_grid):
        return None

    transforms = [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
    random.shuffle(transforms)
    for flip_ud, flip_lr in transforms:
        candidate = _apply_symmetry(input_grid, flip_ud=flip_ud, flip_lr=flip_lr)
        try:
            candidate_metrics = _behavior_metrics_from_grid(candidate)
            candidate_result = main(candidate)
        except Exception:
            continue
        if _meets_behavior_thresholds(candidate_metrics) and candidate_result.shape == candidate.shape:
            return candidate
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
