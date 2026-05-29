from __future__ import annotations

# Variant of 21897d95
# Dimension: D10 — 重着色的同时性与顺序性 (multi_step)
# Change: 所有T标记的重着色同时执行（simultaneous），不存在处理顺序
#   → T标记分组处理：先normal后colored；colored在第二步看到的是normal已经改写后的网格
#
# Design analysis:
# - concrete_behavior:
#   - T 的几何识别不变，normal 仍是 4 格纯 1 的 T，colored 仍是 3 个单独的 1 围住一个彩色中心。
#   - 改动落在“何时根据当前网格识别 source / destination 连通块并执行重着色”。
#   - 先执行所有 normal，得到中间网格；此时 normal 标记已被源侧最终颜色填回，colored 与小碎片/L9 仍保留为控制符号。
#   - 再在中间网格上解析 colored 的 source/destination 连通块，所以 colored 可能命中一个被 normal 扩大的同色连通块。
# - example_trace:
#   - 若 normal 是 B→C，则 phase1 后 C 变成 B 的颜色，与相邻的 B 合并。
#   - 若 colored 是 S→B，原题 simultaneous 只会把 B 改成中心色；本变体会把 phase1 后的 (B∪C) 一起改成中心色。
#   - 这让同一 colored T 在两阶段模型下会改动更多像素，视觉上可见。
# - complexity_calibration:
#   - 只引入“phase order”这一个新自由度，不增加新的符号或新的朝向规则。
#   - 生成器固定至少 1 个 colored，并强制至少 1 个 normal→colored 交互链，避免退化回原题。
# - old_assumption_audit:
#   - 原 solver 把所有 T 一次性放进同一个 DSU / final_color 更新里，默认 colored 的目标区域来自原始输入分区。
#   - 这在新规则下失效，因为 colored 的 destination/source 必须基于 phase1 后的网格重新判定。
#   - 因此不能复用“先解析一次所有 comp_id 再统一上色”的写法，必须分阶段重新标号连通块。
# - placement_legality:
#   - 仍沿用原 T 放置几何约束：source 对角格在源区，stem 方向 step>=2 的首个非控制格在目标区。
#   - 额外要求交互链中的 normal 与 colored 拥有不同 destination，但它们围绕同一中间区域，确保 phase2 的目标扩张合法且可解。
# - observability_and_cleanliness:
#   - 区域底色不使用 1 或 9；9 只保留给 L 标记，避免与底色混淆。
#   - 变体差异通过整块区域在第二步被“连带改色”体现，不依赖颜色编号排序或不可见元数据。
# - color_inference:
#   - colored 的中心色直接写在输入里，是 phase2 的字面替换色；生成器保证它不同于 phase1 后的目标颜色，使变化稳定可见。

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


def _get_bar(stem: tuple[int, int]) -> list[tuple[int, int]]:
    if int(stem[0]) == 0:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _t_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (int(r), int(c)),
        (int(r + stem[0]), int(c + stem[1])),
        (int(r + bar[0][0]), int(c + bar[0][1])),
        (int(r + bar[1][0]), int(c + bar[1][1])),
    }


def _t_branch_cells(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    r, c = center
    bar = _get_bar(stem)
    return {
        (int(r + stem[0]), int(c + stem[1])),
        (int(r + bar[0][0]), int(c + bar[0][1])),
        (int(r + bar[1][0]), int(c + bar[1][1])),
    }


def _t_neighborhood(center: tuple[int, int], stem: tuple[int, int]) -> set[tuple[int, int]]:
    cells = _t_cells(center, stem)
    neighborhood: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr, dc in DIRS:
            nb = (int(r + dr), int(c + dc))
            if nb not in cells:
                neighborhood.add(nb)
    return neighborhood


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


def _label_base_components(g: np.ndarray, control: np.ndarray) -> tuple[np.ndarray, list[int], list[int]]:
    comp_id = np.full(g.shape, -1, dtype=int)
    comp_colors: list[int] = []
    comp_areas: list[int] = []
    cid = 0
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == int(col)) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == int(k))
            if coords.size == 0:
                continue
            for rr, cc in coords:
                comp_id[int(rr), int(cc)] = cid
            comp_colors.append(int(col))
            comp_areas.append(int(len(coords)))
            cid += 1
    return comp_id, comp_colors, comp_areas


def _adjacent_component_counts(cells: set[tuple[int, int]], comp_id: np.ndarray) -> Counter:
    counts: Counter = Counter()
    for r, c in cells:
        for dr, dc in DIRS:
            rr, cc = int(r + dr), int(c + dc)
            if _in_bounds(comp_id.shape, rr, cc) and int(comp_id[rr, cc]) >= 0:
                counts[int(comp_id[rr, cc])] += 1
    return counts


def _first_component_from(
    center: tuple[int, int], direction: tuple[int, int], comp_id: np.ndarray, start: int
) -> int | None:
    r, c = center
    step = int(start)
    while True:
        rr = int(r + direction[0] * step)
        cc = int(c + direction[1] * step)
        if not _in_bounds(comp_id.shape, rr, cc):
            return None
        if int(comp_id[rr, cc]) >= 0:
            return int(comp_id[rr, cc])
        step += 1


def _mark_cells(mask: np.ndarray, markers: list[dict]) -> None:
    for marker in markers:
        for r, c in marker["cells"]:
            mask[int(r), int(c)] = True


def _parse_controls(g: np.ndarray) -> tuple[list[dict], list[dict], list[dict]]:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        int(k): {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == int(k))}
        for k in range(1, int(n_1) + 1)
    }
    one_sizes = {int(k): len(cells) for k, cells in one_components.items()}

    normal_markers: list[dict] = []
    colored_markers: list[dict] = []
    used_ones: set[tuple[int, int]] = set()
    control = np.zeros(g.shape, dtype=bool)
    next_mid = 0

    for cells in one_components.values():
        parsed = _parse_tetromino(cells)
        if parsed is None:
            continue
        center, stem, bar = parsed
        marker = {
            "mid": next_mid,
            "kind": "normal",
            "center": center,
            "stem": stem,
            "bar": bar,
            "cells": set(cells),
        }
        next_mid += 1
        normal_markers.append(marker)
        used_ones |= cells
        _mark_cells(control, [marker])

    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            if int(g[r, c]) == 1:
                continue
            branches: list[tuple[int, int]] = []
            cells: list[tuple[int, int]] = []
            ok = True
            for d in DIRS:
                rr, cc = int(r + d[0]), int(c + d[1])
                if not _in_bounds(g.shape, rr, cc) or int(g[rr, cc]) != 1:
                    continue
                label = int(labels_1[rr, cc])
                if one_sizes.get(label, 0) == 1 and (rr, cc) not in used_ones:
                    branches.append(d)
                    cells.append((rr, cc))
                else:
                    ok = False
            if ok and len(branches) == 3:
                stem = _stem_from_branches(branches)
                bar = [d for d in branches if d != stem]
                assert len(bar) == 2 and bar[0] == _opp(bar[1])
                marker = {
                    "mid": next_mid,
                    "kind": "colored",
                    "center": (int(r), int(c)),
                    "stem": stem,
                    "bar": bar,
                    "cells": set(cells) | {(int(r), int(c))},
                    "target_color": int(g[r, c]),
                }
                next_mid += 1
                colored_markers.append(marker)
                used_ones |= set(cells)
                _mark_cells(control, [marker])

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a T control marker"

    small_markers: list[dict] = []
    for col in sorted(set(map(int, g.ravel())) - {1}):
        labels, n_labels = label_mask((g == int(col)) & (~control), connectivity=4)
        for k in range(1, int(n_labels) + 1):
            coords = np.argwhere(labels == int(k))
            if len(coords) > 5:
                continue
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            if len(coords) == (int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1):
                continue
            marker = {
                "mid": next_mid,
                "kind": "small",
                "cells": {tuple(map(int, xy)) for xy in coords},
            }
            next_mid += 1
            small_markers.append(marker)
            _mark_cells(control, [marker])

    return normal_markers, colored_markers, small_markers


def _apply_controls(
    g: np.ndarray,
    current_t_markers: list[dict],
    active_t_markers: list[dict],
    small_markers: list[dict],
    fill_small: bool,
) -> np.ndarray:
    control = np.zeros(g.shape, dtype=bool)
    _mark_cells(control, current_t_markers)
    _mark_cells(control, small_markers)

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"

    dsu = _DSU(len(comp_colors))
    active_ids = {int(marker["mid"]) for marker in active_t_markers}
    marker_source_comp: dict[int, int] = {}
    marker_dest_comp: dict[int, int] = {}

    for marker in current_t_markers:
        center = marker["center"]
        stem = marker["stem"]
        dest = _first_component_from(center, stem, comp_id, start=2)
        assert dest is not None, "T marker must point at a destination region"

        source_parts: list[int] = []
        for bar_dir in marker["bar"]:
            rr = int(center[0] + stem[0] + bar_dir[0])
            cc = int(center[1] + stem[1] + bar_dir[1])
            assert _in_bounds(g.shape, rr, cc) and int(comp_id[rr, cc]) >= 0
            source_parts.append(int(comp_id[rr, cc]))

        source = source_parts[0]
        for part in source_parts[1:]:
            assert comp_colors[part] == comp_colors[source], "T source diagonals must share a color"
            dsu.union(source, part)

        for comp in _adjacent_component_counts(marker["cells"], comp_id):
            if comp_colors[comp] == comp_colors[source]:
                dsu.union(source, comp)

        marker_source_comp[int(marker["mid"])] = source
        if int(marker["mid"]) in active_ids:
            marker_dest_comp[int(marker["mid"])] = int(dest)

    small_owner_comp: dict[int, int] = {}
    for marker in small_markers:
        counts = _adjacent_component_counts(marker["cells"], comp_id)
        assert counts, "small marker must touch its owner region"
        owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
        for comp in counts:
            if comp_colors[comp] == comp_colors[owner]:
                dsu.union(owner, comp)
        small_owner_comp[int(marker["mid"])] = int(owner)

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    final_color = dict(root_color)
    for marker in active_t_markers:
        source = dsu.find(marker_source_comp[int(marker["mid"])])
        dest = dsu.find(marker_dest_comp[int(marker["mid"])])
        replacement = int(marker["target_color"]) if marker["kind"] == "colored" else int(root_color[source])
        assert final_color[dest] in {root_color[dest], replacement}, "conflicting T controls for one region"
        final_color[dest] = replacement

    out = g.copy()
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in active_t_markers:
        source = dsu.find(marker_source_comp[int(marker["mid"])])
        color = int(final_color[source])
        for r, c in marker["cells"]:
            out[int(r), int(c)] = color

    if fill_small:
        for marker in small_markers:
            owner = dsu.find(small_owner_comp[int(marker["mid"])])
            color = int(final_color[owner])
            for r, c in marker["cells"]:
                out[int(r), int(c)] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _solve_unrotated_sequential(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    mid = _apply_controls(g, normal_markers + colored_markers, normal_markers, small_markers, fill_small=False)
    return _apply_controls(mid, colored_markers, colored_markers, small_markers, fill_small=True)


def _solve_unrotated_simultaneous(g: np.ndarray) -> np.ndarray:
    normal_markers, colored_markers, small_markers = _parse_controls(g)
    return _apply_controls(
        g,
        normal_markers + colored_markers,
        normal_markers + colored_markers,
        small_markers,
        fill_small=True,
    )


def _find_l9_orientation(g: np.ndarray) -> str | None:
    mask9 = g == 9
    if not mask9.any():
        return None
    labels, n = label_mask(mask9, connectivity=4)
    for k in range(1, int(n) + 1):
        coords = {tuple(map(int, xy)) for xy in np.argwhere(labels == int(k))}
        if len(coords) != 5:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 != 2 or c1 - c0 != 2:
            continue
        for cr, cc in ((r0, c0), (r0, c1), (r1, c0), (r1, c1)):
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
    Repaint regions from T-shaped control markers in two phases, then optionally rotate via a color-9 L marker.

    Input structure:
    - The grid is a mosaic of irregular monochrome regions.
    - Color 1 still makes T controls: either a full 4-cell T or three singleton 1s around a colored center.
    - A colored-center T stores its replacement color at the center cell itself.
    - An optional 5-cell color-9 L remains embedded inside one region and controls a quarter-turn of the final image.

    Rule:
    - Each T stem points to the destination region, while the two diagonals behind the stem identify the source side.
    - Phase 1 applies every normal T simultaneously: each destination takes the source side's current color, and each processed marker is filled with its source side's final phase-1 color.
    - Phase 2 then applies every colored T on the already modified grid from phase 1: each destination takes the colored center value, and those marker cells are filled with the source side's final phase-2 color.
    - Any small non-rectangular scraps, including the optional L marker, are absorbed into their touching owner region at the end; L orientation is still ┐ → CCW 90° and ┌ → CW 90°.

    Key insight:
    - The geometry of a T never changes, but colored T markers are evaluated one phase later, so their destination can expand after normal recoloring merges regions.
    """
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated_sequential(g)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


# --------------- generator (sequential recolor) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.5
REGION_COUNTS = [5, 6, 7, 8]
REGION_WEIGHTS = [3, 3, 2, 1]

MARKER_MIN = 4
MARKER_MAX = 8
NORMAL_MAX = 6
COLORED_COUNT = 1

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}

SEED_ATTEMPTS = 200
GEN_RETRIES = 400
PLACEMENT_RETRIES = 16


def _grow_regions(H: int, W: int, n_regions: int) -> np.ndarray | None:
    region = np.full((H, W), -1, dtype=int)
    seeds: set[tuple[int, int]] = set()
    for _ in range(SEED_ATTEMPTS):
        if len(seeds) >= n_regions:
            break
        seeds.add((random.randint(1, H - 2), random.randint(1, W - 2)))
    if len(seeds) < n_regions:
        return None

    frontier: list[tuple[int, int, int]] = []
    for i, (r, c) in enumerate(list(seeds)):
        region[r, c] = i
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == -1:
                frontier.append((nr, nc, i))
    random.shuffle(frontier)

    while frontier:
        idx = random.randint(0, len(frontier) - 1)
        r, c, rid = frontier[idx]
        frontier[idx] = frontier[-1]
        frontier.pop()
        if int(region[r, c]) != -1:
            continue
        region[r, c] = rid
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and int(region[nr, nc]) == -1:
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
    undirected: set[tuple[int, int]] = set()
    for r in range(H):
        for c in range(W):
            for dr, dc in DIRS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < H and 0 <= nc < W:
                    a, b = int(region[r, c]), int(region[nr, nc])
                    if a != b:
                        undirected.add((min(a, b), max(a, b)))
    directed: list[tuple[int, int]] = []
    for a, b in undirected:
        directed.append((a, b))
        directed.append((b, a))
    return directed


def _region_neighbors(region: np.ndarray) -> dict[int, set[int]]:
    neighbors = {rid: set() for rid in range(int(region.max()) + 1)}
    for a, b in _find_adjacent_pairs(region):
        neighbors[int(a)].add(int(b))
    return neighbors


def _choose_marker_counts(n_regions: int) -> tuple[int, int, int] | None:
    max_possible = min(MARKER_MAX, n_regions)
    max_with_fixed_colored = min(max_possible, NORMAL_MAX + COLORED_COUNT)
    if max_with_fixed_colored < MARKER_MIN:
        return None
    n_markers = random.randint(MARKER_MIN, max_with_fixed_colored)
    return n_markers, n_markers - COLORED_COUNT, COLORED_COUNT


def _choose_interaction(neighbors: dict[int, set[int]]) -> tuple[int, int, int] | None:
    candidates = [rid for rid, nbs in neighbors.items() if len(nbs) >= 2]
    random.shuffle(candidates)
    for bridge in candidates:
        nbs = list(neighbors[bridge])
        random.shuffle(nbs)
        for i, dst_after_normal in enumerate(nbs):
            for src_colored in nbs[i + 1 :]:
                return int(bridge), int(dst_after_normal), int(src_colored)
    return None


def _find_valid_placements(
    region: np.ndarray,
    src: int,
    dst: int,
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    H, W = region.shape
    candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            for stem in DIRS:
                bar = _get_bar(stem)
                cells = _t_cells((r, c), stem)
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
                    nr, nc = int(r + stem[0] * step), int(c + stem[1] * step)
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
        return

    branch = _t_branch_cells(center, stem)
    occupied_cells |= branch
    occupied_cells.add(center)
    for br, bc in branch:
        for dr, dc in DIRS:
            nb = (int(br + dr), int(bc + dc))
            if nb != center and nb not in branch:
                occupied_neighbors.add(nb)
    for dr, dc in DIRS:
        nb = (int(center[0] + dr), int(center[1] + dc))
        if nb not in branch:
            occupied_neighbors.add(nb)


def _place_markers(region: np.ndarray, marker_defs: list[tuple[str, int, int, int | None]]) -> list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] | None:
    for _ in range(PLACEMENT_RETRIES):
        occupied_cells: set[tuple[int, int]] = set()
        occupied_neighbors: set[tuple[int, int]] = set()
        pending = list(marker_defs)
        placed: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]] = []

        while pending:
            options = []
            dead = False
            for idx, (kind, src, dst, tc) in enumerate(pending):
                cands = _find_valid_placements(region, src, dst, occupied_cells, occupied_neighbors)
                if not cands:
                    dead = True
                    break
                options.append((len(cands), random.random(), idx, cands))
            if dead:
                break

            _, _, pick_idx, cands = min(options, key=lambda item: (item[0], item[1]))
            kind, src, dst, tc = pending.pop(int(pick_idx))
            center, stem = random.choice(cands)
            _register_marker(kind, center, stem, occupied_cells, occupied_neighbors)
            placed.append((kind, center, stem, src, dst, tc))

        if not pending:
            return placed
    return None


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _stamp_markers_input(grid: np.ndarray, placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]]) -> None:
    for kind, center, stem, _src, _dst, tc in placed_markers:
        if kind == "normal":
            for cr, cc in _t_cells(center, stem):
                grid[int(cr), int(cc)] = 1
        else:
            grid[int(center[0]), int(center[1])] = int(tc)
            for cr, cc in _t_branch_cells(center, stem):
                grid[int(cr), int(cc)] = 1


def _try_place_l9(
    region: np.ndarray,
    placed_markers: list[tuple[str, tuple[int, int], tuple[int, int], int, int, int | None]],
    input_grid: np.ndarray,
) -> None:
    if random.random() >= L9_PROBABILITY:
        return
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    all_marker_cells: set[tuple[int, int]] = set()
    for _kind, center, stem, *_ in placed_markers:
        all_marker_cells |= _t_cells(center, stem)

    blocked = set(all_marker_cells)
    for _kind, center, stem, *_ in placed_markers:
        blocked |= _t_neighborhood(center, stem)

    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            cells = [(int(r + dr), int(c + dc)) for dr, dc in offsets]
            if not all(0 <= cr < H and 0 <= cc < W for cr, cc in cells):
                continue
            rid = int(region[r, c])
            if not all(int(region[cr, cc]) == rid for cr, cc in cells):
                continue
            if any((cr, cc) in blocked for cr, cc in cells):
                continue
            candidates.append((int(r), int(c)))

    if not candidates:
        return
    lr, lc = random.choice(candidates)
    for dr, dc in offsets:
        input_grid[int(lr + dr), int(lc + dc)] = 9


def _sequential_delta_cells(g: np.ndarray) -> int:
    seq = _solve_unrotated_sequential(g)
    sim = _solve_unrotated_simultaneous(g)
    return int(np.count_nonzero(seq != sim))


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(9, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    counts = _choose_marker_counts(n_regions)
    if counts is None:
        return None
    n_markers, n_normal, n_colored = counts
    assert n_colored == 1

    neighbors = _region_neighbors(region)
    interaction = _choose_interaction(neighbors)
    if interaction is None:
        return None
    bridge, normal_dst, colored_src = interaction

    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    interaction_target_choices = [
        c for c in pool if c not in {int(input_colors[bridge]), int(input_colors[normal_dst])}
    ]
    if not interaction_target_choices:
        return None
    interaction_target = random.choice(interaction_target_choices)

    fixed_defs: list[tuple[str, int, int, int | None]] = [
        ("normal", int(bridge), int(normal_dst), None),
        ("colored", int(colored_src), int(bridge), int(interaction_target)),
    ]

    directed_pairs = _find_adjacent_pairs(region)
    random.shuffle(directed_pairs)
    dest_used = {int(normal_dst), int(bridge)}
    extra_defs: list[tuple[str, int, int, int | None]] = []
    for src, dst in directed_pairs:
        if int(dst) in dest_used:
            continue
        extra_defs.append(("normal", int(src), int(dst), None))
        dest_used.add(int(dst))
        if len(extra_defs) >= n_markers - len(fixed_defs):
            break
    if len(extra_defs) < n_markers - len(fixed_defs):
        return None

    marker_defs = fixed_defs + extra_defs
    random.shuffle(marker_defs)
    placed_markers = _place_markers(region, marker_defs)
    if placed_markers is None:
        return None

    input_grid = _build_color_map(region, input_colors)
    _stamp_markers_input(input_grid, placed_markers)

    # Enforce that the new phase order changes the output on this exact input.
    try:
        delta = _sequential_delta_cells(input_grid)
    except Exception:
        return None
    if delta < MIN_REGION_CELLS:
        return None

    _try_place_l9(region, placed_markers, input_grid)

    try:
        _ = main(input_grid)
    except Exception:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
