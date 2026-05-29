# Combo variant of 21897d95 (index 18)
# combo_spec: D1/0+D9/2+D10/0
# Name: 碎片序列交换链
#
# design_analysis:
#   a) concretization_and_trace:
#      - 视觉外壳沿用原题: 不规则单色大区 + 边界上的普通 T 标记 + 可选的 9 色 L 旋转标记。
#      - 融合后新增 2-4 个小碎片, 每个都是 <=5 格的非矩形单色小块; 它们的颜色按从左到右的顺序定义阶段序列。
#      - 每个阶段收集“源区初始颜色 == 当前碎片颜色”的全部 T, 再在该阶段内按 T 自身的左上扫描顺序逐个执行。
#      - 具体小例子: 若碎片颜色序列是 [2, 3, 4], 三个源区 A/B/C 的输入色分别也是 2/3/4。
#        设阶段 2 内有两枚 T: A<->X, A<->B; 阶段 3 内有一枚 T: B<->C。
#        原题会变成单向改色, 且通常同时读源色; 这里先做 A<->X, A 拿到 X 的当前色, 再做 A<->B,
#        所以 B 拿到的是“已经变过的 A 色”; 接着阶段 3 的 B<->C 又继续搬运这个中间结果。
#      - 差异不是某一条规则单独变化, 而是“碎片决定阶段顺序 + 阶段内顺序执行 + 每步交换”三者叠加后的链式传播。
#   b) complexity_calibration:
#      - 若每个碎片只对应一枚 T, D10 的阶段内顺序就不可见; 因此生成器强制至少一个阶段含有两枚同源色 T。
#      - 若所有 T 的空间顺序恰好与碎片顺序一致, D9 的阶段分组会退化成普通 row-major; 因此生成器拒绝这类样本。
#      - 若把碎片颜色设成任意新颜色而不对应某个源区, 人类无法知道它指向哪些 T; 因此碎片颜色直接复用源区输入色。
#   c) implicit_assumptions_review:
#      - 原 solver 假设普通 T 只改 destination, 这里失效; 交换要求 source 和 destination 都变。
#      - 原 solver 假设所有 T 可同时解释, 这里失效; 需要先完整解析场景, 再按碎片阶段与阶段内顺序执行。
#      - 原 solver 的“小碎片只负责被吸收”假设也失效; 现在小碎片既是可被吸收的对象, 也是阶段排序令牌。
#      - 新 solver 额外假设: 每枚 T 的源区初始颜色恰好出现在某个碎片令牌里, 且令牌颜色互异。生成器会显式保证。
#   d) placement_legality:
#      - T 仍遵守原题几何: 4 格普通 T、对角源格落在 source 区、stem 前方第一次命中的基础区域是 destination。
#      - 标记之间保留缓冲区, 小碎片与 L9 也不与 T 或彼此重叠、越界。
#      - 小碎片放在与其同色源区不同的区域内部, 保证它在输入里是可见的独立对象。
#   e) visual_observability:
#      - 阶段顺序只依赖碎片在画面中的左右位置, 不依赖颜色编号或隐藏排序键。
#      - 每个基础区域初始颜色都互异, 所以“哪种颜色被交换到了哪里”能直接从大区颜色变化看出来。
#      - 生成器保证至少有一个后续阶段读取到前一阶段交换后的颜色, 否则链式特征不明显就丢弃。
#   f) cleanliness:
#      - 输入里只有三类控制元素: T、少量小碎片、可选 L9, 没有额外纹理。
#      - 小碎片数控制在 2-4, T 数控制在 4-5, 并加缓冲, 避免控制物挤成一团。
#   g) inference_of_change:
#      - 颜色变化的线索是结构性的: “碎片颜色 -> 匹配到哪些源区 T -> 按顺序交换”。
#      - 为防退化, 生成器同时检查:
#        1. 正确规则与忽略碎片顺序的 row-major 交换输出不同;
#        2. 正确规则与同顺序单向重着色输出不同;
#        3. 正确规则与阶段内逆序交换输出不同。

from __future__ import annotations

from collections import Counter, defaultdict
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


def _scan_key(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return min(cells)


def _scrap_key(cells: set[tuple[int, int]]) -> tuple[int, int]:
    min_col = min(c for _r, c in cells)
    min_row = min(r for r, _c in cells)
    return int(min_col), int(min_row)


def _label_base_components(
    g: np.ndarray,
    control: np.ndarray,
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


def _parse_scene(g: np.ndarray) -> dict:
    labels_1, n_1 = label_mask(g == 1, connectivity=4)
    one_components = {
        k: {tuple(map(int, xy)) for xy in np.argwhere(labels_1 == k)}
        for k in range(1, int(n_1) + 1)
    }

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
                "kind": "t",
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

    all_one_cells = {tuple(map(int, xy)) for xy in np.argwhere(g == 1)}
    assert all_one_cells == used_ones, "all color-1 pixels must belong to a normal T marker"

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
            markers.append(
                {
                    "kind": "small",
                    "cells": cells,
                    "marker_color": int(col),
                    "sequence_key": _scrap_key(cells),
                    "is_sequence_token": int(col) != 9,
                }
            )
            for rr, cc in cells:
                control[rr, cc] = True

    comp_id, comp_colors, comp_areas = _label_base_components(g, control)
    assert comp_colors, "at least one base region is required"
    dsu = _DSU(len(comp_colors))

    def first_component_from(
        center: tuple[int, int],
        direction: tuple[int, int],
        start: int,
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
        if marker["kind"] == "t":
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
        else:
            counts = adjacent_component_counts(marker["cells"])
            assert counts, "small marker must touch an owner region"
            owner = max(counts, key=lambda c: (counts[c], comp_areas[c]))
            for comp in counts:
                if comp_colors[comp] == comp_colors[owner]:
                    dsu.union(owner, comp)
            marker["owner_comp"] = owner

    root_color: dict[int, int] = {}
    for i, col in enumerate(comp_colors):
        root = dsu.find(i)
        root_color.setdefault(root, int(col))
        assert root_color[root] == int(col), "merged region colors must match"

    t_markers: list[dict] = []
    small_markers: list[dict] = []
    for marker in markers:
        if marker["kind"] == "t":
            marker["source_root"] = dsu.find(marker["source_comp"])
            marker["dest_root"] = dsu.find(marker["dest_comp"])
            marker["source_input_color"] = int(root_color[marker["source_root"]])
            assert marker["source_root"] != marker["dest_root"], "swap T must connect distinct regions"
            t_markers.append(marker)
        else:
            marker["owner_root"] = dsu.find(marker["owner_comp"])
            small_markers.append(marker)

    sequence_tokens = [m for m in small_markers if m["is_sequence_token"]]
    sequence_tokens = sorted(sequence_tokens, key=lambda m: m["sequence_key"])
    assert 2 <= len(sequence_tokens) <= 4, "combo variant expects 2-4 ordering scraps"
    token_colors = [int(m["marker_color"]) for m in sequence_tokens]
    assert len(set(token_colors)) == len(token_colors), "ordering scraps must have distinct colors"

    return {
        "shape": g.shape,
        "comp_id": comp_id,
        "dsu": dsu,
        "root_color": root_color,
        "t_markers": t_markers,
        "small_markers": small_markers,
        "sequence_tokens": sequence_tokens,
    }


def _phase_groups(scene: dict, *, reverse_within_phase: bool = False) -> list[list[dict]]:
    by_color: dict[int, list[dict]] = defaultdict(list)
    for marker in scene["t_markers"]:
        by_color[int(marker["source_input_color"])].append(marker)

    groups: list[list[dict]] = []
    used = 0
    for token in scene["sequence_tokens"]:
        color = int(token["marker_color"])
        group = sorted(by_color[color], key=lambda m: m["order_key"], reverse=reverse_within_phase)
        assert group, "every ordering scrap color must activate at least one T marker"
        groups.append(group)
        used += len(group)

    assert used == len(scene["t_markers"]), "all T markers must be covered by the scrap color sequence"
    return groups


def _apply_rule(scene: dict, mode: str) -> dict[int, int]:
    current_color = dict(scene["root_color"])

    if mode == "combo":
        groups = _phase_groups(scene, reverse_within_phase=False)
        for group in groups:
            for marker in group:
                src = int(marker["source_root"])
                dst = int(marker["dest_root"])
                current_color[src], current_color[dst] = current_color[dst], current_color[src]
        return current_color

    if mode == "phase_reverse_swap":
        groups = _phase_groups(scene, reverse_within_phase=True)
        for group in groups:
            for marker in group:
                src = int(marker["source_root"])
                dst = int(marker["dest_root"])
                current_color[src], current_color[dst] = current_color[dst], current_color[src]
        return current_color

    if mode == "row_major_swap":
        ordered = sorted(scene["t_markers"], key=lambda m: m["order_key"])
        for marker in ordered:
            src = int(marker["source_root"])
            dst = int(marker["dest_root"])
            current_color[src], current_color[dst] = current_color[dst], current_color[src]
        return current_color

    if mode == "grouped_repaint":
        groups = _phase_groups(scene, reverse_within_phase=False)
        for group in groups:
            for marker in group:
                src = int(marker["source_root"])
                dst = int(marker["dest_root"])
                current_color[dst] = int(current_color[src])
        return current_color

    raise ValueError(f"unknown mode: {mode}")


def _render_scene(scene: dict, final_color: dict[int, int]) -> np.ndarray:
    comp_id = scene["comp_id"]
    dsu = scene["dsu"]
    out = np.full(scene["shape"], -1, dtype=int)
    for r in range(scene["shape"][0]):
        for c in range(scene["shape"][1]):
            cid = int(comp_id[r, c])
            if cid >= 0:
                out[r, c] = int(final_color[dsu.find(cid)])

    for marker in scene["t_markers"]:
        color = int(final_color[marker["source_root"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    for marker in scene["small_markers"]:
        color = int(final_color[marker["owner_root"]])
        for r, c in marker["cells"]:
            out[r, c] = color

    assert bool(np.all(out >= 0)), "all cells must be assigned"
    return out


def _solve_unrotated(g: np.ndarray, *, mode: str = "combo") -> np.ndarray:
    scene = _parse_scene(g)
    final_color = _apply_rule(scene, mode=mode)
    return _render_scene(scene, final_color)


def _solve_with_mode(grid: np.ndarray, *, mode: str) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = _solve_unrotated(g, mode=mode)
    l9 = _find_l9_orientation(g)
    if l9 == "top_right":
        return np.rot90(out, 1)
    if l9 == "top_left":
        return np.rot90(out, 3)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Exchange region colors through T controls, using a left-to-right scrap sequence to choose which source colors act first, then erase the controls and optionally rotate.

    Input structure:
    - The board is partitioned into irregular monochrome regions with distinct base colors.
    - Color 1 forms ordinary 4-cell T markers on region boundaries.
    - There are 2-4 small non-rectangular scraps; reading their colors from left to right gives a processing sequence.
    - An optional 5-cell color-9 L marker is embedded in one region and requests a 90-degree rotation after everything else is finished.

    Rule:
    - For each T, the stem points at the destination region, while the two diagonal cells behind the stem identify the source-side region that owns that T.
    - Group T markers by the source region's original color, then process the groups in the left-to-right order given by the scrap colors.
    - Inside one color group, process the corresponding T markers in top-left to bottom-right order, and each T swaps the current colors of its source and destination regions.
    - After all swaps, fill every T cell with the final color of its source-side region, absorb each small scrap back into its neighboring owner region, and rotate the whole result if the L marker is present: ┐ → CCW 90°, ┌ → CW 90°.

    Key insight:
    - The scraps do not repaint anything by themselves; they define which source colors get to act first. Because every action is a swap, later groups may inherit colors that earlier groups already moved through the chain.
    """
    return _solve_with_mode(grid, mode="combo")


def variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    scene = _parse_scene(g)
    phase_groups = _phase_groups(scene)
    combo = _solve_with_mode(g, mode="combo")
    row_major = _solve_with_mode(g, mode="row_major_swap")
    grouped_repaint = _solve_with_mode(g, mode="grouped_repaint")
    phase_reverse = _solve_with_mode(g, mode="phase_reverse_swap")
    return {
        "phase_count": int(len(phase_groups)),
        "t_markers": int(len(scene["t_markers"])),
        "max_phase_size": int(max(len(group) for group in phase_groups)),
        "diff_vs_row_major_swap": int(np.sum(combo != row_major)),
        "diff_vs_grouped_repaint": int(np.sum(combo != grouped_repaint)),
        "diff_vs_phase_reverse": int(np.sum(combo != phase_reverse)),
    }


# --------------- generator (scrap-sequence swap-chain + l9rot) ---------------

GRID_SIZE_MAX = 30
MIN_REGION_CELLS = 6
REGION_SIZE_FACTOR = 2.8
REGION_COUNTS = [5, 6, 7]
REGION_WEIGHTS = [2, 3, 1]

COLOR_POOL = [0, 2, 3, 4, 5, 6, 7, 8]

PHASE_COUNT_OPTIONS = [2, 3, 4]
PHASE_COUNT_WEIGHTS = [5, 4, 2]

L9_PROBABILITY = 0.5
L9_OFFSETS = {
    "top_right": [(0, 0), (0, -1), (0, -2), (1, 0), (2, 0)],
    "top_left": [(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)],
}
SCRAP_SHAPES = [
    [(0, 0), (1, 0), (0, 1)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 1), (1, 0), (1, 1)],
]

SEED_ATTEMPTS = 200
GEN_RETRIES = 500
PLACEMENT_BRANCH_LIMIT = 24

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
    out = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


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


def _choose_phase_sources(adjacency: dict[int, set[int]], phase_count: int) -> list[int] | None:
    sizes = _phase_sizes(phase_count)
    nodes = [rid for rid, nbs in adjacency.items() if len(nbs) >= sizes[0]]
    for _ in range(300):
        start = random.choice(nodes)
        path = [int(start)]
        used = {int(start)}
        while len(path) < phase_count:
            pos = len(path)
            options = [
                rid
                for rid in adjacency[path[-1]]
                if rid not in used and len(adjacency[rid]) >= sizes[pos]
            ]
            if not options:
                break
            random.shuffle(options)
            options.sort(key=lambda rid: (-len(adjacency[rid]), rid))
            nxt = int(options[0])
            path.append(nxt)
            used.add(nxt)
        if len(path) == phase_count:
            return path
    return None


def _phase_sizes(phase_count: int) -> list[int]:
    if phase_count == 2:
        return [2, 2]
    if phase_count == 3:
        return [2, 1, 1]
    if phase_count == 4:
        return [2, 1, 1, 1]
    raise ValueError(f"unsupported phase count: {phase_count}")


def _choose_marker_defs(adjacency: dict[int, set[int]], phase_sources: list[int]) -> list[list[dict]] | None:
    phase_defs: list[list[dict]] = []
    sizes = _phase_sizes(len(phase_sources))

    for i, src in enumerate(phase_sources):
        chain_dst = phase_sources[i + 1] if i + 1 < len(phase_sources) else None
        need = sizes[i]
        defs: list[dict] = []
        if chain_dst is not None:
            defs.append({"src": int(src), "dst": int(chain_dst)})
            need -= 1

        extras = [
            int(dst)
            for dst in adjacency[src]
            if chain_dst is None or int(dst) != int(chain_dst)
        ]
        random.shuffle(extras)
        extras.sort(key=lambda dst: (dst in phase_sources, dst))
        picked: list[int] = []
        if need > 0:
            for dst in extras:
                picked.append(int(dst))
                if len(picked) == need:
                    break
        if len(picked) != need:
            return None

        extra_defs = [{"src": int(src), "dst": int(dst)} for dst in picked]
        phase_defs.append(extra_defs + defs)

    return phase_defs


def _find_valid_t_placements(
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
                    candidates.append(((int(r), int(c)), (int(stem[0]), int(stem[1]))))
    return candidates


def _register_t_marker(
    center: tuple[int, int],
    stem: tuple[int, int],
    occupied_cells: set[tuple[int, int]],
    occupied_neighbors: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    new_cells = set(occupied_cells)
    new_neighbors = set(occupied_neighbors)
    cells = _t_cells(center, stem)
    new_cells |= cells
    new_neighbors |= (_t_neighborhood(center, stem) - cells)
    return new_cells, new_neighbors


def _place_t_markers(
    region: np.ndarray,
    phase_defs: list[list[dict]],
    H: int,
    W: int,
) -> list[dict] | None:
    flat_defs: list[tuple[int, int, dict]] = []
    for phase_idx, defs in enumerate(phase_defs):
        for within_idx, base in enumerate(defs):
            flat_defs.append((phase_idx, within_idx, base))

    def dfs(
        index: int,
        occupied_cells: set[tuple[int, int]],
        occupied_neighbors: set[tuple[int, int]],
        last_keys: dict[int, tuple[int, int]],
        placed: list[dict],
    ) -> list[dict] | None:
        if index == len(flat_defs):
            return placed

        phase_idx, _within_idx, base = flat_defs[index]
        prev_key = last_keys.get(phase_idx)
        cands = _find_valid_t_placements(
            region,
            base["src"],
            base["dst"],
            occupied_cells,
            occupied_neighbors,
            H,
            W,
        )
        if prev_key is not None:
            cands = [cand for cand in cands if _placement_key(*cand) > prev_key]
        random.shuffle(cands)
        if len(cands) > PLACEMENT_BRANCH_LIMIT:
            cands = cands[:PLACEMENT_BRANCH_LIMIT]

        for center, stem in cands:
            order_key = _placement_key(center, stem)
            next_cells, next_neighbors = _register_t_marker(center, stem, occupied_cells, occupied_neighbors)
            next_last = dict(last_keys)
            next_last[phase_idx] = order_key
            marker = {
                "src": int(base["src"]),
                "dst": int(base["dst"]),
                "center": center,
                "stem": stem,
                "order_key": order_key,
                "phase_idx": int(phase_idx),
            }
            out = dfs(index + 1, next_cells, next_neighbors, next_last, placed + [marker])
            if out is not None:
                return out
        return None

    return dfs(0, set(), set(), {}, [])


def _build_color_map(region: np.ndarray, colors: list[int]) -> np.ndarray:
    lut = np.array(colors, dtype=int)
    return lut[region]


def _simulate_combo_region_colors(
    input_colors: list[int],
    phase_sources: list[int],
    placed_markers: list[dict],
) -> list[int]:
    current = list(map(int, input_colors))
    grouped: dict[int, list[dict]] = defaultdict(list)
    for marker in placed_markers:
        grouped[int(marker["src"])].append(marker)
    for src in phase_sources:
        group = sorted(grouped[int(src)], key=lambda m: m["order_key"])
        for marker in group:
            a = int(marker["src"])
            b = int(marker["dst"])
            current[a], current[b] = current[b], current[a]
    return current


def _stamp_t_input(grid: np.ndarray, placed_markers: list[dict]) -> None:
    for marker in placed_markers:
        for r, c in _t_cells(marker["center"], marker["stem"]):
            grid[r, c] = 1


def _stamp_t_output(grid: np.ndarray, placed_markers: list[dict], out_colors: list[int]) -> None:
    for marker in placed_markers:
        color = int(out_colors[marker["src"]])
        for r, c in _t_cells(marker["center"], marker["stem"]):
            grid[r, c] = color


def _scrap_neighborhood(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set()
    for r, c in cells:
        for dr, dc in _DIRS:
            nb = (r + dr, c + dc)
            if nb not in cells:
                out.add(nb)
    return out


def _find_scrap_candidates(
    region: np.ndarray,
    blocked_cells: set[tuple[int, int]],
    blocked_neighbors: set[tuple[int, int]],
    source_rid: int,
) -> list[dict]:
    H, W = region.shape
    candidates: list[dict] = []
    for shape in SCRAP_SHAPES:
        max_r = max(dr for dr, _dc in shape)
        max_c = max(dc for _dr, dc in shape)
        for r in range(H - max_r):
            for c in range(W - max_c):
                cells = {(int(r + dr), int(c + dc)) for dr, dc in shape}
                if cells & blocked_cells:
                    continue
                if cells & blocked_neighbors:
                    continue
                owner = int(region[r, c])
                if owner == int(source_rid):
                    continue
                if any(int(region[rr, cc]) != owner for rr, cc in cells):
                    continue
                candidates.append(
                    {
                        "cells": cells,
                        "owner": owner,
                        "sequence_key": _scrap_key(cells),
                    }
                )
    return candidates


def _place_scraps(
    region: np.ndarray,
    phase_sources: list[int],
    input_colors: list[int],
    blocked_cells: set[tuple[int, int]],
    blocked_neighbors: set[tuple[int, int]],
) -> list[dict] | None:
    token_colors = [int(input_colors[src]) for src in phase_sources]
    candidate_lists: list[list[dict]] = []
    for src in phase_sources:
        cands = _find_scrap_candidates(region, blocked_cells, blocked_neighbors, int(src))
        random.shuffle(cands)
        cands.sort(key=lambda cand: cand["sequence_key"])
        candidate_lists.append(cands[:80])
        if not candidate_lists[-1]:
            return None

    def dfs(
        idx: int,
        prev_col: int,
        cur_cells: set[tuple[int, int]],
        placed: list[dict],
    ) -> list[dict] | None:
        if idx == len(phase_sources):
            return placed
        for cand in candidate_lists[idx]:
            seq_col, _seq_row = cand["sequence_key"]
            if seq_col <= prev_col:
                continue
            cells = cand["cells"]
            if cells & cur_cells:
                continue
            marker = {
                "cells": cells,
                "owner": int(cand["owner"]),
                "color": int(token_colors[idx]),
                "sequence_key": cand["sequence_key"],
            }
            out = dfs(
                idx + 1,
                int(seq_col),
                cur_cells | cells,
                placed + [marker],
            )
            if out is not None:
                return out
        return None

    return dfs(0, -10, set(blocked_cells) | set(blocked_neighbors), [])


def _stamp_scrap_input(grid: np.ndarray, scraps: list[dict]) -> None:
    for scrap in scraps:
        for r, c in scrap["cells"]:
            grid[r, c] = int(scrap["color"])


def _stamp_scrap_output(grid: np.ndarray, scraps: list[dict], out_colors: list[int]) -> None:
    for scrap in scraps:
        color = int(out_colors[scrap["owner"]])
        for r, c in scrap["cells"]:
            grid[r, c] = color


def _try_place_l9(
    region: np.ndarray,
    blocked_cells: set[tuple[int, int]],
    blocked_neighbors: set[tuple[int, int]],
    input_grid: np.ndarray,
    output_grid: np.ndarray,
    out_colors: list[int],
) -> str | None:
    if random.random() >= L9_PROBABILITY:
        return None
    orient = random.choice(list(L9_OFFSETS))
    offsets = L9_OFFSETS[orient]
    H, W = region.shape

    candidates = []
    for r in range(H):
        for c in range(W):
            cells = {(r + dr, c + dc) for dr, dc in offsets}
            if not all(0 <= rr < H and 0 <= cc < W for rr, cc in cells):
                continue
            if cells & blocked_cells or cells & blocked_neighbors:
                continue
            owner = int(region[r, c])
            if not all(int(region[rr, cc]) == owner for rr, cc in cells):
                continue
            candidates.append((cells, owner))

    if not candidates:
        return None
    cells, owner = random.choice(candidates)
    for r, c in cells:
        input_grid[r, c] = 9
        output_grid[r, c] = int(out_colors[owner])
    return orient


def _occupied_from_t_markers(placed_markers: list[dict]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    cells: set[tuple[int, int]] = set()
    neighbors: set[tuple[int, int]] = set()
    for marker in placed_markers:
        cells, neighbors = _register_t_marker(marker["center"], marker["stem"], cells, neighbors)
    return cells, neighbors


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choices(REGION_COUNTS, weights=REGION_WEIGHTS)[0]
    min_size = max(10, int(n_regions * REGION_SIZE_FACTOR))
    H = random.randint(min_size, GRID_SIZE_MAX)
    W = random.randint(min_size, GRID_SIZE_MAX)

    region = _grow_regions(H, W, n_regions)
    if region is None:
        return None

    adjacency = _region_adjacency(region)
    pool = list(COLOR_POOL)
    random.shuffle(pool)
    if n_regions > len(pool):
        return None
    input_colors = pool[:n_regions]

    max_phase_count = min(max(PHASE_COUNT_OPTIONS), n_regions)
    phase_choices = [p for p in PHASE_COUNT_OPTIONS if p <= max_phase_count]
    weights = [PHASE_COUNT_WEIGHTS[PHASE_COUNT_OPTIONS.index(p)] for p in phase_choices]
    phase_count = random.choices(phase_choices, weights=weights)[0]
    phase_sources = _choose_phase_sources(adjacency, phase_count)
    if phase_sources is None:
        return None

    phase_defs = _choose_marker_defs(adjacency, phase_sources)
    if phase_defs is None:
        return None

    placed_markers = _place_t_markers(region, phase_defs, H, W)
    if placed_markers is None:
        return None

    occupied_cells, occupied_neighbors = _occupied_from_t_markers(placed_markers)
    scraps = _place_scraps(region, phase_sources, input_colors, occupied_cells, occupied_neighbors)
    if scraps is None:
        return None

    out_colors = _simulate_combo_region_colors(input_colors, phase_sources, placed_markers)
    input_grid = _build_color_map(region, input_colors)
    output_grid = _build_color_map(region, out_colors)

    _stamp_t_input(input_grid, placed_markers)
    _stamp_t_output(output_grid, placed_markers, out_colors)
    _stamp_scrap_input(input_grid, scraps)
    _stamp_scrap_output(output_grid, scraps, out_colors)

    scrap_cells = set().union(*(scrap["cells"] for scrap in scraps)) if scraps else set()
    scrap_neighbors = set().union(*(_scrap_neighborhood(scrap["cells"]) for scrap in scraps)) if scraps else set()
    l9_orient = _try_place_l9(
        region,
        occupied_cells | scrap_cells,
        occupied_neighbors | scrap_neighbors,
        input_grid,
        output_grid,
        out_colors,
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
    if metrics["phase_count"] < 2:
        return None
    if metrics["max_phase_size"] < 2:
        return None
    if metrics["diff_vs_row_major_swap"] <= 0:
        return None
    if metrics["diff_vs_grouped_repaint"] <= 0:
        return None
    if metrics["diff_vs_phase_reverse"] <= 0:
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
