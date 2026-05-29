# Combo variant of 80a900e0 (index 11)
# combo_spec: D7/2+D14/0
# Name: 颜色编码数量加激活条件
"""
design_analysis:
  fused_rule_summary:
    - 保留原题的 0/1 棋盘格背景、8 像素空心菱形、以及贴着菱形边外侧的短对角线段。
    - D7/2 在这里具体化为“菱形颜色=该菱形外边框数量”，只使用 2/3/4 三种菱形色，
      因而红菱形有 2 条外边框，绿菱形有 3 条，黄菱形有 4 条。
    - D14/0 叠加为“只有外边框颜色值大于菱形颜色值时才发射”；颜色值不够大的外边框
      仍然在输入中可见，但输出里不延伸射线。
  concrete_behavior_trace:
    - 原题：每个菱形周围凡是存在的短对角线段，都会从两个端点沿外法线方向发射两条对角射线。
    - 融合后：先看菱形颜色决定“应该有几条外边框”，再对这些边框逐条检查颜色关系。
      例：颜色 3 的菱形必须有 3 条外边框；若这三条边框颜色分别是 2/4/5，则只有 4 和 5
      那两条发射，颜色 2 的那条保持短线段本体不动。
    - 这和原题的差异有两层：输入里每种菱形的边框数量不再随意取 2-4，而是被颜色固定；
      输出里也不再是“所有短线段都发射”，而是同一个菱形周围会同时出现激活边和惰性边。
  complexity_calibration:
    - 为了让颜色→数量和颜色大小规律在 2-4 个样例内可见，每个样例固定同时出现 2/3/4
      三种菱形色，因此三档数量 2/3/4 会反复出现。
    - 外边框颜色限制在 2/3/4/5 这条小链上：2 菱形只连更大的 3/4/5，3 菱形混入惰性的 2，
      4 菱形混入惰性的 2/3 与激活的 5。这样从样例中能观察到稳定的相对次序，而不是无结构乱色。
    - 不增加新的形状类型，只改变“有几条边”和“哪些边会发射”，视觉差异足够明显但不会过载。
  implicit_assumption_audit:
    - 原 `main()` 隐含“每个 diamond 周围找到的所有合法外线段都要发射”；D14/0 破坏了这个前提，
      因此必须在发射前加上 `outer_color > diamond_color` 条件。
    - 原 `main()` 对外线段数量没有语义要求；D7/2 让数量本身变成颜色编码，所以 solver 不能再把
      任意个候选线段都接受。这里显式要求候选段数必须等于该菱形颜色编码的数量。
    - 原 `top-2 colors = bg` 仍然成立，但这要求任一前景单色都不能比 0 或 1 更常见；
      generator 保留原题的大网格和棋盘底，并做频次检查来保证这一点。
    - 原“相邻非背景像素都是本菱形的外边框”依赖于菱形 8 邻域内没有别的对象；generator 为每个
      菱形保留相邻空壳，只允许本菱形自己的短线段占用这些邻位。
    - 若 inactive 外边框外侧立刻被别的对象挡住，则按融合逻辑与按原逻辑的输出可能一样；
      generator 因此为 inactive 方向也保留完整空走廊，确保激活条件真实起作用。
  placement_legality:
    - 仍用 output-first。先放菱形、短线段和所有 active 射线得到输出，再删去 active 射线恢复输入。
    - 搜索放置时不仅禁止输出像素重叠，还禁止不同对象占用彼此“潜在射线走廊”，包括 inactive 边
      本来会走过但最终保持背景的通道，从而避免逻辑退化和视觉遮挡。
    - 允许边界裁切后的短线段长度为 2 或 3；少于 2 就拒绝该摆放。
  visual_observability:
    - 同一张图里同时有 2/3/4 三种菱形，解题者能直接数出“红=2 边、绿=3 边、黄=4 边”。
    - 对 3 色和 4 色菱形，输入中可同时看到会发射与不会发射的边框；输出只延伸其中一部分，
      使 D14/0 的效果在单样例内也可见。
    - 虽然激活条件最终仍由颜色值大小决定，但生成器把参与者收缩到 2<3<4<5 这条小链，并在
      多个对象上重复同一相对关系，使其可从样例中归纳，而不是依赖黑箱数值。
  visual_cleanliness_and_inference:
    - 每条走廊只服务于一个外边框，active 射线与 inactive 空走廊都不与别的对象交叉，图面清爽。
    - 同色复用只发生在同类角色上，不会把菱形和外边框混成一个连通块。
    - 没有随机排序、隐藏 tie-break 或额外元信息；全部规律都体现在可数的线段数量和可见的发射/不发射结果中。
  new_solver_assumptions_and_generator_guarantees:
    - solver 新增假设：合法输入只含颜色 2/3/4 的菱形，并且该颜色正好编码外边框数；
      generator 固定使用这三种颜色并精确构造 2/3/4 条边。
    - solver 新增假设：每条识别到的外边框都是单色对角短线段组件；generator 用模板几何并保留菱形壳层避免误检。
    - solver 新增假设：inactive 边外侧若按原题发射，本应能走出一段背景通道；generator 为其保留完整空走廊，
      并额外验证融合版输出与“所有边都发射”的基线输出确实不同。
"""

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, neighbors8


def _cc8(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8)]


def _bbox(pts: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.asarray(pts))


def _top_k_colors(g: np.ndarray, k: int) -> list[int]:
    return [int(c) for c, _ in Counter(map(int, g.ravel())).most_common(k)]


def _components8_by_color(g: np.ndarray, colors: set[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    return [(int(col), pts) for col in colors for pts in _cc8(g == col)]


_DIAMOND8_OFFSETS = {
    (0, 2),
    (1, 1),
    (1, 3),
    (2, 0),
    (2, 4),
    (3, 1),
    (3, 3),
    (4, 2),
}


def _is_diamond8(comp: list[tuple[int, int]]) -> bool:
    if len(comp) != 8:
        return False
    rs = [r for r, _ in comp]
    cs = [c for _, c in comp]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    if (r1 - r0) != 4 or (c1 - c0) != 4:
        return False
    norm = {(r - r0, c - c0) for r, c in comp}
    return norm == _DIAMOND8_OFFSETS


def _outer_line_info(pts: list[tuple[int, int]]) -> tuple[str, list[tuple[int, int]]] | None:
    if len(pts) < 2:
        return None
    rs = [r for r, _ in pts]
    cs = [c for _, c in pts]
    if all((r - c) == (rs[0] - cs[0]) for r, c in pts):
        return ("rc_diff", sorted(pts, key=lambda p: (p[0], p[1])))
    if all((r + c) == (rs[0] + cs[0]) for r, c in pts):
        return ("rc_sum", sorted(pts, key=lambda p: (p[0], -p[1])))
    return None


def _expected_outer_count(diamond_color: int) -> int | None:
    if diamond_color in (2, 3, 4):
        return diamond_color
    return None


def _edge_rank_for_component(
    oc: list[tuple[int, int]], diamond_bbox: tuple[int, int, int, int]
) -> int:
    r0, c0, _, _ = diamond_bbox
    center_r = r0 + 2
    center_c = c0 + 2
    mean_r = sum(r for r, _ in oc) / len(oc)
    mean_c = sum(c for _, c in oc) / len(oc)
    if mean_r < center_r and mean_c < center_c:
        return 0  # TL
    if mean_r < center_r and mean_c > center_c:
        return 1  # TR
    if mean_r > center_r and mean_c < center_c:
        return 2  # BL
    return 3  # BR


def _solve(grid: np.ndarray, *, gated: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg_colors = set(_top_k_colors(g, 2))
    non_bg = set(int(v) for v in np.unique(g)) - bg_colors
    comps = _components8_by_color(g, non_bg)
    diamonds: list[tuple[int, list[tuple[int, int]]]] = [(col, pts) for col, pts in comps if _is_diamond8(pts)]
    out = g.copy()

    def bbox_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[int, int, int, int]:
        _, pts = item
        return _bbox(pts)

    for dcol, dpts in sorted(diamonds, key=bbox_key):
        expected_count = _expected_outer_count(dcol)
        if expected_count is None:
            continue

        dset = set(dpts)
        diamond_bbox = _bbox(dpts)
        adj: list[tuple[int, int]] = []
        for r, c in dpts:
            for rr, cc in neighbors8(r, c):
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset:
                    if int(g[rr, cc]) not in bg_colors:
                        adj.append((rr, cc))

        if not adj:
            continue

        adj_mask = np.zeros((h, w), dtype=bool)
        for r, c in adj:
            adj_mask[r, c] = True

        candidates = []
        for oc in _cc8(adj_mask):
            info = _outer_line_info(oc)
            if info is None:
                continue
            candidates.append((oc, info, _edge_rank_for_component(oc, diamond_bbox)))

        if len(candidates) != expected_count:
            continue

        candidates.sort(key=lambda item: item[2])
        for oc, (slope_key, endpoints), _ in candidates:
            ocol = int(g[oc[0][0], oc[0][1]])
            if gated and ocol <= dcol:
                continue

            if slope_key == "rc_diff":
                normals = [(-1, 1), (1, -1)]
            else:
                normals = [(-1, -1), (1, 1)]

            best_n = None
            best_score = -1
            for n in normals:
                score = sum(((r + n[0], c + n[1]) in dset) for r, c in oc)
                if score > best_score:
                    best_score = score
                    best_n = n
            if best_n is None or best_score <= 0:
                continue
            outward = (-best_n[0], -best_n[1])

            for pr, pc in (endpoints[0], endpoints[-1]):
                rr, cc = pr + outward[0], pc + outward[1]
                while 0 <= rr < h and 0 <= cc < w:
                    if int(g[rr, cc]) in bg_colors:
                        out[rr, cc] = ocol
                    else:
                        break
                    rr += outward[0]
                    cc += outward[1]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend endpoint rays from short outer segments around hollow diamonds on a checkerboard, but only for color-qualified segments.

    Input structure:
    - A 0/1 checkerboard background, identified as the two most common colors.
    - Hollow 8-pixel diamonds in non-background colors; in this fused variant the diamond color also encodes how many outer segments belong to it.
    - Short diagonal outer segments adjacent to selected diamond edges, each in its own non-background color.

    Rule:
    - Detect each diamond and gather its adjacent diagonal short segments; keep only diamonds whose segment count matches the number encoded by the diamond color.
    - For every such segment, infer the outward diagonal exactly as in the base task.
    - Emit the two endpoint rays only when the segment color value is greater than the diamond color value; otherwise the segment stays inert.

    Key insight:
    - Color now plays two different roles at once: the diamond color determines how many outer segments should be present, and the segment-versus-diamond color ordering determines which visible segments actually fire.
    """
    return _solve(grid, gated=True)


GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (24, 30)
DIAMOND_COLORS = (2, 3, 4)
MIN_SEGMENT_LEN = 2
MIN_RAY_CELLS = 4
MAX_GEN_RETRIES = 500

_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TR": {"segment": ((-1, 3), (0, 4), (1, 5)), "outward": (-1, 1), "slope": "diff"},
    "TL": {"segment": ((-1, 1), (0, 0), (1, -1)), "outward": (-1, -1), "slope": "sum"},
    "BL": {"segment": ((3, -1), (4, 0), (5, 1)), "outward": (1, -1), "slope": "diff"},
    "BR": {"segment": ((3, 5), (4, 4), (5, 3)), "outward": (1, 1), "slope": "sum"},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _sort_segment(cells: list[tuple[int, int]], slope: str) -> list[tuple[int, int]]:
    if slope == "diff":
        return sorted(cells, key=lambda p: (p[0], p[1]))
    return sorted(cells, key=lambda p: (p[0], -p[1]))


def _ray_cells(shape: tuple[int, int], start: tuple[int, int], direction: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = direction
    r, c = start[0] + dr, start[1] + dc
    cells: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        cells.append((r, c))
        r += dr
        c += dc
    return cells


def _diamond_shell(
    shape: tuple[int, int],
    diamond: list[tuple[int, int]],
    allowed_adjacent: set[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    dset = set(diamond)
    shell: set[tuple[int, int]] = set()
    for r, c in diamond:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in dset and (rr, cc) not in allowed_adjacent:
                shell.add((rr, cc))
    return shell


def _segment_color_multiset(diamond_color: int) -> list[int]:
    if diamond_color == 2:
        cols = random.sample([3, 4, 5], 2)
    elif diamond_color == 3:
        cols = [2, 4, 5]
    elif diamond_color == 4:
        cols = [2, 3, 5, 5]
    else:
        raise ValueError(f"unsupported diamond color: {diamond_color}")
    random.shuffle(cols)
    return cols


def _build_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for dcol in DIAMOND_COLORS:
        edge_count = dcol
        edges = random.sample(list(_EDGE_NAMES), edge_count)
        seg_colors = _segment_color_multiset(dcol)
        specs.append(
            {
                "diamond_color": dcol,
                "segments": [
                    {
                        "edge": edge,
                        "color": seg_colors[i],
                        "active": seg_colors[i] > dcol,
                    }
                    for i, edge in enumerate(edges)
                ],
            }
        )
    random.shuffle(specs)
    return specs


def _candidate_placement(
    h: int,
    w: int,
    r0: int,
    c0: int,
    spec: dict[str, object],
) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    all_cells = set(diamond)
    guard_cells = set(diamond)
    allowed_adjacent: set[tuple[int, int]] = set()
    seg_parts: list[dict[str, object]] = []

    for seg_spec in spec["segments"]:
        info = _EDGE_INFO[seg_spec["edge"]]
        segment = [
            (r0 + dr, c0 + dc)
            for dr, dc in info["segment"]
            if 0 <= r0 + dr < h and 0 <= c0 + dc < w
        ]
        if len(segment) < MIN_SEGMENT_LEN:
            return None
        segment = _sort_segment(segment, info["slope"])
        potential_rays = _ray_cells((h, w), segment[0], info["outward"]) + _ray_cells(
            (h, w), segment[-1], info["outward"]
        )
        if len(potential_rays) < MIN_RAY_CELLS:
            return None

        part_guard = segment + potential_rays
        if len(set(part_guard)) != len(part_guard):
            return None
        if any(cell in guard_cells for cell in part_guard):
            return None

        allowed_adjacent.update(segment)
        guard_cells.update(part_guard)

        active_rays = potential_rays if seg_spec["active"] else []
        output_cells = segment + active_rays
        if any(cell in all_cells for cell in output_cells):
            return None
        all_cells.update(output_cells)

        seg_parts.append(
            {
                "edge": seg_spec["edge"],
                "color": seg_spec["color"],
                "active": seg_spec["active"],
                "segment": segment,
                "rays": active_rays,
                "potential_rays": potential_rays,
            }
        )

    shell = _diamond_shell((h, w), diamond, allowed_adjacent)
    if not guard_cells.isdisjoint(shell):
        return None

    return {
        "diamond_color": spec["diamond_color"],
        "diamond": diamond,
        "segments": seg_parts,
        "all_cells": frozenset(all_cells),
        "guard_cells": frozenset(guard_cells),
        "shell": frozenset(shell),
    }


def _search_placements(h: int, w: int, specs: list[dict[str, object]]) -> list[dict[str, object]] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: (
            sum(1 for seg in specs[i]["segments"] if seg["active"]),
            len(specs[i]["segments"]),
        ),
        reverse=True,
    )
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[int, list[dict[str, object]]] = {}
    for idx in order:
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[idx] = []
        for r0, c0 in positions:
            placement = _candidate_placement(h, w, r0, c0, specs[idx])
            if placement is not None:
                candidate_cache[idx].append(placement)
        if not candidate_cache[idx]:
            return None

    occupied_all: set[tuple[int, int]] = set()
    occupied_guard: set[tuple[int, int]] = set()
    blocked_shell: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        for placement in candidate_cache[spec_i]:
            if not occupied_guard.isdisjoint(placement["guard_cells"]):
                continue
            if not blocked_shell.isdisjoint(placement["all_cells"]):
                continue
            if not occupied_all.isdisjoint(placement["shell"]):
                continue

            occupied_all.update(placement["all_cells"])
            occupied_guard.update(placement["guard_cells"])
            blocked_shell.update(placement["shell"])
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_all.difference_update(placement["all_cells"])
            occupied_guard.difference_update(placement["guard_cells"])
            blocked_shell.difference_update(placement["shell"])
            placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in range(2, 10)), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    specs = _build_specs()
    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    out = _checkerboard(h, w)
    active_ray_cells: set[tuple[int, int]] = set()
    inactive_potential_cells = 0
    inactive_segment_count = 0
    active_segment_count = 0

    for placement in placements:
        _paint_cells(out, placement["diamond"], placement["diamond_color"])
        for seg in placement["segments"]:
            _paint_cells(out, seg["segment"], seg["color"])
            if seg["active"]:
                _paint_cells(out, seg["rays"], seg["color"])
                active_ray_cells.update(seg["rays"])
                active_segment_count += 1
            else:
                inactive_segment_count += 1
                inactive_potential_cells += len(seg["potential_rays"])

    if inactive_segment_count == 0 or active_segment_count == 0 or inactive_potential_cells == 0:
        return None
    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if active_ray_cells:
        rs, cs = zip(*sorted(active_ray_cells))
        rs_arr = np.array(rs)
        cs_arr = np.array(cs)
        inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

    fused_out = main(inp)
    if not np.array_equal(fused_out, out):
        return None
    if np.array_equal(_solve(inp, gated=False), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
