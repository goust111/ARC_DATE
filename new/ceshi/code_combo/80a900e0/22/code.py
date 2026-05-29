# Combo variant of 80a900e0 (index 22)
# combo_spec: D2/1+D6/0+D7/0
# Name: 极简单射
#
# Design analysis for the fused combo
# a) Concrete behavior:
#    原题的可见外物体是 3 像素对角线段，每条线段从两个端点各发 1 条射线；现在三项改动
#    叠加后，输入里每个菱形只保留 1 个外标记，而且该标记缩成单像素。这个单像素固定落在
#    5x5 菱形 bbox 的四个角之一，分别对应 TL/TR/BL/BR 四条边。融合描述额外规定：
#    这个唯一像素本身就是唯一发射点，因此不再恢复“虚拟 3 像素线段”再从两个端点发射，
#    而是直接从该角点沿对应 outward 对角方向发出 1 条射线。
#    例子：若菱形 bbox 为 rows 10..14, cols 20..24，并在右上角 (10,24) 放一个橙色点，
#    原题会需要 3 像素 TR 外线段并从两个端点向 (-1,+1) 发射两条射线；D6 单变体会把该点
#    当作线段中点，再恢复两个虚拟端点继续发双射线；本融合版本则只从 (10,24) 这个点本身
#    向 (-1,+1) 连续涂色，得到一条更稀疏也更极简的单射线。
# b) Complexity calibration:
#    只保留 4 个离散角点位置和 4 个离散 outward 方向，避免自由度过多；同时每个菱形严格
#    只有 1 个点标记，视觉上与原题的“多条 3 像素线段”差异足够明显。生成器仍保留 1-3 个
#    菱形，并尽量让同一网格内出现至少两种 corner 类型，保证人类能从少量样例中看到“角点
#    决定方向、每个点只延长一条线”的重复证据。
# c) Implicit assumptions audit:
#    - 原 main() 依赖 `_outer_line_info` 从 >=2 像素外线段恢复斜率；单像素标记下该假设失效，
#      必须改成 `corner -> edge -> outward` 查表。
#    - 原 main() 假设每个 outer component 都有两个端点；融合后不存在可见双端点，必须删除
#      “两个端点都发射”的逻辑，改成“角点标记本身就是唯一发射点”。
#    - 原 main() 通过 diamond 邻接非背景像素搜外线段；融合后若周围对象贴得太近，其他对象也
#      可能落到某个 diamond bbox 角上并被误判为 marker，因此生成器必须保证每个 diamond 的
#      四个 bbox 角除了它自己的唯一 marker 外都保持背景。
#    - D2/1 在单独实现里需要比较两端点离中心远近；但在 D6/0 + D7/0 叠加后，融合描述已经把
#      “最远端点”收缩成唯一可见像素，所以 solver 不再做距离比较，generator 也必须生成这种
#      唯一可观察端点，而不是对称点位。
# d) Placement legality:
#    生成器采用 output-first。先放 diamond，再放唯一 marker，再计算该 marker 的实际单射线。
#    搜索时同时为“本融合输出会出现的单射线”和“若错误按 D6 双端点逻辑恢复时会出现的额外
#    射线”预留空间，确保对象不重叠、不越界，也确保融合行为不会被别的物体遮掉而退化。
# e) Visual observability:
#    解题者能直接看到：每个菱形只有 1 个角点标记；输出里每个点只延长 1 条同色对角线，
#    而不是一对平行线。corner 位置提供了全部可见线索，不依赖颜色值、生成顺序或隐藏状态。
# f) Visual cleanliness:
#    输入只含 checkerboard、空心菱形和稀疏角点，输出只额外增加单条细对角线。生成器对输入
#    物体加 8 邻域缓冲，并禁止实际/对照射线互撞，避免多个对象堆在同一区域导致难以辨认。
# g) Color inferability:
#    颜色规则保持简单：marker 与其射线同色，diamond 使用不同颜色。颜色只起到“哪条线属于
#    哪个点”的分组作用，不参与角点判定或发射方向选择。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np


def _find_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "solvers" / "arc_utils.py").exists():
            return parent
    raise RuntimeError("could not locate project root containing solvers/arc_utils.py")


ROOT = _find_root()
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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


def _checkerboard(h: int, w: int) -> np.ndarray:
    rows, cols = np.indices((h, w))
    return ((rows + cols + 1) % 2).astype(int)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


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


def _expand8(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out = set(cells)
    for r, c in cells:
        for rr, cc in neighbors8(r, c):
            if 0 <= rr < h and 0 <= cc < w:
                out.add((rr, cc))
    return out


def _legacy_dual_rays(
    shape: tuple[int, int], marker: tuple[int, int], tangent: tuple[int, int], outward: tuple[int, int]
) -> list[tuple[int, int]]:
    a = (marker[0] - tangent[0], marker[1] - tangent[1])
    b = (marker[0] + tangent[0], marker[1] + tangent[1])
    return _ray_cells(shape, a, outward) + _ray_cells(shape, b, outward)


# ── Tunable parameters ──────────────────────────────────────
GRID_H_RANGE = (20, 24)
GRID_W_RANGE = (14, 30)
DIAMOND_COUNT_RANGE = (1, 3)
MIN_RAY_CELLS = 1
COLOR_RANGE = range(2, 10)
MAX_GEN_RETRIES = 400

# ── Geometry constants ──────────────────────────────────────
_DIAMOND_COORDS = (
    (0, 2), (1, 1), (1, 3),
    (2, 0), (2, 4),
    (3, 1), (3, 3), (4, 2),
)

_EDGE_INFO = {
    "TL": {"marker": (0, 0), "outward": (-1, -1), "tangent": (1, -1)},
    "TR": {"marker": (0, 4), "outward": (-1, 1), "tangent": (1, 1)},
    "BL": {"marker": (4, 0), "outward": (1, -1), "tangent": (1, 1)},
    "BR": {"marker": (4, 4), "outward": (1, 1), "tangent": (1, -1)},
}
_EDGE_NAMES = tuple(_EDGE_INFO)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend one diagonal ray outward from a single corner marker beside each hollow diamond.

    Input structure:
    - A checkerboard background made from the two dominant colors.
    - One or more 8-pixel hollow diamonds in 5x5 bounding boxes.
    - For each diamond, exactly one extra non-background pixel at one bbox corner; that pixel is
      the only outer marker for the diamond.

    Rule:
    - Use the marker's corner position (TL/TR/BL/BR) to determine which diamond edge it refers to
      and which diagonal direction points away from the diamond.
    - Treat the marker itself as the unique emitter and cast exactly one ray from the next cell in
      that outward diagonal direction.
    - Color background cells along that ray with the marker color until the path hits a non-background
      cell or leaves the grid.

    Key insight:
    - The sparse corner point already encodes everything: which edge is active, which outward
      diagonal to follow, and that only a single ray should appear. There is no hidden 3-pixel segment
      to reconstruct and no second endpoint to use.
    """
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

    for _, dpts in sorted(diamonds, key=bbox_key):
        r0, c0, r1, c1 = _bbox(dpts)
        corner_to_edge = {
            (r0, c0): "TL",
            (r0, c1): "TR",
            (r1, c0): "BL",
            (r1, c1): "BR",
        }

        markers: list[tuple[int, int, str, int]] = []
        for (mr, mc), edge in corner_to_edge.items():
            if not (0 <= mr < h and 0 <= mc < w):
                continue
            mcol = int(g[mr, mc])
            if mcol in bg_colors:
                continue
            markers.append((mr, mc, edge, mcol))

        # D7/0: each diamond must have exactly one visible outer marker.
        if len(markers) != 1:
            continue

        mr, mc, edge, mcol = markers[0]
        dr, dc = _EDGE_INFO[edge]["outward"]
        rr, cc = mr + dr, mc + dc
        while 0 <= rr < h and 0 <= cc < w:
            if int(out[rr, cc]) in bg_colors:
                out[rr, cc] = mcol
            else:
                break
            rr += dr
            cc += dc

    return out


def _candidate_placement(h: int, w: int, r0: int, c0: int, edge: str) -> dict[str, object] | None:
    diamond = [(r0 + dr, c0 + dc) for dr, dc in _DIAMOND_COORDS]
    info = _EDGE_INFO[edge]
    marker = (r0 + info["marker"][0], c0 + info["marker"][1])
    actual_rays = _ray_cells((h, w), marker, info["outward"])
    if len(actual_rays) < MIN_RAY_CELLS:
        return None

    legacy_rays = _legacy_dual_rays((h, w), marker, info["tangent"], info["outward"])
    if not legacy_rays:
        return None

    actual_set = set(actual_rays)
    legacy_set = set(legacy_rays)
    if not (legacy_set - actual_set):
        return None

    input_cells = set(diamond)
    if marker in input_cells:
        return None
    input_cells.add(marker)

    reserve_cells = set(diamond)
    reserve_cells.add(marker)
    reserve_cells.update(actual_set)
    reserve_cells.update(legacy_set)
    if len(reserve_cells) != len(diamond) + 1 + len(actual_set | legacy_set):
        return None

    return {
        "diamond": diamond,
        "edge": {
            "name": edge,
            "marker": marker,
            "rays": actual_rays,
            "legacy_only": sorted(legacy_set - actual_set),
            "weight": 1 + len(actual_rays),
        },
        "all_cells": frozenset(set(diamond) | {marker} | actual_set),
        "reserve_cells": frozenset(reserve_cells),
        "input_cells": frozenset(input_cells),
        "input_halo": frozenset(_expand8(input_cells, (h, w))),
    }


def _search_placements(h: int, w: int, specs: list[str]) -> list[dict[str, object]] | None:
    order = list(range(len(specs)))
    even_toplefts = [(r, c) for r in range(h - 4) for c in range(w - 4) if (r + c) % 2 == 0]

    candidate_cache: dict[str, list[dict[str, object]]] = {}
    for edge in set(specs):
        positions = even_toplefts[:]
        random.shuffle(positions)
        candidate_cache[edge] = [
            p for r0, c0 in positions
            if (p := _candidate_placement(h, w, r0, c0, edge)) is not None
        ]
        if not candidate_cache[edge]:
            return None

    occupied_reserve: set[tuple[int, int]] = set()
    occupied_input_halo: set[tuple[int, int]] = set()
    placed: list[dict[str, object] | None] = [None] * len(specs)

    def backtrack(i: int) -> bool:
        if i == len(order):
            return True
        spec_i = order[i]
        edge = specs[spec_i]
        for placement in candidate_cache[edge]:
            if not occupied_reserve.isdisjoint(placement["reserve_cells"]):
                continue
            if not occupied_input_halo.isdisjoint(placement["input_cells"]):
                continue

            occupied_reserve.update(placement["reserve_cells"])
            occupied_input_halo.update(placement["input_halo"])
            placed[spec_i] = placement
            if backtrack(i + 1):
                return True
            occupied_reserve.difference_update(placement["reserve_cells"])
            occupied_input_halo.difference_update(placement["input_halo"])
            placed[spec_i] = None
        return False

    if not backtrack(0):
        return None
    return [p for p in placed if p is not None]


def _assign_marker_colors(placements: list[dict[str, object]], diamond_colors: list[int]) -> list[int]:
    diamond_set = set(diamond_colors)
    palette = [c for c in COLOR_RANGE if c not in diamond_set]
    counts = Counter({c: 0 for c in palette})
    for c in diamond_colors:
        counts[c] += len(_DIAMOND_COORDS)

    weighted = sorted(
        ((pl["edge"]["weight"], random.random(), i) for i, pl in enumerate(placements)),
        reverse=True,
    )
    marker_colors = [0] * len(placements)
    for _, _, i in weighted:
        chosen = min(palette, key=lambda c: (counts[c], random.random()))
        marker_colors[i] = chosen
        counts[chosen] += placements[i]["edge"]["weight"]
    return marker_colors


def _bg_counts_safe(grid: np.ndarray) -> bool:
    counts = Counter(map(int, grid.ravel()))
    max_non_bg = max((counts[c] for c in COLOR_RANGE), default=0)
    return counts[0] > max_non_bg and counts[1] > max_non_bg


def _sample_specs(diamond_count: int) -> list[str]:
    specs = [random.choice(_EDGE_NAMES) for _ in range(diamond_count)]
    if diamond_count >= 2 and len(set(specs)) == 1:
        idx = random.randrange(diamond_count)
        alternatives = [edge for edge in _EDGE_NAMES if edge != specs[idx]]
        specs[idx] = random.choice(alternatives)
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    diamond_count = random.randint(*DIAMOND_COUNT_RANGE)
    specs = _sample_specs(diamond_count)

    placements = _search_placements(h, w, specs)
    if placements is None:
        return None

    diamond_colors = random.sample(list(COLOR_RANGE), len(placements))
    marker_colors = _assign_marker_colors(placements, diamond_colors)

    out = _checkerboard(h, w)
    ray_coords: list[tuple[int, int]] = []
    for i, placement in enumerate(placements):
        _paint_cells(out, placement["diamond"], diamond_colors[i])
        edge = placement["edge"]
        _paint_cells(out, [edge["marker"]], marker_colors[i])
        _paint_cells(out, edge["rays"], marker_colors[i])
        ray_coords.extend(edge["rays"])

    if not _bg_counts_safe(out):
        return None

    inp = out.copy()
    if ray_coords:
        rs, cs = zip(*ray_coords)
        rs_arr = np.array(rs)
        cs_arr = np.array(cs)
        inp[rs_arr, cs_arr] = (rs_arr + cs_arr + 1) % 2

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GEN_RETRIES} attempts")
