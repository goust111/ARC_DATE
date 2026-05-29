# Variant of db695cfb
# Dimension: D11 — 蓝色像素对的配对规则 (meta)
# Change: 网格中恰好两个蓝色像素，天然配对... → 蓝色像素按距离配对（最近邻）
#
# Design analysis
# source_task: None
# a) 具体化与行为追踪:
#    - 原题里每条有效对角线只需要看到一对蓝点；现在输入里会在同一条 45 度对角线上放 4 或 6 个蓝点。
#    - 这些蓝点不再由“同一条线上的最外侧两点”天然决定，而是要先在所有蓝点之间做完美配对，再分别画出每一对的短对角线。
#    - 具体小例子: 同一条对角线上蓝点位于偏移 0,3,6,9。原逻辑会把 0 和 9 连成一整条长线；新逻辑要把
#      (0,3) 与 (6,9) 这两对短线分别连起来，中间 4..5 这段保持背景。若某个品红点只落在左侧短线上，
#      只有左侧短线会触发垂直对角射线。
# b) 复杂度校准:
#    - 只改变“蓝点如何结对”这一维，其余颜色语义、射线方向、覆盖规则都保持原题风格。
#    - 为了让 D11 肉眼可见，生成器强制至少一条有效对角线上出现 2 对或 3 对蓝点；因此新旧规则会在这条线的
#      中间空隙处产生显著差异，而不是仅仅把原题复制多份。
#    - 蓝点总数限制为 4 或 6，避免匹配自由度过高。
# c) 隐式假设审查:
#    - 原 main() 先按颜色找“最长对角对”，再把每条同向对角线的极值连起来；这隐含了“每条相关对角线上至多一对端点”。
#      该前提在变体中失效，所以必须改成“先求蓝点最优完美匹配，再逐对画线”。
#    - 原 draw_segment() 依赖“路径上遇到的非蓝像素都是触发器”这一点；该点在变体中仍成立，只要生成器保证别的蓝点
#      不会落到另一对的路径上。
#    - 新 solver 会引入一个新假设: 蓝点数量为偶数，且存在由对角候选边组成的完美匹配。生成器必须显式保证这一点。
# d) 放置合法性:
#    - 生成器按 cluster 放置: 一个 cluster 是同一条对角线上的 2 对或 3 对蓝点，相邻两对之间保留 1-3 个背景格。
#    - 不同 cluster 之间禁止任何蓝点再形成额外对角候选边，避免 solver 产生非预期跨 cluster 配对。
#    - 每一对蓝点的内部可放 0-1 个品红触发点；触发射线和所有已有输入/输出占位都不能碰撞。
# e) 视觉可观察性:
#    - 关键可见信号是“同一条对角线上多段短蓝线，中间留空”，这与原题“整条线被一次性连通”形成稳定对比。
#    - 不使用颜色 id 参与排序；配对仅由几何距离和可见坐标顺序决定。
# f) 可视化整洁度:
#    - 触发点数量下降到每对最多 1 个，且射线不得穿过已有非背景像素，减少杂乱叠加。
#    - 非作用品红干扰点只放在所有活动对角线之外，避免看上去像“漏触发”的点。
# g) 颜色变化可推断性:
#    - 颜色语义完全沿用原题: 蓝色负责成段，品红负责射线，背景为第三色。没有新增颜色映射。

from functools import lru_cache
import random

import numpy as np

from arc_utils import get_bg


Coord = tuple[int, int]
Pair = tuple[Coord, Coord]


def _is_diag_pair(p: Coord, q: Coord) -> bool:
    return p != q and abs(p[0] - q[0]) == abs(p[1] - q[1]) and p[0] != q[0]


def _pair_distance(p: Coord, q: Coord) -> int:
    return abs(p[0] - q[0])


def _canonical_pair(p: Coord, q: Coord) -> Pair:
    return (p, q) if p <= q else (q, p)


def _diag_path(p: Coord, q: Coord) -> list[Coord]:
    r0, c0 = p
    r1, c1 = q
    dr = 1 if r1 > r0 else -1
    dc = 1 if c1 > c0 else -1
    steps = abs(r1 - r0)
    return [(r0 + k * dr, c0 + k * dc) for k in range(steps + 1)]


def _best_blue_matching(points: list[Coord]) -> list[Pair]:
    pts = tuple(sorted(points))
    assert pts and len(pts) % 2 == 0, "blue endpoints must form a non-empty even-sized set"

    @lru_cache(maxsize=None)
    def solve(remaining: tuple[Coord, ...]) -> tuple[Pair, ...] | None:
        if not remaining:
            return ()

        first = remaining[0]
        best_key = None
        best_pairs = None
        for i in range(1, len(remaining)):
            other = remaining[i]
            if not _is_diag_pair(first, other):
                continue
            rest = solve(remaining[1:i] + remaining[i + 1 :])
            if rest is None:
                continue
            pairs = tuple(sorted((_canonical_pair(first, other),) + rest))
            lengths = tuple(sorted(_pair_distance(a, b) for a, b in pairs))
            key = (sum(lengths), lengths, pairs)
            if best_key is None or key < best_key:
                best_key = key
                best_pairs = pairs
        return best_pairs

    result = solve(pts)
    assert result is not None, "blue endpoints must admit a diagonal perfect matching"
    return list(result)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair multiple blue diagonal endpoints by nearest matching, then emit perpendicular rays from trigger pixels on the chosen segments.

    Input structure:
    - The background is the dominant color. Blue pixels mark an even-sized set of diagonal endpoints, often with 4 or 6 blue pixels total, and magenta-like trigger pixels may sit on the interiors of the eventual matched segments.

    Rule:
    - Compute a perfect pairing among blue pixels using only diagonal-aligned candidates; choose the pairing with the smallest total diagonal span, and if several tie, use the coordinate-smallest pairing.
    - Draw each matched segment in blue. Whenever a non-blue pixel lies on one of those segments, extend rays from it along the perpendicular diagonal directions until the grid border.

    Key insight:
    - The decisive step is no longer “which diagonal exists” but “which blue endpoints belong together”: nearby blue endpoints form short local segments, leaving visible gaps that the original extreme-to-extreme logic would have filled.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    line_color = 1

    blue_points = [(r, c) for r in range(h) for c in range(w) if int(grid[r, c]) == line_color]
    pairs = _best_blue_matching(blue_points)

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}

    def draw_segment(p: Coord, q: Coord) -> None:
        path = _diag_path(p, q)
        dr = 1 if q[0] > p[0] else -1
        dc = 1 if q[1] > p[1] else -1
        perp_dirs = [(dr, -dc), (-dr, dc)]
        for r, c in path:
            value = int(out[r, c])
            if value == bg:
                out[r, c] = line_color
            elif value != line_color:
                key = (r, c, value)
                triggers.setdefault(key, set()).update(perp_dirs)

    for p, q in pairs:
        draw_segment(p, q)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                if int(out[r, c]) == bg:
                    out[r, c] = tcol
                r += ddr
                c += ddc

    return out


# --------------- generator (c) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_PAIR_LEN_RANGE = (3, 5)
_PAIR_GAP_RANGE = (1, 3)       # background cells left between consecutive paired segments
_CLUSTER_PATTERNS = ((2,), (3,), (2, 1))
_TRIGGERS_PER_PAIR = (0, 1)

# === CFG: distractor counts ===
_OFF_DIAG_TRIGGERS = (0, 2)

# === CFG: retry limits ===
_CLUSTER_PLACE_RETRIES = 120
_DISTRACTOR_RETRIES = 60
_GEN_RETRIES = 400


def _compute_ray_cells(triggers: list[Coord], perp_dirs: list[tuple[int, int]], H: int, W: int) -> set[Coord]:
    cells: set[Coord] = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(inp: np.ndarray, bg: int, segments) -> np.ndarray:
    H, W = inp.shape
    expected = inp.copy()
    for _, _, trigger_positions, path, perp_dirs in segments:
        for r, c in path:
            if expected[r, c] == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    if expected[r, c] == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc
    return expected


def _segment_to_path(start: Coord, end: Coord) -> list[Coord]:
    return _diag_path(start, end)


def _try_place_cluster(
    pair_count: int,
    H: int,
    W: int,
    occupied: set[Coord],
    drawn: set[Coord],
    reserved_gaps: set[Coord],
    existing_blue: list[Coord],
    used_backslash: set[int],
    used_slash: set[int],
):
    for _ in range(_CLUSTER_PLACE_RETRIES):
        slope_type = random.choice(["backslash", "slash"])
        pair_lengths = [random.randint(*_PAIR_LEN_RANGE) for _ in range(pair_count)]
        gap_cells = [random.randint(*_PAIR_GAP_RANGE) for _ in range(pair_count - 1)]

        offsets: list[tuple[int, int]] = []
        cursor = 0
        for idx, seg_len in enumerate(pair_lengths):
            start_offset = cursor
            end_offset = start_offset + seg_len
            offsets.append((start_offset, end_offset))
            if idx < pair_count - 1:
                cursor = end_offset + gap_cells[idx] + 1
        max_offset = offsets[-1][1]

        if slope_type == "backslash":
            if H - 1 - max_offset < 0 or W - 1 - max_offset < 0:
                continue
            r0 = random.randint(0, H - 1 - max_offset)
            c0 = random.randint(0, W - 1 - max_offset)
            dc = 1
            diag_key = r0 - c0
            if diag_key in used_backslash:
                continue
        else:
            if H - 1 - max_offset < 0 or max_offset > W - 1:
                continue
            r0 = random.randint(0, H - 1 - max_offset)
            c0 = random.randint(max_offset, W - 1)
            dc = -1
            diag_key = r0 + c0
            if diag_key in used_slash:
                continue

        def point_at(offset: int) -> Coord:
            return (r0 + offset, c0 + offset * dc)

        endpoints: list[Coord] = []
        cluster_segments = []
        cluster_path_cells: set[Coord] = set()
        cluster_ray_cells: set[Coord] = set()
        cluster_gap_cells: set[Coord] = set()
        perp_dirs = [(1, -dc), (-1, dc)]

        valid = True
        for start_offset, end_offset in offsets:
            ep0 = point_at(start_offset)
            ep1 = point_at(end_offset)
            endpoints.extend([ep0, ep1])

            path = _segment_to_path(ep0, ep1)
            if any(cell in occupied or cell in drawn or cell in reserved_gaps or cell in cluster_path_cells for cell in path):
                valid = False
                break

            interior = path[1:-1]
            n_triggers = random.randint(_TRIGGERS_PER_PAIR[0], min(_TRIGGERS_PER_PAIR[1], len(interior)))
            trigger_positions = random.sample(interior, n_triggers) if n_triggers else []
            ray_cells = _compute_ray_cells(trigger_positions, perp_dirs, H, W)
            if any(
                cell in occupied or cell in drawn or cell in reserved_gaps or cell in cluster_path_cells or cell in cluster_ray_cells
                for cell in ray_cells
            ):
                valid = False
                break

            cluster_segments.append((ep0, ep1, trigger_positions, path, perp_dirs))
            cluster_path_cells.update(path)
            cluster_ray_cells.update(ray_cells)

        if not valid:
            continue

        if any(_is_diag_pair(p, q) for p in endpoints for q in existing_blue):
            continue

        for (_, end_a), (start_b, _) in zip(offsets, offsets[1:]):
            cluster_gap_cells.update(point_at(offset) for offset in range(end_a + 1, start_b))

        if any(cell in occupied or cell in drawn or cell in reserved_gaps for cell in cluster_ray_cells):
            continue

        if slope_type == "backslash":
            used_backslash.add(diag_key)
        else:
            used_slash.add(diag_key)
        return cluster_segments, endpoints, cluster_path_cells, cluster_ray_cells, cluster_gap_cells

    return None


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    occupied: set[Coord] = set()
    drawn: set[Coord] = set()
    reserved_gaps: set[Coord] = set()
    segments = []
    blue_points: list[Coord] = []
    used_backslash: set[int] = set()
    used_slash: set[int] = set()

    cluster_pattern = random.choice(_CLUSTER_PATTERNS)
    for pair_count in cluster_pattern:
        placed = _try_place_cluster(
            pair_count,
            H,
            W,
            occupied,
            drawn,
            reserved_gaps,
            blue_points,
            used_backslash,
            used_slash,
        )
        if placed is None:
            return None
        cluster_segments, endpoints, cluster_path_cells, cluster_ray_cells, cluster_gap_cells = placed
        segments.extend(cluster_segments)
        occupied.update(endpoints)
        for _, _, trigger_positions, path, _ in cluster_segments:
            occupied.update(trigger_positions)
            drawn.update(set(path[1:-1]) - set(trigger_positions))
        drawn.update(cluster_ray_cells)
        reserved_gaps.update(cluster_gap_cells)
        blue_points.extend(endpoints)

    if len(blue_points) < 4 or len(blue_points) % 2 != 0:
        return None

    inp = np.full((H, W), bg, dtype=int)
    for ep0, ep1, trigger_positions, _, _ in segments:
        inp[ep0] = _LINE_COLOR
        inp[ep1] = _LINE_COLOR
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR

    active_backslash = {r - c for r, c in blue_points}
    active_slash = {r + c for r, c in blue_points}
    for _ in range(random.randint(*_OFF_DIAG_TRIGGERS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(0, H - 1)
            c = random.randint(0, W - 1)
            if (r, c) in occupied or (r, c) in drawn or (r, c) in reserved_gaps:
                continue
            if (r - c) in active_backslash or (r + c) in active_slash:
                continue
            inp[r, c] = _TRIGGER_COLOR
            occupied.add((r, c))
            break

    try:
        result = main(inp)
    except Exception:
        return None

    expected = _build_expected(inp, bg, segments)
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
