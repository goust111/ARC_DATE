# Combo variant of 88e364bc (index 24)
# combo_spec: D4/3+D15/2
# Name: 滑到中心+链式触发

# Design analysis
# ===============
# combo_spec:
#   a_concretization_and_trace:
#     - 视觉骨架仍是原题：角落图例给出方向，图例填充色与某批封闭围墙颜色相同；黄色像素位于围墙内部。
#     - 融合后新增一种“内部标记像素”：它们颜色与本区域围墙同色，但不是边界的一部分，而是孤立地待在区域内部。
#     - 第一步不再把黄色滑到远端墙边，而是沿图例方向前进，停在该离散轨迹上“离区域中心最近”的位置。
#       中心按该区域内部可活动空间的几何平均估计；若两个连续格与中心同样近，则继续向前取更靠前的那个。
#     - 第二步才检查链式触发：若某个休眠标记与第一步停住的黄色 4 邻接，它被激活、原地清空，并沿同方向按同样的
#       “向中心停靠”规则滑动；新停住的激活标记又可能邻接到下一个休眠标记，于是波次继续传播。
#     - 小例子：方向向右时，原题会把黄点一路推到最右壁。本融合版先让它停在区域中部最靠中心的格；该格右侧若恰有
#       一个墙色单像素标记，则标记被触发，再向右滑到另一个中心附近格并变黄；若它滑完后又贴住第二个标记，则第二个
#       标记继续滑。差异是“先中心聚集，再局部扩散”，而不是“直接贴墙结束”。
#   b_complexity_calibration:
#     - 只保留一种新增对象，即墙同色内部标记，不引入额外颜色语义，避免与原图例颜色体系冲突。
#     - 图例方向仍只取 8 个离散方向；区域数仍保持 1-4 个，链式反应通常集中在一个区域，其他区域最多附带少量普通黄点。
#     - generator 会拒绝两类退化样本：一类是启用链式后输出仍与“只有中心停点”相同，另一类是启用中心停点后输出仍与
#       “边界滑动+链式触发”相同。这样两个维度都必须真正参与最终输出。
#   c_implicit_assumptions_review:
#     - `_legend_directions` 只读图例方向，不依赖终点规则或链式传播，可直接复用。
#     - 原题 `passable=(bg|yellow)` 默认内部只有黄色可动；融合后不成立。solver 先用这个 mask 找到底层封闭区域，
#       再单独侦测“墙色但四邻都在区域内部”的标记，并把它们并入该区域的可活动空间。
#     - 原 `_slide_points` 默认一直滑到边界；这里必须改成可切换的终点策略，且第一波黄色与后续激活标记都共享同一
#       中心停点逻辑，否则两个阶段的行为会不一致。
#     - 原题对黄色数量守恒的隐含前提也失效，因为被激活的标记最终会转成黄色；generator 因此改为检查黄色数增加。
#   d_placement_legality:
#     - 标记只能放在 core cell（四邻都属于区域内部）的单个格子里，保证它既不会与边墙粘连，也不会把区域切断。
#     - 黄色输入位置通过反向搜索得到：只有那些正向执行新规则后能精确回到目标中心停点的起点才会被接受。
#     - 对角移动仍保留防穿角约束：若方向是对角，则两个正交中间格都必须在区域内。
#   e_visual_observability:
#     - 人类能在样例里看到两条线索同时存在：黄色不再贴远端墙，而是停在区域中部；同时少量墙色内点会消失并在附近
#       变成新的黄色，表现出“先停住，再触发”的时序。
#     - 这些差异完全由可见的空间关系定义，不依赖颜色编号大小、数组扫描顺序等人类不可见因素。
#   f_visual_cleanliness:
#     - 标记始终是孤立单像素，数量通常 1-2 个；链式行为优先限制在一个区域，避免全图多个区域同时多波扩散而显得杂乱。
#     - 其余区域只少量补充普通黄色，保留原题的留白感与围墙可读性。
#   g_color_inference:
#     - 颜色语义清晰且复用原题：4 是黄色 mover，2 是图例红色指示像素，1 是图例中心像素；墙色既决定图例匹配，也决定
#       哪些内部标记属于该区域。唯一颜色变化是“墙色标记 -> 黄色”，且触发条件由相邻关系给出。
#   h_solver_assumptions_after_change:
#     - solver 新增假设：内部标记是孤立单像素且四邻都在区域内部；generator 用 core-cell 约束保证这一点。
#     - solver 假设至少有一个区域里同时存在“中心停点差异”和“标记激活差异”；generator 用多基线 oracle 强制过滤。
#     - solver 假设区域中心可由离散内部空间稳定估计；generator 通过足够大的闭合区域与少量标记，避免中心定义被噪声主导。

import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 3)
MAX_TOTAL_REGIONS = 4
REGION_SIZE_BOUNDS = {
    1: (18, 85),
    2: (16, 55),
    3: (14, 35),
    4: (12, 25),
}
BLOB_GROW_RETRIES = 25

MAX_REVERSE_STEPS = 6
CHAIN_SEARCH_ATTEMPTS = 180
CHAIN_MARKER_SAMPLE = 28
EXTRA_YELLOW_PROB = 0.45
OUTPUT_CANDIDATE_POOL = 8
START_CANDIDATE_POOL = 8
GEN_RETRIES = 450

MOVER = 4
EPS = 1e-9


def _projection(p: tuple[int, int], dr: int, dc: int) -> int:
    return int(dr * p[0] + dc * p[1])


def _adjacent4(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _region_center(region_set: set[tuple[int, int]]) -> tuple[float, float]:
    pts = np.array(sorted(region_set), dtype=float)
    ctr = pts.mean(axis=0)
    return float(ctr[0]), float(ctr[1])


def _center_dist_sq(pos: tuple[int, int], center: tuple[float, float]) -> float:
    rr = float(pos[0]) - center[0]
    cc = float(pos[1]) - center[1]
    return rr * rr + cc * cc


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode their center→indicator directions."""
    h, w = g.shape
    non_bg = g != bg
    best: dict[int, tuple[int, tuple[int, int]]] = {}
    for pts in components_mask(non_bg, connectivity=4):
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        if area != len(pts):
            continue

        counts: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            counts[v] = counts.get(v, 0) + 1
        if len(counts) != 3:
            continue

        fill_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != fill_col]
        others.sort()
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            tie_info = []
            for col in (others[0][1], others[1][1]):
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                tie_info.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            tie_info.sort()
            center_col = tie_info[0][1]
            indicator_col = tie_info[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = [(r, c) for r, c in pts if int(g[r, c]) == center_col]
        indic = [(r, c) for r, c in pts if int(g[r, c]) == indicator_col]
        if not center or not indic:
            continue

        cc = np.mean(np.array(center, dtype=float), axis=0)
        ci = np.mean(np.array(indic, dtype=float), axis=0)
        vr, vc = (ci - cc).tolist()
        dr = 0 if abs(vr) < 1e-6 else (1 if vr > 0 else -1)
        dc = 0 if abs(vc) < 1e-6 else (1 if vc > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _center_slide_single(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    center: tuple[float, float],
) -> tuple[int, int]:
    r, c = start
    cur_d = _center_dist_sq((r, c), center)
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        next_d = _center_dist_sq((nr, nc), center)
        if next_d > cur_d + EPS:
            break
        r, c = nr, nc
        cur_d = next_d
    return (r, c)


def _boundary_slide_single(
    region_set: set[tuple[int, int]],
    start: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> tuple[int, int]:
    r, c = start
    while True:
        if not _slide_step_valid(region_set, r, c, dr, dc):
            break
        nr, nc = r + dr, c + dc
        if (nr, nc) in blocked:
            break
        r, c = nr, nc
    return (r, c)


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    *,
    blockers: set[tuple[int, int]] | None = None,
    center_mode: bool,
    center: tuple[float, float] | None = None,
) -> list[tuple[int, int]]:
    if not points:
        return []
    occupied = set(blockers or ())
    occupied -= set(points)
    points_sorted = sorted(points, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    if center_mode:
        assert center is not None
    out_pts = []
    for pt in points_sorted:
        stop = (
            _center_slide_single(region_set, pt, dr, dc, occupied, center)
            if center_mode
            else _boundary_slide_single(region_set, pt, dr, dc, occupied)
        )
        occupied.add(stop)
        out_pts.append(stop)
    return out_pts


def _region_border_color(
    g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int, mover: int
) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        for nr, nc in neighbors4(rr, cc):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in pset:
                continue
            v = int(g[nr, nc])
            if v == bg or v == mover:
                continue
            border_counts[v] = border_counts.get(v, 0) + 1
    if not border_counts:
        return None
    return max(border_counts, key=border_counts.get)


def _detect_region_markers(
    g: np.ndarray, pset: set[tuple[int, int]], border_col: int
) -> list[tuple[int, int]]:
    """Find isolated wall-colored internal seeds whose 4-neighbors all lie in the region."""
    h, w = g.shape
    markers: set[tuple[int, int]] = set()
    for r, c in pset:
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in pset or (nr, nc) in markers:
                continue
            if int(g[nr, nc]) != border_col:
                continue
            nbrs = list(neighbors4(nr, nc))
            if all(0 <= ar < h and 0 <= ac < w and (ar, ac) in pset for ar, ac in nbrs):
                markers.add((nr, nc))
    return sorted(markers)


def _simulate_region(
    region_set: set[tuple[int, int]],
    movers: list[tuple[int, int]],
    markers: set[tuple[int, int]],
    dr: int,
    dc: int,
    *,
    center_mode: bool,
    activate_markers: bool,
) -> dict:
    center = _region_center(region_set)
    dormant = set(markers)
    first_order = sorted(movers, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
    first_stops = _slide_points(
        region_set,
        first_order,
        dr,
        dc,
        blockers=dormant,
        center_mode=center_mode,
        center=center,
    )
    settled = set(first_stops)
    activated: set[tuple[int, int]] = set()
    activated_pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    wave_depth = 0

    if activate_markers:
        while True:
            newly = [
                pos
                for pos in sorted(dormant, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)
                if any(_adjacent4(pos, s) for s in settled)
            ]
            if not newly:
                break
            wave_depth += 1
            newly_set = set(newly)
            dormant -= newly_set
            activated |= newly_set
            moved = _slide_points(
                region_set,
                newly,
                dr,
                dc,
                blockers=settled | dormant,
                center_mode=center_mode,
                center=center,
            )
            activated_pairs.extend(zip(newly, moved))
            settled |= set(moved)

    return {
        "settled": settled,
        "center": center,
        "first_pairs": list(zip(first_order, first_stops)),
        "activated": activated,
        "activated_pairs": activated_pairs,
        "wave_depth": wave_depth,
    }


def _solve_grid(
    grid: np.ndarray,
    *,
    center_mode: bool,
    activate_markers: bool,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == MOVER)
    out = g.copy()
    stats = {
        "activated_markers": 0,
        "activated_regions": 0,
        "max_wave_depth": 0,
    }

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == MOVER]
        if not movers:
            continue

        border_col = _region_border_color(g, pts, pset, bg, MOVER)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        markers = set(_detect_region_markers(g, pset, border_col))
        region_set = pset | markers

        sim = _simulate_region(
            region_set,
            movers,
            markers,
            dr,
            dc,
            center_mode=center_mode,
            activate_markers=activate_markers,
        )

        for r, c in movers:
            out[r, c] = bg
        for r, c in markers:
            out[r, c] = bg
        for r, c in sim["settled"]:
            out[r, c] = MOVER

        if sim["activated"]:
            stats["activated_markers"] += len(sim["activated"])
            stats["activated_regions"] += 1
            stats["max_wave_depth"] = max(stats["max_wave_depth"], sim["wave_depth"])

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move yellow pixels inside closed wall regions using directions from border legends, then let centered stops trigger chained marker motion.

    Input structure:
    - The background is the most common color. Colored wall loops enclose regions that contain yellow pixels and may also contain isolated single-pixel seeds in the same color as the surrounding wall.
    - Border legends are solid rectangles on the outer edge. Their fill color matches one wall color, and the offset from the legend's central motif to its red indicator motif encodes one of eight movement directions.

    Rule:
    - For each enclosed region, match its surrounding wall color to the legend of the same fill color to obtain `(dr, dc)`.
    - First slide the yellow pixels in that direction, processing front-to-back, but stop each one at the reachable point on its trajectory that is closest to the region center; dormant seeds block this first pass.
    - Then repeatedly activate any 4-neighbor-adjacent wall-colored seeds. Each activated seed disappears from its original cell, slides by the same center-seeking rule, turns yellow, and may trigger another wave.

    Key insight:
    - The legends still tell you only the direction. The new behavior comes from combining a center-based stopping criterion with a second stage where adjacency to the centered yellow cluster releases wall-colored seeds into the same motion field.
    """
    return _solve_grid(grid, center_mode=True, activate_markers=True, return_stats=False)


def _center_only_main(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid, center_mode=True, activate_markers=False, return_stats=False)


def _boundary_chain_main(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid, center_mode=False, activate_markers=True, return_stats=False)


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid, center_mode=False, activate_markers=False, return_stats=False)


def _behavior_metrics(grid: np.ndarray) -> dict[str, int | bool]:
    combo_out, combo_stats = _solve_grid(grid, center_mode=True, activate_markers=True, return_stats=True)
    center_only = _center_only_main(grid)
    boundary_chain = _boundary_chain_main(grid)
    legacy = _legacy_main(grid)
    return {
        "activated_markers": int(combo_stats["activated_markers"]),
        "activated_regions": int(combo_stats["activated_regions"]),
        "max_wave_depth": int(combo_stats["max_wave_depth"]),
        "yellow_gain": int((combo_out == MOVER).sum() - (np.asarray(grid) == MOVER).sum()),
        "diff_vs_center_only": not np.array_equal(combo_out, center_only),
        "diff_vs_boundary_chain": not np.array_equal(combo_out, boundary_chain),
        "diff_vs_legacy": not np.array_equal(combo_out, legacy),
    }


def _legend_5x5(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((5, 5), fill, dtype=int)
    g[1, 1] = 1
    g[1, 2] = 1
    g[1, 3] = 2
    g[3, 1] = 1
    g[3, 2] = 1
    g[3, 3] = 2
    rot = {(0, 1): 0, (-1, 0): 1, (0, -1): 2, (1, 0): 3}
    return np.rot90(g, rot[(dr, dc)])


def _legend_6x7(fill: int, dr: int, dc: int) -> np.ndarray:
    g = np.full((6, 7), fill, dtype=int)
    g[1, 3] = 1
    g[2, 1] = 1
    g[2, 4] = 1
    g[3, 2] = 1
    g[3, 5] = 2
    g[4, 3] = 2
    if (dr, dc) == (1, 1):
        return g
    if (dr, dc) == (-1, 1):
        return np.flipud(g)
    if (dr, dc) == (1, -1):
        return np.fliplr(g)
    if (dr, dc) == (-1, -1):
        return np.rot90(g, 2)
    raise ValueError((dr, dc))


def _corner_origin(corner: str, h: int, w: int, ph: int, pw: int) -> tuple[int, int]:
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - pw
    if corner == "BL":
        return h - ph, 0
    if corner == "BR":
        return h - ph, w - pw
    raise ValueError(corner)


def _grow_blob(
    canvas_mask: np.ndarray, seed: tuple[int, int], target_size: int, rng: random.Random
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier: list[tuple[int, int]] = []

    def push_neighbors(r: int, c: int) -> None:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and canvas_mask[nr, nc] and (nr, nc) not in blob:
                frontier.append((nr, nc))

    push_neighbors(sr, sc)
    while len(blob) < target_size and frontier:
        idx = rng.randrange(len(frontier))
        r, c = frontier.pop(idx)
        if (r, c) in blob or not canvas_mask[r, c]:
            continue
        blob.add((r, c))
        push_neighbors(r, c)
    if len(blob) < target_size:
        return None
    return blob


def _blob_wall_cells(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    wall = set()
    for r, c in blob:
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in blob:
                wall.add((nr, nc))
    return wall


def _fill_holes(blob: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    walls = _blob_wall_cells(blob, h, w)
    barrier = np.zeros((h, w), dtype=bool)
    for r, c in blob:
        barrier[r, c] = True
    for r, c in walls:
        barrier[r, c] = True

    reachable = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in (0, w - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not barrier[r, c]:
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        if reachable[r, c] or barrier[r, c]:
            continue
        reachable[r, c] = True
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reachable[nr, nc] and not barrier[nr, nc]:
                stack.append((nr, nc))

    trapped = (~reachable) & (~barrier)
    if not trapped.any():
        return blob
    filled = set(blob)
    for r, c in np.argwhere(trapped):
        filled.add((int(r), int(c)))
    return filled


def _place_legends(
    rng: random.Random, grid: np.ndarray
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wc in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wc, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wc, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wc, (dr, dc)))
    return legend_occupied, legends


def _try_grow_valid_blob(
    rng: random.Random,
    canvas: np.ndarray,
    occupied: np.ndarray,
    target_size: int,
    h: int,
    w: int,
) -> set[tuple[int, int]] | None:
    for _ in range(BLOB_GROW_RETRIES):
        free = np.argwhere(canvas)
        if len(free) == 0:
            break
        sr, sc = free[rng.randrange(len(free))]
        cand = _grow_blob(canvas, (int(sr), int(sc)), target_size, rng)
        if cand is None:
            continue
        while True:
            filled = _fill_holes(cand, h, w)
            if filled == cand:
                break
            cand = filled
        if any(not canvas[r, c] for r, c in cand):
            continue
        walls = _blob_wall_cells(cand, h, w)
        if any(occupied[wr, wc] or wr in (0, h - 1) or wc in (0, w - 1) for wr, wc in walls):
            continue
        return cand
    return None


def _grow_regions(
    rng: random.Random,
    legends: list[tuple[int, tuple[int, int]]],
    legend_occupied: np.ndarray,
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for col in all_colors:
        region_colors.extend([col] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            continue

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            continue

        walls = _blob_wall_cells(blob, h, w)
        regions.append({"color": rc, "dir": color_dir[rc], "interior": blob, "walls": walls})

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

    if not regions:
        return None
    if {reg["color"] for reg in regions} != set(all_colors):
        return None
    return regions


def _validate_topology(grid: np.ndarray, regions: list[dict]) -> bool:
    h, w = grid.shape
    labeled, n_comp = label_mask(grid == 0, connectivity=4)
    bounded = []
    for comp_id in range(1, n_comp + 1):
        coords = np.argwhere(labeled == comp_id)
        if any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords):
            continue
        bounded.append({(int(r), int(c)) for r, c in coords})
    region_sets = [reg["interior"] for reg in regions]
    if len(bounded) != len(region_sets):
        return False
    return all(comp in region_sets for comp in bounded)


def _core_cells(blob: set[tuple[int, int]]) -> list[tuple[int, int]]:
    return [p for p in blob if all((nr, nc) in blob for nr, nc in neighbors4(p[0], p[1]))]


def _available_center_outputs(
    region_set: set[tuple[int, int]],
    allowed_starts: set[tuple[int, int]],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
) -> list[dict]:
    center = _region_center(region_set)
    output_to_starts: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for start in allowed_starts:
        if start in blocked:
            continue
        out_pos = _center_slide_single(region_set, start, dr, dc, blocked, center)
        if out_pos == start:
            continue
        output_to_starts.setdefault(out_pos, []).append(start)

    items = []
    for out_pos, starts in output_to_starts.items():
        starts_sorted = sorted(
            set(starts),
            key=lambda p: (_projection(p, dr, dc), p[0], p[1]),
            reverse=True,
        )
        items.append(
            {
                "out": out_pos,
                "starts": starts_sorted,
                "dist2": _center_dist_sq(out_pos, center),
                "proj": _projection(out_pos, dr, dc),
                "distinctive": _boundary_slide_single(region_set, out_pos, dr, dc, blocked) != out_pos,
            }
        )
    items.sort(key=lambda d: (d["dist2"], -d["proj"], d["out"][0], d["out"][1]))
    return items


def _reverse_start_candidates(
    region_set: set[tuple[int, int]],
    allowed_starts: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    existing_inputs: set[tuple[int, int]],
    center: tuple[float, float],
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    candidates = []
    hard_blocked = set(blocked) | set(existing_inputs)
    for steps in range(1, MAX_REVERSE_STEPS + 1):
        r, c = out_pos
        ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in hard_blocked:
                ok = False
                break
            if not _slide_step_valid(region_set, pr, pc, dr, dc):
                ok = False
                break
            r, c = pr, pc
        if not ok:
            continue
        start = (r, c)
        if start not in allowed_starts:
            continue
        if _center_slide_single(region_set, start, dr, dc, blocked, center) != out_pos:
            continue
        key = (_projection(start, dr, dc), start[0], start[1])
        candidates.append((steps, key, start))
    candidates.sort(key=lambda item: (item[1][0], item[1][1], item[1][2], item[0]), reverse=True)
    return candidates


def _plan_simple_yellow(rng: random.Random, reg: dict) -> list[tuple[int, int]] | None:
    blob = set(reg["interior"])
    dr, dc = reg["dir"]
    center = _region_center(blob)
    outputs = _available_center_outputs(blob, blob, dr, dc, set())
    if not outputs:
        return None
    pool = [item for item in outputs if item["distinctive"]] or outputs
    pool = pool[: min(len(pool), OUTPUT_CANDIDATE_POOL)]
    choice = pool[rng.randrange(len(pool))]
    starts = _reverse_start_candidates(blob, blob, choice["out"], dr, dc, set(), set(), center)
    if not starts:
        return None
    starts = starts[: min(len(starts), START_CANDIDATE_POOL)]
    return [starts[rng.randrange(len(starts))][2]]


def _search_chain_layout(reg: dict, rng: random.Random) -> dict | None:
    blob = set(reg["interior"])
    dr, dc = reg["dir"]
    core = _core_cells(blob)
    if not core:
        return None

    core_pool = rng.sample(core, min(len(core), CHAIN_MARKER_SAMPLE))
    starts = sorted(blob, key=lambda p: (_projection(p, dr, dc), p[0], p[1]), reverse=True)

    for want_two in (True, False):
        for _ in range(CHAIN_SEARCH_ATTEMPTS):
            if want_two:
                if len(core_pool) < 2:
                    break
                m1, m2 = rng.sample(core_pool, 2)
                if _adjacent4(m1, m2):
                    continue
                markers = {m1, m2}
            else:
                markers = {rng.choice(core_pool)}

            region_set = blob | markers
            chosen_start = None
            best_sim = None

            for start in starts:
                if start in markers:
                    continue
                sim_center = _simulate_region(
                    region_set,
                    [start],
                    markers,
                    dr,
                    dc,
                    center_mode=True,
                    activate_markers=True,
                )
                if len(sim_center["activated"]) < 1:
                    continue
                if want_two and len(sim_center["activated"]) < 2:
                    continue
                if sim_center["first_pairs"][0][0] == sim_center["first_pairs"][0][1]:
                    continue
                if not any(src != dst for src, dst in sim_center["activated_pairs"]):
                    continue

                sim_boundary = _simulate_region(
                    region_set,
                    [start],
                    markers,
                    dr,
                    dc,
                    center_mode=False,
                    activate_markers=True,
                )
                if sim_center["settled"] == sim_boundary["settled"]:
                    continue
                chosen_start = start
                best_sim = sim_center
                break

            if chosen_start is None or best_sim is None:
                continue

            return {
                "markers": sorted(markers),
                "yellow_inputs": [chosen_start],
                "activated_markers": len(best_sim["activated"]),
                "wave_depth": best_sim["wave_depth"],
            }
    return None


def _place_contents(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    order = sorted(range(len(regions)), key=lambda i: len(regions[i]["interior"]), reverse=True)
    chain_idx = None
    chain_layout = None
    for idx in order:
        layout = _search_chain_layout(regions[idx], rng)
        if layout is not None:
            chain_idx = idx
            chain_layout = layout
            break
    if chain_idx is None or chain_layout is None:
        return False

    chain_reg = regions[chain_idx]
    for r, c in chain_layout["markers"]:
        grid[r, c] = chain_reg["color"]
    for r, c in chain_layout["yellow_inputs"]:
        grid[r, c] = MOVER

    others = [i for i in order if i != chain_idx]
    rng.shuffle(others)
    for idx in others:
        if rng.random() >= EXTRA_YELLOW_PROB:
            continue
        planned = _plan_simple_yellow(rng, regions[idx])
        if planned is None:
            continue
        for r, c in planned:
            grid[r, c] = MOVER
    return True


def _oracle_check(input_grid: np.ndarray) -> bool:
    try:
        combo_out, combo_stats = _solve_grid(
            input_grid, center_mode=True, activate_markers=True, return_stats=True
        )
        center_only = _center_only_main(input_grid)
        boundary_chain = _boundary_chain_main(input_grid)
        legacy = _legacy_main(input_grid)
    except Exception:
        return False

    if combo_stats["activated_markers"] < 1:
        return False
    if np.array_equal(combo_out, center_only):
        return False
    if np.array_equal(combo_out, boundary_chain):
        return False
    if np.array_equal(combo_out, legacy):
        return False
    if int((combo_out == MOVER).sum()) <= int((input_grid == MOVER).sum()):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h, w = GRID_H, GRID_W
    grid = np.zeros((h, w), dtype=int)

    legend_occupied, legends = _place_legends(rng, grid)
    regions = _grow_regions(rng, legends, legend_occupied)
    if regions is None:
        return None

    for reg in regions:
        for r, c in reg["walls"]:
            grid[r, c] = reg["color"]

    if not _validate_topology(grid, regions):
        return None

    input_grid = grid.copy()
    if not _place_contents(rng, regions, input_grid):
        return None
    if not _oracle_check(input_grid):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            g = _gen_attempt(rng)
        except Exception:
            continue
        if g is not None:
            return g
    raise RuntimeError("generate_input: exceeded retries")
