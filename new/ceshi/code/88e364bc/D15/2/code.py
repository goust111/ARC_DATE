# Variant of 88e364bc
# Dimension: D15 — 多步处理与操作顺序 (multi_step)
# Change: 单步处理——所有区域的黄色像素独立滑动一次即完成... → 链式触发：先滑动后合并

import random
import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 保留原题的边框图例、彩色围墙、黄色像素和封闭区域。
#    - 新增对象是“内部标记像素”：它们位于区域内部，颜色与该区域围墙相同，但不是边界的一部分，
#      生成时保证是孤立单像素，四邻都在区域内部，因此视觉上像“墙色种子”。
#    - 原题流程：黄色像素按图例方向滑到远端，过程结束。
#    - 变体流程：第一步仍是黄色先滑。第二步开始按波次检查：若某个休眠标记与已停住的黄色
#      4 邻接，则该标记被激活，原地清空，并像黄色一样沿同方向滑动、并入黄色堆叠。新停住的
#      激活标记还可以继续触发下一批休眠标记，直到没有新触发。
#    - 例子：向右滑时，黄色先到最右端；其下方有一个同墙色标记，因与黄色相邻而触发，也向右滑。
#      它滑完后如果又贴住另一个休眠标记，则第二个标记再滑，形成链式反应。
# b) 复杂度校准
#    - 只引入一种新增对象：与墙同色的内部单像素标记，避免颜色体系膨胀。
#    - 标记数量控制在 1-2 个，至少 1 个真正会被激活；大多数样例只在 1 个区域内发生链式反应，
#      保证人眼能聚焦到“先黄后扩散”的时序差异。
# c) 隐式假设审查
#    - 原 solver 的 `passable=(bg|yellow)` 假设所有可动对象都是黄色。变体下不成立；不过为了不把
#      休眠标记误当墙，solver 先用原 passable 找到底层区域，再额外检测内部标记并把它们加入区域空间。
#    - 原 `_slide_points` 只有同波次对象彼此堆叠。变体下还必须让休眠标记和已停住的激活对象充当 blocker，
#      否则黄色会穿过标记，或后续波次会覆盖前一波结果。
#    - 原逻辑默认黄色数量守恒。变体下激活标记会转成黄色，因此输出黄色数会增加；generator 也据此校验。
# d) 放置合法性
#    - 标记只能放在“四邻全在区域内部”的 core cell，避免与边界连成一体，也避免把区域切断。
#    - 黄色输入位置用反向滑动构造，显式检查路径、阻挡关系和最终停点，保证 solver 一定可复现。
# e) 视觉可观察性
#    - 输入里能看到少量墙同色内点，输出里这些点消失并在远端转为黄色，且黄色总数上升。
#      这比纯粹改变处理顺序更容易肉眼归纳。
# f) 可视化整洁度
#    - 标记是孤立单像素，不会与边墙重叠；链反应通常只发生在一个区域，避免全图同时多层变化。
# g) 颜色变化可推断性
#    - 只有“墙同色内部点 -> 黄色”这一种颜色变化，且触发条件由空间邻接给出，线索明确。
# h) main() 的新假设
#    - 标记是单像素且四邻均在区域内部；generator 通过 core-cell 约束保证。
#    - 邻接定义使用 4 邻，不使用对角；generator 的触发布局也按 4 邻构造。
#    - 每个会发生链式反应的区域至少有一个普通黄色起点；generator 强制满足。


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

MAX_SLIDE_STEPS = 5
CHAIN_SEARCH_ATTEMPTS = 120
EXTRA_YELLOW_PROB = 0.55
GEN_RETRIES = 300


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Detect border legends and decode center->indicator directions."""
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

        bg_col = max(counts, key=counts.get)
        others = [(cnt, col) for col, cnt in counts.items() if col != bg_col]
        others.sort()
        if len(others) < 2:
            continue
        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            cols = [others[0][1], others[1][1]]
            cents = []
            for col in cols:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                cents.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            cents.sort()
            center_col = cents[0][1]
            indicator_col = cents[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[-1][1]

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

        prev = best.get(bg_col)
        if prev is None or len(pts) > prev[0]:
            best[bg_col] = (len(pts), (dr, dc))

    return {k: v for k, (_, v) in best.items()}


def _slide_step_valid(region_set: set[tuple[int, int]], r: int, c: int, dr: int, dc: int) -> bool:
    nr, nc = r + dr, c + dc
    if (nr, nc) not in region_set:
        return False
    if dr != 0 and dc != 0:
        if (r + dr, c) not in region_set or (r, c + dc) not in region_set:
            return False
    return True


def _slide_points(
    region_set: set[tuple[int, int]],
    points: list[tuple[int, int]],
    dr: int,
    dc: int,
    blockers: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Slide a batch from far to near, respecting already-occupied blocker cells."""
    if not points:
        return []
    moving = set(points)
    occupied = set(blockers or ())
    occupied -= moving
    points_sorted = sorted(points, key=lambda p: (dr * p[0] + dc * p[1], p[0], p[1]), reverse=True)
    out_pts = []
    for r0, c0 in points_sorted:
        r, c = r0, c0
        while True:
            if not _slide_step_valid(region_set, r, c, dr, dc):
                break
            nr, nc = r + dr, c + dc
            if (nr, nc) in occupied:
                break
            r, c = nr, nc
        occupied.add((r, c))
        out_pts.append((r, c))
    return out_pts


def _region_border_color(
    g: np.ndarray, pts: np.ndarray, pset: set[tuple[int, int]], bg: int, mover: int
) -> int | None:
    h, w = g.shape
    border_counts: dict[int, int] = {}
    for r, c in pts:
        for nr, nc in neighbors4(int(r), int(c)):
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
    """Internal wall-colored seeds: isolated single pixels whose 4-neighbors stay inside the region."""
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


def _solve_grid(
    grid: np.ndarray, *, activate_markers: bool = True, return_stats: bool = False
) -> np.ndarray | tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == mover)
    out = g.copy()
    stats = {"activated_markers": 0, "activated_regions": 0, "max_wave_depth": 0}

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        border_col = _region_border_color(g, pts, pset, bg, mover)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]
        markers = _detect_region_markers(g, pset, border_col)
        dormant = set(markers)
        region_set = pset | dormant

        settled = set(_slide_points(region_set, movers, dr, dc, blockers=dormant))
        activated: set[tuple[int, int]] = set()
        wave_depth = 0

        if activate_markers:
            while True:
                newly = [
                    pos
                    for pos in dormant
                    if any(nb in settled for nb in neighbors4(pos[0], pos[1]))
                ]
                if not newly:
                    break
                wave_depth += 1
                newly_set = set(newly)
                dormant -= newly_set
                activated |= newly_set
                moved = _slide_points(region_set, newly, dr, dc, blockers=settled | dormant)
                settled |= set(moved)

        for r, c in movers:
            out[r, c] = bg
        for r, c in activated:
            out[r, c] = bg
        for r, c in settled:
            out[r, c] = mover

        if activated:
            stats["activated_markers"] += len(activated)
            stats["activated_regions"] += 1
            stats["max_wave_depth"] = max(stats["max_wave_depth"], wave_depth)

    if return_stats:
        return out, stats
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Yellow pixels slide as in the original task.
    2. Wall-colored internal marker pixels adjacent to the settled yellow mass activate wave by wave.
    3. Activated markers also slide in the same direction and turn yellow, possibly triggering more markers.
    """
    return _solve_grid(grid, activate_markers=True, return_stats=False)


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


def _forward_slide(
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
    rng: random.Random, canvas: np.ndarray, occupied: np.ndarray, target_size: int, h: int, w: int
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
    rng: random.Random, legends: list[tuple[int, tuple[int, int]]], legend_occupied: np.ndarray
) -> list[dict] | None:
    h, w = legend_occupied.shape
    color_dir = {wc: d for wc, d in legends}
    all_colors = [wc for wc, _ in legends]

    region_colors = []
    for c in all_colors:
        region_colors.extend([c] * rng.randint(*REGIONS_PER_COLOR))
    if len(region_colors) > MAX_TOTAL_REGIONS:
        region_colors = rng.sample(region_colors, MAX_TOTAL_REGIONS)
    rng.shuffle(region_colors)

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for rc in region_colors:
        dr, dc = color_dir[rc]
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
        regions.append({"color": rc, "dir": (dr, dc), "interior": blob, "walls": walls})

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


def _proj(p: tuple[int, int], dr: int, dc: int) -> int:
    return dr * p[0] + dc * p[1]


def _adjacent4(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def _core_cells(blob: set[tuple[int, int]]) -> list[tuple[int, int]]:
    core = []
    for r, c in blob:
        if all((nr, nc) in blob for nr, nc in neighbors4(r, c)):
            core.append((r, c))
    return core


def _find_reverse_slide(
    region_set: set[tuple[int, int]],
    out_pos: tuple[int, int],
    dr: int,
    dc: int,
    blocked: set[tuple[int, int]],
    existing_inputs: list[tuple[int, int]],
    rng: random.Random,
) -> tuple[int, int] | None:
    target_steps = rng.randint(1, MAX_SLIDE_STEPS)
    extra_blocked = set(blocked) | set(existing_inputs)
    for steps in range(target_steps, 0, -1):
        r, c = out_pos
        path_ok = True
        for _ in range(steps):
            pr, pc = r - dr, c - dc
            if (pr, pc) not in region_set or (pr, pc) in extra_blocked:
                path_ok = False
                break
            if dr != 0 and dc != 0:
                if (pr + dr, pc) not in region_set or (pr, pc + dc) not in region_set:
                    path_ok = False
                    break
            r, c = pr, pc
        if not path_ok:
            continue
        if _forward_slide(region_set, (r, c), dr, dc, extra_blocked) == out_pos:
            return (r, c)
    return None


def _search_chain_layout(reg: dict, rng: random.Random) -> dict | None:
    blob = set(reg["interior"])
    dr, dc = reg["dir"]
    core = _core_cells(blob)
    if not core:
        return None

    top_cells = sorted(blob, key=lambda p: (_proj(p, dr, dc), p[0], p[1]), reverse=True)[:24]

    for want_two_markers in (True, False):
        for _ in range(CHAIN_SEARCH_ATTEMPTS):
            yf = rng.choice(top_cells)
            adj_core = [m for m in core if _adjacent4(m, yf)]
            if not adj_core:
                continue
            m1 = rng.choice(adj_core)

            if want_two_markers:
                m2_pool = [m for m in core if m != m1 and not _adjacent4(m, yf)]
                if not m2_pool:
                    continue
                m2 = rng.choice(m2_pool)
                region_set = blob | {m1, m2}
                f1 = _forward_slide(region_set, m1, dr, dc, {yf, m2})
                if f1 == m1 or not _adjacent4(f1, m2):
                    continue
                f2 = _forward_slide(region_set, m2, dr, dc, {yf, f1})
                if f2 == m2:
                    continue
                yi = _find_reverse_slide(region_set, yf, dr, dc, {m1, m2}, [], rng)
                if yi is None or yi in {m1, m2}:
                    continue
                return {"markers": [m1, m2], "yellow_inputs": [yi]}

            region_set = blob | {m1}
            f1 = _forward_slide(region_set, m1, dr, dc, {yf})
            if f1 == m1:
                continue
            yi = _find_reverse_slide(region_set, yf, dr, dc, {m1}, [], rng)
            if yi is None or yi == m1:
                continue
            return {"markers": [m1], "yellow_inputs": [yi]}
    return None


def _place_simple_yellow(rng: random.Random, reg: dict, grid: np.ndarray) -> bool:
    blob = set(reg["interior"])
    dr, dc = reg["dir"]
    outs = sorted(blob, key=lambda p: (_proj(p, dr, dc), p[0], p[1]), reverse=True)[:18]
    for out_pos in rng.sample(outs, len(outs)):
        inp = _find_reverse_slide(blob, out_pos, dr, dc, set(), [], rng)
        if inp is None:
            continue
        grid[inp] = 4
        return True
    return False


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

    reg = regions[chain_idx]
    for r, c in chain_layout["markers"]:
        grid[r, c] = reg["color"]
    for r, c in chain_layout["yellow_inputs"]:
        grid[r, c] = 4

    others = [i for i in order if i != chain_idx]
    rng.shuffle(others)
    for idx in others:
        if rng.random() < EXTRA_YELLOW_PROB:
            _place_simple_yellow(rng, regions[idx], grid)
    return True


def _oracle_check(input_grid: np.ndarray) -> bool:
    try:
        solver_out, stats = _solve_grid(input_grid, activate_markers=True, return_stats=True)
        baseline = _solve_grid(input_grid, activate_markers=False, return_stats=False)
    except Exception:
        return False
    if stats["activated_markers"] < 1:
        return False
    if np.array_equal(solver_out, baseline):
        return False
    if int((solver_out == 4).sum()) <= int((input_grid == 4).sum()):
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
