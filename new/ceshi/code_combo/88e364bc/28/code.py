# Combo variant of 88e364bc (index 28)
# combo_spec: D8/1+D10/2
# Name: 反向堆叠顺序+密集像素

"""
设计分析：88e364bc 的融合变体（D8/1 + D10/2）。

a) 具体化与行为追踪
  - 仍然是“图例颜色 -> 墙色 -> 滑动方向”的结构；变化只发生在每个封闭区域内部的黄色像素。
  - 每个区域现在固定放 3-5 个黄色像素，并且它们不是随便散放，而是放在同一条终端滑道上：
    终态车道记作 [p0, p1, p2, p3, ...]，其中 p0 紧贴目标墙，p1 在其后一格，依此类推。
  - 生成器先决定输出要占据 p0..p{k-1}，再把输入放成 p1..p{k}。这样输入是“离墙一格的稠密带”。
  - 小例子：方向向右、k=4，输入在 [p1, p2, p3, p4]。
    融合规则（近处先处理）按 [p1, p2, p3, p4] 依次滑动：
      p1 -> p0, p2 -> p1, p3 -> p2, p4 -> p3，得到 [p0, p1, p2, p3]。
    若按相反顺序（远处先处理）：
      p4 -> p0, p3 -> p1, p2 -> p2, p1 会被已停像素顶住而留在 p1，
      于是只能看到 [p0, p1, p2]，和融合规则明显不同。
  - 因此这里不是“单纯多几个黄色像素”，而是利用稠密输入让处理顺序成为决定性的可见规则。

b) 复杂度校准
  - 保留原题的 1-2 个图例色、1-3 个封闭区域、20x20 尺寸和不规则墙体，避免把任务改成完全不同的题。
  - 但去掉空区域，并把总区域数压到最多 3 个，否则每区 3-5 个黄色像素会让画面过挤。
  - 终端车道只取“一条直滑道”，并把输入固定成紧挨终态的一格偏移；顺序差异非常明显，同时不会引入第二套规则。

c) 隐式假设审查
  - _legend_directions 依赖：图例是贴边实心矩形、恰有三种颜色、填充色最多。融合版不改图例，所以该假设仍成立。
  - main() 依赖：黄色仍是唯一会移动的颜色，且位于被墙包围的 passable 连通块内。密集黄色仍满足此条件。
  - _slide_points 最敏感的前提是“排序方向”。这里必须按离目标墙最近者优先处理。
    对于代码中的投影 key，`reverse=True` 恰好表示“墙边最近的先动”；不能把它反过来。
  - 清空旧黄色再重绘仍然安全，因为区域内没有其他会复用颜色 4 的静态对象。
  - border color 的判定仍靠邻接墙色多数票；生成器继续保证每个 interior 只被一种墙色包围。

d) 放置合法性
  - 图例先落在角上，并预留 1 格缓冲区。
  - 区域仍由不规则 blob + 其 4 邻域墙环构成；blob、墙、缓冲互不重叠，也不碰外边框。
  - 黄色只放在某条合法终端车道上，输入为 `lane[1:count+1]`，因此天然不越界、不重叠，且全都能滑到同一区域内。

e) 视觉可观察性
  - 人类在 2-4 个样例里能直接看到：输入中的每个区域都有 3-5 个黄色像素，且它们离目标墙只差一格；
    输出则变成紧贴远端墙的连续密集堆叠。
  - 图例仍提供方向，墙色仍提供区域归属，新的可见差异完全来自“密集输入 + 最近者先处理”的堆叠结果。

f) 可视化整洁度
  - 不引入额外标记色，不叠加第二种局部编码。
  - 通过减少区域数、增大区域最小尺寸、禁止空区域，保持每个黄色带和各区域边界都清晰可读。

g) 变化可推断性
  - 本融合不涉及颜色重映射；颜色的职责仍然稳定：
    背景 0、中心 1、指示 2、墙色若干、黄色 4。
  - 变化只有一个：区域内多个黄色像素在同一方向上按“近 -> 远”顺序处理，且数量变成 3-5。
    由于输入总是紧邻终态带后一格，人类可以从样例中稳定归纳这一点。
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, components_mask, get_bg, label_mask, neighbors4, paste


def _projection_key(p: tuple[int, int], dr: int, dc: int) -> tuple[int, int, int]:
    return dr * p[0] + dc * p[1], p[0], p[1]


def _legend_directions(g: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """
    Detect border legends and decode fill-color -> movement-direction.
    """
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
        others = sorted((cnt, col) for col, cnt in counts.items() if col != fill_col)
        if len(others) != 2:
            continue

        if others[0][0] == others[1][0]:
            rc = np.array([(r0 + r1) / 2.0, (c0 + c1) / 2.0])
            scored = []
            for _, col in others:
                pts_col = np.array([(r, c) for r, c in pts if int(g[r, c]) == col], dtype=float)
                scored.append((np.sum((pts_col.mean(axis=0) - rc) ** 2), col))
            scored.sort()
            center_col = scored[0][1]
            indicator_col = scored[1][1]
        else:
            indicator_col = others[0][1]
            center_col = others[1][1]

        center = np.array([(r, c) for r, c in pts if int(g[r, c]) == center_col], dtype=float)
        indicator = np.array([(r, c) for r, c in pts if int(g[r, c]) == indicator_col], dtype=float)
        if len(center) == 0 or len(indicator) == 0:
            continue

        delta = indicator.mean(axis=0) - center.mean(axis=0)
        dr = 0 if abs(float(delta[0])) < 1e-6 else (1 if delta[0] > 0 else -1)
        dc = 0 if abs(float(delta[1])) < 1e-6 else (1 if delta[1] > 0 else -1)
        if dr == 0 and dc == 0:
            continue

        prev = best.get(fill_col)
        if prev is None or len(pts) > prev[0]:
            best[fill_col] = (len(pts), (dr, dc))

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
    *,
    near_first: bool = True,
) -> list[tuple[int, int]]:
    if not points:
        return []
    points_sorted = sorted(points, key=lambda p: _projection_key(p, dr, dc), reverse=near_first)
    occupied: set[tuple[int, int]] = set()
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
    g: np.ndarray,
    pts: list[tuple[int, int]],
    pset: set[tuple[int, int]],
    bg: int,
    mover: int,
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide dense yellow groups inside colored enclosures using the border legends, with nearest-to-wall pixels moving first.

    Input structure:
    - Background 0 fills the canvas; closed colored wall loops enclose passable regions made of background and yellow cells.
    - Border legends are solid edge rectangles whose dominant fill color matches one wall color; the inner blue/red markers encode a direction.
    - Each occupied region now contains a dense group of 3-5 yellow pixels instead of the sparse 1-2 pattern from the source task.

    Rule:
    - For each enclosed passable region, identify its surrounding wall color and look up the legend with the same fill color.
    - Decode the motion vector from the legend and slide every yellow pixel in that region along the vector.
    - When several yellows share one region, process them from the pixel already closest to the destination wall toward the farther ones.
      Earlier pixels become blockers, so later pixels stack immediately behind them.

    Key insight:
    - The legend still tells you where a region moves, but the fused variant adds a second visible requirement:
      the yellows must be resolved in wall-nearest-first order.
      Because each region contains a compact 3-5-pixel band, using the wrong order produces a visibly different stack.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    mover = 4
    dirs = _legend_directions(g, bg)
    assert dirs, "legend_directions must find at least one legend"

    passable = (g == bg) | (g == mover)
    out = g.copy()

    for pts in components_mask(passable, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        movers = [(int(r), int(c)) for r, c in pts if int(g[r, c]) == mover]
        if not movers:
            continue

        pset = {(int(r), int(c)) for r, c in pts}
        border_col = _region_border_color(g, pts, pset, bg, mover)
        if border_col is None or border_col not in dirs:
            continue
        dr, dc = dirs[border_col]

        for r, c in movers:
            out[r, c] = bg

        for r, c in _slide_points(pset, movers, dr, dc, near_first=True):
            out[r, c] = mover

    return out


GRID_H, GRID_W = 20, 20

CARDINAL_DIRS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
DIAGONAL_DIRS = [(1, 1), (1, -1), (-1, 1), (-1, -1)]
CORNERS = ["TL", "TR", "BL", "BR"]
WALL_COLOR_CHOICES = [3, 5, 6, 7, 8, 9]
CARDINAL_PROB = 0.5
LEGEND_BUFFER = 1

REGIONS_PER_COLOR = (1, 2)
MAX_TOTAL_REGIONS = 3
REGION_SIZE_BOUNDS = {
    1: (28, 85),
    2: (24, 55),
    3: (20, 38),
}
BLOB_GROW_RETRIES = 25

YELLOWS_PER_REGION = [3, 4, 5]
INPUT_OFFSET = 1

GEN_RETRIES = 400


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
    canvas_mask: np.ndarray,
    seed: tuple[int, int],
    target_size: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    h, w = canvas_mask.shape
    sr, sc = seed
    if not canvas_mask[sr, sc]:
        return None
    blob = {(sr, sc)}
    frontier = []

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
    stack = []
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
    rng: random.Random,
    grid: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, tuple[int, int]]]]:
    h, w = grid.shape
    n_legends = rng.choice([1, 2])
    corners = rng.sample(CORNERS, n_legends)
    wall_colors = rng.sample(WALL_COLOR_CHOICES, n_legends)

    legend_occupied = np.zeros((h, w), dtype=bool)
    legends = []
    for corner, wall_color in zip(corners, wall_colors):
        if rng.random() < CARDINAL_PROB:
            dr, dc = rng.choice(CARDINAL_DIRS)
            patch = _legend_5x5(wall_color, dr, dc)
        else:
            dr, dc = rng.choice(DIAGONAL_DIRS)
            patch = _legend_6x7(wall_color, dr, dc)
        ph, pw = patch.shape
        r0, c0 = _corner_origin(corner, h, w, ph, pw)
        paste(grid, patch, r0, c0)
        r_lo, r_hi = max(0, r0 - LEGEND_BUFFER), min(h, r0 + ph + LEGEND_BUFFER)
        c_lo, c_hi = max(0, c0 - LEGEND_BUFFER), min(w, c0 + pw + LEGEND_BUFFER)
        legend_occupied[r_lo:r_hi, c_lo:c_hi] = True
        legends.append((wall_color, (dr, dc)))

    return legend_occupied, legends


def _assigned_region_colors(rng: random.Random, all_colors: list[int]) -> list[int]:
    counts = {color: 1 for color in all_colors}
    remaining = MAX_TOTAL_REGIONS - len(all_colors)
    for color in rng.sample(all_colors, len(all_colors)):
        if remaining <= 0:
            break
        extra_cap = REGIONS_PER_COLOR[1] - counts[color]
        if extra_cap <= 0:
            continue
        extra = rng.randint(0, min(extra_cap, remaining))
        counts[color] += extra
        remaining -= extra

    region_colors = []
    for color in all_colors:
        region_colors.extend([color] * counts[color])
    rng.shuffle(region_colors)
    return region_colors


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
    color_dir = {wall_color: direction for wall_color, direction in legends}
    region_colors = _assigned_region_colors(rng, [wall_color for wall_color, _ in legends])

    size_min, size_max = REGION_SIZE_BOUNDS[len(region_colors)]
    occupied = legend_occupied.copy()
    regions = []

    for wall_color in region_colors:
        target_size = rng.randint(size_min, size_max)
        canvas = np.ones((h, w), dtype=bool)
        canvas[0, :] = False
        canvas[-1, :] = False
        canvas[:, 0] = False
        canvas[:, -1] = False
        canvas &= ~occupied
        if not canvas.any():
            return None

        blob = _try_grow_valid_blob(rng, canvas, occupied, target_size, h, w)
        if blob is None:
            return None

        walls = _blob_wall_cells(blob, h, w)
        regions.append(
            {
                "color": wall_color,
                "dir": color_dir[wall_color],
                "interior": blob,
                "walls": walls,
            }
        )

        reserve = blob | walls
        extra = set()
        for r, c in walls:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in reserve:
                    extra.add((nr, nc))
        for r, c in reserve | extra:
            occupied[r, c] = True

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


def _terminal_lanes(blob: set[tuple[int, int]], dr: int, dc: int) -> list[list[tuple[int, int]]]:
    lanes = []
    for cell in blob:
        if _slide_step_valid(blob, cell[0], cell[1], dr, dc):
            continue
        lane = [cell]
        while True:
            prev = (lane[-1][0] - dr, lane[-1][1] - dc)
            if prev not in blob:
                break
            if not _slide_step_valid(blob, prev[0], prev[1], dr, dc):
                break
            lane.append(prev)
        if len(lane) >= min(YELLOWS_PER_REGION) + INPUT_OFFSET:
            lanes.append(lane)
    lanes.sort(key=lambda lane: (_projection_key(lane[0], dr, dc), len(lane)), reverse=True)
    return lanes


def _choose_dense_reverse_plan(
    rng: random.Random,
    blob: set[tuple[int, int]],
    dr: int,
    dc: int,
) -> dict | None:
    plans = []
    for lane in _terminal_lanes(blob, dr, dc):
        max_count = min(max(YELLOWS_PER_REGION), len(lane) - INPUT_OFFSET)
        for count in YELLOWS_PER_REGION:
            if count > max_count:
                continue
            plans.append(
                {
                    "count": count,
                    "lane_len": len(lane),
                    "output": lane[:count],
                    "input": lane[INPUT_OFFSET : INPUT_OFFSET + count],
                }
            )
    if not plans:
        return None
    plans.sort(key=lambda plan: (plan["count"], plan["lane_len"]), reverse=True)
    top = plans[: min(12, len(plans))]
    return top[rng.randrange(len(top))]


def _place_yellows(rng: random.Random, regions: list[dict], grid: np.ndarray) -> bool:
    for reg in regions:
        blob = reg["interior"]
        dr, dc = reg["dir"]
        plan = _choose_dense_reverse_plan(rng, blob, dr, dc)
        if plan is None:
            return False
        reg["input_yellows"] = list(plan["input"])
        reg["output_yellows"] = list(plan["output"])
        reg["yellow_count"] = int(plan["count"])
        for r, c in reg["input_yellows"]:
            grid[r, c] = 4
    return True


def _oracle_check(input_grid: np.ndarray, regions: list[dict]) -> bool:
    try:
        solver_out = main(input_grid)
    except Exception:
        return False
    expected = input_grid.copy()
    expected[expected == 4] = 0
    for reg in regions:
        count = len(reg.get("output_yellows", []))
        if count not in YELLOWS_PER_REGION:
            return False
        for r, c in reg["output_yellows"]:
            expected[r, c] = 4
    return np.array_equal(solver_out, expected)


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = np.zeros((GRID_H, GRID_W), dtype=int)
        legend_occupied, legends = _place_legends(rng, grid)
        regions = _grow_regions(rng, legends, legend_occupied)
        if regions is None:
            continue

        for reg in regions:
            for r, c in reg["walls"]:
                grid[r, c] = reg["color"]

        if not _validate_topology(grid, regions):
            continue

        input_grid = grid.copy()
        if not _place_yellows(rng, regions, input_grid):
            continue
        if not _oracle_check(input_grid, regions):
            continue
        return input_grid
    raise RuntimeError("generate_input: exceeded retries")
