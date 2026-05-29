# Variant of 13e47133
# Dimension: D7 — 区域分割方式 (obstacle)
# Change: 单一颜色的轴对齐墙壁线段将网格分割为 3-6 个矩形或 L/T 形区域，墙壁像素不变... → 墙壁形成封闭矩形框架
"""
design_analysis:
  variant_geometry:
    - 墙壁不再是穿过整块画布的分割线，而是 3-6 个同色、1 像素厚、彼此嵌套的闭合矩形框。
    - 最外层框直接贴在网格边界上，这样不存在“框外”的自由区域；非墙区域正好分成若干层：
      每对相邻框之间是一圈矩形环带，最内层框内部是一个中心矩形区域。
    - 因而自由区域总数 = 框数，保持 3-6；其中大多数区域带有内外双边界，视觉上明显区别于原题的矩形/L/T 分区。

  behavior_trace:
    original_local_example:
      - 原题若有一条竖墙和一条横墙，solver 把每块非墙连通域当作一个普通区域。
      - 一个 6x6 矩形区域里若 marker 距边界深度依次为 0,1,2，则 palette 为 [m0,m1,m2]，
        整块区域按 distance mod 3 画成同心矩形框。
    variant_local_example:
      - 现在最外框贴边，下一层框向内缩 4 格，则两框之间形成一圈厚度 3 的环带。
      - solver 仍然把这圈环带当成一个非墙连通域；其边界同时来自外框和内框。
      - 若 marker 仍按该环带自身的深度 0,1 放置，则填色会沿着环带两侧同时向中间收缩，
        形成“绕孔”的同心色带，而不是在实心矩形里向中心收缩。
    difference:
      - 规则核心没变，变化点在“区域几何形状”。
      - 原逻辑依赖 bbox 角落放 marker；变体中若继续按矩形 bbox 假设选点，会把 marker 放进洞里或墙上，
        因此生成器必须改为依据真实连通域的 boundary-distance 选合法点。

  complexity_calibration:
    - 框数限制为 3-6，避免只有 1-2 层时退化回“普通分区”。
    - 相邻框之间的自由厚度限定为 1-4 格；薄环带允许更深的嵌套层数，厚环带则保留可见的内部色带。
    - marker 仍限制为每区 0-3 个、深度连续、颜色互异，保持原题的学习负担和颜色语义。

  implicit_assumption_audit:
    - main() 中 `free = g != wall` 只要求“墙色唯一”，并不要求墙是直线；闭合框仍满足。
    - main() 中 `_cc4(free)` 默认每个区域是 4 连通块；嵌套矩形环带确实是单个 4 连通块。
    - `_boundary_distance(component_mask)` 会把靠墙和靠网格边缘都视作边界。最外框贴边后，所有自由区域都只靠墙，
      不会错误吸收网格边缘作为额外边界；中心区与环带也都能正确得到深度。
    - `base = _mode(vals)` 依赖每个区域绝大多数像素仍是底色；生成器只放少量 marker，保证成立。
    - `dist_to_color[d]` 每个深度只取一个 marker；生成器必须保证同一区域同一深度最多放一个 marker。
    - `wall = second-most-common color` 不是几何推理，而是频数假设；生成器必须让墙色像素数严格大于任一 marker 颜色。

  placement_legality:
    - 通过对称内缩生成每一层框，天然不会越界、不会互相接触、不会重叠。
    - marker 位置直接从真实连通域中按指定深度筛选，避免放到洞内、墙上或越过内边界。
    - 最内层框尺寸保持至少 5x5，使中心自由区至少 3x3；这样中心区也能承载非平凡填充。

  observability_and_cleanliness:
    - 人眼能从少量样例看到：墙本身是一组闭合矩形框，输出色带沿着环带绕行，而不是落在被切开的矩形块上。
    - 颜色顺序只由可见的 marker 深度决定，不使用颜色 id 排序等不可见规则。
    - 每个区域仍只有“底色 + 少量 marker”两层信息，避免墙、marker、填充线索互相遮挡。

  generator_guarantees_for_new_main:
    - 保证墙色唯一且为全局第二众数。
    - 保证至少使用 3 种 marker 颜色，避免样例过于单调。
    - 保证框数分布覆盖 3-6，从而真正触发 D7 的“闭合框架分区”行为，而非偶然退化。
"""

import random

import numpy as np

from arc_utils import components_mask

DIR8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def _cc4(mask):
    return components_mask(mask, connectivity=4)


def _mode(arr):
    vals, counts = np.unique(np.asarray(arr), return_counts=True)
    return int(vals[counts.argmax()])


def _boundary_distance(component_mask: np.ndarray) -> np.ndarray:
    """
    Compute 8-neigh (Chebyshev) distance to boundary for True-cells in component_mask.
    Boundary cells are those that touch False or the grid edge in their 8-neighborhood.
    Returns dist array with -1 outside component.
    """
    h, w = component_mask.shape
    dist = np.full((h, w), -1, dtype=int)
    q: list[tuple[int, int]] = []

    for r in range(h):
        for c in range(w):
            if not component_mask[r, c]:
                continue
            is_boundary = False
            for dr, dc in DIR8:
                rr, cc = r + dr, c + dc
                if rr < 0 or rr >= h or cc < 0 or cc >= w:
                    is_boundary = True
                    break
                if not component_mask[rr, cc]:
                    is_boundary = True
                    break
            if is_boundary:
                dist[r, c] = 0
                q.append((r, c))

    qi = 0
    while qi < len(q):
        r, c = q[qi]
        qi += 1
        d = dist[r, c]
        for dr, dc in DIR8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and component_mask[rr, cc] and dist[rr, cc] == -1:
                dist[rr, cc] = d + 1
                q.append((rr, cc))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested framed regions with cyclic color bands derived from marker depths.

    Input structure:
    - A single wall color forms several 1px closed rectangular frames nested inside one another, with the outermost frame on the grid border.
    - Every non-wall region is either a rectangular ring between two frames or the center chamber inside the innermost frame, and may contain a few marker pixels of other colors.

    Rule:
    - Treat each 4-connected non-wall component independently and order its marker colors by Chebyshev distance from that component's own boundary.
    - If the shallowest marker starts deeper than distance 0, prepend the region's base color to the palette.
    - Repaint the whole component with palette[distance mod palette_length], so rings and the center chamber both become concentric local bands.

    Key insight: The closed frames only change the shape of the connected regions; once marker depths are read from the true component boundary, the same cyclic distance rule naturally wraps color bands around both the outer and inner edges of each ring.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    out = g.copy()
    free = g != wall
    comps = _cc4(free)

    for pts in comps:
        comp_mask = np.zeros((h, w), dtype=bool)
        for r, c in pts:
            comp_mask[r, c] = True

        vals = g[comp_mask]
        base = _mode(vals)
        dist = _boundary_distance(comp_mask)

        dist_to_color: dict[int, int] = {}
        for r, c in pts:
            v = int(g[r, c])
            if v == base:
                continue
            d = int(dist[r, c])
            if d < 0:
                continue
            if d not in dist_to_color:
                dist_to_color[d] = v

        palette: list[int] = []
        if dist_to_color:
            ds = sorted(dist_to_color.keys())
            if ds[0] > 0:
                palette.append(int(base))
            for d in ds:
                palette.append(int(dist_to_color[d]))
        else:
            palette = [int(base)]

        m = len(palette)
        for r, c in pts:
            d = int(dist[r, c])
            out[r, c] = palette[d % m]

    return out


H_MIN, H_MAX = 17, 30
W_MIN, W_MAX = 17, 30

N_FRAMES_MIN, N_FRAMES_MAX = 3, 6
MIN_FRAME_STEP = 2          # 1 wall cell + at least 1 free cell before next frame
MAX_FRAME_STEP = 5
MIN_INNER_FRAME_DIM = 5     # ensures a >=3x3 center region

MARKERS_PER_REGION_MAX = 3
MARKER_COUNT_WEIGHTS = [1, 2, 3, 3]
START_DIST_MAX = 1

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

GEN_MAX_ATTEMPTS = 300


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _frame_rects(H: int, W: int, rng: random.Random):
    max_inset = min((H - MIN_INNER_FRAME_DIM) // 2, (W - MIN_INNER_FRAME_DIM) // 2)
    max_frames = min(N_FRAMES_MAX, max_inset // MIN_FRAME_STEP + 1)
    if max_frames < N_FRAMES_MIN:
        return None

    n_frames = rng.randint(N_FRAMES_MIN, max_frames)
    remaining_inset = max_inset
    remaining_steps = n_frames - 1
    steps = []
    for _ in range(n_frames - 1):
        min_needed = MIN_FRAME_STEP * (remaining_steps - 1)
        step_hi = min(MAX_FRAME_STEP, remaining_inset - min_needed)
        step = rng.randint(MIN_FRAME_STEP, step_hi)
        steps.append(step)
        remaining_inset -= step
        remaining_steps -= 1

    rects = [(0, 0, H - 1, W - 1)]
    inset = 0
    for step in steps:
        inset += step
        rects.append((inset, inset, H - 1 - inset, W - 1 - inset))
    return rects


def _draw_frame(mask: np.ndarray, rect):
    r0, c0, r1, c1 = rect
    mask[r0, c0:c1 + 1] = 1
    mask[r1, c0:c1 + 1] = 1
    mask[r0:r1 + 1, c0] = 1
    mask[r0:r1 + 1, c1] = 1


def _corner_key(corner: int, r: int, c: int, r0: int, c0: int, r1: int, c1: int):
    rr = r - r0
    cc = c - c0
    hh = r1 - r0
    ww = c1 - c0
    if corner == 0:   # top-left
        return (rr + cc, rr, cc)
    if corner == 1:   # top-right
        return (rr + (ww - cc), rr, -cc)
    if corner == 2:   # bottom-left
        return ((hh - rr) + cc, -rr, cc)
    return ((hh - rr) + (ww - cc), -rr, -cc)  # bottom-right


def _plan_markers(pts: np.ndarray, shape: tuple[int, int], marker_pool: list[int], rng: random.Random):
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        comp_mask[r, c] = True

    dist = _boundary_distance(comp_mask)
    ds = dist[comp_mask]
    max_depth = int(ds.max()) if ds.size else -1
    if max_depth < 0:
        return []

    upper = min(MARKERS_PER_REGION_MAX, max_depth + 1)
    n = rng.choices(range(upper + 1), weights=MARKER_COUNT_WEIGHTS[:upper + 1])[0]
    if n == 0:
        return []

    start_hi = min(START_DIST_MAX, max_depth - (n - 1))
    start_d = rng.randint(0, start_hi)
    corner = rng.randrange(4)

    pts_list = [(int(r), int(c)) for r, c in pts]
    r0 = min(r for r, _ in pts_list)
    c0 = min(c for _, c in pts_list)
    r1 = max(r for r, _ in pts_list)
    c1 = max(c for _, c in pts_list)

    colors = rng.sample(marker_pool, n)
    markers = []
    used_cells = set()
    for offset, color in enumerate(colors):
        target_d = start_d + offset
        candidates = [(r, c) for r, c in pts_list if int(dist[r, c]) == target_d and (r, c) not in used_cells]
        if not candidates:
            return None
        candidates.sort(key=lambda rc: _corner_key(corner, rc[0], rc[1], r0, c0, r1, c1))
        rr, cc = candidates[0]
        used_cells.add((rr, cc))
        markers.append((target_d, color, rr, cc))
    return markers


def _stamp_markers(grid: np.ndarray, comps: list[np.ndarray], bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    for pts in comps:
        markers = _plan_markers(pts, grid.shape, marker_pool, rng)
        if markers is None:
            return None
        for _d, col, rr, cc in markers:
            if grid[rr, cc] != bg:
                return None
            grid[rr, cc] = col
            used.add(col)
    return used


def _validate_color_order(grid: np.ndarray, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _gen_attempt(rng: random.Random):
    H = rng.randint(H_MIN, H_MAX)
    W = rng.randint(W_MIN, W_MAX)

    rects = _frame_rects(H, W, rng)
    if rects is None:
        return None

    bg, wall, marker_pool = _pick_palette(rng)

    wall_mask = np.zeros((H, W), dtype=int)
    for rect in rects:
        _draw_frame(wall_mask, rect)

    grid = np.full((H, W), bg, dtype=int)
    grid[wall_mask == 1] = wall

    comps = _cc4(grid != wall)
    if not (N_FRAMES_MIN <= len(comps) <= N_FRAMES_MAX):
        return None

    used = _stamp_markers(grid, comps, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
