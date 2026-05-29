# Combo variant of 13e47133 (index 16)
# combo_spec: D6/1+D7/0
# Name: 区域拆分拼接+嵌套框架墙壁

"""
design_analysis:
  concrete_behavior_and_trace:
    - 原题先把 `g != wall` 的每个连通域当作一个普通区域，按该区域到边界的切比雪夫距离读取 marker，
      得到 palette 后直接回写到原画布，因此输出尺寸与输入一致。
    - 这个融合版先把墙改成嵌套闭合矩形框：最外框贴边，内层继续向内缩，因而得到
      `外环 / 中环 / 中心` 三个区域。外环和中环都不是实心矩形，而是带洞的环带。
    - 小例子：若输入是 25x15，框的 inset 为 0,2,4，则三个自由区域的 bbox 宽度依次约为 13,9,5。
      原题/单 D7 会把三个区域都填回原画布；本融合版会先分别完成填充，再把三块裁成
      `甜甜圈 tile、甜甜圈 tile、中心 tile`，按外到内顺序水平并排，用墙色列隔开。
    - 差异不只是“换几何”或“换输出形状”之一，而是两者联动：因为区域本身是带洞环带，
      被裁出的 tile 也保留墙色孔洞，人眼能直接看出是从嵌套框中拆出来的独立小网格。

  complexity_calibration:
    - 若仍允许任意 3-6 层框，再把所有 tile 单行并排，输出宽度很容易超过 30。
      因此生成器固定为 3 个区域：两个环带 + 一个中心区。这已经足够显式触发 D7 的环形区域，
      同时也让 D6 的“拆出并排拼接”稳定落在 ARC 尺寸范围内。
    - 宽度固定在 18，并把两次 inward step 固定为 `(4,2)`：
      这样外环有 3 格自由厚度，确实能出现 `distance=1` 的内层色带；同时三块 tile 并排后宽度刚好不超过 30。
    - 每区仍只放 1-3 个 marker，且深度连续；同时偏向产生至少一个 palette 长度 >= 2 的环带，
      避免全部退化成纯色甜甜圈。

  implicit_assumption_audit:
    - `wall = second-most-common color`:
      原代码依赖墙色是全局第二众数。嵌套框会显著增加墙像素数，因此输入画布必须做成足够高，
      让底色像素仍多于墙色。生成器用 20-29 x 18 的细长画布来保证这点。
    - `_cc4(g != wall)`:
      原代码不要求区域必须是矩形，只要求非墙区域是 4 连通块。矩形环带满足这一点，
      因此 solver 无须新增拓扑特判。
    - `_boundary_distance(component_mask)`:
      该函数默认“接触 False 或网格边缘即为边界”。最外框贴边后，所有自由区域都由墙围出，
      不会错误把画布外侧当成额外自由空间；环带的内外两圈边界也会自然同时起作用。
    - `dist_to_color[d]`:
      solver 只保留每个深度遇到的第一个 marker，因此生成器必须保证同一区域同一深度至多一个 marker。
      这里直接从真实连通域的某个目标深度层里挑一个格子放 marker。
    - `sorted(comps, key=min_row_col)`:
      融合版新增了“按顺序拼接”的假设。对嵌套框来说，外环、中环、中心区的左上角严格沿对角线向内递增，
      所以按 `(min_row, min_col)` 排序恰好等于从外到内的自然顺序。生成器按此结构构造输入。

  placement_legality:
    - 框通过合法 inset 生成，天然不越界、不重叠，且相邻框之间至少隔 1 格自由空间。
    - marker 不再基于 bbox 角点猜位置，而是先求真实连通域的 boundary-distance，
      再在目标深度层中挑格子，因此不会落进洞里、不会盖到墙上，也不会跑出连通域。
    - 输出 tile 统一用墙色打底，再把区域像素贴进去，于是环带的孔洞自然保留成墙色空腔。

  observability_cleanliness_and_inferability:
    - 输入一眼能看见是三层同色闭合矩形框；输出一眼能看见是拆开的三个小块，其中前两个带孔。
      这让解题者能同时观察到 D7 的“环形区域”与 D6 的“拆分重组”。
    - 颜色顺序完全由可见的 marker 深度决定，没有按颜色 id 排序等不可视规则。
    - tile 之间只用墙色分隔列，不再叠加其他装饰，避免“墙、marker、填充”三类信息互相遮挡。

  generator_guarantees_for_main:
    - 保证恰有 3 个非墙连通域，其中恰有 2 个是带洞环带。
    - 保证输出宽高都 <= 30，且输出形状不同于输入。
    - 保证至少一个环带拥有非平凡 palette，使“同心带填充”不是只剩下纯色重涂。
"""

import random

import numpy as np

from arc_utils import components_mask, paste

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


def _component_key(pts: np.ndarray) -> tuple[int, int]:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.min()), int(cols.min())


def _bbox_area(pts: np.ndarray) -> int:
    rows = pts[:, 0]
    cols = pts[:, 1]
    return int(rows.max() - rows.min() + 1) * int(cols.max() - cols.min() + 1)


def _filled_region_tile(g: np.ndarray, pts: np.ndarray, wall: int) -> np.ndarray:
    h, w = g.shape
    comp_mask = np.zeros((h, w), dtype=bool)
    for r, c in pts:
        comp_mask[int(r), int(c)] = True

    vals = g[comp_mask]
    base = _mode(vals)
    dist = _boundary_distance(comp_mask)

    dist_to_color: dict[int, int] = {}
    for r, c in sorted((int(r), int(c)) for r, c in pts):
        v = int(g[r, c])
        if v == base:
            continue
        d = int(dist[r, c])
        if d >= 0 and d not in dist_to_color:
            dist_to_color[d] = v

    if dist_to_color:
        ds = sorted(dist_to_color)
        palette = [int(base)] if ds[0] > 0 else []
        palette.extend(int(dist_to_color[d]) for d in ds)
    else:
        palette = [int(base)]

    r0 = int(pts[:, 0].min())
    c0 = int(pts[:, 1].min())
    r1 = int(pts[:, 0].max())
    c1 = int(pts[:, 1].max())
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), wall, dtype=int)

    m = len(palette)
    for r, c in pts:
        rr = int(r)
        cc = int(c)
        tile[rr - r0, cc - c0] = palette[int(dist[rr, cc]) % m]
    return tile


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill nested framed regions by boundary depth, then split the solved rings into standalone tiles.

    Input structure:
    - One wall color forms nested 1px rectangular frames, with the outermost frame on the grid border.
    - The non-wall components are rectangular rings plus a center chamber, each dominated by one base color and containing a few marker pixels at specific Chebyshev depths from that component's own boundary.

    Rule:
    - Solve each 4-connected non-wall component independently: order marker colors by boundary distance, optionally prepend the base color when the first marker is deeper than 0, then repaint the component with palette[distance mod palette_length].
    - After filling, crop every solved component to its own bounding box, keeping wall-colored cells wherever the cropped bbox contains the ring's hole.
    - Reassemble those cropped tiles from outermost region to innermost region in reading order, placing them side by side with one wall-colored separator column between neighboring tiles.

    Key insight:
    - The same distance-to-boundary palette rule still explains the local coloring, but the final answer no longer lives on the original canvas: the closed-frame geometry creates donut-shaped components, and D6 turns those solved components into independent wall-backed tiles.
    """
    g = np.asarray(grid, dtype=int)

    bg = _mode(g)
    other = g[g != bg]
    wall = bg if other.size == 0 else _mode(other)

    comps = sorted(_cc4(g != wall), key=_component_key)
    if not comps:
        return g.copy()

    tiles = [_filled_region_tile(g, pts, wall) for pts in comps]
    out_h = max(tile.shape[0] for tile in tiles)
    out_w = sum(tile.shape[1] for tile in tiles) + (len(tiles) - 1)
    out = np.full((out_h, out_w), wall, dtype=int)

    c = 0
    for tile in tiles:
        paste(out, tile, 0, c)
        c += tile.shape[1] + 1
    return out


# --------------- generator ---------------

H_MIN, H_MAX = 19, 29
W_MIN, W_MAX = 17, 19
OUT_MAX = 30

N_FRAMES = 3
FRAME_INNER_GAP = 2
FRAME_OUTER_STEPS = (4, 5)
MIN_INNER_FRAME_DIM = 5

MARKERS_PER_REGION_MAX = 2
START_DIST_MAX = 1
MARKER_COUNT_WEIGHTS = [4, 3]  # for 1,2 markers
EMPTY_REGION_WEIGHT = 2
START_DEPTH_WEIGHTS = [3, 2]  # prefer distance 0, but keep distance 1 common enough
CENTER_FORCE_NONTRIVIAL_PROB = 0.4
CENTER_FORCE_OFFSET_PROB = 0.4

DISTINCT_MARKER_COLORS_MIN, DISTINCT_MARKER_COLORS_MAX = 3, 9

GEN_MAX_ATTEMPTS = 500


def _pick_palette(rng: random.Random):
    colors = list(range(10))
    bg = rng.choice(colors)
    colors.remove(bg)
    wall = rng.choice(colors)
    colors.remove(wall)
    return bg, wall, colors


def _frame_rects(H: int, W: int, rng: random.Random):
    candidates = []
    for step1 in FRAME_OUTER_STEPS:
        step2 = FRAME_INNER_GAP
        total_inset = step1 + step2
        if min(H, W) - 2 * total_inset < MIN_INNER_FRAME_DIM:
            continue
        packed_w = 3 * W - 4 * step1 - 2 * step2 - 4
        if packed_w <= OUT_MAX:
            candidates.append((step1, step2))
    if not candidates:
        return None
    step1, step2 = rng.choice(candidates)
    insets = [0, step1, step1 + step2]
    return [(inset, inset, H - 1 - inset, W - 1 - inset) for inset in insets]


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
    if corner == 0:
        return (rr + cc, rr, cc)
    if corner == 1:
        return (rr + (ww - cc), rr, -cc)
    if corner == 2:
        return ((hh - rr) + cc, -rr, cc)
    return ((hh - rr) + (ww - cc), -rr, -cc)


def _plan_markers(
    pts: np.ndarray,
    shape: tuple[int, int],
    rng: random.Random,
    *,
    force_nontrivial: bool = False,
    prefer_offset: bool = False,
):
    comp_mask = np.zeros(shape, dtype=bool)
    for r, c in pts:
        comp_mask[int(r), int(c)] = True

    dist = _boundary_distance(comp_mask)
    ds = dist[comp_mask]
    max_depth = int(ds.max()) if ds.size else -1
    if max_depth < 0:
        return []

    upper = min(MARKERS_PER_REGION_MAX, max_depth + 1)
    if upper <= 0:
        return []

    if force_nontrivial:
        if upper < 2:
            return None
        n = 2
        start_d = 0
    elif prefer_offset and max_depth >= 1:
        n = 1
        start_d = 1
    else:
        choices = list(range(0, upper + 1))
        weights = [EMPTY_REGION_WEIGHT] + MARKER_COUNT_WEIGHTS[:upper]
        n = rng.choices(choices, weights=weights)[0]
        if n == 0:
            return []
        start_hi = min(START_DIST_MAX, max_depth - (n - 1))
        if start_hi <= 0:
            start_d = 0
        else:
            start_d = rng.choices(list(range(start_hi + 1)), weights=START_DEPTH_WEIGHTS[:start_hi + 1])[0]
    corner = rng.randrange(4)

    pts_list = [(int(r), int(c)) for r, c in pts]
    r0 = min(r for r, _ in pts_list)
    c0 = min(c for _, c in pts_list)
    r1 = max(r for r, _ in pts_list)
    c1 = max(c for _, c in pts_list)

    markers = []
    used_cells = set()
    for offset in range(n):
        target_d = start_d + offset
        candidates = [
            (r, c)
            for r, c in pts_list
            if int(dist[r, c]) == target_d and (r, c) not in used_cells
        ]
        if not candidates:
            return None
        candidates.sort(key=lambda rc: _corner_key(corner, rc[0], rc[1], r0, c0, r1, c1))
        rr, cc = candidates[0]
        used_cells.add((rr, cc))
        markers.append((target_d, rr, cc))
    return markers


def _stamp_markers(grid: np.ndarray, comps: list[np.ndarray], bg: int, marker_pool: list[int], rng: random.Random):
    used = set()
    markers_per_region = []
    last_idx = len(comps) - 1
    for idx, pts in enumerate(comps):
        force_nontrivial = idx == 0 or (idx == last_idx and rng.random() < CENTER_FORCE_NONTRIVIAL_PROB)
        prefer_offset = (idx == last_idx and not force_nontrivial and rng.random() < CENTER_FORCE_OFFSET_PROB)
        markers = _plan_markers(
            pts,
            grid.shape,
            rng,
            force_nontrivial=force_nontrivial,
            prefer_offset=prefer_offset,
        )
        if markers is None:
            return None, None
        if len(markers) == 0:
            markers_per_region.append([])
            continue
        fresh = [c for c in marker_pool if c not in used]
        if len(fresh) < len(markers):
            return None, None
        colors = rng.sample(fresh, len(markers))
        placed = []
        for (d, rr, cc), col in zip(markers, colors):
            if grid[rr, cc] != bg:
                return None, None
            grid[rr, cc] = col
            used.add(col)
            placed.append((d, col, rr, cc))
        markers_per_region.append(placed)
    return used, markers_per_region


def _validate_color_order(grid: np.ndarray, bg: int, wall: int):
    vals, counts = np.unique(grid, return_counts=True)
    order = np.argsort(-counts)
    if int(vals[order[0]]) != bg:
        return False
    return len(order) >= 2 and int(vals[order[1]]) == wall


def _palette_is_nontrivial(markers) -> bool:
    if not markers:
        return False
    return int(markers[0][0]) > 0 or len(markers) >= 2


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

    comps = sorted(_cc4(grid != wall), key=_component_key)
    if len(comps) != N_FRAMES:
        return None
    if sum(int(len(pts) < _bbox_area(pts)) for pts in comps) != N_FRAMES - 1:
        return None

    used, markers_per_region = _stamp_markers(grid, comps, bg, marker_pool, rng)
    if used is None:
        return None
    if not (DISTINCT_MARKER_COLORS_MIN <= len(used) <= DISTINCT_MARKER_COLORS_MAX):
        return None
    if not any(_palette_is_nontrivial(markers_per_region[i]) for i in range(N_FRAMES - 1)):
        return None
    if not _validate_color_order(grid, bg, wall):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape == grid.shape:
        return None
    if out.shape[0] > OUT_MAX or out.shape[1] > OUT_MAX:
        return None
    if len(set(map(int, out.ravel()))) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_MAX_ATTEMPTS):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError(f"generate_input failed after {GEN_MAX_ATTEMPTS} attempts")
