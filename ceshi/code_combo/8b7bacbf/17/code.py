# Combo variant of 8b7bacbf (index 17)
# combo_spec: D3/1+D17/1
# Name: 交替色填充+菱形框架

"""
design_analysis:
  fusion_goal:
    preserve:
      - 边缘唯一色单像素 seed 通过 8 连通线连接到若干闭合框架
      - 只有属于该 seed 连通绘制网络的闭合腔体会被改写
      - 仍允许存在不应被填的干扰菱形或无 seed 的干扰网络
    change:
      - 框架从圆角矩形改为曼哈顿几何的菱形环
      - 填充从纯 seed 色 flood-fill 改为 seed 色与环色交替
      - 交替不按全局网格奇偶，而按菱形对角轴上的斜条纹分带

  a_concretization_and_trace:
    input_objects:
      - 菱形环由单色 ring 组成，满足 |dr|+|dc|=R
      - 腔体为背景色菱形，满足 |dr|+|dc|<R
      - 线条只在腔体外侧与 ring 8 邻接，不直接贴到 cavity
    tiny_trace:
      - 设 seed 色为 7，ring 色为 3，菱形中心在 (6,6)，半径 R=3
      - 原题输出：13 个腔体像素全部填成 7
      - 仅做 D3 时若用全局棋盘：同一菱形内按格点奇偶交替，模式与形状几乎解耦
      - 本融合版输出：沿固定菱形对角轴产生双宽斜条纹
        - 相邻两条平行对角线共享一种颜色
        - 再往另一侧的两条对角线切换到另一种颜色
      - 差异点不只是“多了一种颜色”，而是双色分布和菱形对角方向绑定

  b_complexity_calibration:
    - 若仍用全局奇偶棋盘，D3 和 D17 基本是彼此独立叠加，融合感很弱
    - 采用沿菱形对角轴的双宽斜条纹后，只有识别出局部菱形坐标系，才能得到正确填充
    - 为避免半径 2 的小菱形和全局棋盘偶然近似，attached 菱形半径偏向 3-4
    - 颜色规则只使用 seed 色与该环自身颜色，保持低自由度和可归纳性

  c_implicit_assumption_audit_main:
    - seed 检测仍依赖“唯一非背景色且位于边界”，该前提不变
    - 从 seed 出发追踪 8 连通非背景组件的逻辑不依赖环形状，仍成立
    - 背景区域按 4 连通分解依旧成立；菱形腔体在 4 连通下是单一连通块
    - len(boundary colors)==1 现在更重要，因为环色直接参与内部图案
    - 原版 `out[r,c]=seed_col` 的逐点纯填不再正确，必须先恢复菱形中心/半径
    - 新增假设：要被填的 cavity 必须真的是规则菱形；generator 会显式保证

  d_placement_legality:
    - 菱形中心必须满足整颗菱形 bbox 离边界至少 1 格
    - ring 与 cavity 均不能覆盖已有像素
    - cavity 不能 8 邻接 line_col，避免线条看起来钻入腔体
    - ring 不能 8 邻接外来非背景颜色，避免不同结构粘连成歧义组件
    - attached ring 至少要有一个像素 8 邻接到所属 line_col

  e_visual_observability:
    - 输入里能直接看出“框架是菱形而非矩形”
    - 输出里能直接看出“填充沿菱形对角方向成斜条纹”，不是普通 flood-fill
    - 多个半径 3-4 的样例足以让人从少量 I/O 对中归纳出图案方向来自菱形本身

  f_visual_cleanliness:
    - 保持 1px 线条、1 层 ring、1 个 cavity，不叠额外标记
    - 通过 margin、foreign-color adjacency、cavity-vs-line 间隔约束避免画面粘连
    - 限制总环数，防止多 seed 多菱形叠在一起导致阅读负担过大

  g_change_inferability:
    - 色彩来源完全局部可见：一种来自边缘 seed，一种来自包围该 cavity 的 ring
    - 图案来源也局部可见：由菱形的中心和对角轴分带决定，不依赖颜色编号大小
    - 额外用生成校验排除“融合版输出恰好等于全局棋盘输出”的退化样本

  post_change_main_assumptions_and_generator_guards:
    - 每个被 main() 真正改写的 cavity 都应能解析为规则菱形
    - 被改写 cavity 的输出应只包含 seed 色和对应 ring 色
    - 至少一个被改写 cavity 对比全局棋盘填法存在显著差异
    - generate_input() 用 oracle + variant_metrics() 联合保证这些条件
"""

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        r, c = map(int, np.argwhere(g == v)[0])
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            seeds.append((r, c))
    return seeds


def _nonbg_component8_from(g: np.ndarray, start: tuple[int, int], bg: int) -> set[tuple[int, int]]:
    h, w = g.shape
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and int(g[nr, nc]) != bg:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return seen


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    regions: list[tuple[set[tuple[int, int]], set[tuple[int, int]]]] = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (region, _) in enumerate(regions):
        if any(p in region for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    out = []
    for i, (region, boundary) in enumerate(regions):
        if i in outside_idxs or not boundary:
            continue
        out.append((region, boundary, {int(g[p]) for p in boundary}))
    return out


def _diamond_center_radius(cavity: set[tuple[int, int]]) -> tuple[int, int, int] | None:
    rows: dict[int, list[int]] = {}
    for r, c in cavity:
        rows.setdefault(r, []).append(c)

    row_keys = sorted(rows)
    lengths: list[int] = []
    for r in row_keys:
        cols = sorted(rows[r])
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        lengths.append(len(cols))

    peak = max(lengths)
    peak_idx = lengths.index(peak)
    if lengths.count(peak) != 1 or peak % 2 == 0:
        return None

    if lengths[: peak_idx + 1] != list(range(1, peak + 1, 2)):
        return None
    if lengths[peak_idx:] != list(range(peak, 0, -2)):
        return None

    center_r = row_keys[peak_idx]
    center_cols = sorted(rows[center_r])
    center_c = (center_cols[0] + center_cols[-1]) // 2
    radius = (peak + 1) // 2

    expected = {
        (center_r + dr, center_c + dc)
        for dr in range(-(radius - 1), radius)
        for dc in range(-(radius - 1), radius)
        if abs(dr) + abs(dc) < radius
    }
    if expected != cavity:
        return None
    return center_r, center_c, radius


def _diamond_ring_match(
    cavity: set[tuple[int, int]], boundary: set[tuple[int, int]]
) -> tuple[int, int, int] | None:
    parsed = _diamond_center_radius(cavity)
    if parsed is None:
        return None

    center_r, center_c, radius = parsed
    expected_boundary = {
        (center_r + dr, center_c + dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if abs(dr) + abs(dc) == radius
    }
    if boundary != expected_boundary:
        return None
    return parsed


def _diamond_stripe_color(
    r: int,
    c: int,
    center_r: int,
    center_c: int,
    radius: int,
    seed_col: int,
    alt_col: int,
) -> int:
    diag_index = (r - c) - (center_r - center_c) + (radius - 1)
    return seed_col if (diag_index // 2) % 2 == 0 else alt_col


def _paint_diamond_stripes(
    out: np.ndarray,
    cavity: set[tuple[int, int]],
    boundary: set[tuple[int, int]],
    seed_col: int,
    alt_col: int,
    bg: int,
) -> bool:
    parsed = _diamond_ring_match(cavity, boundary)
    if parsed is None:
        return False

    center_r, center_c, radius = parsed
    for r, c in cavity:
        if int(out[r, c]) != bg:
            continue
        out[r, c] = _diamond_stripe_color(r, c, center_r, center_c, radius, seed_col, alt_col)
    return True


def _checkerboard_mismatch(cavity: set[tuple[int, int]], seed_col: int, alt_col: int) -> int:
    parsed = _diamond_center_radius(cavity)
    if parsed is None:
        return -1

    center_r, center_c, radius = parsed
    anchor_r, anchor_c = min(cavity)
    mismatch = 0
    for r, c in cavity:
        stripe_col = _diamond_stripe_color(r, c, center_r, center_c, radius, seed_col, alt_col)
        checker_col = seed_col if (r - anchor_r + c - anchor_c) % 2 == 0 else alt_col
        if stripe_col != checker_col:
            mismatch += 1
    return mismatch


def variant_metrics(grid: np.ndarray, out: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    painted = np.asarray(out, dtype=int)
    bg = get_bg(g)

    filled_diamonds = 0
    max_radius = 0
    max_checkerboard_mismatch = 0

    for cavity, boundary, bcols in _cavities(g, bg):
        if len(bcols) != 1:
            continue
        if not any(int(painted[r, c]) != int(g[r, c]) for r, c in cavity):
            continue

        parsed = _diamond_ring_match(cavity, boundary)
        if parsed is None:
            return {
                "filled_diamonds": -1,
                "max_radius": -1,
                "max_checkerboard_mismatch": -1,
            }

        alt_col = next(iter(bcols))
        cavity_cols = {int(painted[r, c]) for r, c in cavity}
        seed_candidates = cavity_cols - {alt_col}
        if len(seed_candidates) != 1:
            return {
                "filled_diamonds": -1,
                "max_radius": -1,
                "max_checkerboard_mismatch": -1,
            }

        seed_col = next(iter(seed_candidates))
        mismatch = _checkerboard_mismatch(cavity, seed_col, alt_col)
        if mismatch < 0:
            return {
                "filled_diamonds": -1,
                "max_radius": -1,
                "max_checkerboard_mismatch": -1,
            }

        filled_diamonds += 1
        max_radius = max(max_radius, parsed[2])
        max_checkerboard_mismatch = max(max_checkerboard_mismatch, mismatch)

    return {
        "filled_diamonds": filled_diamonds,
        "max_radius": max_radius,
        "max_checkerboard_mismatch": max_checkerboard_mismatch,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace each seed-linked drawing and repaint only the enclosed diamond cavities it owns.

    Input structure:
    - The background is the most frequent color.
    - Each actionable structure has a unique border seed pixel, an 8-connected line, and one or more single-color diamond rings attached to that line.
    - Each ring encloses a background diamond cavity; extra rings or disconnected line+ring networks may appear as distractors.

    Rule:
    - From each unique border seed, trace its whole 8-connected non-background component and infer the adjacent line color.
    - Among enclosed background regions, keep only single-boundary-color cavities whose traced boundary belongs to that component and is genuinely adjacent to the seed's line.
    - If such a cavity is a diamond, fill it with alternating diagonal stripes using the seed color and the ring color, with the stripes aligned to one axis of the local diamond.

    Key insight:
    - The alternation is tied to the diamond's own diagonal coordinate, not to a global checkerboard parity, so the fill pattern and the rotated frame geometry must be inferred together.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    cavities = _cavities(g, bg)

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])

        neigh_cols = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w:
                v = int(g[nr, nc])
                if v != bg:
                    neigh_cols.append(v)
        assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]

        traced = _nonbg_component8_from(g, (sr, sc), bg)

        for cavity, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue

            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            touches_line = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        touches_line = True
                        break
                if touches_line:
                    break
            if not touches_line:
                continue

            ring_col = next(iter(bcols))
            _paint_diamond_stripes(out, cavity, boundary, seed_col, ring_col, bg)

    return out


# --------------- generator (diamond-band fusion variant) ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (9, 30)
GRID_W_RANGE = (9, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 7
ATTACHED_DIAMOND_RADII = [3, 3, 3, 4]
DISTRACTOR_DIAMOND_RADII = [2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 60

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 60
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 400
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ADJ8_DIRS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _valid_radii(h: int, w: int, radii: list[int]) -> list[int]:
    max_r = (min(h, w) - 2 * FRAME_PLACEMENT_PAD - 1) // 2
    return [r for r in radii if r <= max_r]


def _diamond_ring_coords(center_r: int, center_c: int, radius: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    ring = []
    cavity = []
    for dr in range(-radius, radius + 1):
        span = radius - abs(dr)
        if span == 0:
            ring.append((center_r + dr, center_c))
        else:
            ring.append((center_r + dr, center_c - span))
            ring.append((center_r + dr, center_c + span))
        for dc in range(-(radius - 1), radius):
            if abs(dr) + abs(dc) < radius:
                cavity.append((center_r + dr, center_c + dc))
    return ring, cavity


def _diamond_ring_offsets(radius: int) -> list[tuple[int, int]]:
    offsets = []
    for dr in range(-radius, radius + 1):
        span = radius - abs(dr)
        if span == 0:
            offsets.append((dr, 0))
        else:
            offsets.append((dr, -span))
            offsets.append((dr, span))
    return offsets


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    line_col: int,
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if not _in_bounds(h, w, rr, cc) or canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}

    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = ddr == dr_bias or ddc == dc_bias
            dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))

        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue
            cur = (nr, nc)
            path.append(cur)
            path_set.add(cur)
            grid[nr, nc] = line_col
            placed = True
            break

        if not placed:
            if rng.random() < BIAS_FLIP_PROB:
                dr_bias *= -1
            if rng.random() < BIAS_FLIP_PROB:
                dc_bias *= -1
            break

    return path


def _candidate_centers_for_line_pixel(lp: tuple[int, int], radius: int) -> list[tuple[int, int]]:
    centers = set()
    for rr_off, cc_off in _diamond_ring_offsets(radius):
        for ar, ac in ADJ8_DIRS:
            ring_r = lp[0] + ar
            ring_c = lp[1] + ac
            centers.add((ring_r - rr_off, ring_c - cc_off))
    return list(centers)


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int] | None:
    h, w = canvas.shape
    valid_radii = _valid_radii(h, w, ATTACHED_DIAMOND_RADII)
    if not valid_radii:
        return None

    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        radius = rng.choice(valid_radii)
        lp = rng.choice(line_path)
        centers = _candidate_centers_for_line_pixel(lp, radius)
        rng.shuffle(centers)
        for center_r, center_c in centers:
            if not (pad + radius <= center_r < h - pad - radius and pad + radius <= center_c < w - pad - radius):
                continue
            ring, cavity = _diamond_ring_coords(center_r, center_c, radius)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity, radius
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray,
    frame_col: int,
    rng: random.Random,
    max_tries: int = DISTRACTOR_FRAME_TRIES,
) -> bool:
    h, w = canvas.shape
    valid_radii = _valid_radii(h, w, DISTRACTOR_DIAMOND_RADII)
    if not valid_radii:
        return False

    for _ in range(max_tries):
        radius = rng.choice(valid_radii)
        center_r = rng.randint(FRAME_PLACEMENT_PAD + radius, h - FRAME_PLACEMENT_PAD - radius - 1)
        center_c = rng.randint(FRAME_PLACEMENT_PAD + radius, w - FRAME_PLACEMENT_PAD - radius - 1)

        ok = True
        for rr in range(center_r - radius - 1, center_r + radius + 2):
            for cc in range(center_c - radius - 1, center_c + radius + 2):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        ring, _ = _diamond_ring_coords(center_r, center_c, radius)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True

    return False


def _try_place_distractor_network(canvas: np.ndarray, line_col: int, frame_colors_pool: list[int], rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue

        canvas[r, c] = line_col
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, (r, c), line_col, target_len, rng)
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue

        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached > 0:
            return True

        for pr, pc in path:
            canvas[pr, pc] = -1

    return False


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        seed = (0, 0)
        first = rng.choice([(0, 1), (1, 0)])
    elif corner_kind == "TR":
        seed = (0, w - 1)
        first = rng.choice([(0, w - 2), (1, w - 1)])
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        first = rng.choice([(h - 1, 1), (h - 2, 0)])
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        first = rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds: int, num_attached_target: int, rng: random.Random) -> list[int]:
    seed_order = list(range(num_seeds))
    rng.shuffle(seed_order)
    per_seed = [0] * num_seeds
    remaining = num_attached_target

    while remaining > 0 and any(per_seed[i] == 0 for i in range(num_seeds)):
        for i in seed_order:
            if per_seed[i] == 0 and remaining > 0:
                per_seed[i] += 1
                remaining -= 1

    while remaining > 0:
        per_seed[rng.choice(seed_order)] += 1
        remaining -= 1

    return per_seed


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    if not _valid_radii(h, w, ATTACHED_DIAMOND_RADII):
        return None

    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners_used = corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]] = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]

        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    attached_radii: list[int] = []
    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            frame_col = rng.choice(frame_colors_pool)
            placed = _try_attach_frame(canvas, path, line_col, frame_col, rng)
            if placed is not None:
                attached += 1
                attached_radii.append(placed[2])
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, min(DISTRACTOR_SOLO_MAX, num_distractors_target))):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    if not attached_radii or max(attached_radii) < 3:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    metrics = variant_metrics(grid, out)
    if metrics["filled_diamonds"] <= 0:
        return None
    if metrics["max_radius"] < 3:
        return None
    if metrics["max_checkerboard_mismatch"] < 4:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
