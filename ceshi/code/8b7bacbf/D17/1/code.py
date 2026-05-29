# Variant of 8b7bacbf
# Dimension: D17 — 框架环形状 (obstacle)
# Change: 圆角矩形环——矩形边框去除四个角像素，腔体为内部矩形区域 → 菱形/旋转45度的框架环

"""
design_analysis:
  variant_goal:
    preserve:
      - 边缘唯一色 seed 通过 8 连通线连接到若干闭合框架
      - 只有与该 seed 所在线网相连的闭合腔体会被填成 seed 色
      - 非连接框架仍是干扰项
    change:
      - 框架不再是“矩形边框减四角”
      - 框架改为曼哈顿距离定义的菱形环: |dr|+|dc|=R
      - 腔体改为其内部菱形: |dr|+|dc|<R

  a_concretization_and_trace:
    input_object:
      - 每个框架由单色像素组成的 8 连通菱形轮廓构成
      - 腔体是背景色构成的 4 连通菱形区域
      - 线条只贴在菱形外缘, 与框架 8 邻接, 不直接贴到腔体
    tiny_example:
      original:
        - 矩形环的 cavity 是一个小矩形, boundary 也是单色
      variant:
        - 取中心 (5,5), 半径 R=3
        - ring = 所有 |dr|+|dc|=3 的点, 共 12 个
        - cavity = 所有 |dr|+|dc|<3 的点, 共 13 个, 呈菱形
        - 当 line 像素贴在 ring 外缘时, solver 仍会把内部背景连通块识别成可填腔体
    flow_difference:
      - solver 的“找闭合背景腔体并 flood-fill”逻辑本身是形状无关的
      - 真正变化的是 generator 的几何构造: 原先按矩形 bbox 画环, 现在按曼哈顿半径画菱形环

  b_complexity_calibration:
    - 半径限制在 2-4, 视觉上明确是对角线封闭区域, 不会退化成单点或过大杂乱图形
    - 每题仍保留 1-2 个 seed、多个框架颜色和若干干扰框架, 规则辨识难度与原题接近
    - 多个菱形可重复使用同一 frame_color, 保留“颜色不是唯一索引”的原题特征

  c_implicit_assumption_audit_main:
    - _seed_pixels: 只要求 seed 是边界唯一色单像素, 与框架形状无关, 仍成立
    - _nonbg_component8_from: 只要求线与框架 8 连通; 菱形环像素间本来就是 8 连通, 仍成立
    - 背景分量按 4 连通分解: 菱形腔体在 4 连通下仍是一个整体, 且被 ring 阻断, 仍成立
    - boundary 用 neighbors4 采样: 菱形腔体每个边界背景像素都能在 4 邻域碰到 ring, 仍成立
    - len(bcols)==1: 仍要求每个框架边界单色, generator 会继续保证
    - traced boundary 需 8 邻接 line_col: 仍可区分“贴在线上的框架”和“只是碰到别的非背景结构的区域”
    conclusion:
      - main 的核心推理不需要按“矩形”特判, 只需继续保证 cavity 真正被单色 ring 封闭即可

  d_placement_legality:
    - 菱形 bbox 为 (2R+1)x(2R+1), 需至少离边界 1 格
    - ring 与 cavity 不能覆盖已有像素
    - cavity 任何像素都不能 8 邻接 line_col, 防止“线穿进腔体边上”导致规则模糊
    - ring 不能 8 邻接外来颜色, 以免不同结构粘连
    - attached frame 必须至少有一个 ring 像素 8 邻接 line_col

  e_visual_observability:
    - 与原题相比, 可观察差异集中在“闭合区域边缘沿对角方向扩张”
    - radius 2-4 的菱形腔体在 2-4 个样例中足够显眼, 人类能直接看出不再是矩形框

  f_visual_cleanliness:
    - 通过空白 margin 和 foreign-color adjacency 检查避免多个框架挤成一团
    - 仍保持 1px 宽线条和单层框架, 不叠加额外标记

  g_color_inference:
    - 颜色规则不变: 被填颜色来自 border seed, 不是 frame_color
    - 颜色种类仍受原题限制, 结构线索清晰, 不需要依赖颜色值大小或排序

  post_change_main_assumptions:
    - 每个可填腔体必须是被单色 ring 封闭的 4 连通背景块
    - ring 与其所属 line 至少在一个位置 8 邻接
    - seed 邻接到的主要非背景颜色就是其 line_color
    - generator 已显式构造并校验这些条件
"""

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


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
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, boundary) in enumerate(regions):
        if i in outside_idxs or not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((reg, boundary, bcols))
    return cavities


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each enclosed cavity that belongs to the 8-connected drawing reached from a unique border seed.

    The D17 variant changes only the visual geometry of the loops: generator now produces diamond rings
    instead of rounded rectangles. The solver remains shape-agnostic and operates on enclosed background
    regions, which still correctly identifies the new diamond cavities.
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

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue

            ok = False
            for r, c in boundary_in_traced:
                for nr, nc in neighbors8(r, c):
                    if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                        ok = True
                        break
                if ok:
                    break
            if not ok:
                continue

            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = seed_col

    return out


# --------------- generator (diamond-frame variant) ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 7
ATTACHED_DIAMOND_RADII = [2, 2, 2, 3, 3, 4]
DISTRACTOR_DIAMOND_RADII = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50
DISTRACTOR_FRAME_TRIES = 50

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

GEN_RETRIES = 300
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


def _diamond_radius_from_cavity(cav: set[tuple[int, int]]) -> int | None:
    rows: dict[int, list[int]] = {}
    for r, c in cav:
        rows.setdefault(r, []).append(c)

    row_keys = sorted(rows)
    lengths = []
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
    return (peak + 1) // 2


def _grow_line(grid: np.ndarray, start: tuple[int, int], line_col: int, target_len: int, rng: random.Random) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start
    path_set = {start}
    while len(path) < target_len:
        dirs = []
        for ddr, ddc in DIAG_DIRS:
            aligned = (ddr == dr_bias or ddc == dc_bias)
            w_d = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            w_d = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * w_d)
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
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
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
            return ring, cavity
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
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
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

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
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

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    placed_distractors = 0
    if rng.random() < DISTRACTOR_NETWORK_PROB:
        if _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng):
            placed_distractors += 1
    else:
        for _ in range(rng.randint(0, min(DISTRACTOR_SOLO_MAX, num_distractors_target))):
            if _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng):
                placed_distractors += 1

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, _, _, _ in lines_info:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    # Variant-specific guard:
    # 1. every single-color enclosed cavity should match the intended diamond profile
    # 2. at least one filled cavity should be a visible diamond with radius >= 2
    variant_ok = False
    for cav, boundary, bcols in _cavities(grid, bg):
        if len(bcols) != 1:
            continue
        radius = _diamond_radius_from_cavity(cav)
        if radius is None:
            return None
        changed = any(out[r, c] != grid[r, c] for r, c in cav)
        if changed and radius < 2:
            return None
        if changed and radius >= 2:
            variant_ok = True
    if not variant_ok:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
