# Combo variant of 8b7bacbf (index 15)
# combo_spec: D10/0+D17/1
# Name: 菱形框架+不规则腔体
#
# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 原题里，seed 在边界，seed 接一条 8 连通细线，细线再贴着若干单色闭环；solver 找到不触边的背景腔体并填成 seed 色。
#    - 这次融合后，框架不再是圆角矩形，而是“菱形系”闭环；腔体也不再是规则菱形，而是带削肩/偏心/截边的
#      不规则菱形区域。一个 5x5 小例子可以是 rows = [00100, 01110, 11110, 01110, 00100]，它仍然有尖顶尖底和
#      对角向扩张感，但中段不再是标准对称菱形。对该腔体取 4 邻外壳层后，就得到对应的单色菱形风格框架环。
#    - solver 的执行链保持不变：先追踪 seed 所在的非背景 8 连通网络，再枚举所有不触边的 4 连通背景区域，
#      只要该区域的 4 邻边界是单色、且边界属于 seed 追踪到的网络并真正贴到 line_color，就整体 flood-fill。
#
# b) 复杂度校准
#    - 不把腔体做成任意 blob，而是只从少量“菱形风格模板”中采样，再做旋转/镜像；这样样本里会反复出现
#      “尖顶尖底 + 斜向扩张 + 一处不规则”的视觉规律，人类能从 2-4 个样例中归纳。
#    - 模板只用 5x5 和 7x6/6x7 量级，既足够显眼，又不会把网格挤满。
#
# c) 隐式假设审查
#    - 原版用“背景角点属于 outside”判断外部区域，对自由形状较脆弱；这里改成“任一背景分量只要触边就视为 outside”，
#      以适配不规则菱形和更复杂的局部凹凸。
#    - _seed_pixels、_nonbg_component8_from、单色边界判定、line_color 邻接判定都不依赖矩形假设，融合后仍成立。
#    - 新增假设是：被填腔体必须仍是 4 连通、其边界必须完全由单色壳层围住。generator 会显式校验。
#
# d) 放置合法性
#    - cavity mask 与 ring 都必须落在空白处且不越界。
#    - cavity 不能 8 邻接 line_color，避免线直接贴到腔体内部边。
#    - ring 不能 8 邻接外来颜色；attached frame 至少有一个 ring 像素 8 邻接到本 seed 的 line。
#    - 多 seed 情况下，最终还要校验 seed 对应的非背景 8 连通分量彼此分离。
#
# e) 视觉可观察性
#    - 规则变化集中在“被填区域的几何外观”：输入里能直接看到菱形风格闭环，输出里则把这些不规则菱形腔体整体染成 seed 色。
#    - 颜色规则没有新增任意映射，被填颜色仍来自边界 seed；人类无需利用颜色编号等不可视因素。
#
# f) 可视化整洁度
#    - 通过 1 格 margin、foreign-color adjacency 检查和较保守的模板尺寸，避免不同框架挤在一起。
#    - line 保持 1px 宽，框架只保留单层壳，不叠加额外标记。
#
# g) 变化的可推断性
#    - 生成后强制要求：所有可填腔体都必须是“非规则菱形但仍具菱形剖面”的区域，不能退化成原题矩形，也不能退化成
#      D17 的标准规则菱形。这样 input->output 的关键变化会稳定出现。

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
    stack = [start]
    vis = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                stack.append((nr, nc))
    return vis


def _bg_regions_with_boundaries(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], bool]]:
    h, w = g.shape
    regions = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, _touches_border(pts, h, w)))
    return regions


def _line_color_for_seed(g: np.ndarray, sr: int, sc: int, bg: int) -> int:
    h, w = g.shape
    neigh_cols = []
    for nr, nc in neighbors4(sr, sc):
        if 0 <= nr < h and 0 <= nc < w:
            v = int(g[nr, nc])
            if v != bg:
                neigh_cols.append(v)
    assert neigh_cols, "seed pixel must have at least one non-bg neighbor"
    return Counter(neigh_cols).most_common(1)[0][0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flood-fill the enclosed diamond-style cavities that belong to a seeded line network.

    Input structure:
    - The background is the most frequent color.
    - One or two unique single-pixel seed colors appear on the grid border.
    - Each seed touches a 1px-wide 8-connected line, and that line may connect to single-color frame loops.
    - In this combo, those loops are 45-degree diamond-flavored shells whose enclosed bg regions are irregular
      diamond-like shapes rather than plain rectangles or perfectly regular diamonds.

    Rule:
    - For each seed, trace the full non-background 8-connected component that starts at that seed.
    - Find every 4-connected background region that does not touch the grid border.
    - If such a region has a single-color non-background boundary, and that boundary belongs to the traced component,
      and at least one traced boundary pixel is 8-neighbor to the seed's line color, fill the whole region with the seed color.

    Key insight:
    - The geometry changed, but the decisive invariant did not: a fillable target is still a closed 4-connected background
      cavity with a single-color shell attached to exactly one seed's drawing. The solver can stay shape-agnostic as long
      as the generator guarantees these cavity and attachment conditions for the new irregular diamond frames.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    cavities = []
    for region, boundary, touches_border in _bg_regions_with_boundaries(g, bg):
        if touches_border or not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, bcols))

    for sr, sc in _seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        line_col = _line_color_for_seed(g, sr, sc, bg)
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


# ========================== CFG ==========================
GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 5
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 7
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 120
DISTRACTOR_FRAME_TRIES = 120

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_PROB = 0.55
DISTRACTOR_NETWORK_TRIES = 60
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

GEN_RETRIES = 700
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


IRREGULAR_DIAMOND_TEMPLATE_BANK = [
    _parse_mask(["00100", "01110", "01111", "01110", "00100"]),
    _parse_mask(["00100", "01110", "11110", "01110", "00100"]),
    _parse_mask(["00100", "01100", "11110", "01110", "00100"]),
    _parse_mask(["00100", "01110", "11110", "01100", "00100"]),
    _parse_mask(["001000", "011100", "111110", "111111", "111110", "011100", "001000"]),
    _parse_mask(["001000", "001100", "011110", "111111", "111110", "011100", "001000"]),
    _parse_mask(["001000", "011100", "111110", "111111", "011110", "001100", "000100"]),
    _parse_mask(["001000", "001100", "011110", "111111", "011110", "001100", "000100"]),
]
IRREGULAR_DIAMOND_TEMPLATE_WEIGHTS = [3, 3, 2, 2, 2, 2, 2, 2]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _irregular_frame_coords(r0: int, c0: int, cavity_mask: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    mh, mw = cavity_mask.shape
    ring = set()
    cavity = []
    for rr in range(mh):
        for cc in range(mw):
            if not cavity_mask[rr, cc]:
                continue
            cavity.append((r0 + 1 + rr, c0 + 1 + cc))
            for nr, nc in neighbors4(rr + 1, cc + 1):
                if 1 <= nr < mh + 1 and 1 <= nc < mw + 1 and cavity_mask[nr - 1, nc - 1]:
                    continue
                ring.add((r0 + nr, c0 + nc))
    return sorted(ring), cavity


def _mask_has_single_cavity(cavity_mask: np.ndarray) -> bool:
    ring, cavity = _irregular_frame_coords(0, 0, cavity_mask)
    local_h, local_w = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
    local = np.zeros((local_h, local_w), dtype=int)
    for rr, cc in ring:
        local[rr, cc] = 1
    inner = []
    for pts in _components_mask(local == 0, conn8=False):
        if not _touches_border(pts, local_h, local_w):
            inner.append(set(pts))
    return len(inner) == 1 and inner[0] == set(cavity)


def _diamond_radius_from_region(region: set[tuple[int, int]]) -> int | None:
    rows: dict[int, list[int]] = {}
    for r, c in region:
        rows.setdefault(r, []).append(c)

    row_keys = sorted(rows)
    lengths = []
    centers = []
    for r in row_keys:
        cols = sorted(rows[r])
        if cols != list(range(cols[0], cols[-1] + 1)):
            return None
        lengths.append(len(cols))
        centers.append(cols[0] + cols[-1])

    peak = max(lengths)
    peak_idx = lengths.index(peak)
    if peak_idx == 0 or peak_idx == len(lengths) - 1 or lengths.count(peak) != 1 or peak % 2 == 0:
        return None
    if lengths[: peak_idx + 1] != list(range(1, peak + 1, 2)):
        return None
    if lengths[peak_idx:] != list(range(peak, 0, -2)):
        return None
    if len(set(centers)) != 1:
        return None
    return (peak + 1) // 2


def _is_irregular_diamond_region(region: set[tuple[int, int]]) -> bool:
    rows: dict[int, list[int]] = {}
    for r, c in region:
        rows.setdefault(r, []).append(c)

    row_keys = sorted(rows)
    widths = []
    centers = []
    for r in row_keys:
        cols = sorted(rows[r])
        if cols != list(range(cols[0], cols[-1] + 1)):
            return False
        widths.append(len(cols))
        centers.append(cols[0] + cols[-1])

    if len(widths) < 5 or widths[0] != 1 or widths[-1] != 1:
        return False

    peak = max(widths)
    peak_idx = widths.index(peak)
    if peak_idx == 0 or peak_idx == len(widths) - 1 or widths.count(peak) != 1:
        return False

    if not all(widths[i] < widths[i + 1] for i in range(peak_idx)):
        return False
    if not all(widths[i] > widths[i + 1] for i in range(peak_idx, len(widths) - 1)):
        return False
    if not all(1 <= widths[i + 1] - widths[i] <= 2 for i in range(peak_idx)):
        return False
    if not all(1 <= widths[i] - widths[i + 1] <= 2 for i in range(peak_idx, len(widths) - 1)):
        return False
    if not all(abs(centers[i + 1] - centers[i]) <= 1 for i in range(len(centers) - 1)):
        return False

    return _diamond_radius_from_region(region) is None


def _random_cavity_mask(rng: random.Random) -> np.ndarray:
    for _ in range(64):
        mask = rng.choices(IRREGULAR_DIAMOND_TEMPLATE_BANK, weights=IRREGULAR_DIAMOND_TEMPLATE_WEIGHTS, k=1)[0].copy()
        if rng.random() < 0.5:
            mask = np.fliplr(mask)
        if rng.random() < 0.5:
            mask = np.flipud(mask)
        rot = rng.randrange(4)
        if rot:
            mask = np.rot90(mask, rot)
        mask = _trim_mask(mask)
        if not _mask_has_single_cavity(mask):
            continue
        region = {(int(r), int(c)) for r, c in np.argwhere(mask)}
        if _is_irregular_diamond_region(region):
            return mask
    raise RuntimeError("failed to sample valid irregular diamond cavity mask")


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
            aligned = ddr == dr_bias or ddc == dc_bias
            weight = DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
        for ddr, ddc in ORTHO_DIRS:
            aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
            weight = ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER
            dirs.extend([(ddr, ddc)] * weight)
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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
        lp = rng.choice(line_path)
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _irregular_frame_coords(r0, c0, cavity_mask)
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
    for _ in range(max_tries):
        cavity_mask = _random_cavity_mask(rng)
        fh, fw = cavity_mask.shape[0] + 2, cavity_mask.shape[1] + 2
        if h - fh - 1 < 1 or w - fw - 1 < 1:
            return False
        r0 = rng.randint(1, h - fh - 1)
        c0 = rng.randint(1, w - fw - 1)
        ok = True
        for rr in range(r0 - 1, r0 + fh + 1):
            for cc in range(c0 - 1, c0 + fw + 1):
                if _in_bounds(h, w, rr, cc) and canvas[rr, cc] != -1:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue
        ring, _ = _irregular_frame_coords(r0, c0, cavity_mask)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
) -> bool:
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


def _seed_components_are_disjoint(
    grid: np.ndarray,
    lines_info: list[tuple[int, tuple[int, int], int, list[tuple[int, int]]]],
    bg: int,
) -> bool:
    traces = []
    seed_positions = [seed_rc for _, seed_rc, _, _ in lines_info]
    for _, seed_rc, _, _ in lines_info:
        traced = _nonbg_component8_from(grid, seed_rc, bg)
        for other_seed in seed_positions:
            if other_seed != seed_rc and other_seed in traced:
                return False
        traces.append(traced)
    for i in range(len(traces)):
        for j in range(i + 1, len(traces)):
            if traces[i] & traces[j]:
                return False
    return True


def _fillable_cavity_profile(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    seed_infos = []
    for sr, sc in _seed_pixels(grid, bg):
        traced = _nonbg_component8_from(grid, (sr, sc), bg)
        line_col = _line_color_for_seed(grid, sr, sc, bg)
        seed_infos.append((traced, line_col))

    irregular = 0
    perfect = 0
    other = 0
    for region, boundary, touches_border in _bg_regions_with_boundaries(grid, bg):
        if touches_border or not boundary:
            continue
        if len({int(grid[p]) for p in boundary}) != 1:
            continue
        fillable = False
        for traced, line_col in seed_infos:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            for r, c in boundary_in_traced:
                if any(
                    0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and int(grid[nr, nc]) == line_col
                    for nr, nc in neighbors8(r, c)
                ):
                    fillable = True
                    break
            if fillable:
                break
        if not fillable:
            continue

        if _diamond_radius_from_region(region) is not None:
            perfect += 1
        elif _is_irregular_diamond_region(region):
            irregular += 1
        else:
            other += 1
    return irregular, perfect, other


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
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
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
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

    if not _seed_components_are_disjoint(grid, lines_info, bg):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    irregular_fillable, perfect_fillable, other_fillable = _fillable_cavity_profile(grid, bg)
    if irregular_fillable == 0 or perfect_fillable != 0 or other_fillable != 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
