# Variant of 8b7bacbf
# Dimension: D1 — 路径追踪连通性 (path)
# Change: 8连通追踪（path_tracing）——从边缘种子像素出发，沿8连通（含对角线）的非背景像素追踪出完整线路组件... → 混合连通追踪（线用8连通，腔体判定用8连通）

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


# ---------------------------------------------------------------------------
# Design analysis for the variant
# ---------------------------------------------------------------------------
# spec_variant:
#   visual_core:
#     - 线路仍然是从边缘唯一色种子出发的 8 连通细线。
#     - 被填充对象改成“闭角矩形环”，因为原题的缺角环在 8 连通背景下会漏到外部，
#       不再能形成腔体。
#     - 至少一个与线路相连的环内部含有同色“对角隔墙”。这道隔墙从环的一侧斜接到
#       对侧，4 连通背景下像墙，8 连通背景下则会在隔墙两侧通过对角缝连通。
#   concrete_behavior:
#     - 原逻辑：背景按 4 连通分量分腔体，隔墙会把腔体分成两块；只有靠近线路接触边
#       的那块会被视为与线路相关并填色。
#     - 变体逻辑：背景按 8 连通分量分腔体，隔墙两侧通过对角相邻背景像素合并为一个
#       更大的腔体；因为这个大腔体的边界中包含靠线路的一侧，所以整块都被填色。
#   tiny_trace:
#     - 一个闭角矩形环内部若有对角链 `# . / . #` 形式的同色墙：
#       4 连通背景把左袋和右袋分开；8 连通背景允许两袋经对角背景像素互通。
#     - 若线路只接触环的左边，那么 4 连通下只会填左袋；8 连通下左袋+右袋整体填。
#   hidden_assumptions_checked:
#     - 原 `bg_comps = ... conn8=False` 直接依赖“腔体用 4 连通”。这里必须改成 8。
#     - 原边界扫描用 `neighbors4`，默认“背景分量的边界由 4 邻接定义”。
#       既然腔体改成 8 连通背景，边界也必须按 8 邻接收集，否则对角围挡会漏记边界。
#     - 原 generator 的缺角环恰好依赖 4 连通背景才闭合；在新规则下会退化成开口，
#       所以必须把环改成闭角矩形。
#     - 原 solver 的“边界中有像素 8 邻接到 line_color”这一筛选仍然成立，但 generator
#       必须保证线路只接触环的一侧，否则 4/8 两种规则会得到相同填充结果。
#   placement_guarantees:
#     - 闭角矩形环与内部对角隔墙全部使用同一种 frame_color。
#     - 线路只允许贴住环的单侧外边；腔体内部背景不直接接触线路。
#     - generator 用新旧两版 solver 对同一输入做差分，强制至少出现一次
#       “8 连通腔体填得比 4 连通更多”的情况，避免退化回原任务。


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
    st = [start]
    vis = {start}
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) != bg:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _solve_with_bg_connectivity(grid: np.ndarray, bg_conn8: bool) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    boundary_neighbors = neighbors8 if bg_conn8 else neighbors4
    bg_comps = _components_mask(g == bg, conn8=bg_conn8)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        boundary = set()
        for r, c in region:
            for nr, nc in boundary_neighbors(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for region, boundary in regions:
        if not boundary:
            continue
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, bcols))

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


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_with_bg_connectivity(grid, bg_conn8=True)


# --------------- generator ---------------

GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_CHOICES = [1, 2, 2]
MIN_LINE_LEN = 6
LINE_LEN_RANGE = (10, 36)

FRAME_COLORS_MAX = 4
FRAME_PAD = 1
ATTACH_TRIES = 70
DISTRACTOR_TRIES = 60
GEN_RETRIES = 500

VARIANT_H_CHOICES = [3, 4, 5]
VARIANT_W_CHOICES = [5, 6, 7]
PLAIN_H_CHOICES = [2, 3, 4]
PLAIN_W_CHOICES = [2, 3, 4, 5]
EXTRA_ATTACHED_MAX = 2
DISTRACTOR_MAX = 2

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("left", "right", "top", "bottom")


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _rect_ring_coords(r0: int, c0: int, ch: int, cw: int) -> list[tuple[int, int]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            if dr == 0 or dc == 0 or dr == fh - 1 or dc == fw - 1:
                ring.append((r0 + dr, c0 + dc))
    return ring


def _variant_divider_coords(r0: int, c0: int, ch: int, cw: int, side: str) -> list[tuple[int, int]]:
    divider = []
    if side in ("left", "right"):
        base_c = c0 + 1 + max(1, cw // 2 - 1)
        for i in range(ch):
            divider.append((r0 + 1 + i, base_c + (i % 2)))
    else:
        base_r = r0 + 1 + max(1, ch // 2 - 1)
        for i in range(cw):
            divider.append((base_r + (i % 2), c0 + 1 + i))
    return divider


def _touch_side_cells(r0: int, c0: int, ch: int, cw: int, side: str) -> set[tuple[int, int]]:
    if side == "left":
        return {(r0 + dr, c0) for dr in range(1, ch + 1)}
    if side == "right":
        return {(r0 + dr, c0 + cw + 1) for dr in range(1, ch + 1)}
    if side == "top":
        return {(r0, c0 + dc) for dc in range(1, cw + 1)}
    return {(r0 + ch + 1, c0 + dc) for dc in range(1, cw + 1)}


def _chamber_cells(
    r0: int,
    c0: int,
    ch: int,
    cw: int,
    side: str,
    partitioned: bool,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]], set[tuple[int, int]]]:
    ring = _rect_ring_coords(r0, c0, ch, cw)
    divider = _variant_divider_coords(r0, c0, ch, cw, side) if partitioned else []
    wall_set = set(ring) | set(divider)
    cavity_bg = []
    for rr in range(r0 + 1, r0 + ch + 1):
        for cc in range(c0 + 1, c0 + cw + 1):
            if (rr, cc) not in wall_set:
                cavity_bg.append((rr, cc))
    return list(wall_set), cavity_bg, _touch_side_cells(r0, c0, ch, cw, side)


def _fits_bbox(h: int, w: int, r0: int, c0: int, ch: int, cw: int) -> bool:
    fh, fw = ch + 2, cw + 2
    return r0 >= FRAME_PAD and c0 >= FRAME_PAD and r0 + fh <= h - FRAME_PAD and c0 + fw <= w - FRAME_PAD


def _adjacent_to_color(canvas: np.ndarray, pts: list[tuple[int, int]], color: int) -> set[tuple[int, int]]:
    h, w = canvas.shape
    out = set()
    for r, c in pts:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == color:
                out.add((r, c))
                break
    return out


def _validate_chamber_placement(
    canvas: np.ndarray,
    wall: list[tuple[int, int]],
    cavity_bg: list[tuple[int, int]],
    line_col: int,
    touch_side: set[tuple[int, int]],
) -> bool:
    h, w = canvas.shape
    for rr, cc in wall:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity_bg:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity_bg:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and int(canvas[nr, nc]) == line_col:
                return False

    wall_set = set(wall)
    for rr, cc in wall:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in wall_set:
                continue
            v = int(canvas[nr, nc])
            if v != -1 and v != line_col:
                return False

    touching = _adjacent_to_color(canvas, wall, line_col)
    if not touching:
        return False
    if not touching.issubset(touch_side):
        return False
    return True


def _place_wall(canvas: np.ndarray, wall: list[tuple[int, int]], frame_col: int) -> None:
    for rr, cc in wall:
        canvas[rr, cc] = frame_col


def _side_bbox(lp: tuple[int, int], side: str, ch: int, cw: int, offset: int) -> tuple[int, int]:
    if side == "left":
        return lp[0] - offset, lp[1] + 1
    if side == "right":
        return lp[0] - offset, lp[1] - (cw + 2)
    if side == "top":
        return lp[0] + 1, lp[1] - offset
    return lp[0] - (ch + 2), lp[1] - offset


def _try_attach_chamber(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    line_col: int,
    frame_col: int,
    rng: random.Random,
    *,
    partitioned: bool,
    max_tries: int = ATTACH_TRIES,
) -> bool:
    h, w = canvas.shape
    h_choices = VARIANT_H_CHOICES if partitioned else PLAIN_H_CHOICES
    w_choices = VARIANT_W_CHOICES if partitioned else PLAIN_W_CHOICES

    for _ in range(max_tries):
        ch = rng.choice(h_choices)
        cw = rng.choice(w_choices)
        lp = rng.choice(line_path)
        sides = list(SIDES)
        rng.shuffle(sides)
        for side in sides:
            offset = rng.randint(1, ch if side in ("left", "right") else cw)
            r0, c0 = _side_bbox(lp, side, ch, cw, offset)
            if not _fits_bbox(h, w, r0, c0, ch, cw):
                continue
            wall, cavity_bg, touch_side = _chamber_cells(r0, c0, ch, cw, side, partitioned)
            if not _validate_chamber_placement(canvas, wall, cavity_bg, line_col, touch_side):
                continue
            _place_wall(canvas, wall, frame_col)
            return True
    return False


def _try_place_distractor_chamber(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_TRIES):
        ch = rng.choice(PLAIN_H_CHOICES)
        cw = rng.choice(PLAIN_W_CHOICES)
        side = rng.choice(SIDES)
        fh, fw = ch + 2, cw + 2
        r0 = rng.randint(FRAME_PAD, h - fh - FRAME_PAD)
        c0 = rng.randint(FRAME_PAD, w - fw - FRAME_PAD)
        wall, cavity_bg, _ = _chamber_cells(r0, c0, ch, cw, side, partitioned=False)

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
        for rr, cc in cavity_bg:
            if canvas[rr, cc] != -1:
                ok = False
                break
        if not ok:
            continue
        _place_wall(canvas, wall, frame_col)
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


def _corner_config(corner_kind: str, h: int, w: int, rng: random.Random) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), rng.choice([(0, 1), (1, 0)])
    if corner_kind == "TR":
        return (0, w - 1), rng.choice([(0, w - 2), (1, w - 1)])
    if corner_kind == "BL":
        return (h - 1, 0), rng.choice([(h - 1, 1), (h - 2, 0)])
    if corner_kind == "BR":
        return (h - 1, w - 1), rng.choice([(h - 1, w - 2), (h - 2, w - 1)])
    raise ValueError(corner_kind)


def _variant_delta(grid: np.ndarray) -> int:
    old_out = _solve_with_bg_connectivity(grid, bg_conn8=False)
    new_out = main(grid)
    return int(np.count_nonzero(old_out != new_out))


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_CHOICES)
    if num_seeds * 2 > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners = corners[:num_seeds]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners[i], h, w, rng)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w + 6))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], line_colors[i], path))

    variant_hits = 0
    for _, line_col, path in lines_info:
        frame_col = rng.choice(frame_colors_pool)
        if not _try_attach_chamber(canvas, path, line_col, frame_col, rng, partitioned=True):
            return None
        variant_hits += 1

    for _, line_col, path in lines_info:
        extra = rng.randint(0, EXTRA_ATTACHED_MAX)
        for _ in range(extra):
            frame_col = rng.choice(frame_colors_pool)
            _try_attach_chamber(canvas, path, line_col, frame_col, rng, partitioned=False)

    for _ in range(rng.randint(0, DISTRACTOR_MAX)):
        _try_place_distractor_chamber(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col in seed_colors:
        if int((grid == seed_col).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    if variant_hits == 0:
        return None
    if _variant_delta(grid) <= 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
