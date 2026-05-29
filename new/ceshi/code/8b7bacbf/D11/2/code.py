# Variant of 8b7bacbf
# Dimension: D11 — 干扰项策略 (obstacle)
# Change: 0-5个未连接到种子路径的干扰框架环，可能包含独立的线路+框架网络... → 干扰路径（同色线路不连通到种子）

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis for this D11 variant
# -----------------------------------
# a) 具体化与行为追踪
#    输入里仍然有 1-2 个边缘单像素 seed，每个 seed 连到一条 8 连通细线，再连到若干闭合框架环。
#    输出仍然是把“属于该 seed 的线路网络所接触到的环内部腔体”填成 seed 颜色。
#    变体的关键新增是：画面里会稳定出现至少一个“与某条 seed 线路同色、但 8 连通上完全断开”的
#    干扰路径网络；它通常也会挂着 1-3 个框架环。视觉上，人会看到同一种 line_color 在两个区域出现，
#    但只有和 seed 真正连通的那一簇会触发填充。
#    小例子：
#      原逻辑输入：seed(6) 接 line_color=2，右侧有一个 color=2 路径连到框架，框内被填 6。
#      变体输入：除了这条真正的 color=2 路径，左下角再放一条断开的 color=2 路径，也挂一个框架。
#      错误做法：只要看到任何 color=2 邻接框架，就把两个框都填 6。
#      正确做法：只能从 seed 出发沿 8 连通追到“属于该 seed 的 color=2 分量”，断开的 color=2
#      不能算，因此左下角框架保持背景。
# b) 复杂度校准
#    干扰网络不只是裸线条；它通常带至少一个腔体，这样“同色但不应填”的差异在 input/output 上可见。
#    干扰网络数量控制在 1-3 个，挂框 1-3 个，避免画面被无关线条淹没，同时足以反制颜色匹配型 solver。
# c) 隐式假设审查
#    原 solver 中最敏感的旧假设是：
#      - 用 seed 邻域多数色推 line_color：这个前提仍成立，因为 seed 仍只接自家线路。
#      - 只要 cavity 边界附近出现该 line_color 就够：在变体里不够，必须限定为“与 seed 连通的
#        line_color 分量”。否则全局同色干扰路径会误触发填充。
#      - 用 seed 可达的非背景组件 traced 判断框架归属：这个前提仍成立，但需要再叠加一个更细的
#        owned_line 分量，防止实现者把“颜色相同”误写成“连通归属相同”。
# d) 放置合法性
#    生成器保证：干扰路径从空白内部起步，路径像素对已有非背景像素保持 8 邻隔离；挂接框架时，框环和腔体
#    也必须落在空白区，并且框环只允许接触这条干扰路径本身。这样干扰网络和任何 seed 网络都保持断开。
# e) 视觉可观察性
#    解题者在 2-4 个样例里能直接看到：同一种线路颜色会在多个位置出现，但只有连到边缘 seed 的那一簇
#    周围框架会被填色。这个规律来自可见的拓扑连通性，而不是颜色编号或隐藏排序。
# f) 可视化整洁度
#    干扰网络仍采用细线 + 小矩形环，和主对象同语法；同时通过 1-cell 以上间隔避免不同网络糊成一团。
#    总框数与路径长度延续原题量级，不额外引入新形状。
# g) 颜色变化可推断性
#    颜色规则没变：腔体被填成对应 seed 的唯一颜色。新增干扰只复用已有 line_color，不新增难以归纳的
#    颜色映射；人类只需推断“同色不代表同属一条路径”。


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


def _color_component8_from(
    g: np.ndarray, starts: list[tuple[int, int]] | set[tuple[int, int]], color: int
) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [p for p in starts if int(g[p]) == color]
    vis = set(st)
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and int(g[nr, nc]) == color:
                vis.add((nr, nc))
                st.append((nr, nc))
    return vis


def _background_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
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
    cavities = []
    for region, boundary in regions:
        if not boundary:
            continue
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        bcols = {int(g[p]) for p in boundary}
        cavities.append((region, boundary, bcols))
    return cavities


def _seed_infos(g: np.ndarray, bg: int) -> list[dict]:
    h, w = g.shape
    infos = []
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

        line_starts = []
        for nr, nc in neighbors4(sr, sc):
            if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) == line_col:
                line_starts.append((nr, nc))
        assert line_starts, "seed pixel must touch its line color orthogonally"

        infos.append(
            {
                "seed_rc": (sr, sc),
                "seed_col": seed_col,
                "line_col": line_col,
                "owned_nonbg": _nonbg_component8_from(g, (sr, sc), bg),
                "owned_line": _color_component8_from(g, line_starts, line_col),
                "all_line": {(int(r), int(c)) for r, c in np.argwhere(g == line_col)},
            }
        )
    return infos


def _boundary_adjacent_to_pixels(
    boundary: set[tuple[int, int]], pixels: set[tuple[int, int]], h: int, w: int
) -> bool:
    if not pixels:
        return False
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in pixels:
                return True
    return False


def _solve_with_connectivity_mode(g: np.ndarray, require_connected_line: bool) -> np.ndarray:
    g = np.asarray(g, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cavities = _background_cavities(g, bg)

    for info in _seed_infos(g, bg):
        traced_nonbg = info["owned_nonbg"]
        line_pixels = info["owned_line"] if require_connected_line else info["all_line"]
        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            if require_connected_line and not (boundary & traced_nonbg):
                continue
            if not _boundary_adjacent_to_pixels(boundary, line_pixels, h, w):
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = info["seed_col"]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill cavities only when their frame is touched by the seed's own reachable line component.

    D11 variant note:
    - The same line color may appear in a disconnected distractor path elsewhere.
    - Therefore we must trace the seed-owned line-color component, not just match color globally.
    """
    return _solve_with_connectivity_mode(np.asarray(grid, dtype=int), require_connected_line=True)


# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_TRIES = 60
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_NETWORK_FRAMES = (1, 3)
DISTRACTOR_NETWORK_COUNT = (1, 3)
DISTRACTOR_SOLO_MAX = 1

GEN_RETRIES = 400
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dr == h - 1 or dc == 0 or dc == w - 1
            corner = (dr in (0, h - 1)) and (dc in (0, w - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray, ring: list[tuple[int, int]], cavity: list[tuple[int, int]], line_col: int, h: int, w: int
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _clear_cells(canvas: np.ndarray, pts: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    for r, c in pts:
        canvas[r, c] = -1


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
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(line_path)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_distractor_frame(
    canvas: np.ndarray, frame_col: int, rng: random.Random, max_tries: int = DISTRACTOR_FRAME_TRIES
) -> bool:
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 1 <= 1 or w - fw - 1 <= 1:
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
        ring, _ = _frame_ring_coords(r0, c0, ch, cw)
        for rr, cc in ring:
            canvas[rr, cc] = frame_col
        return True
    return False


def _path_is_isolated(canvas: np.ndarray, path: list[tuple[int, int]]) -> bool:
    h, w = canvas.shape
    path_set = set(path)
    for r, c in path:
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in path_set:
                continue
            if canvas[nr, nc] != -1:
                return False
    return True


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    frame_goal: int,
) -> int:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        if h < 6 or w < 6:
            return 0
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue

        canvas[r, c] = line_col
        target_len = rng.randint(*DISTRACTOR_LINE_LEN)
        path = _grow_line(canvas, (r, c), line_col, target_len, rng)
        if len(path) < 3 or not _path_is_isolated(canvas, path):
            _clear_cells(canvas, path)
            continue

        placed_cells: set[tuple[int, int]] = set(path)
        attached = 0
        attempts = max(frame_goal * 2, frame_goal + 1)
        for _ in range(attempts):
            if attached >= frame_goal:
                break
            frame_col = rng.choice(frame_colors_pool)
            attached_frame = _try_attach_frame(canvas, path, line_col, frame_col, rng)
            if attached_frame is None:
                continue
            ring, _ = attached_frame
            placed_cells.update(ring)
            attached += 1

        if attached >= frame_goal:
            return attached

        _clear_cells(canvas, placed_cells)
    return 0


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

    distractor_loops_target = rng.randint(1, min(4, TOTAL_LOOPS_MAX - num_seeds))
    num_attached_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX - distractor_loops_target)
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    target_networks = rng.randint(*DISTRACTOR_NETWORK_COUNT)
    placed_loops = 0
    placed_networks = 0
    while placed_loops < distractor_loops_target and placed_networks < target_networks:
        remaining = distractor_loops_target - placed_loops
        frame_goal = rng.randint(1, min(DISTRACTOR_NETWORK_FRAMES[1], remaining))
        gained = _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng, frame_goal)
        if gained > 0:
            placed_loops += gained
            placed_networks += 1
        else:
            break
    if placed_loops == 0:
        return None

    for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
        _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

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

    naive = _solve_with_connectivity_mode(grid, require_connected_line=False)
    if np.array_equal(naive, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
