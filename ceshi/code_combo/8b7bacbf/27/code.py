# Combo variant of 8b7bacbf (index 27)
# combo_spec: D1/2+D11/2
# Name: 有限步追踪+干扰假路径

from __future__ import annotations

from collections import deque
import os
import random
import sys

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../.."))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)

from arc_utils import components_mask, get_bg, neighbors4, neighbors8

# Design analysis
# ---------------
# a) 具体化与行为追踪
#    画面里每个 seed 仍是边角上的唯一色单像素，但它现在正交连着两条分支：
#    一条是通往内部的 8 连通 line，另一条是沿边框延伸的直条 meter。meter 长度 N 编码可追踪步数上限。
#    同时，网格中还会放入至少一个“复用某条真实 line_color、但与 seed 完全断开的假路径网络”，它也挂着闭合框。
#    原题会从 seed 追完整个非背景连通块，因而同一路径上的远框都会填；本融合变体只能填 N 步内真正可达的框。
#    另外，若有人只按全局同色 line_color 匹配，假路径上的框也会被误填；正确规则必须同时满足“步数内可达”
#    和“与 seed 真正 8 连通”这两个条件。
#    小例子：seed 在左上角，meter 长度 4；真路径前段挂一个近框，后段第 8 步附近挂一个远框；右侧另有一条断开的
#    同色假路径也挂一框。输出里只填近框：远框因超步数不填，假框因不连通不填。
# b) 复杂度校准
#    meter 只取 2-5 这几个小档位，便于肉眼计数；每个 seed 至少同时出现 1 个近框和 1 个远框，保证“有限步”
#    不是偶然。假路径至少带 1 个框，且复用已有真实 line_color，保证“同色但不连通”的陷阱在 I/O 中可见。
# c) 隐式假设审查
#    原 solver 默认 seed 只接一条非背景分支，这里已失效：必须先区分 meter 和 line。原 solver 追完整个非背景
#    连通块也已失效：这里必须做带上限的 BFS。原 solver 只要 cavity 边界接触 traced 组件就填，也不够精确：
#    这里还要确认边界接触的是 seed 自己那条 connected line_color，而不是别处断开的同色路径。
#    新 solver 的额外假设是：seed 位于角上；meter 是 seed 邻接分支里更长的那条边框直条；真实 line 在离开 seed 后
#    很快转入内部；generator 会保证这些前提成立。
# d) 放置合法性
#    meter 和 line 使用角点两条可用正交方向，互不重叠。line 生长时禁止与已有非背景像素发生 8 邻接粘连。
#    框环放置要求：环和腔体全为空、腔体不贴目标路径、环只允许贴它所属的那条路径，因此不会误连到其他网络。
#    假路径从内部空区起步，并先验证整条 path 与已有对象保持 8 邻隔离，再挂框，保证它确实断开。
# e) 视觉可观察性
#    人类可以直接看到三个重复现象：边框直条长度可数；同一路径上有近框与远框且输出不同；同色细线在别处再次出现，
#    但其框不填。规则由空间拓扑和距离共同决定，而不是由颜色编号、排序或隐式元数据决定。
# f) 可视化整洁度
#    所有对象仍沿用“边角 seed + 细线 + 小矩形框”的原题语法，没有引入新图元。假路径网络数量控制为 1 个主网络，
#    其框数也很少，避免与主路径堆叠。框与路径之间保留至少 1 格缓冲，减少糊团。
# g) 变化的可推断性
#    输出颜色规则没有变化：被填腔体仍填成对应 seed 的唯一色。新增变化只体现在“哪些框会被填”，并且每个条件都
#    有可见线索：meter 编码步数，路径连通性由像素接触关系直接呈现，同色假路径用重复 line_color 强化这一点。


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


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


def _border_run_length(g: np.ndarray, seed: tuple[int, int], nbr: tuple[int, int], bg: int) -> int:
    h, w = g.shape
    sr, sc = seed
    nr, nc = nbr
    col = int(g[nr, nc])
    if col == bg:
        return 0
    dr = nr - sr
    dc = nc - sc
    run = 0
    r, c = nr, nc
    while _in_bounds(h, w, r, c) and int(g[r, c]) == col and (r == 0 or c == 0 or r == h - 1 or c == w - 1):
        run += 1
        r += dr
        c += dc
    return run


def _seed_info(g: np.ndarray, seed: tuple[int, int], bg: int) -> dict[str, object]:
    h, w = g.shape
    sr, sc = seed
    nonbg_neighbors = []
    for nr, nc in neighbors4(sr, sc):
        if _in_bounds(h, w, nr, nc) and int(g[nr, nc]) != bg:
            nonbg_neighbors.append((nr, nc))
    assert nonbg_neighbors, "seed pixel must have at least one non-bg neighbor"

    meter_pt: tuple[int, int] | None = None
    step_limit: int | None = None
    if len(nonbg_neighbors) >= 2:
        runs = [(_border_run_length(g, seed, nbr, bg), nbr) for nbr in nonbg_neighbors]
        runs.sort(key=lambda x: x[0], reverse=True)
        if runs[0][0] >= 2:
            step_limit = int(runs[0][0])
            meter_pt = runs[0][1]

    if meter_pt is None:
        line_pt = nonbg_neighbors[0]
        line_col = int(g[line_pt])
        line_starts = [nbr for nbr in nonbg_neighbors if int(g[nbr]) == line_col]
        return {
            "seed_rc": (sr, sc),
            "seed_col": int(g[sr, sc]),
            "line_col": line_col,
            "line_starts": line_starts,
            "step_limit": None,
        }

    line_candidates = [nbr for nbr in nonbg_neighbors if nbr != meter_pt]
    assert line_candidates, "meter branch cannot consume every seed neighbor"
    line_pt = line_candidates[0]
    line_col = int(g[line_pt])
    line_starts = [nbr for nbr in line_candidates if int(g[nbr]) == line_col]
    assert line_starts, "seed must retain a real line branch"
    return {
        "seed_rc": (sr, sc),
        "seed_col": int(g[sr, sc]),
        "line_col": line_col,
        "line_starts": line_starts,
        "step_limit": step_limit,
    }


def _trace_nonbg_component8_from(
    g: np.ndarray, start: tuple[int, int], bg: int, step_limit: int | None
) -> set[tuple[int, int]]:
    h, w = g.shape
    q = deque([(start, 0)])
    dist = {start: 0}
    while q:
        (r, c), d = q.popleft()
        if step_limit is not None and d >= step_limit:
            continue
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc):
                continue
            if (nr, nc) in dist or int(g[nr, nc]) == bg:
                continue
            dist[(nr, nc)] = d + 1
            q.append(((nr, nc), d + 1))
    return set(dist)


def _color_component8_from(
    g: np.ndarray, starts: list[tuple[int, int]] | set[tuple[int, int]], color: int
) -> set[tuple[int, int]]:
    h, w = g.shape
    st = [p for p in starts if int(g[p]) == color]
    vis = set(st)
    while st:
        r, c = st.pop()
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) not in vis and int(g[nr, nc]) == color:
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
                if _in_bounds(h, w, nr, nc) and (nr, nc) not in region and int(g[nr, nc]) != bg:
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


def _boundary_adjacent_to_pixels(
    boundary: set[tuple[int, int]], pixels: set[tuple[int, int]], h: int, w: int
) -> bool:
    if not boundary or not pixels:
        return False
    for r, c in boundary:
        for nr, nc in neighbors8(r, c):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in pixels:
                return True
    return False


def _solve_core(
    grid: np.ndarray,
    *,
    use_step_limit: bool,
    line_scope: str,
    require_boundary_in_trace: bool,
    limit_line_to_trace: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cavities = _cavities(g, bg)

    for seed in _seed_pixels(g, bg):
        info = _seed_info(g, seed, bg)
        step_limit = info["step_limit"] if use_step_limit else None
        traced = _trace_nonbg_component8_from(g, seed, bg, step_limit)
        connected_line = _color_component8_from(g, info["line_starts"], int(info["line_col"]))
        if line_scope == "connected":
            line_pixels = connected_line
        elif line_scope == "global":
            line_pixels = {(int(r), int(c)) for r, c in np.argwhere(g == int(info["line_col"]))}
        else:
            raise ValueError(f"unsupported line_scope: {line_scope}")
        if limit_line_to_trace:
            line_pixels = line_pixels & traced

        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_candidates = boundary & traced if require_boundary_in_trace else boundary
            if not _boundary_adjacent_to_pixels(boundary_candidates, line_pixels, h, w):
                continue
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = int(info["seed_col"])
    return out


def _combo_stats(grid: np.ndarray) -> list[dict[str, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    cavities = _cavities(g, bg)
    stats: list[dict[str, int]] = []
    for seed in _seed_pixels(g, bg):
        info = _seed_info(g, seed, bg)
        full_trace = _trace_nonbg_component8_from(g, seed, bg, None)
        limited_trace = _trace_nonbg_component8_from(g, seed, bg, int(info["step_limit"]))
        connected_line = _color_component8_from(g, info["line_starts"], int(info["line_col"]))
        limited_line = connected_line & limited_trace
        all_line = {(int(r), int(c)) for r, c in np.argwhere(g == int(info["line_col"]))}
        near = 0
        far = 0
        fake = 0
        for _cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            full_hit = _boundary_adjacent_to_pixels(boundary & full_trace, connected_line, h, w)
            limited_hit = _boundary_adjacent_to_pixels(boundary & limited_trace, limited_line, h, w)
            global_hit = _boundary_adjacent_to_pixels(boundary, all_line, h, w)
            if limited_hit:
                near += 1
            if full_hit and not limited_hit:
                far += 1
            if global_hit and not full_hit:
                fake += 1
        stats.append(
            {
                "step_limit": int(info["step_limit"]),
                "near": near,
                "far": far,
                "fake": fake,
            }
        )
    return stats


def _cell_delta(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(np.asarray(a, dtype=int) != np.asarray(b, dtype=int)))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the seed-owned cavities that are both connected and close enough to the seed.

    Input structure:
    - The background is the most common color.
    - Each seed is a unique single pixel on a corner, with one branch forming a straight border meter and the other branch forming an 8-connected line into the grid.
    - Closed single-color frame rings may attach to the true line, and an extra disconnected same-color fake path may also carry rings.

    Rule:
    - Read the step cap N from the longer straight border run next to the seed.
    - From that seed, trace non-background pixels by 8-connected BFS but stop after N steps.
    - Fill a cavity with the seed color only if its ring is touched inside this limited reachable set and that touched ring is adjacent to the seed's own connected line-color component.

    Key insight:
    The meter limits how far the seed can claim, and connectivity still matters after that: far true loops and nearby-looking disconnected same-color loops both remain background.
    """
    return _solve_core(
        np.asarray(grid, dtype=int),
        use_step_limit=True,
        line_scope="connected",
        require_boundary_in_trace=True,
        limit_line_to_trace=True,
    )


# ========================== CFG ==========================
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

SEED_COUNT_CHOICES = [1, 1, 2]
METER_LEN_RANGE = (2, 5)
LINE_LEN_RANGE = (12, 28)
MIN_LINE_LEN = 10

FRAME_COLORS_MAX = 3
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_ATTACH_TRIES = 60
DISTRACTOR_FRAME_TRIES = 40
FRAME_PLACEMENT_PAD = 1

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.3

SAME_COLOR_DISTRACTOR_TRIES = 80
DISTRACTOR_LINE_LEN = (5, 12)
DISTRACTOR_NETWORK_FRAMES = (1, 2)
EXTRA_TRUE_FRAMES_MAX = 2
SOLO_DISTRACTOR_MAX = 1

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = ch + 2, cw + 2
    ring = []
    for dr in range(h):
        for dc in range(w):
            on_bdr = dr == 0 or dr == h - 1 or dc == 0 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    allowed_touch_pixels: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in allowed_touch_pixels:
                return False

    ring_set = set(ring)
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] != -1 and (nr, nc) not in allowed_touch_pixels:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and (nr, nc) in allowed_touch_pixels:
                return True
    return False


def _weighted_dirs(dr_bias: int, dc_bias: int) -> list[tuple[int, int]]:
    dirs: list[tuple[int, int]] = []
    for ddr, ddc in DIAG_DIRS:
        aligned = ddr == dr_bias or ddc == dc_bias
        dirs.extend([(ddr, ddc)] * (DIAG_WEIGHT_ALIGNED if aligned else DIAG_WEIGHT_OTHER))
    for ddr, ddc in ORTHO_DIRS:
        aligned = (ddr == dr_bias and ddc == 0) or (ddc == dc_bias and ddr == 0)
        dirs.extend([(ddr, ddc)] * (ORTHO_WEIGHT_ALIGNED if aligned else ORTHO_WEIGHT_OTHER))
    return dirs


def _grow_line_from_prefix(
    grid: np.ndarray,
    prefix: list[tuple[int, int]],
    line_col: int,
    target_len: int,
    rng: random.Random,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = list(prefix)
    cur = path[-1]
    path_set = set(path)
    dr_bias = 1 if cur[0] < h // 2 else -1
    dc_bias = 1 if cur[1] < w // 2 else -1

    while len(path) < target_len:
        dirs = _weighted_dirs(dr_bias, dc_bias)
        rng.shuffle(dirs)
        placed = False
        for ddr, ddc in dirs:
            nr, nc = cur[0] + ddr, cur[1] + ddc
            if not _in_bounds(h, w, nr, nc) or grid[nr, nc] != -1:
                continue
            if any((nnr, nnc) != cur and (nnr, nnc) in path_set for nnr, nnc in neighbors8(nr, nc)):
                continue
            foreign_touch = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc):
                    continue
                if (nnr, nnc) == cur:
                    continue
                if grid[nnr, nnc] != -1:
                    foreign_touch = True
                    break
            if foreign_touch:
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
            if not placed:
                break
    return path


def _corner_layout(
    corner_kind: str, h: int, w: int, meter_len: int, meter_horizontal: bool
) -> tuple[tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]:
    if corner_kind == "TL":
        seed = (0, 0)
        if meter_horizontal:
            meter = [(0, c) for c in range(1, meter_len + 1)]
            prefix = [(1, 0), (2, 1)]
        else:
            meter = [(r, 0) for r in range(1, meter_len + 1)]
            prefix = [(0, 1), (1, 2)]
    elif corner_kind == "TR":
        seed = (0, w - 1)
        if meter_horizontal:
            meter = [(0, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(1, w - 1), (2, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(1, meter_len + 1)]
            prefix = [(0, w - 2), (1, w - 3)]
    elif corner_kind == "BL":
        seed = (h - 1, 0)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(1, meter_len + 1)]
            prefix = [(h - 2, 0), (h - 3, 1)]
        else:
            meter = [(r, 0) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, 1), (h - 2, 2)]
    elif corner_kind == "BR":
        seed = (h - 1, w - 1)
        if meter_horizontal:
            meter = [(h - 1, c) for c in range(w - 2, w - 2 - meter_len, -1)]
            prefix = [(h - 2, w - 1), (h - 3, w - 2)]
        else:
            meter = [(r, w - 1) for r in range(h - 2, h - 2 - meter_len, -1)]
            prefix = [(h - 1, w - 2), (h - 2, w - 3)]
    else:
        raise ValueError(corner_kind)
    return seed, meter, prefix


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _clear_cells(canvas: np.ndarray, pts: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    for r, c in pts:
        canvas[r, c] = -1


def _try_attach_frame(
    canvas: np.ndarray,
    anchor_points: list[tuple[int, int]],
    allowed_touch_pixels: set[tuple[int, int]],
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    if not anchor_points:
        return None
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    for _ in range(max_tries):
        ch = rng.choice(CAVITY_SIZE_WEIGHTS)
        cw = rng.choice(CAVITY_SIZE_WEIGHTS)
        lp = rng.choice(anchor_points)
        fh, fw = ch + 2, cw + 2
        offsets = _halo_offsets(fh, fw)
        rng.shuffle(offsets)
        for dr, dc in offsets:
            r0 = lp[0] - dr
            c0 = lp[1] - dc
            if r0 < pad or c0 < pad or r0 + fh > h - pad or c0 + fw > w - pad:
                continue
            ring, cavity = _frame_ring_coords(r0, c0, ch, cw)
            if not _validate_frame_placement(canvas, ring, cavity, allowed_touch_pixels, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_attach_detached_frame(canvas: np.ndarray, frame_col: int, rng: random.Random) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_FRAME_TRIES):
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


def _try_place_same_color_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    frame_goal: int,
) -> int:
    h, w = canvas.shape
    for _ in range(SAME_COLOR_DISTRACTOR_TRIES):
        if h < 8 or w < 8:
            return 0
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue

        canvas[r, c] = line_col
        path = _grow_line_from_prefix(canvas, [(r, c)], line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng)
        if len(path) < 5 or not _path_is_isolated(canvas, path):
            _clear_cells(canvas, path)
            continue

        allowed_touch = set(path)
        placed_cells: set[tuple[int, int]] = set(path)
        attached = 0
        for _ in range(max(frame_goal * 3, frame_goal + 2)):
            if attached >= frame_goal:
                break
            attached_frame = _try_attach_frame(canvas, path, allowed_touch, rng.choice(frame_colors_pool), rng)
            if attached_frame is None:
                continue
            ring, _ = attached_frame
            placed_cells.update(ring)
            attached += 1

        if attached >= frame_goal:
            return attached

        _clear_cells(canvas, placed_cells)
    return 0


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_CHOICES)
    if 3 * num_seeds >= len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    meter_colors = [palette.pop() for _ in range(num_seeds)]

    max_frame_cols = min(FRAME_COLORS_MAX, len(palette))
    if max_frame_cols <= 0:
        return None
    num_frame_cols = rng.randint(1, max_frame_cols)
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(corners)
    corners_used = corners[:num_seeds]

    lines_info: list[dict[str, object]] = []
    for i in range(num_seeds):
        meter_len = rng.randint(*METER_LEN_RANGE)
        seed_rc, meter_cells, prefix = _corner_layout(corners_used[i], h, w, meter_len, rng.choice([False, True]))
        all_cells = [seed_rc] + meter_cells + prefix
        if any(not _in_bounds(h, w, r, c) for r, c in all_cells):
            return None
        if any(canvas[r, c] != -1 for r, c in all_cells):
            return None

        canvas[seed_rc] = seed_colors[i]
        for r, c in meter_cells:
            canvas[r, c] = meter_colors[i]
        for r, c in prefix:
            canvas[r, c] = line_colors[i]

        lo = max(LINE_LEN_RANGE[0], meter_len + 8)
        hi = min(LINE_LEN_RANGE[1], h + w)
        if lo > hi:
            return None
        target_len = rng.randint(lo, hi)
        path = _grow_line_from_prefix(canvas, prefix, line_colors[i], target_len, rng)
        if len(path) < max(MIN_LINE_LEN, meter_len + 6):
            return None
        lines_info.append(
            {
                "seed_col": seed_colors[i],
                "seed_rc": seed_rc,
                "line_col": line_colors[i],
                "meter_col": meter_colors[i],
                "meter_len": meter_len,
                "path": path,
            }
        )

    for info in lines_info:
        path = list(info["path"])
        path_set = set(path)
        meter_len = int(info["meter_len"])
        near_points = path[: min(len(path), max(2, meter_len))]
        far_start = meter_len + 3
        far_points = path[far_start:] if far_start < len(path) else []
        if not near_points or not far_points:
            return None

        if _try_attach_frame(canvas, near_points, path_set, rng.choice(frame_colors_pool), rng) is None:
            return None
        if _try_attach_frame(canvas, far_points, path_set, rng.choice(frame_colors_pool), rng) is None:
            return None

        for _ in range(rng.randint(0, EXTRA_TRUE_FRAMES_MAX)):
            if rng.random() < 0.5:
                candidate_points = near_points
            else:
                candidate_points = path[max(2, meter_len + 2) :]
            _try_attach_frame(canvas, candidate_points, path_set, rng.choice(frame_colors_pool), rng)

    fake_line_col = int(rng.choice([info["line_col"] for info in lines_info]))
    fake_goal = rng.randint(*DISTRACTOR_NETWORK_FRAMES)
    if _try_place_same_color_distractor_network(canvas, fake_line_col, frame_colors_pool, rng, fake_goal) <= 0:
        return None

    for _ in range(rng.randint(0, SOLO_DISTRACTOR_MAX)):
        _try_attach_detached_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None
    for info in lines_info:
        if int((grid == int(info["seed_col"])).sum()) != 1:
            return None

    try:
        combo_out = main(grid)
        full_connected = _solve_core(
            grid,
            use_step_limit=False,
            line_scope="connected",
            require_boundary_in_trace=True,
            limit_line_to_trace=False,
        )
        global_color = _solve_core(
            grid,
            use_step_limit=False,
            line_scope="global",
            require_boundary_in_trace=False,
            limit_line_to_trace=False,
        )
    except Exception:
        return None

    if np.array_equal(combo_out, grid):
        return None
    if np.array_equal(combo_out, full_connected):
        return None
    if np.array_equal(combo_out, global_color):
        return None

    stats = _combo_stats(grid)
    if not stats:
        return None
    if any(s["near"] < 1 for s in stats):
        return None
    if any(s["far"] < 1 for s in stats):
        return None
    if max(s["fake"] for s in stats) < 1:
        return None
    if _cell_delta(combo_out, full_connected) <= 0:
        return None
    if _cell_delta(combo_out, global_color) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
