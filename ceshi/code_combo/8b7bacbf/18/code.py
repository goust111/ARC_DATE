# Combo variant of 8b7bacbf (index 18)
# combo_spec: D6/0+D11/2+D13/0
# Name: 框架重涂+干扰假路径+线路种子同色
#
# Design analysis
# a) 具体化与行为追踪
#    融合后输入仍是“角点 seed -> 细线路径 -> 单色框架环”的语法，但有两点同时变化：
#    1. seed 与其真线路共享同一种颜色；
#    2. 画面里还会出现一个或多个同色但断开的假路径网络，也可能挂着框架。
#    具体小例子：左上角角点是 7，向内的真线路也是 7，并贴着一个 3 色框环；右下角另有一条断开的
#    7 色假路径，也贴着一个 4 色框环。原题会先靠“唯一色边缘像素”找 seed，再把真线路触到的框内腔体
#    填成 7。融合后这两步都变了：seed 不能再靠颜色唯一性识别，而要靠“角点非背景像素”；并且输出不再
#    填腔体，而是只把真线路可达框环本身改成 7。右下角那条同色假路径虽然颜色完全相同，但因不和角点 seed
#    8 连通，不能触发重涂。差异点正好覆盖三项维度：识别 seed 的方式变了、真假路径必须靠连通性区分、
#    最终改色目标从腔体切到环边界。
# b) 复杂度校准
#    仍保留 1-2 个 seed、多个矩形环、1px 8 连通线路，不引入新对象种类。为了让融合效果肉眼可见，
#    generator 强制至少放一个“同色但断开”的假路径网络，且至少给它挂一个框环，这样输出里一定能看到
#    “同色但不变”的对照。同时 frame 颜色仍限制在少量颜色池内，避免把问题变成杂乱的全局颜色匹配。
# c) 隐式假设审查
#    - 原 `_seed_pixels()` 假设 seed 颜色全局唯一。D13 打破该前提，因此改成只读四角非背景像素。
#    - 原 `line_col` 由 seed 的 4 邻颜色统计得到。这里 line 与 seed 同色，且首像素是对角相邻，因此不再
#      用邻域多数色推断，直接令 `line_col = seed_col`，并用 seed 出发的同色 8 连通分量表示“owned_line”。
#    - 原“只要边界附近出现 line_color 即可”的想法会被 D11 同色假路径击穿。融合后必须限定为：
#      边界既属于 seed 的 non-bg 可达组件，又必须贴着 seed 拥有的同色分量，而不是全局同色像素。
#    - 原最终写回 `cavity` 内部的逻辑会退化回原题；D6 要求写回 `boundary` 才对。
#    - 原生成器允许“同 line_color 的任何像素都可作为框的合法贴线对象”。融合后这会让真/假同色网络在生成
#      阶段互相串线，因此放框合法性也必须改为“只允许接触这条指定路径像素集合”。
# d) 放置合法性
#    generator 保证只有角点可以出现 seed；线路从角点的对角内侧起步，框架离边至少 1 格；每个框环与腔体都
#    占据空白格，且框环只能 8 邻接它所归属的那条路径像素。假路径网络从内部空白起步，放置后再校验与已有
#    非背景像素 8 断开。最终还会检查不同角点 seed 的 non-bg 组件彼此不相交，避免 solver 的 traced drawing
#    归属变得含混。
# e) 视觉可观察性
#    人类从少量样例里可以直接看到三件事：角点和线路同色；同色线路会在别处重复出现但那些框不变；真正发生
#    的变化是“框边变成 seed 色而内部空白保持原样”。这三条都是纯拓扑和可见颜色变化，不依赖颜色编号大小。
# f) 可视化整洁度
#    仍使用原题的细线和小型矩形环，不叠加额外标记。假路径数量控制在 1-2 个网络，框数控制在小范围内，并用
#    8 邻隔离避免不同网络糊成大团。框环颜色与 seed 颜色分离，保证输出中的重涂效果明显。
# g) 变化的可推断性
#    输出颜色变化只有一种：某个真 seed 可达框环整体改成该 seed/线路颜色。由于输入里同色假路径稳定存在，
#    解题者可以归纳出“同色不够，必须连到角点”；由于腔体不被填充，也能归纳出“变的是环，不是内部”。颜色
#    总种类仍受控在少数几种，结构线索充足。
# post_edit_main_assumptions:
#    - 假设 1：角点上的非背景像素就是 seed。generator 通过角点专用放置和边缘留白保证。
#    - 假设 2：每个 seed 至少连到一个同色像素，因此从 seed 出发的同色 8 连通分量不只 seed 自身。
#    - 假设 3：目标框环的边界既在 seed 的 non-bg traced drawing 内，也至少贴着该 seed 拥有的同色线路。
#    - 假设 4：不同 seed 的 non-bg 组件彼此 8 断开；否则“哪个 seed 拥有该框环”会不稳定。

from __future__ import annotations

import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _corner_seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return [(r, c) for r, c in corners if int(g[r, c]) != bg]


def _legacy_unique_border_seeds(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
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


def _single_color_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], int]]:
    h, w = g.shape
    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    cavities = []
    for pts in _components_mask(g == bg, conn8=False):
        region = set(pts)
        if any(p in region for p in corners if int(g[p]) == bg):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if not boundary:
            continue
        bcols = {int(g[p]) for p in boundary}
        if len(bcols) != 1:
            continue
        cavities.append((region, boundary, next(iter(bcols))))
    return cavities


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


def _seed_infos(g: np.ndarray, bg: int) -> list[dict]:
    infos = []
    for sr, sc in _corner_seed_pixels(g, bg):
        seed_col = int(g[sr, sc])
        owned_nonbg = _nonbg_component8_from(g, (sr, sc), bg)
        owned_line = _color_component8_from(g, {(sr, sc)}, seed_col)
        assert len(owned_line) > 1, "corner seed must connect to a same-color line"
        all_line = {(int(r), int(c)) for r, c in np.argwhere(g == seed_col)}
        infos.append(
            {
                "seed_rc": (sr, sc),
                "seed_col": seed_col,
                "owned_nonbg": owned_nonbg,
                "owned_line": owned_line,
                "all_line": all_line,
            }
        )
    return infos


def _solve_variant(
    grid: np.ndarray,
    *,
    require_connected_line: bool,
    recolor_ring: bool,
    require_traced_boundary: bool,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()
    cavities = _single_color_cavities(g, bg)

    for info in _seed_infos(g, bg):
        line_pixels = info["owned_line"] if require_connected_line else info["all_line"]
        for cavity, boundary, _frame_col in cavities:
            if require_traced_boundary and not (boundary & info["owned_nonbg"]):
                continue
            if not _boundary_adjacent_to_pixels(boundary, line_pixels, h, w):
                continue
            target = boundary if recolor_ring else cavity
            for r, c in target:
                if recolor_ring or int(out[r, c]) == bg:
                    out[r, c] = info["seed_col"]
    return out


def _recolorable_ring_count(g: np.ndarray) -> int:
    g = np.asarray(g, dtype=int)
    bg = get_bg(g)
    count = 0
    for info in _seed_infos(g, bg):
        for _cavity, boundary, _frame_col in _single_color_cavities(g, bg):
            if not (boundary & info["owned_nonbg"]):
                continue
            if _boundary_adjacent_to_pixels(boundary, info["owned_line"], g.shape[0], g.shape[1]):
                count += 1
    return count


def _same_color_distractor_component_count(g: np.ndarray) -> int:
    g = np.asarray(g, dtype=int)
    bg = get_bg(g)
    total = 0
    for info in _seed_infos(g, bg):
        comps = _components_mask(g == info["seed_col"], conn8=True)
        for comp in comps:
            comp_set = set(comp)
            if info["seed_rc"] not in comp_set:
                total += 1
    return total


def _seed_components_are_disjoint(g: np.ndarray) -> bool:
    bg = get_bg(g)
    comps = [_nonbg_component8_from(g, seed, bg) for seed in _corner_seed_pixels(g, bg)]
    for i in range(len(comps)):
        for j in range(i + 1, len(comps)):
            if comps[i] & comps[j]:
                return False
    return True


def _border_is_clean_except_corner_seeds(g: np.ndarray, bg: int) -> bool:
    h, w = g.shape
    allowed = set(_corner_seed_pixels(g, bg))
    border_pts = {(0, c) for c in range(w)} | {(h - 1, c) for c in range(w)} | {(r, 0) for r in range(h)} | {
        (r, w - 1) for r in range(h)
    }
    for r, c in border_pts:
        if int(g[r, c]) != bg and (r, c) not in allowed:
            return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor reachable frame rings from corner seeds whose lines share the seed color.

    Input structure:
    - bg is the most frequent color.
    - One or two non-background corner pixels act as seeds.
    - Each seed is 8-connected to a same-color meandering line, and those lines touch single-color frame rings.
    - Elsewhere there may be disconnected fake paths in that very same color, sometimes with their own rings.

    Rule:
    - Treat every non-background corner pixel as a seed and trace its reachable non-background drawing.
    - Within that drawing, find enclosed background cavities whose single-color boundary is touched by the seed's own
      same-color line component.
    - Do not flood-fill the cavity interior.
    - Instead recolor the entire boundary ring to the seed/line color.
    - Disconnected same-color fake paths do nothing because they are not connected back to the corner seed.

    Key insight:
    - The same color can appear on both a true seed-line network and a false distractor path, so color matching alone
      is insufficient; the solver must combine corner-based seed detection, same-color connectivity, and ring-only
      recoloring.
    """
    return _solve_variant(
        np.asarray(grid, dtype=int),
        require_connected_line=True,
        recolor_ring=True,
        require_traced_boundary=True,
    )


# ========================== CFG ==========================
GRID_H_RANGE = (10, 30)
GRID_W_RANGE = (10, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 1, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 50

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.9
BIAS_FLIP_PROB = 0.3

DISTRACTOR_NETWORK_TRIES = 80
DISTRACTOR_LINE_LEN = (5, 14)
DISTRACTOR_NETWORK_FRAMES = (1, 3)
DISTRACTOR_NETWORK_COUNT = (1, 2)

GEN_RETRIES = 500
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    fh, fw = ch + 2, cw + 2
    ring = []
    for dr in range(fh):
        for dc in range(fw):
            on_bdr = dr == 0 or dr == fh - 1 or dc == 0 or dc == fw - 1
            corner = (dr in (0, fh - 1)) and (dc in (0, fw - 1))
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
    ring_set = set(ring)
    cavity_set = set(cavity)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in cavity_set or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] != -1:
                return False

    touching = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set or (nr, nc) in cavity_set:
                continue
            if (nr, nc) in allowed_touch_pixels:
                touching = True
                continue
            if canvas[nr, nc] != -1:
                return False
    return touching


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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _clear_cells(canvas: np.ndarray, pts: list[tuple[int, int]] | set[tuple[int, int]]) -> None:
    for r, c in pts:
        canvas[r, c] = -1


def _path_is_isolated(
    canvas: np.ndarray,
    path: list[tuple[int, int]],
    allowed_extra: set[tuple[int, int]] | None = None,
) -> bool:
    allowed_extra = set() if allowed_extra is None else set(allowed_extra)
    h, w = canvas.shape
    path_set = set(path)
    for r, c in path:
        for nr, nc in neighbors8(r, c):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in path_set or (nr, nc) in allowed_extra:
                continue
            if canvas[nr, nc] != -1:
                return False
    return True


def _try_attach_frame(
    canvas: np.ndarray,
    line_path: list[tuple[int, int]],
    allowed_touch_pixels: set[tuple[int, int]],
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
            if not _validate_frame_placement(canvas, ring, cavity, allowed_touch_pixels, h, w):
                continue
            for rr, cc in ring:
                canvas[rr, cc] = frame_col
            return ring, cavity
    return None


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors_pool: list[int],
    rng: random.Random,
    frame_goal: int,
) -> int:
    h, w = canvas.shape
    if h < 8 or w < 8:
        return 0

    for _ in range(DISTRACTOR_NETWORK_TRIES):
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
        allowed_touch = set(path)
        attached = 0
        attempts = max(frame_goal * 4, frame_goal + 2)
        for _ in range(attempts):
            if attached >= frame_goal:
                break
            frame_col = rng.choice(frame_colors_pool)
            attached_frame = _try_attach_frame(canvas, path, allowed_touch, frame_col, rng)
            if attached_frame is None:
                continue
            ring, _cavity = attached_frame
            placed_cells.update(ring)
            attached += 1

        if attached >= frame_goal:
            return attached

        _clear_cells(canvas, placed_cells)
    return 0


def _corner_config(corner_kind: str, h: int, w: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if corner_kind == "TL":
        return (0, 0), (1, 1)
    if corner_kind == "TR":
        return (0, w - 1), (1, w - 2)
    if corner_kind == "BL":
        return (h - 1, 0), (h - 2, 1)
    if corner_kind == "BR":
        return (h - 1, w - 1), (h - 2, w - 2)
    raise ValueError(corner_kind)


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
    if num_seeds + 1 >= len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = list(seed_colors)

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    num_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    frame_colors_pool = [palette.pop() for _ in range(num_frame_cols)]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN or not _path_is_isolated(canvas, path, {seed_rc}):
            return None
        lines_info.append((seed_colors[i], seed_rc, path))

    total_loops_target = rng.randint(num_seeds + 1, TOTAL_LOOPS_MAX)
    distractor_loops_target = rng.randint(1, min(3, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - distractor_loops_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_seed_col, seed_rc, path) in enumerate(lines_info):
        attach_path = path[1:] if len(path) > 1 else path
        allowed_touch = set(path) | {seed_rc}
        attached = 0
        attempts = max(per_seed[i] * 4, per_seed[i] + 2)
        for _ in range(attempts):
            if attached >= per_seed[i]:
                break
            frame_col = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, attach_path, allowed_touch, frame_col, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    target_networks = rng.randint(*DISTRACTOR_NETWORK_COUNT)
    placed_distractor_loops = 0
    for _ in range(target_networks):
        remaining = distractor_loops_target - placed_distractor_loops
        if remaining <= 0:
            break
        frame_goal = rng.randint(1, min(DISTRACTOR_NETWORK_FRAMES[1], remaining))
        gained = _try_place_distractor_network(canvas, rng.choice(seed_colors), frame_colors_pool, rng, frame_goal)
        if gained > 0:
            placed_distractor_loops += gained
    if placed_distractor_loops == 0:
        return None

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    if len(_corner_seed_pixels(grid, bg)) != num_seeds:
        return None
    if not _border_is_clean_except_corner_seeds(grid, bg):
        return None
    if not _seed_components_are_disjoint(grid):
        return None
    if _same_color_distractor_component_count(grid) == 0:
        return None
    if _legacy_unique_border_seeds(grid, bg):
        return None

    for seed_col, seed_rc, _path in lines_info:
        if int((grid == seed_col).sum()) <= 1:
            return None
        if int(grid[seed_rc]) != seed_col:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    if _recolorable_ring_count(grid) == 0:
        return None

    naive_color_match = _solve_variant(
        grid,
        require_connected_line=False,
        recolor_ring=True,
        require_traced_boundary=False,
    )
    if np.array_equal(naive_color_match, out):
        return None

    fill_interior = _solve_variant(
        grid,
        require_connected_line=True,
        recolor_ring=False,
        require_traced_boundary=True,
    )
    if np.array_equal(fill_interior, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
