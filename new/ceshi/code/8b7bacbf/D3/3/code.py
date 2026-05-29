# Variant of 8b7bacbf
# Dimension: D3 — 腔体填充方式 (trajectory)
# Change: flood-fill——将连通腔体内部所有背景像素用种子颜色完全填满... → 填充为框架环颜色而非种子颜色
#
# Design analysis
# ===============
# variant_spec:
#   core_change:
#     input_objects: 与原题相同。边界唯一色 seed -> 8 连通细线 -> 若干单色框架环；框架环包围的 cavity 初始仍是背景色。
#     output_effect:
#       original: 某条 seed 所连到的所有可填 cavity，统一刷成该 seed 的颜色。
#       variant: 某个 cavity 一旦被判定属于这条 seed 所连通的结构，填充色改为该 cavity 自身框架环的唯一颜色。
#     concrete_visual_example:
#       - seed 是红色，线是蓝色。
#       - 同一条线旁边挂着两个框：左框边界黄色，右框边界绿色。
#       - 原题输出会把两个 cavity 都填成红色。
#       - 本变体输出会把左 cavity 填成黄色、右 cavity 填成绿色。
#       - 差异落在 “找到 cavity 之后最终写什么颜色” 这一步，而不是 cavity 匹配逻辑。
#   complexity_calibration:
#     - 仍保留 1-2 个 seed、1-9 个 loop、1-4 种 frame color，避免任务结构退化。
#     - 为了让颜色来源反转肉眼可见，生成器会更积极地制造多 frame color 的附着 loop。
#     - seed color、line color、frame color 仍从互不重叠的调色板中抽取，防止“刚好同色”掩盖变体。
#   implicit_assumptions_audit_main:
#     - `_seed_pixels`: 依赖 seed 是边界上的唯一色单像素；变体不改变这一点。
#     - `line_col = Counter(neigh_cols)...`: 依赖 seed 至少接触一个非背景像素，且邻近线色可识别；变体不改变这一点。
#     - `bg_comps` / `outside_idxs`: 依赖 cavity 仍是 4 连通背景区域；变体不改变几何定义。
#     - `bcols = {int(g[p]) for p in bnd}` 与 `len(bcols) == 1`:
#         这是本变体最关键的旧前提。只有当 cavity 边界完全是单一框架色时，才能把 “边界唯一色” 当成填充色。
#         因此生成器必须继续保证线只贴在框架外侧，不进入 cavity 的 4 邻边界。
#     - `if int(out[r, c]) == bg`: 原题里即使多个 seed 错误命中同一 cavity，也只是第一次写 seed 色。
#         变体里仍保留这层保护，避免后续覆盖；生成器同时继续保证每个可填 cavity 只属于一条 seed 结构。
#   placement_legality:
#     - 框架 ring 和 cavity 必须落在空白区域，不能重叠、越界。
#     - cavity 不得 8 邻接 line 色，否则 boundary 色集合可能混入 line 色，破坏 `len(bcols)==1`。
#     - ring 不得 8 邻接无关非背景色，否则多结构会黏连，solver 的 traced component 会错误扩大。
#     - 至少一个 ring cell 8 邻接对应 line，确保 cavity 的可达性仍由原题机制决定。
#   visual_observability:
#     - 关键可见信号不是“seed 连到了 cavity”，而是“填进去的颜色与 seed 不同，却与各自框架一致”。
#     - 因此生成器偏向在有多个附着 loop 时使用多个 frame color，增加同图中不同 cavity 填不同色的概率。
#   cleanliness:
#     - 不新增额外标记，不叠加新图层；仍沿用原题的线、框、背景三层视觉结构。
#     - 变体只改变输出着色来源，所以画面复杂度与原题基本一致。
#   color_inference:
#     - 颜色规则完全由局部结构决定：每个 cavity 看它自己的包围环颜色。
#     - 不依赖颜色编号大小、排序或隐藏元信息。
#   post_edit_main_assumptions_to_match_generator:
#     - 每个可填 cavity 的 boundary colors 必须是单元素集合。
#     - seed colors 与 frame colors 不重合，这样新输出与原题输出在所有成功样本上都会不同。
#     - 至少存在一个被填 cavity，否则无法观察到 D3 变体。

import random
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

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


def _main_seed_fill(grid: np.ndarray) -> np.ndarray:
    """Original task behavior, kept only for generator-side anti-degeneration checks."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

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
    """
    Variant rule:
    - Keep the original seed/line tracing and cavity selection logic.
    - After a cavity is identified as belonging to a traced seed structure,
      flood-fill it with the cavity's own single boundary color, not the seed color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    # Background regions are unchanged from the original task:
    # interior 4-connected bg components are candidate cavities.
    bg_comps = _components_mask(g == bg, conn8=False)
    regions = []
    for pts in bg_comps:
        region = set(pts)
        touches = _touches_border(pts, h, w)
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        regions.append((region, boundary, touches, len(region)))

    corners = {(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)}
    outside_idxs = set()
    for i, (reg, _, _, _) in enumerate(regions):
        if any(p in reg for p in corners if int(g[p]) == bg):
            outside_idxs.add(i)

    cavities = []
    for i, (reg, bnd, _, _) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for sr, sc in _seed_pixels(g, bg):
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

            frame_col = next(iter(bcols))
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = frame_col

    return out


# --------------- generator (attempt_1_distractor_network) ---------------

# ========================== CFG ==========================
# Canvas / grid
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

# Seeds
SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

# Frame loops
FRAME_COLORS_MAX = 4
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40
MIN_ATTACHED_FRAME_COLORS_WHEN_POSSIBLE = 2

# Line growth
DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

# Distractor network
DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

# Generation
GEN_RETRIES = 300
# =========================================================

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h, w, r, c):
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0, c0, ch, cw):
    """Return (ring_coords, cavity_coords) for a rounded-rect frame."""
    H, W = ch + 2, cw + 2
    ring = []
    for dr in range(H):
        for dc in range(W):
            on_bdr = (dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1)
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
    """
    Check that ring+cavity can be placed on canvas without conflicts.

    The single-color boundary assumption used by `main()` requires that the line
    only touches the ring externally, never the cavity boundary itself.
    """
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
            v = canvas[nr, nc]
            if v != -1 and v != line_col:
                return False

    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] == line_col:
                return True
    return False


def _grow_line(grid, start, line_col, target_len, rng):
    """Grow a meandering 8-connected 1px-wide path of line_col from start."""
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


def _halo_offsets(fh, fw):
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas, line_path, line_col, frame_col, rng, max_tries=FRAME_ATTACH_TRIES):
    """Try to place a frame loop 8-adjacent to some line pixel."""
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


def _try_attach_distractor_frame(canvas, frame_col, rng, max_tries=DISTRACTOR_FRAME_TRIES):
    """Place a loop in empty space, not touching any non-bg pixel."""
    h, w = canvas.shape
    for _ in range(max_tries):
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
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
    return None


def _try_place_distractor_network(canvas, line_col, frame_colors_pool, rng):
    """Place a disconnected line+frame distractor (no seed)."""
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


def _corner_config(corner_kind, h, w, rng):
    """Return (seed_rc, first_line_rc) for corner_kind in {TL,TR,BL,BR}."""
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


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
    """Distribute attached-loop budget across seeds, guaranteeing >= 1 each."""
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


def _planned_frame_colors(per_seed, frame_colors_pool, rng):
    """
    Plan frame colors for attached loops.

    This biases generation toward visible D3 evidence:
    when there are multiple attached loops and multiple frame colors available,
    at least two attached slots receive different frame colors.
    """
    plans = [[rng.choice(frame_colors_pool) for _ in range(n)] for n in per_seed]
    flat_slots = [(i, j) for i, cols in enumerate(plans) for j in range(len(cols))]
    if len(flat_slots) >= 2 and len(frame_colors_pool) >= MIN_ATTACHED_FRAME_COLORS_WHEN_POSSIBLE:
        rng.shuffle(flat_slots)
        c1, c2 = rng.sample(frame_colors_pool, 2)
        i1, j1 = flat_slots[0]
        i2, j2 = flat_slots[1]
        plans[i1][j1] = c1
        plans[i2][j2] = c2
    for cols in plans:
        if len(cols) >= 2 and len(set(cols)) == 1 and len(frame_colors_pool) >= 2:
            alt = rng.choice([c for c in frame_colors_pool if c != cols[0]])
            cols[-1] = alt
            break
    return plans


def _filled_interior_colors(inp: np.ndarray, out: np.ndarray) -> set[int]:
    bg = get_bg(inp)
    mask = (inp == bg) & (out != inp)
    return {int(v) for v in out[mask]}


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

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target

    min_frame_cols = 1
    if num_attached_target >= 2 and len(palette) >= 2:
        min_frame_cols = MIN_ATTACHED_FRAME_COLORS_WHEN_POSSIBLE
    num_frame_cols = rng.randint(min_frame_cols, min(FRAME_COLORS_MAX, len(palette)))
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

    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)
    color_plans = _planned_frame_colors(per_seed, frame_colors_pool, rng)

    for i, (_, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for fc in color_plans[i]:
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
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
        original_out = _main_seed_fill(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if np.array_equal(out, original_out):
        return None

    filled_colors = _filled_interior_colors(grid, out)
    if not filled_colors:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
