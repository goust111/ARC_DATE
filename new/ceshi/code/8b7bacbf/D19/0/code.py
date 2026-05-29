# Variant of 8b7bacbf
# Dimension: D19 — 符号指令编码 (symbol_instructions)
# Change: 无符号指令——所有规则直接从空间结构推断 → 颜色映射图例
#
# Design analysis
# ===============
# spec_variant:
#   preserved_core:
#     - 原题的主体仍然是：边界 seed 接一条 8 连通细线；线外侧挂若干单色闭合框；
#       solver 追踪 seed 所在非背景连通块，并把与该块相接的单色 cavity flood-fill。
#     - cavity 的判定、line/frame 的拓扑关系、distractor 的作用都保持原题风格。
#   d19_change:
#     - 输入左上角固定放一个 4x5 的 legend box。
#     - 外环是一圈统一的 legend_color，内部两行分别放两个颜色点：
#       row = [seed_color, bg, fill_color]，表示 "该 seed 最终要填成 fill_color"。
#     - 真正的边界 seed 仍在其余三个角里，但 output 不再用 seed 原色填洞，而是先查 legend。
#   concrete_trace:
#     - 原题某 seed 为红色，线追踪到几个 cavity 后，直接把 cavity 变红。
#     - 变体中 legend 某行若是 [红, bg, 黄]，则同一条线仍然追踪同样的 cavity，
#       但最后把 cavity 变黄。
#     - 差异只发生在“写入颜色”的最后一步；图形拓扑判断不变。
#   complexity_calibration:
#     - legend 固定为 2 行，支持原题的 1-2 个 seed。
#     - 若本例只有 1 个 seed，另一行放一个未使用的 dummy 映射，避免图例退化成
#       “只是一对无解释颜色”的弱提示；这样人类更容易看出这是映射表而不是噪声。
#     - fill_color 始终与 seed_color 不同，确保 D19 的视觉效果可观察，不会退化回原题。
#   implicit_assumption_audit:
#     - 原 `_seed_pixels()` 依赖“seed 颜色在整图只出现 1 次”。变体里 seed 颜色还会在
#       legend 里再出现 1 次，所以这个前提失效。修复：只在 legend 之外统计唯一色。
#     - 原 `out[r, c] = seed_col` 依赖“填充色就是 seed 色”。变体里该前提失效。
#       修复：先解析 legend，使用 `legend_map[seed_col]`。
#     - 原 solver 默认四角中任意背景角都可作为 outside bg 的锚点。现在左上角被 legend
#       占据，不再是背景角；生成器保证最多只再使用 2 个 seed 角，因此剩余角里至少还有
#       1 个背景角，outside 检测仍然成立。
#   placement_legality:
#     - legend 固定在左上角，并为其右侧/下侧预留 1 格缓冲带，避免 seed 线或 frame
#       紧贴 legend 造成视觉混叠，也防止 traced component 意外碰到 legend。
#     - seed 只放在 TR / BL / BR 三个角，避免与 legend 冲突。
#     - frame 仍要求 cavity 自身不接触 line，ring 至少一处 8 邻接到对应 line。
#   visual_observability:
#     - 解题者能在 input 中直接看到：左上角有一个小框，框中左列颜色恰好复用了边界 seed
#       的颜色，而 output 中被填充的颜色改成了同一行右列的颜色。
#     - 多个训练样例里映射会变化，因此不能靠固定颜色记忆，只能靠读图例。
#   cleanliness:
#     - legend 与主体图形分区明显；主体拓扑不叠在 legend 上，减少视觉噪声。
#   color_inference:
#     - 颜色种类被控制在少量档位：1 个 legend_color、1 个 frame_color、1-2 个 seed、
#       1-2 个 line、1-2 个 fill，再加 0-1 条 dummy 映射。
#     - 颜色映射不是凭颜色编号，而是凭 legend 中“同一行左右配对”的显式结构推断。

from collections import Counter
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


LEGEND_H = 4
LEGEND_W = 5
LEGEND_ROWS = 2
LEGEND_PAD = 1
RESERVED = -2


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _legend_bbox() -> tuple[int, int, int, int]:
    return (0, 0, LEGEND_H - 1, LEGEND_W - 1)


def _legend_mask(shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[:LEGEND_H, :LEGEND_W] = True
    return mask


def _legend_border_coords() -> list[tuple[int, int]]:
    coords = []
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                coords.append((r, c))
    return coords


def _extract_legend(g: np.ndarray, bg: int) -> dict[int, int]:
    assert g.shape[0] >= LEGEND_H and g.shape[1] >= LEGEND_W
    box = g[:LEGEND_H, :LEGEND_W]
    legend_col = int(box[0, 0])
    assert legend_col != bg
    for r, c in _legend_border_coords():
        assert int(box[r, c]) == legend_col

    mapping: dict[int, int] = {}
    for row in range(1, 1 + LEGEND_ROWS):
        left = int(box[row, 1])
        mid = int(box[row, 2])
        right = int(box[row, 3])
        assert left != bg and right != bg
        assert left != legend_col and right != legend_col
        assert mid == bg
        mapping[left] = right
    return mapping


def _seed_pixels(g: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = g.shape
    outside = ~legend_mask
    vals, cnts = np.unique(g[outside], return_counts=True)
    seeds: list[tuple[int, int]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        if v == bg or int(cnt) != 1:
            continue
        rc = np.argwhere((g == v) & outside)
        if rc.shape[0] != 1:
            continue
        r, c = map(int, rc[0])
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    1. Read the top-left legend box to obtain seed_color -> fill_color pairs.
    2. Find border seed pixels outside the legend.
    3. Trace each seed's connected drawing exactly as in the original task.
    4. Fill attached single-color cavities, but use the mapped fill_color rather than the seed color itself.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    legend_map = _extract_legend(g, bg)
    legend_mask = _legend_mask(g.shape)

    out = g.copy()

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
    for i, (reg, bnd) in enumerate(regions):
        if i in outside_idxs or not bnd:
            continue
        bcols = {int(g[p]) for p in bnd}
        cavities.append((reg, bnd, bcols))

    for sr, sc in _seed_pixels(g, bg, legend_mask):
        seed_col = int(g[sr, sc])
        assert seed_col in legend_map
        fill_col = int(legend_map[seed_col])

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
                    out[r, c] = fill_col

    return out


# --------------- generator ---------------

# Canvas / grid
GRID_H_RANGE = (12, 30)
GRID_W_RANGE = (12, 30)
BG_COLORS = [0, 5]

# Seeds / lines
SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

# Frame loops
FRAME_COLORS_MAX = 1
TOTAL_LOOPS_MAX = 8
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 40
DISTRACTOR_FRAME_TRIES = 40

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

DIAG_DIRS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
ORTHO_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _in_bounds(h: int, w: int, r: int, c: int) -> bool:
    return 0 <= r < h and 0 <= c < w


def _frame_ring_coords(r0: int, c0: int, ch: int, cw: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
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


def _validate_frame_placement(canvas: np.ndarray, ring: list[tuple[int, int]], cavity: list[tuple[int, int]], line_col: int, h: int, w: int) -> bool:
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


def _halo_offsets(fh: int, fw: int) -> list[tuple[int, int]]:
    return [(dr, dc) for dr in range(-1, fh + 1) for dc in range(-1, fw + 1) if not (0 <= dr < fh and 0 <= dc < fw)]


def _try_attach_frame(canvas: np.ndarray, line_path: list[tuple[int, int]], line_col: int, frame_col: int, rng: random.Random, max_tries: int = FRAME_ATTACH_TRIES):
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


def _try_attach_distractor_frame(canvas: np.ndarray, frame_col: int, rng: random.Random, max_tries: int = DISTRACTOR_FRAME_TRIES):
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
    if corner_kind == "TR":
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


def _place_legend(canvas: np.ndarray, legend_col: int, legend_rows: list[tuple[int, int]]) -> None:
    for r in range(LEGEND_H):
        for c in range(LEGEND_W):
            if r in (0, LEGEND_H - 1) or c in (0, LEGEND_W - 1):
                canvas[r, c] = legend_col
    for i, (seed_col, fill_col) in enumerate(legend_rows, start=1):
        canvas[i, 1] = seed_col
        canvas[i, 3] = fill_col


def _reserve_legend_moat(canvas: np.ndarray) -> None:
    h, w = canvas.shape
    max_r = min(h, LEGEND_H + LEGEND_PAD)
    max_c = min(w, LEGEND_W + LEGEND_PAD)
    for r in range(max_r):
        for c in range(max_c):
            if r < LEGEND_H and c < LEGEND_W:
                continue
            if canvas[r, c] == -1:
                canvas[r, c] = RESERVED


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    needed = 1 + num_seeds + num_seeds + num_seeds + (LEGEND_ROWS - num_seeds) * 2 + 1
    if needed > len(palette):
        return None

    canvas = -np.ones((h, w), dtype=int)

    legend_col = palette.pop()
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = [palette.pop() for _ in range(num_seeds)]
    fill_colors = [palette.pop() for _ in range(num_seeds)]
    dummy_pairs = [(palette.pop(), palette.pop()) for _ in range(LEGEND_ROWS - num_seeds)]
    frame_colors_pool = [palette.pop()]

    active_pairs = list(zip(seed_colors, fill_colors))
    legend_rows = active_pairs + dummy_pairs
    rng.shuffle(legend_rows)
    _place_legend(canvas, legend_col, legend_rows)
    _reserve_legend_moat(canvas)

    all_corners = ["TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

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
        lines_info.append((seed_colors[i], fill_colors[i], seed_rc, line_colors[i], path))

    total_loops_target = rng.randint(num_seeds, TOTAL_LOOPS_MAX)
    num_distractors_target = rng.randint(0, max(0, total_loops_target - num_seeds))
    num_attached_target = total_loops_target - num_distractors_target
    per_seed = _distribute_loop_budget(num_seeds, num_attached_target, rng)

    for i, (_, _, _, line_col, path) in enumerate(lines_info):
        attached = 0
        for _ in range(per_seed[i]):
            fc = rng.choice(frame_colors_pool)
            if _try_attach_frame(canvas, path, line_col, fc, rng) is not None:
                attached += 1
        if attached == 0:
            return None

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas < 0, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    legend_mask = _legend_mask(grid.shape)
    for seed_col, _, _, _, _ in lines_info:
        if int(((grid == seed_col) & ~legend_mask).sum()) != 1:
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None

    # Generator-side non-degeneracy check: at least one active mapped fill color must appear outside the legend in output.
    if not any(int(((out == fill_col) & ~legend_mask).sum()) > 0 for _, fill_col, _, _, _ in lines_info):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
