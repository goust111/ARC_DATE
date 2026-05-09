# Variant of 8b7bacbf
# Dimension: D4 — 种子源对象定义 (src)
# Change: 网格边缘的唯一色单像素——在网格最外圈、颜色在整个网格中只出现一次的像素作为种子... → 边缘色块（非单像素）
#
# Design analysis
# ---
# variant_spec:
#   core_change:
#     original: "seed 是边缘上的唯一色单像素"
#     variant: "seed 是边缘上的唯一色块，大小为 2-3 个像素，颜色在全网格只出现于这个小块"
#   concrete_shape:
#     seed_block:
#       size: [2, 3]
#       connectivity: "4-connected"
#       placement: "整块都压在最外圈上，仍然靠近角落，形状为短横段或短竖段"
#       attachment: "线条从该短段的内侧端点向网格内部伸出，保证 seed 周围唯一的非背景邻色就是对应的 line_color"
#     unchanged_objects:
#       lines: "仍为 8-connected、1px 宽的蜿蜒路径"
#       frame_loops: "仍为单色闭环，包围背景腔体"
#       fill_rule: "闭环腔体仍用该 seed 的颜色填充"
#   behavior_trace:
#     original_flow:
#       - "按颜色计数找到只出现 1 次的边缘像素"
#       - "从该像素的 4 邻域投票出 line_color"
#       - "从该像素出发追踪整块非背景连通体，再找与之接触的闭环并填色"
#     variant_flow:
#       - "按颜色计数找到只出现 2-3 次、形成单个边缘小块的颜色"
#       - "汇总整个 seed 色块所有像素的 4 邻域，统计其中非背景且非 seed_color 的颜色，得到 line_color"
#       - "从 seed 色块任一像素出发追踪 8 连通非背景结构，后续闭环筛选与填色保持原样"
#     local_example:
#       - "原题 TL 角可能只有 (0,0)=7 这一点作为 seed"
#       - "变体 TL 角改成 [(0,0),(0,1),(0,2)] 都是 7，线从 (1,2) 伸入"
#       - "原 solver 的 `_seed_pixels()` 会漏掉这种输入，因为颜色 7 不再只出现 1 次；新 solver 改为识别整块"
#   complexity_calibration:
#     kept_small:
#       - "seed 大小只允许 2 或 3，避免过多自由度"
#       - "仍限制在角落附近，避免引入新的全局搜索负担"
#     visibility:
#       - "边缘短段相比单点更显眼，2-4 个样例足以让人看出“种子变成小色块”"
#   implicit_assumption_audit_main:
#     old_sensitive_lines:
#       - "_seed_pixels(): 假设 seed color 全局计数为 1；变体下失效"
#       - "neigh_cols around one pixel: 假设一个像素足以暴露 line_color；变体下应改成整块聚合"
#       - "_nonbg_component8_from(start=single_pixel): 这里仍然成立，因为色块任一点都在同一非背景连通体中"
#     new_assumptions:
#       - "seed color 的全部像素形成单个 4 连通小块"
#       - "seed 色块至少有一个 4 邻非背景像素，且其主导颜色就是 line_color"
#       - "不存在其他颜色也恰好出现 2-3 次并构成边缘小块冒充 seed"
#   placement_legality:
#     guarantees_needed:
#       - "seed block 不能与其他 seed、line、frame 重叠"
#       - "line 起点必须与 seed block 4 邻接"
#       - "seed block 仍需保持颜色全局唯一，因此生成器不能复用 seed 颜色"
#       - "线与 frame 的旧约束保持不变，确保 solver 的 cavity 检测仍可用"
#   visual_cleanliness:
#     choices:
#       - "只增加 seed 的长度，不叠加额外标记"
#       - "seed 仍在边缘，线和闭环仍在内部，层次关系清晰"
#   color_inference:
#     note:
#       - "输出仍然把每条线关联的闭环内部填成 seed 颜色；颜色变化规则与原题一致，只是 seed 载体从单点变成小块"
#   generator_validation:
#     checks:
#       - "每个 seed color 必须只出现 2-3 次"
#       - "这些像素必须形成单个边缘 4 连通块"
#       - "新 solver 必须改变至少一个像素"
#       - "旧 solver 在同一输入上应与新 solver 输出显著不同，否则说明变体退化"

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _touches_border(pts: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


def _is_border_cell(r: int, c: int, h: int, w: int) -> bool:
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _seed_components(g: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = g.shape
    vals, cnts = np.unique(g, return_counts=True)
    seeds: list[list[tuple[int, int]]] = []
    for v, cnt in zip(vals, cnts):
        v = int(v)
        cnt = int(cnt)
        if v == bg or cnt not in (2, 3):
            continue
        comps = _components_mask(g == v, conn8=False)
        if len(comps) != 1:
            continue
        pts = comps[0]
        if len(pts) != cnt:
            continue
        if not all(_is_border_cell(r, c, h, w) for r, c in pts):
            continue
        if not _touches_border(pts, h, w):
            continue
        seeds.append(sorted(pts))
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
    Fill enclosed regions along a line seeded by unique border color blocks.

    Input structure:
    - bg = most common color.
    - Colored lines/drawings on the grid.
    - Each seed is a unique-color 2-3 pixel block lying on the outer border.

    Rule:
    - Each border seed block connects (8-connected through non-bg pixels) to a line drawing.
    - Find closed bg cavities whose boundary touches that line.
    - Flood-fill each such cavity with the seed block's color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    out = g.copy()

    # Precompute background regions (4-connected). Exclude those containing background corners as "outside".
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

    for seed_pts in _seed_components(g, bg):
        seed_col = int(g[seed_pts[0]])

        neigh_cols = []
        for sr, sc in seed_pts:
            for nr, nc in neighbors4(sr, sc):
                if 0 <= nr < h and 0 <= nc < w:
                    v = int(g[nr, nc])
                    if v != bg and v != seed_col:
                        neigh_cols.append(v)
        assert neigh_cols, "seed block must have at least one non-bg non-seed neighbor"
        line_col = Counter(neigh_cols).most_common(1)[0][0]

        traced = _nonbg_component8_from(g, seed_pts[0], bg)

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


# --------------- generator (attempt_1_distractor_network) ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
SEED_BLOCK_LENGTHS = [2, 2, 3, 3]
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

DISTRACTOR_NETWORK_PROB = 0.6
DISTRACTOR_NETWORK_TRIES = 50
DISTRACTOR_LINE_LEN = (5, 15)
DISTRACTOR_SOLO_MAX = 2
DISTRACTOR_NETWORK_FRAMES = (1, 3)

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
            on_bdr = dr == 0 or dr == H - 1 or dc == 0 or dc == W - 1
            corner = (dr in (0, H - 1)) and (dc in (0, W - 1))
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(canvas, ring, cavity, line_col, h, w):
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


def _corner_seed_block(corner_kind, h, w, rng):
    length = rng.choice(SEED_BLOCK_LENGTHS)
    orientation = rng.choice(["h", "v"])
    if corner_kind == "TL":
        if orientation == "h":
            seed = [(0, c) for c in range(length)]
            first = (1, length - 1)
        else:
            seed = [(r, 0) for r in range(length)]
            first = (length - 1, 1)
    elif corner_kind == "TR":
        if orientation == "h":
            seed = [(0, w - 1 - c) for c in range(length)]
            first = (1, w - length)
        else:
            seed = [(r, w - 1) for r in range(length)]
            first = (length - 1, w - 2)
    elif corner_kind == "BL":
        if orientation == "h":
            seed = [(h - 1, c) for c in range(length)]
            first = (h - 2, length - 1)
        else:
            seed = [(h - 1 - r, 0) for r in range(length)]
            first = (h - length, 1)
    elif corner_kind == "BR":
        if orientation == "h":
            seed = [(h - 1, w - 1 - c) for c in range(length)]
            first = (h - 2, w - length)
        else:
            seed = [(h - 1 - r, w - 1) for r in range(length)]
            first = (h - length, w - 2)
    else:
        raise ValueError(corner_kind)
    return seed, first


def _distribute_loop_budget(num_seeds, num_attached_target, rng):
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
        seed_pts, first_rc = _corner_seed_block(corners_used[i], h, w, rng)
        if any(canvas[r, c] != -1 for r, c in seed_pts):
            return None
        if canvas[first_rc] != -1:
            return None
        for r, c in seed_pts:
            canvas[r, c] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng)
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_pts, line_colors[i], path))

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

    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(line_colors), frame_colors_pool, rng)
    else:
        for _ in range(rng.randint(0, DISTRACTOR_SOLO_MAX)):
            _try_attach_distractor_frame(canvas, rng.choice(frame_colors_pool), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    seen_seed_cols = set()
    for seed_col, seed_pts, _, _ in lines_info:
        seed_count = int((grid == seed_col).sum())
        if seed_count != len(seed_pts) or seed_count not in (2, 3):
            return None
        seen_seed_cols.add(seed_col)

    seed_components = _seed_components(grid, bg)
    if len(seed_components) != num_seeds:
        return None
    if {int(grid[pts[0]]) for pts in seed_components} != seen_seed_cols:
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("generate_input: exhausted attempts")
