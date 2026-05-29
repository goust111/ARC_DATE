# Combo variant of 8b7bacbf (index 22)
# combo_spec: D3/3+D9/1+D13/0
# Name: 边界色匹配线路+种子线路同色+填充为框架色

# Design analysis
# ---------------
# concretization_and_trace:
#   - 融合后输入仍是: 角点 seed -> 8 连通细线路 -> 若干闭合框环 -> 框内背景 cavity。
#   - 但真实 seed 不再靠“全图唯一色”识别, 而是靠角点位置识别; seed 与本线路共享同色。
#   - 可填框环必须满足两层条件:
#       1. 它属于该角点出发的 traced non-bg 组件;
#       2. 它的整圈边界色必须等于这条线路色, 也就是等于该 seed 色。
#   - 填色来源按 D3/3 走“框环色”, 不是“seed 色”。在本组合里, 对所有真正可填框来说
#     frame_color == line_color == seed_color, 所以写入来源与写入数值合流成一个 shared color。
#   - 具体小例子:
#       * 左上角有颜色 2 的 seed, 对角内侧延出一条颜色 2 的线。
#       * 线上挂两个框: 一个边界是 2, 一个边界是 7。
#       * 原题会试图找唯一 seed 色并常常失效; 即使强行沿原题思路, 也不会要求框边界等于线色。
#       * 融合后只填边界为 2 的那个 cavity, 且写入的是框环色 2; 边界为 7 的 attached cavity 保持背景。
#
# complexity_calibration:
#   - 保留 1-2 个 seed 与多框结构, 不把任务退化成单对象着色题。
#   - 生成器强制“每个 seed 至少 1 个 matching loop + 全图至少 1 个 mismatching attached loop”,
#     让 D9 的筛选效应在 2-3 个样例内肉眼可见。
#   - 颜色自由度保持低: seed/line 颜色 1-2 种, 失配框颜色来自少量 extra frame colors 或另一条线色。
#
# implicit_assumption_audit:
#   - 原 `_seed_pixels()` 假设 seed 颜色全图只出现一次; D13/0 破坏它, 所以改为 `_corner_seed_pixels()`。
#   - 原“从 seed 的 4 邻统计 line color”假设 seed 与 line 异色且 4 邻接; D13/0 破坏它,
#     所以直接令 `line_col = seed_col`。
#   - 原“边界 8 邻到某个 line_color 像素即可”在本组合里不够, 因为 matching frame 与 line 同色,
#     边界邻到边界自身就会产生伪证据; 因此必须改成 `_has_external_line_contact()`:
#     要求 boundary pixel 8 邻到 traced 组件里、且不属于 full boundary 的同色像素, 即真正的外部线路接触。
#   - D3/3 要求填充色取自边界色。这里数值上等于 seed/line 色, 但代码仍显式从 `bcols` 取 `frame_col`
#     再写入, 避免把“来源反转”偷换回 seed 填色。
#
# placement_legality:
#   - seed 固定占据四角之一, 线路第一步取对角内侧像素, 既保证与 seed 8 连通, 又避免依赖 4 邻。
#   - 框环 placement 只允许接触预定 line_path 上的像素, 不能 8 邻到其他任何已占像素;
#     这是本组合最关键的几何约束, 因为 matching frame 与 line 同色, 不能靠颜色区分“合法接触”和“误粘连”。
#   - cavity 周围必须在放置前保持空白, 防止边界色集合混入别的对象或让 solver 看到脏腔体。
#
# observability_and_cleanliness:
#   - 解题者能直接看到两个稳定信号:
#       1. 角点 seed 与其线路同色;
#       2. 同样挂在线路上的框, 只有与线路同色的那些才被填。
#   - D3 的“填框色”在该组合下与 shared color 合流, 单靠输出无法和“填 seed 色”区分;
#     这是组合规格本身导致的数值塌缩, 不是生成退化。为避免真正退化, 生成器额外保证 D9 仍然可见:
#     `main(grid)` 必须与“不检查边界是否匹配线路色”的 relaxed baseline 不同。
#
# post_edit_main_assumptions:
#   - 非背景角点就是 seed; generator 通过边缘留白和角点专用放置保证这一点。
#   - 每个 seed 至少连出一条同色线路; generator 通过对角首像素和最短长度检查保证。
#   - 每个被判定为 matching 的 cavity, 其整圈边界色是单色且等于对应 line color。
#   - 若某 cavity 被认为 attached, 则其 boundary 一定能在 traced 组件里找到一个不属于 boundary 的外部线路接触点。

import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import components_mask, get_bg, neighbors4, neighbors8


def _components_mask(mask: np.ndarray, conn8: bool) -> list[list[tuple[int, int]]]:
    return [[(int(r), int(c)) for r, c in comp] for comp in components_mask(mask, connectivity=8 if conn8 else 4)]


def _corner_seed_pixels(g: np.ndarray, bg: int) -> list[tuple[int, int]]:
    h, w = g.shape
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    return [(r, c) for r, c in corners if int(g[r, c]) != bg]


def _touches_border(pts: set[tuple[int, int]] | list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)


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


def _collect_cavities(g: np.ndarray, bg: int) -> list[tuple[set[tuple[int, int]], set[tuple[int, int]], set[int]]]:
    h, w = g.shape
    bg_comps = _components_mask(g == bg, conn8=False)
    cavities = []
    for pts in bg_comps:
        region = set(pts)
        if _touches_border(region, h, w):
            continue
        boundary = set()
        for r, c in region:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in region and int(g[nr, nc]) != bg:
                    boundary.add((nr, nc))
        if boundary:
            cavities.append((region, boundary, {int(g[p]) for p in boundary}))
    return cavities


def _has_external_line_contact(
    g: np.ndarray,
    boundary_in_traced: set[tuple[int, int]],
    full_boundary: set[tuple[int, int]],
    traced: set[tuple[int, int]],
    line_col: int,
) -> bool:
    h, w = g.shape
    for r, c in boundary_in_traced:
        for nr, nc in neighbors8(r, c):
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in full_boundary or (nr, nc) not in traced:
                continue
            if int(g[nr, nc]) == line_col:
                return True
    return False


def _main_relaxed_samecolor_frame_fill(grid: np.ndarray) -> np.ndarray:
    """
    Relaxed baseline used only by the generator.

    It keeps D13 and D3, but removes D9's boundary-color match check.
    If combo samples are well-formed, this baseline should fill extra mismatching
    attached loops that the true combo solver leaves untouched.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _collect_cavities(g, bg)

    for seed in _corner_seed_pixels(g, bg):
        line_col = int(g[seed])
        traced = _nonbg_component8_from(g, seed, bg)
        for cav, boundary, bcols in cavities:
            if len(bcols) != 1:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            if not _has_external_line_contact(g, boundary_in_traced, boundary, traced, line_col):
                continue
            frame_col = next(iter(bcols))
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = frame_col
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill only the matched cavities in a same-color seed-line-frame system.

    Input structure:
    - bg is the most common color.
    - One or two non-bg corner pixels act as seeds.
    - Each seed is 8-connected to a meandering line of the same color.
    - Closed single-color frame rings sit along those traced drawings, enclosing bg cavities.

    Rule:
    - Use each non-bg corner as a seed and trace its full 8-connected non-bg component.
    - A cavity is fillable for that seed only when its boundary is monochrome and that boundary color equals the seed/line color.
    - The boundary must also have an external 8-neighbor contact with the traced line structure, not just with itself.
    - Fill every such cavity with the frame ring's color.

    Key insight:
    - The operative color is shared across seed, line, and every valid frame on that route.
    - Attached loops of the wrong color are deliberate distractors: connectivity alone is insufficient, color agreement with the route is required.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()
    cavities = _collect_cavities(g, bg)

    for seed in _corner_seed_pixels(g, bg):
        line_col = int(g[seed])
        traced = _nonbg_component8_from(g, seed, bg)

        for cav, boundary, bcols in cavities:
            if bcols != {line_col}:
                continue
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            if not _has_external_line_contact(g, boundary_in_traced, boundary, traced, line_col):
                continue

            frame_col = next(iter(bcols))
            for r, c in cav:
                if int(out[r, c]) == bg:
                    out[r, c] = frame_col
    return out


# --------------- generator ---------------

# ========================== CFG ==========================
GRID_H_RANGE = (8, 30)
GRID_W_RANGE = (8, 30)
BG_COLORS = [0, 5]

SEED_COUNT_WEIGHTS = [1, 2, 2]
MIN_LINE_LEN = 4
LINE_LEN_RANGE = (8, 35)

FRAME_COLORS_MAX = 3
TOTAL_LOOPS_MAX = 9
CAVITY_SIZE_WEIGHTS = [2, 2, 2, 2, 3]
DISTRACTOR_CAVITY_WEIGHTS = [2, 2, 3]
FRAME_PLACEMENT_PAD = 1
FRAME_ATTACH_TRIES = 45
DISTRACTOR_FRAME_TRIES = 45

DIAG_WEIGHT_ALIGNED = 5
DIAG_WEIGHT_OTHER = 1
ORTHO_WEIGHT_ALIGNED = 2
ORTHO_WEIGHT_OTHER = 1
BORDER_AVOIDANCE_PROB = 0.85
BIAS_FLIP_PROB = 0.3

MATCHING_FRAME_PROB = 0.6

DISTRACTOR_NETWORK_PROB = 0.45
DISTRACTOR_NETWORK_TRIES = 40
DISTRACTOR_LINE_LEN = (5, 14)
DISTRACTOR_SOLO_MAX = 1
DISTRACTOR_NETWORK_FRAMES = (1, 2)

GEN_RETRIES = 300
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
            on_bdr = dr == 0 or dc == 0 or dr == h - 1 or dc == w - 1
            corner = dr in (0, h - 1) and dc in (0, w - 1)
            if on_bdr and not corner:
                ring.append((r0 + dr, c0 + dc))
    cavity = [(r0 + 1 + dr, c0 + 1 + dc) for dr in range(ch) for dc in range(cw)]
    return ring, cavity


def _validate_frame_placement(
    canvas: np.ndarray,
    ring: list[tuple[int, int]],
    cavity: list[tuple[int, int]],
    allowed_touch: set[tuple[int, int]],
) -> bool:
    h, w = canvas.shape
    ring_set = set(ring)

    for rr, cc in ring:
        if canvas[rr, cc] != -1:
            return False
    for rr, cc in cavity:
        if canvas[rr, cc] != -1:
            return False

    for cr, cc in cavity:
        for nr, nc in neighbors8(cr, cc):
            if _in_bounds(h, w, nr, nc) and canvas[nr, nc] != -1:
                return False

    touches_line = False
    for rr, cc in ring:
        for nr, nc in neighbors8(rr, cc):
            if not _in_bounds(h, w, nr, nc) or (nr, nc) in ring_set:
                continue
            if canvas[nr, nc] == -1:
                continue
            if (nr, nc) not in allowed_touch:
                return False
            touches_line = True
    return touches_line


def _grow_line(
    grid: np.ndarray,
    start: tuple[int, int],
    line_col: int,
    target_len: int,
    rng: random.Random,
    allowed_contacts: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    path = [start]
    dr_bias = 1 if start[0] < h // 2 else -1
    dc_bias = 1 if start[1] < w // 2 else -1
    cur = start

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
            if (nr == 0 or nc == 0 or nr == h - 1 or nc == w - 1) and rng.random() < BORDER_AVOIDANCE_PROB:
                continue

            bad = False
            for nnr, nnc in neighbors8(nr, nc):
                if not _in_bounds(h, w, nnr, nnc):
                    continue
                if (nnr, nnc) == cur or (nnr, nnc) in allowed_contacts:
                    continue
                if grid[nnr, nnc] != -1:
                    bad = True
                    break
            if bad:
                continue

            cur = (nr, nc)
            path.append(cur)
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
    frame_col: int,
    rng: random.Random,
    max_tries: int = FRAME_ATTACH_TRIES,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    h, w = canvas.shape
    pad = FRAME_PLACEMENT_PAD
    allowed_touch = set(line_path)

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
            if not _validate_frame_placement(canvas, ring, cavity, allowed_touch):
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
        ch = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        cw = rng.choice(DISTRACTOR_CAVITY_WEIGHTS)
        fh, fw = ch + 2, cw + 2
        if h - fh - 2 < 1 or w - fw - 2 < 1:
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


def _try_place_distractor_network(
    canvas: np.ndarray,
    line_col: int,
    frame_colors: list[int],
    rng: random.Random,
) -> bool:
    h, w = canvas.shape
    for _ in range(DISTRACTOR_NETWORK_TRIES):
        r = rng.randint(2, h - 3)
        c = rng.randint(2, w - 3)
        if any(canvas[r + dr, c + dc] != -1 for dr in range(-1, 2) for dc in range(-1, 2)):
            continue
        canvas[r, c] = line_col
        path = _grow_line(canvas, (r, c), line_col, rng.randint(*DISTRACTOR_LINE_LEN), rng, set())
        if len(path) < 3:
            for pr, pc in path:
                canvas[pr, pc] = -1
            continue
        attached = 0
        for _ in range(rng.randint(*DISTRACTOR_NETWORK_FRAMES)):
            fc = rng.choice(frame_colors)
            if _try_attach_frame(canvas, path, fc, rng) is not None:
                attached += 1
        if attached > 0:
            return True
        for pr, pc in path:
            canvas[pr, pc] = -1
    return False


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


def _choose_mismatch_color(line_idx: int, line_colors: list[int], extra_frame_colors: list[int], rng: random.Random) -> int:
    options = [c for c in extra_frame_colors + line_colors if c != line_colors[line_idx]]
    return rng.choice(options)


def _build_frame_plan(
    num_seeds: int,
    line_colors: list[int],
    extra_frame_colors: list[int],
    total_attached_target: int,
    rng: random.Random,
) -> list[list[int]]:
    plans = [[line_colors[i]] for i in range(num_seeds)]

    mismatch_seed = rng.randrange(num_seeds)
    plans[mismatch_seed].append(_choose_mismatch_color(mismatch_seed, line_colors, extra_frame_colors, rng))

    remaining = total_attached_target - (num_seeds + 1)
    while remaining > 0:
        i = rng.randrange(num_seeds)
        if rng.random() < MATCHING_FRAME_PROB:
            plans[i].append(line_colors[i])
        else:
            plans[i].append(_choose_mismatch_color(i, line_colors, extra_frame_colors, rng))
        remaining -= 1
    return plans


def _variant_metrics(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    corner_seeds = _corner_seed_pixels(g, bg)
    cavities = _collect_cavities(g, bg)

    matching = 0
    mismatched = 0
    repeated_seed_colors = 0
    for seed in corner_seeds:
        line_col = int(g[seed])
        if int((g == line_col).sum()) > 1:
            repeated_seed_colors += 1
        traced = _nonbg_component8_from(g, seed, bg)
        for _, boundary, bcols in cavities:
            boundary_in_traced = boundary & traced
            if not boundary_in_traced:
                continue
            if not _has_external_line_contact(g, boundary_in_traced, boundary, traced, line_col):
                continue
            if len(bcols) != 1:
                continue
            if bcols == {line_col}:
                matching += 1
            else:
                mismatched += 1

    combo_out = main(g)
    relaxed_out = _main_relaxed_samecolor_frame_fill(g)
    combo_gap_cells = int(np.sum(combo_out != relaxed_out))
    filled_cells = int(np.sum((g == bg) & (combo_out != g)))
    return {
        "corner_seeds": len(corner_seeds),
        "repeated_seed_colors": repeated_seed_colors,
        "matching_attached": matching,
        "mismatched_attached": mismatched,
        "combo_gap_cells": combo_gap_cells,
        "filled_cells": filled_cells,
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    bg = rng.choice(BG_COLORS)
    palette = [c for c in range(10) if c != bg]
    rng.shuffle(palette)

    num_seeds = rng.choice(SEED_COUNT_WEIGHTS)
    if len(palette) < num_seeds + 1:
        return None

    canvas = -np.ones((h, w), dtype=int)
    seed_colors = [palette.pop() for _ in range(num_seeds)]
    line_colors = list(seed_colors)

    num_extra_frame_cols = rng.randint(1, min(FRAME_COLORS_MAX, len(palette)))
    extra_frame_colors = [palette.pop() for _ in range(num_extra_frame_cols)]

    all_corners = ["TL", "TR", "BL", "BR"]
    rng.shuffle(all_corners)
    corners_used = all_corners[:num_seeds]

    lines_info = []
    for i in range(num_seeds):
        seed_rc, first_rc = _corner_config(corners_used[i], h, w)
        if canvas[seed_rc] != -1 or canvas[first_rc] != -1:
            return None
        canvas[seed_rc] = seed_colors[i]
        canvas[first_rc] = line_colors[i]
        target_len = rng.randint(LINE_LEN_RANGE[0], min(LINE_LEN_RANGE[1], h + w))
        path = _grow_line(canvas, first_rc, line_colors[i], target_len, rng, {seed_rc})
        if len(path) < MIN_LINE_LEN:
            return None
        lines_info.append((seed_colors[i], seed_rc, line_colors[i], path))

    min_attached = max(2, num_seeds + 1)
    max_attached = min(TOTAL_LOOPS_MAX, num_seeds + 4)
    total_attached_target = rng.randint(min_attached, max_attached)
    frame_plans = _build_frame_plan(num_seeds, line_colors, extra_frame_colors, total_attached_target, rng)

    for i, (_, _, _, path) in enumerate(lines_info):
        for frame_col in frame_plans[i]:
            if _try_attach_frame(canvas, path, frame_col, rng) is None:
                return None

    distractor_frame_colors = extra_frame_colors + line_colors
    if rng.random() < DISTRACTOR_NETWORK_PROB:
        _try_place_distractor_network(canvas, rng.choice(distractor_frame_colors), distractor_frame_colors, rng)
    elif rng.random() < 0.5:
        _try_attach_distractor_frame(canvas, rng.choice(distractor_frame_colors), rng)

    grid = np.where(canvas == -1, bg, canvas).astype(int)

    vals, cnts = np.unique(grid, return_counts=True)
    if int(vals[np.argmax(cnts)]) != bg:
        return None

    for seed_col, seed_rc, _, _ in lines_info:
        if int((grid == seed_col).sum()) <= 1:
            return None
        if int(grid[seed_rc]) != seed_col:
            return None

    metrics = _variant_metrics(grid)
    if metrics["corner_seeds"] != num_seeds:
        return None
    if metrics["repeated_seed_colors"] != num_seeds:
        return None
    if metrics["matching_attached"] < num_seeds:
        return None
    if metrics["mismatched_attached"] < 1:
        return None
    if metrics["combo_gap_cells"] <= 0:
        return None
    if metrics["filled_cells"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
