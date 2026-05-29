# Combo variant of c4d067a0 (index 19)
# combo_spec: D1/3+D4/0+D10/0
# Name: 旋转+颜色映射+单锚点（三元组）

from __future__ import annotations

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


"""
design_analysis:
  a_concretization_and_trace:
    baseline_trace:
      - 原题把左上角 step=2 的单像素图例，直接按 `(rr, cc) -> (base_r + rr*step, base_c + cc*step)` 放大成同色方块。
      - 输入里有两块相邻锚点，因此原 solver 能同时从“颜色对应 + 位置差分”直接读出 offset 与 block step。
    fused_trace:
      - 新变体仍保留左上角楼梯状图例，但图例只用两种颜色，其中一种只出现 1 次，另一种出现 5-9 次。
      - 主区域只保留 1 个可见 NxN 方块。这个方块不是原色复制，而是那颗“唯一色”像素在旋转后、再经过两色互换映射后的结果。
      - 例: 图例里只有 1 个红点，其余都是绿点。若采用 270 度旋转，则这颗红点先从 `(0,2)` 变到 `(2,3)`，再映射到主区唯一可见锚点；颜色则从红换成绿。解题时必须先意识到 visible block 对应“唯一色像素”，再枚举旋转和 step，直到整个旋转后的图例能合法落进主区。
    difference_from_baseline:
      - 坐标映射不再是原向投影，而是先做 90/180/270 度旋转。
      - 颜色不再保留，而是固定做两色互换。
      - 锚点从 2 块降到 1 块，因此 step 不能再靠差分读取，只能离散枚举并用全局合法性筛选。
  b_complexity_calibration:
    - 若保留 3-4 种图例色，单锚点不足以恢复整张颜色映射；因此收紧为“恰好 2 色 + 互换映射”。
    - 若两种颜色都多次出现，则唯一锚点无法知道自己对应哪一颗 legend 像素；因此强制其中一种颜色只出现 1 次，使锚点来源唯一。
    - step 仍限制在 3-5，旋转仅限 90/180/270 三档，避免自由度爆炸。
    - 形状继续使用不规则 staircase，避免旋转后视觉上过于接近原向。
  c_implicit_assumption_audit:
    - 原代码假设 `block color == legend color`；融合后失效，必须改成基于“2 色互换”的显式颜色规则。
    - 原代码假设至少 2 个 block，且 block 的行列差能给出 step；融合后失效，step 必须从合法候选集合中枚举。
    - 原代码假设 `(rr, cc)` 直接投影；融合后必须先旋转。
    - 原代码对 base 只做颜色投票；融合后一个锚点无法投票，只能把 `(rotation, step, base)` 作为联合候选，并检查整张投影是否越界、是否侵入图例区、是否与唯一锚点颜色一致。
  d_placement_legality:
    - generator 先生成完整输出，再只保留“唯一色像素”对应的一个锚点块。
    - 所有 block 都满足 `step = block_size + gap` 且 `gap in [1,3]`，因此块间不会重叠。
    - 主区最左列保持在图例右侧，且所有投影 block 都不能覆盖左上角 legend 单像素。
    - 生成时显式搜索 base，使 solver 恰好得到唯一候选。
  e_visual_observability:
    - 肉眼可见的核心线索有三个：主区只有一块、块颜色与 legend 中那颗唯一色不相同、块位置像是把整张 staircase 旋转后“钉”在画布上的某个角。
    - 由于两色互换且唯一色只出现一次，观察者能从“唯一色没有在主区复现、反而变成另一色锚点”归纳出换色规则。
    - 旋转不能靠颜色 id 数值推断，而是靠 staircase 轮廓在不同朝向下与画布边界/图例区的相容性来推断。
  f_visual_cleanliness:
    - 仍维持左上 legend 与右侧/下方 block 区分离的双区布局。
    - 主区只含规则纯色方块，无额外噪声；唯一锚点让观察重点非常集中。
  g_color_inference:
    - 颜色映射固定是“两色互换”，不用依赖任何不可视的颜色编号顺序。
    - 图例两色之一只出现一次，另一色重复出现，因此一个锚点即可暴露完整映射，而输出中仍会出现大量互换后的重复方块，帮助人类确认规则。
  new_solver_assumptions_checked_against_generator:
    - solver 假设图例恰好 2 色，且“与锚点目标色相反的源色”在图例中恰好出现 1 次；generator 强制满足。
    - solver 假设唯一解存在于 `rotation in {1,2,3}` 与合法 `step` 枚举中；generator 通过搜索 base 和 `main(inp) == out` 反验保证。
    - solver 假设主区 block 不会覆盖 legend 单像素，且整体位于 legend 右侧；generator 放置时显式保证。
"""


# === CFG: canvas ===
GRID_SIZE_MIN, GRID_SIZE_MAX = 23, 27
NUM_COLORS = 10

# === CFG: legend lattice ===
LEGEND_ORIGIN = (1, 1)
LEGEND_STEP = 2
LEGEND_ROWS = (3, 4)
LEGEND_COLS = (2, 3)
LEGEND_EMPTIES = (0, 1, 2)
LEGEND_FILLED_MIN, LEGEND_FILLED_MAX = 6, 10
LEGEND_PALETTE_SIZE = 2

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1
ROTATIONS = (1, 2, 3)

# === CFG: generation ===
MAX_RETRIES = 600
BASE_SEARCH_TRIES = 40


def _rotate_coord(rr: int, cc: int, height: int, width: int, k: int) -> tuple[int, int]:
    k %= 4
    if k == 0:
        return int(rr), int(cc)
    if k == 1:
        return int(width - 1 - cc), int(rr)
    if k == 2:
        return int(height - 1 - rr), int(width - 1 - cc)
    return int(cc), int(height - 1 - rr)


def _rotate_legend_points(
    legend_pts: list[tuple[int, int, int]], k: int
) -> list[tuple[int, int, int]]:
    height = max(rr for rr, _, _ in legend_pts) + 1
    width = max(cc for _, cc, _ in legend_pts) + 1
    rotated = []
    for rr, cc, col in legend_pts:
        nr, nc = _rotate_coord(rr, cc, height, width, k)
        rotated.append((int(nr), int(nc), int(col)))
    return rotated


def _extract_singles_and_blocks(
    g: np.ndarray, bg: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    blocks: list[tuple[int, int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 2:
            continue
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        col = int(sub[0, 0])
        if col == bg:
            continue
        if np.all(sub == col) and sub.size == len(pts):
            blocks.append((int(r0), int(c0), int(hh), int(col)))
    return singles, blocks


def _legend_from_singles(
    singles: list[tuple[int, int, int]], h: int, w: int
) -> tuple[list[tuple[int, int, int]], dict[tuple[int, int], int], int, int, int]:
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    diffs = drs + dcs
    assert diffs, "legend lattice must expose a measurable step"
    step_l = reduce(gcd, diffs)
    assert step_l > 0, "legend lattice step must be positive"

    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)
    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
    legend_by_pos = {(rr, cc): col for rr, cc, col in legend_pts}
    return legend_pts, legend_by_pos, step_l, rmin, cmin


def _step_candidates(block_size: int) -> list[int]:
    return [
        step
        for step in range(STEP_MIN, STEP_MAX + 1)
        if 1 <= step - block_size <= max(GAP_RANGE)
    ]


def _candidate_valid(
    rotated_pts: list[tuple[int, int, int]],
    step: int,
    base_r: int,
    base_c: int,
    block_size: int,
    grid_shape: tuple[int, int],
    legend_canvas_pts: list[tuple[int, int, int]],
) -> bool:
    h, w = grid_shape
    projected: set[tuple[int, int]] = set()
    min_c = None
    for rr, cc, _col in rotated_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if tr0 < 0 or tc0 < 0 or tr0 > h - block_size or tc0 > w - block_size:
            return False
        if (tr0, tc0) in projected:
            return False
        projected.add((int(tr0), int(tc0)))
        min_c = tc0 if min_c is None else min(min_c, tc0)
        for lr, lc, _ in legend_canvas_pts:
            if tr0 <= lr < tr0 + block_size and tc0 <= lc < tc0 + block_size:
                return False
    assert min_c is not None
    if min_c < BASE_C_MIN:
        return False
    return True


def _find_candidates(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    singles, blocks = _extract_singles_and_blocks(g, bg)
    assert singles and blocks, "grid must contain legend pixels and main blocks"

    legend_pts, legend_by_pos, _step_l, _rmin, _cmin = _legend_from_singles(singles, h, w)
    color_counts: dict[int, int] = {}
    for _rr, _cc, col in legend_pts:
        color_counts[col] = color_counts.get(col, 0) + 1
    assert len(color_counts) == LEGEND_PALETTE_SIZE, "legend must contain exactly two colors"

    rare_colors = [col for col, cnt in color_counts.items() if cnt == 1]
    assert len(rare_colors) == 1, "exactly one legend color must appear once"
    rare_color = int(rare_colors[0])
    common_color = int(next(col for col, cnt in color_counts.items() if cnt != 1))
    swap_map = {rare_color: common_color, common_color: rare_color}

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) == 1, "single-anchor variant expects exactly one visible block"
    br0, bc0, _s, block_color = blocks[0]

    anchor_src_color = next((src for src, dst in swap_map.items() if dst == block_color), None)
    assert anchor_src_color is not None, "anchor color must be explained by the two-color swap"
    anchor_src_positions = [
        (rr, cc) for rr, cc, col in legend_pts if col == anchor_src_color
    ]
    assert len(anchor_src_positions) == 1, "visible block must correspond to the unique legend color"
    anchor_src = anchor_src_positions[0]

    height = max(rr for rr, _, _ in legend_pts) + 1
    width = max(cc for _, cc, _ in legend_pts) + 1

    candidates: list[dict[str, object]] = []
    seen_keys: set[tuple[int, int, int, int]] = set()
    for rot_k in ROTATIONS:
        rotated_pts = _rotate_legend_points(legend_pts, rot_k)
        anchor_rr, anchor_cc = _rotate_coord(anchor_src[0], anchor_src[1], height, width, rot_k)
        for step in _step_candidates(block_size):
            base_r = int(br0 - anchor_rr * step)
            base_c = int(bc0 - anchor_cc * step)
            if not _candidate_valid(
                rotated_pts, step, base_r, base_c, block_size, (h, w), singles
            ):
                continue
            key = (int(rot_k), int(step), int(base_r), int(base_c))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidates.append(
                {
                    "rotation": int(rot_k),
                    "step": int(step),
                    "base_r": int(base_r),
                    "base_c": int(base_c),
                    "block_size": int(block_size),
                    "rotated_pts": rotated_pts,
                    "swap_map": dict(swap_map),
                    "rare_color": int(rare_color),
                    "common_color": int(common_color),
                }
            )
    return candidates


def _render_solution(grid: np.ndarray, candidate: dict[str, object]) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    block_size = int(candidate["block_size"])
    base_r = int(candidate["base_r"])
    base_c = int(candidate["base_c"])
    step = int(candidate["step"])
    swap_map = dict(candidate["swap_map"])
    rotated_pts = list(candidate["rotated_pts"])
    h, w = out.shape

    for rr, cc, src_col in rotated_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "rotated legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = swap_map[int(src_col)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a sparse block layout whose single anchor comes from a rotated, recolored legend.

    Input structure:
    - The background is the most frequent color.
    - Top-left: a staircase-like legend of singleton pixels on a regular lattice, using exactly two colors, with one color appearing only once.
    - Main area: exactly one visible NxN square block; it belongs to a rotated copy of the legend and already uses the swapped target color.

    Rule:
    - Interpret the lone block as the rotated projection of the legend's unique-color pixel.
    - Enumerate the allowed 90/180/270-degree rotations and the small set of possible block-lattice steps, then derive the matching offset from that single anchor.
    - Keep only the candidate whose full rotated projection fits the canvas, stays clear of the legend, and respects the fixed two-color swap; then paint every missing block with swapped colors.

    Key insight: one anchor is enough because the unique legend color identifies which source pixel survived, the two legend colors must swap, and the remaining uncertainty is a tiny geometric search over rotation and scale.
    """
    candidates = _find_candidates(np.asarray(grid, dtype=int))
    assert len(candidates) == 1, "rotation, step, and offset must be uniquely determined"
    return _render_solution(np.asarray(grid, dtype=int), candidates[0])


def _enumerate_staircase_shapes():
    shapes = []
    for rows in LEGEND_ROWS:
        for cols in LEGEND_COLS:
            total = rows * cols
            for empties in LEGEND_EMPTIES:
                count = total - empties
                if not (LEGEND_FILLED_MIN <= count <= LEGEND_FILLED_MAX):
                    continue

                def rec(prefix, remaining, r):
                    if r == rows:
                        if remaining == 0 and max(prefix) == cols:
                            shapes.append(tuple(prefix))
                        return
                    left = rows - r - 1
                    lo = max(1, remaining - left * cols)
                    hi = min(cols, remaining - left)
                    for width in range(lo, hi + 1):
                        rec(prefix + [width], remaining - width, r + 1)

                rec([], count, 0)
    return shapes


_BLOCK_GAP_PAIRS = [
    (n, g) for n in BLOCK_SIZE_RANGE for g in GAP_RANGE if STEP_MIN <= n + g <= STEP_MAX
]
_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    return pts


def _coord_signature(pts: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in pts))


def _rotate_positions(pts: list[tuple[int, int]], k: int) -> list[tuple[int, int]]:
    height = max(r for r, _ in pts) + 1
    width = max(c for _, c in pts) + 1
    return [_rotate_coord(r, c, height, width, k) for r, c in pts]


def _pick_unique_cell(pts: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    preferred = [
        p for p in pts if p[0] in (0, max_r) or p[1] in (0, max_c)
    ]
    return random.choice(preferred or pts)


def _legend_coloring(
    pts: list[tuple[int, int]], unique_cell: tuple[int, int], palette: list[int]
) -> dict[tuple[int, int], int]:
    rare_color, common_color = random.sample(palette, 2)
    colors = {pos: int(common_color) for pos in pts}
    colors[unique_cell] = int(rare_color)
    return colors


def _render_output(
    size: int,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    rot_k: int,
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int], dict[int, int]]:
    lr0, lc0 = LEGEND_ORIGIN
    unique_cell = next(pos for pos, col in legend_colors.items() if list(legend_colors.values()).count(col) == 1)
    rare_color = legend_colors[unique_cell]
    common_color = next(col for pos, col in legend_colors.items() if pos != unique_cell)
    color_map = {int(rare_color): int(common_color), int(common_color): int(rare_color)}

    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = int(col)

    legend_pts = [(rr, cc, col) for (rr, cc), col in legend_colors.items()]
    rotated_pts = _rotate_legend_points(legend_pts, rot_k)
    anchor_rot = _rotate_coord(
        unique_cell[0],
        unique_cell[1],
        max(r for r, _ in legend_colors) + 1,
        max(c for _, c in legend_colors) + 1,
        rot_k,
    )
    for rr, cc, src_col in rotated_pts:
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = color_map[int(src_col)]
    return out, unique_cell, anchor_rot, color_map


def _base_candidates(r_hi: int, c_hi: int) -> list[tuple[int, int]]:
    rows = list(range(BASE_R_MIN, r_hi + 1))
    cols = list(range(BASE_C_MIN, c_hi + 1))
    if not rows or not cols:
        return []
    head_rows = rows[: min(len(rows), 5)]
    head_cols = cols[: min(len(cols), 6)]
    tail_rows = rows[-min(len(rows), 3) :]
    tail_cols = cols[-min(len(cols), 3) :]
    pool = {(r, c) for r in head_rows for c in head_cols}
    pool |= {(r, c) for r in tail_rows for c in head_cols}
    pool |= {(r, c) for r in head_rows for c in tail_cols}
    pool = list(pool)
    random.shuffle(pool)
    return pool[:BASE_SEARCH_TRIES]


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts = _pick_staircase()
    if any(_coord_signature(_rotate_positions(pts, k)) == _coord_signature(pts) for k in ROTATIONS):
        return None

    unique_cell = _pick_unique_cell(pts)
    rot_k = random.choice(ROTATIONS)
    bg = random.randrange(NUM_COLORS)
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], LEGEND_PALETTE_SIZE)
    legend_colors = _legend_coloring(pts, unique_cell, palette)

    rotated_pts = _rotate_positions(pts, rot_k)
    max_r = max(r for r, _ in rotated_pts)
    max_c = max(c for _, c in rotated_pts)
    r_hi = size - block_size - max_r * step
    c_hi = size - block_size - max_c * step
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None

    for base_r, base_c in _base_candidates(r_hi, c_hi):
        out, _unique_cell, anchor_rot, _color_map = _render_output(
            size=size,
            bg=bg,
            legend_colors=legend_colors,
            rot_k=rot_k,
            block_size=block_size,
            step=step,
            base_r=base_r,
            base_c=base_c,
        )
        inp = out.copy()
        for rr, cc in rotated_pts:
            if (rr, cc) == anchor_rot:
                continue
            br = base_r + rr * step
            bc = base_c + cc * step
            inp[br : br + block_size, bc : bc + block_size] = bg

        try:
            candidates = _find_candidates(inp)
            if len(candidates) != 1:
                continue
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp
    return None


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
