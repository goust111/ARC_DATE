# Combo variant of c4d067a0 (index 3)
# combo_spec: D1/3+D10/0
# Name: 旋转映射+单锚点参考

from __future__ import annotations

import random
from functools import reduce
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


"""
design_analysis:
  a_concretization_and_trace:
    input_view:
      - 左上角仍是 step=2 的单像素楼梯图例，颜色不变，也不旋转显示。
      - 主区域不再保留两个块，而是只保留一个 NxN 纯色方块作为唯一锚点。
      - 真正的块阵列对应“先旋转图例，再按 block-step 放大并平移”的结果。
    trace_example:
      - 假设图例里唯一出现一次的颜色位于格点 (0,2)，其方块边长是 3，所以 block-step 只可能是 4 或 5。
      - 若实际旋转为 90 度，则该锚点会变成旋转后格点 (0,0) 或 (0,?) 的某个位置，再映射到唯一已知块的左上角。
      - solver 不能再像原题那样用两个块的间距算 step；它必须枚举少量 `(rotation, step)`，把锚点反投影成 base，再检查整套投影是否完整落在主区域且不会压到左上角图例。
      - 与原题差异在于：原题依赖“多块间距 + 颜色投票”，融合后改为“单块反推 + 边界/占用合法性筛选”。
  b_complexity_calibration:
    - 旋转仍只允许 90/180/270 三档，避免自由度失控。
    - step 仍保持原题 3-5 的离散范围，并由 `block_size + gap` 给出；单锚点下只枚举少量合法 step。
    - 锚点颜色在图例中强制唯一出现，让人和 solver 都能先锁定“这是哪一个图例像素”，再去判断旋转与尺度。
    - 图例保持不规则楼梯形，避免旋转后几何轮廓过于对称，导致多个方向肉眼不可区分。
  c_implicit_assumption_audit:
    - 原代码用多个块顶点差分的 gcd 推出 block-step；单锚点后这个前提失效，必须改成显式枚举少量 step 候选。
    - 原代码把候选只写成 `(base_r, base_c)`；旋转加入后候选至少要扩成 `(rotation, step, base_r, base_c)`。
    - 原代码默认“只要颜色能投票出一个 base 就够了”；单锚点后颜色票数太稀疏，必须把“其余投影位置都为空、所有投影都不盖到图例、所有块都在画布内”并入判定。
    - 原代码绘制阶段直接使用未旋转坐标；融合后必须绘制旋转后的格点集合，否则会静默回退到原题行为。
  d_placement_legality:
    - 生成器先渲染完整输出，再只留下一个锚点块构成输入。
    - 所有块都放在 `base_c >= 8` 的主区域，避免覆盖左上 12x12 图例窗口。
    - 生成器逐个尝试合法 base，只接受那些在 solver 的 `(rotation, step)` 枚举里恰好产生唯一候选的样本。
  e_visual_observability:
    - 唯一锚点颜色在图例里只出现一次，解题者能先把单块和图例中的那一个像素对应起来。
    - 真实 base 会被放到靠边的位置，使错误旋转或错误 step 往往越界，或把块阵列压进图例区；这些差异在输入输出样例里肉眼可见。
    - 生成器额外拒绝任何“未旋转也合法”的样本，确保旋转维度不是摆设。
  f_visual_cleanliness:
    - 图例和块区继续分区显示，没有新增标记、噪声或叠层对象。
    - 输入里主区域只有一个方块，视觉焦点很明确；输出则是同一方块族的完整补全，不会产生杂乱重叠。
  g_color_inference:
    - 不做颜色替换，所有新块直接继承图例像素颜色。
    - 颜色结构只承担“唯一锚点定位”这一项职责，不叠加额外颜色规则，避免少样本下不可归纳。
  new_solver_assumptions_checked_against_generator:
    - solver 假设输入里至少有一个图例像素和一个主区方块；generator 始终满足。
    - solver 假设主区只需关注 dominant block size；generator 始终只保留一个该尺寸方块。
    - solver 假设在 `rotation in {1,2,3}` 与离散 step 候选中存在且仅存在一个合法投影；generator 用同一套候选检查和 `main(inp) == out` 反验。
    - solver 假设合法投影不会覆盖任何现有非背景像素，除了那唯一锚点块本身；generator 的放置逻辑保证这一点。
"""


def _rotate_coord(rr: int, cc: int, height: int, width: int, k: int) -> tuple[int, int]:
    """Rotate a lattice coord inside a height x width box by k*90 degrees CCW."""
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


def _extract_scene(
    grid: np.ndarray,
) -> tuple[np.ndarray, int, list[tuple[int, int, int]], list[tuple[int, int, int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

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
            blocks.append((int(r0), int(c0), int(hh), col))

    win_r = min(g.shape[0], 12)
    win_c = min(g.shape[1], 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    diffs = drs + dcs
    assert diffs, "legend lattice must expose a measurable step"
    stepL = reduce(gcd, diffs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
    return g, bg, legend_pts, blocks


def _dominant_block_size(blocks: list[tuple[int, int, int, int]]) -> int:
    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    return int(vals[cnts.argmax()])


def _candidate_steps(block_size: int) -> list[int]:
    return sorted(
        {
            block_size + gap
            for gap in GAP_RANGE
            if STEP_MIN <= block_size + gap <= STEP_MAX
        }
    )


def _projection_valid(
    g: np.ndarray,
    bg: int,
    rotated_pts: list[tuple[int, int, int]],
    anchor: tuple[int, int, int, int],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> bool:
    h, w = g.shape
    anchor_r, anchor_c, _anchor_s, anchor_col = anchor
    anchor_hits = 0

    for rr, cc, col in rotated_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
            return False
        region = g[tr0 : tr0 + block_size, tc0 : tc0 + block_size]
        if tr0 == anchor_r and tc0 == anchor_c:
            if col != anchor_col or not np.all(region == col):
                return False
            anchor_hits += 1
        else:
            if np.any(region != bg):
                return False
    return anchor_hits == 1


def _collect_candidates(
    g: np.ndarray,
    bg: int,
    legend_pts: list[tuple[int, int, int]],
    blocks: list[tuple[int, int, int, int]],
    rotations: tuple[int, ...],
) -> tuple[int, list[tuple[int, int, int, int, list[tuple[int, int, int]]]]]:
    assert blocks, "grid must contain at least one main block"
    block_size = _dominant_block_size(blocks)
    anchors = [b for b in blocks if b[2] == block_size]
    assert len(anchors) == 1, "single-anchor variant expects exactly one dominant block"
    anchor = anchors[0]

    candidates: dict[
        tuple[int, int, int, int], list[tuple[int, int, int]]
    ] = {}
    for rot_k in rotations:
        rotated_pts = _rotate_legend_points(legend_pts, rot_k)
        for step in _candidate_steps(block_size):
            for rr, cc, col in rotated_pts:
                if col != anchor[3]:
                    continue
                base_r = anchor[0] - rr * step
                base_c = anchor[1] - cc * step
                if not _projection_valid(
                    g, bg, rotated_pts, anchor, block_size, step, base_r, base_c
                ):
                    continue
                candidates[(int(rot_k), int(step), int(base_r), int(base_c))] = rotated_pts

    result = [
        (rot_k, step, base_r, base_c, rotated_pts)
        for (rot_k, step, base_r, base_c), rotated_pts in candidates.items()
    ]
    result.sort(key=lambda item: item[:4])
    return block_size, result


def _render_blocks(
    out: np.ndarray,
    rotated_pts: list[tuple[int, int, int]],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> np.ndarray:
    painted = out.copy()
    for rr, cc, col in rotated_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        painted[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = col
    return painted


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a rotated block layout from a single visible anchor block.

    Input structure:
    - bg is the dominant background color.
    - The top-left corner contains a sparse single-pixel legend on a step-2 lattice in its original orientation.
    - The main area contains exactly one uniformly colored square block, which is one enlarged pixel from a rotated copy of the legend.

    Rule:
    - Convert the legend pixels into lattice coordinates, then test 90/180/270-degree rotations of that coordinate pattern.
    - For each rotation, test the few block-step values allowed by the square block size, using the lone block as the only anchor to back-solve the base offset.
    - Keep the unique candidate whose full projected block family fits on the canvas and does not cover any existing non-background cells except the anchor itself, then paint every missing block.

    Key insight: The single block is not enough for pairwise spacing inference, so the solution comes from matching that lone color anchor and rejecting every rotation/scale placement that would force the rotated pattern out of bounds or into the legend.
    """
    g, bg, legend_pts, blocks = _extract_scene(grid)
    assert blocks, "grid must contain legend pixels and one main block"
    block_size, candidates = _collect_candidates(g, bg, legend_pts, blocks, ROTATIONS)
    assert candidates, "single anchor must determine a valid rotated projection"
    assert len(candidates) == 1, "rotation, step, and base offset must be unique"
    rot_k, step, base_r, base_c, rotated_pts = candidates[0]
    _ = rot_k
    return _render_blocks(g, rotated_pts, block_size, step, base_r, base_c)


# --------------- generator (rotation + single-anchor variant) ---------------

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 400
ROTATIONS = (1, 2, 3)


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
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _random_color_counts(total: int, k: int) -> list[int]:
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, value in enumerate(counts) if value < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(
    pts: list[tuple[int, int]],
    anchor_pt: tuple[int, int],
    palette: list[int],
) -> dict[tuple[int, int], int]:
    anchor_color = random.choice(palette)
    other_colors = [col for col in palette if col != anchor_color]
    assert other_colors, "single-anchor variant needs at least two colors"

    counts = _random_color_counts(len(pts) - 1, len(other_colors))
    colors = {anchor_pt: int(anchor_color)}
    bag: list[int] = []
    for col, count in zip(other_colors, counts):
        bag.extend([int(col)] * int(count))
    random.shuffle(bag)

    rest = [pos for pos in pts if pos != anchor_pt]
    random.shuffle(rest)
    for pos, col in zip(rest, bag):
        colors[pos] = int(col)
    return colors


def _rotate_color_map(
    colors: dict[tuple[int, int], int], k: int
) -> dict[tuple[int, int], int]:
    height = max(r for r, _ in colors) + 1
    width = max(c for _, c in colors) + 1
    rotated: dict[tuple[int, int], int] = {}
    for (rr, cc), col in colors.items():
        nr, nc = _rotate_coord(rr, cc, height, width, k)
        rotated[(int(nr), int(nc))] = int(col)
    return rotated


def _color_signature(colors: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(col)) for (r, c), col in colors.items()))


def _legend_points_from_map(colors: dict[tuple[int, int], int]) -> list[tuple[int, int, int]]:
    return [(int(r), int(c), int(col)) for (r, c), col in sorted(colors.items())]


def _candidate_metrics(inp: np.ndarray) -> dict[str, object]:
    g, bg, legend_pts, blocks = _extract_scene(inp)
    block_size, rotated_candidates = _collect_candidates(g, bg, legend_pts, blocks, ROTATIONS)
    _block_size_0, all_candidates = _collect_candidates(g, bg, legend_pts, blocks, (0, 1, 2, 3))
    unrotated = [cand for cand in all_candidates if cand[0] == 0]
    return {
        "block_size": int(block_size),
        "rotated_candidates": rotated_candidates,
        "all_candidates": all_candidates,
        "unrotated_count": len(unrotated),
    }


def _build_case(
    size: int,
    bg: int,
    legend_colors: dict[tuple[int, int], int],
    rotated_colors: dict[tuple[int, int], int],
    anchor_rot: tuple[int, int],
    block_size: int,
    step: int,
    base_r: int,
    base_c: int,
) -> tuple[np.ndarray, np.ndarray]:
    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
    for (rr, cc), col in rotated_colors.items():
        br = base_r + rr * step
        bc = base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = col

    inp = out.copy()
    for rr, cc in rotated_colors:
        if (rr, cc) == anchor_rot:
            continue
        br = base_r + rr * step
        bc = base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg
    return inp, out


def _gen_attempt(return_meta: bool = False):
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts = _pick_staircase()
    anchor_pt = random.choice(pts)
    rot_k = random.choice(ROTATIONS)

    bg = random.randrange(NUM_COLORS)
    k_min = max(
        PALETTE_SIZE_MIN,
        1 + (len(pts) - 1 + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR,
    )
    k_colors = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    palette = random.sample([c for c in range(NUM_COLORS) if c != bg], k_colors)
    legend_colors = _assign_legend_colors(pts, anchor_pt, palette)
    rotated_colors = _rotate_color_map(legend_colors, rot_k)
    if _color_signature(rotated_colors) == _color_signature(legend_colors):
        return None

    legend_h = max(r for r, _ in legend_colors) + 1
    legend_w = max(c for _, c in legend_colors) + 1
    anchor_rot = _rotate_coord(anchor_pt[0], anchor_pt[1], legend_h, legend_w, rot_k)

    max_r = max(r for r, _ in rotated_colors)
    max_c = max(c for _, c in rotated_colors)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None

    bases = [(r, c) for r in range(BASE_R_MIN, r_hi + 1) for c in range(BASE_C_MIN, c_hi + 1)]
    random.shuffle(bases)
    bases.sort(
        key=lambda rc: min(rc[0] - BASE_R_MIN, r_hi - rc[0], rc[1] - BASE_C_MIN, c_hi - rc[1])
    )

    for base_r, base_c in bases:
        inp, out = _build_case(
            size, bg, legend_colors, rotated_colors, anchor_rot, block_size, step, base_r, base_c
        )
        metrics = _candidate_metrics(inp)
        rotated_candidates = metrics["rotated_candidates"]
        if len(rotated_candidates) != 1:
            continue
        cand_rot, cand_step, cand_base_r, cand_base_c, _rotated_pts = rotated_candidates[0]
        if (cand_rot, cand_step, cand_base_r, cand_base_c) != (rot_k, step, base_r, base_c):
            continue
        if metrics["unrotated_count"] != 0:
            continue
        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, out):
            continue
        if return_meta:
            return {
                "input": inp,
                "output": out,
                "rotation": int(rot_k),
                "step": int(step),
                "block_size": int(block_size),
                "anchor_color": int(legend_colors[anchor_pt]),
                "unrotated_count": int(metrics["unrotated_count"]),
            }
        return inp
    return None


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(return_meta=False)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
