# Combo variant of c4d067a0 (index 1)
# combo_spec: D1/3+D4/0
# Name: 旋转映射+颜色映射变换

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
      - 左上角仍是 step=2 的单像素楼梯形图例，图例本身不旋转，且只使用 2 种源色。
      - 主区域仍只显示 2 个同尺寸的锚点方块；它们对应的是“先旋转图例格点，再按固定 color map 换色”后的结果。
      - 因为原始锚点是一对左右相邻的图例点，旋转后这 2 个可见块可能变成上下相邻、左右次序反转，或保持水平但方向改变。
    trace_example:
      - 设图例中原始锚点是 (1,0,A) 与 (1,1,B)。
      - 原题会直接投影到 (base_r+1*step, base_c+0*step, A) 和 (base_r+1*step, base_c+1*step, B)。
      - 本融合题若旋转 90 度且颜色映射 A->X, B->Y，则先得到旋转格点 (W-1,1) 与 (W-2,1)，再投影成一对竖直相邻块，颜色分别是 X、Y。
      - 因此 solver 不能再先用颜色找 base、再单独谈旋转；它必须联合枚举 `(rotation, base, color_map)`。
  b_complexity_calibration:
    - 旋转只允许 90/180/270 三档，避免自由度失控。
    - 颜色映射限制为恰好 2 个源色到 2 个目标色的一一映射，这样仅凭 2 个锚点就能暴露完整映射。
    - 目标色集合与图例色集合强制分离，使“换色”肉眼可见；同时拒绝旋转后彩图例与原彩图例完全一致的样本，使“旋转”也肉眼可见。
  c_implicit_assumption_audit:
    - 原题的 `legend_pts -> base + (rr,cc)*step` 默认未旋转；这里必须改成 `rotate(rr,cc) -> base + rotated*step`。
    - 原题的颜色投票假设 `block_color == legend_color`；这里不成立，候选必须携带 `source_color -> target_color`。
    - 原题只看横向图例锚点与同色块；这里旋转后锚点可能竖直，所以 solver 需要检查四邻接的有向相邻对，而不是只看向右的相邻对。
    - 原题绘制阶段直接保留图例颜色；这里若不经过 color_map，会静默退化成纯旋转版本。
  d_placement_legality:
    - 生成器先在图例坐标系里造楼梯形，再旋转为主区域坐标，随后依据旋转后最大行列坐标计算合法 base，保证所有块不越界。
    - `base_c >= 8` 继续保留，确保主区域与左上图例分离。
    - 输入只删除非锚点块，不改动图例像素，也不引入重叠或额外标记。
  e_visual_observability:
    - 90/270 度样本会把原本水平锚点变成竖直块对；180 度样本会让左右次序翻转。
    - 由于目标色与图例色不重合，解题者能直接观察到“块颜色不是照抄图例，而是固定换色”。
    - 两种源色都会重复出现，多处缺失块重复使用同一映射，便于从少量样例中归纳。
  f_visual_cleanliness:
    - 图例仍局限在左上 12x12 小窗，主区是规则方块阵列，区域职责清晰。
    - 不叠加新装饰层，不引入噪声对象；视觉变化只来自旋转后的空间布局和统一的换色规则。
  g_change_inferability:
    - 颜色映射是题内固定的一一映射，并且 2 个源色都在锚点中出现，所以两块可见方块已经给出完整映射证据。
    - 旋转方向由“锚点从水平变成哪种相邻方向”以及整张图例在主区中的投影形状共同约束。
    - 两条线索交织：若旋转猜错，锚点方向/位置对不上；若颜色映射猜错，锚点颜色对不上。
  h_new_solver_assumptions_checked_against_generator:
    - solver 假设图例只有 2 种源色且可见块有 2 种目标色；generator 固定按此生成。
    - solver 假设可见块在块格上构成一对步长为 `step` 的四邻接块；generator 总是保留旋转后对应的一对相邻锚点。
    - solver 假设正确解在 `(rotation, base, color_map)` 空间中唯一；generator 用 `main(inp) == out` 反验，不唯一就重试。
    - solver 假设旋转后布局与原图例彩图样不同；generator 显式拒绝该类退化样本。
"""


def _rotate_coord(rr: int, cc: int, height: int, width: int, k: int) -> tuple[int, int]:
    """Rotate a lattice coordinate inside a height x width box by k*90° CCW."""
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
    return [
        (*_rotate_coord(rr, cc, height, width, k), int(col))
        for rr, cc, col in legend_pts
    ]


def _rotate_color_grid(colors: dict[tuple[int, int], int], k: int) -> dict[tuple[int, int], int]:
    height = max(r for r, _ in colors) + 1
    width = max(c for _, c in colors) + 1
    rotated: dict[tuple[int, int], int] = {}
    for (rr, cc), col in colors.items():
        rotated[_rotate_coord(rr, cc, height, width, k)] = int(col)
    return rotated


def _color_signature(colors: dict[tuple[int, int], int]) -> tuple[tuple[int, int, int], ...]:
    return tuple(sorted((int(r), int(c), int(col)) for (r, c), col in colors.items()))


def _directed_neighbor_pairs(
    pos_to_color: dict[tuple[int, int], int], step: int
) -> list[tuple[int, int, int, int, int, int]]:
    pairs: list[tuple[int, int, int, int, int, int]] = []
    for (r, c), col in sorted(pos_to_color.items()):
        for dr, dc in ((0, step), (step, 0), (0, -step), (-step, 0)):
            nxt = (r + dr, c + dc)
            if nxt in pos_to_color:
                pairs.append((int(r), int(c), int(dr), int(dc), int(col), int(pos_to_color[nxt])))
    return pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a block layout whose legend must be rotated and recolored before projection.

    Input structure:
    - The background is the most common color.
    - Top-left: a sparse single-pixel legend on a regular lattice, shown in its original orientation and using two source colors.
    - Main area: two or more uniformly colored square blocks on a coarser lattice; these visible blocks are both rotated copies of legend positions and recolored versions of legend pixels.

    Rule:
    - Convert legend pixels to lattice coordinates, then test the 90/180/270-degree rotations of that coordinate pattern.
    - For each rotation, try to align a directed adjacent legend pair with the visible adjacent block pair, which simultaneously proposes a base offset and a two-color source-to-target mapping.
    - After the unique rotation, offset, and color map are identified, fill every missing NxN block at the rotated positions using the mapped block colors.

    Key insight: The anchor blocks are evidence for two coupled hidden choices at once: which rotated legend adjacency they come from, and how the legend's two colors are remapped in the enlarged block area.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
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

    assert singles and blocks, "grid must contain legend pixels and main blocks"

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
    stepL = reduce(gcd, diffs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))
    legend_colors = sorted({col for _, _, col in legend_pts})
    assert len(legend_colors) == 2, "this combo variant expects exactly two legend colors"

    sizes = [s for _, _, s, _ in blocks]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    block_size = int(vals[cnts.argmax()])
    blocks = [b for b in blocks if b[2] == block_size]
    assert len(blocks) >= 2, "must contain at least two anchor blocks"

    tops = sorted({r0 for r0, _, _, _ in blocks})
    lefts = sorted({c0 for _, c0, _, _ in blocks})
    step_candidates = [tops[i + 1] - tops[i] for i in range(len(tops) - 1)]
    step_candidates += [lefts[i + 1] - lefts[i] for i in range(len(lefts) - 1)]
    assert step_candidates, "visible blocks must expose a block-lattice step"
    step = reduce(gcd, step_candidates)
    assert step > 0, "block step must be positive"

    block_by_pos = {(r0, c0): int(col) for r0, c0, _s, col in blocks}
    block_pairs = _directed_neighbor_pairs(block_by_pos, step)
    assert block_pairs, "visible blocks must contain an adjacent anchor pair"

    def candidate_valid(
        rotated_by_pos: dict[tuple[int, int], int],
        base_r: int,
        base_c: int,
        mapping: dict[int, int],
    ) -> bool:
        if set(mapping.keys()) != set(legend_colors):
            return False
        if len(set(mapping.values())) != len(mapping):
            return False
        seen_visible = 0
        visible_lookup = {(r0, c0): int(dst_col) for r0, c0, _s, dst_col in blocks}
        for r0, c0, _s, dst_col in blocks:
            dr = r0 - base_r
            dc = c0 - base_c
            if dr < 0 or dc < 0 or dr % step != 0 or dc % step != 0:
                return False
            pos = (dr // step, dc // step)
            src_col = rotated_by_pos.get(pos)
            if src_col is None or mapping.get(src_col) != dst_col:
                return False
        for (rr, cc), src_col in rotated_by_pos.items():
            tr0 = base_r + rr * step
            tc0 = base_c + cc * step
            if not (0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size):
                return False
            sub = g[tr0 : tr0 + block_size, tc0 : tc0 + block_size]
            mapped_col = mapping[src_col]
            if np.all(sub == bg):
                continue
            if np.all(sub == mapped_col) and visible_lookup.get((tr0, tc0)) == mapped_col:
                seen_visible += 1
                continue
            return False
        if seen_visible != len(blocks):
            return False
        return True

    candidates: dict[
        tuple[int, int, int, tuple[tuple[int, int], ...]],
        tuple[int, int, int, dict[int, int], list[tuple[int, int, int]]],
    ] = {}
    for rot_k in (1, 2, 3):
        rotated_pts = _rotate_legend_points(legend_pts, rot_k)
        rotated_by_pos = {(rr, cc): col for rr, cc, col in rotated_pts}
        legend_pairs = _directed_neighbor_pairs(rotated_by_pos, 1)
        for rr, cc, dr, dc, src_a, src_b in legend_pairs:
            if src_a == src_b:
                continue
            for br, bc, bdr, bdc, dst_a, dst_b in block_pairs:
                if dst_a == dst_b:
                    continue
                if (dr, dc) != (bdr // step, bdc // step):
                    continue
                mapping = {int(src_a): int(dst_a), int(src_b): int(dst_b)}
                base_r = br - rr * step
                base_c = bc - cc * step
                if not candidate_valid(rotated_by_pos, base_r, base_c, mapping):
                    continue
                key = (
                    int(rot_k),
                    int(base_r),
                    int(base_c),
                    tuple(sorted((int(k), int(v)) for k, v in mapping.items())),
                )
                candidates[key] = (int(rot_k), int(base_r), int(base_c), dict(mapping), rotated_pts)

    assert len(candidates) == 1, "rotation, offset, and recolor map must be uniquely determined"
    rot_k, base_r, base_c, color_map, rotated_pts = next(iter(candidates.values()))

    for rr, cc, src_col in rotated_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - block_size and 0 <= tc0 <= w - block_size, (
            "rotated legend block projection must fit in canvas"
        )
        out[tr0 : tr0 + block_size, tc0 : tc0 + block_size] = color_map[src_col]
    return out


# --------------- generator ---------------

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
MIN_COUNT_PER_COLOR, MAX_COUNT_PER_COLOR = 2, 6

# === CFG: block area ===
BLOCK_SIZE_RANGE = range(2, 5)
GAP_RANGE = range(1, 4)
STEP_MIN, STEP_MAX = 3, 5
BASE_C_MIN = 8
BASE_R_MIN = 1

# === CFG: generation ===
MAX_RETRIES = 300
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
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    pairs = [(r, c) for r, width in enumerate(widths) for c in range(width - 1)]
    return pts, pairs


def _two_color_counts(total: int) -> list[int]:
    lo = max(MIN_COUNT_PER_COLOR, total - MAX_COUNT_PER_COLOR)
    hi = min(MAX_COUNT_PER_COLOR, total - MIN_COUNT_PER_COLOR)
    assert lo <= hi
    first = random.randint(lo, hi)
    counts = [first, total - first]
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _two_color_counts(len(pts))))
    colors = {first: a, second: b}
    counts[a] -= 1
    counts[b] -= 1
    if counts[a] < 0 or counts[b] < 0:
        return None

    anchor_colors = (a, b)
    cells = [p for p in pts if p not in colors]

    def allowed(pos, col):
        r, c = pos
        left = (r, c - 1)
        if left in pos_set and left in colors:
            if (colors[left], col) == anchor_colors and (left, pos) != (first, second):
                return False
        right = (r, c + 1)
        if right in pos_set and right in colors:
            if (col, colors[right]) == anchor_colors and (pos, right) != (first, second):
                return False
        return True

    def search():
        if not cells:
            return all(v == 0 for v in counts.values())
        best_i, best_opts = None, None
        for i, pos in enumerate(cells):
            opts = [col for col in palette if counts[col] > 0 and allowed(pos, col)]
            if best_opts is None or len(opts) < len(best_opts):
                best_i, best_opts = i, opts
        if not best_opts:
            return False
        pos = cells.pop(best_i)
        random.shuffle(best_opts)
        for col in best_opts:
            colors[pos] = col
            counts[col] -= 1
            if search():
                return True
            counts[col] += 1
            del colors[pos]
        cells.insert(best_i, pos)
        return False

    if not search():
        return None
    return colors


def _gen_attempt(return_meta: bool = False):
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    block_size, gap = random.choice(_BLOCK_GAP_PAIRS)
    step = block_size + gap
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)
    rot_k = random.choice(ROTATIONS)

    bg = random.randrange(NUM_COLORS)
    legend_palette = random.sample([c for c in range(NUM_COLORS) if c != bg], LEGEND_PALETTE_SIZE)
    legend_colors = _assign_legend_colors(pts, anchor_pair, legend_palette)
    if legend_colors is None:
        return None

    target_pool = [c for c in range(NUM_COLORS) if c != bg and c not in legend_palette]
    target_palette = random.sample(target_pool, LEGEND_PALETTE_SIZE)
    color_map = dict(zip(legend_palette, target_palette))

    rotated_colors = _rotate_color_grid(legend_colors, rot_k)
    if _color_signature(rotated_colors) == _color_signature(legend_colors):
        return None

    max_r = max(r for r, _ in rotated_colors)
    max_c = max(c for _, c in rotated_colors)
    r_hi = size - block_size - max_r * step - 1
    c_hi = size - block_size - max_c * step - 1
    if r_hi < BASE_R_MIN or c_hi < BASE_C_MIN:
        return None
    base_r = random.randint(BASE_R_MIN, r_hi)
    base_c = random.randint(BASE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), src_col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = src_col
    for (rr, cc), src_col in rotated_colors.items():
        br, bc = base_r + rr * step, base_c + cc * step
        out[br : br + block_size, bc : bc + block_size] = color_map[src_col]

    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    legend_h = max(r for r, _ in legend_colors) + 1
    legend_w = max(c for _, c in legend_colors) + 1
    keep = {
        _rotate_coord(first[0], first[1], legend_h, legend_w, rot_k),
        _rotate_coord(second[0], second[1], legend_h, legend_w, rot_k),
    }

    inp = out.copy()
    for rr, cc in rotated_colors:
        if (rr, cc) in keep:
            continue
        br, bc = base_r + rr * step, base_c + cc * step
        inp[br : br + block_size, bc : bc + block_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    if return_meta:
        return {
            "input": inp,
            "output": out,
            "rotation": int(rot_k),
            "color_map": {int(k): int(v) for k, v in color_map.items()},
            "legend_colors": {tuple(map(int, pos)): int(col) for pos, col in legend_colors.items()},
            "rotated_colors": {tuple(map(int, pos)): int(col) for pos, col in rotated_colors.items()},
            "base_r": int(base_r),
            "base_c": int(base_c),
            "step": int(step),
            "block_size": int(block_size),
            "bg": int(bg),
            "legend_signature": _color_signature(legend_colors),
            "rotated_signature": _color_signature(rotated_colors),
        }
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt(return_meta=False)
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
