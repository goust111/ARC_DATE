# Variant of c4d067a0
# Dimension: D3 — 目标位置：色块放置区域 (dst)
# Change: 网格中通过缩放因子计算出的对应位置，每个图例像素映射为一个 NxN 的实心色块，色块之间以背景色分隔... → 放置到框架内部的洞中

import numpy as np
from functools import reduce
from math import gcd

from arc_utils import bbox_coords, components_mask, get_bg
import random

# Design analysis
# spec_variant:
#   preserved_structure:
#     - 左上角仍然有规则步长=2 的单像素图例，图例坐标仍是主区域布局的缩小版。
#     - 图例仍是 3-4 行、2-3 列的阶梯形连通点集，颜色数仍受控在 2-4。
#     - 输入里仍只暴露两个相邻锚点颜色，用于通过颜色+相对位置投票推断偏移。
#   d3_core_change:
#     original_dst:
#       - 每个图例像素直接映射到主区域中的一个实心 NxN 色块坐标。
#     variant_dst:
#       - 主区域先摆放一批统一的空心方框；图例颜色不再直接画到裸背景上，
#         而是填入对应方框内部的洞。
#       - 输入中所有方框都已存在，但只有 2 个相邻方框的洞已被填色，其余洞为空。
#       - 输出保持方框边框不变，只补全所有空洞的内部颜色。
#   concrete_behavior_trace:
#     - 例: 图例格点 (rr=1, cc=2) 在原题中会直接映射到某个 block top-left，
#       然后整块 block_size x block_size 区域被涂成图例颜色。
#     - 在本变体中，同一 (rr, cc) 仍映射到同一主区域晶格，但该晶格位置现在对应
#       一个空心方框的“洞左上角”。solver 先定位那个方框，再只填充方框内部的
#       inner_size x inner_size 洞，不覆盖方框边框。
#     - 差异不在图例本身，也不在偏移推断，而在最终着色目标：原题是“往背景上放块”，
#       这里是“往既有框架的内部填洞”。
#   complexity_calibration:
#     - 如果直接把空心框当作可选目标而不保留锚点，任务会退化为纯形状对齐，失去原题
#       的“用局部已知块反推全局偏移”的特征。
#     - 如果只放 1 个空框或所有洞都已填满，D3 变化不可观察。因此固定为“所有位置都有框，
#       但仅 2 个相邻洞预填”，确保输入能肉眼看出“框存在、洞待填”的新机制。
#     - 方框统一为 1 像素边框、统一外尺寸，减少自由度；洞尺寸只取 2 或 3，
#       既明显又不会把 23-27 的画布挤满。
#   implicit_assumption_audit:
#     - 原代码把主区域对象识别为“整块同色正方形”，这在变体中失效，因为主区域主体
#       变成“同色边框 + 中央非同色洞”。因此 block 提取逻辑必须改为 frame 提取逻辑。
#     - 原代码用 block 的 top-left 推导 step；这里空洞未填时看不到实心块，所以必须
#       改为用所有方框的洞坐标推导 step。
#     - 原代码最终可直接覆盖目标区域；这里若直接写整个外框 bbox 会抹掉边框，因此只能
#       写 inner hole 区域。
#     - 原代码默认主区域已有若干实心块；这里输入里只有 2 个洞被填色，但所有方框都要
#       提前摆好，否则“匹配图例位置与框架位置”的 D3 变化无法发生。
#   placement_legality:
#     - 生成器保证所有方框外框互不重叠、与左上图例按列分隔，且每个洞都完整落在画布内。
#     - 方框只放在图例实际存在的阶梯格点上，不在图例空位处生成框，避免 solver 被错误
#       地引导为完整矩形布局。
#     - 预填的两个洞对应同一行的相邻格点，沿用原题唯一偏移投票机制。
#   visual_observability:
#     - 输入肉眼可见：左上是单像素图例；右侧/下侧是一串空心框，其中只有两个框内部有色。
#       输出则是所有空心框内部都按图例颜色填满。
#     - 颜色本身不参与排序或数值运算；唯一依赖颜色的步骤是“按图例颜色与已填锚点颜色匹配”。
#   cleanliness:
#     - 图例与主区域分栏放置，且主区域只包含一种边框色，视觉层次清晰。
#     - 边框色不出现在图例调色板中，避免把“边框”误读成“需传播的图例颜色”。


def _is_square_frame(sub: np.ndarray, color: int) -> bool:
    """Return True iff `sub` is a 1-cell-thick square frame of `color`."""
    hh, ww = sub.shape
    if hh != ww or hh < 4:
        return False
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    return not np.any(sub[1:-1, 1:-1] == color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a framed block layout so the holes match the minimap legend in the top-left corner.

    Input structure:
    - bg = most common color.
    - Top-left corner: isolated single-pixel legend dots on a regular lattice.
    - Main area: a lattice of same-size hollow square frames; only two adjacent frames have their inner holes already filled.

    Rule:
    - The legend dots still define the color pattern on lattice coordinates.
    - The two pre-filled frame holes determine how legend coordinates align with the frame lattice.
    - For every legend pixel, fill the interior hole of the corresponding frame with that legend color, keeping frame borders unchanged.

    Key insight: The original scale/offset inference still applies, but the target is no longer a bare square on the canvas — it is the empty interior of a pre-existing frame at the matched lattice position.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))

    assert singles, "grid must contain legend pixels"

    # Legend pixels: singleton colored pixels in a small top-left window.
    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    # Infer legend lattice step (usually 2).
    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    stepL = reduce(gcd, drs + dcs)
    assert stepL > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    for r, c, v in legend:
        rr = (r - rmin) // stepL
        cc = (c - cmin) // stepL
        legend_pts.append((int(rr), int(cc), int(v)))

    # Detect hollow square frames by scanning monochrome components color-by-color.
    frame_candidates: list[tuple[int, int, int, int]] = []  # (r0,c0,outer_size,frame_color)
    for col in sorted(set(map(int, np.unique(g))) - {int(bg)}):
        for pts in components_mask(g == int(col), connectivity=4):
            if len(pts) < 12:
                continue
            r0, c0, r1, c1 = bbox_coords(pts)
            hh, ww = r1 - r0 + 1, c1 - c0 + 1
            sub = g[r0 : r1 + 1, c0 : c1 + 1]
            if _is_square_frame(sub, int(col)):
                frame_candidates.append((int(r0), int(c0), int(hh), int(col)))

    assert frame_candidates, "grid must contain hollow square frames"

    # Dominant outer size and color identify the frame family used by the puzzle.
    sizes = [s for _, _, s, _ in frame_candidates]
    size_vals, size_cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    outer_size = int(size_vals[size_cnts.argmax()])
    frame_candidates = [f for f in frame_candidates if f[2] == outer_size]
    frame_colors = [col for _, _, _, col in frame_candidates]
    col_vals, col_cnts = np.unique(np.array(frame_colors, dtype=int), return_counts=True)
    frame_color = int(col_vals[col_cnts.argmax()])
    frames = [f for f in frame_candidates if f[3] == frame_color]
    assert frames, "must contain at least one dominant frame family"

    inner_size = outer_size - 2
    frame_by_hole: dict[tuple[int, int], tuple[int, int]] = {}
    filled_holes: list[tuple[int, int, int]] = []  # (hole_r0,hole_c0,fill_color)
    for fr0, fc0, _, _ in frames:
        hr0, hc0 = fr0 + 1, fc0 + 1
        frame_by_hole[(int(hr0), int(hc0))] = (int(fr0), int(fc0))
        inner = g[hr0 : hr0 + inner_size, hc0 : hc0 + inner_size]
        vals = {int(v) for v in np.unique(inner)}
        if len(vals) == 1:
            only = next(iter(vals))
            if only == bg:
                continue
            assert only != frame_color, "frame interior cannot reuse frame color"
            filled_holes.append((int(hr0), int(hc0), int(only)))
            continue
        raise AssertionError("frame interior must be either empty or uniformly pre-filled")

    assert filled_holes, "at least one frame hole must already be filled"

    # Compute step from frame-hole top-left positions.
    hole_rows = sorted({r0 for r0, _ in frame_by_hole})
    hole_cols = sorted({c0 for _, c0 in frame_by_hole})
    step_candidates = []
    step_candidates += [hole_rows[i + 1] - hole_rows[i] for i in range(len(hole_rows) - 1)]
    step_candidates += [hole_cols[i + 1] - hole_cols[i] for i in range(len(hole_cols) - 1)]
    assert step_candidates, "frame lattice must contain repeated rows or columns"
    step = reduce(gcd, step_candidates)
    assert step > 0, "frame lattice step must be positive"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for hr0, hc0, col in filled_holes:
        by_color.setdefault(int(col), []).append((int(hr0), int(hc0)))
    for col in by_color:
        by_color[col].sort()

    # Infer base offset exactly as in the original task, but using pre-filled hole origins.
    bases: dict[tuple[int, int], int] = {}
    for rr, cc, col in legend_pts:
        if col not in by_color:
            continue
        for hr0, hc0 in by_color[col]:
            base_r = hr0 - rr * step
            base_c = hc0 - cc * step
            bases[(base_r, base_c)] = bases.get((base_r, base_c), 0) + 1
    assert bases, "pre-filled holes must determine the legend-to-frame offset"
    max_votes = max(bases.values())
    best_bases = [base for base, votes in bases.items() if votes == max_votes]
    assert len(best_bases) == 1, "legend-to-frame offset must be unique"
    base_r, base_c = best_bases[0]

    # Fill every mapped frame hole; the frame border stays untouched.
    for rr, cc, col in legend_pts:
        tr0 = base_r + rr * step
        tc0 = base_c + cc * step
        assert 0 <= tr0 <= h - inner_size and 0 <= tc0 <= w - inner_size, \
            "legend hole projection must fit in canvas"
        assert (tr0, tc0) in frame_by_hole, "every legend point must map to an existing frame hole"
        out[tr0 : tr0 + inner_size, tc0 : tc0 + inner_size] = col
    return out


# --------------- generator (d3-hole-fill) ---------------

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

# === CFG: frame area ===
INNER_SIZE_RANGE = (2, 3)            # hole side length
FRAME_GAP = 1                        # gap between outer frame boxes
BASE_HOLE_C_MIN = 8                  # keeps frames visually separate from legend
BASE_HOLE_R_MIN = 2

# === CFG: generation ===
MAX_RETRIES = 400


def _enumerate_staircase_shapes():
    """Enumerate all valid (widths) tuples for the staircase legend shape."""
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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    """Pick a random staircase shape; return (filled_pts, adjacent_pairs)."""
    widths = random.choice(_STAIRCASE_SHAPES)
    pts = [(r, c) for r, w in enumerate(widths) for c in range(w)]
    pairs = [(r, c) for r, w in enumerate(widths) for c in range(w - 1)]
    return pts, pairs


def _random_color_counts(total, k):
    """Distribute `total` among `k` bins, each in [1, MAX_COUNT_PER_COLOR]."""
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, v in enumerate(counts) if v < MAX_COUNT_PER_COLOR]
        i = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[i]))
        counts[i] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    """Assign colors to legend pts while keeping the anchor pair uniquely identifiable."""
    pos_set = set(pts)
    first = anchor_pair
    second = (anchor_pair[0], anchor_pair[1] + 1)
    a, b = random.sample(palette, 2)
    counts = dict(zip(palette, _random_color_counts(len(pts), len(palette))))
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


def _draw_frame(canvas: np.ndarray, fr0: int, fc0: int, outer_size: int, frame_color: int) -> None:
    """Draw a hollow square frame with 1-cell border."""
    canvas[fr0, fc0 : fc0 + outer_size] = frame_color
    canvas[fr0 + outer_size - 1, fc0 : fc0 + outer_size] = frame_color
    canvas[fr0 : fr0 + outer_size, fc0] = frame_color
    canvas[fr0 : fr0 + outer_size, fc0 + outer_size - 1] = frame_color


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    inner_size = random.randint(*INNER_SIZE_RANGE)
    outer_size = inner_size + 2
    step = outer_size + FRAME_GAP
    pts, pairs = _pick_staircase()
    anchor_pair = random.choice(pairs)

    bg = random.randrange(NUM_COLORS)
    frame_color = random.choice([c for c in range(NUM_COLORS) if c != bg])
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    palette_pool = [c for c in range(NUM_COLORS) if c not in (bg, frame_color)]
    if len(palette_pool) < k_min:
        return None
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts), len(palette_pool)))
    palette = random.sample(palette_pool, k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_r = max(r for r, _ in pts)
    max_c = max(c for _, c in pts)
    r_hi = size - inner_size - 1 - max_r * step
    c_hi = size - inner_size - 1 - max_c * step
    if r_hi < BASE_HOLE_R_MIN or c_hi < BASE_HOLE_C_MIN:
        return None
    base_r = random.randint(BASE_HOLE_R_MIN, r_hi)
    base_c = random.randint(BASE_HOLE_C_MIN, c_hi)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    for rr, cc in pts:
        hr, hc = base_r + rr * step, base_c + cc * step
        _draw_frame(out, hr - 1, hc - 1, outer_size, frame_color)
        out[hr : hr + inner_size, hc : hc + inner_size] = legend_colors[(rr, cc)]

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for rr, cc in pts:
        if (rr, cc) in keep:
            continue
        hr, hc = base_r + rr * step, base_c + cc * step
        inp[hr : hr + inner_size, hc : hc + inner_size] = bg

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input():
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid c4d067a0 D3 input")
