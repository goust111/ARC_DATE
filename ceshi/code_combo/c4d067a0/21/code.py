# Combo variant of c4d067a0 (index 21)
# combo_spec: D3/3+D4/1+D11/0
# Name: 重力堆叠+棋盘格背景+空心框放大

import random
from functools import reduce
from itertools import combinations
from math import gcd

import numpy as np

from arc_utils import bbox_coords, components_mask


# Design analysis
# ===============
# spec_variant:
#   preserved_backbone:
#     - 仍保留原题的双层结构: 左上角 step=2 的单像素图例 + 右侧主区域的放大对象。
#     - 仍保留“输入只给两个相邻锚点，用它们反推整张布局”的信息结构。
#     - 图例形状仍使用原题的 staircase 连通点集，颜色数与填充数保持同量级。
#   fusion_concretized:
#     - D11/0 先作用在整个画布: 画布不再有单一背景色，而是完整的双色棋盘格。
#       图例单像素与主区域对象都只是叠加在该底纹上。
#     - D4/1 再改变放大对象的形状: 每个 legend 像素不再放成实心 NxN 方块，而是放成
#       同色的 1 像素厚空心方框，方框内部保持棋盘格不变。
#     - D3/3 最后改变目的位置: legend 坐标仍决定“投到哪一列”，但不再决定绝对行坐标。
#       同一 legend 列中的方框按 rr 从小到大依次落下，先投的停在更低处，后投的堆在上方。
#   concrete_trace:
#     - 例: outer_size=4, step=5, 某列 cc=0 有 legend 像素
#         (0,0)=红, (1,0)=蓝, (2,0)=黄。
#     - 原题会把 3 个 4x4 实心块放到
#         base_r+0*5, base_r+1*5, base_r+2*5
#       三者之间有 1 行背景间隔。
#     - 本融合版中，同样 3 个颜色仍投到同一主区域列，但它们都从顶部下落:
#         红框先落到底部，蓝框停在红框上方，黄框再停在蓝框上方；
#         每个对象只染外框 12 个像素，中间 2x2 仍显示棋盘格。
#     - 因此 solver 需要同时看懂三件事:
#         1. 哪两种颜色构成背景棋盘格。
#         2. 主区域对象是空心框而不是实心块。
#         3. legend 的 rr 表示下落顺序，而不是固定 y 坐标。
#   complexity_calibration:
#     - 棋盘格只用 2 种背景色，避免“先识别背景”这一步自由度过高。
#     - 空心框统一为 outer_size=4、边框厚度=1，既明显可见，又避免尺寸过多导致检测不稳。
#     - step 只取 5 或 6，保证列间留缝，且重力堆叠后的无竖向间隔效果能肉眼看到。
#     - 仍保持 2-4 种前景色、6-10 个 legend 点，复杂度接近原题，不退化成只读颜色计数。
#   implicit_assumption_audit:
#     - 原 solver 用“最常见颜色 = 背景”识别前景；这里失效，因为背景有两色且都很常见。
#       新 solver 必须先从全画布反推出棋盘格的两色与相位。
#     - 原 solver 把主区域对象识别为实心正方形；这里失效，因为对象是空心框，内部是背景。
#       新 solver 改为识别“单色边框 + 内部完全等于棋盘格”的正方框。
#     - 原 solver 依赖块的绝对 top-left 行列投票 base_r/base_c；重力后 base_r 不再存在。
#       新 solver 只能用两个相邻锚框恢复 block_size/step/base_c，并把 rr 当作投放顺序。
#     - 原 solver 直接整块覆盖目标区域；这里若覆盖整个 bbox 会抹掉空心结构。
#       新 solver 只能绘制方框边界，保持内部棋盘格原样。
#     - 若同一列任意两层颜色相同，而它们之间没有别的同色框隔开，重力后可能直接拼成
#       一个更高的同色组件，破坏“每个颜色边框对应一个对象”的可见性。
#       生成器因此禁止同一列重复使用同一种前景色。
#   placement_legality:
#     - 主区域列步长满足 step > outer_size，不同列之间始终有棋盘格缝隙，不会横向接触。
#     - outer_size=4、最高堆叠层数=4，在 23-27 的画布上总高度最多 16，始终能落下。
#     - base_c >= 8，主区域与左上图例分栏，避免框和图例互相接触。
#     - 输入只保留 row0 的一对相邻锚框；它们在完整输出中本来就是各自列的最底层，
#       删除其他框后不会留下悬空结构。
#   visual_observability:
#     - 输入里肉眼可见三种变化同时存在:
#         1. 背景是交替两色而不是纯底色。
#         2. 主区域对象是空心框，内部透出背景。
#         3. 输出中同列框会紧密竖直堆叠，而不是按 legend 行距散开。
#     - 这些规律都来自位置与形状，不依赖颜色编号大小等人类不可视因素。
#   cleanliness_and_inference:
#     - 图例与主区域分开摆放，背景底纹贯穿全图，使“空洞保留背景”一眼可见。
#     - 前景调色板明确排除两种背景色，避免把棋盘格误读为可传播的图例颜色。
#     - 由于第一列必然贯穿所有 legend 行，至少有一列会出现 3-4 层堆叠，重力行为不会退化。
#   main_new_assumptions:
#     - 输入的前景颜色都不复用棋盘格两色。
#     - 输入恰有两个主区域空心锚框，且它们对应 legend 顶行中的唯一一对相邻有序颜色对。
#     - 所有主区域对象共享相同 outer_size=4，内部严格保留原棋盘格。
#     - generate_input() 会构造并反向验算这些条件。


def _make_checkerboard(shape: tuple[int, int], color_a: int, color_b: int, phase: int) -> np.ndarray:
    """Build a checkerboard where parity^phase==0 uses color_a and the other uses color_b."""
    h, w = map(int, shape)
    parity = np.indices((h, w)).sum(axis=0) & 1
    return np.where((parity ^ int(phase)) == 0, int(color_a), int(color_b)).astype(int)


def _infer_checkerboard(grid: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """Recover the dominant 2-color checkerboard background and return (pattern, a, b, phase)."""
    colors = sorted(map(int, np.unique(grid)))
    best = None
    parity = np.indices(grid.shape).sum(axis=0) & 1
    for color_a, color_b in combinations(colors, 2):
        for phase in (0, 1):
            pattern = np.where((parity ^ phase) == 0, color_a, color_b)
            score = int(np.sum(grid == pattern))
            cand = (score, int(color_a), int(color_b), int(phase))
            if best is None or cand > best:
                best = cand
    assert best is not None, "grid must contain at least two colors"
    _score, color_a, color_b, phase = best
    pattern = _make_checkerboard(grid.shape, color_a, color_b, phase)
    return pattern, int(color_a), int(color_b), int(phase)


def _is_hollow_frame(sub: np.ndarray, bg_sub: np.ndarray, color: int) -> bool:
    """Return True iff sub is a 1-pixel-thick square frame whose hole matches the checkerboard."""
    hh, ww = sub.shape
    if hh != ww or hh < 3:
        return False
    if not np.all(sub[0, :] == color):
        return False
    if not np.all(sub[-1, :] == color):
        return False
    if not np.all(sub[:, 0] == color):
        return False
    if not np.all(sub[:, -1] == color):
        return False
    return bool(np.array_equal(sub[1:-1, 1:-1], bg_sub[1:-1, 1:-1]))


def _paint_frame(grid: np.ndarray, r0: int, c0: int, outer_size: int, color: int) -> None:
    """Paint only the border of a square frame, leaving the interior untouched."""
    s = int(outer_size)
    r0 = int(r0)
    c0 = int(c0)
    color = int(color)
    grid[r0, c0:c0 + s] = color
    grid[r0 + s - 1, c0:c0 + s] = color
    grid[r0:r0 + s, c0] = color
    grid[r0:r0 + s, c0 + s - 1] = color


def _drop_frame(grid: np.ndarray, bg_pattern: np.ndarray, outer_size: int, c0: int, color: int) -> int:
    """Drop a hollow frame within one projected column until the next step would collide."""
    h, w = grid.shape
    s = int(outer_size)
    c0 = int(c0)
    assert 0 <= c0 <= w - s

    def clear_at(r0: int) -> bool:
        return bool(np.array_equal(grid[r0:r0 + s, c0:c0 + s], bg_pattern[r0:r0 + s, c0:c0 + s]))

    r0 = 0
    assert clear_at(r0), "drop column must be clear at the top"
    while r0 + 1 <= h - s and clear_at(r0 + 1):
        r0 += 1
    _paint_frame(grid, r0, c0, s, color)
    return int(r0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a checkerboard scene by dropping hollow square frames according to the top-left legend.

    Input structure:
    - The canvas background is a full two-color checkerboard rather than a single bg color.
    - Top-left corner: isolated single-pixel legend dots on a regular lattice, overlaid on that checkerboard.
    - Main area: exactly two same-size hollow square frames already resting at the bottoms of their projected columns; their interiors remain checkerboard.

    Rule:
    - First recover the checkerboard background, then treat every non-checkerboard cell as foreground.
    - Match the two existing frames to the unique adjacent color pair in the legend, which fixes the frame size, projected column spacing, and horizontal offset.
    - For every other legend pixel, use its legend column to choose a main-area column and drop a same-color hollow frame from the top; process legend rows from top to bottom so earlier rows settle lower.

    Key insight: The legend still controls colors and columns, but three original assumptions are gone at once: the background is patterned, the enlarged object is hollow, and legend rows encode drop order instead of absolute y-position.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()

    bg_pattern, bg_a, bg_b, _phase = _infer_checkerboard(g)
    fg_mask = g != bg_pattern
    comps = components_mask(fg_mask, connectivity=4)

    singles: list[tuple[int, int, int]] = []
    frame_candidates: list[tuple[int, int, int, int]] = []  # (r0, c0, outer_size, color)
    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww:
            continue
        sub = g[r0:r1 + 1, c0:c1 + 1]
        bg_sub = bg_pattern[r0:r1 + 1, c0:c1 + 1]
        color = int(g[r0, c0])
        if color in {bg_a, bg_b}:
            continue
        if len(pts) == 4 * hh - 4 and _is_hollow_frame(sub, bg_sub, color):
            frame_candidates.append((int(r0), int(c0), int(hh), int(color)))

    assert singles, "grid must contain legend pixels"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"

    rs = sorted({r for r, _, _ in legend})
    cs = sorted({c for _, c, _ in legend})
    drs = [rs[i + 1] - rs[i] for i in range(len(rs) - 1)]
    dcs = [cs[i + 1] - cs[i] for i in range(len(cs) - 1)]
    step_l = reduce(gcd, drs + dcs)
    assert step_l > 0, "legend lattice step must be positive"
    rmin = min(r for r, _, _ in legend)
    cmin = min(c for _, c, _ in legend)

    legend_pts: list[tuple[int, int, int]] = []
    legend_color: dict[tuple[int, int], int] = {}
    for r, c, v in legend:
        rr = (r - rmin) // step_l
        cc = (c - cmin) // step_l
        legend_pts.append((int(rr), int(cc), int(v)))
        legend_color[(int(rr), int(cc))] = int(v)

    assert frame_candidates, "grid must contain hollow frame anchors"
    sizes = [s for _, _, s, _ in frame_candidates]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    outer_size = int(vals[cnts.argmax()])
    frames = [f for f in frame_candidates if f[2] == outer_size]
    assert len(frames) == 2, "fusion variant expects exactly two anchor frames"

    frames.sort(key=lambda item: (item[1], item[0]))
    left_frame, right_frame = frames
    assert left_frame[1] < right_frame[1], "anchor frames must occupy distinct columns"
    step = int(right_frame[1] - left_frame[1])
    assert step > outer_size, "projected frame columns must remain separated"

    pair_colors = (int(left_frame[3]), int(right_frame[3]))
    anchor_matches = []
    for rr, cc, col in legend_pts:
        right_col = legend_color.get((rr, cc + 1))
        if right_col is not None and (int(col), int(right_col)) == pair_colors:
            anchor_matches.append((int(rr), int(cc)))
    assert len(anchor_matches) == 1, "anchor frames must match a unique adjacent legend pair"

    anchor_rr, anchor_cc = anchor_matches[0]
    base_c = int(left_frame[1] - anchor_cc * step)
    assert int(right_frame[1]) == base_c + (anchor_cc + 1) * step

    keep = {(anchor_rr, anchor_cc), (anchor_rr, anchor_cc + 1)}
    for cc in sorted({cc for _, cc, _ in legend_pts}):
        col_items = sorted(
            [(rr, col) for rr, ccol, col in legend_pts if ccol == cc and (rr, ccol) not in keep],
            key=lambda item: item[0],
        )
        c0 = base_c + cc * step
        assert 0 <= c0 <= w - outer_size, "projected frame column must fit in canvas"
        for _rr, col in col_items:
            _drop_frame(out, bg_pattern, outer_size, c0, col)

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
PALETTE_SIZE_MIN, PALETTE_SIZE_MAX = 2, 4
MAX_COUNT_PER_COLOR = 6

# === CFG: frame area ===
OUTER_SIZE = 4
GAP_RANGE = (1, 2)
BASE_C_MIN = 8

# === CFG: generation ===
MAX_RETRIES = 500


def _enumerate_staircase_shapes():
    """Enumerate all valid staircase width tuples for the legend."""
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
_ANCHORABLE_SHAPES = [widths for widths in _STAIRCASE_SHAPES if widths[0] >= 2]


def _pick_staircase():
    """Pick a staircase shape and an anchor pair from its top row."""
    widths = random.choice(_ANCHORABLE_SHAPES)
    pts = [(r, c) for r, width in enumerate(widths) for c in range(width)]
    row0_pairs = [(0, c) for c in range(widths[0] - 1)]
    assert row0_pairs
    return pts, random.choice(row0_pairs)


def _random_color_counts(total: int, k: int) -> list[int]:
    """Distribute total cells among k colors, each count in [1, MAX_COUNT_PER_COLOR]."""
    counts = [1] * k
    remaining = total - k
    while remaining:
        open_idxs = [i for i, val in enumerate(counts) if val < MAX_COUNT_PER_COLOR]
        idx = random.choice(open_idxs)
        add = random.randint(1, min(remaining, MAX_COUNT_PER_COLOR - counts[idx]))
        counts[idx] += add
        remaining -= add
    random.shuffle(counts)
    return counts


def _assign_legend_colors(pts, anchor_pair, palette):
    """Backtrack a color pattern with unique anchor pair and no repeated color within a column."""
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

        for rr2, cc2 in pos_set:
            if cc2 == c and (rr2, cc2) in colors and colors[(rr2, cc2)] == col:
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


def _render_output(size: int, bg_a: int, bg_b: int, phase: int, step: int, base_c: int, legend_colors):
    """Render the full output and record where each frame landed."""
    out = _make_checkerboard((size, size), bg_a, bg_b, phase)
    lr0, lc0 = LEGEND_ORIGIN
    for (rr, cc), col in legend_colors.items():
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    bg_pattern = _make_checkerboard((size, size), bg_a, bg_b, phase)
    placed = {}
    for rr, cc in sorted(legend_colors.keys()):
        c0 = base_c + cc * step
        r0 = _drop_frame(out, bg_pattern, OUTER_SIZE, c0, legend_colors[(rr, cc)])
        placed[(rr, cc)] = (r0, c0)
    return out, bg_pattern, placed


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    step = OUTER_SIZE + random.choice(GAP_RANGE)
    pts, anchor_pair = _pick_staircase()

    bg_a, bg_b = random.sample(range(NUM_COLORS), 2)
    phase = random.randint(0, 1)
    available = [c for c in range(NUM_COLORS) if c not in {bg_a, bg_b}]
    max_col_height = max(sum(1 for rr, cc in pts if cc == target_c) for target_c in {cc for _, cc in pts})
    k_min = max(
        PALETTE_SIZE_MIN,
        max_col_height,
        (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR,
    )
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(available), len(pts)))
    palette = random.sample(available, k)
    legend_colors = _assign_legend_colors(pts, anchor_pair, palette)
    if legend_colors is None:
        return None

    max_c = max(c for _, c in pts)
    c_hi = size - OUTER_SIZE - max_c * step
    if c_hi < BASE_C_MIN:
        return None
    base_c = random.randint(BASE_C_MIN, c_hi)

    out, bg_pattern, placed = _render_output(size, bg_a, bg_b, phase, step, base_c, legend_colors)

    inp = out.copy()
    keep = {anchor_pair, (anchor_pair[0], anchor_pair[1] + 1)}
    for pos, (r0, c0) in placed.items():
        if pos in keep:
            continue
        inp[r0:r0 + OUTER_SIZE, c0:c0 + OUTER_SIZE] = bg_pattern[r0:r0 + OUTER_SIZE, c0:c0 + OUTER_SIZE]

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
    raise RuntimeError("failed to generate valid c4d067a0 combo input")
