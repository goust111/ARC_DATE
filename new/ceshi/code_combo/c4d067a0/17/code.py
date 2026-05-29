# Combo variant of c4d067a0 (index 17)
# combo_spec: D1/0+D3/0
# Name: 颜色匹配映射+框架填充

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg

# Design analysis
# ---------------
# combo_variant:
#   concrete_layout:
#     legend:
#       左上角保留原题的 step=2 稀疏单像素楼梯图例，仍使用 6-10 个点、2-4 种非背景色。
#       图例在融合后只承担“颜色需求表”的角色：每个像素颜色表示主区需要一个同色框位。
#     main_area:
#       主区对象统一改成同尺寸正方形框架。空框是 1 像素厚单色边框，内部洞为背景色；
#       已填框则是在同一框内把洞填成和边框相同的颜色，因此整个外接正方形看起来变成实心块。
#       另外加入 1-2 个干扰空框，其边框颜色完全不出现在图例中，输出必须保持为空。
#   flow_trace_example:
#     original:
#       图例若有 red / blue 两个相邻像素，原题会利用主区已有的两个实心锚块推断 step 和 base，
#       再把整张 legend 作为缩略图投影到主区规则晶格上。
#     combo:
#       现在图例里若 red 出现 3 次、blue 出现 2 次，则主区会散布 3 个红框、2 个蓝框和若干非图例色干扰框。
#       输入中部分红/蓝框已填成实心，其余保持空框。solver 统计 legend 颜色出现次数，
#       然后只把边框颜色属于 legend 的空框内部填满；干扰框颜色不在 legend 中，因此继续保持空心。
#     difference:
#       原题的“坐标缩放+偏移”被完全删除，目标也不再是直接往背景上放色块，而是往既有框架内部填洞。
#       新规则是颜色配对驱动的槽位补全：图例颜色 <-> 框架边框颜色。
#   complexity_calibration:
#     不能让输出变成“所有框都填满”，否则 legend 失去作用；因此强制放置至少 1 个非图例色干扰框。
#     也不能让每种图例颜色只出现 1 次，否则颜色多重集太弱；沿用原题 6-10 个图例点和 2-4 种颜色，
#     自然形成重复颜色。对每个图例颜色，主区对应框位总数恰好等于该颜色在图例中的计数，避免歧义。
#   implicit_assumption_audit:
#     - 原 solver 假设主区关键对象是实心块，并依赖块坐标推 affine step；融合后主区主体是空框/已填框混合，
#       所以必须改成槽位识别，不能再从几何间距反推坐标映射。
#     - 原 solver 依赖 legend 的 (rr,cc) 参与投影；融合后这些坐标不再参与主区定位，只统计颜色计数。
#     - D3/0 引入“填框内洞而非画裸块”的目标，因此输出时不能盲目覆盖整个 bbox；应只填内部洞区域。
#       由于洞颜色与边框颜色相同，填完后视觉上变成实心方块，但其生成过程仍是框架填充。
#     - 新 solver 还假设主区合法对象只有同尺寸的空框或同尺寸的已填框；生成器通过统一尺寸和留白来保证。
#   placement_legality:
#     所有框位共享一个外尺寸 3 或 4，并放在从第 9 列开始的粗网格候选位置上，彼此至少隔 1 格，不重叠不越界。
#     图例区只含单像素点，主区只含槽位，视觉上分栏。输入至少保留 2 个已填框且至少留 1 个待填框。
#   visual_observability:
#     人类会看到：左上图例给出若干颜色；主区散布这些颜色的空框/实心块混合物，另有少量颜色不在图例中的空框。
#     输出只补完图例颜色对应的框位，非图例色框保持空心，因此“颜色匹配 + 框内填充”都能肉眼观察到。
#   cleanliness_and_color_reasoning:
#     颜色规则只依赖显式的“相同颜色匹配”，不依赖颜色编号大小。
#     主区颜色数控制在图例 2-4 色加 1-2 个干扰色；框位尺寸统一，避免视觉噪声。
#   anti_degeneracy:
#     生成器显式拒绝任何把“图例格点 -> 对应图例色框位”解释成单一 affine step/base 的布局，
#     防止样例退化回原题或 D3 单变体的坐标投影思路。
#   new_solver_assumptions:
#     - 左上角存在可分离的单像素图例。
#     - 主区存在若干同尺寸单色方框槽位；状态要么为空框，要么为同色已填框。
#     - 每个图例颜色在主区的槽位总数恰好等于它在图例中的出现次数；干扰框颜色不在图例中。
#     - 至少一个图例颜色对应的槽位在输入中仍为空，且至少一个非图例色干扰框保持为空。
#     - 上述假设都由 generate_input() 构造并用 main() 自验证。


def _ring_mask(size: int) -> np.ndarray:
    ring = np.zeros((size, size), dtype=bool)
    ring[0, :] = True
    ring[-1, :] = True
    ring[:, 0] = True
    ring[:, -1] = True
    return ring


def _collect_slots(grid: np.ndarray, bg: int):
    comps = components_mask(grid != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    slots: list[tuple[int, int, int, int, bool]] = []  # (r0,c0,size,color,filled)

    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(grid[r, c])))
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 3:
            continue

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        fg = sub != bg
        cols = {int(v) for v in sub[fg]}
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        ring = _ring_mask(int(hh))

        if np.array_equal(fg, ring):
            slots.append((int(r0), int(c0), int(hh), int(col), False))
        elif np.all(sub == col):
            slots.append((int(r0), int(c0), int(hh), int(col), True))

    return singles, slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete a scattered field of colored square frames using the legend as a color-to-slot key.

    Input structure:
    - bg is the most common color.
    - Top-left: a sparse legend of single-pixel colors on a regular lattice; repeated colors mean multiple target pieces of that color.
    - Main area: equal-sized square frames scattered away from the legend. Some are hollow frames with background-colored holes, and some are already filled into solid same-color squares.

    Rule:
    - Do not infer any coordinate scale or offset from the legend.
    - Count how many times each color appears in the legend.
    - In the main area, treat each hollow frame as an empty slot whose identity is given by its border color; a solid square of that color is the same slot already filled.
    - For every legend color, fill the interior hole of the matching same-colored empty frames until that color's legend count is satisfied. Frames whose border colors never appear in the legend stay unchanged.

    Key insight: The legend is no longer a minimap of positions. It is a multiset of colors, and each color selects which existing frames should have their inner holes filled.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    singles, slots = _collect_slots(g, bg)
    assert singles and slots, "grid must contain legend pixels and square frame slots"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"
    need = Counter(int(v) for _, _, v in legend)

    sizes = [size for _, _, size, _, _ in slots]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    slot_size = int(vals[cnts.argmax()])
    slots = [slot for slot in slots if slot[2] == slot_size]
    assert slots, "must contain at least one slot of the dominant size"

    by_color: dict[int, list[tuple[int, int, bool]]] = {}
    for r0, c0, size, col, filled in slots:
        by_color.setdefault(int(col), []).append((int(r0), int(c0), bool(filled)))
    for col in by_color:
        by_color[col].sort()

    for col, required in need.items():
        placements = by_color.get(int(col), [])
        assert placements, "every legend color must have matching slots"
        filled = [(r0, c0) for r0, c0, is_filled in placements if is_filled]
        empties = [(r0, c0) for r0, c0, is_filled in placements if not is_filled]
        assert len(filled) <= required, "already-filled slots cannot exceed legend demand"
        assert len(filled) + len(empties) == required, \
            "matching slot count must equal the legend multiplicity for that color"
        for r0, c0 in empties:
            out[r0 + 1 : r0 + slot_size - 1, c0 + 1 : c0 + slot_size - 1] = col

    return out


# --------------- generator (combo d1+d3) ---------------

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

# === CFG: slot area ===
SLOT_SIZE_RANGE = range(3, 5)
BASE_C_MIN = 9
BASE_R_MIN = 1
DISTRACTOR_SLOT_COUNTS = (1, 2)
PREFILLED_MIN, PREFILLED_MAX = 2, 4

# === CFG: generation ===
MAX_RETRIES = 400


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


_STAIRCASE_SHAPES = _enumerate_staircase_shapes()


def _pick_staircase():
    widths = random.choice(_STAIRCASE_SHAPES)
    return [(r, c) for r, width in enumerate(widths) for c in range(width)]


def _random_color_counts(total: int, k: int) -> list[int]:
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


def _assign_legend_colors(pts, palette):
    counts = _random_color_counts(len(pts), len(palette))
    bag = []
    for col, count in zip(palette, counts):
        bag.extend([col] * count)
    random.shuffle(bag)
    order = pts[:]
    random.shuffle(order)
    return {pos: col for pos, col in zip(order, bag)}


def _draw_slot(arr: np.ndarray, r0: int, c0: int, size: int, col: int, bg: int, filled: bool) -> None:
    arr[r0 : r0 + size, c0 : c0 + size] = bg
    arr[r0, c0 : c0 + size] = col
    arr[r0 + size - 1, c0 : c0 + size] = col
    arr[r0 : r0 + size, c0] = col
    arr[r0 : r0 + size, c0 + size - 1] = col
    if filled:
        arr[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = col


def _pick_prefilled_positions(pts, legend_colors):
    max_keep = min(PREFILLED_MAX, len(pts) - 1)
    keep_count = random.randint(PREFILLED_MIN, max_keep)
    distinct_goal = min(2, len(set(legend_colors.values())))
    for _ in range(80):
        keep = set(random.sample(pts, keep_count))
        if len({legend_colors[p] for p in keep}) >= distinct_goal:
            return keep
    return set(random.sample(pts, keep_count))


def _assigned_mapping_is_affine(placement, grid_size):
    for step in range(1, grid_size + 1):
        base_rs = {int(r0) - int(rr) * step for (rr, _cc), (r0, _c0) in placement.items()}
        base_cs = {int(c0) - int(cc) * step for (_rr, cc), (_r0, c0) in placement.items()}
        if len(base_rs) == 1 and len(base_cs) == 1:
            return True
    return False


def _gen_attempt():
    size = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    slot_size = random.choice(tuple(SLOT_SIZE_RANGE))
    pts = _pick_staircase()

    bg = random.randrange(NUM_COLORS)
    k_min = max(PALETTE_SIZE_MIN, (len(pts) + MAX_COUNT_PER_COLOR - 1) // MAX_COUNT_PER_COLOR)
    usable = [c for c in range(NUM_COLORS) if c != bg]
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts), len(usable)))
    palette = random.sample(usable, k)
    legend_colors = _assign_legend_colors(pts, palette)

    distractor_count = random.choice(DISTRACTOR_SLOT_COUNTS)
    distractor_palette = [c for c in usable if c not in palette]
    if len(distractor_palette) < distractor_count:
        return None
    distractor_colors = random.sample(distractor_palette, distractor_count)

    pitch = slot_size + 1
    rows = list(range(BASE_R_MIN, size - slot_size + 1, pitch))
    cols = list(range(BASE_C_MIN, size - slot_size + 1, pitch))
    candidates = [(r0, c0) for r0 in rows for c0 in cols]
    total_slots = len(pts) + distractor_count
    if len(candidates) < total_slots:
        return None

    random.shuffle(candidates)
    chosen = candidates[:total_slots]
    legend_positions = chosen[: len(pts)]
    distractor_positions = chosen[len(pts) :]
    placement = {pos: slot for pos, slot in zip(pts, legend_positions)}
    if _assigned_mapping_is_affine(placement, size):
        return None

    prefilled = _pick_prefilled_positions(pts, legend_colors)

    lr0, lc0 = LEGEND_ORIGIN
    out = np.full((size, size), bg, dtype=int)
    inp = np.full((size, size), bg, dtype=int)

    for (rr, cc), col in legend_colors.items():
        rr0 = lr0 + rr * LEGEND_STEP
        cc0 = lc0 + cc * LEGEND_STEP
        out[rr0, cc0] = col
        inp[rr0, cc0] = col

    for pos in pts:
        col = legend_colors[pos]
        r0, c0 = placement[pos]
        _draw_slot(out, r0, c0, slot_size, col, bg, True)
        _draw_slot(inp, r0, c0, slot_size, col, bg, pos in prefilled)

    for (r0, c0), col in zip(distractor_positions, distractor_colors):
        _draw_slot(out, r0, c0, slot_size, col, bg, False)
        _draw_slot(inp, r0, c0, slot_size, col, bg, False)

    if not any(pos not in prefilled for pos in pts):
        return None

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
