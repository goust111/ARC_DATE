# Variant of c4d067a0
# Dimension: D1 — 匹配条件：坐标映射方式 (match_cond)
# Change: 图例像素位置通过统一缩放因子+偏移量映射到网格位置，缩放因子由已有色块的间距与图例像素间距的比值确定... → 颜色匹配对接
#
# Design analysis
# ---------------
# variant_spec:
#   concrete_layout:
#     legend:
#       top-left 仍然是 step=2 的稀疏单像素楼梯图案，颜色 2-4 种，单个颜色可重复出现。
#       这里的坐标只负责把“有哪些颜色、每种颜色要放几次”展示出来；不再承担主区域定位功能。
#     main_area:
#       原题的规则方阵实心色块改成同尺寸的彩色方框槽位。槽位有两种状态：
#       1. 空框：1 像素厚边框、中心为 bg。
#       2. 已放入：整个外接正方形都变成该边框颜色的实心块。
#       另外加入 1-2 个干扰空框，其边框颜色不在图例中，输出必须保持为空。
#   flow_trace_example:
#     original:
#       图例里 (0,0)=red, (0,1)=blue；主区已有两个实心块，通过它们的间距推出 step，
#       再把剩余 legend 像素按 affine projection 投到主区。
#     variant:
#       图例里 red 出现 3 次、blue 出现 2 次；主区散布 3 个红框、2 个蓝框和 1 个黄框。
#       输入里其中一些红/蓝框已实心，其余为空。solver 统计图例颜色多重集合，然后只把匹配颜色的空框补成实心，
#       黄框因为颜色不在图例中保持为空。坐标不参与任何对应。
#     difference:
#       原逻辑的“同一 step + base 的位置投影”被彻底移除；新的关键关系是“图例颜色多重集合 ↔ 方框边框颜色”。
#   complexity_calibration:
#     不能把任务简化成“所有彩框都填满”，否则 legend 冗余。因此生成器强制加入非图例颜色的干扰框。
#     也不能让每种颜色只出现 1 次，否则多重集合信息太弱。所以沿用原题 6-10 个 legend 像素、2-4 种颜色，
#     自然形成重复颜色；solver 按每色需要的数量补全。
#   implicit_assumption_audit:
#     - 原 solver 把 legend 像素正规化成 (rr,cc) 并依赖它们参与投影；变体中这些坐标不再用于主区定位。
#     - 原 solver 假设主区关键对象全是实心块；变体中主区主对象多数是空框，因此“由实心块 gcd 推 step”的代码会失效。
#     - 原 solver 对每个 legend 点无条件向 affine 目标位置上色；变体中这样会覆盖错误区域，所以必须改成只填充检测出的空框槽位。
#     - 新 solver 需要区分“同尺寸空框”和“同尺寸实心块”这两种合法槽位状态；生成器保证除此之外没有别的同色复杂组件。
#   placement_legality:
#     所有槽位共享一个外边长 3 或 4，并放在从第 9 列开始的粗网格候选位置上，彼此至少隔 1 格，避免重叠和粘连。
#     图例区只含单像素点，主区只含槽位，二者视觉分离。输出仅把图例颜色对应的空框变成实心，其余对象不变。
#   visual_observability:
#     人类看到样例时，会明显发现原题的“坐标缩放后生成规则阵列”消失了，取而代之的是“散布的彩色框按颜色被填满”。
#     干扰框保持空心，使 legend 的作用可见；已有实心槽位则提示输出是在“补完同类槽位”而不是整体重绘。
#   cleanliness_and_color_reasoning:
#     颜色映射只有“相同颜色匹配”这一条显式规则，不依赖颜色 id 排序或隐藏数值关系。
#     颜色种类上限仍为 4 个图例色 + 2 个干扰色，足够可读，不会让视觉过乱。
#   anti-degeneracy:
#     生成器显式拒绝任何把“legend 点 → 对应槽位”解释成单一 affine step/base 投影的布局，防止退化回原题。
#   new_solver_assumptions:
#     - 至少存在一个 top-left legend 单像素和若干个同尺寸槽位。
#     - 槽位都是单色正方形外接框：要么是空心 ring，要么是同尺寸实心 square。
#     - 每个 legend 颜色在主区的槽位总数恰好等于图例中该颜色出现次数；干扰颜色不出现在 legend。
#     - 上述假设由 generate_input() 保证。

import random
from collections import Counter

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete scattered square slots by matching legend colors to frame-border colors.

    Input structure:
    - bg = most common color.
    - Top-left corner: a small sparse legend of single colored pixels on a regular lattice; repeated colors mean multiple pieces of that color.
    - Main area: equal-sized monochrome square slots scattered away from the legend. A slot is either a hollow frame (empty) or a solid square (already filled).

    Rule:
    - Ignore geometric alignment between the legend and the main area.
    - Count how many times each legend color appears.
    - In the main area, treat a hollow square frame with border color X as an empty slot for color X; solid squares of color X count as already placed.
    - Fill enough matching hollow frames to realize the legend's color multiset, leaving slots of colors absent from the legend untouched.

    Key insight: The legend no longer acts like a minimap. It supplies a multiset of colors, and border colors on the hollow square frames determine where those colored blocks belong.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    out = g.copy()
    bg = get_bg(g)

    comps = components_mask(g != bg, connectivity=4)
    singles: list[tuple[int, int, int]] = []
    slots: list[tuple[int, int, int, int, bool]] = []  # (r0,c0,size,color,filled)

    for pts in comps:
        if len(pts) == 1:
            r, c = pts[0]
            singles.append((int(r), int(c), int(g[r, c])))
            continue

        r0, c0, r1, c1 = bbox_coords(pts)
        hh, ww = r1 - r0 + 1, c1 - c0 + 1
        if hh != ww or hh < 3:
            continue

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        fg = sub != bg
        cols = {int(v) for v in sub[fg]}
        if len(cols) != 1:
            continue
        col = next(iter(cols))
        size = int(hh)

        ring = np.zeros((size, size), dtype=bool)
        ring[0, :] = True
        ring[-1, :] = True
        ring[:, 0] = True
        ring[:, -1] = True

        if np.array_equal(fg, ring):
            slots.append((int(r0), int(c0), size, col, False))
        elif np.all(sub == col):
            slots.append((int(r0), int(c0), size, col, True))

    assert singles and slots, "grid must contain legend pixels and square slots"

    win_r = min(h, 12)
    win_c = min(w, 12)
    legend = [(r, c, v) for (r, c, v) in singles if r < win_r and c < win_c]
    assert legend, "top-left legend must contain singleton pixels"
    need = Counter(int(v) for _, _, v in legend)

    sizes = [s for _, _, s, _, _ in slots]
    vals, cnts = np.unique(np.array(sizes, dtype=int), return_counts=True)
    slot_size = int(vals[cnts.argmax()])
    slots = [slot for slot in slots if slot[2] == slot_size]
    assert slots, "must contain at least one slot of the dominant size"

    by_color: dict[int, list[tuple[int, int, bool]]] = {}
    have = Counter()
    for r0, c0, size, col, filled in slots:
        by_color.setdefault(int(col), []).append((int(r0), int(c0), bool(filled)))
        if filled:
            have[int(col)] += 1
    for col in by_color:
        by_color[col].sort()

    for col, required in need.items():
        filled = have[col]
        empties = [(r0, c0) for r0, c0, is_filled in by_color.get(col, []) if not is_filled]
        assert filled <= required, "already-filled slots cannot exceed legend demand"
        assert filled + len(empties) >= required, "matching slots must cover the legend demand"
        for r0, c0 in empties[: required - filled]:
            out[r0 : r0 + slot_size, c0 : c0 + slot_size] = col

    return out


# --------------- generator (o5x) ---------------

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
    """Enumerate all valid staircase shapes as tuples of row widths."""
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


def _random_color_counts(total, k):
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


def _paint_slot(arr, r0, c0, size, col, bg, filled):
    if filled:
        arr[r0 : r0 + size, c0 : c0 + size] = col
        return
    arr[r0 : r0 + size, c0 : c0 + size] = bg
    arr[r0, c0 : c0 + size] = col
    arr[r0 + size - 1, c0 : c0 + size] = col
    arr[r0 : r0 + size, c0] = col
    arr[r0 : r0 + size, c0 + size - 1] = col


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
    k = random.randint(k_min, min(PALETTE_SIZE_MAX, len(pts)))
    usable = [c for c in range(NUM_COLORS) if c != bg]
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
        out[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col
        inp[lr0 + rr * LEGEND_STEP, lc0 + cc * LEGEND_STEP] = col

    for pos in pts:
        col = legend_colors[pos]
        r0, c0 = placement[pos]
        _paint_slot(out, r0, c0, slot_size, col, bg, True)
        _paint_slot(inp, r0, c0, slot_size, col, bg, pos in prefilled)

    for (r0, c0), col in zip(distractor_positions, distractor_colors):
        _paint_slot(out, r0, c0, slot_size, col, bg, False)
        _paint_slot(inp, r0, c0, slot_size, col, bg, False)

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
    raise RuntimeError("failed to generate valid c4d067a0 D1 input")
