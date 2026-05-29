# Combo variant of 8f215267 (index 9)
# combo_spec: D4/1+D5/0
# Name: 计数取反+从左侧放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# ---------------------------------------------------------------------------
# Design analysis for the combo variant
#
# a) 具体化与行为追踪
#    - 视觉对象沿用原题: 左侧竖直堆叠 3-4 个 5x11 空心彩框, 右侧散布小型单色精灵.
#    - 输入里框中心行为空; 输出里仍只在各框中心行画点, 但融合后同时改两件事:
#      1. 点数不再等于同色精灵数 N, 而是等于 (MAX-N), 这里 MAX=4.
#      2. 点位不再从右内壁累积, 而是从左内壁按 2 列间隔向右累积.
#    - 例子: 某框 bbox 为 (r0,c0,r1,c1)=(1,1,5,11), 中心行 r=3.
#      若该框同色精灵数 N=1:
#        原题输出会在列 9 画 1 个点.
#        本融合变体会画 (4-1)=3 个点, 且位置变为列 3,5,7.
#      若 N=4:
#        原题是 4 个点.
#        本融合变体是 0 个点.
#      差异同时体现在“数量反向映射”和“左右方向镜像”.
#
# b) 复杂度校准
#    - 不改变对象种类、颜色匹配方式、框/精灵分区, 只改 tally 的解释方式.
#    - 为了让两种变化都在 2-4 个样例中可见, 生成器强制每个样例至少包含:
#      一个 N=0 框, 一个 N=4 框, 以及一个中间档 N in {1,2,3} 的框.
#      这样能同时观察到:
#      * 反向计数: 0->4 点, 4->0 点;
#      * 左侧布局: 中间档会产生 1-3 个非满行点, 明显贴左而不是贴右.
#
# c) 隐式假设审查（针对 main）
#    - `get_bg(g)` 依赖背景是全局出现最多的颜色. 框为空心且精灵稀疏, 仍成立.
#    - `find_components(... monochromatic=True)` 依赖框与精灵都是单色 4 连通块.
#      生成器继续用单色空心框和单色连通 polyomino, 成立.
#    - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 依赖框明显大且稀疏.
#      固定 5x11 空心框面积 28, 仍远小于 55*0.7, 可与小精灵区分.
#    - “先擦除所有 pattern 再画点” 依赖精灵不会接触或覆盖框. 生成器把精灵限定在右侧独立画布.
#    - `dots_to_draw = len(slots) - num_patterns` 引入新假设: 同色精灵数不超过可用槽位数.
#      生成器将真实计数限制在 0..4, 保证不会出现负槽位需求.
#    - 左起点布局依赖至少有一个样例出现 1..3 个显示点, 否则 0 点或 4 点都可能与右起点视觉重合.
#      生成器显式保证这一点.
#
# d) 放置合法性
#    - 框仍位于左侧固定列区间; 精灵先放在宽 9 的右侧子画布, 再粘贴到全局列 13 开始的位置.
#    - `random_free_location(... padding=1, padding_connectivity=8)` 保证精灵彼此不重叠、不接触且不越界.
#    - 采用 output-first: 先构造融合规则下的目标输出, 再抹掉框内点形成输入, 最后放置右侧精灵.
#      末尾用 `main(inp) == out` 做严格验证.
#
# e) 视觉可观察性
#    - 人类可直接看到“精灵越多, 点越少”的反直觉映射, 因为同一张图中同时存在空框(0 点)和满框(4 点)两端案例.
#    - 人类也能看到“点从左边开始排”, 因为至少有一个框显示 1-3 个点, 不会退化成左右都一样的满四点或零点.
#    - 颜色只承担同类匹配作用, 不参与任何不可见数值排序.
#
# f) 可视化整洁度
#    - 左区只有框与框内点, 右区只有精灵, 分区清晰.
#    - 标记仍只占中心行四个离散槽位, 不会与框边混成实心块.
#
# g) 变化的可推断性
#    - 没有新增颜色映射. 解题者只需追踪“同色精灵属于哪个框”, 再从多个样例归纳出:
#      tally 数量取反, 且 tally 的累积方向改为从左到右.
#
# h) 量化非退化目标
#    - 记每个样例的真实匹配计数为 match_counts, 输出显示计数为 display_counts=4-match_counts.
#    - 一个样例被视为表达充分, 需同时满足:
#      1. `0 in match_counts` and `4 in match_counts`, 证明计数取反确实可见;
#      2. `any(1 <= d <= 3 for d in display_counts)`, 证明左侧起排可见;
#      3. 至少一个框满足 `display_count != match_count`, 否则会局部退化成原题计数.
#      本生成器通过采样约束直接保证这些条件.
#
# i) 新增假设与生成器对应
#    - main() 默认所有框共享同样的四个合法点位模式. 固定 5x11 框保证这一点.
#    - main() 不区分 distractor 颜色, 只按是否与某框同色来计数. 生成器保证干扰色与框色不重合.
#    - main() 依据 bbox 中心行绘点. 所有框都是奇数高 5, 因而中心行唯一且稳定.
# ---------------------------------------------------------------------------


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300
COUNT_SAMPLE_ATTEMPTS = 200


def _global_dot_slots_from_left(c0: int, c1: int) -> list[int]:
    """Legal center-row dot columns inside a frame, ordered from left to right."""
    return [int(cc) for cc in range(int(c0) + 2, int(c1), 2)]


def _local_dot_cols_from_left(n_dots: int) -> list[int]:
    """Interior patch columns for left-anchored dots in the fixed 5x11 frame."""
    return [2 + 2 * i for i in range(int(n_dots))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert right-side color classes into inverse tallies inside the matching frames.

    Input structure:
    - The background is the most frequent color.
    - Several large hollow monochrome frames are stacked on the left.
    - Small monochrome sprites of various colors are scattered to the right of the frames.

    Rule:
    - Erase every small sprite from the grid.
    - For each frame color C, count matching sprites N and compute the inverse tally (4 - N).
    - Draw that many dots on the frame center row, spaced every 2 columns, starting from the left inner wall.

    Key insight:
    - The frames act as displays, but this variant flips both semantics and direction:
      more same-color sprites produce fewer dots, and the remaining dots accumulate from left to right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == color)
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        slots = _global_dot_slots_from_left(c0, c1)
        dots_to_draw = max(0, len(slots) - num_patterns)
        for cc in slots[:dots_to_draw]:
            out[int(r_mid), int(cc)] = color

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with left-anchored dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    patch[FRAME_H // 2, _local_dot_cols_from_left(n_dots)] = color
    return patch


def _sample_match_counts(rng: np.random.Generator, n_frames: int, n_distractors: int) -> list[int] | None:
    """
    Sample true same-color sprite counts per frame.

    The sample must simultaneously expose:
    - count inversion via both extremes (0 and 4),
    - left-anchored placement via a partial displayed tally (1..3 dots).
    """
    lo_total, hi_total = TOTAL_PATTERNS_RANGE
    for _ in range(COUNT_SAMPLE_ATTEMPTS):
        counts = [int(rng.integers(0, MAX_DOTS_PER_FRAME + 1)) for _ in range(n_frames)]
        if 0 not in counts:
            continue
        if MAX_DOTS_PER_FRAME not in counts:
            continue
        if not any(0 < c < MAX_DOTS_PER_FRAME for c in counts):
            continue
        total_patterns = sum(counts) + n_distractors
        if lo_total <= total_patterns < hi_total:
            return counts
    return None


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w bounding box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    """Generate a small random connected sprite, tight-cropped to its bounding box."""
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    match_counts = _sample_match_counts(rng, n_frames, n_distractors)
    if match_counts is None:
        return None

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    display_counts = [MAX_DOTS_PER_FRAME - n for n in match_counts]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, n_dots) in enumerate(zip(frame_colors, display_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, n_dots), r0, FRAME_C0)

    inp = out.copy()
    for r0, n_dots in zip(frame_rows, display_counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + cc for cc in _local_dot_cols_from_left(n_dots)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, match_counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    """
    Generate validated combo-variant inputs.

    Guarantees:
    - each frame's true same-color sprite count lies in 0..4;
    - each sample contains 0, 4, and an intermediate true count;
    - therefore the output shows both count inversion and a visible left-anchored partial tally;
    - only grids satisfying `main(inp) == out` are returned.
    """
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
