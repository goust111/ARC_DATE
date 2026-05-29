# Combo variant of 8f215267 (index 12)
# combo_spec: D4/0+D16/0
# Name: 面积计数+多行排列

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) 具体化与行为追踪
#    - 视觉母体保持原题的左右分区：左边是竖直堆叠的彩色空心框，右边是散落的小单色精灵。
#    - 原题对每个框只做“同色精灵个数”统计，N 最大自然接近 4，所以单行中心行足够。
#    - 本融合版把 N 改成“同色精灵总面积 / 2”。因此同样都是 2 个精灵，
#      若一个框收到面积 [2,2]，则 N=(2+2)/2=2；另一个框收到面积 [2,6]，则 N=(2+6)/2=4。
#      两个框精灵个数相同，但输出点数不同，解题者必须观察精灵大小。
#    - 同时引入多行容量：点位顺序固定为中心行从右往左 4 格，满了之后改到上方一行，
#      还不够再改到下方一行。于是 N 可达到 12。
#    - 例子：某蓝框右侧有 3 个蓝精灵，面积分别为 6、6、6，总面积 18，所以 N=9。
#      原题会把它误看成 3 个点；本题会画 9 个点：中心行 4 个、上行 4 个、下行 1 个。
#
# b) 复杂度校准
#    - 面积贡献只使用 1/2/3 三档，对应精灵面积 2/4/6；避免连续大范围取值导致观察者难以归纳。
#    - 每个样例都强制出现：
#      1. 一个高值框（N>=9），使多行尤其第三行可见；
#      2. 一对“精灵个数相同但面积和不同”的框，直接打破“按个数计数”的旧解释。
#    - 除此之外不改颜色语义、不改左右布局，复杂度集中在两个目标维度上。
#
# c) 隐式假设审查（main 逐段）
#    - `find_components(..., monochromatic=True)` 仍成立：输入里框和精灵都保持单色且彼此分离。
#    - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 仍成立：
#      输入中的框内部为空心；精灵 bbox 仍限制在 4x3 以内，不会伪装成框。
#    - “先移除所有 patterns” 仍安全：右侧只有精灵，没有别的记号；框内在输入态为空。
#    - 原 solver 的关键旧假设是“每个精灵贡献 1 点且 4 个点就够用”。融合后这两个前提都失效：
#      需要把 `sum(1 for p...)` 改成 `sum(area for p...) // 2`，并把绘点逻辑改成跨三行枚举合法位置。
#    - 新增假设是“可计入的同色总面积能被 2 整除且不超过 24 像素”。生成器通过只采样面积 2/4/6
#      且总贡献 N<=12 来保证，因此不会出现半点或越界点位。
#
# d) 放置合法性
#    - 生成器仍采用 output-first：先决定每个框应显示多少点，再反推右侧要放哪些面积的精灵。
#    - 右侧精灵继续用 `random_free_location(..., padding=1, padding_connectivity=8)` 放置，
#      既避免重叠，也避免同色精灵贴边后合并成一个更大组件。
#    - 输出点位严格限制在 5x11 框的 3 条内部行和 4 个右对齐列上，总容量 12。
#
# e) 视觉可观察性
#    - 人类能直接看到精灵大小差异，因为所有计数对象都是实际像素块，不依赖颜色编号或隐藏标签。
#    - 每个样例都会出现“同样 2 个精灵却点数不同”的配对，并且至少一个框溢出到多行，
#      因此两条融合规则都能在少量样例中被肉眼支持。
#
# f) 可视化整洁度
#    - 框区和精灵区仍完全分开；框内点位采用规整的 4 列 x 3 行右对齐排布，不会与边框混成噪声。
#    - 精灵只用紧凑的 2/4/6 像素 polyomino，既能呈现面积差异，又不会挤满右侧区域。
#
# g) 变化的可推断性
#    - 颜色仍只承担“哪堆精灵属于哪个框”的分组作用，没有额外颜色映射。
#    - 真正变化来自可见形状面积与框内多行点阵的对应关系；生成器刻意重复这些档位，
#      让观察者能从多个框之间对比出“面积/2”而不是“颜色值”或“精灵个数”。


# --- shared geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- tally semantics ---
AREA_PER_DOT = 2
MAX_DOTS_PER_ROW = 4
DOT_ROW_OFFSETS = (0, -1, 1)
MAX_DOTS_PER_FRAME = MAX_DOTS_PER_ROW * len(DOT_ROW_OFFSETS)
SPRITE_VALUE_CHOICES = (1, 2, 3)

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (8, 13)
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- generator templates ---
HEAVY_VALUE_OPTIONS = (
    (3, 3, 3),       # 9 dots
    (3, 3, 2, 1),    # 9 dots
    (3, 2, 2, 2),    # 9 dots
    (3, 3, 3, 1),    # 10 dots
    (3, 3, 2, 2),    # 10 dots
    (3, 3, 3, 2),    # 11 dots
    (3, 3, 3, 3),    # 12 dots
)
PAIR_VALUE_OPTIONS = (
    ((1, 1), (1, 2)),  # 2 sprites vs 2 sprites, 2 dots vs 3 dots
    ((1, 1), (2, 2)),  # 2 vs 4
    ((1, 2), (1, 3)),  # 3 vs 4
    ((1, 2), (2, 3)),  # 3 vs 5
    ((2, 2), (2, 3)),  # 4 vs 5
)
EXTRA_VALUE_OPTIONS = (
    (),
    (1,),
    (2,),
    (3,),
    (1, 1),
)

# --- retry budget ---
MAX_ATTEMPTS = 300


def _is_frame(comp: dict) -> bool:
    """Heuristic frame detector inherited from the base task."""
    b = comp["bbox"]
    h = bbox_h(b)
    w = bbox_w(b)
    area = int(comp["area"])
    return (h >= 5) and (w >= 5) and (area < (h * w * 0.7))


def _tally_positions_from_bbox(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]]:
    """Enumerate the frame's legal tally cells in the fixed multi-row order."""
    r0, c0, r1, c1 = map(int, bbox)
    r_mid = (r0 + r1) // 2
    positions: list[tuple[int, int]] = []
    remaining = int(count)
    for row_offset in DOT_ROW_OFFSETS:
        rr = r_mid + row_offset
        if not (r0 < rr < r1):
            continue
        for i in range(MAX_DOTS_PER_ROW):
            if remaining <= 0:
                return positions
            cc = int(c1 - 2 - 2 * i)
            if cc > c0:
                positions.append((rr, cc))
                remaining -= 1
    return positions


def _split_scene(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    """Return background color plus separated frame/pattern component lists."""
    bg = int(get_bg(grid))
    regions = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        if _is_frame(comp):
            frames.append(comp)
        else:
            patterns.append(comp)
    return bg, frames, patterns


def _frame_value(patterns: list[dict], color: int) -> int:
    """Compute a frame's displayed tally from same-color sprite area."""
    total_area = sum(int(p["area"]) for p in patterns if int(p["color"]) == int(color))
    return total_area // AREA_PER_DOT


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert same-color sprite area into a right-aligned multi-row tally inside each frame.

    Input structure:
    - The background is the most frequent color.
    - Several hollow monochrome frames of distinct colors are stacked on the left.
    - Small monochrome sprites are scattered on the right; some share a frame color and some are distractors.

    Rule:
    - Remove every scattered sprite from the grid.
    - For each frame color C, sum the pixel area of all sprites with color C and divide by 2 to get N.
    - Draw N dots of color C inside that frame using fixed slots: center row from the right every 2 columns, then the row above, then the row below.

    Key insight:
    - The frames no longer count how many same-color sprites exist; they measure how much same-color sprite area exists, and overflow is shown by spilling the tally into extra rows.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg, frames, patterns = _split_scene(g)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        tally = _frame_value(patterns, color)
        if tally <= 0:
            continue
        for rr, cc in _tally_positions_from_bbox(tuple(map(int, f["bbox"])), tally):
            out[int(rr), int(cc)] = color

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a hollow frame with its multi-row tally already drawn."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    for rr, cc in _tally_positions_from_bbox((0, 0, FRAME_H - 1, FRAME_W - 1), n_dots):
        patch[int(rr), int(cc)] = color
    return patch


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    """Grow a connected polyomino of the given area inside an h x w box."""
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < int(area):
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite_from_value(rng: np.random.Generator, color: int, bg: int, value: int) -> np.ndarray:
    """Generate a sprite whose area encodes `value` tally units."""
    area = int(value) * AREA_PER_DOT
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
        patch[int(r - r_min), int(c - c_min)] = color
    return patch


def _sample_matched_value_specs(rng: np.random.Generator, n_frames: int) -> list[tuple[int, ...]]:
    """
    Sample per-frame sprite-value lists.

    Guarantees:
    - one heavy frame with tally >= 9, so third-row usage is visible;
    - one pair of frames with the same sprite count (2) but different tally sums.
    """
    heavy = tuple(HEAVY_VALUE_OPTIONS[int(rng.integers(0, len(HEAVY_VALUE_OPTIONS)))])
    pair_left, pair_right = PAIR_VALUE_OPTIONS[int(rng.integers(0, len(PAIR_VALUE_OPTIONS)))]
    specs: list[tuple[int, ...]] = [heavy, tuple(pair_left), tuple(pair_right)]
    while len(specs) < int(n_frames):
        extra = tuple(EXTRA_VALUE_OPTIONS[int(rng.integers(0, len(EXTRA_VALUE_OPTIONS)))])
        specs.append(extra)
    rng.shuffle(specs)
    return specs


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    matched_value_specs = _sample_matched_value_specs(rng, n_frames)
    total_match = sum(len(spec) for spec in matched_value_specs)

    valid_distractors = [
        d
        for d in range(*N_DISTRACTORS_RANGE)
        if TOTAL_PATTERNS_RANGE[0] <= total_match + d < TOTAL_PATTERNS_RANGE[1]
    ]
    if not valid_distractors:
        return None
    n_distractors = int(valid_distractors[int(rng.integers(0, len(valid_distractors)))])

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    tallies = [sum(spec) for spec in matched_value_specs]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, tally) in enumerate(zip(frame_colors, tallies)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, tally), r0, FRAME_C0)

    inp = out.copy()
    for r0, tally in zip(frame_rows, tallies):
        bbox = (r0, FRAME_C0, r0 + FRAME_H - 1, FRAME_C0 + FRAME_W - 1)
        for rr, cc in _tally_positions_from_bbox(bbox, tally):
            inp[int(rr), int(cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    sprite_values_by_color = list(zip(frame_colors, matched_value_specs))

    sprite_jobs: list[tuple[int, int]] = []
    for color, spec in sprite_values_by_color:
        for value in spec:
            sprite_jobs.append((int(color), int(value)))
    for color in distractor_colors:
        value = int(SPRITE_VALUE_CHOICES[int(rng.integers(0, len(SPRITE_VALUE_CHOICES)))])
        sprite_jobs.append((int(color), value))
    rng.shuffle(sprite_jobs)

    try:
        for color, value in sprite_jobs:
            sprite = _make_sprite_from_value(rng, color, bg, value)
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
    Construct validated inputs for the area-count + multi-row combo.

    Generator-side guarantees for the solver's new assumptions:
    - all counted sprite areas are multiples of 2, so `total_area // 2` is exact;
    - at least one frame needs more than one tally row, usually all three rows;
    - at least one pair of frames has the same sprite count but different area totals;
    - every returned grid is checked by `main(inp) == out`.
    """
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
