# Combo variant of 8f215267 (index 14)
# combo_spec: D5/1+D7/0
# Name: 垂直点+保留精灵

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace
#    原题流程是：右侧散布的小单色精灵先被清除，再按“框颜色 = 精灵颜色”的计数，
#    在框的中心行从右向左隔 2 列画点。这个融合版同时改两个环节：
#    - D7/0：右侧精灵不再被移除，输出中保持原位。
#    - D5/1：框内 tally 不再横排，而是在中心列自底向上竖排。
#    一个 3 计数例子：
#    - 原题：蓝色精灵消失，蓝框中心行的右侧 3 个位置出现蓝点。
#    - 融合后：蓝色精灵仍留在右侧，蓝框中心列的底部 3 个内部位置出现蓝点。
#    人类对比输入输出时看到的是“框里新增了竖直点列，同时源精灵没动”。
#
# b) Complexity calibration
#    若沿用 5x11 框，高度只够 3 个内部竖直点，最大计数会退化。
#    因此沿用 D5 单变体的最小适配：把框高改成 6，宽仍为 11，计数上限保留 4。
#    这样每个样例仍有 3-4 个框、5-10 个总精灵、1-2 个干扰颜色，视觉证据足够。
#
# c) Implicit-assumption audit for main()
#    - out = g.copy()：在 D7/0 下仍成立，因为输出保留输入里的精灵。
#    - find_components + frame 判定：依赖“大而稀疏的是框，小而紧凑的是精灵”。
#      6x11 空心框仍满足；右侧精灵尺寸继续限制在 4x3 以内，避免误判。
#    - 原题的“删除 patterns”循环依赖“精灵会消失”这一旧前提；融合后该前提失效，必须去掉。
#    - 画点逻辑改为依赖“框内部中心列有足够空位”。生成器通过 FRAME_H=6、MAX_DOTS_PER_FRAME=4
#      和输入空框构造来保证这个前提成立。
#    - 由于精灵被保留，solver 还隐含要求精灵不会和左侧框接触，否则组件分割会失真。
#      生成器继续把精灵限制在独立右侧区域。
#
# d) Placement legality
#    框固定在左侧，右侧精灵放进单独画布后再 paste 到主网格，且 RIGHT_C0=13 与框区域分离。
#    random_free_location(..., padding=1) 负责避免精灵彼此重叠或粘连；精灵 bbox 也受尺寸上限约束。
#
# e) Visual observability
#    该融合版要求样例同时暴露两条规律：右侧精灵输入输出完全不变，框内新增标记改成竖向堆叠。
#    只要有 1 个以上匹配精灵，竖直列与原题横向条的区别肉眼就非常明显。
#
# f) Visual cleanliness
#    左侧只放框和新增点，右侧只放源精灵，中间保留背景缝隙。输出只比输入多一层框内点，
#    不会遮挡右侧对象，也不会把多种标记混进同一块区域。
#
# g) Inferability of the change
#    颜色规则仍是直接的同色匹配，没有任何依赖颜色数值的隐藏排序。
#    干扰色精灵继续存在，但不会触发任何框内新增点，能帮助解题者排除错误归纳。

# --- layout geometry ---
FRAME_H = 6                    # raised so 4 vertical dots fit inside each frame
FRAME_W = 11
FRAME_R0 = 0                   # moved up so 4 taller frames still fit in <= 30 rows
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)        # 3 or 4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1 or 2 distractor sprites
TOTAL_PATTERNS_RANGE = (5, 11) # 5..10 total sprites
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)     # 2..8 cells
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _vertical_dot_rows(bbox: tuple[int, int, int, int], n_dots: int) -> list[int]:
    """Rows for bottom-up tally dots inside a hollow frame bbox."""
    r0, _c0, r1, _c1 = map(int, bbox)
    return [int(r1 - 1 - i) for i in range(int(n_dots)) if int(r1 - 1 - i) > int(r0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites for each frame and show the tally as vertical dots while
    leaving the source sprites in place.

    Input structure:
    - The background is the most frequent color.
    - Several large hollow rectangles of different colors are stacked on the left.
    - Small monochrome sprites are scattered in a separate right-side region.

    Rule:
    - Keep all scattered sprites unchanged in the output.
    - For each frame of color C, count how many small sprites also have color C.
    - Draw that many same-color dots on the frame's center column, stacked bottom-up.

    Key insight:
    - The scattered sprites are evidence to be counted, not objects to erase.
    - The only change from input to output is a vertical tally added inside each matching frame.
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

    for f in frames:
        color = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == color)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        c_mid = (c0 + c1) // 2
        for rr in _vertical_dot_rows((r0, c0, r1, c1), num_patterns):
            out[int(rr), int(c_mid)] = color

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with vertical tally dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    c_mid = FRAME_W // 2
    for rr in range(FRAME_H - 2, FRAME_H - 2 - int(n_dots), -1):
        patch[int(rr), int(c_mid)] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute total matching sprites across frames, capped per frame."""
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
        hi = min(MAX_DOTS_PER_FRAME, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val
    return counts


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
    """Generate a small random connected sprite, tight-cropped to its bbox."""
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
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors
    if total_match <= 0:
        return None

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    if sum(counts) != total_match:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # Build the input skeleton: empty left frames only.
    inp = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, color in enumerate(frame_colors):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(inp, _frame_patch(color, bg, 0), r0, FRAME_C0)

    # Scatter right-side sprites; these remain unchanged in the output.
    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [color for color, count in zip(frame_colors, counts) for _ in range(count)]
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

    # Expected output: identical to input, plus vertical tally dots inside each frame.
    out = inp.copy()
    c_mid = FRAME_C0 + FRAME_W // 2
    for r0, color, count in zip(frame_rows, frame_colors, counts):
        for rr in range(r0 + FRAME_H - 2, r0 + FRAME_H - 2 - int(count), -1):
            out[int(rr), int(c_mid)] = int(color)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
