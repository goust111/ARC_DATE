# Combo variant of 8f215267 (index 10)
# combo_spec: D4/1+D11/0
# Name: 计数取反+统一点颜色
#
# Design analysis for this combo variant
# a) 具体化与行为追踪
#    - 画面对象仍是原题两大类：左侧竖直堆叠的 5x11 彩色空心框，右侧散布的小型单色精灵。
#    - 输入里框内部保持空白；输出里只在框的中心行放置圆点，位置仍是从右往左每隔 2 列的 4 个槽位。
#    - 原题流程：
#      1. 按框色统计右侧同色精灵数 N；
#      2. 在该框里画 N 个、且颜色也等于框色的点。
#    - 融合后流程：
#      1. 仍先按框色统计同色精灵数 N；
#      2. 但显示数量改成 (4-N)；
#      3. 且所有框内的点都改用同一个固定颜色，而不是各框自己的颜色。
#    - 例子：若三个框的真实匹配数是 [4, 1, 0]，
#      原题会画 [4, 1, 0] 个对应框色的点；
#      本变体会画 [0, 3, 4] 个统一颜色的点。
#      因此解题者必须同时发现“计数取反”和“点色统一”这两个变化。
# b) 复杂度校准
#    - 不改框形状、精灵形状、左右分区和颜色匹配机制，只改最后的显示规则，避免把任务复杂度转移到别的维度。
#    - 为了让融合效果在 2-3 个样例里足够醒目，生成器强制每个样例至少出现：
#      一个 N=0 的框、一个 N=4 的框、以及一个 1..3 的中间档位。
#      这样输出里一定同时出现“满 4 点”“0 点”“中间点数”，反比关系不会退化。
#    - 同时因为 N=0 和中间档位都会产生可见点，所以每个样例里至少有两个不同框显示同一种点色，
#      统一颜色这条规则也会被重复支撑，而不是只出现一次偶然现象。
# c) 隐式假设审查（针对 main）
#    - `find_components(... monochromatic=True)` 仍成立：输入中的框和精灵都还是单色连通块。
#    - `is_frame = (h>=5 and w>=5 and area < h*w*0.7)` 仍成立：输入中框内部没有点，不会把框误判成实心块。
#    - “先擦掉 patterns” 仍安全：精灵只出现在右侧区域，不接触框，擦除不会伤到框。
#    - 原代码默认“要画的点数 = 同色精灵数”；这里必须显式改成“槽位总数 - 同色精灵数”。
#    - 原代码默认“点色 = 框色”；这里必须显式改成全局常量 `UNIFIED_DOT_COLOR`。
#    - 新 solver 额外依赖 `0 <= N <= 4`，否则 `4-N` 会越界；生成器通过采样约束保证这一点。
#    - 新 solver 还依赖固定点色不会与框色混淆；生成器把该颜色从背景、框色、干扰色候选里排除。
# d) 放置合法性
#    - 仍沿用 output-first：先构造带统一点色的目标输出，再删点得到输入，再把真实精灵放到右侧。
#    - 右侧精灵继续使用 `random_free_location(... padding=1)`，保证彼此不重叠、不越界，也不粘连成大块。
#    - 所有候选输入最后都要通过 `np.array_equal(main(inp), out)` 验证后才返回。
# e) 视觉可观察性
#    - 颜色值本身不参与数值比较；人类只需要用“颜色是否相同”来做框-精灵匹配。
#    - 反比关系通过 [0, 4, 中间] 三档真实计数直接可见；统一点色通过“不同框里出现相同颜色的点”直接可见。
#    - 点色固定且不与任何框色重合，避免解题者把它误解为某个额外类别的框。
# f) 可视化整洁度
#    - 左侧只放框，右侧只放精灵，区域职责单一；框内标记仍限制在中心行四个离散槽位。
#    - 统一点色虽然跨框复用，但因为点始终待在各自框内，不会和右侧精灵或框边界混叠。
# g) 变化的可推断性
#    - 颜色变化只有一条：所有点都共享一个固定颜色。
#    - 这条规则有明确结构线索：同一位置类型（框中心行槽位）跨多个不同框重复出现同色标记。
#    - 生成器避免让该固定颜色出现在框或精灵里，从而把“这是显示色，不是类别色”表达得更清楚。

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# --------------- shared geometry ---------------
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9
MAX_DOTS_PER_FRAME = 4
UNIFIED_DOT_COLOR = 2


def _dot_columns(c0: int, c1: int) -> list[int]:
    """Valid center-row dot columns inside a frame, ordered from right to left."""
    return [int(cc) for cc in range(int(c1) - 2, int(c0), -2)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Turn color-matched sprite counts into inverse tallies shown with one shared marker color.

    Input structure:
    - The background is the most frequent color.
    - Several large hollow rectangles of distinct colors are stacked on the left.
    - Small monochrome sprites are scattered in the right-side region.
    - Frame interiors are empty in the input; only frame color tells which sprites belong to which frame.

    Rule:
    - Remove every small sprite from the grid.
    - For each frame, count same-color sprites N.
    - Compute how many center-row slots remain after inversion: MAX - N, with MAX given by the frame's legal dot slots.
    - Fill that many rightmost slots, spaced every other cell, using the single fixed dot color for every frame.

    Key insight:
    - Frame colors still define the counting groups, but the displayed tally breaks two original expectations at once:
      more matching sprites produce fewer dots, and those dots all share one global display color instead of copying the frame color.
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
        frame_color = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == frame_color)

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        slots = _dot_columns(c0, c1)
        dots_to_draw = max(0, len(slots) - num_patterns)
        for cc in slots[:dots_to_draw]:
            out[int(r_mid), int(cc)] = UNIFIED_DOT_COLOR

    return out


# --------------- generator ---------------
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
BOTTOM_PAD_RANGE = (0, 7)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MAX_ATTEMPTS = 300
COUNT_SAMPLE_ATTEMPTS = 200


def _frame_patch(frame_color: int, bg: int, n_dots: int, dot_color: int) -> np.ndarray:
    """Build a hollow frame whose center-row tally dots use the shared dot color."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    mid_r = FRAME_H // 2
    local_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, local_cols] = dot_color
    return patch


def _sample_match_counts(rng: np.random.Generator, n_frames: int, n_distractors: int) -> list[int] | None:
    """
    Sample same-color sprite counts per frame.

    Observability constraints:
    - include N=0 so one frame shows 4 dots,
    - include N=4 so one frame shows 0 dots,
    - include one intermediate N in 1..3 so inversion is not reducible to a binary rule.
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

    available_colors = [c for c in range(10) if c != UNIFIED_DOT_COLOR]
    palette = rng.choice(available_colors, size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    display_counts = [MAX_DOTS_PER_FRAME - n for n in match_counts]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (frame_color, n_dots) in enumerate(zip(frame_colors, display_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(frame_color, bg, n_dots, UNIFIED_DOT_COLOR), r0, FRAME_C0)

    inp = out.copy()
    for r0, n_dots in zip(frame_rows, display_counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(n_dots)]
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
    Generator guarantees for the fused solver assumptions:
    - every frame's true same-color sprite count stays in 0..4,
    - each sample contains low/high/intermediate counts so inversion is visible,
    - the unified dot color is excluded from bg, frame colors, and distractor colors,
    - at least two frames display visible dots of that shared color in each accepted sample,
    - only validated input grids are returned.
    """
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
