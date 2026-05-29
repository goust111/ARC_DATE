# Variant of 8f215267
# Dimension: D5 — 变换规则：点的布局方式 (transform)
# Change: 在框架中心行从右侧起每隔2列放置N个同色点（列坐标: c1-2, c1-4, c1-6, ...）... → 填充连续线段而非离散点
#
# Design analysis
# - requested_reference: 16de56c4
# - reference_solver_status: 当前工作区中不存在 16de56c4 的 solver 文件，只找到题目数据与说明文本；
#   因此“先阅读参考 solver”这一步按“不存在就跳过”处理，不引入外部实现。
# - concretization:
#   输入仍然由左侧多色空心矩形框和右侧散落的小同色 sprite 构成。输出仍然会清除右侧 sprite，
#   并在每个框的中心行写回同色计数。唯一核心变化是：原题把 N 表示成从右往左隔 1 格放置的 N 个离散点；
#   本变体改为把同一个 N 表示成从右往左的长度为 N 的连续水平线段。
#   为了尽量只改 D5，本实现保留原题的右侧锚点：线段最右端仍落在 c1-2，而不是贴住框右边。
#   因此长度为 N 的线段占据列 c1-2, c1-3, ..., c1-(N+1)。
# - behavior trace:
#   例：若某个颜色框原来匹配到 3 个 sprite，原 solver 会在中心行写入 c1-2/c1-4/c1-6 三个点；
#   现在改为在 c1-2/c1-3/c1-4 写一段长度 3 的连续条。计数来源没变，只有可视化布局变了。
# - complexity calibration:
#   原题因为隔列放点，5x11 框的可容纳上限是 4。连续线段可以在不碰边框的前提下容纳 8 格，
#   所以生成器把单框上限提升到 8，这是与 D5 直接相关的适配，不改变“按颜色计数”的本质。
#   同时强制每个样本至少有一个 N>=2 的框，避免输出退化成单点而让新规则不可见。
# - implicit assumption review for main():
#   1. `find_components(..., monochromatic=True)` 假设框和 sprite 都是单色连通块；生成器继续满足。
#   2. `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 假设 sprite 不会长成大空心框；生成器继续把
#      sprite 限制在 4x3 以内，因此不会误判。
#   3. 画线时依赖“框内部至少有 8 个可用格”的前提；生成器固定框宽 11，因此 `c1-2` 到 `c0+1` 恰好可放 8 格。
#   4. 如果真实输入里计数大于可放长度，solver 会像原题一样做截断式安全处理；生成器则保证不会触发。
# - placement legality:
#   框仍然固定在左侧纵向堆叠，sprite 仍然只放在右侧子画布，通过 `random_free_location` 保证不重叠、
#   不越界，并保留 padding。输出先构造带线段的目标图，再删线段得到输入，所以 `main(input)==output`
#   的校验仍然直接适用。
# - visual observability:
#   新旧规则在 N>=2 时肉眼差异明显：原题是稀疏点列，本变体是短横线。生成器显式制造至少一个 N>=2
#   的框，并允许出现 N>4 的长条，保证少量样例就能看出“不是点，而是连续填充”。
# - cleanliness:
#   线段只出现在框中心行，右侧 sprite 区域与框区域完全分离，因此不会引入新的视觉遮挡。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as a right-anchored line segment.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Draw a continuous horizontal segment of that length on the frame's center row,
      filling from the right anchor at c1-2 leftward.
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
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        max_fill = max(0, c1 - c0 - 2)
        fill_len = min(int(num_patterns), max_fill)
        for cc in range(int(c1 - 2), int(c1 - 2 - fill_len), -1):
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out


# --------------- generator ---------------

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
TOTAL_PATTERNS_RANGE = (6, 12)
MAX_LINE_LEN = FRAME_W - 3
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _line_cols(n_fill: int) -> list[int]:
    """Interior-relative columns for a right-anchored continuous line segment."""
    if n_fill <= 0:
        return []
    start = FRAME_W - 3
    return list(range(start, start - int(n_fill), -1))


def _frame_patch(color: int, bg: int, n_fill: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with a center-row line segment."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    patch[mid_r, _line_cols(n_fill)] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """
    Distribute total_match sprites across n_frames, each capped at MAX_LINE_LEN.

    One frame is seeded with at least 2 matches so the output visibly differs
    from the original dotted rule.
    """
    counts = [0] * n_frames
    prominent_idx = int(rng.integers(0, n_frames))
    seed_fill = int(rng.integers(2, min(MAX_LINE_LEN, total_match) + 1))
    counts[prominent_idx] = seed_fill
    remaining = total_match - seed_fill

    order = [int(x) for x in rng.permutation(n_frames)]
    for pos, idx in enumerate(order):
        capacity = MAX_LINE_LEN - counts[idx]
        future_capacity = sum(MAX_LINE_LEN - counts[j] for j in order[pos + 1 :])
        lo = max(0, remaining - future_capacity)
        hi = min(capacity, remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

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
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    if max(counts) < 2:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        line_cols = [FRAME_C0 + c for c in _line_cols(count)]
        inp[mid_r, line_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
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
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
