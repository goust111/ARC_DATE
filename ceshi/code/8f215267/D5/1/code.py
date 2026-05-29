# Variant of 8f215267
# Dimension: D5 — 变换规则：点的布局方式 (transform)
# Change: 在框架中心行从右侧起每隔2列放置N个同色点（列坐标: c1-2, c1-4, c1-6, ...）... → 在中心列垂直放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the D5 variant
# ----------------------------------
# variant_spec:
#   unchanged_core:
#     - 背景色仍为整张图里出现最多的颜色。
#     - 左侧仍是若干个不同颜色的空心矩形框；右侧仍是同色/干扰色的小单色精灵。
#     - 颜色匹配关系不变：每个框只统计与自身颜色相同的散布精灵个数。
#     - 输出仍然先清除右侧所有精灵，再把计数结果显示回对应颜色的框内。
#   changed_dimension:
#     - 原题把计数点画在框的中心行，从右往左隔一列放置。
#     - 本变体把计数点改画在框的中心列，从底部向上逐行堆叠。
#     - 具体坐标：若框 bbox 为 (r0, c0, r1, c1)，中心列 cc=(c0+c1)//2，
#       第 i 个点画在 rr=r1-1-i，i=0,1,2,...，要求 rr>r0 才是框内可见位置。
#   behavior_trace:
#     - 原逻辑例子：某个蓝框统计到 3 个蓝色精灵，则在该框中心行的列 c1-2, c1-4, c1-6 画 3 个蓝点。
#     - 新逻辑例子：同样统计到 3 个蓝色精灵，则改为在中心列的行 r1-1, r1-2, r1-3 画 3 个蓝点。
#     - 差异不在“计数”而在“呈现轴向”：横向计数条变成竖向计数柱。
#   complexity_calibration:
#     - 若继续使用原来的 5x11 框，高度只能容纳 3 个可见内部竖点，计数上限会退化。
#     - 为保留原题“每框最多 4 个计数”的复杂度，最小适配是把框高增至 6，同时保持宽度 11。
#     - 为了让 4 个 6x11 框仍落在 ARC 常见的 30 行以内，首框顶边改到第 0 行；其余间距不变。
#   solver_assumption_audit:
#     - 组件分类仍依赖“框足够大且面积远小于外接矩形面积”，6x11 空心框仍满足该判定。
#     - 原 main() 里真正依赖旧布局的只有 r_mid 与 cc=c1-2-2*i 这一段；必须整体替换。
#     - 删除散布精灵的循环不依赖点布局，保留即可。
#     - 新布局依赖“框内部至少有 4 个可见行位”；生成器通过 FRAME_H=6 和 MAX_DOTS_PER_FRAME=4 保证。
#   placement_legality:
#     - generator 仍采用 output-first：先构造带竖直点列的输出，再删除点得到输入，再在右侧区域放精灵。
#     - 框之间保留 2 行背景带；右侧精灵只放在 cols [13, 21]，不会接触左侧框。
#     - 每个精灵使用 random_free_location 并带 1 格 padding，避免重叠和粘连。
#   observability_and_cleanliness:
#     - 竖直点列与原题横向点列在肉眼上差异直接可见，尤其在计数 2-4 时尤为明显。
#     - 同一框内只保留一种内部标记方式，不叠加其他结构；视觉仍然整洁。
#     - 颜色含义未变，不引入任何基于颜色数值大小的不可见规则。


def _vertical_dot_rows(bbox: tuple[int, int, int, int], n_dots: int) -> list[int]:
    """Rows for bottom-up dots inside a hollow frame bbox."""
    r0, _c0, _r1, _c1 = map(int, bbox)
    return [int(_r1 - 1 - i) for i in range(int(n_dots)) if int(_r1 - 1 - i) > int(r0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as vertical dots inside.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors.
    - Small monochrome pattern sprites scattered outside frames.

    Rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count how many patterns share color C.
    - Draw that many dots on the frame's center column, stacked bottom-up.

    Key insight: Frames still act as tally displays; only the in-frame tally layout changes
    from horizontal to vertical.
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
        c_mid = (c0 + c1) // 2
        for rr in _vertical_dot_rows((r0, c0, r1, c1), num_patterns):
            out[int(rr), int(c_mid)] = col

    return out


# --------------- generator (codex-att2, D5 vertical tally variant) ---------------

# --- layout geometry ---
FRAME_H = 6                    # frame height; raised from 5 so 4 vertical dots fit visibly
FRAME_W = 11                   # frame width
FRAME_R0 = 0                   # top-left row of first frame; moved up so 4 frames still fit
FRAME_C0 = 1                   # top-left col of all frames
FRAME_GAP = 2                  # bg rows between consecutive frames
GRID_W = 23                    # fixed grid width
RIGHT_C0 = 13                  # first column of the right (sprite) region
RIGHT_W = 9                    # width of the right region

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)        # [lo, hi) for number of frames
N_DISTRACTORS_RANGE = (1, 3)   # [lo, hi) for distractor sprite count
TOTAL_PATTERNS_RANGE = (5, 11) # [lo, hi) for total sprite count
MAX_DOTS_PER_FRAME = 4         # preserved: 6-row frame has 4 visible interior rows
BOTTOM_PAD_RANGE = (0, 7)      # [lo, hi) extra bg rows when n_frames==3

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)     # [lo, hi) connected-cell count per sprite
SPRITE_MAX_H = 4               # max bbox height for sprite
SPRITE_MAX_W = 3               # max bbox width for sprite
SPRITE_PADDING = 1             # gap between placed sprites

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center column."""
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
    """Distribute total_match sprites across n_frames, each capped at MAX_DOTS_PER_FRAME."""
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
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    c_mid = FRAME_C0 + FRAME_W // 2
    for r0, count in zip(frame_rows, counts):
        for rr in range(r0 + FRAME_H - 2, r0 + FRAME_H - 2 - int(count), -1):
            inp[int(rr), int(c_mid)] = bg

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
