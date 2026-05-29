# Combo variant of 8f215267 (index 13)
# combo_spec: D5/4+D11/1
# Name: 连续线段+互补色
#
# Design analysis
# - concretization:
#   左侧仍是纵向堆叠的 5x11 空心框，右侧仍是散落的小单色 sprite。输入里的 sprite 继续用“框本身的颜色”
#   表示归属关系，所以计数逻辑没有改成跨色匹配；融合后的变化发生在输出编码阶段：
#   原题把计数 N 画成框同色的离散点列，这里改为从右侧锚点开始、长度为 N 的连续水平线段，
#   并且线段颜色不是框色，而是该框色在一个固定少量颜色对表中的互补色。
# - behavior trace:
#   例：若蓝框匹配到 3 个蓝 sprite，原题会在中心行画 3 个蓝点；D5/4 会改成 3 格蓝线；
#   本融合版进一步把这条 3 格线改成“蓝色的互补色”红色，所以同一个 N 会变成右锚定的红色短横线。
#   若样本里同时出现红框和蓝框，两者会各自按自己的 sprite 数量画线，但线色互换，有助于观察配对关系。
# - complexity calibration:
#   连续线段比隔列点更省空间，11 列宽的框内部可安全容纳 8 格，所以单框上限从原题的 4 提到 8。
#   同时每个样本强制包含一对完整互补框，且这两个框都有非零计数，其中至少一个计数 >= 2，
#   这样既能看出“不是散点而是线段”，也能看出“线色不是框色本身”。
# - implicit assumption review for main():
#   1. `find_components(..., monochromatic=True)` 仍要求输入对象都是单色连通块；生成器保留这一点。
#   2. `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 仍假设 sprite 不会像大空心框；
#      生成器继续把 sprite 限制在 4x3 以内，避免误判。
#   3. 新增假设是“框色属于固定互补表，且互补色不等于背景色”；生成器把背景限制在 0/9，
#      而框色只从 1-8 的四组互补对中选，确保输出线段始终可见。
#   4. 画线逻辑依赖 `c1-2` 到 `c0+1` 的 8 格内部空间；固定 11 列框宽恰好满足。
#   5. 若真实输入中的计数超过可容纳长度，solver 仍做安全截断；生成器不会触发该分支。
# - placement legality:
#   生成器采用 output-first：先构造带互补色线段的目标图，再删除线段得到输入，最后只在右侧子画布放置
#   不重叠 sprite。`random_free_location` 继续保证 sprite 彼此之间留空隙且不越界。
# - visual observability:
#   少量固定颜色对比“任意颜色重映射”更容易从 2-4 组样例归纳。每个样本至少出现一整对互补框，
#   输出里两种框会表现为“边框颜色 A / 线段颜色 B”与“边框颜色 B / 线段颜色 A”的呼应关系。
#   再加上至少一条长度 >= 2 的连续线段，人眼能同时看出颜色交换和几何布局变化。
# - cleanliness:
#   线段仍只出现在框中心行，且与边框保持 1 格间隔；右侧 sprite 区域与左侧框区分离，不会产生覆盖混淆。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


COMPLEMENT_COLOR = {
    1: 2,
    2: 1,
    3: 6,
    6: 3,
    4: 8,
    8: 4,
    5: 7,
    7: 5,
}
COMPLEMENT_PAIRS = ((1, 2), (3, 6), (4, 8), (5, 7))
PAIRABLE_COLORS = tuple(sorted(COMPLEMENT_COLOR))
BG_CHOICES = (0, 9)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count frame-matching sprites and encode each tally as a complementary-color line segment.

    Input structure:
    - The background is the most common color.
    - Several hollow rectangular frames sit on the left, each in a color from a small fixed pairing table.
    - Small monochrome sprites are scattered away from the frames and still use the frame's own color.

    Rule:
    - Remove every small sprite from the scene.
    - For each frame of color C, count how many sprites also have color C.
    - On the frame's center row, draw a right-anchored continuous horizontal segment of length N,
      but color that segment with the paired complement of C rather than with C itself.

    Key insight:
    - The sprites still determine the quantity by same-color counting, but the displayed mark goes through
      a second learned color-pair lookup, so both the geometry and the displayed color differ from the base task.
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
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        max_fill = max(0, c1 - c0 - 2)
        fill_len = min(int(num_patterns), max_fill)
        line_color = int(COMPLEMENT_COLOR.get(frame_color, frame_color))
        for cc in range(int(c1 - 2), int(c1 - 2 - fill_len), -1):
            if cc > int(c0):
                out[int(r_mid), int(cc)] = line_color

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
N_FRAMES_RANGE = (3, 5)        # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)   # 1-2 distractors
TOTAL_PATTERNS_RANGE = (6, 13) # 6-12 total sprites
MAX_LINE_LEN = FRAME_W - 3     # 8 cells from c1-2 down to c0+1
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


def _frame_patch(frame_color: int, line_color: int, bg: int, n_fill: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with a complementary-color center-row segment."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = frame_color
    patch[-1, :] = frame_color
    patch[:, 0] = frame_color
    patch[:, -1] = frame_color
    patch[FRAME_H // 2, _line_cols(n_fill)] = line_color
    return patch


def _sample_frame_colors(
    rng: np.random.Generator, n_frames: int
) -> tuple[list[int], tuple[int, int], list[int]]:
    """
    Choose distinct frame colors while forcing one full complement pair to appear.

    Returns:
        frame_colors: colors in display order
        focal_pair: the forced complete pair
        focal_positions: indices of the pair members inside frame_colors
    """
    pair_idx = int(rng.integers(0, len(COMPLEMENT_PAIRS)))
    focal_pair = tuple(int(x) for x in COMPLEMENT_PAIRS[pair_idx])
    remaining_pool = [c for c in PAIRABLE_COLORS if c not in focal_pair]
    extra = [int(x) for x in rng.choice(np.array(remaining_pool), size=n_frames - 2, replace=False)]
    frame_colors = [*focal_pair, *extra]
    rng.shuffle(frame_colors)
    focal_positions = [i for i, color in enumerate(frame_colors) if color in focal_pair]
    return frame_colors, focal_pair, focal_positions


def _sample_counts(
    rng: np.random.Generator, n_frames: int, total_match: int, focal_positions: list[int]
) -> list[int]:
    """
    Distribute matching sprites across frames with three visibility constraints:
    - both focal complement frames get non-zero counts;
    - at least one frame has count >= 2;
    - each frame stays within the 8-cell line capacity.
    """
    counts = [0] * n_frames
    seed = [1, 2]
    rng.shuffle(seed)
    for idx, val in zip(focal_positions, seed):
        counts[int(idx)] = int(val)
    remaining = total_match - sum(seed)

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

    bg = int(BG_CHOICES[int(rng.integers(0, len(BG_CHOICES)))])
    frame_colors, _, focal_positions = _sample_frame_colors(rng, n_frames)
    counts = _sample_counts(rng, n_frames, total_match, focal_positions)

    available_distractors = [c for c in PAIRABLE_COLORS if c not in frame_colors]
    distractor_colors = [
        int(x)
        for x in rng.choice(np.array(available_distractors), size=n_distractors, replace=False)
    ]

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (frame_color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        line_color = int(COMPLEMENT_COLOR[frame_color])
        paste(out, _frame_patch(frame_color, line_color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        line_cols = [FRAME_C0 + c for c in _line_cols(count)]
        inp[mid_r, line_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [color for color, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, int(color), bg)
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
