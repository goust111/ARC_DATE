# Variant of 8f215267
# Dimension: D15 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——点的数量和位置完全由精灵计数和固定间距规则决定... → 图例指定点间距

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


"""
Design analysis for the D15 variant
==================================

variant_spec:
  preserved_core:
    - 输入里仍然有若干个不同颜色的 5x11 空心框架，竖直堆叠在左侧。
    - 右侧仍然有若干同色小精灵；输出仍然需要删除这些精灵。
    - 每个框内仍然画出与同色精灵数量相等的圆点，并且圆点都落在框的中心行上，从右侧开始依次摆放。
  changed_dimension:
    name: D15 symbol_instructions
    old_behavior: 点位步长恒为 2，solver 可以硬编码 "每隔 2 列"。
    new_behavior: 点位步长不再固定，而是由右上角图例给出。
  legend_geometry:
    shape: 右上角 2x(span+1) 小图例，表现为两根同色竖柱。
    encoding: 两根竖柱的列差就是 frame 内圆点的水平步长 spacing。
    allowed_spacing: [2, 3, 4]
    location: 顶部两行、最右若干列；该区域不会放置右侧精灵。
    persistence: 图例在 input/output 中保持不变，它是说明符而不是待消除对象。
  why_this_is_visible:
    - spacing=2 时框内点较密，spacing=4 时只有最右侧两个位置可用，视觉差异明显。
    - 图例位于固定角落且在输入输出中都保留，解题者可把它识别为“说明”而不是普通精灵。
    - 在生成时强制至少一个框的计数 >= 2，避免图例存在但步长变化在该样例中不可见。

behavior_trace:
  original_local_step:
    - 某框颜色为红，右侧有 3 个红精灵。
    - 原 solver 在框中心行写入列 c1-2, c1-4, c1-6。
  variant_local_step:
    - 同样有 3 个红精灵，但右上角图例两根竖柱相距 3 列。
    - 新 solver 先读出 spacing=3，再在框中心行写入 c1-2, c1-5, c1-8。
  difference:
    - 原代码把 "2" 当常量；变体要求该值从输入中的可见符号解码。

complexity_calibration:
  - spacing 只取 2/3/4 三档，避免自由度过大。
  - frame 宽度不变，因此每档 spacing 都对应稳定且有限的最大点数:
      spacing=2 -> max 4, spacing=3 -> max 3, spacing=4 -> max 2
  - 生成器按 spacing 调整计数上限，避免出现“数得出来但画不下”的非法样例。

implicit_assumption_audit:
  old_main_dependency_1:
    code: cc = c1 - 2 - 2 * i
    assumption: 点步长始终为 2。
    status_in_variant: 不成立，必须改为从图例解析出的 spacing。
  old_main_dependency_2:
    code: 所有非 frame 的 monochromatic component 都视为 pattern
    assumption: 输入中不存在其他小组件。
    status_in_variant: 不成立，因为角落图例本身也是小组件；必须先识别并屏蔽图例坐标。
  old_main_dependency_3:
    code: 删除 patterns 时直接把所有 pattern 坐标刷成 bg
    assumption: 所有小组件都应该消失。
    status_in_variant: 不成立，图例是说明符，输出里必须保留。

placement_legality:
  - 图例固定在右上角，右侧精灵从第 3 行开始放置，因此不与图例重叠。
  - frame 仍在左侧固定列范围，图例位于右侧固定角落，互不接触。
  - 按 spacing 动态计算 max_dots，再据此分配每个 frame 的计数，保证所有点都能落在框内。

visual_cleanliness:
  - 图例只占右上角一个很小区域，且形状规则，不与随机精灵混杂。
  - 精灵仍只在右区散布，框仍只在左区，三类元素（框、精灵、图例）分区明确。

new_main_assumptions_and_generator_guarantees:
  - main 假设顶部两行最右侧窗口中恰好存在两根竖柱图例；generator 始终这样构造。
  - main 假设图例窗口内没有其他非背景像素；generator 为图例预留专用区域，不在此放 sprite。
  - main 假设每个 frame 的计数不超过当前 spacing 可容纳的上限；generator 按 spacing 约束计数。
"""


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = GRID_W - RIGHT_C0

# --- legend geometry ---
LEGEND_ROW = 0
LEGEND_H = 2
LEGEND_ALLOWED_SPACINGS = (2, 3, 4)
LEGEND_MAX_SPACING = max(LEGEND_ALLOWED_SPACINGS)
LEGEND_WINDOW_W = LEGEND_MAX_SPACING + 2
SPRITE_R0 = 3

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)  # inclusive support: 5..10 via rng upper-bound handling below
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _max_dots_for_spacing(spacing: int) -> int:
    """Number of usable dot slots in an 11-wide frame when stepping by `spacing`."""
    return 1 + (FRAME_W - 4) // int(spacing)


def _legend_patch(color: int, bg: int, spacing: int) -> np.ndarray:
    """Two vertical posts whose column difference encodes the frame dot spacing."""
    patch = np.full((LEGEND_H, spacing + 1), bg, dtype=int)
    patch[:, 0] = color
    patch[:, spacing] = color
    return patch


def _legend_cols(width: int, spacing: int) -> tuple[int, int]:
    """Absolute legend columns after right-aligning the patch to the grid edge."""
    c_right = int(width - 1)
    c_left = int(c_right - spacing)
    return c_left, c_right


def _legend_mask(shape: tuple[int, int], spacing: int) -> np.ndarray:
    """Boolean mask for the legend cells."""
    h, w = map(int, shape)
    mask = np.zeros((h, w), dtype=bool)
    c_left, c_right = _legend_cols(w, spacing)
    mask[LEGEND_ROW : LEGEND_ROW + LEGEND_H, c_left] = True
    mask[LEGEND_ROW : LEGEND_ROW + LEGEND_H, c_right] = True
    return mask


def _extract_spacing(grid: np.ndarray, bg: int) -> tuple[int, np.ndarray]:
    """
    Read the top-right legend and return (spacing, legend_mask).

    The generator guarantees that within the top-two-row, right-corner window there are
    exactly two occupied columns, each forming a 2-cell vertical post.
    """
    h, w = grid.shape
    if h < LEGEND_H:
        raise ValueError("grid too short for legend")

    c0 = max(0, w - LEGEND_WINDOW_W)
    window = grid[LEGEND_ROW : LEGEND_ROW + LEGEND_H, c0:w]
    cols = []
    for local_c in range(window.shape[1]):
        col = window[:, local_c]
        if np.all(col != bg):
            cols.append(c0 + local_c)

    if len(cols) != 2:
        raise ValueError("failed to parse two legend posts")

    spacing = int(cols[1] - cols[0])
    if spacing not in LEGEND_ALLOWED_SPACINGS:
        raise ValueError(f"unsupported legend spacing: {spacing}")

    return spacing, _legend_mask((h, w), spacing)


def _frame_patch(color: int, bg: int, n_dots: int, spacing: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with spacing-controlled dots."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - spacing * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(
    rng: np.random.Generator, n_frames: int, total_match: int, max_per_frame: int
) -> list[int]:
    """Distribute total_match sprites across n_frames, each capped by current spacing capacity."""
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - max_per_frame * frames_left)
        hi = min(max_per_frame, remaining)
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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and render that tally as dots.

    D15 variant:
    - A top-right legend encodes the dot spacing.
    - The legend remains unchanged in output.
    - Only scattered sprites are removed and tallied.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    spacing, legend_mask = _extract_spacing(g, bg)

    work = g.copy()
    work[legend_mask] = bg
    regions = find_components(work, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

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

        r0, c0, _r1, c1 = map(int, f["bbox"])
        r_mid = int(r0 + FRAME_H // 2)
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - spacing * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    spacing = int(rng.choice(np.array(LEGEND_ALLOWED_SPACINGS)))
    max_dots_per_frame = _max_dots_for_spacing(spacing)

    total_match_lo = max(2, TOTAL_PATTERNS_RANGE[0] - n_distractors)
    total_match_hi = min(TOTAL_PATTERNS_RANGE[1] - 1 - n_distractors, n_frames * max_dots_per_frame)
    if total_match_lo > total_match_hi:
        return None

    palette = rng.choice(np.arange(10), size=2 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    legend_color = int(palette[1])
    frame_colors = [int(x) for x in palette[2 : 2 + n_frames]]
    distractor_colors = [int(x) for x in palette[2 + n_frames :]]

    counts = None
    for _ in range(50):
        total_match = int(rng.integers(total_match_lo, total_match_hi + 1))
        candidate = _sample_counts(rng, n_frames, total_match, max_dots_per_frame)
        if max(candidate) >= min(2, max_dots_per_frame):
            counts = candidate
            break
    if counts is None:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    paste(out, _legend_patch(legend_color, bg, spacing), LEGEND_ROW, GRID_W - (spacing + 1))

    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count, spacing), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - spacing * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height - SPRITE_R0, RIGHT_W), bg, dtype=int)
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

    paste(inp, right_canvas, SPRITE_R0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
