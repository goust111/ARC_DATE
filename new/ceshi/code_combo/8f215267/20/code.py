# Combo variant of 8f215267 (index 20)
# combo_spec: D5/1+D15/2
# Name: 控制像素指定方向+垂直放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior / trace
#    - 原题: 每个彩色空心框统计右侧同色精灵个数，然后在框中心行从右往左画水平点列。
#    - D5/1 改动: 计数点改为画在中心列，竖直堆叠。
#    - D15/2 改动: 每个框多一个同色控制像素，控制“从哪一端开始画”。
#    - 融合后: 若某个蓝框匹配到 3 个蓝色精灵，且控制像素在框上方，
#      则输出在该框中心列的上侧内边起连续画 3 个蓝点；若控制像素在下方，
#      则改为从下侧内边向上连续画 3 个蓝点。
#    - 与原题的核心差异有两层同时生效:
#      1) 呈现轴向由水平变为垂直。
#      2) 垂直点列不是固定自底向上，而是由每个框自己的控制像素决定起始端。
#
# b) Complexity calibration
#    - 方向只保留 top / bottom 两档，样例少时仍可归纳。
#    - 为保留“每框最多 4 个点”的原始复杂度，真实框高度调到 6，
#      这样中心列内部恰有 4 个可见位置可连续堆叠。
#    - 控制像素额外占用 1 行，因此每个框保留块是 7x11；4 个框加间隔仍可稳定生成。
#    - generator 强制 top / bottom 两种方向都出现，且两类中至少各有一个 1..3 点的框，
#      避免“正好填满 4 格时上下都一样”的退化样本。
#
# c) Implicit assumptions audit on main()
#    - 原代码把 bbox 直接当作框本体；融合后不成立，因为 bbox 会把外接控制像素也包进去。
#      所以 solver 必须先解析“顶行单像素还是底行单像素”，再恢复真实框 bbox。
#    - 原代码默认点总在中心行、按固定水平间距绘制；融合后改为中心列连续竖排，
#      且起始端取决于方向控制。
#    - 原代码按“大且稀疏”为框、“小组件”为精灵来分组；这里仍成立，
#      因为框面积约 31，而精灵面积保持在 2-8。
#    - 原代码删除所有小组件即可清空右侧精灵；这里仍成立，前提是控制像素必须与框连通。
#      generator 固定将控制像素贴在框中心正上/正下方，保证不会误判成独立组件。
#
# d) Placement legality
#    - 左侧框统一放在固定列范围，右侧精灵只放在独立右区域，不会与框重叠。
#    - 框保留块之间保留 1 行背景，避免视觉拥挤；不同颜色也避免组件误并。
#    - 右侧精灵继续用 random_free_location 加 1 格 padding，防止相互粘连。
#
# e) Visual observability
#    - 解题者能直接看到每个框在上边或下边多出一个同色单像素。
#    - 输出中的点列严格沿中心列靠控制侧连续生长；比较两个方向相反但计数相近的框，
#      规则非常直观。
#    - 颜色只承担匹配关系，不承担任何“数值大小”类的不可视语义。
#
# f) Visual cleanliness
#    - 控制像素与框同色，位置固定在中心轴上，看起来像框的方向标记而不是新对象。
#    - 框内只有一种编码方式: 单列竖点；不再混入水平间隔或额外符号，画面保持整洁。
#
# g) Inferability / non-degeneracy
#    - 融合规则真正依赖两个变量: 颜色匹配给出数量，控制像素给出起始方向。
#    - 若忽略控制像素并一律从底部画，含 top 正计数字框的样例会得到不同输出。
#      generator 通过方向与计数约束，系统性避免退化回“纯 D5/1 底部堆叠”。
#
# h) New solver assumptions and generator guarantees
#    - solver 假设框组件满足: 顶行或底行恰有 1 个同色像素，另一端是完整横边。
#      generator 固定按此构造。
#    - solver 假设单框计数不超过 4，且真实框中心列内部有 4 个可用格。
#      generator 通过 FRAME_RING_H=6 和 MAX_DOTS_PER_FRAME=4 保证。
#    - solver 假设精灵是小型单色连通块；generator 保持该约束不变。


FRAME_RING_H = 6             # actual hollow frame height
FRAME_W = 11                 # frame width
PATCH_H = FRAME_RING_H + 1   # one extra row for the top/bottom control pixel
FRAME_R0 = 0
FRAME_C0 = 1
FRAME_GAP = 1
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

N_FRAMES_RANGE = (3, 5)       # 3-4 frames
N_DISTRACTORS_RANGE = (1, 3)  # 1-2 distractors
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 4)

SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

MIN_FRAME_AREA = 30
MAX_ATTEMPTS = 300


def _frame_geometry(direction: str) -> tuple[int, int, int]:
    ring_r0 = 1 if direction == "top" else 0
    ring_r1 = ring_r0 + FRAME_RING_H - 1
    c_mid = FRAME_W // 2
    return int(ring_r0), int(ring_r1), int(c_mid)


def _dot_rows_for_ring(direction: str, ring_r0: int, ring_r1: int, n_dots: int) -> list[int]:
    if direction == "top":
        return [int(ring_r0 + 1 + i) for i in range(int(n_dots)) if int(ring_r0 + 1 + i) < int(ring_r1)]
    return [int(ring_r1 - 1 - i) for i in range(int(n_dots)) if int(ring_r1 - 1 - i) > int(ring_r0)]


def _frame_patch(color: int, bg: int, n_dots: int, direction: str) -> np.ndarray:
    """Build a 7x11 patch: 6x11 hollow frame plus one control pixel above or below."""
    patch = np.full((PATCH_H, FRAME_W), bg, dtype=int)
    ring_r0, ring_r1, c_mid = _frame_geometry(direction)
    patch[ring_r0, :] = color
    patch[ring_r1, :] = color
    patch[ring_r0 : ring_r1 + 1, 0] = color
    patch[ring_r0 : ring_r1 + 1, -1] = color

    ctrl_r = 0 if direction == "top" else PATCH_H - 1
    patch[int(ctrl_r), int(c_mid)] = color

    for rr in _dot_rows_for_ring(direction, ring_r0, ring_r1, n_dots):
        patch[int(rr), int(c_mid)] = color
    return patch


def _parse_frame(comp: dict) -> tuple[str, tuple[int, int, int, int]]:
    """
    Recover the real hollow-frame bbox from a frame+control component.

    Expected component silhouettes:
    - top control:    bbox top row has exactly 1 pixel, bottom row is the full frame edge
    - bottom control: bbox bottom row has exactly 1 pixel, top row is the full frame edge
    """
    r0, c0, r1, c1 = map(int, comp["bbox"])
    rows = comp["coords"][:, 0]
    top_count = int(np.sum(rows == int(r0)))
    bottom_count = int(np.sum(rows == int(r1)))
    width = int(c1 - c0 + 1)

    if top_count == 1 and bottom_count >= width:
        return "top", (int(r0 + 1), int(c0), int(r1), int(c1))
    if bottom_count == 1 and top_count >= width:
        return "bottom", (int(r0), int(c0), int(r1 - 1), int(c1))
    raise ValueError("frame component does not match the expected control-pixel layout")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and display the tally as a directed vertical stack.

    Input structure:
    - bg is the most frequent color.
    - Several hollow rectangular frames appear on the left, each with one same-color control
      pixel attached at the center of its top side or bottom side.
    - Small monochrome sprites are scattered outside the frames, mainly in the right region.

    Rule:
    - Erase all small sprites from the grid.
    - For each frame of color C, count how many sprites also use color C.
    - Draw that many C-colored dots on the frame's center column.
    - If the control pixel is above the frame, start at the top interior cell and grow downward.
      If the control pixel is below the frame, start at the bottom interior cell and grow upward.

    Key insight:
    - The frames still behave like color-matched counters, but the control pixel now selects
      which end of the vertical tally column is the starting side for that specific frame.
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
        is_frame = (area >= MIN_FRAME_AREA) and (h >= PATCH_H) and (w >= FRAME_W)
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        n_match = sum(1 for p in patterns if int(p["color"]) == color)
        if n_match <= 0:
            continue

        direction, ring_bbox = _parse_frame(f)
        r0, c0, r1, c1 = map(int, ring_bbox)
        c_mid = (int(c0) + int(c1)) // 2
        for rr in _dot_rows_for_ring(direction, r0, r1, n_match):
            out[int(rr), int(c_mid)] = color

    return out


def _sample_directions(rng: np.random.Generator, n_frames: int) -> list[str]:
    while True:
        directions = [str(rng.choice(["top", "bottom"])) for _ in range(n_frames)]
        if "top" in directions and "bottom" in directions:
            return directions


def _sample_counts(
    rng: np.random.Generator, n_frames: int, total_match: int, directions: list[str]
) -> list[int]:
    """Distribute matches while preserving one visibly directional frame per side."""
    counts = [0] * n_frames
    top_choices = [i for i, d in enumerate(directions) if d == "top"]
    bottom_choices = [i for i, d in enumerate(directions) if d == "bottom"]

    top_idx = int(top_choices[int(rng.integers(0, len(top_choices)))])
    bottom_idx = int(bottom_choices[int(rng.integers(0, len(bottom_choices)))])
    counts[top_idx] += 1
    counts[bottom_idx] += 1
    remaining = int(total_match - 2)

    while remaining > 0:
        candidates = []
        for i, v in enumerate(counts):
            cap = MAX_DOTS_PER_FRAME
            if i in (top_idx, bottom_idx):
                cap = MAX_DOTS_PER_FRAME - 1
            if v < cap:
                candidates.append(i)
        idx = int(candidates[int(rng.integers(0, len(candidates)))])
        counts[idx] += 1
        remaining -= 1
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
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
        patch[int(r - r_min), int(c - c_min)] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = int(total_patterns - n_distractors)

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    directions = _sample_directions(rng, n_frames)
    counts = _sample_counts(rng, n_frames, total_match, directions)

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * PATCH_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    inp = np.full((height, GRID_W), bg, dtype=int)

    for i, (color, count, direction) in enumerate(zip(frame_colors, counts, directions)):
        r0 = FRAME_R0 + i * (PATCH_H + FRAME_GAP)
        paste(out, _frame_patch(color, bg, count, direction), r0, FRAME_C0)
        paste(inp, _frame_patch(color, bg, 0, direction), r0, FRAME_C0)

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
