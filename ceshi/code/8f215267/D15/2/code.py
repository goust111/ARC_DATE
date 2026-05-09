# Variant of 8f215267
# Dimension: D15 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——点的数量和位置完全由精灵计数和固定间距规则决定
#   -> 控制像素指定绘制方向。每个框与框同色的单个外接像素位于上方或下方，
#      指示该框内的点要从顶部向下画，还是从底部向上画。
#
# Design analysis
# a) Concrete behavior / trace
#    - 原题输入: 左侧是若干同色空心框，右侧是散落的小精灵；输出把右侧精灵清掉，
#      并在对应颜色的框内用若干点表示计数。
#    - 变体输入: 仍然是按颜色计数，但每个框多一个“控制像素”。
#      该像素与框同色，固定贴在框的正上方或正下方中央，因此框组件仍然是单色连通块。
#      若控制像素在上方，则输出点沿框的中心列从顶部内侧开始，每隔 2 行向下绘制；
#      若控制像素在下方，则从底部内侧开始，每隔 2 行向上绘制。
#    - 具体小例子: 某个蓝框匹配到 3 个蓝色精灵。
#      原题会在中心行右侧画 3 个水平点；本变体若控制像素在上方，则改成
#      center-column 的第 1/3/5 个内侧行画 3 个点；若控制像素在下方，则画在
#      倒数第 1/3/5 个内侧行。计数没变，显示方向变了。
# b) Complexity calibration
#    - 只保留两个控制档位: top / bottom，避免方向过多导致少量样例不可归纳。
#    - 为了让“从上画/从下画”在视觉上真正可见，需要把框高从原来的 5 增到 11；
#      否则原题的最多 4 个点无法在竖直方向按 2 间隔容纳。
#    - 仍保留 3-4 个框、1-2 个干扰色和 0-4 的单框计数上限，难度与原题接近。
# c) Implicit assumptions audit on main()
#    - 原代码用 bbox + 面积把大框和小精灵区分开；新版本依然成立，但框现在带一个
#      外接控制像素，所以 bbox 高度会是 12 而不是 5。solver 不能再假定框 bbox
#      就是矩形外框本身，必须先从首尾行的像素数识别控制像素所在侧，再恢复真正的框 bbox。
#    - 原代码默认点总在中心行且从右向左画；这个前提已失效，必须改为读取控制方向，
#      并在中心列沿竖向绘制。
#    - 原代码无条件按颜色计数所有小组件；新版本仍然这样做，但前提是控制像素不能被
#      误判为独立小组件。生成器因此保证控制像素与框同色且直接相连。
# d) Placement legality
#    - 框 patch 统一为 12x11：其中 11x11 是空心框，额外 1 行用于上/下控制像素。
#      相邻框之间保留 1 行背景，故不会粘连。
#    - 右侧精灵仍放在独立右区域，使用 random_free_location 保证不重叠、不越界。
#    - 控制像素只占框区域内的预留行，不会与右侧精灵或其他框冲突。
# e) Visual observability
#    - 人类可直接看到: 同色框上方或下方有一个“角标式”单像素；输出中的点列也确实
#      从该侧开始堆叠。只看 2-3 组样例就能归纳“上标记=>从上往下，下标记=>从下往上”。
# f) Visual cleanliness
#    - 控制像素与框同色且位于中央，视觉上像框的附加标记，不会和右侧杂散精灵混淆。
#    - 点全部落在中心列，规则统一，不会与边框重叠。
# g) Color inferability
#    - 颜色关系仍与原题一致: 仅按“框颜色 == 精灵颜色”匹配；控制像素不引入新颜色。
# h) Variant-specific generator guard
#    - 每个样例都至少包含一个 top 框和一个 bottom 框，并强制这两类方向至少各有一个
#      正计数框，避免退化成“控制像素存在但不起作用”的样本。
# i) New solver assumptions and generator guarantees
#    - solver 假设大组件的首行或末行恰有 1 个像素，且另一端是完整框边；生成器固定按此构造。
#    - solver 假设匹配精灵仍是小型单色连通块；生成器继续把精灵面积限制在 2-8。
#    - solver 假设单框最多 4 个点；生成器通过分配逻辑保证 count <= 4。

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


FRAME_RING_H = 11             # actual hollow frame height
FRAME_W = 11                  # frame width
PATCH_H = FRAME_RING_H + 1    # extra row for top/bottom control pixel
FRAME_R0 = 0                  # top row of first reserved patch block
FRAME_C0 = 1                  # left column of frames
FRAME_GAP = 1                 # bg rows between reserved patch blocks
GRID_W = 23                   # fixed total width
RIGHT_C0 = 13                 # first column of sprite region
RIGHT_W = 9                   # width of sprite region

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
        return [int(ring_r0 + 1 + 2 * i) for i in range(int(n_dots))]
    return [int(ring_r1 - 1 - 2 * i) for i in range(int(n_dots))]


def _frame_patch(color: int, bg: int, n_dots: int, direction: str) -> np.ndarray:
    """Build a 12x11 patch: 11x11 hollow frame plus one control pixel above or below."""
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
    Infer direction from the single-pixel protrusion and return the true ring bbox.

    The generator guarantees one of:
    - top control:    top row has 1 pixel, bottom row is full border
    - bottom control: bottom row has 1 pixel, top row is full border
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
    raise ValueError("frame component does not match the expected control-pixel shape")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame, then draw that tally vertically from the control side.

    Input structure:
    - bg = most common color.
    - Hollow rectangular frames in various colors, each with one same-color control pixel
      attached above or below its center.
    - Small monochrome sprites scattered outside the frames.

    Rule:
    - Remove all small sprites from the grid.
    - For each frame of color C, count sprites whose color is C.
    - If the frame's control pixel is above, draw C-colored dots on the center column
      starting from the top interior row and stepping down by 2.
    - If the control pixel is below, draw from the bottom interior row upward by 2.
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
        is_frame = (area >= MIN_FRAME_AREA) and (h >= FRAME_RING_H) and (w >= FRAME_W)
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

        direction, ring_bbox = _parse_frame(f)
        r0, c0, r1, c1 = map(int, ring_bbox)
        c_mid = (int(c0) + int(c1)) // 2
        for rr in _dot_rows_for_ring(direction, r0, r1, num_patterns):
            if int(r0) < int(rr) < int(r1):
                out[int(rr), int(c_mid)] = col

    return out


def _sample_directions(rng: np.random.Generator, n_frames: int) -> list[str]:
    while True:
        directions = [str(rng.choice(["top", "bottom"])) for _ in range(n_frames)]
        if "top" in directions and "bottom" in directions:
            return directions


def _sample_counts(
    rng: np.random.Generator, n_frames: int, total_match: int, directions: list[str]
) -> list[int]:
    """Distribute matches with visibility guarantees: both directions appear with positive count."""
    counts = [0] * n_frames
    top_choices = [i for i, d in enumerate(directions) if d == "top"]
    bottom_choices = [i for i, d in enumerate(directions) if d == "bottom"]

    top_idx = int(top_choices[int(rng.integers(0, len(top_choices)))])
    bottom_idx = int(bottom_choices[int(rng.integers(0, len(bottom_choices)))])
    counts[top_idx] += 1
    counts[bottom_idx] += 1
    remaining = int(total_match - 2)

    while remaining > 0:
        candidates = [i for i, v in enumerate(counts) if v < MAX_DOTS_PER_FRAME]
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
