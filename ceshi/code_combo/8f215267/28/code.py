# Combo variant of 8f215267 (index 28)
# combo_spec: D5/4+D7/0+D11/1
# Name: 连续线段+互补色+保留精灵

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


COMPLEMENT = {
    1: 2,
    2: 1,
    3: 4,
    4: 3,
    6: 7,
    7: 6,
    8: 9,
    9: 8,
}
COMPLEMENT_PAIRS = (
    (1, 2),
    (3, 4),
    (6, 7),
    (8, 9),
)
BG_CHOICES = (0, 5)


# ---------------------------------------------------------------------------
# Design analysis for combo D5/4 + D7/0 + D11/1
# ---------------------------------------------------------------------------
#
# a) Concretization and behavior trace
#    原题流程：
#    - 输入左侧有若干不同颜色的 5x11 空心框，右侧有若干同色小精灵。
#    - 输出会先删除所有右侧精灵。
#    - 每个框统计“与自己同色”的精灵个数 N，再在框中心行从右向左每隔 2 列画 N 个同色点。
#    融合后流程：
#    - 右侧精灵保留在原位，不再删除。
#    - 计数关系不变：框颜色 C 仍然只统计颜色也是 C 的精灵。
#    - 框内显示方式改为连续线段：仍从右锚点 c1-2 出发，但连续向左填 N 格。
#    - 线段颜色不再等于框色，而是使用框色的互补色 `COMPLEMENT[C]`。
#    具体追踪例子：若蓝框(1)右侧有 3 个蓝色精灵，则原题输出会删除那 3 个蓝精灵，并在蓝框中心行
#    的 c1-2 / c1-4 / c1-6 放 3 个蓝点；本融合版会保留那 3 个蓝精灵，并在该框中心行的
#    c1-2 / c1-3 / c1-4 放一段长度 3 的红线(2)，因为 1<->2 是互补对。
#
# b) Complexity calibration
#    仅靠“换颜色”会让颜色规则在少量样例中不易归纳；仅靠“保留精灵”又容易弱化 D5 的可见差异。
#    因此生成器做了三项约束来避免过简化：
#    - 固定 4 个框，并且总是选 2 组完整互补对，所以每个样例内都能同时看到 A->B 与 B->A。
#    - 每个框至少有 1 个匹配精灵，因此每个框都会出现一段互补色标记，不会出现“有些框没证据”。
#    - 至少一个框的计数 >= 3，使连续线段在视觉上明显区别于原题的离散点列。
#    这样样例既保留原题“按颜色计数”的核心，又让三项改动在单个样例里都可见。
#
# c) Implicit-assumption audit for main()
#    - `find_components(..., monochromatic=True)` 依赖框和精灵都是单色连通块；生成器继续满足。
#    - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 依赖精灵不会长成大空心结构；
#      生成器继续将精灵限制在 4x3 以内，因此不会被误判为框。
#    - 原题“删除精灵”的循环依赖“所有小组件输出都应消失”；D7/0 破坏了这一前提，所以该循环必须移除。
#    - 原题按 `frame_color` 直接着色；D11/1 破坏了“显示色=统计色”的前提，因此需要显式调用互补映射。
#      这又引入了新前提：每个框色都必须在 `COMPLEMENT` 中，生成器通过只从 8 个可配对颜色里选框色来保证。
#    - D5/4 把离散点改成连续线段后，`main()` 依赖框内部至少有足够的水平空间。固定 11 列宽的框内部
#      可容纳 8 格连续线段，生成器也把单框计数上限控制在 8 以内。
#
# d) Placement legality
#    左侧框区域与右侧精灵区域仍然硬分区：框固定在列 1..11，精灵只放在右侧子画布列 13..21。
#    `random_free_location(..., padding=1)` 保证精灵彼此不重叠、不贴边合并，也不会触碰框区域。
#    输出使用 output-first 构造：先画带互补线段的目标框，再拷贝出无线段输入骨架，最后把同一批精灵粘到
#    输入和输出上，因此天然满足“精灵保留、框内新增线段”的组合规则。
#
# e) Visual observability
#    解题者从 2-3 组样例中应能直接看到：
#    - 右侧精灵输入输出完全不动，说明它们不是被搬运或擦除，而是作为计数来源被保留。
#    - 框内标记是连续横条而不是隔列点列，且长度与同色精灵数量一致。
#    - 横条颜色总是框色的互补色，而且因为每个样例内都有完整互补对，映射关系可从同一张图中交叉验证。
#    不依赖颜色编号大小、排序或其它人类不可见因素。
#
# f) Visual cleanliness
#    线段只位于框的中心行内侧；精灵只位于右侧专用区域；二者没有重叠层。互补色线段即使恰好等于另一个框
#    的边框色，也只出现在当前框内部，不会造成组件粘连或角色混淆。
#
# g) Inferability of the color change
#    颜色变化不是任意置换，而是固定的双向互补对：1<->2, 3<->4, 6<->7, 8<->9。
#    生成器每次只使用其中两对，并且四个框把这两对完整展示出来，所以“哪种框色会变成哪种线段色”
#    在单个样例里就有重复支撑，不需要跨很多样例猜测。


def _complement_color(color: int) -> int:
    return int(COMPLEMENT[int(color)])


def _line_cols(n_fill: int) -> list[int]:
    """Interior-relative columns for a right-anchored continuous line segment."""
    if n_fill <= 0:
        return []
    start = FRAME_W - 3
    return list(range(start, start - int(n_fill), -1))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color sprites per frame and display the tally as a complementary line segment.

    Input structure:
    - bg = most common color.
    - Several hollow rectangular frames in distinct colors on the left.
    - Small monochrome sprites scattered in a separate right-side region.

    Rule:
    - Keep every scattered sprite exactly where it is.
    - For each frame of color C, count how many small sprites share color C.
    - On the frame's center row, draw a continuous horizontal segment of length N from the
      right anchor c1-2 leftward, using the complementary color paired with C.

    Key insight:
    - The sprites are not consumed; they remain visible as evidence. Each frame acts as a
      tally display whose length comes from same-color sprite count, while the display color
      is transferred to the fixed complementary partner of the frame color.
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
        frame_color = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == frame_color)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        max_fill = max(0, c1 - c0 - 2)
        fill_len = min(int(num_patterns), max_fill)
        line_color = _complement_color(frame_color)
        for cc in range(int(c1 - 2), int(c1 - 2 - fill_len), -1):
            if cc > int(c0):
                out[int(r_mid), int(cc)] = line_color

    return out


# --------------- generator (combo variant) ---------------

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
N_FRAMES = 4
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_MATCH_RANGE = (6, 11)
MAX_LINE_LEN = FRAME_W - 3

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(border_color: int, bg: int, fill_color: int, n_fill: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with a center-row line segment."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = border_color
    patch[-1, :] = border_color
    patch[:, 0] = border_color
    patch[:, -1] = border_color
    if n_fill > 0:
        patch[FRAME_H // 2, _line_cols(n_fill)] = fill_color
    return patch


def _sample_counts(rng: np.random.Generator, total_match: int) -> list[int]:
    """
    Distribute matches across the 4 frames.

    Every frame gets at least one sprite, and one frame gets at least three total
    so the continuous-line behavior is visually obvious.
    """
    counts = [1] * N_FRAMES
    remaining = total_match - N_FRAMES
    if remaining < 2:
        raise ValueError("total_match too small for the combo visibility constraints")

    prominent_idx = int(rng.integers(0, N_FRAMES))
    prominent_extra = int(rng.integers(2, min(MAX_LINE_LEN - 1, remaining) + 1))
    counts[prominent_idx] += prominent_extra
    remaining -= prominent_extra

    order = [int(x) for x in rng.permutation(N_FRAMES)]
    for pos, idx in enumerate(order):
        capacity = MAX_LINE_LEN - counts[idx]
        future_capacity = sum(MAX_LINE_LEN - counts[j] for j in order[pos + 1 :])
        lo = max(0, remaining - future_capacity)
        hi = min(capacity, remaining)
        add = int(rng.integers(lo, hi + 1))
        counts[idx] += add
        remaining -= add

    if remaining != 0 or min(counts) < 1 or max(counts) < 3:
        raise ValueError("failed to sample visible non-degenerate counts")
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


def _sample_frame_colors(rng: np.random.Generator) -> list[int]:
    """Pick two full complement pairs and shuffle the four resulting frame colors."""
    pair_ids = rng.choice(np.arange(len(COMPLEMENT_PAIRS)), size=2, replace=False)
    frame_colors = [color for idx in pair_ids for color in COMPLEMENT_PAIRS[int(idx)]]
    rng.shuffle(frame_colors)
    return [int(x) for x in frame_colors]


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    bg = int(BG_CHOICES[int(rng.integers(0, len(BG_CHOICES)))])
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_match = int(rng.integers(*TOTAL_MATCH_RANGE))

    frame_colors = _sample_frame_colors(rng)
    available_distractors = [c for pair in COMPLEMENT_PAIRS for c in pair if c not in frame_colors]
    rng.shuffle(available_distractors)
    distractor_colors = [int(c) for c in available_distractors[:n_distractors]]
    if len(distractor_colors) != n_distractors:
        return None

    try:
        counts = _sample_counts(rng, total_match)
    except ValueError:
        return None

    height = FRAME_R0 + N_FRAMES * FRAME_H + (N_FRAMES - 1) * FRAME_GAP

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (frame_color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        line_color = _complement_color(frame_color)
        paste(out, _frame_patch(frame_color, bg, line_color, count), r0, FRAME_C0)

    inp = np.full((height, GRID_W), bg, dtype=int)
    for r0, frame_color in zip(frame_rows, frame_colors):
        paste(inp, _frame_patch(frame_color, bg, bg, 0), r0, FRAME_C0)

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [c for c, n in zip(frame_colors, counts) for _ in range(n)]
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
    paste(out, right_canvas, 0, RIGHT_C0, transparent=bg)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
