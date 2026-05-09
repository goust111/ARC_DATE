# Variant of 8f215267
# Dimension: D7 — 精灵移除方式 (trajectory)
# Change: 所有精灵在输出中完全移除（设为背景色），无轨迹残留... → 保留精灵在原位

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location

# ---------------------------------------------------------------------------
# Design analysis for the D7 variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
#    原题的像素流程是：
#    - 输入左侧有若干空心框，右侧有散布的小单色精灵。
#    - 输出先把右侧所有精灵抹成背景色，再按“框颜色 = 精灵颜色”的匹配计数，
#      在对应框的中心行从右向左每隔 2 格补上若干同色点。
#    这个 D7 变体只改“精灵如何处理”：
#    - 输入仍然没有框内点，右侧仍然有小精灵。
#    - 输出不再删除精灵；右侧精灵完整保留在原位置，同时框内新增点。
#    一个具体追踪例子：
#      某个蓝框对应 3 个蓝色精灵，原题输出 = 右侧蓝精灵被抹掉，蓝框中心行出现 3 个蓝点；
#      本变体输出 = 右侧蓝精灵仍在，蓝框中心行同样出现 3 个蓝点。
#    因此解题者观察到的是“框里新增了 tally dots，而右侧源精灵还在”，
#    而不是“右侧源精灵消失后被框吸收”。
#
# b) Complexity calibration
#    不改对象种类、颜色匹配关系、计数上限、布局骨架，只替换 trajectory。
#    这样复杂度与原题同级，避免把任务降成“单纯复制点数”的过易版本。
#    为了让视觉差异稳定可见，生成器保留：
#    - 3-4 个框
#    - 5-10 个总精灵
#    - 1-2 个 distractor 颜色
#    所以每个样例都会同时出现“匹配精灵”“不匹配精灵”“框内新增点”三类证据。
#
# c) Implicit-assumption audit for main()
#    原 solver 与 D7 相关的关键前提是：
#    - out = g.copy()：输出先复制输入。这个前提仍成立。
#    - 组件分类依赖“框 bbox 至少 5x5 且面积明显小于实心矩形面积”，
#      这在变体中仍成立，因为框形状和精灵尺寸都不变。
#    - 原代码中的“Remove patterns from output”循环依赖旧任务前提：
#      所有小组件都应该在输出中消失。该前提在变体中失效，若保留这段代码会错误删除
#      本应保留的精灵，所以必须删去。
#    - 后续画点逻辑隐含假设：输入框内部没有预先存在与 tally 重叠的前景像素。
#      生成器继续保证点只画在框内部中心行，右侧精灵永不进入框区域，因此仍安全。
#
# d) Placement legality
#    生成器继续把框固定在左侧，把精灵全部放到右侧独立画布中，再整体 paste 到主网格。
#    由于框宽 11、右侧区域起始列 13，中间至少隔 1 列背景，不会接触或并成同一连通块。
#    精灵使用 random_free_location + padding=1，保证彼此不重叠、不过近、不过界。
#    精灵 bbox 最大 4x3，小于框判定阈值，因此不会被 solver 误识别为 frame。
#
# e) Visual observability
#    看到 2-3 组样例时，人类会发现：
#    - 右侧精灵在输入和输出中位置完全不变；
#    - 左侧框内部新增的点数，恰好等于同色精灵数量；
#    - 不属于任何框颜色的 distractor 精灵也被保留，但不会触发加点。
#    这比“消失型”原题更直接表达了“新增了什么”。
#
# f) Visual cleanliness
#    框与精灵分区明确：左侧只放框，右侧只放精灵，中间有背景缝隙。
#    输出叠加只有一层：在原输入基础上往框内部加点，不会覆盖右侧精灵。
#    因此可视上仍然整洁，不会出现元素堆叠难分辨的问题。
#
# g) Color inferability
#    颜色关系保持为一对一的“同色匹配”，没有引入颜色值排序或隐藏编码。
#    每个样例中框颜色各不相同，且有少量 distractor 颜色作为反例，
#    足以让解题者归纳“只有与框同色的精灵才会贡献框内点数”。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-color patterns per frame and represent the tally as dots inside.

    D7 variant rule:
    - Keep all scattered sprites in place.
    - For each hollow frame of color C, count how many small sprites share color C.
    - Draw that many dots along the frame's center row, spaced by 2, starting from the right.
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

    # D7 change: keep the source sprites instead of erasing them.
    for f in frames:
        col = int(f["color"])
        num_patterns = sum(1 for p in patterns if int(p["color"]) == col)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col

    return out


# --------------- generator (D7 keep_source variant) ---------------

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
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    """Distribute matching sprites across frames, each capped at MAX_DOTS_PER_FRAME."""
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

    # --- choose structure ---
    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors
    if total_match <= 0:
        return None

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1: 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames:]]

    counts = _sample_counts(rng, n_frames, total_match)
    if sum(counts) != total_match:
        return None

    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    # --- build the input skeleton: empty frames only ---
    inp = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, color in enumerate(frame_colors):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(inp, _frame_patch(color, bg, 0), r0, FRAME_C0)

    # --- scatter right-side sprites; these stay in both input and output ---
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

    # --- expected output: identical to input plus tally dots inside the frames ---
    out = inp.copy()
    for r0, color, count in zip(frame_rows, frame_colors, counts):
        mid_r = int(r0 + FRAME_H // 2)
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        for cc in dot_cols:
            out[int(mid_r), int(cc)] = int(color)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
