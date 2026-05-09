from __future__ import annotations

# Variant of 8f215267
# Dimension: D4 — 变换规则：计数方式 (transform)
# Change: 直接计数与框架同色的精灵数量 N... → 计数后取反
#
# Design analysis for this variant
# a) 具体化与行为追踪
#    - 视觉对象保持原题不变：左侧是 5x11 的彩色空心矩形框，右侧是散布的小型单色精灵。
#    - 输入中框内部为空；输出中仍只在框的中心行打点，点位仍然从右往左每隔 2 列放置。
#    - 变化只发生在“点数”的定义：
#      原题: 若某框同色精灵数为 N，则画 N 个点。
#      变体: 若某框同色精灵数为 N，则画 (MAX-N) 个点，这里 MAX 是该框可容纳的最大点数。
#      在当前固定 5x11 框里，MAX=4，对应四个合法点位。
#    - 小例子：
#      同一张图里三个框的同色精灵数若分别为 [4, 1, 0]，
#      原题输出点数为 [4, 1, 0]，
#      本变体输出点数为 [0, 3, 4]。
#      差异集中体现在 solver 的“计数→绘点”阶段，输入构造也必须按这个反向映射反推。
# b) 复杂度校准
#    - 不改对象种类、颜色匹配方式、空间布局，避免引入和 D4 无关的新推理负担。
#    - 为了让“精灵越多，点越少”在 2-3 个样例中肉眼可见，生成器强制每个样例至少出现：
#      一个 0 精灵框、一个 4 精灵框、以及一个中间档位框。这样能同时看到 4 点、0 点和中间点数。
# c) 隐式假设审查（针对 main）
#    - 组件分割 `find_components(... monochromatic=True)` 仍成立：输入里框和精灵都还是单色连通块。
#    - `is_frame = (h>=5 and w>=5 and area < h*w*0.7)` 仍成立：输入中的框内部依旧为空，没有额外填充。
#    - “移除 patterns” 这一步仍安全：右侧精灵仍不与框接触，且输入中框内没有点，不会误擦框内容。
#    - 原代码把 `num_patterns` 直接当作要画的点数；这是旧属性的关键依赖，变体中必须改为 `max_slots - num_patterns`。
#    - 新代码额外假设“每个框的同色精灵数不超过可容纳点位数”，生成器通过将匹配数限制在 0..4 来保证。
# d) 放置合法性
#    - 框布局、右侧散点区、精灵尺寸与原题一致；仍用 `random_free_location` 保证不重叠、不越界。
#    - 输出先画反向点数，再清空框内点位得到输入，再按真实精灵数在右侧放置精灵，最后用 `main(inp)==out` 校验。
# e) 视觉可观察性
#    - 颜色并不参与排序或数值计算，只作为“哪类精灵属于哪个框”的可见匹配线索。
#    - 每个样例都包含高/低/中三档真实计数，观察者能稳定看到“更多精灵对应更少点”的反直觉映射。
# f) 可视化整洁度
#    - 框仍集中在左侧、精灵仍集中在右侧，两类元素不混叠；框内标记仍只占中心行四个离散点位，清晰可读。
# g) 颜色变化可推断性
#    - 颜色用途未变：仅用于匹配精灵类别与框类别。没有额外颜色映射，因此不会引入新的不可见规则。

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


def _dot_columns(c0: int, c1: int) -> list[int]:
    """Valid center-row dot columns inside a frame, ordered from right to left."""
    return [int(cc) for cc in range(int(c1) - 2, int(c0), -2)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Remove all small patterns from the grid.
    - For each frame of color C, count matching patterns N.
    - Let MAX be the number of legal dot slots in that frame's center row.
    - Draw (MAX - N) dots of color C, from the right, every 2 columns.

    New assumptions introduced by the inversion:
    - Matching pattern counts should stay within the frame's slot capacity.
      The generator guarantees 0 <= N <= MAX (MAX=4 under the fixed 5x11 frame layout).
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

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        slots = _dot_columns(c0, c1)
        dots_to_draw = max(0, len(slots) - num_patterns)
        for cc in slots[:dots_to_draw]:
            out[int(r_mid), int(cc)] = col

    return out


# --------------- generator ---------------

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300
COUNT_SAMPLE_ATTEMPTS = 200


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center row."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    local_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, local_cols] = color
    return patch


def _sample_match_counts(rng: np.random.Generator, n_frames: int, n_distractors: int) -> list[int] | None:
    """
    Sample true same-color sprite counts per frame.

    Observability constraint:
    - include one empty frame (0 sprites -> 4 dots),
    - include one saturated frame (4 sprites -> 0 dots),
    - include one intermediate count (1..3 sprites -> 3..1 dots).
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

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    display_counts = [MAX_DOTS_PER_FRAME - n for n in match_counts]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, n_dots) in enumerate(zip(frame_colors, display_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, n_dots), r0, FRAME_C0)

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
    Generator guarantees for the solver's new assumptions:
    - every frame's true same-color sprite count is sampled in 0..4,
    - each sample contains low/high/intermediate counts so inversion is visible,
    - all sprites are placed disjointly in the right region,
    - only validated input grids are returned.
    """
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
