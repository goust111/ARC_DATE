# Variant of 8f215267
# Dimension: D1 — 匹配条件：颜色匹配 (match_cond)
# Change: 精灵颜色 == 框架颜色 (s.color == d.color)，同色精灵被归属于对应框架... → 行对齐匹配
#
# Design analysis
# ===============
# variant_spec:
#   unchanged_structure:
#     - 背景仍为单一底色。
#     - 左侧仍是 3-4 个 5x11 的空心矩形框，竖直堆叠，框内中心行用于显示圆点计数。
#     - 右侧仍是若干个小型单色精灵，输出仍要把它们擦掉，并在对应框中从右往左每隔 2 列画点。
#   core_change:
#     - 原题把精灵按“颜色等于框色”归类。
#     - 本变体把精灵按“精灵 bbox 中心行最接近哪个框的中心行”归类。
#     - 输出点的颜色仍使用框本身的颜色，因为框是 tally display；变的是“谁被算进哪个框”，不是框的外观。
#   concrete_trace:
#     - 例：三个框中心行分别为 3, 10, 17。
#     - 原逻辑：若一个红色精灵位于第 3 个框旁边，只要红框在第 1 行带，它仍会计入红框。
#     - 新逻辑：同一个精灵若 bbox 中心行为 16，则它离 17 最近，应计入第 3 个框；即使它颜色与第 1 个框相同，也不能计入第 1 个框。
#     - 差异点出现在“统计每个框的精灵数量”这一步；输入构造、擦除精灵、在框中画点的形式保持不变。
#   complexity_calibration:
#     - 若仍让精灵颜色经常等于框色，则很多样本会退化为原题，肉眼看不出新规则。
#     - 因此生成器会刻意让精灵颜色来自非框色的小调色板，使“颜色不再是可用线索”在样例中可观察。
#     - 但不引入额外形状语义；精灵仍只是小单色连通块，复杂度与原题同级。
#   implicit_assumption_audit:
#     - 原 main() 中 `num_patterns = sum(... p['color'] == col)` 直接依赖“颜色就是匹配键”，这一行必须替换。
#     - 原 main() 其他部分：
#       - `find_components(... monochromatic=True)` 仍成立，因为框和精灵仍是单色连通块。
#       - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 仍成立，因为框仍是大空心矩形，精灵仍远小于框。
#       - 擦除 patterns 的循环仍成立，因为精灵仍都位于框外。
#       - 画点时直接用框色写入仍安全，因为输出语义没有改。
#     - 新增假设：每个精灵都必须对某个框有唯一最近的中心行。生成器通过只生成奇数高度精灵、并把精灵中心行限制在框中心行 ±1 内来避免平局。
#   placement_legality:
#     - 右侧区域仍独立于框区域，不会与框重叠。
#     - 每个框对应 4 个固定列槽位；精灵最大宽度为 2，因此最多 4 个精灵可在同一行带中无重叠放下。
#     - 精灵高度只取 1 或 3，且以目标中心行为锚点放置，保证都落在网格内。
#   visual_observability:
#     - 人类能直接看到：右侧精灵按“高度/横带”分组，并在输出中被吸收到同高度的左侧框里。
#     - 由于精灵颜色与框色故意脱钩，解题者不会被错误地诱导到“按颜色匹配”。
#   visual_cleanliness:
#     - 左侧只放框，右侧只放精灵；同一框对应的精灵在右侧只占一条窄横带，阅读顺序清楚。
#     - 精灵最多 4 个/框，且列槽位固定，避免堆叠杂乱。
#   color_inference:
#     - 颜色不再参与分类，因此生成器只保留 2-3 种精灵色并复用，强调“颜色只是外观差异”。
#   generator_adaptation:
#     - 原题的 distractor 是“颜色不匹配任何框，因此不计数”。
#     - 新规则下颜色不影响计数，所以不再保留这种 distractor；所有精灵都必须被最近框吸收。
#     - 为防退化，生成器会显式拒绝那些在同一输入上让旧颜色规则与新行对齐规则得到相同输出的样本。

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, get_bg, paste


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
TOTAL_PATTERNS_RANGE = (5, 11) # 5-10 total sprites, all counted by nearest-row rule
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite appearance / placement ---
SPRITE_SLOT_COLS = (0, 2, 4, 6)
CENTER_OFFSETS = (-1, 0, 1)
MAX_ATTEMPTS = 300

SPRITE_MASKS = [
    np.array([[1, 1]], dtype=int),
    np.array([[1], [1], [1]], dtype=int),
    np.array([[1, 0], [1, 1], [0, 1]], dtype=int),
    np.array([[0, 1], [1, 1], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 0], [1, 1]], dtype=int),
    np.array([[0, 1], [0, 1], [1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 1]], dtype=int),
    np.array([[1, 1], [0, 1], [1, 1]], dtype=int),
]


def _split_objects(g: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
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
    return frames, patterns


def _center_row(comp: dict) -> float:
    r0, _, r1, _ = map(int, comp["bbox"])
    return 0.5 * (r0 + r1)


def _nearest_row_counts(frames: list[dict], patterns: list[dict]) -> list[int]:
    frame_centers = [_center_row(f) for f in frames]
    counts = [0] * len(frames)
    for p in patterns:
        p_center = _center_row(p)
        idx = min(
            range(len(frames)),
            key=lambda i: (abs(p_center - frame_centers[i]), frame_centers[i]),
        )
        counts[int(idx)] += 1
    return counts


def _legacy_color_counts(frames: list[dict], patterns: list[dict]) -> list[int]:
    return [sum(1 for p in patterns if int(p["color"]) == int(f["color"])) for f in frames]


def _render_counts(base: np.ndarray, bg: int, frames: list[dict], counts: list[int]) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for f, num_patterns in zip(frames, counts):
        if int(num_patterns) <= 0:
            continue
        col = int(f["color"])
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(int(num_patterns)):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = col
    return out


def _erase_patterns(g: np.ndarray, bg: int, patterns: list[dict]) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = int(bg)
    return out


def _legacy_color_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_objects(g, bg)
    base = _erase_patterns(g, bg, patterns)
    counts = _legacy_color_counts(frames, patterns)
    return _render_counts(base, bg, frames, counts)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count outside sprites by nearest frame center row, then display the tally inside each frame.

    The input still consists of large hollow frames on the left and small monochrome sprites on the right.
    The only changed semantic is the matching key:
    - old task: sprite color == frame color
    - variant: sprite bbox center row is closest to the frame's center row
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_objects(g, bg)

    base = _erase_patterns(g, bg, patterns)
    counts = _nearest_row_counts(frames, patterns)
    return _render_counts(base, bg, frames, counts)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_patterns: int) -> list[int]:
    counts = [1] * n_frames
    remaining = total_patterns - n_frames
    order = rng.permutation(n_frames)
    extra_cap = MAX_DOTS_PER_FRAME - 1
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - extra_cap * frames_left)
        hi = min(extra_cap, remaining)
        extra = int(rng.integers(lo, hi + 1))
        counts[int(idx)] += extra
        remaining -= extra
    return counts


def _make_sprite(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _place_frame_sprites(
    rng: np.random.Generator,
    canvas: np.ndarray,
    frame_mid: int,
    count: int,
    sprite_colors: list[int],
    bg: int,
) -> None:
    slot_cols = list(rng.permutation(SPRITE_SLOT_COLS)[:count])
    for c0 in slot_cols:
        mask = SPRITE_MASKS[int(rng.integers(0, len(SPRITE_MASKS)))]
        color = int(sprite_colors[int(rng.integers(0, len(sprite_colors)))])
        sprite = _make_sprite(mask, color, bg)
        target_center = int(frame_mid + int(rng.choice(CENTER_OFFSETS)))
        r0 = int(target_center - sprite.shape[0] // 2)
        paste(canvas, sprite, r0, int(c0), transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    total_patterns = int(rng.integers(max(TOTAL_PATTERNS_RANGE[0], n_frames), TOTAL_PATTERNS_RANGE[1]))

    all_colors = list(range(10))
    rng.shuffle(all_colors)
    bg = int(all_colors[0])
    frame_colors = [int(x) for x in all_colors[1 : 1 + n_frames]]

    remaining_colors = [int(c) for c in all_colors if int(c) not in {bg, *frame_colors}]
    if len(remaining_colors) < 2:
        return None
    n_sprite_colors = min(len(remaining_colors), int(rng.integers(2, 4)))
    sprite_colors = [int(x) for x in remaining_colors[:n_sprite_colors]]

    counts = _sample_counts(rng, n_frames, total_patterns)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    frame_mids: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        frame_mids.append(r0 + FRAME_H // 2)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    for frame_mid, count in zip(frame_mids, counts):
        _place_frame_sprites(rng, right_canvas, int(frame_mid), int(count), sprite_colors, bg)

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(_legacy_color_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
