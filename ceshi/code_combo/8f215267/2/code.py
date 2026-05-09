# Combo variant of 8f215267 (index 2)
# combo_spec: D1/4+D4/1
# Name: 行对齐匹配+计数取反
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 视觉骨架保持原题：左侧竖直堆叠 3-4 个 5x11 彩色空心框，右侧是若干个小单色精灵。
#    - 原题流程：
#      1. 按“精灵颜色 == 框颜色”把右侧精灵归类到某个框。
#      2. 对每个框统计匹配精灵数 N。
#      3. 在框中心行从右往左每隔 2 列画 N 个点。
#    - 融合后流程：
#      1. 精灵不再看颜色，而是看 bbox 中心行最接近哪个框的中心行。
#      2. 先得到该框的真实计数 N，再改成显示计数 (4-N)。
#      3. 仍在框中心行固定四个槽位里，从右往左画点。
#    - 例：三个框中心行分别为 3, 10, 17。若右侧某条上方横带有 4 个精灵、中央横带有 1 个精灵、下方横带有 0 个精灵，
#      原题若按颜色数可能得到别的分配；本变体必须先按横带就近归类得到 [4,1,0]，再显示成 [0,3,4] 个点。
#      因而差异同时发生在“谁属于哪个框”和“计数如何转成点数”两层。
# b) 复杂度校准
#    - 若仍大量使用框色作为精灵色，解题者可能误以为颜色匹配仍成立，所以生成器刻意让精灵颜色只来自非框色的小调色板。
#    - 若所有框都只有同一档位计数，计数取反不明显；因此每个样例强制同时出现 0、4、以及 1..3 这三类真实计数。
#    - 若精灵在右侧随机散布，最近行匹配会不易肉眼读出；因此每个框的精灵都限制在对应框中心行附近的一条窄横带中。
# c) 隐式假设审查
#    - `find_components(... monochromatic=True)` 仍成立：框和精灵都保持单色连通块。
#    - `is_frame = (h >= 5) and (w >= 5) and (area < h*w*0.7)` 仍成立：框仍是大空心矩形，精灵远小于框。
#    - 擦除精灵的循环仍安全：输入里框内部为空，所有精灵都在右侧独立区域。
#    - 原题核心假设“颜色就是匹配键”已失效，必须改成“按中心行最近框匹配”。
#    - 新增假设 1：每个精灵对最近框的归属必须唯一。生成器只使用奇数高度精灵，并把精灵中心行限制在目标框中心行 ±1，
#      而相邻框中心行间隔固定为 7，因此不会出现平局。
#    - 新增假设 2：每个框的真实计数 N 不能超过 4，否则 `4-N` 会出现负值；生成器将每框计数限制在 0..4。
# d) 放置合法性
#    - 右侧区域宽 9 列，使用 4 个固定列槽位；每个精灵 bbox 宽最多 2，因此同一横带最多安全放 4 个精灵。
#    - 相邻槽位的精灵颜色强制不同，避免同色组件横向相连而被 `find_components` 误合并。
#    - 相邻框横带之间至少留 2 行净空，且精灵高度只取 1 或 3，保证不同横带不会接触。
# e) 视觉可观察性
#    - 解题者能直接看到：右侧精灵按上下横带对应左侧不同框，但颜色与框脱钩。
#    - 同一个样例里同时出现“精灵多但点少”和“精灵少但点多”，因此反向映射可由少量例子归纳。
# f) 可视化整洁度
#    - 左边只放框，右边只放精灵；每个框对应一条窄横带，阅读路径简单。
#    - 框内点只落在四个离散槽位，右侧精灵也落在固定列槽位，画面不会因为随机拥挤而混乱。
# g) 变化的可推断性
#    - 本融合不引入新的颜色映射。颜色只用于区分对象外观，真正可推断的结构线索是“纵向行带”和“4-N”。
#    - 生成器还会拒绝那些在同一输入上被“原题颜色匹配 + 计数取反”也能解释的退化样本。

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
MAX_DOTS_PER_FRAME = 4

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)        # 3-4 frames
TOTAL_PATTERNS_RANGE = (5, 11) # 5-10 total sprites, all counted by row alignment
BOTTOM_PAD_RANGE = (0, 7)
COUNT_SAMPLE_ATTEMPTS = 200
MAX_ATTEMPTS = 300

# --- sprite placement ---
SPRITE_SLOT_COLS = (0, 2, 4, 6)
CENTER_OFFSETS = (-1, 0, 1)

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
    frames.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    return frames, patterns


def _center_row(comp: dict) -> float:
    r0, _, r1, _ = map(int, comp["bbox"])
    return 0.5 * (r0 + r1)


def _dot_columns(c0: int, c1: int) -> list[int]:
    return [int(cc) for cc in range(int(c1) - 2, int(c0), -2)]


def _erase_patterns(g: np.ndarray, bg: int, patterns: list[dict]) -> np.ndarray:
    out = np.array(g, dtype=int).copy()
    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = int(bg)
    return out


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


def _render_inverted_counts(base: np.ndarray, frames: list[dict], counts: list[int]) -> np.ndarray:
    out = np.array(base, dtype=int).copy()
    for f, true_count in zip(frames, counts):
        col = int(f["color"])
        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        slots = _dot_columns(c0, c1)
        dots_to_draw = max(0, len(slots) - int(true_count))
        for cc in slots[:dots_to_draw]:
            out[int(r_mid), int(cc)] = col
    return out


def _legacy_color_inversion_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_objects(g, bg)
    base = _erase_patterns(g, bg, patterns)
    return _render_inverted_counts(base, frames, _legacy_color_counts(frames, patterns))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassign the outside sprites by nearest frame row, then show the complementary tally inside each frame.

    Input structure:
    - One dominant background color fills the canvas.
    - Several large hollow frames with distinct colors are stacked vertically on the left.
    - Small monochrome sprites sit in the right region, roughly aligned with one frame row-band each, but their colors do not determine ownership.

    Rule:
    - Remove all right-side sprites from the grid.
    - For each sprite, find the frame whose center row is closest to the sprite's bbox center row; that frame claims the sprite.
    - If a frame claims N sprites, draw (4 - N) dots in that frame's center row, using the frame color and the usual right-to-left spacing of 2 columns.

    Key insight:
    - The important cue is vertical alignment, not color matching: sprites are absorbed by the nearest frame row-band.
    - The displayed dot count is the complement of the true count, so more nearby sprites produce fewer dots.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, patterns = _split_objects(g, bg)

    base = _erase_patterns(g, bg, patterns)
    counts = _nearest_row_counts(frames, patterns)
    return _render_inverted_counts(base, frames, counts)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    local_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, local_cols] = color
    return patch


def _sample_true_counts(rng: np.random.Generator, n_frames: int) -> list[int] | None:
    lo_total, hi_total = TOTAL_PATTERNS_RANGE
    for _ in range(COUNT_SAMPLE_ATTEMPTS):
        counts = [int(rng.integers(0, MAX_DOTS_PER_FRAME + 1)) for _ in range(n_frames)]
        if 0 not in counts:
            continue
        if MAX_DOTS_PER_FRAME not in counts:
            continue
        if not any(0 < c < MAX_DOTS_PER_FRAME for c in counts):
            continue
        total = sum(counts)
        if lo_total <= total < hi_total:
            return counts
    return None


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
    local_colors = list(rng.permutation(sprite_colors))
    while len(local_colors) < count:
        local_colors.extend(list(rng.permutation(sprite_colors)))
    # Sort by slot so adjacency checks are deterministic.
    placed = sorted(zip(slot_cols, local_colors[:count]), key=lambda item: item[0])
    for idx, (c0, color) in enumerate(placed):
        if idx > 0 and int(color) == int(placed[idx - 1][1]):
            color = next(c for c in sprite_colors if int(c) != int(placed[idx - 1][1]))
        mask = SPRITE_MASKS[int(rng.integers(0, len(SPRITE_MASKS)))]
        sprite = _make_sprite(mask, int(color), bg)
        target_center = int(frame_mid + int(rng.choice(CENTER_OFFSETS)))
        r0 = int(target_center - sprite.shape[0] // 2)
        paste(canvas, sprite, r0, int(c0), transparent=bg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    true_counts = _sample_true_counts(rng, n_frames)
    if true_counts is None:
        return None

    all_colors = list(range(10))
    rng.shuffle(all_colors)
    bg = int(all_colors[0])
    frame_colors = [int(x) for x in all_colors[1 : 1 + n_frames]]
    remaining_colors = [int(c) for c in all_colors if int(c) not in {bg, *frame_colors}]
    if len(remaining_colors) < 3:
        return None
    sprite_colors = [int(x) for x in remaining_colors[:3]]

    display_counts = [MAX_DOTS_PER_FRAME - n for n in true_counts]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    frame_mids: list[int] = []
    for i, (color, n_dots) in enumerate(zip(frame_colors, display_counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        frame_mids.append(r0 + FRAME_H // 2)
        paste(out, _frame_patch(color, bg, n_dots), r0, FRAME_C0)

    inp = out.copy()
    for r0, n_dots in zip(frame_rows, display_counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(n_dots)]
        inp[mid_r, dot_cols] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    for frame_mid, count in zip(frame_mids, true_counts):
        _place_frame_sprites(rng, right_canvas, int(frame_mid), int(count), sprite_colors, bg)

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_legacy_color_inversion_output(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
