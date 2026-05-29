# Variant of db0c5428
# Dimension: D7 — 输出网格的整体形状 (grid_shape)
# Change: 输出与输入同尺寸（same），外扩块在网格范围内原地放置，超出部分被截断... → 分帧输出：原始图案和外扩图案并排展示
#
# Design analysis
# ===============
# variant:
#   core_change:
#     - 输入仍然只有一个 9x9 motif；分帧只出现在输出。
#     - 输出固定为两帧横向拼接：左帧展示“补全中心后的原始 9x9 motif”，
#       右帧展示完整的 5x5 block 外扩图案，中间用一列纯色分隔线隔开。
#   concrete_layout:
#     - 输入: N x N 方阵，bg=8；其中仅有一个 9x9 前景 bbox。
#     - 左帧: 15 x 9，bg 画布上垂直居中放置补全后的 9x9 motif。
#     - 分隔线: 1 列、贯穿全高、颜色固定为 0；该颜色不出现在生成的 motif 中。
#     - 右帧: 15 x 15，完整 5x5 block 图案，不再依赖输入画布位置，也不做裁切。
#     - 最终输出: 15 x 25。
#   behavior_trace:
#     - 原任务在同尺寸输出上做 `paste(out, expanded, r0 - 3, c0 - 3)`，
#       结果绑定到输入中的绝对位置，且越界部分会被截断或根本无法容纳。
#     - 本变体改为先在独立 15 x 15 画布中构造完整外扩图案，再与补全后的
#       9x9 motif 做横向分帧拼接。输入中的 `(r0, c0)` 只用于定位 motif，
#       不再决定输出对象的最终坐标。
#   complexity_calibration:
#     - 保留原题的 D4 对称 block 生成和“内边界 8 像素重建中心块”机制，
#       所以真正的推理难点没有被删掉。
#     - 新增的 D7 变化只有一个明确档位：单帧 same-size → 双帧 side-by-side。
#       视觉上稳定，不引入额外分支爆炸。
#   hidden_assumptions_review:
#     - `bbox(g != bg)` 仍然成立，因为输入里仍然只有一个 motif。
#     - `_center_patch(...)` 依赖 block=3；生成器继续只产出 3x3 blocks。
#     - 原版 `out = g.copy()` 在这里失效，因为输出尺寸已不再等于输入尺寸。
#     - 原版 `expanded_r0/expanded_c0` 也失效，因为输出不再锚定输入坐标。
#     - 外扩复制必须写入一个专用 15 x 15 画布，不能再直接写回原输入副本。
#     - 分隔线颜色必须与 bg 不同且不出现在 motif 内；生成器用固定 palette
#       避免颜色冲突，从而保证“整列纯色”能被人眼直接识别为 frame divider。
#   placement_legality:
#     - 输入放置仍保证 9x9 motif 完全位于 N x N 画布内，且周围留白，bbox 清晰。
#     - 输出完全由专用帧画布组成，因此不存在越界、重叠或裁切歧义。
#   observability:
#     - 同一输入经过变体后一定得到 15 x 25 输出，并出现明确分隔列。
#     - 左帧和右帧内容不同但相关：左边只补中心，右边则补中心并外扩复制，
#       人类从少量样例即可看出“同一对象的两个结果被并排展示”。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


DIVIDER_COLOR = 0


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert block == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    offsets = [-1, 1, 3]
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + offsets[r], hole_c + offsets[c]])
    return patch


def _filled_motif(motif: np.ndarray, center: np.ndarray, block: int) -> np.ndarray:
    out = motif.copy()
    paste(out, center, block, block)
    return out


def _expanded_output(motif: np.ndarray, center: np.ndarray, bg: int, block: int) -> np.ndarray:
    expanded = np.full((block * 5, block * 5), bg, dtype=int)
    output_motif = _filled_motif(motif, center, block)
    paste(expanded, output_motif, block, block)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * block:(br + 1) * block, bc * block:(bc + 1) * block]
            paste(expanded, src, (4 - 2 * br) * block, (4 - 2 * bc) * block)
    return expanded


def _framed_output(motif_with_center: np.ndarray, expanded: np.ndarray, bg: int) -> np.ndarray:
    frame_h = int(expanded.shape[0])
    left_h, left_w = map(int, motif_with_center.shape)
    right_w = int(expanded.shape[1])
    out = np.full((frame_h, left_w + 1 + right_w), bg, dtype=int)
    out[:, left_w] = DIVIDER_COLOR
    paste(out, motif_with_center, (frame_h - left_h) // 2, 0)
    paste(out, expanded, 0, left_w + 1)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center, then present the local and expanded motifs as two frames.

    Input structure:
    - A square teal canvas contains exactly one 9x9 motif made from a 3x3 array of 3x3 blocks.
    - The middle 3x3 block is blank teal, while the eight surrounding blocks remain visible.
    - The pixels facing the hole encode the eight non-central cells of the missing center block.

    Rule:
    - Recover the center block from those inner-boundary pixels, with its middle cell set to the motif's majority foreground color.
    - Build two outputs from the same reconstruction: a left frame containing only the repaired 9x9 motif, and a right frame containing the full 5x5 block expansion obtained by copying each visible block to its point-symmetric outer slot.
    - Place the two frames side by side on teal background, separated by one solid divider column.

    Key insight: The input still describes one hidden center block and one larger block pattern, but the variant asks for both answers at once: the repaired local motif and the complete outer expansion.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    assert DIVIDER_COLOR != bg, "divider color must differ from background"

    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3

    motif = crop_bbox(g, motif_bbox)
    center = _center_patch(g, int(r0), int(c0), block, bg)
    motif_with_center = _filled_motif(motif, center, block)
    expanded = _expanded_output(motif, center, bg, block)
    return _framed_output(motif_with_center, expanded, bg)


# --------------- generator (D7 framed variant) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
MOTIF_PAD = 3

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 300

# -- Symmetry orbits for 3x3 blocks --
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)

_CENTER_OFFSETS = np.array([-1, 1, 3])


def _major_orbits(orbits, forced_minor, target_count):
    forced_minor = set(forced_minor)
    usable = [
        idx for idx, orbit in enumerate(orbits)
        if all(coord not in forced_minor for coord in orbit)
    ]
    choices = []
    for mask in range(1 << len(usable)):
        selected = set()
        total = 0
        for bit, idx in enumerate(usable):
            if mask & (1 << bit):
                selected.add(idx)
                total += len(orbits[idx])
        if total == target_count:
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, target_count)
    minor_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                block[r, c] = majority
        else:
            minor_orbits.append((block, orbit))
    return block, minor_orbits


def _assign_minor_colors(minor_orbits, colors):
    random.shuffle(minor_orbits)
    guaranteed = list(colors)
    random.shuffle(guaranteed)
    assignments = guaranteed + [
        random.choice(colors) for _ in range(len(minor_orbits) - len(guaranteed))
    ]
    for (block, orbit), color in zip(minor_orbits, assignments):
        for r, c in orbit:
            block[r, c] = color


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00, b01):
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): b00,
        (0, 1): b01,
        (0, 2): np.fliplr(b00),
        (1, 0): b01.T,
        (1, 2): np.rot90(b01, -1),
        (2, 0): np.flipud(b00),
        (2, 1): np.flipud(b01),
        (2, 2): np.rot90(b00, 2),
    }
    for (br, bc), blk in blocks.items():
        paste(motif, blk, br * BLOCK, bc * BLOCK)
    return motif


def _gen_center_patch(motif, majority):
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = majority
    return patch


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.randint(COLOR_COUNT_MIN, COLOR_COUNT_MAX)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = colors[0]
    minor_colors = colors[1:]

    possible_totals = [
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= len(minor_colors)
    ]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    corner_major, edge_major = random.choice(pairs)

    b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    motif = _assemble_motif(b00, b01)
    center = _gen_center_patch(motif, majority)
    motif_with_center = _filled_motif(motif, center, BLOCK)
    expanded = _expanded_output(motif, center, BG, BLOCK)
    expected = _framed_output(motif_with_center, expanded, BG)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D7 input")
