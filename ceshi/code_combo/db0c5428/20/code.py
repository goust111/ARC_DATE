# Combo variant of db0c5428 (index 20)
# combo_spec: D10/2+D14/0
# Name: 颜色数编码规则+先重建再外扩

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


# 设计分析：db0c5428 的融合变体（D10/2 + D14/0）
# a) 具体化与行为追踪
#    - 原题：先把 8 个可见块复制到 5x5 块网格外圈的点对称槽位，再从 hole 周围 8 个朝向像素
#      重建中心块；中心像素固定为主色。即使先后顺序对调，最终外圈也不受中心影响。
#    - 本融合版：
#      1. 先数前景色个数，决定中心块重建模式。
#         * 2 色：中心 3x3 整块都填主色。
#         * 3 色：中心 8 个外圈像素继续读取 hole 周围 8 个朝向像素，中心像素填主色。
#         * 4 色：中心 8 个外圈像素仍按朝向像素拼接，但中心像素改成全图最稀有前景色。
#      2. 得到完整 3x3 块阵列后，再做外扩：
#         * 8 个原始非中心块仍复制到原题那 8 个“远端对称槽位”。
#         * 新重建出的中心块再复制到外圈剩余的 8 个空槽位。
#      一个 2 色小例子：原题中心会是“边界拼接 + 主色中心”，本融合版却先变成纯主色 3x3，
#      然后这个纯色块不仅出现在 motif 正中，还会重复出现在外圈原本为空的 8 个 block 槽位。
#      一个 4 色小例子：中心外框仍来自 hole 朝向像素，但中心点换成最稀有色，这个差异会被
#      外圈 8 次重复放大。
# b) 复杂度校准
#    - 颜色数只保留 2/3/4 三档，对应 3 个离散模式；规则数少，样例间可归纳。
#    - 2 色与 4 色模式都会显式偏离原题中心重建；3 色模式保留原题边界拼接，提供中间档。
#    - D14/0 不能只做“先填中心再按原题外扩”，那样输出与原题相同。这里让中心块占据外圈剩余
#      8 个槽位，顺序依赖才真正可见。
# c) 隐式假设审查
#    - `bbox(g != bg)` 依赖输入只有一个前景 motif。生成器仍只放一个 9x9 motif。
#    - `h == w == 9`、`block == 3` 依赖固定 3x3 block。生成器保持原题尺寸。
#    - 读取 hole 周围 8 个像素依赖中心 block 为空且周围 8 个采样位都在 bbox 内。生成器沿用
#      原题 9x9 结构并保留中心空洞。
#    - 4 色分支新增“最稀有前景色可唯一确定”的假设。生成器显式拒绝稀有色计数并列的样本。
#    - 外扩逻辑新增“15x15 输出区能容纳完整 5x5 block 网格，且输入 outer-gap 槽位初始为空”。
#      生成器沿用原题 `MOTIF_PAD = 3`，因此不会越界，也不会和输入前景重叠。
# d) 放置合法性
#    - motif 仍放在距边至少 3 格的位置，所以 15x15 最终输出区总能完整落在画布内。
#    - 外圈 16 个 block 槽位分成两组：8 个远端对称槽位由可见块填充，另外 8 个 gap 槽位由中心块
#      填充；两组互不重合，且都不与中间 3x3 motif 区重叠。
# e) 视觉可观察性
#    - 解题者肉眼能先看出“仍是原题的 9x9 缺中心 motif”，再看出输出不再是稀疏 5x5，而是被
#      中心块补满外圈空位的满 5x5 块网格。
#    - 同时，2/3/4 色对中心块长相的控制完全来自可见的“颜色种数”，不依赖颜色编号大小。
# f) 可视化整洁度
#    - 所有新增内容都按 block 槽位对齐；没有局部覆盖、旋转混排或像素级叠加。
#    - 中心块重复到一整圈 gap 槽位，模式清晰，比只改一个中心像素更容易肉眼捕捉。
# g) 变化的可推断性
#    - 颜色种类仍限制在 2-4 个前景色。
#    - 4 色模式虽然需要识别“最稀有色”，但该色会被生成器做成唯一最少出现次数，且最终在 9 个
#      中心块副本的中心像素上重复出现，具备足够结构线索。
# 非退化度量
#    - `gap_copy_cells`：输出中 8 个 outer-gap 槽位里非背景像素数。原题恒为 0，本变体恒为 72，
#      因而直接量化 D14/0 是否生效。
#    - `center_rule_delta`：按本变体中心规则与原题中心规则得到的中心块差异像素数。2 色与 4 色
#      样本应大于 0，3 色样本可为 0；这能量化 D10/2 是否真的控制了中心重建。
#    - 生成器额外要求 `main(input)` 与原题输出不同，防止退化回原任务。


BLOCK = 3
MOTIF_BLOCKS = 3
OUTPUT_BLOCKS = 5
MOTIF_SIZE = BLOCK * MOTIF_BLOCKS
EXPANDED_SIZE = BLOCK * OUTPUT_BLOCKS
MOTIF_PAD = 3
_CENTER_OFFSETS = np.array([-1, 1, 3])
OUTER_GAP_SLOTS = (
    (0, 1), (0, 3),
    (1, 0), (1, 4),
    (3, 0), (3, 4),
    (4, 1), (4, 3),
)


def _foreground_colors(g: np.ndarray, bg: int) -> list[int]:
    return sorted(int(v) for v in np.unique(g[g != int(bg)]))


def _foreground_count_map(g: np.ndarray, bg: int) -> dict[int, int]:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    return {int(v): int(c) for v, c in zip(vals, counts)}


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    counts = _foreground_count_map(g, bg)
    assert counts, "input must contain foreground colors"
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


def _rarest_foreground_color(g: np.ndarray, bg: int) -> int:
    counts = _foreground_count_map(g, bg)
    assert counts, "input must contain foreground colors"
    return min(counts.items(), key=lambda item: (item[1], item[0]))[0]


def _stitched_center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    """Original center rule: read the 8 hole-facing pixels and use majority at the center."""
    majority = _major_foreground_color(motif, bg)
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
    return patch


def _center_patch(motif: np.ndarray, bg: int) -> np.ndarray:
    """D10/2: choose the center reconstruction mode from the foreground color count."""
    fg_colors = _foreground_colors(motif, bg)
    count = len(fg_colors)
    assert 2 <= count <= 4, "this combo uses 2-4 foreground colors"

    majority = _major_foreground_color(motif, bg)
    if count == 2:
        return np.full((BLOCK, BLOCK), int(majority), dtype=int)

    patch = _stitched_center_patch(motif, bg)
    if count == 4:
        patch[1, 1] = int(_rarest_foreground_color(motif, bg))
    return patch


def _completed_motif(motif: np.ndarray, center: np.ndarray) -> np.ndarray:
    full = motif.copy()
    paste(full, center, BLOCK, BLOCK)
    return full


def _expanded_output(motif: np.ndarray, center: np.ndarray, bg: int) -> np.ndarray:
    """
    D14/0: first rebuild the center, then expand the completed 3x3 block motif.
    The 8 original visible blocks go to the far symmetric slots; the rebuilt center
    fills the 8 remaining outer gaps, producing a full 5x5 block grid.
    """
    completed = _completed_motif(motif, center)
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), bg, dtype=int)
    paste(expanded, completed, BLOCK, BLOCK)

    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            src = completed[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            dst_br = 4 - 2 * br
            dst_bc = 4 - 2 * bc
            paste(expanded, src, dst_br * BLOCK, dst_bc * BLOCK)

    for dst_br, dst_bc in OUTER_GAP_SLOTS:
        paste(expanded, center, dst_br * BLOCK, dst_bc * BLOCK)
    return expanded


def _original_like_output(motif: np.ndarray, bg: int) -> np.ndarray:
    """Baseline original-task output used only for generator anti-degeneration checks."""
    center = _stitched_center_patch(motif, bg)
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), bg, dtype=int)
    paste(expanded, _completed_motif(motif, center), BLOCK, BLOCK)
    for br in range(MOTIF_BLOCKS):
        for bc in range(MOTIF_BLOCKS):
            if (br, bc) == (1, 1):
                continue
            src = motif[br * BLOCK : (br + 1) * BLOCK, bc * BLOCK : (bc + 1) * BLOCK]
            paste(expanded, src, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the missing center from a color-count-coded rule, then expand the completed motif.

    Input structure:
    - A single 9x9 foreground motif sits on a uniform background.
    - The motif is a 3x3 array of 3x3 blocks whose middle block is empty.
    - The visible foreground uses exactly 2, 3, or 4 non-background colors, and
      the 8 pixels facing the hole still provide the directional boundary cue.

    Rule:
    - First count the foreground colors. With 2 colors, fill the whole center
      block with the motif's majority color. With 3 colors, rebuild the center's
      outer ring from the 8 hole-facing pixels and use the majority color at the
      center. With 4 colors, use the same stitched outer ring but place the
      rarest foreground color at the center pixel.
    - Insert that rebuilt center to obtain a complete 3x3 block motif.
    - Expand only after the center exists: copy each of the 8 visible source
      blocks to its original far point-symmetric outer slot, and copy the rebuilt
      center block to the other 8 outer-ring slots that were empty in the source.

    Key insight:
    - The color count is not decorative metadata; it selects how the hole is
      reconstructed. Because the second step expands the completed motif instead of
      the hollow one, the rebuilt center now propagates outward and must be solved
      before the 5x5 block grid can be completed.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w == MOTIF_SIZE, "motif must be a 9x9 square"

    motif = crop_bbox(g, motif_bbox)
    center = _center_patch(motif, bg)
    expanded = _expanded_output(motif, center, bg)

    out = g.copy()
    paste(out, expanded, int(r0) - BLOCK, int(c0) - BLOCK)
    return out


# --------------- generator (combo D10/2 + D14/0) ---------------

# -- CFG: canvas & structure --
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19

# -- CFG: colors --
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_CHOICES = (2, 3, 4)

# -- CFG: majority pixel counts (corner block + edge block combined) --
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8

# -- CFG: generation --
MAX_RETRIES = 400

# Diagonal-symmetric orbits (b00 == b00.T)
DIAG_ORBITS = (
    ((0, 0),),
    ((1, 1),),
    ((2, 2),),
    ((0, 1), (1, 0)),
    ((0, 2), (2, 0)),
    ((1, 2), (2, 1)),
)

# LR-symmetric orbits (b01 == fliplr(b01))
LR_ORBITS = (
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
)


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
    free_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for r, c in orbit:
                block[r, c] = int(majority)
        else:
            free_orbits.append((block, orbit))
    return block, free_orbits


def _assign_minor_colors(free_orbits, minor_colors):
    """Color free orbits while biasing 4-color samples toward a unique rarest color."""
    assert free_orbits
    random.shuffle(free_orbits)
    minors = list(minor_colors)

    if len(minors) == 1:
        for block, orbit in free_orbits:
            for r, c in orbit:
                block[r, c] = int(minors[0])
        return

    if len(minors) == 2:
        assignments = [int(minors[0]), int(minors[1])]
        assignments += [int(random.choice(minors)) for _ in range(len(free_orbits) - 2)]
        random.shuffle(assignments)
        for (block, orbit), color in zip(free_orbits, assignments):
            for r, c in orbit:
                block[r, c] = int(color)
        return

    assert len(minors) == 3
    rare = int(minors[2])
    singleton_idxs = [i for i, (_block, orbit) in enumerate(free_orbits) if len(orbit) == 1]
    if not singleton_idxs or len(free_orbits) < 4:
        raise AssertionError("need enough free orbits to realize a unique rarest color")

    rare_idx = random.choice(singleton_idxs)
    ordered = [free_orbits[rare_idx]] + [
        free_orbits[i] for i in range(len(free_orbits)) if i != rare_idx
    ]

    tail = [int(minors[0]), int(minors[1]), int(minors[0])]
    tail += [int(random.choice(minors[:2])) for _ in range(len(ordered) - 1 - len(tail))]
    random.shuffle(tail)
    assignments = [rare] + tail
    for (block, orbit), color in zip(ordered, assignments):
        for r, c in orbit:
            block[r, c] = int(color)


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_free = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_free = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    free_orbits = corner_free + edge_free
    assert len(free_orbits) >= len(minor_colors)
    if len(minor_colors) == 3:
        assert len(free_orbits) >= 4
    _assign_minor_colors(free_orbits, minor_colors)
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


def _valid_color_count_profile(motif: np.ndarray) -> bool:
    counts = _foreground_count_map(motif, BG)
    if not (2 <= len(counts) <= 4):
        return False
    if len(counts) == 4:
        vals = sorted(counts.values())
        if vals[0] == vals[1]:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    color_count = random.choice(COLOR_COUNT_CHOICES)
    colors = random.sample(FOREGROUND_COLORS, color_count)
    majority = int(colors[0])
    minor_colors = [int(c) for c in colors[1:]]

    possible_totals = [
        total for total in range(MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX + 1)
        if 18 - total >= len(minor_colors)
    ]
    if color_count == 4:
        possible_totals = [total for total in possible_totals if 18 - total >= 4]
    majority_total = random.choice(possible_totals)
    pairs = [
        (corner, majority_total - corner)
        for corner in range(CORNER_MAJOR_MIN, CORNER_MAJOR_MAX + 1)
        if EDGE_MAJOR_MIN <= majority_total - corner <= EDGE_MAJOR_MAX
    ]
    if color_count == 4:
        pairs = [
            (corner, edge)
            for corner, edge in pairs
            if (9 - corner) + (9 - edge) >= 4
        ]
    if not pairs:
        return None
    corner_major, edge_major = random.choice(pairs)

    try:
        b00, b01 = _make_blocks(majority, minor_colors, corner_major, edge_major)
    except AssertionError:
        return None
    motif = _assemble_motif(b00, b01)
    if not _valid_color_count_profile(motif):
        return None

    center = _center_patch(motif, BG)
    stitched = _stitched_center_patch(motif, BG)
    if color_count in (2, 4) and np.array_equal(center, stitched):
        return None

    expanded = _expanded_output(motif, center, BG)
    original = _original_like_output(motif, BG)
    if np.array_equal(expanded, original):
        return None

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)

    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 combo input")
