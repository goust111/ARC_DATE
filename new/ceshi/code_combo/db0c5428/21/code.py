# Combo variant of db0c5428 (index 21)
# combo_spec: D3/0+D13/1
# Name: 插值填充空白+完全填充5x5网格

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, get_bg, paste

# Design analysis for combo D3/0 + D13/1
# ----------------------------------------
# a) Concrete behavior and trace
#    - 原题输入仍然只有一个 9x9 motif: 3x3 个 3x3 block，中心 block 缺失，中心内容由 hole 周围
#      8 个朝内像素编码。
#    - 原题输出先补中心，再把 8 个非中心 block 复制到 5x5 block 网格的对称外槽，得到 17 个有内容
#      的 block；其余 gap 槽位为空。这里要把这些 gap 也全部补满，因此输出是完整 15x15 密集图案。
#    - 具体追踪一个顶部 gap 槽位：原题在 block 坐标 (0,1) 留背景；融合后它位于两个已知块
#      (0,0) 与 (0,2) 之间，因此读取左块最右列和右块最左列，生成一个 3x3 插值块：
#      左列=左 donor 边，右列=右 donor 边，中列按行奇偶在两条 donor 边之间交替。
#    - 纵向 gap 同理：上行=上 donor 底边，下行=下 donor 顶边，中行按列奇偶交替。
#    - 说明里 D13/1 提到“12 个空白位”，但对 db0c5428 的真实 5x5 几何，原题实际只有 8 个空白
#      gap 槽位；本实现按真实几何把这 8 个 gap 全部补满，最终 25 个 block 全有内容。
# b) Complexity calibration
#    - 插值规则只分两类：横向 gap 与纵向 gap。每类都只依赖相邻 donor 的可见边，避免过多档位。
#    - 为了让变化肉眼可见，generator 强制若干 gap 的 donor 边不完全相同；这样输出不会退化成
#      “只是把空格涂成单色多数色”。
#    - 中心补全规则保持原题不变，因此解题者仍能先识别“hole -> center block”，再识别
#      “5x5 外扩 + gap 插值”，推理链清晰。
# c) Hidden-assumption audit of the original main()
#    - 原 `for br,bc` 循环只把 8 个非中心 block 粘到外侧，隐含其余外部槽位保持背景；该前提失效。
#    - 原实现只需要 `out = g.copy()` 再 paste 8 次；融合后必须先构造完整 5x5 block map，再统一渲染。
#    - 原 `dst_r = expanded_r0 + (4 - 2*br) * block` / `dst_c = ...` 仍然可用于那 8 个直接复制槽位，
#      但不能覆盖全部输出结构。
#    - 原中心重建依赖 `block == 3`、hole 周围 8 个像素编码、majority 填中心像素；这些前提仍成立。
# d) Placement legality
#    - 输出仍是和原题一样的 15x15 外扩区域，因此输入放置约束无需扩大：只要 motif 四周各保留
#      1 个 block 的边距，完整密集 5x5 就不会越界。
#    - generator 延续 output-first：先构造 motif 和完整 dense output，再把 motif 放回输入。
# e) Visual observability
#    - 样例里解题者会看到：中心 hole 被补上；外围不再是“块-空-块”的稀疏脚手架，而是连续 5x5
#      密集图；gap block 明显继承左右或上下 donor 边的颜色条纹。
#    - 因为 gap 只出现在已知块之间，局部因果关系很直接：看相邻边即可解释中间新块。
# f) Visual cleanliness
#    - 没有额外 marker 或跨区域覆盖；所有新像素都局限在原题 already-implied 的 15x15 外扩框内。
#    - gap 插值只用相邻边，不会引入与周围完全无关的噪声颜色。
# g) Color inference
#    - 不做任何“按颜色 id 映射”的隐藏规则；颜色完全来自可见 donor 边的逐像素继承与交替。
# h) New solver assumptions after fusion
#    - 5x5 中所有待填槽位都恰好是横向或纵向 gap，不会出现需要四邻域同时推断的未知块；
#      generator 通过保留原题 block-level 几何保证这一点。
#    - 相邻 donor block 已存在且未越界；generator 继续要求 motif 至少离边界 3 格。
# i) Anti-degeneracy metric
#    - 统计 8 个 gap 槽位里有多少个 donor 边存在逐像素差异 (`mixed_gap_slots`) 以及 donor 边
#      总差异像素数 (`total_edge_mismatches`)。
#    - 如果这些值太低，视觉上会退化成“密集但几乎纯色的填满”，generator 会丢弃该样本。


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5
GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
BG = 8
FOREGROUND_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
COLOR_COUNT_MIN, COLOR_COUNT_MAX = 2, 4
MAJORITY_TOTAL_MIN, MAJORITY_TOTAL_MAX = 10, 16
CORNER_MAJOR_MIN, CORNER_MAJOR_MAX = 5, 8
EDGE_MAJOR_MIN, EDGE_MAJOR_MAX = 5, 8
MAX_RETRIES = 600

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


def _major_foreground_color(g: np.ndarray, bg: int) -> int:
    vals, counts = np.unique(g[g != int(bg)], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int) -> np.ndarray:
    assert int(block) == 3, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg)
    patch = np.full((block, block), main, dtype=int)
    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for rr in range(block):
        for cc in range(block):
            if (rr, cc) == (block // 2, block // 2):
                continue
            patch[rr, cc] = int(g[hole_r + int(_CENTER_OFFSETS[rr]), hole_c + int(_CENTER_OFFSETS[cc])])
    return patch


def _extract_blocks(g: np.ndarray, r0: int, c0: int, block: int, center: np.ndarray | None = None) -> list[list[np.ndarray]]:
    blocks: list[list[np.ndarray]] = []
    for br in range(3):
        row: list[np.ndarray] = []
        for bc in range(3):
            if (br, bc) == (1, 1) and center is not None:
                row.append(np.array(center, dtype=int))
            else:
                r1 = int(r0) + br * block
                c1 = int(c0) + bc * block
                row.append(np.array(g[r1:r1 + block, c1:c1 + block], dtype=int))
        blocks.append(row)
    return blocks


def _interp_horizontal(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    block = int(left.shape[0])
    assert left.shape == right.shape == (block, block) and block == 3
    patch = np.empty_like(left)
    left_edge = left[:, -1]
    right_edge = right[:, 0]
    patch[:, 0] = left_edge
    patch[:, -1] = right_edge
    for rr in range(block):
        patch[rr, 1] = left_edge[rr] if rr % 2 == 0 else right_edge[rr]
    return patch


def _interp_vertical(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    block = int(top.shape[0])
    assert top.shape == bottom.shape == (block, block) and block == 3
    patch = np.empty_like(top)
    top_edge = top[-1, :]
    bottom_edge = bottom[0, :]
    patch[0, :] = top_edge
    patch[-1, :] = bottom_edge
    for cc in range(block):
        patch[1, cc] = top_edge[cc] if cc % 2 == 0 else bottom_edge[cc]
    return patch


def _seed_block_map(blocks: list[list[np.ndarray]]) -> list[list[np.ndarray | None]]:
    block_map: list[list[np.ndarray | None]] = [[None for _ in range(5)] for _ in range(5)]
    for br in range(3):
        for bc in range(3):
            block_map[br + 1][bc + 1] = np.array(blocks[br][bc], dtype=int)
    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            block_map[4 - 2 * br][4 - 2 * bc] = np.array(blocks[br][bc], dtype=int)
    return block_map


def _complete_block_map(blocks: list[list[np.ndarray]]) -> list[list[np.ndarray]]:
    block_map = _seed_block_map(blocks)
    changed = True
    while changed:
        changed = False
        for br in range(5):
            for bc in range(5):
                if block_map[br][bc] is not None:
                    continue
                left = block_map[br][bc - 1] if bc - 1 >= 0 else None
                right = block_map[br][bc + 1] if bc + 1 < 5 else None
                top = block_map[br - 1][bc] if br - 1 >= 0 else None
                bottom = block_map[br + 1][bc] if br + 1 < 5 else None
                if left is not None and right is not None:
                    block_map[br][bc] = _interp_horizontal(left, right)
                    changed = True
                elif top is not None and bottom is not None:
                    block_map[br][bc] = _interp_vertical(top, bottom)
                    changed = True
    assert all(block_map[br][bc] is not None for br in range(5) for bc in range(5))
    return [[np.array(block_map[br][bc], dtype=int) for bc in range(5)] for br in range(5)]


def _render_block_map(block_map: list[list[np.ndarray]], bg: int) -> np.ndarray:
    canvas = np.full((EXPANDED_SIZE, EXPANDED_SIZE), int(bg), dtype=int)
    for br in range(5):
        for bc in range(5):
            paste(canvas, block_map[br][bc], br * BLOCK, bc * BLOCK)
    return canvas


def _dense_expansion_from_blocks(blocks: list[list[np.ndarray]], bg: int) -> np.ndarray:
    return _render_block_map(_complete_block_map(blocks), bg)


def _gap_metrics_from_blocks(blocks: list[list[np.ndarray]]) -> dict[str, int]:
    block_map = _seed_block_map(blocks)
    mixed_gap_slots = 0
    total_edge_mismatches = 0
    gap_slots = 0
    for br in range(5):
        for bc in range(5):
            if block_map[br][bc] is not None:
                continue
            gap_slots += 1
            left = block_map[br][bc - 1] if bc - 1 >= 0 else None
            right = block_map[br][bc + 1] if bc + 1 < 5 else None
            top = block_map[br - 1][bc] if br - 1 >= 0 else None
            bottom = block_map[br + 1][bc] if br + 1 < 5 else None
            if left is not None and right is not None:
                mismatch = int(np.count_nonzero(left[:, -1] != right[:, 0]))
            else:
                assert top is not None and bottom is not None
                mismatch = int(np.count_nonzero(top[-1, :] != bottom[0, :]))
            total_edge_mismatches += mismatch
            if mismatch > 0:
                mixed_gap_slots += 1
    return {
        "gap_slots": int(gap_slots),
        "mixed_gap_slots": int(mixed_gap_slots),
        "total_edge_mismatches": int(total_edge_mismatches),
    }


def analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0
    block = h // 3
    center = _center_patch(g, int(r0), int(c0), block, bg)
    blocks = _extract_blocks(g, int(r0), int(c0), block, center)
    return _gap_metrics_from_blocks(blocks)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand the 3x3-block motif into a fully filled 5x5 block mosaic.

    Input structure:
    - One 9x9 motif sits on the background and is itself a 3x3 arrangement of 3x3 blocks.
    - The middle block is missing, and the 8 pixels facing that hole encode the missing block.
    - The visible 8 surrounding blocks already imply the larger 5x5 layout used by the output.

    Rule:
    - Rebuild the center 3x3 block from the inner-boundary pixels exactly as in the original task.
    - Keep the original 3x3 block region in place and mirror each non-center block to its point-symmetric
      outer slot of the 5x5 block grid.
    - Fill every remaining gap slot of that 5x5 grid by interpolating between its two neighboring donor
      blocks: horizontal gaps mix left/right edge columns, vertical gaps mix top/bottom edge rows.

    Key insight:
    - The output is no longer the sparse 5x5 scaffold of the source task. After center repair, the solver
      should treat the known blocks as anchors of a complete block lattice and synthesize the in-between
      blocks from the adjacent visible edges, yielding a dense 15x15 pattern.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h % 3 == 0, "motif must be a square 3x3 block grid"
    block = h // 3
    assert int(block) == 3, "this task uses 3x3 blocks"

    center = _center_patch(g, int(r0), int(c0), block, bg)
    blocks = _extract_blocks(g, int(r0), int(c0), block, center)
    dense = _dense_expansion_from_blocks(blocks, bg)

    out = np.array(g, copy=True)
    paste(out, dense, int(r0) - block, int(c0) - block)
    return out


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
        if total == int(target_count):
            choices.append(selected)
    assert choices, "unreachable symmetric majority count"
    return random.choice(choices)


def _seed_block(orbits, forced_minor, majority, target_count):
    block = np.full((BLOCK, BLOCK), -1, dtype=int)
    major_idxs = _major_orbits(orbits, forced_minor, target_count)
    minor_orbits = []
    for idx, orbit in enumerate(orbits):
        if idx in major_idxs:
            for rr, cc in orbit:
                block[rr, cc] = int(majority)
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
        for rr, cc in orbit:
            block[rr, cc] = int(color)


def _make_blocks(majority, minor_colors, corner_major, edge_major):
    b00, corner_minors = _seed_block(DIAG_ORBITS, [(2, 2)], majority, corner_major)
    b01, edge_minors = _seed_block(LR_ORBITS, [(2, 1)], majority, edge_major)
    minor_orbits = corner_minors + edge_minors
    assert len(minor_orbits) >= len(minor_colors)
    _assign_minor_colors(minor_orbits, minor_colors)
    assert np.all(b00 != -1) and np.all(b01 != -1)
    return b00, b01


def _assemble_motif(b00: np.ndarray, b01: np.ndarray) -> np.ndarray:
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


def _gen_center_patch(motif: np.ndarray, majority: int) -> np.ndarray:
    rows = BLOCK + _CENTER_OFFSETS
    cols = BLOCK + _CENTER_OFFSETS
    patch = motif[np.ix_(rows, cols)].copy()
    patch[1, 1] = int(majority)
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
    filled = motif.copy()
    paste(filled, center, BLOCK, BLOCK)
    blocks = _extract_blocks(filled, 0, 0, BLOCK)
    metrics = _gap_metrics_from_blocks(blocks)
    if metrics["mixed_gap_slots"] < 4 or metrics["total_edge_mismatches"] < 8:
        return None

    expanded = _dense_expansion_from_blocks(blocks, BG)
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
