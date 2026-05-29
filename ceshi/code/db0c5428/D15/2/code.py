# Variant of db0c5428
# Dimension: D15 — 块内像素的结构模式 (meta)
# Change: 每个3x3块内部为自由的多色图案，遵循整体图案的双轴对称约束，无额外的块内结构要求
#   → 块内有一个特殊标记像素指示块的朝向；外扩时需要依据该标记决定旋转方式。
#
# Design analysis
# spec:
#   original_frame:
#     - 输入仍是单个 9x9 motif，按 3x3 个 3x3 block 排列，中心 block 为空背景。
#     - 输出仍是在原位置保留 motif，把 8 个非中心 block 放到 5x5 block 网格的对称外槽，并补出中心 block。
#   concretization_and_trace:
#     - 每个可见 3x3 block 恰好有 1 个“控制色”角像素；它只出现在四个角之一，用来表示 block 当前朝向。
#     - 中心 block 的补法保持原题：仍然从 hole 周围的 8 个内边界像素取值，中心像素填主体色。
#     - 原题在外扩环节直接 `paste(src)`；变体在这里改变：
#       先读取 src 的控制角，再按“角所在方位 = 顺时针旋转档位”来旋转整个 3x3 block，
#       然后把旋转后的 block 粘贴到点对称外槽。
#     - 具体小例子：若某个 block 的控制角在右上角，则其外扩副本要顺时针转 90 度；
#       原题这里是不转的，所以外圈该 block 的图案和控制角位置都会不同。
#   complexity_calibration:
#     - 若还沿用原题的块内镜像/对角对称，很多旋转会在视觉上退化，因此必须让 seed block 内部变为不对称图案。
#     - 同时，单个控制色角点本身已足够显著；其余只保留 1-3 个普通杂色，避免噪声过多。
#     - 整体 9x9 motif 不再强制原题的双轴镜像，因为“每个 block 只有一个角标记”与轴上 block 的镜像约束冲突；
#       这里改为用两个不对称 seed block 经过 90 度旋转铺出 8 个位置，这是 D15 变体成立所必需的适配。
#   implicit_assumptions_audit:
#     - `bbox(g != bg)` 仍成立：输入还是唯一的 9x9 非背景 motif。
#     - `block == 3` 仍成立：变体没有改 block 尺寸。
#     - `_center_patch()` 仍可用“hole 周围 8 像素 + 中心主体色”的逻辑，因为 D15 只改外扩时的旋转控制。
#     - 原 `for br, bc ... paste(out, src, ...)` 依赖“外扩 copy 不旋转”，这是变体中唯一必须替换的关键旧前提。
#     - 新 solver 额外假设：
#       控制色在每个可见 block 中恰好出现一次且位于角上；generator 会显式验证这一点。
#   placement_legality:
#     - 仍使用 16..19 的正方形画布，motif 左上角满足至少 3 格边距，因此 15x15 expanded 区域始终不越界。
#     - 只有一个 motif，没有其他物体，因此不存在重叠匹配歧义。
#   visual_observability:
#     - 解题者可直接看见每个 block 的控制角，再对照外圈对应副本，归纳出“角位置决定旋转量”。
#     - 每个样例里四种控制角都会各出现两次，因此旋转规则在单个样例中就有重复支撑，不依赖跨样例记忆颜色 id。
#   cleanliness_and_color_inferability:
#     - 控制色只承担一个职责：标记朝向。它不参与中心重建，不做隐式颜色映射。
#     - 普通前景色总量控制在 2-4 种（含主体色），再额外加入 1 种控制色；数量足以做不对称，但不会花哨到难以读图。
#   behavior_metric_for_non_degeneracy:
#     - 新规则的量化指标取“8 个 block 的外扩旋转量分布”和“新输出相对原 solver 的差异格数”。
#     - 生成器被构造成每个样例必有 2 个 0 转、2 个 90 度、2 个 180 度、2 个 270 度的外扩 copy；
#       因此至少 6/8 个外圈 block 必然与原题的‘不旋转复制’不同，不能退化回原行为。
#   new_main_assumptions_checked_against_generator:
#     - generator 保证控制色唯一可识别，保证主体色严格多数，保证四种控制角都出现，保证中心边界像素不被控制色污染。
#     - solver 的每个循环只依赖这些局部性质，不依赖旧版的全局镜像对称。

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox, crop_bbox, get_bg, paste


BLOCK = 3
MOTIF_SIZE = BLOCK * 3
EXPANDED_SIZE = BLOCK * 5

GRID_SIZE_MIN, GRID_SIZE_MAX = 16, 19
MOTIF_PAD = 3
MAX_RETRIES = 400

BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX = 2, 4

_CENTER_OFFSETS = np.array([-1, 1, 3])
_VISIBLE_BLOCKS = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
_CORNERS = ((0, 0), (0, 2), (2, 2), (2, 0))
_CORNER_TO_CW = {(0, 0): 0, (0, 2): 1, (2, 2): 2, (2, 0): 3}


def _major_foreground_color(g: np.ndarray, bg: int, exclude: set[int] | None = None) -> int:
    mask = g != int(bg)
    if exclude:
        for color in exclude:
            mask &= g != int(color)
    vals, counts = np.unique(g[mask], return_counts=True)
    assert vals.size, "input must contain foreground colors"
    return int(vals[int(np.argmax(counts))])


def _block_slice(r0: int, c0: int, br: int, bc: int) -> tuple[int, int, int, int]:
    return (
        int(r0) + br * BLOCK,
        int(c0) + bc * BLOCK,
        int(r0) + (br + 1) * BLOCK - 1,
        int(c0) + (bc + 1) * BLOCK - 1,
    )


def _iter_visible_blocks(g: np.ndarray, r0: int, c0: int) -> list[np.ndarray]:
    return [crop_bbox(g, _block_slice(r0, c0, br, bc)) for br, bc in _VISIBLE_BLOCKS]


def _marker_color(g: np.ndarray, r0: int, c0: int, bg: int) -> int:
    blocks = _iter_visible_blocks(g, r0, c0)
    colors = [int(c) for c in np.unique(g) if int(c) != int(bg)]
    candidates: list[int] = []
    for color in colors:
        ok = True
        for block in blocks:
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            rc = tuple(int(x) for x in hits[0])
            if rc not in _CORNERS:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    assert len(candidates) == 1, "expected one unique control color"
    return int(candidates[0])


def _center_patch(g: np.ndarray, r0: int, c0: int, block: int, bg: int, marker: int) -> np.ndarray:
    assert block == BLOCK, "this task uses 3x3 blocks"
    main = _major_foreground_color(g, bg, exclude={marker})
    patch = np.full((block, block), main, dtype=int)

    hole_r = int(r0) + block
    hole_c = int(c0) + block
    for r in range(block):
        for c in range(block):
            if r == block // 2 and c == block // 2:
                continue
            patch[r, c] = int(g[hole_r + _CENTER_OFFSETS[r], hole_c + _CENTER_OFFSETS[c]])
    return patch


def _extract_motif(g: np.ndarray, bg: int) -> tuple[int, int, int]:
    motif_bbox = bbox(g != bg)
    assert motif_bbox is not None, "expected a foreground motif"
    r0, c0, r1, c1 = motif_bbox
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    assert h == w and h == MOTIF_SIZE, "motif must be a 9x9 3x3-block arrangement"
    return int(r0), int(c0), BLOCK


def _marker_corner(block: np.ndarray, marker: int) -> tuple[int, int]:
    hits = np.argwhere(block == int(marker))
    assert hits.shape[0] == 1, "each block must contain the control color exactly once"
    rc = tuple(int(x) for x in hits[0])
    assert rc in _CORNERS, "control color must sit on a corner"
    return rc


def _rotation_steps_cw(block: np.ndarray, marker: int) -> int:
    return int(_CORNER_TO_CW[_marker_corner(block, marker)])


def _rotate_cw(block: np.ndarray, steps: int) -> np.ndarray:
    return np.rot90(block, -int(steps))


def rotation_profile(grid: np.ndarray) -> list[int]:
    """Required outer-copy rotations in clockwise quarter-turns for the 8 visible blocks."""
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, _block = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)
    return [_rotation_steps_cw(block, marker) for block in _iter_visible_blocks(g, r0, c0)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a 3x3 block motif into a 5x5 block motif, using a corner marker to control rotation.

    Input structure:
    - Teal background contains one 9x9 motif made from eight visible 3x3 blocks around a teal center hole.
    - Each visible block has exactly one special corner pixel of the same control color; that corner encodes the block's orientation.
    - The eight colored cells bordering the hole still provide the non-center pixels of the missing middle block.

    Rule:
    - Keep the original 9x9 motif in place and reconstruct the center block from the inner boundary, with its center set to the main body color.
    - Copy each of the eight visible blocks into the opposite outer slot of the implied 5x5 block layout.
    - Before pasting an outer copy, rotate the whole 3x3 block according to which corner holds the control pixel.

    Key insight: The corner marker is not a decorative pixel; it is a local control signal telling you how many quarter-turns the entire source block needs before it is pasted outward.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    r0, c0, block = _extract_motif(g, bg)
    marker = _marker_color(g, r0, c0, bg)

    out = g.copy()
    expanded_r0 = int(r0) - block
    expanded_c0 = int(c0) - block

    for br in range(3):
        for bc in range(3):
            if (br, bc) == (1, 1):
                continue
            src = crop_bbox(g, _block_slice(int(r0), int(c0), br, bc))
            rotated = _rotate_cw(src, _rotation_steps_cw(src, marker))
            dst_r = expanded_r0 + (4 - 2 * br) * block
            dst_c = expanded_c0 + (4 - 2 * bc) * block
            paste(out, rotated, dst_r, dst_c)

    center = _center_patch(g, int(r0), int(c0), block, bg, marker)
    paste(out, center, int(r0) + block, int(c0) + block)
    return out


def _random_seed_block(
    marker_corner: tuple[int, int],
    marker_color: int,
    majority: int,
    accent_colors: list[int],
    force_body_cell: tuple[int, int] | None = None,
) -> np.ndarray:
    block = np.full((BLOCK, BLOCK), int(majority), dtype=int)
    mr, mc = marker_corner
    block[mr, mc] = int(marker_color)

    cells = [(r, c) for r in range(BLOCK) for c in range(BLOCK) if (r, c) != marker_corner]
    random.shuffle(cells)

    chosen: list[tuple[int, int]] = []
    if force_body_cell is not None and force_body_cell != marker_corner:
        chosen.append(force_body_cell)
    for cell in cells:
        if cell not in chosen:
            chosen.append(cell)
        if len(chosen) >= random.randint(max(2, len(accent_colors)), min(4, len(cells))):
            break

    forced = list(accent_colors)
    random.shuffle(forced)
    colors = forced + [random.choice(accent_colors) for _ in range(len(chosen) - len(forced))]
    random.shuffle(colors)
    for (r, c), color in zip(chosen, colors):
        block[int(r), int(c)] = int(color)
    return block


def _all_rotations_distinct(block: np.ndarray) -> bool:
    seen = {tuple(np.rot90(block, k).flatten()) for k in range(4)}
    return len(seen) == 4


def _assemble_motif(corner_seed: np.ndarray, edge_seed: np.ndarray) -> np.ndarray:
    motif = np.full((MOTIF_SIZE, MOTIF_SIZE), BG, dtype=int)
    blocks = {
        (0, 0): corner_seed,
        (0, 1): edge_seed,
        (0, 2): np.rot90(corner_seed, -1),
        (1, 0): np.rot90(edge_seed, 1),
        (1, 2): np.rot90(edge_seed, -1),
        (2, 0): np.rot90(corner_seed, 1),
        (2, 1): np.rot90(edge_seed, 2),
        (2, 2): np.rot90(corner_seed, 2),
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


def _expanded_output(motif: np.ndarray, center: np.ndarray, marker_color: int) -> np.ndarray:
    expanded = np.full((EXPANDED_SIZE, EXPANDED_SIZE), BG, dtype=int)
    output_motif = motif.copy()
    paste(output_motif, center, BLOCK, BLOCK)
    paste(expanded, output_motif, BLOCK, BLOCK)
    for br, bc in _VISIBLE_BLOCKS:
        src = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
        rotated = _rotate_cw(src, _rotation_steps_cw(src, marker_color))
        paste(expanded, rotated, (4 - 2 * br) * BLOCK, (4 - 2 * bc) * BLOCK)
    return expanded


def _marker_histogram_ok(motif: np.ndarray, marker_color: int) -> bool:
    counts = {corner: 0 for corner in _CORNERS}
    for br, bc in _VISIBLE_BLOCKS:
        block = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
        counts[_marker_corner(block, marker_color)] += 1
    return all(counts[corner] == 2 for corner in _CORNERS)


def _marker_unique_in_motif(motif: np.ndarray, marker_color: int) -> bool:
    colors = [int(c) for c in np.unique(motif) if int(c) != BG]
    candidates = []
    for color in colors:
        ok = True
        for br, bc in _VISIBLE_BLOCKS:
            block = motif[br * BLOCK:(br + 1) * BLOCK, bc * BLOCK:(bc + 1) * BLOCK]
            hits = np.argwhere(block == int(color))
            if hits.shape[0] != 1:
                ok = False
                break
            if tuple(int(x) for x in hits[0]) not in _CORNERS:
                ok = False
                break
        if ok:
            candidates.append(int(color))
    return candidates == [int(marker_color)]


def _body_majority_ok(motif: np.ndarray, majority: int, marker_color: int) -> bool:
    vals, counts = np.unique(motif[(motif != BG) & (motif != int(marker_color))], return_counts=True)
    freq = {int(v): int(c) for v, c in zip(vals, counts)}
    if not freq or int(majority) not in freq:
        return False
    return freq[int(majority)] > max(v for k, v in freq.items() if k != int(majority)) if len(freq) > 1 else True


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(GRID_SIZE_MIN, GRID_SIZE_MAX)
    total_color_count = random.randint(BODY_COLOR_COUNT_MIN, BODY_COLOR_COUNT_MAX + 1)
    colors = random.sample(PALETTE, total_color_count)
    marker_color = int(colors[0])
    body_colors = [int(c) for c in colors[1:]]
    if len(body_colors) < BODY_COLOR_COUNT_MIN:
        return None
    majority = int(body_colors[0])
    accent_colors = body_colors[1:] if len(body_colors) > 1 else [body_colors[0]]

    corner_seed = _random_seed_block((0, 0), marker_color, majority, accent_colors, force_body_cell=(2, 2))
    edge_seed = _random_seed_block((0, 2), marker_color, majority, accent_colors, force_body_cell=(2, 1))

    if not _all_rotations_distinct(corner_seed) or not _all_rotations_distinct(edge_seed):
        return None

    motif = _assemble_motif(corner_seed, edge_seed)
    center = _gen_center_patch(motif, majority)
    if np.any(center == int(marker_color)):
        return None
    if not _marker_histogram_ok(motif, marker_color):
        return None
    if not _marker_unique_in_motif(motif, marker_color):
        return None
    if not _body_majority_ok(motif, majority, marker_color):
        return None

    expanded = _expanded_output(motif, center, marker_color)

    r0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    c0 = random.randint(MOTIF_PAD, n - MOTIF_SIZE - MOTIF_PAD)
    inp = np.full((n, n), BG, dtype=int)
    out = np.full((n, n), BG, dtype=int)
    paste(inp, motif, r0, c0)
    paste(out, expanded, r0 - BLOCK, c0 - BLOCK)

    profile = rotation_profile(inp)
    if sorted(profile) != [0, 0, 1, 1, 2, 2, 3, 3]:
        return None
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise AssertionError("failed to generate a valid db0c5428 D15 input")
