# Combo variant of 1818057f (index 6)
# combo_spec: D1/4+D6/0
# Name: 孤立与嵌入混合膨胀填充

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for combo D1/4 + D6/0
# -------------------------------------
# spec:
#   fused_rule:
#     original:
#       - 只允许嵌入在更大前景成分中的 '+'。
#       - 检测到 '+' 后，只把十字本身 5 格改成 teal。
#     combo:
#       - '+' 既可以是恰好 5 格的孤立十字，也可以仍然嵌入在更大的前景形状里。
#       - 只要某个前景中心满足“上下左右四邻都为前景”，输出就把该中心的整个 3x3 邻域都填成 teal。
#   concrete_pixel_behavior:
#     input:
#       - 背景仍是全局最多色；生成器固定用 2/red，并保证前景不到一半。
#       - 仍只有一种非背景色；生成器固定用 4/yellow。
#       - 目标 '+' 的检测条件完全沿用原题：中心 + 四个正交邻居都为前景。
#       - 同一张图里同时出现两类目标：
#         1. 孤立十字：组件恰好只有 5 格。
#         2. 嵌入十字：十字沿某些臂继续延伸，属于更大连通分量。
#       - 为了让 D1/4 的“膨胀补对角”肉眼可见，所有目标 '+' 的 3x3 对角格都保持背景。
#     output:
#       - 每个检测到的 '+' 都产出一个 teal 3x3 实心块。
#       - 孤立十字从 5 格直接长成 9 格方块。
#       - 嵌入十字则表现为“中间膨胀成 teal 方块，外侧延伸臂仍保持 yellow”，视觉上像 teal 方块与周围 fg 融合。
#   behavior_trace:
#     isolated_example:
#       input:
#         2 2 2 2 2
#         2 2 4 2 2
#         2 4 4 4 2
#         2 2 4 2 2
#         2 2 2 2 2
#       combo_output:
#         2 8 8 8 2
#         2 8 8 8 2
#         2 8 8 8 2
#         2 2 2 2 2
#         2 2 2 2 2
#     embedded_example:
#       input:
#         2 2 2 2 2 2 2
#         2 2 2 4 2 2 2
#         2 2 4 4 4 4 2
#         2 2 2 4 2 2 2
#         2 2 2 4 2 2 2
#         2 2 2 2 2 2 2
#       original_output_would_be:
#         2 2 2 8 2 2 2
#         2 2 8 8 8 4 2
#         2 2 2 8 2 2 2
#         2 2 2 4 2 2 2
#         2 2 2 2 2 2 2
#       combo_output:
#         2 2 8 8 8 2 2
#         2 2 8 8 8 4 2
#         2 2 8 8 8 2 2
#         2 2 2 4 2 2 2
#         2 2 2 2 2 2 2
#     difference:
#       - 原题不会出现“干净的 5 格十字作为完整组件”，而这里会。
#       - 原题只改已有十字 5 格；这里每个目标都会额外把 4 个对角背景格补成 teal。
#   complexity_calibration:
#     - solver 仍只有一个局部检测器，不把“孤立”和“嵌入”拆成两套变换，避免规则分叉。
#     - generator 强制每板至少有一个孤立 '+' 和一个嵌入 '+'，这样 D6/0 不是纸面允许，而是样例里稳定可见。
#     - 目标中心之间使用更保守的间距约束，避免多个 3x3 teal 块或延伸臂挤在一起造成视觉歧义。
#   implicit_assumptions_review_for_main:
#     - `bg = get_bg(g)` 依赖背景为最常见色；生成器继续保证 fg 总量 < 50%。
#     - `colors[0]` 当作唯一前景色仍成立，因为输入只含 bg 与一种 fg。
#     - 遍历 `r in 1..H-2, c in 1..W-2` 只依赖 '+' 中心至少有一圈正交邻居；融合后依旧成立。
#     - 检测条件依然只看正交四邻，不能把“对角为空”写进 solver；
#       否则会把“输出膨胀规则”错误地改造成“更严格的输入匹配规则”。
#     - 3x3 写入会覆盖背景或前景，因此 generator 必须保持目标对角为背景、并让不同目标的 3x3 不重叠。
#   placement_legality:
#     - 中心只从距边界至少 2 格的位置选取，使轴向 distance-2 护栏格也在棋盘内。
#     - 每个中心的保护区包含：
#       1. 中心 3x3：保证输出膨胀的 4 个对角在输入中可见。
#       2. 四个轴向 distance-2 格：用于控制“孤立 vs 嵌入”。
#     - 孤立 '+' 不允许任何额外前景进入这 8 个护栏位，因此其连通分量保持恰好 5 格。
#     - 嵌入 '+' 只允许沿选定臂从 distance-2 开始延伸，并且延伸不会穿入别的中心保护区。
#   visual_observability:
#     - 每个真 '+' 都贡献 4 个 bg -> teal 的对角补格，因此 D1/4 的变化在单个样例中就能看见。
#     - 同板同时出现“完全独立的 teal 方块”和“带黄色尾巴的 teal 方块”，人类能直接读出 D6/0 的混合性。
#   cleanliness:
#     - 干扰物永远不会进入任何目标中心的 3x3 或轴向 distance-2 护栏区。
#     - 干扰物只用少量小线段/小块，并在每次加入后验证不会新造 '+'，防止画面退化成大面积噪声。
#   generator_validations:
#     - `set(_find_plus_centers(inp, fg)) == intended_centers`
#     - 每个孤立中心的 8 个护栏格都不是 fg。
#     - 每个嵌入中心至少有一个轴向 distance-2 格是 fg。
#     - `np.sum((inp == bg) & (out == 8)) == 4 * len(intended_centers)`：
#       每个目标都确实把 4 个背景对角补成 teal。
#     - `len(isolated_centers) >= 1 and len(embedded_centers) >= 1`：
#       量化保证融合行为在生成样本中实际出现，而不是退化回单一类型。
#   new_main_assumptions_after_edit:
#     - 多个 3x3 写入的顺序应当无关，因此 generator 通过中心间距确保这些输出块不重叠。
#     - 因为 solver 不区分孤立/嵌入，generator 必须显式构造两种类型，并用护栏位锁定它们的类别。


CROSS5 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect local '+' motifs in a single foreground color and dilate each one into a teal 3x3 block.

    Input structure:
    - bg is the most common color.
    - One foreground color draws mixed shapes on the bg; some targets are isolated 5-cell crosses and some are crosses embedded in larger fg components.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - For every such center, recolor the entire 3x3 neighborhood around it to teal (8), including the four diagonals.
    - All non-targeted cells outside those 3x3 neighborhoods stay unchanged.

    Key insight: The detection cue is still the same thin orthogonal cross as in the original task, but the allowed contexts are broader and the output footprint is larger, so both clean isolated crosses and embedded crosses must trigger the same square-filling transform.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(g[r, c]) != fg:
                continue
            if all(int(g[r + dr, c + dc]) == fg for dr, dc in DIRS4):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        out[r + dr, c + dc] = 8
    return out


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if all(int(grid[r + dr, c + dc]) == int(fg) for dr, dc in DIRS4):
                centers.append((int(r), int(c)))
    return centers


def _moat_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c + 1),
        (r + 1, c - 1),
        (r + 1, c + 1),
        (r - 2, c),
        (r + 2, c),
        (r, c - 2),
        (r, c + 2),
    ]


def _protected_zone(r: int, c: int) -> set[tuple[int, int]]:
    zone: set[tuple[int, int]] = set()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            zone.add((int(r + dr), int(c + dc)))
    zone.update(_moat_cells(int(r), int(c)))
    return zone


def _mark_mask(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    H, W = mask.shape
    for r, c in cells:
        if 0 <= int(r) < H and 0 <= int(c) < W:
            mask[int(r), int(c)] = True


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], fg: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(fg)


def _can_paint_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    protected: np.ndarray,
    allowed_centers: set[tuple[int, int]],
    bg: int,
) -> bool:
    H, W = grid.shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if bool(protected[int(r), int(c)]):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False

    candidate = grid.copy()
    _paint_cells(candidate, cells, fg)
    return set(_find_plus_centers(candidate, fg)) == allowed_centers


def _can_paint_extension(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    allowed_centers: set[tuple[int, int]],
    bg: int,
) -> bool:
    H, W = grid.shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False

    candidate = grid.copy()
    _paint_cells(candidate, cells, fg)
    return set(_find_plus_centers(candidate, fg)) == allowed_centers


def _is_well_separated(r: int, c: int, centers: list[tuple[int, int]]) -> bool:
    for er, ec in centers:
        if max(abs(int(r) - er), abs(int(c) - ec)) < 4:
            return False
        if abs(int(r) - er) + abs(int(c) - ec) < 7:
            return False
    return True


def _corridors_for_center(
    center: tuple[int, int],
    zones_by_center: dict[tuple[int, int], set[tuple[int, int]]],
    shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    H, W = shape
    cr, cc = center
    blocked_by_others: set[tuple[int, int]] = set()
    for other, cells in zones_by_center.items():
        if other == center:
            continue
        blocked_by_others.update(cells)

    corridors: list[list[tuple[int, int]]] = []
    for dr, dc in DIRS4:
        line: list[tuple[int, int]] = []
        step = 2
        while True:
            nr = int(cr + dr * step)
            nc = int(cc + dc * step)
            if not (0 <= nr < H and 0 <= nc < W):
                break
            if (nr, nc) in blocked_by_others:
                break
            line.append((nr, nc))
            step += 1
        if line:
            corridors.append(line)
    return corridors


def _center_is_isolated(grid: np.ndarray, center: tuple[int, int], fg: int) -> bool:
    cr, cc = center
    return all(int(grid[r, c]) != int(fg) for r, c in _moat_cells(cr, cc))


def _center_is_embedded(grid: np.ndarray, center: tuple[int, int], fg: int) -> bool:
    cr, cc = center
    return any(int(grid[cr + 2 * dr, cc + 2 * dc]) == int(fg) for dr, dc in DIRS4)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    target_k = int(rng.integers(2, 8))
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if _is_well_separated(int(cr), int(cc), centers):
            centers.append((int(cr), int(cc)))
            if len(centers) >= target_k:
                break
    if len(centers) < 2:
        return None

    zones_by_center = {center: _protected_zone(*center) for center in centers}
    protected = np.zeros((H, W), dtype=bool)
    for cells in zones_by_center.values():
        _mark_mask(protected, cells)

    corridors_by_center = {
        center: _corridors_for_center(center, zones_by_center, grid.shape) for center in centers
    }
    embeddable = [center for center in centers if corridors_by_center[center]]
    if not embeddable:
        return None

    rng.shuffle(centers)
    rng.shuffle(embeddable)
    max_embedded = min(len(embeddable), len(centers) - 1)
    if max_embedded < 1:
        return None
    n_embedded = int(rng.integers(1, max_embedded + 1))
    embedded_centers = set(embeddable[:n_embedded])
    isolated_centers = [center for center in centers if center not in embedded_centers]
    if not isolated_centers:
        return None

    for cr, cc in centers:
        for dr, dc in CROSS5:
            grid[cr + dr, cc + dc] = fg

    intended_centers = set(centers)
    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    for center in embedded_centers:
        corridors = corridors_by_center[center]
        if not corridors:
            return None
        n_dirs = int(rng.integers(1, min(2, len(corridors)) + 1))
        chosen_ids = np.atleast_1d(rng.choice(len(corridors), size=n_dirs, replace=False))
        added_any = False
        for idx in chosen_ids:
            corridor = corridors[int(idx)]
            max_len = min(3, len(corridor))
            ext_len = int(rng.integers(1, max_len + 1))
            cells = corridor[:ext_len]
            if not _can_paint_extension(grid, cells, fg, intended_centers, bg):
                continue
            _paint_cells(grid, cells, fg)
            added_any = True
        if not added_any:
            return None

    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    for center in isolated_centers:
        if not _center_is_isolated(grid, center, fg):
            return None
    for center in embedded_centers:
        if not _center_is_embedded(grid, center, fg):
            return None

    shapes = [
        [(0, 0)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0), (2, 0)],
        [(0, 0), (0, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1)],
        [(0, 1), (1, 0), (1, 1)],
        [(0, 0), (0, 1), (1, 1)],
        [(0, 0), (1, 0), (0, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1)],
        [(0, 0), (1, 0), (2, 0), (3, 0)],
        [(0, 0), (0, 1), (0, 2), (0, 3)],
    ]

    total = H * W
    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    for _ in range(total * 30):
        if current_fg >= target_fg:
            break
        remaining = target_fg - current_fg
        usable_shapes = [shape for shape in shapes if len(shape) <= max(1, remaining)]
        shape = usable_shapes[int(rng.integers(len(usable_shapes)))]
        anchor_r = int(rng.integers(0, H))
        anchor_c = int(rng.integers(0, W))
        cells = [(anchor_r + dr, anchor_c + dc) for dr, dc in shape]
        if not _can_paint_cells(grid, cells, fg, protected, intended_centers, bg):
            continue
        _paint_cells(grid, cells, fg)
        current_fg += len(cells)

    current_fg = int(np.sum(grid != bg))
    if current_fg < int(total * 0.24):
        return None
    if current_fg >= total // 2:
        return None

    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    isolated_count = sum(_center_is_isolated(grid, center, fg) for center in intended_centers)
    embedded_count = sum(_center_is_embedded(grid, center, fg) for center in intended_centers)
    if isolated_count < 1 or embedded_count < 1:
        return None
    if isolated_count + embedded_count != len(intended_centers):
        return None

    inp = grid.copy()
    out = main(inp)
    if not np.any(out == 8):
        return None

    if int(np.sum(out == 8)) != 9 * len(intended_centers):
        return None

    bg_to_teal = int(np.sum((inp == bg) & (out == 8)))
    if bg_to_teal != 4 * len(intended_centers):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
