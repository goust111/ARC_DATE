# Combo variant of 1818057f (index 22)
# combo_spec: D1/4+D9/2+D11/0
# Name: 最大分量内两步膨胀

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# -------------------------------------
# combo_rule:
#   - 先找唯一最大前景4连通分量。
#   - 只在该分量内检测 '+' 中心。
#   - 处理顺序拆成三步：
#       1) 先把中心点染成 teal。
#       2) 再把这些中心的4个正交前景邻居染成 teal，得到标准 5 格十字。
#       3) 最后补上同一中心 3x3 邻域里的4个对角格，把十字闭成 3x3 实心块。
#   - 小分量中的同形 '+' 完全忽略。
#
# behavior_trace:
#   example_input:
#     - 大分量里有一个嵌入式 '+':
#         2 4 2
#         4 4 4
#         2 4 2
#       且远处还有一个较小独立分量，内部也含同样的 '+'。
#   original_task:
#     - 两个分量里的 '+' 都会被处理，而且每个只改十字 5 格。
#   combo_variant:
#     - 先比较分量大小，只保留大分量。
#     - 第一步输出概念上只标记中心 1 格。
#     - 第二步把中心扩到十字 5 格。
#     - 第三步只补这批中心各自的4个对角，最终每个目标 '+' 变成一个 3x3 teal 方块。
#     - 小分量里的 '+' 连中心都不会被标记，因此整个小分量保持黄色。
#   observable_difference:
#     - 与原题相比：不仅出现“有的 '+' 不处理”，被处理的那些还从 5 格扩成 9 格。
#     - 与单独 D11/0 相比：大分量中的 '+' 不再只是十字，而是补成方块。
#     - 与单独 D1/4 相比：并非所有 '+' 都会补成方块，只有最大分量内的才会。
#
# complexity_calibration:
#   - 若只生成一个前景分量，D11/0 的全局筛选就不可见，因此 generator 强制至少再放一个较小的含 '+'
#     分量，让“相同局部形状、不同命运”的对比稳定出现。
#   - 若允许目标中心太近，3x3 方块会并在一起，解题者会更像在看大块涂抹而不是“逐个 '+' 被补满”；
#     因此主分量里的中心间距固定为 4 或 5。
#   - 若不保护对角格为空，D1/4 的膨胀特征会退化成纯换色；因此所有目标 '+' 的 3x3 保留区都不许噪声侵入。
#
# hidden_assumptions_review_for_main:
#   - `bg = get_bg(g)`: 仍假设背景是最多颜色；generator 保持前景严格少于半数。
#   - `fg = colors[0]`: 仍假设输入只有一种非背景色；generator 继续只用 2 和 4。
#   - `_largest_component_mask(g, fg)`: 新 solver 依赖“最大分量唯一”；generator 验证最大分量严格大于第二大分量。
#   - `_plus_centers(g, fg, keep)`: 新的 '+' 判定不只要求局部 4 邻域为 fg，还要求这些像素都属于最大分量。
#     若不加 `keep` 掩码，小分量中的 '+' 会被误处理。
#   - `centers -> cross cells -> square cells` 的三阶段写入假设检测集合固定，不会因为中间写 teal 反过来制造新中心；
#     因此 solver 始终基于原输入 `g` 检测中心，再把三步拆开执行。
#   - 第三步补方块不是对整片 teal 做通用形态学膨胀；若那样做，十字的5个 teal 会扩成比 3x3 更大的块，
#     偏离 D1/4 的真实语义。这里补的是“每个已确认中心对应的4个对角格”。
#
# placement_legality:
#   - 各组件在放置时禁止与已有组件 4 邻接，确保连通分量彼此独立。
#   - 每个含 '+' 的组件都携带局部中心列表；放置时额外检查这些中心的全局 3x3 保留区不越界、不重叠、
#     且不覆盖其他组件已有像素。
#   - 主分量模板自带桥接线与远端延伸，保证目标 '+' 不是孤立 5 格，而是嵌在更大的黄色连通块里。
#
# visual_observability_and_cleanliness:
#   - 每个样例同时含有：一个最大黄分量里的多个 '+'，以及至少一个较小黄分量里的 '+'。
#   - 所有目标 '+' 的输入对角格都留空，输出里这些格统一变 teal，因此人眼能直接看出“十字 -> 方块”。
#   - 小分量中的同形 '+' 保持黄十字不变，提供清晰的对照，避免解题者把规则误判为“所有 '+' 都补满”。
#   - 干扰组件只用短线、L 形、2x2 方块等简单非 '+' 结构，减少局部纹理冲突。
#
# generator_validations:
#   - 最大分量中的 '+' 数量 `main_count >= 2`，非最大分量中的 '+' 数量 `ignored_count >= 1`。
#   - `main(input)` 产生的 teal 总数必须是 `9 * main_count`，且 `bg -> teal` 数量必须是 `4 * main_count`；
#     这说明每个被处理的 '+' 都确实从十字补成了独立 3x3 方块。
#   - 至少一个被忽略中心在输出中仍保持 fg，证明 D11/0 的分量筛选确实生效。


ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]
PLUS5 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ORTHO:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _largest_component_mask(grid: np.ndarray, fg: int) -> np.ndarray:
    keep = np.zeros_like(grid, dtype=bool)
    comps = _connected_components(grid == fg)
    if not comps:
        return keep
    largest = max(comps, key=len)
    for r, c in largest:
        keep[r, c] = True
    return keep


def _plus_centers(
    grid: np.ndarray, fg: int, allowed: np.ndarray | None = None
) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if allowed is not None and not allowed[r, c]:
                continue
            ok = True
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if int(grid[nr, nc]) != int(fg):
                    ok = False
                    break
                if allowed is not None and not allowed[nr, nc]:
                    ok = False
                    break
            if ok:
                centers.append((int(r), int(c)))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect '+' motifs only inside the largest foreground component, then expand them into teal 3x3 blocks in two stages.

    Input structure:
    - bg is the most common color.
    - One foreground color forms several disconnected 4-connected components.
    - Some components contain thin '+' motifs, but only the unique largest foreground component is eligible for processing.

    Rule:
    - Find the largest 4-connected foreground component.
    - Inside that component, mark every '+' center teal, where a '+' center is a foreground pixel whose 4 orthogonal neighbors are also foreground in the same kept component.
    - In a second step, recolor the center and its 4 orthogonal neighbors to teal.
    - In a third step, fill the 4 diagonal cells around each marked center as well, producing a teal 3x3 solid block.
    - Ignore '+' motifs in all smaller components; leave every other pixel unchanged.

    Key insight: The transformation is a fusion of a global filter and a local multi-step expansion: first select the only component that matters, then infer that each accepted '+' grows from center to cross to full square, while visually identical '+' motifs elsewhere stay untouched.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    keep = _largest_component_mask(g, fg)
    centers = _plus_centers(g, fg, keep)
    if not centers:
        return out

    center_mask = np.zeros_like(g, dtype=bool)
    for r, c in centers:
        center_mask[r, c] = True

    # Step 1: mark centers.
    out[center_mask] = 8

    # Step 2: expand from the marked centers to their 4 foreground neighbors.
    cross_mask = center_mask.copy()
    for r, c in centers:
        for dr, dc in ORTHO:
            nr, nc = r + dr, c + dc
            if keep[nr, nc] and int(g[nr, nc]) == fg:
                cross_mask[nr, nc] = True
    out[cross_mask] = 8

    # Step 3: add the four diagonals around each confirmed center to complete a 3x3 block.
    square_mask = cross_mask.copy()
    for r, c in centers:
        for dr in (-1, 1):
            for dc in (-1, 1):
                square_mask[r + dr, c + dc] = True
    out[square_mask] = 8
    return out


def _normalize(cells: set[tuple[int, int]], centers: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm_cells = {(r - min_r, c - min_c) for r, c in cells}
    norm_centers = {(r - min_r, c - min_c) for r, c in centers}
    return norm_cells, norm_centers


def _component_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _add_plus(cells: set[tuple[int, int]], center: tuple[int, int]) -> None:
    r, c = center
    for dr, dc in PLUS5:
        cells.add((r + dr, c + dc))


def _make_main_component(
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    orientation = str(rng.choice(["h", "v"]))
    count = int(rng.integers(2, 5))
    centers: list[tuple[int, int]] = [(0, 0)]
    for _ in range(1, count):
        gap = int(rng.integers(4, 6))
        pr, pc = centers[-1]
        if orientation == "h":
            centers.append((pr, pc + gap))
        else:
            centers.append((pr + gap, pc))

    cells: set[tuple[int, int]] = set()
    for center in centers:
        _add_plus(cells, center)

    for (r1, c1), (r2, c2) in zip(centers, centers[1:]):
        if r1 == r2:
            for cc in range(c1 + 2, c2 - 1):
                cells.add((r1, cc))
        else:
            for rr in range(r1 + 2, r2 - 1):
                cells.add((rr, c1))

    for idx, (r, c) in enumerate(centers):
        if orientation == "h":
            branch_dir = 1 if rng.random() < 0.5 else -1
            for step in range(2, 2 + int(rng.integers(1, 3))):
                cells.add((r + branch_dir * step, c))
            if idx == 0:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r, c - step))
            if idx == len(centers) - 1:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r, c + step))
        else:
            branch_dir = 1 if rng.random() < 0.5 else -1
            for step in range(2, 2 + int(rng.integers(1, 3))):
                cells.add((r, c + branch_dir * step))
            if idx == 0:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r - step, c))
            if idx == len(centers) - 1:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r + step, c))

    return _normalize(cells, set(centers))


def _make_small_plus_component(
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    center = (0, 0)
    cells: set[tuple[int, int]] = set()
    _add_plus(cells, center)
    num_dirs = int(rng.integers(1, 3))
    chosen = rng.choice(len(ORTHO), size=num_dirs, replace=False)
    for idx in np.atleast_1d(chosen):
        dr, dc = ORTHO[int(idx)]
        for step in range(2, 2 + int(rng.integers(1, 3))):
            cells.add((center[0] + dr * step, center[1] + dc * step))
    return _normalize(cells, {center})


def _make_distractor_component(
    rng: np.random.Generator,
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    shape = str(rng.choice(["hline", "vline", "elbow", "box"]))
    cells: set[tuple[int, int]] = set()
    if shape == "hline":
        for c in range(int(rng.integers(2, 5))):
            cells.add((0, c))
    elif shape == "vline":
        for r in range(int(rng.integers(2, 5))):
            cells.add((r, 0))
    elif shape == "box":
        for r in range(2):
            for c in range(2):
                cells.add((r, c))
    else:
        arm_r = int(rng.integers(2, 4))
        arm_c = int(rng.integers(2, 4))
        for r in range(arm_r):
            cells.add((r, 0))
        for c in range(arm_c):
            cells.add((0, c))
    return _normalize(cells, set())


def _fits_component(
    occupied: np.ndarray,
    protected: np.ndarray,
    cells: set[tuple[int, int]],
    plus_centers: set[tuple[int, int]],
    top: int,
    left: int,
) -> bool:
    H, W = occupied.shape
    abs_cells = {(top + r, left + c) for r, c in cells}

    for rr, cc in abs_cells:
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if occupied[rr, cc]:
            return False
        if protected[rr, cc]:
            return False

    for rr, cc in abs_cells:
        for dr, dc in ORTHO:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc] and (nr, nc) not in abs_cells:
                return False

    for cr, cc in plus_centers:
        base_r = top + cr
        base_c = left + cc
        if not (1 <= base_r < H - 1 and 1 <= base_c < W - 1):
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = base_r + dr, base_c + dc
                if occupied[nr, nc] and (nr, nc) not in abs_cells:
                    return False
                if protected[nr, nc] and (nr, nc) not in abs_cells:
                    return False
    return True


def _place_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    protected: np.ndarray,
    cells: set[tuple[int, int]],
    plus_centers: set[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
    max_tries: int = 300,
) -> tuple[bool, set[tuple[int, int]]]:
    H, W = grid.shape
    h, w = _component_dims(cells)
    if h > H or w > W:
        return False, set()

    for _ in range(max_tries):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        if not _fits_component(occupied, protected, cells, plus_centers, top, left):
            continue

        placed_centers = {(top + r, left + c) for r, c in plus_centers}
        for r, c in cells:
            rr, cc = top + r, left + c
            grid[rr, cc] = fg
            occupied[rr, cc] = True
        for cr, cc in placed_centers:
            protected[cr - 1 : cr + 2, cc - 1 : cc + 2] = True
        return True, placed_centers
    return False, set()


def _foreground_count(grid: np.ndarray, bg: int) -> int:
    return int(np.sum(grid != bg))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(12, 23))
    W = int(rng.integers(12, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)
    protected = np.zeros((H, W), dtype=bool)

    main_cells, main_local_centers = _make_main_component(rng)
    ok, main_centers = _place_component(
        grid, occupied, protected, main_cells, main_local_centers, fg, rng, max_tries=500
    )
    if not ok:
        return None

    all_plus_centers = set(main_centers)
    small_plus_count = int(rng.integers(1, 4))
    for _ in range(small_plus_count):
        cells, local_centers = _make_small_plus_component(rng)
        ok, placed_centers = _place_component(
            grid, occupied, protected, cells, local_centers, fg, rng, max_tries=400
        )
        if not ok:
            return None
        all_plus_centers |= placed_centers

    target_density = float(rng.uniform(0.26, 0.39))
    target_fg = int(H * W * target_density)

    distractor_budget = int(rng.integers(6, 12))
    for _ in range(distractor_budget):
        if _foreground_count(grid, bg) >= target_fg:
            break
        cells, local_centers = _make_distractor_component(rng)
        _place_component(grid, occupied, protected, cells, local_centers, fg, rng, max_tries=120)

    while _foreground_count(grid, bg) < target_fg:
        cells, local_centers = _make_distractor_component(rng)
        ok, _ = _place_component(
            grid, occupied, protected, cells, local_centers, fg, rng, max_tries=80
        )
        if not ok:
            break

    fg_count = _foreground_count(grid, bg)
    if fg_count < int(H * W * 0.24):
        return None
    if fg_count >= (H * W) // 2:
        return None

    comps = sorted(_connected_components(grid == fg), key=len, reverse=True)
    if len(comps) < 2:
        return None
    if len(comps[0]) <= len(comps[1]):
        return None

    largest_mask = _largest_component_mask(grid, fg)
    main_detected = set(_plus_centers(grid, fg, largest_mask))
    all_detected = set(_plus_centers(grid, fg))
    ignored_detected = all_detected - main_detected

    if main_detected != main_centers:
        return None
    if len(main_detected) < 2:
        return None
    if len(ignored_detected) < 1:
        return None
    if all_detected != all_plus_centers:
        return None

    out = main(grid)
    if not np.any(out == 8):
        return None

    main_count = len(main_detected)
    if int(np.sum(out == 8)) != 9 * main_count:
        return None
    if int(np.sum((grid == bg) & (out == 8))) != 4 * main_count:
        return None

    if not any(int(out[r, c]) == fg for r, c in ignored_detected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
