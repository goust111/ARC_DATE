# Variant of 1818057f
# Dimension: D11 — 前景像素的连通性利用方式 (meta)
# Change: 不直接使用连通分量分析，仅在每个像素的局部4-邻域上做判断... → 仅在最大连通分量中检测'+'

from __future__ import annotations

import numpy as np
from arc_utils import get_bg


# Design analysis
# ===============
# spec_variant:
#   core_change:
#     original: "任何前景像素只要其4个正交邻居也都是前景，就把该局部'+'染成青色。"
#     variant: "先对前景做4连通分量分解，只保留唯一最大连通分量；仅在该分量内部检测局部'+'。"
#   concrete_visualization:
#     - 输入仍然只有背景色2和前景色4。
#     - 网格里会同时出现一个明显更大的黄色连通块，以及若干较小的黄色连通块。
#     - 最大连通块和某些较小连通块里都故意包含'+'形局部结构。
#     - 输出只把最大连通块里的这些'+'改成青色8；较小连通块里的同形状'+'保持黄色。
#   flow_trace:
#     - 小例子：若一条长横向链状组件里有3个'+'，旁边另一个独立小组件里有1个'+'。
#     - 原规则会把4个'+'全部染色。
#     - 变体先比较连通块大小，只保留那条长链；于是只染色其中3个'+'，小组件不动。
#     - 差异点不在'+'判定本身，而在“哪些像素有资格进入判定范围”。
#   complexity_calibration:
#     - 若只生成一个前景连通块，变体会退化回原题，视觉上不可观察，因此生成器强制加入含'+'的小组件。
#     - 若允许多个最大块并列，solver 需要额外定义并列规则，容易引入不必要歧义；因此生成器保证最大块唯一。
#     - 仍然只保留一种前景色，避免把难度转移到颜色映射。
#   hidden_assumption_review:
#     - 原 main 中 `fg = colors[0]` 依赖“只有一种前景色”；该前提保持不变。
#     - 原双重循环默认“所有前景像素都可参与'+'检测”；变体下不成立，必须先求最大4连通块并用掩码限域。
#     - 原邻域判断只检查 `g[...] == fg`；在变体里如果不先限域，会错误处理小连通块中的'+'。
#     - 原直接写 `out[...] = 8` 不会出错，因为输出色仍固定为8，且多个'+'重叠写入仍一致。
#   placement_legality:
#     - 所有组件都以局部模板生成，再平移到全局网格。
#     - 组件之间禁止4邻接，从而保证它们确实属于不同连通分量；允许对角接触以保留放置自由度。
#     - 主组件使用同一行或同一列上的多个'+'加短桥连接，既稳定连通，又不会意外生成额外'+'中心。
#   visual_observability:
#     - 样例中会同时出现“形状一样但命运不同”的'+'：大块中的被染色，小块中的不染色。
#     - 这让人类能从2-3组样例归纳出：不是所有'+'都变色，而是只处理最大黄色团块里的'+'。
#   visual_cleanliness:
#     - 主组件是一条规则的链状结构，小组件是独立的小十字或短折线，区域层次清楚。
#     - 非'+'干扰物只用短线和L形，避免同一区域堆叠过多纹理。
#   color_inference:
#     - 颜色角色不变：2是背景、4是待检测前景、8是输出标记。
#     - 颜色变化完全由“是否在最大前景连通块内且构成'+'”决定，而不是由颜色id排序等不可见因素决定。
#   generator_guarantees_for_new_main:
#     - 至少一个'+'位于最大连通块中，因此输出一定出现青色。
#     - 至少一个'+'位于非最大连通块中，因此同一输入上新旧规则的输出一定不同。
#     - 最大连通块大小严格大于其他每个连通块，避免 solver 依赖未说明的并列决策。


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
    comps = _connected_components(grid == fg)
    keep = np.zeros_like(grid, dtype=bool)
    if not comps:
        return keep
    largest = max(comps, key=len)
    for r, c in largest:
        keep[r, c] = True
    return keep


def _plus_centers(grid: np.ndarray, fg: int, allowed: np.ndarray | None = None) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if allowed is not None and not allowed[r, c]:
                continue
            ok = True
            for dr, dc in ORTHO:
                nr, nc = r + dr, c + dc
                if int(grid[nr, nc]) != fg:
                    ok = False
                    break
                if allowed is not None and not allowed[nr, nc]:
                    ok = False
                    break
            if ok:
                centers.append((r, c))
    return centers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect '+' motifs only inside the largest foreground component and recolor them to teal.

    Input structure:
    - bg is the most common color.
    - One foreground color forms several disconnected 4-connected components; the largest one is the only component that matters.

    Rule:
    - Find the unique largest 4-connected foreground component.
    - Within that component, any foreground pixel whose 4 orthogonal neighbors are also foreground defines a '+'.
    - Recolor that center and its 4 orthogonal neighbors to teal (8).
    - Ignore '+' motifs in all smaller foreground components; leave every other pixel unchanged.

    Key insight:
    - The local '+' test is unchanged, but it must be gated by a global component-selection step first, otherwise smaller disconnected copies of the same motif would also be recolored.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    keep = _largest_component_mask(g, fg)
    for r, c in _plus_centers(g, fg, keep):
        for dr, dc in PLUS5:
            out[r + dr, c + dc] = 8
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _component_size(cells: set[tuple[int, int]]) -> int:
    return len(cells)


def _component_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _add_plus(cells: set[tuple[int, int]], r: int, c: int) -> None:
    for dr, dc in PLUS5:
        cells.add((r + dr, c + dc))


def _make_main_component(rng: np.random.Generator) -> set[tuple[int, int]]:
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
    for r, c in centers:
        _add_plus(cells, r, c)

    for (r1, c1), (r2, c2) in zip(centers, centers[1:]):
        if r1 == r2:
            for cc in range(c1 + 2, c2 - 1):
                cells.add((r1, cc))
        else:
            for rr in range(r1 + 2, r2 - 1):
                cells.add((rr, c1))

    for idx, (r, c) in enumerate(centers):
        if orientation == "h":
            dr, dc = (1, 0) if rng.random() < 0.5 else (-1, 0)
            for step in range(2, 2 + int(rng.integers(1, 3))):
                cells.add((r + dr * step, c))
            if idx == 0:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r, c - step))
            if idx == len(centers) - 1:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r, c + step))
        else:
            dr, dc = (0, 1) if rng.random() < 0.5 else (0, -1)
            for step in range(2, 2 + int(rng.integers(1, 3))):
                cells.add((r, c + dc * step))
            if idx == 0:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r - step, c))
            if idx == len(centers) - 1:
                for step in range(2, 2 + int(rng.integers(0, 2))):
                    cells.add((r + step, c))

    return _normalize(cells)


def _make_small_plus_component(rng: np.random.Generator) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    _add_plus(cells, 0, 0)
    num_dirs = int(rng.integers(1, 3))
    chosen = rng.choice(len(ORTHO), size=num_dirs, replace=False)
    for idx in chosen:
        dr, dc = ORTHO[int(idx)]
        for step in range(2, 2 + int(rng.integers(1, 2))):
            cells.add((dr * step, dc * step))
    return _normalize(cells)


def _make_distractor_component(rng: np.random.Generator) -> set[tuple[int, int]]:
    shape = str(rng.choice(["hline", "vline", "elbow"]))
    cells: set[tuple[int, int]] = set()
    if shape == "hline":
        length = int(rng.integers(2, 5))
        for c in range(length):
            cells.add((0, c))
    elif shape == "vline":
        length = int(rng.integers(2, 5))
        for r in range(length):
            cells.add((r, 0))
    else:
        arm_r = int(rng.integers(2, 4))
        arm_c = int(rng.integers(2, 4))
        for r in range(arm_r):
            cells.add((r, 0))
        for c in range(arm_c):
            cells.add((0, c))
    return _normalize(cells)


def _fits_component(
    occupied: np.ndarray, cells: set[tuple[int, int]], top: int, left: int
) -> bool:
    H, W = occupied.shape
    for r, c in cells:
        rr, cc = top + r, left + c
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if occupied[rr, cc]:
            return False
        for dr, dc in ORTHO:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < H and 0 <= nc < W and occupied[nr, nc]:
                return False
    return True


def _place_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    cells: set[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
    max_tries: int = 300,
) -> bool:
    H, W = grid.shape
    h, w = _component_dims(cells)
    if h > H or w > W:
        return False
    for _ in range(max_tries):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        if not _fits_component(occupied, cells, top, left):
            continue
        for r, c in cells:
            rr, cc = top + r, left + c
            grid[rr, cc] = fg
            occupied[rr, cc] = True
        return True
    return False


def _foreground_count(grid: np.ndarray, bg: int) -> int:
    return int(np.sum(grid != bg))


def _global_main_teal_count(grid: np.ndarray, fg: int) -> tuple[int, int]:
    largest = _largest_component_mask(grid, fg)
    return len(_plus_centers(grid, fg, largest)), len(_plus_centers(grid, fg, None))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(12, 19))
    W = int(rng.integers(12, 21))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    main_cells = _make_main_component(rng)
    if not _place_component(grid, occupied, main_cells, fg, rng):
        return None

    small_plus_count = int(rng.integers(1, 4))
    for _ in range(small_plus_count):
        if not _place_component(grid, occupied, _make_small_plus_component(rng), fg, rng):
            return None

    target_density = float(rng.uniform(0.24, 0.38))
    target_fg = int(H * W * target_density)

    distractor_budget = int(rng.integers(5, 10))
    for _ in range(distractor_budget):
        if _foreground_count(grid, bg) >= target_fg:
            break
        _place_component(grid, occupied, _make_distractor_component(rng), fg, rng, max_tries=120)

    while _foreground_count(grid, bg) < target_fg:
        if not _place_component(grid, occupied, _make_distractor_component(rng), fg, rng, max_tries=80):
            break

    if _foreground_count(grid, bg) >= (H * W) // 2:
        return None

    comps = sorted(_connected_components(grid == fg), key=len, reverse=True)
    if len(comps) < 2:
        return None
    if len(comps[0]) <= len(comps[1]):
        return None
    if len(comps[0]) != _component_size(main_cells):
        return None

    main_plus_count, all_plus_count = _global_main_teal_count(grid, fg)
    if main_plus_count < 1:
        return None
    if all_plus_count <= main_plus_count:
        return None

    out = main(grid)
    if not np.any(out == 8):
        return None

    # Enforce that at least one smaller-component '+' remains yellow after solving.
    global_centers = _plus_centers(grid, fg, None)
    largest_mask = _largest_component_mask(grid, fg)
    ignored = [(r, c) for r, c in global_centers if not largest_mask[r, c]]
    if not ignored:
        return None
    if not any(int(out[r, c]) == fg for r, c in ignored):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
