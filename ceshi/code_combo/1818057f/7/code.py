# Combo variant of 1818057f (index 7)
# combo_spec: D9/0+D11/0
# Name: 最大连通分量内传播式重着色

from __future__ import annotations

from collections import deque
import numpy as np

from arc_utils import get_bg


# Design analysis
# ---------------
# concretization_and_trace:
#   - 输入仍然只有背景色 2 和单一前景色 4，但前景会被拆成多个互相 4-断开的连通分量。
#   - 原题做法是：对整张图逐点检查，只要某个前景像素上下左右也都是前景，就立刻把这个中心及四臂共 5 格全部染成青色。
#   - 本融合规则先做全局筛选：只保留“唯一最大”的前景 4 连通分量。随后只在这个分量里找 '+' 中心。
#   - 找到中心后并不一次性染 5 格，而是先只把中心当作种子；再做固定 3 轮同步传播，每轮把该最大分量内与现有青色正交相邻的前景也染青。
#   - 具体小例子：若最大分量里有一个 '+'，其右臂继续向右延伸 5 格，而旁边还有一个较小独立组件也含有 '+'：
#     原题会把两个组件中的所有局部 '+' 都染成 5 格十字；
#     本题只会处理最大分量中的那个 '+'，且青色会从中心向外沿连通路径继续扩到距离 2、3 的格子，距离 4 及更远的格子仍保持黄色。
#   - 因此差异是双重的：不是“所有 '+' 都处理”，也不是“处理了就只染十字 5 格”，而是“只在最大分量里，从中心向外限步传播”。
# complexity_calibration:
#   - 若传播只做 1 轮，就退化成“最大分量版原题”；传播轮数太大又会把整个最大分量全染青，失去“限步”可见性。
#   - 这里固定为 3 轮传播，连同种子轮一共 4 个阶段：比原题明显更远，但仍会留下最大分量中的黄色尾端。
#   - 生成器强制同时出现三类证据：最大分量内有距离 2..3 的可达格、最大分量内有距离 >=4 的未达格、较小分量里也有 '+' 但完全不参与染色。
# implicit_assumptions_review:
#   - `out = g.copy()` 仍然安全，因为背景、较小分量、以及最大分量中传播未到达的前景都应保持原色。
#   - `fg = colors[0]` 依赖“输入只有一个非背景色”，这一点由生成器继续保证。
#   - 原题的局部 '+' 判定必须改成“先算最大分量掩码，再在掩码内部找中心”；否则较小分量中的同形结构会被误处理。
#   - '+' 中心检测必须只基于原图 `g`，不能用传播中的青色结果反推新种子，否则会凭空制造额外中心。
#   - 传播必须按轮同步更新，若一轮内原地连锁赋值，就会让扫描顺序影响传播深度，违背 D9 的多步规则。
#   - solver 本身对“最大分量并列”没有额外规则，因此生成器必须保证最大分量唯一；solver 只用 `max` 兜底。
# placement_legality:
#   - 所有组件先在局部坐标里造型，再平移到全局网格。
#   - 组件之间禁止 4 邻接，从而保证连通分量互不合并；允许对角接触以保留放置自由度。
#   - 主组件由一串间隔 4~5 的 '+' 加桥接线段组成，并在端点强制接一条更长尾巴，既保证存在 '+' 种子，也保证存在传播见证路径。
# visual_observability:
#   - 解题者能直接比较“同样是 '+'，为什么大块里的会变、旁边小块里的不变”，从而看出最大分量筛选。
#   - 同时，大块内部的青色又不是只停留在 5 格十字，而是沿着细线继续前进几步后停止，传播深度也能从 2~3 个样例里稳定看见。
# visual_cleanliness:
#   - 主组件保持细线和链状结构，较小组件只用小十字加短臂，干扰块只用短线或小折角，避免多个模式在同一区域互相遮蔽。
# change_inferability:
#   - 颜色角色始终固定：2 是背景，4 是待判断前景，8 是输出标记。
#   - 颜色变化完全由两条可见结构条件决定：是否属于最大前景分量，以及与该分量内 '+' 中心的图距离是否不超过固定传播深度。
# generator_contract_for_main:
#   - 输入始终只有一个前景色。
#   - 最大前景分量唯一，且其中至少有一个 '+' 中心。
#   - 至少一个较小分量也含有 '+'，确保“只处理最大分量”在样例中可见。
#   - 最大分量内至少有一个格子会因为传播而变青但不属于经典 5 格十字，也至少有一个格子因超过传播深度而保持黄色。

TEAL = 8
BG = 2
FG = 4
PROPAGATION_STEPS = 3
ORTHO = ((1, 0), (-1, 0), (0, 1), (0, -1))
PLUS5 = ((0, 0),) + ORTHO


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(H):
        for c in range(W):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((int(cr), int(cc)))
                for dr, dc in ORTHO:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if 0 <= nr < H and 0 <= nc < W and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _largest_component_mask(grid: np.ndarray, fg: int) -> np.ndarray:
    keep = np.zeros_like(grid, dtype=bool)
    comps = _connected_components(grid == int(fg))
    if not comps:
        return keep
    largest = max(comps, key=len)
    for r, c in largest:
        keep[int(r), int(c)] = True
    return keep


def _plus_centers(grid: np.ndarray, fg: int, allowed: np.ndarray | None = None) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != int(fg):
                continue
            if allowed is not None and not bool(allowed[r, c]):
                continue
            ok = True
            for dr, dc in ORTHO:
                nr, nc = int(r + dr), int(c + dc)
                if int(grid[nr, nc]) != int(fg):
                    ok = False
                    break
                if allowed is not None and not bool(allowed[nr, nc]):
                    ok = False
                    break
            if ok:
                centers.append((int(r), int(c)))
    return centers


def _propagation_mask(grid: np.ndarray, fg: int, allowed: np.ndarray, steps: int) -> np.ndarray:
    teal = np.zeros_like(grid, dtype=bool)
    fg_allowed = (grid == int(fg)) & allowed

    for r, c in _plus_centers(grid, fg, allowed):
        teal[int(r), int(c)] = True

    for _ in range(int(max(0, steps))):
        grown = teal.copy()
        H, W = grid.shape
        for r in range(H):
            for c in range(W):
                if not bool(fg_allowed[r, c]) or bool(teal[r, c]):
                    continue
                if (
                    (r > 0 and bool(teal[r - 1, c]))
                    or (r + 1 < H and bool(teal[r + 1, c]))
                    or (c > 0 and bool(teal[r, c - 1]))
                    or (c + 1 < W and bool(teal[r, c + 1]))
                ):
                    grown[r, c] = True
        if np.array_equal(grown, teal):
            break
        teal = grown
    return teal


def _classic_largest_mask(grid: np.ndarray, fg: int, allowed: np.ndarray) -> np.ndarray:
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in _plus_centers(grid, fg, allowed):
        for dr, dc in PLUS5:
            mask[int(r + dr), int(c + dc)] = True
    return mask


def _distance_from_largest_centers(grid: np.ndarray, fg: int, allowed: np.ndarray) -> np.ndarray:
    H, W = grid.shape
    dist = np.full((H, W), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in _plus_centers(grid, fg, allowed):
        dist[int(r), int(c)] = 0
        q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in ORTHO:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if not bool(allowed[nr, nc]) or int(grid[nr, nc]) != int(fg) or int(dist[nr, nc]) != -1:
                continue
            dist[nr, nc] = int(dist[r, c] + 1)
            q.append((int(nr), int(nc)))
    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor only the largest foreground component, and let teal spread outward from '+' centers.

    Input structure:
    - The most common color is the background.
    - Exactly one other color forms several disconnected thin components; one of them is strictly larger than the rest.

    Rule:
    - Find the unique largest 4-connected foreground component.
    - Inside that component, mark every '+' center, meaning a foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Use those centers as teal seeds, then perform 3 synchronous propagation steps restricted to that same largest component: any remaining foreground cell orthogonally adjacent to teal also becomes teal.
    - Smaller foreground components are ignored completely, and every non-reached cell keeps its original color.

    Key insight:
    - The local '+' test is only the starting point. The actual output is determined by a global component-selection step followed by bounded graph-distance growth inside that chosen component.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    keep = _largest_component_mask(g, fg)
    teal = _propagation_mask(g, fg, keep, PROPAGATION_STEPS)
    out[teal] = TEAL
    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return {(int(r - min_r), int(c - min_c)) for r, c in cells}


def _component_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _add_plus(cells: set[tuple[int, int]], r: int, c: int) -> None:
    for dr, dc in PLUS5:
        cells.add((int(r + dr), int(c + dc)))


def _make_main_component(rng: np.random.Generator) -> set[tuple[int, int]]:
    orientation = str(rng.choice(["h", "v"]))
    count = int(rng.integers(2, 5))
    centers: list[tuple[int, int]] = [(0, 0)]

    for _ in range(1, count):
        gap = int(rng.integers(4, 6))
        pr, pc = centers[-1]
        if orientation == "h":
            centers.append((int(pr), int(pc + gap)))
        else:
            centers.append((int(pr + gap), int(pc)))

    cells: set[tuple[int, int]] = set()
    for r, c in centers:
        _add_plus(cells, r, c)

    for (r1, c1), (r2, c2) in zip(centers, centers[1:]):
        if r1 == r2:
            for cc in range(int(c1 + 2), int(c2 - 1)):
                cells.add((int(r1), int(cc)))
        else:
            for rr in range(int(r1 + 2), int(r2 - 1)):
                cells.add((int(rr), int(c1)))

    if orientation == "h":
        endpoint = centers[0] if rng.random() < 0.5 else centers[-1]
        direction = (0, -1) if endpoint == centers[0] else (0, 1)
    else:
        endpoint = centers[0] if rng.random() < 0.5 else centers[-1]
        direction = (-1, 0) if endpoint == centers[0] else (1, 0)

    dr, dc = direction
    tail_end = int(rng.integers(PROPAGATION_STEPS + 2, PROPAGATION_STEPS + 5))
    for step in range(2, tail_end + 1):
        cells.add((int(endpoint[0] + dr * step), int(endpoint[1] + dc * step)))

    for r, c in centers:
        if rng.random() < 0.75:
            if orientation == "h":
                side_dr = -1 if rng.random() < 0.5 else 1
                side_len = int(rng.integers(1, 3))
                for step in range(2, 2 + side_len):
                    cells.add((int(r + side_dr * step), int(c)))
            else:
                side_dc = -1 if rng.random() < 0.5 else 1
                side_len = int(rng.integers(1, 3))
                for step in range(2, 2 + side_len):
                    cells.add((int(r), int(c + side_dc * step)))

    return _normalize(cells)


def _make_small_plus_component(rng: np.random.Generator) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    _add_plus(cells, 0, 0)
    extra_dirs = rng.choice(len(ORTHO), size=int(rng.integers(1, 3)), replace=False)
    for idx in extra_dirs:
        dr, dc = ORTHO[int(idx)]
        ext_len = int(rng.integers(1, 3))
        for step in range(2, 2 + ext_len):
            cells.add((int(dr * step), int(dc * step)))
    return _normalize(cells)


DISTRACTOR_SHAPES = [
    {(0, 0)},
    {(0, 0), (0, 1)},
    {(0, 0), (1, 0)},
    {(0, 0), (0, 1), (0, 2)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2)},
]


def _make_distractor_component(rng: np.random.Generator) -> set[tuple[int, int]]:
    shape = DISTRACTOR_SHAPES[int(rng.integers(0, len(DISTRACTOR_SHAPES)))]
    return set((int(r), int(c)) for r, c in shape)


def _fits_component(occupied: np.ndarray, cells: set[tuple[int, int]], top: int, left: int) -> bool:
    H, W = occupied.shape
    for r, c in cells:
        rr, cc = int(top + r), int(left + c)
        if not (0 <= rr < H and 0 <= cc < W):
            return False
        if bool(occupied[rr, cc]):
            return False
        for dr, dc in ORTHO:
            nr, nc = int(rr + dr), int(cc + dc)
            if 0 <= nr < H and 0 <= nc < W and bool(occupied[nr, nc]):
                return False
    return True


def _place_component(
    grid: np.ndarray,
    occupied: np.ndarray,
    cells: set[tuple[int, int]],
    fg: int,
    rng: np.random.Generator,
    max_tries: int = 400,
) -> bool:
    H, W = grid.shape
    h, w = _component_dims(cells)
    if h > H or w > W:
        return False
    for _ in range(int(max_tries)):
        top = int(rng.integers(0, H - h + 1))
        left = int(rng.integers(0, W - w + 1))
        if not _fits_component(occupied, cells, top, left):
            continue
        for r, c in cells:
            rr, cc = int(top + r), int(left + c)
            grid[rr, cc] = int(fg)
            occupied[rr, cc] = True
        return True
    return False


def _foreground_count(grid: np.ndarray, bg: int) -> int:
    return int(np.sum(grid != int(bg)))


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(12, 21))
    W = int(rng.integers(12, 21))
    grid = np.full((H, W), BG, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    main_cells = _make_main_component(rng)
    if not _place_component(grid, occupied, main_cells, FG, rng):
        return None

    small_plus_count = int(rng.integers(1, 4))
    for _ in range(small_plus_count):
        if not _place_component(grid, occupied, _make_small_plus_component(rng), FG, rng, max_tries=500):
            return None

    target_density = float(rng.uniform(0.24, 0.34))
    target_fg = int(H * W * target_density)

    filler_attempts = 0
    while _foreground_count(grid, BG) < target_fg and filler_attempts < H * W * 8:
        filler_attempts += 1
        _place_component(grid, occupied, _make_distractor_component(rng), FG, rng, max_tries=80)

    if _foreground_count(grid, BG) >= (H * W) // 2:
        return None

    comps = sorted(_connected_components(grid == FG), key=len, reverse=True)
    if len(comps) < 2:
        return None
    if len(comps[0]) <= len(comps[1]):
        return None
    if len(comps[0]) != len(main_cells):
        return None

    largest = _largest_component_mask(grid, FG)
    centers_main = _plus_centers(grid, FG, largest)
    centers_all = _plus_centers(grid, FG, None)
    if not centers_main:
        return None
    if len(centers_all) <= len(centers_main):
        return None

    dist = _distance_from_largest_centers(grid, FG, largest)
    if not np.any((dist >= 2) & (dist <= PROPAGATION_STEPS)):
        return None
    if not np.any(dist > PROPAGATION_STEPS):
        return None

    out = main(grid)
    if not np.any(out == TEAL):
        return None

    variant_mask = out == TEAL
    classic_mask = _classic_largest_mask(grid, FG, largest)
    if int(np.sum(variant_mask & largest & ~classic_mask)) < 2:
        return None
    if not np.any(largest & (grid == FG) & ~variant_mask):
        return None

    ignored_centers = [(r, c) for r, c in centers_all if not bool(largest[r, c])]
    if not ignored_centers:
        return None
    if not any(int(out[r, c]) == FG for r, c in ignored_centers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 600 attempts")
