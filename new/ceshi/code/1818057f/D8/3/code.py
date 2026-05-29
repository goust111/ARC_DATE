# Variant of 1818057f
# Dimension: D8 — 干扰项的结构和复杂度 (meta)
# Change: 散布的小型前景像素组(1-4格)作为干扰项，生成时确保不会形成新的'+'图案... → 干扰项本身形成另一种需要不同变换的图案

from __future__ import annotations

import numpy as np

from arc_utils import get_bg


# Design analysis
# ----------------
# a) 具体化与行为追踪
#    - 保留原任务的主目标：在单一前景色里寻找 5 格薄 '+'，输出时把这 5 格改成 teal(8)。
#    - D8 变体新增第二类局部图案：前景色 2x2 方块。它们属于原来“干扰项”的位置层级，
#      但现在不再只是无意义杂点，而是另一条规则的目标，输出时改成 orange(7)。
#    - 为了让第二类图案肉眼可辨，生成器会放置若干独立 2x2 方块；其中一部分会额外带一个
#      单格侧向小尾巴，使它看起来像“嵌在干扰区域中的方块”，而不是纯粹的标记。
#    - 小例子追踪：
#      原逻辑：一个 2x2 方块只是普通前景，solver 完全忽略，只把 '+' 染成 8。
#      新逻辑：同样的输入若含 2x2 方块，则 '+' 的 5 格仍染成 8，而该 2x2 的 4 格改染成 7。
#      差异点在“局部匹配集合”扩大为两类，且必须在同一张输入上同时执行两条重着色规则。
# b) 复杂度校准
#    - 如果把第二图案做成“任意矩形”或“任意 4 连通块”，规则会太松，肉眼难归纳。
#      2x2 是低歧义、可重复出现、又明显不同于 '+' 的局部结构。
#    - 如果允许任意大实心块，则会产生多个重叠 2x2 子块，导致规则边界含混。
#      因此生成器显式阻止意外的新 2x2，只保留刻意放置的方块。
# c) 隐式假设审查（原 main 的关键前提）
#    - `colors[0]` 依赖输入仍只有一种非背景色。该前提在变体输入里继续成立；新增颜色只出现在输出。
#    - 原 solver 一边扫描一边写 `out`，因为只有一种规则且写成 8 不会影响尚未检查的 `g`。
#      变体若仍边扫描边写，可能出现第二规则覆盖第一规则的结果，或顺序依赖。因此新 main 先在原图 `g`
#      上冻结两类匹配，再统一写回 `out`。
#    - 原 solver 默认所有需要改色的目标都使用同一种颜色 8。变体不再成立，所以必须区分两类 footprint。
# d) 放置合法性
#    - '+' 与 2x2 方块在生成时使用 1 格 halo 隔开，避免重叠、越界、以及相互拼出额外模式。
#    - 后续补充前景密度时，每次落子都检查“不会新造 '+'、不会新造 2x2”，确保 intended motifs 唯一。
# e) 视觉可观察性
#    - 解题者在 2-4 组样例里应能看到：十字形都变 teal，方块都变 orange，其余黄块不变。
#      规则完全由可见结构决定，不依赖颜色 id 排序、生成顺序或隐藏属性。
# f) 可视化整洁度
#    - 目标图案之间保留空隙，方块附近不塞随机噪声；这样十字与方块都能直接被眼睛分组识别。
# g) 颜色变化可推断性
#    - 输出新增只有两种功能色：8 对应 '+'，7 对应 2x2 方块。颜色种类少，且和明确形状绑定。
#
# main() 的新假设审查
#    - 假设 1：输入仍是“背景 + 单前景色”。generate_input() 保证这一点。
#    - 假设 2：有意放置的 '+' 与 2x2 不共享像素。generate_input() 用 halo 和显式检测保证。
#    - 假设 3：意外的大实心块不会出现，否则会产生额外 2x2 匹配。generate_input() 的落子检查保证不会退化。


PLUS_COLOR = 8
BLOCK_COLOR = 7
ORTHO = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def _plus_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)]


def _block_cells(r: int, c: int) -> list[tuple[int, int]]:
    return [(r, c), (r + 1, c), (r, c + 1), (r + 1, c + 1)]


def _find_plus_footprints(grid: np.ndarray, fg: int) -> list[tuple[tuple[int, int], ...]]:
    H, W = grid.shape
    footprints: list[tuple[tuple[int, int], ...]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO):
                footprints.append(tuple(_plus_cells(r, c)))
    return footprints


def _find_block_footprints(
    grid: np.ndarray, fg: int, blocked_cells: set[tuple[int, int]] | None = None
) -> list[tuple[tuple[int, int], ...]]:
    H, W = grid.shape
    blocked = blocked_cells or set()
    footprints: list[tuple[tuple[int, int], ...]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for r in range(H - 1):
        for c in range(W - 1):
            cells = tuple(_block_cells(r, c))
            if any((rr, cc) in blocked for rr, cc in cells):
                continue
            if all(int(grid[rr, cc]) == fg for rr, cc in cells):
                key = frozenset(cells)
                if key not in seen:
                    seen.add(key)
                    footprints.append(cells)
    return footprints


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect two local motifs in a single foreground color and recolor each motif family differently.

    Input structure:
    - bg is still the most common color.
    - One foreground color forms mixed shapes on the bg: some regions contain thin '+' crosses, and some distractor regions contain compact 2x2 squares.

    Rule:
    - Any foreground pixel whose 4 orthogonal neighbors are also foreground defines a '+' motif; recolor that center and its 4 neighbors to teal (8).
    - Any all-foreground 2x2 block whose 4 cells do not belong to a detected '+' footprint is a square motif; recolor those 4 cells to orange (7).
    - Every other cell stays unchanged.

    Key insight:
    - This is now a multi-rule local-pattern task. The safe way to solve it is to detect both motif families on the original foreground mask first, then apply the two recolorings from those frozen matches so processing order cannot create false interactions.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    plus_footprints = _find_plus_footprints(g, fg)
    plus_cells = {cell for footprint in plus_footprints for cell in footprint}
    block_footprints = _find_block_footprints(g, fg, plus_cells)

    for footprint in plus_footprints:
        for r, c in footprint:
            out[r, c] = PLUS_COLOR
    for footprint in block_footprints:
        for r, c in footprint:
            out[r, c] = BLOCK_COLOR
    return out


def _halo(cells: list[tuple[int, int]], H: int, W: int, margin: int = 1) -> set[tuple[int, int]]:
    halo: set[tuple[int, int]] = set()
    for r, c in cells:
        for nr in range(r - margin, r + margin + 1):
            for nc in range(c - margin, c + margin + 1):
                if 0 <= nr < H and 0 <= nc < W:
                    halo.add((nr, nc))
    return halo


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        ok = True
        for dr, dc in [(0, 0), *ORTHO]:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                ok = False
                break
        if ok:
            return True
    return False


def _would_create_block(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    H, W = grid.shape
    for top in (r - 1, r):
        for left in (c - 1, c):
            if top < 0 or left < 0 or top >= H - 1 or left >= W - 1:
                continue
            ok = True
            for rr, cc in _block_cells(top, left):
                val = fg if (rr, cc) == (r, c) else int(grid[rr, cc])
                if val != fg:
                    ok = False
                    break
            if ok:
                return True
    return False


def _safe_to_add(grid: np.ndarray, r: int, c: int, bg: int, fg: int) -> bool:
    if int(grid[r, c]) != bg:
        return False
    return not _would_create_plus(grid, r, c, fg) and not _would_create_block(grid, r, c, fg)


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = color


def _place_plus_component(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    fg: int,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], list[tuple[int, int]]] | None:
    H, W = grid.shape
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(160):
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        core = _plus_cells(cr, cc)
        if any((r, c) in reserved for r, c in core):
            continue

        possible_dirs: list[tuple[int, int, int]] = []
        for dr, dc in dirs:
            max_len = 0
            tip_r, tip_c = cr + dr, cc + dc
            for step in range(1, 4):
                nr, nc = tip_r + dr * step, tip_c + dc * step
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if (nr, nc) in reserved:
                    break
                max_len = step
            if max_len > 0:
                possible_dirs.append((dr, dc, max_len))
        if not possible_dirs:
            continue

        n_ext = int(rng.integers(1, min(2, len(possible_dirs)) + 1))
        chosen_idx = rng.choice(len(possible_dirs), size=n_ext, replace=False)
        attachments: list[tuple[int, int]] = []
        for idx in np.atleast_1d(chosen_idx):
            dr, dc, max_len = possible_dirs[int(idx)]
            tip_r, tip_c = cr + dr, cc + dc
            ext_len = int(rng.integers(1, max_len + 1))
            for step in range(1, ext_len + 1):
                attachments.append((tip_r + dr * step, tip_c + dc * step))

        component = core + attachments
        if any((r, c) in reserved for r, c in component):
            continue
        return (cr, cc), tuple(core), attachments
    return None


def _place_block_component(
    rng: np.random.Generator,
    grid: np.ndarray,
    reserved: set[tuple[int, int]],
    fg: int,
) -> tuple[tuple[int, int], tuple[tuple[int, int], ...], list[tuple[int, int]]] | None:
    H, W = grid.shape
    tail_offsets = [(-1, 0), (-1, 1), (2, 0), (2, 1), (0, -1), (1, -1), (0, 2), (1, 2)]
    for _ in range(160):
        r = int(rng.integers(1, H - 2))
        c = int(rng.integers(1, W - 2))
        core = _block_cells(r, c)
        if any((rr, cc) in reserved for rr, cc in core):
            continue

        attachments: list[tuple[int, int]] = []
        if rng.random() < 0.7:
            rng.shuffle(tail_offsets)
            for dr, dc in tail_offsets:
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in reserved:
                    attachments = [(nr, nc)]
                    break

        component = core + attachments
        if any((rr, cc) in reserved for rr, cc in component):
            continue
        return (r, c), tuple(core), attachments
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)
    reserved: set[tuple[int, int]] = set()

    area = H * W
    n_plus = int(rng.integers(1, min(6, max(2, area // 40)) + 1))
    n_blocks = int(rng.integers(1, min(5, max(2, area // 55 + 1)) + 1))

    plus_footprints: list[tuple[tuple[int, int], ...]] = []
    block_footprints: list[tuple[tuple[int, int], ...]] = []

    for _ in range(n_plus):
        placed = _place_plus_component(rng, grid, reserved, fg)
        if placed is None:
            return None
        _, core, attachments = placed
        component = list(core) + attachments
        _paint_cells(grid, component, fg)
        reserved |= _halo(component, H, W, margin=1)
        plus_footprints.append(core)

    for _ in range(n_blocks):
        placed = _place_block_component(rng, grid, reserved, fg)
        if placed is None:
            return None
        _, core, attachments = placed
        component = list(core) + attachments
        _paint_cells(grid, component, fg)
        reserved |= _halo(component, H, W, margin=1)
        block_footprints.append(core)

    total = H * W
    target_density = float(rng.uniform(0.24, 0.38))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != bg))

    candidates = [(r, c) for r in range(H) for c in range(W) if (r, c) not in reserved]
    rng.shuffle(candidates)
    for r, c in candidates:
        if current_fg >= target_fg:
            break
        if _safe_to_add(grid, r, c, bg, fg):
            grid[r, c] = fg
            current_fg += 1

    if int(np.sum(grid != bg)) >= total // 2:
        return None

    expected = grid.copy()
    for footprint in plus_footprints:
        for r, c in footprint:
            expected[r, c] = PLUS_COLOR
    for footprint in block_footprints:
        for r, c in footprint:
            expected[r, c] = BLOCK_COLOR

    if not np.any(expected == PLUS_COLOR) or not np.any(expected == BLOCK_COLOR):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
