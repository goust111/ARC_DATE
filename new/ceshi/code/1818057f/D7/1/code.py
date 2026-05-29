# Variant of 1818057f
# Dimension: D7 — 输出网格形状 (grid_shape)
# Change: 输出网格与输入网格尺寸完全相同(10x12到22x22)，原位变换... → 每个'+'单独裁剪并排列

from __future__ import annotations

import numpy as np
from arc_utils import get_bg


# Design analysis
# ----------------
# spec:
#   concretization:
#     input: |
#       仍然是红色背景(2)上的单一黄色前景(4)。若某个黄色像素的四个正交邻居也都是黄色，
#       它就是一个 plus 中心。每个真实 plus 都会带至少一条额外延伸臂，因此输入里看到的
#       不是孤立 5 像素十字，而是嵌在更大黄色连通结构中的局部十字。
#     output: |
#       不再返回整幅原图。改为把每个 plus 的 5x5 局部窗口单独切下；窗口中该 plus 的
#       5 个像素改成青色(8)，其他像素保持输入颜色；再按输入中的阅读顺序(先行后列)
#       横向排成一行，相邻窗口之间保留 1 列背景分隔。输出高度固定为 5，宽度随 plus 数量变化。
#   behavior_trace:
#     original: |
#       原题在原位扫描整张图，对每个满足条件的中心直接把十字 5 像素染成青色，所以输出形状
#       与输入完全相同。
#     variant: |
#       现在仍先扫描出所有 plus 中心，但之后不保留原布局：例如两个中心分别位于 (4,5) 和
#       (9,10) 时，原题输出仍是 HxW；变体则得到两个 5x5 tile，把各自中心十字染青后并排，
#       输出变成 5x11。
#     difference: |
#       差异发生在“命中 plus 之后”的阶段：原题是原位重着色；变体是分割 -> 居中裁剪 ->
#       横向 packed row。
#   complexity_calibration:
#     decisions:
#       - 固定使用 5x5 centered crop，而不是可变最小外接框，避免再引入额外对齐规则。
#       - 保留原始“单前景色 + 干扰像素 + 嵌入式 plus”检测难度，避免退化成直接读出孤立对象。
#       - plus 数量限制在 2..4，使 packed row 肉眼可见，同时输出不会长到难以阅读。
#   implicit_assumptions_review:
#     original_main_dependencies:
#       - `out = g.copy()` 依赖“输出与输入同形”；变体不成立，必须新建 packed 输出。
#       - 扫描范围 `1..H-2` 只保证检测所需的 1 格边距；变体裁 5x5 时需要额外处理边缘。
#       - 直接原位赋值 `out[r+dr, c+dc] = 8` 依赖“结果仍放回原图”；变体应写入 tile。
#       - 取 `colors[0]` 依赖“只有一种非背景色”；此约束仍保留。
#     new_handling:
#       - 求解器用背景填充式 5x5 提取函数，因此即使非常规输入把 plus 放得更靠边也不会崩溃。
#       - 生成器仍把中心放在距边界至少 2 格处，让 5x5 裁剪完整且视觉更稳定。
#   placement_legality:
#     guarantees:
#       - 中心只从 `r in [2, H-3], c in [2, W-3]` 中采样，保证 5x5 局部窗口完整落在输入内。
#       - 任意两中心的 L1 距离至少为 4，沿用原题的“plus footprint 不共享像素”约束。
#       - 延伸臂和干扰像素的加入都经过 `_would_create_plus` 检查，避免生成额外未计划的 plus。
#   observability:
#     human_cues:
#       - 输入中仍能看见多个局部十字埋在杂乱黄图案里。
#       - 输出高度统一为 5，明显不是原位结果，而是把多个局部片段抽出后按顺序并排。
#       - 每个输出 tile 里只有命中的十字被染青，说明不是简单裁剪，而是“先识别 plus 再裁剪”。
#   cleanliness:
#     choices:
#       - 输出只保留 plus 周边 5x5，上下不再混入无关大区域，减少 packed 后的视觉噪声。
#       - tile 间保留单列背景分隔，避免多个裁片黏连成新的形状。
#   color_inference:
#     rule: |
#       颜色变化仍只有一种：被识别到的 plus 5 像素统一变为青色。背景和其它黄色像素不变，
#       所以颜色规则可以从少量样例稳定归纳。
#   main_generator_contract:
#     solver_assumptions:
#       - 非背景色只有一种，且有效样例里至少有 2 个 plus。
#       - packed 顺序采用扫描顺序，因此 generator 也必须按同样定义验证中心列表。
#     generator_support:
#       - 生成器显式复用同一 plus-center 检测器，并检查“检测到的中心集合 == 计划中心集合”。


CROSS_OFFSETS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
ORTHO_OFFSETS = CROSS_OFFSETS[1:]
BG = 2
FG = 4
TEAL = 8
TILE_SIZE = 5
CROP_RADIUS = 2
TILE_GAP = 1


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    """Return plus centers in row-major order."""
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in ORTHO_OFFSETS):
                centers.append((int(r), int(c)))
    return centers


def _extract_tile(grid: np.ndarray, r: int, c: int, bg: int) -> np.ndarray:
    """Extract a centered 5x5 tile, padding with bg if the window hits an edge."""
    H, W = grid.shape
    tile = np.full((TILE_SIZE, TILE_SIZE), int(bg), dtype=int)
    for tr in range(TILE_SIZE):
        rr = int(r - CROP_RADIUS + tr)
        if not (0 <= rr < H):
            continue
        for tc in range(TILE_SIZE):
            cc = int(c - CROP_RADIUS + tc)
            if 0 <= cc < W:
                tile[tr, tc] = int(grid[rr, cc])
    return tile


def _packed_width(n_tiles: int) -> int:
    return int(n_tiles * TILE_SIZE + max(0, n_tiles - 1) * TILE_GAP)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs, crop a local window around each one, and pack the crops into a row.

    Input structure:
    - bg = most common color.
    - One non-bg foreground color forms cluttered shapes on the bg.
    - Some foreground pixels are centers of local '+' motifs: the 4 orthogonal neighbors are also foreground.

    Rule:
    - Find every '+' center in scan order.
    - For each center, take its 5x5 neighborhood, recolor the center and its 4 orthogonal neighbors to teal (8) inside that crop, and keep all other pixels unchanged.
    - Concatenate the resulting 5x5 crops left-to-right with a 1-column bg gap between adjacent crops.

    Key insight: The task still hinges on recognizing the same local plus condition, but the answer is no longer an in-place highlight; it is a packed gallery of per-plus local views.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return g.copy()
    fg = int(colors[0])

    centers = _find_plus_centers(g, fg)
    if not centers:
        return g.copy()

    out = np.full((TILE_SIZE, _packed_width(len(centers))), int(bg), dtype=int)
    for i, (r, c) in enumerate(centers):
        tile = _extract_tile(g, r, c, bg)
        for dr, dc in CROSS_OFFSETS:
            tile[CROP_RADIUS + dr, CROP_RADIUS + dc] = TEAL
        c0 = int(i * (TILE_SIZE + TILE_GAP))
        out[:, c0 : c0 + TILE_SIZE] = tile
    return out


def _would_create_plus(grid: np.ndarray, r: int, c: int, fg: int) -> bool:
    """Check whether painting one new fg cell would create any plus center nearby."""
    H, W = grid.shape
    candidates = [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    for cr, cc in candidates:
        if cr < 1 or cr >= H - 1 or cc < 1 or cc >= W - 1:
            continue
        all_fg = True
        for dr, dc in CROSS_OFFSETS:
            nr, nc = cr + dr, cc + dc
            val = fg if (nr, nc) == (r, c) else int(grid[nr, nc])
            if val != fg:
                all_fg = False
                break
        if all_fg:
            return True
    return False


def _place_centers(rng: np.random.Generator, H: int, W: int, target_k: int) -> list[tuple[int, int]]:
    """Greedily sample plus centers with the original L1 separation constraint."""
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(abs(cr - er) + abs(cc - ec) >= 4 for er, ec in centers):
            centers.append((int(cr), int(cc)))
            if len(centers) >= target_k:
                break
    return centers


def _extend_pluses(grid: np.ndarray, centers: list[tuple[int, int]], rng: np.random.Generator) -> bool:
    """Add at least one extra arm cell for each plus so it is visibly embedded."""
    dirs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    H, W = grid.shape

    for cr, cc in centers:
        desired_dirs = int(rng.integers(1, 3))
        added_dirs = 0
        for di in rng.permutation(len(dirs)):
            dr, dc = dirs[int(di)]
            ext_len = int(rng.integers(1, 3))
            added_here = 0
            for step in range(2, 2 + ext_len):
                nr = int(cr + dr * step)
                nc = int(cc + dc * step)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if int(grid[nr, nc]) == FG:
                    break
                if _would_create_plus(grid, nr, nc, FG):
                    break
                grid[nr, nc] = FG
                added_here += 1
            if added_here > 0:
                added_dirs += 1
            if added_dirs >= desired_dirs:
                break
        if added_dirs == 0:
            return False
    return True


def _sprinkle_distractors(grid: np.ndarray, rng: np.random.Generator) -> None:
    """Raise fg density without introducing extra plus centers."""
    total = int(grid.size)
    target_density = float(rng.uniform(0.26, 0.38))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(grid != BG))

    bg_positions = list(zip(*np.where(grid == BG)))
    rng.shuffle(bg_positions)
    for r, c in bg_positions:
        if current_fg >= target_fg:
            break
        if not _would_create_plus(grid, int(r), int(c), FG):
            grid[int(r), int(c)] = FG
            current_fg += 1


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(12, 23))

    grid = np.full((H, W), BG, dtype=int)

    target_k = int(rng.integers(2, 5))  # 2..4 crops keeps the packed-row effect obvious.
    centers = _place_centers(rng, H, W, target_k)
    if len(centers) < 2:
        return None

    for cr, cc in centers:
        for dr, dc in CROSS_OFFSETS:
            grid[cr + dr, cc + dc] = FG

    if not _extend_pluses(grid, centers, rng):
        return None

    _sprinkle_distractors(grid, rng)

    if int(np.sum(grid != BG)) >= grid.size // 2:
        return None

    found = _find_plus_centers(grid, FG)
    if found != sorted(centers):
        return None

    out = main(grid)
    if out.shape != (TILE_SIZE, _packed_width(len(centers))):
        return None
    if out.shape == grid.shape:
        return None
    if not np.any(out == TEAL):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
