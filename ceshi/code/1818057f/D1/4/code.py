# Variant of 1818057f
# Dimension: D1 — 核心变换操作：重着色规则 (transform)
# Change: 识别'+'字图案后，将构成'+'的5个像素从前景色(黄色4)统一重着色为青色(teal 8)，其余像素不变
#   → 识别'+'图案后，将中心 3x3 邻域整体填成青色，令十字膨胀为 3x3 实心方块。

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg


# Design analysis for the D1 variant
# ----------------------------------
# spec:
#   core_change:
#     original: "检测到一个局部 '+' 后，只把十字本身 5 格改成 teal。"
#     variant: "检测到同样的局部 '+' 后，把以该中心为核心的整个 3x3 区域都改成 teal。"
#   concrete_pixel_behavior:
#     input:
#       - 背景仍为最常见色，生成器固定用 2/red。
#       - 唯一前景色仍为 4/yellow。
#       - 真正触发变换的对象仍是细十字：中心和四个正交邻居为前景。
#       - 为了让新规则肉眼可见，十字四个对角格在输入中必须保留为背景，而不是预先填满。
#     output:
#       - 检测条件完全不变；变化只在输出 footprint。
#       - 每个被检测到的十字从 5 格 teal 变成 9 格 teal，即围出 3x3 实心块。
#   behavior_trace:
#     example:
#       input_3x3:
#         2 4 2
#         4 4 4
#         2 4 2
#       original_output:
#         2 8 2
#         8 8 8
#         2 8 2
#       variant_output:
#         8 8 8
#         8 8 8
#         8 8 8
#     difference:
#       - 原题只重着色已有十字像素。
#       - 变体额外把四个对角背景格“补满”，因此输出出现 bg -> teal 的新增像素。
#   complexity_calibration:
#     - 检测逻辑仍然是单一局部模式，避免把任务复杂度抬到多对象匹配级别。
#     - 但生成器强制每个真十字都带有 4 个空对角，保证新旧任务在视觉上稳定不同。
#     - 中心之间保留较大间距，让 3x3 teal 块不互相覆盖，避免解题者误把结果看成大块涂抹。
#   implicit_assumptions_review_for_main:
#     - `bg = get_bg(g)`: 仍依赖背景是最多的颜色；生成器继续保证前景不到一半。
#     - `colors[0]` 作为唯一前景色：仍成立，因为输入只含 bg 与一种 fg。
#     - 遍历 `r in 1..H-2, c in 1..W-2`：检测 '+' 只需 1 格边距；3x3 膨胀也只需这同一格边距。
#     - 检测条件只看中心与四邻：在变体中仍成立，不能把对角是否为空写进 solver，
#       否则会把“需要推断的膨胀规则”误改成“更严格的输入模式识别规则”。
#     - 输出写入现在是整块 3x3，而不是仅写十字 5 格；因此 generator 必须避免两个目标中心过近，
#       否则多个 3x3 输出块会相互覆盖并弱化规则可读性。
#   placement_legality:
#     - 中心放在距边界至少 2 格的位置，既保留 3x3 空心/实心对比，也给延伸臂留空间。
#     - 任意两个中心满足 Chebyshev 距离 >= 4，使各自的 3x3 保护区互不接触。
#     - 每个中心至少沿一个臂向外延伸 1 格以上，确保 '+' 不是孤立 5 格，而是嵌在更大前景成分中。
#   visual_observability:
#     - 每个真 '+' 的四个对角格在输入中都保持背景，并在输出中统一变成 teal。
#     - 因此单个样例里就能看到“从十字补成方块”的规律，而不是仅看到颜色替换。
#   cleanliness:
#     - 生成器把所有目标十字的 3x3 邻域设为保留区，不让噪声侵入对角或压到相邻方块上。
#     - 干扰物只在保留区外生成，并且每次加入后都验证不会新造 '+' 中心。
#   generator_validations:
#     - `set(_find_plus_centers(inp, fg)) == intended_centers`
#     - `main(inp)` 必须成功。
#     - `np.sum((inp == bg) & (out == 8)) == 4 * len(intended_centers)`：
#       每个真十字都要产生 4 个由背景补成的 teal 对角，这是本变体的量化指纹。


CROSS5 = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
DIRS4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect thin '+' motifs in one foreground color and expand each into a teal 3x3 block.

    Input structure:
    - bg = most common color.
    - One foreground color forms noisy lines/blobs on the bg, and some local patterns are thin '+' crosses.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Recolor the entire 3x3 neighborhood around every such center to teal (8), not just the 5 cells of the cross.
    - All other pixels remain unchanged.

    Key insight: The detection cue is still the orthogonal cross, but the output footprint is larger than the detected support, so the four diagonal cells must be inferred and filled to complete a solid square.
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


def _can_paint_cells(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    fg: int,
    reserved: np.ndarray,
    allowed_centers: set[tuple[int, int]],
    bg: int,
) -> bool:
    H, W = grid.shape
    if not cells:
        return False
    for r, c in cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        if bool(reserved[int(r), int(c)]):
            return False
        if int(grid[int(r), int(c)]) != int(bg):
            return False

    candidate = grid.copy()
    for r, c in cells:
        candidate[int(r), int(c)] = int(fg)
    return set(_find_plus_centers(candidate, fg)) == allowed_centers


def _paint_cells(grid: np.ndarray, cells: list[tuple[int, int]], fg: int) -> None:
    for r, c in cells:
        grid[int(r), int(c)] = int(fg)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    grid = np.full((H, W), bg, dtype=int)

    target_k = int(rng.integers(1, 7))
    candidates = [(r, c) for r in range(2, H - 2) for c in range(2, W - 2)]
    rng.shuffle(candidates)

    centers: list[tuple[int, int]] = []
    for cr, cc in candidates:
        if all(max(abs(int(cr) - er), abs(int(cc) - ec)) >= 4 for er, ec in centers):
            centers.append((int(cr), int(cc)))
            if len(centers) >= target_k:
                break
    if not centers:
        return None

    reserved = np.zeros((H, W), dtype=bool)
    for cr, cc in centers:
        reserved[cr - 1 : cr + 2, cc - 1 : cc + 2] = True

    for cr, cc in centers:
        for dr, dc in CROSS5:
            grid[cr + dr, cc + dc] = fg

    intended_centers = set(centers)
    if set(_find_plus_centers(grid, fg)) != intended_centers:
        return None

    # Each plus must be embedded in a larger component, so extend 1-2 arms outward
    # starting beyond the 3x3 reserved zone.
    for cr, cc in centers:
        corridors: list[list[tuple[int, int]]] = []
        for dr, dc in DIRS4:
            cells: list[tuple[int, int]] = []
            step = 2
            while True:
                nr, nc = int(cr + dr * step), int(cc + dc * step)
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                if bool(reserved[nr, nc]):
                    break
                cells.append((nr, nc))
                step += 1
            if cells:
                corridors.append(cells)
        if not corridors:
            return None

        n_ext = int(rng.integers(1, min(2, len(corridors)) + 1))
        chosen_ids = rng.choice(len(corridors), size=n_ext, replace=False)
        added_any = False
        for idx in np.atleast_1d(chosen_ids):
            corridor = corridors[int(idx)]
            max_len = min(3, len(corridor))
            ext_len = int(rng.integers(1, max_len + 1))
            cells = corridor[:ext_len]
            if not _can_paint_cells(grid, cells, fg, reserved, intended_centers, bg):
                return None
            _paint_cells(grid, cells, fg)
            added_any = True
        if not added_any:
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
        if not _can_paint_cells(grid, cells, fg, reserved, intended_centers, bg):
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

    inp = grid.copy()
    out = main(inp)
    if not np.any(out == 8):
        return None

    bg_to_teal = int(np.sum((inp == bg) & (out == 8)))
    if bg_to_teal != 4 * len(intended_centers):
        return None

    if int(np.sum(out == 8)) != 9 * len(intended_centers):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
