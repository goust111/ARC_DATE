# Variant of 1818057f
# Dimension: D11 — 前景像素的连通性利用方式 (meta)
# Change: 不直接使用连通分量分析，仅在每个像素的局部4-邻域上做判断
#   → 连通分量的边界像素重着色
#
# Design analysis
# ===============
# variant_spec:
#   concrete_behavior:
#     - 仍然是单一背景色 + 单一前景色的二维网格，颜色约束沿用原题：
#       background=2/red, foreground=4/yellow, output highlight=8/teal。
#     - 输入中的前景不再围绕“嵌入式十字”组织，而是由 2~4 个彼此分离的
#       4-连通厚块组成。每个厚块通常是一个实心矩形再拼接 0~2 个矩形凸起，
#       所以视觉上会出现矩形、胖 L、带台阶的 blob 等形状。
#     - 输出中，对每个前景连通分量的外露轮廓着色为青色，内部像素保持黄色。
#       这里“外露”指某个正交方向上接触到背景或网格外部。
#   behavior_trace:
#     - 原逻辑的小例子：若输入含一个 3x3 实心方块，中心像素有四个正交前景邻居，
#       原 solver 会把中心及四邻域共 5 个像素改成青色，四角仍是黄色。
#     - 变体逻辑：同一个 3x3 方块只有中心不是边界，因此输出应是 8 个外圈像素变青，
#       中心保留黄色。
#     - 差异：原题强调局部“+”模板命中；变体强调整块前景的整体连通性和外轮廓。
#   complexity_calibration:
#     - 若只生成细线或单像素噪声，则所有前景都会成为边界，规则退化得过于简单，
#       也不容易和“十字检测”区分。
#     - 因此生成器强制至少放置一个具有内部像素的厚分量，并在大多数样本中放置多个
#       不同外形的厚分量，让“边界变青、内部保黄”的模式能被肉眼直接看出。
#     - 颜色仍限制为 3 种，避免引入新的不可见映射关系。
#   implicit_assumption_audit:
#     - `get_bg(g)` 依赖背景色是全局最多色；生成器保持前景密度在 26%~41%，确保成立。
#     - `colors[0]` 依赖输入只有一个非背景色；生成器在输入阶段只放 4/yellow，成立。
#     - 连通性改为 4-连通 BFS。原 solver 的逐点局部判断不再足够，因为现在需要先知道
#       哪些像素属于同一个分量，再在该分量上提取边界。
#     - 边界判断把网格外也视作“外露边”。生成器仍给对象留 1 格外边框，这样即便不触边，
#       规则也完全可观察；若偶然贴边，solver 仍有明确行为。
#     - `out` 先复制输入再局部改色，依赖“不存在需要恢复成背景色的操作”；变体仍只重着色，
#       所以前提成立。
#   placement_legality:
#     - 组件放置时要求：不越界、组件间至少留 1 格背景缓冲、且通常避免对角相贴，
#       防止视觉拥挤和意外并块。
#     - 组件本体由紧密拼接的实心矩形组成，因此天然 4-连通、无孔洞，solver 的组件遍历
#       与生成器的结构假设一致。
#   observability:
#     - 在 2~3 个样例中，人类可以看到：同色 blob 的外圈总变青，而厚块中央留下黄色核。
#       这是稳定、重复的结构线索，不依赖颜色 id 排序或隐式数值规则。
#   cleanliness:
#     - 使用少量大块而不是大量碎噪声，保证边界轮廓清楚。
#     - 分量间留空，减少多个轮廓交错导致的视觉歧义。
#   new_main_assumptions_checked_against_generator:
#     - 假设存在单一前景色：已保证。
#     - 假设至少一个样本里存在边界与内部之分：生成器要求至少一个分量拥有 >=2 个内部像素。
#     - 假设组件遍历顺序不影响结果：本规则只把满足条件的 fg 改成固定颜色 8，顺序无副作用。

from __future__ import annotations

import numpy as np

from arc_utils import get_bg

ORTHO4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _collect_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    H, W = mask.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[list[tuple[int, int]]] = []

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
                for dr, dc in ORTHO4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < H and 0 <= nc < W and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _boundary_cells(mask: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    H, W = mask.shape
    boundary: list[tuple[int, int]] = []
    for r, c in cells:
        exposed = False
        for dr, dc in ORTHO4:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < H and 0 <= nc < W) or not mask[nr, nc]:
                exposed = True
                break
        if exposed:
            boundary.append((r, c))
    return boundary


def _component_interior_count(mask: np.ndarray) -> int:
    total = 0
    for cells in _collect_components(mask):
        total += len(cells) - len(_boundary_cells(mask, cells))
    return total


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the exposed outline of each foreground component to teal.

    Input structure:
    - bg is the most common color.
    - One foreground color forms several 4-connected blobs on the bg, often thick enough to have interior pixels.

    Rule:
    - Split the foreground into 4-connected components.
    - In each component, recolor every foreground pixel that touches background or the grid exterior in one of the 4 orthogonal directions to teal (8).
    - Foreground pixels surrounded on all 4 orthogonal sides by the same component stay in the original foreground color.

    Key insight:
    - This is no longer a local '+' template check. The solver first groups pixels by connectivity, then extracts each component's outer shell; only those boundary pixels change color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    fg_mask = g == fg
    for cells in _collect_components(fg_mask):
        for r, c in _boundary_cells(fg_mask, cells):
            out[r, c] = 8
    return out


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(mask)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _make_component_shape(rng: np.random.Generator, max_h: int, max_w: int) -> np.ndarray | None:
    canvas = np.zeros((9, 9), dtype=bool)

    base_h = int(rng.integers(3, 6))
    base_w = int(rng.integers(3, 6))
    base_r = int(rng.integers(2, 8 - base_h))
    base_c = int(rng.integers(2, 8 - base_w))
    canvas[base_r : base_r + base_h, base_c : base_c + base_w] = True

    for _ in range(int(rng.integers(0, 3))):
        r0, c0, r1, c1 = _bbox(canvas)
        side = str(rng.choice(["top", "bottom", "left", "right"]))
        placed = False

        for _ in range(12):
            if side in {"top", "bottom"}:
                ph = int(rng.integers(2, 5))
                pw = int(rng.integers(2, 6))
                overlap = int(rng.integers(2, min(pw, c1 - c0 + 1) + 1))
                seg0 = int(rng.integers(c0, c1 - overlap + 2))
                left = seg0 - int(rng.integers(0, pw - overlap + 1))
                top = r0 - ph if side == "top" else r1 + 1
                if 0 <= top and top + ph <= 9 and 0 <= left and left + pw <= 9:
                    canvas[top : top + ph, left : left + pw] = True
                    placed = True
                    break
            else:
                ph = int(rng.integers(2, 6))
                pw = int(rng.integers(2, 5))
                overlap = int(rng.integers(2, min(ph, r1 - r0 + 1) + 1))
                seg0 = int(rng.integers(r0, r1 - overlap + 2))
                top = seg0 - int(rng.integers(0, ph - overlap + 1))
                left = c0 - pw if side == "left" else c1 + 1
                if 0 <= top and top + ph <= 9 and 0 <= left and left + pw <= 9:
                    canvas[top : top + ph, left : left + pw] = True
                    placed = True
                    break

        if not placed:
            continue

    shape = _crop(canvas)
    if shape.shape[0] > max_h or shape.shape[1] > max_w:
        return None

    area = int(shape.sum())
    if not (9 <= area <= 42):
        return None
    if _component_interior_count(shape) < 2:
        return None
    return shape


def _can_place(occupied: np.ndarray, shape: np.ndarray, top: int, left: int) -> bool:
    H, W = occupied.shape
    h, w = shape.shape
    cells = np.argwhere(shape)
    for sr, sc in cells:
        r = top + int(sr)
        c = left + int(sc)
        if occupied[r, c]:
            return False
        for nr in range(max(0, r - 1), min(H, r + 2)):
            for nc in range(max(0, c - 1), min(W, c + 2)):
                if occupied[nr, nc]:
                    return False
    return True


def _paint_shape(grid: np.ndarray, shape: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = shape.shape
    region = grid[top : top + h, left : left + w]
    region[shape] = color


def _build_output_from_input(inp: np.ndarray, fg: int) -> np.ndarray:
    fg_mask = inp == fg
    out = inp.copy()
    for cells in _collect_components(fg_mask):
        for r, c in _boundary_cells(fg_mask, cells):
            out[r, c] = 8
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    total = H * W
    min_fg = int(np.ceil(total * 0.26))
    max_fg = int(np.floor(total * 0.41))

    grid = np.full((H, W), bg, dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    if total < 170:
        desired_components = int(rng.integers(2, 4))
    else:
        desired_components = int(rng.integers(3, 5))

    placements = 0
    for idx in range(desired_components):
        remaining_area = max(min_fg - int(occupied.sum()), 9)
        components_left = desired_components - idx
        soft_cap = max_fg - int(occupied.sum()) - 9 * (components_left - 1)
        if soft_cap < 9:
            break

        placed = False
        for _ in range(120):
            max_h = min(8, H - 2)
            max_w = min(8, W - 2)
            shape = _make_component_shape(rng, max_h=max_h, max_w=max_w)
            if shape is None:
                continue
            area = int(shape.sum())
            avg_target = max(remaining_area // components_left, 9)
            if area > soft_cap or area > avg_target + 16:
                continue

            h, w = shape.shape
            if h > H - 2 or w > W - 2:
                continue

            top = int(rng.integers(1, H - h))
            left = int(rng.integers(1, W - w))
            if not _can_place(occupied, shape, top, left):
                continue

            _paint_shape(grid, shape, top, left, fg)
            _paint_shape(occupied, shape, top, left, True)
            placed = True
            placements += 1
            break

        if not placed and placements < 2:
            return None

    fg_count = int(occupied.sum())
    if placements < 2:
        return None
    if not (min_fg <= fg_count <= max_fg):
        return None
    if _component_interior_count(occupied) < 4:
        return None

    out = _build_output_from_input(grid, fg)
    if np.count_nonzero(out == 8) < 12:
        return None
    if np.count_nonzero(out == fg) < 4:
        return None

    inp = grid.copy()
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
