# Variant of 1818057f
# Dimension: D9 — 扫描方向和处理顺序 (multi_step)
# Change: 从上到下、从左到右扫描内部像素(r in 1..H-2, c in 1..W-2)，基于原始输入判断'+'，一次性完成所有标记
#   → 迭代传播式重着色：首轮只把'+'中心染成青色，随后再做固定轮数的同步传播

from __future__ import annotations

from collections import deque
import numpy as np

from arc_utils import get_bg


# Design analysis
# ---------------
# variant:
#   source_baseline:
#     - 参考任务 800d221b 的复杂度基线不是“单次局部替换”，而是“先识别种子/关键结构，再分阶段扩散到别处”。
#     - 这里借用它的“先找种子，再按邻接关系传播”的思路，但保持本题只有两种输入色、目标仍是给 '+' 相关区域染青色。
#   concretization_and_trace:
#     - 输入仍是红底(2)上单一前景黄(4)的细线/十字混合图。
#     - 旧规则：若某格上下左右都是前景，则立即把这个中心和四个正交邻居都改成青色。
#     - 新规则固定做 4 轮：
#       1. 第 1 轮只把所有 '+' 中心改成青色。
#       2. 第 2~4 轮中，凡是原输入里的前景格只要与当前青色格正交相邻，就在本轮结束时变青。
#     - 小例子：若一枚 '+' 的右臂继续向右多延伸 3 格，旧规则只会把中心和离中心距离 1 的四臂染青；
#       新规则会把距离 0/1/2/3 的前景都染青，距离 4 及以上仍保持黄色，因此“传播深度”能被看到。
#   complexity_calibration:
#     - 若只做 2 轮，会退化回原题；因此固定为 4 轮，让传播比原题多走 2 步，但又不会一口气染完整个连通块。
#     - 生成器强制至少存在一条见证臂：同一前景连通块中既有距离种子正好 3 的格子，也有距离至少 4 的格子。
#       这样样例里既能看到“比原题多染了”，也能看到“传播会停止”，规则才可推断。
#   implicit_assumptions_review:
#     - 原 main() 的 `out = g.copy()` 仍成立，因为背景和未传播到的前景都应保持原色。
#     - 原先检测 '+' 时只看原图 `g`；新规则仍必须如此，否则后续传播出的青色会反过来制造假种子。
#     - 原先在检测到中心后立即给 5 格赋值；这一行在变体里不再安全，因为它会跳过“只种中心、再逐轮扩散”的中间状态。
#     - 传播阶段必须基于“上一轮的青色集合”同步更新；若在单轮扫描时原地连锁赋值，会引入扫描顺序伪效应，破坏固定步数。
#   placement_legality:
#     - '+' 中心只放在有 1 格边距的位置。
#     - 每个分支只允许通过前驱格附着，避免与其他组件侧向接触、重叠或意外连成大块。
#     - 额外前景干扰块与主组件完全断开，并经过“不会新增 '+' 中心”的检查。
#   visual_observability:
#     - 所有输入都只用背景色和一种前景色；输出只新增青色，不依赖颜色 id 排序等不可见信息。
#     - 传播深度在细长臂上表现为“从中心向外延长 3 格的青色前沿”，与原题的 5 格十字有稳定、肉眼可见的差别。
#   neatness:
#     - 组件之间保持较大间距；分支至多转一次弯，避免满屏噪声。
#     - 干扰块限制为小型孤立形状，不与传播组件叠压，保证青色前沿清晰可读。
#   generator_contract_for_main:
#     - main() 假设输入只有一个非背景色前景，且至少有一个 '+' 种子；生成器显式保证这两点。
#     - main() 假设传播发生在原前景图的 4 邻接图上；生成器保证主组件都是 1 格粗细、无重叠，便于该假设稳定成立。

TEAL = 8
BG = 2
FG = 4
PROPAGATION_ROUNDS = 4
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_plus_centers(grid: np.ndarray, fg: int) -> list[tuple[int, int]]:
    H, W = grid.shape
    centers: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if int(grid[r, c]) != fg:
                continue
            if all(int(grid[r + dr, c + dc]) == fg for dr, dc in DIRS):
                centers.append((r, c))
    return centers


def _propagation_mask(grid: np.ndarray, fg: int, rounds: int) -> np.ndarray:
    fg_mask = grid == int(fg)
    teal = np.zeros(grid.shape, dtype=bool)

    for r, c in _find_plus_centers(grid, fg):
        teal[int(r), int(c)] = True

    for _ in range(max(0, int(rounds) - 1)):
        grown = teal.copy()
        H, W = grid.shape
        for r in range(H):
            for c in range(W):
                if not bool(fg_mask[r, c]) or bool(teal[r, c]):
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


def _classic_teal_mask(grid: np.ndarray, fg: int) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    for r, c in _find_plus_centers(grid, fg):
        mask[int(r), int(c)] = True
        for dr, dc in DIRS:
            mask[int(r + dr), int(c + dc)] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Start from '+' centers and let teal spread outward for a fixed number of passes.

    Input structure:
    - The most common color is the background.
    - Exactly one other color forms thin '+' motifs, longer attached arms, and separate distractor pieces.

    Rule:
    - Pass 1 recolors only foreground pixels whose 4 orthogonal neighbors are also foreground.
    - Then 3 more synchronous passes are applied; on each pass, any remaining foreground pixel orthogonally touching teal also becomes teal.
    - Background cells and disconnected foreground beyond the 4-pass reach stay unchanged.

    Key insight: The whole 5-cell cross is no longer recolored at once. The centers are the only seeds, so the final teal region is the original foreground graph within Manhattan graph-distance 3 of any '+' center.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    teal = _propagation_mask(g, fg, PROPAGATION_ROUNDS)
    out[teal] = TEAL
    return out


def _would_create_unplanned_plus(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    for cr, cc in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if not (1 <= cr < H - 1 and 1 <= cc < W - 1):
            continue
        cells = [(cr, cc)] + [(cr + dr, cc + dc) for dr, dc in DIRS]
        if all((nr, nc) == (r, c) or int(grid[nr, nc]) == fg for nr, nc in cells):
            if (int(cr), int(cc)) not in allowed_centers:
                return True
    return False


def _can_add_path_cell(
    grid: np.ndarray,
    r: int,
    c: int,
    fg: int,
    prev: tuple[int, int],
    allowed_centers: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    if not (0 <= r < H and 0 <= c < W):
        return False
    if int(grid[r, c]) != BG:
        return False
    for dr, dc in DIRS:
        nr, nc = int(r + dr), int(c + dc)
        if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == fg and (nr, nc) != prev:
            return False
    if _would_create_unplanned_plus(grid, r, c, fg, allowed_centers):
        return False
    return True


def _try_place_branch(
    grid: np.ndarray,
    center: tuple[int, int],
    direction: tuple[int, int],
    total_len: int,
    rng: np.random.Generator,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    dr, dc = direction
    tip = (int(center[0] + dr), int(center[1] + dc))
    options: list[list[tuple[int, int]]] = [[direction] * int(total_len)]
    if total_len >= 3:
        turns = [(-dc, dr), (dc, -dr)]
        for split in range(2, int(total_len)):
            for turn in turns:
                options.append([direction] * int(split) + [turn] * int(total_len - split))
    rng.shuffle(options)

    for plan in options:
        temp = grid.copy()
        prev = tip
        cur_r, cur_c = map(int, tip)
        ok = True
        for step_dr, step_dc in plan:
            cur_r += int(step_dr)
            cur_c += int(step_dc)
            if not _can_add_path_cell(temp, cur_r, cur_c, FG, prev, allowed_centers):
                ok = False
                break
            temp[cur_r, cur_c] = FG
            prev = (cur_r, cur_c)
        if ok:
            grid[:, :] = temp
            return True
    return False


def _distance_from_centers(grid: np.ndarray, fg: int) -> np.ndarray:
    H, W = grid.shape
    dist = np.full((H, W), -1, dtype=int)
    q: deque[tuple[int, int]] = deque()
    for r, c in _find_plus_centers(grid, fg):
        dist[int(r), int(c)] = 0
        q.append((int(r), int(c)))

    while q:
        r, c = q.popleft()
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                continue
            if int(grid[nr, nc]) != fg or dist[nr, nc] != -1:
                continue
            dist[nr, nc] = int(dist[r, c] + 1)
            q.append((nr, nc))
    return dist


DISTRACTOR_SHAPES = [
    [(0, 0)],
    [(0, 0), (0, 1)],
    [(0, 0), (1, 0)],
    [(0, 0), (0, 1), (0, 2)],
    [(0, 0), (1, 0), (2, 0)],
    [(0, 0), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1)],
    [(0, 0), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (1, 1), (2, 0), (2, 1)],
    [(0, 0), (0, 1), (1, 0), (1, 1)],
]


def _try_place_distractor(
    grid: np.ndarray,
    shape: list[tuple[int, int]],
    anchor_r: int,
    anchor_c: int,
    allowed_centers: set[tuple[int, int]],
) -> bool:
    H, W = grid.shape
    cells = [(int(anchor_r + dr), int(anchor_c + dc)) for dr, dc in shape]
    cell_set = set(cells)

    for r, c in cells:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if int(grid[r, c]) != BG:
            return False

    for r, c in cells:
        for dr, dc in DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if 0 <= nr < H and 0 <= nc < W and int(grid[nr, nc]) == FG and (nr, nc) not in cell_set:
                return False

    temp = grid.copy()
    for r, c in cells:
        temp[r, c] = FG

    if set(_find_plus_centers(temp, FG)) != allowed_centers:
        return False

    grid[:, :] = temp
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    grid = np.full((H, W), BG, dtype=int)

    area = H * W
    max_centers = min(5, max(1, area // 85 + 1))
    target_centers = int(rng.integers(1, max_centers + 1))

    centers: list[tuple[int, int]] = []
    for _ in range(target_centers * 80):
        if len(centers) >= target_centers:
            break
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        if all(abs(cr - er) + abs(cc - ec) >= 7 for er, ec in centers):
            centers.append((cr, cc))

    if not centers:
        return None

    for cr, cc in centers:
        cells = [(cr, cc)] + [(cr + dr, cc + dc) for dr, dc in DIRS]
        if any(int(grid[r, c]) != BG for r, c in cells):
            return None
        for r, c in cells:
            grid[r, c] = FG

    allowed_centers = set((int(r), int(c)) for r, c in centers)

    witness_built = False
    for center in centers:
        dirs = list(DIRS)
        rng.shuffle(dirs)
        branch_count = int(rng.integers(1, 4))
        built_here = 0

        if not witness_built:
            witness_dirs = dirs[:]
            rng.shuffle(witness_dirs)
            for direction in witness_dirs:
                if _try_place_branch(grid, center, direction, int(rng.integers(3, 6)), rng, allowed_centers):
                    witness_built = True
                    built_here += 1
                    break

        for direction in dirs:
            if built_here >= branch_count:
                break
            if witness_built and built_here == 0 and branch_count == 1:
                length = int(rng.integers(1, 4))
            else:
                length = int(rng.integers(1, 6))
            if _try_place_branch(grid, center, direction, length, rng, allowed_centers):
                built_here += 1

        if built_here == 0:
            return None

    if not witness_built:
        return None

    target_density = float(rng.uniform(0.23, 0.33))
    target_fg = int(area * target_density)
    attempts = 0
    while int(np.sum(grid == FG)) < target_fg and attempts < area * 10:
        attempts += 1
        shape = DISTRACTOR_SHAPES[int(rng.integers(0, len(DISTRACTOR_SHAPES)))]
        anchor_r = int(rng.integers(0, H))
        anchor_c = int(rng.integers(0, W))
        _try_place_distractor(grid, shape, anchor_r, anchor_c, allowed_centers)

    if int(np.sum(grid == FG)) >= area // 2:
        return None

    centers_found = _find_plus_centers(grid, FG)
    if set(centers_found) != allowed_centers:
        return None

    dist = _distance_from_centers(grid, FG)
    reachable = dist >= 0
    max_reachable = int(dist[reachable].max()) if np.any(reachable) else -1
    if max_reachable < PROPAGATION_ROUNDS:
        return None
    if not np.any(dist == PROPAGATION_ROUNDS - 1):
        return None

    variant_out = main(grid)
    classic_mask = _classic_teal_mask(grid, FG)
    variant_mask = variant_out == TEAL
    if int(np.sum(variant_mask)) <= int(np.sum(classic_mask)):
        return None
    if not np.any((dist > PROPAGATION_ROUNDS - 1) & (grid == FG)):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
