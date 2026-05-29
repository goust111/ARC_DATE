# Combo variant of b99e7126 (index 4)
# combo_spec: D2/0+D7/0
# Name: 多异常组合源图案 + 多异常独立放大

from __future__ import annotations

from functools import reduce
from itertools import combinations
from math import gcd
from pathlib import Path
import random
import sys

import numpy as np

_SOLVER_DIR = Path(__file__).resolve().parents[4] / "solvers"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import paste


# Design analysis
# a) 具体化与行为追踪
#    - 画面仍是 7x7 个 3x3 小格组成的规则晶格，分隔线颜色统一。
#    - 原题：输入里有一个“少数格”图案，读取它内部 3x3 的活跃像素，再把这个掩码平移到
#      大网格上，只补全那一个全局 3x3 粗块。
#    - 本融合：输入里改成 2-3 个单例异常格，它们都位于各自将被放大的粗块中心，但每个
#      异常格只展示最终模板的一部分碎片；这些碎片先在同一 3x3 坐标系里做并集，得到一
#      个完整模板。随后，不是只放大一次，而是以每个异常格为中心，各自独立地把同一个
#      “组合后模板”放大成 3x3 粗块。
#    - 例子：若三个异常格分别显示“中心+上边”“中心+左下”“中心+右下”，solver 先把
#      三者并成完整模板，再在三个异常中心周围都画出同一份完整模板，而不是分别画出三份
#      不同小模板。
# b) 复杂度校准
#    - 异常格数量固定为 2 或 3，防止退回原题的单异常。
#    - 完整模板活跃像素固定为 6 或 7，保持和原题相近的视觉密度。
#    - 每个碎片都必须是完整模板的真子集、都含中心像素、都有 3-5 个活跃像素，并且至少
#      两个碎片图案不同。这样输入中“多个不同异常格”肉眼可见，输出中“每处都扩成同一
#      个更完整模板”也明显可见。
# c) 隐式假设审查
#    - 原 `min(counts)` 假设只有一种少数模式；这里失效，因为 2-3 个异常格都各不相同。
#      新 solver 只能先找唯一多数模式，再把其余单例都当作种子。
#    - 原“异常格里出现次数最多的颜色就是 active”在碎片场景不稳定，因为碎片可能稀疏；
#      新 solver 改为：active 色是异常格中出现、但多数格中不出现的那种颜色。
#    - 原 solver 假设多个异常位置共同约束一个全局平移；融合后不再搜索单一 `(top,left)`，
#      而是把每个异常格自身当作一个局部放大中心。
#    - 原输出循环默认所有活跃粗格都写入“观察到的异常图案”；融合后必须先合成一个可能
#      从未完整出现过的 `full_pattern`，再在每个中心重复写它。
# d) 放置合法性
#    - 所有中心都放在 1..5 的粗坐标内，保证其 3x3 放大块完全落在 7x7 晶格内。
#    - 任意两中心对应的 3x3 放大块都两两不重叠，避免覆盖冲突，也让多个局部操作清晰。
#    - 每个碎片都使用相同的 shared 色和 anomaly 色，只在像素形状上不同，保证“组合”
#      是可见的像素并集，而不是依赖颜色编号的隐藏匹配。
# e) 视觉可观察性
#    - 输入一眼能看见 2-3 个位置分散的不同异常格；输出一眼能看见这些位置周围都长出
#      同一份更完整的 3x3 粗块图案。
#    - 颜色角色只有 4 个：分隔线、多数格专属、共享背景、异常活跃色。没有使用颜色 id
#      排序或数值大小这种人类不可见因素。
# f) 可视化整洁度
#    - 各放大块不重叠，且全局仍只有规则网格 + 少数局部改写区域，画面不会混乱。
#    - 输入中的碎片仅出现在中心格，不会在同一局部块里堆叠多层对象。
# g) 变化的可推断性
#    - 所有异常格共享同一对颜色，区别只来自像素子图案；输出里每个中心都复现相同的
#      完整模板。这给了解题者稳定的结构线索：先组合，再重复，而不是猜每个颜色各自映射。
# 9) 新 solver 假设与 generator 保证
#    - solver 假设多数模式是唯一最高频模式；generator 用它铺满所有非种子格。
#    - solver 假设非多数模式恰好有 2-3 个，且都只出现一次；generator 强制单例。
#    - solver 假设所有异常格共享同一个 shared 色和 anomaly 色；generator 强制一致。
#    - solver 假设每个碎片都是真子集且中心 active；generator 采样时强制，并拒绝任何会
#      退化成“各自独立放大自己的碎片”即可得到同样输出的样本。


def _infer_cell_grid_params(g: np.ndarray) -> tuple[int, list[int], list[int]]:
    """
    Infer a regular cell grid separated by full rows/cols of the same line color.
    Returns (step, row_divs, col_divs) where:
      - row_divs/col_divs are divider indices
      - step is the spacing between dividers (cell size including divider).
    """
    h, w = g.shape
    line = int(g[0, 0])
    row_divs = [r for r in range(h) if np.all(g[r, :] == line)]
    col_divs = [c for c in range(w) if np.all(g[:, c] == line)]
    assert len(row_divs) >= 2 and len(col_divs) >= 2, "need at least 2 row and 2 col dividers"
    row_divs.sort()
    col_divs.sort()
    drs = [row_divs[i + 1] - row_divs[i] for i in range(len(row_divs) - 1) if row_divs[i + 1] > row_divs[i]]
    dcs = [col_divs[i + 1] - col_divs[i] for i in range(len(col_divs) - 1) if col_divs[i + 1] > col_divs[i]]
    step = reduce(gcd, drs + dcs)
    assert step > 1, "divider step must be > 1"
    assert row_divs[0] == 0 and col_divs[0] == 0, "first divider must be at index 0"
    assert (h - 1) % step == 0 and (w - 1) % step == 0, "grid dimensions must be (k*step + 1)"
    assert row_divs[-1] == h - 1 and col_divs[-1] == w - 1, "last divider must be at boundary"
    return int(step), row_divs, col_divs


def _extract_cell_patterns(
    g: np.ndarray, step: int
) -> tuple[dict[tuple[int, ...], np.ndarray], dict[tuple[int, ...], list[tuple[int, int]]]]:
    n = (g.shape[0] - 1) // step
    inner = step - 1
    patterns: dict[tuple[int, ...], np.ndarray] = {}
    positions: dict[tuple[int, ...], list[tuple[int, int]]] = {}
    for i in range(n):
        for j in range(n):
            r0 = i * step + 1
            c0 = j * step + 1
            arr = g[r0 : r0 + inner, c0 : c0 + inner].copy()
            key = tuple(arr.ravel().tolist())
            if key not in patterns:
                patterns[key] = arr
            positions.setdefault(key, []).append((i, j))
    return patterns, positions


def _connected(mask: np.ndarray) -> bool:
    coords = [tuple(x) for x in np.argwhere(mask)]
    if not coords:
        return False
    seen = {coords[0]}
    frontier = [coords[0]]
    while frontier:
        r, c = frontier.pop()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                frontier.append((nr, nc))
    return len(seen) == len(coords)


def _compose_pattern_from_seeds(
    patterns: dict[tuple[int, ...], np.ndarray], positions: dict[tuple[int, ...], list[tuple[int, int]]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[int, int]], list[np.ndarray]]:
    counts = {pat: len(pos_list) for pat, pos_list in positions.items()}
    assert len(counts) >= 3, "need a majority pattern plus multiple singleton anomalies"

    majority_pat = max(counts.items(), key=lambda kv: kv[1])[0]
    majority_arr = patterns[majority_pat]
    majority_colors = set(map(int, majority_arr.ravel().tolist()))

    seed_pats = [pat for pat, cnt in counts.items() if pat != majority_pat]
    assert 2 <= len(seed_pats) <= 3, "variant expects 2-3 singleton anomaly cells"
    assert all(counts[pat] == 1 for pat in seed_pats), "every anomaly pattern must appear once"

    inner = majority_arr.shape[0]
    radius = inner // 2
    shared_color: int | None = None
    active_color: int | None = None
    union_mask = np.zeros_like(majority_arr, dtype=bool)
    centers: list[tuple[int, int]] = []
    fragment_masks: list[np.ndarray] = []

    for pat in seed_pats:
        arr = patterns[pat]
        (ci, cj) = positions[pat][0]
        colors = set(map(int, arr.ravel().tolist()))
        shared = colors & majority_colors
        extra = colors - majority_colors
        assert len(shared) == 1 and len(extra) == 1, "each fragment must share one color and add one anomaly color"
        this_shared = next(iter(shared))
        this_active = next(iter(extra))
        if shared_color is None:
            shared_color = this_shared
        else:
            assert this_shared == shared_color, "all fragments must share the same background color"
        if active_color is None:
            active_color = this_active
        else:
            assert this_active == active_color, "all fragments must share the same anomaly color"

        frag_mask = arr == this_active
        assert frag_mask[radius, radius], "each anomaly cell must mark itself as the center seed"
        assert 3 <= int(frag_mask.sum()) <= inner * inner - 1, "fragment must be visible but incomplete"
        centers.append((ci, cj))
        fragment_masks.append(frag_mask)
        union_mask |= frag_mask

    assert shared_color is not None and active_color is not None
    assert union_mask[radius, radius], "combined template must stay centered on each seed"
    assert 2 <= len({tuple(mask.ravel().tolist()) for mask in fragment_masks}) <= 3, "fragments must not collapse to one pattern"
    assert all(np.any(union_mask & ~mask) for mask in fragment_masks), "every fragment must be a strict subset of the combined template"

    full_pattern = np.where(union_mask, active_color, shared_color).astype(int)
    assert not np.array_equal(full_pattern, majority_arr), "combined anomaly pattern must differ from the majority"
    return majority_arr, full_pattern, union_mask, centers, fragment_masks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combine several singleton fragment cells into one full motif, then expand that same motif around every seed.

    Input structure:
    - The board is partitioned into equal cells by full divider rows and columns of one color.
    - One cell interior pattern is the clear majority background.
    - There are 2 or 3 singleton anomaly cells, all using the same shared background color and the same anomaly color,
      but each singleton shows only a different fragment of a larger 3x3 motif.

    Rule:
    - Read every non-majority singleton cell, identify the anomaly-only color, and overlay those anomaly-colored pixels
      in the common 3x3 coordinate frame to reconstruct one combined source motif.
    - Treat each singleton anomaly cell as the center of its own 3x3 coarse block on the cell lattice.
    - For every seed center, write the combined motif at cell scale: active coarse positions receive the reconstructed
      full anomaly pattern, and inactive coarse positions keep the majority pattern.

    Key insight:
    The exceptional cells are not separate self-contained blueprints. They are partial views of one hidden template,
    and that shared template must be reconstructed first before being repeated independently around every anomaly seed.
    """
    g = np.asarray(grid, dtype=int)
    step, _, _ = _infer_cell_grid_params(g)
    n = (g.shape[0] - 1) // step
    inner = step - 1
    assert inner % 2 == 1, "seed-centered expansion needs an odd interior size"
    radius = inner // 2

    patterns, positions = _extract_cell_patterns(g, step)
    majority_arr, full_pattern, full_mask, centers, fragment_masks = _compose_pattern_from_seeds(patterns, positions)
    assert full_mask.shape == (inner, inner), "combined source motif must match the cell interior size"
    assert _connected(full_mask), "combined motif must be readable as one connected shape"
    assert 6 <= int(full_mask.sum()) <= 7, "combined motif density should remain close to the seed task"

    occupied = np.zeros((n, n), dtype=bool)
    for (ci, cj), frag_mask in zip(centers, fragment_masks):
        assert radius <= ci < n - radius and radius <= cj < n - radius, "seed block must fit in bounds"
        assert _connected(frag_mask), "fragment masks must stay visually coherent"
        block = np.zeros((n, n), dtype=bool)
        block[ci - radius : ci + radius + 1, cj - radius : cj + radius + 1] = True
        assert not np.any(occupied & block), "seed expansions must be disjoint"
        occupied |= block

    out = g.copy()
    for ci, cj in centers:
        for di in range(inner):
            for dj in range(inner):
                target_pat = full_pattern if full_mask[di, dj] else majority_arr
                r0 = (ci + di - radius) * step + 1
                c0 = (cj + dj - radius) * step + 1
                out[r0 : r0 + inner, c0 : c0 + inner] = target_pat
    return out


# --------------- generator (o5x) ---------------

CELL_GRID_N = 7
CELL_INNER = 3
CELL_STEP = 4
GRID_SIZE = CELL_GRID_N * CELL_STEP + 1
SEED_COUNTS = (2, 3)
FULL_ACTIVE_PIXEL_COUNTS = (6, 7)
COLOR_POOL = 10
MAX_RETRIES = 600

_LR_GROUPS = [
    ((0, 0), (0, 2)),
    ((1, 0), (1, 2)),
    ((2, 0), (2, 2)),
    ((0, 1),),
    ((1, 1),),
    ((2, 1),),
]

_VALID_CENTERS = [(r, c) for r in range(1, CELL_GRID_N - 1) for c in range(1, CELL_GRID_N - 1)]


def _precompute_full_masks() -> list[np.ndarray]:
    masks: list[np.ndarray] = []
    for bits in range(1 << (CELL_INNER * CELL_INNER)):
        mask = np.array([(bits >> k) & 1 for k in range(CELL_INNER * CELL_INNER)], dtype=bool).reshape(CELL_INNER, CELL_INNER)
        active = int(mask.sum())
        if active not in FULL_ACTIVE_PIXEL_COUNTS:
            continue
        if not mask[1, 1]:
            continue
        if not mask[0, :].any() or not mask[2, :].any() or not mask[:, 0].any() or not mask[:, 2].any():
            continue
        if not _connected(mask):
            continue
        masks.append(mask)
    return masks


def _precompute_center_sets() -> dict[int, list[tuple[tuple[int, int], ...]]]:
    by_count: dict[int, list[tuple[tuple[int, int], ...]]] = {}
    for count in SEED_COUNTS:
        valid = []
        for combo in combinations(_VALID_CENTERS, count):
            ok = True
            for (r1, c1), (r2, c2) in combinations(combo, 2):
                if abs(r1 - r2) < CELL_INNER and abs(c1 - c2) < CELL_INNER:
                    ok = False
                    break
            if ok:
                valid.append(combo)
        by_count[count] = valid
    return by_count


_FULL_MASKS = _precompute_full_masks()
_CENTER_SETS = _precompute_center_sets()


def _random_lr_pattern(color_a: int, color_b: int) -> np.ndarray:
    arr = np.empty((CELL_INNER, CELL_INNER), dtype=int)
    choices = [random.choice((color_a, color_b)) for _ in _LR_GROUPS]
    if all(v == choices[0] for v in choices):
        choices[random.randrange(len(choices))] = color_b if choices[0] == color_a else color_a
    for group, color in zip(_LR_GROUPS, choices):
        for r, c in group:
            arr[r, c] = color
    return arr


def _cell_origin(i: int, j: int) -> tuple[int, int]:
    return i * CELL_STEP + 1, j * CELL_STEP + 1


def _build_base_grid(separator: int, majority: np.ndarray) -> np.ndarray:
    grid = np.full((GRID_SIZE, GRID_SIZE), separator, dtype=int)
    for i in range(CELL_GRID_N):
        for j in range(CELL_GRID_N):
            paste(grid, majority, *_cell_origin(i, j))
    return grid


def _apply_block(grid: np.ndarray, majority: np.ndarray, full_pattern: np.ndarray, full_mask: np.ndarray, center: tuple[int, int]) -> None:
    ci, cj = center
    radius = CELL_INNER // 2
    for di in range(CELL_INNER):
        for dj in range(CELL_INNER):
            target = full_pattern if full_mask[di, dj] else majority
            paste(grid, target, *_cell_origin(ci + di - radius, cj + dj - radius))


def _fragment_masks_from_full(full_mask: np.ndarray, fragment_count: int) -> list[np.ndarray] | None:
    active_coords = [tuple(x) for x in np.argwhere(full_mask)]
    center = (CELL_INNER // 2, CELL_INNER // 2)
    non_center = [coord for coord in active_coords if coord != center]
    for _ in range(300):
        masks = [np.zeros_like(full_mask, dtype=bool) for _ in range(fragment_count)]
        for mask in masks:
            mask[center] = True
        for coord in non_center:
            subset_size = random.randint(1, fragment_count)
            for idx in random.sample(range(fragment_count), subset_size):
                masks[idx][coord] = True

        sizes = [int(mask.sum()) for mask in masks]
        if any(size < 3 or size >= int(full_mask.sum()) for size in sizes):
            continue
        if any(not _connected(mask) for mask in masks):
            continue
        keys = [tuple(mask.ravel().tolist()) for mask in masks]
        if len(set(keys)) != fragment_count:
            continue
        if any(np.array_equal(mask, full_mask) for mask in masks):
            continue
        if any(not np.any(full_mask & ~mask) for mask in masks):
            continue
        union = np.zeros_like(full_mask, dtype=bool)
        for mask in masks:
            union |= mask
        if not np.array_equal(union, full_mask):
            continue
        return masks
    return None


def _build_wrong_independent_output(
    separator: int, majority: np.ndarray, fragment_patterns: list[np.ndarray], fragment_masks: list[np.ndarray], centers: tuple[tuple[int, int], ...]
) -> np.ndarray:
    wrong = _build_base_grid(separator, majority)
    radius = CELL_INNER // 2
    for center, frag_pattern, frag_mask in zip(centers, fragment_patterns, fragment_masks):
        ci, cj = center
        for di in range(CELL_INNER):
            for dj in range(CELL_INNER):
                target = frag_pattern if frag_mask[di, dj] else majority
                paste(wrong, target, *_cell_origin(ci + di - radius, cj + dj - radius))
    return wrong


def _gen_attempt() -> np.ndarray | None:
    separator, maj_exclusive, shared, anomaly = random.sample(range(COLOR_POOL), 4)
    majority = _random_lr_pattern(shared, maj_exclusive)

    seed_count = random.choice(SEED_COUNTS)
    centers = random.choice(_CENTER_SETS[seed_count])
    full_mask = random.choice(_FULL_MASKS)
    fragment_masks = _fragment_masks_from_full(full_mask, seed_count)
    if fragment_masks is None:
        return None

    full_pattern = np.where(full_mask, anomaly, shared).astype(int)
    fragment_patterns = [np.where(mask, anomaly, shared).astype(int) for mask in fragment_masks]

    out = _build_base_grid(separator, majority)
    for center in centers:
        _apply_block(out, majority, full_pattern, full_mask, center)

    wrong = _build_wrong_independent_output(separator, majority, fragment_patterns, fragment_masks, centers)
    if np.array_equal(wrong, out):
        return None

    inp = _build_base_grid(separator, majority)
    for center, frag_pattern in zip(centers, fragment_patterns):
        paste(inp, frag_pattern, *_cell_origin(*center))

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid b99e7126 combo input")
