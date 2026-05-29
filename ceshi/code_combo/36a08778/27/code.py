# Combo variant of 36a08778 (index 27)
# combo_spec: D4/0+D8/0
# Name: 间歇涂色+架子溶解
#
# 设计分析：
# a) 具体化与行为追踪
#    原题里，水从顶边源头垂直落下；若下一格是架子，则在架子上一行向左右铺开，再从两端继续下落。
#    这个融合版把两个关键环节一起改掉：
#    - D4/0: 纵向轨迹不再连续上色，而是按相对源底行“隔一行”上色，形成虚线瀑布。
#    - D8/0: 架子不再阻挡和分叉；水撞到架子时，只把命中的那个架子像素擦成背景，并沿同一列继续下落。
#    例子：源底行为 r=1。原题若在 r=4 处遇架子，会在 r=3 这一行横向扩散并分叉。
#    本题则在 r=2 跳过、r=3 涂色、r=4 把红架子擦空、r=5 涂色、r=6 跳过……整列向下继续。
#    差异在像素层面非常明确：没有横向水幕，只有竖直虚线和被打穿的红架子孔洞。
#
# b) 复杂度校准
#    若只放一个源和一条架子，这个变体会过于简单，所以生成器强制：
#    - 至少 3 条会被击中的架子；
#    - 1-2 个源；
#    - 至少 1 条不会被击中的对照架子。
#    这样训练样例里会重复出现“虚线列”和“只有命中处才侵蚀”的规律，避免歧义。
#
# c) 隐式假设审查（对应原 main 的关键前提）
#    - 原代码依赖 `find_shelf_seg()`：默认架子是阻挡物，并且命中后要读出整段左右边界。
#      这里前提失效，因为架子不再触发横向传播；solver 不再需要段级碰撞逻辑。
#    - 原代码依赖 BFS `seen`：默认一次命中会产生左右两个子流。
#      这里前提失效，因为新流永远不分叉；每个源只需单列扫描到底。
#    - 原代码在每一步都 `out[...] = water`：默认轨迹连续。
#      这里前提失效，改为由相对源底行控制的奇偶相位决定是否上色。
#    - 原代码读 `below = g[nr, cc]` 并在碰架后停止当前下落。
#      这里改为“碰架后擦除但不停流”；因此读取输入网格做判定、写输出网格做侵蚀是安全的。
#
# d) 放置合法性
#    生成器只放 1x2 顶边水源、水平连续架子；所有对象都在边界内。
#    会被击中的架子必须覆盖某个源列；不会被击中的架子明确避开所有源列，避免语义混杂。
#
# e) 视觉可观察性
#    人类从少量样例能直接看到：
#    - 水不再横向扩散；
#    - 竖直路径呈隔行点状；
#    - 红架子只在源列位置被“打孔”，未命中的红架子保持不变。
#
# f) 可视化整洁度
#    生成器避免把多个源列放得过近，也避免同一行放多段相互干扰的架子。
#    这样红架子、品红虚线和背景空洞三者始终可分辨。
#
# g) 颜色变化可推断性
#    本变体不引入新颜色映射；只保留原题三色体系。
#    颜色含义稳定：橙=背景，品红=水，红=架子；架子被侵蚀后回到背景色。

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC2_DIR = ROOT / "arc2_solver_select"
if str(ARC2_DIR) not in sys.path:
    sys.path.insert(0, str(ARC2_DIR))

from arc_utils import components_mask


BG = 7
WATER = 6
SHELF = 2


def _detect_roles(grid: np.ndarray) -> tuple[int, int, int]:
    cnt = Counter(map(int, grid.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert non_bg

    top_touching = [int(c) for c in non_bg if bool((grid[0] == int(c)).any())]
    pool = top_touching if top_touching else non_bg
    water = int(min(pool, key=lambda c: (int(cnt[int(c)]), int(c))))

    shelves = [int(c) for c in non_bg if int(c) != int(water)]
    assert shelves
    shelf = int(max(shelves, key=lambda c: (int(cnt[int(c)]), -int(c))))
    return int(bg), int(water), int(shelf)


def _find_sources(grid: np.ndarray, water: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for coords in components_mask(grid == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))
    return sorted(set(sources))


def _should_paint(start_row: int, row: int) -> bool:
    # Keep the source itself solid and extend the new trail on every second row after it.
    return int(row - start_row) % 2 == 0


def _simulate_variant(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _ = g.shape
    bg, water, shelf = _detect_roles(g)
    out = g.copy()

    for start_row, col in _find_sources(g, water):
        for r in range(int(start_row) + 1, H):
            cell = int(g[int(r), int(col)])
            if cell == int(shelf):
                out[int(r), int(col)] = int(bg)
                continue
            if cell == int(bg) and _should_paint(int(start_row), int(r)):
                out[int(r), int(col)] = int(water)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _simulate_variant(grid)


def _choose_source_cols(width: int) -> list[int]:
    if width >= 12 and random.random() < 0.55:
        for _ in range(100):
            c1 = random.randint(1, width - 7)
            c2 = random.randint(c1 + 5, min(width - 2, c1 + 10))
            if c2 - c1 >= 5:
                return [int(c1), int(c2)]
    return [int(random.randint(1, width - 2))]


def _active_target_sets(source_cols: list[int], count: int) -> list[set[int]]:
    if len(source_cols) == 1:
        return [{int(source_cols[0])} for _ in range(count)]

    left, right = (int(source_cols[0]), int(source_cols[1]))
    seeds: list[set[int]] = [{left}, {right}, {left, right}, {random.choice([left, right])}]
    while len(seeds) < count:
        mode = random.random()
        if mode < 0.35:
            seeds.append({left, right})
        elif mode < 0.675:
            seeds.append({left})
        else:
            seeds.append({right})
    random.shuffle(seeds)
    return seeds[:count]


def _choose_segment(width: int, targets: set[int], source_cols: list[int]) -> tuple[int, int] | None:
    include = sorted(int(c) for c in targets)
    forbid = [int(c) for c in source_cols if int(c) not in targets]
    lo = int(include[0])
    hi = int(include[-1])

    for _ in range(80):
        left_ext = random.randint(1, 3)
        right_ext = random.randint(1, 3)
        a = max(0, int(lo - left_ext))
        b = min(width - 1, int(hi + right_ext))
        if b - a + 1 < 2:
            continue
        if any(int(a) <= int(c) <= int(b) for c in forbid):
            continue
        return int(a), int(b)

    if any(int(lo) <= int(c) <= int(hi) for c in forbid):
        return None
    return int(lo), int(hi)


def _place_distractor(
    grid: np.ndarray,
    used_rows: set[int],
    source_cols: list[int],
) -> bool:
    H, W = grid.shape
    rows = [int(r) for r in range(3, H) if int(r) not in used_rows]
    random.shuffle(rows)
    for r in rows[:20]:
        for _ in range(40):
            length = random.randint(2, min(7, W))
            a = random.randint(0, W - length)
            b = int(a + length - 1)
            if any(int(a) <= int(c) <= int(b) for c in source_cols):
                continue
            if np.any(grid[int(r), int(a):int(b) + 1] != int(BG)):
                continue
            grid[int(r), int(a):int(b) + 1] = int(SHELF)
            used_rows.add(int(r))
            return True
    return False


def _behavior_metrics(grid: np.ndarray, out: np.ndarray) -> dict[str, int]:
    bg, water, shelf = _detect_roles(grid)
    eroded = int(np.sum((grid == int(shelf)) & (out == int(bg))))
    intact = int(np.sum((grid == int(shelf)) & (out == int(shelf))))

    skipped_bg = 0
    painted_bg = 0
    for start_row, col in _find_sources(grid, water):
        for r in range(int(start_row) + 1, grid.shape[0]):
            if int(grid[int(r), int(col)]) != int(bg):
                continue
            if _should_paint(int(start_row), int(r)):
                if int(out[int(r), int(col)]) == int(water):
                    painted_bg += 1
            else:
                if int(out[int(r), int(col)]) == int(bg):
                    skipped_bg += 1

    return {
        "eroded": int(eroded),
        "intact": int(intact),
        "skipped_bg": int(skipped_bg),
        "painted_bg": int(painted_bg),
    }


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(10, 22)
    W = random.randint(8, 22)
    grid = np.full((H, W), int(BG), dtype=int)

    source_cols = _choose_source_cols(W)
    for c in source_cols:
        grid[0, int(c)] = int(WATER)
        grid[1, int(c)] = int(WATER)

    min_active = 3 if len(source_cols) == 1 else 4
    max_active = min(6, H - 3)
    if max_active < min_active:
        return None

    active_count = random.randint(min_active, max_active)
    active_rows = sorted(random.sample(range(3, H), active_count))
    target_sets = _active_target_sets(source_cols, active_count)
    used_rows = set()

    for row, targets in zip(active_rows, target_sets):
        seg = _choose_segment(W, targets, source_cols)
        if seg is None:
            return None
        a, b = seg
        if np.any(grid[int(row), int(a):int(b) + 1] != int(BG)):
            return None
        grid[int(row), int(a):int(b) + 1] = int(SHELF)
        used_rows.add(int(row))

    distractor_goal = random.randint(1, 3)
    distractors = 0
    while distractors < distractor_goal and _place_distractor(grid, used_rows, source_cols):
        distractors += 1

    if distractors == 0:
        return None

    if int(np.sum(grid == int(SHELF))) <= int(np.sum(grid == int(WATER))):
        return None

    out = main(grid.copy())
    if np.array_equal(out, grid):
        return None

    metrics = _behavior_metrics(grid, out)
    expected_hits = sum(len(targets) for targets in target_sets)
    if metrics["eroded"] < max(3, expected_hits - 1):
        return None
    if metrics["skipped_bg"] < len(source_cols) * 2:
        return None
    if metrics["painted_bg"] < len(source_cols) * 2:
        return None
    if metrics["intact"] <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 400 attempts")
