# Combo variant of 36a08778 (index 28)
# combo_spec: D2/1+D14/2
# Name: 缺口穿透+交替左右溢出
#
# 设计分析
# a) 具体化与行为追踪
#    原题：水流垂直下落，撞到完整红架子后，在架子上方一行同时向左右两端扩散，并从两端继续下落。
#    本融合：分两类架子。
#    - 完整架子：不再双侧分叉，只从一侧溢出；同一条分支第一次撞完整架子向左，第二次向右，之后左右交替。
#    - 缺口架子：同一行由两段红色水平段组成，中间正好空 1 格；水流撞到这类架子时，会在架子上方一行横向流到缺口列，
#      然后从缺口继续单路下落，不发生溢出分叉。
#    这里选择“缺口穿透不计入左右交替计数”。因此一个具体链条可以是：
#    1. 第 1 个完整架子：左溢出；
#    2. 第 2 个架子有缺口：穿透，交替计数保持不变；
#    3. 第 3 个完整架子：仍视为“第 2 次完整架子碰撞”，所以右溢出；
#    4. 第 4 个完整架子：再切回左溢出。
#    这和“缺口也计数”的解释不同；后者在第 3 步会走左边。
#
# b) 复杂度校准
#    只保留两类清晰对象：完整架子、单格缺口架子。缺口宽度固定为 1，避免自由度过多。
#    交替规则只绑定到“完整架子命中次数”，而不是行号、全局顺序或所有碰撞次数，视觉上更稳定。
#
# c) 隐式假设审查
#    原 solver 的关键前提是：
#    - 架子是水平连续段；
#    - 碰撞后一定双侧扩散；
#    - BFS 的状态只需记位置，不需记命中历史。
#    融合后前两条都失效：
#    - 单行里可能出现两个红段 + 1 格缺口，不能只看“包含当前列的单段”；
#    - 完整架子的行为依赖该分支此前命中过多少个完整架子，所以状态必须扩展为 (row, col, solid_hits)。
#    同时，缺口穿透与完整架子交替规则叠加后引入新假设：生成器必须确保至少出现“完整 -> 缺口 -> 完整”的链条，
#    否则“缺口不计数”这一点在样例中不可观察。
#
# d) 放置合法性
#    生成器按证据链直接构造 4 个主架子，并预留所有必经下落列与横向流动行，避免和干扰架子重叠。
#    缺口架子总是两段红色，中间 1 格背景；完整架子总是单段连续红色。
#
# e) 视觉可观察性
#    单个样例中就会出现事件序列 L -> G -> R -> L：
#    - L/R 表示完整架子的左右单侧溢出；
#    - G 表示缺口穿透。
#    人类可以直接看到“有缺口时不是左右选边，而是穿过去”，以及“下一次完整架子并没有因为穿透而翻转方向”。
#
# f) 可视化整洁度
#    只使用背景、品红水源、红架子三种颜色；对象层次固定。
#    主路径单链式布置，横向移动和竖向下落交替出现，避免同一区域多层覆盖。
#
# g) 颜色变化可推断性
#    本融合不引入颜色映射变化，所有可推断信息都由几何结构给出：完整段 vs 缺口段、左/右单侧溢出、是否穿透。

from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(SOLVER_DIR))

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
    if top_touching:
        water = int(min(top_touching, key=lambda c: (int(cnt[int(c)]), int(c))))
    else:
        water = int(min(non_bg, key=lambda c: (int(cnt[int(c)]), int(c))))

    shelf_candidates = [int(c) for c in non_bg if int(c) != int(water)]
    assert shelf_candidates
    shelf = int(max(shelf_candidates, key=lambda c: (int(cnt[int(c)]), -int(c))))
    return bg, water, shelf


def _get_sources(grid: np.ndarray, water: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for coords in components_mask(grid == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))
    return sources


def _build_shelf_segments(grid: np.ndarray, shelf: int) -> list[list[tuple[int, int]]]:
    H, W = grid.shape
    segs: list[list[tuple[int, int]]] = [[] for _ in range(H)]
    for r in range(H):
        c = 0
        while c < W:
            if int(grid[r, c]) != int(shelf):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < W and int(grid[r, c + 1]) == int(shelf):
                c += 1
            segs[r].append((int(c0), int(c)))
            c += 1
    return segs


def _find_segment(row_segs: list[tuple[int, int]], col: int) -> tuple[int, int] | None:
    for a, b in row_segs:
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b))
    return None


def _find_gap_route(row_segs: list[tuple[int, int]], col: int) -> int | None:
    hit = _find_segment(row_segs, col)
    if hit is None:
        return None
    idx = row_segs.index(hit)
    a, b = hit
    if idx + 1 < len(row_segs):
        na, _nb = row_segs[idx + 1]
        if int(na) == int(b) + 2:
            return int(b + 1)
    if idx - 1 >= 0:
        _pa, pb = row_segs[idx - 1]
        if int(a) == int(pb) + 2:
            return int(pb + 1)
    return None


def _paint_run(
    grid: np.ndarray,
    out: np.ndarray,
    bg: int,
    row: int,
    start_col: int,
    stop_col: int,
    water: int,
) -> tuple[bool, int]:
    step = 1 if int(stop_col) >= int(start_col) else -1
    c = int(start_col)
    last = int(start_col)
    while True:
        if not (0 <= c < grid.shape[1]):
            return False, int(last)
        if c != int(start_col) and int(grid[row, c]) != int(bg):
            return False, int(last)
        if int(grid[row, c]) == int(bg):
            out[int(row), int(c)] = int(water)
        else:
            out[int(row), int(c)] = int(water)
        last = int(c)
        if int(c) == int(stop_col):
            return True, int(last)
        c += step


def _simulate(
    grid: np.ndarray,
    *,
    allow_gap_penetration: bool,
    count_gap_in_alternation: bool,
    split_on_solid: bool,
) -> tuple[np.ndarray, dict[str, object]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, water, shelf = _detect_roles(g)
    out = g.copy()
    shelf_segs = _build_shelf_segments(g, shelf)
    sources = _get_sources(g, water)

    q = deque((int(r), int(c), 0) for r, c in sources)
    seen: set[tuple[int, int, int]] = set()
    stats: dict[str, object] = {
        "gap_hits": 0,
        "solid_hits": 0,
        "left_overflows": 0,
        "right_overflows": 0,
        "events": [],
    }

    while q:
        r, c, solid_hits = q.popleft()
        key = (int(r), int(c), int(solid_hits))
        if key in seen:
            continue
        seen.add(key)

        rr, cc = int(r), int(c)
        out[int(rr), int(cc)] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= H:
                break

            if int(g[int(nr), int(cc)]) == int(shelf):
                row_segs = shelf_segs[int(nr)]
                gap_col = _find_gap_route(row_segs, cc) if allow_gap_penetration else None
                if gap_col is not None:
                    ok, last = _paint_run(g, out, bg, rr, cc, int(gap_col), water)
                    if not ok:
                        break
                    stats["gap_hits"] = int(stats["gap_hits"]) + 1
                    cast_events = stats["events"]
                    assert isinstance(cast_events, list)
                    cast_events.append("G")
                    cc = int(last)
                    if count_gap_in_alternation:
                        solid_hits = int(solid_hits + 1)
                    continue

                seg = _find_segment(row_segs, cc)
                if seg is None:
                    break
                a, b = seg
                left = int(max(0, a - 1))
                right = int(min(W - 1, b + 1))
                targets = [left, right] if split_on_solid else [int(left if solid_hits % 2 == 0 else right)]
                next_hits = int(solid_hits + 1)

                stats["solid_hits"] = int(stats["solid_hits"]) + 1
                cast_events = stats["events"]
                assert isinstance(cast_events, list)

                if split_on_solid:
                    cast_events.extend(["L", "R"])
                else:
                    if int(targets[0]) < int(cc):
                        stats["left_overflows"] = int(stats["left_overflows"]) + 1
                        cast_events.append("L")
                    elif int(targets[0]) > int(cc):
                        stats["right_overflows"] = int(stats["right_overflows"]) + 1
                        cast_events.append("R")
                    else:
                        cast_events.append("S")

                for target in targets:
                    ok, last = _paint_run(g, out, bg, rr, cc, int(target), water)
                    if ok and int(last) != int(cc):
                        q.append((int(rr), int(last), int(next_hits)))
                break

            rr = int(nr)
            out[int(rr), int(cc)] = int(water)

    return out, stats


def _behavior_metrics(grid: np.ndarray) -> dict[str, object]:
    combo_out, combo_stats = _simulate(
        grid,
        allow_gap_penetration=True,
        count_gap_in_alternation=False,
        split_on_solid=False,
    )
    counted_out, _ = _simulate(
        grid,
        allow_gap_penetration=True,
        count_gap_in_alternation=True,
        split_on_solid=False,
    )
    base_out, _ = _simulate(
        grid,
        allow_gap_penetration=False,
        count_gap_in_alternation=False,
        split_on_solid=True,
    )
    return {
        "gap_hits": int(combo_stats["gap_hits"]),
        "solid_hits": int(combo_stats["solid_hits"]),
        "events": list(combo_stats["events"]),
        "diff_vs_gap_counted": int(np.sum(combo_out != counted_out)),
        "diff_vs_base": int(np.sum(combo_out != base_out)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    out, _stats = _simulate(
        grid,
        allow_gap_penetration=True,
        count_gap_in_alternation=False,
        split_on_solid=False,
    )
    return out


def _reserve_vertical(cells: set[tuple[int, int]], col: int, row0: int, row1: int) -> None:
    for r in range(int(row0), int(row1) + 1):
        cells.add((int(r), int(col)))


def _reserve_horizontal(cells: set[tuple[int, int]], row: int, col0: int, col1: int) -> None:
    lo, hi = sorted((int(col0), int(col1)))
    for c in range(lo, hi + 1):
        cells.add((int(row), int(c)))


def _add_segment(
    grid: np.ndarray,
    shelf_cells: set[tuple[int, int]],
    row: int,
    a: int,
    b: int,
) -> bool:
    H, W = grid.shape
    if not (0 <= int(row) < H and 0 <= int(a) <= int(b) < W):
        return False
    for c in range(int(a), int(b) + 1):
        if (int(row), int(c)) in shelf_cells:
            return False
    for c in range(int(a), int(b) + 1):
        shelf_cells.add((int(row), int(c)))
        grid[int(row), int(c)] = int(SHELF)
    return True


def _place_distractors(
    grid: np.ndarray,
    shelf_cells: set[tuple[int, int]],
    reserved_clear: set[tuple[int, int]],
    blocked_rows: set[int],
) -> None:
    H, W = grid.shape
    for _ in range(random.randint(0, 2)):
        for _attempt in range(50):
            row = random.randint(2, H - 2)
            if int(row) in blocked_rows:
                continue
            length = random.randint(2, min(6, W - 2))
            a = random.randint(1, W - length - 1)
            b = int(a + length - 1)
            cells = {(int(row), int(c)) for c in range(int(a), int(b) + 1)}
            if cells & shelf_cells:
                continue
            if cells & reserved_clear:
                continue
            if _add_segment(grid, shelf_cells, int(row), int(a), int(b)):
                break


def _gen_attempt() -> np.ndarray | None:
    W = random.randint(14, 22)
    source_col = random.randint(5, W - 6)
    c1 = int(source_col - random.randint(2, 3))

    gap_shift_choices = [d for d in (-3, -2, -1, 1, 2, 3) if 3 <= int(c1 + d) <= W - 4]
    if not gap_shift_choices:
        return None
    gap_col = int(c1 + random.choice(gap_shift_choices))

    c3_choices = [gap_col + d for d in (2, 3) if int(gap_col + d) <= W - 4]
    if not c3_choices:
        return None
    c3 = int(random.choice(c3_choices))

    c4_choices = [c3 - d for d in (2, 3) if int(c3 - d) >= 2]
    if not c4_choices:
        return None
    c4 = int(random.choice(c4_choices))

    r1 = random.randint(4, 5)
    r2 = int(r1 + random.randint(3, 4))
    r3 = int(r2 + random.randint(3, 4))
    r4 = int(r3 + random.randint(3, 4))
    H = random.randint(r4 + 3, min(28, r4 + 8))

    grid = np.full((H, W), int(BG), dtype=int)
    grid[0, int(source_col)] = int(WATER)
    grid[1, int(source_col)] = int(WATER)

    shelf_cells: set[tuple[int, int]] = set()

    a1 = int(c1 + 1)
    b1_lo = max(int(source_col), int(a1 + 1))
    b1_hi = min(W - 3, int(source_col + 4))
    if b1_lo > b1_hi:
        return None
    b1 = int(random.randint(b1_lo, b1_hi))
    if not _add_segment(grid, shelf_cells, r1, a1, b1):
        return None

    if int(gap_col) > int(c1):
        a2_left = random.randint(max(1, c1 - 2), c1)
        b2_left = int(gap_col - 1)
        b2_right = int(gap_col + random.randint(2, 4))
        if b2_right > W - 2 or a2_left > b2_left:
            return None
        if not _add_segment(grid, shelf_cells, r2, a2_left, b2_left):
            return None
        if not _add_segment(grid, shelf_cells, r2, gap_col + 1, b2_right):
            return None
    else:
        a2_left = int(gap_col - random.randint(2, 4))
        b2_left = int(gap_col - 1)
        b2_right = random.randint(c1, min(W - 2, c1 + 2))
        if a2_left < 1 or a2_left > b2_left:
            return None
        if gap_col + 1 > b2_right:
            return None
        if not _add_segment(grid, shelf_cells, r2, a2_left, b2_left):
            return None
        if not _add_segment(grid, shelf_cells, r2, gap_col + 1, b2_right):
            return None

    b3 = int(c3 - 1)
    a3 = int(random.randint(max(1, gap_col - 3), gap_col))
    if a3 > int(gap_col) or b3 < int(gap_col) or b3 > W - 2:
        return None
    if not _add_segment(grid, shelf_cells, r3, a3, b3):
        return None

    a4 = int(c4 + 1)
    b4_lo = max(int(c3), int(a4 + 1))
    b4_hi = min(W - 2, int(c3 + 3))
    if b4_lo > b4_hi:
        return None
    b4 = int(random.randint(b4_lo, b4_hi))
    if not _add_segment(grid, shelf_cells, r4, a4, b4):
        return None

    reserved_clear: set[tuple[int, int]] = {(0, int(source_col)), (1, int(source_col))}
    _reserve_vertical(reserved_clear, int(source_col), 2, r1 - 1)
    _reserve_horizontal(reserved_clear, r1 - 1, int(source_col), int(c1))
    _reserve_vertical(reserved_clear, int(c1), r1, r2 - 1)
    _reserve_horizontal(reserved_clear, r2 - 1, int(c1), int(gap_col))
    _reserve_vertical(reserved_clear, int(gap_col), r2, r3 - 1)
    _reserve_horizontal(reserved_clear, r3 - 1, int(gap_col), int(c3))
    _reserve_vertical(reserved_clear, int(c3), r3, r4 - 1)
    _reserve_horizontal(reserved_clear, r4 - 1, int(c3), int(c4))
    _reserve_vertical(reserved_clear, int(c4), r4, H - 1)

    _place_distractors(grid, shelf_cells, reserved_clear, {r1, r2, r3, r4, r1 - 1, r2 - 1, r3 - 1, r4 - 1})

    out = main(grid.copy())
    if np.array_equal(out, grid):
        return None

    metrics = _behavior_metrics(grid)
    if int(metrics["gap_hits"]) < 1 or int(metrics["solid_hits"]) < 3:
        return None
    events = metrics["events"]
    if not isinstance(events, list) or events[:4] != ["L", "G", "R", "L"]:
        return None
    if int(metrics["diff_vs_gap_counted"]) <= 0:
        return None
    if int(metrics["diff_vs_base"]) <= 0:
        return None

    for row, col in shelf_cells:
        if int(out[int(row), int(col)]) != int(SHELF):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid combo input after 400 attempts")
