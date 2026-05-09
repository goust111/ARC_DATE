# Combo variant of 36a08778 (index 24)
# combo_spec: D7/0+D9/2
# Name: 颜色停止+网格扩展
#
# Design analysis:
# - a) 具体化与行为追踪:
#   原题只有三种角色: 顶边品红水源、红色水平架子、背景。水流垂直下落，遇到架子后在上一行向
#   左右扩散到“架子端点外再多一格”，再从两个溢出端继续下落；越界部分会被原网格尺寸截掉。
#   本融合版新增蓝色排水口，并允许输出最小限度横向扩展来容纳真正的溢出路径。
#   小例子:
#   源头在第 2 列；第 4 行有架子覆盖第 0..3 列。原题命中该架子时，左溢本应去到第 -1 列，但
#   被输入宽度裁掉，只剩到第 0 列。融合后输出增加 1 列左边界，让左支在新列继续下落到底部。
#   同时若右支下一步将进入蓝色排水口，则该分支在排水口正上方停止，排水口本身保持蓝色不改色。
#   差异集中在两个节点:
#   1. “下一步是不是排水口”优先于“继续下落/命中架子”。
#   2. 架子端点外溢不再被输入宽度裁掉，而是在最小必要的一列扩展中继续存在。
# - b) 复杂度校准:
#   若只加排水口、不制造边缘架子，则 D9/2 不可见；若只做边缘溢出、不让任何支流遇到排水口，
#   D7/0 不可见。生成器因此强制:
#   1. 第一层有效架子触碰左或右边界，保证至少一条分支越界并触发宽度扩展。
#   2. 至少一条内部支流被蓝色排水口截断，保证与“无排水口版本”存在可量化差异。
#   3. 至少三条架子段存在，避免视觉上退化成单次碰撞的小例子。
# - c) 隐式假设审查:
#   1. 原代码 `water = rarest non-bg` 在有蓝色排水口后不再可靠，因为排水口可能比水源更稀少。
#      新 solver 改成“触顶的非背景连通块里选水源色”，保留人类可见的顶部源头线索。
#   2. 原代码默认“除水源外唯一另一种颜色就是架子”。融合后必须区分“有水平连续段的颜色=架子”
#      与“其余非背景颜色=排水口”。
#   3. 原代码把输出数组固定为输入大小，且 `a-1/b+1` 会被 `0..W-1` 裁切；融合后该前提失效，
#      改成在左右各预留 1 列世界边界，并在结束后只保留真正被使用的扩展列。
#   4. 原代码只把“到底边”当作垂直停止条件。融合后新增“下一格是排水口则停止且不涂蓝格”。
#   5. 原代码的 BFS `seen[(r,c)]` 依赖分支起点坐标唯一。这里仍成立，因为新扩展只会把边缘溢出
#      推到输入外 1 列；该列中不会再命中新架子，因此不会制造无限新状态。
# - d) 放置合法性:
#   生成器只放 1 个顶部水源、3 条主动架子和 2 个排水口。架子彼此不重叠，排水口不放在架子或
#   水源上，也不放在计划中的水平扩散线上；第三条架子显式避开第二层另一支的列，防止意外交叉。
# - e) 视觉可观察性:
#   观察者会同时看到两类稳定信号:
#   1. 触边架子总会让输出比输入多出一列，新增列里继续出现品红竖流。
#   2. 蓝色像素总恰好出现在品红支流的正下方，且水在其上方戛然而止、蓝色保持不变。
#   这两类信号在 2-4 组样例里都可复现，不需要依赖计数或隐藏状态。
# - f) 可视化整洁度:
#   架子只用横线、排水口只用单像素、源头只用 1x2 竖条。生成器避免把排水口放到水平扩散层上，
#   也避免多条主动架子落在同一行，从而保证对象层次清楚。
# - g) 颜色变化可推断性:
#   融合版只有 4 种颜色角色: 背景、品红水、红架子、蓝排水口。蓝色不发生颜色映射，只承担
#   “吸收停止”的结构作用；颜色数保持很少，解题者可以直接从位置关系推断规则。
# - h) main() 的新增隐式假设:
#   1. 只有一个“触顶水源颜色”；生成器保证顶边只有水源色。
#   2. 架子颜色至少出现一段长度 >= 2 的水平连续段；生成器的三条主动架子都满足。
#   3. 排水口不相邻成横向段，否则会干扰架子检测；生成器把排水口分开放置。
#   4. 真正需要的横向扩展最多各侧 1 列。对本任务成立，因为所有架子都位于原输入内部，越界后
#      的支流只会在新增空白列中直落，不可能再命中输入外的架子。

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
DRAIN = 1


def _detect_roles(grid: np.ndarray) -> tuple[int, int, int, set[int]]:
    g = np.array(grid, dtype=int)
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert non_bg

    top_touching = [int(c) for c in non_bg if bool((g[0] == int(c)).any())]
    assert top_touching
    water = int(min(top_touching, key=lambda c: (int(cnt[int(c)]), int(c))))

    rest = [int(c) for c in non_bg if int(c) != int(water)]
    assert rest

    def has_horizontal_run(color: int) -> bool:
        for r in range(g.shape[0]):
            for c in range(g.shape[1] - 1):
                if int(g[r, c]) == int(color) and int(g[r, c + 1]) == int(color):
                    return True
        return False

    shelf_candidates = [int(c) for c in rest if has_horizontal_run(int(c))]
    assert shelf_candidates
    shelf = int(max(shelf_candidates, key=lambda c: (int(cnt[int(c)]), -int(c))))
    drains = {int(c) for c in rest if int(c) != int(shelf)}
    return bg, water, shelf, drains


def _extract_sources(grid: np.ndarray, water: int) -> list[tuple[int, int]]:
    sources: list[tuple[int, int]] = []
    for coords in components_mask(grid == int(water), connectivity=8):
        if coords.size == 0:
            continue
        if not bool((coords[:, 0] == 0).any()):
            continue
        rmax = int(coords[:, 0].max())
        cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        for c in cols:
            sources.append((int(rmax), int(c)))
    return sources


def _shelf_segments(grid: np.ndarray, shelf: int) -> list[list[tuple[int, int]]]:
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
            segs[int(r)].append((int(c0), int(c)))
            c += 1
    return segs


def _find_segment(segments: list[list[tuple[int, int]]], row: int, col: int) -> tuple[int, int] | None:
    for a, b in segments[int(row)]:
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b))
    return None


def _trim_output(out: np.ndarray, original_width: int, bg: int) -> np.ndarray:
    used = np.argwhere(out != int(bg))
    if used.size == 0:
        return out[:, 1:1 + int(original_width)]
    cols = used[:, 1]
    left = min(1, int(cols.min()))
    right = max(int(original_width), int(cols.max()))
    return out[:, int(left) : int(right) + 1]


def _run_solver(grid: np.ndarray, *, stop_on_drains: bool, allow_expand: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, water, shelf, drain_colors = _detect_roles(g)

    side_pad = 1 if allow_expand else 0
    world = np.full((H, W + 2 * side_pad), int(bg), dtype=int)
    out = world.copy()
    world[:, side_pad : side_pad + W] = g
    out[:, side_pad : side_pad + W] = g

    sources = [(int(r), int(c + side_pad)) for r, c in _extract_sources(g, water)]
    segments = _shelf_segments(world, shelf)

    seen: set[tuple[int, int]] = set()
    q: deque[tuple[int, int]] = deque(sources)

    while q:
        r, c = q.popleft()
        if (int(r), int(c)) in seen:
            continue
        seen.add((int(r), int(c)))

        rr, cc = int(r), int(c)
        out[int(rr), int(cc)] = int(water)

        while True:
            nr = int(rr + 1)
            if nr >= H:
                break
            below = int(world[int(nr), int(cc)])

            if stop_on_drains and below in drain_colors:
                break

            if below == int(shelf):
                seg = _find_segment(segments, nr, cc)
                if seg is None:
                    break
                a, b = seg
                nominal_left = int(a - 1)
                nominal_right = int(b + 1)

                eff_left = int(cc)
                for x in range(int(cc) - 1, nominal_left - 1, -1):
                    if not (0 <= int(x) < world.shape[1]):
                        break
                    if int(world[int(rr), int(x)]) != int(bg):
                        break
                    eff_left = int(x)
                    out[int(rr), int(x)] = int(water)

                eff_right = int(cc)
                for x in range(int(cc) + 1, nominal_right + 1):
                    if not (0 <= int(x) < world.shape[1]):
                        break
                    if int(world[int(rr), int(x)]) != int(bg):
                        break
                    eff_right = int(x)
                    out[int(rr), int(x)] = int(water)

                if eff_left == nominal_left and nominal_left != int(cc):
                    q.append((int(rr), int(nominal_left)))
                if eff_right == nominal_right and nominal_right != int(cc):
                    q.append((int(rr), int(nominal_right)))
                break

            rr = nr
            out[int(rr), int(cc)] = int(water)

    if not allow_expand:
        return out[:, side_pad : side_pad + W]
    return _trim_output(out, W, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Waterfall with two fused modifications:
    1. water stops before entering blue drain pixels;
    2. edge overflows may widen the output by one column on the touched side.
    """
    return _run_solver(grid, stop_on_drains=True, allow_expand=True)


def _baseline_without_combo(grid: np.ndarray) -> np.ndarray:
    """
    Counterfactual control solver used only by the generator:
    same role detection, but no drain stopping and no grid expansion.
    """
    return _run_solver(grid, stop_on_drains=False, allow_expand=False)


def _choose_internal_segment(
    width: int,
    impact_col: int,
    *,
    forbid_cols: set[int] | None = None,
    min_len: int = 2,
    max_len: int = 4,
) -> tuple[int, int] | None:
    forbid = set() if forbid_cols is None else {int(c) for c in forbid_cols}
    max_len = min(int(max_len), int(width - 2))
    if max_len < min_len:
        return None
    for _ in range(100):
        length = random.randint(int(min_len), int(max_len))
        offset = random.randint(0, length - 1)
        a = int(impact_col - offset)
        b = int(a + length - 1)
        if a < 1 or b > int(width - 2):
            continue
        if any(int(a) <= int(c) <= int(b) for c in forbid):
            continue
        return (int(a), int(b))
    return None


def _find_input_offset(out: np.ndarray, grid: np.ndarray) -> int | None:
    H, W = grid.shape
    bg = int(Counter(map(int, grid.flatten())).most_common(1)[0][0])
    non_bg = [(int(r), int(c)) for r, c in np.argwhere(grid != int(bg))]
    for offset in range(out.shape[1] - W + 1):
        if all(int(out[int(r), int(c + offset)]) == int(grid[int(r), int(c)]) for r, c in non_bg):
            return int(offset)
    return None


def _count_effective_drains(out: np.ndarray, grid: np.ndarray, offset: int) -> int:
    count = 0
    for r, c in np.argwhere(grid == int(DRAIN)):
        rr, cc = int(r), int(c)
        if rr == 0:
            continue
        if int(out[int(rr), int(cc + offset)]) != int(DRAIN):
            continue
        if int(out[int(rr - 1), int(cc + offset)]) == int(WATER):
            count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(10, 16)
    W = random.randint(6, 10)
    grid = np.full((H, W), int(BG), dtype=int)

    expand_side = random.choice(["left", "right"])
    if expand_side == "left":
        src = random.randint(1, max(1, min(W - 3, W // 2 + 1)))
    else:
        src = random.randint(min(W - 2, max(2, W // 2)), W - 2)
    grid[0, int(src)] = int(WATER)
    grid[1, int(src)] = int(WATER)

    r1 = random.randint(4, min(5, H - 6))
    if expand_side == "left":
        a1 = 0
        b1 = random.randint(int(src), min(W - 2, int(src) + 2))
        main_col = int(b1 + 1)
    else:
        b1 = W - 1
        a1 = random.randint(max(1, int(src) - 2), int(src))
        main_col = int(a1 - 1)

    if not (0 <= main_col < W):
        return None
    grid[int(r1), int(a1) : int(b1) + 1] = int(SHELF)

    r2_lo = int(r1 + 3)
    r2_hi = int(H - 4)
    if r2_lo > r2_hi:
        return None
    r2 = random.randint(r2_lo, min(r2_hi, int(r1 + 5)))
    seg2 = _choose_internal_segment(W, main_col, min_len=2, max_len=3)
    if seg2 is None:
        return None
    a2, b2 = seg2
    left2, right2 = int(a2 - 1), int(b2 + 1)
    if not (0 <= left2 < W and 0 <= right2 < W):
        return None
    children2 = [left2, right2]
    continue_col = random.choice(children2)
    drain2_col = int(children2[1] if continue_col == children2[0] else children2[0])
    grid[int(r2), int(a2) : int(b2) + 1] = int(SHELF)

    r3_lo = int(r2 + 2)
    r3_hi = int(H - 2)
    if r3_lo > r3_hi:
        return None
    r3 = random.randint(r3_lo, min(r3_hi, int(r2 + 4)))
    seg3 = _choose_internal_segment(
        W,
        continue_col,
        forbid_cols={drain2_col},
        min_len=2,
        max_len=3,
    )
    if seg3 is None:
        return None
    a3, b3 = seg3
    left3, right3 = int(a3 - 1), int(b3 + 1)
    if not (0 <= left3 < W and 0 <= right3 < W):
        return None
    terminal3_col = random.choice([left3, right3])
    grid[int(r3), int(a3) : int(b3) + 1] = int(SHELF)

    spread2 = int(r2 - 1)
    rd2_lo = int(spread2 + 2)
    rd2_hi = int(H - 1)
    if rd2_lo > rd2_hi:
        return None
    rd2 = random.randint(rd2_lo, rd2_hi)

    spread3 = int(r3 - 1)
    rd3_lo = int(spread3 + 2)
    rd3_hi = int(H - 1)
    if rd3_lo > rd3_hi:
        return None
    rd3 = random.randint(rd3_lo, rd3_hi)

    if len({(int(rd2), int(drain2_col)), (int(rd3), int(terminal3_col))}) < 2:
        return None
    grid[int(rd2), int(drain2_col)] = int(DRAIN)
    grid[int(rd3), int(terminal3_col)] = int(DRAIN)

    # Optional distractor shelf: placed only if it can avoid all planned active columns.
    active_cols = {int(src), int(main_col), int(left2), int(right2), int(continue_col), int(drain2_col), int(left3), int(right3), int(terminal3_col)}
    if random.random() < 0.45:
        for _ in range(40):
            dr = random.randint(2, H - 1)
            if dr in {int(r1), int(r2), int(r3), int(rd2), int(rd3)}:
                continue
            length = random.randint(2, min(4, W))
            da = random.randint(0, W - length)
            db = int(da + length - 1)
            if any(int(da) <= col <= int(db) for col in active_cols):
                continue
            if np.any(grid[int(dr), int(da) : int(db) + 1] != int(BG)):
                continue
            grid[int(dr), int(da) : int(db) + 1] = int(SHELF)
            break

    # Reject obvious geometric conflicts before solver validation.
    if np.any(grid[2:, :] == int(WATER)):
        return None
    if int(np.sum(grid == int(SHELF))) <= int(np.sum(grid == int(WATER))):
        return None

    try:
        out = main(grid.copy())
        base = _baseline_without_combo(grid.copy())
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    if out.shape[1] <= W:
        return None

    offset = _find_input_offset(out, grid)
    if offset is None:
        return None

    out_sub = out[:, int(offset) : int(offset) + W]
    if np.any(out_sub[grid == int(SHELF)] != int(SHELF)):
        return None
    if np.any(out_sub[grid == int(DRAIN)] != int(DRAIN)):
        return None

    drain_effect = int(np.sum((base == int(WATER)) & (out_sub != int(WATER))))
    if drain_effect <= 0:
        return None

    if _count_effective_drains(out, grid, offset) <= 0:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("Failed to generate valid input after 500 attempts")
